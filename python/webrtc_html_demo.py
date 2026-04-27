#!/usr/bin/env python3
"""
apps/python/webrtc_html_demo.py

Browser streaming demo: RTSP / file / USB camera → WebRTC → browser.

Source codec requirements
--------------------------
File / RTSP:
    The source must carry an encoded stream (H.264, H.265, VP8, VP9).
    Encoded packets pass through RTPMuxerFrameFilter without re-encoding.

USB camera:
    The camera outputs raw decoded frames, so an encoder is required.

    --hw-accel  H.264 via NVENC (hardware).  Requires an NVIDIA GPU.
    (default)   H.264 via libx264 (software).

Pipelines
---------
File / RTSP:
  [LiveStreamThread | MediaFileThread]
       → RTPMuxerFrameFilter
       → WebRTCServerThread     HTTP signaling on WEBRTC_PORT
       → nginx (static HTML)    HTTP on HTTP_PORT

USB + --hw-accel (H.264 NVENC):
  USBCameraThread
       → EncodingFrameFilter(NVENC H.264)
       → RTPMuxerFrameFilter
       → WebRTCServerThread  ...

USB (H.264 libx264):
  USBCameraThread
       → EncodingFrameFilter(libx264 H.264)
       → RTPMuxerFrameFilter
       → WebRTCServerThread  ...

Usage:
    python3 apps/python/webrtc_html_demo.py --file video.mp4
    python3 apps/python/webrtc_html_demo.py --rtsp rtsp://user:pass@cam/stream
    python3 apps/python/webrtc_html_demo.py --usb /dev/video0 --hw-accel
    python3 apps/python/webrtc_html_demo.py --usb /dev/video0

Options:
    --rtsp         URL   RTSP stream URL
    --file         PATH  local media file
    --usb          DEV   V4L2 device (default /dev/video0)
    --webrtc-port        WebRTC signaling HTTP port (default 8090)
    --http-port          nginx static-file HTTP port (default 8091)
    --uuid               stream UUID (default 'stream', exposed as /stream)
    --fps                playback/capture rate for file/USB (default 25)
    --loop               loop file source
    --width              USB capture width (default 640)
    --height             USB capture height (default 480)
    --bitrate            USB encoder bitrate in bps (default 4_000_000)
    --hw-accel           USB: use NVENC H.264 instead of libx264

Press Ctrl+C to stop.
"""

import os
import time
import shlex
import argparse
import textwrap
import tempfile
import subprocess
import pathlib

import limef

SLOT = 1

_SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
_STATIC_DIR = _SCRIPT_DIR / "webrtc_html_demo" / "static"


# ── nginx ─────────────────────────────────────────────────────────────────────

_NGINX_CONF_TEMPLATE = textwrap.dedent("""\
    user {USER};
    worker_processes 1;
    daemon off;
    error_log  {TMPDIR}/error.log warn;
    pid        {TMPDIR}/nginx.pid;

    events {{ worker_connections 1024; }}

    http {{
        include      /etc/nginx/mime.types;
        default_type application/octet-stream;
        access_log   {TMPDIR}/access.log;
        sendfile     on;
        keepalive_timeout 65;

        server {{
            listen {HTTP_PORT};

            location / {{
                root  {STATIC_DIR};
                index index.html;
                add_header Last-Modified $date_gmt;
                add_header Cache-Control 'no-store, no-cache';
                if_modified_since off;
                expires off;
                etag off;
            }}
        }}
    }}
""")


def _start_nginx(tmpdir: pathlib.Path, http_port: int) -> subprocess.Popen:
    conf = _NGINX_CONF_TEMPLATE.format(
        USER=os.environ["USER"],
        TMPDIR=tmpdir,
        HTTP_PORT=http_port,
        STATIC_DIR=_STATIC_DIR,
    )
    conf_path = tmpdir / "nginx.conf"
    conf_path.write_text(conf)

    subprocess.run(["killall", "-9", "nginx"], capture_output=True)
    time.sleep(0.3)

    proc = subprocess.Popen(shlex.split(f"nginx -p {tmpdir} -c {conf_path}"))
    time.sleep(0.5)
    if proc.poll() is not None:
        raise RuntimeError(
            f"nginx failed to start (exit {proc.returncode}). "
            f"Check {tmpdir}/error.log for details."
        )
    return proc


def _stop_nginx(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ── pipeline builders ─────────────────────────────────────────────────────────

def _build_file_source(args):
    ctx      = limef.MediaFileContext(args.file, SLOT)
    ctx.fps  = args.fps
    ctx.loop = 0 if args.loop else -1
    source   = limef.MediaFileThread("source", ctx)
    return source, None   # (source_thread, encoder_ff_or_None)


def _build_rtsp_source(args):
    ctx    = limef.LiveStreamContext(args.rtsp, SLOT)
    source = limef.LiveStreamThread("source", ctx)
    return source, None


def _build_usb_source(args):
    cam_ctx               = limef.USBCameraContext(args.usb, SLOT)
    cam_ctx.width         = args.width
    cam_ctx.height        = args.height
    cam_ctx.fps           = args.fps

    enc_params              = limef.FFmpegEncoderParams()
    enc_params.codec_id     = limef.AV_CODEC_ID_H264
    enc_params.bitrate      = args.bitrate
    enc_params.max_b_frames = 0
    enc_params.gop_size     = max(1, args.fps // 2)
    enc_params.profile      = "baseline"   # Constrained Baseline — required for Firefox WebRTC
    if args.hw_accel:
        cam_ctx.output_format = limef.AV_PIX_FMT_NV12
        enc_params.hw_accel   = limef.HWACCEL_CUDA
        enc_params.preset     = "p1"
        enc_params.tune       = "ull"
    else:
        cam_ctx.output_format = limef.AV_PIX_FMT_YUV420P

    source  = limef.USBCameraThread("source", cam_ctx)
    encoder = limef.EncodingFrameFilter("encoder", enc_params)
    return source, encoder


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Limef browser streaming demo (WebRTC + nginx)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--rtsp",  metavar="URL",  help="RTSP stream URL")
    src.add_argument("--file",  metavar="PATH", help="local media file")
    src.add_argument("--usb",   metavar="DEV",  help="V4L2 device, e.g. /dev/video0")

    p.add_argument("--webrtc-port", type=int, default=8090, metavar="PORT",
                   help="WebRTC signaling HTTP port (loopback only)")
    p.add_argument("--http-port",   type=int, default=8091, metavar="PORT",
                   help="nginx external HTTP port")
    p.add_argument("--uuid",        default="stream", metavar="UUID",
                   help="stream UUID (exposed as /<uuid> on the WebRTC server)")
    p.add_argument("--fps",         type=int, default=25, metavar="FPS",
                   help="playback speed (file) / capture rate (USB)")
    p.add_argument("--loop",        action="store_true",
                   help="loop file source")
    p.add_argument("--width",       type=int, default=640,
                   help="USB capture width")
    p.add_argument("--height",      type=int, default=480,
                   help="USB capture height")
    p.add_argument("--bitrate",     type=int, default=4_000_000,
                   help="USB encoder bitrate in bps")
    p.add_argument("--hw-accel",    action="store_true",
                   help="USB: use NVENC H.264 instead of libx264")
    p.add_argument("--dump",        action="store_true",
                   help="log every RTP packet leaving the muxer (debug)")
    p.add_argument("--debug",       action="store_true",
                   help="set WebRTCServerThread log level to DEBUG (shows raw SDP exchange)")
    args = p.parse_args()

    stream_uuid = f"/{args.uuid}"

    # ── build source ───────────────────────────────────────────────────────────
    if args.file:
        source, encoder = _build_file_source(args)
    elif args.rtsp:
        source, encoder = _build_rtsp_source(args)
    else:
        source, encoder = _build_usb_source(args)

    # ── build RTP muxer + WebRTC server ───────────────────────────────────────
    rtp    = limef.RTPMuxerFrameFilter("rtp")
    wrtc   = limef.WebRTCServerThread("webrtc", port=args.webrtc_port)
    dump   = limef.DumpFrameFilter("dump") if args.dump else None

    if encoder:
        source.cc(encoder).cc(rtp)
    else:
        source.cc(rtp)
    if dump:
        rtp.cc(dump).cc(wrtc.getInput())
    else:
        rtp.cc(wrtc.getInput())

    # ── start WebRTC server ────────────────────────────────────────────────────
    if args.debug:
        wrtc.setLogLevel(limef.LOG_LEVEL_DEBUG)
    wrtc.start()
    wrtc.expose(SLOT, stream_uuid)

    # ── start nginx ────────────────────────────────────────────────────────────
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="limef-webrtc-demo-"))
    nginx_proc = _start_nginx(tmpdir, args.http_port)

    # ── start source ───────────────────────────────────────────────────────────
    source.start()

    player_url = f"http://localhost:{args.http_port}/?uuid={args.uuid}&wport={args.webrtc_port}"
    print("=================================")
    print("  WebRTC HTML Demo")
    print("=================================")
    if args.file:
        print(f"Source:      file  {args.file}")
    elif args.rtsp:
        print(f"Source:      rtsp  {args.rtsp}")
    else:
        codec = "H.264/NVENC" if args.hw_accel else "H.264/libx264"
        print(f"Source:      usb   {args.usb}  {args.width}x{args.height}@{args.fps}  [{codec}]")
    print(f"Stream UUID: {stream_uuid}")
    print(f"WebRTC port: {args.webrtc_port}  (loopback)")
    print(f"HTTP port:   {args.http_port}")
    print(f"Player:      {player_url}")
    print("=================================")
    print("Press Ctrl+C to stop\n")

    # ── main loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down...")

    # ── stop (downstream first) ───────────────────────────────────────────────
    _stop_nginx(nginx_proc)
    wrtc.stop()
    source.stop()

    print("Done.")


if __name__ == "__main__":
    main()
