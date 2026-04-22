#!/usr/bin/env python3
"""
apps/python/ws_html_demo.py

Browser streaming demo: RTSP / file / USB camera → WebSocket → browser MSE.

Source codec requirements
--------------------------
File / RTSP:
    The source must carry an fMP4-compatible encoded stream (H.264 / AVC).
    Encoded packets pass directly into FMP4FrameFilter without re-encoding.

USB camera:
    The camera outputs raw decoded frames, so an encoder is required.  Two options:

    --hw-accel  H.264 via NVENC (hardware).  Requires an NVIDIA GPU.
                Uses FMP4FrameFilter → WebM container NOT needed.

    (default)   VP8 via libvpx (software).
                VP8 cannot be carried in fMP4 for browser MSE, so this path
                uses WebMFrameFilter → WebM container instead.

Pipelines
---------
File / RTSP (H.264 → fMP4):
  [LiveStreamThread | MediaFileThread]
       → FMP4FrameFilter
       → WebSocketServerThread          ws://127.0.0.1:WS_PORT
       → nginx (reverse proxy + static HTML)

USB + --hw-accel (H.264 NVENC → fMP4):
  USBCameraThread
       → EncodingFrameFilter(NVENC H.264)
       → FMP4FrameFilter
       → WebSocketServerThread  ...

USB (VP8 software → WebM):
  USBCameraThread
       → EncodingFrameFilter(libvpx VP8)
       → WebMFrameFilter
       → WebSocketServerThread  ...

Usage:
    python3 apps/python/ws_html_demo.py --file video.mp4
    python3 apps/python/ws_html_demo.py --rtsp rtsp://user:pass@cam/stream
    python3 apps/python/ws_html_demo.py --usb /dev/video0 --hw-accel
    python3 apps/python/ws_html_demo.py --usb /dev/video0

Options:
    --rtsp   URL   RTSP stream URL (H.264)
    --file   PATH  local media file (H.264)
    --usb    DEV   V4L2 device (default /dev/video0)
    --ws-port      local WebSocket port (default 18080)
    --http-port    nginx HTTP port (default 8090)
    --uuid         stream UUID (default 'stream')
    --token        access token (default 'demo')
    --fps          playback/capture rate for file/USB (default 25)
    --loop         loop file source
    --width        USB capture width (default 640)
    --height       USB capture height (default 480)
    --bitrate      USB encoder bitrate in bps (default 4_000_000)
    --hw-accel     USB only: use NVENC H.264 instead of libvpx VP8

Press Ctrl+C to stop.
"""

import os
import time
import shlex
import signal
import argparse
import textwrap
import tempfile
import subprocess
import pathlib

import limef

SLOT = 1

_SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
_STATIC_DIR = _SCRIPT_DIR / "ws_html_demo" / "static"


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

            location /ws {{
                proxy_pass         http://127.0.0.1:{WS_PORT};
                proxy_http_version 1.1;
                proxy_set_header   Upgrade    $http_upgrade;
                proxy_set_header   Connection "upgrade";
                proxy_read_timeout 3600s;
            }}

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


def _start_nginx(tmpdir: pathlib.Path, ws_port: int, http_port: int) -> subprocess.Popen:
    conf = _NGINX_CONF_TEMPLATE.format(
        USER=os.environ["USER"],
        TMPDIR=tmpdir,
        WS_PORT=ws_port,
        HTTP_PORT=http_port,
        STATIC_DIR=_STATIC_DIR,
    )
    conf_path = tmpdir / "nginx.conf"
    conf_path.write_text(conf)

    # kill any stale nginx from a previous dirty exit
    subprocess.run(["killall", "-9", "nginx"], capture_output=True)
    time.sleep(0.3)

    proc = subprocess.Popen(shlex.split(f"nginx -p {tmpdir} -c {conf_path}"))
    time.sleep(0.5)  # let nginx bind the port
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
    ctx         = limef.LiveStreamContext(args.rtsp, SLOT)
    source      = limef.LiveStreamThread("source", ctx)
    return source, None


def _build_usb_source(args):
    cam_ctx               = limef.USBCameraContext(args.usb, SLOT)
    cam_ctx.width         = args.width
    cam_ctx.height        = args.height
    cam_ctx.fps           = args.fps

    enc_params              = limef.FFmpegEncoderParams()
    enc_params.bitrate      = args.bitrate
    enc_params.max_b_frames = 0
    enc_params.gop_size     = max(1, args.fps // 2)
    if args.hw_accel:
        # H.264 via NVENC — requires NVIDIA GPU.
        cam_ctx.output_format = limef.AV_PIX_FMT_NV12
        enc_params.codec_id   = limef.AV_CODEC_ID_H264
        enc_params.hw_accel   = limef.HWACCEL_CUDA
        enc_params.preset     = "p1"
        enc_params.tune       = "ull"
    else:
        # VP8 via libvpx — software encoding.  YUV420P is libvpx's native format.
        cam_ctx.output_format = limef.AV_PIX_FMT_YUV420P
        enc_params.codec_id   = limef.AV_CODEC_ID_VP8

    # USBCameraThread copies cam_ctx by value — construct AFTER output_format is set.
    source  = limef.USBCameraThread("source", cam_ctx)
    encoder = limef.EncodingFrameFilter("encoder", enc_params)
    return source, encoder


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Limef browser streaming demo (fMP4 over WebSocket + nginx)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--rtsp",  metavar="URL",  help="RTSP stream URL")
    src.add_argument("--file",  metavar="PATH", help="local media file")
    src.add_argument("--usb",   metavar="DEV",  help="V4L2 device, e.g. /dev/video0")

    p.add_argument("--ws-port",   type=int, default=18080, metavar="PORT",
                   help="local WebSocket port (loopback only)")
    p.add_argument("--http-port", type=int, default=8090,  metavar="PORT",
                   help="nginx external HTTP port")
    p.add_argument("--uuid",      default="stream", metavar="UUID",
                   help="stream UUID")
    p.add_argument("--token",     default="demo",   metavar="TOKEN",
                   help="access token")
    p.add_argument("--fps",       type=int, default=25, metavar="FPS",
                   help="playback speed (file) / capture rate (USB)")
    p.add_argument("--loop",      action="store_true",
                   help="loop file source")
    p.add_argument("--width",     type=int, default=640,
                   help="USB capture width")
    p.add_argument("--height",    type=int, default=480,
                   help="USB capture height")
    p.add_argument("--bitrate",   type=int, default=4_000_000,
                   help="USB encoder bitrate in bps")
    p.add_argument("--hw-accel",  action="store_true",
                   help="USB: use NVENC instead of libx264")
    args = p.parse_args()

    # ── build source ───────────────────────────────────────────────────────────
    if args.file:
        source, encoder = _build_file_source(args)
    elif args.rtsp:
        source, encoder = _build_rtsp_source(args)
    else:
        source, encoder = _build_usb_source(args)

    # ── build muxer (fMP4 for H.264, WebM for VP8) ────────────────────────────
    use_webm = args.usb and not args.hw_accel
    if use_webm:
        muxer = limef.WebMFrameFilter("webm")
    else:
        muxer = limef.FMP4FrameFilter("fmp4")

    ws_ctx   = limef.FrameFifoContext(False, 32, 64)
    wsserver = limef.WebSocketServerThread("wsserver", ws_ctx)

    if encoder:
        source.cc(encoder).cc(muxer)
    else:
        source.cc(muxer)
    muxer.cc(wsserver.getInput())

    # ── start WebSocket server ─────────────────────────────────────────────────
    wsserver.start()
    wsserver.startServer(args.ws_port)
    wsserver.setSlotUUID(SLOT, args.uuid)
    wsserver.addStreamToken(args.token, [args.uuid])

    # ── start nginx ────────────────────────────────────────────────────────────
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="limef-ws-demo-"))
    nginx_proc = _start_nginx(tmpdir, args.ws_port, args.http_port)

    # ── open gate and start source ────────────────────────────────────────────
    muxer.open()
    source.start()

    player_url = (
        f"http://localhost:{args.http_port}/"
        f"?token={args.token}&stream={args.uuid}"
    )
    print("=================================")
    print("  WebSocket HTML Demo")
    print("=================================")
    if args.file:
        print(f"Source:    file  {args.file}")
    elif args.rtsp:
        print(f"Source:    rtsp  {args.rtsp}")
    else:
        codec = "VP8/WebM" if use_webm else "H.264/fMP4 (NVENC)"
        print(f"Source:    usb   {args.usb}  {args.width}x{args.height}@{args.fps}  [{codec}]")
    print(f"Container: {'WebM' if use_webm else 'fMP4'}")
    print(f"WS port:   {args.ws_port}  (loopback)")
    print(f"HTTP port: {args.http_port}")
    print(f"Player:    {player_url}")
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
    wsserver.stop()
    source.stop()

    print("Done.")


if __name__ == "__main__":
    main()
