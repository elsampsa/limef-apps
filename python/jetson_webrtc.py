#!/usr/bin/env python3
"""
apps/python/jetson_webrtc.py

MediaFile → FFmpeg SW decode → [HW|SW] encode → WebRTC → browser.

No USB camera required — source is a local media file.

Encoder options (--encoder):
  v4l2   Jetson V4L2 NVENC H.264  [requires /dev/v4l2-nvenc]
  cuda   FFmpeg CUDA NVENC H.264  [requires NVIDIA GPU]
  sw     FFmpeg software VP8      [any host, no GPU needed]

Pipeline (v4l2 / cuda):
  MediaFileThread
    → DecodingFrameFilter        (FFmpeg SW decode, CPU YUV420P)
    → DecodedUploadFrameFilter   (CPU YUV420P → CUDA YUV420P)
    → CUDAScaleFrameFilter       (YUV420P CUDA → NV12 CUDA)
    → EncodingFrameFilter        (V4L2NVEncoder or FFmpeg CUDA NVENC)
    → WebRTCMuxerFrameFilter
    → WebRTCServerThread

Pipeline (sw):
  MediaFileThread
    → DecodingFrameFilter
    → SwScaleFrameFilter         (ensure YUV420P on CPU)
    → EncodingFrameFilter        (libvpx VP8)
    → WebRTCMuxerFrameFilter
    → WebRTCServerThread

Frontend:
  nginx serves webrtc_html_demo/static/index.html on --http-port.
  The page accepts a bare positional server spec in the URL:
    http://<anywhere>:<http-port>/?<jetson-host>
    http://<anywhere>:<http-port>/?<jetson-host>:<webrtc-port>
    http://<anywhere>:<http-port>/?<jetson-host>&uuid=mystream
  or classic named params:
    http://<anywhere>:<http-port>/?server=<jetson-host>&wport=9090&uuid=stream

Usage:
    python3 apps/python/jetson_webrtc.py --file fixtures/jontxu.mp4
    python3 apps/python/jetson_webrtc.py --file fixtures/jontxu.mp4 --encoder v4l2
    python3 apps/python/jetson_webrtc.py --file fixtures/jontxu.mp4 --encoder cuda
    python3 apps/python/jetson_webrtc.py --file fixtures/jontxu.mp4 --encoder sw

Press Ctrl+C to stop.
"""

import os
import sys
import time
import shlex
import socket
import argparse
import textwrap
import tempfile
import subprocess
import pathlib

import limef

sys.stdout.reconfigure(line_buffering=True)

SLOT = 1

_SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
_STATIC_DIR = _SCRIPT_DIR / "webrtc_html_demo" / "static"


def _lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


# ── nginx ──────────────────────────────────────────────────────────────────────

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
            f"Check {tmpdir}/error.log"
        )
    return proc


def _stop_nginx(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ── pipeline ───────────────────────────────────────────────────────────────────

class Pipeline:
    """Owns every filterchain object so Python GC never destroys a live chain node."""

    def __init__(self, args):
        file_ctx      = limef.MediaFileContext(args.file, SLOT)
        file_ctx.fps  = args.fps
        file_ctx.loop = 0 if args.loop else -1

        self._src       = limef.MediaFileThread("src", file_ctx)
        self._sw_dec    = limef.DecodingFrameFilter("sw-dec")
        self._pre_dump  = limef.DumpFrameFilter("pre-enc",  verbose=args.pre_dump)
        self._post_dump = limef.DumpFrameFilter("post-enc", verbose=args.post_dump)

        enc = args.encoder.lower()

        if enc == "v4l2":
            self._upload  = limef.DecodedUploadFrameFilter("upload")
            self._to_nv12 = limef.CUDAScaleFrameFilter("to-nv12")
            ep                  = limef.V4L2NVEncoderParams()
            ep.device           = args.device
            ep.codec_fourcc     = limef.V4L2_PIX_FMT_H264
            ep.bitrate          = args.bitrate
            ep.gop_size         = args.gop_size
            ep.global_header    = False
            ep.h264_profile     = limef.V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE
            ep.h264_level       = limef.V4L2_MPEG_VIDEO_H264_LEVEL_4_1
            self._encoder = limef.EncodingFrameFilter("encoder", ep)
            (self._src
                .cc(self._sw_dec)
                .cc(self._upload)
                .cc(self._to_nv12)
                .cc(self._pre_dump)
                .cc(self._encoder)
                .cc(self._post_dump))
            self.codec_desc = f"V4L2 NVENC H.264  device={args.device}  profile=baseline  level=4.1"

        elif enc == "cuda":
            self._upload  = limef.DecodedUploadFrameFilter("upload")
            self._to_nv12 = limef.CUDAScaleFrameFilter("to-nv12")
            ep                  = limef.FFmpegEncoderParams()
            ep.codec_id         = limef.AV_CODEC_ID_H264
            ep.hw_accel         = limef.HWACCEL_CUDA
            ep.bitrate          = args.bitrate
            ep.gop_size         = args.gop_size
            ep.max_b_frames     = 0
            ep.preset           = "p1"
            ep.tune             = "ull"
            ep.profile          = "baseline"
            ep.global_header    = False
            self._encoder = limef.EncodingFrameFilter("encoder", ep)
            (self._src
                .cc(self._sw_dec)
                .cc(self._upload)
                .cc(self._to_nv12)
                .cc(self._pre_dump)
                .cc(self._encoder)
                .cc(self._post_dump))
            self.codec_desc = "FFmpeg CUDA NVENC H.264  profile=baseline"

        else:  # sw
            self._swscale = limef.SwScaleFrameFilter("swscale", limef.AV_PIX_FMT_YUV420P)
            ep                  = limef.FFmpegEncoderParams()
            ep.codec_id         = limef.AV_CODEC_ID_VP8
            ep.bitrate          = args.bitrate
            ep.gop_size         = args.gop_size
            ep.max_b_frames     = 0
            self._encoder = limef.EncodingFrameFilter("encoder", ep)
            (self._src
                .cc(self._sw_dec)
                .cc(self._swscale)
                .cc(self._pre_dump)
                .cc(self._encoder)
                .cc(self._post_dump))
            self.codec_desc = "FFmpeg SW VP8 (libvpx)"

        self.chain = self._post_dump  # tail: connect WebRTC muxer here

    def start(self): self._src.start()
    def stop(self):  self._src.stop()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Limef Jetson: MediaFile → SW decode → [HW|SW] encode → WebRTC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file",         required=True, metavar="PATH",
                   help="Input video file")
    p.add_argument("--encoder",      default="v4l2",
                   choices=["v4l2", "cuda", "sw"],
                   help="v4l2=Jetson NVENC  cuda=FFmpeg CUDA NVENC  sw=libvpx VP8")
    p.add_argument("--fps",          type=int, default=-1,
                   help="Playback fps (-1 = natural speed from file)")
    p.add_argument("--loop",         action="store_true",
                   help="Loop file source")
    p.add_argument("--bitrate",      type=int, default=2_000_000,
                   help="Encoder target bitrate in bits/sec")
    p.add_argument("--gop-size",     type=int, default=30,
                   help="Keyframe interval in frames")
    p.add_argument("--device",       default="/dev/v4l2-nvenc", metavar="DEV",
                   help="V4L2 NVENC device node (--encoder v4l2 only)")
    p.add_argument("--webrtc-port",  type=int, default=9090, metavar="PORT",
                   help="WebRTC signaling HTTP port")
    p.add_argument("--http-port",    type=int, default=9091, metavar="PORT",
                   help="nginx static-file HTTP port")
    p.add_argument("--uuid",         default="stream", metavar="UUID",
                   help="Stream UUID (exposed as /<uuid> on the WebRTC server)")
    p.add_argument("--pre-dump",     action="store_true",
                   help="Verbose dump before encoder (decoded frames)")
    p.add_argument("--post-dump",    action="store_true",
                   help="Verbose dump after encoder (encoded packets)")
    p.add_argument("--debug",        action="store_true",
                   help="Set WebRTCServerThread log level to DEBUG (raw SDP exchange)")
    args = p.parse_args()

    if not os.path.exists(args.file):
        print(f"ERROR: file not found: {args.file}")
        sys.exit(1)

    if args.encoder == "v4l2" and not os.path.exists(args.device):
        print(f"ERROR: V4L2 NVENC device not found: {args.device}")
        print("  Is this a Jetson Orin?  Try --encoder cuda or --encoder sw")
        sys.exit(1)

    lan_ip      = _lan_ip()
    stream_uuid = f"/{args.uuid}"
    wport       = args.webrtc_port
    hport       = args.http_port

    # ── build pipeline ──────────────────────────────────────────────────────────
    pipeline = Pipeline(args)

    # ── WebRTC muxer + server ───────────────────────────────────────────────────
    rtp  = limef.WebRTCMuxerFrameFilter("webrtc-muxer")
    wrtc = limef.WebRTCServerThread("webrtc", port=wport)
    pipeline.chain.cc(rtp).cc(wrtc.getInput())

    # ── banner ──────────────────────────────────────────────────────────────────
    print("==============================================")
    print("  Jetson WebRTC: MediaFile → decode → encode → WebRTC")
    print("==============================================")
    print(f"File:         {args.file}")
    print(f"FPS:          {'natural' if args.fps == -1 else args.fps}")
    print(f"Loop:         {'yes' if args.loop else 'no'}")
    print(f"Encoder:      {pipeline.codec_desc}")
    print(f"Bitrate:      {args.bitrate // 1000} kbps   gop={args.gop_size}")
    print(f"Stream UUID:  {stream_uuid}")
    print(f"WebRTC port:  {wport}")
    print(f"HTTP port:    {hport}")
    print(f"LAN IP:       {lan_ip}")
    print()
    print("Open in browser on THIS machine:")
    print(f"  http://localhost:{hport}/?{lan_ip}:{wport}&uuid={args.uuid}")
    print()
    print("Open from a REMOTE machine (HTML served from this host):")
    print(f"  http://{lan_ip}:{hport}/?{lan_ip}:{wport}&uuid={args.uuid}")
    print()
    print("Open from a REMOTE machine (HTML served elsewhere, stream from here):")
    print(f"  http://<static-host>:<port>/?{lan_ip}:{wport}&uuid={args.uuid}")
    print("  (replace LAN IP with Tailscale hostname if using VPN)")
    print("==============================================")
    print("Press Ctrl+C to stop\n")

    # ── start WebRTC server ─────────────────────────────────────────────────────
    if args.debug:
        wrtc.setLogLevel(limef.LOG_LEVEL_DEBUG)
    wrtc.start()
    wrtc.expose(SLOT, stream_uuid)

    # ── start nginx ─────────────────────────────────────────────────────────────
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="limef-jetson-webrtc-"))
    nginx_proc = _start_nginx(tmpdir, hport)

    # ── start file source ───────────────────────────────────────────────────────
    pipeline.start()

    # ── run until Ctrl+C ────────────────────────────────────────────────────────
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down...")

    # ── stop (nginx + WebRTC first, then source) ────────────────────────────────
    _stop_nginx(nginx_proc)
    wrtc.stop()
    pipeline.stop()
    print("Done.")


if __name__ == "__main__":
    main()
