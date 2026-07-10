#!/usr/bin/env python3
"""
apps/python/jetson_webrtc.py

MediaFile → FFmpeg SW decode → [HW|SW] encode → WebRTC → browser.

No USB camera required — source is a local media file.

Encoder options (--encoder):
  v4l2   Jetson V4L2 NVENC H.264  [requires /dev/v4l2-nvenc]
  cuda   FFmpeg CUDA NVENC H.264  [requires NVIDIA GPU]
  sw     FFmpeg software VP8      [any host, no GPU needed]

Pipelines
---------
Sources and encoders are independent classes — pick an encoder and wire:

  FileSource  → V4L2Encoder  ─┐
              → CUDAEncoder   ├─ WebRTCMuxerFrameFilter → WebRTCServerThread
              → VP8Encoder   ─┘

FileSource:
  MediaFileThread → MuteAudioFrameFilter → DecodingFrameFilter

V4L2Encoder (Jetson NVENC):
  DecodedUploadFrameFilter → CUDAScaleFrameFilter(NV12) → EncodingFrameFilter(V4L2NVEncoder H.264 baseline)

CUDAEncoder (FFmpeg CUDA NVENC):
  DecodedUploadFrameFilter → CUDAScaleFrameFilter(NV12) → EncodingFrameFilter(NVENC H.264 baseline)

VP8Encoder (software):
  SwScaleFrameFilter(YUV420P) → EncodingFrameFilter(libvpx VP8)

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


# ── source ─────────────────────────────────────────────────────────────────────
# Exposes:
#   .chain  — tail framefilter producing DecodedFrames; connect an encoder here
#   .start() / .stop()

class FileSource:
    """MediaFileThread → MuteAudioFrameFilter → DecodingFrameFilter.

    Audio is stripped before decoding so video-only decoded frames reach the
    encoder.  Wire an encoder's .input downstream of .chain.
    """

    def __init__(self, args):
        self._ctx      = limef.MediaFileContext(args.file, SLOT)
        self._ctx.fps  = args.fps
        self._ctx.loop = 0 if args.loop else -1

        self._source = limef.MediaFileThread("source", self._ctx)
        self._mute   = limef.MuteAudioFrameFilter("mute")
        self._decode = limef.DecodingFrameFilter("decode")
        self._source.cc(self._mute).cc(self._decode)
        self.chain = self._decode

    def start(self): self._source.start()
    def stop(self):  self._source.stop()


# ── encoders ───────────────────────────────────────────────────────────────────
# All encoders expose:
#   .input      — first framefilter; connect source.chain here
#   .chain      — tail framefilter (EncodingFrameFilter); connect muxer here
#   .codec_desc — human-readable description string

class V4L2Encoder:
    """DecodedUploadFrameFilter → CUDAScaleFrameFilter(NV12) → EncodingFrameFilter(V4L2 NVENC H.264).

    Jetson-specific.  Requires /dev/v4l2-nvenc (Jetson Orin).
    """

    def __init__(self, args):
        self._ep              = limef.V4L2NVEncoderParams()
        self._ep.device       = args.device
        self._ep.codec_fourcc = limef.V4L2_PIX_FMT_H264
        self._ep.bitrate      = args.bitrate
        self._ep.gop_size     = args.gop_size
        self._ep.global_header = False
        self._ep.h264_profile = limef.V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE
        self._ep.h264_level   = limef.V4L2_MPEG_VIDEO_H264_LEVEL_4_1

        self._upload  = limef.DecodedUploadFrameFilter("upload")
        self._to_nv12 = limef.CUDAScaleFrameFilter("to-nv12")
        self._encode  = limef.EncodingFrameFilter("encoder", self._ep)
        self._upload.cc(self._to_nv12).cc(self._encode)
        self.input      = self._upload
        self.chain      = self._encode
        self.codec_desc = f"V4L2 NVENC H.264  device={args.device}  profile=baseline  level=4.1"


class CUDAEncoder:
    """GPU pipeline: DecodedUploadFrameFilter → CUDAScaleFrameFilter(NV12) → CudaToFFmpegFrameFilter → EncodingFrameFilter(h264_nvenc).

    Requires an NVIDIA GPU.  Baseline profile for Firefox compatibility.
    Color conversion (YUV420P→NV12) happens on GPU; CudaToFFmpegFrameFilter
    wraps the raw CUDA frame in an AVHWFramesContext so NVENC can consume it.
    delay=0 + rc=cbr produce smooth, evenly-spaced packet output.
    """

    def __init__(self, args):
        self._ep               = limef.FFmpegEncoderParams()
        self._ep.codec_id      = limef.AV_CODEC_ID_H264
        self._ep.hw_accel      = limef.HWACCEL_CUDA
        self._ep.bitrate       = args.bitrate
        self._ep.gop_size      = args.gop_size
        self._ep.max_b_frames  = 0
        self._ep.preset        = "p1"
        self._ep.tune          = "ull"
        self._ep.profile       = "baseline"
        self._ep.global_header = False
        self._ep.options       = {"delay": "0", "rc": "cbr"}

        self._upload    = limef.DecodedUploadFrameFilter("upload")
        self._to_nv12   = limef.CUDAScaleFrameFilter("to-nv12")
        self._to_ffmpeg = limef.CudaToFFmpegFrameFilter("to-ffmpeg")
        self._encode    = limef.EncodingFrameFilter("encoder", self._ep)
        self._upload.cc(self._to_nv12).cc(self._to_ffmpeg).cc(self._encode)
        self.input      = self._upload
        self.chain      = self._encode
        self.codec_desc = "FFmpeg CUDA NVENC H.264  profile=baseline"


class VP8Encoder:
    """SwScaleFrameFilter(YUV420P) → EncodingFrameFilter(VP8/libvpx).

    Software encoder — works on any host without a GPU.
    VP8 is mandatory per RFC 8834 and works in every WebRTC-capable browser.
    """

    def __init__(self, args):
        self._ep              = limef.FFmpegEncoderParams()
        self._ep.codec_id     = limef.AV_CODEC_ID_VP8
        self._ep.bitrate      = args.bitrate
        self._ep.gop_size     = args.gop_size
        self._ep.max_b_frames = 0

        self._swscale = limef.SwScaleFrameFilter("swscale", limef.AV_PIX_FMT_YUV420P)
        self._encode  = limef.EncodingFrameFilter("encoder", self._ep)
        self._swscale.cc(self._encode)
        self.input      = self._swscale
        self.chain      = self._encode
        self.codec_desc = "FFmpeg SW VP8 (libvpx)"


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(
        description="Limef Jetson: MediaFile → SW decode → [HW|SW] encode → WebRTC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file",        required=True, metavar="PATH",
                   help="Input video file")
    p.add_argument("--encoder",     default="v4l2",
                   choices=["v4l2", "cuda", "sw"],
                   help="v4l2=Jetson NVENC  cuda=FFmpeg CUDA NVENC  sw=libvpx VP8")
    p.add_argument("--fps",         type=int, default=-1,
                   help="Playback fps (-1 = natural speed from file)")
    p.add_argument("--loop",        action="store_true",
                   help="Loop file source")
    p.add_argument("--bitrate",     type=int, default=2_000_000,
                   help="Encoder target bitrate in bits/sec")
    p.add_argument("--gop-size",    type=int, default=30,
                   help="Keyframe interval in frames")
    p.add_argument("--device",      default="/dev/v4l2-nvenc", metavar="DEV",
                   help="V4L2 NVENC device node (--encoder v4l2 only)")
    p.add_argument("--webrtc-port", type=int, default=9090, metavar="PORT",
                   help="WebRTC signaling HTTP port")
    p.add_argument("--http-port",   type=int, default=9091, metavar="PORT",
                   help="nginx static-file HTTP port")
    p.add_argument("--uuid",        default="stream", metavar="UUID",
                   help="Stream UUID (exposed as /<uuid> on the WebRTC server)")
    p.add_argument("--pre-dump",    action="store_true",
                   help="Verbose dump before encoder (decoded frames)")
    p.add_argument("--post-dump",   action="store_true",
                   help="Verbose dump after encoder (encoded packets)")
    p.add_argument("--debug",       action="store_true",
                   help="Set WebRTCServerThread log level to DEBUG (raw SDP exchange)")
    p.add_argument("--stun",        default="", metavar="URL",
                   help="STUN server URL, e.g. stun:100.84.182.90:3478 (default: "
                        "disabled). Needed for Firefox over Tailscale: Firefox "
                        "filters 100.64.0.0/10 from ICE host candidates, so "
                        "without a STUN server reachable on the tailnet it never "
                        "offers its Tailscale address and ICE fails. The STUN "
                        "server itself must already be running at that address. "
                        "Use a literal IP or the full Tailscale FQDN (e.g. "
                        "host.tailXXXX.ts.net) — NOT the bare Tailscale hostname: "
                        "/etc/hosts resolves that to 127.0.1.1 before DNS search "
                        "domains ever apply, so ICE gathering silently times out. "
                        "This only sets up the server side — the browser page also "
                        "needs '&stun=true' in its URL (or '&stun=<url>' for a "
                        "custom one) to actually use it, see webrtc_html_demo/static/index.html.")
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

    # ── build source ────────────────────────────────────────────────────────────
    source = FileSource(args)

    # ── build encoder ───────────────────────────────────────────────────────────
    enc = args.encoder.lower()
    if enc == "v4l2":
        encoder = V4L2Encoder(args)
    elif enc == "cuda":
        encoder = CUDAEncoder(args)
    else:
        encoder = VP8Encoder(args)

    # ── pre/post dump filters (always in chain; verbose only when flag is set) ───
    pre_dump  = limef.DumpFrameFilter("pre-enc",  verbose=args.pre_dump)
    post_dump = limef.DumpFrameFilter("post-enc", verbose=args.post_dump)

    # ── wire source → pre_dump → encoder → post_dump ────────────────────────────
    source.chain.cc(pre_dump).cc(encoder.input)
    chain = encoder.chain.cc(post_dump)

    # ── WebRTC muxer + server ───────────────────────────────────────────────────
    rtp  = limef.WebRTCMuxerFrameFilter("webrtc-muxer")
    wrtc = limef.WebRTCServerThread("webrtc", port=wport,
                                    stack_size=200, fifo_size=400,
                                    stun_server=args.stun)
    chain.cc(rtp).cc(wrtc.getInput())

    # ── banner ──────────────────────────────────────────────────────────────────
    print("==============================================")
    print("  Jetson WebRTC: MediaFile → decode → encode → WebRTC")
    print("==============================================")
    print(f"File:         {args.file}")
    print(f"FPS:          {'natural' if args.fps == -1 else args.fps}")
    print(f"Loop:         {'yes' if args.loop else 'no'}")
    print(f"Encoder:      {encoder.codec_desc}")
    print(f"Bitrate:      {args.bitrate // 1000} kbps   gop={args.gop_size}")
    print(f"Stream UUID:  {stream_uuid}")
    print(f"WebRTC port:  {wport}")
    print(f"HTTP port:    {hport}")
    print(f"LAN IP:       {lan_ip}")
    print(f"STUN server:  {args.stun if args.stun else 'disabled'}")
    # The browser only uses the STUN server configured above (on the Jetson side)
    # if the page URL also carries &stun=true — see index.html. Reflect that in
    # every example URL below so it's not forgotten mid-session.
    stun_qs = "&stun=true" if args.stun else ""
    if args.stun:
        print("              (browser page needs '&stun=true' in its URL to use this — see below)")
    print()
    print("Open in browser on THIS machine:")
    print(f"  http://localhost:{hport}/?{lan_ip}:{wport}&uuid={args.uuid}{stun_qs}")
    print()
    print("Open from a REMOTE machine (HTML served from this host):")
    print(f"  http://{lan_ip}:{hport}/?{lan_ip}:{wport}&uuid={args.uuid}{stun_qs}")
    print()
    print("Open from a REMOTE machine (HTML served elsewhere, stream from here):")
    print(f"  http://<static-host>:<port>/?{lan_ip}:{wport}&uuid={args.uuid}{stun_qs}")
    print("  (replace LAN IP with Tailscale hostname if using VPN)")
    print()
    print("To serve HTML from your local machine instead of this host:")
    print(f"  python3 -m http.server {hport} --directory {_STATIC_DIR}")
    print(f"  then open: http://localhost:{hport}/?{lan_ip}:{wport}&uuid={args.uuid}{stun_qs}")
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

    # ── start source ────────────────────────────────────────────────────────────
    source.start()

    # ── run until Ctrl+C ────────────────────────────────────────────────────────
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down...")

    # ── stop (nginx + WebRTC first, then source) ─────────────────────────────────
    _stop_nginx(nginx_proc)
    wrtc.stop()
    source.stop()
    print("Done.")


if __name__ == "__main__":
    main()
