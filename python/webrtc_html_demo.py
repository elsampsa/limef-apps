#!/usr/bin/env python3
"""
apps/python/webrtc_html_demo.py

Browser streaming demo: RTSP / file / USB camera → WebRTC → browser.

Codec validation
----------------
WebRTCMuxerFrameFilter contains a CodecAssertFrameFilter as its first stage.
If the incoming stream carries an incompatible codec pair, CodecAssert prints a
warning and inutilizes the filter chain — no crash, no silent corruption.  The
upstream pipeline is responsible for providing the right codec; no conversion is
performed here.

WebRTC codec constraints are extremely strict (RFC 8834).  Allowed pairs:

  Video  | Audio           | Notes
  -------|-----------------|-----------------------------------------------
  H264   | Opus/PCMU/PCMA  | Baseline profile recommended for Firefox
  H264   | —               | Video-only
  VP8    | Opus/PCMU/PCMA  | Mandatory per RFC 8834
  VP8    | —               | Video-only
  VP9    | Opus            | Widely supported
  VP9    | —               | Video-only
  AV1    | Opus            | Chrome 90+, Firefox 93+
  AV1    | —               | Video-only

Not accepted: AAC, MP3, H.265 (not in the WebRTC spec; browsers reject them).
For file/RTSP sources the upstream stream must already use one of these codecs —
re-encoding is not performed.  Incompatible streams trip CodecAssert and inutilize
the chain.

Pipelines
---------
Sources and encoders are independent classes — pick one of each and wire them:

  FileSource      → MuteAudio → Decode  ─┐
                                          ├─ VP8Encoder  ─┐
  USBCameraSource ──────────────────────-─┘               ├─ WebRTCMuxerFrameFilter → WebRTCServerThread
                                          ├─ CUDAEncoder ─┘
                                         (chosen by --hw-accel)

USB + --hw-accel (H.264 NVENC):
  USBCameraThread
       → EncodingFrameFilter(NVENC H.264)
       → WebRTCMuxerFrameFilter
       → WebRTCServerThread  ...

USB default (VP8 libvpx, software):
  USBCameraThread
       → EncodingFrameFilter(libvpx VP8)
       → WebRTCMuxerFrameFilter
       → WebRTCServerThread  ...

Usage:
    python3 apps/python/webrtc_html_demo.py --file video.mp4
    python3 apps/python/webrtc_html_demo.py --file video.mp4 --hw-accel
    python3 apps/python/webrtc_html_demo.py --usb /dev/video0
    python3 apps/python/webrtc_html_demo.py --usb /dev/video0 --hw-accel

Options:
    --file         PATH  local media file (decoded + re-encoded)
    --usb          DEV   V4L2 device, e.g. /dev/video0
    --hw-accel           use NVENC H.264 encoder instead of libvpx VP8
    --webrtc-port        WebRTC signaling HTTP port (default 9090)
    --http-port          nginx static-file HTTP port (default 9091)
    --uuid               stream UUID (default 'stream', exposed as /stream)
    --fps                playback/capture rate (default 25)
    --width              USB capture width (default 640)
    --height             USB capture height (default 480)
    --bitrate            encoder bitrate in bps (default 4_000_000)

Press Ctrl+C to stop.
"""

import os
import sys
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


# ── sources ───────────────────────────────────────────────────────────────────
# All sources expose:
#   .chain  — tail framefilter producing DecodedFrames; connect an encoder here
#   .start() / .stop()

class FileSource:
    """MediaFileThread → MuteAudioFrameFilter → DecodingFrameFilter.

    Outputs decoded video frames (audio stripped).  Wire an encoder downstream.
    """

    def __init__(self, args):
        self._ctx      = limef.MediaFileContext(args.file, SLOT)
        self._ctx.fps  = args.fps
        self._ctx.loop = 0

        self._source = limef.MediaFileThread("source", self._ctx)
        self._mute   = limef.MuteAudioFrameFilter("mute")
        self._decode = limef.DecodingFrameFilter("decode")
        self._source.cc(self._mute).cc(self._decode)
        self.chain = self._decode

    def start(self): self._source.start()
    def stop(self):  self._source.stop()


class USBCameraSource:
    """USBCameraThread outputting raw DecodedFrames (YUYV422).

    Wire an encoder downstream; the encoder's SwScale handles format conversion.
    """

    def __init__(self, args):
        self._cam_ctx                = limef.USBCameraContext(args.usb, SLOT)
        self._cam_ctx.width          = args.width
        self._cam_ctx.height         = args.height
        self._cam_ctx.fps            = args.fps
        self._cam_ctx.capture_format = limef.AV_PIX_FMT_YUYV422

        self._thread = limef.USBCameraThread("source", self._cam_ctx)
        self.chain   = self._thread

    def start(self): self._thread.start()
    def stop(self):  self._thread.stop()


# ── encoders ──────────────────────────────────────────────────────────────────
# All encoders expose:
#   .input  — first framefilter (SwScaleFrameFilter); connect a source here
#   .chain  — tail framefilter (EncodingFrameFilter); connect the muxer here

class VP8Encoder:
    """SwScaleFrameFilter(YUV420P) → EncodingFrameFilter(VP8/libvpx).

    VP8 is mandatory per RFC 8834 and works in every WebRTC-capable browser.
    """

    def __init__(self, args):
        self._ep              = limef.FFmpegEncoderParams()
        self._ep.codec_id     = limef.AV_CODEC_ID_VP8
        self._ep.bitrate      = args.bitrate
        self._ep.max_b_frames = 0
        self._ep.gop_size     = max(1, args.fps // 2)

        self._swscale = limef.SwScaleFrameFilter("swscale", limef.AV_PIX_FMT_YUV420P)
        self._encode  = limef.EncodingFrameFilter("encode", self._ep)
        self._swscale.cc(self._encode)
        self.input = self._swscale
        self.chain = self._encode


class CUDAEncoder:
    """SwScaleFrameFilter(NV12) → EncodingFrameFilter(H.264 NVENC, baseline).

    Requires an NVIDIA GPU.  Baseline profile is used for Firefox compatibility.
    """

    def __init__(self, args):
        self._ep                   = limef.FFmpegEncoderParams()
        self._ep.codec_id          = limef.AV_CODEC_ID_H264
        self._ep.hw_accel          = limef.HWACCEL_CUDA
        self._ep.bitrate           = args.bitrate
        self._ep.max_b_frames      = 0
        self._ep.gop_size          = max(1, args.fps // 2)
        self._ep.preset            = "p1"
        self._ep.tune              = "ull"
        self._ep.profile           = "baseline"
        self._ep.global_header     = False
        self._ep.options           = {"delay": "0", "rc": "cbr"}

        self._swscale = limef.SwScaleFrameFilter("swscale", limef.AV_PIX_FMT_NV12)
        self._encode  = limef.EncodingFrameFilter("encode", self._ep)
        self._swscale.cc(self._encode)
        self.input = self._swscale
        self.chain = self._encode


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(
        description="Limef browser streaming demo (WebRTC + nginx)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file",  metavar="PATH", help="local media file")
    src.add_argument("--usb",   metavar="DEV",  help="V4L2 device, e.g. /dev/video0")

    p.add_argument("--webrtc-port", type=int, default=9090, metavar="PORT",
                   help="WebRTC signaling HTTP port (loopback only)")
    p.add_argument("--http-port",   type=int, default=9091, metavar="PORT",
                   help="nginx external HTTP port")
    p.add_argument("--uuid",        default="stream", metavar="UUID",
                   help="stream UUID (exposed as /<uuid> on the WebRTC server)")
    p.add_argument("--fps",         type=int, default=25, metavar="FPS",
                   help="playback speed (file) / capture rate (USB)")
    p.add_argument("--width",       type=int, default=640,
                   help="USB capture width")
    p.add_argument("--height",      type=int, default=480,
                   help="USB capture height")
    p.add_argument("--bitrate",     type=int, default=4_000_000,
                   help="USB encoder bitrate in bps")
    p.add_argument("--hw-accel",    action="store_true",
                   help="use NVENC H.264 encoder instead of libvpx VP8 (requires GPU)")
    p.add_argument("--packetdump",   action="store_true",
                   help="log every packet leaving the encoder, before the WebRTC muxer (debug)")
    p.add_argument("--dump",        action="store_true",
                   help="log every RTP packet leaving the muxer (debug)")
    p.add_argument("--debug",       action="store_true",
                   help="set WebRTCServerThread log level to DEBUG (shows raw SDP exchange)")
    args = p.parse_args()

    stream_uuid = f"/{args.uuid}"

    # ── build source ───────────────────────────────────────────────────────────
    if args.file:
        source = FileSource(args)
    else:
        source = USBCameraSource(args)

    # ── build encoder ──────────────────────────────────────────────────────────
    encoder = CUDAEncoder(args) if args.hw_accel else VP8Encoder(args)

    # ── wire source → encoder ──────────────────────────────────────────────────
    source.chain.cc(encoder.input)

    # ── build RTP muxer + WebRTC server ───────────────────────────────────────
    rtp        = limef.WebRTCMuxerFrameFilter("webrtc_muxer")
    wrtc       = limef.WebRTCServerThread("webrtc", port=args.webrtc_port,
                                          stack_size=200, fifo_size=400)
    packetdump = limef.DumpFrameFilter("packetdump") if args.packetdump else None
    dump       = limef.DumpFrameFilter("dump") if args.dump else None

    chain = encoder.chain
    chain = chain.cc(packetdump) if packetdump else chain
    chain = chain.cc(rtp)
    if dump:
        rtp.cc(dump).cc(wrtc.getInput())
    else:
        rtp.cc(wrtc.getInput())

    enc_name   = "H.264/NVENC" if args.hw_accel else "VP8/libvpx"
    player_url = f"http://localhost:{args.http_port}/?uuid={args.uuid}&wport={args.webrtc_port}"
    print("=================================")
    print("  WebRTC HTML Demo")
    print("=================================")
    if args.file:
        print(f"Source:      file  {args.file}")
    else:
        print(f"Source:      usb   {args.usb}  {args.width}x{args.height}@{args.fps}")
    print(f"Encoder:     {enc_name}")
    print(f"Stream UUID: {stream_uuid}")
    print(f"WebRTC port: {args.webrtc_port}  (loopback)")
    print(f"HTTP port:   {args.http_port}")
    print(f"Player:      {player_url}")
    print("=================================")
    print("Press Ctrl+C to stop\n")

    # ── start WebRTC server ────────────────────────────────────────────────────
    if args.debug:
        wrtc.setLogLevel(limef.LOG_LEVEL_DEBUG)
    # Forward browser keyframe requests (PLI) to the encoder. Single-slot demo,
    # so no slot→encoder lookup is needed — a multi-stream app would keep a
    # {slot: EncodingFrameFilter} dict here instead.
    wrtc.onKeyframeRequested(lambda slot: encoder.chain.requestKeyFrame())
    wrtc.start()
    wrtc.expose(SLOT, stream_uuid)

    # ── start nginx ────────────────────────────────────────────────────────────
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="limef-webrtc-demo-"))
    nginx_proc = _start_nginx(tmpdir, args.http_port)

    # ── start source ───────────────────────────────────────────────────────────
    source.start()

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
