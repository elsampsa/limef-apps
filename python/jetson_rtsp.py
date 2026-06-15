#!/usr/bin/env python3
"""
apps/python/jetson_rtsp.py

MediaFile → FFmpeg SW decode → CUDA upload → V4L2NVEncode (H.264) → RTSP
for Jetson Orin Nano (NVENC via /dev/v4l2-nvenc).

No USB camera required — source is a local media file read by MediaFileThread.

Pipeline:

    MediaFileThread           [encoded packets from file]
        → DecodingFrameFilter     (FFmpeg SW decoder, CPU YUV420P output)
        → DecodedUploadFrameFilter (CPU YUV420P → CUDA NV12, AV_PIX_FMT_CUDA)
        → DumpFrameFilter          (pre-encode, verbose with --pre-dump)
        → EncodingFrameFilter      (V4L2NVEncoder, H.264 via /dev/v4l2-nvenc)
        → DumpFrameFilter          (post-encode, verbose with --post-dump)
        → RTSPMuxerFrameFilter
        → RTSPServerThread

Usage:
    python3 apps/python/jetson_rtsp.py --file fixtures/jontxu.mp4

Then connect from another machine:
    ffplay rtsp://<jetson-ip>:8554/live/stream
    ffplay -rtsp_transport tcp rtsp://<jetson-ip>:8554/live/stream

Press Ctrl+C to stop.
"""

import os
import sys
import time
import socket
import argparse

import limef

sys.stdout.reconfigure(line_buffering=True)


def _lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main():
    p = argparse.ArgumentParser(
        description='Limef Jetson: MediaFile → SW decode → CUDA upload → V4L2NVEncode → RTSP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--file',       required=True, metavar='PATH',
                   help='Input video file')
    p.add_argument('--fps',        type=int, default=-1,
                   help='Playback fps (-1 = natural speed)')
    p.add_argument('--loop',       type=int, default=0,
                   help='Loop count (0 = infinite loop)')
    p.add_argument('--bitrate',    type=int, default=2_000_000,
                   help='NVENC target bitrate in bits/sec')
    p.add_argument('--gop-size',   type=int, default=30,
                   help='Keyframe interval (frames)')
    p.add_argument('--device',     default='/dev/v4l2-nvenc', metavar='DEV',
                   help='NVENC V4L2 device')
    p.add_argument('--port',       type=int, default=8554,
                   help='RTSP server port')
    p.add_argument('--url-tail',   default='/live/stream', metavar='PATH',
                   help='RTSP URL path')
    p.add_argument('--pre-dump',   action='store_true',
                   help='Verbose dump before the encoder (decoded CUDA frames)')
    p.add_argument('--post-dump',  action='store_true',
                   help='Verbose dump after the encoder (encoded H.264 packets)')
    args = p.parse_args()

    if not os.path.exists(args.file):
        print(f"ERROR: file not found: {args.file}")
        sys.exit(1)

    if not os.path.exists(args.device):
        print(f"ERROR: NVENC device not found: {args.device}")
        print("  Is this a Jetson Orin with NVENC enabled?")
        sys.exit(1)

    lan_ip   = _lan_ip()
    port     = args.port
    url_tail = args.url_tail
    SLOT     = 1

    # ── Encoder params ──────────────────────────────────────────────────────────
    enc_params              = limef.V4L2NVEncoderParams()
    enc_params.device       = args.device
    enc_params.codec_fourcc = limef.V4L2_PIX_FMT_H264
    enc_params.bitrate      = args.bitrate
    enc_params.gop_size     = args.gop_size
    # enc_params.global_header = False # test: repeat sps and pps
    enc_params.global_header = True # carries sps and pps only in CodecFrame

    # ── Media file source ───────────────────────────────────────────────────────
    file_ctx      = limef.MediaFileContext(args.file, SLOT)
    file_ctx.fps  = args.fps
    file_ctx.loop = args.loop

    # ── Build pipeline objects ──────────────────────────────────────────────────
    src      = limef.MediaFileThread('src', file_ctx)
    sw_dec   = limef.DecodingFrameFilter('sw-dec')
    upload   = limef.DecodedUploadFrameFilter('upload')
    pre_dump = limef.DumpFrameFilter('pre-enc',  verbose=args.pre_dump)
    encoder  = limef.EncodingFrameFilter('encoder', enc_params)
    post_dump = limef.DumpFrameFilter('post-enc', verbose=args.post_dump)
    rtp      = limef.RTSPMuxerFrameFilter('rtp-muxer')
    rtsp     = limef.RTSPServerThread('rtsp-server', port=port,
                                      stack_size=30, fifo_size=100)

    src.cc(sw_dec).cc(upload).cc(pre_dump).cc(encoder).cc(post_dump).cc(rtp).cc(rtsp.getInput())

    # ── Banner ──────────────────────────────────────────────────────────────────
    print("==============================================")
    print("  Jetson RTSP: MediaFile → SW decode → V4L2NVEncode")
    print("==============================================")
    print(f"File:        {args.file}")
    print(f"FPS:         {'natural' if args.fps == -1 else args.fps}")
    print(f"Loop:        {'infinite' if args.loop == 0 else args.loop}")
    print(f"NVENC device:{args.device}")
    print(f"Codec:       H.264  bitrate={args.bitrate // 1000} kbps  gop={args.gop_size}")
    print(f"RTSP port:   {port}")
    print(f"LAN IP:      {lan_ip}")
    print(f"URL:         rtsp://{lan_ip}:{port}{url_tail}")
    print(f"Pre-dump:    {'on' if args.pre_dump else 'off'}   Post-dump: {'on' if args.post_dump else 'off'}")
    print("==============================================")
    print("Connect with:")
    print(f"  ffplay rtsp://{lan_ip}:{port}{url_tail}")
    print(f"  ffplay -rtsp_transport tcp rtsp://{lan_ip}:{port}{url_tail}")
    print("==============================================")
    print("Press Ctrl+C to stop\n")

    # ── Start (downstream first, source last) ──────────────────────────────────
    print("Starting RTSP server ...")
    rtsp.start()
    time.sleep(0.1)
    rtsp.expose(SLOT, url_tail)
    time.sleep(0.05)

    print("Starting file source ...")
    src.start()
    time.sleep(0.5)

    print(f"\nReady.  Connect from another device:")
    print(f"  ffplay rtsp://{lan_ip}:{port}{url_tail}\n")

    # ── Main loop ───────────────────────────────────────────────────────────────
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down ...")

    # ── Cleanup (source first, then downstream) ─────────────────────────────────
    print("Stopping file source ...")
    try:
        src.stop()
    except KeyboardInterrupt:
        sys.exit(1)

    print("Stopping RTSP server ...")
    try:
        rtsp.stop()
    except KeyboardInterrupt:
        sys.exit(1)

    print("Done.")


if __name__ == '__main__':
    main()
