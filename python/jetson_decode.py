#!/usr/bin/env python3
"""
apps/python/jetson_decode.py

File decode via NVDEC (V4L2 M2M) → CUDA scale → VP8 → RTSP streaming demo
for Jetson Orin.

Decodes a video file using Jetson NVDEC hardware (V4L2 M2M API), scales and
converts NV12→YUV420P on the GPU in a single pass, and re-encodes as VP8 for
RTSP streaming.

Pipeline:

    MediaFileThread  [encoded packets]
        → DecodingFrameFilter     (V4L2/NVDEC — hardware decode, outputs CUDA NV12)
        → CUDAScaleFrameFilter    (scale + NV12→YUV420P on GPU)
        → DecodedDownloadFrameFilter  (CUDA YUV420P → CPU YUV420P)
        → EncodingFrameFilter     (libvpx VP8 software encode)
        → RTSPMuxerFrameFilter
        → RTSPServerThread

Usage:
    python3 apps/python/jetson_decode.py --file video.mp4 [options]

Then connect from another machine:
    ffplay rtsp://<jetson-ip>:8554/live/stream
    ffplay -rtsp_transport tcp rtsp://<jetson-ip>:8554/live/stream

Press Ctrl+C to stop.
"""

import sys
import time
import socket
import argparse

import limef

sys.stdout.reconfigure(line_buffering=True)


def _lan_ip():
    """Return the first LAN IP address (used to reach the internet)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main():
    p = argparse.ArgumentParser(
        description='Limef Jetson NVDEC file decode → VP8 RTSP demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--file',       required=True, metavar='PATH',
                   help='Input video file (H.264 recommended for NVDEC on Orin Nano)')
    p.add_argument('--width',      type=int, default=1280,
                   help='Encode width  (CUDAScaleFrameFilter target)')
    p.add_argument('--height',     type=int, default=720,
                   help='Encode height (CUDAScaleFrameFilter target)')
    p.add_argument('--fps',        type=int, default=30,
                   help='Nominal frame rate for VP8 GOP sizing')
    p.add_argument('--bitrate',    type=int, default=4_000_000,
                   help='VP8 encoder bitrate in bits/sec')
    p.add_argument('--port',       type=int, default=8554,
                   help='RTSP server port')
    p.add_argument('--url-tail',   default='/live/stream', metavar='PATH',
                   help='RTSP URL path')
    p.add_argument('--dec-device', default=None, metavar='DEV',
                   help='V4L2 decoder device (e.g. /dev/v4l2-nvdec); auto-discovered if not set')
    args = p.parse_args()

    lan_ip   = _lan_ip()
    port     = args.port
    url_tail = args.url_tail
    SLOT     = 1

    # ── Discover V4L2 decoder ─────────────────────────────────────────────────
    if args.dec_device:
        dec_device = args.dec_device
        codec_info = '(device overridden)'
    else:
        devs = limef.v4l2.scan_devices()
        sel  = limef.v4l2.select_best_decoder(devs)
        if sel is None:
            print("ERROR: no V4L2 decoder found on this board.")
            print("  Check /dev/v4l2-nvdec and verify the driver is bound:")
            print("  cd limef/testing && ./runone.bash v4l2_test:0")
            sys.exit(1)
        dec_device = sel.device
        codec_info = sel.codec_name

    # ── Build pipeline objects ────────────────────────────────────────────────
    # V4L2 decoder: codec identity is auto-detected from the upstream CodecFrame.
    # target=HWACCEL_CUDA tells NVDEC to emit frames directly into CUDA memory,
    # so no separate upload step is needed before CUDAScaleFrameFilter.
    dec_params        = limef.V4L2DecoderParams()
    dec_params.device = dec_device
    dec_params.target = limef.HWACCEL_CUDA

    # Scale + deinterleave NV12→YUV420P on GPU in one pass (same as jetson_cam.py branch b).
    scale_params               = limef.CUDAScaleParams(args.width, args.height)
    scale_params.output_format = limef.AV_PIX_FMT_YUV420P

    enc_params          = limef.FFmpegEncoderParams()
    enc_params.codec_id = limef.AV_CODEC_ID_VP8
    enc_params.bitrate  = args.bitrate
    enc_params.gop_size = max(1, args.fps // 2)

    file_ctx      = limef.MediaFileContext(args.file, SLOT)
    file_ctx.fps  = 1   # feed at very slow speed ftm
    file_ctx.loop = 0    # play once; set to 0 for gapless looping

    src      = limef.MediaFileThread('src', file_ctx)
    dec      = limef.DecodingFrameFilter('dec', dec_params)
    scale    = limef.CUDAScaleFrameFilter('scale', scale_params)
    download = limef.DecodedDownloadFrameFilter('download')
    encoder  = limef.EncodingFrameFilter('encoder', enc_params)
    rtp      = limef.RTSPMuxerFrameFilter('rtp-muxer')
    rtsp     = limef.RTSPServerThread('rtsp-server', port=port,
                                      stack_size=30, fifo_size=100)

    src.cc(dec).cc(scale).cc(download).cc(encoder).cc(rtp).cc(rtsp.getInput())

    # ── Banner ────────────────────────────────────────────────────────────────
    print("==============================================")
    print("  Jetson NVDEC File Decode → VP8 RTSP")
    print("==============================================")
    print(f"File:        {args.file}")
    print(f"V4L2 device: {dec_device}  ({codec_info})")
    print(f"Encode res:  {args.width}x{args.height}")
    print(f"GOP size:    {enc_params.gop_size}  (fps={args.fps})")
    print(f"Bitrate:     {args.bitrate // 1000} kbps  (libvpx VP8)")
    print(f"RTSP port:   {port}")
    print(f"LAN IP:      {lan_ip}")
    print(f"URL:         rtsp://{lan_ip}:{port}{url_tail}")
    print("==============================================")
    print("Connect with:")
    print(f"  ffplay rtsp://{lan_ip}:{port}{url_tail}")
    print(f"  ffplay -rtsp_transport tcp rtsp://{lan_ip}:{port}{url_tail}")
    print("==============================================")
    print("Press Ctrl+C to stop\n")

    # ── Start (downstream first, source last) ─────────────────────────────────
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

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down ...")

    # ── Cleanup (source first, then downstream) ───────────────────────────────
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
