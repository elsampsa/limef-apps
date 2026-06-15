#!/usr/bin/env python3
"""
apps/python/jetson_encode.py

Debug script: encode with V4L2NVEncoder then decode back and save PNG frames.

Pipeline:

    MediaFileThread            [encoded packets from file]
        → DecodingFrameFilter  (FFmpeg SW → CPU YUV420P)
        → DumpFrameFilter      [1: post-SW-decode]
        → DecodedUploadFrameFilter (CPU YUV420P → CUDA NV12)
        → DumpFrameFilter      [2: pre-encode, CUDA frames]
        → EncodingFrameFilter  (V4L2NVEncoder, H.264)
        → DumpFrameFilter      [3: post-encode, H.264 packets]
        → DecodingFrameFilter  (FFmpeg SW → CPU YUV420P, decodes H.264)
        → DumpFrameFilter      [4: pre-PNG, CPU decoded frames]
        → WritePNGFrameFilter  (saves frame000001.png, …)

Usage:
    python3 apps/python/jetson_encode.py --file ../../fixtures/jontxu.mkv

Inspect output:
    ls -lh frames/
    eog frames/
"""

import os
import sys
import time
import argparse

import limef

sys.stdout.reconfigure(line_buffering=True)


def main():
    p = argparse.ArgumentParser(
        description='Jetson encode debug: SW-dec → CUDA upload → V4L2NVEnc → SW-dec → PNG',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--file',      required=True, metavar='PATH',
                   help='Input video file')
    p.add_argument('--fps',       type=int, default=1,
                   help='Playback fps (default 1 for slow capture)')
    p.add_argument('--secs',      type=int, default=5,
                   help='How many seconds to run')
    p.add_argument('--bitrate',   type=int, default=2_000_000,
                   help='NVENC target bitrate in bits/sec')
    p.add_argument('--gop-size',  type=int, default=30,
                   help='Keyframe interval (frames)')
    p.add_argument('--device',    default='/dev/v4l2-nvenc', metavar='DEV',
                   help='NVENC V4L2 device')
    p.add_argument('--out-dir',   default='frames', metavar='DIR',
                   help='Output directory for PNG files')
    args = p.parse_args()

    if not os.path.exists(args.file):
        print(f"ERROR: file not found: {args.file}")
        sys.exit(1)

    if not os.path.exists(args.device):
        print(f"ERROR: NVENC device not found: {args.device}")
        sys.exit(1)

    SLOT = 1

    # ── Encoder params ────────────────────────────────────────────────────────
    enc_params              = limef.V4L2NVEncoderParams()
    enc_params.device       = args.device
    enc_params.codec_fourcc = limef.V4L2_PIX_FMT_H264
    enc_params.bitrate      = args.bitrate
    enc_params.gop_size     = args.gop_size
    enc_params.global_header = True   # SPS/PPS in CodecFrame extradata for decoder

    # ── Media file source ─────────────────────────────────────────────────────
    file_ctx       = limef.MediaFileContext(args.file, SLOT)
    file_ctx.fps   = args.fps
    file_ctx.loop  = 1          # single pass

    # ── Pipeline objects ──────────────────────────────────────────────────────
    src        = limef.MediaFileThread('src', file_ctx)
    sw_dec1    = limef.DecodingFrameFilter('sw-dec1')
    dump1      = limef.DumpFrameFilter('post-dec1', verbose=True)
    upload     = limef.DecodedUploadFrameFilter('upload')
    # yuv420p→NV12 on GPU: DecodedUploadFrameFilter now preserves yuv420p layout;
    # CUDAScaleFrameFilter interleaves U+V into NV12 for V4L2NVEncoder.
    to_nv12    = limef.CUDAScaleFrameFilter('to-nv12')   # dst 0×0 = same size, NV12 out
    dump2      = limef.DumpFrameFilter('pre-enc',   verbose=True)
    encoder    = limef.EncodingFrameFilter('encoder', enc_params)
    dump3      = limef.DumpFrameFilter('post-enc',  verbose=True)
    sw_dec2    = limef.DecodingFrameFilter('sw-dec2')
    dump4      = limef.DumpFrameFilter('pre-png',   verbose=True)
    png_writer = limef.WritePNGFrameFilter('png-writer', args.out_dir)

    src.cc(sw_dec1).cc(dump1).cc(upload).cc(to_nv12).cc(dump2).cc(encoder) \
       .cc(dump3).cc(sw_dec2).cc(dump4).cc(png_writer)

    # ── Banner ────────────────────────────────────────────────────────────────
    print("==============================================")
    print("  Jetson encode debug: encode → decode → PNG")
    print("==============================================")
    print(f"File:    {args.file}")
    print(f"FPS:     {args.fps}  Secs: {args.secs}")
    print(f"Bitrate: {args.bitrate // 1000} kbps  GOP: {args.gop_size}")
    print(f"Device:  {args.device}")
    print(f"Out dir: {args.out_dir}")
    print("==============================================\n")

    # ── Start ─────────────────────────────────────────────────────────────────
    src.start()
    time.sleep(args.secs)

    # ── Stop ──────────────────────────────────────────────────────────────────
    print("\nStopping ...")
    try:
        src.stop()
    except KeyboardInterrupt:
        sys.exit(1)

    print(f"Done. PNGs written to: {args.out_dir}/")


if __name__ == '__main__':
    main()
