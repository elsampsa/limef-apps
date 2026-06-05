#!/usr/bin/env python3
"""
apps/python/jetson_decode0.py

Diagnostic: NVDEC (V4L2/CPU) → WritePNGFrameFilter

Pipeline:
    MediaFileThread
        → H264StripParamSetsFrameFilter   (prevents repeated SOURCE_CHANGE on keyframes)
        → DecodingFrameFilter             (V4L2/NVDEC, target=CPU)
        → DumpFrameFilter                 (log each frame: format, location, pts)
        → WritePNGFrameFilter             (pngs/frame000001.png, …)

Purpose: verify that NVDEC delivers sequential (non-stale) frames.

WARNING: V4L2 CPU target on Jetson NVDEC outputs Block-Linear (GOB-tiled) NV12.
WritePNGFrameFilter passes that as-is through libswscale which treats it as
pitch-linear → the PNGs will look visually corrupted (tiled pattern).  That is
expected.  Frame-to-frame pixel variation is still detectable even in BL layout,
which is what this test checks.

NOTE: SwScaleFrameFilter is NOT needed.  WritePNGFrameFilter has its own internal
libswscale conversion from any CPU pixel format to RGB24 before PNG encoding.

Usage:
    cd /home/sampsa/limef
    source go_debug.bash
    python3 limef/apps/python/jetson_decode0.py

    # or with options:
    python3 limef/apps/python/jetson_decode0.py --fps 5 --dir /tmp/pngs
"""

import sys
import os
import time
import argparse

import limef

sys.stdout.reconfigure(line_buffering=True)

_HERE    = os.path.dirname(os.path.abspath(__file__))
FIXTURE  = os.path.normpath(os.path.join(_HERE, '../../fixtures/jontxu_short_no_audio.mkv'))
PNG_DIR  = 'pngs'
SLOT     = 1


def main():
    p = argparse.ArgumentParser(
        description='Jetson NVDEC CPU → PNG diagnostic',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--file',       default=FIXTURE, metavar='PATH',
                   help='Input video file')
    p.add_argument('--fps',        type=int, default=5,
                   help='Feed rate in frames/sec (throttles MediaFileThread)')
    p.add_argument('--dir',        default=PNG_DIR, metavar='DIR',
                   help='Output directory for PNG files')
    p.add_argument('--dec-device', default=None, metavar='DEV',
                   help='V4L2 decoder device; auto-discovered if omitted')
    args = p.parse_args()

    # ── Discover V4L2 decoder ─────────────────────────────────────────────────
    if args.dec_device:
        dec_device  = args.dec_device
        codec_label = '(device overridden)'
    else:
        devs = limef.v4l2.scan_devices()
        sel  = limef.v4l2.select_best_decoder(devs)
        if sel is None:
            print("ERROR: no V4L2 decoder found on this board.")
            print("  Verify /dev/v4l2-nvdec and that the tegra-nvdec driver is bound:")
            print("  cd limef/testing && ./runone.bash v4l2_test:0")
            sys.exit(1)
        dec_device  = sel.device
        codec_label = sel.codec_name

    # ── Build pipeline objects ────────────────────────────────────────────────
    file_ctx      = limef.MediaFileContext(args.file, SLOT)
    file_ctx.fps  = args.fps
    file_ctx.loop = -1   # -1 = play once, no looping

    dec_params        = limef.V4L2NVDecoderParams()
    dec_params.device = dec_device
    dec_params.target = limef.HWACCEL_NONE   # CPU output (Block-Linear on Jetson — expected)

    src       = limef.MediaFileThread('src', file_ctx)
    annexb    = limef.AnnexBFrameFilter("annex-b")
    # h264strip = limef.H264StripParamSetsFrameFilter('h264strip')
    dumpp     = limef.DumpFrameFilter('packet', verbose=True)
    dec       = limef.DecodingFrameFilter('dec', dec_params)
    dump      = limef.DumpFrameFilter('decoded', verbose=True)
    writer    = limef.WritePNGFrameFilter('writer', args.dir)

    dec.setLogLevel(limef.LOG_LEVEL_DEBUG)

    src.cc(annexb).cc(dumpp).cc(dec).cc(dump).cc(writer)

    # ── Banner ────────────────────────────────────────────────────────────────
    abs_dir = os.path.abspath(args.dir)
    print("==============================================")
    print("  Jetson NVDEC CPU → PNG diagnostic")
    print("==============================================")
    print(f"File:       {args.file}")
    print(f"V4L2:       {dec_device}  ({codec_label})")
    print(f"Feed rate:  {args.fps} fps")
    print(f"PNG output: {abs_dir}/")
    print("NOTE: PNGs will be visually corrupted (BL layout)")
    print("      but should vary frame-to-frame if decoder is working.")
    print("==============================================\n")

    # ── Run ───────────────────────────────────────────────────────────────────
    src.start()

    try:
        while src.isRunning():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    src.stop()

    n = len([f for f in os.listdir(abs_dir) if f.endswith('.png')]) if os.path.isdir(abs_dir) else 0
    print(f"\nDone.  {n} PNG(s) written to {abs_dir}/")


if __name__ == '__main__':
    main()
