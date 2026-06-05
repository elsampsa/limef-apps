#!/usr/bin/env python3
"""
apps/python/jetson_cam0.py

Diagnostic: Argus CSI camera → PNG dump

Captures --duration seconds of raw frames from the CSI camera and writes
them as PNG files to --out-dir.

Argus delivers pitch-linear NV12 (NVBUF_LAYOUT_PITCH) — no BL→PL conversion
needed, unlike NVDEC.  WritePNGFrameFilter converts NV12 → RGB24 internally.

Resolution is chosen via --sensor-mode.  Run --list-modes to see available
modes.  On the IMX219 (Raspberry Pi Camera v2), mode 2 is typically 1920×1080.

Pipeline:
    ArgusCameraThread [CPU, NV12, pitch-linear]
        → DumpFrameFilter  (log one line per frame)
        → WritePNGFrameFilter

Usage:
    python3 apps/python/jetson_cam0.py --list-modes
    python3 apps/python/jetson_cam0.py --sensor-mode 2 --duration 1.0
"""

import sys
import time
import argparse
import limef

sys.stdout.reconfigure(line_buffering=True)


def main():
    p = argparse.ArgumentParser(
        description='Argus CSI camera diagnostic: dump N seconds of PNGs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--camera',      type=int, default=0,     metavar='IDX',
                   help='Argus camera device index')
    p.add_argument('--sensor-mode', type=int, default=0,     metavar='MODE',
                   help='Sensor mode index — determines resolution (run --list-modes)')
    p.add_argument('--duration',    type=float, default=1.0,
                   help='Seconds to capture')
    p.add_argument('--out-dir',     default='cam0_frames',
                   help='Directory for PNG output (created if absent)')
    p.add_argument('--list-modes',  action='store_true',
                   help='List available Argus cameras and sensor modes, then exit')
    args = p.parse_args()

    if args.list_modes:
        modes = limef.argus.list_sensor_modes()
        if not modes:
            print("No cameras found.  Is nvargus-daemon running?")
            print("  sudo systemctl start nvargus-daemon")
            sys.exit(1)
        for m in modes:
            print(f"  Camera {m.camera_idx}  Mode {m.mode_idx}: {m.width}x{m.height}")
        sys.exit(0)

    print("==============================================")
    print("  Argus CSI camera → PNG diagnostic")
    print("==============================================")
    print(f"Camera:      index={args.camera}  sensor_mode={args.sensor_mode}")
    print(f"Duration:    {args.duration}s")
    print(f"Output dir:  {args.out_dir}")
    print("Note: Argus NV12 is pitch-linear — no BL→PL needed")
    print("==============================================")
    print("(is nvargus-daemon running?  sudo systemctl start nvargus-daemon)")

    ctx = limef.ArgusCameraContext()
    ctx.camera_index      = args.camera
    ctx.sensor_mode_index = args.sensor_mode
    ctx.output_location   = limef.HWACCEL_NONE  # CPU, pitch-linear NV12

    camera = limef.ArgusCameraThread('argus-cam', ctx)
    dump   = limef.DumpFrameFilter('dump')
    png    = limef.WritePNGFrameFilter('png', args.out_dir)

    camera.cc(dump).cc(png)

    print("\nStarting camera...")
    camera.start()
    time.sleep(args.duration)

    print("\nStopping...")
    try:
        camera.stop()
    except KeyboardInterrupt:
        sys.exit(1)

    print(f"Done.  PNGs written to '{args.out_dir}/'")


if __name__ == '__main__':
    main()
