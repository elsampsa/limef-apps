#!/usr/bin/env python3
"""
apps/python/jetson_cam0.py

Diagnostic: Argus CSI camera frame-rate measurement and optional PNG dump.

Pipeline:
    ArgusCameraThread [CPU, NV12, pitch-linear]
        → DumpFrameFilter       (one log line per frame)
        → CountDecodedFrameFilter
        [→ WritePNGFrameFilter]  (only with --png)

Usage:
    python3 apps/python/jetson_cam0.py --list-modes
    python3 apps/python/jetson_cam0.py --sensor-mode 4 --fps 30 --duration 10
    python3 apps/python/jetson_cam0.py --sensor-mode 2 --png --duration 2
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
    p.add_argument('--fps',         type=int, default=30,
                   help='Capture frame rate in frames per second')
    p.add_argument('--png',         action="store_true",
                   help='dump png (will slow down your pipeline)')
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
            print(f"  Camera {m.camera_idx}  Mode {m.mode_idx}: {m.width}x{m.height}  {m.min_fps}-{m.max_fps} fps")
        sys.exit(0)

    print("==============================================")
    print("  Argus CSI camera → PNG diagnostic")
    print("==============================================")
    print(f"Camera:      index={args.camera}  sensor_mode={args.sensor_mode}  fps={args.fps}")
    print(f"Duration:    {args.duration}s")
    print(f"Output dir:  {args.out_dir}")
    print("Note: Argus NV12 is pitch-linear — no BL→PL needed")
    print("==============================================")
    print("(is nvargus-daemon running?  sudo systemctl start nvargus-daemon)")

    ctx = limef.ArgusCameraContext()
    ctx.camera_index      = args.camera
    ctx.sensor_mode_index = args.sensor_mode
    ctx.fps               = args.fps
    ctx.output_location   = limef.HWACCEL_NONE  # CPU, pitch-linear NV12

    camera  = limef.ArgusCameraThread('argus-cam', ctx)
    dump    = limef.DumpFrameFilter('dump')
    counter = limef.CountDecodedFrameFilter('counter')
    png     = limef.WritePNGFrameFilter('png', args.out_dir)

    tail = camera.cc(dump).cc(counter)
    if args.png:
        tail.cc(png)

    print("\nStarting camera...")
    camera.start()
    time.sleep(args.duration)

    print("\nStopping...")
    try:
        camera.stop()
    except KeyboardInterrupt:
        sys.exit(1)

    counter.report()
    if args.png:
        print(f"PNGs written to '{args.out_dir}/'")


if __name__ == '__main__':
    main()
