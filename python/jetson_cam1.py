#!/usr/bin/env python3
"""
apps/python/jetson_cam1.py

Argus CSI camera via CUDA path: capture → GPU scale → CPU download → optional PNG dump.

Pipeline:
    ArgusCameraThread [CUDA, NV12, EGL-mapped]
        → CUDAScaleFrameFilter   (GPU bilinear resize; 0×0 = keep source size)
        → DecodedDownloadFrameFilter  (CUDA NV12 → CPU NV12)
        → CountDecodedFrameFilter
        [→ WritePNGFrameFilter]  (only with --png)

Usage:
    python3 apps/python/jetson_cam1.py --list-modes
    python3 apps/python/jetson_cam1.py --sensor-mode 4 --fps 30 --duration 10
    python3 apps/python/jetson_cam1.py --sensor-mode 4 --width 640 --height 360 --png --duration 2
"""

import sys
import time
import argparse
import limef

sys.stdout.reconfigure(line_buffering=True)


def main():
    p = argparse.ArgumentParser(
        description='Argus CSI camera CUDA path: capture → GPU scale → CPU download → optional PNG',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--camera',      type=int, default=0,     metavar='IDX',
                   help='Argus camera device index')
    p.add_argument('--sensor-mode', type=int, default=0,     metavar='MODE',
                   help='Sensor mode index — determines resolution (run --list-modes)')
    p.add_argument('--fps',         type=int, default=30,
                   help='Capture frame rate in frames per second')
    p.add_argument('--width',       type=int, default=0,
                   help='Scale output width (0 = keep sensor resolution)')
    p.add_argument('--height',      type=int, default=0,
                   help='Scale output height (0 = keep sensor resolution)')
    p.add_argument('--png',         action='store_true',
                   help='Write PNG frames (slows pipeline)')
    p.add_argument('--duration',    type=float, default=1.0,
                   help='Seconds to capture')
    p.add_argument('--out-dir',     default='cam1_frames',
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

    scale_str = f"{args.width}x{args.height}" if (args.width or args.height) else "passthrough"
    print("==============================================")
    print("  Argus CSI camera → CUDA scale → CPU download")
    print("==============================================")
    print(f"Camera:      index={args.camera}  sensor_mode={args.sensor_mode}  fps={args.fps}")
    print(f"Scale:       {scale_str}")
    print(f"Duration:    {args.duration}s")
    print(f"Output dir:  {args.out_dir}")
    print("==============================================")
    print("(is nvargus-daemon running?  sudo systemctl start nvargus-daemon)")

    ctx = limef.ArgusCameraContext()
    ctx.camera_index      = args.camera
    ctx.sensor_mode_index = args.sensor_mode
    ctx.fps               = args.fps
    ctx.output_location   = limef.HWACCEL_CUDA

    camera   = limef.ArgusCameraThread('argus-cam', ctx)
    dump     = limef.DumpFrameFilter('dump', verbose=False)
    scale    = limef.CUDAScaleFrameFilter('scale', limef.CUDAScaleParams(args.width, args.height))
    download = limef.DecodedDownloadFrameFilter('download')
    counter  = limef.CountDecodedFrameFilter('counter')
    png      = limef.WritePNGFrameFilter('png', args.out_dir)

    tail = camera.cc(dump).cc(scale).cc(download).cc(counter)
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
