#!/usr/bin/env python3
"""
apps/python/usb_info.py

USB camera → TensorPythonInterface → InfoFrameFilter demo.

The Python consumer counts incoming TensorFrames and every 10 frames pushes
an InfoFrame carrying JSON {"frames": N} downstream.  A separate reader thread
waits on an EventFd and prints each message as it arrives.

No video output — this demo is purely about the InfoFrame message channel.

Pipeline:
    USBCameraThread
      → DecodedToTensorFrameFilter
      → TensorPythonInterface
          ↑ Python consumer: count frames, push InfoFrame every 10
      → InfoFrameFilter(efd)
          ↑ reader thread: select on efd, popMessage()

Usage:
    source go_debug.bash
    python3 apps/python/usb_info.py
    python3 apps/python/usb_info.py --device /dev/video2
"""

import sys
import time
import json
import select
import argparse
import threading

import limef

sys.stdout.reconfigure(line_buffering=True)


def main():
    p = argparse.ArgumentParser(
        description='limef USB camera → InfoFrame message channel demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-d', '--device',  default='/dev/video0', metavar='DEV',
                   help='V4L2 camera device')
    p.add_argument('-w', '--width',   type=int, default=640)
    p.add_argument('-H', '--height',  type=int, default=480)
    p.add_argument('-f', '--fps',     type=int, default=30)
    p.add_argument('--interval',      type=int, default=10, metavar='N',
                   help='push an InfoFrame every N TensorFrames')
    args = p.parse_args()

    SLOT       = 1
    TIMEOUT_MS = 200

    print("==============================================")
    print("  USB Camera → InfoFrame message channel")
    print("==============================================")
    print(f"Device:     {args.device}")
    print(f"Resolution: {args.width}x{args.height} @ {args.fps} fps")
    print(f"Interval:   every {args.interval} frames")
    print("==============================================\n")

    # ── Camera source ──────────────────────────────────────────────────────────
    cam_ctx                = limef.USBCameraContext(args.device, SLOT)
    cam_ctx.width          = args.width
    cam_ctx.height         = args.height
    cam_ctx.fps            = args.fps
    cam_ctx.capture_format = limef.AV_PIX_FMT_YUYV422

    camera = limef.USBCameraThread('usb-camera', cam_ctx)
    d2t    = limef.DecodedToTensorFrameFilter('d2t', limef.CHANNEL_ORDER_RGB)

    # ── TensorPythonInterface ──────────────────────────────────────────────────
    pyf    = limef.TensorPythonInterface(stack_size=10, leaky=True,
                                         hw_accel=limef.HWACCEL_NONE, fifo_size=0)
    client = pyf.client()

    # ── InfoFrameFilter downstream ─────────────────────────────────────────────
    efd  = limef.EventFd()
    info = limef.InfoFrameFilter('info', efd)

    # ── Wire pipeline ──────────────────────────────────────────────────────────
    camera.cc(d2t).cc(pyf.getInput())
    pyf.getOutput().cc(info)

    # ── Consumer thread ────────────────────────────────────────────────────────
    stop_event  = threading.Event()
    frame_count = [0]

    def consumer():
        while not stop_event.is_set():
            frame = client.pull(timeout_ms=TIMEOUT_MS)

            if frame is None:
                continue

            # Pass StreamFrames through (codec init signal).
            if isinstance(frame, limef.StreamFrame):
                client.push(frame)
                continue

            if not isinstance(frame, limef.TensorFrame):
                continue

            frame_count[0] += 1

            if frame_count[0] % args.interval == 0:
                msg = json.dumps({"frames": frame_count[0]})
                client.push(limef.InfoFrame(msg))

    consumer_thread = threading.Thread(target=consumer, daemon=True,
                                       name='tensor-consumer')

    # ── Reader thread ──────────────────────────────────────────────────────────
    def reader():
        fd = efd.getFd()
        while not stop_event.is_set():
            r, _, _ = select.select([fd], [], [], 0.2)
            if not r:
                continue
            efd.clear()                     # drain the eventfd counter
            while info.hasMessage():
                data = json.loads(info.popMessage())
                print(f"[reader] {data}")

    reader_thread = threading.Thread(target=reader, daemon=True,
                                     name='info-reader')

    # ── Start (downstream filters need no explicit start; camera last) ─────────
    consumer_thread.start()
    reader_thread.start()
    camera.start()

    print("Running — press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down...")

    # ── Stop ───────────────────────────────────────────────────────────────────
    stop_event.set()

    try:
        camera.stop()
    except KeyboardInterrupt:
        sys.exit(1)

    consumer_thread.join(timeout=2.0)
    reader_thread.join(timeout=1.0)

    print(f"\nDone.  {frame_count[0]} TensorFrames received.")


if __name__ == '__main__':
    main()
