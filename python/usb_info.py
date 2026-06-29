#!/usr/bin/env python3
"""
apps/python/usb_info.py

USB camera → PutInfoFrameFilter → TensorPythonInterface → InfoFrameFilter demo.

Two message sources flow through the pipeline:

1. A timer thread calls put_ff.put("message N") every --inject-interval seconds.
   PutInfoFrameFilter emits this as an InfoFrame just before the next TensorFrame.
   TensorPythonInterface.pull() surfaces it; the consumer captures it and merges
   it into the next outgoing InfoFrame.

2. The consumer itself counts TensorFrames and pushes an InfoFrame every
   --frame-interval frames carrying {"frames": N, "injected": "..."}.

InfoFrameFilter downstream catches all outgoing InfoFrames and queues them.
A reader thread wakes via EventFd and drains popMessage().

Pipeline:
    USBCameraThread
      → DecodedToTensorFrameFilter
      → PutInfoFrameFilter         ← timer thread: put("message N") every 10 s
      → TensorPythonInterface
          ↑ consumer: on InfoFrame capture msg; every N TensorFrames push merged InfoFrame
      → InfoFrameFilter(efd)
          ↑ reader thread: select on efd, popMessage()

Usage:
    source go_debug.bash
    python3 apps/python/usb_info.py
    python3 apps/python/usb_info.py --inject-interval 5 --frame-interval 20
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
    p.add_argument('-d', '--device',          default='/dev/video0', metavar='DEV',
                   help='V4L2 camera device')
    p.add_argument('-w', '--width',            type=int, default=640)
    p.add_argument('-H', '--height',           type=int, default=480)
    p.add_argument('-f', '--fps',              type=int, default=30)
    p.add_argument('--frame-interval',         type=int, default=10, metavar='N',
                   help='push an InfoFrame every N TensorFrames')
    p.add_argument('--inject-interval',        type=float, default=10.0, metavar='SECS',
                   help='inject a put() message every SECS seconds')
    args = p.parse_args()

    SLOT       = 1
    TIMEOUT_MS = 200

    print("==============================================")
    print("  USB Camera → InfoFrame message channel")
    print("==============================================")
    print(f"Device:          {args.device}")
    print(f"Resolution:      {args.width}x{args.height} @ {args.fps} fps")
    print(f"Frame interval:  every {args.frame_interval} frames")
    print(f"Inject interval: every {args.inject_interval:.0f} s")
    print("==============================================\n")

    # ── Camera source ──────────────────────────────────────────────────────────
    cam_ctx                = limef.USBCameraContext(args.device, SLOT)
    cam_ctx.width          = args.width
    cam_ctx.height         = args.height
    cam_ctx.fps            = args.fps
    cam_ctx.capture_format = limef.AV_PIX_FMT_YUYV422

    camera = limef.USBCameraThread('usb-camera', cam_ctx)
    d2t    = limef.DecodedToTensorFrameFilter('d2t', limef.CHANNEL_ORDER_RGB)

    # ── PutInfoFrameFilter — injects messages from the timer thread ────────────
    put_ff = limef.PutInfoFrameFilter('put')

    # ── TensorPythonInterface ──────────────────────────────────────────────────
    pyf    = limef.TensorPythonInterface(stack_size=10, leaky=True,
                                         hw_accel=limef.HWACCEL_NONE, fifo_size=0)
    client = pyf.client()

    # ── InfoFrameFilter downstream ─────────────────────────────────────────────
    efd  = limef.EventFd()
    info = limef.InfoFrameFilter('info', efd)

    # ── Wire pipeline ──────────────────────────────────────────────────────────
    camera.cc(d2t).cc(put_ff).cc(pyf.getInput())
    pyf.getOutput().cc(info)

    # ── Timer thread — injects messages upstream via put_ff ────────────────────
    stop_event    = threading.Event()
    inject_count  = [0]

    def injector():
        while not stop_event.is_set():
            stop_event.wait(timeout=args.inject_interval)
            if stop_event.is_set():
                break
            inject_count[0] += 1
            msg = f"message {inject_count[0]}"
            put_ff.put(msg)
            print(f"[injector] put: {msg!r}")

    injector_thread = threading.Thread(target=injector, daemon=True,
                                       name='injector')

    # ── Consumer thread ────────────────────────────────────────────────────────
    frame_count   = [0]
    last_injected = [None]   # most recent message received from put_ff

    def consumer():
        while not stop_event.is_set():
            frame = client.pull(timeout_ms=TIMEOUT_MS)

            if frame is None:
                continue

            # StreamFrame: pass through so downstream sees codec init.
            if isinstance(frame, limef.StreamFrame):
                client.push(frame)
                continue

            # InfoFrame injected by PutInfoFrameFilter: capture and echo it.
            if isinstance(frame, limef.InfoFrame):
                last_injected[0] = frame.message
                print(f"[consumer] received injected: {frame.message!r}")
                continue

            if not isinstance(frame, limef.TensorFrame):
                continue

            frame_count[0] += 1

            if frame_count[0] % args.frame_interval == 0:
                payload = {"frames": frame_count[0]}
                if last_injected[0] is not None:
                    payload["injected"] = last_injected[0]
                client.push(limef.InfoFrame(json.dumps(payload)))

    consumer_thread = threading.Thread(target=consumer, daemon=True,
                                       name='tensor-consumer')

    # ── Reader thread ──────────────────────────────────────────────────────────
    def reader():
        fd = efd.getFd()
        while not stop_event.is_set():
            r, _, _ = select.select([fd], [], [], 0.2)
            if not r:
                continue
            efd.clear()
            while info.hasMessage():
                data = json.loads(info.popMessage())
                print(f"[reader]   {data}")

    reader_thread = threading.Thread(target=reader, daemon=True,
                                     name='info-reader')

    # ── Start ──────────────────────────────────────────────────────────────────
    consumer_thread.start()
    reader_thread.start()
    injector_thread.start()
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
    injector_thread.join(timeout=args.inject_interval + 1.0)

    print(f"\nDone.  {frame_count[0]} TensorFrames, "
          f"{inject_count[0]} injected messages.")


if __name__ == '__main__':
    main()
