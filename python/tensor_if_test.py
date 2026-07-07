#!/usr/bin/env python3
"""
apps/python/tensor_if_test.py

Minimal pipeline to test TensorPythonInterface receiving GPU TensorFrames from
a software-decoded file source.

Pipeline:
    MediaFileThread → DecodingFrameFilter(sw) → SwScaleFrameFilter(NV12)
        → DecodedUploadFrameFilter(CUDA) → DecodedToTensorFrameFilter(RGB)
        → TensorPythonInterface(HWACCEL_CUDA) ← Python (inspect & pass through)
        → TensorToDecodedFrameFilter(RGB)  ← bug reproduction: needs CUDA hwcontext
        → DumpFrameFilter

Usage:
    python3 apps/python/tensor_if_test.py
    python3 apps/python/tensor_if_test.py --file /path/to/video.mkv --frames 20
"""

import sys
import os
import time
import argparse
import threading

import limef

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_FIXTURE_FILE = os.path.join(_SCRIPT_DIR, '../../fixtures/jontxu.mkv')


def main():
    p = argparse.ArgumentParser(
        description='TensorPythonInterface smoke test (file → SW decode → GPU tensor)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--file',   default=_FIXTURE_FILE, metavar='PATH',
                   help='input video file')
    p.add_argument('--frames', type=int, default=10,
                   help='stop after this many TensorFrames')
    args = p.parse_args()

    SLOT       = 1
    TIMEOUT_MS = 500

    print("=== tensor_if_test ===")
    print(f"File:   {args.file}")
    print(f"Frames: {args.frames}")
    print(f"CUDA available (runtime): {limef.isCUDARuntimeAvailable()}")
    print(f"CUDA available (FFmpeg):  {limef.isHWAccelAvailable(limef.HWACCEL_CUDA)}")
    print()

    # ── Pipeline ──────────────────────────────────────────────────────────────
    file_ctx      = limef.MediaFileContext(args.file, SLOT)
    file_ctx.fps  = 0   # as fast as possible
    file_ctx.loop = -1  # no loop

    src    = limef.MediaFileThread('src', file_ctx)
    dec    = limef.DecodingFrameFilter('dec')                           # software decode
    scale  = limef.SwScaleFrameFilter('scale', limef.AV_PIX_FMT_NV12) # → NV12
    upload = limef.DecodedUploadFrameFilter('upload')                   # CPU → CUDA
    d2t    = limef.DecodedToTensorFrameFilter('d2t', limef.CHANNEL_ORDER_RGB)

    # HWACCEL_CUDA: fifo stack frames are allocated on GPU (BufferLocation::CUDA).
    # This is the configuration used in the mantis-server pipeline and is the
    # suspected trigger for the "no cuda context" error we want to reproduce.
    pyf    = limef.TensorPythonInterface(stack_size=10, leaky=False,
                                         hw_accel=limef.HWACCEL_CUDA, fifo_size=0)
    client = pyf.client()

    # TensorToDecodedFrameFilter downstream of TensorPythonInterface — this is
    # the filter that fails with "no cuda context" in the mantis-server pipeline.
    # Frames pushed back by the Python consumer flow through it.
    t2d_out = limef.TensorToDecodedFrameFilter('t2d_out', limef.CHANNEL_ORDER_RGB)
    dump    = limef.DumpFrameFilter('t2d-dump')

    src.cc(dec).cc(scale).cc(upload).cc(d2t).cc(pyf.getInput())
    pyf.getOutput().cc(t2d_out).cc(dump)

    # ── Consumer ──────────────────────────────────────────────────────────────
    done        = threading.Event()
    frame_count = [0]

    def consumer():
        while not done.is_set():
            frame = client.pull(timeout_ms=TIMEOUT_MS)
            if frame is None:
                continue

            if isinstance(frame, limef.StreamFrame):
                print(f"  StreamFrame")
                client.push(frame)
                continue

            if not isinstance(frame, limef.TensorFrame):
                client.push(frame)
                continue

            n = frame_count[0] + 1
            frame_count[0] = n

            if frame.num_planes > 0:
                plane = frame.planes[0]
                if frame.is_gpu:
                    try:
                        import torch
                        shape = list(torch.from_dlpack(plane).shape)
                    except Exception:
                        shape = '(gpu-no-torch)'
                else:
                    shape = list(plane.shape)
            else:
                shape = '?'
            print(f"  TensorFrame #{n:3d}  gpu={frame.is_gpu}"
                  f"  shape={shape}  ts={frame.timestamp}")

            client.push(frame)

            if n >= args.frames:
                print(f"\nGot {n} frames — done.")
                done.set()

    t = threading.Thread(target=consumer, daemon=True, name='consumer')

    # ── Start ─────────────────────────────────────────────────────────────────
    t.start()
    src.start()

    try:
        t.join(timeout=30)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        done.set()
        src.stop()

    if not frame_count[0]:
        print("ERROR: no TensorFrames received", file=sys.stderr)
        sys.exit(1)

    print("Done.")


if __name__ == '__main__':
    main()
