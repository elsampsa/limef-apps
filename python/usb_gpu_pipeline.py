#!/usr/bin/env python3
"""
apps/python/usb_gpu_pipeline.py

Python version of apps/cpp/usb_gpu_pipeline.cpp.

All camera capture, upload, encoding, and RTSP serving run in C++ threads.
Python only "visits" the frame data to apply Gaussian blur (--modify flag),
using TensorPythonInterface as the pass-through point.

Pipeline (default — passthrough, no Python processing):
    [C++] USBCameraThread → UploadGPUFrameFilter(CUDA)
        → DecodedToTensorFrameFilter(RGB)
        → TensorPythonInterface  ← Python consumer (passes frame straight through)
        → TensorToDecodedFrameFilter(RGB)
        → EncodingFrameFilter(NVENC H264) → RTSPMuxerFrameFilter → RTSPServerThread

Pipeline (--modify — Gaussian blur in Python, mirrors C++ GPUOpenCVThread):
    [C++] USBCameraThread → UploadGPUFrameFilter(CUDA)
        → DecodedToTensorFrameFilter(RGB)
        → TensorPythonInterface  ← Python: pull GPU tensor, blur on CPU, push back
        → TensorToDecodedFrameFilter(RGB)
        → EncodingFrameFilter(NVENC H264) → RTSPMuxerFrameFilter → RTSPServerThread

Usage:
    python3 apps/python/usb_gpu_pipeline.py [options]
    python3 apps/python/usb_gpu_pipeline.py --modify   # enable Gaussian blur

Then connect with:
    ffplay rtsp://localhost:8554/live/stream
    ffplay -rtsp_transport tcp rtsp://localhost:8554/live/stream
    ffplay -fflags nobuffer -flags low_delay -framedrop -probesize 32 -analyzeduration 0 rtsp://localhost:8554/live/stream

Press Ctrl+C to stop.
"""

import sys
import time
import argparse
import threading

import limef

try:
    import torch
    import torch.nn.functional as F_torch
    _TORCH = True
except ImportError:
    _TORCH = False


def _make_gauss_kernel(device):
    """Build a 15×15 Gaussian kernel matching OpenCV GaussianBlur(15,15,0)."""
    ksize = 15
    # OpenCV sigma formula for ksize=15: 0.3*((15-1)*0.5 - 1) + 0.8 = 2.6
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    x = torch.arange(ksize, dtype=torch.float32, device=device) - ksize // 2
    gauss = torch.exp(-x.pow(2.0) / (2.0 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss.outer(gauss)                            # (15, 15)
    # Shape for grouped conv: (out_channels, in_channels/groups, H, W)
    # groups=C means one filter per channel
    return kernel_2d.view(1, 1, ksize, ksize)


def main():
    p = argparse.ArgumentParser(
        description='limef USB camera → GPU → RTSP pipeline with optional Python frame processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-m', '--modify',  action='store_true',
                   help='enable Gaussian blur in Python (mirrors C++ GPUOpenCVThread)')
    p.add_argument('-d', '--device',  default='/dev/video0', metavar='DEV',
                   help='V4L2 camera device')
    p.add_argument('-p', '--port',    type=int, default=8554,
                   help='RTSP server port')
    p.add_argument('-w', '--width',   type=int, default=640)
    p.add_argument('-H', '--height',  type=int, default=480)
    p.add_argument('-f', '--fps',     type=int, default=30)
    p.add_argument('--bitrate',       type=int, default=4_000_000,
                   help='NVENC target bitrate in bits/sec')
    p.add_argument('--url-tail',      default='/live/stream', metavar='PATH',
                   help='RTSP URL path component')
    args = p.parse_args()

    port     = args.port
    url_tail = args.url_tail
    SLOT     = 1
    TIMEOUT_MS = 200

    print("==========================================")
    print("  USB Camera → GPU → RTSP (Python)")
    print("==========================================")
    print(f"Device:     {args.device}")
    print(f"Resolution: {args.width}x{args.height} @ {args.fps} fps")
    print(f"Port:       {port}")
    print(f"URL:        rtsp://localhost:{port}{url_tail}")
    print(f"Modify:     {args.modify}  (Gaussian blur 15×15 in Python)")
    print("==========================================")
    print("Press Ctrl+C to stop\n")

    # ── C++ camera source ──────────────────────────────────────────────────────
    # USBCameraThread: YUYV → internal SwScale → NV12 DecodedFrame
    cam_ctx               = limef.USBCameraContext(args.device, SLOT)
    cam_ctx.width         = args.width
    cam_ctx.height        = args.height
    cam_ctx.fps           = args.fps
    cam_ctx.output_format = limef.AV_PIX_FMT_NV12  # suitable for GPU upload

    camera = limef.USBCameraThread('usb-camera', cam_ctx)

    # ── C++ upstream chain (before Python visit) ───────────────────────────────
    # NV12 DecodedFrame → GPU upload → TensorFrame (3, H, W) RGB on GPU
    upload = limef.UploadGPUFrameFilter('gpu-upload', limef.HWACCEL_CUDA)
    d2t    = limef.DecodedToTensorFrameFilter('d2t', limef.CHANNEL_ORDER_RGB)

    # ── TensorPythonInterface — Python visits here ─────────────────────────────
    # GPU TensorFrames arrive via pull(); Python may process and must push() back.
    #
    # TensorPythonInterface fifo: stack_size=10, leaky=True, hw_accel=CUDA
    #   10 TensorFrame slots absorb a pull() delay in the Python consumer loop.
    #   leaky=True: drop GPU frames if the Python loop falls behind rather than
    #   stalling the camera → upload → d2t chain.
    #   hw_accel=CUDA: CPU TensorFrames are uploaded to the GPU at the thread
    #   boundary (H2D copy) before they become available from pull().
    pyf    = limef.TensorPythonInterface(stack_size=10, leaky=True,
                                         hw_accel=limef.HWACCEL_CUDA, fifo_size=0)
    client = pyf.client()

    # ── C++ downstream chain (after Python visit) ──────────────────────────────
    # TensorFrame (3, H, W) RGB GPU → DecodedFrame CUDA NV12 → NVENC → RTP → RTSP
    t2d = limef.TensorToDecodedFrameFilter('t2d', limef.CHANNEL_ORDER_RGB)

    enc_params              = limef.FFmpegEncoderParams()
    enc_params.codec_id     = limef.AV_CODEC_ID_H264
    enc_params.hw_accel     = limef.HWACCEL_CUDA
    enc_params.bitrate      = args.bitrate
    enc_params.preset       = 'p1'
    enc_params.tune         = 'ull'
    enc_params.max_b_frames = 0
    enc_params.gop_size     = args.fps // 2

    encoder   = limef.EncodingFrameFilter('encoder', enc_params)
    rtp_muxer = limef.RTSPMuxerFrameFilter('rtp-muxer')
    # stack_size=30: absorbs I-frame bursts (several RTP packets in rapid succession).
    # fifo_size=100: cap on queued RTP packets; prevents unbounded growth under slow clients.
    # leaky is always False for RTSPServerThread — RTP sequence gaps corrupt the stream.
    rtsp      = limef.RTSPServerThread('rtsp-server', port=port, stack_size=30, fifo_size=100)

    # ── Wire the pipeline ──────────────────────────────────────────────────────
    # camera → upload → d2t → [Python] → t2d → encoder → rtp → rtsp
    camera.cc(upload).cc(d2t).cc(pyf.getInput())
    pyf.getOutput().cc(t2d).cc(encoder).cc(rtp_muxer).cc(rtsp.getInput())

    # ── Python consumer thread ─────────────────────────────────────────────────
    stop_event  = threading.Event()
    frame_count = [0]
    t_start     = [time.monotonic()]

    # Gaussian kernel is built lazily on the first frame so we get the right device.
    gauss_kernel    = [None]   # [Tensor or None]
    warned_cpu_push = [False]  # one-shot format-mismatch warning

    def consumer():
        while not stop_event.is_set():
            frame = client.pull(timeout_ms=TIMEOUT_MS)

            if frame is None:
                continue

            if isinstance(frame, limef.StreamFrame):
                # Forward stream metadata so EncodingFrameFilter can configure itself
                client.push(frame)
                continue

            if not isinstance(frame, limef.TensorFrame):
                continue

            frame_count[0] += 1

            if args.modify and frame.is_gpu and _TORCH:
                # ── GPU Gaussian blur — mirrors C++ GPUOpenCVThread ─────────────
                # PIPELINE RULE: always push GPU TensorFrames into this pipeline.
                # TensorToDecodedFrameFilter selects its output format based on the
                # incoming frame: GPU → AV_PIX_FMT_CUDA (NV12) → NVENC ✓
                #                 CPU → AV_PIX_FMT_GBRP             → NVENC ✗ (wrong colours)
                # So any processing that produces CPU data must upload back to GPU
                # before pushing. Here we stay on GPU the whole time via torch.
                t = torch.from_dlpack(frame.planes[0]).float()  # (3,H,W) float32 CUDA
                C, H, W = t.shape

                # Lazy-init kernel on the same device as the tensor
                if gauss_kernel[0] is None:
                    gauss_kernel[0] = _make_gauss_kernel(t.device).expand(C, 1, 15, 15).contiguous()

                # Grouped depthwise conv: one filter per channel, padding=7 → same size
                blurred = F_torch.conv2d(
                    t.unsqueeze(0), gauss_kernel[0], padding=7, groups=C
                ).squeeze(0).clamp(0, 255).to(torch.uint8)    # (3,H,W) uint8 CUDA

                # Write into a new owned GPU TensorFrame.
                # reserve_gpu_plane allocates a fresh cudaMalloc buffer.
                # torch.from_dlpack gives a writable view; copy_ fills it.
                # The C++ Plane.d_data_ remains valid after the DLPack capsule
                # is consumed by torch, so t2d can read it downstream.
                out = limef.TensorFrame()
                out.reserve_gpu_plane(0, [C, H, W], 'uint8')
                torch.from_dlpack(out.planes[0]).copy_(blurred)
                out.timestamp = frame.timestamp
                out.slot      = frame.slot
                client.push(out)
            else:
                # No processing (or torch unavailable) — pass frame straight through.
                # Warn once if a CPU frame is about to enter the CUDA pipeline.
                if (isinstance(frame, limef.TensorFrame)
                        and not frame.is_gpu
                        and not warned_cpu_push[0]):
                    print("WARNING: pushing CPU TensorFrame into a CUDA/NVENC pipeline — "
                          "TensorToDecodedFrameFilter will output GBRP instead of "
                          "CUDA NV12; NVENC will produce wrong colours. "
                          "Ensure your processing keeps frames on GPU.",
                          file=sys.stderr)
                    warned_cpu_push[0] = True
                client.push(frame)

            if frame_count[0] % 100 == 1:
                elapsed = time.monotonic() - t_start[0]
                print(f"  frame #{frame_count[0]:5d}"
                      f"  gpu={frame.is_gpu}"
                      f"  ts={frame.timestamp / 1e6:7.3f} s"
                      f"  elapsed={elapsed:.1f} s"
                      f"  fps={frame_count[0] / max(elapsed, 1e-9):.1f}")

    consumer_thread = threading.Thread(target=consumer, daemon=True,
                                       name='limef-tensor-consumer')

    # ── Start (downstream first, then upstream) ────────────────────────────────
    print("Starting RTSP server ...")
    rtsp.start()
    time.sleep(0.1)

    rtsp.expose(SLOT, url_tail)
    time.sleep(0.05)

    print("Starting consumer thread ...")
    consumer_thread.start()

    print("Starting USB camera ...")
    camera.start()
    time.sleep(0.5)

    print(f"\nReady!  Connect with:")
    print(f"  ffplay rtsp://localhost:{port}{url_tail}")
    print(f"  ffplay -rtsp_transport tcp rtsp://localhost:{port}{url_tail}")
    print(f"  ffplay -fflags nobuffer -flags low_delay -framedrop "
          f"-probesize 32 -analyzeduration 0 rtsp://localhost:{port}{url_tail}\n")

    # ── Main loop ──────────────────────────────────────────────────────────────
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down...")

    # ── Cleanup (upstream first, then downstream) ──────────────────────────────
    stop_event.set()

    print("Stopping USB camera ...")
    try:
        camera.stop()
    except KeyboardInterrupt:
        print("Interrupted — forcing exit.")
        sys.exit(1)

    # Wait for the consumer thread to exit before destroying C++ objects.
    # Same rationale as pipeline_test_thread_tensor.py: pull() may be blocking
    # in fifo_.read() with the GIL released; destroying TensorFrameFifo while
    # a thread waits on its condition variable → std::terminate().
    while consumer_thread.is_alive():
        try:
            consumer_thread.join(timeout=TIMEOUT_MS / 1000 + 0.5)
        except KeyboardInterrupt:
            pass

    print("Stopping RTSP server ...")
    try:
        rtsp.stop()
    except KeyboardInterrupt:
        print("Interrupted — forcing exit.")
        sys.exit(1)

    elapsed = time.monotonic() - t_start[0]
    print(f"\nDone.  {elapsed:.1f} s, {frame_count[0]} frames"
          f"  ({frame_count[0] / max(elapsed, 1e-9):.1f} fps avg)")


if __name__ == '__main__':
    main()
