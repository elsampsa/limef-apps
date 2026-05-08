#!/usr/bin/env python3
"""
apps/python/usb_pipeline.py

USB camera → CPU tensorframe → RTSP pipeline with optional Python frame processing.

All processing uses CPU tensors (CpuSwScaleConverter path) — no GPU upload before the
tensor stage.  Use this to:
  - Test and develop the CPU tensorframe pipeline on any Linux desktop
  - Deploy to Raspberry Pi (with --encoder v4l2m2m for hardware H.264 encoding)

For desktop testing NVENC is available as the encoder (--encoder nvenc, default).
FFmpegEncoder handles the CPU→GPU upload internally (av_hwframe_transfer_data) so no
explicit UploadGPUFrameFilter is needed — the pipeline is identical for both encoders.

Pipeline (passthrough, both encoders):
    [C++] USBCameraThread
        → DecodedToTensorFrameFilter(RGB, CPU via CpuSwScaleConverter)
        → TensorPythonInterface  ← Python (pass straight through)
        → TensorToDecodedFrameFilter(RGB → GBRP)
        → SwScaleFrameFilter(NV12)
        → EncodingFrameFilter(NVENC H264 or V4L2) → RTSPMuxerFrameFilter → RTSPServerThread

Usage:
    python3 apps/python/usb_pipeline.py [options]
    python3 apps/python/usb_pipeline.py --modify               # CPU Gaussian blur
    python3 apps/python/usb_pipeline.py --encoder v4l2m2m      # RPi / vicodec

Then connect with:
    ffplay rtsp://localhost:8554/live/stream

Press Ctrl+C to stop.
"""

import sys
import time
import argparse
import threading

import numpy as np
import limef

try:
    import torch
    import torch.nn.functional as F_torch
    _TORCH = True
except ImportError:
    _TORCH = False


_V4L2_CODEC_FOURCCS = {
    'fwht': limef.V4L2_PIX_FMT_FWHT,   # vicodec (laptop testing)
    'h264': limef.V4L2_PIX_FMT_H264,   # Jetson / RPi
    'h265': limef.V4L2_PIX_FMT_HEVC,   # Jetson / RPi
}


def _make_gauss_kernel(device):
    """Build a 15×15 Gaussian kernel matching OpenCV GaussianBlur(15,15,0)."""
    ksize = 15
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    x = torch.arange(ksize, dtype=torch.float32, device=device) - ksize // 2
    gauss = torch.exp(-x.pow(2.0) / (2.0 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss.outer(gauss)
    return kernel_2d.view(1, 1, ksize, ksize)


def _build_encoder(args):
    """Return a configured EncodingFrameFilter based on --encoder choice.

    NVENC: FFmpegEncoder handles CPU→GPU upload internally (av_hwframe_transfer_data)
    so no explicit UploadGPUFrameFilter is needed before the encoder.
    """
    if args.encoder == 'nvenc':
        enc_params              = limef.FFmpegEncoderParams()
        enc_params.codec_id     = limef.AV_CODEC_ID_H264
        enc_params.hw_accel     = limef.HWACCEL_CUDA
        enc_params.bitrate      = args.bitrate
        enc_params.preset       = 'p1'
        enc_params.tune         = 'ull'
        enc_params.max_b_frames = 0
        enc_params.gop_size     = args.fps // 2
        return limef.EncodingFrameFilter('encoder', enc_params)
    else:  # v4l2m2m
        v4l2_params              = limef.V4L2EncoderParams()
        v4l2_params.device       = args.enc_device
        v4l2_params.codec_fourcc = _V4L2_CODEC_FOURCCS[args.enc_codec]
        v4l2_params.bitrate      = args.bitrate
        v4l2_params.gop_size     = args.fps // 2
        return limef.EncodingFrameFilter('encoder', v4l2_params)


def main():
    p = argparse.ArgumentParser(
        description='limef USB camera → CPU tensors → RTSP (NVENC or V4L2 M2M encoder)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-m', '--modify',     action='store_true',
                   help='enable Gaussian blur in Python (CPU, via torch)')
    p.add_argument('-d', '--device',     default='/dev/video0', metavar='DEV',
                   help='V4L2 camera device')
    p.add_argument('-p', '--port',       type=int, default=8554,
                   help='RTSP server port')
    p.add_argument('-w', '--width',      type=int, default=640)
    p.add_argument('-H', '--height',     type=int, default=480)
    p.add_argument('-f', '--fps',        type=int, default=30)
    p.add_argument('--bitrate',          type=int, default=4_000_000,
                   help='encoder target bitrate in bits/sec')
    p.add_argument('--url-tail',         default='/live/stream', metavar='PATH',
                   help='RTSP URL path component')
    p.add_argument('--encoder',          choices=['nvenc', 'v4l2m2m'], default='nvenc',
                   help='encoder backend: nvenc (desktop) or v4l2m2m (RPi / vicodec)')
    p.add_argument('--enc-device',       default='/dev/video2', metavar='DEV',
                   help='V4L2 encoder device (v4l2m2m only; vicodec: /dev/video2, RPi: /dev/video11)')
    p.add_argument('--enc-codec',        choices=['fwht', 'h264', 'h265'], default='h264',
                   help='V4L2 output codec (v4l2m2m only; fwht for vicodec laptop testing)')
    args = p.parse_args()

    port     = args.port
    url_tail = args.url_tail
    SLOT     = 1
    TIMEOUT_MS = 200

    print("==============================================")
    print("  USB Camera → CPU tensors → RTSP")
    print("==============================================")
    print(f"Device:     {args.device}")
    print(f"Resolution: {args.width}x{args.height} @ {args.fps} fps")
    print(f"Port:       {port}")
    print(f"URL:        rtsp://localhost:{port}{url_tail}")
    print(f"Encoder:    {args.encoder}"
          + (f"  device={args.enc_device}  codec={args.enc_codec}"
             if args.encoder == 'v4l2m2m' else ''))
    print(f"Modify:     {args.modify}  (CPU Gaussian blur 15×15 in Python)")
    print("==============================================")
    print("Press Ctrl+C to stop\n")

    # ── C++ camera source ──────────────────────────────────────────────────────
    # Camera emits native YUYV422; CpuSwScaleConverter in DecodedToTensorFrameFilter
    # handles any CPU pixel format directly.
    cam_ctx                = limef.USBCameraContext(args.device, SLOT)
    cam_ctx.width          = args.width
    cam_ctx.height         = args.height
    cam_ctx.fps            = args.fps
    cam_ctx.capture_format = limef.AV_PIX_FMT_YUYV422

    camera = limef.USBCameraThread('usb-camera', cam_ctx)

    # ── C++ upstream chain (before Python visit) ───────────────────────────────
    # No GPU upload — frames stay on CPU. CpuSwScaleConverter handles NV12 directly.
    d2t = limef.DecodedToTensorFrameFilter('d2t', limef.CHANNEL_ORDER_RGB)

    # ── TensorPythonInterface ─────────────────────────────────────────────────
    # hw_accel=HWACCEL_SW: CPU TensorFrames stay on CPU at the thread boundary.
    # leaky=True: drop frames if Python loop falls behind.
    pyf    = limef.TensorPythonInterface(stack_size=10, leaky=True,
                                         hw_accel=limef.HWACCEL_SW, fifo_size=0)
    client = pyf.client()

    # ── C++ downstream chain (after Python visit) ──────────────────────────────
    # T2D CPU path outputs GBRP. SwScale converts to NV12 for the encoder.
    t2d      = limef.TensorToDecodedFrameFilter('t2d', limef.CHANNEL_ORDER_RGB)
    swscale  = limef.SwScaleFrameFilter('swscale', limef.AV_PIX_FMT_NV12)

    encoder = _build_encoder(args)

    rtp_muxer = limef.RTSPMuxerFrameFilter('rtp-muxer')
    rtsp      = limef.RTSPServerThread('rtsp-server', port=port, stack_size=30, fifo_size=100)

    # ── Wire the pipeline ──────────────────────────────────────────────────────
    camera.cc(d2t).cc(pyf.getInput())
    pyf.getOutput().cc(t2d).cc(swscale).cc(encoder).cc(rtp_muxer).cc(rtsp.getInput())

    # ── Python consumer thread ─────────────────────────────────────────────────
    stop_event  = threading.Event()
    frame_count = [0]
    t_start     = [time.monotonic()]

    gauss_kernel = [None]

    def consumer():
        while not stop_event.is_set():
            frame = client.pull(timeout_ms=TIMEOUT_MS)

            if frame is None:
                continue

            if isinstance(frame, limef.StreamFrame):
                client.push(frame)
                continue

            if not isinstance(frame, limef.TensorFrame):
                continue

            frame_count[0] += 1

            if args.modify and _TORCH:
                # CPU Gaussian blur via torch.
                # frame.planes[0] is a zero-copy numpy view (CPU TensorFrame).
                arr = frame.planes[0]                    # numpy (3,H,W) uint8
                t   = torch.from_numpy(arr).float()      # CPU tensor, same memory
                C, H, W = t.shape

                if gauss_kernel[0] is None:
                    gauss_kernel[0] = _make_gauss_kernel(t.device).expand(C, 1, 15, 15).contiguous()

                blurred = F_torch.conv2d(
                    t.unsqueeze(0), gauss_kernel[0], padding=7, groups=C
                ).squeeze(0).clamp(0, 255).to(torch.uint8)

                out = limef.TensorFrame()
                out.reserve_cpu_plane(0, [C, H, W], 'uint8')
                np.copyto(out.planes[0], blurred.numpy())
                out.timestamp = frame.timestamp
                out.slot      = frame.slot
                client.push(out)
            else:
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

    # ── Cleanup ────────────────────────────────────────────────────────────────
    stop_event.set()

    print("Stopping USB camera ...")
    try:
        camera.stop()
    except KeyboardInterrupt:
        print("Interrupted — forcing exit.")
        sys.exit(1)

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
