#!/usr/bin/env python3
"""
apps/python/jetson_cam.py

CSI camera (ArgusCameraThread, CUDA output) → RTSP streaming demo for Jetson Orin.

Encoder choices (--codec):

  vp8    FFmpeg libvpx VP8 software encoder  [default]
         Set --bitrate high (default 8 Mbps) — high bitrate = less compression work
         = fastest CPU path.

  mjpeg  FFmpeg MJPEG software encoder (all-intra, quality-based, ignores --bitrate)
         Faster than VP8 on ARM; not suitable for WebRTC.

  v4l2   V4L2 M2M hardware encoder (H.264/H.265)
         Requires a Jetson with a working NVENC V4L2 node (e.g. Orin AGX/NX).
         Orin Nano: NVENC is disabled in the BSP.

Pipeline (vp8 / mjpeg):

    ArgusCameraThread [CUDA, NV12, native sensor resolution]
        → CUDAScaleFrameFilter  (scale to --width x --height; NV12→YUV420P on GPU)
        → DecodedDownloadFrameFilter  (CUDA YUV420P → CPU YUV420P)
        → EncodingFrameFilter   (vp8 or mjpeg)
        → RTSPMuxerFrameFilter
        → RTSPServerThread

Usage:
    python3 apps/python/jetson_cam.py [options]

Then connect from another machine:
    ffplay rtsp://<jetson-ip>:8554/live/stream
    ffplay -rtsp_transport tcp rtsp://<jetson-ip>:8554/live/stream

Press Ctrl+C to stop.
"""

import sys
import time
import socket
import argparse

import limef

sys.stdout.reconfigure(line_buffering=True)


# ── LAN IP helpers ─────────────────────────────────────────────────────────────

def _lan_ip():
    """Return the IP address on the LAN interface (used to reach the internet)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


# ── Encoder builders ────────────────────────────────────────────────────────────

def _build_vp8_encoder(bitrate, fps):
    """FFmpeg libvpx VP8 software encoder (input: CPU YUV420P).

    High bitrate = less compression work = fastest CPU path on ARM.
    """
    p          = limef.FFmpegEncoderParams()
    p.codec_id = limef.AV_CODEC_ID_VP8
    p.bitrate  = bitrate
    p.gop_size = max(1, fps // 2)
    return limef.EncodingFrameFilter('encoder', p)


def _build_mjpeg_encoder(fps):
    """FFmpeg MJPEG software encoder (input: CPU YUV420P, quality-based).

    All-intra codec; faster than VP8 on ARM.  Bitrate is not used.
    The encoder remaps YUV420P→YUVJ420P internally.
    """
    p          = limef.FFmpegEncoderParams()
    p.codec_id = limef.AV_CODEC_ID_MJPEG
    p.gop_size = 1
    return limef.EncodingFrameFilter('encoder', p)


def _build_v4l2_encoder(args):
    """V4L2 M2M H.264 hardware encoder (input: CPU NV12).

    Auto-discovers the best encoder device unless --enc-device / --enc-codec
    are provided.  Exits if no V4L2 encoder is found.
    """
    _V4L2_FOURCCS = {
        'h264': limef.V4L2_PIX_FMT_H264,
        'h265': limef.V4L2_PIX_FMT_HEVC,
        'fwht': limef.V4L2_PIX_FMT_FWHT,
    }

    devs = limef.v4l2.scan_devices()
    sel  = limef.v4l2.select_best_encoder(devs)
    if sel is None and args.enc_device is None:
        print("ERROR: no V4L2 encoder found on this board.")
        print("  Orin Nano: NVENC is disabled in the BSP — use --codec vp8 or --codec mjpeg.")
        print("  Other Jetson: check /dev/video10,11 and v4l2-ctl --list-devices")
        sys.exit(1)

    device     = args.enc_device or sel.enc_device
    fourcc     = _V4L2_FOURCCS[args.enc_codec] if args.enc_codec else (sel.codec_fourcc if sel else limef.V4L2_PIX_FMT_H264)
    codec_name = args.enc_codec or (sel.codec_name if sel else 'h264')

    p              = limef.V4L2EncoderParams()
    p.device       = device
    p.codec_fourcc = fourcc
    p.bitrate      = args.bitrate
    p.gop_size     = max(1, args.fps // 2)
    return limef.EncodingFrameFilter('encoder', p), device, codec_name


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Limef Jetson CSI camera → RTSP demo (ArgusCameraThread)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--codec',         choices=['vp8', 'mjpeg', 'v4l2'], default='vp8',
                   help='Encoder: vp8=libvpx SW, mjpeg=MJPEG SW (quality-based), v4l2=HW H.264 (needs NVENC)')
    p.add_argument('--camera',        type=int, default=0,    metavar='IDX',
                   help='Argus camera device index')
    p.add_argument('--sensor-mode',   type=int, default=0,    metavar='MODE',
                   help='Argus sensor mode index (0=full resolution; run --list-modes to enumerate)')
    p.add_argument('--width',         type=int, default=640,
                   help='Encode width (CUDAScaleFrameFilter target)')
    p.add_argument('--height',        type=int, default=360,
                   help='Encode height (CUDAScaleFrameFilter target)')
    p.add_argument('--fps',           type=int, default=5,
                   help='Nominal frame rate (used for GOP sizing and sensor)')
    p.add_argument('--bitrate',       type=int, default=8_000_000,
                   help='Encoder bitrate in bits/sec (VP8 only; set high for fastest ARM encoding; ignored for MJPEG)')
    p.add_argument('--port',          type=int, default=8554,
                   help='RTSP server port')
    p.add_argument('--url-tail',      default='/live/stream', metavar='PATH',
                   help='RTSP URL path')
    p.add_argument('--enc-device',    default=None, metavar='DEV',
                   help='(v4l2) V4L2 encoder device; auto-discovered if not set')
    p.add_argument('--enc-codec',     choices=['h264', 'h265', 'fwht'], default=None,
                   help='(v4l2) V4L2 output codec; auto-discovered if not set')
    p.add_argument('--list-modes',    action='store_true',
                   help='List available Argus cameras and sensor modes, then exit')
    args = p.parse_args()

    if args.list_modes:
        modes = limef.argus.list_sensor_modes()
        if not modes:
            print("No cameras found (is nvargus-daemon running?)")
            sys.exit(1)
        for m in modes:
            print(f"  Camera {m.camera_idx}  Mode {m.mode_idx}: {m.width}x{m.height}")
        sys.exit(0)

    lan_ip   = _lan_ip()
    port     = args.port
    url_tail = args.url_tail
    SLOT     = 1

    # ── Build encoder ──────────────────────────────────────────────────────────
    enc_info = ''
    if args.codec == 'vp8':
        encoder  = _build_vp8_encoder(args.bitrate, args.fps)
        enc_info = f'libvpx VP8 (software)  bitrate={args.bitrate // 1000} kbps'
    elif args.codec == 'mjpeg':
        encoder  = _build_mjpeg_encoder(args.fps)
        enc_info = 'MJPEG (software, quality-based)'
    else:
        encoder, enc_device, enc_codec_name = _build_v4l2_encoder(args)
        enc_info = f'V4L2 H.264  device={enc_device}  codec={enc_codec_name}'

    # ── Scale params (GPU): NV12→YUV420P in one GPU pass ──────────────────────
    scale_params               = limef.CUDAScaleParams(args.width, args.height)
    scale_params.output_format = limef.AV_PIX_FMT_YUV420P  # works for both VP8 and MJPEG

    # ── Print banner ───────────────────────────────────────────────────────────
    print("==============================================")
    print("  Jetson CSI Camera → RTSP")
    print("==============================================")
    print(f"Camera:      index={args.camera}  sensor_mode={args.sensor_mode}")
    print(f"Encode res:  {args.width}x{args.height} @ {args.fps} fps (nominal)")
    print(f"Encoder:     {enc_info}")
    print(f"RTSP port:   {port}")
    print(f"LAN IP:      {lan_ip}")
    print(f"URL:         rtsp://{lan_ip}:{port}{url_tail}")
    print("==============================================")
    print("Connect with:")
    print(f"  ffplay rtsp://{lan_ip}:{port}{url_tail}")
    print(f"  ffplay -rtsp_transport tcp rtsp://{lan_ip}:{port}{url_tail}")
    print("==============================================")
    print("Press Ctrl+C to stop\n")

    # ── Build pipeline ─────────────────────────────────────────────────────────
    ctx                   = limef.ArgusCameraContext()
    ctx.camera_index      = args.camera
    ctx.sensor_mode_index = args.sensor_mode
    ctx.output_location   = limef.HWACCEL_CUDA
    ctx.fps               = args.fps

    camera   = limef.ArgusCameraThread('argus-cam', ctx)
    scale    = limef.CUDAScaleFrameFilter('scale', scale_params)
    download = limef.DecodedDownloadFrameFilter('download')
    dump1    = limef.DumpFrameFilter('cpu-decoded', verbose=False)
    dump2    = limef.DumpFrameFilter('encoded',     verbose=False)
    rtp      = limef.RTSPMuxerFrameFilter('rtp-muxer')
    rtsp     = limef.RTSPServerThread('rtsp-server', port=port, stack_size=30, fifo_size=100)

    camera.cc(scale).cc(download).cc(dump1).cc(encoder).cc(dump2).cc(rtp).cc(rtsp.getInput())

    # ── Start (downstream first) ───────────────────────────────────────────────
    print("Starting RTSP server ...")
    rtsp.start()
    time.sleep(0.1)
    rtsp.expose(SLOT, url_tail)
    time.sleep(0.05)

    print("Starting Argus camera ...")
    camera.start()
    time.sleep(1.0)

    print(f"\nReady.  Connect from another device:")
    print(f"  ffplay rtsp://{lan_ip}:{port}{url_tail}\n")

    # ── Main loop ──────────────────────────────────────────────────────────────
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down ...")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    print("Stopping camera ...")
    try:
        camera.stop()
    except KeyboardInterrupt:
        sys.exit(1)

    print("Stopping RTSP server ...")
    try:
        rtsp.stop()
    except KeyboardInterrupt:
        sys.exit(1)

    print("Done.")


if __name__ == '__main__':
    main()
