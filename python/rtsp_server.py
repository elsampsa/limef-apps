#!/usr/bin/env python3
"""
apps/python/rtsp_server.py

RTSP server with in-process Python frame processing via PythonInterface.

Full pipeline:
  Upstream   (C++):    MediaFileThread → Decode → SwScale(BGR24) → PythonInterface
  Consumer   (Python): pull → Gaussian blur → push
  Downstream (C++):    SwScale(YUV420P) → EncodingFrameFilter(H264) → RTPMuxerFrameFilter
                            → RTSPServerThread

Usage:
    python3 apps/python/rtsp_server.py --file fixtures/jontxu.mp4 --port 8554

Then connect with:
    ffplay rtsp://localhost:8554/live/stream
    ffplay -rtsp_transport tcp rtsp://localhost:8554/live/stream
    vlc rtsp://localhost:8554/live/stream

Press Ctrl+C to stop.

OpenCV is optional (install with pip install opencv-python).
Without it, frames are passed through without blurring.
"""

import os
import time
import argparse
import threading
from pathlib import Path

import limef
import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False


def main():
    p = argparse.ArgumentParser(
        description='limef RTSP server with Python frame processing (Gaussian blur)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-f', '--file',     default=os.environ.get('MEDIA_FILE'), metavar='PATH',
                   help='media file to stream (or set $MEDIA_FILE)')
    p.add_argument('-p', '--port',     type=int, default=8554,
                   help='RTSP server port')
    p.add_argument('--url-tail',       default='/live/stream', metavar='PATH',
                   help='RTSP URL path component')
    p.add_argument('-d', '--duration', type=float, default=0.0, metavar='SECS',
                   help='stop after N seconds (0 = run until Ctrl-C)')
    p.add_argument('--bitrate',        type=int, default=2_000_000,
                   help='H.264 target bitrate in bits/sec')
    args = p.parse_args()

    if not _CV2:
        print("WARNING: opencv-python not found — blur step will be skipped (pip install opencv-python)")

    if not args.file:
        p.error('--file PATH is required (or set $MEDIA_FILE)')
    media_file = str(Path(args.file).resolve())
    port       = args.port
    url_tail   = args.url_tail
    SLOT       = 1
    TIMEOUT_MS = 200

    print("=================================")
    print("  RTSP Server (Python)")
    print("=================================")
    print(f"Media file: {media_file}")
    print(f"Port:       {port}")
    print(f"URL:        rtsp://localhost:{port}{url_tail}")
    print(f"Bitrate:    {args.bitrate // 1000} kbps")
    print("=================================")
    print("Press Ctrl+C to stop\n")

    # ── Upstream filter chain ──────────────────────────────────────────────────
    # MediaFileThread → decode → scale_bgr → pyf.getInput()
    #
    # PythonInterface fifo: stack_size=10, leaky=True
    #   10 DecodedFrame slots are enough to absorb a pull() delay in the Python
    #   consumer loop.  leaky=True: drop frames if the Python thread falls behind
    #   rather than stalling the upstream decode/scale chain.
    dumpu = limef.DumpFrameFilter("upstream_dump")
    pyf    = limef.PythonInterface(stack_size=10, leaky=True, fifo_size=0)
    scale  = limef.SwScaleFrameFilter('scale_bgr', limef.AV_PIX_FMT_BGR24)
    decode = limef.DecodingFrameFilter('decode')
    decode.cc(scale).cc(pyf.getInput())

    # ── Downstream filter chain ────────────────────────────────────────────────
    # pyf.getOutput() → scale_yuv → encode → rtp_muxer → rtsp.getInput()
    #
    # scale_yuv:  BGR24 (from Python push) → YUV420P (required by H.264 encoder)
    # encode:     DecodedFrame → CodecFrame + PacketFrames
    # rtp_muxer:  CodecFrame + PacketFrames → SDPFrame + RTPPacketFrames
    # rtsp:       serves RTP packets to connected RTSP clients
    dumpd = limef.DumpFrameFilter("downstream_dump")
    scale_yuv = limef.SwScaleFrameFilter('scale_yuv', limef.AV_PIX_FMT_YUV420P)

    enc_params            = limef.FFmpegEncoderParams()
    enc_params.codec_id   = limef.AV_CODEC_ID_VP8
    enc_params.bitrate    = args.bitrate
    #enc_params.preset     = 'ultrafast'
    #enc_params.tune       = 'zerolatency'
    #enc_params.gop_size   = 30
    #enc_params.max_b_frames = 0   # no B-frames for low-latency live

    encode    = limef.EncodingFrameFilter('encode', enc_params)
    rtp_muxer = limef.RTPMuxerFrameFilter('rtp_muxer')
    # stack_size=30: absorbs I-frame bursts (several RTP packets in rapid succession).
    # fifo_size=100: cap on queued RTP packets; prevents unbounded growth under slow clients.
    # leaky is always False for RTSPServerThread — RTP sequence gaps corrupt the stream.
    rtsp      = limef.RTSPServerThread('rtsp_server', port=port, stack_size=30, fifo_size=100)
    
    # pyf.getOutput().cc(scale_yuv).cc(encode).cc(rtp_muxer).cc(dumpd).cc(rtsp.getInput())
    pyf.getOutput().cc(scale_yuv).cc(encode).cc(rtp_muxer).cc(rtsp.getInput())
    
    # ── Media source ───────────────────────────────────────────────────────────
    dumpm = limef.DumpFrameFilter("media-source-dump")
    ctx      = limef.MediaFileContext(media_file, SLOT)
    ctx.fps  = -1   # native playback speed
    ctx.loop = 0    # loop immediately at EOF
    thread   = limef.MediaFileThread('reader', ctx)
    thread.cc(decode)

    # ── RTSP callbacks (for logging) ───────────────────────────────────────────
    rtsp.onStreamRequired(
        lambda slot: print(f"[event] Client subscribed to slot {slot}"))
    rtsp.onStreamNotRequired(
        lambda slot: print(f"[event] No more clients on slot {slot}"))

    # ── Python consumer thread ─────────────────────────────────────────────────
    stop_event  = threading.Event()
    client      = pyf.client()
    video_count = [0]
    audio_count = [0]
    t_start     = [0.0]

    def consumer():
        while not stop_event.is_set():
            frame = client.pull(timeout_ms=TIMEOUT_MS)
            # print(">", frame)
            if frame is None:
                continue   # timeout — recheck stop_event

            if isinstance(frame, limef.StreamFrame):
                print(f"  StreamFrame  slot={frame.slot}  streams={len(frame.streams)}")
                for i, s in enumerate(frame.streams):
                    if s.codec_type == limef.AVMEDIA_TYPE_VIDEO:
                        print(f"    [{i}] video  {s.width}x{s.height}"
                              f"  fps={s.r_frame_rate[0]}/{s.r_frame_rate[1]}")
                    elif s.codec_type == limef.AVMEDIA_TYPE_AUDIO:
                        print(f"    [{i}] audio  rate={s.sample_rate}"
                              f"  ch={s.channels}")
                # Forward StreamFrame downstream so EncodingFrameFilter can configure
                client.push(frame)
                continue

            if not isinstance(frame, limef.DecodedFrame):
                continue

            if frame.is_video:
                video_count[0] += 1
                w, h = frame.width, frame.height

                # BGR24: single plane, shape (H, linesize) — copy before next pull()
                plane = np.array(frame.planes[0])
                img   = plane[:, :w * 3].reshape(h, w, 3)

                blurred = cv2.GaussianBlur(img, (21, 21), 0) if _CV2 else img

                # Construct a new owned DecodedFrame for push() downstream.
                # reserve_video() allocates AVFrame buffers; planes setter copies in.
                out = limef.DecodedFrame()
                out.reserve_video(w, h, frame.format)
                out.timestamp = frame.timestamp
                out.pts       = frame.pts
                out.slot      = frame.slot
                out.planes    = [np.ascontiguousarray(blurred.reshape(h, w * 3))]
                client.push(out)

                if video_count[0] % 100 == 1:
                    elapsed = time.monotonic() - t_start[0]
                    print(f"  video #{video_count[0]:5d}  {w}x{h}"
                          f"  ts={frame.timestamp / 1e6:7.3f} s"
                          f"  elapsed={elapsed:.1f} s")

            elif frame.is_audio:
                audio_count[0] += 1
                client.push(frame)   # pass audio downstream as-is

    consumer_thread = threading.Thread(
        target=consumer, daemon=True, name='limef-consumer')

    # ── Start everything ───────────────────────────────────────────────────────
    print("Starting RTSP server ...")
    rtsp.start()
    time.sleep(0.1)

    rtsp.expose(SLOT, url_tail)
    time.sleep(0.05)

    print("Starting media playback ...")
    thread.start()
    t_start[0] = time.monotonic()
    consumer_thread.start()

    time.sleep(1.0)   # wait for first frames to flow
    print(f"\nReady!  Connect with:")
    print(f"  ffplay rtsp://localhost:{port}{url_tail}")
    print(f"  ffplay -rtsp_transport tcp rtsp://localhost:{port}{url_tail}\n")

    # ── Main loop ──────────────────────────────────────────────────────────────
    # Let Python's default SIGINT handler raise KeyboardInterrupt — simpler and
    # reliably interruptible.  time.sleep() in a loop is fine for this purpose.
    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down...")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    stop_event.set()
    consumer_thread.join(timeout=2.0)
    elapsed = time.monotonic() - t_start[0]

    # Wrap each blocking C++ stop() so a second Ctrl+C forces an immediate exit.
    print("Stopping media playback ...")
    try:
        thread.stop()
    except KeyboardInterrupt:
        print("Interrupted during stop — forcing exit.")
        sys.exit(1)

    print("Stopping RTSP server ...")
    try:
        rtsp.stop()
    except KeyboardInterrupt:
        print("Interrupted during stop — forcing exit.")
        sys.exit(1)

    print(f"\nDone.  {elapsed:.1f} s, "
          f"{video_count[0]} video frames, "
          f"{audio_count[0]} audio chunks received")


if __name__ == '__main__':
    main()
