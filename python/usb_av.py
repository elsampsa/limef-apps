#!/usr/bin/env python3
"""
apps/python/usb_av.py

USB webcam + microphone → one RTSP stream with video (VP9) and audio (Opus).

A hardware demo (needs a V4L2 camera and an ALSA capture device), not a unit test.

    [C++] USBCameraThread → SwScaleFrameFilter(YUV420P) → EncodingFrameFilter(VP9, libvpx) ─┐
                                                                                            ├→ CombineFrameFilter
    [C++] AvDeviceThread(alsa) → InfoFrameFilter → AudioTranscodeFrameFilter(Opus) ─────────┘
        → RTSPMuxerFrameFilter → RTSPServerThread

The two sources run in their own threads and both write into the same CombineFrameFilter, which announces
one stream with a video and an audio track.  If the microphone is absent or busy, the video starts anyway
after --wait-ms and the audio joins in when the device shows up (AvDeviceThread retries with a backoff).

The microphone is opened as a native "hw:" device (stereo, 48 kHz): "plughw:" would resample inside
libasound, and its speex resampler plugin crashes on some systems.  Find the device with ``arecord -l``:
"card 2: Camera ... device 0" is ``hw:2,0``.

Usage:
    python3 apps/python/usb_av.py                      # /dev/video0, hw:2,0
    python3 apps/python/usb_av.py --audio-device hw:1,0 --video-device /dev/video2

Then connect with (VLC and ffplay both do VP9 + Opus):
    ffplay -rtsp_transport tcp rtsp://localhost:8554/live/stream

Press Ctrl+C to stop.
"""

import argparse
import select
import threading
import time

import limef.core as limef


def main():
    p = argparse.ArgumentParser(
        description='limef: USB camera + ALSA microphone -> RTSP (VP9 + Opus)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-d', '--video-device', default='/dev/video0', metavar='DEV',
                   help='V4L2 camera device')
    p.add_argument('-a', '--audio-device', default='hw:2,0', metavar='DEV',
                   help='ALSA capture device (hw:CARD,DEV; see arecord -l)')
    p.add_argument('--audio-rate', type=int, default=48000,
                   help='capture sample rate: must be one the device supports natively')
    p.add_argument('--audio-channels', type=int, default=2,
                   help='capture channels: must be a count the device supports natively (USB mics often: 2)')
    p.add_argument('-w', '--width', type=int, default=640)
    p.add_argument('-H', '--height', type=int, default=480)
    p.add_argument('-f', '--fps', type=int, default=30)
    p.add_argument('--bitrate', type=int, default=1_500_000,
                   help='VP9 target bitrate in bits/sec')
    p.add_argument('--audio-bitrate', type=int, default=64_000,
                   help='Opus target bitrate in bits/sec')
    p.add_argument('--wait-ms', type=int, default=2000,
                   help='how long to wait for the other of video/audio before streaming the one that is there '
                        '(0: don\'t wait, <0: wait for both for ever)')
    p.add_argument('-p', '--port', type=int, default=8554, help='RTSP server port')
    p.add_argument('--url-tail', default='/live/stream', metavar='PATH', help='RTSP URL path component')
    p.add_argument('-t', '--duration', type=float, default=0.0, metavar='SECS',
                   help='stop after N seconds (0 = run until Ctrl-C)')
    args = p.parse_args()

    SLOT = 1  # both sources must use the same slot: CombineFrameFilter doesn't rewrite them
    port = args.port
    url_tail = args.url_tail

    print('==============================================')
    print('  USB camera + microphone -> RTSP (VP9 + Opus)')
    print('==============================================')
    print(f'Video:      {args.video_device}  {args.width}x{args.height} @ {args.fps} fps  VP9 {args.bitrate} bit/s')
    print(f'Audio:      alsa {args.audio_device}  {args.audio_rate} Hz  {args.audio_channels} ch  Opus {args.audio_bitrate} bit/s')
    print(f'URL:        rtsp://localhost:{port}{url_tail}')
    print('==============================================')
    print('Press Ctrl+C to stop\n')

    # ── video: camera → YUV420P → VP9 ──────────────────────────────────────────
    cam_ctx = limef.USBCameraContext(args.video_device, SLOT)
    cam_ctx.width = args.width
    cam_ctx.height = args.height
    cam_ctx.fps = args.fps
    cam_ctx.capture_format = limef.AV_PIX_FMT_YUYV422   # the native format of most webcams
    camera = limef.USBCameraThread('usb-camera', cam_ctx)
    scale = limef.SwScaleFrameFilter('scale', limef.AV_PIX_FMT_YUV420P)

    enc_params = limef.FFmpegEncoderParams()
    enc_params.codec_id = limef.AV_CODEC_ID_VP9
    enc_params.bitrate = args.bitrate
    enc_params.gop_size = args.fps                       # a keyframe every second: clients can join quickly
    enc_params.options = {                               # libvpx-vp9: live settings
        'deadline': 'realtime',
        'cpu-used': '6',
        'lag-in-frames': '0',
        'row-mt': '1',
    }
    encoder = limef.EncodingFrameFilter('encoder', enc_params)

    # ── audio: alsa → Opus ─────────────────────────────────────────────────────
    mic_ctx = limef.AvDeviceContext('alsa', args.audio_device, SLOT)
    mic_ctx.options = {'sample_rate': str(args.audio_rate), 'channels': str(args.audio_channels)}
    mic = limef.AvDeviceThread('microphone', mic_ctx)
    efd = limef.EventFd()
    events = limef.InfoFrameFilter('mic-events', efd)    # "connected" / "lost" of the microphone; passes all else on
    opus = limef.AudioTranscodeFrameFilter(
        'opus', limef.AudioTranscodeContext.Opus(args.audio_bitrate))

    # ── join, mux, serve ───────────────────────────────────────────────────────
    combine = limef.CombineFrameFilter('combine', args.wait_ms)
    rtp_muxer = limef.RTSPMuxerFrameFilter('rtp-muxer')
    rtsp = limef.RTSPServerThread('rtsp-server', port=port, stack_size=30, fifo_size=100)

    camera.cc(scale).cc(encoder).cc(combine)
    mic.cc(events).cc(opus).cc(combine)
    combine.cc(rtp_muxer).cc(rtsp.getInput())

    # ── microphone events (connected / lost) ───────────────────────────────────
    stop_event = threading.Event()

    def watch_events():
        while not stop_event.is_set():
            r, _, _ = select.select([efd.getFd()], [], [], 0.2)
            if not r:
                continue
            efd.clear()
            while events.hasMessage():
                print(f'  [microphone] {events.popMessage()}')

    watcher = threading.Thread(target=watch_events, daemon=True, name='mic-events')

    # ── start: downstream first, then the sources ──────────────────────────────
    print('Starting RTSP server ...')
    rtsp.start()
    time.sleep(0.1)
    rtsp.expose(SLOT, url_tail)
    watcher.start()
    print('Starting microphone ...')
    mic.start()
    print('Starting USB camera ...')
    camera.start()
    time.sleep(0.5)

    print('\nReady!  Connect with:')
    print(f'  ffplay -rtsp_transport tcp rtsp://localhost:{port}{url_tail}')
    print(f'  vlc rtsp://localhost:{port}{url_tail}\n')

    t0 = time.monotonic()
    try:
        while args.duration <= 0 or time.monotonic() - t0 < args.duration:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    print('\nStopping ...')
    stop_event.set()
    camera.stop()
    mic.stop()
    rtsp.stop()
    watcher.join(timeout=1.0)
    print(f'Dropped by the combiner before the stream was announced: {combine.getDropped()}')


if __name__ == '__main__':
    main()
