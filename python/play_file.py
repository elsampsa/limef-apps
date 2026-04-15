#!/usr/bin/env python3
"""
apps/python/play_file.py

Media file player: reads a local file and displays it in a window.

Pipeline:
  MediaFileThread → [dump_src] → DecodingFrameFilter → [dump_dec] → <presenter>

MediaFileThread internally chains MediaFileThread1 (file reader + AAC transcode)
and OrderedPacketBufferThread (DTS ordering), so no separate buffer thread is
needed here.

Usage:
    python3 apps/python/play_file.py --file video.mp4

Options:
    --file       Input file path (required)
    --loop       Loop the file: pause this many ms at EOF before restarting
                 (default: -1 = no loop).  Use 0 for gapless looping.
    --decode     Decoder backend: sw (default), cuda, vaapi
    --buffer     De-jitter buffer in milliseconds (default: 0 = disabled)
                 Frames whose absolute timestamp is older than this are dropped
                 by the presenter.  Useful if the file's timestamps have gaps.
    --presenter  Window backend: sdl (default) or glx (OpenGL/X11, CUDA zero-copy)
    --bypass-compositor
                 (GLX only) Set _NET_WM_BYPASS_COMPOSITOR hint — needed on KWin/PRIME
                 when NVIDIA GLX and the compositor run on different GPUs.
    --verbose    Print one line per frame at each pipeline stage.
                 Useful for diagnosing timestamp issues or pipeline stalls.

Press Ctrl+C to stop.
"""

import time
import argparse

import limef


_DECODE_MAP = {
    'sw':    limef.HWACCEL_SW,
    'cuda':  limef.HWACCEL_CUDA,
    'vaapi': limef.HWACCEL_VAAPI,
}


def main():
    p = argparse.ArgumentParser(
        description='Limef media file player',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--file', required=True, metavar='PATH',
                   help='input file, e.g. video.mp4')
    p.add_argument('--loop', type=int, default=-1, metavar='MS',
                   help='loop at EOF: pause this many ms then restart '
                        '(-1 = no loop, 0 = gapless)')
    p.add_argument('--decode', choices=['sw', 'cuda', 'vaapi'], default='sw',
                   help='decoder backend')
    p.add_argument('--buffer', type=int, default=0, metavar='MS',
                   help='drop frames older than this many ms '
                        '(0 = disabled)')
    p.add_argument('--presenter', choices=['glx', 'sdl'], default='sdl',
                   help='window backend: sdl (SDL2, default) or glx (OpenGL/X11, CUDA zero-copy)')
    p.add_argument('--bypass-compositor', dest='bypass_compositor',
                   action='store_true', default=False,
                   help='(GLX only) set _NET_WM_BYPASS_COMPOSITOR — needed on KWin/PRIME '
                        'when NVIDIA GLX and compositor run on different GPUs')
    p.add_argument('--verbose', action='store_true', default=False,
                   help='print one line per frame at each pipeline stage for debugging')
    args = p.parse_args()

    SLOT = 1

    # ── MediaFileContext ───────────────────────────────────────────────────────
    # fps=-1: feed packets at their natural playback speed (wall-clock paced).
    # loop: -1 = play once; >=0 = pause that many ms at EOF then restart.
    file_ctx = limef.MediaFileContext(args.file, SLOT)
    file_ctx.fps  = -1
    file_ctx.loop = args.loop

    # ── Build presenter ────────────────────────────────────────────────────────
    if args.presenter == 'glx':
        pctx = limef.PresenterContext()
        pctx.max_age_ms        = args.buffer
        pctx.bypass_compositor = args.bypass_compositor
        presenter = limef.GLXPresenterThread('presenter', pctx=pctx,
                        stack_size=20,  # a few screen-refresh cycles of headroom
                        fifo_size=40)   # evict oldest if display falls behind a burst
    else:
        if args.bypass_compositor:
            print("WARNING: --bypass-compositor is only applicable with --presenter glx, ignoring.")
        presenter = limef.SDLVideoPresenterThread(
            'presenter',
            max_age_ms=args.buffer,
            stack_size=20,  # a few screen-refresh cycles of headroom
            fifo_size=40,   # evict oldest if display falls behind a burst
        )

    # ── Build filter chain ─────────────────────────────────────────────────────
    #
    #   src → dump_src → dec → dump_dec → presenter
    #
    # MediaFileThread already contains an internal OrderedPacketBufferThread
    # (stack_size=30, leaky=False) so no separate buf thread is needed here.
    # Back-pressure is intentional for file playback: the source must never
    # outrun the decoder.
    #
    # presenter: DecodedFrame fifo (leaky=True, stack=20, fifo cap=40)
    #   stack_size=20: a few screen-refresh cycles of headroom.  For file playback
    #   the decoder feeds at a steady rate, so 20 slots is ample.
    #   fifo_size=40:  hard cap on queued-but-not-yet-displayed frames so memory
    #   stays bounded if the display momentarily falls behind; oldest is evicted.
    #   Tunable via stack_size= on the presenter constructor if needed.
    #
    # DumpFrameFilters are always in the chain; verbose=False makes them silent
    # pass-throughs with no overhead.  Use --verbose to activate both.
    dump_src = limef.DumpFrameFilter('dump_src', verbose=args.verbose)
    dec      = limef.DecodingFrameFilter('dec', hw_accel=_DECODE_MAP[args.decode])
    dump_dec = limef.DumpFrameFilter('dump_dec', verbose=args.verbose)

    src = limef.MediaFileThread('src', file_ctx)

    src.cc(dump_src)
    dump_src.cc(dec)
    dec.cc(dump_dec)
    dump_dec.cc(presenter.getInput())

    print("=================================")
    print("  File Player")
    print("=================================")
    print(f"File:      {args.file}")
    print(f"Loop:      {args.loop} ms" if args.loop >= 0 else "Loop:      off")
    print(f"Decode:    {args.decode}")
    print(f"Presenter: {args.presenter}"
          + (f"  (bypass_compositor={args.bypass_compositor})" if args.presenter == 'glx' else ""))
    print(f"Buffer:    {args.buffer} ms  (presenter max_age; 0 = disabled)")
    print(f"Verbose:   {args.verbose}  (dump_src / dump_dec)")
    print("=================================")
    print("Press Ctrl+C to stop\n")

    # ── Start threads (downstream first, source last) ─────────────────────────
    presenter.start()
    src.start()

    # ── Main loop ──────────────────────────────────────────────────────────────
    # Note: src.isRunning() stays True even after EOF because MediaFileThread is a
    # ComposeThread whose second sub-thread (OrderedPacketBufferThread) never stops
    # on its own — it blocks waiting for more input indefinitely.  Press Ctrl+C to stop.
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down...")

    # ── Stop (downstream first to relieve any backpressure) ───────────────────
    presenter.stop()
    src.stop()

    print("Done.")


if __name__ == '__main__':
    main()
