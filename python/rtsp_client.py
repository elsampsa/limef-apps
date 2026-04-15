#!/usr/bin/env python3
"""
apps/python/rtsp_client.py

RTSP / IP camera client: connects to an RTSP stream and displays it.

Pipeline:
  LiveStreamThread → [dump_live] → OrderedPacketBufferThread → [dump_buf]
    → DecodingFrameFilter → [dump_dec] → <presenter>

Usage:
    python3 apps/python/rtsp_client.py --rtsp rtsp://user:pass@192.168.1.10/stream

Options:
    --rtsp       RTSP URL (required)
    --timeout    Read timeout in seconds (default: 5)
    --use-ntp    Use NTP wall-clock from RTCP Sender Reports (default: off,
                 only enable if camera NTP is known to be reliable)
    --decode     Decoder backend: sw (default), cuda, vaapi
    --buffer     De-jitter buffer in milliseconds (default: 0 = disabled)
                 Frames whose absolute timestamp is older than this are dropped by
                 the presenter.  0 means all frames are displayed regardless of latency.
                 Note: the OrderedPacketBufferThread stack (fixed at 30 frames) provides
                 capacity for DTS reordering; the --buffer value controls the presenter's
                 max_age tolerance in wall-clock time.
    --presenter  Window backend: glx (default) or sdl
    --bypass-compositor
                 (GLX only) Set _NET_WM_BYPASS_COMPOSITOR hint — needed on KWin/PRIME
                 when NVIDIA GLX and the compositor run on different GPUs.
    --verbose    Print one line per frame at each pipeline stage (dump_live,
                 dump_buf, dump_dec).  Useful for diagnosing where the pipeline
                 stalls.

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
        description='Limef RTSP / IP camera client',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--rtsp', required=True,
                   help='RTSP URL, e.g. rtsp://user:pass@192.168.1.10/stream')
    p.add_argument('--timeout', type=int, default=5, metavar='SECS',
                   help='read timeout in seconds (LiveStreamContext.timeout_sec)')
    p.add_argument('--use-ntp', action='store_true', default=False,
                   help='use NTP wall-clock timestamps from RTCP Sender Reports '
                        '(only if camera NTP is known to be reliable; default: off)')
    p.add_argument('--decode', choices=['sw', 'cuda', 'vaapi'], default='sw',
                   help='decoder backend (default: sw)')
    p.add_argument('--buffer', type=int, default=0, metavar='MS',
                   help='de-jitter buffer: drop frames older than this many ms '
                        '(0 = disabled, keep all frames regardless of latency)')
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

    # ── LiveStreamContext ──────────────────────────────────────────────────────
    stream_ctx = limef.LiveStreamContext(args.rtsp, SLOT)
    stream_ctx.timeout_sec = args.timeout
    stream_ctx.use_ntp     = args.use_ntp

    # ── Build presenter ────────────────────────────────────────────────────────
    # max_future_age_ms: discard frames whose NTP timestamp is more than this
    # many ms in the future.  Protects against cameras that send wrong RTCP
    # wallclock (e.g. year 2030) after reconnect: without this guard those
    # frames fill pending_frames_ indefinitely and block the whole pipeline.
    # 10 s is a safe default — a legitimately buffered live stream would never
    # be more than a few seconds ahead of wall clock.
    MAX_FUTURE_AGE_MS = 10_000

    if args.presenter == 'glx':
        # PresenterContext bundles all GLX window options.
        # max_age_ms: drop frames whose NTP timestamp is older than this —
        # the de-jitter tolerance.  0 disables dropping.
        # bypass_compositor: set _NET_WM_BYPASS_COMPOSITOR on the window;
        # required on KWin + PRIME when the NVIDIA GPU renders into a GLX window
        # that the compositor (running on Intel/Mesa) would otherwise try to
        # re-composite — doing so breaks CUDA GL interop.
        pctx = limef.PresenterContext()
        pctx.max_age_ms         = args.buffer
        pctx.max_future_age_ms  = MAX_FUTURE_AGE_MS
        pctx.bypass_compositor  = args.bypass_compositor
        presenter = limef.GLXPresenterThread('presenter', pctx=pctx,
                        stack_size=20,  # a few screen-refresh cycles of headroom
                        fifo_size=40)   # evict oldest if display falls behind a burst
    else:
        # SDLVideoPresenterThread: simpler, cross-platform, no GLX/EGL setup.
        # GPU frames are downloaded to CPU before display — no zero-copy path.
        # bypass_compositor is not applicable for SDL.
        if args.bypass_compositor:
            print("WARNING: --bypass-compositor is only applicable with --presenter glx, ignoring.")
        presenter = limef.SDLVideoPresenterThread(
            'presenter',
            max_age_ms=args.buffer,
            max_future_age_ms=MAX_FUTURE_AGE_MS,
            stack_size=20,  # a few screen-refresh cycles of headroom
            fifo_size=40,   # evict oldest if display falls behind a burst
        )

    # ── Build filter chain ─────────────────────────────────────────────────────
    #
    #   live → dump_live → buf → dump_buf → dec → dump_dec → presenter
    #
    # Each consumer thread owns a FrameFifo (stack + fifo).  Key parameters:
    #
    #   stack_size: number of pre-allocated Frame slots.  The Frame *objects* are
    #       created upfront, but their data buffers (AVPacket payload, pixel planes)
    #       grow lazily on first use and are then reused.  So memory footprint
    #       stabilises after ~1 GOP, not at construction time.
    #       The stack is a hard cap on frames in flight; when exhausted the thread
    #       either drops (leaky=True) or blocks upstream (leaky=False).
    #
    #   leaky: for live streaming, drop rather than block — a missed display frame
    #       is acceptable; stalling the source thread is not.
    #       Exception: RTSPServerThread uses leaky=False because RTP is
    #       sequence-numbered and every dropped packet corrupts the stream.
    #
    # buf: OrderedPacketBufferThread
    #   stack_size=30: capacity for one full GOP at 25–60 fps plus network jitter.
    #   leaky=True: if the decoder stalls, drop incoming packets rather than
    #       blocking LiveStreamThread.
    #
    # presenter: DecodedFrame fifo (leaky=True, stack=20, fifo cap=40)
    #   stack_size=20: a few screen-refresh cycles of headroom (~667 ms at 30 fps).
    #   fifo_size=40:  hard cap on queued-but-not-yet-displayed frames; oldest is
    #       evicted when full so memory stays bounded under a decode burst.
    #   Both values are set via stack_size= on the presenter constructor.
    #
    # DumpFrameFilters are always in the chain; verbose=False makes them silent
    # pass-throughs with no overhead.  Use --verbose to activate all three.
    dump_live = limef.DumpFrameFilter('dump_live', verbose=args.verbose)
    buf       = limef.OrderedPacketBufferThread('buf',
                    stack_size=30,   # one GOP at 25-60 fps
                    leaky=True,      # drop if decoder stalls rather than blocking live source
                    fifo_size=0)     # unlimited queue; back-pressure via stack only
    dump_buf  = limef.DumpFrameFilter('dump_buf',  verbose=args.verbose)
    dec       = limef.DecodingFrameFilter('dec', hw_accel=_DECODE_MAP[args.decode])
    dump_dec  = limef.DumpFrameFilter('dump_dec',  verbose=args.verbose)

    live = limef.LiveStreamThread('live', stream_ctx)

    live.cc(dump_live)
    dump_live.cc(buf.getInput())
    buf.cc(dump_buf)
    dump_buf.cc(dec)
    dec.cc(dump_dec)
    dump_dec.cc(presenter.getInput())

    # ── Status callbacks ───────────────────────────────────────────────────────
    # live.onConnected(lambda slot: print(f"[live] connected  slot={slot}"))
    # live.onLost(     lambda slot: print(f"[live] stream lost slot={slot}  (will reconnect)"))

    print("=================================")
    print("  RTSP Client")
    print("=================================")
    print(f"URL:       {args.rtsp}")
    print(f"Decode:    {args.decode}")
    print(f"NTP:       {args.use_ntp}")
    print(f"Presenter: {args.presenter}"
          + (f"  (bypass_compositor={args.bypass_compositor})" if args.presenter == 'glx' else ""))
    print(f"Buffer:    {args.buffer} ms  (presenter max_age; 0 = disabled)")
    print(f"Verbose:   {args.verbose}  (dump_live / dump_buf / dump_dec)")
    print("=================================")
    print("Press Ctrl+C to stop\n")

    # ── Start threads (downstream first, source last) ─────────────────────────
    presenter.start()
    buf.start()
    live.start()

    # ── Main loop ──────────────────────────────────────────────────────────────
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down...")

    # ── Stop (downstream first so backpressure is relieved before waitStop) ──
    # Calling requestStop() now (1) sets stop_requested_ directly, (2) closes
    # the thread's own fifo.  But a thread blocked in output_ff.go() →
    # downstream_fifo.writeCopy() can only be unblocked by closing *that*
    # downstream fifo — which happens when we requestStop() the downstream
    # thread.  Stopping presenter first closes its fifo and unblocks buf if
    # buf is stuck in writeCopy(); stopping buf next does the same for live.
    presenter.stop()
    buf.stop()
    live.stop()

    print("Done.")


if __name__ == '__main__':
    main()
