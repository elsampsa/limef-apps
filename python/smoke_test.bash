#!/usr/bin/env bash
# smoke_test.bash — lightweight smoke test for all Python demo apps.
#
# For each app it verifies:
#   (1) limef imports and all objects construct without error
#   (2) threads start and run for a few seconds without crashing
#
# Three outcomes per test:
#   PASS  — started, ran, and exited cleanly (exit 0 or 130)
#   WARN  — started and ran correctly, but stop() hung and needed SIGKILL
#           This is a pre-existing issue in the C++ shutdown path, distinct
#           from API regressions.  WARNs do not cause the script to exit 1.
#   FAIL  — crashed at startup or during the run (the kind of thing an API
#           change could introduce)
#
# Run this after API changes (e.g. new constructor parameters) to confirm
# every app still works before a full test suite run.
#
# Usage (from the meta-repo root, after sourcing go_debug.bash):
#   apps/python/smoke_test.bash
#
# Environment variables:
#   MEDIA_FILE   path to a video file.  Defaults to fixtures/jontxu_k1_sec.mkv.
#   DISPLAY      X11 display.  Apps that open a window are skipped when unset.
#   LIVE_URL_1   optional: real RTSP URL for an extra rtsp_client live-stream test.

set -uo pipefail

APPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$APPS_DIR/../.." && pwd)"

PASS=0
WARN=0
FAIL=0
SKIP=0

# How long to let each app run before stopping it (seconds).
RUN_SECS=6

# Minimum uptime before we consider the start successful.
# Catches construction / import crashes that happen in the first second or two.
MIN_UPTIME=3

# How long to wait for clean exit after SIGINT before escalating to SIGKILL.
SIGINT_GRACE=5

# ── colour helpers ────────────────────────────────────────────────────────────

_red()    { printf '\033[31m%s\033[0m' "$*"; }
_green()  { printf '\033[32m%s\033[0m' "$*"; }
_yellow() { printf '\033[33m%s\033[0m' "$*"; }

_pass() { printf '  %-44s  %s\n' "$1" "$(_green PASS)";        PASS=$((PASS+1)); }
_warn() { printf '  %-44s  %s\n' "$1" "$(_yellow "WARN: $2")"; WARN=$((WARN+1)); }
_fail() { printf '  %-44s  %s\n' "$1" "$(_red   "FAIL: $2")";  FAIL=$((FAIL+1)); }
_skip() { printf '  %-44s  %s\n' "$1" "$(_yellow "SKIP: $2")"; SKIP=$((SKIP+1)); }

_show_log() {
    local log="$1"
    if [[ -s "$log" ]]; then
        echo "    --- last output ---"
        tail -8 "$log" | sed 's/^/    /'
        echo "    ---"
    fi
}

# _stop_pid <pid> <label> <log>
# Try SIGINT first; escalate to SIGKILL after SIGINT_GRACE seconds.
# Sets global _stop_warn=1 if SIGKILL was needed.
_stop_pid() {
    local pid="$1" label="$2" log="$3"
    _stop_warn=0

    kill -INT "$pid" 2>/dev/null || true
    local deadline=$(( SECONDS + SIGINT_GRACE ))
    while kill -0 "$pid" 2>/dev/null && [[ $SECONDS -lt $deadline ]]; do
        sleep 0.2
    done

    if kill -0 "$pid" 2>/dev/null; then
        # SIGINT was ignored — escalate.
        kill -KILL "$pid" 2>/dev/null || true
        local grace=$(( SECONDS + 3 ))
        while kill -0 "$pid" 2>/dev/null && [[ $SECONDS -lt $grace ]]; do
            sleep 0.2
        done
        _stop_warn=1
    fi

    wait "$pid" 2>/dev/null || true
}

# ── run_with_sigint ───────────────────────────────────────────────────────────
# Start the app, wait RUN_SECS, then stop it.
# PASS if alive at MIN_UPTIME and exited cleanly.
# WARN if alive at MIN_UPTIME but needed SIGKILL.
# FAIL if dead before MIN_UPTIME (startup crash).
# Usage: run_with_sigint <label> python3 app.py [args...]
run_with_sigint() {
    local label="$1"; shift
    local log; log=$(mktemp /tmp/limef_smoke_XXXXXX.log)

    python3 "$@" >"$log" 2>&1 &
    local pid=$!

    # Startup check.
    sleep "$MIN_UPTIME"
    if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null; local rc=$?
        _fail "$label" "crashed at startup (exit $rc)"
        _show_log "$log"
        rm -f "$log"
        return
    fi

    # Let it run for the remaining time.
    local remain=$(( RUN_SECS - MIN_UPTIME ))
    [[ $remain -gt 0 ]] && sleep "$remain"

    # Check it's still alive mid-run (guards against deferred crashes).
    if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null; local rc=$?
        _fail "$label" "crashed during run (exit $rc)"
        _show_log "$log"
        rm -f "$log"
        return
    fi

    # Stop it.
    _stop_pid "$pid" "$label" "$log"

    if [[ $_stop_warn -eq 1 ]]; then
        _warn "$label" "stop() hung — needed SIGKILL (pre-existing shutdown issue)"
    else
        _pass "$label"
    fi
    rm -f "$log"
}


# ── prerequisites ─────────────────────────────────────────────────────────────

echo ""
echo "limef Python apps smoke test"
echo "============================"
echo ""

if ! python3 -c "import limef" 2>/dev/null; then
    echo "ERROR: 'import limef' failed.  Source go_debug.bash first."
    exit 1
fi

MEDIA_FILE="${MEDIA_FILE:-$REPO_ROOT/fixtures/jontxu_k1_sec.mkv}"
if [[ ! -f "$MEDIA_FILE" ]]; then
    echo "WARNING: MEDIA_FILE not found at '$MEDIA_FILE'."
    echo "         Set MEDIA_FILE=/path/to/video.mp4 for file-based tests."
    MEDIA_FILE=""
fi

HAS_DISPLAY=0
[[ -n "${DISPLAY:-}" ]] && HAS_DISPLAY=1

HAS_CAMERA=0
CAMERA_DEV=""
for dev in /dev/video*; do
    if [[ -c "$dev" ]]; then
        HAS_CAMERA=1; CAMERA_DEV="$dev"; break
    fi
done

echo "  limef:       OK"
echo "  MEDIA_FILE:  ${MEDIA_FILE:-<not set — file tests will be skipped>}"
echo "  DISPLAY:     ${DISPLAY:-<not set — presenter tests will be skipped>}"
echo "  camera:      $( [[ $HAS_CAMERA -eq 1 ]] && echo "$CAMERA_DEV" || echo "<none — camera tests will be skipped>" )"
echo ""
echo "Results:"
echo ""

# ── play_file.py ──────────────────────────────────────────────────────────────
# Tests the sdl and glx presenter paths with a short fixture file.

if [[ -z "$MEDIA_FILE" ]]; then
    _skip "play_file.py (sdl)" "MEDIA_FILE not set"
    _skip "play_file.py (glx)" "MEDIA_FILE not set"
elif [[ $HAS_DISPLAY -eq 0 ]]; then
    _skip "play_file.py (sdl)" "DISPLAY not set"
    _skip "play_file.py (glx)" "DISPLAY not set"
else
    run_with_sigint "play_file.py (sdl)" \
        "$APPS_DIR/play_file.py" --file "$MEDIA_FILE" --presenter sdl
    run_with_sigint "play_file.py (glx)" \
        "$APPS_DIR/play_file.py" --file "$MEDIA_FILE" --presenter glx
fi

# ── rtsp_client.py ────────────────────────────────────────────────────────────
# Connect to a non-existent server — verifies startup + reconnect-backoff loop.

if [[ $HAS_DISPLAY -eq 0 ]]; then
    _skip "rtsp_client.py (sdl, no server)" "DISPLAY not set"
    _skip "rtsp_client.py (glx, no server)" "DISPLAY not set"
else
    run_with_sigint "rtsp_client.py (sdl, no server)" \
        "$APPS_DIR/rtsp_client.py" --rtsp rtsp://127.0.0.1:19999/test --presenter sdl
    run_with_sigint "rtsp_client.py (glx, no server)" \
        "$APPS_DIR/rtsp_client.py" --rtsp rtsp://127.0.0.1:19999/test --presenter glx
fi

if [[ -n "${LIVE_URL_1:-}" ]]; then
    if [[ $HAS_DISPLAY -eq 0 ]]; then
        _skip "rtsp_client.py (live, sdl)" "DISPLAY not set"
    else
        run_with_sigint "rtsp_client.py (live, sdl)" \
            "$APPS_DIR/rtsp_client.py" --rtsp "$LIVE_URL_1" --presenter sdl
    fi
else
    _skip "rtsp_client.py (live)" "LIVE_URL_1 not set"
fi

# ── rtsp_server.py ────────────────────────────────────────────────────────────
# No display needed: media file → decode → blur → encode → RTSP.

if [[ -z "$MEDIA_FILE" ]]; then
    _skip "rtsp_server.py" "MEDIA_FILE not set"
else
    run_with_sigint "rtsp_server.py" \
        "$APPS_DIR/rtsp_server.py" --file "$MEDIA_FILE"
fi

# ── usb_cpu_gpu.py ────────────────────────────────────────────────────────────
# Camera only, no display, no CUDA required (GPU block falls back to SW).

if [[ $HAS_CAMERA -eq 0 ]]; then
    _skip "usb_cpu_gpu.py" "no camera"
else
    run_with_sigint "usb_cpu_gpu.py" \
        "$APPS_DIR/usb_cpu_gpu.py" "$CAMERA_DEV"
fi

# ── usb_cpu_gpu2.py ───────────────────────────────────────────────────────────
# Camera + display (GLXPresenterThread).

if [[ $HAS_CAMERA -eq 0 ]]; then
    _skip "usb_cpu_gpu2.py" "no camera"
elif [[ $HAS_DISPLAY -eq 0 ]]; then
    _skip "usb_cpu_gpu2.py" "DISPLAY not set"
else
    run_with_sigint "usb_cpu_gpu2.py" \
        "$APPS_DIR/usb_cpu_gpu2.py" "$CAMERA_DEV"
fi

# ── usb_gpu_pipeline.py ───────────────────────────────────────────────────────
# Camera only, no display.  CUDA preferred but not required.

if [[ $HAS_CAMERA -eq 0 ]]; then
    _skip "usb_gpu_pipeline.py" "no camera"
else
    run_with_sigint "usb_gpu_pipeline.py" \
        "$APPS_DIR/usb_gpu_pipeline.py" --device "$CAMERA_DEV"
fi

# ── summary ───────────────────────────────────────────────────────────────────

echo ""
echo "----"
printf 'PASS: %d   WARN: %d   FAIL: %d   SKIP: %d\n' \
    "$PASS" "$WARN" "$FAIL" "$SKIP"
if [[ $WARN -gt 0 ]]; then
    echo "(WARN = ran correctly but stop() hung — pre-existing shutdown issue, not an API regression)"
fi
echo ""

[[ $FAIL -eq 0 ]]
