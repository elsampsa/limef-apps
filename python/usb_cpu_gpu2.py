"""
Demo: USB camera → CPUBlock → GPUBlock → EncoderBlock → DumpFF

Same topology as usb_cpu_gpu.py but the TensorThread slots inside CPUBlock and
GPUBlock are replaced with TensorPythonInterface + Python consumer threads.
Each block exposes two member functions (_run_p1 / _run_p2) that become the
thread execution targets — override them to add real per-frame work.

Topology (see limef-md/diagrams.md):

    USBCameraTR --- Dec2TensorFF --> CPUBlock --> GPUBlock --> EncoderBlock --- DumpFF

CPUBlock / GPUBlock internals (Python-thread variant):

    SwitchFF
        terminal 0 → output              (skip: straight to block output)
        terminal 1 → TensorPythonIF → _run_p1 thread → output
        terminal 2 → TensorPythonIF → _run_p2 thread → output

EncoderBlock is unchanged (TensorThread + TensorToDecodedFF + NVENC EncodingFF).

Usage:
    python usb_cpu_gpu2.py [/dev/videoN]

Runtime switching:
    cpu_block.toggle(1)   # route through cpu p1
    gpu_block.toggle(2)   # route through gpu p2
    cpu_block.toggle(0)   # back to skip (pass-through)
"""

import time
import sys
import threading
import numpy as np
import limef

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

DEVICE     = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"
TIMEOUT_MS = 200   # pull() timeout; controls how quickly threads notice stop_event

sys.stdout.reconfigure(line_buffering=True)

# ── Block definitions ─────────────────────────────────────────────────────────

class CPUBlock:
    """
    TensorFrame routing block; all processing stays on CPU.

    SwitchFF (3 terminals):
        0 → output                       (skip: straight to block output)
        1 → p1 TensorPythonIF → output   (Python thread 1)
        2 → p2 TensorPythonIF → output   (Python thread 2)

    p1 keeps only the red channel (zeros G and B).
    p2 keeps only the green channel (zeros R and B).

    TensorPythonInterface fifo: stack_size=10, leaky=True, hw_accel=SW.
    10 CPU TensorFrame slots absorb a pull() delay in the Python thread.
    leaky=True: drop frames if the Python thread falls behind rather than
    stalling the upstream camera chain.
    """

    def __init__(self, name, verbose=False):
        self._name = name
        self._stop = threading.Event()

        self.switch = limef.SwitchFrameFilter(f"{name}_sw", 3)
        self.output = limef.DumpFrameFilter(f"{name}_out", verbose=verbose)

        self.p1 = limef.TensorPythonInterface(stack_size=10, leaky=True,
                                               hw_accel=limef.HWACCEL_SW, fifo_size=0)
        self.p2 = limef.TensorPythonInterface(stack_size=10, leaky=True,
                                               hw_accel=limef.HWACCEL_SW, fifo_size=0)
        self._c1 = self.p1.client()
        self._c2 = self.p2.client()

        self.switch.cc(0, self.output)
        self.switch.cc(1, self.p1.getInput())
        self.switch.cc(2, self.p2.getInput())
        self.p1.getOutput().cc(self.output)
        self.p2.getOutput().cc(self.output)
        self.switch.toggle(0)

        self._t1 = threading.Thread(target=self._run_p1, daemon=True,
                                    name=f"{name}_p1")
        self._t2 = threading.Thread(target=self._run_p2, daemon=True,
                                    name=f"{name}_p2")

    # ── Python thread targets ────────────────────────────────────────────────

    def _run_p1(self):
        """CPU thread 1: keep only the red channel (zero G and B)."""
        while not self._stop.is_set():
            frame = self._c1.pull(timeout_ms=TIMEOUT_MS)
            if frame is None:
                continue
            if isinstance(frame, limef.StreamFrame):
                self._c1.push(frame)
                continue
            # planes[0] is a zero-copy numpy (3,H,W) uint8 borrowed from the fifo.
            # Clone to get an owned, writable copy; then zero out G and B channels.
            out = frame.clone()
            arr = out.planes[0]   # (3, H, W): [0]=R  [1]=G  [2]=B
            arr[1] = 0            # zero green
            arr[2] = 0            # zero blue
            self._c1.push(out)

    def _run_p2(self):
        """CPU thread 2: keep only the green channel (zero R and B)."""
        while not self._stop.is_set():
            frame = self._c2.pull(timeout_ms=TIMEOUT_MS)
            if frame is None:
                continue
            if isinstance(frame, limef.StreamFrame):
                self._c2.push(frame)
                continue
            out = frame.clone()
            arr = out.planes[0]   # (3, H, W): [0]=R  [1]=G  [2]=B
            arr[0] = 0            # zero red
            arr[2] = 0            # zero blue
            self._c2.push(out)

    # ── Block interface ──────────────────────────────────────────────────────

    def getInput(self):
        return self.switch

    def cc(self, next_ff):
        self.output.cc(next_ff)
        return next_ff

    def toggle(self, i):
        self.switch.toggle(i)

    def start(self):
        self._t1.start()
        self._t2.start()

    def stop(self):
        self._stop.set()
        for t in (self._t1, self._t2):
            while t.is_alive():
                try:
                    t.join(timeout=TIMEOUT_MS / 1000 + 0.5)
                except KeyboardInterrupt:
                    pass


class GPUBlock:
    """
    Same topology as CPUBlock; TensorPythonInterfaces request CUDA H2D upload
    at the thread boundary.  Falls back to SW if CUDA is unavailable.

    SwitchFF (3 terminals):
        0 → output                       (skip: straight to block output)
        1 → p1 TensorPythonIF → output   (Python thread 1, CUDA)
        2 → p2 TensorPythonIF → output   (Python thread 2, CUDA)

    p1 keeps only the blue channel (zeros R and G).
    p2 converts to greyscale (BW) by averaging R, G, B.

    TensorPythonInterface fifo: stack_size=10, leaky=True, hw_accel=CUDA.
    Same rationale as CPUBlock.  hw_accel=CUDA triggers a CPU→CUDA H2D
    upload at the thread boundary so pull() always returns GPU tensors.
    """

    def __init__(self, name, verbose=False):
        self._name = name
        self._stop = threading.Event()
        hw = (limef.HWACCEL_CUDA
              if limef.isHWAccelAvailable(limef.HWACCEL_CUDA)
              else limef.HWACCEL_SW)

        self.switch = limef.SwitchFrameFilter(f"{name}_sw", 3)
        self.output = limef.DumpFrameFilter(f"{name}_out", verbose=verbose)

        self.p1 = limef.TensorPythonInterface(stack_size=10, leaky=True,
                                               hw_accel=hw, fifo_size=0)
        self.p2 = limef.TensorPythonInterface(stack_size=10, leaky=True,
                                               hw_accel=hw, fifo_size=0)
        self._c1 = self.p1.client()
        self._c2 = self.p2.client()

        self.switch.cc(0, self.output)
        self.switch.cc(1, self.p1.getInput())
        self.switch.cc(2, self.p2.getInput())
        self.p1.getOutput().cc(self.output)
        self.p2.getOutput().cc(self.output)
        self.switch.toggle(0)

        self._t1 = threading.Thread(target=self._run_p1, daemon=True,
                                    name=f"{name}_p1")
        self._t2 = threading.Thread(target=self._run_p2, daemon=True,
                                    name=f"{name}_p2")

    # ── Python thread targets ────────────────────────────────────────────────

    def _run_p1(self):
        """GPU thread 1: keep only the blue channel (zero R and G)."""
        while not self._stop.is_set():
            frame = self._c1.pull(timeout_ms=TIMEOUT_MS)
            if frame is None:
                continue
            if isinstance(frame, limef.StreamFrame):
                self._c1.push(frame)
                continue
            if not _TORCH:
                self._c1.push(frame)
                continue
            # planes[0] is a DLPack capsule for GPU frames.
            # Clone via torch before modifying; write into a new owned TensorFrame.
            t = torch.from_dlpack(frame.planes[0]).clone()  # (3,H,W) uint8 CUDA
            C, H, W = t.shape
            t[0] = 0   # zero red
            t[1] = 0   # zero green  → only blue remains
            out = limef.TensorFrame()
            out.reserve_gpu_plane(0, [C, H, W], 'uint8')
            torch.from_dlpack(out.planes[0]).copy_(t)
            out.timestamp = frame.timestamp
            out.slot      = frame.slot
            self._c1.push(out)

    def _run_p2(self):
        """GPU thread 2: convert to greyscale (BW) by averaging R, G, B."""
        while not self._stop.is_set():
            frame = self._c2.pull(timeout_ms=TIMEOUT_MS)
            if frame is None:
                continue
            if isinstance(frame, limef.StreamFrame):
                self._c2.push(frame)
                continue
            if not _TORCH:
                self._c2.push(frame)
                continue
            t = torch.from_dlpack(frame.planes[0])          # (3,H,W) uint8 CUDA (borrowed)
            C, H, W = t.shape
            # Mean across channels → (1,H,W) → expand to (3,H,W) so downstream
            # TensorToDecodedFrameFilter sees the expected 3-channel layout.
            bw = (t.float().mean(dim=0, keepdim=True)
                   .to(torch.uint8)
                   .expand(3, -1, -1)
                   .contiguous())
            out = limef.TensorFrame()
            out.reserve_gpu_plane(0, [C, H, W], 'uint8')
            torch.from_dlpack(out.planes[0]).copy_(bw)
            out.timestamp = frame.timestamp
            out.slot      = frame.slot
            self._c2.push(out)

    # ── Block interface ──────────────────────────────────────────────────────

    def getInput(self):
        return self.switch

    def cc(self, next_ff):
        self.output.cc(next_ff)
        return next_ff

    def toggle(self, i):
        self.switch.toggle(i)

    def start(self):
        self._t1.start()
        self._t2.start()

    def stop(self):
        self._stop.set()
        for t in (self._t1, self._t2):
            while t.is_alive():
                try:
                    t.join(timeout=TIMEOUT_MS / 1000 + 0.5)
                except KeyboardInterrupt:
                    pass


class EncoderBlock:
    """
    DecodedFrame → H.264 PacketFrame (NVENC).

    Receives DecodedFrames (AV_PIX_FMT_CUDA NV12) directly — TensorThread and
    TensorToDecodedFrameFilter live upstream (outside this block) so their output
    can be shared with the display branch via SplitFrameFilter.

    Internal chain:
        NVENC EncodingFF --- DumpFF
    """

    def __init__(self, name, verbose=False):
        enc_params              = limef.FFmpegEncoderParams()
        enc_params.codec_id     = limef.AV_CODEC_ID_H264
        enc_params.hw_accel     = limef.HWACCEL_CUDA
        enc_params.preset       = 'p1'
        enc_params.tune         = 'ull'
        enc_params.max_b_frames = 0

        self.enc    = limef.EncodingFrameFilter(f"{name}_enc", enc_params)
        self.output = limef.DumpFrameFilter(f"{name}_out", verbose=verbose)

        self.enc.cc(self.output)

    def getInput(self):
        """Return the EncodingFrameFilter (accepts DecodedFrames)."""
        return self.enc

    def cc(self, next_ff):
        self.output.cc(next_ff)
        return next_ff

    def start(self):
        pass   # no C++ threads owned by this block

    def stop(self):
        pass


# ── Instantiate blocks ────────────────────────────────────────────────────────

cpu_block = CPUBlock("cpu", verbose=False)
gpu_block = GPUBlock("gpu", verbose=False)
enc_block = EncoderBlock("enc", verbose=True)

# ── Tensor → Decoded conversion stage (shared by display and encoder) ─────────
# TensorThread breaks the synchronous chain and handles CPU→CUDA upload.
# TensorToDecodedFrameFilter converts GPU TensorFrame → AV_PIX_FMT_CUDA NV12.
# Their output is a DecodedFrame that both GLXPresenterThread and EncoderBlock need.

input_tr = limef.TensorThread("enc_input",
                               hw_accel=limef.HWACCEL_CUDA,
                               verbose_interval=20)
t2d = limef.TensorToDecodedFrameFilter("t2d")

# SplitFrameFilter fans out DecodedFrames to display and encoder in parallel.
split = limef.SplitFrameFilter("split")

# GLXPresenterThread fifo: default stack_size=20, leaky=True, fifo cap=40.
# 20 DecodedFrame slots cover a few screen-refresh cycles of headroom.
# leaky=True: drop a display frame rather than stalling the encoder branch.
# Tunable via stack_size= if needed.
pctx                    = limef.PresenterContext()
pctx.width              = 640
pctx.height             = 480
pctx.bypass_compositor  = True   # required on KWin + PRIME/GLX
glx = limef.GLXPresenterThread("glx", pctx,
          stack_size=20,  # a few screen-refresh cycles of headroom
          fifo_size=40)   # evict oldest if display falls behind a burst

# ── Camera source ─────────────────────────────────────────────────────────────

ctx               = limef.USBCameraContext(DEVICE, slot=1)
ctx.width         = 640
ctx.height        = 480
ctx.fps           = 30
ctx.output_format = limef.AV_PIX_FMT_GBRP   # required by DecodedToTensorFrameFilter

d2t = limef.DecodedToTensorFrameFilter("d2t")
cam = limef.USBCameraThread("usb-cam", ctx)

# ── Wire the pipeline ─────────────────────────────────────────────────────────
#
#   cam --- d2t --> cpu_block --> gpu_block --> input_tr --- t2d --> split --> glx
#                                                                         \--> enc_block

cam.cc(d2t)                      # DecodedFrame (GBRP)
d2t.cc(cpu_block.getInput())     # TensorFrame  (CPU, RGB uint8)
cpu_block.cc(gpu_block.getInput())  # TensorFrame  (CPU, RGB uint8)
gpu_block.cc(input_tr.getInput())   # TensorFrame  (CPU or CUDA, RGB uint8)
input_tr.cc(t2d)                 # TensorFrame  (CUDA, RGB uint8)  — after H2D upload
t2d.cc(split)                    # DecodedFrame (AV_PIX_FMT_CUDA NV12)
split.cc(glx.getInput())         # DecodedFrame (AV_PIX_FMT_CUDA NV12)
split.cc(enc_block.getInput())   # DecodedFrame (AV_PIX_FMT_CUDA NV12)

# ── Start threads (consumers before producers) ────────────────────────────────

glx.start()
enc_block.start()
input_tr.start()
gpu_block.start()
cpu_block.start()
cam.start()

print("Pipeline running for 30 s.  Press Ctrl-C to stop early.")
print("  cpu_block.toggle(1)  → red only   (CPU p1)")
print("  cpu_block.toggle(2)  → green only (CPU p2)")
print("  gpu_block.toggle(1)  → blue only  (GPU p1, CUDA)")
print("  gpu_block.toggle(2)  → greyscale  (GPU p2, CUDA)")
print("  *.toggle(0)          → skip (pass-through)")

try:
    cpu_block.toggle(0)
    time.sleep(5)

    print("Switching to cpu thread 1")
    cpu_block.toggle(1)
    time.sleep(5)

    print("Switching to cpu thread 2")
    cpu_block.toggle(2)
    time.sleep(5)

    cpu_block.toggle(0)

    print("Switching to gpu thread 1")
    gpu_block.toggle(1)
    time.sleep(5)

    print("Switching to gpu thread 2")
    gpu_block.toggle(2)
    time.sleep(5)

    gpu_block.toggle(0)

    time.sleep(5)

except KeyboardInterrupt:
    pass

# ── Stop (source first, then join Python threads, then C++ encoder) ───────────
# IMPORTANT: Python threads must be joined before their TensorPythonInterface
# objects are destroyed.  pull() may be blocking in fifo_.read() with the GIL
# released; destroying the fifo while a thread waits → std::terminate().

cam.stop()
cpu_block.stop()
gpu_block.stop()
input_tr.stop()
enc_block.stop()
glx.stop()

print("Done.")
