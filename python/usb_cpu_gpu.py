"""
Demo: USB camera → CPUBlock → GPUBlock → EncoderBlock → DumpFF

Topology (see limef-md/diagrams.md for the diagram convention):

    USBCameraTR --- Dec2TensorFF --> CPUBlock --> GPUBlock --> EncoderBlock --- DumpFF

Each processing block (CPUBlock, GPUBlock) wraps:
    SwitchFF
        terminal 0 → output        (skip: straight to block output)
        terminal 1 → p1 TensorTR → output
        terminal 2 → p2 TensorTR → output

The block output is a DumpFrameFilter whose verbosity can be toggled at
construction time, making it easy to peek at what each block emits.

EncoderBlock:
    TensorTR (CUDA) --- TensorToDecodedFF --- NVENC EncodingFF

    GPU TensorFrame → AV_PIX_FMT_CUDA (NV12) → NVENC — no SwScale needed.

Usage:
    python usb_cpu_gpu.py [/dev/videoN]

Runtime switching (from an interactive Python session or a second thread):
    cpu_block.switch.toggle(1)   # route through cpu p1
    gpu_block.switch.toggle(2)   # route through gpu p2
    cpu_block.switch.toggle(0)   # back to skip (pass-through)
"""

import time
import sys
import limef

DEVICE           = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"
VERBOSE_INTERVAL = 20   # TensorThreads log frame count every N frames

sys.stdout.reconfigure(line_buffering=True) # necessary if we want to interleave cpp and python printouts

# ── Block definitions ─────────────────────────────────────────────────────────

class CPUBlock:
    """
    TensorFrame routing block; all processing stays on CPU.

    SwitchFF (3 terminals):
        0 → output              (skip: straight to block output)
        1 → p1 TensorTR → output
        2 → p2 TensorTR → output

    output is a DumpFrameFilter that serves as the convergence point.
    Set verbose=True to see every frame that leaves this block.

    TensorThread fifo: default stack_size=5, leaky=False.
    5 slots are sufficient at camera frame rate (no burst source).
    leaky=False: back-pressure is preferred here — if a thread stalls,
    we want to notice rather than silently drop frames.
    """

    def __init__(self, name, verbose=False):
        self.switch = limef.SwitchFrameFilter(f"{name}_sw", 3)
        self.output = limef.DumpFrameFilter(f"{name}_out", verbose=verbose)
        self.p1     = limef.TensorThread(f"{name}_p1",
                                         hw_accel=limef.HWACCEL_SW,
                                         verbose_interval=VERBOSE_INTERVAL)
        self.p2     = limef.TensorThread(f"{name}_p2",
                                         hw_accel=limef.HWACCEL_SW,
                                         verbose_interval=VERBOSE_INTERVAL)

        self.switch.cc(0, self.output)          # skip terminal → output directly
        self.switch.cc(1, self.p1.getInput())
        self.switch.cc(2, self.p2.getInput())
        self.p1.cc(self.output)
        self.p2.cc(self.output)
        self.switch.toggle(0)                   # start on skip branch

    def getInput(self):
        return self.switch

    def cc(self, next_ff):
        self.output.cc(next_ff)
        return next_ff

    def start(self):
        self.p1.start()
        self.p2.start()

    def stop(self):
        self.p1.stop()
        self.p2.stop()

    def toggle(self, i):
        self.switch.toggle(i)


class GPUBlock:
    """
    Same topology as CPUBlock; TensorThreads request a CUDA H2D upload at the
    thread boundary.  Falls back to SW if CUDA is unavailable.

    SwitchFF (3 terminals):
        0 → output              (skip: straight to block output)
        1 → p1 TensorTR → output
        2 → p2 TensorTR → output

    TensorThread fifo: default stack_size=5, leaky=False.
    Same reasoning as CPUBlock — camera rate is steady, 5 slots is ample.
    hw_accel=CUDA triggers an H2D upload at the thread boundary so downstream
    code always sees CUDA tensors regardless of what the upstream block emits.
    """

    def __init__(self, name, verbose=False):
        hw = (limef.HWACCEL_CUDA
              if limef.isHWAccelAvailable(limef.HWACCEL_CUDA)
              else limef.HWACCEL_SW)

        self.switch = limef.SwitchFrameFilter(f"{name}_sw", 3)
        self.output = limef.DumpFrameFilter(f"{name}_out", verbose=verbose)
        self.p1     = limef.TensorThread(f"{name}_p1",
                                         hw_accel=hw,
                                         verbose_interval=VERBOSE_INTERVAL)
        self.p2     = limef.TensorThread(f"{name}_p2",
                                         hw_accel=hw,
                                         verbose_interval=VERBOSE_INTERVAL)

        self.switch.cc(0, self.output)
        self.switch.cc(1, self.p1.getInput())
        self.switch.cc(2, self.p2.getInput())
        self.p1.cc(self.output)
        self.p2.cc(self.output)
        self.switch.toggle(0)

    def getInput(self):
        return self.switch

    def cc(self, next_ff):
        self.output.cc(next_ff)
        return next_ff

    def start(self):
        self.p1.start()
        self.p2.start()

    def stop(self):
        self.p1.stop()
        self.p2.stop()

    def toggle(self, i):
        self.switch.toggle(i)


class EncoderBlock:
    """
    DecodedFrame → H.264 PacketFrame (NVENC).

    Receives DecodedFrames (AV_PIX_FMT_CUDA NV12) directly.
    TensorThread and TensorToDecodedFrameFilter live upstream (outside this
    block) as standalone objects.

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
# Set verbose=True on a block to see every frame leaving that block.

cpu_block = CPUBlock("cpu", verbose=False)
gpu_block = GPUBlock("gpu", verbose=False)
enc_block = EncoderBlock("enc", verbose=True)   # show encoded output

# ── Tensor → Decoded conversion stage ─────────────────────────────────────────
# Standalone, outside EncoderBlock, so the boundary is explicit in the wiring.

input_tr = limef.TensorThread("enc_input",
                               hw_accel=limef.HWACCEL_CUDA,
                               verbose_interval=VERBOSE_INTERVAL)
t2d = limef.TensorToDecodedFrameFilter("t2d")

# ── Camera source ─────────────────────────────────────────────────────────────
# USBCamera outputs GBRP — required by DecodedToTensorFrameFilter (CPU path).
# GPUBlock TensorThreads upload CPU→CUDA at the thread boundary, so the same
# GBRP TensorFrames flow into the GPU block without a separate UploadGPU filter.

ctx               = limef.USBCameraContext(DEVICE, slot=1)
ctx.width         = 640
ctx.height        = 480
ctx.fps           = 30
ctx.output_format = limef.AV_PIX_FMT_GBRP

d2t = limef.DecodedToTensorFrameFilter("d2t")
cam = limef.USBCameraThread("usb-cam", ctx)

# ── Wire the pipeline ─────────────────────────────────────────────────────────
#
#   cam --- d2t --> cpu_block --> gpu_block --> input_tr --- t2d --> enc_block

cam.cc(d2t)                          # DecodedFrame (GBRP)
d2t.cc(cpu_block.getInput())         # TensorFrame  (CPU, RGB uint8)
cpu_block.cc(gpu_block.getInput())   # TensorFrame  (CPU, RGB uint8)
gpu_block.cc(input_tr.getInput())    # TensorFrame  (CPU or CUDA, RGB uint8)
input_tr.cc(t2d)                     # TensorFrame  (CUDA, RGB uint8)  — after H2D upload
t2d.cc(enc_block.getInput())         # DecodedFrame (AV_PIX_FMT_CUDA NV12)

# ── Start threads (consumers before producers) ────────────────────────────────

enc_block.start()
input_tr.start()
gpu_block.start()
cpu_block.start()
cam.start()

print("Pipeline running for 30 s.  Press Ctrl-C to stop early.")
print("  cpu_block.switch.toggle(1)  → route through cpu p1")
print("  gpu_block.switch.toggle(2)  → route through gpu p2")
print("  *.switch.toggle(0)          → back to skip (pass-through)")

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

# ── Stop threads (source first, then consumers in order) ─────────────────────

cam.stop()
cpu_block.stop()
gpu_block.stop()
input_tr.stop()
enc_block.stop()

print("Done.")
