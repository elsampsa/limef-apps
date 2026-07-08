# Python Example Apps

> [!IMPORTANT]
> `FrameFilter` or `Thread` objects that participate in a chain **must be
> kept alive by a Python variable** for the entire duration of the pipeline run.

Demo apps for complex live streaming pipelines in python:

- Stream live from USB camera
- Do preprocessing and analysis both on the CPU and GPU
- You can do image manipulation / machine vision analysis on the GPU only
- Encode stream in the GPU
- Machine vision modules can be toggled/switched on/off on-the-fly
- Visualize on your linux desktop and/or transmit your video stream over the internet with RTSP

## Setup

Install limef using the deb package, or setup your staging/development environment.

## rtsp_server.py

*Stream a media file over the internet with in-process Python frame processing*

Reads a media file, decodes it, passes frames to a Python thread via
`PythonInterface` (here: Gaussian blur via OpenCV), re-encodes, and serves the
result as an RTSP stream that any player can consume over the network.
Decoding and encoding both run on the **CPU**.  Use this as a starting point
when you want to intercept and modify frames in Python before streaming —
no GPU required.

```
python3 apps/python/rtsp_server.py --file PATH [--port 8554] [--bitrate N]
```

Connect with `ffplay rtsp://localhost:8554/live/stream`.

### Pipeline

```mermaid
flowchart TD
    mediatr[MediaFileTR]
    decff(DecFF)
    scale_bgr(SwScaleFF)
    pyif[PythonInterface]
    scale_yuv(SwScaleFF)
    encff(EncFF)
    rtpmux(RTPMuxerFF)
    rtsptr[RTSPServerTR]

    mediatr --- decff
    decff --- scale_bgr
    scale_bgr -->|DecodedFrame BGR24| pyif
    pyif --- scale_yuv
    scale_yuv --- encff
    encff --- rtpmux
    rtpmux --> rtsptr

    classDef thread fill:#4a90d9,stroke:#2c5f8a,color:#fff
    classDef pytr   fill:#7b5ea7,stroke:#4a3570,color:#fff
    classDef ff     fill:#5ba85a,stroke:#3d6e3d,color:#fff
    class mediatr,rtsptr thread
    class pyif pytr
    class decff,scale_bgr,scale_yuv,encff,rtpmux ff
```

`PythonInterface` acts as a thread boundary: frames flow in, the Python consumer
processes them (Gaussian blur via OpenCV), and pushes them back downstream.
Audio frames and `StreamFrame`s are forwarded unchanged.

---

## rtsp_client.py

*Connect to an RTSP / IP camera and display the stream in a window*

Connects to any RTSP source (IP camera, re-streamer, etc.), decodes the stream,
and presents it in a window.  Reconnects automatically on stream loss.  Supports
software, VAAPI and CUDA (NVDEC) decoding.  Timestamps are derived from the
stream's own PTS progression anchored to wall-clock at connect time (`t0+PTS_delta`)
— safe for all cameras.  Pass `--use-ntp` to use NTP wallclock from RTCP Sender
Reports instead (only if the camera's NTP is known to be reliable).

```
python3 apps/python/rtsp_client.py --rtsp RTSP_URL
                                   [--timeout SECS]
                                   [--use-ntp]
                                   [--decode sw|cuda|vaapi]
                                   [--buffer MS]
                                   [--presenter sdl|glx]
                                   [--bypass-compositor]
                                   [--verbose]
```

Example:

```bash
python3 apps/python/rtsp_client.py \
    --rtsp rtsp://admin:pass@192.168.1.10/stream \
    --decode cuda \
    --buffer 200
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--rtsp URL` | (required) | RTSP URL |
| `--timeout SECS` | 5 | Read timeout before reconnect |
| `--use-ntp` | off | Use NTP wall-clock from RTCP Sender Reports (only if camera NTP is reliable) |
| `--decode sw\|cuda\|vaapi` | `sw` | Decoder backend |
| `--buffer MS` | 0 | De-jitter tolerance in ms (see note below) |
| `--presenter sdl\|glx` | `sdl` | Window backend: SDL2 (default) or GLX/OpenGL (CUDA zero-copy) |
| `--bypass-compositor` | off | (GLX only) set `_NET_WM_BYPASS_COMPOSITOR` — needed on KWin/PRIME |
| `--verbose` | off | Print one line per frame at each pipeline stage for debugging |

**`--buffer` note:** maps to `PresenterContext.max_age_ms`.  Frames whose
absolute timestamp is older than this are dropped by the presenter, allowing the
display to catch up after a jitter burst.  `0` (default) keeps all frames.
A value of 100–200 ms is a good starting point for cameras with moderate network
jitter.  The `OrderedPacketBufferThread` holds a fixed 30-frame DTS-ordered queue
upstream of the decoder to absorb packet reordering; that is separate from this
tolerance.

**`--presenter` note:** `sdl` (default) works on any display setup.  `glx` uses
OpenGL/GLX; on NVIDIA hardware with `--decode cuda` it enables zero-copy
`cudaGraphicsGLRegisterImage` so decoded frames never touch the CPU.  Use
`--bypass-compositor` with GLX on KWin + PRIME (NVIDIA render offload) to prevent
the compositor from re-compositing through Mesa.

### Pipeline

```mermaid
flowchart TD
    livetr[LiveStreamTR]
    dumplive(DumpFF)
    buftr[OrderedPacketBufTR]
    dumpbuf(DumpFF)
    decff(DecFF)
    dumpdec(DumpFF)
    pres[PresenterTR SDL/GLX]

    livetr ---|PacketFrame| dumplive
    dumplive --> buftr
    buftr ---|PacketFrame DTS-ordered| dumpbuf
    dumpbuf --- decff
    decff ---|DecodedFrame| dumpdec
    dumpdec --> pres

    classDef thread fill:#4a90d9,stroke:#2c5f8a,color:#fff
    classDef ff     fill:#5ba85a,stroke:#3d6e3d,color:#fff
    class livetr,buftr,pres thread
    class decff,dumplive,dumpbuf,dumpdec ff
```

`LiveStreamThread` opens the RTSP URL and reconnects with exponential back-off on
failure.  `OrderedPacketBufferThread` re-orders packets by DTS before they reach
the decoder — important for cameras that send audio and video packets interleaved
out of order.  The three `DumpFrameFilter` nodes are silent pass-throughs by
default; pass `--verbose` to activate them for per-frame pipeline tracing.

---

## play_file.py

*Play a local media file in a window*

Reads a local file at its natural playback speed and presents decoded frames in
a window.  Supports software, VAAPI and CUDA (NVDEC) decoding.  The file can be
looped continuously or played once.

`MediaFileThread` internally chains a file-reader and an `OrderedPacketBufferThread`
(DTS ordering), so no separate buffer thread is needed.

```
python3 apps/python/play_file.py --file PATH
                                 [--loop MS]
                                 [--decode sw|cuda|vaapi]
                                 [--buffer MS]
                                 [--presenter sdl|glx]
                                 [--bypass-compositor]
                                 [--verbose]
```

Example:

```bash
python3 apps/python/play_file.py --file /path/to/video.mp4 --loop 0 --decode vaapi
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--file PATH` | (required) | Input file |
| `--loop MS` | -1 | Loop at EOF: pause this many ms then restart (-1 = play once, 0 = gapless) |
| `--decode sw\|cuda\|vaapi` | `sw` | Decoder backend |
| `--buffer MS` | 0 | Drop frames older than this many ms (0 = disabled) |
| `--presenter sdl\|glx` | `sdl` | Window backend |
| `--bypass-compositor` | off | (GLX only) set `_NET_WM_BYPASS_COMPOSITOR` |
| `--verbose` | off | Print one line per frame at each stage for debugging |

The process exits automatically when the file ends (no loop).

### Pipeline

```mermaid
flowchart TD
    srctr[MediaFileTR]
    dumpsrc(DumpFF)
    decff(DecFF)
    dumpdec(DumpFF)
    pres[PresenterTR SDL/GLX]

    srctr ---|PacketFrame| dumpsrc
    dumpsrc --- decff
    decff ---|DecodedFrame| dumpdec
    dumpdec --> pres

    classDef thread fill:#4a90d9,stroke:#2c5f8a,color:#fff
    classDef ff     fill:#5ba85a,stroke:#3d6e3d,color:#fff
    class srctr,pres thread
    class decff,dumpsrc,dumpdec ff
```

---

## tensor_if_test.py

*Minimal smoke test for the GPU TensorFrame → TensorPythonInterface path*

Reads a local file, software-decodes it, converts to NV12, uploads to the GPU,
converts to a CUDA RGB tensor, and passes it through `TensorPythonInterface` to a
Python consumer thread.  The consumer prints each frame's GPU status and shape,
then exits after `--frames` frames.

Use this to verify that the CUDA runtime, `DecodedUploadFrameFilter`,
`DecodedToTensorFrameFilter`, and `TensorPythonInterface` all work end-to-end on
the current platform — including Jetson, where FFmpeg's own CUDA hwcontext is
intentionally disabled.

```
python3 apps/python/tensor_if_test.py [--file PATH] [--frames N]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--file PATH` | `fixtures/jontxu.mkv` | Input video file |
| `--frames N` | 10 | Stop after this many `TensorFrame`s |

### Pipeline

```mermaid
flowchart TD
    srctr[MediaFileTR]
    decff(DecFF sw)
    scale(SwScaleFF NV12)
    upload(DecodedUploadFF CUDA)
    d2t(Dec2TensorFF RGB)
    pyif[TensorPythonInterface CUDA]
    py[Python consumer]

    srctr ---|PacketFrame| decff
    decff ---|DecodedFrame YUV| scale
    scale ---|DecodedFrame NV12| upload
    upload ---|DecodedFrame CUDA| d2t
    d2t -->|TensorFrame CUDA| pyif
    pyif -.->|pull / push| py

    classDef thread fill:#4a90d9,stroke:#2c5f8a,color:#fff
    classDef pytr   fill:#7b5ea7,stroke:#4a3570,color:#fff
    classDef ff     fill:#5ba85a,stroke:#3d6e3d,color:#fff
    class srctr thread
    class pyif pytr
    class decff,scale,upload,d2t ff
    class py pytr
```

`TensorPythonInterface` is constructed with `hw_accel=HWACCEL_CUDA` so its
internal `TensorFrameFifo` pre-allocates stack frames in GPU memory
(`BufferLocation::CUDA`).  Frames arrive already on the GPU from
`DecodedToTensorFrameFilter`, so no CPU↔GPU transfer happens at the Python
boundary.  The GPU path relies on the CUDA runtime (`cudaGetDeviceCount`) only —
FFmpeg's own CUDA hwcontext (`CONFIG_CUDA`) is not required, making it suitable
for Jetson builds where FFmpeg CUDA is disabled.

---

## usb_gpu_pipeline_cuda.py

*USB camera → CUDA GPU tensors → RTSP (NVENC or V4L2 M2M encoder)*

The full zero-copy GPU pipeline: camera frames are uploaded to the GPU immediately,
Python receives them as CUDA tensors via `torch.from_dlpack()`, and encoding is done
on the GPU.  Target platform is **Jetson Orin** (V4L2 M2M encoder) or any CUDA
desktop (NVENC).  The demo shows a 15×15 Gaussian blur (`--modify`) that stays
entirely on the GPU.

```
python3 apps/python/usb_gpu_pipeline_cuda.py [--modify]
                                              [--device /dev/video0]
                                              [--width 640] [--height 480] [--fps 30]
                                              [--port 8554] [--bitrate N]
                                              [--encoder nvenc|v4l2m2m]
                                              [--enc-device /dev/video11]
                                              [--enc-codec fwht|h264|h265]
```

Connect with `ffplay rtsp://localhost:8554/live/stream`.

| Encoder | Target | Notes |
|---------|--------|-------|
| `nvenc` (default) | Desktop CUDA GPU | HWACCEL_CUDA, H.264 NVENC |
| `v4l2m2m` | Jetson Orin | `--enc-device /dev/video11 --enc-codec h264` |

> **Note (Jetson):** zero-copy CUDA→V4L2 handoff (roadmap Step 6) is not yet
> implemented; `--encoder v4l2m2m` will incur a GPU→CPU download at the encoder
> boundary until Step 6 lands.

> **Note (GPU):** keep frames on the GPU throughout.  If you push a CPU
> `TensorFrame` into this pipeline, `Tensor2DecFF` outputs `GBRP` instead of CUDA
> NV12 and the encoder receives the wrong format.

### Pipeline

```mermaid
flowchart TD
    camtr[USBCameraTR]
    swscale(SwScaleFF NV12)
    uploadff(DecodedUploadFF)
    d2t(Dec2TensorFF)
    pyif[TensorPythonInterface CUDA]
    t2d(Tensor2DecFF)
    encff(EncFF NVENC or V4L2)
    rtpmux(RTPMuxerFF)
    rtsptr[RTSPServerTR]

    camtr ---|DecodedFrame YUYV| swscale
    swscale ---|DecodedFrame NV12| uploadff
    uploadff --- d2t
    d2t -->|TensorFrame CUDA| pyif
    pyif --- t2d
    t2d --- encff
    encff --- rtpmux
    rtpmux --> rtsptr

    classDef thread fill:#4a90d9,stroke:#2c5f8a,color:#fff
    classDef pytr   fill:#7b5ea7,stroke:#4a3570,color:#fff
    classDef ff     fill:#5ba85a,stroke:#3d6e3d,color:#fff
    class camtr,rtsptr thread
    class pyif pytr
    class swscale,uploadff,d2t,t2d,encff,rtpmux ff
```

---

## usb_pipeline.py

*USB camera → CPU tensors → RTSP (NVENC or V4L2 M2M encoder)*

The CPU tensorframe pipeline: tensors never go to the GPU before the encoding stage.
`CpuSwScaleConverter` handles any CPU pixel format (YUYV422, NV12, YUV420P, …) directly inside
`DecodedToTensorFrameFilter` — no explicit SwScale needed between the camera and `Dec2TensorFF`.  Use this to:

- **Test and develop** the CPU tensor path on any Linux desktop (use `--encoder nvenc`
  for CUDA H.264 encoding)
- **Deploy to Raspberry Pi** with hardware H.264 (`--encoder v4l2m2m --enc-codec h264`)
- **Test V4L2 encoding on desktop** with vicodec (`--encoder v4l2m2m --enc-codec fwht`)

`TensorToDecodedFrameFilter` (CPU path) outputs `GBRP`; `SwScaleFrameFilter` converts
to `NV12` before the encoder.  For `--encoder nvenc`, `FFmpegEncoder` handles the
CPU→GPU upload internally (`av_hwframe_transfer_data`) — no explicit `UploadGPUFF`
is needed.  The pipeline wiring is identical for both encoder choices.

Python receives CPU `TensorFrame`s — `frame.planes[0]` is a zero-copy numpy array.
The `--modify` Gaussian blur uses `torch.from_numpy()` and stays on the CPU.

```
python3 apps/python/usb_pipeline.py [--modify]
                                     [--device /dev/video0]
                                     [--width 640] [--height 480] [--fps 30]
                                     [--port 8554] [--bitrate N]
                                     [--encoder nvenc|v4l2m2m]
                                     [--enc-device /dev/video2]
                                     [--enc-codec fwht|h264|h265]
```

Connect with `ffplay rtsp://localhost:8554/live/stream`.

| Encoder | Target | Notes |
|---------|--------|-------|
| `nvenc` (default) | Desktop CUDA GPU | `DecodedUploadFF` inserted before encoder |
| `v4l2m2m` | RPi / Jetson | `--enc-device /dev/video2` for vicodec, `/dev/video11` for Jetson |

### Pipeline (both encoders)

```mermaid
flowchart TD
    camtr[USBCameraTR]
    d2t(Dec2TensorFF CPU)
    pyif[TensorPythonInterface CPU]
    t2d(Tensor2DecFF)
    swscale(SwScaleFF NV12)
    encff(EncFF NVENC or V4L2)
    rtpmux(RTPMuxerFF)
    rtsptr[RTSPServerTR]

    camtr --- d2t
    d2t -->|TensorFrame CPU| pyif
    pyif --- t2d
    t2d --- swscale
    swscale --- encff
    encff --- rtpmux
    rtpmux --> rtsptr

    classDef thread fill:#4a90d9,stroke:#2c5f8a,color:#fff
    classDef pytr   fill:#7b5ea7,stroke:#4a3570,color:#fff
    classDef ff     fill:#5ba85a,stroke:#3d6e3d,color:#fff
    class camtr,rtsptr thread
    class pyif pytr
    class d2t,t2d,swscale,encff,rtpmux ff
```


---

## usb_info.py

*USB camera → TensorPythonInterface → InfoFrame message channel*

Demonstrates pushing `InfoFrame`s (JSON strings) downstream from a Python
consumer and reading them in a separate thread via `InfoFrameFilter` + `EventFd`.

The Python consumer counts incoming `TensorFrame`s and every `--interval` frames
(default 10) pushes `limef.InfoFrame(json.dumps({"frames": N}))` into the pipeline
output.  A reader thread blocks on `select()` waiting for the `EventFd` signal, then
drains `InfoFrameFilter.popMessage()`.  No video output — purely a message-channel demo.

```
python3 apps/python/usb_info.py [--device /dev/video0]
                                 [--width 640] [--height 480] [--fps 30]
                                 [--interval 10]
```

### Pipeline

```mermaid
flowchart TD
    camtr[USBCameraTR]
    d2t(Dec2TensorFF CPU)
    putff(PutInfoFrameFilter)
    timer[timer thread]
    pyif[TensorPythonInterface]
    infoff(InfoFrameFilter)
    reader[reader thread]

    camtr --- d2t
    d2t --- putff
    timer -.->|put msg every 10 s| putff
    putff -->|TensorFrame + InfoFrame| pyif
    pyif ---|InfoFrame every N frames| infoff
    infoff -.->|popMessage via EventFd| reader

    classDef thread fill:#4a90d9,stroke:#2c5f8a,color:#fff
    classDef pytr   fill:#7b5ea7,stroke:#4a3570,color:#fff
    classDef ff     fill:#5ba85a,stroke:#3d6e3d,color:#fff
    class camtr,timer,reader thread
    class pyif pytr
    class d2t,putff,infoff ff
```

A timer thread calls `put_ff.put("message N")` every 10 s.  `PutInfoFrameFilter`
queues the string and emits it as an `InfoFrame` (on the driving thread) just before
the next `TensorFrame`.  `TensorPythonInterface.pull()` surfaces both frame types to
the Python consumer: injected `InfoFrame`s are captured; every N `TensorFrame`s the
consumer pushes a merged `InfoFrame({"frames": N, "injected": ...})` downstream.
`InfoFrameFilter` queues it and signals the `EventFd`; the reader thread drains
`popMessage()`.

---

## usb_cpu_gpu.py

*Live video processing on both CPU and GPU; processing stages can be switched and toggled on and off*

Demonstrates the `CPUBlock` / `GPUBlock` / `EncoderBlock` pattern.  Each block
wraps a `SwitchFrameFilter` with three terminals: terminal 0 is a direct
pass-through (skip), terminals 1 and 2 route through a `TensorThread` slot where
per-frame work lives.  The active branch can be switched at runtime without
stopping the pipeline — swap your CPU or GPU processing stage on the fly.
Encoding is done on the GPU with NVENC.

```
python3 apps/python/usb_cpu_gpu.py [--device /dev/video0] [--verbose]
```

### CPUBlock / GPUBlock internals

Both blocks have the same topology; `GPUBlock` threads use `hw_accel=HWACCEL_CUDA`.

```mermaid
flowchart TD
    input(SwitchFF)
    p1[TensorTR]
    p2[TensorTR]
    output(DumpFF)

    input ---|skip| output
    input --> p1
    input --> p2
    p1 --- output
    p2 --- output

    classDef thread fill:#4a90d9,stroke:#2c5f8a,color:#fff
    classDef ff     fill:#5ba85a,stroke:#3d6e3d,color:#fff
    class p1,p2 thread
    class input,output ff
```

### EncoderBlock internals

Receives `DecodedFrame AV_PIX_FMT_CUDA NV12` directly — the `TensorThread` and
`Tensor2DecFF` conversion sit outside this block as standalone objects.

```mermaid
flowchart TD
    encff(EncFF NVENC H.264)
    dumpff(DumpFF)

    encff ---|PacketFrame H.264| dumpff

    classDef ff fill:#5ba85a,stroke:#3d6e3d,color:#fff
    class encff,dumpff ff
```

### Pipeline

```mermaid
flowchart TD
    camthread[USBCameraTR]
    dec2tensor(Dec2TensorFF)
    cpublock[[CPUBlock]]
    gpublock[[GPUBlock]]
    input_tr[TensorTR]
    t2d(Tensor2DecFF)
    encoderblock[[EncoderBlock]]

    camthread ---|DecodedFrame| dec2tensor
    dec2tensor -->|TensorFrame| cpublock
    cpublock -->|TensorFrame| gpublock
    gpublock -->|TensorFrame CUDA| input_tr
    input_tr ---|TensorFrame CUDA| t2d
    t2d ---|DecodedFrame CUDA NV12| encoderblock

    classDef thread fill:#4a90d9,stroke:#2c5f8a,color:#fff
    classDef ff     fill:#5ba85a,stroke:#3d6e3d,color:#fff
    classDef block  fill:#e07b39,stroke:#9e4f1b,color:#fff
    class camthread,input_tr thread
    class dec2tensor,t2d ff
    class cpublock,gpublock,encoderblock block
```

---

## usb_cpu_gpu2.py

*Change machine vision module on your live stream on-the-fly, encode the modified video, visualize on your Linux desktop (you could continue by transmitting the video over the internet)*

The same `CPUBlock` / `GPUBlock` / `EncoderBlock` structure as above, but the
`TensorThread` slots inside each block are replaced with `TensorPythonInterface`
+ Python consumer threads.  This is where you put your real work: run a neural
network, draw bounding boxes, apply filters — all in Python, either on CPU
(CPUBlock) or on the GPU via `torch.from_dlpack()` (GPUBlock).  Switch between
processing modules at runtime without restarting.  After processing, a
`SplitFrameFilter` fans the result out to both a local `GLXPresenterThread` (live
window on your Linux desktop) and `EncoderBlock` for NVENC encoding and RTSP
streaming over the network.

```
python3 apps/python/usb_cpu_gpu2.py [--device /dev/video0] [--verbose]
```

### CPUBlock / GPUBlock internals (Python threads)

```mermaid
flowchart TD
    input(SwitchFF)
    p1[PyTR]
    p2[PyTR]
    output(DumpFF)

    input ---|skip| output
    input --> p1
    input --> p2
    p1 --- output
    p2 --- output

    classDef pytr fill:#7b5ea7,stroke:#4a3570,color:#fff
    classDef ff   fill:#5ba85a,stroke:#3d6e3d,color:#fff
    class p1,p2 pytr
    class input,output ff
```

### EncoderBlock internals

Identical to `usb_cpu_gpu.py`: receives `DecodedFrame AV_PIX_FMT_CUDA NV12` (the
`TensorThread` + `Tensor2DecFF` output is shared with `GLXPresenterTR` via
`SplitFrameFilter` upstream).

```mermaid
flowchart TD
    encff(EncFF NVENC H.264)
    dumpff(DumpFF)

    encff ---|PacketFrame H.264| dumpff

    classDef ff fill:#5ba85a,stroke:#3d6e3d,color:#fff
    class encff,dumpff ff
```

### Pipeline

```mermaid
flowchart TD
    camthread[USBCameraTR]
    dec2tensor(Dec2TensorFF)
    cpublock[[CPUBlock]]
    gpublock[[GPUBlock]]
    input_tr[TensorTR]
    t2d(Tensor2DecFF)
    split(SplitFF)
    encoderblock[[EncoderBlock]]
    glx[GLXPresenterTR]

    camthread ---|DecodedFrame| dec2tensor
    dec2tensor -->|TensorFrame| cpublock
    cpublock -->|TensorFrame| gpublock
    gpublock -->|TensorFrame CUDA| input_tr
    input_tr ---|TensorFrame CUDA| t2d
    t2d --- split
    split ---|DecodedFrame CUDA NV12| encoderblock
    split --> glx

    classDef thread fill:#4a90d9,stroke:#2c5f8a,color:#fff
    classDef pytr   fill:#7b5ea7,stroke:#4a3570,color:#fff
    classDef ff     fill:#5ba85a,stroke:#3d6e3d,color:#fff
    classDef block  fill:#e07b39,stroke:#9e4f1b,color:#fff
    class camthread,input_tr,glx thread
    class dec2tensor,t2d,split ff
    class cpublock,gpublock,encoderblock block
```

---

## Jetson examples

For NVIDIA Jetson-specific examples (Argus CSI camera, NVDEC/NVENC hardware
codec, CUDA pipelines) see **[jetson.md](jetson.md)**.

Covered there: `jetson_cam0.py`, `jetson_cam1.py`, `jetson_cam.py`,
`jetson_decode0.py`, `jetson_decode.py`, `jetson_encode.py`, `jetson_rtsp.py`.

---

## ws_html_demo.py

*Stream live video directly to a browser tab — no plugins, no RTSP player required*

Reads from an RTSP stream, a local media file, or a USB/V4L2 camera, muxes to
fragmented MP4, and serves it over WebSocket.  An embedded nginx process acts as
reverse proxy and static file server: open the printed URL in any modern browser
and the page plays the stream via the Media Source Extensions API.

**Codec validation:**

`FMP4FrameFilter` and `WebMFrameFilter` each contain a `CodecAssertFrameFilter`
as their first stage.  If the incoming stream carries an incompatible codec,
`CodecAssert` prints a warning and inutilizes the filter chain — no crash, no
silent corruption.  The upstream pipeline is responsible for providing the right
codec; these filters do not perform any conversion.

- `--file` / `--rtsp`: packets are forwarded directly into `FMP4FrameFilter`
  (no re-encoding).  The source is expected to carry H.264 (the most common
  case); other fMP4-compatible codecs (H.265, AV1) also work.
- `--usb`: the camera outputs raw frames; an encoder is inserted automatically.
  `--hw-accel` selects NVENC H.264 → `FMP4FrameFilter` (GPU required).
  Default (no flag) selects libvpx VP8 → `WebMFrameFilter` (software, no GPU needed).

```
python3 apps/python/ws_html_demo.py --file PATH [options]
python3 apps/python/ws_html_demo.py --rtsp URL  [options]
python3 apps/python/ws_html_demo.py --usb  DEV  [--hw-accel] [options]
```

Player URL is printed at startup:
```
http://localhost:{HTTP_PORT}/?token={TOKEN}&stream={UUID}
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--file PATH` | — | Local media file (H.264 or other fMP4-compatible codec) |
| `--rtsp URL` | — | RTSP stream URL (same codec requirement) |
| `--usb DEV` | — | V4L2 device, e.g. `/dev/video0` |
| `--ws-port PORT` | 18080 | Local WebSocket port (loopback only) |
| `--http-port PORT` | 8090 | nginx external HTTP port |
| `--uuid UUID` | `stream` | Stream UUID embedded in the WebSocket URL |
| `--token TOKEN` | `demo` | Access token embedded in the player URL |
| `--fps FPS` | 25 | Playback speed (file) or capture rate (USB) |
| `--loop` | off | Loop file source |
| `--width W` | 640 | USB capture width |
| `--height H` | 480 | USB capture height |
| `--bitrate BPS` | 4 000 000 | USB encoder bitrate |
| `--hw-accel` | off | USB: NVENC H.264/fMP4 instead of libvpx VP8/WebM (GPU required) |

### Pipeline

```mermaid
flowchart TD
    fileTR[MediaFileTR]
    liveTR[LiveStreamTR]
    usbTR[USBCameraTR]
    encH264(EncFF NVENC H.264)
    encVP8(EncFF libvpx VP8)
    fmp4(FMP4FF)
    webm(WebMFF)
    wssvr[WSSrvrTR]

    fileTR ---|PacketFrame H.264| fmp4
    liveTR ---|PacketFrame H.264| fmp4
    usbTR ---|DecodedFrame| encH264
    usbTR ---|DecodedFrame| encVP8
    encH264 ---|PacketFrame H.264| fmp4
    encVP8 ---|PacketFrame VP8| webm
    fmp4 -->|RawFrame fMP4| wssvr
    webm -->|RawFrame WebM| wssvr

    classDef thread fill:#4a90d9,stroke:#2c5f8a,color:#fff
    classDef ff     fill:#5ba85a,stroke:#3d6e3d,color:#fff
    class fileTR,liveTR,usbTR,wssvr thread
    class encH264,encVP8,fmp4,webm ff
```

`FMP4FrameFilter` and `WebMFrameFilter` each validate the incoming codec pair via
an internal `CodecAssertFrameFilter`, then handle muxing and box/cluster
partitioning into `RawFrame`s ready for the WebSocket server.
`WebSocketServerThread` (`WSSrvrTR`) caches the init segment per slot and replays
it to each new browser client before the next keyframe, so late-joiners always get
a clean stream start.

nginx is launched as a subprocess with `daemon off;` so it can be cleanly
terminated on Ctrl+C.  It proxies `/ws` to the loopback WebSocket port and
serves `ws_html_demo/static/index.html` at `/`.  The browser JS reads `token` and
`stream` from the page's own URL query string — no server-side templating needed.

---

## webrtc_html_demo.py

*Stream live video directly to a browser tab via WebRTC — lowest possible latency*

Reads from an RTSP stream, a local media file, or a USB/V4L2 camera, muxes to
RTP, and delivers it via WebRTC to any modern browser.  A `WebRTCServerThread`
handles the ICE/SDP signaling over HTTP; an embedded nginx serves the static page.

Sources and encoders are independent classes — pick one of each with `--file`/`--usb`
and `--hw-accel`:

| Source | Encoder | Command |
|--------|---------|---------|
| `FileSource` | `VP8Encoder` | `--file PATH` |
| `FileSource` | `CUDAEncoder` | `--file PATH --hw-accel` |
| `USBCameraSource` | `VP8Encoder` | `--usb /dev/video0` |
| `USBCameraSource` | `CUDAEncoder` | `--usb /dev/video0 --hw-accel` |

`FileSource` always strips audio (`MuteAudioFrameFilter`) and decodes video before
handing decoded frames to the encoder.  File source loops forever.
`USBCameraSource` outputs raw YUYV422 frames; the encoder's `SwScaleFrameFilter`
handles format conversion.  `WebRTCMuxerFrameFilter`'s internal
`CodecAssertFrameFilter` validates the codec — VP8 is universally accepted by
browsers (RFC 8834 mandatory); H.264 baseline (NVENC) works in Chrome and Firefox
when OpenH264 is available.

```
python3 apps/python/webrtc_html_demo.py --file PATH [options]
python3 apps/python/webrtc_html_demo.py --usb  DEV  [--hw-accel] [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--file PATH` | — | Local media file (decoded + re-encoded; audio muted) |
| `--usb DEV` | — | V4L2 device, e.g. `/dev/video0` |
| `--hw-accel` | off | Use NVENC H.264 encoder instead of libvpx VP8 (GPU required) |
| `--webrtc-port PORT` | 9090 | WebRTC signaling HTTP port (loopback only) |
| `--http-port PORT` | 9091 | nginx static-file HTTP port |
| `--uuid UUID` | `stream` | Stream UUID (exposed as `/<uuid>` on the signaling server) |
| `--fps FPS` | 25 | Playback speed (file) or capture rate (USB) |
| `--width W` | 640 | USB capture width |
| `--height H` | 480 | USB capture height |
| `--bitrate BPS` | 4 000 000 | Encoder bitrate |
| `--packetdump` | off | Log every packet before the WebRTC muxer (debug) |
| `--dump` | off | Log every RTP packet leaving the muxer (debug) |
| `--debug` | off | Set WebRTCServerThread log level to DEBUG (raw SDP exchange) |

### Pipeline

```mermaid
flowchart TD
    fileTR[MediaFileTR]
    muteFF(MuteAudioFF)
    decFF(DecFF)
    usbTR[USBCameraTR]
    swVP8(SwScaleFF YUV420P)
    encVP8(EncFF VP8)
    swCUDA(SwScaleFF NV12)
    encCUDA(EncFF H.264 NVENC)
    webrtcmux(WebRTCMuxerFF)
    wrtcsvr[WebRTCServerTR]

    fileTR --- muteFF
    muteFF --- decFF
    decFF --- swVP8
    decFF --- swCUDA
    usbTR --- swVP8
    usbTR --- swCUDA
    swVP8 --- encVP8
    swCUDA --- encCUDA
    encVP8 --- webrtcmux
    encCUDA --- webrtcmux
    webrtcmux -->|RTPFrames| wrtcsvr

    classDef thread fill:#4a90d9,stroke:#2c5f8a,color:#fff
    classDef ff     fill:#5ba85a,stroke:#3d6e3d,color:#fff
    class fileTR,usbTR,wrtcsvr thread
    class muteFF,decFF,swVP8,encVP8,swCUDA,encCUDA,webrtcmux ff
```

Only one of the two encoder branches is active at runtime (`VP8Encoder` or
`CUDAEncoder`).  `WebRTCMuxerFrameFilter` chains `CodecAssertFrameFilter` →
`AnnexBFrameFilter` (H.264 AVCC → Annex B for RTP) → `RTPMuxerFrameFilter`.
`WebRTCServerThread` handles ICE/STUN negotiation and SDP exchange; nginx serves
the static page on a separate port.

> **FrameFifo sizing:** the only thread boundary in this pipeline where packets
> can be lost is `WebRTCServerThread`'s incoming `FrameFifo`.  Hardware encoders
> (NVENC) emit packets in bursts — the default `stack_size=50, fifo_size=100` is
> too small and causes periodic frame drops that show up as video jerkiness.
> The demo uses `stack_size=200, fifo_size=400`; increase further if jitter
> returns (e.g. at higher bitrates or slower machines).
