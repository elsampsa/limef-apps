/*
 * Copyright (c) 2026 Sampsa Riikonen <sampsa.riikonen@iki.fi>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 */
/*
 * usb_gpu_pipeline - Full USB camera to RTSP pipeline via GPU
 *
 * Pipeline (default, no --modify):
 *   USBCameraThread → UploadGPUFrameFilter → EncodingFrameFilter(NVENC) → RTSPMuxerFrameFilter → RTSPServer
 *
 * Pipeline (with --modify, GPU Gaussian blur via OpenCV):
 *   USBCameraThread → UploadGPUFrameFilter → DecodedToTensorFrameFilter
 *       → GPUOpenCVThread (Gaussian blur on TensorFrame)
 *       → TensorToDecodedFrameFilter → EncodingFrameFilter(NVENC) → RTSPMuxerFrameFilter → RTSPServer
 *
 * NOTE: one copy could be eliminated here:
 * DecodedToTensorFrameFilter copies DecodedFrame -> TensorFrame
 * GPUOpenCVThread copies TensorFrame -> TensorFrame
 * We could have a FrameFifo that copies DecodedFrame directly into a TensorFrame taken from the FrameFifo's stack
 * 
 * Usage:
 *   ./usb_gpu_pipeline [--device /dev/video0] [--port 8554] [--width 640] [--height 480]
 *   ./usb_gpu_pipeline --modify    # enable GPU Gaussian blur
 *
 * Then connect with:
 *   ffplay rtsp://localhost:8554/live/stream
 *   ffplay -rtsp_transport tcp rtsp://localhost:8554/live/stream
 *   ffplay -fflags nobuffer -flags low_delay -framedrop -probesize 32 -analyzeduration 0 rtsp://localhost:8554/live/stream
 *
 * Press Ctrl+C to stop.
 *
 * ## Latency analysis (~500 ms end-to-end measured)
 *
 * Stage                          Latency    Notes
 * ─────────────────────────────────────────────────────────────────────
 * USB camera sensor              33–100 ms  Hardware, unavoidable at 30 fps
 * V4L2 capture (1 buffer)           ~0 ms  Single buffer is correct for low latency
 * SwScale YUYV→NV12                 ~1 ms  Synchronous in camera thread
 * GPU upload + NVENC encode        5–15 ms  Fast with preset=p1, tune=ull
 * GOP wait (gop_size=5 @ 30fps)   0–166 ms  IDR interval; larger → more client wait on connect
 * RTSP + network (localhost)         ~5 ms
 * ffplay decode + display         50–200 ms  Largest variable; use -fflags nobuffer -framedrop
 *
 * Key tuning levers:
 *  - enc_params.gop_size: smaller = faster client connect, higher bitrate.
 *    gop_size=1 (all-intra) is maximum responsiveness; gop_size=5-10 is a good tradeoff.
 *  - NVENC preset p1 + tune ull: already at minimum encoder latency.
 *  - V4L2 req.count=1 in usbcamera.cpp: keeps only the freshest frame, no camera-side queue.
 *  - ffplay flags: -fflags nobuffer -flags low_delay -framedrop -probesize 32 -analyzeduration 0
 */

#include <iostream>
#include <string>
#include <atomic>
#include <csignal>
#include <thread>
#include <chrono>
#include <getopt.h>

#include "gpu_opencv_thread.h"
#include "limef/thread/usbcamera.h"
#include "limef/framefilter/uploadgpu.h"
#include "limef/framefilter/decoded_to_tensor.h"
#include "limef/framefilter/tensor_to_decoded.h"
#include "limef/framefilter/encoding.h"
#include "limef/framefilter/rtspmuxer.h"
#include "limef/framefilter/dump.h"
#include "limef/rtsp/rtspserverthread.h"
#include "limef/encode/ffmpeg_encoder.h"

static std::atomic<bool> g_running{true};

static void signal_handler(int signum) {
    if (signum == SIGINT || signum == SIGTERM) {
        std::cout << "\nShutting down..." << std::endl;
        g_running = false;
    }
}

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\n"
              << "Options:\n"
              << "  --modify, -m          Demo modify the image by blurring on the GPU\n"
              << "  --device, -d <path>   USB camera device (default: /dev/video0)\n"
              << "  --port,   -p <port>   RTSP port (default: 8554)\n"
              << "  --width,  -w <px>     Capture width (default: 640)\n"
              << "  --height, -H <px>     Capture height (default: 480)\n"
              << "  --fps,    -f <fps>    Capture FPS (default: 30)\n"
              << "  --help,   -h          Show this help\n"
              << "\n"
              << "Example:\n"
              << "  " << prog << " --device /dev/video0 --port 8554\n"
              << "\n"
              << "Then connect with:\n"
              << "  ffplay rtsp://localhost:8554/live/stream\n"
              << std::endl;
}

int main(int argc, char** argv) {
    std::string device = "/dev/video0";
    int port = 8554;
    int width = 640;
    int height = 480;
    int fps = 30;
    bool modify = false;

    static struct option long_options[] = {
        {"modify", no_argument, 0, 'm'},
        {"device", required_argument, 0, 'd'},
        {"port",   required_argument, 0, 'p'},
        {"width",  required_argument, 0, 'w'},
        {"height", required_argument, 0, 'H'},
        {"fps",    required_argument, 0, 'f'},
        {"help",   no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "md:p:w:H:f:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'd': device = optarg; break;
            case 'p': port = std::stoi(optarg); break;
            case 'w': width = std::stoi(optarg); break;
            case 'H': height = std::stoi(optarg); break;
            case 'f': fps = std::stoi(optarg); break;
            case 'm': modify = true; break;
            case 'h':
            default:
                print_usage(argv[0]);
                return (opt == 'h') ? 0 : 1;
        }
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    using namespace Limef;
    using namespace Limef::thread;
    using namespace Limef::ff;

    const int SLOT = 1;
    const char* URL_TAIL = "/live/stream";

    std::cout << "==========================================\n";
    std::cout << "  USB Camera → GPU TensorFrame → RTSP Pipeline\n";
    std::cout << "==========================================\n";
    std::cout << "Device:     " << device << "\n";
    std::cout << "Resolution: " << width << "x" << height << " @ " << fps << " fps\n";
    std::cout << "Port:       " << port << "\n";
    std::cout << "URL:        rtsp://localhost:" << port << URL_TAIL << "\n";
    std::cout << "==========================================\n";
    std::cout << "Press Ctrl+C to stop\n\n";

    // --- 1. USB Camera (producer) ---
    USBCameraContext cam_ctx(device, SLOT);
    cam_ctx.width = width;
    cam_ctx.height = height;
    cam_ctx.fps = fps;
    cam_ctx.output_format = AV_PIX_FMT_NV12;

    USBCameraThread camera("usb-camera", cam_ctx);

    // --- 2. GPU Upload ---
    UploadGPUParams upload_params(HWAccel::CUDA);
    UploadGPUFrameFilter upload("gpu-upload", upload_params);

    // --- 3. DecodedFrame → TensorFrame (GPU, NV12 → CHW RGB) ---
    DecodedToTensorFrameFilter d2t("d2t", ChannelOrder::RGB);

    // --- 4. GPU OpenCV Processing (TensorFrame in, TensorFrame out) ---
    FrameFifoContext opencv_ctx(false, 5, 0, HWAccel::CUDA, "");
    LimefApp::GPUOpenCVThread opencv("gpu-opencv", opencv_ctx);

    // --- 5. TensorFrame → DecodedFrame (GPU, CHW RGB → NV12 CUDA) ---
    TensorToDecodedFrameFilter t2d("t2d", ChannelOrder::RGB);

    // --- 6. NVENC H.264 Encoding ---
    encode::FFmpegEncoderParams enc_params;
    enc_params.codec_id = AV_CODEC_ID_H264;
    enc_params.hw_accel = HWAccel::CUDA;
    enc_params.bitrate = 4*1024*1024; // 4Mbps
    enc_params.preset = std::string("p1");
    enc_params.tune = std::string("ull");
    enc_params.max_b_frames = 0;
    enc_params.gop_size = fps/2;
    EncodingFrameFilter encoder("nvenc-encoder", enc_params);

    // --- 7. RTP Muxer ---
    RTSPMuxerFrameFilter rtp_muxer("rtp-muxer");

    // --- 8. RTSP Server ---
    FrameFifoContext rtsp_ctx(false, 5, 100);
    Limef::rtsp::RTSPServerThread rtsp_server("rtsp-server", rtsp_ctx, port);

    // --- Debug: dump filters (uncomment wiring below to enable) ---
    DumpFrameFilter dump_after_camera("DUMP-after-camera");
    DumpFrameFilter dump_after_upload("DUMP-after-upload");
    DumpFrameFilter dump_after_opencv("DUMP-after-opencv");
    DumpFrameFilter dump_after_encoder("DUMP-after-encoder");
    DumpFrameFilter dump_after_rtp("DUMP-after-rtp");

    // --- Wire the pipeline ---

    if (modify) {
        // USBCamera → Upload → d2t → GPUOpenCV(TensorFrame) → t2d → Encoder → RTPMuxer → RTSPServer
        camera.getOutput().cc(upload).cc(d2t).cc(opencv.getInput());
        opencv.getOutput().cc(t2d).cc(encoder).cc(rtp_muxer).cc(rtsp_server.getInput());
    }
    else {
        // just encode and transmit
        camera.getOutput().cc(upload).cc(encoder).cc(rtp_muxer).cc(rtsp_server.getInput());
    }

    // Debug version with dumps at each stage:
    // camera.getOutput().cc(dump_after_camera).cc(upload).cc(dump_after_upload).cc(d2t).cc(opencv.getInput());
    // opencv.getOutput().cc(dump_after_opencv).cc(t2d).cc(encoder).cc(dump_after_encoder).cc(rtp_muxer).cc(dump_after_rtp).cc(rtsp_server.getInput());

    // --- Start (reverse order: downstream first) ---
    std::cout << "Starting RTSP server..." << std::endl;
    rtsp_server.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    rtsp_server.expose(SLOT, URL_TAIL);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::cout << "Starting GPU OpenCV thread..." << std::endl;
    opencv.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << "Starting USB camera..." << std::endl;
    camera.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    std::cout << "\nReady! Connect with:\n";
    std::cout << "  ffplay rtsp://localhost:" << port << URL_TAIL << "\n";
    std::cout << "  ffplay -rtsp_transport tcp rtsp://localhost:" << port << URL_TAIL << "\n";
    std::cout << "  ffplay -fflags nobuffer -flags low_delay -framedrop -probesize 32 -analyzeduration 0 rtsp://localhost:" << port << URL_TAIL  << "\n";

    // --- Main loop ---
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // --- Stop (reverse order: upstream first) ---
    std::cout << "Stopping camera..." << std::endl;
    camera.stop();

    std::cout << "Stopping GPU OpenCV..." << std::endl;
    opencv.stop();

    std::cout << "Stopping RTSP server..." << std::endl;
    rtsp_server.stop();

    std::cout << "Done." << std::endl;
    return 0;
}
