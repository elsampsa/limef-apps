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
/**
 * @file    gpu_opencv_test.cpp
 * @date    2025
 * @version 1.6.1
 *
 * @brief Unit test for GPUOpenCVThread (TensorFrame path)
 *
 * Sends a CPU TensorFrame (3, H, W) UInt8 filled with a known value into
 * GPUOpenCVThread (gpu_target=CUDA, so the fifo uploads it).  The thread
 * applies per-channel Gaussian blur on the GPU and emits a TensorFrame.
 * The test captures the output, downloads to CPU, and verifies:
 *   - shape (3, H, W) preserved
 *   - frame is on GPU
 *   - interior pixel values ≈ fill value (blur of uniform region is identity)
 *
 * Requires NVIDIA GPU with CUDA support.
 */

#include "gpu_opencv_thread.h"
#include "limef/framefilter/simple.h"
#include "limef/framefilter/dump.h"
#include "limef/hwaccel.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>
#include <chrono>
#include <vector>

using namespace Limef;

std::shared_ptr<spdlog::logger> logger = spdlog::default_logger();

// Capture filter: grabs TensorFrame output for verification
class CaptureFrameFilter : public Limef::ff::SimpleFrameFilter {
public:
    explicit CaptureFrameFilter(std::string name)
        : SimpleFrameFilter(std::move(name)) {}

    void go(const Limef::frame::Frame* frame) override {
        count++;
        auto* tf = frame->as<Limef::frame::TensorFrame>();
        if (!tf) {
            return;
        }
        received = true;
        num_planes = tf->getNumPlanes();
        if (num_planes < 1) return;

        const auto& p = tf->planes[0];
        is_gpu  = p.isGPU();
        ndim    = p.ndim;
        c       = static_cast<int>(p.shape[0]);
        h       = static_cast<int>(p.shape[1]);
        w       = static_cast<int>(p.shape[2]);

        // Download plane 0 to CPU for pixel inspection
        if (is_gpu && p.ndim == 3) {
            size_t nbytes = static_cast<size_t>(c) * h * w;
            cpu_buf.resize(nbytes);
            cudaMemcpy(cpu_buf.data(), p.d_data_, nbytes, cudaMemcpyDeviceToHost);
            has_cpu_data = true;
        }
    }

    std::atomic<int>  count{0};
    bool              received{false};
    int               num_planes{0};
    bool              is_gpu{false};
    int               ndim{0};
    int               c{0}, h{0}, w{0};
    bool              has_cpu_data{false};
    std::vector<uint8_t> cpu_buf;
};

/**
 * Test 1: Gaussian blur round-trip on TensorFrame
 *
 * CPU TensorFrame (3,64,64) filled with 200 →
 * GPUOpenCVThread (uploads H2D, blurs, outputs TensorFrame GPU) →
 * CaptureFilter
 *
 * Verify: shape preserved, frame on GPU, interior pixel ≈ 200
 * (Gaussian blur of a uniform field is identity in the interior)
 */
int test_1()
{
    const char* name = "@TEST: gpu_opencv_test: test 1: ";
    printf("%s ** GPU TensorFrame Gaussian blur round-trip **\n", name);

    if (!isHWAccelAvailable(HWAccel::CUDA)) {
        printf("%s CUDA not available on this system\n", name);
        return 1;
    }
    printf("%s CUDA available\n", name);

    const int W = 64;
    const int H = 64;
    const uint8_t FILL = 200;

    // Output capture
    CaptureFrameFilter capture("capture");

    // GPUOpenCVThread with CUDA fifo (incoming CPU frames get H2D uploaded)
    FrameFifoContext ctx(false, 5, 0, HWAccel::CUDA, "");
    LimefApp::GPUOpenCVThread thread("gpu-opencv", ctx);
    thread.shortLogFormat();
    thread.getOutput().cc(capture);

    // Build CPU TensorFrame (3, H, W) filled with FILL
    frame::TensorFrame src;
    {
        int64_t shape[3] = {3, H, W};
        src.reserveCPUPlane(0, 3, shape, frame::DType::UInt8);
        src.setNumPlanes(1);
        std::memset(src.planes[0].data_, FILL, src.planes[0].contentBytes());
        src.setAbsoluteTimestamp(Limef::timestamp::Timestamp{100000});
        src.setSlot(1);
    }
    printf("%s Source frame: CPU TensorFrame (3,%d,%d) filled with %d\n",
           name, H, W, FILL);

    thread.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    printf("%s Sending frame to GPUOpenCVThread...\n", name);
    thread.getInput().go(&src);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    thread.stop();

    // --- Verify ---
    printf("%s Output frame received: %s\n", name, capture.received ? "yes" : "no");
    if (!capture.received) {
        printf("%s ERROR: no output TensorFrame received\n", name);
        return 1;
    }

    printf("%s num_planes=%d  is_gpu=%d  ndim=%d  shape=(%d,%d,%d)\n",
           name, capture.num_planes, capture.is_gpu,
           capture.ndim, capture.c, capture.h, capture.w);

    if (capture.num_planes != 1) {
        printf("%s ERROR: expected 1 plane, got %d\n", name, capture.num_planes);
        return 1;
    }
    if (!capture.is_gpu) {
        printf("%s ERROR: output plane is not on GPU\n", name);
        return 1;
    }
    if (capture.c != 3 || capture.h != H || capture.w != W) {
        printf("%s ERROR: shape mismatch, expected (3,%d,%d)\n", name, H, W);
        return 1;
    }

    if (!capture.has_cpu_data) {
        printf("%s ERROR: could not download pixel data\n", name);
        return 1;
    }

    // Check interior pixel (center of each channel) ≈ FILL
    // With a 15x15 kernel on a 64x64 frame, center pixels are unaffected by borders
    for (int c = 0; c < 3; ++c) {
        int idx = c * H * W + (H / 2) * W + (W / 2);
        uint8_t pix = capture.cpu_buf[idx];
        printf("%s channel %d center pixel: %d (expected ~%d)\n", name, c, pix, FILL);
        if (pix < FILL - 5 || pix > FILL + 5) {
            printf("%s ERROR: pixel value out of range!\n", name);
            return 1;
        }
    }

    printf("%s PASSED\n", name);
    return 0;
}

int main(int argc, char** argv)
{
    int test_num = 1;
    if (argc > 1) {
        test_num = atoi(argv[1]);
    }

    switch (test_num) {
        case 1: return test_1();
        default:
            printf("Unknown test number: %d\n", test_num);
            return 1;
    }
}
