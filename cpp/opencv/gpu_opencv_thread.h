#pragma once
/**
 * @file    gpu_opencv_thread.h
 * @date    2025
 * @version 1.6.1
 *
 * @brief Thread that applies Gaussian blur to GPU TensorFrames using OpenCV CUDA.
 *
 * Receives TensorFrames with shape (3, H, W) UInt8 on GPU, applies per-channel
 * Gaussian blur via OpenCV CUDA, and emits TensorFrames downstream.
 *
 * The thread uses TensorFrameFifo with gpu_target=CUDA so that incoming CPU
 * frames are automatically uploaded to GPU in a single H2D copy.
 *
 * Typical pipeline:
 *   USBCamera → UploadGPU → DecodedToTensorFrameFilter
 *       → GPUOpenCVThread → TensorToDecodedFrameFilter → NVENC → RTSP
 *
 * Non-TensorFrame frames (SignalFrame, StreamFrame, etc.) pass through unchanged.
 * Non-video or wrongly-shaped TensorFrames are logged and dropped.
 */

#include "limef/thread/thread.h"
#include "limef/framefifo/tensorframefifo.h"
#include "limef/frame/tensorframe.h"

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>

namespace LimefApp {

class GPUOpenCVThread : public Limef::thread::Thread<Limef::frame::TensorFrame> {
    THREAD_CLASS(GPUOpenCVThread);

public:
    GPUOpenCVThread(std::string name, const Limef::FrameFifoContext& ctx)
        : Thread(std::move(name), ctx)
    {}

protected:
    void createFrameFifo() override {
        framefifo = std::make_unique<Limef::TensorFrameFifo>(name_, ctx);
    }

    void processFrame(Limef::frame::Frame* frame) override {
        auto* tf = frame->as<Limef::frame::TensorFrame>();
        if (!tf) {
            output_ff.go(frame);
            return;
        }

        if (tf->getNumPlanes() != 1) {
            logger->warn("{}: expected 1 plane, got {}", name_, tf->getNumPlanes());
            return;
        }

        const auto& p = tf->planes[0];
        if (!p.isGPU() || p.ndim != 3 || p.shape[0] != 3 ||
            p.dtype != Limef::frame::DType::UInt8) {
            logger->warn("{}: expected GPU plane (3, H, W) UInt8, "
                         "got ndim={} shape[0]={} isGPU={} dtype={}",
                         name_, p.ndim, p.shape[0], p.isGPU(),
                         static_cast<int>(p.dtype));
            return;
        }

        const int C = static_cast<int>(p.shape[0]);
        const int H = static_cast<int>(p.shape[1]);
        const int W = static_cast<int>(p.shape[2]);

        if (!ensureOutput(H, W)) {
            return;
        }

        // Lazy-init Gaussian filter; recreate on size change
        if (!gauss_filter_ || last_h_ != H || last_w_ != W) {
            gauss_filter_ = cv::cuda::createGaussianFilter(
                CV_8UC1, CV_8UC1, cv::Size(15, 15), 0.0);
            last_h_ = H;
            last_w_ = W;
        }

        // CHW layout: channel c starts at d_data_ + c * strides[0]
        // Each row within a channel is strides[1] bytes wide
        const size_t ch_stride  = static_cast<size_t>(p.strides[0]);
        const size_t row_stride = static_cast<size_t>(p.strides[1]);
        auto& op = output_.planes[0];

        for (int c = 0; c < C; ++c) {
            cv::cuda::GpuMat ch_in (H, W, CV_8UC1,
                                    p.d_data_  + c * ch_stride, row_stride);
            cv::cuda::GpuMat ch_out(H, W, CV_8UC1,
                                    op.d_data_ + c * ch_stride, row_stride);
            gauss_filter_->apply(ch_in, ch_out);
        }

        output_.setAbsoluteTimestamp(tf->getAbsoluteTimestamp());
        output_.setSlot(tf->getSlot());
        output_ff.go(&output_);
    }

private:
    bool ensureOutput(int H, int W) {
        auto& op = output_.planes[0];
        if (op.isGPU() && op.shape[1] == H && op.shape[2] == W) {
            return true;
        }
        int64_t shape[3] = {3, static_cast<int64_t>(H), static_cast<int64_t>(W)};
        output_.setNumPlanes(1);
        return output_.reserveGPUPlane(0, 3, shape, Limef::frame::DType::UInt8, 0);
    }

    Limef::frame::TensorFrame  output_;
    cv::Ptr<cv::cuda::Filter>  gauss_filter_;
    int                        last_h_ = 0;
    int                        last_w_ = 0;
};

} // namespace LimefApp
