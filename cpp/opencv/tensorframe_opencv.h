#pragma once
/**
 * @file    tensorframe_opencv.h
 * @brief   Zero-copy helpers: TensorFrame::Plane ↔ OpenCV GpuMat / Mat
 *
 * Drop this header into any project that uses both limef TensorFrame and
 * OpenCV.  No limef internals are modified — these are free functions only.
 *
 * Requirements:
 *   - OpenCV built with CUDA support (opencv_core, opencv_cudaarithm, ...)
 *   - LIMEF_CUDA defined for GPU helpers
 *
 * All functions are zero-copy: they wrap the existing buffer pointer.
 * The caller is responsible for keeping the TensorFrame alive while the
 * returned Mat / GpuMat is in use.
 *
 * Plane layout assumed: CHW, C-contiguous, dtype UInt8 (CV_8U).
 * Other dtypes are not checked — map manually if needed.
 */

#include "limef/frame/tensorframe.h"

#include <opencv2/core.hpp>
#ifdef LIMEF_CUDA
#include <opencv2/core/cuda.hpp>
#endif

namespace LimefApp {

// ---------------------------------------------------------------------------
// CPU helpers
// ---------------------------------------------------------------------------

/**
 * Wrap a CPU plane as a cv::Mat.
 *
 * For a CHW plane (ndim==3), the returned Mat has size (shape[0]*shape[1], shape[2])
 * and type CV_8UC1 — i.e. the full plane as a single-channel 2D matrix.
 * Use channelMat() to get a per-channel view.
 *
 * For a 2D plane (ndim==2), returns (shape[0], shape[1]) CV_8UC1.
 */
inline cv::Mat toMat(const Limef::frame::Plane& p)
{
    assert(p.data_ != nullptr && "toMat: plane is not a CPU plane");
    assert(p.ndim >= 2 && "toMat: plane must have at least 2 dims");

    int rows, cols;
    size_t step;
    if (p.ndim == 3) {
        rows = static_cast<int>(p.shape[0]) * static_cast<int>(p.shape[1]);
        cols = static_cast<int>(p.shape[2]);
        step = static_cast<size_t>(p.strides[1]);
    } else {
        rows = static_cast<int>(p.shape[0]);
        cols = static_cast<int>(p.shape[1]);
        step = static_cast<size_t>(p.strides[0]);
    }
    return cv::Mat(rows, cols, CV_8UC1, p.data_, step);
}

/**
 * Wrap channel c of a CHW CPU plane as a (H, W) cv::Mat CV_8UC1.
 * Zero-copy: points directly into the plane buffer.
 */
inline cv::Mat channelMat(const Limef::frame::Plane& p, int c)
{
    assert(p.data_ != nullptr && "channelMat: plane is not a CPU plane");
    assert(p.ndim == 3 && "channelMat: plane must be CHW (ndim==3)");
    assert(c >= 0 && c < static_cast<int>(p.shape[0]) && "channelMat: channel index out of range");

    const int H    = static_cast<int>(p.shape[1]);
    const int W    = static_cast<int>(p.shape[2]);
    const size_t step = static_cast<size_t>(p.strides[1]);
    uint8_t* ptr   = p.data_ + static_cast<size_t>(c) * static_cast<size_t>(p.strides[0]);
    return cv::Mat(H, W, CV_8UC1, ptr, step);
}

// ---------------------------------------------------------------------------
// GPU helpers (require LIMEF_CUDA)
// ---------------------------------------------------------------------------

#ifdef LIMEF_CUDA

/**
 * Wrap a GPU plane as a cv::cuda::GpuMat.
 *
 * For a CHW plane (ndim==3), returns a (shape[0]*shape[1], shape[2]) CV_8UC1
 * GpuMat covering the full plane.  Use channelGpuMat() for per-channel views.
 *
 * For a 2D plane (ndim==2), returns (shape[0], shape[1]) CV_8UC1.
 */
inline cv::cuda::GpuMat toGpuMat(const Limef::frame::Plane& p)
{
    assert(p.d_data_ != nullptr && "toGpuMat: plane is not a GPU plane");
    assert(p.ndim >= 2 && "toGpuMat: plane must have at least 2 dims");

    int rows, cols;
    size_t step;
    if (p.ndim == 3) {
        rows = static_cast<int>(p.shape[0]) * static_cast<int>(p.shape[1]);
        cols = static_cast<int>(p.shape[2]);
        step = static_cast<size_t>(p.strides[1]);
    } else {
        rows = static_cast<int>(p.shape[0]);
        cols = static_cast<int>(p.shape[1]);
        step = static_cast<size_t>(p.strides[0]);
    }
    return cv::cuda::GpuMat(rows, cols, CV_8UC1, p.d_data_, step);
}

/**
 * Wrap channel c of a CHW GPU plane as a (H, W) cv::cuda::GpuMat CV_8UC1.
 * Zero-copy: points directly into the device buffer.
 */
inline cv::cuda::GpuMat channelGpuMat(const Limef::frame::Plane& p, int c)
{
    assert(p.d_data_ != nullptr && "channelGpuMat: plane is not a GPU plane");
    assert(p.ndim == 3 && "channelGpuMat: plane must be CHW (ndim==3)");
    assert(c >= 0 && c < static_cast<int>(p.shape[0]) && "channelGpuMat: channel index out of range");

    const int H       = static_cast<int>(p.shape[1]);
    const int W       = static_cast<int>(p.shape[2]);
    const size_t step = static_cast<size_t>(p.strides[1]);
    uint8_t* ptr      = p.d_data_ + static_cast<size_t>(c) * static_cast<size_t>(p.strides[0]);
    return cv::cuda::GpuMat(H, W, CV_8UC1, ptr, step);
}

#endif // LIMEF_CUDA

} // namespace LimefApp
