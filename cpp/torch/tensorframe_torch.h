#pragma once
/**
 * @file    tensorframe_torch.h
 * @brief   Zero-copy helpers: TensorFrame::Plane → torch::Tensor (C++ libtorch)
 *
 * Drop this header into any project that uses both limef TensorFrame and
 * libtorch (C++ PyTorch).  No limef internals are modified — these are free
 * functions only.
 *
 * Requirements:
 *   - libtorch (torch/torch.h)
 *   - For GPU planes: LIMEF_CUDA defined, torch built with CUDA support
 *
 * All functions are zero-copy: they wrap the existing buffer pointer via
 * torch::from_blob().  The caller is responsible for keeping the TensorFrame
 * alive while the returned tensor is in use.
 *
 * Note: Python users do not need this header.  The Python binding already
 * exports GPU planes as DLPack capsules consumed via torch.from_dlpack(),
 * and CPU planes as numpy arrays consumed via torch.from_numpy() — all
 * without linking against libtorch in C++.  This header is for pure C++
 * inference pipelines that link libtorch directly.
 *
 * TODO: implement once libtorch is available in apps/ext/
 */

// #include "frame/tensorframe.h"
// #include <torch/torch.h>

// namespace LimefApp {

// /**
//  * Wrap a CPU plane as a torch::Tensor (zero-copy, via torch::from_blob).
//  * Returns a tensor with shape = plane.shape[0..ndim-1], dtype mapped from DType.
//  * The tensor does NOT own the data — keep the TensorFrame alive.
//  */
// torch::Tensor toTorchTensor(const Limef::frame::Plane& p);

// #ifdef LIMEF_CUDA
// /**
//  * Wrap a GPU plane as a CUDA torch::Tensor (zero-copy, via torch::from_blob
//  * with device=cuda:device_id).
//  */
// torch::Tensor toTorchTensorGPU(const Limef::frame::Plane& p);
// #endif

// } // namespace LimefApp
