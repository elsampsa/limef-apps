#pragma once
/**
 * @file    tensorframe_ort.h
 * @brief   Zero-copy helpers: TensorFrame::Plane → ONNX Runtime Ort::Value
 *
 * Drop this header into any project that uses both limef TensorFrame and
 * ONNX Runtime.  No limef internals are modified — these are free functions only.
 *
 * Requirements:
 *   - ONNX Runtime C++ API (onnxruntime_cxx_api.h)
 *   - For GPU planes: OrtCUDAProviderOptions must be configured
 *
 * All functions are zero-copy: they wrap the existing buffer pointer.
 * The caller is responsible for keeping the TensorFrame alive while the
 * returned Ort::Value is in use.
 *
 * TODO: implement once ONNX Runtime is available in apps/ext/
 */

// #include "frame/tensorframe.h"
// #include <onnxruntime_cxx_api.h>

// namespace LimefApp {

// /**
//  * Wrap a CPU plane as an Ort::Value tensor (zero-copy).
//  * shape and strides are taken directly from the Plane struct.
//  * dtype must be UInt8, Float32 etc — mapped to ONNXTensorElementDataType.
//  */
// Ort::Value toOrtValue(const Limef::frame::Plane& p, Ort::MemoryInfo& mem_info);

// } // namespace LimefApp
