#!/bin/bash
# Download and build OpenCV with CUDA support
# Builds only the modules needed for GPU image processing
#
# Prerequisites:
#   - CUDA toolkit (nvcc, libcudart) - e.g. nvidia-cuda-toolkit package
#   - CMake 3.14+
#   - Standard build tools (gcc, g++, make)
#
# Result:
#   apps/ext/opencv/install/  - headers and libraries
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect CUDA major version (nvcc is authoritative; fall back to 12)
CUDA_MAJOR=12
if command -v nvcc &>/dev/null; then
    CUDA_MAJOR=$(nvcc --version | grep -oP 'release \K[0-9]+' | head -1)
fi
echo "Detected CUDA major version: $CUDA_MAJOR"

# CUDA 12 → OpenCV 4.10.0 (no upstream patches needed)
# CUDA 13+ → OpenCV 4.13.0 (requires three source patches for CCCL 2.x / NPP 13.x)
if [ "$CUDA_MAJOR" -ge 13 ]; then
    OPENCV_VERSION="4.13.0"
    CUDA13_CMAKE_FLAGS=(
        "-DCMAKE_CUDA_STANDARD=17"
        "-DCUDA_NVCC_FLAGS=-std=c++17 --expt-relaxed-constexpr"
    )
else
    OPENCV_VERSION="4.10.0"
    CUDA13_CMAKE_FLAGS=()
fi
echo "Using OpenCV $OPENCV_VERSION"

INSTALL_DIR="$SCRIPT_DIR/opencv/install"
BUILD_DIR="$SCRIPT_DIR/opencv/build"

# Skip if already built
if [ -f "$INSTALL_DIR/lib/libopencv_core.so" ]; then
    echo "OpenCV already built at $INSTALL_DIR"
    echo "Remove $SCRIPT_DIR/opencv/ to rebuild"
    exit 0
fi

echo "========================================"
echo "Building OpenCV $OPENCV_VERSION with CUDA"
echo "========================================"

# Download OpenCV
if [ ! -d "$SCRIPT_DIR/opencv/opencv-$OPENCV_VERSION" ]; then
    echo ">>> Downloading OpenCV $OPENCV_VERSION..."
    mkdir -p "$SCRIPT_DIR/opencv"
    cd "$SCRIPT_DIR/opencv"
    wget -q --show-progress "https://github.com/opencv/opencv/archive/refs/tags/$OPENCV_VERSION.tar.gz" -O opencv.tar.gz
    tar xf opencv.tar.gz
    rm opencv.tar.gz
fi

# Download OpenCV contrib (for cuda modules)
if [ ! -d "$SCRIPT_DIR/opencv/opencv_contrib-$OPENCV_VERSION" ]; then
    echo ">>> Downloading OpenCV contrib $OPENCV_VERSION..."
    cd "$SCRIPT_DIR/opencv"
    wget -q --show-progress "https://github.com/opencv/opencv_contrib/archive/refs/tags/$OPENCV_VERSION.tar.gz" -O opencv_contrib.tar.gz
    tar xf opencv_contrib.tar.gz
    rm opencv_contrib.tar.gz
fi

# --- CUDA 13.x source patches -------------------------------------------------
# These fix cudev/NPP API changes in CUDA 13 / CCCL 2.x that aren't yet fixed
# upstream in OpenCV.  Applied once after download; idempotent (sed -i is safe
# to run on already-patched files).
if [ "$CUDA_MAJOR" -ge 13 ]; then

# Patch 1: cudev detail/tuple.hpp
#   thrust::tuple in CCCL 2.x (CUDA 13) confuses nvcc C++17 template lookup for
#   3+-element tuple overloads.  std::tuple is equivalent and works correctly.
TUPLE_HPP="$SCRIPT_DIR/opencv/opencv_contrib-$OPENCV_VERSION/modules/cudev/include/opencv2/cudev/util/detail/tuple.hpp"
if [ -f "$TUPLE_HPP" ] && grep -q "thrust/tuple.h" "$TUPLE_HPP"; then
    echo ">>> Patching cudev/tuple.hpp (thrust::tuple → std::tuple for CCCL 2.x)..."
    sed -i 's|#include <thrust/tuple.h>|#include <tuple>|'          "$TUPLE_HPP"
    sed -i 's/using thrust::tuple;/using std::tuple;/'              "$TUPLE_HPP"
    sed -i 's/using thrust::tuple_size;/using std::tuple_size;/'    "$TUPLE_HPP"
    sed -i 's/using thrust::get;/using std::get;/'                  "$TUPLE_HPP"
    sed -i 's/using thrust::tuple_element;/using std::tuple_element;/' "$TUPLE_HPP"
    sed -i 's/using thrust::make_tuple;/using std::make_tuple;/'    "$TUPLE_HPP"
    sed -i 's/using thrust::tie;/using std::tie;/'                  "$TUPLE_HPP"
fi

# Patch 2: cudev ptr2d/zip.hpp
#   _LIBCUDACXX_BEGIN/END_NAMESPACE_STD was renamed to _CCCL_BEGIN/END_NAMESPACE_STD
#   in CCCL 2.x.  The CUDA ≥12.4 block in zip.hpp uses the old name.
ZIP_HPP="$SCRIPT_DIR/opencv/opencv_contrib-$OPENCV_VERSION/modules/cudev/include/opencv2/cudev/ptr2d/zip.hpp"
if [ -f "$ZIP_HPP" ] && grep -q "_LIBCUDACXX_BEGIN_NAMESPACE_STD" "$ZIP_HPP"; then
    echo ">>> Patching cudev/zip.hpp (_LIBCUDACXX → _CCCL namespace macros)..."
    sed -i 's/_LIBCUDACXX_BEGIN_NAMESPACE_STD/_CCCL_BEGIN_NAMESPACE_STD/g' "$ZIP_HPP"
    sed -i 's/_LIBCUDACXX_END_NAMESPACE_STD/_CCCL_END_NAMESPACE_STD/g'     "$ZIP_HPP"
fi

# Patch 3: core/private.cuda.hpp
#   nppGetStreamContext() was removed in NPP 13.x.  Populate NppStreamContext
#   manually via cudaGetDeviceProperties instead.
PRIVATE_CUDA_HPP="$SCRIPT_DIR/opencv/opencv-$OPENCV_VERSION/modules/core/include/opencv2/core/private.cuda.hpp"
if [ -f "$PRIVATE_CUDA_HPP" ] && grep -q "nppGetStreamContext" "$PRIVATE_CUDA_HPP"; then
    echo ">>> Patching private.cuda.hpp (nppGetStreamContext removed in NPP 13.x)..."
    python3 - "$PRIVATE_CUDA_HPP" <<'PYEOF'
import sys
path = sys.argv[1]
text = open(path).read()
old = (
    '            nppStreamContext = {};\n'
    '            nppSafeCall(nppGetStreamContext(&nppStreamContext));\n'
    '            nppStreamContext.hStream = newStream;\n'
    '            cudaSafeCall(cudaStreamGetFlags(nppStreamContext.hStream, &nppStreamContext.nStreamFlags));'
)
new = (
    '            nppStreamContext = {};\n'
    '#if NPP_VERSION < 13000\n'
    '            nppSafeCall(nppGetStreamContext(&nppStreamContext));\n'
    '            nppStreamContext.hStream = newStream;\n'
    '            cudaSafeCall(cudaStreamGetFlags(nppStreamContext.hStream, &nppStreamContext.nStreamFlags));\n'
    '#else\n'
    '            // nppGetStreamContext removed in NPP 13.x\n'
    '            nppStreamContext.hStream = newStream;\n'
    '            cudaSafeCall(cudaGetDevice(&nppStreamContext.nCudaDeviceId));\n'
    '            cudaDeviceProp props = {};\n'
    '            cudaSafeCall(cudaGetDeviceProperties(&props, nppStreamContext.nCudaDeviceId));\n'
    '            nppStreamContext.nMultiProcessorCount             = props.multiProcessorCount;\n'
    '            nppStreamContext.nMaxThreadsPerMultiProcessor     = props.maxThreadsPerMultiProcessor;\n'
    '            nppStreamContext.nMaxThreadsPerBlock              = props.maxThreadsPerBlock;\n'
    '            nppStreamContext.nSharedMemPerBlock               = props.sharedMemPerBlock;\n'
    '            nppStreamContext.nCudaDevAttrComputeCapabilityMajor = props.major;\n'
    '            nppStreamContext.nCudaDevAttrComputeCapabilityMinor = props.minor;\n'
    '            cudaSafeCall(cudaStreamGetFlags(nppStreamContext.hStream, &nppStreamContext.nStreamFlags));\n'
    '#endif'
)
if old in text:
    open(path, 'w').write(text.replace(old, new, 1))
    print("  applied.")
else:
    print("  already patched or pattern not found.")
PYEOF
fi

fi  # end CUDA 13+ patches
# ------------------------------------------------------------------------------

# Detect CUDA compute capability from GPU
CUDA_ARCH=""
if command -v nvidia-smi &>/dev/null; then
    CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
    echo "Detected GPU compute capability: $CUDA_ARCH"
fi
if [ -z "$CUDA_ARCH" ]; then
    echo "WARNING: Could not detect GPU. Using common architectures."
    CUDA_ARCH="7.5;8.0;8.6;8.9"
fi

echo ">>> Configuring OpenCV with CUDA (arch=$CUDA_ARCH)..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# CUDA 13.x supports GCC 13; keep gcc-12 fallback for older BSPs
CUDA_HOST_COMPILER=""
if [ -x /usr/bin/gcc-12 ]; then
    CUDA_HOST_COMPILER="/usr/bin/gcc-12"
    echo "Using GCC 12 as CUDA host compiler"
fi

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    ${CUDA_HOST_COMPILER:+-DCUDA_HOST_COMPILER="$CUDA_HOST_COMPILER"} \
    -DOPENCV_EXTRA_MODULES_PATH="$SCRIPT_DIR/opencv/opencv_contrib-$OPENCV_VERSION/modules" \
    \
    "${CUDA13_CMAKE_FLAGS[@]}" \
    -DWITH_CUDA=ON \
    -DCUDA_ARCH_BIN="$CUDA_ARCH" \
    -DCUDA_FAST_MATH=ON \
    -DWITH_CUBLAS=ON \
    -DWITH_CUFFT=OFF \
    -DWITH_NVCUVID=OFF \
    -DWITH_NVCUVENC=OFF \
    \
    -DBUILD_LIST=core,imgproc,cudev,cudaimgproc,cudawarping,cudaarithm,cudafilters \
    \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_opencv_python3=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_DOCS=OFF \
    -DBUILD_opencv_java=OFF \
    -DBUILD_opencv_js=OFF \
    \
    -DWITH_GTK=OFF \
    -DWITH_QT=OFF \
    -DWITH_OPENGL=OFF \
    -DWITH_V4L=OFF \
    -DWITH_FFMPEG=OFF \
    -DWITH_GSTREAMER=OFF \
    -DWITH_1394=OFF \
    -DWITH_OPENEXR=OFF \
    -DWITH_JASPER=OFF \
    -DWITH_TIFF=OFF \
    -DWITH_WEBP=OFF \
    -DWITH_OPENJPEG=OFF \
    \
    "$SCRIPT_DIR/opencv/opencv-$OPENCV_VERSION"

echo ">>> Building OpenCV (this may take a while)..."
make -j$(nproc)

echo ">>> Installing OpenCV to $INSTALL_DIR..."
make install

echo ""
echo "========================================"
echo "OpenCV $OPENCV_VERSION with CUDA built!"
echo "Install dir: $INSTALL_DIR"
echo "CUDA arch:   $CUDA_ARCH"
echo "========================================"
