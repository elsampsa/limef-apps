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

OPENCV_VERSION="4.10.0"
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
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# CUDA 12.0 requires GCC <= 12 as host compiler
CUDA_HOST_COMPILER=""
if [ -x /usr/bin/gcc-12 ]; then
    CUDA_HOST_COMPILER="/usr/bin/gcc-12"
    echo "Using GCC 12 as CUDA host compiler (CUDA 12.0 doesn't support GCC 13+)"
fi

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    ${CUDA_HOST_COMPILER:+-DCUDA_HOST_COMPILER="$CUDA_HOST_COMPILER"} \
    -DOPENCV_EXTRA_MODULES_PATH="$SCRIPT_DIR/opencv/opencv_contrib-$OPENCV_VERSION/modules" \
    \
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
