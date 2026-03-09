#!/bin/bash
#
# Run this from the build directory:
#   mkdir build && cd build
#   ../run_cmake.bash
#   make -j$(nproc)
#
# Binaries will be in bin/. Run with:
#   LD_LIBRARY_PATH=$HOME/limef-stage/lib:<opencv_install>/lib ./bin/<app>
#
# LIMEF_PREFIX selects where to find limef headers and library:
#   ~/limef-stage  (default) — dev staging prefix, set up by ./staging.bash
#   /usr           — if limef was installed from the .deb package
#
# Examples:
#   ../run_cmake.bash                          # use staging prefix
#   LIMEF_PREFIX=/usr ../run_cmake.bash        # use .deb installation
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIMEF_PREFIX="${LIMEF_PREFIX:-$HOME/limef-stage}"
OPENCV_INSTALL="${OPENCV_INSTALL:-$SCRIPT_DIR/../../ext/opencv/install}"

BUILD_TYPE="Debug"
# BUILD_TYPE="Release"

echo
echo "Limef OpenCV Apps - CMake Configuration"
echo "========================================"
echo "LIMEF_PREFIX:   $LIMEF_PREFIX"
echo "OPENCV_INSTALL: $OPENCV_INSTALL"
echo "Source dir:     $SCRIPT_DIR"
echo

cmake \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DLIMEF_PREFIX=$LIMEF_PREFIX \
    -DOPENCV_INSTALL=$OPENCV_INSTALL \
    $SCRIPT_DIR

echo
echo "Run 'make -j\$(nproc)' to build"
echo "Binaries will be in bin/"
echo "Run with:"
echo "  LD_LIBRARY_PATH=$LIMEF_PREFIX/lib:$OPENCV_INSTALL/lib ./bin/<app>"
echo
