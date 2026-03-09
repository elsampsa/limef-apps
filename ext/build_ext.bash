#!/bin/bash
# Build all external dependencies for limef apps
# This script is idempotent - safe to run multiple times
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Building apps external dependencies"
echo "========================================"

# OpenCV with CUDA
echo ""
echo ">>> OpenCV (CUDA)"
./download_opencv.bash

echo ""
echo "========================================"
echo "All apps external dependencies built!"
echo "========================================"
