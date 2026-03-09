# CMakeLists.txt for limef ONNX Runtime apps
#
# Placeholder — no targets yet.
#
# When ready:
#   1. Install ONNX Runtime into apps/ext/onnxruntime/
#   2. Set ONNXRUNTIME_ROOT here or via -DONNXRUNTIME_ROOT=...
#   3. Add app .cpp files and wire up find_library / include_directories
#
# Build:
#   mkdir build && cd build
#   ../run_cmake.bash
#   make -j 16
#

cmake_minimum_required(VERSION 3.14)
project(LimefONNXApps)

message(STATUS "")
message(STATUS "LimefONNXApps: no targets yet (placeholder)")
message(STATUS "See tensorframe_ort.h for the planned API.")
message(STATUS "")
