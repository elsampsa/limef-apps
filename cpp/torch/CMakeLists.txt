# CMakeLists.txt for limef libtorch apps
#
# Placeholder — no targets yet.
#
# When ready:
#   1. Download libtorch into apps/ext/libtorch/
#   2. Set TORCH_ROOT here or via -DTORCH_ROOT=...
#   3. Use find_package(Torch REQUIRED HINTS ${TORCH_ROOT})
#   4. Add app .cpp files and link against ${TORCH_LIBRARIES}
#
# Note: Python pipelines don't need this — GPU planes are already exported
# as DLPack capsules (torch.from_dlpack) and CPU planes as numpy arrays
# (torch.from_numpy) with no libtorch linkage required.
#
# Build:
#   mkdir build && cd build
#   ../run_cmake.bash
#   make -j 16
#

cmake_minimum_required(VERSION 3.14)
project(LimefTorchApps)

message(STATUS "")
message(STATUS "LimefTorchApps: no targets yet (placeholder)")
message(STATUS "See tensorframe_torch.h for the planned API.")
message(STATUS "")
