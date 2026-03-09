# CMakeLists.txt for limef base apps (no CUDA/OpenCV deps)
#
# Build:
#   mkdir build && cd build
#   ../run_cmake.bash
#   make -j$(nproc)
#
# Requires staging prefix (run ./staging.bash from the limef repo root first):
#   cmake -DLIMEF_PREFIX=$HOME/limef-stage ..
#

cmake_minimum_required(VERSION 3.14)
project(LimefBaseApps)

set(LIMEF_PREFIX "$ENV{HOME}/limef-stage" CACHE PATH "Path to limef staging prefix")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(PkgConfig REQUIRED)
pkg_search_module(X11 REQUIRED x11)
pkg_search_module(ALSA REQUIRED alsa)
pkg_search_module(OPENSSL REQUIRED openssl)
find_package(ZLIB REQUIRED)

exec_program(pkg-config ARGS --libs x11 glew OUTPUT_VARIABLE GLEW_LIBRARIES)

exec_program(python3-config ARGS --includes OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS_TMP)
string(REPLACE "-I" " " PYTHON_INCLUDE_DIRS_TMP ${PYTHON_INCLUDE_DIRS_TMP})
string(REPLACE " " ";" PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS_TMP})
exec_program(python3 ARGS "-c \"import numpy; print(numpy.get_include(), end='')\"" OUTPUT_VARIABLE NUMPY_INCLUDE_DIR)

# All limef headers and bundled ext deps live under include/limef/
# Apps write: #include "limef/thread/thread.h"   (needs include/ in path)
# Bundled ext deps (spdlog, flatbuffers, uws) are also under include/limef/
#   so <spdlog/spdlog.h> etc. resolve with include/limef/ in path
set(COMMON_INCLUDES
    ${LIMEF_PREFIX}/include              # for "limef/xxx.h" includes
    ${LIMEF_PREFIX}/include/limef        # for <spdlog/...>, <flatbuffers/...>, <uws/...>
    ${LIMEF_PREFIX}/include/limef/ffmpeg # for <libavcodec/avcodec.h> etc.
    ${PYTHON_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
)

set(COMMON_LIBS
    ${X11_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${ALSA_LIBRARIES}
    ${OPENSSL_LIBRARIES}
    ${ZLIB_LIBRARIES}
    pthread
)

function(add_limef_app name)
    add_executable(${name} ${name}.cpp)
    target_include_directories(${name} PRIVATE ${COMMON_INCLUDES})
    target_link_libraries(${name} PRIVATE
        ${COMMON_LIBS}
        "-L${LIMEF_PREFIX}/lib"
        "Limef.so"
    )
endfunction()

file(GLOB APP_SOURCES "*.cpp")
foreach(source ${APP_SOURCES})
    get_filename_component(name ${source} NAME_WE)
    message(STATUS "Adding app: ${name}")
    add_limef_app(${name})
endforeach()

message(STATUS "")
message(STATUS "Configuration:")
message(STATUS "  LIMEF_PREFIX: ${LIMEF_PREFIX}")
message(STATUS "  Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "")
