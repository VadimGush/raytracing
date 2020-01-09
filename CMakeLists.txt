cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

project(raytracing LANGUAGES CXX CUDA)

add_executable(raytracing src/Kernel.cu src/Display.cpp src/RayTracer.cu src/utils/logger.cpp src/utils/png.cpp src/utils/file.cpp)

# when we compile C++ files, GCC compiler doesn't know anything about
# CUDA headers
target_include_directories(raytracing PUBLIC /usr/local/cuda/include)

target_link_libraries(raytracing curand png)

