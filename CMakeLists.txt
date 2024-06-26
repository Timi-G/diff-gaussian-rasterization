cmake_minimum_required(VERSION 3.20)

project(DiffRast LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_library(CudaRasterizer
    cuda_rasterizer/backward.h
    cuda_rasterizer/backward.cu
    cuda_rasterizer/forward.h
    cuda_rasterizer/forward.cu
    cuda_rasterizer/auxiliary.h
    cuda_rasterizer/rasterizer_impl.cu
    cuda_rasterizer/rasterizer_impl.h
    cuda_rasterizer/rasterizer.h
)

target_include_directories(CudaRasterizer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer
    PRIVATE third_party/glm
)
