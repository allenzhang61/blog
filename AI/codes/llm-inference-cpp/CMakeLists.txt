cmake_minimum_required(VERSION 3.18)

project(llm_inference_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(llm-inference-cpp
    src/core/cli.cpp
    src/core/common.cpp
    src/core/config.cpp
    src/core/profile.cpp
    src/safetensors/safetensors.cpp
    src/core/tokenizer.cpp
    src/core/cuda_kernels.cu
    src/kernels/cuda/cuda_common.cpp
    src/kernels/cuda/cuda_ops.cpp
    src/kernels/cuda/cuda_scratch_buffer.cpp
    src/kernels/cuda/cuda_weight_cache.cpp
    src/main.cpp
    src/model/QwenModel.cpp
    src/model/weights.cpp
)

target_include_directories(llm-inference-cpp SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

find_package(CUDAToolkit REQUIRED)
if(CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET llm-inference-cpp PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
else()
    set_property(TARGET llm-inference-cpp PROPERTY CUDA_ARCHITECTURES 86)
endif()
target_link_libraries(llm-inference-cpp PRIVATE CUDA::cudart CUDA::cublas)
set_property(TARGET llm-inference-cpp PROPERTY CUDA_STANDARD 17)
set_property(TARGET llm-inference-cpp PROPERTY CUDA_STANDARD_REQUIRED ON)

if(MSVC)
    target_compile_options(llm-inference-cpp PRIVATE /W4)
else()
    target_compile_options(llm-inference-cpp PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    )
endif()
