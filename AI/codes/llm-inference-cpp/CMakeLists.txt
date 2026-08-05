cmake_minimum_required(VERSION 3.18)

project(llm_inference_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LLM_INFERENCE_ENABLE_CUDA "Use CUDA/cuBLAS for matvec when CUDA Toolkit is available" ON)

add_executable(llm-inference-cpp
    src/core/cli.cpp
    src/core/common.cpp
    src/core/config.cpp
    src/core/profile.cpp
    src/safetensors/safetensors.cpp
    src/core/tensor_ops.cpp
    src/core/tokenizer.cpp
    src/kernels/cpu/dtype.cpp
    src/kernels/cpu/matvec.cpp
    src/kernels/cpu/embedding.cpp
    src/kernels/cpu/rmsnorm.cpp
    src/kernels/cpu/elementwise.cpp
    src/ops/ops.cpp
    src/main.cpp
    src/model/QwenModel.cpp
    src/model/weights.cpp
)

target_include_directories(llm-inference-cpp SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(llm-inference-cpp PRIVATE ${ACCELERATE_FRAMEWORK})
    target_compile_definitions(llm-inference-cpp PRIVATE ACCELERATE_NEW_LAPACK=1)
else()
    find_package(BLAS REQUIRED)
    target_link_libraries(llm-inference-cpp PRIVATE ${BLAS_LIBRARIES})
    include(CheckIncludeFileCXX)
    check_include_file_cxx(cblas.h LLM_INFERENCE_HAS_CBLAS_H)
    if(NOT LLM_INFERENCE_HAS_CBLAS_H)
        message(FATAL_ERROR "默认 CPU BLAS 路径需要 cblas.h")
    endif()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(llm-inference-cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

if(LLM_INFERENCE_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_sources(llm-inference-cpp PRIVATE src/kernels/cuda/cuda_ops.cpp)
        target_sources(llm-inference-cpp PRIVATE src/core/cuda_kernels.cu)
        if(CMAKE_CUDA_ARCHITECTURES)
            set_property(TARGET llm-inference-cpp PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
        else()
            set_property(TARGET llm-inference-cpp PROPERTY CUDA_ARCHITECTURES 86)
        endif()
        target_link_libraries(llm-inference-cpp PRIVATE CUDA::cudart CUDA::cublas)
        set_property(TARGET llm-inference-cpp PROPERTY CUDA_STANDARD 17)
        set_property(TARGET llm-inference-cpp PROPERTY CUDA_STANDARD_REQUIRED ON)
    else()
        target_sources(llm-inference-cpp PRIVATE src/kernels/cuda/cuda_ops_stub.cpp)
    endif()
else()
    target_sources(llm-inference-cpp PRIVATE src/kernels/cuda/cuda_ops_stub.cpp)
endif()

if(MSVC)
    target_compile_options(llm-inference-cpp PRIVATE /W4)
else()
    target_compile_options(llm-inference-cpp PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    )
endif()
