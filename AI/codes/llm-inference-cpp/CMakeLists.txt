cmake_minimum_required(VERSION 3.18)

project(llm_inference_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LLM_INFERENCE_ENABLE_CBLAS "Use CBLAS for CPU matvec; requires enough memory for float weight cache" OFF)
option(LLM_INFERENCE_ENABLE_CUDA "Use CUDA/cuBLAS for matvec when CUDA Toolkit is available" ON)

add_executable(llm-inference-cpp
    src/core/cli.cpp
    src/core/common.cpp
    src/core/config.cpp
    src/core/profile.cpp
    src/core/safetensors.cpp
    src/core/tensor_ops.cpp
    src/kernels/cuda/cuda_ops.cpp
    src/core/tokenizer.cpp
    src/kernels/cpu/dtype.cpp
    src/kernels/cpu/matvec.cpp
    src/kernels/cpu/embedding.cpp
    src/kernels/cpu/rmsnorm.cpp
    src/kernels/cpu/elementwise.cpp
    src/tensor/tensor.cpp
    src/ops/ops.cpp
    src/main.cpp
    src/model/QwenModel.cpp
    src/model/weights.cpp
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(llm-inference-cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

if(LLM_INFERENCE_ENABLE_CBLAS)
    find_package(BLAS REQUIRED)
    if(BLAS_FOUND)
        target_link_libraries(llm-inference-cpp PRIVATE ${BLAS_LIBRARIES})
        include(CheckIncludeFileCXX)
        check_include_file_cxx(cblas.h LLM_INFERENCE_HAS_CBLAS_H)
        if(LLM_INFERENCE_HAS_CBLAS_H)
            target_compile_definitions(llm-inference-cpp PRIVATE LLM_INFERENCE_USE_CBLAS=1)
        else()
            message(FATAL_ERROR "LLM_INFERENCE_ENABLE_CBLAS=ON but cblas.h was not found")
        endif()
    endif()
endif()

if(LLM_INFERENCE_ENABLE_CUDA)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        target_sources(llm-inference-cpp PRIVATE src/core/cuda_kernels.cu)
        if(CMAKE_CUDA_ARCHITECTURES)
            set_property(TARGET llm-inference-cpp PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
        else()
            set_property(TARGET llm-inference-cpp PROPERTY CUDA_ARCHITECTURES 86)
        endif()
        target_link_libraries(llm-inference-cpp PRIVATE CUDA::cudart CUDA::cublas)
        target_compile_definitions(llm-inference-cpp PRIVATE LLM_INFERENCE_USE_CUDA_CUBLAS=1)
        set_property(TARGET llm-inference-cpp PROPERTY CUDA_STANDARD 17)
        set_property(TARGET llm-inference-cpp PROPERTY CUDA_STANDARD_REQUIRED ON)
    endif()
endif()

if(MSVC)
    target_compile_options(llm-inference-cpp PRIVATE /W4)
else()
    target_compile_options(llm-inference-cpp PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    )
endif()
