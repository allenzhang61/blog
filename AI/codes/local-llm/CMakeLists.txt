cmake_minimum_required(VERSION 3.22.1)

# 在 project() 启用 CUDA 语言前，兜底定位 nvcc：CLion 等 IDE 的默认 profile 常常
# 不会显式传 -DCMAKE_CUDA_COMPILER，而 nvcc 通常不在默认 PATH（在 /usr/local/cuda/bin）。
# 仅在用户/环境未指定时探测常见路径，不覆盖显式传入的值。
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    find_program(_LOCAL_LLM_NVCC nvcc
            HINTS ${CUDA_TOOLKIT_ROOT_DIR} ENV CUDA_PATH
            PATHS /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.8 /usr/local/cuda-12.4
            PATH_SUFFIXES bin)
    if(_LOCAL_LLM_NVCC)
        set(CMAKE_CUDA_COMPILER "${_LOCAL_LLM_NVCC}")
    endif()
endif()

project(local_llm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# 目标架构：RTX A4000 为 sm_86。可用 -DCMAKE_CUDA_ARCHITECTURES=xx 覆盖。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_executable(local_llm main.cpp
        src/utils/cli/Args.cpp
        src/utils/cli/Args.h
        src/backend/common.h
        src/utils/log/Log.cpp
        src/utils/log/Log.h
        src/llm/model/qwen/QwenConfig.cpp
        src/llm/model/qwen/QwenConfig.h
        src/llm/model/qwen/QwenWeights.cpp
        src/llm/model/qwen/QwenWeights.h
        src/backend/cuda/common.cpp
        src/backend/cuda/common.h
        src/backend/cuda/mem/CudaWeightPool.cpp
        src/backend/cuda/mem/CudaWeightPool.h
        src/backend/cuda/mem/CudaWeight.cpp
        src/backend/cuda/mem/CudaWeight.h
        src/backend/cuda/mem/CudaWeightDequantPool.cpp
        src/backend/cuda/mem/CudaWeightDequantPool.h
        src/backend/cuda/mem/CudaScratchBuffer.cpp
        src/backend/cuda/mem/CudaScratchBuffer.h
        src/backend/cuda/mem/CudaScratch.cpp
        src/backend/cuda/mem/CudaScratch.h
        src/backend/cuda/mem/SessionBase.h
        src/backend/cuda/mem/Quant.cpp
        src/backend/cuda/mem/Quant.h
        src/llm/model/qwen/QwenSession.cpp
        src/llm/model/qwen/QwenSession.h
        src/backend/cuda/ops/kernel.cu
        src/backend/cuda/ops/kernel.cuh
        src/backend/cuda/ops/kernel_internal.cuh
        src/backend/cuda/ops/launch.cu
        src/backend/cuda/ops/gemm.cpp
        src/backend/cuda/ops/gemm.h
        src/format/gguf/GgufFile.cpp
        src/format/gguf/GgufFile.h
        src/format/MF.cpp
        src/format/MF.h
        src/tensor/TensorCommon.h
        src/tensor/StorageTensor.cpp
        src/tensor/StorageTensor.h
        src/tensor/CPUTensor.cpp
        src/tensor/CPUTensor.h
        src/tensor/GPUTensor.cpp
        src/tensor/GPUTensor.h
        src/tensor/TensorTool.cpp
        src/tensor/TensorTool.h
        src/format/MFFactory.cpp
        src/format/MFFactory.h
        src/format/hf/HFFile.cpp
        src/format/hf/HFFile.h
        src/format/hf/HFTokenizer.cpp
        src/format/hf/HFTokenizer.h
        src/llm/model/BaseModel.h
        src/llm/module/Module.h
        src/llm/model/ModelFactory.cpp
        src/llm/model/ModelFactory.h
        src/utils/sampling/Sampler.cpp
        src/utils/sampling/Sampler.h
        src/llm/module/common/Embedding.cpp
        src/llm/module/common/Embedding.h
        src/llm/module/common/RMSNorm.cpp
        src/llm/module/common/RMSNorm.h
        src/llm/module/common/LMHead.cpp
        src/llm/module/common/LMHead.h
        src/llm/module/qwen/SwiGLUMlp.cpp
        src/llm/module/qwen/SwiGLUMlp.h
        src/llm/module/qwen/FullAttention.cpp
        src/llm/module/qwen/FullAttention.h
        src/llm/module/qwen/LinearAttention.cpp
        src/llm/module/qwen/LinearAttention.h
        src/llm/module/qwen/DecoderLayer.cpp
        src/llm/module/qwen/DecoderLayer.h
        src/llm/model/qwen/QwenModel.cpp
        src/llm/model/qwen/QwenModel.h
        src/llm/model/deepseek/DeepseekConfig.cpp
        src/llm/model/deepseek/DeepseekConfig.h
        src/llm/model/deepseek/DeepseekSession.cpp
        src/llm/model/deepseek/DeepseekSession.h
        src/llm/model/deepseek/DeepseekWeights.cpp
        src/llm/model/deepseek/DeepseekWeights.h
        src/format/gguf/GGUFTokenizer.cpp
        src/format/gguf/GGUFTokenizer.h
        src/llm/module/deepseek/DenseFFN.cpp
        src/llm/module/deepseek/DenseFFN.h
        src/llm/module/deepseek/MLA.cpp
        src/llm/module/deepseek/MLA.h
        src/llm/module/deepseek/MoE.cpp
        src/llm/module/deepseek/MoE.h
        src/llm/module/deepseek/MoERouter.cpp
        src/llm/module/deepseek/MoERouter.h
        src/llm/module/deepseek/MLP.cpp
        src/llm/module/deepseek/MLP.h
        src/llm/module/deepseek/RoutedExperts.cpp
        src/llm/module/deepseek/RoutedExperts.h
        src/llm/module/deepseek/SharedExperts.cpp
        src/llm/module/deepseek/SharedExperts.h
        src/llm/model/deepseek/DeepseekModel.cpp
        src/llm/model/deepseek/DeepseekModel.h
        src/utils/stats/StatsReport.h
        src/utils/stats/MemoryUsageProvider.h
        src/utils/stats/Profiler.cpp
        src/utils/stats/Profiler.h
        src/utils/stats/ScopedTimer.cpp
        src/utils/stats/ScopedTimer.h
        src/utils/stats/MemoryReporter.cpp
        src/utils/stats/MemoryReporter.h
        src/utils/stats/DeviceMonitor.cpp
        src/utils/stats/DeviceMonitor.h
        src/utils/stats/WeightLoadTracker.cpp
        src/utils/stats/WeightLoadTracker.h
        src/llm/module/common/Embedding.cpp
)

target_include_directories(local_llm PRIVATE src)

# 外部依赖：PCRE2（pre-tokenize 的 Unicode 正则）、utf8proc（NFC 规范化）。
# 直接用 find_path/find_library 定位，不强依赖 pkg-config；并把 Homebrew 的
# 默认前缀加入搜索路径，方便 macOS 本机构建。
list(APPEND CMAKE_PREFIX_PATH /opt/homebrew /usr/local)

find_path(PCRE2_INCLUDE_DIR NAMES pcre2.h)
find_library(PCRE2_LIBRARY NAMES pcre2-8 libpcre2-8)
find_path(UTF8PROC_INCLUDE_DIR NAMES utf8proc.h)
find_library(UTF8PROC_LIBRARY NAMES utf8proc libutf8proc)

if(NOT PCRE2_INCLUDE_DIR OR NOT PCRE2_LIBRARY)
    message(FATAL_ERROR "未找到 PCRE2（pcre2.h / libpcre2-8）。macOS 可执行：brew install pcre2")
endif()
if(NOT UTF8PROC_INCLUDE_DIR OR NOT UTF8PROC_LIBRARY)
    message(FATAL_ERROR "未找到 utf8proc（utf8proc.h / libutf8proc）。macOS 可执行：brew install utf8proc")
endif()

# PCRE2 是 8-bit code unit 接口，必须在包含 pcre2.h 前定义该宏。
target_compile_definitions(local_llm PRIVATE PCRE2_CODE_UNIT_WIDTH=8)
target_include_directories(local_llm PRIVATE ${PCRE2_INCLUDE_DIR} ${UTF8PROC_INCLUDE_DIR})
target_link_libraries(local_llm PRIVATE ${PCRE2_LIBRARY} ${UTF8PROC_LIBRARY})

# CUDA：project() 已启用 CUDA 语言，.cu kernel 由 nvcc 编译。这里使用 CMake 官方
# CUDAToolkit 包导出 include 目录和 imported targets，CLion 的 code model 识别更稳定。
# 兼容旧参数 -DCUDA_TOOLKIT_ROOT_DIR=/path，同时允许 CMake 原生 -DCUDAToolkit_ROOT=/path。
if(DEFINED CUDA_TOOLKIT_ROOT_DIR AND NOT DEFINED CUDAToolkit_ROOT)
    set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT_DIR}")
endif()
find_package(CUDAToolkit REQUIRED)

target_include_directories(local_llm SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(local_llm PRIVATE CUDA::cudart CUDA::cublas)

# NVML（libnvidia-ml）：DeviceMonitor 用于采样 GPU 利用率 / 带宽 / 功耗 / 温度。
# nvml.h 随 CUDA toolkit 提供；运行库 libnvidia-ml 随驱动安装（stubs 也可链接）。
find_library(NVML_LIBRARY NAMES nvidia-ml
        HINTS ${CUDA_TOOLKIT_ROOT_DIR} ${CUDAToolkit_ROOT}
        PATHS /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.8 /usr/local/cuda-12.4
        PATH_SUFFIXES lib64 lib lib64/stubs lib/stubs targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)
if(NOT NVML_LIBRARY)
    message(FATAL_ERROR "未找到 NVML（libnvidia-ml）。可用 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda 指定，或安装 NVIDIA 驱动")
endif()
target_link_libraries(local_llm PRIVATE ${NVML_LIBRARY})

option(LOCAL_LLM_BUILD_TESTS "Build local-llm tests" ON)

if(LOCAL_LLM_BUILD_TESTS)
    include(CTest)
    include(FetchContent)

    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        FetchContent_Declare(
                googletest
                URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        )
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(local_llm_tests
            test/LaunchAddTest.cpp
            src/backend/cuda/ops/kernel.cu
            src/backend/cuda/ops/kernel.cuh
            src/backend/cuda/ops/kernel_internal.cuh
            src/backend/cuda/ops/launch.cu
            src/utils/stats/Profiler.cpp
            src/utils/stats/Profiler.h
            src/utils/stats/ScopedTimer.cpp
            src/utils/stats/ScopedTimer.h
    )

    target_include_directories(local_llm_tests SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_include_directories(local_llm_tests PRIVATE src)
    target_link_libraries(local_llm_tests PRIVATE GTest::gtest_main CUDA::cudart)

    include(GoogleTest)
    gtest_discover_tests(local_llm_tests)
endif()
