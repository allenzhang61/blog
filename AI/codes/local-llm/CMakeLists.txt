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
        src/llm/qwen/QwenConfig.cpp
        src/llm/qwen/QwenConfig.h
        src/llm/qwen/QwenWeights.cpp
        src/llm/qwen/QwenWeights.h
        src/llm/qwen/QwenTokenizer.cpp
        src/llm/qwen/QwenTokenizer.h
        src/backend/cuda/common.cpp
        src/backend/cuda/common.h
        src/backend/cuda/mem/CudaWeightPool.cpp
        src/backend/cuda/mem/CudaWeightPool.h
        src/backend/cuda/mem/CudaWeight.cpp
        src/backend/cuda/mem/CudaWeight.h
        src/backend/cuda/mem/CudaScratchBuffer.cpp
        src/backend/cuda/mem/CudaScratchBuffer.h
        src/llm/qwen/QwenSession.cpp
        src/llm/qwen/QwenSession.h
        src/llm/qwen/QwenForwardScratch.cpp
        src/llm/qwen/QwenForwardScratch.h
        src/backend/cuda/ops/kernel.cu
        src/backend/cuda/ops/kernel.cuh
        src/backend/cuda/ops/gemm.cpp
        src/backend/cuda/ops/gemm.h
        src/format/gguf/GgufFile.cpp
        src/format/gguf/GgufFile.h
        src/format/gguf/Q4K.cpp
        src/format/gguf/Q4K.h
        src/format/TensorContainer.h
        src/format/safetensors/SafeTensorsFile.cpp
        src/format/safetensors/SafeTensorsFile.h
        src/llm/BaseModel.h
        src/llm/ModelFactory.cpp
        src/llm/ModelFactory.h
        src/llm/qwen/model/Module.h
        src/llm/qwen/model/RMSNorm.cpp
        src/llm/qwen/model/RMSNorm.h
        src/llm/qwen/model/Embedding.cpp
        src/llm/qwen/model/Embedding.h
        src/llm/qwen/model/SwiGLUMlp.cpp
        src/llm/qwen/model/SwiGLUMlp.h
        src/llm/qwen/model/LMHead.cpp
        src/llm/qwen/model/LMHead.h
        src/llm/qwen/model/FullAttention.cpp
        src/llm/qwen/model/FullAttention.h
        src/llm/qwen/model/LinearAttention.cpp
        src/llm/qwen/model/LinearAttention.h
        src/llm/qwen/model/DecoderLayer.cpp
        src/llm/qwen/model/DecoderLayer.h
        src/llm/qwen/model/QwenModel.cpp
        src/llm/qwen/model/QwenModel.h
        src/llm/deepseek/DeepseekConfig.cpp
        src/llm/deepseek/DeepseekConfig.h
        src/llm/deepseek/DeepseekWeights.cpp
        src/llm/deepseek/DeepseekWeights.h
        src/llm/deepseek/DeepseekTokenizer.cpp
        src/llm/deepseek/DeepseekTokenizer.h
        src/llm/deepseek/model/DeepseekForwardScratch.h
        src/llm/deepseek/model/DeepseekSession.cpp
        src/llm/deepseek/model/DeepseekSession.h
        src/llm/deepseek/model/DeepseekModel.cpp
        src/llm/deepseek/model/DeepseekModel.h
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

# CUDA：project() 已启用 CUDA 语言，.cu kernel 由 nvcc 编译。
# 这里再补上 cuBLAS 的头文件与链接库路径（cudart 由 CUDA 工具链自动链接，
# 但显式链接更稳妥）。允许通过 -DCUDA_TOOLKIT_ROOT_DIR=/path 覆盖探测位置。
find_path(CUDA_INCLUDE_DIR NAMES cublas_v2.h cuda_runtime.h
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}
        PATHS /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.8 /usr/local/cuda-12.4
        PATH_SUFFIXES include targets/x86_64-linux/include)
find_library(CUDART_LIBRARY NAMES cudart
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}
        PATHS /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.8 /usr/local/cuda-12.4
        PATH_SUFFIXES lib64 lib targets/x86_64-linux/lib)
find_library(CUBLAS_LIBRARY NAMES cublas
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}
        PATHS /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.8 /usr/local/cuda-12.4
        PATH_SUFFIXES lib64 lib targets/x86_64-linux/lib)

if(NOT CUDA_INCLUDE_DIR OR NOT CUDART_LIBRARY OR NOT CUBLAS_LIBRARY)
    message(FATAL_ERROR "未找到 CUDA toolkit（cublas_v2.h / libcudart / libcublas）。可用 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda 指定")
endif()

target_include_directories(local_llm PRIVATE ${CUDA_INCLUDE_DIR})
target_link_libraries(local_llm PRIVATE ${CUDART_LIBRARY} ${CUBLAS_LIBRARY})

# NVML（libnvidia-ml）：DeviceMonitor 用于采样 GPU 利用率 / 带宽 / 功耗 / 温度。
# nvml.h 随 CUDA toolkit 提供；运行库 libnvidia-ml 随驱动安装（stubs 也可链接）。
find_library(NVML_LIBRARY NAMES nvidia-ml
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}
        PATHS /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.8 /usr/local/cuda-12.4
        PATH_SUFFIXES lib64 lib lib64/stubs lib/stubs targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)
if(NOT NVML_LIBRARY)
    message(FATAL_ERROR "未找到 NVML（libnvidia-ml）。可用 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda 指定，或安装 NVIDIA 驱动")
endif()
target_link_libraries(local_llm PRIVATE ${NVML_LIBRARY})


