cmake_minimum_required(VERSION 4.2)
project(local_llm)

set(CMAKE_CXX_STANDARD 20)

add_executable(local_llm main.cpp
        src/utils/cli/Args.cpp
        src/utils/cli/Args.h
        src/backend/common.h
        src/utils/log/Log.cpp
        src/utils/log/Log.h
        src/llm/qwen/QwenLLM.cpp
        src/llm/qwen/QwenLLM.h
        src/llm/deepseek/DeepseekLLM.cpp
        src/llm/deepseek/DeepseekLLM.h
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


