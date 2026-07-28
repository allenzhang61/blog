cmake_minimum_required(VERSION 3.18)
project(llm_train_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LLM_CPP_ENABLE_METAL "Build the Metal backend on Apple platforms" OFF)
option(LLM_CPP_ENABLE_CUDA "Build the CUDA backend when CUDA Toolkit is available" OFF)

set(LLM_CPP_SOURCES
    src/utils/device.cpp
    src/utils/utils.cpp
    src/tensor/tensor.cpp
    src/ops/ops.cpp
    src/backend/Backend.cpp
    src/backend/BackendRegistry.cpp
    src/backend/cpu/CPUBackend.cpp
    src/backend/metal/MetalDeviceBackend.cpp
    src/backend/cuda/CUDADeviceBackend.cpp
    src/kernels/cpu/elementwise.cpp
    src/kernels/cpu/matmul.cpp
    src/kernels/cpu/softmax.cpp
    src/kernels/cpu/layernorm.cpp
    src/kernels/cpu/gelu.cpp
    src/kernels/cpu/embedding.cpp
    src/model/Module.cpp
    src/model/Linear.cpp
    src/model/Embedding.cpp
    src/model/LayerNorm.cpp
    src/model/GELU.cpp
    src/model/MultiHeadAttention.cpp
    src/model/FeedForward.cpp
    src/model/TransformerBlock.cpp
    src/model/GPTModel.cpp
    src/data/GPT2BPETokenizer.cpp
    src/data/DataLoader.cpp
    src/train/AdamW.cpp
    src/train/Trainer.cpp
)

set(LLM_CPP_ENABLE_METAL_COMPILED 0)
if(LLM_CPP_ENABLE_METAL AND APPLE)
    enable_language(OBJCXX)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    list(APPEND LLM_CPP_SOURCES src/backend/metal/MetalBackend.mm)
    list(APPEND LLM_CPP_SOURCES src/kernels/metal/MetalKernels.mm)
    set(LLM_CPP_ENABLE_METAL_COMPILED 1)
else()
    list(APPEND LLM_CPP_SOURCES src/backend/metal/MetalBackend.cpp)
endif()

set(LLM_CPP_ENABLE_CUDA_COMPILED 0)
if(LLM_CPP_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        # CUDA 开启：真实 kernel + 委派实现（对应 Metal 的 MetalKernels.mm + MetalBackend.mm）。
        list(APPEND LLM_CPP_SOURCES src/kernels/cuda/cuda_ops.cu)
        list(APPEND LLM_CPP_SOURCES src/kernels/cuda/cuda_device_kernels.cu)
        list(APPEND LLM_CPP_SOURCES src/kernels/cuda/cuda_runtime.cu)
        list(APPEND LLM_CPP_SOURCES src/backend/cuda/CUDABackend.cpp)
        set(LLM_CPP_ENABLE_CUDA_COMPILED 1)
    endif()
endif()
if(NOT LLM_CPP_ENABLE_CUDA_COMPILED)
    # CUDA 关闭：抛错 stub（对应 Metal 的 MetalBackend.cpp）。
    list(APPEND LLM_CPP_SOURCES src/backend/cuda/CUDAStub.cpp)
endif()

add_library(llm_train_cpp STATIC
    ${LLM_CPP_SOURCES}
)
target_include_directories(llm_train_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(llm_train_cpp PUBLIC
    LLM_CPP_ENABLE_METAL_COMPILED=${LLM_CPP_ENABLE_METAL_COMPILED}
    LLM_CPP_ENABLE_CUDA_COMPILED=${LLM_CPP_ENABLE_CUDA_COMPILED}
)
target_compile_options(llm_train_cpp PUBLIC
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/utf-8>
)
if(APPLE)
    # macOS 自带 Accelerate 框架（含 BLAS）。用 cblas_dgemm 替换手写 matmul，
    # 可获得数量级的矩阵乘加速（多线程 + 向量化 + 缓存分块）。
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(llm_train_cpp PUBLIC ${ACCELERATE_FRAMEWORK})
    target_compile_definitions(llm_train_cpp PUBLIC LLM_CPP_USE_BLAS=1)
endif()
if(LLM_CPP_ENABLE_CUDA_COMPILED)
    target_compile_options(llm_train_cpp PRIVATE
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=/utf-8>
    )
endif()
if(LLM_CPP_ENABLE_METAL_COMPILED)
    # 构建期把 .metal 源码编译成 .metallib（业界主流做法），运行期直接加载二进制。
    set(LLM_CPP_METAL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/metal/metal_kernels.metal)
    set(LLM_CPP_METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/metal_kernels.air)
    set(LLM_CPP_METALLIB ${CMAKE_CURRENT_BINARY_DIR}/metal_kernels.metallib)
    add_custom_command(
        OUTPUT ${LLM_CPP_METALLIB}
        COMMAND xcrun -sdk macosx metal -c ${LLM_CPP_METAL_SOURCE} -o ${LLM_CPP_METAL_AIR}
        COMMAND xcrun -sdk macosx metallib ${LLM_CPP_METAL_AIR} -o ${LLM_CPP_METALLIB}
        DEPENDS ${LLM_CPP_METAL_SOURCE}
        COMMENT "Compiling Metal kernels to metallib"
        VERBATIM
    )
    add_custom_target(llm_cpp_metallib DEPENDS ${LLM_CPP_METALLIB})
    add_dependencies(llm_train_cpp llm_cpp_metallib)

    target_compile_definitions(llm_train_cpp PRIVATE
        LLM_CPP_METALLIB_PATH="${LLM_CPP_METALLIB}"
    )
    target_link_libraries(llm_train_cpp PUBLIC ${FOUNDATION_FRAMEWORK} ${METAL_FRAMEWORK})
endif()

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(llm_cpp_tests tests/test_cpu.cpp tests/utils/test_utils.cpp tests/utils/test_device.cpp tests/tensor/test_tensor.cpp)
target_link_libraries(llm_cpp_tests PRIVATE llm_train_cpp GTest::gtest_main)
target_compile_definitions(llm_cpp_tests PRIVATE LLM_CPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(train_gpt examples/train_gpt.cpp)
target_link_libraries(train_gpt PRIVATE llm_train_cpp)
target_compile_definitions(train_gpt PRIVATE LLM_CPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(generate_text examples/generate_text.cpp)
target_link_libraries(generate_text PRIVATE llm_train_cpp)
target_compile_definitions(generate_text PRIVATE LLM_CPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(backend_benchmark examples/backend_benchmark.cpp)
target_link_libraries(backend_benchmark PRIVATE llm_train_cpp)

add_executable(train_benchmark examples/train_benchmark.cpp)
target_link_libraries(train_benchmark PRIVATE llm_train_cpp)

enable_testing()
include(GoogleTest)
gtest_discover_tests(llm_cpp_tests)

if(LLM_CPP_ENABLE_METAL_COMPILED)
    add_executable(llm_cpp_metal_smoke tests/test_metal.cpp)
    target_link_libraries(llm_cpp_metal_smoke PRIVATE llm_train_cpp GTest::gtest_main)
    gtest_discover_tests(llm_cpp_metal_smoke)

    add_executable(metal_benchmark examples/metal_benchmark.cpp)
    target_link_libraries(metal_benchmark PRIVATE llm_train_cpp)
endif()

if(LLM_CPP_ENABLE_CUDA_COMPILED)
    add_executable(llm_cpp_cuda_smoke tests/test_cuda.cpp)
    target_link_libraries(llm_cpp_cuda_smoke PRIVATE llm_train_cpp GTest::gtest_main)
    gtest_discover_tests(llm_cpp_cuda_smoke)
endif()
