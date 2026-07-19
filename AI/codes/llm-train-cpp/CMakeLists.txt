cmake_minimum_required(VERSION 3.16)
project(llm_train_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(llm_train_cpp STATIC
    src/core/core.cpp
    src/tensor/tensor.cpp
    src/ops/ops.cpp
    src/backend/BackendRegistry.cpp
    src/backend/CPUBackend.cpp
    src/backend/CUDABackend.cpp
    src/backend/MetalBackend.cpp
    src/kernels/cpu/elementwise.cpp
    src/kernels/cpu/matmul.cpp
    src/kernels/cpu/softmax.cpp
    src/kernels/cpu/layernorm.cpp
    src/kernels/cpu/gelu.cpp
    src/kernels/cpu/embedding.cpp
    src/model/model.cpp
    src/data/data.cpp
    src/train/train.cpp
)
target_include_directories(llm_train_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(llm_cpp_tests tests/test_main.cpp)
target_link_libraries(llm_cpp_tests PRIVATE llm_train_cpp)
target_compile_definitions(llm_cpp_tests PRIVATE LLM_CPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(train_gpt examples/train_gpt.cpp)
target_link_libraries(train_gpt PRIVATE llm_train_cpp)
target_compile_definitions(train_gpt PRIVATE LLM_CPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(generate_text examples/generate_text.cpp)
target_link_libraries(generate_text PRIVATE llm_train_cpp)
target_compile_definitions(generate_text PRIVATE LLM_CPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

enable_testing()
add_test(NAME llm_cpp_tests COMMAND llm_cpp_tests)
