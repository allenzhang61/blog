cmake_minimum_required(VERSION 3.18)

project(llm_inference_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(llm-inference-cpp
    src/main.cpp
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(llm-inference-cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(llm-inference-cpp PRIVATE /W4)
else()
    target_compile_options(llm-inference-cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()
