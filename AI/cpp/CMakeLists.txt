cmake_minimum_required(VERSION 3.14)
project(cpp)

set(CMAKE_CXX_STANDARD 20)

# ---- 业务目标 ------------------------------------------------
add_executable(cpp
    main.cpp
    tensor.cpp
    function.cpp
)

# ---- Google Test（自动下载）---------------------------------
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(tensor_test
    test.cpp
    tensor.cpp
    function.cpp
)
target_link_libraries(tensor_test GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(tensor_test)
