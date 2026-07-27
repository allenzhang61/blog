#include "llm/device.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

using namespace llm;

TEST(DeviceToString, DeviceType) {
    EXPECT_EQ(to_string(DeviceType::CPU), "cpu");
    EXPECT_EQ(to_string(DeviceType::CUDA), "cuda");
    EXPECT_EQ(to_string(DeviceType::Metal), "metal");
}

TEST(DeviceToString, DType) {
    EXPECT_EQ(to_string(DType::Float32), "float32");
    EXPECT_EQ(to_string(DType::Int64), "int64");
}

TEST(DeviceToString, UnknownDeviceType) {
    // 用非法枚举值触发 switch 之外的兜底分支。
    EXPECT_EQ(to_string(static_cast<DeviceType>(999)), "unknown");
}

TEST(DeviceToString, UnknownDType) {
    // 用非法枚举值触发 switch 之外的兜底分支。
    EXPECT_EQ(to_string(static_cast<DType>(999)), "unknown");
}

TEST(DeviceParse, KnownDevices) {
    EXPECT_EQ(Device::parse("cpu").type, DeviceType::CPU);
    EXPECT_EQ(Device::parse("cuda").type, DeviceType::CUDA);
    EXPECT_EQ(Device::parse("cuda:0").type, DeviceType::CUDA);
    EXPECT_EQ(Device::parse("metal").type, DeviceType::Metal);
    EXPECT_EQ(Device::parse("metal:0").type, DeviceType::Metal);
}

TEST(DeviceParse, IndexIsZero) {
    EXPECT_EQ(Device::parse("cuda:0").index, 0);
    EXPECT_EQ(Device::parse("metal:0").index, 0);
}

TEST(DeviceParse, UnknownThrows) {
    EXPECT_THROW(Device::parse("gpu"), std::runtime_error);
    EXPECT_THROW(Device::parse("cuda:1"), std::runtime_error);
    EXPECT_THROW(Device::parse(""), std::runtime_error);
}

TEST(DeviceStr, FormatsTypeAndIndex) {
    Device cpu{DeviceType::CPU, 0};
    Device cuda{DeviceType::CUDA, 0};
    Device metal{DeviceType::Metal, 1};
    EXPECT_EQ(cpu.str(), "cpu:0");
    EXPECT_EQ(cuda.str(), "cuda:0");
    EXPECT_EQ(metal.str(), "metal:1");
}

TEST(SelectDevice, ArgTakesPriority) {
    EXPECT_EQ(select_device_from_arg_or_env("metal").type, DeviceType::Metal);
    EXPECT_EQ(select_device_from_arg_or_env("cuda").type, DeviceType::CUDA);
}

TEST(SelectDevice, EmptyArgAndEnvReturnsDefaultCpu) {
    const char* env_name = "LLM_CPP_TEST_BACKEND";
    unsetenv(env_name);
    Device d = select_device_from_arg_or_env("", env_name);
    EXPECT_EQ(d.type, DeviceType::CPU);
    EXPECT_EQ(d.index, 0);
}

TEST(SelectDevice, FallsBackToEnv) {
    const char* env_name = "LLM_CPP_TEST_BACKEND";
    setenv(env_name, "metal", 1);
    EXPECT_EQ(select_device_from_arg_or_env("", env_name).type, DeviceType::Metal);
    unsetenv(env_name);
}

TEST(SelectDevice, EmptyEnvValueReturnsDefaultCpu) {
    const char* env_name = "LLM_CPP_TEST_BACKEND";
    setenv(env_name, "", 1);
    EXPECT_EQ(select_device_from_arg_or_env("", env_name).type, DeviceType::CPU);
    unsetenv(env_name);
}
