#include "llm/device.hpp"
#include "llm/backend/Backend.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

using namespace llm;

namespace {

class ScopedBackendEnv {
public:
    ScopedBackendEnv() {
        const char* value = std::getenv("LLM_CPP_BACKEND");
        if (value != nullptr) {
            had_value_ = true;
            value_ = value;
        }
    }

    ~ScopedBackendEnv() {
        if (had_value_) {
            setenv("LLM_CPP_BACKEND", value_.c_str(), 1);
        } else {
            unsetenv("LLM_CPP_BACKEND");
        }
    }

private:
    bool had_value_{false};
    std::string value_;
};

} // namespace

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
    ScopedBackendEnv scoped_env;
    setenv("LLM_CPP_BACKEND", "cpu", 1);
    EXPECT_EQ(select_device_from_arg_or_env("cpu").type, DeviceType::CPU);
}

TEST(SelectDevice, ArgUnavailableBackendThrows) {
    if (!backend::available(DeviceType::Metal)) {
        EXPECT_THROW(select_device_from_arg_or_env("metal"), std::runtime_error);
    }
    if (!backend::available(DeviceType::CUDA)) {
        EXPECT_THROW(select_device_from_arg_or_env("cuda"), std::runtime_error);
    }
}

TEST(SelectDevice, EmptyArgAndEnvReturnsDefaultCpu) {
    ScopedBackendEnv scoped_env;
    unsetenv("LLM_CPP_BACKEND");
    Device d = select_device_from_arg_or_env("");
    EXPECT_EQ(d.type, DeviceType::CPU);
    EXPECT_EQ(d.index, 0);
}

TEST(SelectDevice, FallsBackToEnv) {
    ScopedBackendEnv scoped_env;
    if (backend::available(DeviceType::Metal)) {
        setenv("LLM_CPP_BACKEND", "metal", 1);
        EXPECT_EQ(select_device_from_arg_or_env("").type, DeviceType::Metal);
        return;
    }
    if (backend::available(DeviceType::CUDA)) {
        setenv("LLM_CPP_BACKEND", "cuda", 1);
        EXPECT_EQ(select_device_from_arg_or_env("").type, DeviceType::CUDA);
        return;
    }
    setenv("LLM_CPP_BACKEND", "metal", 1);
    EXPECT_THROW(select_device_from_arg_or_env(""), std::runtime_error);
}

TEST(SelectDevice, EmptyEnvValueReturnsDefaultCpu) {
    ScopedBackendEnv scoped_env;
    setenv("LLM_CPP_BACKEND", "", 1);
    EXPECT_EQ(select_device_from_arg_or_env("").type, DeviceType::CPU);
}
