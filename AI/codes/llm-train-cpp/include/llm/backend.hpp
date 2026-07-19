#pragma once

#include "llm/core.hpp"

namespace llm {

class Backend {
public:
    virtual ~Backend();
    virtual DeviceType type() const = 0;
    virtual std::string name() const = 0;
};

class CPUBackend : public Backend {
public:
    DeviceType type() const override;
    std::string name() const override;
};

class UnimplementedBackend : public Backend {
public:
    explicit UnimplementedBackend(DeviceType type);
    DeviceType type() const override;
    std::string name() const override;

private:
    DeviceType type_;
};

class BackendRegistry {
public:
    static Backend& get(Device device);
};

Device select_device(const std::string& backend);
Device select_device_from_arg_or_env(const std::string& arg = "", const char* env_name = "LLM_CPP_BACKEND");
bool cuda_backend_available();
bool metal_backend_available();
std::string cuda_backend_status();
std::string metal_backend_status();

} // namespace llm
