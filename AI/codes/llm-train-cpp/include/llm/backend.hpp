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

std::string cuda_backend_status();
std::string metal_backend_status();

} // namespace llm
