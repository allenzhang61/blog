//
// Created by zhangyoulun on 9/8/2026.
//

#include "utils/stats/DeviceMonitor.h"

#include <algorithm>
#include <chrono>

#include <nvml.h>

DeviceMonitor::DeviceMonitor(int device_index) : device_index_(device_index) {}

DeviceMonitor::~DeviceMonitor() {
    stop();
}

void DeviceMonitor::start(int interval_ms) {
    if (running_.load()) {
        return;
    }
    samples_.clear();
    running_.store(true);
    worker_ = std::thread(&DeviceMonitor::run, this, interval_ms);
}

void DeviceMonitor::stop() {
    if (!running_.load()) {
        return;
    }
    running_.store(false);
    if (worker_.joinable()) {
        worker_.join();
    }
}

void DeviceMonitor::run(int interval_ms) {
    // NVML 采样失败不抛异常（监控是旁路），静默退出即可。
    if (nvmlInit_v2() != NVML_SUCCESS) {
        return;
    }
    nvmlDevice_t dev;
    if (nvmlDeviceGetHandleByIndex_v2(static_cast<unsigned int>(device_index_), &dev) !=
        NVML_SUCCESS) {
        nvmlShutdown();
        return;
    }

    while (running_.load()) {
        Sample s;
        s.ts = std::chrono::steady_clock::now();

        nvmlUtilization_t util;
        if (nvmlDeviceGetUtilizationRates(dev, &util) == NVML_SUCCESS) {
            s.sm_util_percent = util.gpu;   // SM（计算）利用率
            s.mem_util_percent = util.memory; // 显存控制器（带宽）利用率
        }
        unsigned int power = 0;
        if (nvmlDeviceGetPowerUsage(dev, &power) == NVML_SUCCESS) {
            s.power_milliwatt = power;
        }
        unsigned int temp = 0;
        if (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
            s.temperature_celsius = temp;
        }
        nvmlMemory_t mem;
        if (nvmlDeviceGetMemoryInfo(dev, &mem) == NVML_SUCCESS) {
            s.mem_used_bytes = mem.used;
        }
        samples_.push_back(s);

        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }

    nvmlShutdown();
}

void DeviceMonitor::write_jsonl(std::ostream &os) const {
    if (samples_.empty()) {
        return;
    }
    const auto t0 = samples_.front().ts;
    for (const Sample &s : samples_) {
        const double ts_ms = std::chrono::duration<double, std::milli>(s.ts - t0).count();
        os << "{\"kind\":\"device\""
           << ",\"ts_ms\":" << ts_ms
           << ",\"sm_util_percent\":" << s.sm_util_percent
           << ",\"mem_util_percent\":" << s.mem_util_percent
           << ",\"power_w\":" << static_cast<double>(s.power_milliwatt) / 1000.0
           << ",\"temperature_c\":" << s.temperature_celsius
           << ",\"mem_used_bytes\":" << s.mem_used_bytes
           << "}\n";
    }
}

namespace {
struct Agg {
    double sm_sum = 0, mem_sum = 0, power_sum = 0;
    unsigned int sm_max = 0, mem_max = 0, power_max = 0, temp_max = 0;
    size_t mem_used_max = 0;
    size_t n = 0;
};

Agg aggregate(const std::vector<DeviceMonitor::Sample> &samples) {
    Agg a;
    for (const auto &s : samples) {
        a.sm_sum += s.sm_util_percent;
        a.mem_sum += s.mem_util_percent;
        a.power_sum += s.power_milliwatt;
        a.sm_max = std::max(a.sm_max, s.sm_util_percent);
        a.mem_max = std::max(a.mem_max, s.mem_util_percent);
        a.power_max = std::max(a.power_max, s.power_milliwatt);
        a.temp_max = std::max(a.temp_max, s.temperature_celsius);
        a.mem_used_max = std::max(a.mem_used_max, s.mem_used_bytes);
    }
    a.n = samples.size();
    return a;
}
} // namespace

void DeviceMonitor::write_json_summary(std::ostream &os) const {
    const Agg a = aggregate(samples_);
    const double n = a.n > 0 ? static_cast<double>(a.n) : 1.0;
    os << "{\"kind\":\"device_summary\""
       << ",\"samples\":" << a.n
       << ",\"sm_util_avg\":" << a.sm_sum / n
       << ",\"sm_util_max\":" << a.sm_max
       << ",\"mem_util_avg\":" << a.mem_sum / n
       << ",\"mem_util_max\":" << a.mem_max
       << ",\"power_w_avg\":" << a.power_sum / n / 1000.0
       << ",\"power_w_max\":" << static_cast<double>(a.power_max) / 1000.0
       << ",\"temp_c_max\":" << a.temp_max
       << ",\"mem_used_max_bytes\":" << a.mem_used_max
       << "}\n";
}

void DeviceMonitor::write_markdown_summary(std::ostream &os) const {
    const Agg a = aggregate(samples_);
    const double n = a.n > 0 ? static_cast<double>(a.n) : 1.0;
    os << "## Device summary (" << a.n << " samples)\n\n";
    os << "| metric | avg | max |\n";
    os << "|---|---:|---:|\n";
    os << "| SM util (%) | " << a.sm_sum / n << " | " << a.sm_max << " |\n";
    os << "| mem-bw util (%) | " << a.mem_sum / n << " | " << a.mem_max << " |\n";
    os << "| power (W) | " << a.power_sum / n / 1000.0 << " | "
       << static_cast<double>(a.power_max) / 1000.0 << " |\n";
    os << "| temp (C) | - | " << a.temp_max << " |\n";
    os << "| mem used (MiB) | - | " << static_cast<double>(a.mem_used_max) / (1024.0 * 1024.0)
       << " |\n";
    os << "\n";
}
