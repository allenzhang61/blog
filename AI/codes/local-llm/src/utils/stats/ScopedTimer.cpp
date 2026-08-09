//
// Created by zhangyoulun on 9/8/2026.
//

#include "utils/stats/ScopedTimer.h"

#include <utility>

ScopedGpuTimer::ScopedGpuTimer(std::string name, cudaStream_t stream, size_t bytes)
    : name_(std::move(name)), stream_(stream), bytes_(bytes) {
    if (!Profiler::instance().enabled()) {
        return; // 关闭时不创建 event，零开销。
    }
    // 创建失败则退化为不计时，避免影响主流程。
    if (cudaEventCreate(&start_) != cudaSuccess || cudaEventCreate(&stop_) != cudaSuccess) {
        if (start_) { cudaEventDestroy(start_); start_ = nullptr; }
        if (stop_) { cudaEventDestroy(stop_); stop_ = nullptr; }
        return;
    }
    active_ = true;
    cudaEventRecord(start_, stream_);
}

ScopedGpuTimer::~ScopedGpuTimer() {
    if (!active_) {
        return;
    }
    cudaEventRecord(stop_, stream_);
    // 取耗时需要 stop event 完成；profile 模式下这次同步可接受。
    cudaEventSynchronize(stop_);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_, stop_);
    Profiler::instance().record(name_, static_cast<double>(ms), bytes_);
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

ScopedCpuTimer::ScopedCpuTimer(std::string name) : name_(std::move(name)) {
    if (!Profiler::instance().enabled()) {
        return;
    }
    active_ = true;
    start_ = std::chrono::steady_clock::now();
}

ScopedCpuTimer::~ScopedCpuTimer() {
    if (!active_) {
        return;
    }
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
    Profiler::instance().record(name_, ms, 0);
}
