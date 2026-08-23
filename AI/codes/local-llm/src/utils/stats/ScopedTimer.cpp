//
// Created by zhangyoulun on 9/8/2026.
//

#include "utils/stats/ScopedTimer.h"

ScopedGpuTimer::ScopedGpuTimer(const char *name, cudaStream_t stream, size_t bytes)
    : name_(name), stream_(stream), bytes_(bytes) {
    if (name_ == nullptr || !Profiler::instance().capturing()) {
        return; // 关闭 / 非采样步 / 匿名段时不借 event，零开销。
    }
    // 向 Profiler 借一对可复用 event（池化，避免反复 create/destroy）。
    start_ = Profiler::instance().acquire_event();
    stop_ = Profiler::instance().acquire_event();
    if (start_ == nullptr || stop_ == nullptr) {
        // 借用失败则退化为不计时，避免影响主流程。
        return;
    }
    active_ = true;
    cudaEventRecord(start_, stream_);
}

ScopedGpuTimer::~ScopedGpuTimer() {
    if (!active_) {
        return;
    }
    // 只 record stop，不同步、不销毁：把这对 event 交给 Profiler 延迟批量结算，
    // 避免逐算子 cudaEventSynchronize 打断流水线。name_ 是字面量，仅传指针。
    cudaEventRecord(stop_, stream_);
    Profiler::instance().push_gpu_event(name_, bytes_, start_, stop_);
}

ScopedCpuTimer::ScopedCpuTimer(const char *name) : name_(name) {
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
