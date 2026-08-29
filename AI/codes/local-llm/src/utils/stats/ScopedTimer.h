//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_SCOPEDTIMER_H
#define LOCAL_LLM_SCOPEDTIMER_H

#include <chrono>
#include <cstddef>
#include <string>

#include <cuda_runtime.h>

#include "utils/stats/Profiler.h"

// RAII 的 GPU 段计时：用 CUDA event 夹住一段 kernel，回填 Profiler。
// GPU kernel 异步执行，CPU 侧 std::chrono 只能测到 launch 开销，故算子 /
// 层级 GPU 计时必须走 event。
//
// 用法：在某段 kernel 序列的作用域开头放一行
//     ScopedGpuTimer t("mlp.gemm.down", bytes_read);
// 作用域结束即完成计时。bytes 传该段从显存读取的权重字节数，用于算有效带宽；
// 不关心带宽时传 0。
//
// 低扰动实现：构造时向 Profiler 借一对可复用 event 并 record(start)；析构时只
// record(stop) 并把这对 event 连同 name/bytes 交给 Profiler 排队（push_gpu_event），
// **不做 cudaEventSynchronize、不销毁 event**，因此不会打断 CUDA 流水线——避免
// “发一个 kernel、等一个 kernel”的逐算子同步开销。真正取耗时延迟到
// Profiler::flush_gpu_events()（调用方须先保证 event 已完成，如 cudaDeviceSynchronize）。
// 仅在 Profiler::enabled() 时借用 event，关闭时为零开销。
class ScopedGpuTimer {
public:
    // name 必须是静态存储期字符串（字面量）：热路径只存指针，不拷贝，避免每算子
    // 一次 std::string 分配。实测该分配是 profile 开销的主因（远大于 event 本身）。
    ScopedGpuTimer(const char *name, size_t bytes = 0);
    ~ScopedGpuTimer();

    ScopedGpuTimer(const ScopedGpuTimer &) = delete;
    ScopedGpuTimer &operator=(const ScopedGpuTimer &) = delete;

private:
    const char *name_ = nullptr;
    cudaStream_t stream_ = nullptr;
    size_t bytes_ = 0;
    bool active_ = false; // Profiler 未开启时不创建 event
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

// RAII 的 CPU 段计时：用于 tokenizer encode/decode、权重加载等纯 CPU 阶段。
// 与 GPU 版共用同一个 Profiler 聚合出口。
class ScopedCpuTimer {
public:
    // 同样要求 name 为静态存储期字符串（字面量）。
    explicit ScopedCpuTimer(const char *name);
    ~ScopedCpuTimer();

    ScopedCpuTimer(const ScopedCpuTimer &) = delete;
    ScopedCpuTimer &operator=(const ScopedCpuTimer &) = delete;

private:
    const char *name_ = nullptr;
    bool active_ = false;
    std::chrono::steady_clock::time_point start_;
};

#endif // LOCAL_LLM_SCOPEDTIMER_H
