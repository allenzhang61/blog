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

// RAII 的 GPU 段计时：用 CUDA event 夹住一段 kernel，析构时算出 elapsed 并回填
// Profiler。GPU kernel 异步执行，CPU 侧 std::chrono 只能测到 launch 开销，故算子 /
// 层级 GPU 计时必须走 event。
//
// 用法：在某段 kernel 序列的作用域开头放一行
//     ScopedGpuTimer t("mlp.gemm.down", stream, bytes_read);
// 作用域结束即完成计时。bytes 传该段从显存读取的权重字节数，用于算有效带宽；
// 不关心带宽时传 0。
//
// 说明：析构时需要 cudaEventSynchronize(stop) 才能取到耗时，会引入一次同步。
// 因当前每层 forward 内本就有 cudaDeviceSynchronize，profile 模式下这点额外同步
// 可接受；仅在 Profiler::enabled() 时创建 event，关闭时为零开销。
class ScopedGpuTimer {
public:
    ScopedGpuTimer(std::string name, cudaStream_t stream, size_t bytes = 0);
    ~ScopedGpuTimer();

    ScopedGpuTimer(const ScopedGpuTimer &) = delete;
    ScopedGpuTimer &operator=(const ScopedGpuTimer &) = delete;

private:
    std::string name_;
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
    explicit ScopedCpuTimer(std::string name);
    ~ScopedCpuTimer();

    ScopedCpuTimer(const ScopedCpuTimer &) = delete;
    ScopedCpuTimer &operator=(const ScopedCpuTimer &) = delete;

private:
    std::string name_;
    bool active_ = false;
    std::chrono::steady_clock::time_point start_;
};

#endif // LOCAL_LLM_SCOPEDTIMER_H
