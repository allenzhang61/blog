//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_PROFILER_H
#define LOCAL_LLM_PROFILER_H

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "utils/stats/StatsReport.h"

// 轻量的性能聚合器：进程内单例，按埋点名称收集耗时与访存字节，最后聚合输出。
// 供各 Module 的 ScopedGpuTimer / ScopedCpuTimer 在析构时回填。
//
// 设计取舍（见 doc/2.md：decode 为访存瓶颈）：
//   - 除了耗时，还记录每段读取的字节数，聚合时算出“有效访存带宽 = bytes / ms”，
//     这是判断“是否已接近显存带宽上限、还有没有优化空间”的关键指标。
//   - 除了按名称聚合，还按调用顺序追加原始时间线（每条含时间戳），可还原各段
//     的先后顺序、观察阶段间 gap，供导出 timeline 做进一步分析。
//   - 未 enable 时所有埋点应退化为零开销（配合 ScopedGpuTimer 内部短路），
//     从而可常驻代码、只在 --profile 时打开，不污染生产路径。
//   - 非线程安全：当前推理为单请求串行；如需并发采集再另加锁。
//
// GPU 计时的低扰动结算（关键）：
//   ScopedGpuTimer 析构时只做 cudaEventRecord(stop) 并把 (name/bytes/start/stop)
//   一对 event 交给这里排队（push_gpu_event），既不 cudaEventSynchronize、也不销毁
//   event，从而不打断 CUDA 流水线——避免“发一个 kernel 等一个”的逐算子同步开销。
//   真正的耗时结算延迟到 flush_gpu_events()：调用前须保证这些 event 已完成
//   （main.cpp 在 decode 循环后有一次 cudaDeviceSynchronize），再统一
//   cudaEventElapsedTime 取值、record 聚合，并把 event 归还内部池复用。
//   代价：相邻算子在流水线里可能部分重叠，各算子 event 区间可能交叠，逐项耗时
//   的“绝对值可比、但边界归因略模糊”，占比之和可能不严格等于墙钟。
class Profiler : public StatsReport {
public:
    // 进程内唯一实例。
    static Profiler &instance();

    // 打开 / 关闭采集；关闭时 record 直接返回，埋点零开销。
    void enable(bool on) { enabled_ = on; }
    bool enabled() const { return enabled_; }

    // 采样式 profile 的逐步开关：即便 enabled_，也仅在 capture_ 为 true 的步真正
    // 借 event / 插 event。非采样步 acquire_event 返回 nullptr，timer 短路为零 event
    // 开销（不 cudaEventRecord），从而把每步上百次 event 插流的开销降到接近 base。
    // 由 main.cpp 每步按采样间隔设置。默认 true（不采样时等价全量）。
    void set_capture(bool on) { capture_ = on; }
    bool capturing() const { return enabled_ && capture_; }

    // 记录一段耗时（毫秒）与该段读取的字节数（0 表示不统计带宽）。
    // 同名多次调用会累加到聚合统计，同时按调用顺序追加一条原始时间线记录。
    // name 为静态存储期字符串（字面量）；仅此处按名聚合时才转成 std::string 做 key，
    // 热路径（timer 构造/析构）不产生 std::string。
    void record(const char *name, double ms, size_t bytes = 0);

    // 低扰动 GPU 计时：ScopedGpuTimer 析构时把一对已 record 的 CUDA event
    // （start/stop）连同 name/bytes 交给这里排队，不做同步、不销毁 event。
    // 结算延迟到 flush_gpu_events()。enabled_ 为 false 时应无调用（timer 会短路）。
    // name 为字面量指针，仅入队存指针，不拷贝字符串。
    void push_gpu_event(const char *name, size_t bytes,
                        cudaEvent_t start, cudaEvent_t stop);

    // 结算所有排队中的 GPU event：调用方须先保证这些 event 已完成
    // （如 cudaDeviceSynchronize）。逐对 cudaEventElapsedTime 取耗时后 record 聚合，
    // 并把 event 归还内部池复用，清空待结算队列。可多次调用（增量结算）。
    void flush_gpu_events();

    // 借一个可复用的 CUDA event（优先从池取，池空则新建）。供 ScopedGpuTimer 使用，
    // 避免每次埋点 create/destroy 上万个 event 句柄。enabled_ 为 false 时返回 nullptr。
    cudaEvent_t acquire_event();

    // 清空已采集数据（聚合统计 + 原始时间线）。
    // 用于 warmup 之后、正式计时之前归零。
    // 注意：会先结算并回收 warmup 阶段残留的待结算 GPU event，避免混入稳态或泄漏。
    void reset();

    // === StatsReport：三类报告 ===
    // JSONL 原始日志：每行一条 record 的原始时间线记录（ts_ms / name / ms / bytes）。
    void write_jsonl(std::ostream &os) const override;
    // JSON summary：按名称聚合的 count / total_ms / avg_ms / 占比 / 有效带宽(GB/s)。
    void write_json_summary(std::ostream &os) const override;
    // Markdown summary：同源聚合结论渲染成表格。
    void write_markdown_summary(std::ostream &os) const override;

private:
    Profiler() = default;

    // 单个埋点名称的累计统计。
    struct Stat {
        uint64_t count = 0;    // 调用次数
        double total_ms = 0.0; // 累计耗时（毫秒）
        uint64_t total_bytes = 0; // 累计读取字节数（用于算有效带宽）
    };

    // 单条原始时间线记录：一次 record 调用的现场快照。
    struct Event {
        std::string name;     // 埋点名称
        double ms = 0.0;      // 本次耗时（毫秒）
        size_t bytes = 0;     // 本次读取字节数
        // record 被调用时的时间戳（用于还原调用顺序、计算相对偏移与 gap）。
        std::chrono::steady_clock::time_point ts;
    };

    // 一对待结算的 GPU event：flush 时用 cudaEventElapsedTime(start,stop) 取耗时。
    // name 为字面量指针，不拥有字符串。
    struct PendingGpuEvent {
        const char *name = nullptr;
        size_t bytes = 0;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
    };

    bool enabled_ = false;
    // 采样式 profile 的逐步开关（见 set_capture）。默认 true。
    bool capture_ = true;
    // 按名称聚合的累计统计。
    std::unordered_map<std::string, Stat> stats_;
    // 按调用顺序追加的原始时间线。
    std::vector<Event> events_;
    // 待结算的 GPU event（低扰动模式：入队时不同步，flush 时统一取耗时）。
    std::vector<PendingGpuEvent> pending_gpu_;
    // 已归还、可复用的 CUDA event 池（避免反复 create/destroy）。
    std::vector<cudaEvent_t> event_pool_;
};

#endif // LOCAL_LLM_PROFILER_H
