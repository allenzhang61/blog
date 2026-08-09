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
class Profiler : public StatsReport {
public:
    // 进程内唯一实例。
    static Profiler &instance();

    // 打开 / 关闭采集；关闭时 record 直接返回，埋点零开销。
    void enable(bool on) { enabled_ = on; }
    bool enabled() const { return enabled_; }

    // 记录一段耗时（毫秒）与该段读取的字节数（0 表示不统计带宽）。
    // 同名多次调用会累加到聚合统计，同时按调用顺序追加一条原始时间线记录。
    void record(const std::string &name, double ms, size_t bytes = 0);

    // 清空已采集数据（聚合统计 + 原始时间线）。
    // 用于 warmup 之后、正式计时之前归零。
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

    bool enabled_ = false;
    // 按名称聚合的累计统计。
    std::unordered_map<std::string, Stat> stats_;
    // 按调用顺序追加的原始时间线。
    std::vector<Event> events_;
};

#endif // LOCAL_LLM_PROFILER_H
