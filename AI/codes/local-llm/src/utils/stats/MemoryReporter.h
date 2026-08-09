//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_MEMORYREPORTER_H
#define LOCAL_LLM_MEMORYREPORTER_H

#include <chrono>
#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include "utils/stats/StatsReport.h"

class CudaWeightPool;
class MemoryUsageProvider;

// 显存分层采集器：把推理过程中三类 device 内存的占用按时间线采样并汇总，配合耗时
// Profiler 一起指导优化。三类内存与代码里的职责分离一一对应：
//   - 持久权重（CudaWeightPool）：跨请求常驻，占用最大且固定。
//   - 跨 token 状态（KV cache / recurrent state 等）：随序列长度增长，是 decode
//     唯一随长度增长的访存量，量化 / f16 KV 的主战场（见 doc/2.md）。
//   - 前向临时激活（grow-only）：per-request，当前容量即峰值。
// 另附设备侧总量（cudaMemGetInfo），用于判断容量水位与 KV cache 增长空间。
//
// 与只取终值的单点快照不同，这里保留每次 sample() 的时间线（带时间戳与阶段标签），
// 从而能看清 KV cache / 显存随生成长度的增长曲线，指导长上下文与小内存优化。
// 时间戳与 Profiler / DeviceMonitor 用同一时钟（steady_clock），便于对齐阶段。
//
// 与具体模型解耦：权重经通用的 CudaWeightPool 查询；跨 token 状态与临时激活经
// MemoryUsageProvider 抽象接口获取，各模型（Qwen / 后续其它模型）自报字节数。
class MemoryReporter : public StatsReport {
public:
    // 分层显存用量快照（单位：字节），附时间戳与阶段标签。
    struct Snapshot {
        std::chrono::steady_clock::time_point ts; // 采样时间戳
        std::string label;         // 阶段标签（如 "prefill" / "decode"）

        size_t weight_bytes = 0;   // 持久权重（CudaWeightPool::cached_bytes）
        size_t kv_cache_bytes = 0; // 跨 token 状态：KV cache + recurrent / conv state
        size_t scratch_bytes = 0;  // 前向临时激活峰值

        // 设备侧总量（来自 cudaMemGetInfo）。
        size_t device_total_bytes = 0; // 显存总量
        size_t device_free_bytes = 0;  // 当前空闲
        size_t device_used_bytes = 0;  // 当前已用（total - free）
    };

    // 从各来源采集当前字节数，追加一条带时间戳 / 标签的快照到时间线。
    // usage 由具体模型的 session 实现，上报跨 token 状态与临时激活字节数。
    void sample(const CudaWeightPool &pool, const MemoryUsageProvider &usage,
                const std::string &label);

    // 采到的原始快照时间线。
    const std::vector<Snapshot> &samples() const { return samples_; }

    // 静态采集单条快照（不入时间线），供只需终值的场景使用。
    static Snapshot collect(const CudaWeightPool &pool, const MemoryUsageProvider &usage,
                            const std::string &label = "");

    // === StatsReport：三类报告 ===
    // JSONL 原始日志：每行一条快照（ts_ms / label / 各层字节 / 设备用量）。
    void write_jsonl(std::ostream &os) const override;
    // JSON summary：各层峰值 + 末值 + 占设备总量百分比。
    void write_json_summary(std::ostream &os) const override;
    // Markdown summary：分层显存表（峰值按 MiB / 占比），人类可读。
    void write_markdown_summary(std::ostream &os) const override;

private:
    std::vector<Snapshot> samples_;
};

#endif // LOCAL_LLM_MEMORYREPORTER_H
