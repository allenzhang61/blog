//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_WEIGHTLOADTRACKER_H
#define LOCAL_LLM_WEIGHTLOADTRACKER_H

#include <chrono>
#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include "utils/stats/StatsReport.h"

// 权重懒加载事件类型：
//   Alloc    —— cudaMalloc 分配一块 device 显存
//   Upload   —— host->device 拷贝（H2D）
//   EvictAll —— 历史/预留事件：旧版权重池超限时整体清空
// 定义在接口之外，便于抽象基类的回调直接使用。
enum class WeightLoadEventKind { Alloc, Upload, EvictAll };

// 权重懒加载追踪：记录“模型权重一块一块被搬进显存”的过程。与只给终值的
// CudaWeightPool::cached_bytes() 不同，这里保留每一次事件的原始时间线
// （类型 / 名称 / 字节 / 耗时 / 时间戳 / 事件后累计驻留量），包括分配（Alloc）
// 与拷贝（Upload）。EvictAll 仅保留用于兼容历史报告格式。
//
// 用途（面向小内存设备优化）：
//   - 看清权重按什么顺序、什么时机被换入，指导预取 / 分块 / layer offload。
//   - 观察驻留量随时间的增长曲线与峰值，判断能否在给定显存预算内容纳。
//   - 分配与拷贝分开记录，便于将来做预分配池 / 换入换出时区分二者开销。
//
// 设计为抽象接口：CudaWeightPool 只在分配 / 拷贝 / 驱逐时回调本接口，不反向依赖具体
// 采集实现，从而保持 pool 通用。默认实现 InMemoryWeightLoadTracker 把事件存在内存中。
class WeightLoadTracker {
public:
    virtual ~WeightLoadTracker() = default;

    // 记录一条权重懒加载事件。
    //   kind           事件类型（Alloc / Upload / EvictAll）
    //   name           相关 tensor / 组合名称（EvictAll 时为空）
    //   bytes          本次事件涉及的字节数（分配 / 拷贝 / 释放）
    //   ms             本次事件耗时（毫秒；EvictAll 通常为 0）
    //   resident_bytes 事件发生后缓存内的累计驻留字节数
    virtual void record(WeightLoadEventKind kind, const std::string &name, size_t bytes,
                        double ms, size_t resident_bytes) = 0;
};

// 默认实现：把懒加载事件按时间顺序存在内存里，供结束后打印或导出。
// 时间戳与 Profiler / DeviceMonitor 用同一时钟（steady_clock），便于把权重换入
// 曲线对齐到 prefill / decode 各阶段。
class InMemoryWeightLoadTracker : public WeightLoadTracker, public StatsReport {
public:
    using Kind = WeightLoadEventKind;

    // 单条懒加载事件记录。
    struct Event {
        Kind kind = Kind::Upload;
        std::string name;        // 相关权重名（EvictAll 时为空）
        size_t bytes = 0;        // 分配 / 拷贝 / 释放字节数
        double ms = 0.0;         // 事件耗时（毫秒）
        size_t resident_bytes = 0; // 事件发生后缓存驻留字节数
        std::chrono::steady_clock::time_point ts; // 事件时间戳
    };

    void record(WeightLoadEventKind kind, const std::string &name, size_t bytes,
                double ms, size_t resident_bytes) override;

    // 采到的原始事件时间线。
    const std::vector<Event> &events() const { return events_; }

    // 上传总量与观察到的驻留峰值（用于小内存预算评估）。
    size_t total_uploaded_bytes() const { return total_uploaded_bytes_; }
    size_t peak_resident_bytes() const { return peak_resident_bytes_; }

    // === StatsReport：三类报告 ===
    // JSONL 原始日志：每行一条懒加载事件（ts_ms / kind / name / bytes / ms / resident）。
    void write_jsonl(std::ostream &os) const override;
    // JSON summary：分配 / 上传 / 驱逐次数、总上传字节、驻留峰值、分配与拷贝总耗时。
    void write_json_summary(std::ostream &os) const override;
    // Markdown summary：懒加载事件表 + 总量 / 峰值 / 耗时，人类可读。
    void write_markdown_summary(std::ostream &os) const override;

private:
    std::vector<Event> events_;
    size_t total_uploaded_bytes_ = 0;
    size_t peak_resident_bytes_ = 0;
};

#endif // LOCAL_LLM_WEIGHTLOADTRACKER_H
