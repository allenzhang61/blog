//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_DEVICEMONITOR_H
#define LOCAL_LLM_DEVICEMONITOR_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <ostream>
#include <thread>
#include <vector>

#include "utils/stats/StatsReport.h"

// 设备侧时序监控：后台线程按固定间隔用 NVML 采样 GPU 运行时状态，形成带时间戳的
// 时间序列。与 Profiler 的应用内埋点互补——埋点回答“各算子/层耗时占比”，本类回答
// “采样期间 GPU 到底忙不忙、卡在算力还是访存”：
//   - SM 利用率低 + 显存带宽利用率高 => 访存瓶颈（印证 doc/2.md 中 decode 的判断）。
//   - 功耗 / 温度用于交叉验证是否真在满负荷运行。
//
// 依赖 NVML（libnvidia-ml，CUDA 自带）。因采样有开销且需起后台线程，仅在 --profile
// 时启动，不常驻。非线程安全接口：start / stop 由主线程串行调用。
class DeviceMonitor : public StatsReport {
public:
    // 单次采样快照：一个时间点的设备状态。
    struct Sample {
        // 采样时刻的时间戳，与 Profiler timeline 用同一时钟（steady_clock），
        // 便于把设备利用率曲线对齐到 prefill / decode 各阶段。
        std::chrono::steady_clock::time_point ts;
        unsigned int sm_util_percent = 0;      // SM（计算）利用率 %
        unsigned int mem_util_percent = 0;     // 显存控制器（带宽）利用率 %
        unsigned int power_milliwatt = 0;      // 瞬时功耗（mW）
        unsigned int temperature_celsius = 0;  // GPU 温度（℃）
        size_t mem_used_bytes = 0;             // 已用显存字节数
    };

    // 绑定要监控的 GPU 序号（对应 nvmlDeviceGetHandleByIndex）。
    explicit DeviceMonitor(int device_index = 0);
    ~DeviceMonitor();

    DeviceMonitor(const DeviceMonitor &) = delete;
    DeviceMonitor &operator=(const DeviceMonitor &) = delete;

    // 启动后台采样线程，每 interval_ms 采一次；重复调用前需先 stop。
    void start(int interval_ms = 100);
    // 停止采样线程并 join。
    void stop();

    // 采到的原始时间序列（stop 后读取）。
    const std::vector<Sample> &samples() const { return samples_; }

    // === StatsReport：三类报告 ===
    // JSONL 原始日志：每行一条采样（ts_ms / sm% / mem-bw% / power / temp / mem）。
    void write_jsonl(std::ostream &os) const override;
    // JSON summary：各指标的均值 / 峰值 + 采样数（判断算力 vs 访存瓶颈）。
    void write_json_summary(std::ostream &os) const override;
    // Markdown summary：均值 / 峰值表格，人类可读。
    void write_markdown_summary(std::ostream &os) const override;

private:
    // 后台线程循环体：初始化 NVML、按 interval 采样、退出前清理。
    void run(int interval_ms);

    int device_index_ = 0;
    std::atomic<bool> running_{false};
    std::thread worker_;
    std::vector<Sample> samples_;
};

#endif // LOCAL_LLM_DEVICEMONITOR_H
