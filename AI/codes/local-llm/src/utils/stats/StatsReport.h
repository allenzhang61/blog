//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_STATSREPORT_H
#define LOCAL_LLM_STATSREPORT_H

#include <ostream>

// 统一的三类报告输出约定：stats 目录下所有采集器（Profiler / MemoryReporter /
// DeviceMonitor / WeightLoadTracker）都按同一套语义产出报告，便于统一落盘与后续
// 脚本消费。三类内容职责区分明确：
//
//   1) JSONL 原始日志（write_jsonl）：
//      面向“完整明细”。每行一个独立 JSON 对象，对应一条原始事件 / 采样 / 时间线
//      记录，不做聚合。适合流式追加、大体量、逐行 grep / 喂给分析脚本。
//      约定每条至少含 { "kind": <采集器标识>, "ts_ms": <相对起点毫秒>, ... }。
//
//   2) JSON summary（write_json_summary）：
//      面向“机器消费的聚合结论”。单个 JSON 对象，含聚合统计（总量 / 均值 / 峰值 /
//      占比 / 有效带宽等），用于不同优化前后的自动化回归对比。
//
//   3) Markdown summary（write_markdown_summary）：
//      面向“人读”。与 JSON summary 同源的聚合结论，渲染成标题 + 表格，便于直接贴进
//      doc/ 或 MR 描述。
//
// 说明：三者都写入调用方给定的 ostream，由上层决定落到 .jsonl / .json / .md 文件。
class StatsReport {
public:
    virtual ~StatsReport() = default;

    // 逐行输出 JSONL 原始日志（一行一条记录，无外层数组）。
    virtual void write_jsonl(std::ostream &os) const = 0;

    // 输出单对象 JSON summary（聚合结论）。
    virtual void write_json_summary(std::ostream &os) const = 0;

    // 输出 Markdown summary（标题 + 表格，人类可读）。
    virtual void write_markdown_summary(std::ostream &os) const = 0;
};

#endif // LOCAL_LLM_STATSREPORT_H
