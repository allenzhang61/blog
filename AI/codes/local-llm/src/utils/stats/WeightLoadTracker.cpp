//
// Created by zhangyoulun on 9/8/2026.
//

#include "utils/stats/WeightLoadTracker.h"

#include <algorithm>

void InMemoryWeightLoadTracker::record(WeightLoadEventKind kind, const std::string &name,
                                       size_t bytes, double ms, size_t resident_bytes) {
    Event e;
    e.kind = kind;
    e.name = name;
    e.bytes = bytes;
    e.ms = ms;
    e.resident_bytes = resident_bytes;
    e.ts = std::chrono::steady_clock::now();
    events_.push_back(std::move(e));

    // 只把 H2D 拷贝计入“上传总量”，分配（Alloc）不重复计字节。
    if (kind == Kind::Upload) {
        total_uploaded_bytes_ += bytes;
    }
    if (kind != Kind::EvictAll) {
        peak_resident_bytes_ = std::max(peak_resident_bytes_, resident_bytes);
    }
}

namespace {
const char *kind_str(InMemoryWeightLoadTracker::Kind k) {
    switch (k) {
        case InMemoryWeightLoadTracker::Kind::Alloc: return "alloc";
        case InMemoryWeightLoadTracker::Kind::Upload: return "upload";
        case InMemoryWeightLoadTracker::Kind::EvictAll: return "evict_all";
    }
    return "unknown";
}
double to_mib(size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }
} // namespace

void InMemoryWeightLoadTracker::write_jsonl(std::ostream &os) const {
    if (events_.empty()) {
        return;
    }
    const auto t0 = events_.front().ts;
    for (const Event &e : events_) {
        const double ts_ms = std::chrono::duration<double, std::milli>(e.ts - t0).count();
        os << "{\"kind\":\"weight_load\""
           << ",\"event\":\"" << kind_str(e.kind) << "\""
           << ",\"ts_ms\":" << ts_ms
           << ",\"name\":\"" << e.name << "\""
           << ",\"bytes\":" << e.bytes
           << ",\"ms\":" << e.ms
           << ",\"resident_bytes\":" << e.resident_bytes
           << "}\n";
    }
}

void InMemoryWeightLoadTracker::write_json_summary(std::ostream &os) const {
    size_t alloc_count = 0;
    size_t upload_count = 0;
    size_t evict_count = 0;
    double total_alloc_ms = 0.0;
    double total_h2d_ms = 0.0;
    for (const Event &e : events_) {
        switch (e.kind) {
            case Kind::Alloc: ++alloc_count; total_alloc_ms += e.ms; break;
            case Kind::Upload: ++upload_count; total_h2d_ms += e.ms; break;
            case Kind::EvictAll: ++evict_count; break;
        }
    }
    os << "{\"kind\":\"weight_load_summary\""
       << ",\"alloc_count\":" << alloc_count
       << ",\"upload_count\":" << upload_count
       << ",\"evict_count\":" << evict_count
       << ",\"total_uploaded_bytes\":" << total_uploaded_bytes_
       << ",\"peak_resident_bytes\":" << peak_resident_bytes_
       << ",\"total_alloc_ms\":" << total_alloc_ms
       << ",\"total_h2d_ms\":" << total_h2d_ms
       << "}\n";
}

void InMemoryWeightLoadTracker::write_markdown_summary(std::ostream &os) const {
    if (events_.empty()) {
        os << "## Weight load summary\n\n（无懒加载事件）\n\n";
        return;
    }
    const auto t0 = events_.front().ts;

    double total_alloc_ms = 0.0;
    double total_h2d_ms = 0.0;
    for (const Event &e : events_) {
        if (e.kind == Kind::Alloc) {
            total_alloc_ms += e.ms;
        } else if (e.kind == Kind::Upload) {
            total_h2d_ms += e.ms;
        }
    }

    os << "## Weight load summary\n\n";
    os << "- 上传总量：" << to_mib(total_uploaded_bytes_) << " MiB\n";
    os << "- 驻留峰值：" << to_mib(peak_resident_bytes_) << " MiB\n";
    os << "- 分配总耗时（cudaMalloc）：" << total_alloc_ms << " ms\n";
    os << "- 拷贝总耗时（H2D）：" << total_h2d_ms << " ms\n\n";

    os << "| ts_ms | event | name | bytes(MiB) | ms | resident(MiB) |\n";
    os << "|---:|---|---|---:|---:|---:|\n";
    for (const Event &e : events_) {
        const double ts_ms = std::chrono::duration<double, std::milli>(e.ts - t0).count();
        os << "| " << ts_ms
           << " | " << kind_str(e.kind)
           << " | " << (e.name.empty() ? "-" : e.name)
           << " | " << to_mib(e.bytes)
           << " | " << e.ms
           << " | " << to_mib(e.resident_bytes)
           << " |\n";
    }
    os << "\n";
}
