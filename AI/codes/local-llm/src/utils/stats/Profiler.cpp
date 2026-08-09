//
// Created by zhangyoulun on 9/8/2026.
//

#include "utils/stats/Profiler.h"

#include <algorithm>
#include <vector>

Profiler &Profiler::instance() {
    static Profiler inst;
    return inst;
}

void Profiler::record(const std::string &name, double ms, size_t bytes) {
    if (!enabled_) {
        return;
    }
    Stat &s = stats_[name];
    s.count += 1;
    s.total_ms += ms;
    s.total_bytes += bytes;

    events_.push_back(Event{name, ms, bytes, std::chrono::steady_clock::now()});
}

void Profiler::reset() {
    stats_.clear();
    events_.clear();
}

namespace {
// GB/s：bytes / (ms/1000) / 1e9 = bytes / ms / 1e6。
double bandwidth_gbps(uint64_t bytes, double ms) {
    if (ms <= 0.0 || bytes == 0) {
        return 0.0;
    }
    return static_cast<double>(bytes) / ms / 1.0e6;
}

// 每秒 token 数：count / (ms/1000) = count / ms * 1000。
double tokens_per_sec(uint64_t count, double ms) {
    if (ms <= 0.0 || count == 0) {
        return 0.0;
    }
    return static_cast<double>(count) / ms * 1000.0;
}
} // namespace

void Profiler::write_jsonl(std::ostream &os) const {
    if (events_.empty()) {
        return;
    }
    const auto t0 = events_.front().ts;
    for (const Event &e : events_) {
        const double ts_ms = std::chrono::duration<double, std::milli>(e.ts - t0).count();
        os << "{\"kind\":\"profiler\""
           << ",\"ts_ms\":" << ts_ms
           << ",\"name\":\"" << e.name << "\""
           << ",\"ms\":" << e.ms
           << ",\"bytes\":" << e.bytes
           << "}\n";
    }
}

void Profiler::write_json_summary(std::ostream &os) const {
    // 总耗时用于算各项占比。
    double total_ms = 0.0;
    for (const auto &kv : stats_) {
        total_ms += kv.second.total_ms;
    }

    // decode 稳态吞吐：用逐 token 计时的 decode_token 聚合派生（最纯净的稳态口径）。
    double decode_tps = 0.0;
    double decode_avg_ms = 0.0;
    if (auto it = stats_.find("decode_token"); it != stats_.end()) {
        const Stat &s = it->second;
        decode_tps = tokens_per_sec(s.count, s.total_ms);
        decode_avg_ms = s.count > 0 ? s.total_ms / static_cast<double>(s.count) : 0.0;
    }

    os << "{\"kind\":\"profiler_summary\",\"total_ms\":" << total_ms
       << ",\"decode_tokens_per_sec\":" << decode_tps
       << ",\"decode_avg_ms_per_token\":" << decode_avg_ms
       << ",\"items\":[";
    bool first = true;
    for (const auto &kv : stats_) {
        const Stat &s = kv.second;
        const double avg_ms = s.count > 0 ? s.total_ms / static_cast<double>(s.count) : 0.0;
        const double pct = total_ms > 0.0 ? s.total_ms / total_ms * 100.0 : 0.0;
        if (!first) {
            os << ",";
        }
        first = false;
        os << "{\"name\":\"" << kv.first << "\""
           << ",\"count\":" << s.count
           << ",\"total_ms\":" << s.total_ms
           << ",\"avg_ms\":" << avg_ms
           << ",\"pct\":" << pct
           << ",\"bandwidth_gbps\":" << bandwidth_gbps(s.total_bytes, s.total_ms)
           << "}";
    }
    os << "]}\n";
}

void Profiler::write_markdown_summary(std::ostream &os) const {
    double total_ms = 0.0;
    for (const auto &kv : stats_) {
        total_ms += kv.second.total_ms;
    }

    // 按累计耗时降序排列，突出瓶颈。
    std::vector<const std::pair<const std::string, Stat> *> items;
    items.reserve(stats_.size());
    for (const auto &kv : stats_) {
        items.push_back(&kv);
    }
    std::sort(items.begin(), items.end(),
              [](const auto *a, const auto *b) { return a->second.total_ms > b->second.total_ms; });

    os << "## Profiler summary\n\n";

    // decode 稳态吞吐：逐 token 计时的 decode_token 聚合派生。
    if (auto it = stats_.find("decode_token"); it != stats_.end()) {
        const Stat &s = it->second;
        const double avg_ms = s.count > 0 ? s.total_ms / static_cast<double>(s.count) : 0.0;
        os << "- decode 吞吐：" << tokens_per_sec(s.count, s.total_ms) << " tokens/s"
           << "（" << s.count << " tokens，平均 " << avg_ms << " ms/token）\n\n";
    }

    os << "| name | count | total_ms | avg_ms | pct | bandwidth(GB/s) |\n";
    os << "|---|---:|---:|---:|---:|---:|\n";
    for (const auto *kv : items) {
        const Stat &s = kv->second;
        const double avg_ms = s.count > 0 ? s.total_ms / static_cast<double>(s.count) : 0.0;
        const double pct = total_ms > 0.0 ? s.total_ms / total_ms * 100.0 : 0.0;
        os << "| " << kv->first
           << " | " << s.count
           << " | " << s.total_ms
           << " | " << avg_ms
           << " | " << pct << "%"
           << " | " << bandwidth_gbps(s.total_bytes, s.total_ms)
           << " |\n";
    }
    os << "\n";
}
