//
// Created by zhangyoulun on 9/8/2026.
//

#include "utils/stats/MemoryReporter.h"

#include <algorithm>

#include <cuda_runtime.h>

#include "backend/cuda/mem/CudaWeightDequantPool.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "utils/stats/CudaAllocTracker.h"
#include "utils/stats/MemoryUsageProvider.h"

namespace {
// 字节转 MiB。
double to_mib(size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

// 占设备总量百分比。
double pct_of_total(size_t bytes, size_t total) {
    return total > 0 ? static_cast<double>(bytes) / static_cast<double>(total) * 100.0 : 0.0;
}

size_t alloc_active(const CudaAllocSnapshot &snapshot, CudaAllocKind kind) {
    return snapshot.by_kind[static_cast<size_t>(kind)].active_bytes;
}
} // namespace

MemoryReporter::Snapshot MemoryReporter::collect(const CudaWeightPool &pool,
                                                 const MemoryUsageProvider &usage,
                                                 const std::string &label) {
    Snapshot snap;
    snap.ts = std::chrono::steady_clock::now();
    snap.label = label;
    snap.weight_bytes = pool.cached_bytes();
    snap.kv_cache_bytes = usage.kv_state_bytes();
    snap.scratch_bytes = usage.scratch_bytes();
    if (auto *dequant_pool = global_cuda_weight_dequant_pool()) {
        snap.dequant_cache_bytes = dequant_pool->cached_bytes();
    }
    snap.alloc = cuda_alloc_snapshot();

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    // 失败时保持 0；不抛异常以免影响主流程。
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        snap.device_total_bytes = total_bytes;
        snap.device_free_bytes = free_bytes;
        snap.device_used_bytes = total_bytes - free_bytes;
    }
    return snap;
}

void MemoryReporter::sample(const CudaWeightPool &pool, const MemoryUsageProvider &usage,
                            const std::string &label) {
    samples_.push_back(collect(pool, usage, label));
}

void MemoryReporter::write_jsonl(std::ostream &os) const {
    if (samples_.empty()) {
        return;
    }
    const auto t0 = samples_.front().ts;
    for (const Snapshot &s : samples_) {
        const double ts_ms = std::chrono::duration<double, std::milli>(s.ts - t0).count();
        os << "{\"kind\":\"memory\""
           << ",\"ts_ms\":" << ts_ms
           << ",\"label\":\"" << s.label << "\""
           << ",\"weight_bytes\":" << s.weight_bytes
           << ",\"kv_cache_bytes\":" << s.kv_cache_bytes
           << ",\"scratch_bytes\":" << s.scratch_bytes
           << ",\"dequant_cache_bytes\":" << s.dequant_cache_bytes
           << ",\"tracked_alloc_bytes\":" << s.alloc.total.active_bytes
           << ",\"tracked_weight_bytes\":" << alloc_active(s.alloc, CudaAllocKind::Weight)
           << ",\"tracked_dequant_bytes\":" << alloc_active(s.alloc, CudaAllocKind::Dequant)
           << ",\"tracked_scratch_bytes\":" << alloc_active(s.alloc, CudaAllocKind::Scratch)
           << ",\"tracked_state_bytes\":" << alloc_active(s.alloc, CudaAllocKind::State)
           << ",\"tracked_other_bytes\":" << alloc_active(s.alloc, CudaAllocKind::Other)
           << ",\"tracked_allocs\":" << s.alloc.total.active_allocs
           << ",\"tracked_weight_allocs\":" << s.alloc.by_kind[static_cast<size_t>(CudaAllocKind::Weight)].active_allocs
           << ",\"tracked_dequant_allocs\":" << s.alloc.by_kind[static_cast<size_t>(CudaAllocKind::Dequant)].active_allocs
           << ",\"tracked_scratch_allocs\":" << s.alloc.by_kind[static_cast<size_t>(CudaAllocKind::Scratch)].active_allocs
           << ",\"tracked_state_allocs\":" << s.alloc.by_kind[static_cast<size_t>(CudaAllocKind::State)].active_allocs
           << ",\"tracked_other_allocs\":" << s.alloc.by_kind[static_cast<size_t>(CudaAllocKind::Other)].active_allocs
           << ",\"device_total_bytes\":" << s.device_total_bytes
           << ",\"device_free_bytes\":" << s.device_free_bytes
           << ",\"device_used_bytes\":" << s.device_used_bytes
           << "}\n";
    }
}

void MemoryReporter::write_json_summary(std::ostream &os) const {
    if (samples_.empty()) {
        os << "{\"kind\":\"memory_summary\",\"samples\":0}\n";
        return;
    }
    // 各层取时间线峰值；设备用量与总量取末值（末值即最大 seq_len 时的水位）。
    size_t peak_weight = 0, peak_kv = 0, peak_scratch = 0, peak_used = 0;
    size_t peak_dequant = 0, peak_tracked = 0, peak_untracked = 0;
    std::array<size_t, static_cast<size_t>(CudaAllocKind::Count)> peak_by_kind{};
    for (const Snapshot &s : samples_) {
        peak_weight = std::max(peak_weight, s.weight_bytes);
        peak_kv = std::max(peak_kv, s.kv_cache_bytes);
        peak_scratch = std::max(peak_scratch, s.scratch_bytes);
        peak_used = std::max(peak_used, s.device_used_bytes);
        peak_dequant = std::max(peak_dequant, s.dequant_cache_bytes);
        peak_tracked = std::max(peak_tracked, s.alloc.total.active_bytes);
        peak_untracked = std::max(peak_untracked, s.device_used_bytes > s.alloc.total.active_bytes
                                                      ? s.device_used_bytes - s.alloc.total.active_bytes
                                                      : 0);
        for (size_t i = 0; i < peak_by_kind.size(); ++i) {
            peak_by_kind[i] = std::max(peak_by_kind[i], s.alloc.by_kind[i].active_bytes);
        }
    }
    const Snapshot &last = samples_.back();
    const size_t total = last.device_total_bytes;

    os << "{\"kind\":\"memory_summary\""
       << ",\"samples\":" << samples_.size()
       << ",\"peak_weight_bytes\":" << peak_weight
       << ",\"peak_kv_cache_bytes\":" << peak_kv
       << ",\"peak_scratch_bytes\":" << peak_scratch
       << ",\"peak_dequant_cache_bytes\":" << peak_dequant
       << ",\"peak_tracked_alloc_bytes\":" << peak_tracked
       << ",\"peak_tracked_weight_bytes\":" << peak_by_kind[static_cast<size_t>(CudaAllocKind::Weight)]
       << ",\"peak_tracked_dequant_bytes\":" << peak_by_kind[static_cast<size_t>(CudaAllocKind::Dequant)]
       << ",\"peak_tracked_scratch_bytes\":" << peak_by_kind[static_cast<size_t>(CudaAllocKind::Scratch)]
       << ",\"peak_tracked_state_bytes\":" << peak_by_kind[static_cast<size_t>(CudaAllocKind::State)]
       << ",\"peak_tracked_other_bytes\":" << peak_by_kind[static_cast<size_t>(CudaAllocKind::Other)]
       << ",\"peak_untracked_device_bytes\":" << peak_untracked
       << ",\"peak_device_used_bytes\":" << peak_used
       << ",\"device_total_bytes\":" << total
       << ",\"last_kv_cache_bytes\":" << last.kv_cache_bytes
       << ",\"peak_weight_pct\":" << pct_of_total(peak_weight, total)
       << ",\"peak_kv_cache_pct\":" << pct_of_total(peak_kv, total)
       << ",\"peak_scratch_pct\":" << pct_of_total(peak_scratch, total)
       << ",\"peak_dequant_cache_pct\":" << pct_of_total(peak_dequant, total)
       << ",\"peak_tracked_alloc_pct\":" << pct_of_total(peak_tracked, total)
       << ",\"peak_untracked_device_pct\":" << pct_of_total(peak_untracked, total)
       << ",\"peak_device_used_pct\":" << pct_of_total(peak_used, total)
       << "}\n";
}

void MemoryReporter::write_markdown_summary(std::ostream &os) const {
    os << "## Memory summary\n\n";
    if (samples_.empty()) {
        os << "（无显存采样）\n\n";
        return;
    }
    size_t peak_weight = 0, peak_kv = 0, peak_scratch = 0, peak_used = 0;
    size_t peak_dequant = 0, peak_tracked = 0, peak_untracked = 0;
    std::array<size_t, static_cast<size_t>(CudaAllocKind::Count)> peak_by_kind{};
    for (const Snapshot &s : samples_) {
        peak_weight = std::max(peak_weight, s.weight_bytes);
        peak_kv = std::max(peak_kv, s.kv_cache_bytes);
        peak_scratch = std::max(peak_scratch, s.scratch_bytes);
        peak_used = std::max(peak_used, s.device_used_bytes);
        peak_dequant = std::max(peak_dequant, s.dequant_cache_bytes);
        peak_tracked = std::max(peak_tracked, s.alloc.total.active_bytes);
        peak_untracked = std::max(peak_untracked, s.device_used_bytes > s.alloc.total.active_bytes
                                                      ? s.device_used_bytes - s.alloc.total.active_bytes
                                                      : 0);
        for (size_t i = 0; i < peak_by_kind.size(); ++i) {
            peak_by_kind[i] = std::max(peak_by_kind[i], s.alloc.by_kind[i].active_bytes);
        }
    }
    const size_t total = samples_.back().device_total_bytes;

    os << "采样数：" << samples_.size() << "（下表为时间线峰值）\n\n";
    os << "| layer | peak(MiB) | pct of device |\n";
    os << "|---|---:|---:|\n";
    os << "| weights | " << to_mib(peak_weight)
       << " | " << pct_of_total(peak_weight, total) << "% |\n";
    os << "| kv cache / state | " << to_mib(peak_kv)
       << " | " << pct_of_total(peak_kv, total) << "% |\n";
    os << "| scratch (peak) | " << to_mib(peak_scratch)
       << " | " << pct_of_total(peak_scratch, total) << "% |\n";
    os << "| dequant cache | " << to_mib(peak_dequant)
       << " | " << pct_of_total(peak_dequant, total) << "% |\n";
    os << "| tracked cuda allocs | " << to_mib(peak_tracked)
       << " | " << pct_of_total(peak_tracked, total) << "% |\n";
    os << "| tracked weight allocs | " << to_mib(peak_by_kind[static_cast<size_t>(CudaAllocKind::Weight)])
       << " | " << pct_of_total(peak_by_kind[static_cast<size_t>(CudaAllocKind::Weight)], total) << "% |\n";
    os << "| tracked dequant allocs | " << to_mib(peak_by_kind[static_cast<size_t>(CudaAllocKind::Dequant)])
       << " | " << pct_of_total(peak_by_kind[static_cast<size_t>(CudaAllocKind::Dequant)], total) << "% |\n";
    os << "| tracked scratch allocs | " << to_mib(peak_by_kind[static_cast<size_t>(CudaAllocKind::Scratch)])
       << " | " << pct_of_total(peak_by_kind[static_cast<size_t>(CudaAllocKind::Scratch)], total) << "% |\n";
    os << "| tracked state allocs | " << to_mib(peak_by_kind[static_cast<size_t>(CudaAllocKind::State)])
       << " | " << pct_of_total(peak_by_kind[static_cast<size_t>(CudaAllocKind::State)], total) << "% |\n";
    os << "| tracked other allocs | " << to_mib(peak_by_kind[static_cast<size_t>(CudaAllocKind::Other)])
       << " | " << pct_of_total(peak_by_kind[static_cast<size_t>(CudaAllocKind::Other)], total) << "% |\n";
    os << "| untracked device used | " << to_mib(peak_untracked)
       << " | " << pct_of_total(peak_untracked, total) << "% |\n";
    os << "| device used | " << to_mib(peak_used)
       << " | " << pct_of_total(peak_used, total) << "% |\n";
    os << "| device total | " << to_mib(total) << " | 100% |\n\n";

    // KV cache 增长曲线：随生成长度变化，指导长上下文 / 小内存优化。
    os << "### KV cache growth\n\n";
    os << "| ts_ms | label | kv(MiB) | scratch(MiB) | tracked(MiB) | untracked(MiB) | device used(MiB) |\n";
    os << "|---:|---|---:|---:|---:|---:|---:|\n";
    const auto t0 = samples_.front().ts;
    for (const Snapshot &s : samples_) {
        const double ts_ms = std::chrono::duration<double, std::milli>(s.ts - t0).count();
        const size_t untracked = s.device_used_bytes > s.alloc.total.active_bytes
                                     ? s.device_used_bytes - s.alloc.total.active_bytes
                                     : 0;
        os << "| " << ts_ms
           << " | " << (s.label.empty() ? "-" : s.label)
           << " | " << to_mib(s.kv_cache_bytes)
           << " | " << to_mib(s.scratch_bytes)
           << " | " << to_mib(s.alloc.total.active_bytes)
           << " | " << to_mib(untracked)
           << " | " << to_mib(s.device_used_bytes)
           << " |\n";
    }
    os << "\n";
}
