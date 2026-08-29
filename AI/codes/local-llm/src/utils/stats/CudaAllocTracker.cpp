//
// Created by zhangyoulun on 29/8/2026.
//

#include "utils/stats/CudaAllocTracker.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace {
struct Allocation {
    size_t bytes = 0;
    CudaAllocKind kind = CudaAllocKind::Other;
};

std::mutex g_mutex;
std::unordered_map<void *, Allocation> g_allocations;
CudaAllocSnapshot g_stats;

CudaAllocStats &stats_for(CudaAllocSnapshot &snapshot, CudaAllocKind kind) {
    return snapshot.by_kind[static_cast<size_t>(kind)];
}

void add_alloc(CudaAllocStats &stats, size_t bytes) {
    stats.active_bytes += bytes;
    stats.total_allocated_bytes += bytes;
    ++stats.active_allocs;
    ++stats.total_allocs;
    stats.peak_active_bytes = std::max(stats.peak_active_bytes, stats.active_bytes);
    stats.peak_active_allocs = std::max(stats.peak_active_allocs, stats.active_allocs);
}

void add_free(CudaAllocStats &stats, size_t bytes) {
    stats.active_bytes -= bytes;
    stats.total_freed_bytes += bytes;
    --stats.active_allocs;
    ++stats.total_frees;
}

bool contains(const std::string &s, const char *needle) {
    return s.find(needle) != std::string::npos;
}
} // namespace

const char *cuda_alloc_kind_name(CudaAllocKind kind) {
    switch (kind) {
        case CudaAllocKind::Weight:
            return "weight";
        case CudaAllocKind::Dequant:
            return "dequant";
        case CudaAllocKind::Scratch:
            return "scratch";
        case CudaAllocKind::State:
            return "state";
        case CudaAllocKind::Other:
            return "other";
        case CudaAllocKind::Count:
            return "count";
    }
    return "unknown";
}

CudaAllocKind classify_cuda_alloc(const std::string &what) {
    if (contains(what, "s_weight")) {
        return CudaAllocKind::Weight;
    }
    if (contains(what, ".dequant.f16") || contains(what, "dequant")) {
        return CudaAllocKind::Dequant;
    }
    if (contains(what, "scratch[")) {
        return CudaAllocKind::Scratch;
    }
    if (contains(what, "key cache") || contains(what, "value cache") ||
        contains(what, "latent kv cache") || contains(what, "recurrent") ||
        contains(what, "conv state") || contains(what, "state")) {
        return CudaAllocKind::State;
    }
    return CudaAllocKind::Other;
}

void record_cuda_alloc(void *ptr, size_t bytes, CudaAllocKind kind, const std::string &) {
    if (ptr == nullptr || bytes == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto [_, inserted] = g_allocations.emplace(ptr, Allocation{bytes, kind});
    if (!inserted) {
        return;
    }
    add_alloc(stats_for(g_stats, kind), bytes);
    add_alloc(g_stats.total, bytes);
}

void record_cuda_free(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto found = g_allocations.find(ptr);
    if (found == g_allocations.end()) {
        return;
    }
    const Allocation allocation = found->second;
    g_allocations.erase(found);
    add_free(stats_for(g_stats, allocation.kind), allocation.bytes);
    add_free(g_stats.total, allocation.bytes);
}

CudaAllocSnapshot cuda_alloc_snapshot() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_stats;
}
