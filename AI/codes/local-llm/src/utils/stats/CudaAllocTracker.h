//
// Created by zhangyoulun on 29/8/2026.
//

#ifndef LOCAL_LLM_CUDAALLOCTRACKER_H
#define LOCAL_LLM_CUDAALLOCTRACKER_H

#include <array>
#include <cstddef>
#include <string>

enum class CudaAllocKind {
    Weight = 0,
    Dequant,
    Scratch,
    State,
    Other,
    Count,
};

struct CudaAllocStats {
    size_t active_bytes = 0;
    size_t peak_active_bytes = 0;
    size_t total_allocated_bytes = 0;
    size_t total_freed_bytes = 0;
    size_t active_allocs = 0;
    size_t peak_active_allocs = 0;
    size_t total_allocs = 0;
    size_t total_frees = 0;
};

struct CudaAllocSnapshot {
    std::array<CudaAllocStats, static_cast<size_t>(CudaAllocKind::Count)> by_kind{};
    CudaAllocStats total{};
};

const char *cuda_alloc_kind_name(CudaAllocKind kind);
CudaAllocKind classify_cuda_alloc(const std::string &what);
void record_cuda_alloc(void *ptr, size_t bytes, CudaAllocKind kind, const std::string &what);
void record_cuda_free(void *ptr);
CudaAllocSnapshot cuda_alloc_snapshot();

#endif // LOCAL_LLM_CUDAALLOCTRACKER_H
