//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_CUDAWEIGHTPOOL_H
#define LOCAL_LLM_CUDAWEIGHTPOOL_H

#include "CudaWeight.h"

#include "format/MF.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include <cublas_v2.h>

class WeightLoadTracker;

// 通用的 CUDA 权重缓存：持有 cuBLAS handle，并按 tensor 名称惰性地把 host 端
// （mmap）权重上传到 device 后缓存复用。与具体模型结构无关——只依赖通用的
// TensorView（safetensors tensor 引用）。超过字节上限时整体清空重来。
//
// 注意：本类只负责“权重”这一类持久 device 内存；前向过程中反复覆盖的临时中间
// 结果请用 CudaScratchBuffer，二者职责分离。
class CudaWeightPool {
public:
    // 创建 cuBLAS handle。
    CudaWeightPool();
    // 释放权重缓存和 cuBLAS handle。
    ~CudaWeightPool();

    CudaWeightPool(const CudaWeightPool &) = delete;
    CudaWeightPool &operator=(const CudaWeightPool &) = delete;

    // 全局复用的 cuBLAS handle。
    cublasHandle_t handle = nullptr;

    // 获取 device 权重缓存；首次访问时从 mmap host 权重上传到 GPU。
    // BF16/F16/F32 权重可直接用于 GEMM；量化权重以 CUDA_R_8I 标记原始字节，
    // 使用前需由 Quant 反量化。单个权重超过上限时返回 nullptr。
    CudaWeight *cached_weight(const MFTensorView &weight);

    // 已缓存权重的总字节数。
    size_t cached_bytes() const { return bytes_; }

    // 设置懒加载追踪器（可选）；非空时会在每次 miss 上传 / 整体驱逐时回调。
    // 传 nullptr 关闭。追踪器生命周期由调用方管理。
    void set_load_tracker(WeightLoadTracker *tracker) { tracker_ = tracker; }

private:
    // 按 tensor 名称或组合名称索引的 device 权重缓存。
    std::unordered_map<std::string, CudaWeight> items_;
    // items_ 中已缓存权重的总字节数。
    size_t bytes_ = 0;
    // 懒加载追踪器（可选，不拥有）。
    WeightLoadTracker *tracker_ = nullptr;
    // 返回本进程允许缓存的 CUDA 权重总字节数（可由环境变量覆盖）。
    static size_t cache_limit_bytes();

    // 将 safetensors dtype 映射为 CUDA / cuBLAS dtype。
    static cudaDataType_t cuda_type_for(const MFTensorView &weight);

    // 返回当前支持 dtype 的单元素字节数。
    static size_t dtype_size_for(const MFTensorView &weight);

    // 计时版显存分配：timed 为 true 时用 CUDA event 测 cudaMalloc 耗时（毫秒，写入 out_ms），
    // 否则退化为普通分配、耗时返回 0。
    static void cuda_malloc_timed(void **ptr, size_t bytes, const std::string &what,
                                  bool timed, double &out_ms);

    // 计时版 H2D 拷贝：timed 为 true 时用 CUDA event 测 host->device 耗时（毫秒，写入 out_ms），
    // 否则退化为普通同步拷贝、耗时返回 0，避免非 profile 路径产生额外开销。
    static void memcpy_h2d_timed(void *dst, const void *src, size_t bytes, const std::string &what,
                                 bool timed, double &out_ms);
};

void set_global_cuda_weight_pool(CudaWeightPool *pool);
CudaWeightPool &global_cuda_weight_pool();

#endif // LOCAL_LLM_CUDAWEIGHTPOOL_H
