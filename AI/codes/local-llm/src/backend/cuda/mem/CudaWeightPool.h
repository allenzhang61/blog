//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_CUDAWEIGHTPOOL_H
#define LOCAL_LLM_CUDAWEIGHTPOOL_H

#include "CudaWeight.h"

#include "llm/qwen/QwenWeights.h" // 仅依赖其中通用的 WeightData / WeightMeta

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include <cublas_v2.h>

class WeightLoadTracker;

// 通用的 CUDA 权重缓存：持有 cuBLAS handle，并按 tensor 名称惰性地把 host 端
// （mmap）权重上传到 device 后缓存复用。与具体模型结构无关——只依赖通用的
// WeightData（safetensors tensor 引用）。超过字节上限时整体清空重来。
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

    // 获取普通 device 权重缓存；首次访问时从 mmap host 权重上传到 GPU。
    // 单个权重超过上限时返回 nullptr。
    CudaWeight *cached_weight(const WeightData &weight);

    // 将多个二维权重按行拼接后上传并缓存，用于合并 projection（如 QKV / gate+up）。
    // 要求各权重 dtype 相同、列数（shape[1]）一致；不满足或超限时返回 nullptr。
    CudaWeight *cached_concat_weight(const std::string &name, const std::vector<WeightData> &weights);

    // 将一段 GGUF Q4_K 原始字节常驻上传并缓存（按 name 复用）。返回的 CudaWeight
    // 以 type=CUDA_R_8I 标记“原始量化字节”（非可直接 gemm 的 dtype）；src_bytes 为字节数。
    // 超过上限时返回 nullptr。使用时先经 dequantize_q4k_to_f16 反量化到临时 f16 buffer。
    CudaWeight *cached_q4k_weight(const std::string &name, const uint8_t *host_src, size_t src_bytes);

    // 把常驻的 Q4_K 权重反量化到 device 端 f16 输出（d_out_f16，元素数 >= num_elements），
    // 返回一个包装该 f16 buffer 的非拥有 CudaWeight 视图，可直接传给 gemm_weight。
    // d_out_f16 通常取自 CudaScratchBuffer<uint16_t>（grow-only，跨调用复用）。
    static CudaWeight dequantize_q4k_to_f16(const CudaWeight &q4k, uint16_t *d_out_f16,
                                            int64_t num_elements);

    // 通用量化字节常驻上传：与 cached_q4k_weight 相同（type=CUDA_R_8I 标记原始字节），
    // 但不限定量化类型，适用于 Q4_K/Q6_K/Q8_0/Q5_0/F32 等任意 GGUF 原始张量字节。
    CudaWeight *cached_quant_weight(const std::string &name, const uint8_t *host_src, size_t src_bytes);

    // 按 GGML 类型码把常驻量化权重反量化到 device f16（d_out_f16 元素数 >= num_elements），
    // 返回非拥有 f16 视图。ggml_type：0=F32,6=Q5_0,8=Q8_0,12=Q4_K,14=Q6_K。stream 为 CUDA stream。
    static CudaWeight dequantize_to_f16(const CudaWeight &quant, uint16_t *d_out_f16,
                                        int64_t num_elements, int ggml_type, void *stream = nullptr);

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
    static cudaDataType_t cuda_type_for(const WeightData &weight);

    // 返回当前支持 dtype 的单元素字节数。
    static size_t dtype_size_for(const WeightData &weight);

    // 计时版显存分配：timed 为 true 时用 CUDA event 测 cudaMalloc 耗时（毫秒，写入 out_ms），
    // 否则退化为普通分配、耗时返回 0。
    static void cuda_malloc_timed(void **ptr, size_t bytes, const std::string &what,
                                  bool timed, double &out_ms);

    // 计时版 H2D 拷贝：timed 为 true 时用 CUDA event 测 host->device 耗时（毫秒，写入 out_ms），
    // 否则退化为普通同步拷贝、耗时返回 0，避免非 profile 路径产生额外开销。
    static void memcpy_h2d_timed(void *dst, const void *src, size_t bytes, const std::string &what,
                                 bool timed, double &out_ms);
};

#endif // LOCAL_LLM_CUDAWEIGHTPOOL_H
