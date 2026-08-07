#pragma once

#include "CudaScratchBuffer.h"
#include "DeviceWeight.h"

#include "../../../model/weights.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace llm_inference {

// CUDA 推理过程复用的全局 cache：cuBLAS handle、权重缓存和临时 device buffer。
class CudaWeightCache {
public:
    // 全局复用的 cuBLAS handle。
    cublasHandle_t handle = nullptr;
    // items 中已缓存权重的总字节数。
    size_t bytes = 0;
    // 通用输出向量 / logits buffer。
    CudaScratchBuffer<float> y_buffer;
    // MLP gate projection 临时输出。
    CudaScratchBuffer<float> gate_buffer;
    // MLP up projection 临时输出。
    CudaScratchBuffer<float> up_buffer;
    // MLP SiLU(gate) * up 的 float 临时结果。
    CudaScratchBuffer<float> prod_buffer;
    // MLP product 转成 BF16/F16 后的低精度输入。
    CudaScratchBuffer<uint16_t> prod_lowp_buffer;
    // attention/mixer 子层输出缓存。
    CudaScratchBuffer<float> mixer_buffer;
    // MLP 子层输出缓存。
    CudaScratchBuffer<float> mlp_out_buffer;
    // 完整 transformer layer 输出缓存。
    CudaScratchBuffer<float> layer_out_buffer;
    // token hidden 双缓冲 A，用于 decode 层间传递。
    CudaScratchBuffer<float> token_hidden_a;
    // token hidden 双缓冲 B，用于 decode 层间传递。
    CudaScratchBuffer<float> token_hidden_b;
    // post-attention RMSNorm 后的低精度 hidden。
    CudaScratchBuffer<uint16_t> post_norm_lowp_buffer;
    // RMSNorm 输出的 BF16/F16 hidden。
    CudaScratchBuffer<uint16_t> norm_lowp_buffer;
    // 合并 gate/up projection 后的 float 输出。
    CudaScratchBuffer<float> gate_up_buffer;
    // argmax 每个 block 的最大值缓存。
    CudaScratchBuffer<float> argmax_block_values;
    // argmax 每个 block 的最大值索引缓存。
    CudaScratchBuffer<int> argmax_block_indices;
    // argmax 最终最大值的单元素 device buffer。
    CudaScratchBuffer<float> argmax_best_value;
    // argmax 最终 token id 的单元素 device buffer。
    CudaScratchBuffer<int> argmax_best_index;
    // token ids device staging buffer；decode 保存 generated ids，prefill 保存 prompt ids。
    CudaScratchBuffer<int> token_id_buffer;
    // 按 tensor 名称或组合名称索引的 device 权重缓存。
    std::unordered_map<std::string, DeviceWeight> items;

    // 创建 cuBLAS handle。
    CudaWeightCache();
    // 释放权重缓存和 cuBLAS handle；临时 buffer 由 CudaScratchBuffer 自动释放。
    ~CudaWeightCache();
    CudaWeightCache(const CudaWeightCache &) = delete;
    CudaWeightCache & operator=(const CudaWeightCache &) = delete;

    // 获取普通 device 权重缓存；首次访问时从 mmap host 权重上传到 GPU。
    DeviceWeight * cached_weight(const WeightData & weight);

    // 将多个 BF16 二维权重按行拼接后上传并缓存，用于合并 projection。
    DeviceWeight * cached_concat_weight(const std::string & name, const std::vector<WeightData> & weights);

private:
    // 返回本进程允许缓存的 CUDA 权重总字节数。
    static size_t cache_limit_bytes();

    // 将 safetensors dtype 映射为 CUDA/cuBLAS dtype。
    static cudaDataType_t cuda_type_for(const WeightData & weight);

    // 返回当前支持 dtype 的单元素字节数。
    static size_t dtype_size_for(const WeightData & weight);
};

// 返回进程内唯一的 CUDA cache 实例。
CudaWeightCache & cuda_weight_cache();

} // namespace llm_inference
