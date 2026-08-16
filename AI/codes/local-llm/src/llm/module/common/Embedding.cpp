//
// Created by zhangyoulun on 16/8/2026.
//

#include "llm/module/common/Embedding.h"

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratchBuffer.h"
#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/kernel.cuh"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace common {

Embedding::Embedding(const MFTensorView &weight, CudaWeightPool *pool)
    : weight_(weight), pool_(pool) {
    vocab_size_ = static_cast<int>(weight_.shape[0]);
    hidden_size_ = static_cast<int>(weight_.shape[1]);
}

void Embedding::forward(const std::vector<int> &input, float *d_hidden,
                        CudaScratchBuffer<int> &input_buffer,
                        const std::string &input_buffer_name) {
    CudaWeight *resident = pool_->cached_weight(weight_);
    if (!resident) {
        throw std::runtime_error("Embedding 权重上传失败：" + weight_.name);
    }
    CudaWeight table = resident->try_dequant();

    const int lowp_type = (table.dtype == DType::F16) ? 1 : 0;
    const size_t input_size = input.size();

    int *d_input = input_buffer.ensure(input_size, input_buffer_name);
    cuda_memcpy_h2d(d_input, input.data(), input_size * sizeof(int),
                    "cudaMemcpy embedding token ids 失败");

    launch_embedding_lookup(d_input, d_hidden, static_cast<const uint16_t *>(table.ptr),
                            static_cast<int>(input_size), vocab_size_, hidden_size_, lowp_type,
                            nullptr);
}

} // namespace common
