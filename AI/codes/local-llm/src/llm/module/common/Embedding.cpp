//
// Created by zhangyoulun on 16/8/2026.
//

#include "llm/module/common/Embedding.h"

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/kernel.cuh"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace common {

Embedding::Embedding(const Tensor &weight)
    : weight_(weight) {
    vocab_size_ = static_cast<int>(weight_.shape[0]);
    hidden_size_ = static_cast<int>(weight_.shape[1]);
}

void Embedding::forward(const Tensor &input, const Tensor &hidden, CudaScratch &scratch) {
    CudaWeight table = weight_.cached_weight()->try_dequant();

    const int lowp_type = (table.dtype == DType::F16) ? 1 : 0;
    const size_t input_size = static_cast<size_t>(input.numel());

    int *d_input = scratch.ensure<int>(scratch_key::kInput, input_size);
    cuda_memcpy_h2d(d_input, input.host_i32(), input_size * sizeof(int),
                    "cudaMemcpy embedding token ids 失败");

    launch_embedding_lookup(d_input, hidden.gpu_f32(), static_cast<const uint16_t *>(table.ptr),
                            static_cast<int>(input_size), vocab_size_, hidden_size_, lowp_type,
                            nullptr);
}

} // namespace common
