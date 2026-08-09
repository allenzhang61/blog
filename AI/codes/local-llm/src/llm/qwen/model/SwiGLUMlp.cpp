//
// Created by zhangyoulun on 9/8/2026.
//

#include "SwiGLUMlp.h"

#include <stdexcept>

#include <cuda_runtime.h>

#include "llm/qwen/QwenWeights.h"
#include "llm/qwen/QwenForwardScratch.h"
#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"

SwiGLUMlp::SwiGLUMlp(const MlpWeights &weights, CudaWeightPool *pool)
    : weights_(weights), pool_(pool) {}

void SwiGLUMlp::forward(const float *d_in, float *d_out, int rows, int hidden_size,
                        QwenForwardScratch &scratch) {
    CudaWeight *gate = pool_->cached_weight(weights_.gate);
    CudaWeight *up = pool_->cached_weight(weights_.up);
    CudaWeight *down = pool_->cached_weight(weights_.down);
    if (!gate || !up || !down) {
        throw std::runtime_error("SwiGLUMlp 权重上传失败");
    }

    // gate / up：[intermediate, hidden]；down：[hidden, intermediate]。
    const int intermediate = static_cast<int>(weights_.gate.info->shape[0]);
    const size_t n = static_cast<size_t>(rows) * intermediate;

    float *d_gate = scratch.gate_buffer.ensure(n, "mlp gate");
    float *d_up = scratch.up_buffer.ensure(n, "mlp up");
    float *d_prod = scratch.prod_buffer.ensure(n, "mlp prod");

    // 激活以 float 直接参与 gemm（x_type=CUDA_R_32F）。
    gemm_weight(pool_->handle, *gate, intermediate, hidden_size, d_in, CUDA_R_32F, rows, d_gate);
    gemm_weight(pool_->handle, *up, intermediate, hidden_size, d_in, CUDA_R_32F, rows, d_up);

    // prod = SiLU(gate) * up。
    launch_silu_mul(d_gate, d_up, d_prod, static_cast<int>(n), /*stream=*/nullptr);

    // down：[hidden, intermediate] · prod[intermediate, rows] -> [hidden, rows]。
    gemm_weight(pool_->handle, *down, hidden_size, intermediate, d_prod, CUDA_R_32F, rows, d_out);

    check_cuda(cudaDeviceSynchronize(), "SwiGLUMlp 同步失败");
}
