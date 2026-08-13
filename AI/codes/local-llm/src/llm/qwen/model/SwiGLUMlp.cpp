//
// Created by zhangyoulun on 9/8/2026.
//

#include "SwiGLUMlp.h"

#include <cstddef>
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

void SwiGLUMlp::forward(const float *d_in, float *d_out, size_t rows, int hidden_size,
                        QwenForwardScratch &scratch) {
    CudaWeight *gate = pool_->cached_weight(weights_.gate);
    CudaWeight *up = pool_->cached_weight(weights_.up);
    CudaWeight *down = pool_->cached_weight(weights_.down);
    if (!gate || !up || !down) {
        throw std::runtime_error("SwiGLUMlp 权重上传失败");
    }

    // gate / up：[intermediate, hidden]；down：[hidden, intermediate]。
    const int intermediate = static_cast<int>(weights_.gate.shape[0]);
    const size_t n = rows * static_cast<size_t>(intermediate);

    float *d_gate = scratch.gate_buffer.ensure(n, "mlp gate");
    float *d_up = scratch.up_buffer.ensure(n, "mlp up");
    float *d_prod = scratch.prod_buffer.ensure(n, "mlp prod");

    // 输入激活转成权重 dtype（BF16/F16）后再投影；gate/up 同 dtype，只需转一次。
    uint16_t *d_in_lowp =
        scratch.input_lowp_buffer.ensure(rows * static_cast<size_t>(hidden_size), "mlp in lowp");
    to_weight_lowp(d_in, d_in_lowp, rows * static_cast<size_t>(hidden_size), *gate, nullptr);

    gemm_weight(pool_->handle, *gate, intermediate, hidden_size, d_in_lowp, gate->type, rows, d_gate, "mlp.gate");
    gemm_weight(pool_->handle, *up, intermediate, hidden_size, d_in_lowp, up->type, rows, d_up, "mlp.up");

    // prod = SiLU(gate) * up。
    launch_silu_mul(d_gate, d_up, d_prod, static_cast<int>(n), /*stream=*/nullptr);

    // prod 转成 down 权重 dtype 后做 down 投影。
    uint16_t *d_prod_lowp = scratch.prod_lowp_buffer.ensure(n, "mlp prod lowp");
    to_weight_lowp(d_prod, d_prod_lowp, n, *down, nullptr);

    // down：[hidden, intermediate] · prod[intermediate, rows] -> [hidden, rows]。
    gemm_weight(pool_->handle, *down, hidden_size, intermediate, d_prod_lowp, down->type, rows, d_out, "mlp.down");

    check_cuda(cudaDeviceSynchronize(), "SwiGLUMlp 同步失败");
}
