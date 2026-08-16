//
// Created by zhangyoulun on 9/8/2026.
//

#include "SwiGLUMlp.h"

#include <cstddef>
#include <stdexcept>

#include <cuda_runtime.h>

#include "llm/model/qwen/QwenWeights.h"
#include "llm/model/qwen/QwenForwardScratch.h"
#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"

SwiGLUMlp::SwiGLUMlp(const MlpWeights &weights, CudaWeightPool *pool)
    : weights_(weights), pool_(pool) {}

void SwiGLUMlp::forward(const float *d_in, float *d_out, size_t rows, int hidden_size,
                        QwenForwardScratch &scratch) {
    // gate / up：[intermediate, hidden]；down：[hidden, intermediate]。
    const int intermediate = static_cast<int>(weights_.gate.shape[0]);
    const size_t n = rows * static_cast<size_t>(intermediate);

    float *d_gate = scratch.gate_buffer.ensure(n, "mlp gate");
    float *d_up = scratch.up_buffer.ensure(n, "mlp up");
    float *d_prod = scratch.prod_buffer.ensure(n, "mlp prod");

    // 输入激活转成权重 dtype（BF16/F16）后再投影；gate/up 同 dtype，只需转一次。
    uint16_t *d_in_lowp =
        scratch.input_lowp_buffer.ensure(rows * static_cast<size_t>(hidden_size), "mlp in lowp");
    CudaWeight gate = pool_->cached_weight(weights_.gate)->try_dequant();
    GemmInput in = prepare_gemm_input(d_in, d_in_lowp, rows * static_cast<size_t>(hidden_size), gate.type, nullptr);
    gemm_weight(pool_->handle, gate, in.ptr, d_gate, intermediate, hidden_size, rows, in.type, "mlp.gate");

    CudaWeight up = pool_->cached_weight(weights_.up)->try_dequant();
    gemm_weight(pool_->handle, up, in.ptr, d_up, intermediate, hidden_size, rows, in.type, "mlp.up");

    // prod = SiLU(gate) * up。
    launch_silu_mul(d_gate, d_up, d_prod, static_cast<int>(n), /*stream=*/nullptr);

    // prod 转成 down 权重 dtype 后做 down 投影。
    uint16_t *d_prod_lowp = scratch.prod_lowp_buffer.ensure(n, "mlp prod lowp");
    CudaWeight down = pool_->cached_weight(weights_.down)->try_dequant();
    GemmInput prod_in = prepare_gemm_input(d_prod, d_prod_lowp, n, down.type, nullptr);

    // down：[hidden, intermediate] · prod[intermediate, rows] -> [hidden, rows]。
    gemm_weight(pool_->handle, down, prod_in.ptr, d_out, hidden_size, intermediate, rows, prod_in.type, "mlp.down");

    check_cuda(cudaDeviceSynchronize(), "SwiGLUMlp 同步失败");
}
