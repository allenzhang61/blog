//
// Created by zhangyoulun on 9/8/2026.
//

#include "SwiGLUMlp.h"

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "llm/model/qwen/QwenWeights.h"
#include "llm/model/qwen/QwenSession.h"
#include "backend/cuda/common.h"
#include "tensor/GPUTensor.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "tensor/TensorTool.h"

SwiGLUMlp::SwiGLUMlp(const MlpWeights &weights)
    : weights_(weights) {}

void SwiGLUMlp::forward(QwenSession &session, const GPUTensor &g_in, const GPUTensor &g_out) {
    const size_t rows = static_cast<size_t>(g_in.rows());
    CudaScratch &scratch = session.scratch;
    // g_gate / g_up：[intermediate, hidden]；down：[hidden, intermediate]。
    const int intermediate = static_cast<int>(weights_.s_gate_proj.shape[0]);
    const std::vector<int64_t> intermediate_shape = {static_cast<int64_t>(rows),
                                                     static_cast<int64_t>(intermediate)};
    GPUTensor g_gate = GPUTensor(scratch, scratch_key::kGate, intermediate_shape, DType::F32);
    GPUTensor g_up = GPUTensor(scratch, scratch_key::kUp, intermediate_shape, DType::F32);
    GPUTensor g_prod = GPUTensor(scratch, scratch_key::kProd, intermediate_shape, DType::F32);

    // 输入激活转成权重 dtype（BF16/F16）后再投影；g_gate/g_up 同 dtype，只需转一次。
    TensorTool::gemm(weights_.s_gate_proj, g_in, g_gate, scratch, scratch_key::kInputLowp, "mlp.gate");
    TensorTool::gemm(weights_.s_up_proj, g_in, g_up, scratch, scratch_key::kInputLowp, "mlp.up");

    // g_prod = SiLU(g_gate) * g_up。
    TensorTool::silu_mul(g_gate, g_up, g_prod);

    // g_prod 转成 down 权重 dtype 后做 down 投影。
    // down：[hidden, intermediate] · g_prod[intermediate, rows] -> [hidden, rows]。
    TensorTool::gemm(weights_.s_down_proj, g_prod, g_out, scratch, scratch_key::kProdLowp, "mlp.down");

    check_cuda(cudaDeviceSynchronize(), "SwiGLUMlp 同步失败");
}
