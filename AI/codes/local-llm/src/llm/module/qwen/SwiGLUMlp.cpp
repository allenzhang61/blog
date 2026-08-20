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
#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/ops/kernel.cuh"

SwiGLUMlp::SwiGLUMlp(const MlpWeights &weights)
    : weights_(weights) {}

void SwiGLUMlp::forward(QwenSession &session, const Tensor &in, const Tensor &out) {
    const size_t rows = static_cast<size_t>(in.rows());
    CudaScratch &scratch = session.scratch;
    // gate / up：[intermediate, hidden]；down：[hidden, intermediate]。
    const int intermediate = static_cast<int>(weights_.gate_proj.shape[0]);
    const size_t n = rows * static_cast<size_t>(intermediate);

    const std::vector<int64_t> intermediate_shape = {static_cast<int64_t>(rows),
                                                     static_cast<int64_t>(intermediate)};
    Tensor gate = Tensor::gpu_scratch(scratch, scratch_key::kGate, intermediate_shape);
    Tensor up = Tensor::gpu_scratch(scratch, scratch_key::kUp, intermediate_shape);
    Tensor prod = Tensor::gpu_scratch(scratch, scratch_key::kProd, intermediate_shape);

    // 输入激活转成权重 dtype（BF16/F16）后再投影；gate/up 同 dtype，只需转一次。
    weights_.gate_proj.to_gpu();
    weights_.gate_proj.gemm(in, gate, scratch, scratch_key::kInputLowp, "mlp.gate");
    weights_.up_proj.to_gpu();
    weights_.up_proj.gemm(in, up, scratch, scratch_key::kInputLowp, "mlp.up");

    // prod = SiLU(gate) * up。
    launch_silu_mul(gate.gpu_f32(), up.gpu_f32(), prod.gpu_f32(), static_cast<int>(n), /*stream=*/nullptr);

    // prod 转成 down 权重 dtype 后做 down 投影。
    // down：[hidden, intermediate] · prod[intermediate, rows] -> [hidden, rows]。
    weights_.down_proj.to_gpu();
    weights_.down_proj.gemm(prod, out, scratch, scratch_key::kProdLowp, "mlp.down");

    check_cuda(cudaDeviceSynchronize(), "SwiGLUMlp 同步失败");
}
