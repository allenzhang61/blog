//
// Created by zhangyoulun on 9/8/2026.
//

#include "SwiGLUMlp.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "llm/model/qwen/QwenWeights.h"
#include "llm/model/qwen/QwenSession.h"
#include "backend/cuda/common.h"
#include "tensor/GPUTensor.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "tensor/TensorTool.h"

void SwiGLUMlp::forward(const MlpWeights &weights, QwenSession &session,
                        const GPUTensor &g_in_f32, const GPUTensor &g_out_f32) {
    const int64_t rows = g_in_f32.rows();
    CudaScratch &scratch = session.cuda_scratch;
    // g_gate / g_up：[intermediate, hidden]；down：[hidden, intermediate]。
    const int64_t intermediate = weights.s_gate_proj.shape[0];
    const std::vector<int64_t> intermediate_shape = {rows, intermediate};
    GPUTensor g_gate_f32 = GPUTensor(scratch, scratch_key::kGate, intermediate_shape, DType::F32);
    GPUTensor g_up_f32 = GPUTensor(scratch, scratch_key::kUp, intermediate_shape, DType::F32);
    GPUTensor g_prod_f32 = GPUTensor(scratch, scratch_key::kProd, intermediate_shape, DType::F32);

    // gate/up 共享同一份 hidden 输入且权重同 dtype：只转一次 bf16/f16 复用，
    // 省掉原先每个 GEMM 各自一次 f32->bf16 拷贝（旧路径每层 2 次冗余转换）。
    const void *d_in_lowp = TensorTool::prepare_lowp_input(
        g_in_f32, weights.s_gate_proj.dtype, scratch, scratch_key::kInputLowp);
    const int64_t in_rows = g_in_f32.rows();
    TensorTool::gemm_lowp(weights.s_gate_proj, d_in_lowp, in_rows, g_gate_f32, "mlp.gate");
    TensorTool::gemm_lowp(weights.s_up_proj, d_in_lowp, in_rows, g_up_f32, "mlp.up");

    // g_prod = SiLU(g_gate) * g_up。
    TensorTool::silu_mul(g_gate_f32, g_up_f32, g_prod_f32);

    // g_prod 转成 down 权重 dtype 后做 down 投影。
    // down：[hidden, intermediate] · g_prod[intermediate, rows] -> [hidden, rows]。
    TensorTool::gemm(weights.s_down_proj, g_prod_f32, g_out_f32, scratch, scratch_key::kProdLowp, "mlp.down");

    // 不做全设备同步：同流顺序保证依赖，barrier 交给前向末尾的 lm_head 同步 D2H。
    check_cuda(cudaGetLastError(), "SwiGLUMlp kernel launch 失败");
}
