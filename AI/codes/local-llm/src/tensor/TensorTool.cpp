//
// Created by zhangyoulun on 21/8/2026.
//

#include "tensor/TensorTool.h"

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/mem/Quant.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"
#include "llm/model/deepseek/DeepseekRuntimeOptions.h"
#include "tensor/GPUTensor.h"
#include "utils/stats/ScopedTimer.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    int norm_weight_type_of(DType dtype) {
        if (dtype == DType::BF16) return 0;
        if (dtype == DType::F16) return 1;
        if (dtype == DType::F32) return 2;
        throw std::runtime_error(std::string("RMSNorm 不支持的 norm dtype：") + dtype_name(dtype));
    }

    int embedding_weight_type_of(DType dtype) {
        if (dtype == DType::BF16) return 0;
        if (dtype == DType::F16) return 1;
        throw std::runtime_error(std::string("Embedding 不支持的 norm dtype：") + dtype_name(dtype));
    }

    cudaDataType_t cuda_type_of(DType dtype) {
        if (dtype == DType::BF16) return CUDA_R_16BF;
        if (dtype == DType::F16) return CUDA_R_16F;
        if (dtype == DType::F32) return CUDA_R_32F;
        throw std::runtime_error(std::string("gemm 不支持 dtype: ") + dtype_name(dtype));
    }

    //todo 删掉这层封装
    const uint16_t *lowp_data(const StorageTensor &s_weight) {
        GPUTensor g_device_weight = s_weight.to_gpu(true);
        return g_device_weight.data<uint16_t>();
    }

    bool is_deepseek_gemm_op(const std::string &op_name) {
        return op_name.rfind("ds.gemm.", 0) == 0;
    }

    bool should_use_safe_dequant_gemm(const char *name, int m) {
        const std::string op_name = name ? name : "";
        if (!is_deepseek_gemm_op(op_name)) return false;
        if (m == 1 && deepseek_runtime_options().quant_direct) return false;
        return true;
    }

    bool should_use_moe_fused_swiglu(const char *name) {
        const std::string op_name = name ? name : "";
        if (!is_deepseek_gemm_op(op_name)) return false;
        const DeepseekRuntimeOptions options = deepseek_runtime_options();
        return options.experimental_moe_fused_swiglu || options.quant_direct;
    }

    bool device_indexed_moe_reject(const char *reason) {
        if (deepseek_runtime_options().debug_device_indexed_moe) {
            std::cerr << "[device-indexed-moe] fallback: " << reason << std::endl;
        }
        return false;
    }

    StorageTensor expert_tensor_view(const StorageTensor &s_weight, int expert, int n_experts,
                                     std::vector<int64_t> shape) {
        const int64_t n_per_expert = s_weight.numel() / n_experts;
        const size_t bytes_per_expert = s_weight.nbytes / static_cast<size_t>(n_experts);
        return s_weight.slice(static_cast<size_t>(expert) * bytes_per_expert,
                              shape.empty() ? std::vector<int64_t>{n_per_expert} : std::move(shape),
                              bytes_per_expert,
                              s_weight.name + ".e" + std::to_string(expert));
    }

} // namespace

//w*x=y
void TensorTool::gemm(const StorageTensor &s_weight, const GPUTensor &g_input_f32, const GPUTensor &g_output_f32,
                      CudaScratch &scratch, const std::string &lowp_key, const char *name) {
    // DeepSeek 层内 decode GEMV 默认走 quant_direct_gemm；显式
    // LOCAL_LLM_DEEPSEEK_QUANT_DIRECT=0 时回退到 dequantize-to-F16 + cuBLAS。
    if (Quant::is_quantized_dtype(s_weight.dtype)) {
        const int m = static_cast<int>(g_input_f32.rows());
        if (should_use_safe_dequant_gemm(name, m)) {
            safe_dequant_gemm(s_weight, g_input_f32, g_output_f32, scratch, lowp_key, name);
            return;
        }
        quant_direct_gemm(s_weight, g_input_f32, g_output_f32, scratch, lowp_key, name);
        return;
    }

    const GPUTensor g_weight = s_weight.to_gpu(true);
    auto *d_input_lowp = scratch.ensure<uint16_t>(lowp_key, static_cast<size_t>(g_input_f32.numel()));

    const void *d_input = g_input_f32.data<float>();
    cudaDataType_t d_input_type = CUDA_R_32F;
    if (g_weight.dtype == DType::BF16 || g_weight.dtype == DType::F16) {
        if (g_weight.dtype == DType::BF16) {
            launch_f32_to_bf16_copy(g_input_f32.data<float>(), d_input_lowp,
                                    g_input_f32.numel());
        } else {
            launch_f32_to_f16_copy(g_input_f32.data<float>(), d_input_lowp,
                                   g_input_f32.numel());
        }
        d_input = d_input_lowp;
        d_input_type = (g_weight.dtype == DType::BF16) ? CUDA_R_16BF : CUDA_R_16F;
    } else if (g_weight.dtype != DType::F32) {
        throw std::runtime_error(std::string("gemm 不支持 dtype: ") + dtype_name(g_weight.dtype));
    }

    gemm_main(global_cuda_weight_pool().handle, g_weight.data(), d_input, g_output_f32.data<float>(),
              static_cast<int>(s_weight.shape[0]), static_cast<int>(s_weight.shape[1]),
              static_cast<size_t>(g_input_f32.rows()),
              cuda_type_of(g_weight.dtype), d_input_type, g_weight.nbytes,
              name);
}

void TensorTool::safe_dequant_gemm(const StorageTensor &s_weight, const GPUTensor &g_input_f32,
                                   const GPUTensor &g_output_f32, CudaScratch &scratch,
                                   const std::string &lowp_key, const char *name) {
    const int out_dim = static_cast<int>(s_weight.shape[0]);
    const int in_dim = static_cast<int>(s_weight.shape[1]);
    const int m = static_cast<int>(g_input_f32.rows());
    const CudaWeight *q = global_cuda_weight_pool().cached_weight(s_weight);
    if (q == nullptr) {
        throw std::runtime_error("TensorTool::safe_dequant_gemm 量化权重超过 CudaWeightPool 上限: " + s_weight.name);
    }
    CudaWeight dequant = q->try_dequant();
    auto *d_input_f16 = scratch.ensure<uint16_t>(lowp_key + ".safe.f16",
                                                static_cast<size_t>(g_input_f32.numel()));
    launch_f32_to_f16_copy(g_input_f32.data<float>(), d_input_f16, g_input_f32.numel());
    gemm_main(global_cuda_weight_pool().handle, dequant.ptr, d_input_f16, g_output_f32.data<float>(),
              out_dim, in_dim, static_cast<size_t>(m), CUDA_R_16F, CUDA_R_16F,
              dequant.bytes, name);
}

void TensorTool::quant_direct_gemm(const StorageTensor &s_weight, const GPUTensor &g_input_f32,
                                   const GPUTensor &g_output_f32, CudaScratch &scratch,
                                   const std::string &lowp_key, const char *name) {
    const int out_dim = static_cast<int>(s_weight.shape[0]);
    const int in_dim = static_cast<int>(s_weight.shape[1]);
    const int m = static_cast<int>(g_input_f32.rows());
    const CudaWeight *q = global_cuda_weight_pool().cached_weight(s_weight);
    if (q == nullptr) {
        throw std::runtime_error("TensorTool::quant_direct_gemm 量化权重超过 CudaWeightPool 上限: " + s_weight.name);
    }
    const size_t row_bytes = q->bytes / static_cast<size_t>(out_dim);
    ScopedGpuTimer timer(name && name[0] ? name : nullptr, q->bytes);

    // Correctness-first quant-direct: keep activations in F32 and only decode
    // quantized weights on the fly. Q8_1 activation/MMQ variants stay outside
    // this generic GEMM path until their numerical drift is isolated.
    (void) scratch;
    (void) lowp_key;
    if (m > 1) {
        launch_quant_matmul(s_weight.dtype, static_cast<const uint8_t *>(q->ptr),
                            row_bytes, g_input_f32.data<float>(), g_output_f32.data<float>(),
                            out_dim, in_dim, m);
        return;
    }
    launch_quant_gemv(s_weight.dtype, static_cast<const uint8_t *>(q->ptr),
                      row_bytes, g_input_f32.data<float>(), g_output_f32.data<float>(),
                      out_dim, in_dim, m);
}

const void *TensorTool::prepare_lowp_input(const GPUTensor &g_input_f32, DType weight_dtype,
                                           CudaScratch &scratch, const std::string &lowp_key) {
    // 权重是 f32 时无需转换，GEMM 直接吃 f32 输入。
    if (weight_dtype == DType::F32) {
        return g_input_f32.data<float>();
    }
    uint16_t *d_input_lowp = scratch.ensure<uint16_t>(lowp_key, static_cast<size_t>(g_input_f32.numel()));
    if (weight_dtype == DType::BF16) {
        launch_f32_to_bf16_copy(g_input_f32.data<float>(), d_input_lowp, g_input_f32.numel());
    } else if (weight_dtype == DType::F16) {
        launch_f32_to_f16_copy(g_input_f32.data<float>(), d_input_lowp, g_input_f32.numel());
    } else {
        throw std::runtime_error(std::string("prepare_lowp_input 不支持 dtype: ") + dtype_name(weight_dtype));
    }
    return d_input_lowp;
}

void TensorTool::gemm_lowp(const StorageTensor &s_weight, const void *d_input_lowp, int64_t input_rows,
                           const GPUTensor &g_output_f32, const char *name) {
    GPUTensor g_weight = s_weight.to_gpu(true);
    cudaDataType_t d_input_type;
    if (g_weight.dtype == DType::BF16) {
        d_input_type = CUDA_R_16BF;
    } else if (g_weight.dtype == DType::F16) {
        d_input_type = CUDA_R_16F;
    } else if (g_weight.dtype == DType::F32) {
        d_input_type = CUDA_R_32F;
    } else {
        throw std::runtime_error(std::string("gemm_lowp 不支持 dtype: ") + dtype_name(g_weight.dtype));
    }
    gemm_main(global_cuda_weight_pool().handle, g_weight.data(), d_input_lowp, g_output_f32.data<float>(),
              static_cast<int>(s_weight.shape[0]), static_cast<int>(s_weight.shape[1]),
              static_cast<size_t>(input_rows),
              cuda_type_of(g_weight.dtype), d_input_type, g_weight.nbytes,
              name);
}

bool TensorTool::can_quant_swiglu(const StorageTensor &s_gate_weight, const StorageTensor &s_up_weight,
                                  const GPUTensor &g_input_f32, const GPUTensor &g_act_f32,
                                  const char *name) {
    if (!should_use_moe_fused_swiglu(name)) return false;
    if (g_input_f32.rows() != 1 || g_input_f32.dtype != DType::F32 || g_act_f32.dtype != DType::F32) return false;
    if (!Quant::is_quantized_dtype(s_gate_weight.dtype) || s_gate_weight.dtype != s_up_weight.dtype) return false;
    if (s_gate_weight.shape.size() != 2 || s_up_weight.shape.size() != 2) return false;
    if (s_gate_weight.shape != s_up_weight.shape) return false;
    const int ffn_dim = static_cast<int>(s_gate_weight.shape[0]);
    const int in_dim = static_cast<int>(s_gate_weight.shape[1]);
    if (g_input_f32.cols() != in_dim || g_act_f32.rows() != 1 || g_act_f32.cols() != ffn_dim) return false;
    return true;
}

void TensorTool::quant_swiglu(const StorageTensor &s_gate_weight, const StorageTensor &s_up_weight,
                              const GPUTensor &g_input_f32, const GPUTensor &g_act_f32,
                              const char *name) {
    const int ffn_dim = static_cast<int>(s_gate_weight.shape[0]);
    const int in_dim = static_cast<int>(s_gate_weight.shape[1]);
    const CudaWeight *gate = global_cuda_weight_pool().cached_weight(s_gate_weight);
    const CudaWeight *up = global_cuda_weight_pool().cached_weight(s_up_weight);
    if (gate == nullptr || up == nullptr) {
        throw std::runtime_error("TensorTool::quant_swiglu 量化权重超过 CudaWeightPool 上限");
    }
    const size_t gate_row_bytes = gate->bytes / static_cast<size_t>(ffn_dim);
    const size_t up_row_bytes = up->bytes / static_cast<size_t>(ffn_dim);
    ScopedGpuTimer timer(name && name[0] ? name : nullptr, gate->bytes + up->bytes);
    launch_quant_swiglu(s_gate_weight.dtype,
                        static_cast<const uint8_t *>(gate->ptr),
                        static_cast<const uint8_t *>(up->ptr),
                        gate_row_bytes, up_row_bytes,
                        g_input_f32.data<float>(), g_act_f32.data<float>(),
                        ffn_dim, in_dim);
}

bool TensorTool::quant_gemv_add(const StorageTensor &s_weight, const GPUTensor &g_input_f32,
                                const GPUTensor &g_output_f32, const char *name) {
    if (!deepseek_runtime_options().quant_direct) return false;
    if (g_input_f32.rows() != 1 || g_input_f32.dtype != DType::F32 || g_output_f32.rows() != 1 ||
        g_output_f32.dtype != DType::F32) {
        return false;
    }
    if (!Quant::is_quantized_dtype(s_weight.dtype) || s_weight.shape.size() != 2) return false;
    const int out_dim = static_cast<int>(s_weight.shape[0]);
    const int in_dim = static_cast<int>(s_weight.shape[1]);
    if (g_input_f32.cols() != in_dim || g_output_f32.cols() != out_dim) return false;

    const CudaWeight *q = global_cuda_weight_pool().cached_weight(s_weight);
    if (q == nullptr) {
        throw std::runtime_error("TensorTool::quant_gemv_add 量化权重超过 CudaWeightPool 上限: " + s_weight.name);
    }
    const size_t row_bytes = q->bytes / static_cast<size_t>(out_dim);
    ScopedGpuTimer timer(name && name[0] ? name : nullptr, q->bytes);
    launch_quant_gemv_add(s_weight.dtype, static_cast<const uint8_t *>(q->ptr), row_bytes,
                          g_input_f32.data<float>(), g_output_f32.data<float>(),
                          out_dim, in_dim);
    return true;
}

bool TensorTool::moe_routed_decode_indexed(const StorageTensor &s_gate_exps, const StorageTensor &s_up_exps,
                                           const StorageTensor &s_down_exps, const GPUTensor &g_input_f32,
                                           const int *d_expert_ids, const float *d_weights,
                                           const GPUTensor &g_out_f32, CudaScratch &scratch,
                                           int n_experts, int k, const char *name) {
    /*参数校验*/
    if (g_input_f32.rows() != 1 || g_input_f32.dtype != DType::F32 || g_out_f32.dtype != DType::F32) {
        return device_indexed_moe_reject("not decode F32 input/output");
    }
    if (d_expert_ids == nullptr || d_weights == nullptr) return device_indexed_moe_reject("null route device pointers");
    if (!Quant::is_quantized_dtype(s_gate_exps.dtype) || s_gate_exps.dtype != s_up_exps.dtype) {
        return device_indexed_moe_reject("gate/up dtype mismatch or not quantized");
    }
    if (s_down_exps.dtype != DType::Q4_K && s_down_exps.dtype != DType::Q6_K &&
        s_down_exps.dtype != DType::Q5_0 && s_down_exps.dtype != DType::Q8_0) {
        return device_indexed_moe_reject((std::string("down dtype unsupported: ") +
                                          dtype_name(s_down_exps.dtype)).c_str());
    }
    if (s_gate_exps.dtype != DType::Q4_K && s_gate_exps.dtype != DType::Q6_K) {
        return device_indexed_moe_reject("gate/up dtype unsupported");
    }
    if (s_gate_exps.shape.size() != 3 || s_up_exps.shape.size() != 3 || s_down_exps.shape.size() != 3) {
        return device_indexed_moe_reject("expert tensor rank is not 3");
    }
    if (s_gate_exps.shape[0] != n_experts || s_up_exps.shape[0] != n_experts || s_down_exps.shape[0] != n_experts) {
        return device_indexed_moe_reject("expert tensor n_experts mismatch");
    }
    if (s_gate_exps.shape != s_up_exps.shape) return device_indexed_moe_reject("gate/up shape mismatch");

    const int ffn_dim = static_cast<int>(s_gate_exps.shape[1]);
    const int in_dim = static_cast<int>(s_gate_exps.shape[2]);
    const int hidden_size = static_cast<int>(s_down_exps.shape[1]);
    if (s_down_exps.shape[2] != ffn_dim || g_input_f32.cols() != in_dim || g_out_f32.cols() != hidden_size) {
        return device_indexed_moe_reject("expert tensor dimensions mismatch");
    }

    const CudaWeight *gate = global_cuda_weight_pool().cached_weight(s_gate_exps);
    const CudaWeight *up = global_cuda_weight_pool().cached_weight(s_up_exps);
    const CudaWeight *down = global_cuda_weight_pool().cached_weight(s_down_exps);
    if (gate == nullptr || up == nullptr || down == nullptr) {
        throw std::runtime_error("TensorTool::moe_routed_decode_indexed 量化专家权重超过 CudaWeightPool 上限");
    }
    const size_t gate_expert_bytes = gate->bytes / static_cast<size_t>(n_experts);
    const size_t up_expert_bytes = up->bytes / static_cast<size_t>(n_experts);
    const size_t down_expert_bytes = down->bytes / static_cast<size_t>(n_experts);
    const size_t gate_row_bytes = gate_expert_bytes / static_cast<size_t>(ffn_dim);
    const size_t up_row_bytes = up_expert_bytes / static_cast<size_t>(ffn_dim);
    const size_t down_row_bytes = down_expert_bytes / static_cast<size_t>(hidden_size);

    ScopedGpuTimer timer(name && name[0] ? name : "ds.gemm.e_indexed_moe",
                         gate->bytes + up->bytes + down->bytes);
    auto g_act_f32 = GPUTensor(scratch, scratch_key::kAct, {k, ffn_dim}, DType::F32);
    launch_quant_swiglu_indexed(s_gate_exps.dtype,
                                static_cast<const uint8_t *>(gate->ptr),
                                static_cast<const uint8_t *>(up->ptr),
                                gate_expert_bytes, up_expert_bytes,
                                gate_row_bytes, up_row_bytes,
                                g_input_f32.data<float>(), d_expert_ids,
                                g_act_f32.data<float>(), k, ffn_dim, in_dim);
    launch_quant_down_f32_indexed_accum_ordered(s_down_exps.dtype,
                                                static_cast<const uint8_t *>(down->ptr),
                                                down_expert_bytes, down_row_bytes,
                                                g_act_f32.data<float>(), d_expert_ids, d_weights,
                                                g_out_f32.data<float>(), k, hidden_size, ffn_dim);
    return true;
}

void TensorTool::embedding_lookup(const StorageTensor &s_table, const GPUTensor &g_input_i32,
                                  const GPUTensor &g_hidden_f32) {
    // 量化表（如 DeepSeek 的 Q4_K token_embd）直接走量化直算查表：量化常驻、按行反量化，
    // 不把整张 [vocab,hidden] 表展开成 F16（省近 0.4GB 显存 + 消除大 F16 lease）。
    if (Quant::is_quantized_dtype(s_table.dtype)) {
        const GPUTensor g_table_q = s_table.to_gpu(false);
        const int vocab = static_cast<int>(s_table.shape[0]);
        const size_t row_bytes = g_table_q.nbytes / static_cast<size_t>(vocab);
        launch_quant_embedding(s_table.dtype, g_input_i32.data<int>(),
                               g_hidden_f32.data<float>(), static_cast<const uint8_t *>(g_table_q.data()),
                               row_bytes, static_cast<int>(s_table.shape[1]),
                               static_cast<int>(g_input_i32.numel()));
        return;
    }
    GPUTensor g_table_u16 = s_table.to_gpu(true);
    launch_embedding_lookup(g_input_i32.data<int>(), g_hidden_f32.data<float>(),
                            g_table_u16.data<uint16_t>(),
                            static_cast<int>(g_input_i32.numel()), static_cast<int>(s_table.shape[0]),
                            static_cast<int>(s_table.shape[1]), embedding_weight_type_of(g_table_u16.dtype));
}

void TensorTool::rms_norm(const StorageTensor &s_weight_u16, const GPUTensor &g_input_f32,
                          const GPUTensor &g_output_f32,
                          float eps, bool one_plus) {
    GPUTensor g_weight_u16 = s_weight_u16.to_gpu(true);
    const int f16_or_bf16 = norm_weight_type_of(g_weight_u16.dtype);
    // norm 权重可能为 f32（DeepSeek GGUF），此时按字节取指针，kernel 内按 weight_type 重解释。
    launch_rms_norm(g_input_f32.data<float>(), g_output_f32.data<float>(),
                    static_cast<const uint16_t *>(g_weight_u16.data()),
                    f16_or_bf16, static_cast<int>(g_input_f32.rows()),
                    static_cast<int>(g_input_f32.cols()), eps, one_plus);
}

void TensorTool::add_rms_norm(const StorageTensor &s_weight, const GPUTensor &g_x_f32,
                              const GPUTensor &g_residual_f32, const GPUTensor &g_residual_io_f32,
                              const GPUTensor &g_norm_out_f32, float eps, bool one_plus, void *stream) {
    const GPUTensor g_weight_u16 = s_weight.to_gpu(true);
    const int f16_or_bf16 = norm_weight_type_of(g_weight_u16.dtype);
    launch_add_rms_norm(g_x_f32.data<float>(), g_residual_f32.data<float>(), g_residual_io_f32.data<float>(),
                        g_norm_out_f32.data<float>(), static_cast<const uint16_t *>(g_weight_u16.data()),
                        f16_or_bf16, static_cast<int>(g_x_f32.rows()), static_cast<int>(g_x_f32.cols()),
                        eps, one_plus);
}

void TensorTool::add(const GPUTensor &g_a_f32, const GPUTensor &g_b_f32, const GPUTensor &g_out_f32, void *stream) {
    launch_add(g_a_f32.data<float>(), g_b_f32.data<float>(), g_out_f32.data<float>(),
               static_cast<int>(g_out_f32.numel()));
}

void TensorTool::silu_mul(const GPUTensor &g_gate_f32, const GPUTensor &g_up_f32, const GPUTensor &g_out_f32,
                          void *stream) {
    launch_silu_mul(g_gate_f32.data<float>(), g_up_f32.data<float>(), g_out_f32.data<float>(),
                    static_cast<int>(g_out_f32.numel()));
}

void TensorTool::full_attention_q(const GPUTensor &g_q_and_gate_f32, const StorageTensor &s_q_norm_weight,
                                  const GPUTensor &g_q_f32, const GPUTensor &g_gate_f32, int n_heads, int head_dim,
                                  const int *pos_dev,
                                  float rope_theta, float partial_rotary_factor, float eps,
                                  void *stream) {
    launch_full_attention_q(g_q_and_gate_f32.data<float>(), lowp_data(s_q_norm_weight), g_q_f32.data<float>(),
                            g_gate_f32.data<float>(), n_heads, head_dim, pos_dev,
                            rope_theta, partial_rotary_factor, eps);
}

void TensorTool::full_attention_q_batch(const GPUTensor &g_q_and_gate_f32, const StorageTensor &s_q_norm_weight_f32,
                                        const GPUTensor &g_q_f32, const GPUTensor &g_gate_f32, int n_heads,
                                        int head_dim, int start_pos, float rope_theta,
                                        float partial_rotary_factor, float eps, void *stream) {
    launch_full_attention_q_batch(g_q_and_gate_f32.data<float>(), lowp_data(s_q_norm_weight_f32), g_q_f32.data<float>(),
                                  g_gate_f32.data<float>(),
                                  static_cast<int>(g_q_and_gate_f32.rows()), n_heads,
                                  head_dim, start_pos, rope_theta, partial_rotary_factor, eps);
}

void TensorTool::full_attention_kv(const GPUTensor &g_k_in_f32, const GPUTensor &g_v_in_f32,
                                   const StorageTensor &s_k_norm_weight, const GPUTensor &g_key_cache_f32,
                                   const GPUTensor &g_value_cache_f32, int kv_heads, int head_dim,
                                   int max_seq_len, const int *pos_dev, float rope_theta,
                                   float partial_rotary_factor, float eps, void *stream) {
    const bool kv_bf16 = g_key_cache_f32.dtype == DType::BF16;
    launch_full_attention_kv(g_k_in_f32.data<float>(), g_v_in_f32.data<float>(), lowp_data(s_k_norm_weight),
                             g_key_cache_f32.data(), g_value_cache_f32.data(), kv_bf16, kv_heads,
                             head_dim, max_seq_len, pos_dev, rope_theta, partial_rotary_factor,
                             eps);
}

void TensorTool::full_attention_kv_batch(const GPUTensor &g_k_in_f32, const GPUTensor &g_v_in_f32,
                                         const StorageTensor &s_k_norm_weight, const GPUTensor &g_key_cache_f32,
                                         const GPUTensor &g_value_cache_f32, int kv_heads,
                                         int head_dim, int max_seq_len, int start_pos,
                                         float rope_theta, float partial_rotary_factor,
                                         float eps, void *stream) {
    const bool kv_bf16 = g_key_cache_f32.dtype == DType::BF16;
    launch_full_attention_kv_batch(g_k_in_f32.data<float>(), g_v_in_f32.data<float>(), lowp_data(s_k_norm_weight),
                                   g_key_cache_f32.data(), g_value_cache_f32.data(), kv_bf16,
                                   static_cast<int>(g_k_in_f32.rows()), kv_heads, head_dim, max_seq_len, start_pos,
                                   rope_theta, partial_rotary_factor, eps);
}

void TensorTool::full_attention_attend(const GPUTensor &g_q_f32, const GPUTensor &g_gate_f32,
                                       const GPUTensor &g_key_cache_f32, const GPUTensor &g_value_cache_f32,
                                       const GPUTensor &g_attn_f32, int n_heads, int kv_heads, int head_dim,
                                       int max_seq_len, const int *pos_dev, void *stream) {
    const bool kv_bf16 = g_key_cache_f32.dtype == DType::BF16;
    launch_full_attention_attend(g_q_f32.data<float>(), g_gate_f32.data<float>(), g_key_cache_f32.data(),
                                 g_value_cache_f32.data(), kv_bf16, g_attn_f32.data<float>(), n_heads, kv_heads,
                                 head_dim, max_seq_len, pos_dev);
}

void TensorTool::full_attention_attend_batch(const GPUTensor &g_q_f32, const GPUTensor &g_gate_f32,
                                             const GPUTensor &g_key_cache_f32, const GPUTensor &g_value_cache_f32,
                                             const GPUTensor &g_attn_f32, int n_heads, int kv_heads,
                                             int head_dim, int max_seq_len, int start_pos,
                                             void *stream) {
    const bool kv_bf16 = g_key_cache_f32.dtype == DType::BF16;
    launch_full_attention_attend_batch(g_q_f32.data<float>(), g_gate_f32.data<float>(), g_key_cache_f32.data(),
                                       g_value_cache_f32.data(), kv_bf16, g_attn_f32.data<float>(),
                                       static_cast<int>(g_q_f32.rows()),
                                       n_heads, kv_heads, head_dim, max_seq_len, start_pos);
}

void TensorTool::linear_attention_conv(const GPUTensor &g_mixed_f32, const StorageTensor &s_conv_weight,
                                       const GPUTensor &g_conv_state_f32, const GPUTensor &g_conv_out_f32,
                                       int kernel, void *stream) {
    launch_linear_attention_conv(g_mixed_f32.data<float>(), lowp_data(s_conv_weight), g_conv_state_f32.data<float>(),
                                 g_conv_out_f32.data<float>(), static_cast<int>(g_mixed_f32.cols()),
                                 kernel);
}

void TensorTool::linear_attention_conv_batch(const GPUTensor &g_mixed_f32, const StorageTensor &s_conv_weight,
                                             const GPUTensor &g_conv_state_f32, const GPUTensor &g_conv_out_f32,
                                             int kernel, void *stream) {
    launch_linear_attention_conv_batch(g_mixed_f32.data<float>(), lowp_data(s_conv_weight),
                                       g_conv_state_f32.data<float>(), g_conv_out_f32.data<float>(),
                                       static_cast<int>(g_mixed_f32.rows()), static_cast<int>(g_mixed_f32.cols()),
                                       kernel);
}

void TensorTool::linear_attention_recurrent(const GPUTensor &g_conv_out_f32, const GPUTensor &g_z_f32,
                                            const GPUTensor &g_b_f32, const GPUTensor &g_a_f32,
                                            const StorageTensor &s_a_log, const StorageTensor &s_dt_bias,
                                            const StorageTensor &s_norm_weight, const GPUTensor &g_recurrent_state,
                                            const GPUTensor &g_gated_f32, int key_heads, int value_heads,
                                            int k_dim, int v_dim, float eps, void *stream) {
    GPUTensor g_a_log_f32 = s_a_log.to_gpu(true);
    GPUTensor g_norm_weight_f32 = s_norm_weight.to_gpu(true);
    const bool state_bf16 = g_recurrent_state.dtype == DType::BF16;
    launch_linear_attention_recurrent(g_conv_out_f32.data<float>(), g_z_f32.data<float>(), g_b_f32.data<float>(),
                                      g_a_f32.data<float>(),
                                      g_a_log_f32.data<float>(), lowp_data(s_dt_bias), g_norm_weight_f32.data<float>(),
                                      g_recurrent_state.data(), state_bf16, g_gated_f32.data<float>(), key_heads,
                                      value_heads,
                                      k_dim, v_dim, eps);
}

void TensorTool::linear_attention_recurrent_batch(const GPUTensor &g_conv_out_f32, const GPUTensor &g_z_f32,
                                                  const GPUTensor &g_b_f32, const GPUTensor &g_a_f32,
                                                  const StorageTensor &s_a_log, const StorageTensor &s_dt_bias,
                                                  const StorageTensor &s_norm_weight,
                                                  const GPUTensor &g_recurrent_state_f32, const GPUTensor &g_gated_f32,
                                                  int key_heads, int value_heads,
                                                  int k_dim, int v_dim, float eps,
                                                  void *stream) {
    GPUTensor g_a_log_f32 = s_a_log.to_gpu(true);
    GPUTensor g_norm_weight_f32 = s_norm_weight.to_gpu(true);
    launch_linear_attention_recurrent_batch(g_conv_out_f32.data<float>(), g_z_f32.data<float>(), g_b_f32.data<float>(),
                                            g_a_f32.data<float>(),
                                            g_a_log_f32.data<float>(), lowp_data(s_dt_bias),
                                            g_norm_weight_f32.data<float>(),
                                            g_recurrent_state_f32.data<float>(), g_gated_f32.data<float>(),
                                            static_cast<int>(g_conv_out_f32.rows()), key_heads,
                                            value_heads, k_dim, v_dim, eps);
}

void TensorTool::linear_attention_fused(const GPUTensor &g_mixed_f32, const StorageTensor &s_conv_weight,
                                        const GPUTensor &g_conv_state_f32, const GPUTensor &g_z_f32,
                                        const GPUTensor &g_b_f32, const GPUTensor &g_a_f32,
                                        const StorageTensor &s_a_log, const StorageTensor &s_dt_bias,
                                        const StorageTensor &s_norm_weight, const GPUTensor &g_recurrent_state,
                                        const GPUTensor &g_gated_f32, int key_heads, int value_heads,
                                        int k_dim, int v_dim, int kernel, float eps, void *stream) {
    GPUTensor g_a_log_f32 = s_a_log.to_gpu(true);
    GPUTensor g_norm_weight_f32 = s_norm_weight.to_gpu(true);
    const bool state_bf16 = g_recurrent_state.dtype == DType::BF16;
    launch_linear_attention_fused(g_mixed_f32.data<float>(), lowp_data(s_conv_weight), g_conv_state_f32.data<float>(),
                                  g_z_f32.data<float>(), g_b_f32.data<float>(), g_a_f32.data<float>(),
                                  g_a_log_f32.data<float>(), lowp_data(s_dt_bias), g_norm_weight_f32.data<float>(),
                                  g_recurrent_state.data(), state_bf16, g_gated_f32.data<float>(),
                                  key_heads, value_heads, k_dim, v_dim, kernel, eps);
}

void TensorTool::linear_attention_fused_batch(const GPUTensor &g_mixed_f32, const StorageTensor &s_conv_weight,
                                              const GPUTensor &g_conv_state_f32, const GPUTensor &g_z_f32,
                                              const GPUTensor &g_b_f32, const GPUTensor &g_a_f32,
                                              const StorageTensor &s_a_log, const StorageTensor &s_dt_bias,
                                              const StorageTensor &s_norm_weight, const GPUTensor &g_recurrent_state,
                                              const GPUTensor &g_gated_f32, int key_heads, int value_heads,
                                              int k_dim, int v_dim, int kernel, float eps, void *stream) {
    GPUTensor g_a_log_f32 = s_a_log.to_gpu(true);
    GPUTensor g_norm_weight_f32 = s_norm_weight.to_gpu(true);
    const bool state_bf16 = g_recurrent_state.dtype == DType::BF16;
    launch_linear_attention_fused_batch(g_mixed_f32.data<float>(), lowp_data(s_conv_weight),
                                        g_conv_state_f32.data<float>(), g_z_f32.data<float>(),
                                        g_b_f32.data<float>(), g_a_f32.data<float>(),
                                        g_a_log_f32.data<float>(), lowp_data(s_dt_bias),
                                        g_norm_weight_f32.data<float>(), g_recurrent_state.data(), state_bf16,
                                        g_gated_f32.data<float>(), static_cast<int>(g_mixed_f32.rows()),
                                        key_heads, value_heads, k_dim, v_dim, kernel, eps);
}

void TensorTool::mla_kv_a(const GPUTensor &g_kv_a_f32, const StorageTensor &s_kv_a_norm_weight,
                          const GPUTensor &g_kv_cache_f32, //这个是 output
                          int64_t input_size, int kv_lora, int qk_rope,
                          int start_pos, const GPUTensor &g_inv_freq_f32, float eps, void *stream) {
    GPUTensor g_kv_a_norm_weight_f32 = s_kv_a_norm_weight.to_gpu(true);
    launch_mla_kv_a(g_kv_a_f32.data<float>(), g_kv_a_norm_weight_f32.data<float>(),
                    g_kv_cache_f32.data<float>(), static_cast<int>(input_size), kv_lora, qk_rope,
                    start_pos, g_inv_freq_f32.data<float>(), eps);
}

void TensorTool::mla_kv_a_device_pos(const GPUTensor &g_kv_a_f32, const StorageTensor &s_kv_a_norm_weight,
                                     const GPUTensor &g_kv_cache_f32, int64_t input_size, int kv_lora, int qk_rope,
                                     const int *d_pos, const GPUTensor &g_inv_freq_f32, float eps, void *stream) {
    GPUTensor g_kv_a_norm_weight_f32 = s_kv_a_norm_weight.to_gpu(true);
    launch_mla_kv_a_device_pos(g_kv_a_f32.data<float>(), g_kv_a_norm_weight_f32.data<float>(),
                               g_kv_cache_f32.data<float>(), static_cast<int>(input_size), kv_lora, qk_rope,
                               d_pos, g_inv_freq_f32.data<float>(), eps);
}

void TensorTool::mla_rope_q(const GPUTensor &g_q_f32, int64_t input_size, int n_heads, int qk_nope,
                            int qk_rope, int start_pos, const GPUTensor &g_inv_freq_f32,
                            void *stream) {
    launch_mla_rope_q(g_q_f32.data<float>(), static_cast<int>(input_size), n_heads, qk_nope,
                      qk_rope, start_pos, g_inv_freq_f32.data<float>());
}

void TensorTool::mla_rope_q_device_pos(const GPUTensor &g_q_f32, int64_t input_size, int n_heads, int qk_nope,
                                       int qk_rope, const int *d_pos, const GPUTensor &g_inv_freq_f32,
                                       void *stream) {
    launch_mla_rope_q_device_pos(g_q_f32.data<float>(), static_cast<int>(input_size), n_heads, qk_nope,
                                 qk_rope, d_pos, g_inv_freq_f32.data<float>());
}

void TensorTool::mla_attend(const GPUTensor &g_q_f32, const GPUTensor &g_kv_b_out_f32,
                            const GPUTensor &g_kv_cache_f32, const GPUTensor &g_attn_f32,
                            int64_t input_size, int n_heads, int qk_nope, int qk_rope,
                            int v_head, int kv_lora, int start_pos, float softmax_scale, void *stream) {
    launch_mla_attend_batch(g_q_f32.data<float>(), g_kv_b_out_f32.data<float>(),
                            g_kv_cache_f32.data<float>(), g_attn_f32.data<float>(),
                            static_cast<int>(input_size), n_heads, qk_nope,
                            qk_rope, v_head, kv_lora, start_pos,
                            softmax_scale);
}

void TensorTool::mla_attend_device_pos(const GPUTensor &g_q_f32, const GPUTensor &g_kv_b_out_f32,
                                       const GPUTensor &g_kv_cache_f32, const GPUTensor &g_attn_f32,
                                       int64_t input_size, int n_heads, int qk_nope, int qk_rope,
                                       int v_head, int kv_lora, const int *d_pos, int max_seq_len,
                                       float softmax_scale, void *stream) {
    launch_mla_attend_batch_device_pos(g_q_f32.data<float>(), g_kv_b_out_f32.data<float>(),
                                       g_kv_cache_f32.data<float>(), g_attn_f32.data<float>(),
                                       static_cast<int>(input_size), n_heads, qk_nope, qk_rope,
                                       v_head, kv_lora, d_pos, max_seq_len, softmax_scale);
}

void TensorTool::mla_gather_latent_device_pos(const GPUTensor &g_kv_cache_f32, const GPUTensor &g_latent_f32,
                                              int kv_lora, int qk_rope, const int *d_pos, void *stream) {
    launch_mla_gather_latent_device_pos(g_kv_cache_f32.data<float>(), g_latent_f32.data<float>(),
                                        kv_lora, qk_rope, d_pos);
}

void TensorTool::mla_store_kv_b_device_pos(const GPUTensor &g_kv_b_new_f32, const GPUTensor &g_kv_b_cache_f32,
                                           int kvb_out, const int *d_pos, void *stream) {
    launch_mla_store_kv_b_device_pos(g_kv_b_new_f32.data<float>(), g_kv_b_cache_f32.data<float>(),
                                     kvb_out, d_pos);
}

void TensorTool::mla_store_latent_q8_1(const GPUTensor &g_latent_f32, uint8_t *latent_q8_1_cache,
                                       int kv_lora, size_t row_bytes, int start_pos, bool store_raw_sum,
                                       void *stream) {
    launch_quantize_q8_1(g_latent_f32.data<float>(),
                         latent_q8_1_cache + static_cast<size_t>(start_pos) * row_bytes,
                         kv_lora, static_cast<int>(g_latent_f32.rows()), store_raw_sum);
}

void TensorTool::mla_store_latent_q8_1_device_pos(const GPUTensor &g_kv_cache_f32, uint8_t *latent_q8_1_cache,
                                                  int kv_lora, int qk_rope, size_t row_bytes, const int *d_pos) {
    launch_mla_store_latent_q8_1_device_pos(g_kv_cache_f32.data<float>(), latent_q8_1_cache,
                                            kv_lora, qk_rope, row_bytes, d_pos);
}

bool TensorTool::mla_absorb_components(const StorageTensor &s_kv_b_weight, const GPUTensor &g_q_f32,
                                       const uint8_t *latent_q8_1_cache, size_t latent_q8_1_row_bytes,
                                       const GPUTensor &g_kv_cache_f32, const GPUTensor &g_q_abs_f32,
                                       const GPUTensor &g_q_abs_xsum_delta_f32,
                                       const GPUTensor &g_attn_latent_f32, const GPUTensor &g_attn_scores_f32,
                                       const GPUTensor &g_attn_f32,
                                       const GPUTensor &g_attn_xsum_delta_f32,
                                       int n_heads, int qk_nope, int qk_rope, int v_head, int kv_lora,
                                       const int *d_pos, int max_seq_len, float softmax_scale) {
    if (latent_q8_1_cache == nullptr || latent_q8_1_row_bytes < q8_1_row_bytes(kv_lora)) return false;
    if (!Quant::is_quantized_dtype(s_kv_b_weight.dtype) || s_kv_b_weight.shape.size() != 2) {
        if (deepseek_runtime_options().debug_mla_absorb_compare) {
            std::cerr << "[mla_absorb_compare] kv_b dtype/rank mismatch dtype="
                      << dtype_name(s_kv_b_weight.dtype) << " rank=" << s_kv_b_weight.shape.size() << "\n";
        }
        return false;
    }
    const int kvb_out = n_heads * (qk_nope + v_head);
    if (s_kv_b_weight.shape[0] != kvb_out || s_kv_b_weight.shape[1] != kv_lora) {
        if (deepseek_runtime_options().debug_mla_absorb_compare) {
            std::cerr << "[mla_absorb_compare] kv_b shape mismatch: shape0=" << s_kv_b_weight.shape[0]
                      << " shape1=" << s_kv_b_weight.shape[1]
                      << " expected=" << kvb_out << "x" << kv_lora << "\n";
        }
        return false;
    }
    if (g_q_f32.numel() != n_heads * (qk_nope + qk_rope) || g_attn_f32.rows() != 1 ||
        g_q_abs_f32.numel() != n_heads * kv_lora ||
        g_attn_latent_f32.numel() != n_heads * kv_lora ||
        g_q_abs_xsum_delta_f32.numel() != n_heads * static_cast<int>(q8_1_row_bytes(kv_lora) / 36) ||
        g_attn_xsum_delta_f32.numel() != n_heads * static_cast<int>(q8_1_row_bytes(kv_lora) / 36) ||
        g_attn_scores_f32.numel() != n_heads * max_seq_len) {
        if (deepseek_runtime_options().debug_mla_absorb_compare) {
            std::cerr << "[mla_absorb_compare] tensor shape mismatch q_numel=" << g_q_f32.numel()
                      << " attn_rows=" << g_attn_f32.rows()
                      << " q_abs_numel=" << g_q_abs_f32.numel()
                      << " latent_numel=" << g_attn_latent_f32.numel()
                      << " expected_latent=" << (n_heads * kv_lora) << "\n";
        }
        return false;
    }

    const GPUTensor g_kv_b_weight = s_kv_b_weight.to_gpu(false);
    const size_t row_bytes = g_kv_b_weight.nbytes / static_cast<size_t>(kvb_out);
    launch_mla_absorb_q_nope(s_kv_b_weight.dtype, g_q_f32.data<float>(),
                             static_cast<const uint8_t *>(g_kv_b_weight.data()),
                             row_bytes, g_q_abs_f32.data<float>(), n_heads, qk_nope, qk_rope, v_head, kv_lora);
    const int blocks_per_row = static_cast<int>(q8_1_row_bytes(kv_lora) / 36);
    if (s_kv_b_weight.dtype == DType::Q4_K) {
        launch_mla_absorb_q4_xsum_delta(g_q_f32.data<float>(),
                                        static_cast<const uint8_t *>(g_kv_b_weight.data()),
                                        row_bytes, g_q_abs_xsum_delta_f32.data<float>(), n_heads,
                                        qk_nope, qk_rope, v_head, kv_lora);
    } else {
        cuda_memset_async(g_q_abs_xsum_delta_f32.data<float>(), 0,
                          static_cast<size_t>(n_heads * blocks_per_row) * sizeof(float),
                          "mla_absorb q_abs_xsum_delta memset");
    }
    launch_mla_absorb_scores_device_pos(g_q_abs_f32.data<float>(), g_q_f32.data<float>(),
                                        latent_q8_1_cache, latent_q8_1_row_bytes,
                                        g_q_abs_xsum_delta_f32.data<float>(), g_kv_cache_f32.data<float>(),
                                        g_attn_scores_f32.data<float>(), n_heads, qk_nope, qk_rope, kv_lora,
                                        d_pos, max_seq_len, softmax_scale);
    launch_mla_absorb_context_device_pos(g_attn_scores_f32.data<float>(), latent_q8_1_cache,
                                         latent_q8_1_row_bytes, g_attn_xsum_delta_f32.data<float>(),
                                         g_attn_latent_f32.data<float>(), n_heads, kv_lora,
                                         d_pos, max_seq_len);
    launch_mla_absorb_v(s_kv_b_weight.dtype, static_cast<const uint8_t *>(g_kv_b_weight.data()), row_bytes,
                        g_attn_latent_f32.data<float>(), g_attn_xsum_delta_f32.data<float>(),
                        g_attn_f32.data<float>(),
                        n_heads, qk_nope, v_head, kv_lora);
    return true;
}

bool TensorTool::mla_absorb_decode(const StorageTensor &s_kv_b_weight, const GPUTensor &g_q_f32,
                                   const uint8_t *latent_q8_1_cache, size_t latent_q8_1_row_bytes,
                                   const GPUTensor &g_kv_cache_f32, const GPUTensor &g_attn_f32,
                                   int n_heads, int qk_nope, int qk_rope, int v_head, int kv_lora,
                                   const int *d_pos, int max_seq_len, float softmax_scale,
                                   CudaScratch &scratch, void *stream) {
    auto g_q_abs_f32 = GPUTensor(scratch, "mla_absorb_q", {static_cast<int64_t>(n_heads), static_cast<int64_t>(kv_lora)},
                                 DType::F32);
    auto g_attn_latent_f32 = GPUTensor(scratch, "mla_absorb_latent",
                                       {static_cast<int64_t>(n_heads), static_cast<int64_t>(kv_lora)},
                                       DType::F32);
    auto g_attn_scores_f32 = GPUTensor(scratch, "mla_absorb_scores",
                                       {static_cast<int64_t>(n_heads), static_cast<int64_t>(max_seq_len)},
                                       DType::F32);
    const int blocks_per_row = static_cast<int>(q8_1_row_bytes(kv_lora) / 36);
    auto g_q_abs_xsum_delta_f32 = GPUTensor(scratch, "mla_absorb_q_xsum_delta",
                                           {static_cast<int64_t>(n_heads), static_cast<int64_t>(blocks_per_row)},
                                           DType::F32);
    auto g_attn_xsum_delta_f32 = GPUTensor(scratch, "mla_absorb_attn_xsum_delta",
                                          {static_cast<int64_t>(n_heads), static_cast<int64_t>(blocks_per_row)},
                                          DType::F32);
    return mla_absorb_components(s_kv_b_weight, g_q_f32, latent_q8_1_cache, latent_q8_1_row_bytes,
                                 g_kv_cache_f32, g_q_abs_f32, g_q_abs_xsum_delta_f32,
                                 g_attn_latent_f32, g_attn_scores_f32, g_attn_f32, g_attn_xsum_delta_f32,
                                 n_heads, qk_nope, qk_rope, v_head, kv_lora, d_pos, max_seq_len,
                                 softmax_scale);
}

bool TensorTool::mla_absorb_decode_v_cache(const StorageTensor &s_kv_b_weight, const GPUTensor &g_q_f32,
                                           const uint8_t *latent_q8_1_cache, size_t latent_q8_1_row_bytes,
                                           const GPUTensor &g_kv_cache_f32, const GPUTensor &g_kv_b_cache_f32,
                                           const GPUTensor &g_attn_f32,
                                           int n_heads, int qk_nope, int qk_rope, int v_head, int kv_lora,
                                           const int *d_pos, int max_seq_len, float softmax_scale,
                                           CudaScratch &scratch) {
    if (latent_q8_1_cache == nullptr || latent_q8_1_row_bytes < q8_1_row_bytes(kv_lora)) return false;
    if (!Quant::is_quantized_dtype(s_kv_b_weight.dtype) || s_kv_b_weight.shape.size() != 2) return false;
    const int kvb_out = n_heads * (qk_nope + v_head);
    if (s_kv_b_weight.shape[0] != kvb_out || s_kv_b_weight.shape[1] != kv_lora) return false;
    if (g_q_f32.numel() != n_heads * (qk_nope + qk_rope) ||
        g_kv_b_cache_f32.cols() != kvb_out || g_attn_f32.numel() != n_heads * v_head) {
        return false;
    }

    const GPUTensor g_kv_b_weight = s_kv_b_weight.to_gpu(false);
    const size_t row_bytes = g_kv_b_weight.nbytes / static_cast<size_t>(kvb_out);
    const int blocks_per_row = static_cast<int>(q8_1_row_bytes(kv_lora) / 36);
    auto g_q_abs_f32 = GPUTensor(scratch, "mla_absorb_q", {static_cast<int64_t>(n_heads), static_cast<int64_t>(kv_lora)},
                                 DType::F32);
    auto g_q_abs_xsum_delta_f32 = GPUTensor(scratch, "mla_absorb_q_xsum_delta",
                                           {static_cast<int64_t>(n_heads), static_cast<int64_t>(blocks_per_row)},
                                           DType::F32);
    auto g_attn_scores_f32 = GPUTensor(scratch, "mla_absorb_scores",
                                       {static_cast<int64_t>(n_heads), static_cast<int64_t>(max_seq_len)},
                                       DType::F32);

    launch_mla_absorb_q_nope(s_kv_b_weight.dtype, g_q_f32.data<float>(),
                             static_cast<const uint8_t *>(g_kv_b_weight.data()),
                             row_bytes, g_q_abs_f32.data<float>(), n_heads, qk_nope, qk_rope, v_head, kv_lora);
    if (s_kv_b_weight.dtype == DType::Q4_K) {
        launch_mla_absorb_q4_xsum_delta(g_q_f32.data<float>(),
                                        static_cast<const uint8_t *>(g_kv_b_weight.data()),
                                        row_bytes, g_q_abs_xsum_delta_f32.data<float>(), n_heads,
                                        qk_nope, qk_rope, v_head, kv_lora);
    } else {
        cuda_memset_async(g_q_abs_xsum_delta_f32.data<float>(), 0,
                          static_cast<size_t>(n_heads * blocks_per_row) * sizeof(float),
                          "mla_absorb_v_cache q_abs_xsum_delta memset");
    }
    launch_mla_absorb_scores_device_pos(g_q_abs_f32.data<float>(), g_q_f32.data<float>(),
                                        latent_q8_1_cache, latent_q8_1_row_bytes,
                                        g_q_abs_xsum_delta_f32.data<float>(), g_kv_cache_f32.data<float>(),
                                        g_attn_scores_f32.data<float>(), n_heads, qk_nope, qk_rope, kv_lora,
                                        d_pos, max_seq_len, softmax_scale);
    launch_mla_project_v_device_pos(s_kv_b_weight.dtype, static_cast<const uint8_t *>(g_kv_b_weight.data()), row_bytes,
                                    latent_q8_1_cache, latent_q8_1_row_bytes, g_kv_b_cache_f32.data<float>(),
                                    n_heads, qk_nope, v_head, kv_lora, d_pos);
    launch_mla_absorb_context_v_device_pos(g_attn_scores_f32.data<float>(), g_kv_b_cache_f32.data<float>(),
                                           g_attn_f32.data<float>(), n_heads, qk_nope, v_head,
                                           d_pos, max_seq_len);
    return true;
}

void TensorTool::moe_router_topk(const GPUTensor &g_router_logits_f32, const GPUTensor &g_top_idx_i32,
                                 const GPUTensor &g_top_w_f32,
                                 int n_experts, int k, float routed_scaling) {
    launch_moe_router_topk(g_router_logits_f32.data<float>(), g_top_idx_i32.data<int>(), g_top_w_f32.data<float>(),
                           static_cast<int>(g_router_logits_f32.rows()), n_experts, k,
                           routed_scaling);
}

void TensorTool::moe_accumulate(const GPUTensor &g_expert_out_f32, float weight, const GPUTensor &g_out_f32) {
    launch_moe_accumulate(g_expert_out_f32.data<float>(), weight, g_out_f32.data<float>(),
                          static_cast<int>(g_out_f32.numel()));
}

void TensorTool::moe_accumulate_device(const GPUTensor &g_expert_out_f32, const float *d_weight,
                                       const GPUTensor &g_out_f32) {
    launch_moe_accumulate_device(g_expert_out_f32.data<float>(), d_weight, g_out_f32.data<float>(),
                                 static_cast<int>(g_out_f32.numel()));
}

void TensorTool::argmax(const GPUTensor &g_logits_f32, int *d_out_idx, int vocab) {
    launch_argmax(g_logits_f32.data<float>(), vocab, d_out_idx);
}
