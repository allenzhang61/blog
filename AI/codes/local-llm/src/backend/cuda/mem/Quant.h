//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_QUANT_H
#define LOCAL_LLM_QUANT_H

#include "backend/cuda/mem/CudaWeight.h"
#include "format/MF.h"

#include <cstdint>

namespace Quant {

// TensorView.dtype（DType）数值与 ggml_type 保持一致，反量化 kernel 按该 code 分发。
int dtype_code(DType dtype);

// 是否为 GGUF/ggml 量化张量类型。量化权重缓存时保留原始字节，GEMM 前再反量化。
bool is_quantized_dtype(DType dtype);

// shape 各维乘积 = 张量元素总数。
int64_t num_elements(const Tensor &tensor);

// 按 GGML 类型码把常驻量化权重反量化到 device f16。
// ggml_type：0=F32, 6=Q5_0, 8=Q8_0, 12=Q4_K, 14=Q6_K。
CudaWeight dequantize_to_f16(const CudaWeight &quant, uint16_t *d_out_f16,
                             int64_t num_elements, int ggml_type, void *stream = nullptr);

} // namespace Quant

#endif // LOCAL_LLM_QUANT_H
