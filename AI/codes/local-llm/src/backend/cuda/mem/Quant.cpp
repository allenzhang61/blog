//
// Created by zhangyoulun on 15/8/2026.
//

#include "Quant.h"

#include "backend/cuda/ops/kernel.cuh"

#include <stdexcept>

namespace Quant {

int dtype_code(DType dtype) {
    return static_cast<int>(dtype);
}

bool is_quantized_dtype(DType dtype) {
    switch (dtype) {
        case DType::Q4_0:
        case DType::Q4_1:
        case DType::Q5_0:
        case DType::Q5_1:
        case DType::Q8_0:
        case DType::Q8_1:
        case DType::Q2_K:
        case DType::Q3_K:
        case DType::Q4_K:
        case DType::Q5_K:
        case DType::Q6_K:
        case DType::Q8_K:
            return true;
        default:
            return false;
    }
}

int64_t num_elements(const TensorView &tensor) {
    int64_t n = 1;
    for (int64_t dim : tensor.shape) {
        n *= dim;
    }
    return n;
}

CudaWeight dequantize_to_f16(const CudaWeight &quant, uint16_t *d_out_f16,
                             int64_t num_elements, int ggml_type, void *stream) {
    const uint8_t *src = static_cast<const uint8_t *>(quant.ptr);
    switch (ggml_type) {
        case 0: // F32
            launch_f32_to_f16_copy(reinterpret_cast<const float *>(src), d_out_f16,
                                   num_elements, stream);
            break;
        case 6: // Q5_0
            launch_dequantize_q50_to_f16(src, d_out_f16, num_elements, stream);
            break;
        case 8: // Q8_0
            launch_dequantize_q80_to_f16(src, d_out_f16, num_elements, stream);
            break;
        case 12: // Q4_K
            launch_dequantize_q4k_to_f16(src, d_out_f16, num_elements, stream);
            break;
        case 14: // Q6_K
            launch_dequantize_q6k_to_f16(src, d_out_f16, num_elements, stream);
            break;
        default:
            throw std::runtime_error("dequantize_to_f16: 不支持的 GGML 类型码 " +
                                     std::to_string(ggml_type));
    }
    return CudaWeight::make_view(d_out_f16, static_cast<size_t>(num_elements) * sizeof(uint16_t),
                                 CUDA_R_16F, DType::F16, num_elements);
}

} // namespace Quant
