//
// Created by zhangyoulun on 15/8/2026.
//

#include "tensor/Tensor.h"

const char *dtype_name(DType dt) {
    switch (dt) {
        case DType::F32: return "F32";
        case DType::F16: return "F16";
        case DType::Q4_0: return "Q4_0";
        case DType::Q4_1: return "Q4_1";
        case DType::Q5_0: return "Q5_0";
        case DType::Q5_1: return "Q5_1";
        case DType::Q8_0: return "Q8_0";
        case DType::Q8_1: return "Q8_1";
        case DType::Q2_K: return "Q2_K";
        case DType::Q3_K: return "Q3_K";
        case DType::Q4_K: return "Q4_K";
        case DType::Q5_K: return "Q5_K";
        case DType::Q6_K: return "Q6_K";
        case DType::Q8_K: return "Q8_K";
        case DType::I32: return "I32";
        case DType::BF16: return "BF16";
        default: return "UNKNOWN";
    }
}

bool is_supported_dtype(DType dt) {
    switch (dt) {
        case DType::F32:
        case DType::F16:
        case DType::BF16:
        case DType::Q4_K:
        case DType::Q5_0:
        case DType::Q6_K:
        case DType::Q8_0:
            return true;
        // 以下量化类型能被识别，但尚未实现反量化 kernel，暂不启用。
        // 要启用需先在 Quant::dequantize_to_f16 补对应 launch_dequantize_*_to_f16。
        // case DType::Q5_K: // 唯一实战常见（Q5_K_M），后续最该优先补
        // case DType::Q4_0: // legacy，已被 K-quant 取代，新模型基本不发布
        // case DType::Q4_1: // legacy，几乎绝迹
        // case DType::Q5_1: // legacy，少见
        // case DType::Q8_1: // legacy，少见
        // case DType::Q2_K: // 极限压缩档，质量损失大，小众
        // case DType::Q3_K: // 极限压缩档，小众
        // case DType::Q8_K: // 多为 llama.cpp 内部中间格式，权重文件基本不落盘
        default:
            return false;
    }
}

// 注意：Tensor::cached_weight() 的实现定义在
// backend/cuda/mem/CudaWeightPool.cpp，以便访问 CudaWeightPool 全量类型，
// 保持本 tensor 层的 .cpp 零 CUDA 依赖。
