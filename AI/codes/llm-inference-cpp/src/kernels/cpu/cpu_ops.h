#pragma once

#include "../../core/safetensors.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llm_inference {
namespace cpu {

// 将 BF16 原始 bit 表示转换为 float。
float bf16_to_float(uint16_t value);

// 将 IEEE FP16 原始 bit 表示转换为 float。
float f16_to_float(uint16_t h);

// 读取 tensor 中指定线性下标的标量值，并按 dtype 转为 float。
float tensor_value(const TensorRef & ref, size_t index);

// 计算 weight 指定行与向量 x 的点积。
float dot_row(const TensorRef & weight, int row, const std::vector<float> & x);

// CPU embedding lookup，按 token id 取出 embedding 行。
void embedding_lookup(const TensorRef & emb, int token_id, std::vector<float> & y);

// CPU RMSNorm；one_plus 为 true 时使用 (1 + weight) 口径。
void rms_norm(const TensorRef & weight, const std::vector<float> & x, std::vector<float> & y, float eps, bool one_plus);

// sigmoid 激活函数。
float sigmoid(float x);

// SiLU 激活函数。
float silu(float x);

// softplus 激活函数。
float softplus(float x);

// 原地 L2 normalize。
void l2_norm_inplace(float * x, int dim, float eps = 1e-6f);

// 对单个 head 做 gated RMSNorm。
void gated_rms_norm_head(const TensorRef & weight, const float * x, const float * gate, float * y, int dim, float eps);

// x += y。
void add_inplace(std::vector<float> & x, const std::vector<float> & y);

} // namespace cpu
} // namespace llm_inference
