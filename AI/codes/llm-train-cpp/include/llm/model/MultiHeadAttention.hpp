#pragma once

#include "llm/model/Linear.hpp"

namespace llm {

// GPT 使用的多头因果自注意力。
// 每个 token 只能看见自己和左侧 token，不能看见未来 token。
class MultiHeadAttention : public Module {
public:
    // 输出 embedding 维度。
    int64_t d_out;

    // 注意力头数量。
    int64_t num_heads;

    // 每个 head 的维度，通常是 d_out / num_heads。
    int64_t head_dim;

    // 最大上下文长度，用于构造因果 mask。
    int64_t context_length;

    // Query 投影矩阵。
    Linear W_query;

    // Key 投影矩阵。
    Linear W_key;

    // Value 投影矩阵。
    Linear W_value;

    // 多个 head 拼接后的输出投影。
    Linear out_proj;

    // d_in 是输入维度，d_out_ 是输出维度，heads 是注意力头数。
    MultiHeadAttention(int64_t d_in, int64_t d_out_, int64_t context, int64_t heads,
                       bool qkv_bias = false, Device device = {});

    // 输入形状通常是 [batch, seq_len, d_in]，输出是 [batch, seq_len, d_out]。
    Tensor forward(const Tensor& x);

    // 汇总 Q/K/V/out projection 的所有参数。
    std::vector<Tensor*> parameters() override;
};

} // namespace llm
