//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_COMMON_LMHEAD_H
#define LOCAL_LLM_COMMON_LMHEAD_H

#include "llm/module/Module.h"
#include "tensor/StorageTensor.h"

class SessionBase;
class Sampler;
class GPUTensor;

namespace common {

// 通用输出头：对 norm 后的单行隐状态 [1, hidden_size] 计算 logits，拷回 host 交由
// Sampler 采样得到下一个 token id。与具体模型无关，Qwen / DeepSeek 共用。
//
// s_weight：lm_head 投影权重，逻辑形状 [vocab_size, hidden_size]
//   （Qwen tie_word_embeddings 复用 embed_tokens；DeepSeek 用 output，缺失时加载阶段
//    已回退到 token_embd）。
// logits 及输入低精度中间量走 session.scratch；重复惩罚所需历史 token 取自 session.outputs。
class LMHead : public Module {
public:
    LMHead() = default;

    // 对单行隐状态 [1, hidden_size] 计算 logits，拷回 host 后交由 sampler 采样，返回下一个 token id。
    // g_hidden 为已过 final/output norm 的隐状态激活视图（形状最后一维即 hidden_size）；
    // vocab_size 由权重 shape[0] 推出。仅在需要下一个 token 的位置调用（decode 每步、prefill 末位）。
    int forward(const StorageTensor &s_weight, SessionBase &session,
                const GPUTensor &g_hidden_f32, Sampler &sampler);

    // 贪心专用、可纳入 CUDA Graph 的版本：GEMM 出 logits 后直接 GPU argmax，
    // 把下一个 token id 写到 device buffer d_out_token（不做 D2H、不经 host Sampler）。
    // 仅贪心时可用；温度/top-k/top-p/重复惩罚仍需走 host forward。
    void forward_argmax_device(const StorageTensor &s_weight, SessionBase &session,
                               const GPUTensor &g_hidden_f32, int *d_out_token, void *stream = nullptr);
};

} // namespace common

#endif // LOCAL_LLM_COMMON_LMHEAD_H
