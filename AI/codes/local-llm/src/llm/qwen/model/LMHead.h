//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_LMHEAD_H
#define LOCAL_LLM_LMHEAD_H

#include "Module.h"

#include <vector>

struct WeightData;
class QwenForwardScratch;
class CudaWeightPool;
class Sampler;

// 输出头：final_norm 后的 hidden -> logits，交由 Sampler 采样得到下一个 token id。
// 因 tie_word_embeddings=true，投影权重复用 embed_tokens（形状 [vocab_size, hidden_size]）。
// logits 中间量走 QwenForwardScratch；采样在 host 端进行（logits 拷回主机）。
class LMHead : public Module {
public:
    // weight：复用的 embed_tokens 权重（tie）；pool：device 权重缓存。
    LMHead(const WeightData &weight, CudaWeightPool *pool);

    // 对单行隐状态 [1, hidden_size] 计算 logits，拷回 host 后交由 sampler 采样，
    // 返回下一个 token id。prev_tokens 供重复惩罚使用（可为空）。
    // 仅在需要下一个 token 的位置调用（decode 每步、prefill 末位）。
    int forward(const float *d_hidden, int hidden_size, QwenForwardScratch &scratch,
                Sampler &sampler, const std::vector<int> &prev_tokens);

private:
    const WeightData &weight_;
    CudaWeightPool *pool_ = nullptr;
};


#endif //LOCAL_LLM_LMHEAD_H
