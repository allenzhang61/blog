//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_LMHEAD_H
#define LOCAL_LLM_LMHEAD_H

#include "Module.h"

struct WeightData;
class QwenForwardScratch;
class CudaWeightPool;

// 输出头：final_norm 后的 hidden -> logits，并做 argmax 采样得到下一个 token id。
// 因 tie_word_embeddings=true，投影权重复用 embed_tokens（形状 [vocab_size, hidden_size]）。
// logits / argmax 中间量走 QwenForwardScratch。
class LMHead : public Module {
public:
    // weight：复用的 embed_tokens 权重（tie）；pool：device 权重缓存。
    LMHead(const WeightData &weight, CudaWeightPool *pool);

    // 对单行隐状态 [1, hidden_size] 计算 logits 并 argmax，返回下一个 token id。
    // 仅在需要下一个 token 的位置调用（decode 每步、prefill 末位）。
    int forward(const float *d_hidden, int hidden_size, QwenForwardScratch &scratch);

private:
    const WeightData &weight_;
    CudaWeightPool *pool_ = nullptr;
};


#endif //LOCAL_LLM_LMHEAD_H
