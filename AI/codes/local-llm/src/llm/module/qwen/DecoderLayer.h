//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_DECODERLAYER_H
#define LOCAL_LLM_DECODERLAYER_H

#include <cstddef>
#include <memory>

#include "llm/module/Module.h"
#include "RMSNorm.h"
#include "SwiGLUMlp.h"

struct LayerWeights;
struct TextConfig;
class QwenSession;
class QwenForwardScratch;
class CudaWeightPool;

// 单个 Decoder 层，对应 LayerWeights。结构（pre-norm + 残差）：
//   h = x + attn( input_norm(x) )
//   y = h + mlp ( post_norm(h) )
// 其中 attn 子层按层类型二选一：full_attention 或 linear_attention。
// 跨 token 状态（KV cache / recurrent state）从 QwenSession 按 layer_index 取。
class DecoderLayer : public Module {
public:
    // layer_index：本层在 32 层中的下标，用于从 QwenSession 取对应 KV/state。
    DecoderLayer(const LayerWeights &weights, const TextConfig &text_config, CudaWeightPool *pool,
                 size_t layer_index);

    // prefill：处理整段输入 [tokens, hidden_size]，原位更新隐状态。
    void prefill(QwenSession &session, float *d_hidden, size_t input_size);

    // decode：处理位置 pos 的单个 token [1, hidden_size]，原位更新隐状态。
    void decode(QwenSession &session, float *d_hidden, int pos, QwenForwardScratch &scratch);

    // 是否为 full_attention 层（否则为 linear_attention）。
    bool is_full_attention() const { return is_full_; }

private:
    const TextConfig &text_config_;
    size_t layer_index_ = 0;
    // 本层在同类型层中的下标，用于索引 QwenSession 的 fullAttnKVCaches / linearAttnRecurrentStates。
    size_t type_index_ = 0;
    bool is_full_ = false;

    RMSNorm input_norm_;
    RMSNorm post_norm_;
    SwiGLUMlp mlp_;
    // 按层类型二选一：is_full_ 为真时用 full attention 子层，否则用 linear attention 子层。
    // 用基类指针持有，具体类型见 FullAttention / LinearAttention。
    std::unique_ptr<Module> attn_;
};


#endif //LOCAL_LLM_DECODERLAYER_H
