//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_QWENMODEL_H
#define LOCAL_LLM_QWENMODEL_H

#include <vector>

#include "Module.h"
#include "Embedding.h"
#include "DecoderLayer.h"
#include "RMSNorm.h"
#include "LMHead.h"

class QwenConfig;
class QwenWeights;
class QwenSession;
class CudaWeightPool;

// 文本塔顶层，对应整个 text_config：
//   embed_tokens -> 32 × DecoderLayer -> final_norm -> lm_head。
// 负责串起各子 Module，协调 prefill / decode 两条路径。
// 本身无 per-request 状态；跨 token 状态与临时激活分别在 QwenSession / QwenForwardScratch。
class QwenModel : public Module {
public:
    QwenModel(const QwenConfig &config, const QwenWeights &weights, CudaWeightPool *pool);

    // prefill：喂入整段 prompt token，跑完 32 层，返回首个生成 token id。
    // 同时写好 session 内各层的 KV cache / recurrent state，供后续 decode 复用。
    int prefill(QwenSession &session, const std::vector<int> &h_input_ids);

    // decode：喂入上一个 token（位置 pos），跑完 32 层，返回下一个 token id。
    int decode(QwenSession &session, int prev_token_id, int pos);

private:
    const QwenConfig &config_;
    CudaWeightPool *pool_ = nullptr;

    Embedding embed_tokens_;
    std::vector<DecoderLayer> layers_;
    RMSNorm final_norm_;
    LMHead lm_head_;
};


#endif //LOCAL_LLM_QWENMODEL_H
