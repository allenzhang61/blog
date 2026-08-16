//
// Created by zhangyoulun on 15/8/2026.
//

#include "MLP.h"

#include "llm/model/deepseek/DeepseekWeights.h"

MLP::MLP(const DeepseekConfig &config, const DeepseekWeights &weights, CudaWeightPool *pool,
         deepseek::RMSNorm *rms_norm)
    : weights_(weights),
      dense_(config, weights, pool, rms_norm),
      moe_(config, weights, pool, rms_norm) {}

void MLP::forward(DeepseekSession &session, int layer, float *d_hidden, int tokens) {
    if (weights_.layers[layer].is_moe) {
        moe_.forward(session, layer, d_hidden, tokens);
    } else {
        dense_.forward(session, layer, d_hidden, tokens);
    }
}
