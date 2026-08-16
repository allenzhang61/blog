//
// Created by zhangyoulun on 15/8/2026.
//

#include "MLP.h"

#include "llm/model/deepseek/DeepseekWeights.h"

MLP::MLP(const DeepseekConfig &config, const DeepseekWeights &weights, CudaWeightPool *pool)
    : weights_(weights),
      dense_(config, weights, pool),
      moe_(config, weights, pool) {}

void MLP::forward(DeepseekSession &session, int layer, float *d_hidden, int input_size) {
    if (weights_.layers[layer].is_moe) {
        moe_.forward(session, layer, d_hidden, input_size);
    } else {
        dense_.forward(session, layer, d_hidden, input_size);
    }
}
