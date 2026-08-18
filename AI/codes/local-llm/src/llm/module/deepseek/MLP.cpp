//
// Created by zhangyoulun on 15/8/2026.
//

#include "MLP.h"

#include "llm/model/deepseek/DeepseekWeights.h"

MLP::MLP(const DeepseekLayerWeights &weights, const DeepseekConfig &config, CudaWeightPool *pool)
    : weights_(weights),
      dense_(weights, config, pool),
      moe_(weights, config, pool) {}

void MLP::forward(DeepseekSession &session, float *d_hidden, int input_size) {
    if (weights_.is_moe) {
        moe_.forward(session, d_hidden, input_size);
    } else {
        dense_.forward(session, d_hidden, input_size);
    }
}
