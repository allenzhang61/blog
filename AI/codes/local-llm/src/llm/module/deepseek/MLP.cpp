//
// Created by zhangyoulun on 15/8/2026.
//

#include "MLP.h"

#include "llm/model/deepseek/DeepseekWeights.h"

MLP::MLP(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : weights_(weights),
      dense_(weights, config),
      moe_(weights, config) {}

void MLP::forward(DeepseekSession &session, const Tensor &hidden) {
    if (weights_.is_moe) {
        moe_.forward(session, hidden);
    } else {
        dense_.forward(session, hidden);
    }
}
