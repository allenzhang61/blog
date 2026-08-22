//
// Created by zhangyoulun on 15/8/2026.
//

#include "MLP.h"

#include "llm/model/deepseek/DeepseekWeights.h"
#include "tensor/GPUTensor.h"

MLP::MLP(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : weights_(weights),
      dense_(weights, config),
      moe_(weights, config) {
}

void MLP::forward(DeepseekSession &session, const GPUTensor &g_hidden_f32) {
    if (weights_.is_moe) {
        moe_.forward(session, g_hidden_f32);
    } else {
        dense_.forward(session, g_hidden_f32);
    }
}
