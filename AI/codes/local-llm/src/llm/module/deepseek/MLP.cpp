//
// Created by zhangyoulun on 15/8/2026.
//

#include "MLP.h"

#include "llm/model/deepseek/DeepseekWeights.h"
#include "tensor/GPUTensor.h"
#include "utils/stats/ScopedTimer.h"

MLP::MLP(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : weights_(weights),
      dense_(weights, config),
      moe_(weights, config) {
}

void MLP::forward(DeepseekSession &session, const GPUTensor &g_hidden_f32) {
    if (weights_.is_moe) {
        ScopedCpuTimer t("ds.decode.mlp.moe_forward");
        moe_.forward(session, g_hidden_f32);
    } else {
        ScopedCpuTimer t("ds.decode.mlp.dense_forward");
        dense_.forward(session, g_hidden_f32);
    }
}
