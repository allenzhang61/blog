//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_DENSE_FFN_H
#define LOCAL_LLM_DEEPSEEK_DENSE_FFN_H

#include "llm/module/Module.h"
#include "tensor/CPUTensor.h"
#include "tensor/StorageTensor.h"

class DeepseekConfig;
class DeepseekSession;
struct DeepseekLayerWeights;

class DenseFFN : public Module {
public:
    DenseFFN(const DeepseekLayerWeights &weights, const DeepseekConfig &config);

    void forward(DeepseekSession &session, const GPUTensor &g_hidden_f32);

private:
    const DeepseekConfig &config_;
    // Dense FFN 权重：
    //   s_ffn_norm [hidden]
    //   s_ffn_gate/s_ffn_up [dense_ffn, hidden]
    //   s_ffn_down [hidden, dense_ffn]
    const DeepseekLayerWeights &weights_;
};

#endif // LOCAL_LLM_DEEPSEEK_DENSE_FFN_H
