//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_DENSE_FFN_H
#define LOCAL_LLM_DEEPSEEK_DENSE_FFN_H

#include "llm/module/Module.h"
#include "tensor/CPUTensor.h"
#include "tensor/DiskTensor.h"

class DeepseekConfig;
class DeepseekSession;
struct DeepseekLayerWeights;

class DenseFFN : public Module {
public:
    DenseFFN(const DeepseekLayerWeights &weights, const DeepseekConfig &config);

    void forward(DeepseekSession &session, const GPUTensor &hidden);

private:
    const DeepseekConfig &config_;
    const DeepseekLayerWeights &weights_;
};

#endif // LOCAL_LLM_DEEPSEEK_DENSE_FFN_H
