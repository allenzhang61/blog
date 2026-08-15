//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_DENSE_FFN_H
#define LOCAL_LLM_DEEPSEEK_DENSE_FFN_H

#include "llm/module/Module.h"

class CudaWeightPool;
class DeepseekConfig;
class DeepseekSession;
class DeepseekWeights;

class DenseFFN : public Module {
public:
    DenseFFN(const DeepseekConfig &config, const DeepseekWeights &weights, CudaWeightPool *pool);

    void forward(DeepseekSession &session, int layer, int tokens);

private:
    const DeepseekConfig &config_;
    const DeepseekWeights &weights_;
    CudaWeightPool *pool_ = nullptr;
};

#endif // LOCAL_LLM_DEEPSEEK_DENSE_FFN_H
