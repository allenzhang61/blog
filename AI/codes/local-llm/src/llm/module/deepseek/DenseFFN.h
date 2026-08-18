//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_DENSE_FFN_H
#define LOCAL_LLM_DEEPSEEK_DENSE_FFN_H

#include "llm/module/Module.h"

class CudaWeightPool;
class DeepseekConfig;
class DeepseekSession;
struct DeepseekLayerWeights;

class DenseFFN : public Module {
public:
    DenseFFN(const DeepseekLayerWeights &weights, const DeepseekConfig &config, CudaWeightPool *pool);

    void forward(DeepseekSession &session, float *d_hidden, int input_size);

private:
    const DeepseekConfig &config_;
    const DeepseekLayerWeights &weights_;
    CudaWeightPool *pool_ = nullptr;
};

#endif // LOCAL_LLM_DEEPSEEK_DENSE_FFN_H
