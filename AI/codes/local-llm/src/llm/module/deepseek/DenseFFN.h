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

namespace deepseek {
class RMSNorm;
}

class DenseFFN : public Module {
public:
    DenseFFN(const DeepseekConfig &config, const DeepseekWeights &weights, CudaWeightPool *pool,
             deepseek::RMSNorm *rms_norm);

    void forward(DeepseekSession &session, int layer, float *d_hidden, int tokens);

private:
    const DeepseekConfig &config_;
    const DeepseekWeights &weights_;
    CudaWeightPool *pool_ = nullptr;
    deepseek::RMSNorm *rms_norm_ = nullptr;
};

#endif // LOCAL_LLM_DEEPSEEK_DENSE_FFN_H
