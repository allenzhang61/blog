//
// Created by zhangyoulun on 16/8/2026.
//

#ifndef LOCAL_LLM_COMMON_EMBEDDING_H
#define LOCAL_LLM_COMMON_EMBEDDING_H

#include <string>
#include <vector>

#include "format/MF.h"
#include "llm/module/Module.h"

class CudaWeightPool;
class CudaScratch;

namespace common {

// 词嵌入查表：token id -> hidden 向量。
// 权重形状统一约定为 [vocab_size, hidden_size]。
class Embedding : public Module {
public:
    Embedding(const MFTensorView &weight, CudaWeightPool *pool);

    // 按 token id 逐行拷贝嵌入到 d_hidden（device），形状 [tokens, hidden_size]。
    // scratch 提供 token id 的 int 临时缓冲（key = scratch_key::kInput），用于把 host
    // token id 搬到 device。
    void forward(const std::vector<int> &input, float *d_hidden, CudaScratch &scratch);

private:
    const MFTensorView &weight_;
    CudaWeightPool *pool_ = nullptr;
    int vocab_size_ = 0;
    int hidden_size_ = 0;
};

} // namespace common

#endif // LOCAL_LLM_COMMON_EMBEDDING_H
