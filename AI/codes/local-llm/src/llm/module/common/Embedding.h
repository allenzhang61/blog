//
// Created by zhangyoulun on 16/8/2026.
//

#ifndef LOCAL_LLM_COMMON_EMBEDDING_H
#define LOCAL_LLM_COMMON_EMBEDDING_H

#include <string>
#include <vector>

#include "format/MF.h"
#include "llm/module/Module.h"

class CudaScratch;

namespace common {

// 词嵌入查表：token id -> g_hidden 向量。
// 权重形状统一约定为 [vocab_size, hidden_size]。
class Embedding : public Module {
public:
    Embedding(const StorageTensor &s_weight);

    // 按 token id 逐行拷贝嵌入到 g_hidden（device 激活视图），形状 [tokens, hidden_size]。
    // c_input 为 host 侧 token id 视图（CPUTensor，dtype=I32）；scratch 提供
    // TensorTool 内部搬运 token id 到 device 所需的临时缓冲。
    void forward(CPUTensor c_input, const GPUTensor &g_hidden, CudaScratch &scratch);

private:
    const StorageTensor &s_weight_;
};

} // namespace common

#endif // LOCAL_LLM_COMMON_EMBEDDING_H
