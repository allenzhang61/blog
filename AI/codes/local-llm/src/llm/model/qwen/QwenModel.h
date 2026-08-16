//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_QWENMODEL_H
#define LOCAL_LLM_QWENMODEL_H

#include <memory>
#include <string>
#include <vector>

#include "llm/module/Module.h"
#include "llm/module/common/Embedding.h"
#include "llm/module/qwen/DecoderLayer.h"
#include "llm/module/qwen/RMSNorm.h"
#include "llm/module/qwen/LMHead.h"

#include "format/MF.h"
#include "llm/model/BaseModel.h"
#include "llm/model/qwen/QwenConfig.h"
#include "llm/model/qwen/QwenWeights.h"
#include "utils/sampling/Sampler.h"
#include "backend/cuda/mem/CudaWeightPool.h"

class QwenSession;

// Qwen 文本塔顶层，对应整个 text_config：
//   embed_tokens -> 32 × DecoderLayer -> final_norm -> lm_head。
// 负责串起各子 Module，协调 prefill / decode 两条路径。
//
// 作为 BaseModel 的实现，QwenModel 自持一次推理请求所需的全部对象：
//   config / weights / tokenizer / per-request session。
// Module 本身无 per-request 状态；跨 token 状态与临时激活分别在 QwenSession / QwenForwardScratch。
class QwenModel : public Module, public BaseModel {
public:
    // 从已打开的模型文件加载 config / weights / tokenizer，并建立各子 Module。
    // max_output_tokens 用于每次 prefill 时按需分配 session 的 KV cache 容量。
    // sampling 为采样配置（默认贪心）。
    QwenModel(std::unique_ptr<MF> mf, int max_output_tokens, const SamplingConfig &sampling);
    // session_ 持有前向声明的 QwenSession，析构需在 QwenSession 完整定义处（.cpp）生成。
    ~QwenModel() override;

    // === BaseModel ===
    const char *name() const override { return "qwen"; }
    int eos_token_id() const override { return config_.data.text.eos_token_id; }
    std::vector<int> encode(const std::string &text) const override { return mf_->tokenizer_encode(text); }
    std::string decode_text(const std::vector<int> &ids) const override { return mf_->tokenizer_decode(ids); }

    // prefill：为一次新生成开启内部 session（喂入整段 prompt），跑完各层，返回首个生成 token id。
    int prefill(const std::vector<int> &inputs) override;
    // decode：喂入上一个 token（位置 pos），返回下一个 token id，复用 prefill 建立的 session。
    int decode(int prev_token_id, int pos) override;

    const MemoryUsageProvider &memory_usage() const override;
    CudaWeightPool &weight_pool() override { return global_cuda_weight_pool(); }

    // 供 main 追加已确定的生成 token（用于 attention 上下文记录与最终解码）。
    void append_output(int token_id) override;
    // 本次请求已生成的 token id 序列。
    const std::vector<int> &outputs() const override;

private:
    // 内部前向：喂入整段 prompt token，跑完各层，返回首个生成 token id。
    int prefill_session(QwenSession &session, const std::vector<int> &input);
    // 内部前向：喂入上一个 token（位置 pos），跑完各层，返回下一个 token id。
    int decode_session(QwenSession &session, int prev_token_id, int pos);

    std::unique_ptr<MF> mf_;
    QwenConfig config_;
    QwenWeights weights_;
    int max_output_tokens_ = 0;
    Sampler sampler_;

    common::Embedding embedding_;
    std::vector<DecoderLayer> layers_;
    RMSNorm final_norm_;
    LMHead lm_head_;

    // 当前请求的 per-request 状态；prefill() 会重建，decode() 复用。
    // 语法：
    //  session_ 是一个指向 QwenSession 对象的独占所有权指针
    //  std::unique_ptr<T> 表示“这个对象只被一个地方拥有”。它会在 QwenModel 析构时自动 delete 掉 QwenSession，避免手写 new/delete。
    //  它不能被拷贝，只能移动
    //      std::unique_ptr<QwenSession> a;
    //      std::unique_ptr<QwenSession> b = a;             // 不允许
    //      std::unique_ptr<QwenSession> b = std::move(a);  // 允许，所有权转移
    // 比较：
    //   QwenSession* session_;              // 原始指针，需要自己 delete
    //   std::unique_ptr<QwenSession> session_; // 自动管理生命周期，更安全
    std::unique_ptr<QwenSession> session_;
};


#endif //LOCAL_LLM_QWENMODEL_H
