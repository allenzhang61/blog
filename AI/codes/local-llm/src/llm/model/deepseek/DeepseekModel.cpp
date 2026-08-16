//
// Created by zhangyoulun on 9/8/2026.
//

#include "DeepseekModel.h"

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/ops/gemm.h"

#include <cstddef>
#include <cstdint>
#include <utility>

DeepseekModel::DeepseekModel(std::unique_ptr<MF> mf, int max_output_tokens, const SamplingConfig &sampling)
    : mf_(std::move(mf)),
      config_(*mf_),
      weights_(*mf_, config_),
      max_output_tokens_(max_output_tokens),
      sampler_(sampling),
      embedding_(*weights_.token_embd, &global_cuda_weight_pool()),
      mla_(config_, weights_, &global_cuda_weight_pool()),
      mlp_(config_, weights_, &global_cuda_weight_pool()) {
    config_.DebugDump();
}

DeepseekModel::~DeepseekModel() = default;

int DeepseekModel::forward_session(DeepseekSession &session, const std::vector<int> &input, int start_pos) {
    auto &scratch = session.scratch;
    CudaWeightPool &pool = global_cuda_weight_pool();
    const int input_size = static_cast<int>(input.size());
    const int hidden_size = config_.hidden_size;
    const int vocab_size = config_.vocab_size;

    float *d_hidden = scratch.hidden.ensure(input_size * hidden_size, "ds.hidden");
    embedding_.forward(input, d_hidden, scratch.input, "ds.embedding.ids");

    for (int i = 0; i < config_.num_layers; ++i) {
        mla_.forward(session, i, d_hidden, input_size, start_pos);
        mlp_.forward(session, i, d_hidden, input_size);
    }

    const int last = input_size - 1;
    float *d_last = d_hidden + static_cast<size_t>(last) * hidden_size;
    float *d_normed = scratch.normed.ensure(static_cast<size_t>(hidden_size), "ds.final_normed");
    RMSNorm::forward(&global_cuda_weight_pool(), *weights_.output_norm, d_last, d_normed, 1,
                     hidden_size, config_.rms_norm_eps, /*one_plus=*/false);

    float *d_logits = scratch.logits.ensure(static_cast<size_t>(vocab_size), "ds.logits");
    CudaWeight w = pool.cached_weight(*weights_.output)->try_dequant();
    uint16_t *xlow = scratch.logits_in_lowp.ensure(static_cast<size_t>(hidden_size), "ds.logits_in_lowp");
    GemmInput lm_in = prepare_gemm_input(d_normed, xlow, hidden_size, w.type, nullptr);
    gemm_weight(pool.handle, w, lm_in.ptr, d_logits, vocab_size, hidden_size, 1, lm_in.type, "ds.gemm.lm_head");

    scratch.h_logits.resize(static_cast<size_t>(vocab_size));
    cuda_memcpy_d2h(scratch.h_logits.data(), d_logits, static_cast<size_t>(vocab_size) * sizeof(float),
                    "ds.logits.d2h");
    return sampler_.sample(scratch.h_logits.data(), vocab_size, session.h_outputs);
}

int DeepseekModel::prefill(const std::vector<int> &input) {
    session_ = std::make_unique<DeepseekSession>(config_, input, max_output_tokens_);
    return forward_session(*session_, input, 0);
}

int DeepseekModel::decode(int prev_token_id, int pos) {
    return forward_session(*session_, {prev_token_id}, pos);
}

void DeepseekModel::append_output(int token_id) {
    session_->h_outputs.push_back(token_id);
}

const std::vector<int> &DeepseekModel::outputs() const {
    return session_->h_outputs;
}

const MemoryUsageProvider &DeepseekModel::memory_usage() const {
    return *session_;
}
