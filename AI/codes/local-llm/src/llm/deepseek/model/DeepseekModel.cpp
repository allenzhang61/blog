//
// Created by zhangyoulun on 9/8/2026.
//

#include "DeepseekModel.h"

#include "backend/cuda/common.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"
#include "utils/log/Log.h"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
// TensorView.dtype（DType）数值与 ggml_type 一致，直接取整型 code。
int dtype_code(DType t) { return static_cast<int>(t); }
// shape 各维乘积 = 元素总数。
int64_t num_elements(const TensorView *t) {
    int64_t n = 1;
    for (int64_t d : t->shape) {
        n *= d;
    }
    return n;
}
} // namespace

DeepseekModel::DeepseekModel(const std::string &model_dir, int max_output_tokens, const SamplingConfig &sampling)
    : gguf_(model_dir), // model_dir 直接是 .gguf 文件路径
      config_(gguf_),
      weights_(gguf_, config_),
      tokenizer_(gguf_),
      max_output_tokens_(max_output_tokens),
      sampler_(sampling) {
    config_.DebugDump();
}

DeepseekModel::~DeepseekModel() = default;

// ---- 权重 GPU 化 ----

CudaWeight DeepseekModel::dequant_weight(const TensorView *t, CudaScratchBuffer<uint16_t> &buf,
                                         const std::string &tag) {
    CudaWeight *resident = pool_.cached_quant_weight(t->name, t->data, t->nbytes);
    if (resident == nullptr) {
        throw std::runtime_error("dequant_weight: 常驻量化上传失败 " + t->name);
    }
    const int64_t n = num_elements(t);
    uint16_t *out = buf.ensure(static_cast<size_t>(n), tag);
    return CudaWeightPool::dequantize_to_f16(*resident, out, n, dtype_code(t->dtype), nullptr);
}

CudaWeight DeepseekModel::dequant_expert(const TensorView *t, int expert, int n_experts,
                                         CudaScratchBuffer<uint16_t> &buf, const std::string &tag) {
    // 3D 专家权重：n_experts 个连续 2D 块。第 e 个块的原始字节偏移与元素偏移。
    const int64_t n_total = num_elements(t);
    const int64_t n_per = n_total / n_experts;
    const size_t bytes_per = t->nbytes / static_cast<size_t>(n_experts);
    const std::string ename = t->name + ".e" + std::to_string(expert);
    CudaWeight *resident = pool_.cached_quant_weight(ename, t->data + expert * bytes_per, bytes_per);
    if (resident == nullptr) {
        throw std::runtime_error("dequant_expert: 常驻量化上传失败 " + ename);
    }
    uint16_t *out = buf.ensure(static_cast<size_t>(n_per), tag);
    return CudaWeightPool::dequantize_to_f16(*resident, out, n_per, dtype_code(t->dtype), nullptr);
}

const float *DeepseekModel::resident_f32(const TensorView *t, const std::string &tag) {
    (void) tag;
    if (t->dtype != DType::F32) {
        throw std::runtime_error("resident_f32: 期望 F32 张量 " + t->name);
    }
    CudaWeight *resident = pool_.cached_quant_weight(t->name, t->data, t->nbytes);
    if (resident == nullptr) {
        throw std::runtime_error("resident_f32: 上传失败 " + t->name);
    }
    return static_cast<const float *>(resident->ptr);
}

// ---- MLA ----

void DeepseekModel::mla_forward(DeepseekSession &session, int layer, int tokens, int start_pos) {
    auto &s = session.scratch;
    const DeepseekLayerWeights &lw = weights_.layers[layer];
    const int H = config_.hidden_size;
    const int n_heads = config_.num_heads;
    const int qk_nope = config_.qk_nope_head_dim;
    const int qk_rope = config_.qk_rope_head_dim;
    const int qk_head = config_.qk_head_dim();       // 192
    const int v_head = config_.v_head_dim;           // 128
    const int kv_lora = config_.kv_lora_rank;        // 512
    const int kv_total = kv_lora + qk_rope;          // 576
    const int q_dim = n_heads * qk_head;             // 3072
    const int kvb_out = n_heads * (qk_nope + v_head); // 4096

    float *d_normed = s.normed.ensure(static_cast<size_t>(tokens) * H, "ds.normed");
    // attn_norm（标准 RMSNorm，F32 权重）
    launch_rms_norm(s.hidden, resident_f32(lw.attn_norm, "attn_norm"), 2, d_normed, tokens, H,
                    config_.rms_norm_eps, false, nullptr);

    // q = attn_q · normed  -> [tokens, q_dim]
    float *d_q = s.q.ensure(static_cast<size_t>(tokens) * q_dim, "ds.q");
    {
        CudaWeight w = dequant_weight(lw.attn_q, s.deq_a, "ds.deq.attn_q");
        uint16_t *xlow = s.normed_lowp.ensure(static_cast<size_t>(tokens) * H, "ds.normed_lowp");
        to_weight_lowp(d_normed, xlow, tokens * H, w, nullptr);
        gemm_weight(pool_.handle, w, q_dim, H, xlow, w.type, tokens, d_q, "ds.gemm.attn_q");
    }
    // q 的 rope 段旋转
    if (tokens == 1) {
        launch_mla_rope_q(d_q, n_heads, qk_nope, qk_rope, start_pos, static_cast<const float *>(session.inv_freq.ptr), nullptr);
    } else {
        launch_mla_rope_q_batch(d_q, tokens, n_heads, qk_nope, qk_rope, start_pos, static_cast<const float *>(session.inv_freq.ptr), nullptr);
    }

    // kv_a = kv_a_mqa · normed -> [tokens, kv_total]
    float *d_kv_a = s.kv_a.ensure(static_cast<size_t>(tokens) * kv_total, "ds.kv_a");
    {
        CudaWeight w = dequant_weight(lw.attn_kv_a_mqa, s.deq_a, "ds.deq.kv_a_mqa");
        uint16_t *xlow = s.normed_lowp.ensure(static_cast<size_t>(tokens) * H, "ds.normed_lowp");
        to_weight_lowp(d_normed, xlow, tokens * H, w, nullptr);
        gemm_weight(pool_.handle, w, kv_total, H, xlow, w.type, tokens, d_kv_a, "ds.gemm.kv_a");
    }

    // latent RMSNorm + k_rope RoPE + 写 latent KV cache
    float *d_cache = static_cast<float *>(session.kv_caches[layer].cache.ptr);
    const float *kv_a_norm = resident_f32(lw.attn_kv_a_norm, "kv_a_norm");
    if (tokens == 1) {
        launch_mla_kv_a(d_kv_a, kv_a_norm, d_cache, kv_lora, qk_rope, session.max_seq_len,
                        start_pos, static_cast<const float *>(session.inv_freq.ptr),
                        config_.rms_norm_eps, nullptr);
    } else {
        launch_mla_kv_a_batch(d_kv_a, kv_a_norm, d_cache, tokens, kv_lora, qk_rope,
                              session.max_seq_len, start_pos,
                              static_cast<const float *>(session.inv_freq.ptr),
                              config_.rms_norm_eps, nullptr);
    }
    session.kv_caches[layer].seq_len = start_pos + tokens;

    // kv_b 上投影：对 cache 中所有位置的 latent 段做 kv_b -> [seq, kvb_out]，解出每 (pos,head) 的 k_nope||v。
    // MVP：对整段 [0, seq) 重新投影（seq 不大）。
    const int seq = start_pos + tokens;
    float *d_kvb = s.kv_b_out.ensure(static_cast<size_t>(seq) * kvb_out, "ds.kv_b_out");
    {
        CudaWeight w = dequant_weight(lw.attn_kv_b, s.deq_b, "ds.deq.kv_b");
        // 输入是 cache 的 latent 段：cache 行 stride = kv_total，取前 kv_lora。
        // gemm_weight 期望激活按列连续 [in_dim, tokens]，in_dim=kv_lora。
        // cache latent 段非连续（每行含 k_rope 尾巴），先抽取到连续 latent buffer。
        uint16_t *latent_low = s.latent_lowp.ensure(static_cast<size_t>(seq) * kv_lora, "ds.latent_lowp");
        // 用一个小 kernel 抽取+转 lowp：这里复用 float_to_lowp 需连续 float，先 gather。
        // 简化：借用 attn buffer 作为连续 latent f32。
        float *latent_f32 = s.attn.ensure(static_cast<size_t>(seq) * kv_lora, "ds.latent_f32");
        // gather latent（stride kv_total -> kv_lora）：逐行 cudaMemcpy2D。
        check_cuda(cudaMemcpy2D(latent_f32, kv_lora * sizeof(float), d_cache,
                                kv_total * sizeof(float), kv_lora * sizeof(float), seq,
                                cudaMemcpyDeviceToDevice),
                   "ds.gather.latent");
        to_weight_lowp(latent_f32, latent_low, seq * kv_lora, w, nullptr);
        gemm_weight(pool_.handle, w, kvb_out, kv_lora, latent_low, w.type, seq, d_kvb, "ds.gemm.kv_b");
    }

    // attend
    float *d_attn = s.attn.ensure(static_cast<size_t>(tokens) * n_heads * v_head, "ds.attn");
    if (tokens == 1) {
        launch_mla_attend(d_q, d_kvb, d_cache, d_attn, n_heads, qk_nope, qk_rope, v_head, kv_lora,
                          session.max_seq_len, start_pos, session.attn_softmax_scale, nullptr);
    } else {
        launch_mla_attend_batch(d_q, d_kvb, d_cache, d_attn, tokens, n_heads, qk_nope, qk_rope,
                                v_head, kv_lora, session.max_seq_len, start_pos,
                                session.attn_softmax_scale, nullptr);
    }

    // o = attn_output · attn -> [tokens, H]
    float *d_out = s.attn_out.ensure(static_cast<size_t>(tokens) * H, "ds.attn_out");
    {
        CudaWeight w = dequant_weight(lw.attn_output, s.deq_a, "ds.deq.attn_output");
        const int in_dim = n_heads * v_head; // 2048
        uint16_t *xlow = s.attn_lowp.ensure(static_cast<size_t>(tokens) * in_dim, "ds.attn_lowp");
        to_weight_lowp(d_attn, xlow, tokens * in_dim, w, nullptr);
        gemm_weight(pool_.handle, w, H, in_dim, xlow, w.type, tokens, d_out, "ds.gemm.attn_output");
    }

    // 残差
    launch_add(s.hidden, d_out, s.hidden, tokens * H, nullptr);
}

// ---- dense FFN ----

void DeepseekModel::ffn_dense_forward(DeepseekSession &session, int layer, int tokens) {
    auto &s = session.scratch;
    const DeepseekLayerWeights &lw = weights_.layers[layer];
    const int H = config_.hidden_size;
    const int ffn = config_.dense_ffn;

    float *d_normed = s.normed.ensure(static_cast<size_t>(tokens) * H, "ds.normed");
    launch_rms_norm(s.hidden, resident_f32(lw.ffn_norm, "ffn_norm"), 2, d_normed, tokens, H,
                    config_.rms_norm_eps, false, nullptr);

    float *d_gate = s.gate.ensure(static_cast<size_t>(tokens) * ffn, "ds.gate");
    float *d_up = s.up.ensure(static_cast<size_t>(tokens) * ffn, "ds.up");
    uint16_t *xlow = s.ffn_in_lowp.ensure(static_cast<size_t>(tokens) * H, "ds.ffn_in_lowp");
    {
        CudaWeight wg = dequant_weight(lw.ffn_gate, s.deq_a, "ds.deq.ffn_gate");
        to_weight_lowp(d_normed, xlow, tokens * H, wg, nullptr);
        gemm_weight(pool_.handle, wg, ffn, H, xlow, wg.type, tokens, d_gate, "ds.gemm.ffn_gate");
    }
    {
        CudaWeight wu = dequant_weight(lw.ffn_up, s.deq_a, "ds.deq.ffn_up");
        to_weight_lowp(d_normed, xlow, tokens * H, wu, nullptr);
        gemm_weight(pool_.handle, wu, ffn, H, xlow, wu.type, tokens, d_up, "ds.gemm.ffn_up");
    }
    float *d_act = s.act.ensure(static_cast<size_t>(tokens) * ffn, "ds.act");
    launch_silu_mul(d_gate, d_up, d_act, tokens * ffn, nullptr);

    float *d_out = s.ffn_out.ensure(static_cast<size_t>(tokens) * H, "ds.ffn_out");
    {
        CudaWeight wd = dequant_weight(lw.ffn_down, s.deq_a, "ds.deq.ffn_down");
        uint16_t *alow = s.act_lowp.ensure(static_cast<size_t>(tokens) * ffn, "ds.act_lowp");
        to_weight_lowp(d_act, alow, tokens * ffn, wd, nullptr);
        gemm_weight(pool_.handle, wd, H, ffn, alow, wd.type, tokens, d_out, "ds.gemm.ffn_down");
    }
    launch_add(s.hidden, d_out, s.hidden, tokens * H, nullptr);
}

// ---- MoE FFN ----
// 逐 token 处理：router 选 top-k -> 各路由专家 SwiGLU 加权累加 -> 加共享专家 -> 残差。
void DeepseekModel::ffn_moe_forward(DeepseekSession &session, int layer, int tokens) {
    auto &s = session.scratch;
    const DeepseekLayerWeights &lw = weights_.layers[layer];
    const int H = config_.hidden_size;
    const int ffn = config_.expert_ffn;
    const int n_exp = config_.expert_count;
    const int k = config_.expert_used;
    const int shared_ffn = config_.shared_ffn();

    float *d_normed = s.normed.ensure(static_cast<size_t>(tokens) * H, "ds.normed");
    launch_rms_norm(s.hidden, resident_f32(lw.ffn_norm, "ffn_norm"), 2, d_normed, tokens, H,
                    config_.rms_norm_eps, false, nullptr);

    // router logits = ffn_gate_inp · normed -> [tokens, n_exp]（F32 权重）
    float *d_router = s.router_logits.ensure(static_cast<size_t>(tokens) * n_exp, "ds.router");
    {
        // ffn_gate_inp 为 F32，作为 f16 gemm 需转；直接用 dequant 走 type=0(F32->f16)。
        CudaWeight w = dequant_weight(lw.ffn_gate_inp, s.deq_b, "ds.deq.gate_inp");
        uint16_t *xlow = s.ffn_in_lowp.ensure(static_cast<size_t>(tokens) * H, "ds.ffn_in_lowp");
        to_weight_lowp(d_normed, xlow, tokens * H, w, nullptr);
        gemm_weight(pool_.handle, w, n_exp, H, xlow, w.type, tokens, d_router, "ds.gemm.router");
    }
    int *d_topidx = s.top_idx.ensure(static_cast<size_t>(tokens) * k, "ds.topidx");
    float *d_topw = s.top_w.ensure(static_cast<size_t>(tokens) * k, "ds.topw");
    launch_moe_router_topk(d_router, d_topidx, d_topw, tokens, n_exp, k, config_.routed_scaling, nullptr);

    // 取回 host 侧路由结果（tokens 不大）
    std::vector<int> h_idx(static_cast<size_t>(tokens) * k);
    std::vector<float> h_w(static_cast<size_t>(tokens) * k);
    check_cuda(cudaMemcpy(h_idx.data(), d_topidx, h_idx.size() * sizeof(int), cudaMemcpyDeviceToHost), "ds.moe.idx");
    check_cuda(cudaMemcpy(h_w.data(), d_topw, h_w.size() * sizeof(float), cudaMemcpyDeviceToHost), "ds.moe.w");

    // moe_out 清零
    float *d_moe = s.moe_out.ensure(static_cast<size_t>(tokens) * H, "ds.moe_out");
    check_cuda(cudaMemset(d_moe, 0, static_cast<size_t>(tokens) * H * sizeof(float)), "ds.moe.zero");

    float *d_gate = s.gate.ensure(static_cast<size_t>(ffn), "ds.egate");
    float *d_up = s.up.ensure(static_cast<size_t>(ffn), "ds.eup");
    float *d_act = s.act.ensure(static_cast<size_t>(ffn), "ds.eact");
    float *d_eout = s.expert_out.ensure(static_cast<size_t>(H), "ds.eout");
    uint16_t *xlow = s.ffn_in_lowp.ensure(static_cast<size_t>(H), "ds.ffn_in_lowp");
    uint16_t *alow = s.act_lowp.ensure(static_cast<size_t>(ffn), "ds.act_lowp");

    // 逐 token、逐路由专家（MVP：单 token 展开）
    for (int tok = 0; tok < tokens; ++tok) {
        const float *tok_in = d_normed + static_cast<size_t>(tok) * H;
        for (int r = 0; r < k; ++r) {
            const int e = h_idx[static_cast<size_t>(tok) * k + r];
            const float w = h_w[static_cast<size_t>(tok) * k + r];
            // gate/up/down 专家切片
            CudaWeight wg = dequant_expert(lw.ffn_gate_exps, e, n_exp, s.deq_a, "ds.deq.egate");
            to_weight_lowp(tok_in, xlow, H, wg, nullptr);
            gemm_weight(pool_.handle, wg, ffn, H, xlow, wg.type, 1, d_gate, "ds.gemm.egate");
            CudaWeight wu = dequant_expert(lw.ffn_up_exps, e, n_exp, s.deq_a, "ds.deq.eup");
            to_weight_lowp(tok_in, xlow, H, wu, nullptr);
            gemm_weight(pool_.handle, wu, ffn, H, xlow, wu.type, 1, d_up, "ds.gemm.eup");
            launch_silu_mul(d_gate, d_up, d_act, ffn, nullptr);
            CudaWeight wd = dequant_expert(lw.ffn_down_exps, e, n_exp, s.deq_a, "ds.deq.edown");
            to_weight_lowp(d_act, alow, ffn, wd, nullptr);
            gemm_weight(pool_.handle, wd, H, ffn, alow, wd.type, 1, d_eout, "ds.gemm.edown");
            launch_moe_accumulate(d_eout, w, d_moe + static_cast<size_t>(tok) * H, H, nullptr);
        }
    }

    // shared expert（对所有 token 一起做，权重 1.0 累加）
    {
        float *d_sgate = s.gate.ensure(static_cast<size_t>(tokens) * shared_ffn, "ds.sgate");
        float *d_sup = s.up.ensure(static_cast<size_t>(tokens) * shared_ffn, "ds.sup");
        uint16_t *sxlow = s.ffn_in_lowp.ensure(static_cast<size_t>(tokens) * H, "ds.ffn_in_lowp");
        CudaWeight wg = dequant_weight(lw.ffn_gate_shexp, s.deq_a, "ds.deq.sgate");
        to_weight_lowp(d_normed, sxlow, tokens * H, wg, nullptr);
        gemm_weight(pool_.handle, wg, shared_ffn, H, sxlow, wg.type, tokens, d_sgate, "ds.gemm.sgate");
        CudaWeight wu = dequant_weight(lw.ffn_up_shexp, s.deq_a, "ds.deq.sup");
        to_weight_lowp(d_normed, sxlow, tokens * H, wu, nullptr);
        gemm_weight(pool_.handle, wu, shared_ffn, H, sxlow, wu.type, tokens, d_sup, "ds.gemm.sup");
        float *d_sact = s.act.ensure(static_cast<size_t>(tokens) * shared_ffn, "ds.sact");
        launch_silu_mul(d_sgate, d_sup, d_sact, tokens * shared_ffn, nullptr);
        float *d_sout = s.ffn_out.ensure(static_cast<size_t>(tokens) * H, "ds.sout");
        CudaWeight wd = dequant_weight(lw.ffn_down_shexp, s.deq_a, "ds.deq.sdown");
        uint16_t *salow = s.act_lowp.ensure(static_cast<size_t>(tokens) * shared_ffn, "ds.act_lowp");
        to_weight_lowp(d_sact, salow, tokens * shared_ffn, wd, nullptr);
        gemm_weight(pool_.handle, wd, H, shared_ffn, salow, wd.type, tokens, d_sout, "ds.gemm.sdown");
        launch_add(d_moe, d_sout, d_moe, tokens * H, nullptr);
    }

    // 残差
    launch_add(s.hidden, d_moe, s.hidden, tokens * H, nullptr);
}

void DeepseekModel::layer_forward(DeepseekSession &session, int layer, int tokens, int start_pos) {
    mla_forward(session, layer, tokens, start_pos);
    if (weights_.layers[layer].is_moe) {
        ffn_moe_forward(session, layer, tokens);
    } else {
        ffn_dense_forward(session, layer, tokens);
    }
}

// ---- 前向入口 ----

int DeepseekModel::forward(DeepseekSession &session, const std::vector<int> &token_ids, int start_pos) {
    auto &s = session.scratch;
    const int tokens = static_cast<int>(token_ids.size());
    const int H = config_.hidden_size;
    const int vocab = config_.vocab_size;

    // embedding：token_embd 为量化，先反量化整表到 f16（常驻一次），再查表。
    float *d_hidden = s.hidden.ensure(static_cast<size_t>(tokens) * H, "ds.hidden");
    CudaWeight embd = dequant_weight(weights_.token_embd, s.deq_a, "ds.deq.token_embd");
    // token ids 上 device
    int *d_ids = s.top_idx.ensure(static_cast<size_t>(tokens), "ds.ids"); // 借用 int buffer
    check_cuda(cudaMemcpy(d_ids, token_ids.data(), tokens * sizeof(int), cudaMemcpyHostToDevice), "ds.ids.h2d");
    launch_embedding_lookup(static_cast<const uint16_t *>(embd.ptr), d_ids, d_hidden, tokens, vocab, H, 1, nullptr);

    for (int i = 0; i < config_.num_layers; ++i) {
        layer_forward(session, i, tokens, start_pos);
    }

    // final norm 对最后一个 token
    const int last = tokens - 1;
    float *d_last = d_hidden + static_cast<size_t>(last) * H;
    float *d_normed = s.normed.ensure(static_cast<size_t>(H), "ds.final_normed");
    launch_rms_norm(d_last, resident_f32(weights_.output_norm, "output_norm"), 2, d_normed, 1, H,
                    config_.rms_norm_eps, false, nullptr);

    // lm_head：output.weight（Q6_K，非 tie）
    float *d_logits = s.logits.ensure(static_cast<size_t>(vocab), "ds.logits");
    CudaWeight w = dequant_weight(weights_.output, s.deq_b, "ds.deq.lm_head");
    uint16_t *xlow = s.logits_in_lowp.ensure(static_cast<size_t>(H), "ds.logits_in_lowp");
    to_weight_lowp(d_normed, xlow, H, w, nullptr);
    gemm_weight(pool_.handle, w, vocab, H, xlow, w.type, 1, d_logits, "ds.gemm.lm_head");

    // logits 拷回 host，交由 Sampler 采样（greedy 时内部走 argmax）。
    s.h_logits.resize(static_cast<size_t>(vocab));
    check_cuda(cudaMemcpy(s.h_logits.data(), d_logits, static_cast<size_t>(vocab) * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "ds.logits.d2h");
    return sampler_.sample(s.h_logits.data(), vocab, session.h_outputs);
}

int DeepseekModel::prefill(const std::vector<int> &input_ids) {
    session_ = std::make_unique<DeepseekSession>(config_, input_ids, max_output_tokens_);
    return forward(*session_, input_ids, 0);
}

int DeepseekModel::decode(int prev_token_id, int pos) {
    return forward(*session_, {prev_token_id}, pos);
}

void DeepseekModel::append_output(int token_id) { session_->h_outputs.push_back(token_id); }

const std::vector<int> &DeepseekModel::outputs() const { return session_->h_outputs; }

const MemoryUsageProvider &DeepseekModel::memory_usage() const { return *session_; }
