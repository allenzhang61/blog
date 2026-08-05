#include "ops.h"

#include "../kernels/cpu/cpu_ops.h"
#include "../kernels/cuda/cuda_ops.h"
#include "../core/tensor_ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace llm_inference {

LinearLayerState::LinearLayerState(LinearLayerState && other) noexcept
    : conv_state(std::move(other.conv_state)),
      recurrent_state(std::move(other.recurrent_state)),
      cuda_state(other.cuda_state) {
    other.cuda_state = nullptr;
}

LinearLayerState & LinearLayerState::operator=(LinearLayerState && other) noexcept {
    if (this != &other) {
        cuda_free_linear_attention_state(cuda_state);
        conv_state = std::move(other.conv_state);
        recurrent_state = std::move(other.recurrent_state);
        cuda_state = other.cuda_state;
        other.cuda_state = nullptr;
    }
    return *this;
}

LinearLayerState::~LinearLayerState() {
    cuda_free_linear_attention_state(cuda_state);
}

FullAttentionState::FullAttentionState(FullAttentionState && other) noexcept
    : key_cache(std::move(other.key_cache)),
      value_cache(std::move(other.value_cache)),
      max_seq_len(other.max_seq_len),
      cuda_state(other.cuda_state) {
    other.cuda_state = nullptr;
}

FullAttentionState & FullAttentionState::operator=(FullAttentionState && other) noexcept {
    if (this != &other) {
        cuda_free_full_attention_state(cuda_state);
        key_cache = std::move(other.key_cache);
        value_cache = std::move(other.value_cache);
        max_seq_len = other.max_seq_len;
        cuda_state = other.cuda_state;
        other.cuda_state = nullptr;
    }
    return *this;
}

FullAttentionState::~FullAttentionState() {
    cuda_free_full_attention_state(cuda_state);
}

namespace ops {

namespace {

// 对单个 full attention head 应用 RoPE。
void apply_rope(const ModelConfig & config, float * vec, int pos) {
    const int rotary_dim = static_cast<int>(config.head_dim * config.partial_rotary_factor);
    const int half = rotary_dim / 2;
    for (int i = 0; i < half; ++i) {
        const float inv_freq = 1.0f / std::pow(config.rope_theta, static_cast<float>(2 * i) / rotary_dim);
        const float angle = static_cast<float>(pos) * inv_freq;
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        const float x1 = vec[i];
        const float x2 = vec[i + half];
        vec[i] = x1 * c - x2 * s;
        vec[i + half] = x2 * c + x1 * s;
    }
}

} // namespace

bool linear_attention_full_layer(
    const ModelConfig & config,
    const LayerWeights & w,
    const std::vector<float> & x,
    LinearLayerState & state,
    std::vector<float> & out) {
    return cuda_linear_attention_full_layer(
        w.input_norm,
        w.lin.in_proj_qkv,
        w.lin.in_proj_z,
        w.lin.in_proj_b,
        w.lin.in_proj_a,
        w.lin.conv1d,
        w.lin.a_log,
        w.lin.dt_bias,
        w.lin.norm,
        w.lin.out_proj,
        w.post_norm,
        w.mlp.gate,
        w.mlp.up,
        w.mlp.down,
        x,
        state.cuda_state,
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
        config.linear_conv_kernel_dim,
        config.rms_norm_eps,
        true,
        out);
}

bool full_attention_full_layer(
    const ModelConfig & config,
    const LayerWeights & w,
    const std::vector<float> & x,
    FullAttentionState & state,
    int pos,
    std::vector<float> & out) {
    return cuda_full_attention_full_layer(
        w.input_norm,
        w.full.q_proj,
        w.full.k_proj,
        w.full.v_proj,
        w.full.q_norm,
        w.full.k_norm,
        w.full.o_proj,
        w.post_norm,
        w.mlp.gate,
        w.mlp.up,
        w.mlp.down,
        x,
        state.cuda_state,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        state.max_seq_len,
        pos,
        config.rope_theta,
        config.partial_rotary_factor,
        config.rms_norm_eps,
        true,
        out);
}

bool rmsnorm_linear_attention_project(
    const ModelConfig & config,
    const LayerWeights & w,
    const std::vector<float> & x,
    LinearLayerState & state,
    std::vector<float> & out) {
    return cuda_rmsnorm_linear_attention_project_layer(
        w.input_norm,
        w.lin.in_proj_qkv,
        w.lin.in_proj_z,
        w.lin.in_proj_b,
        w.lin.in_proj_a,
        w.lin.conv1d,
        w.lin.a_log,
        w.lin.dt_bias,
        w.lin.norm,
        w.lin.out_proj,
        x,
        state.cuda_state,
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
        config.linear_conv_kernel_dim,
        config.rms_norm_eps,
        true,
        out);
}

bool rmsnorm_full_attention_project(
    const ModelConfig & config,
    const LayerWeights & w,
    const std::vector<float> & x,
    FullAttentionState & state,
    int pos,
    std::vector<float> & out) {
    return cuda_rmsnorm_full_attention_project_layer(
        w.input_norm,
        w.full.q_proj,
        w.full.k_proj,
        w.full.v_proj,
        w.full.q_norm,
        w.full.k_norm,
        w.full.o_proj,
        x,
        state.cuda_state,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        state.max_seq_len,
        pos,
        config.rope_theta,
        config.partial_rotary_factor,
        config.rms_norm_eps,
        true,
        out);
}

bool rmsnorm_mlp(
    const ModelConfig & config,
    const LayerWeights & w,
    const std::vector<float> & x,
    std::vector<float> & out) {
    if (!cuda_rmsnorm_mlp_enabled()) {
        return false;
    }
    return cuda_rmsnorm_mlp_layer(
        w.post_norm,
        w.mlp.gate,
        w.mlp.up,
        w.mlp.down,
        x,
        config.rms_norm_eps,
        true,
        out);
}

void linear_attention(
    const ModelConfig & config,
    const LayerWeights & w,
    const std::vector<float> & x,
    LinearLayerState & state,
    std::vector<float> & out) {
    const int key_heads = config.linear_num_key_heads;
    const int value_heads = config.linear_num_value_heads;
    const int k_dim = config.linear_key_head_dim;
    const int v_dim = config.linear_value_head_dim;
    const int key_total = key_heads * k_dim;
    const int value_total = value_heads * v_dim;
    const int conv_dim = key_total * 2 + value_total;
    const int kernel = config.linear_conv_kernel_dim;

    if (cuda_linear_attention_project_layer(
            w.lin.in_proj_qkv,
            w.lin.in_proj_z,
            w.lin.in_proj_b,
            w.lin.in_proj_a,
            w.lin.conv1d,
            w.lin.a_log,
            w.lin.dt_bias,
            w.lin.norm,
            w.lin.out_proj,
            x,
            state.cuda_state,
            key_heads,
            value_heads,
            k_dim,
            v_dim,
            kernel,
            config.rms_norm_eps,
            out)) {
        return;
    }

    std::vector<float> mixed;
    std::vector<float> z;
    std::vector<float> b;
    std::vector<float> a;
    matvec(w.lin.in_proj_qkv, x, mixed);
    matvec(w.lin.in_proj_z, x, z);
    matvec(w.lin.in_proj_b, x, b);
    matvec(w.lin.in_proj_a, x, a);

    if (cuda_linear_attention_layer(
            w.lin.conv1d,
            w.lin.a_log,
            w.lin.dt_bias,
            w.lin.norm,
            w.lin.out_proj,
            mixed,
            z,
            b,
            a,
            state.cuda_state,
            key_heads,
            value_heads,
            k_dim,
            v_dim,
            kernel,
            config.rms_norm_eps,
            out)) {
        return;
    }

    const TensorRef conv_w = w.lin.conv1d;
    std::vector<float> conv_out(conv_dim);
    for (int d = 0; d < conv_dim; ++d) {
        float * row = state.conv_state.data() + static_cast<size_t>(d) * kernel;
        for (int i = 0; i < kernel - 1; ++i) {
            row[i] = row[i + 1];
        }
        row[kernel - 1] = mixed[d];
        double sum = 0.0;
        for (int k = 0; k < kernel; ++k) {
            sum += static_cast<double>(cpu::tensor_value(conv_w, static_cast<size_t>(d) * kernel + k)) * row[k];
        }
        conv_out[d] = cpu::silu(static_cast<float>(sum));
    }

    const float * query_base = conv_out.data();
    const float * key_base = conv_out.data() + key_total;
    const float * value_base = conv_out.data() + key_total * 2;
    std::vector<float> core(value_total, 0.0f);
    const TensorRef a_log = w.lin.a_log;
    const TensorRef dt_bias = w.lin.dt_bias;

    const int repeat = value_heads / key_heads;
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(k_dim));

    for (int vh = 0; vh < value_heads; ++vh) {
        const int kh = vh / repeat;
        std::array<float, 128> q {};
        std::array<float, 128> k {};
        for (int i = 0; i < k_dim; ++i) {
            q[static_cast<size_t>(i)] = query_base[kh * k_dim + i];
            k[static_cast<size_t>(i)] = key_base[kh * k_dim + i];
        }
        cpu::l2_norm_inplace(q.data(), k_dim);
        cpu::l2_norm_inplace(k.data(), k_dim);
        for (int i = 0; i < k_dim; ++i) {
            q[static_cast<size_t>(i)] *= q_scale;
        }

        const float beta = cpu::sigmoid(b[vh]);
        const float g = -std::exp(cpu::tensor_value(a_log, static_cast<size_t>(vh))) *
                        cpu::softplus(a[vh] + cpu::tensor_value(dt_bias, static_cast<size_t>(vh)));
        const float decay = std::exp(g);
        float * rec = state.recurrent_state.data() +
                      static_cast<size_t>(vh) * k_dim * v_dim;

        for (int i = 0; i < k_dim * v_dim; ++i) {
            rec[i] *= decay;
        }

        std::array<float, 128> kv_mem {};
        for (int kd = 0; kd < k_dim; ++kd) {
            const float kval = k[static_cast<size_t>(kd)];
            const float * rec_row = rec + static_cast<size_t>(kd) * v_dim;
            for (int vd = 0; vd < v_dim; ++vd) {
                kv_mem[static_cast<size_t>(vd)] += rec_row[vd] * kval;
            }
        }

        std::array<float, 128> delta {};
        const float * value = value_base + static_cast<size_t>(vh) * v_dim;
        for (int vd = 0; vd < v_dim; ++vd) {
            delta[static_cast<size_t>(vd)] = (value[vd] - kv_mem[static_cast<size_t>(vd)]) * beta;
        }
        for (int kd = 0; kd < k_dim; ++kd) {
            float * rec_row = rec + static_cast<size_t>(kd) * v_dim;
            const float kval = k[static_cast<size_t>(kd)];
            for (int vd = 0; vd < v_dim; ++vd) {
                rec_row[vd] += kval * delta[static_cast<size_t>(vd)];
            }
        }
        float * core_head = core.data() + static_cast<size_t>(vh) * v_dim;
        for (int kd = 0; kd < k_dim; ++kd) {
            const float qval = q[static_cast<size_t>(kd)];
            const float * rec_row = rec + static_cast<size_t>(kd) * v_dim;
            for (int vd = 0; vd < v_dim; ++vd) {
                core_head[vd] += rec_row[vd] * qval;
            }
        }
    }

    const TensorRef norm_w = w.lin.norm;
    std::vector<float> gated(value_total);
    for (int vh = 0; vh < value_heads; ++vh) {
        cpu::gated_rms_norm_head(
            norm_w,
            core.data() + static_cast<size_t>(vh) * v_dim,
            z.data() + static_cast<size_t>(vh) * v_dim,
            gated.data() + static_cast<size_t>(vh) * v_dim,
            v_dim,
            config.rms_norm_eps);
    }
    matvec(w.lin.out_proj, gated, out);
}

void full_attention(
    const ModelConfig & config,
    const LayerWeights & w,
    const std::vector<float> & x,
    FullAttentionState & state,
    int pos,
    std::vector<float> & out) {
    const int n_heads = config.num_attention_heads;
    const int kv_heads = config.num_key_value_heads;
    const int head_dim = config.head_dim;
    const int q_total = n_heads * head_dim;
    const int kv_total = kv_heads * head_dim;

    if (cuda_full_attention_project_layer(
            w.full.q_proj,
            w.full.k_proj,
            w.full.v_proj,
            w.full.q_norm,
            w.full.k_norm,
            w.full.o_proj,
            x,
            state.cuda_state,
            n_heads,
            kv_heads,
            head_dim,
            state.max_seq_len,
            pos,
            config.rope_theta,
            config.partial_rotary_factor,
            config.rms_norm_eps,
            out)) {
        return;
    }

    std::vector<float> q_and_gate;
    std::vector<float> k;
    std::vector<float> v;
    matvec(w.full.q_proj, x, q_and_gate);
    matvec(w.full.k_proj, x, k);
    matvec(w.full.v_proj, x, v);

    if (cuda_full_attention_layer(
            w.full.q_norm,
            w.full.k_norm,
            w.full.o_proj,
            q_and_gate,
            k,
            v,
            state.cuda_state,
            n_heads,
            kv_heads,
            head_dim,
            state.max_seq_len,
            pos,
            config.rope_theta,
            config.partial_rotary_factor,
            config.rms_norm_eps,
            out)) {
        return;
    }

    std::vector<float> q(q_total);
    std::vector<float> gate(q_total);
    for (int h = 0; h < n_heads; ++h) {
        const int src = h * head_dim * 2;
        const int dst = h * head_dim;
        std::copy(q_and_gate.begin() + src, q_and_gate.begin() + src + head_dim, q.begin() + dst);
        std::copy(q_and_gate.begin() + src + head_dim, q_and_gate.begin() + src + head_dim * 2, gate.begin() + dst);
    }

    const TensorRef q_norm_w = w.full.q_norm;
    const TensorRef k_norm_w = w.full.k_norm;
    for (int h = 0; h < n_heads; ++h) {
        std::vector<float> tmp(q.begin() + h * head_dim, q.begin() + (h + 1) * head_dim);
        std::vector<float> normed;
        cpu::rms_norm(q_norm_w, tmp, normed, config.rms_norm_eps, true);
        std::copy(normed.begin(), normed.end(), q.begin() + h * head_dim);
        apply_rope(config, q.data() + h * head_dim, pos);
    }
    for (int h = 0; h < kv_heads; ++h) {
        std::vector<float> tmp(k.begin() + h * head_dim, k.begin() + (h + 1) * head_dim);
        std::vector<float> normed;
        cpu::rms_norm(k_norm_w, tmp, normed, config.rms_norm_eps, true);
        std::copy(normed.begin(), normed.end(), k.begin() + h * head_dim);
        apply_rope(config, k.data() + h * head_dim, pos);
    }

    std::copy(k.begin(), k.end(), state.key_cache.begin() + static_cast<size_t>(pos) * kv_total);
    std::copy(v.begin(), v.end(), state.value_cache.begin() + static_cast<size_t>(pos) * kv_total);

    std::vector<float> attn(q_total, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(static_cast<size_t>(pos) + 1);

    for (int h = 0; h < n_heads; ++h) {
        const int kh = h / (n_heads / kv_heads);
        const float * qh = q.data() + h * head_dim;
        float max_score = -std::numeric_limits<float>::infinity();
        for (int tpos = 0; tpos <= pos; ++tpos) {
            const float * khp = state.key_cache.data() + static_cast<size_t>(tpos) * kv_total + kh * head_dim;
            double dot = 0.0;
            for (int d = 0; d < head_dim; ++d) {
                dot += static_cast<double>(qh[d]) * khp[d];
            }
            scores[static_cast<size_t>(tpos)] = static_cast<float>(dot) * scale;
            max_score = std::max(max_score, scores[static_cast<size_t>(tpos)]);
        }
        double denom = 0.0;
        for (int tpos = 0; tpos <= pos; ++tpos) {
            const float e = std::exp(scores[static_cast<size_t>(tpos)] - max_score);
            scores[static_cast<size_t>(tpos)] = e;
            denom += e;
        }
        float * ah = attn.data() + h * head_dim;
        for (int tpos = 0; tpos <= pos; ++tpos) {
            const float prob = scores[static_cast<size_t>(tpos)] / static_cast<float>(denom);
            const float * vh = state.value_cache.data() + static_cast<size_t>(tpos) * kv_total + kh * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                ah[d] += prob * vh[d];
            }
        }
    }

    for (int i = 0; i < q_total; ++i) {
        attn[i] *= cpu::sigmoid(gate[i]);
    }
    matvec(w.full.o_proj, attn, out);
}

void mlp(
    const LayerWeights & w,
    const std::vector<float> & x,
    std::vector<float> & out) {
    const TensorRef gate_w = w.mlp.gate;
    const TensorRef up_w = w.mlp.up;
    const TensorRef down_w = w.mlp.down;
    if (cuda_mlp_layer(gate_w, up_w, down_w, x, out)) {
        return;
    }

    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> prod;
    matvec(gate_w, x, gate);
    matvec(up_w, x, up);
    prod.resize(gate.size());
    for (size_t i = 0; i < gate.size(); ++i) {
        prod[i] = cpu::silu(gate[i]) * up[i];
    }
    matvec(down_w, prod, out);
}

int argmax_logits(const ModelParams & params, const std::vector<float> & hidden) {
    const TensorRef emb = params.embed_tokens;
    const int vocab = static_cast<int>(emb.info->shape[0]);
    const int hidden_size = static_cast<int>(emb.info->shape[1]);
    if (static_cast<int>(hidden.size()) != hidden_size) {
        throw std::runtime_error("logits hidden size 不匹配。");
    }

    int best_id = 0;
    if (cuda_argmax_matvec(emb, hidden, best_id)) {
        return best_id;
    }

    std::vector<float> logits;
    matvec(emb, hidden, logits);

    float best = logits[0];
    for (int token = 1; token < vocab; ++token) {
        if (logits[token] > best) {
            best = logits[token];
            best_id = token;
        }
    }
    return best_id;
}

} // namespace ops

} // namespace llm_inference
