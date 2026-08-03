#include "llm_inference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace llm_inference {


struct LinearLayerState {
    std::vector<float> conv_state;
    std::vector<float> recurrent_state;
};

struct FullAttentionState {
    std::vector<float> key_cache;
    std::vector<float> value_cache;
};

struct RunState {
    int seq_len = 0;
    std::vector<LinearLayerState> linear;
    std::vector<FullAttentionState> full;
};

void validate_qwen_tensors(const ModelWeights & weights, const ModelConfig & config) {
    const std::string root = "model.language_model.";
    for (const std::string & name : {
             root + "embed_tokens.weight",
             root + "norm.weight",
         }) {
        if (!has_tensor(weights, name)) {
            throw std::runtime_error("缺少 tensor：" + name);
        }
    }

    for (int layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = root + "layers." + std::to_string(layer) + ".";
        for (const std::string & name : {
                 prefix + "input_layernorm.weight",
                 prefix + "post_attention_layernorm.weight",
                 prefix + "mlp.gate_proj.weight",
                 prefix + "mlp.up_proj.weight",
                 prefix + "mlp.down_proj.weight",
             }) {
            if (!has_tensor(weights, name)) {
                throw std::runtime_error("缺少 tensor：" + name);
            }
        }
        if (config.layer_types[layer] == "linear_attention") {
            for (const std::string & name : {
                     prefix + "linear_attn.A_log",
                     prefix + "linear_attn.norm.weight",
                     prefix + "linear_attn.conv1d.weight",
                     prefix + "linear_attn.dt_bias",
                     prefix + "linear_attn.in_proj_a.weight",
                     prefix + "linear_attn.in_proj_b.weight",
                     prefix + "linear_attn.in_proj_qkv.weight",
                     prefix + "linear_attn.in_proj_z.weight",
                     prefix + "linear_attn.out_proj.weight",
                 }) {
                if (!has_tensor(weights, name)) {
                    throw std::runtime_error("缺少 tensor：" + name);
                }
            }
        } else {
            for (const std::string & name : {
                     prefix + "self_attn.q_proj.weight",
                     prefix + "self_attn.k_proj.weight",
                     prefix + "self_attn.v_proj.weight",
                     prefix + "self_attn.o_proj.weight",
                     prefix + "self_attn.q_norm.weight",
                     prefix + "self_attn.k_norm.weight",
                 }) {
                if (!has_tensor(weights, name)) {
                    throw std::runtime_error("缺少 tensor：" + name);
                }
            }
        }
    }
}

RunState make_run_state(const ModelConfig & config, int max_seq_len) {
    RunState state;
    state.linear.resize(config.num_hidden_layers);
    state.full.resize(config.num_hidden_layers);
    const int conv_dim = config.linear_key_head_dim * config.linear_num_key_heads * 2 +
                         config.linear_value_head_dim * config.linear_num_value_heads;
    for (int layer = 0; layer < config.num_hidden_layers; ++layer) {
        if (config.layer_types[layer] == "linear_attention") {
            state.linear[layer].conv_state.assign(static_cast<size_t>(conv_dim) * config.linear_conv_kernel_dim, 0.0f);
            state.linear[layer].recurrent_state.assign(
                static_cast<size_t>(config.linear_num_value_heads) *
                    config.linear_key_head_dim *
                    config.linear_value_head_dim,
                0.0f);
        } else {
            state.full[layer].key_cache.assign(
                static_cast<size_t>(max_seq_len) * config.num_key_value_heads * config.head_dim,
                0.0f);
            state.full[layer].value_cache.assign(
                static_cast<size_t>(max_seq_len) * config.num_key_value_heads * config.head_dim,
                0.0f);
        }
    }
    return state;
}

namespace {

class NativeQwen {
public:
    NativeQwen(const ModelConfig & config, const ModelWeights & weights)
        : config_(config), weights_(weights) {}

    int generate_next(const std::vector<int> & prompt_ids, RunState & state, Timing & timing) const {
        std::vector<float> hidden;
        const auto prefill_start = Clock::now();
        for (int token : prompt_ids) {
            hidden = forward_token(token, state);
        }
        timing.prefill_s = elapsed_s(prefill_start);
        return argmax_logits(hidden, timing);
    }

    int decode_one(int token, RunState & state, Timing & timing) const {
        const auto decode_start = Clock::now();
        std::vector<float> hidden = forward_token(token, state);
        const int next = argmax_logits(hidden, timing);
        timing.decode_total_s += elapsed_s(decode_start);
        return next;
    }

private:
    TensorRef t(const std::string & name) const {
        return tensor_ref(weights_, name);
    }

    std::vector<float> forward_token(int token, RunState & state) const {
        std::vector<float> x;
        embedding_lookup(t("model.language_model.embed_tokens.weight"), token, x);
        const int pos = state.seq_len;

        for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
            const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
            std::vector<float> residual = x;
            std::vector<float> normed;
            rms_norm(t(prefix + "input_layernorm.weight"), x, normed, config_.rms_norm_eps, true);

            std::vector<float> mixer_out;
            if (config_.layer_types[layer] == "linear_attention") {
                linear_attention_layer(prefix, normed, state.linear[layer], mixer_out);
            } else {
                full_attention_layer(prefix, normed, state.full[layer], pos, mixer_out);
            }
            x = residual;
            add_inplace(x, mixer_out);

            residual = x;
            if (!cuda_rmsnorm_mlp_enabled() || !cuda_rmsnorm_mlp_layer(
                    t(prefix + "post_attention_layernorm.weight"),
                    t(prefix + "mlp.gate_proj.weight"),
                    t(prefix + "mlp.up_proj.weight"),
                    t(prefix + "mlp.down_proj.weight"),
                    x,
                    config_.rms_norm_eps,
                    true,
                    mixer_out)) {
                rms_norm(t(prefix + "post_attention_layernorm.weight"), x, normed, config_.rms_norm_eps, true);
                mlp_layer(prefix, normed, mixer_out);
            }
            x = residual;
            add_inplace(x, mixer_out);
        }

        std::vector<float> normed;
        rms_norm(t("model.language_model.norm.weight"), x, normed, config_.rms_norm_eps, true);
        state.seq_len += 1;
        return normed;
    }

    void mlp_layer(const std::string & prefix, const std::vector<float> & x, std::vector<float> & out) const {
        const TensorRef gate_w = t(prefix + "mlp.gate_proj.weight");
        const TensorRef up_w = t(prefix + "mlp.up_proj.weight");
        const TensorRef down_w = t(prefix + "mlp.down_proj.weight");
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
            prod[i] = silu(gate[i]) * up[i];
        }
        matvec(down_w, prod, out);
    }

    void linear_attention_layer(
        const std::string & prefix,
        const std::vector<float> & x,
        LinearLayerState & state,
        std::vector<float> & out) const {
        const int key_heads = config_.linear_num_key_heads;
        const int value_heads = config_.linear_num_value_heads;
        const int k_dim = config_.linear_key_head_dim;
        const int v_dim = config_.linear_value_head_dim;
        const int key_total = key_heads * k_dim;
        const int value_total = value_heads * v_dim;
        const int conv_dim = key_total * 2 + value_total;
        const int kernel = config_.linear_conv_kernel_dim;

        std::vector<float> mixed;
        std::vector<float> z;
        std::vector<float> b;
        std::vector<float> a;
        matvec(t(prefix + "linear_attn.in_proj_qkv.weight"), x, mixed);
        matvec(t(prefix + "linear_attn.in_proj_z.weight"), x, z);
        matvec(t(prefix + "linear_attn.in_proj_b.weight"), x, b);
        matvec(t(prefix + "linear_attn.in_proj_a.weight"), x, a);

        const TensorRef conv_w = t(prefix + "linear_attn.conv1d.weight");
        std::vector<float> conv_out(conv_dim);
        for (int d = 0; d < conv_dim; ++d) {
            float * row = state.conv_state.data() + static_cast<size_t>(d) * kernel;
            for (int i = 0; i < kernel - 1; ++i) {
                row[i] = row[i + 1];
            }
            row[kernel - 1] = mixed[d];
            double sum = 0.0;
            for (int k = 0; k < kernel; ++k) {
                sum += static_cast<double>(tensor_value(conv_w, static_cast<size_t>(d) * kernel + k)) * row[k];
            }
            conv_out[d] = silu(static_cast<float>(sum));
        }

        const float * query_base = conv_out.data();
        const float * key_base = conv_out.data() + key_total;
        const float * value_base = conv_out.data() + key_total * 2;
        std::vector<float> core(value_total, 0.0f);
        const TensorRef a_log = t(prefix + "linear_attn.A_log");
        const TensorRef dt_bias = t(prefix + "linear_attn.dt_bias");

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
            l2_norm_inplace(q.data(), k_dim);
            l2_norm_inplace(k.data(), k_dim);
            for (int i = 0; i < k_dim; ++i) {
                q[static_cast<size_t>(i)] *= q_scale;
            }

            const float beta = sigmoid(b[vh]);
            const float g = -std::exp(tensor_value(a_log, static_cast<size_t>(vh))) *
                            softplus(a[vh] + tensor_value(dt_bias, static_cast<size_t>(vh)));
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

        const TensorRef norm_w = t(prefix + "linear_attn.norm.weight");
        std::vector<float> gated(value_total);
        for (int vh = 0; vh < value_heads; ++vh) {
            gated_rms_norm_head(
                norm_w,
                core.data() + static_cast<size_t>(vh) * v_dim,
                z.data() + static_cast<size_t>(vh) * v_dim,
                gated.data() + static_cast<size_t>(vh) * v_dim,
                v_dim,
                config_.rms_norm_eps);
        }
        matvec(t(prefix + "linear_attn.out_proj.weight"), gated, out);
    }

    void apply_rope(float * vec, int pos) const {
        const int rotary_dim = static_cast<int>(config_.head_dim * config_.partial_rotary_factor);
        const int half = rotary_dim / 2;
        for (int i = 0; i < half; ++i) {
            const float inv_freq = 1.0f / std::pow(config_.rope_theta, static_cast<float>(2 * i) / rotary_dim);
            const float angle = static_cast<float>(pos) * inv_freq;
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            const float x1 = vec[i];
            const float x2 = vec[i + half];
            vec[i] = x1 * c - x2 * s;
            vec[i + half] = x2 * c + x1 * s;
        }
    }

    void full_attention_layer(
        const std::string & prefix,
        const std::vector<float> & x,
        FullAttentionState & state,
        int pos,
        std::vector<float> & out) const {
        const int n_heads = config_.num_attention_heads;
        const int kv_heads = config_.num_key_value_heads;
        const int head_dim = config_.head_dim;
        const int q_total = n_heads * head_dim;
        const int kv_total = kv_heads * head_dim;

        std::vector<float> q_and_gate;
        std::vector<float> k;
        std::vector<float> v;
        matvec(t(prefix + "self_attn.q_proj.weight"), x, q_and_gate);
        matvec(t(prefix + "self_attn.k_proj.weight"), x, k);
        matvec(t(prefix + "self_attn.v_proj.weight"), x, v);

        std::vector<float> q(q_total);
        std::vector<float> gate(q_total);
        for (int h = 0; h < n_heads; ++h) {
            const int src = h * head_dim * 2;
            const int dst = h * head_dim;
            std::copy(q_and_gate.begin() + src, q_and_gate.begin() + src + head_dim, q.begin() + dst);
            std::copy(q_and_gate.begin() + src + head_dim, q_and_gate.begin() + src + head_dim * 2, gate.begin() + dst);
        }

        const TensorRef q_norm_w = t(prefix + "self_attn.q_norm.weight");
        const TensorRef k_norm_w = t(prefix + "self_attn.k_norm.weight");
        for (int h = 0; h < n_heads; ++h) {
            std::vector<float> tmp(q.begin() + h * head_dim, q.begin() + (h + 1) * head_dim);
            std::vector<float> normed;
            rms_norm(q_norm_w, tmp, normed, config_.rms_norm_eps, true);
            std::copy(normed.begin(), normed.end(), q.begin() + h * head_dim);
            apply_rope(q.data() + h * head_dim, pos);
        }
        for (int h = 0; h < kv_heads; ++h) {
            std::vector<float> tmp(k.begin() + h * head_dim, k.begin() + (h + 1) * head_dim);
            std::vector<float> normed;
            rms_norm(k_norm_w, tmp, normed, config_.rms_norm_eps, true);
            std::copy(normed.begin(), normed.end(), k.begin() + h * head_dim);
            apply_rope(k.data() + h * head_dim, pos);
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
            attn[i] *= sigmoid(gate[i]);
        }
        matvec(t(prefix + "self_attn.o_proj.weight"), attn, out);
    }

    int argmax_logits(const std::vector<float> & hidden, Timing & timing) const {
        const TensorRef emb = t("model.language_model.embed_tokens.weight");
        const int vocab = static_cast<int>(emb.info->shape[0]);
        const int hidden_size = static_cast<int>(emb.info->shape[1]);
        if (static_cast<int>(hidden.size()) != hidden_size) {
            throw std::runtime_error("logits hidden size 不匹配。");
        }

        const auto start = Clock::now();
        std::vector<float> logits;
        matvec(emb, hidden, logits);

        int best_id = 0;
        float best = logits[0];
        for (int token = 1; token < vocab; ++token) {
            if (logits[token] > best) {
                best = logits[token];
                best_id = token;
            }
        }
        timing.logits_s += elapsed_s(start);
        return best_id;
    }

    const ModelConfig & config_;
    const ModelWeights & weights_;

};

} // namespace

std::vector<int> run_generation(
    const ModelConfig & config,
    const ModelWeights & weights,
    const Args & args,
    const std::vector<int> & input_ids,
    Timing & timing) {
    NativeQwen model(config, weights);
    RunState state = make_run_state(config, timing.input_tokens + args.max_new_tokens + 4);
    std::vector<int> generated;
    int next = model.generate_next(input_ids, state, timing);
    for (int i = 0; i < args.max_new_tokens; ++i) {
        generated.push_back(next);
        timing.generated_ids.push_back(next);
        timing.generated_tokens += 1;
        if (next == config.eos_token_id) {
            break;
        }
        if (i + 1 < args.max_new_tokens) {
            next = model.decode_one(next, state, timing);
        }
    }
    return generated;
}

} // namespace llm_inference
