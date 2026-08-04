#include "llm_inference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace llm_inference {


struct LinearLayerState {
    std::vector<float> conv_state;
    std::vector<float> recurrent_state;
    void * cuda_state = nullptr;

    LinearLayerState() = default;
    LinearLayerState(const LinearLayerState &) = delete;
    LinearLayerState & operator=(const LinearLayerState &) = delete;
    LinearLayerState(LinearLayerState && other) noexcept
        : conv_state(std::move(other.conv_state)),
          recurrent_state(std::move(other.recurrent_state)),
          cuda_state(other.cuda_state) {
        other.cuda_state = nullptr;
    }
    LinearLayerState & operator=(LinearLayerState && other) noexcept {
        if (this != &other) {
            cuda_free_linear_attention_state(cuda_state);
            conv_state = std::move(other.conv_state);
            recurrent_state = std::move(other.recurrent_state);
            cuda_state = other.cuda_state;
            other.cuda_state = nullptr;
        }
        return *this;
    }
    ~LinearLayerState() {
        cuda_free_linear_attention_state(cuda_state);
    }
};

struct FullAttentionState {
    std::vector<float> key_cache;
    std::vector<float> value_cache;
    int max_seq_len = 0;
    void * cuda_state = nullptr;

    FullAttentionState() = default;
    FullAttentionState(const FullAttentionState &) = delete;
    FullAttentionState & operator=(const FullAttentionState &) = delete;
    FullAttentionState(FullAttentionState && other) noexcept
        : key_cache(std::move(other.key_cache)),
          value_cache(std::move(other.value_cache)),
          max_seq_len(other.max_seq_len),
          cuda_state(other.cuda_state) {
        other.cuda_state = nullptr;
    }
    FullAttentionState & operator=(FullAttentionState && other) noexcept {
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
    ~FullAttentionState() {
        cuda_free_full_attention_state(cuda_state);
    }
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
            state.full[layer].max_seq_len = max_seq_len;
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
        int next = 0;
        if (generate_next_device(prompt_ids, state, timing, next)) {
            return next;
        }
        std::vector<float> hidden;
        const auto prefill_start = Clock::now();
        for (int token : prompt_ids) {
            hidden = forward_token(token, state);
        }
        timing.prefill_s = elapsed_s(prefill_start);
        return argmax_logits(hidden, timing);
    }

    int decode_one(int token, RunState & state, Timing & timing) const {
        int next = 0;
        if (decode_one_device(token, state, timing, next)) {
            return next;
        }
        const auto decode_start = Clock::now();
        std::vector<float> hidden = forward_token(token, state);
        next = argmax_logits(hidden, timing);
        timing.decode_total_s += elapsed_s(decode_start);
        return next;
    }

    bool generate_sequence_device(
        const std::vector<int> & prompt_ids,
        RunState & state,
        int max_new_tokens,
        int eos_token_id,
        Timing & timing,
        std::vector<int> & generated) const {
        void * generated_device = cuda_generated_token_buffer(max_new_tokens);
        if (!generated_device) {
            return false;
        }

        const auto prefill_start = Clock::now();
        const void * hidden = nullptr;
        {
            std::vector<void *> linear_cuda_states(state.linear.size(), nullptr);
            std::vector<void *> full_cuda_states(state.full.size(), nullptr);
            std::vector<int> full_max_seq_lens(state.full.size(), 0);
            for (size_t i = 0; i < state.linear.size(); ++i) {
                linear_cuda_states[i] = state.linear[i].cuda_state;
            }
            for (size_t i = 0; i < state.full.size(); ++i) {
                full_cuda_states[i] = state.full[i].cuda_state;
                full_max_seq_lens[i] = state.full[i].max_seq_len;
            }
            hidden = cuda_prefill_batch(config_, weights_, prompt_ids, linear_cuda_states, full_cuda_states, full_max_seq_lens, state.seq_len);
            if (hidden) {
                for (size_t i = 0; i < state.linear.size(); ++i) {
                    state.linear[i].cuda_state = linear_cuda_states[i];
                }
                for (size_t i = 0; i < state.full.size(); ++i) {
                    state.full[i].cuda_state = full_cuda_states[i];
                }
            } else {
                for (int token : prompt_ids) {
                    hidden = forward_token_device(token, state);
                    if (!hidden) {
                        return false;
                    }
                }
            }
        }
        if (!cuda_synchronize_device()) {
            return false;
        }
        timing.prefill_s = elapsed_s(prefill_start);

        const TensorRef emb = t("model.language_model.embed_tokens.weight");
        const TensorRef norm = t("model.language_model.norm.weight");
        const int hidden_size = static_cast<int>(emb.info->shape[1]);
        const auto decode_start = Clock::now();
        for (int i = 0; i < max_new_tokens; ++i) {
            int * token_slot = static_cast<int *>(generated_device) + i;
            if (!cuda_final_norm_argmax_to_device(
                    norm,
                    emb,
                    hidden,
                    hidden_size,
                    config_.rms_norm_eps,
                    true,
                    token_slot)) {
                return false;
            }
            if (i + 1 < max_new_tokens) {
                hidden = forward_token_device_from_device(token_slot, state);
                if (!hidden) {
                    return false;
                }
            }
        }
        if (!cuda_copy_generated_tokens_to_host(generated_device, max_new_tokens, generated)) {
            return false;
        }
        timing.decode_total_s += elapsed_s(decode_start);
        if (eos_token_id >= 0) {
            const auto eos = std::find(generated.begin(), generated.end(), eos_token_id);
            if (eos != generated.end()) {
                generated.resize(static_cast<size_t>(std::distance(generated.begin(), eos)) + 1);
            }
        }
        timing.generated_tokens += static_cast<int>(generated.size());
        timing.generated_ids.insert(timing.generated_ids.end(), generated.begin(), generated.end());
        return true;
    }

private:
    TensorRef t(const std::string & name) const {
        return tensor_ref(weights_, name);
    }

    bool generate_next_device(const std::vector<int> & prompt_ids, RunState & state, Timing & timing, int & next) const {
        const auto prefill_start = Clock::now();
        const void * hidden = nullptr;
        for (int token : prompt_ids) {
            hidden = forward_token_device(token, state);
            if (!hidden) {
                return false;
            }
        }
        timing.prefill_s = elapsed_s(prefill_start);
        return argmax_logits_device(hidden, timing, next);
    }

    bool decode_one_device(int token, RunState & state, Timing & timing, int & next) const {
        const auto decode_start = Clock::now();
        const void * hidden = forward_token_device(token, state);
        if (!hidden) {
            return false;
        }
        if (!argmax_logits_device(hidden, timing, next)) {
            return false;
        }
        timing.decode_total_s += elapsed_s(decode_start);
        return true;
    }

    const void * forward_token_device(int token, RunState & state) const {
        const int hidden_size = config_.hidden_size;
        void * current = cuda_token_hidden_buffer(0, hidden_size);
        void * next = cuda_token_hidden_buffer(1, hidden_size);
        if (!current || !next || !cuda_embedding_lookup_device(t("model.language_model.embed_tokens.weight"), token, current)) {
            return nullptr;
        }
        return forward_token_device_layers(current, next, state);
    }

    const void * forward_token_device_from_device(const void * device_token, RunState & state) const {
        const int hidden_size = config_.hidden_size;
        void * current = cuda_token_hidden_buffer(0, hidden_size);
        void * next = cuda_token_hidden_buffer(1, hidden_size);
        if (!current || !next || !cuda_embedding_lookup_device_token(t("model.language_model.embed_tokens.weight"), device_token, current)) {
            return nullptr;
        }
        return forward_token_device_layers(current, next, state);
    }

    const void * forward_token_device_layers(void * current, void * next, RunState & state) const {
        const int hidden_size = config_.hidden_size;
        const int pos = state.seq_len;

        for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
            const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
            bool ok = false;
            if (config_.layer_types[layer] == "linear_attention") {
                ok = cuda_linear_attention_full_layer_device(
                    t(prefix + "input_layernorm.weight"),
                    t(prefix + "linear_attn.in_proj_qkv.weight"),
                    t(prefix + "linear_attn.in_proj_z.weight"),
                    t(prefix + "linear_attn.in_proj_b.weight"),
                    t(prefix + "linear_attn.in_proj_a.weight"),
                    t(prefix + "linear_attn.conv1d.weight"),
                    t(prefix + "linear_attn.A_log"),
                    t(prefix + "linear_attn.dt_bias"),
                    t(prefix + "linear_attn.norm.weight"),
                    t(prefix + "linear_attn.out_proj.weight"),
                    t(prefix + "post_attention_layernorm.weight"),
                    t(prefix + "mlp.gate_proj.weight"),
                    t(prefix + "mlp.up_proj.weight"),
                    t(prefix + "mlp.down_proj.weight"),
                    current,
                    next,
                    hidden_size,
                    state.linear[layer].cuda_state,
                    config_.linear_num_key_heads,
                    config_.linear_num_value_heads,
                    config_.linear_key_head_dim,
                    config_.linear_value_head_dim,
                    config_.linear_conv_kernel_dim,
                    config_.rms_norm_eps,
                    true);
            } else {
                ok = cuda_full_attention_full_layer_device(
                    t(prefix + "input_layernorm.weight"),
                    t(prefix + "self_attn.q_proj.weight"),
                    t(prefix + "self_attn.k_proj.weight"),
                    t(prefix + "self_attn.v_proj.weight"),
                    t(prefix + "self_attn.q_norm.weight"),
                    t(prefix + "self_attn.k_norm.weight"),
                    t(prefix + "self_attn.o_proj.weight"),
                    t(prefix + "post_attention_layernorm.weight"),
                    t(prefix + "mlp.gate_proj.weight"),
                    t(prefix + "mlp.up_proj.weight"),
                    t(prefix + "mlp.down_proj.weight"),
                    current,
                    next,
                    hidden_size,
                    state.full[layer].cuda_state,
                    config_.num_attention_heads,
                    config_.num_key_value_heads,
                    config_.head_dim,
                    state.full[layer].max_seq_len,
                    pos,
                    config_.rope_theta,
                    config_.partial_rotary_factor,
                    config_.rms_norm_eps,
                    true);
            }
            if (!ok) {
                throw std::runtime_error("CUDA device full-layer path 失败，layer=" + std::to_string(layer));
            }
            std::swap(current, next);
        }

        state.seq_len += 1;
        return current;
    }

    std::vector<float> forward_token(int token, RunState & state) const {
        std::vector<float> x;
        embedding_lookup(t("model.language_model.embed_tokens.weight"), token, x);
        const int pos = state.seq_len;

        for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
            const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
            if (config_.layer_types[layer] == "linear_attention") {
                std::vector<float> layer_out;
                if (cuda_linear_attention_full_layer(
                        t(prefix + "input_layernorm.weight"),
                        t(prefix + "linear_attn.in_proj_qkv.weight"),
                        t(prefix + "linear_attn.in_proj_z.weight"),
                        t(prefix + "linear_attn.in_proj_b.weight"),
                        t(prefix + "linear_attn.in_proj_a.weight"),
                        t(prefix + "linear_attn.conv1d.weight"),
                        t(prefix + "linear_attn.A_log"),
                        t(prefix + "linear_attn.dt_bias"),
                        t(prefix + "linear_attn.norm.weight"),
                        t(prefix + "linear_attn.out_proj.weight"),
                        t(prefix + "post_attention_layernorm.weight"),
                        t(prefix + "mlp.gate_proj.weight"),
                        t(prefix + "mlp.up_proj.weight"),
                        t(prefix + "mlp.down_proj.weight"),
                        x,
                        state.linear[layer].cuda_state,
                        config_.linear_num_key_heads,
                        config_.linear_num_value_heads,
                        config_.linear_key_head_dim,
                        config_.linear_value_head_dim,
                        config_.linear_conv_kernel_dim,
                        config_.rms_norm_eps,
                        true,
                        layer_out)) {
                    x = std::move(layer_out);
                    continue;
                }
            } else {
                std::vector<float> layer_out;
                if (cuda_full_attention_full_layer(
                        t(prefix + "input_layernorm.weight"),
                        t(prefix + "self_attn.q_proj.weight"),
                        t(prefix + "self_attn.k_proj.weight"),
                        t(prefix + "self_attn.v_proj.weight"),
                        t(prefix + "self_attn.q_norm.weight"),
                        t(prefix + "self_attn.k_norm.weight"),
                        t(prefix + "self_attn.o_proj.weight"),
                        t(prefix + "post_attention_layernorm.weight"),
                        t(prefix + "mlp.gate_proj.weight"),
                        t(prefix + "mlp.up_proj.weight"),
                        t(prefix + "mlp.down_proj.weight"),
                        x,
                        state.full[layer].cuda_state,
                        config_.num_attention_heads,
                        config_.num_key_value_heads,
                        config_.head_dim,
                        state.full[layer].max_seq_len,
                        pos,
                        config_.rope_theta,
                        config_.partial_rotary_factor,
                        config_.rms_norm_eps,
                        true,
                        layer_out)) {
                    x = std::move(layer_out);
                    continue;
                }
            }

            std::vector<float> residual = x;
            std::vector<float> normed;

            std::vector<float> mixer_out;
            bool mixer_done = false;
            if (config_.layer_types[layer] == "linear_attention") {
                mixer_done = cuda_rmsnorm_linear_attention_project_layer(
                    t(prefix + "input_layernorm.weight"),
                    t(prefix + "linear_attn.in_proj_qkv.weight"),
                    t(prefix + "linear_attn.in_proj_z.weight"),
                    t(prefix + "linear_attn.in_proj_b.weight"),
                    t(prefix + "linear_attn.in_proj_a.weight"),
                    t(prefix + "linear_attn.conv1d.weight"),
                    t(prefix + "linear_attn.A_log"),
                    t(prefix + "linear_attn.dt_bias"),
                    t(prefix + "linear_attn.norm.weight"),
                    t(prefix + "linear_attn.out_proj.weight"),
                    x,
                    state.linear[layer].cuda_state,
                    config_.linear_num_key_heads,
                    config_.linear_num_value_heads,
                    config_.linear_key_head_dim,
                    config_.linear_value_head_dim,
                    config_.linear_conv_kernel_dim,
                    config_.rms_norm_eps,
                    true,
                    mixer_out);
            } else {
                mixer_done = cuda_rmsnorm_full_attention_project_layer(
                    t(prefix + "input_layernorm.weight"),
                    t(prefix + "self_attn.q_proj.weight"),
                    t(prefix + "self_attn.k_proj.weight"),
                    t(prefix + "self_attn.v_proj.weight"),
                    t(prefix + "self_attn.q_norm.weight"),
                    t(prefix + "self_attn.k_norm.weight"),
                    t(prefix + "self_attn.o_proj.weight"),
                    x,
                    state.full[layer].cuda_state,
                    config_.num_attention_heads,
                    config_.num_key_value_heads,
                    config_.head_dim,
                    state.full[layer].max_seq_len,
                    pos,
                    config_.rope_theta,
                    config_.partial_rotary_factor,
                    config_.rms_norm_eps,
                    true,
                    mixer_out);
            }
            if (!mixer_done) {
                rms_norm(t(prefix + "input_layernorm.weight"), x, normed, config_.rms_norm_eps, true);
                if (config_.layer_types[layer] == "linear_attention") {
                    linear_attention_layer(prefix, normed, state.linear[layer], mixer_out);
                } else {
                    full_attention_layer(prefix, normed, state.full[layer], pos, mixer_out);
                }
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

        if (cuda_linear_attention_project_layer(
                t(prefix + "linear_attn.in_proj_qkv.weight"),
                t(prefix + "linear_attn.in_proj_z.weight"),
                t(prefix + "linear_attn.in_proj_b.weight"),
                t(prefix + "linear_attn.in_proj_a.weight"),
                t(prefix + "linear_attn.conv1d.weight"),
                t(prefix + "linear_attn.A_log"),
                t(prefix + "linear_attn.dt_bias"),
                t(prefix + "linear_attn.norm.weight"),
                t(prefix + "linear_attn.out_proj.weight"),
                x,
                state.cuda_state,
                key_heads,
                value_heads,
                k_dim,
                v_dim,
                kernel,
                config_.rms_norm_eps,
                out)) {
            return;
        }

        std::vector<float> mixed;
        std::vector<float> z;
        std::vector<float> b;
        std::vector<float> a;
        matvec(t(prefix + "linear_attn.in_proj_qkv.weight"), x, mixed);
        matvec(t(prefix + "linear_attn.in_proj_z.weight"), x, z);
        matvec(t(prefix + "linear_attn.in_proj_b.weight"), x, b);
        matvec(t(prefix + "linear_attn.in_proj_a.weight"), x, a);

        if (cuda_linear_attention_layer(
                t(prefix + "linear_attn.conv1d.weight"),
                t(prefix + "linear_attn.A_log"),
                t(prefix + "linear_attn.dt_bias"),
                t(prefix + "linear_attn.norm.weight"),
                t(prefix + "linear_attn.out_proj.weight"),
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
                config_.rms_norm_eps,
                out)) {
            return;
        }

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

        if (cuda_full_attention_project_layer(
                t(prefix + "self_attn.q_proj.weight"),
                t(prefix + "self_attn.k_proj.weight"),
                t(prefix + "self_attn.v_proj.weight"),
                t(prefix + "self_attn.q_norm.weight"),
                t(prefix + "self_attn.k_norm.weight"),
                t(prefix + "self_attn.o_proj.weight"),
                x,
                state.cuda_state,
                n_heads,
                kv_heads,
                head_dim,
                state.max_seq_len,
                pos,
                config_.rope_theta,
                config_.partial_rotary_factor,
                config_.rms_norm_eps,
                out)) {
            return;
        }

        std::vector<float> q_and_gate;
        std::vector<float> k;
        std::vector<float> v;
        matvec(t(prefix + "self_attn.q_proj.weight"), x, q_and_gate);
        matvec(t(prefix + "self_attn.k_proj.weight"), x, k);
        matvec(t(prefix + "self_attn.v_proj.weight"), x, v);

        if (cuda_full_attention_layer(
                t(prefix + "self_attn.q_norm.weight"),
                t(prefix + "self_attn.k_norm.weight"),
                t(prefix + "self_attn.o_proj.weight"),
                q_and_gate,
                k,
                v,
                state.cuda_state,
                n_heads,
                kv_heads,
                head_dim,
                state.max_seq_len,
                pos,
                config_.rope_theta,
                config_.partial_rotary_factor,
                config_.rms_norm_eps,
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
        int best_id = 0;
        if (cuda_argmax_matvec(emb, hidden, best_id)) {
            timing.logits_s += elapsed_s(start);
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
        timing.logits_s += elapsed_s(start);
        return best_id;
    }

    bool argmax_logits_device(const void * device_hidden, Timing & timing, int & best_id) const {
        const TensorRef emb = t("model.language_model.embed_tokens.weight");
        const int hidden_size = static_cast<int>(emb.info->shape[1]);
        const auto start = Clock::now();
        if (!cuda_final_norm_argmax_device(
                t("model.language_model.norm.weight"),
                emb,
                device_hidden,
                hidden_size,
                config_.rms_norm_eps,
                true,
                best_id)) {
            return false;
        }
        timing.logits_s += elapsed_s(start);
        return true;
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
    if (args.greedy && model.generate_sequence_device(input_ids, state, args.max_new_tokens, config.eos_token_id, timing, generated)) {
        return generated;
    }
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
