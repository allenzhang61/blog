#include "QwenModel.h"

#include "../kernels/cpu/cpu_ops.h"
#include "../kernels/cuda/cuda_ops.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace llm_inference {

RunState make_run_state(const ModelConfig & config, int max_seq_len) {
    RunState state;
    state.linear.resize(config.text.num_hidden_layers);
    state.full.resize(config.text.num_hidden_layers);
    const int conv_dim = config.text.linear_key_head_dim * config.text.linear_num_key_heads * 2 +
                         config.text.linear_value_head_dim * config.text.linear_num_value_heads;
    for (int layer = 0; layer < config.text.num_hidden_layers; ++layer) {
        if (config.text.layer_types[layer] == "linear_attention") {
            state.linear[layer].conv_state.assign(static_cast<size_t>(conv_dim) * config.text.linear_conv_kernel_dim, 0.0f);
            state.linear[layer].recurrent_state.assign(
                static_cast<size_t>(config.text.linear_num_value_heads) *
                    config.text.linear_key_head_dim *
                    config.text.linear_value_head_dim,
                0.0f);
        } else {
            state.full[layer].max_seq_len = max_seq_len;
            state.full[layer].key_cache.assign(
                static_cast<size_t>(max_seq_len) * config.text.num_key_value_heads * config.text.head_dim,
                0.0f);
            state.full[layer].value_cache.assign(
                static_cast<size_t>(max_seq_len) * config.text.num_key_value_heads * config.text.head_dim,
                0.0f);
        }
    }
    return state;
}

QwenModel::QwenModel(const ModelConfig & config, const ModelWeights & weights, Device device)
    : config_(config), weights_(weights), device_(device), params_(parse_model_params(weights, config)) {
    if (device_ == Device::CUDA && !cuda_cublas_enabled()) {
        throw std::runtime_error("请求 device=cuda，但当前构建未启用 CUDA/cuBLAS 或设备不可用（严格设备匹配，不回退 CPU）。");
    }
}

int QwenModel::generate_next(const std::vector<int> & prompt_ids, RunState & state, Timing & timing) const {
    if (device_ == Device::CUDA) {
        return generate_next_device(prompt_ids, state, timing);
    }
    std::vector<float> hidden;
    const auto prefill_start = Clock::now();
    for (int token : prompt_ids) {
        hidden = forward_token(token, state);
    }
    timing.prefill_s = elapsed_s(prefill_start);
    return argmax_logits(hidden, timing);
}

int QwenModel::decode_one(int token, RunState & state, Timing & timing) const {
    if (device_ == Device::CUDA) {
        return decode_one_device(token, state, timing);
    }
    const auto decode_start = Clock::now();
    std::vector<float> hidden = forward_token(token, state);
    const int next = argmax_logits(hidden, timing);
    timing.decode_total_s += elapsed_s(decode_start);
    return next;
}

bool QwenModel::generate_sequence_device(
        const std::vector<int> & prompt_ids,
        RunState & state,
        int max_new_tokens,
        int eos_token_id,
        Timing & timing,
        std::vector<int> & generated) const {
    void * generated_device = cuda_generated_token_buffer(max_new_tokens);
    if (!generated_device) {
        throw std::runtime_error("CUDA 生成 token 缓冲区分配失败。");
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
            }
        }
    }
    if (!cuda_synchronize_device()) {
        throw std::runtime_error("CUDA 同步失败。");
    }
    timing.prefill_s = elapsed_s(prefill_start);

    const TensorRef emb = params_.embed_tokens;
    const TensorRef norm = params_.final_norm;
    const int hidden_size = static_cast<int>(emb.info->shape[1]);
    const auto decode_start = Clock::now();
    for (int i = 0; i < max_new_tokens; ++i) {
        int * token_slot = static_cast<int *>(generated_device) + i;
        if (!cuda_final_norm_argmax_to_device(
                norm,
                emb,
                hidden,
                hidden_size,
                config_.text.rms_norm_eps,
                true,
                token_slot)) {
            throw std::runtime_error("CUDA final norm + argmax 到设备失败。");
        }
        if (i + 1 < max_new_tokens) {
            hidden = forward_token_device_from_device(token_slot, state);
        }
    }
    if (!cuda_copy_generated_tokens_to_host(generated_device, max_new_tokens, generated)) {
        throw std::runtime_error("CUDA 生成 token 拷回主机失败。");
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

int QwenModel::generate_next_device(const std::vector<int> & prompt_ids, RunState & state, Timing & timing) const {
    const auto prefill_start = Clock::now();
    const void * hidden = nullptr;
    for (int token : prompt_ids) {
        hidden = forward_token_device(token, state);
    }
    timing.prefill_s = elapsed_s(prefill_start);
    return argmax_logits_device(hidden, timing);
}

int QwenModel::decode_one_device(int token, RunState & state, Timing & timing) const {
    const auto decode_start = Clock::now();
    const void * hidden = forward_token_device(token, state);
    const int next = argmax_logits_device(hidden, timing);
    timing.decode_total_s += elapsed_s(decode_start);
    return next;
}

const void * QwenModel::forward_token_device(int token, RunState & state) const {
    const int hidden_size = config_.text.hidden_size;
    void * current = cuda_token_hidden_buffer(0, hidden_size);
    void * next = cuda_token_hidden_buffer(1, hidden_size);
    if (!current || !next || !cuda_embedding_lookup_device(params_.embed_tokens, token, current)) {
        throw std::runtime_error("CUDA embedding lookup（主机 token）失败。");
    }
    return forward_token_device_layers(current, next, state);
}

const void * QwenModel::forward_token_device_from_device(const void * device_token, RunState & state) const {
    const int hidden_size = config_.text.hidden_size;
    void * current = cuda_token_hidden_buffer(0, hidden_size);
    void * next = cuda_token_hidden_buffer(1, hidden_size);
    if (!current || !next || !cuda_embedding_lookup_device_token(params_.embed_tokens, device_token, current)) {
        throw std::runtime_error("CUDA embedding lookup（设备 token）失败。");
    }
    return forward_token_device_layers(current, next, state);
}

const void * QwenModel::forward_token_device_layers(void * current, void * next, RunState & state) const {
    const int hidden_size = config_.text.hidden_size;
    const int pos = state.seq_len;

    for (int layer = 0; layer < config_.text.num_hidden_layers; ++layer) {
        const LayerWeights & w = params_.layers[layer];
        bool ok = false;
        if (w.type == "linear_attention") {
            ok = cuda_linear_attention_full_layer_device(
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
                current,
                next,
                hidden_size,
                state.linear[layer].cuda_state,
                config_.text.linear_num_key_heads,
                config_.text.linear_num_value_heads,
                config_.text.linear_key_head_dim,
                config_.text.linear_value_head_dim,
                config_.text.linear_conv_kernel_dim,
                config_.text.rms_norm_eps,
                true);
        } else {
            ok = cuda_full_attention_full_layer_device(
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
                current,
                next,
                hidden_size,
                state.full[layer].cuda_state,
                config_.text.num_attention_heads,
                config_.text.num_key_value_heads,
                config_.text.head_dim,
                state.full[layer].max_seq_len,
                pos,
                config_.text.rope_parameters.rope_theta,
                config_.text.rope_parameters.partial_rotary_factor,
                config_.text.rms_norm_eps,
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

std::vector<float> QwenModel::forward_token(int token, RunState & state) const {
    std::vector<float> x;
    cpu::embedding_lookup(params_.embed_tokens, token, x);
    const int pos = state.seq_len;

    for (int layer = 0; layer < config_.text.num_hidden_layers; ++layer) {
        const LayerWeights & w = params_.layers[layer];

        std::vector<float> residual = x;
        std::vector<float> normed;
        std::vector<float> mixer_out;

        cpu::rms_norm(w.input_norm, x, normed, config_.text.rms_norm_eps, true);
        if (w.type == "linear_attention") {
            ops::linear_attention(config_, w, normed, state.linear[layer], mixer_out);
        } else {
            ops::full_attention(config_, w, normed, state.full[layer], pos, mixer_out);
        }
        x = residual;
        cpu::add_inplace(x, mixer_out);

        residual = x;
        cpu::rms_norm(w.post_norm, x, normed, config_.text.rms_norm_eps, true);
        ops::mlp(w, normed, mixer_out);
        x = residual;
        cpu::add_inplace(x, mixer_out);
    }

    std::vector<float> normed;
    cpu::rms_norm(params_.final_norm, x, normed, config_.text.rms_norm_eps, true);
    state.seq_len += 1;
    return normed;
}

int QwenModel::argmax_logits(const std::vector<float> & hidden, Timing & timing) const {
    const auto start = Clock::now();
    const int best_id = ops::argmax_logits(params_, hidden);
    timing.logits_s += elapsed_s(start);
    return best_id;
}

int QwenModel::argmax_logits_device(const void * device_hidden, Timing & timing) const {
    const TensorRef emb = params_.embed_tokens;
    const int hidden_size = static_cast<int>(emb.info->shape[1]);
    const auto start = Clock::now();
    int best_id = 0;
    if (!cuda_final_norm_argmax_device(
            params_.final_norm,
            emb,
            device_hidden,
            hidden_size,
            config_.text.rms_norm_eps,
            true,
            best_id)) {
        throw std::runtime_error("CUDA final norm + logits argmax 失败。");
    }
    timing.logits_s += elapsed_s(start);
    return best_id;
}

std::vector<int> QwenModel::run_token_generation(
    RunState & state,
    const Args & args,
    const std::vector<int> & input_ids,
    Timing & timing) const {
    std::vector<int> generated;
    int next = generate_next(input_ids, state, timing);
    for (int i = 0; i < args.max_new_tokens; ++i) {
        generated.push_back(next);
        timing.generated_ids.push_back(next);
        timing.generated_tokens += 1;
        if (next == config_.text.eos_token_id) {
            break;
        }
        if (i + 1 < args.max_new_tokens) {
            next = decode_one(next, state, timing);
        }
    }
    return generated;
}

std::vector<int> QwenModel::run_greedy_generation(
    RunState & state,
    const Args & args,
    const std::vector<int> & input_ids,
    Timing & timing) const {
    // CUDA：走整段设备端 prefill + decode（失败即抛异常，不回退 CPU）。
    if (device_ == Device::CUDA) {
        std::vector<int> generated;
        generate_sequence_device(input_ids, state, args.max_new_tokens, config_.text.eos_token_id, timing, generated);
        return generated;
    }
    // CPU：逐 token 生成。
    return run_token_generation(state, args, input_ids, timing);
}

std::vector<int> QwenModel::run_non_greedy_generation(
    RunState & state,
    const Args & args,
    const std::vector<int> & input_ids,
    Timing & timing) const {
    return run_token_generation(state, args, input_ids, timing);
}

} // namespace llm_inference
