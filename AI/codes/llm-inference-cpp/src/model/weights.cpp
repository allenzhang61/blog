#include "weights.h"

#include <stdexcept>

namespace llm_inference {

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

ModelParams parse_model_params(const ModelWeights & weights, const ModelConfig & config) {
    const std::string root = "model.language_model.";
    ModelParams params;
    params.embed_tokens = tensor_ref(weights, root + "embed_tokens.weight");
    params.final_norm = tensor_ref(weights, root + "norm.weight");
    params.layers.resize(config.num_hidden_layers);

    for (int layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = root + "layers." + std::to_string(layer) + ".";
        LayerWeights & lw = params.layers[layer];
        lw.type = config.layer_types[layer];
        lw.input_norm = tensor_ref(weights, prefix + "input_layernorm.weight");
        lw.post_norm = tensor_ref(weights, prefix + "post_attention_layernorm.weight");
        lw.mlp.gate = tensor_ref(weights, prefix + "mlp.gate_proj.weight");
        lw.mlp.up = tensor_ref(weights, prefix + "mlp.up_proj.weight");
        lw.mlp.down = tensor_ref(weights, prefix + "mlp.down_proj.weight");

        if (lw.type == "linear_attention") {
            lw.lin.in_proj_qkv = tensor_ref(weights, prefix + "linear_attn.in_proj_qkv.weight");
            lw.lin.in_proj_z = tensor_ref(weights, prefix + "linear_attn.in_proj_z.weight");
            lw.lin.in_proj_b = tensor_ref(weights, prefix + "linear_attn.in_proj_b.weight");
            lw.lin.in_proj_a = tensor_ref(weights, prefix + "linear_attn.in_proj_a.weight");
            lw.lin.conv1d = tensor_ref(weights, prefix + "linear_attn.conv1d.weight");
            lw.lin.a_log = tensor_ref(weights, prefix + "linear_attn.A_log");
            lw.lin.dt_bias = tensor_ref(weights, prefix + "linear_attn.dt_bias");
            lw.lin.norm = tensor_ref(weights, prefix + "linear_attn.norm.weight");
            lw.lin.out_proj = tensor_ref(weights, prefix + "linear_attn.out_proj.weight");
        } else {
            lw.full.q_proj = tensor_ref(weights, prefix + "self_attn.q_proj.weight");
            lw.full.k_proj = tensor_ref(weights, prefix + "self_attn.k_proj.weight");
            lw.full.v_proj = tensor_ref(weights, prefix + "self_attn.v_proj.weight");
            lw.full.q_norm = tensor_ref(weights, prefix + "self_attn.q_norm.weight");
            lw.full.k_norm = tensor_ref(weights, prefix + "self_attn.k_norm.weight");
            lw.full.o_proj = tensor_ref(weights, prefix + "self_attn.o_proj.weight");
        }
    }
    return params;
}

} // namespace llm_inference
