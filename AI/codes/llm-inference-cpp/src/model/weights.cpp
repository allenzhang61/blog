#include "weights.h"

#include <stdexcept>

namespace llm_inference {

ModelParams parse_model_params(const ModelWeights & weights, const ModelConfig & config) {
    const std::string root = "model.language_model.";
    ModelParams params;
    params.embed_tokens = weights.tensor_ref(root + "embed_tokens.weight");
    params.final_norm = weights.tensor_ref(root + "norm.weight");
    params.layers.resize(config.text.num_hidden_layers);

    for (int layer = 0; layer < config.text.num_hidden_layers; ++layer) {
        const std::string prefix = root + "layers." + std::to_string(layer) + ".";
        LayerWeights & lw = params.layers[layer];
        lw.type = config.text.layer_types[layer];
        lw.input_norm = weights.tensor_ref(prefix + "input_layernorm.weight");
        lw.post_norm = weights.tensor_ref(prefix + "post_attention_layernorm.weight");
        lw.mlp.gate = weights.tensor_ref(prefix + "mlp.gate_proj.weight");
        lw.mlp.up = weights.tensor_ref(prefix + "mlp.up_proj.weight");
        lw.mlp.down = weights.tensor_ref(prefix + "mlp.down_proj.weight");

        if (lw.type == "linear_attention") {
            lw.lin.in_proj_qkv = weights.tensor_ref(prefix + "linear_attn.in_proj_qkv.weight");
            lw.lin.in_proj_z = weights.tensor_ref(prefix + "linear_attn.in_proj_z.weight");
            lw.lin.in_proj_b = weights.tensor_ref(prefix + "linear_attn.in_proj_b.weight");
            lw.lin.in_proj_a = weights.tensor_ref(prefix + "linear_attn.in_proj_a.weight");
            lw.lin.conv1d = weights.tensor_ref(prefix + "linear_attn.conv1d.weight");
            lw.lin.a_log = weights.tensor_ref(prefix + "linear_attn.A_log");
            lw.lin.dt_bias = weights.tensor_ref(prefix + "linear_attn.dt_bias");
            lw.lin.norm = weights.tensor_ref(prefix + "linear_attn.norm.weight");
            lw.lin.out_proj = weights.tensor_ref(prefix + "linear_attn.out_proj.weight");
        } else {
            lw.full.q_proj = weights.tensor_ref(prefix + "self_attn.q_proj.weight");
            lw.full.k_proj = weights.tensor_ref(prefix + "self_attn.k_proj.weight");
            lw.full.v_proj = weights.tensor_ref(prefix + "self_attn.v_proj.weight");
            lw.full.q_norm = weights.tensor_ref(prefix + "self_attn.q_norm.weight");
            lw.full.k_norm = weights.tensor_ref(prefix + "self_attn.k_norm.weight");
            lw.full.o_proj = weights.tensor_ref(prefix + "self_attn.o_proj.weight");
        }
    }
    return params;
}

} // namespace llm_inference
