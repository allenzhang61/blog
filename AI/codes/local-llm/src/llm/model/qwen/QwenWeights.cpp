//
// Created by zhangyoulun on 8/8/2026.
//

#include "QwenWeights.h"

#include "QwenConfig.h"
#include "utils/log/Log.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {
const StorageTensor &get_tensor_any(const MF &mf, std::initializer_list<std::string> names) {
    for (const std::string &name : names) {
        if (mf.contain_tensor_view(name)) {
            return mf.get_tensor_view(name);
        }
    }

    std::ostringstream out;
    out << "缺少 tensor：";
    bool first = true;
    for (const std::string &name : names) {
        if (!first) {
            out << " 或 ";
        }
        first = false;
        out << name;
    }
    throw std::runtime_error(out.str());
}

void validate_tensor_any(const MF &mf, std::initializer_list<std::string> names) {
    get_tensor_any(mf, names);
}
} // namespace

std::string QwenWeights::shape_to_string(const std::vector<int64_t> &shape) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << shape[i];
    }
    out << "]";
    return out.str();
}

void QwenWeights::validate(const QwenConfig &config) const {
    mf_.validate();

    const int num_hidden_layers = config.data.text.num_hidden_layers;
    const std::vector<std::string> &layer_types = config.data.text.layer_types;
    if (static_cast<int>(layer_types.size()) != num_hidden_layers) {
        throw std::runtime_error("Qwen layer_types 数量与 num_hidden_layers 不一致");
    }

    const std::string root = "model.language_model.";
    for (const std::string &name : {
             root + "embed_tokens.weight",
             root + "norm.weight",
         }) {
        if (!mf_.contain_tensor_view(name)) {
            throw std::runtime_error("缺少 tensor：" + name);
        }
    }

    for (int layer = 0; layer < num_hidden_layers; ++layer) {
        const std::string prefix = root + "layers." + std::to_string(layer) + ".";
        validate_tensor_any(mf_, {prefix + "input_layernorm.weight"});
        validate_tensor_any(mf_, {prefix + "post_attention_layernorm.weight"});
        validate_tensor_any(mf_, {prefix + "mlp.gate_proj.weight", prefix + "mlp.d_gate_proj.weight"});
        validate_tensor_any(mf_, {prefix + "mlp.up_proj.weight", prefix + "mlp.d_up_proj.weight"});
        validate_tensor_any(mf_, {prefix + "mlp.down_proj.weight", prefix + "mlp.d_down_proj.weight"});
        if (layer_types[layer] == "linear_attention") {
            validate_tensor_any(mf_, {prefix + "linear_attn.A_log"});
            validate_tensor_any(mf_, {prefix + "linear_attn.norm.weight", prefix + "linear_attn.d_norm.weight"});
            validate_tensor_any(mf_, {prefix + "linear_attn.conv1d.weight", prefix + "linear_attn.d_conv1d.weight"});
            validate_tensor_any(mf_, {prefix + "linear_attn.dt_bias", prefix + "linear_attn.d_dt_bias"});
            validate_tensor_any(mf_, {prefix + "linear_attn.in_proj_a.weight", prefix + "linear_attn.d_in_proj_a.weight"});
            validate_tensor_any(mf_, {prefix + "linear_attn.in_proj_b.weight", prefix + "linear_attn.d_in_proj_b.weight"});
            validate_tensor_any(mf_, {prefix + "linear_attn.in_proj_qkv.weight", prefix + "linear_attn.d_in_proj_qkv.weight"});
            validate_tensor_any(mf_, {prefix + "linear_attn.in_proj_z.weight", prefix + "linear_attn.d_in_proj_z.weight"});
            validate_tensor_any(mf_, {prefix + "linear_attn.out_proj.weight", prefix + "linear_attn.d_out_proj.weight"});
        } else {
            validate_tensor_any(mf_, {prefix + "self_attn.q_proj.weight", prefix + "self_attn.d_q_proj.weight"});
            validate_tensor_any(mf_, {prefix + "self_attn.k_proj.weight", prefix + "self_attn.d_k_proj.weight"});
            validate_tensor_any(mf_, {prefix + "self_attn.v_proj.weight", prefix + "self_attn.d_v_proj.weight"});
            validate_tensor_any(mf_, {prefix + "self_attn.o_proj.weight", prefix + "self_attn.d_o_proj.weight"});
            validate_tensor_any(mf_, {prefix + "self_attn.q_norm.weight", prefix + "self_attn.d_q_norm.weight"});
            validate_tensor_any(mf_, {prefix + "self_attn.k_norm.weight", prefix + "self_attn.d_k_norm.weight"});
        }
    }

    mf_.validate_tensor_shape(root + "embed_tokens.weight",
                              {config.data.text.vocab_size, config.data.text.hidden_size});
    mf_.validate_tensor_shape(root + "norm.weight", {config.data.text.hidden_size});
}

QwenWeights::QwenWeights(const MF &mf, const QwenConfig &config)
    : mf_(mf) {
    validate(config);

    const int num_hidden_layers = config.data.text.num_hidden_layers;
    const std::vector<std::string> &layer_types = config.data.text.layer_types;

    const std::string root = "model.language_model.";
    s_token_embd = mf_.get_tensor_view(root + "embed_tokens.weight");
    s_output_norm = mf_.get_tensor_view(root + "norm.weight");
    layers.resize(num_hidden_layers);

    // 每种注意力类型独立计数，得到本层在同类型层序列中的下标。
    std::unordered_map<std::string, size_t> type_counts;
    for (int layer = 0; layer < num_hidden_layers; ++layer) {
        const std::string prefix = root + "layers." + std::to_string(layer) + ".";
        LayerWeights &lw = layers[layer];
        lw.type = layer_types[layer];
        lw.type_index = type_counts[lw.type]++;
        lw.s_input_layernorm = mf_.get_tensor_view(prefix + "input_layernorm.weight");
        lw.s_post_attention_layernorm = mf_.get_tensor_view(prefix + "post_attention_layernorm.weight");
        lw.mlp.s_gate_proj = get_tensor_any(mf_, {prefix + "mlp.gate_proj.weight", prefix + "mlp.d_gate_proj.weight"});
        lw.mlp.s_up_proj = get_tensor_any(mf_, {prefix + "mlp.up_proj.weight", prefix + "mlp.d_up_proj.weight"});
        lw.mlp.s_down_proj = get_tensor_any(mf_, {prefix + "mlp.down_proj.weight", prefix + "mlp.d_down_proj.weight"});

        if (lw.type == "linear_attention") {
            lw.lin.type_index = lw.type_index;
            lw.lin.s_in_proj_qkv = get_tensor_any(mf_, {prefix + "linear_attn.in_proj_qkv.weight", prefix + "linear_attn.d_in_proj_qkv.weight"});
            lw.lin.s_in_proj_z = get_tensor_any(mf_, {prefix + "linear_attn.in_proj_z.weight", prefix + "linear_attn.d_in_proj_z.weight"});
            lw.lin.s_in_proj_b = get_tensor_any(mf_, {prefix + "linear_attn.in_proj_b.weight", prefix + "linear_attn.d_in_proj_b.weight"});
            lw.lin.s_in_proj_a = get_tensor_any(mf_, {prefix + "linear_attn.in_proj_a.weight", prefix + "linear_attn.d_in_proj_a.weight"});
            lw.lin.s_conv1d = get_tensor_any(mf_, {prefix + "linear_attn.conv1d.weight", prefix + "linear_attn.d_conv1d.weight"});
            lw.lin.s_a_log = get_tensor_any(mf_, {prefix + "linear_attn.A_log"});
            lw.lin.s_dt_bias = get_tensor_any(mf_, {prefix + "linear_attn.dt_bias", prefix + "linear_attn.d_dt_bias"});
            lw.lin.s_norm = get_tensor_any(mf_, {prefix + "linear_attn.norm.weight", prefix + "linear_attn.d_norm.weight"});
            lw.lin.s_out_proj = get_tensor_any(mf_, {prefix + "linear_attn.out_proj.weight", prefix + "linear_attn.d_out_proj.weight"});
        } else {
            lw.full.type_index = lw.type_index;
            lw.full.s_q_proj = get_tensor_any(mf_, {prefix + "self_attn.q_proj.weight", prefix + "self_attn.d_q_proj.weight"});
            lw.full.s_k_proj = get_tensor_any(mf_, {prefix + "self_attn.k_proj.weight", prefix + "self_attn.d_k_proj.weight"});
            lw.full.s_v_proj = get_tensor_any(mf_, {prefix + "self_attn.v_proj.weight", prefix + "self_attn.d_v_proj.weight"});
            lw.full.s_q_norm = get_tensor_any(mf_, {prefix + "self_attn.q_norm.weight", prefix + "self_attn.d_q_norm.weight"});
            lw.full.s_k_norm = get_tensor_any(mf_, {prefix + "self_attn.k_norm.weight", prefix + "self_attn.d_k_norm.weight"});
            lw.full.s_o_proj = get_tensor_any(mf_, {prefix + "self_attn.o_proj.weight", prefix + "self_attn.d_o_proj.weight"});
        }
    }

    // 解析视觉塔和 MTP 权重；已解析但当前未使用（纯文本推理不走这两条路径）。
    parse_vision_weights(config);
    parse_mtp_weights(config);
}

void QwenWeights::parse_vision_weights(const QwenConfig &config) {
    const std::string root = "model.visual.";
    vision.s_patch_embed_proj_weight = mf_.get_tensor_view(root + "patch_embed.proj.weight");
    vision.s_patch_embed_proj_bias = mf_.get_tensor_view(root + "patch_embed.proj.bias");
    vision.s_pos_embed_weight = mf_.get_tensor_view(root + "pos_embed.weight");
    vision.s_merger_norm_weight = get_tensor_any(mf_, {root + "merger.norm.weight", root + "merger.d_norm.weight"});
    vision.s_merger_norm_bias = get_tensor_any(mf_, {root + "merger.norm.bias", root + "merger.d_norm.bias"});
    vision.s_merger_fc1_weight = mf_.get_tensor_view(root + "merger.linear_fc1.weight");
    vision.s_merger_fc1_bias = mf_.get_tensor_view(root + "merger.linear_fc1.bias");
    vision.s_merger_fc2_weight = mf_.get_tensor_view(root + "merger.linear_fc2.weight");
    vision.s_merger_fc2_bias = mf_.get_tensor_view(root + "merger.linear_fc2.bias");

    const int depth = config.data.vision.depth;
    vision.blocks.resize(depth);
    for (int i = 0; i < depth; ++i) {
        const std::string prefix = root + "blocks." + std::to_string(i) + ".";
        VisionBlockWeights &b = vision.blocks[i];
        b.s_norm1_weight = mf_.get_tensor_view(prefix + "norm1.weight");
        b.s_norm1_bias = mf_.get_tensor_view(prefix + "norm1.bias");
        b.s_norm2_weight = mf_.get_tensor_view(prefix + "norm2.weight");
        b.s_norm2_bias = mf_.get_tensor_view(prefix + "norm2.bias");
        b.s_attn_qkv_weight = mf_.get_tensor_view(prefix + "attn.qkv.weight");
        b.s_attn_qkv_bias = mf_.get_tensor_view(prefix + "attn.qkv.bias");
        b.s_attn_proj_weight = mf_.get_tensor_view(prefix + "attn.proj.weight");
        b.s_attn_proj_bias = mf_.get_tensor_view(prefix + "attn.proj.bias");
        b.s_mlp_fc1_weight = mf_.get_tensor_view(prefix + "mlp.linear_fc1.weight");
        b.s_mlp_fc1_bias = mf_.get_tensor_view(prefix + "mlp.linear_fc1.bias");
        b.s_mlp_fc2_weight = mf_.get_tensor_view(prefix + "mlp.linear_fc2.weight");
        b.s_mlp_fc2_bias = mf_.get_tensor_view(prefix + "mlp.linear_fc2.bias");
    }
}

void QwenWeights::parse_mtp_weights(const QwenConfig &config) {
    const std::string root = "mtp.";
    mtp.s_fc_weight = mf_.get_tensor_view(root + "fc.weight");
    mtp.s_norm_weight = mf_.get_tensor_view(root + "norm.weight");
    mtp.s_pre_fc_norm_embedding_weight = mf_.get_tensor_view(root + "pre_fc_norm_embedding.weight");
    mtp.s_pre_fc_norm_hidden_weight = mf_.get_tensor_view(root + "pre_fc_norm_hidden.weight");

    const int num_layers = config.data.text.mtp_num_hidden_layers;
    mtp.layers.resize(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        const std::string prefix = root + "layers." + std::to_string(i) + ".";
        MtpLayerWeights &m = mtp.layers[i];
        m.s_attn_norm = mf_.get_tensor_view(prefix + "input_layernorm.weight");
        m.s_ffn_norm = mf_.get_tensor_view(prefix + "post_attention_layernorm.weight");
        m.s_self_attn_q_proj = get_tensor_any(mf_, {prefix + "self_attn.q_proj.weight", prefix + "self_attn.d_q_proj.weight"});
        m.s_self_attn_k_proj = get_tensor_any(mf_, {prefix + "self_attn.k_proj.weight", prefix + "self_attn.d_k_proj.weight"});
        m.s_self_attn_v_proj = get_tensor_any(mf_, {prefix + "self_attn.v_proj.weight", prefix + "self_attn.d_v_proj.weight"});
        m.s_self_attn_o_proj = get_tensor_any(mf_, {prefix + "self_attn.o_proj.weight", prefix + "self_attn.d_o_proj.weight"});
        m.s_self_attn_q_norm = get_tensor_any(mf_, {prefix + "self_attn.q_norm.weight", prefix + "self_attn.d_q_norm.weight"});
        m.s_self_attn_k_norm = get_tensor_any(mf_, {prefix + "self_attn.k_norm.weight", prefix + "self_attn.d_k_norm.weight"});
        m.s_mlp_gate = get_tensor_any(mf_, {prefix + "mlp.gate_proj.weight", prefix + "mlp.d_gate_proj.weight"});
        m.s_mlp_up = get_tensor_any(mf_, {prefix + "mlp.up_proj.weight", prefix + "mlp.d_up_proj.weight"});
        m.s_mlp_down = get_tensor_any(mf_, {prefix + "mlp.down_proj.weight", prefix + "mlp.d_down_proj.weight"});
    }
}

void QwenWeights::DebugDump() {
    std::ostringstream out;
    out << "QwenWeights:\n";
    out << "  tensors=" << mf_.tensor_view_names().size() << "\n";

    auto dump_one = [&](const std::string &label, const StorageTensor &s_w) {
        out << "  " << label << ": ";
        if (s_w.data() == nullptr) {
            out << "<null>\n";
            return;
        }
        out << s_w.name
            << " dtype=" << dtype_name(s_w.dtype)
            << " shape=" << shape_to_string(s_w.shape)
            << " bytes=" << s_w.nbytes << "\n";
    };

    dump_one("token_embd", s_token_embd);
    dump_one("output_norm", s_output_norm);
    for (size_t i = 0; i < layers.size(); ++i) {
        const LayerWeights &lw = layers[i];
        out << "  layer[" << i << "] type=" << lw.type << "\n";
        dump_one("  input_layernorm", lw.s_input_layernorm);
        dump_one("  post_attention_layernorm", lw.s_post_attention_layernorm);
        dump_one("  mlp.s_gate_proj", lw.mlp.s_gate_proj);
        dump_one("  mlp.s_up_proj", lw.mlp.s_up_proj);
        dump_one("  mlp.s_down_proj", lw.mlp.s_down_proj);
        if (lw.type == "linear_attention") {
            dump_one("  lin.s_in_proj_qkv", lw.lin.s_in_proj_qkv);
            dump_one("  lin.s_in_proj_z", lw.lin.s_in_proj_z);
            dump_one("  lin.s_in_proj_b", lw.lin.s_in_proj_b);
            dump_one("  lin.s_in_proj_a", lw.lin.s_in_proj_a);
            dump_one("  lin.s_conv1d", lw.lin.s_conv1d);
            dump_one("  lin.s_a_log", lw.lin.s_a_log);
            dump_one("  lin.s_dt_bias", lw.lin.s_dt_bias);
            dump_one("  lin.s_norm", lw.lin.s_norm);
            dump_one("  lin.s_out_proj", lw.lin.s_out_proj);
        } else {
            dump_one("  full.s_q_proj", lw.full.s_q_proj);
            dump_one("  full.s_k_proj", lw.full.s_k_proj);
            dump_one("  full.s_v_proj", lw.full.s_v_proj);
            dump_one("  full.s_q_norm", lw.full.s_q_norm);
            dump_one("  full.s_k_norm", lw.full.s_k_norm);
            dump_one("  full.s_o_proj", lw.full.s_o_proj);
        }
    }

    // 视觉塔权重（model.visual.*）；已解析但当前未使用。
    out << "  vision (model.visual.*, 当前未使用):\n";
    dump_one("  patch_embed.proj.weight", vision.s_patch_embed_proj_weight);
    dump_one("  patch_embed.proj.bias", vision.s_patch_embed_proj_bias);
    dump_one("  pos_embed.weight", vision.s_pos_embed_weight);
    dump_one("  merger.d_norm.weight", vision.s_merger_norm_weight);
    dump_one("  merger.d_norm.bias", vision.s_merger_norm_bias);
    dump_one("  merger.fc1.weight", vision.s_merger_fc1_weight);
    dump_one("  merger.fc1.bias", vision.s_merger_fc1_bias);
    dump_one("  merger.fc2.weight", vision.s_merger_fc2_weight);
    dump_one("  merger.fc2.bias", vision.s_merger_fc2_bias);
    for (size_t i = 0; i < vision.blocks.size(); ++i) {
        const VisionBlockWeights &b = vision.blocks[i];
        out << "  vision.block[" << i << "]\n";
        dump_one("  norm1.weight", b.s_norm1_weight);
        dump_one("  norm1.bias", b.s_norm1_bias);
        dump_one("  norm2.weight", b.s_norm2_weight);
        dump_one("  norm2.bias", b.s_norm2_bias);
        dump_one("  attn.qkv.weight", b.s_attn_qkv_weight);
        dump_one("  attn.qkv.bias", b.s_attn_qkv_bias);
        dump_one("  attn.proj.weight", b.s_attn_proj_weight);
        dump_one("  attn.proj.bias", b.s_attn_proj_bias);
        dump_one("  mlp.fc1.weight", b.s_mlp_fc1_weight);
        dump_one("  mlp.fc1.bias", b.s_mlp_fc1_bias);
        dump_one("  mlp.fc2.weight", b.s_mlp_fc2_weight);
        dump_one("  mlp.fc2.bias", b.s_mlp_fc2_bias);
    }

    // MTP 权重（mtp.*）；已解析但当前未使用。
    out << "  mtp (mtp.*, 当前未使用):\n";
    dump_one("  fc.weight", mtp.s_fc_weight);
    dump_one("  norm.weight", mtp.s_norm_weight);
    dump_one("  pre_fc_norm_embedding.weight", mtp.s_pre_fc_norm_embedding_weight);
    dump_one("  pre_fc_norm_hidden.weight", mtp.s_pre_fc_norm_hidden_weight);
    for (size_t i = 0; i < mtp.layers.size(); ++i) {
        const MtpLayerWeights &m = mtp.layers[i];
        out << "  mtp.layer[" << i << "]\n";
        dump_one("  attn_norm", m.s_attn_norm);
        dump_one("  ffn_norm", m.s_ffn_norm);
        dump_one("  self_attn.d_q_proj", m.s_self_attn_q_proj);
        dump_one("  self_attn.d_k_proj", m.s_self_attn_k_proj);
        dump_one("  self_attn.d_v_proj", m.s_self_attn_v_proj);
        dump_one("  self_attn.d_o_proj", m.s_self_attn_o_proj);
        dump_one("  self_attn.d_q_norm", m.s_self_attn_q_norm);
        dump_one("  self_attn.d_k_norm", m.s_self_attn_k_norm);
        dump_one("  mlp.gate", m.s_mlp_gate);
        dump_one("  mlp.up", m.s_mlp_up);
        dump_one("  mlp.down", m.s_mlp_down);
    }

    Log::debug(out.str());
}
