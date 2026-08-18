//
// Created by zhangyoulun on 8/8/2026.
//

#include "QwenWeights.h"

#include "QwenConfig.h"
#include "utils/log/Log.h"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

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
        for (const std::string &name : {
                 prefix + "input_layernorm.weight",
                 prefix + "post_attention_layernorm.weight",
                 prefix + "mlp.gate_proj.weight",
                 prefix + "mlp.up_proj.weight",
                 prefix + "mlp.down_proj.weight",
             }) {
            if (!mf_.contain_tensor_view(name)) {
                throw std::runtime_error("缺少 tensor：" + name);
            }
        }
        if (layer_types[layer] == "linear_attention") {
            for (const std::string &name : {
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
                if (!mf_.contain_tensor_view(name)) {
                    throw std::runtime_error("缺少 tensor：" + name);
                }
            }
        } else {
            for (const std::string &name : {
                     prefix + "self_attn.q_proj.weight",
                     prefix + "self_attn.k_proj.weight",
                     prefix + "self_attn.v_proj.weight",
                     prefix + "self_attn.o_proj.weight",
                     prefix + "self_attn.q_norm.weight",
                     prefix + "self_attn.k_norm.weight",
                 }) {
                if (!mf_.contain_tensor_view(name)) {
                    throw std::runtime_error("缺少 tensor：" + name);
                }
            }
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
    token_embd = mf_.get_tensor_view(root + "embed_tokens.weight");
    output_norm = mf_.get_tensor_view(root + "norm.weight");
    layers.resize(num_hidden_layers);

    // 每种注意力类型独立计数，得到本层在同类型层序列中的下标。
    std::unordered_map<std::string, size_t> type_counts;
    for (int layer = 0; layer < num_hidden_layers; ++layer) {
        const std::string prefix = root + "layers." + std::to_string(layer) + ".";
        LayerWeights &lw = layers[layer];
        lw.type = layer_types[layer];
        lw.type_index = type_counts[lw.type]++;
        lw.input_layernorm = mf_.get_tensor_view(prefix + "input_layernorm.weight");
        lw.post_attention_layernorm = mf_.get_tensor_view(prefix + "post_attention_layernorm.weight");
        lw.mlp.gate_proj = mf_.get_tensor_view(prefix + "mlp.gate_proj.weight");
        lw.mlp.up_proj = mf_.get_tensor_view(prefix + "mlp.up_proj.weight");
        lw.mlp.down_proj = mf_.get_tensor_view(prefix + "mlp.down_proj.weight");

        if (lw.type == "linear_attention") {
            lw.lin.type_index = lw.type_index;
            lw.lin.in_proj_qkv = mf_.get_tensor_view(prefix + "linear_attn.in_proj_qkv.weight");
            lw.lin.in_proj_z = mf_.get_tensor_view(prefix + "linear_attn.in_proj_z.weight");
            lw.lin.in_proj_b = mf_.get_tensor_view(prefix + "linear_attn.in_proj_b.weight");
            lw.lin.in_proj_a = mf_.get_tensor_view(prefix + "linear_attn.in_proj_a.weight");
            lw.lin.conv1d = mf_.get_tensor_view(prefix + "linear_attn.conv1d.weight");
            lw.lin.a_log = mf_.get_tensor_view(prefix + "linear_attn.A_log");
            lw.lin.dt_bias = mf_.get_tensor_view(prefix + "linear_attn.dt_bias");
            lw.lin.norm = mf_.get_tensor_view(prefix + "linear_attn.norm.weight");
            lw.lin.out_proj = mf_.get_tensor_view(prefix + "linear_attn.out_proj.weight");
        } else {
            lw.full.type_index = lw.type_index;
            lw.full.q_proj = mf_.get_tensor_view(prefix + "self_attn.q_proj.weight");
            lw.full.k_proj = mf_.get_tensor_view(prefix + "self_attn.k_proj.weight");
            lw.full.v_proj = mf_.get_tensor_view(prefix + "self_attn.v_proj.weight");
            lw.full.q_norm = mf_.get_tensor_view(prefix + "self_attn.q_norm.weight");
            lw.full.k_norm = mf_.get_tensor_view(prefix + "self_attn.k_norm.weight");
            lw.full.o_proj = mf_.get_tensor_view(prefix + "self_attn.o_proj.weight");
        }
    }

    // 解析视觉塔和 MTP 权重；已解析但当前未使用（纯文本推理不走这两条路径）。
    parse_vision_weights(config);
    parse_mtp_weights(config);
}

void QwenWeights::parse_vision_weights(const QwenConfig &config) {
    const std::string root = "model.visual.";
    vision.patch_embed_proj_weight = mf_.get_tensor_view(root + "patch_embed.proj.weight");
    vision.patch_embed_proj_bias = mf_.get_tensor_view(root + "patch_embed.proj.bias");
    vision.pos_embed_weight = mf_.get_tensor_view(root + "pos_embed.weight");
    vision.merger_norm_weight = mf_.get_tensor_view(root + "merger.norm.weight");
    vision.merger_norm_bias = mf_.get_tensor_view(root + "merger.norm.bias");
    vision.merger_fc1_weight = mf_.get_tensor_view(root + "merger.linear_fc1.weight");
    vision.merger_fc1_bias = mf_.get_tensor_view(root + "merger.linear_fc1.bias");
    vision.merger_fc2_weight = mf_.get_tensor_view(root + "merger.linear_fc2.weight");
    vision.merger_fc2_bias = mf_.get_tensor_view(root + "merger.linear_fc2.bias");

    const int depth = config.data.vision.depth;
    vision.blocks.resize(depth);
    for (int i = 0; i < depth; ++i) {
        const std::string prefix = root + "blocks." + std::to_string(i) + ".";
        VisionBlockWeights &b = vision.blocks[i];
        b.norm1_weight = mf_.get_tensor_view(prefix + "norm1.weight");
        b.norm1_bias = mf_.get_tensor_view(prefix + "norm1.bias");
        b.norm2_weight = mf_.get_tensor_view(prefix + "norm2.weight");
        b.norm2_bias = mf_.get_tensor_view(prefix + "norm2.bias");
        b.attn_qkv_weight = mf_.get_tensor_view(prefix + "attn.qkv.weight");
        b.attn_qkv_bias = mf_.get_tensor_view(prefix + "attn.qkv.bias");
        b.attn_proj_weight = mf_.get_tensor_view(prefix + "attn.proj.weight");
        b.attn_proj_bias = mf_.get_tensor_view(prefix + "attn.proj.bias");
        b.mlp_fc1_weight = mf_.get_tensor_view(prefix + "mlp.linear_fc1.weight");
        b.mlp_fc1_bias = mf_.get_tensor_view(prefix + "mlp.linear_fc1.bias");
        b.mlp_fc2_weight = mf_.get_tensor_view(prefix + "mlp.linear_fc2.weight");
        b.mlp_fc2_bias = mf_.get_tensor_view(prefix + "mlp.linear_fc2.bias");
    }
}

void QwenWeights::parse_mtp_weights(const QwenConfig &config) {
    const std::string root = "mtp.";
    mtp.fc_weight = mf_.get_tensor_view(root + "fc.weight");
    mtp.norm_weight = mf_.get_tensor_view(root + "norm.weight");
    mtp.pre_fc_norm_embedding_weight = mf_.get_tensor_view(root + "pre_fc_norm_embedding.weight");
    mtp.pre_fc_norm_hidden_weight = mf_.get_tensor_view(root + "pre_fc_norm_hidden.weight");

    const int num_layers = config.data.text.mtp_num_hidden_layers;
    mtp.layers.resize(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        const std::string prefix = root + "layers." + std::to_string(i) + ".";
        MtpLayerWeights &m = mtp.layers[i];
        m.attn_norm = mf_.get_tensor_view(prefix + "input_layernorm.weight");
        m.ffn_norm = mf_.get_tensor_view(prefix + "post_attention_layernorm.weight");
        m.self_attn_q_proj = mf_.get_tensor_view(prefix + "self_attn.q_proj.weight");
        m.self_attn_k_proj = mf_.get_tensor_view(prefix + "self_attn.k_proj.weight");
        m.self_attn_v_proj = mf_.get_tensor_view(prefix + "self_attn.v_proj.weight");
        m.self_attn_o_proj = mf_.get_tensor_view(prefix + "self_attn.o_proj.weight");
        m.self_attn_q_norm = mf_.get_tensor_view(prefix + "self_attn.q_norm.weight");
        m.self_attn_k_norm = mf_.get_tensor_view(prefix + "self_attn.k_norm.weight");
        m.mlp_gate = mf_.get_tensor_view(prefix + "mlp.gate_proj.weight");
        m.mlp_up = mf_.get_tensor_view(prefix + "mlp.up_proj.weight");
        m.mlp_down = mf_.get_tensor_view(prefix + "mlp.down_proj.weight");
    }
}

void QwenWeights::DebugDump() {
    std::ostringstream out;
    out << "QwenWeights:\n";
    out << "  tensors=" << mf_.tensor_view_names().size() << "\n";

    auto dump_one = [&](const std::string &label, const MFTensorView &w) {
        out << "  " << label << ": ";
        if (w.data == nullptr) {
            out << "<null>\n";
            return;
        }
        out << w.name
            << " dtype=" << dtype_name(w.dtype)
            << " shape=" << shape_to_string(w.shape)
            << " bytes=" << w.nbytes << "\n";
    };

    dump_one("token_embd", token_embd);
    dump_one("output_norm", output_norm);
    for (size_t i = 0; i < layers.size(); ++i) {
        const LayerWeights &lw = layers[i];
        out << "  layer[" << i << "] type=" << lw.type << "\n";
        dump_one("  input_layernorm", lw.input_layernorm);
        dump_one("  post_attention_layernorm", lw.post_attention_layernorm);
        dump_one("  mlp.gate_proj", lw.mlp.gate_proj);
        dump_one("  mlp.up_proj", lw.mlp.up_proj);
        dump_one("  mlp.down_proj", lw.mlp.down_proj);
        if (lw.type == "linear_attention") {
            dump_one("  lin.in_proj_qkv", lw.lin.in_proj_qkv);
            dump_one("  lin.in_proj_z", lw.lin.in_proj_z);
            dump_one("  lin.in_proj_b", lw.lin.in_proj_b);
            dump_one("  lin.in_proj_a", lw.lin.in_proj_a);
            dump_one("  lin.conv1d", lw.lin.conv1d);
            dump_one("  lin.a_log", lw.lin.a_log);
            dump_one("  lin.dt_bias", lw.lin.dt_bias);
            dump_one("  lin.norm", lw.lin.norm);
            dump_one("  lin.out_proj", lw.lin.out_proj);
        } else {
            dump_one("  full.q_proj", lw.full.q_proj);
            dump_one("  full.k_proj", lw.full.k_proj);
            dump_one("  full.v_proj", lw.full.v_proj);
            dump_one("  full.q_norm", lw.full.q_norm);
            dump_one("  full.k_norm", lw.full.k_norm);
            dump_one("  full.o_proj", lw.full.o_proj);
        }
    }

    // 视觉塔权重（model.visual.*）；已解析但当前未使用。
    out << "  vision (model.visual.*, 当前未使用):\n";
    dump_one("  patch_embed.proj.weight", vision.patch_embed_proj_weight);
    dump_one("  patch_embed.proj.bias", vision.patch_embed_proj_bias);
    dump_one("  pos_embed.weight", vision.pos_embed_weight);
    dump_one("  merger.norm.weight", vision.merger_norm_weight);
    dump_one("  merger.norm.bias", vision.merger_norm_bias);
    dump_one("  merger.fc1.weight", vision.merger_fc1_weight);
    dump_one("  merger.fc1.bias", vision.merger_fc1_bias);
    dump_one("  merger.fc2.weight", vision.merger_fc2_weight);
    dump_one("  merger.fc2.bias", vision.merger_fc2_bias);
    for (size_t i = 0; i < vision.blocks.size(); ++i) {
        const VisionBlockWeights &b = vision.blocks[i];
        out << "  vision.block[" << i << "]\n";
        dump_one("  norm1.weight", b.norm1_weight);
        dump_one("  norm1.bias", b.norm1_bias);
        dump_one("  norm2.weight", b.norm2_weight);
        dump_one("  norm2.bias", b.norm2_bias);
        dump_one("  attn.qkv.weight", b.attn_qkv_weight);
        dump_one("  attn.qkv.bias", b.attn_qkv_bias);
        dump_one("  attn.proj.weight", b.attn_proj_weight);
        dump_one("  attn.proj.bias", b.attn_proj_bias);
        dump_one("  mlp.fc1.weight", b.mlp_fc1_weight);
        dump_one("  mlp.fc1.bias", b.mlp_fc1_bias);
        dump_one("  mlp.fc2.weight", b.mlp_fc2_weight);
        dump_one("  mlp.fc2.bias", b.mlp_fc2_bias);
    }

    // MTP 权重（mtp.*）；已解析但当前未使用。
    out << "  mtp (mtp.*, 当前未使用):\n";
    dump_one("  fc.weight", mtp.fc_weight);
    dump_one("  norm.weight", mtp.norm_weight);
    dump_one("  pre_fc_norm_embedding.weight", mtp.pre_fc_norm_embedding_weight);
    dump_one("  pre_fc_norm_hidden.weight", mtp.pre_fc_norm_hidden_weight);
    for (size_t i = 0; i < mtp.layers.size(); ++i) {
        const MtpLayerWeights &m = mtp.layers[i];
        out << "  mtp.layer[" << i << "]\n";
        dump_one("  attn_norm", m.attn_norm);
        dump_one("  ffn_norm", m.ffn_norm);
        dump_one("  self_attn.q_proj", m.self_attn_q_proj);
        dump_one("  self_attn.k_proj", m.self_attn_k_proj);
        dump_one("  self_attn.v_proj", m.self_attn_v_proj);
        dump_one("  self_attn.o_proj", m.self_attn_o_proj);
        dump_one("  self_attn.q_norm", m.self_attn_q_norm);
        dump_one("  self_attn.k_norm", m.self_attn_k_norm);
        dump_one("  mlp.gate", m.mlp_gate);
        dump_one("  mlp.up", m.mlp_up);
        dump_one("  mlp.down", m.mlp_down);
    }

    Log::debug(out.str());
}
