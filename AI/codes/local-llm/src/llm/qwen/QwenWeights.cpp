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

void QwenWeights::validate_qwen_tensors(int num_hidden_layers, const std::vector<std::string> &layer_types) const {
    const std::string root = "model.language_model.";
    for (const std::string &name : {
             root + "embed_tokens.weight",
             root + "norm.weight",
         }) {
        if (!tensor_container_.contains(name)) {
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
            if (!tensor_container_.contains(name)) {
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
                if (!tensor_container_.contains(name)) {
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
                if (!tensor_container_.contains(name)) {
                    throw std::runtime_error("缺少 tensor：" + name);
                }
            }
        }
    }
}

QwenWeights::QwenWeights(const TensorContainer &tensor_container, const QwenConfig &config)
    : tensor_container_(tensor_container) {
    const int num_hidden_layers = config.data.text.num_hidden_layers;
    const std::vector<std::string> &layer_types = config.data.text.layer_types;
    validate_qwen_tensors(num_hidden_layers, layer_types);

    const std::string root = "model.language_model.";
    embed_tokens = tensor_container_.get(root + "embed_tokens.weight");
    final_norm = tensor_container_.get(root + "norm.weight");
    layers.resize(num_hidden_layers);

    for (int layer = 0; layer < num_hidden_layers; ++layer) {
        const std::string prefix = root + "layers." + std::to_string(layer) + ".";
        LayerWeights &lw = layers[layer];
        lw.type = layer_types[layer];
        lw.input_norm = tensor_container_.get(prefix + "input_layernorm.weight");
        lw.post_norm = tensor_container_.get(prefix + "post_attention_layernorm.weight");
        lw.mlp.gate = tensor_container_.get(prefix + "mlp.gate_proj.weight");
        lw.mlp.up = tensor_container_.get(prefix + "mlp.up_proj.weight");
        lw.mlp.down = tensor_container_.get(prefix + "mlp.down_proj.weight");

        if (lw.type == "linear_attention") {
            lw.lin.in_proj_qkv = tensor_container_.get(prefix + "linear_attn.in_proj_qkv.weight");
            lw.lin.in_proj_z = tensor_container_.get(prefix + "linear_attn.in_proj_z.weight");
            lw.lin.in_proj_b = tensor_container_.get(prefix + "linear_attn.in_proj_b.weight");
            lw.lin.in_proj_a = tensor_container_.get(prefix + "linear_attn.in_proj_a.weight");
            lw.lin.conv1d = tensor_container_.get(prefix + "linear_attn.conv1d.weight");
            lw.lin.a_log = tensor_container_.get(prefix + "linear_attn.A_log");
            lw.lin.dt_bias = tensor_container_.get(prefix + "linear_attn.dt_bias");
            lw.lin.norm = tensor_container_.get(prefix + "linear_attn.norm.weight");
            lw.lin.out_proj = tensor_container_.get(prefix + "linear_attn.out_proj.weight");
        } else {
            lw.full.q_proj = tensor_container_.get(prefix + "self_attn.q_proj.weight");
            lw.full.k_proj = tensor_container_.get(prefix + "self_attn.k_proj.weight");
            lw.full.v_proj = tensor_container_.get(prefix + "self_attn.v_proj.weight");
            lw.full.q_norm = tensor_container_.get(prefix + "self_attn.q_norm.weight");
            lw.full.k_norm = tensor_container_.get(prefix + "self_attn.k_norm.weight");
            lw.full.o_proj = tensor_container_.get(prefix + "self_attn.o_proj.weight");
        }
    }

    // 解析视觉塔和 MTP 权重；已解析但当前未使用（纯文本推理不走这两条路径）。
    parse_vision_weights(config);
    parse_mtp_weights(config);
}

void QwenWeights::parse_vision_weights(const QwenConfig &config) {
    const std::string root = "model.visual.";
    vision.patch_embed_proj_weight = tensor_container_.get(root + "patch_embed.proj.weight");
    vision.patch_embed_proj_bias = tensor_container_.get(root + "patch_embed.proj.bias");
    vision.pos_embed_weight = tensor_container_.get(root + "pos_embed.weight");
    vision.merger_norm_weight = tensor_container_.get(root + "merger.norm.weight");
    vision.merger_norm_bias = tensor_container_.get(root + "merger.norm.bias");
    vision.merger_fc1_weight = tensor_container_.get(root + "merger.linear_fc1.weight");
    vision.merger_fc1_bias = tensor_container_.get(root + "merger.linear_fc1.bias");
    vision.merger_fc2_weight = tensor_container_.get(root + "merger.linear_fc2.weight");
    vision.merger_fc2_bias = tensor_container_.get(root + "merger.linear_fc2.bias");

    const int depth = config.data.vision.depth;
    vision.blocks.resize(depth);
    for (int i = 0; i < depth; ++i) {
        const std::string prefix = root + "blocks." + std::to_string(i) + ".";
        VisionBlockWeights &b = vision.blocks[i];
        b.norm1_weight = tensor_container_.get(prefix + "norm1.weight");
        b.norm1_bias = tensor_container_.get(prefix + "norm1.bias");
        b.norm2_weight = tensor_container_.get(prefix + "norm2.weight");
        b.norm2_bias = tensor_container_.get(prefix + "norm2.bias");
        b.attn_qkv_weight = tensor_container_.get(prefix + "attn.qkv.weight");
        b.attn_qkv_bias = tensor_container_.get(prefix + "attn.qkv.bias");
        b.attn_proj_weight = tensor_container_.get(prefix + "attn.proj.weight");
        b.attn_proj_bias = tensor_container_.get(prefix + "attn.proj.bias");
        b.mlp_fc1_weight = tensor_container_.get(prefix + "mlp.linear_fc1.weight");
        b.mlp_fc1_bias = tensor_container_.get(prefix + "mlp.linear_fc1.bias");
        b.mlp_fc2_weight = tensor_container_.get(prefix + "mlp.linear_fc2.weight");
        b.mlp_fc2_bias = tensor_container_.get(prefix + "mlp.linear_fc2.bias");
    }
}

void QwenWeights::parse_mtp_weights(const QwenConfig &config) {
    const std::string root = "mtp.";
    mtp.fc_weight = tensor_container_.get(root + "fc.weight");
    mtp.norm_weight = tensor_container_.get(root + "norm.weight");
    mtp.pre_fc_norm_embedding_weight = tensor_container_.get(root + "pre_fc_norm_embedding.weight");
    mtp.pre_fc_norm_hidden_weight = tensor_container_.get(root + "pre_fc_norm_hidden.weight");

    const int num_layers = config.data.text.mtp_num_hidden_layers;
    mtp.layers.resize(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        const std::string prefix = root + "layers." + std::to_string(i) + ".";
        MtpLayerWeights &m = mtp.layers[i];
        m.input_norm = tensor_container_.get(prefix + "input_layernorm.weight");
        m.post_norm = tensor_container_.get(prefix + "post_attention_layernorm.weight");
        m.self_attn_q_proj = tensor_container_.get(prefix + "self_attn.q_proj.weight");
        m.self_attn_k_proj = tensor_container_.get(prefix + "self_attn.k_proj.weight");
        m.self_attn_v_proj = tensor_container_.get(prefix + "self_attn.v_proj.weight");
        m.self_attn_o_proj = tensor_container_.get(prefix + "self_attn.o_proj.weight");
        m.self_attn_q_norm = tensor_container_.get(prefix + "self_attn.q_norm.weight");
        m.self_attn_k_norm = tensor_container_.get(prefix + "self_attn.k_norm.weight");
        m.mlp_gate = tensor_container_.get(prefix + "mlp.gate_proj.weight");
        m.mlp_up = tensor_container_.get(prefix + "mlp.up_proj.weight");
        m.mlp_down = tensor_container_.get(prefix + "mlp.down_proj.weight");
    }
}

void QwenWeights::DebugDump() {
    std::ostringstream out;
    out << "QwenWeights:\n";
    out << "  tensors=" << tensor_container_.names().size() << "\n";

    auto dump_one = [&](const std::string &label, const WeightData &w) {
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

    dump_one("embed_tokens", embed_tokens);
    dump_one("final_norm", final_norm);
    for (size_t i = 0; i < layers.size(); ++i) {
        const LayerWeights &lw = layers[i];
        out << "  layer[" << i << "] type=" << lw.type << "\n";
        dump_one("  input_norm", lw.input_norm);
        dump_one("  post_norm", lw.post_norm);
        dump_one("  mlp.gate", lw.mlp.gate);
        dump_one("  mlp.up", lw.mlp.up);
        dump_one("  mlp.down", lw.mlp.down);
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
        dump_one("  input_norm", m.input_norm);
        dump_one("  post_norm", m.post_norm);
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
