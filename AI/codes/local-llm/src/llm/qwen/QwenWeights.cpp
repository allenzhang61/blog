//
// Created by zhangyoulun on 8/8/2026.
//

#include "QwenWeights.h"

#include "QwenConfig.h"
#include "utils/log/Log.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <regex>
#include <sstream>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;


MmapFile::MmapFile(MmapFile &&other) noexcept {
    *this = std::move(other);
}

MmapFile &MmapFile::operator=(MmapFile &&other) noexcept {
    if (this != &other) {
        close();
        path = std::move(other.path);
        fd = other.fd;
        size = other.size;
        data = other.data;
        other.fd = -1;
        other.size = 0;
        other.data = nullptr;
    }
    return *this;
}

MmapFile::~MmapFile() {
    close();
}

void MmapFile::close() {
    if (data != nullptr && size > 0) {
        munmap(const_cast<uint8_t *>(data), size);
    }
    if (fd >= 0) {
        ::close(fd);
    }
    data = nullptr;
    fd = -1;
    size = 0;
}

MmapFile QwenWeights::mmap_file(const fs::path &path) {
    MmapFile file;
    file.path = path;
    file.fd = ::open(path.c_str(), O_RDONLY);
    if (file.fd < 0) {
        throw std::runtime_error("open 失败：" + path.string() + "，" + std::strerror(errno));
    }

    struct stat st {};
    if (fstat(file.fd, &st) != 0) {
        throw std::runtime_error("fstat 失败：" + path.string() + "，" + std::strerror(errno));
    }
    file.size = static_cast<size_t>(st.st_size);
    if (file.size < 8) {
        throw std::runtime_error("safetensors 文件太小：" + path.string());
    }

    void *ptr = mmap(nullptr, file.size, PROT_READ, MAP_PRIVATE, file.fd, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("mmap 失败：" + path.string() + "，" + std::strerror(errno));
    }
    file.data = static_cast<const uint8_t *>(ptr);
    return file;
}

uint64_t QwenWeights::read_u64_le(const uint8_t *data) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(data[i]) << (8 * i);
    }
    return value;
}

std::vector<int64_t> QwenWeights::parse_i64_array(const std::string &text) {
    std::vector<int64_t> values;
    const std::regex number("-?[0-9]+");
    for (auto it = std::sregex_iterator(text.begin(), text.end(), number); it != std::sregex_iterator(); ++it) {
        values.push_back(std::stoll((*it).str()));
    }
    return values;
}

void QwenWeights::parse_safetensors_header(size_t file_index) {
    const MmapFile &file = mmapFiles[file_index];
    const uint64_t header_len = read_u64_le(file.data);
    if (8 + header_len > file.size) {
        throw std::runtime_error("safetensors header 长度异常：" + file.path.string());
    }

    const std::string header(reinterpret_cast<const char *>(file.data + 8), static_cast<size_t>(header_len));
    const size_t data_base = 8 + static_cast<size_t>(header_len);
    const std::regex tensor_pattern(
        R"REGEX("([^"]+)"\s*:\s*\{\s*"dtype"\s*:\s*"([^"]+)"\s*,\s*"shape"\s*:\s*\[([^\]]*)\]\s*,\s*"data_offsets"\s*:\s*\[\s*([0-9]+)\s*,\s*([0-9]+)\s*\])REGEX");

    for (auto it = std::sregex_iterator(header.begin(), header.end(), tensor_pattern);
         it != std::sregex_iterator();
         ++it) {
        WeightMeta info;
        info.name = (*it)[1].str();
        info.dtype = (*it)[2].str();
        info.shape = parse_i64_array((*it)[3].str());
        info.data_begin = data_base + static_cast<size_t>(std::stoull((*it)[4].str()));
        info.data_end = data_base + static_cast<size_t>(std::stoull((*it)[5].str()));
        info.file_index = file_index;
        if (info.data_begin > info.data_end || info.data_end > file.size) {
            throw std::runtime_error("tensor data_offsets 越界：" + info.name);
        }
        metas.emplace(info.name, std::move(info));
    }
}

std::vector<fs::path> QwenWeights::find_safetensors_files(const fs::path &model_dir) {
    std::vector<fs::path> found;
    for (const auto &entry : fs::directory_iterator(model_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            found.push_back(entry.path());
        }
    }
    std::sort(found.begin(), found.end());
    if (found.empty()) {
        throw std::runtime_error("模型目录下没有 .safetensors 文件：" + model_dir.string());
    }
    return found;
}

void QwenWeights::build_weight_index() {
    for (const auto &[name, info] : metas) {
        weights.emplace(name, WeightData{ &info, mmapFiles[info.file_index].data + info.data_begin });
    }
}

WeightData QwenWeights::weight_data(const std::string &name) const {
    auto it = weights.find(name);
    if (it == weights.end()) {
        throw std::runtime_error("缺少 tensor：" + name);
    }
    return it->second;
}

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
        if (weights.find(name) == weights.end()) {
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
            if (weights.find(name) == weights.end()) {
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
                if (weights.find(name) == weights.end()) {
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
                if (weights.find(name) == weights.end()) {
                    throw std::runtime_error("缺少 tensor：" + name);
                }
            }
        }
    }
}

QwenWeights::QwenWeights(const std::string &model_dir, const QwenConfig &config) {
    const fs::path dir(model_dir);

    // mmap 打开目录下的所有 safetensors 文件，并解析其 header 填充 tensor 索引。
    for (const fs::path &file_path : find_safetensors_files(dir)) {
        mmapFiles.push_back(mmap_file(file_path));
        parse_safetensors_header(mmapFiles.size() - 1);
    }
    // 建立 name -> WeightData 索引（data 指向已 mmap 内存，不拷贝权重数据）。
    build_weight_index();

    const int num_hidden_layers = config.data.text.num_hidden_layers;
    const std::vector<std::string> &layer_types = config.data.text.layer_types;
    validate_qwen_tensors(num_hidden_layers, layer_types);

    const std::string root = "model.language_model.";
    embed_tokens = weight_data(root + "embed_tokens.weight");
    final_norm = weight_data(root + "norm.weight");
    layers.resize(num_hidden_layers);

    for (int layer = 0; layer < num_hidden_layers; ++layer) {
        const std::string prefix = root + "layers." + std::to_string(layer) + ".";
        LayerWeights &lw = layers[layer];
        lw.type = layer_types[layer];
        lw.input_norm = weight_data(prefix + "input_layernorm.weight");
        lw.post_norm = weight_data(prefix + "post_attention_layernorm.weight");
        lw.mlp.gate = weight_data(prefix + "mlp.gate_proj.weight");
        lw.mlp.up = weight_data(prefix + "mlp.up_proj.weight");
        lw.mlp.down = weight_data(prefix + "mlp.down_proj.weight");

        if (lw.type == "linear_attention") {
            lw.lin.in_proj_qkv = weight_data(prefix + "linear_attn.in_proj_qkv.weight");
            lw.lin.in_proj_z = weight_data(prefix + "linear_attn.in_proj_z.weight");
            lw.lin.in_proj_b = weight_data(prefix + "linear_attn.in_proj_b.weight");
            lw.lin.in_proj_a = weight_data(prefix + "linear_attn.in_proj_a.weight");
            lw.lin.conv1d = weight_data(prefix + "linear_attn.conv1d.weight");
            lw.lin.a_log = weight_data(prefix + "linear_attn.A_log");
            lw.lin.dt_bias = weight_data(prefix + "linear_attn.dt_bias");
            lw.lin.norm = weight_data(prefix + "linear_attn.norm.weight");
            lw.lin.out_proj = weight_data(prefix + "linear_attn.out_proj.weight");
        } else {
            lw.full.q_proj = weight_data(prefix + "self_attn.q_proj.weight");
            lw.full.k_proj = weight_data(prefix + "self_attn.k_proj.weight");
            lw.full.v_proj = weight_data(prefix + "self_attn.v_proj.weight");
            lw.full.q_norm = weight_data(prefix + "self_attn.q_norm.weight");
            lw.full.k_norm = weight_data(prefix + "self_attn.k_norm.weight");
            lw.full.o_proj = weight_data(prefix + "self_attn.o_proj.weight");
        }
    }

    // 解析视觉塔和 MTP 权重；已解析但当前未使用（纯文本推理不走这两条路径）。
    parse_vision_weights(config);
    parse_mtp_weights(config);
}

void QwenWeights::parse_vision_weights(const QwenConfig &config) {
    const std::string root = "model.visual.";
    vision.patch_embed_proj_weight = weight_data(root + "patch_embed.proj.weight");
    vision.patch_embed_proj_bias = weight_data(root + "patch_embed.proj.bias");
    vision.pos_embed_weight = weight_data(root + "pos_embed.weight");
    vision.merger_norm_weight = weight_data(root + "merger.norm.weight");
    vision.merger_norm_bias = weight_data(root + "merger.norm.bias");
    vision.merger_fc1_weight = weight_data(root + "merger.linear_fc1.weight");
    vision.merger_fc1_bias = weight_data(root + "merger.linear_fc1.bias");
    vision.merger_fc2_weight = weight_data(root + "merger.linear_fc2.weight");
    vision.merger_fc2_bias = weight_data(root + "merger.linear_fc2.bias");

    const int depth = config.data.vision.depth;
    vision.blocks.resize(depth);
    for (int i = 0; i < depth; ++i) {
        const std::string prefix = root + "blocks." + std::to_string(i) + ".";
        VisionBlockWeights &b = vision.blocks[i];
        b.norm1_weight = weight_data(prefix + "norm1.weight");
        b.norm1_bias = weight_data(prefix + "norm1.bias");
        b.norm2_weight = weight_data(prefix + "norm2.weight");
        b.norm2_bias = weight_data(prefix + "norm2.bias");
        b.attn_qkv_weight = weight_data(prefix + "attn.qkv.weight");
        b.attn_qkv_bias = weight_data(prefix + "attn.qkv.bias");
        b.attn_proj_weight = weight_data(prefix + "attn.proj.weight");
        b.attn_proj_bias = weight_data(prefix + "attn.proj.bias");
        b.mlp_fc1_weight = weight_data(prefix + "mlp.linear_fc1.weight");
        b.mlp_fc1_bias = weight_data(prefix + "mlp.linear_fc1.bias");
        b.mlp_fc2_weight = weight_data(prefix + "mlp.linear_fc2.weight");
        b.mlp_fc2_bias = weight_data(prefix + "mlp.linear_fc2.bias");
    }
}

void QwenWeights::parse_mtp_weights(const QwenConfig &config) {
    const std::string root = "mtp.";
    mtp.fc_weight = weight_data(root + "fc.weight");
    mtp.norm_weight = weight_data(root + "norm.weight");
    mtp.pre_fc_norm_embedding_weight = weight_data(root + "pre_fc_norm_embedding.weight");
    mtp.pre_fc_norm_hidden_weight = weight_data(root + "pre_fc_norm_hidden.weight");

    const int num_layers = config.data.text.mtp_num_hidden_layers;
    mtp.layers.resize(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        const std::string prefix = root + "layers." + std::to_string(i) + ".";
        MtpLayerWeights &m = mtp.layers[i];
        m.input_norm = weight_data(prefix + "input_layernorm.weight");
        m.post_norm = weight_data(prefix + "post_attention_layernorm.weight");
        m.self_attn_q_proj = weight_data(prefix + "self_attn.q_proj.weight");
        m.self_attn_k_proj = weight_data(prefix + "self_attn.k_proj.weight");
        m.self_attn_v_proj = weight_data(prefix + "self_attn.v_proj.weight");
        m.self_attn_o_proj = weight_data(prefix + "self_attn.o_proj.weight");
        m.self_attn_q_norm = weight_data(prefix + "self_attn.q_norm.weight");
        m.self_attn_k_norm = weight_data(prefix + "self_attn.k_norm.weight");
        m.mlp_gate = weight_data(prefix + "mlp.gate_proj.weight");
        m.mlp_up = weight_data(prefix + "mlp.up_proj.weight");
        m.mlp_down = weight_data(prefix + "mlp.down_proj.weight");
    }
}

void QwenWeights::DebugDump() {
    std::ostringstream out;
    out << "QwenWeights:\n";
    out << "  mapped_files=" << mmapFiles.size() << " tensors=" << metas.size() << "\n";
    for (const MmapFile &file : mmapFiles) {
        out << "  file=" << file.path.filename().string() << " bytes=" << file.size << "\n";
    }

    auto dump_one = [&](const std::string &label, const WeightData &w) {
        out << "  " << label << ": ";
        if (w.info == nullptr) {
            out << "<null>\n";
            return;
        }
        out << w.info->name
            << " dtype=" << w.info->dtype
            << " shape=" << shape_to_string(w.info->shape)
            << " bytes=" << (w.info->data_end - w.info->data_begin) << "\n";
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
