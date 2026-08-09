#include "safetensors.h"

#include "../core/config.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <sstream>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace llm_inference {


MappedFile::MappedFile(MappedFile && other) noexcept {
    *this = std::move(other);
}

MappedFile & MappedFile::operator=(MappedFile && other) noexcept {
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

MappedFile::~MappedFile() {
    close();
}

void MappedFile::close() {
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

MappedFile ModelWeights::mmap_file(const fs::path & path) {
    MappedFile file;
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

    void * ptr = mmap(nullptr, file.size, PROT_READ, MAP_PRIVATE, file.fd, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("mmap 失败：" + path.string() + "，" + std::strerror(errno));
    }
    file.data = static_cast<const uint8_t *>(ptr);
    return file;
}

uint64_t ModelWeights::read_u64_le(const uint8_t * data) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(data[i]) << (8 * i);
    }
    return value;
}

std::vector<int64_t> ModelWeights::parse_i64_array(const std::string & text) {
    std::vector<int64_t> values;
    const std::regex number("-?[0-9]+");
    for (auto it = std::sregex_iterator(text.begin(), text.end(), number); it != std::sregex_iterator(); ++it) {
        values.push_back(std::stoll((*it).str()));
    }
    return values;
}

void ModelWeights::parse_safetensors_header(size_t file_index) {
    const MappedFile & file = files[file_index];
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
        tensors.emplace(info.name, std::move(info));
    }
}

std::vector<fs::path> ModelWeights::find_safetensors_files(const fs::path & model_dir) {
    std::vector<fs::path> files;
    for (const auto & entry : fs::directory_iterator(model_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        throw std::runtime_error("模型目录下没有 .safetensors 文件：" + model_dir.string());
    }
    return files;
}

ModelWeights ModelWeights::load_mmap(const fs::path & model_dir) {
    ModelWeights weights;
    for (const fs::path & file_path : find_safetensors_files(model_dir)) {
        weights.files.push_back(mmap_file(file_path));
        weights.parse_safetensors_header(weights.files.size() - 1);
    }
    return weights;
}

WeightData ModelWeights::weight_data(const std::string & name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        throw std::runtime_error("缺少 tensor：" + name);
    }
    const WeightMeta & info = it->second;
    return WeightData{ &info, files[info.file_index].data + info.data_begin };
}

size_t ModelWeights::mapped_file_count() const {
    return files.size();
}

size_t ModelWeights::tensor_count() const {
    return tensors.size();
}

bool ModelWeights::has_tensor(const std::string & name) const {
    return tensors.find(name) != tensors.end();
}

std::string ModelWeights::shape_to_string(const std::vector<int64_t> & shape) {
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

void ModelWeights::dump_tensors() const {
    for (const auto & [name, info] : tensors) {
        std::cerr << name << " dtype=" << info.dtype
                  << " shape=" << shape_to_string(info.shape)
                  << " file=" << files[info.file_index].path.filename().string()
                  << " bytes=" << (info.data_end - info.data_begin)
                  << "\n";
    }
}

void ModelWeights::validate_qwen_tensors(const ModelConfig & config) const {
    const std::string root = "model.language_model.";
    for (const std::string & name : {
             root + "embed_tokens.weight",
             root + "norm.weight",
         }) {
        if (!has_tensor(name)) {
            throw std::runtime_error("缺少 tensor：" + name);
        }
    }

    for (int layer = 0; layer < config.text.num_hidden_layers; ++layer) {
        const std::string prefix = root + "layers." + std::to_string(layer) + ".";
        for (const std::string & name : {
                 prefix + "input_layernorm.weight",
                 prefix + "post_attention_layernorm.weight",
                 prefix + "mlp.gate_proj.weight",
                 prefix + "mlp.up_proj.weight",
                 prefix + "mlp.down_proj.weight",
             }) {
            if (!has_tensor(name)) {
                throw std::runtime_error("缺少 tensor：" + name);
            }
        }
        if (config.text.layer_types[layer] == "linear_attention") {
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
                if (!has_tensor(name)) {
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
                if (!has_tensor(name)) {
                    throw std::runtime_error("缺少 tensor：" + name);
                }
            }
        }
    }
}

} // namespace llm_inference
