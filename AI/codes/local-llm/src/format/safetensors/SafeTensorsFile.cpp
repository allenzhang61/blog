//
// Created by zhangyoulun.
//

#include "SafeTensorsFile.h"

#include "thirdparty/nlohmann/json.hpp"
#include "utils/log/Log.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

SafeTensorsFile::SafeTensorsFile(const std::string &model_dir) {
    const fs::path dir(model_dir);
    for (const fs::path &shard_path: find_shards(dir)) {
        const size_t shard_index = open_shard(shard_path);
        parse_shard_header(shard_index);
    }
}

SafeTensorsFile::~SafeTensorsFile() {
    close();
}

void SafeTensorsFile::close() {
    for (MappedShard &shard: shards_) {
        if (shard.data != nullptr && shard.size > 0) {
            munmap(const_cast<uint8_t *>(shard.data), shard.size);
        }
        if (shard.fd >= 0) {
            ::close(shard.fd);
        }
        shard.data = nullptr;
        shard.fd = -1;
        shard.size = 0;
    }
    shards_.clear();
}

// 将 safetensors文件以 mmap的方式记录到shards_数组中
size_t SafeTensorsFile::open_shard(const fs::path &path) {
    MappedShard shard;
    shard.path = path;
    shard.fd = ::open(path.c_str(), O_RDONLY);
    if (shard.fd < 0) {
        throw std::runtime_error("open 失败：" + path.string() + "，" + std::strerror(errno));
    }

    struct stat st{};
    if (fstat(shard.fd, &st) != 0) {
        ::close(shard.fd);
        throw std::runtime_error("fstat 失败：" + path.string() + "，" + std::strerror(errno));
    }
    shard.size = static_cast<size_t>(st.st_size);
    if (shard.size < 8) {
        ::close(shard.fd);
        throw std::runtime_error("safetensors 文件太小：" + path.string());
    }

    void *ptr = mmap(nullptr, shard.size, PROT_READ, MAP_PRIVATE, shard.fd, 0);
    if (ptr == MAP_FAILED) {
        ::close(shard.fd);
        throw std::runtime_error("mmap 失败：" + path.string() + "，" + std::strerror(errno));
    }
    shard.data = static_cast<const uint8_t *>(ptr);
    shards_.push_back(shard);
    return shards_.size() - 1;
}

uint64_t SafeTensorsFile::read_u64_le(const uint8_t *data) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(data[i]) << (8 * i);
    }
    return value;
}

DType SafeTensorsFile::dtype_from_string(const std::string &s) {
    //todo 待支持其他类型，低优
    if (s == "F32") return DType::F32;
    if (s == "F16") return DType::F16;
    if (s == "BF16") return DType::BF16;
    throw std::runtime_error("safetensors 不支持的 dtype: " + s);
}

//解析safetensors文件中的 header 信息到 infos_ 数组中
void SafeTensorsFile::parse_shard_header(size_t shard_index) {
    const MappedShard &shard = shards_[shard_index];
    const uint64_t header_len = read_u64_le(shard.data);
    if (header_len > shard.size - 8) {//8代表 u64 解析出的前 8 个字节
        throw std::runtime_error("safetensors header 长度异常：" + shard.path.string());
    }

    const std::string header(reinterpret_cast<const char *>(shard.data + 8), static_cast<size_t>(header_len));
    const size_t data_base = 8 + static_cast<size_t>(header_len);

    nlohmann::json root;
    try {
        root = nlohmann::json::parse(header);
    } catch (const nlohmann::json::exception &e) {
        throw std::runtime_error("safetensors header JSON 解析失败：" + shard.path.string() + "，" + e.what());
    }

    for (const auto &[name, tensor_json] : root.items()) {
        if (name == "__metadata__") {
            continue;
        }

        const std::string dtype = tensor_json.at("dtype").get<std::string>();
        const std::vector<int64_t> shape = tensor_json.at("shape").get<std::vector<int64_t>>();
        const std::vector<size_t> offsets = tensor_json.at("data_offsets").get<std::vector<size_t>>();
        if (offsets.size() != 2) {
            throw std::runtime_error("tensor data_offsets 数量异常：" + name);
        }

        const size_t data_begin = data_base + offsets[0];
        const size_t data_end = data_base + offsets[1];
        if (data_begin > data_end || data_end > shard.size) {
            throw std::runtime_error("tensor data_offsets 越界：" + name);
        }
        TensorView view;
        view.name = name;
        view.shape = shape; // safetensors 本就是行主序 [out, in]
        view.dtype = dtype_from_string(dtype);
        view.data = shard.data + data_begin;
        view.nbytes = data_end - data_begin;
        views_.emplace(name, std::move(view));
    }
}

// 找到所有的*.safetensors文件，并根据文件名排序
std::vector<fs::path> SafeTensorsFile::find_shards(const fs::path &model_dir) {
    std::vector<fs::path> found;
    for (const auto &entry: fs::directory_iterator(model_dir)) {
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

std::string SafeTensorsFile::shape_to_string(const std::vector<int64_t> &shape) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) out << ", ";
        out << shape[i];
    }
    out << "]";
    return out.str();
}

bool SafeTensorsFile::contains(const std::string &name) const {
    return views_.contains(name);
}

const TensorView &SafeTensorsFile::get(const std::string &name) const {
    auto it = views_.find(name);
    if (it == views_.end()) {
        throw std::runtime_error("safetensors 缺少张量：" + name);
    }
    return it->second;
}

std::vector<std::string> SafeTensorsFile::names() const {
    std::vector<std::string> names;
    names.reserve(views_.size());
    for (const auto &[name, _]: views_) {
        names.push_back(name);
    }
    return names;
}

void SafeTensorsFile::DebugDump() const {
    std::ostringstream out;
    out << "SafeTensorsFile:\n";
    out << "  shards=" << shards_.size() << " tensors=" << views_.size() << "\n";
    for (const MappedShard &shard: shards_) {
        out << "  shard=" << shard.path.filename().string() << " bytes=" << shard.size << "\n";
    }
    out << "  === tensors (" << views_.size() << ") ===\n";
    for (const auto &[name, view]: views_) {
        out << "  " << name
                << " dtype=" << dtype_name(view.dtype)
                << " shape=" << shape_to_string(view.shape)
                << " nbytes=" << view.nbytes << "\n";
    }
    Log::debug(out.str());
}
