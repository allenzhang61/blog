//
// Created by zhangyoulun.
//

#include "SafeTensorsFile.h"

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

SafeTensorsFile::SafeTensorsFile(const std::string &model_dir) {
    const fs::path dir(model_dir);
    for (const fs::path &shard_path : find_shards(dir)) {
        open_shard(shard_path);
        parse_shard_header(shards_.size() - 1);
    }
}

SafeTensorsFile::~SafeTensorsFile() {
    close();
}

void SafeTensorsFile::close() {
    for (MappedShard &shard : shards_) {
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

void SafeTensorsFile::open_shard(const fs::path &path) {
    MappedShard shard;
    shard.path = path;
    shard.fd = ::open(path.c_str(), O_RDONLY);
    if (shard.fd < 0) {
        throw std::runtime_error("open 失败：" + path.string() + "，" + std::strerror(errno));
    }

    struct stat st {};
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
}

uint64_t SafeTensorsFile::read_u64_le(const uint8_t *data) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(data[i]) << (8 * i);
    }
    return value;
}

std::vector<int64_t> SafeTensorsFile::parse_i64_array(const std::string &text) {
    std::vector<int64_t> values;
    const std::regex number("-?[0-9]+");
    for (auto it = std::sregex_iterator(text.begin(), text.end(), number); it != std::sregex_iterator(); ++it) {
        values.push_back(std::stoll((*it).str()));
    }
    return values;
}

DType SafeTensorsFile::dtype_from_string(const std::string &s) {
    if (s == "F32") return DType::F32;
    if (s == "F16") return DType::F16;
    if (s == "BF16") return DType::BF16;
    throw std::runtime_error("safetensors 不支持的 dtype: " + s);
}

void SafeTensorsFile::parse_shard_header(size_t shard_index) {
    const MappedShard &shard = shards_[shard_index];
    const uint64_t header_len = read_u64_le(shard.data);
    if (8 + header_len > shard.size) {
        throw std::runtime_error("safetensors header 长度异常：" + shard.path.string());
    }

    const std::string header(reinterpret_cast<const char *>(shard.data + 8), static_cast<size_t>(header_len));
    const size_t data_base = 8 + static_cast<size_t>(header_len);
    const std::regex tensor_pattern(
        R"REGEX("([^"]+)"\s*:\s*\{\s*"dtype"\s*:\s*"([^"]+)"\s*,\s*"shape"\s*:\s*\[([^\]]*)\]\s*,\s*"data_offsets"\s*:\s*\[\s*([0-9]+)\s*,\s*([0-9]+)\s*\])REGEX");

    for (auto it = std::sregex_iterator(header.begin(), header.end(), tensor_pattern);
         it != std::sregex_iterator();
         ++it) {
        const std::string name = (*it)[1].str();
        const std::string dtype = (*it)[2].str();
        const std::vector<int64_t> shape = parse_i64_array((*it)[3].str());
        const size_t data_begin = data_base + static_cast<size_t>(std::stoull((*it)[4].str()));
        const size_t data_end = data_base + static_cast<size_t>(std::stoull((*it)[5].str()));
        if (data_begin > data_end || data_end > shard.size) {
            throw std::runtime_error("tensor data_offsets 越界：" + name);
        }
        // 提前校验 dtype 合法（未支持类型此处抛异常）。
        (void) dtype_from_string(dtype);

        SafeTensorInfo info;
        info.name = name;
        info.dtype = dtype;
        info.shape = shape;
        info.data = shard.data + data_begin;
        info.nbytes = data_end - data_begin;
        infos_.emplace(name, std::move(info));
    }
}

std::vector<fs::path> SafeTensorsFile::find_shards(const fs::path &model_dir) {
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

bool SafeTensorsFile::has_tensor(const std::string &name) const {
    return infos_.count(name) > 0;
}

const SafeTensorInfo &SafeTensorsFile::info(const std::string &name) const {
    auto it = infos_.find(name);
    if (it == infos_.end()) {
        throw std::runtime_error("safetensors 缺少张量：" + name);
    }
    return it->second;
}

const TensorView &SafeTensorsFile::tensor(const std::string &name) const {
    auto cached = view_cache_.find(name);
    if (cached != view_cache_.end()) {
        return cached->second;
    }
    const SafeTensorInfo &st = info(name);
    TensorView view;
    view.name = st.name;
    view.shape = st.shape; // safetensors 本就是行主序 [out, in]
    view.dtype = dtype_from_string(st.dtype);
    view.data = st.data;
    view.nbytes = st.nbytes;
    auto [it, _] = view_cache_.emplace(name, std::move(view));
    return it->second;
}

std::vector<std::string> SafeTensorsFile::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(infos_.size());
    for (const auto &[name, _] : infos_) {
        names.push_back(name);
    }
    return names;
}

void SafeTensorsFile::DebugDump() const {
    std::ostringstream out;
    out << "SafeTensorsFile:\n";
    out << "  shards=" << shards_.size() << " tensors=" << infos_.size() << "\n";
    for (const MappedShard &shard : shards_) {
        out << "  shard=" << shard.path.filename().string() << " bytes=" << shard.size << "\n";
    }
    out << "  === tensors (" << infos_.size() << ") ===\n";
    for (const auto &[name, st] : infos_) {
        out << "  " << name
            << " dtype=" << st.dtype
            << " shape=" << shape_to_string(st.shape)
            << " nbytes=" << st.nbytes << "\n";
    }
    Log::debug(out.str());
}
