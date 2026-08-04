#include "safetensors.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <regex>
#include <stdexcept>

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

static MappedFile mmap_file(const fs::path & path) {
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

static uint64_t read_u64_le(const uint8_t * data) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(data[i]) << (8 * i);
    }
    return value;
}

static std::vector<int64_t> parse_i64_array(const std::string & text) {
    std::vector<int64_t> values;
    const std::regex number("-?[0-9]+");
    for (auto it = std::sregex_iterator(text.begin(), text.end(), number); it != std::sregex_iterator(); ++it) {
        values.push_back(std::stoll((*it).str()));
    }
    return values;
}

static void parse_safetensors_header(ModelWeights & weights, size_t file_index) {
    const MappedFile & file = weights.files[file_index];
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
        TensorInfo info;
        info.name = (*it)[1].str();
        info.dtype = (*it)[2].str();
        info.shape = parse_i64_array((*it)[3].str());
        info.data_begin = data_base + static_cast<size_t>(std::stoull((*it)[4].str()));
        info.data_end = data_base + static_cast<size_t>(std::stoull((*it)[5].str()));
        info.file_index = file_index;
        if (info.data_begin > info.data_end || info.data_end > file.size) {
            throw std::runtime_error("tensor data_offsets 越界：" + info.name);
        }
        weights.tensors.emplace(info.name, std::move(info));
    }
}

static std::vector<fs::path> find_safetensors_files(const fs::path & model_dir) {
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

ModelWeights load_weights_mmap(const fs::path & model_dir) {
    ModelWeights weights;
    for (const fs::path & file_path : find_safetensors_files(model_dir)) {
        weights.files.push_back(mmap_file(file_path));
        parse_safetensors_header(weights, weights.files.size() - 1);
    }
    return weights;
}

TensorRef tensor_ref(const ModelWeights & weights, const std::string & name) {
    auto it = weights.tensors.find(name);
    if (it == weights.tensors.end()) {
        throw std::runtime_error("缺少 tensor：" + name);
    }
    const TensorInfo & info = it->second;
    return TensorRef{ &info, weights.files[info.file_index].data + info.data_begin };
}

bool has_tensor(const ModelWeights & weights, const std::string & name) {
    return weights.tensors.find(name) != weights.tensors.end();
}

} // namespace llm_inference
