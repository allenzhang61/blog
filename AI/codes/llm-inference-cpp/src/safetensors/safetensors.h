#pragma once

#include "../core/common.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace llm_inference {

struct ModelConfig;

// mmap 打开的权重文件，负责持有 fd 和映射内存生命周期。
struct MappedFile {
    // 文件路径。
    fs::path path;
    // 打开的文件描述符，-1 表示无效。
    int fd = -1;
    // 文件总字节数。
    size_t size = 0;
    // mmap 后的只读数据起始地址。
    const uint8_t * data = nullptr;

    MappedFile() = default;
    MappedFile(const MappedFile &) = delete;
    MappedFile & operator=(const MappedFile &) = delete;
    MappedFile(MappedFile && other) noexcept;
    MappedFile & operator=(MappedFile && other) noexcept;
    ~MappedFile();

    // 释放 mmap 和文件描述符。
    void close();
};

// 单个 safetensors tensor 的 metadata。
struct TensorInfo {
    // tensor 名称。
    std::string name;
    // tensor dtype，例如 BF16/F16/F32。
    std::string dtype;
    // tensor shape。
    std::vector<int64_t> shape;
    // tensor 数据在文件数据区内的起始偏移。
    size_t data_begin = 0;
    // tensor 数据在文件数据区内的结束偏移。
    size_t data_end = 0;
    // tensor 所在的 files 下标。
    size_t file_index = 0;
};

// 指向已 mmap tensor 数据的轻量引用，不拥有内存。
struct TensorRef {
    // tensor metadata 指针。
    const TensorInfo * info = nullptr;
    // tensor 原始数据起始地址。
    const uint8_t * data = nullptr;
};

// 模型全部权重文件和 tensor metadata 索引。
class ModelWeights {
public:
    // 扫描模型目录下的 safetensors 文件，并 mmap 加载 metadata。
    static ModelWeights load_mmap(const fs::path & model_dir);

    // 校验 Qwen3.5 推理路径需要的 tensor 是否齐全。
    void validate_qwen_tensors(const ModelConfig & config) const;

    // 打印当前权重中所有 tensor 的名称、dtype、shape 和文件信息。
    void dump_tensors() const;

    // 按名称返回 tensor 引用；不存在时抛出异常。
    TensorRef tensor_ref(const std::string & name) const;

    // 已 mmap 的 safetensors 文件数量。
    size_t mapped_file_count() const;

    // 已索引的 tensor 数量。
    size_t tensor_count() const;

private:
    // mmap 后的 safetensors 文件列表。
    std::vector<MappedFile> files;
    // 按 tensor 名称索引的 metadata。
    std::map<std::string, TensorInfo> tensors;

    // 检查权重中是否存在指定 tensor。
    bool has_tensor(const std::string & name) const;

    // 解析单个 safetensors 文件 header 并填充 tensor 索引。
    void parse_safetensors_header(size_t file_index);

    // mmap 打开单个 safetensors 文件。
    static MappedFile mmap_file(const fs::path & path);

    static uint64_t read_u64_le(const uint8_t * data);
    static std::vector<int64_t> parse_i64_array(const std::string & text);
    static std::vector<fs::path> find_safetensors_files(const fs::path & model_dir);
    static std::string shape_to_string(const std::vector<int64_t> & shape);
};

} // namespace llm_inference
