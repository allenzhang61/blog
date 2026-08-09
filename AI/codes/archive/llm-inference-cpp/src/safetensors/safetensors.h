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

// 单个模型权重在 safetensors 文件中的 metadata。
struct WeightMeta {
    // 权重名称。
    std::string name;
    // 权重 dtype，例如 BF16/F16/F32。
    std::string dtype;
    // 权重 shape。
    std::vector<int64_t> shape;
    // 权重数据在文件数据区内的起始偏移。
    size_t data_begin = 0;
    // 权重数据在文件数据区内的结束偏移。
    size_t data_end = 0;
    // 权重所在的 files 下标。
    size_t file_index = 0;
};

// 指向已 mmap 权重数据的轻量引用，不拥有内存。
struct WeightData {
    // 权重 metadata 指针。
    const WeightMeta * info = nullptr;
    // 权重原始数据起始地址。
    const uint8_t * data = nullptr;
};

// 模型全部权重文件和 metadata 索引。
class ModelWeights {
public:
    // 扫描模型目录下的 safetensors 文件，并 mmap 加载 metadata。
    static ModelWeights load_mmap(const fs::path & model_dir);

    // 校验 Qwen3.5 推理路径需要的 tensor 是否齐全。
    void validate_qwen_tensors(const ModelConfig & config) const;

    // 打印当前权重中所有 tensor 的名称、dtype、shape 和文件信息。
    void dump_tensors() const;

    // 按名称返回权重数据引用；不存在时抛出异常。
    WeightData weight_data(const std::string & name) const;

    // 已 mmap 的 safetensors 文件数量。
    size_t mapped_file_count() const;

    // 已索引的 tensor 数量。
    size_t tensor_count() const;

private:
    // mmap 后的 safetensors 文件列表。
    std::vector<MappedFile> files;
    // 按权重名称索引的 metadata。
    std::map<std::string, WeightMeta> tensors;

    // 检查权重中是否存在指定 tensor。
    bool has_tensor(const std::string & name) const;

    // 解析单个 safetensors 文件 header 并填充 tensor 索引。
    void parse_safetensors_header(size_t file_index);

    // mmap 打开单个 safetensors 文件。
    static MappedFile mmap_file(const fs::path & path);

    // 从 safetensors 文件头读取 little-endian uint64 header 长度。
    static uint64_t read_u64_le(const uint8_t * data);

    // 解析 JSON 片段中的 int64 数组，例如 tensor shape。
    static std::vector<int64_t> parse_i64_array(const std::string & text);

    // 查找模型目录下所有 .safetensors 权重文件并排序。
    static std::vector<fs::path> find_safetensors_files(const fs::path & model_dir);

    // 将 shape 转为日志/错误信息中使用的可读字符串。
    static std::string shape_to_string(const std::vector<int64_t> & shape);
};

} // namespace llm_inference
