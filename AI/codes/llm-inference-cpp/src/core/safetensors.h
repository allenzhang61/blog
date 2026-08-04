#pragma once

#include "common.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace llm_inference {

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
struct ModelWeights {
    // mmap 后的 safetensors 文件列表。
    std::vector<MappedFile> files;
    // 按 tensor 名称索引的 metadata。
    std::map<std::string, TensorInfo> tensors;
};

// 扫描模型目录下的 safetensors 文件，并 mmap 加载 metadata。
ModelWeights load_weights_mmap(const fs::path & model_dir);

// 按名称返回 tensor 引用；不存在时抛出异常。
TensorRef tensor_ref(const ModelWeights & weights, const std::string & name);

// 检查权重中是否存在指定 tensor。
bool has_tensor(const ModelWeights & weights, const std::string & name);

} // namespace llm_inference
