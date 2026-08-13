//
// Created by zhangyoulun.
//

#ifndef LOCAL_LLM_SAFETENSORSFILE_H
#define LOCAL_LLM_SAFETENSORSFILE_H

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "format/TensorContainer.h"

// 模型无关的 safetensors 只读解析器：支持目录下多分片 mmap、header 解析、
// dtype 映射与按名取张量，实现 TensorContainer 接口。
//
// 不包含任何特定模型（如 Qwen）的权重组织或校验逻辑。
class SafeTensorsFile : public TensorContainer {
public:
    // 打开模型目录：mmap 目录下全部 *.safetensors 分片并解析各自 header。
    // 目录内无 .safetensors、或 header/offset 越界、或遇未支持 dtype 时抛异常。
    explicit SafeTensorsFile(const std::string &model_dir);
    ~SafeTensorsFile() override;

    SafeTensorsFile(const SafeTensorsFile &) = delete;
    SafeTensorsFile &operator=(const SafeTensorsFile &) = delete;

    // === TensorContainer 接口 ===
    bool contains(const std::string &name) const override;
    const TensorView &get(const std::string &name) const override;
    std::vector<std::string> names() const override;
    void DebugDump() const override;

    // safetensors dtype 字符串 -> 统一 DType；未支持类型抛异常。
    static DType dtype_from_string(const std::string &s);

private:
    // 一个 mmap 打开的分片文件，持有 fd 与映射内存生命周期。
    struct MappedShard {
        std::filesystem::path path;
        int fd = -1;
        size_t size = 0;
        const uint8_t *data = nullptr;
    };

    std::vector<MappedShard> shards_;
    // name -> 张量视图（data 指向对应分片 mmap 区域）。
    std::map<std::string, TensorView> views_;

    // mmap 打开单个分片，返回它在 shards_ 中的下标。
    size_t open_shard(const std::filesystem::path &path);
    // 解析第 shard_index 个分片的 JSON header，填充 infos_。
    void parse_shard_header(size_t shard_index);
    // 释放全部 mmap 与 fd。
    void close();

    // 从分片头读取小端 uint64 header 长度。
    static uint64_t read_u64_le(const uint8_t *data);
    // 查找并排序模型目录下全部 .safetensors 分片。
    static std::vector<std::filesystem::path> find_shards(const std::filesystem::path &model_dir);
    // shape -> 可读字符串。
    static std::string shape_to_string(const std::vector<int64_t> &shape);
};

#endif // LOCAL_LLM_SAFETENSORSFILE_H
