//
// Created by zhangyoulun.
//

#ifndef LOCAL_LLM_HFFILE_H
#define LOCAL_LLM_HFFILE_H

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "format/MF.h"
#include "format/hf/HFTokenizer.h"
#include "thirdparty/nlohmann/json.hpp"

// HuggingFace Transformers 模型目录只读解析器：
//   config.json + tokenizer.json + 多分片 safetensors。
//
// 张量通过 MF 接口统一暴露；config/tokenizer JSON 作为 HF 目录元信息暴露给上层。
class HFFile : public MF {
public:
    // 打开模型目录：解析 config.json / tokenizer.json，并 mmap 全部 *.safetensors 分片。
    // 缺少必要文件、header/offset 越界、或遇未支持 dtype 时抛异常。
    explicit HFFile(const std::string &model_dir);
    ~HFFile() override;

    HFFile(const HFFile &) = delete;
    HFFile &operator=(const HFFile &) = delete;

    std::vector<int> tokenizer_encode(const std::string &text) const override { return tokenizer_.encode(text); }
    std::string tokenizer_decode(const std::vector<int> &ids) const override { return tokenizer_.decode(ids); }

    // === MF tensor 接口 ===
    bool contain_tensor_view(const std::string &name) const override;
    const Tensor &get_tensor_view(const std::string &name) const override;
    std::vector<std::string> tensor_view_names() const override;
    bool contain_metadata(const std::string &key) const override;
    void debug_dump() const override;

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

    std::filesystem::path model_dir_;
    std::filesystem::path config_path_;
    std::filesystem::path tokenizer_path_;
    nlohmann::json config_json_;
    nlohmann::json tokenizer_json_;
    HFTokenizer tokenizer_;

    std::vector<MappedShard> shards_;
    // name -> 张量视图（data 指向对应分片 mmap 区域）。
    std::map<std::string, Tensor> views_;

    // mmap 打开单个分片，返回它在 shards_ 中的下标。
    size_t open_shard(const std::filesystem::path &path);
    // 解析第 shard_index 个分片的 JSON header，填充 TensorView 索引。
    void parse_shard_header(size_t shard_index);
    // 释放全部 mmap 与 fd。
    void close();

    // 从分片头读取小端 uint64 header 长度。
    static uint64_t read_u64_le(const uint8_t *data);
    // 查找并排序模型目录下全部 .safetensors 分片。
    static std::vector<std::filesystem::path> find_shards(const std::filesystem::path &model_dir);
    // 解析 HF 目录里的 JSON 文件。
    static nlohmann::json read_json_file(const std::filesystem::path &path);
    // 通过 "a.b.c" 点路径读取 config.json 中的节点。
    const nlohmann::json *metadata_node(const std::string &key) const;
    Metadata metadata_value(const std::string &key) const override;
    // shape -> 可读字符串。
    static std::string shape_to_string(const std::vector<int64_t> &shape);
};

#endif // LOCAL_LLM_HFFILE_H
