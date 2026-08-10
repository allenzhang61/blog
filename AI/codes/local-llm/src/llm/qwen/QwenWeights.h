//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_QWENWEIGHTS_H
#define LOCAL_LLM_QWENWEIGHTS_H
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "format/TensorContainer.h"

class QwenConfig;

// 单个模型权重在张量容器中的 metadata。
struct WeightMeta {
    // 权重名称。
    std::string name;
    // 权重 dtype，例如 BF16/F16/F32。
    DType dtype = DType::UNKNOWN;
    // 权重 shape。
    std::vector<int64_t> shape;
    // 权重数据字节数。
    size_t nbytes = 0;
};

// 指向已 mmap 权重数据的轻量引用，不拥有内存。
struct WeightData {
    // 权重 metadata 指针。
    const WeightMeta *info = nullptr;
    // 权重原始数据起始地址。
    const uint8_t *data = nullptr;
};

// linear attention 层一次性解析好的权重引用。
struct LinearAttnWeights {
    WeightData in_proj_qkv;
    WeightData in_proj_z;
    WeightData in_proj_b;
    WeightData in_proj_a;
    WeightData conv1d;
    WeightData a_log;
    WeightData dt_bias;
    WeightData norm;
    WeightData out_proj;
};

// full attention 层一次性解析好的权重引用。
struct FullAttnWeights {
    WeightData q_proj;
    WeightData k_proj;
    WeightData v_proj;
    WeightData q_norm;
    WeightData k_norm;
    WeightData o_proj;
};

// MLP 层一次性解析好的权重引用。
struct MlpWeights {
    WeightData gate;
    WeightData up;
    WeightData down;
};

// 单个 transformer 层解析好的权重引用集合。
struct LayerWeights {
    // 层类型："linear_attention" 或 "full_attention"。
    std::string type;
    WeightData input_norm;
    WeightData post_norm;
    // 仅 linear_attention 层有效。
    LinearAttnWeights lin;
    // 仅 full_attention 层有效。
    FullAttnWeights full;
    MlpWeights mlp;
};

// 视觉塔单个 transformer block 的权重引用；当前未使用（纯文本推理不走视觉分支）。
struct VisionBlockWeights {
    WeightData norm1_weight;
    WeightData norm1_bias;
    WeightData norm2_weight;
    WeightData norm2_bias;
    WeightData attn_qkv_weight;
    WeightData attn_qkv_bias;
    WeightData attn_proj_weight;
    WeightData attn_proj_bias;
    WeightData mlp_fc1_weight;
    WeightData mlp_fc1_bias;
    WeightData mlp_fc2_weight;
    WeightData mlp_fc2_bias;
};

// 视觉塔（model.visual.*）解析好的权重引用集合；当前未使用。
struct VisionWeights {
    WeightData patch_embed_proj_weight;
    WeightData patch_embed_proj_bias;
    WeightData pos_embed_weight;
    std::vector<VisionBlockWeights> blocks;
    WeightData merger_norm_weight;
    WeightData merger_norm_bias;
    WeightData merger_fc1_weight;
    WeightData merger_fc1_bias;
    WeightData merger_fc2_weight;
    WeightData merger_fc2_bias;
};

// MTP（多 token 预测）单层权重引用；当前未使用。
struct MtpLayerWeights {
    WeightData input_norm;
    WeightData post_norm;
    WeightData self_attn_q_proj;
    WeightData self_attn_k_proj;
    WeightData self_attn_v_proj;
    WeightData self_attn_o_proj;
    WeightData self_attn_q_norm;
    WeightData self_attn_k_norm;
    WeightData mlp_gate;
    WeightData mlp_up;
    WeightData mlp_down;
};

// MTP（mtp.*）解析好的权重引用集合；当前未使用。
struct MtpWeights {
    WeightData fc_weight;
    WeightData norm_weight;
    WeightData pre_fc_norm_embedding_weight;
    WeightData pre_fc_norm_hidden_weight;
    std::vector<MtpLayerWeights> layers;
};

class QwenWeights {
public:
    QwenWeights(const std::string &model_dir, const QwenConfig &config);

    void DebugDump();

    WeightData embed_tokens;
    WeightData final_norm;
    std::vector<LayerWeights> layers;

    // 视觉塔权重（model.visual.*）；已解析但当前未使用（纯文本推理不走视觉分支）。
    VisionWeights vision;
    // MTP 权重（mtp.*）；已解析但当前未使用（不走多 token 预测路径）。
    MtpWeights mtp;

private:
    // 张量容器（safetensors 格式；负责 mmap、header 解析、按名取张量）。
    std::unique_ptr<TensorContainer> st_;
    // 按权重名称索引的 metadata，持有 WeightMeta 生命周期。
    std::map<std::string, WeightMeta> metas;
    std::map<std::string, WeightData> weights;

    // 校验 Qwen3.5 推理路径需要的 tensor 是否齐全。
    void validate_qwen_tensors(int num_hidden_layers, const std::vector<std::string> &layer_types) const;

    // 从张量容器构建 name -> WeightMeta/WeightData 索引，data 指向已 mmap 内存。
    void build_weight_index();

    // 解析视觉塔（model.visual.*）权重到 this->vision；当前未使用。
    void parse_vision_weights(const QwenConfig &config);

    // 解析 MTP（mtp.*）权重到 this->mtp；当前未使用。
    void parse_mtp_weights(const QwenConfig &config);

    // 按名称返回权重数据引用；不存在时抛出异常。
    WeightData weight_data(const std::string &name) const;

    // 将 shape 转为日志/错误信息中使用的可读字符串。
    static std::string shape_to_string(const std::vector<int64_t> &shape);
};


#endif //LOCAL_LLM_QWENWEIGHTS_H
