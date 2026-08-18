//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_CUDASCRATCH_H
#define LOCAL_LLM_CUDASCRATCH_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// 前向过程中反复覆盖的临时激活暂存区（grow-only 复用），与具体模型无关。
// 内部按 key 维护一批「只增不减」的 device 缓冲：容量足够时复用旧内存，不够时重分配。
// 同一时刻整条前向只用到其中一部分 key，彼此不冲突；随 Session 存活以保证并发隔离。
//
// 底层统一以「字节」为单位存储（一个 map<string, 字节缓冲>），通过模板 ensure<T>
// 按元素个数申请并返回 T* 指针，从而在一套 map 里同时容纳 float / uint16_t / int 等
// 不同元素类型的缓冲。所有 key 预定义在下方 scratch_key 命名空间，qwen / deepseek 共用。
class CudaScratch {
public:
    CudaScratch() = default;
    ~CudaScratch();

    // 独占所有权，禁止拷贝。
    CudaScratch(const CudaScratch &) = delete;
    CudaScratch &operator=(const CudaScratch &) = delete;

    // 确保 key 对应的缓冲至少能容纳 count 个 T 元素，返回 device 指针（T*）。
    // 已够大时复用旧内存，不够时重分配；分配失败信息用 key 标识。
    // 仅对 float / uint16_t / int 做了显式实例化（见 .cpp）。
    template <typename T>
    T *ensure(const std::string &key, size_t count);

    // 所有 grow-only 缓冲当前容量（即峰值）字节数之和，供显存统计用。
    size_t total_bytes() const;

    // 释放全部 device 缓冲并清空。
    void reset();

    // === host 端采样暂存 ===
    // logits 从 device 拷回 host 的暂存（长度 vocab），供 Sampler 做温度 / top-k /
    // top-p / 重复惩罚。不计入 device 显存统计。
    std::vector<float> h_logits;

private:
    // 单个 key 对应的字节缓冲：只增不减。
    struct Buffer {
        void *ptr = nullptr;
        size_t bytes = 0;
    };

    std::unordered_map<std::string, Buffer> buffers_;
};

// 预定义的 scratch key：qwen / deepseek 共用同一套命名空间。
namespace scratch_key {

// ---- 通用 / embedding / lm_head ----
inline constexpr const char *kInput = "input";                 // embedding 输入 token id（int）
inline constexpr const char *kHidden = "hidden";               // 主隐状态 [tokens, hidden]
inline constexpr const char *kLogits = "logits";               // lm_head 输出 [vocab]
inline constexpr const char *kLogitsInLowp = "logits_in_lowp"; // lm_head gemm 输入低精度
inline constexpr const char *kInputLowp = "input_lowp";        // gemm 输入的低精度缓冲

// ---- qwen full attention ----
inline constexpr const char *kFullProjection = "full_projection"; // q+gate 合并投影
inline constexpr const char *kFullQ = "full_q";
inline constexpr const char *kFullGate = "full_gate";
inline constexpr const char *kFullK = "full_k";
inline constexpr const char *kFullV = "full_v";
inline constexpr const char *kFullAttn = "full_attn";
inline constexpr const char *kFullAttnLowp = "full_attn_lowp";

// ---- qwen linear attention ----
inline constexpr const char *kLinearProjection = "linear_projection";
inline constexpr const char *kLinearZ = "linear_z";
inline constexpr const char *kLinearB = "linear_b";
inline constexpr const char *kLinearA = "linear_a";
inline constexpr const char *kLinearConvOut = "linear_conv_out";
inline constexpr const char *kLinearGated = "linear_gated";
inline constexpr const char *kLinearGatedLowp = "linear_gated_lowp";

// ---- qwen mlp / norm / 残差 ----
inline constexpr const char *kGate = "gate";                  // mlp gate
inline constexpr const char *kUp = "up";                      // mlp up
inline constexpr const char *kProd = "prod";                  // SiLU(gate)*up
inline constexpr const char *kProdLowp = "prod_lowp";         // prod 低精度
inline constexpr const char *kMixer = "mixer";                // attention / mixer 子层输出
inline constexpr const char *kMlpOut = "mlp_out";             // mlp 子层输出
inline constexpr const char *kTokenHiddenA = "token_hidden_a"; // decode 层间双缓冲 a
inline constexpr const char *kTokenHiddenB = "token_hidden_b"; // decode 层间双缓冲 b

// ---- deepseek MLA ----
inline constexpr const char *kNormed = "normed";             // [tokens, hidden]
inline constexpr const char *kNormedLowp = "normed_lowp";
inline constexpr const char *kQ = "q";                       // [tokens, n_heads*qk_head]
inline constexpr const char *kKvA = "kv_a";                  // [tokens, kv_lora+qk_rope]
inline constexpr const char *kLatentLowp = "latent_lowp";
inline constexpr const char *kKvBOut = "kv_b_out";           // [tokens, n_heads*(qk_nope+v_head)]
inline constexpr const char *kAttn = "attn";                 // [tokens, n_heads*v_head]
inline constexpr const char *kAttnLowp = "attn_lowp";
inline constexpr const char *kAttnOut = "attn_out";          // [tokens, hidden]

// ---- deepseek FFN / MoE ----
inline constexpr const char *kFfnInLowp = "ffn_in_lowp";
inline constexpr const char *kAct = "act";                   // SiLU(gate)*up
inline constexpr const char *kActLowp = "act_lowp";
inline constexpr const char *kFfnOut = "ffn_out";            // [tokens, hidden]
inline constexpr const char *kMoeOut = "moe_out";            // 累加 [tokens, hidden]
inline constexpr const char *kRouterLogits = "router_logits"; // [tokens, n_experts]
inline constexpr const char *kTopIdx = "top_idx";            // [tokens, k]
inline constexpr const char *kTopW = "top_w";                // [tokens, k]
inline constexpr const char *kExpertOut = "expert_out";      // 单 token 单专家中间 [hidden]

} // namespace scratch_key

#endif // LOCAL_LLM_CUDASCRATCH_H
