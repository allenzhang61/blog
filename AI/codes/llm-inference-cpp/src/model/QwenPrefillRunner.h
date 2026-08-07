#pragma once

#include "Tensor.h"
#include "runtime_state.h"
#include "weights.h"
#include "../core/config.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llm_inference {

class CudaWeightCache;
class DeviceWeight;

// CUDA batch prefill 执行器：一次性处理完整 prompt，并建立 attention cache。
class QwenPrefillRunner {
public:
    QwenPrefillRunner(const ModelConfig & config, const ModelParams & params);

    // 在 device 上处理完整 prompt，返回最后一个 token 的 device hidden。
    Tensor forward(const std::vector<int> & input_ids, RunState & state) const;

private:
    // 获取 prefill 需要的 CUDA device 权重，缓存失败时直接抛错。
    static DeviceWeight & require_device_weight(
        CudaWeightCache & cache,
        const WeightData & weight,
        const std::string & context);

    // 执行 prefill batch MLP：gate/up projection -> SiLU(gate) * up -> down projection。
    static bool forward_mlp(
        const WeightData & gate_w,
        const WeightData & up_w,
        const WeightData & down_w,
        const uint16_t * device_x,
        int tokens,
        float * device_out);

    const ModelConfig & config_;
    const ModelParams & params_;
};

} // namespace llm_inference
