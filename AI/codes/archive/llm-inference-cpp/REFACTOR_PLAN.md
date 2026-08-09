# llm-inference-cpp 分层重构方案（借鉴 llm-train-cpp）

> 状态：**待确认**。本文件是重构前的方案说明，确认后再动代码。
> 目标：借鉴 `llm-train-cpp/src` 的分层架构，重构 `llm-inference-cpp/src`。
> 验证基准：**macOS 本地编译通过**（纯 CPU 路径，CUDA/CBLAS 通过 `find_package` 条件编译，本地自动跳过）。

---

## 1. 借鉴什么、不借鉴什么

`llm-train-cpp` 的可复用设计模式（来自架构分析）：

| 模式 | 是否借鉴 | 说明 |
|------|---------|------|
| 分层解耦 + 单向依赖（model→ops→kernels→tensor→device） | ✅ 借鉴 | 核心目标 |
| `ops::` → `cpu::/cuda::` 模板函数分发 | ✅ 借鉴 | 取代散落的 `#ifdef` + try-fallback |
| 命名空间并列的算子约定（`llm_inference::cpu` / `::cuda`） | ✅ 借鉴 | 同名同签名，后端可替换 |
| Stub + CMake 编译期后端注册 | ✅ 借鉴 | CUDA 未启用时 stub 返回 false |
| 句柄/节点/存储三层 Tensor | ⚠️ 简化 | 推理只需单层轻量 Tensor（见 §3） |
| **闭包式 autograd + backward 断图** | ❌ **不借鉴** | 推理无反向，照搬是纯负担 |
| Module 基类 + 参数指针汇总 | ✅ 借鉴（改造） | 推理版是「Layer 持有已解析权重引用」 |
| 头文件组织（include/ 分模块 + 伞头文件） | ✅ 借鉴 | 见 §6 |

**关键取舍**：推理项目当前**已跑通、已和 Python 对齐**（[8160, 579] "Here's"）。CUDA 的 40+ 融合函数是性能命脉。重构**只改组织结构，不改数值逻辑与 CUDA 融合行为**——把它们从散落的自由函数搬进 `cuda::` 命名空间并由 `ops::` 分发，行为逐字保留。

---

## 2. 目标目录结构

对齐 train 项目的 `src/{tensor,ops,kernels,backend,model,...}` 分层：

```
src/
├── core/                      # 保留：基础设施（不属于计算分层）
│   ├── common.{h,cpp}         # 不动
│   ├── cli.{h,cpp}            # 不动
│   ├── config.{h,cpp}         # 不动
│   ├── safetensors.{h,cpp}    # 不动（权重加载）
│   ├── tokenizer.{h,cpp}      # 不动
│   └── profile.{h,cpp}        # 不动
├── tensor/
│   └── tensor.{h,cpp}         # 新增：统一 Tensor 抽象（见 §3）
├── ops/
│   └── ops.{h,cpp}            # 新增：ops:: 分发层（见 §4）
├── kernels/
│   ├── cpu/
│   │   ├── cpu_ops.h          # namespace cpu 算子声明
│   │   ├── elementwise.cpp    # add_inplace / l2_norm / 激活(silu/sigmoid/softplus)
│   │   ├── matvec.cpp         # matvec / dot_row / dtype 转换 / argmax
│   │   ├── rmsnorm.cpp        # rms_norm / gated_rms_norm_head
│   │   ├── embedding.cpp      # embedding_lookup
│   │   ├── attention.cpp      # full_attention（含 rope、KV cache）
│   │   ├── linear_attention.cpp # linear attention（conv + recurrent）
│   │   └── mlp.cpp            # mlp（gate/up/down + silu）
│   └── cuda/
│       ├── cuda_ops.h         # namespace cuda 算子声明 + available()
│       ├── cuda_ops.cu        # 现 tensor_ops.cpp 里的 cuda_*_layer 封装迁移至此
│       ├── cuda_kernels.{h,cu}# 保留：最底层 launch_* kernel（基本不动）
│       └── cuda_stub.cpp      # 新增：CUDA 未启用时的 stub（全 return false）
├── backend/
│   └── backend.{h,cpp}        # 新增：available/status/设备选择
├── model/
│   ├── layer.{h,cpp}          # 新增：Layer 基类 + 已解析权重持有
│   ├── QwenModel.{h,cpp}      # 由 NativeQwen 重构而来（编排各 Layer）
│   └── weights.{h,cpp}        # 新增：ModelWeights → 结构化 LayerWeights 解析
└── main.{cpp,h}               # 编排：保持 main() 流程，改调用新分层 API
```

> 命名沿用现有风格：类型大驼峰（`Tensor`、`QwenModel`），函数/变量 snake_case，`namespace llm_inference`，中文注释，`#pragma once`。

---

## 3. Tensor 抽象（简化单层，无 autograd）

推理只需表达两类数据：**只读 mmap 权重（BF16/F16/F32）** 和 **可写激活（F32）**。设计成一个轻量类：

```cpp
enum class DType { F32, BF16, F16 };
enum class Device { CPU, CUDA };

class Tensor {
public:
    std::vector<int64_t> shape;
    DType dtype = DType::F32;
    Device device = Device::CPU;

    // 二选一：要么拥有数据(owned_)，要么引用外部只读数据(ext_ptr_)
    // 权重 → 引用 mmap（不拷贝）；激活 → 拥有 std::vector<float>
    float*       data();        // 可写（仅 owned F32）
    const void*  raw() const;   // 原始字节（权重按 dtype 解释）
    float        at(size_t i) const; // 按 dtype 转 float 读取
    int64_t      numel() const;
    // 工厂：from_weight(TensorRef)、zeros(shape)、from_vector(...)
};
```

- **不引入 TensorNode/TensorStorage/backward_fn**。CUDA device 指针用现有的 `void*` 缓存机制（`CudaWeightCache`）继续管理，不塞进 Tensor（保持改动最小、行为不变）。
- 现有 `TensorRef`（`safetensors.h`）**保留**，作为「从 mmap 构造 Tensor」的桥；`Tensor::from_weight(TensorRef)` 封装 dtype 解释。
- 激活从裸 `std::vector<float>` 逐步替换为 `Tensor`（owned F32）。

---

## 4. ops 分发层

对齐 train 的 `dispatch_*` 模板。推理只有 CPU/CUDA 两后端，且 CUDA 是「try 成功即用、否则回退 CPU」语义（不同于 train 的严格设备匹配）。所以分发模板改为：

```cpp
// ops.cpp（示意）
Tensor mlp(const LayerWeights& w, const Tensor& x) {
    Tensor out;
    if (cuda::available() && cuda::mlp(w, x, out)) return out;  // try CUDA
    cpu::mlp(w, x, out);                                        // fallback CPU
    return out;
}
```

- `ops::` 集中「try-CUDA-then-CPU」逻辑，**取代散落在 NativeQwen 里针对每种融合粒度手写的多份 fallback**。
- CUDA 的多级融合（整层 → project → 裸算子）在 `cuda::` 内部消化，对 `ops::` 只暴露一个入口（如 `cuda::linear_attention_layer` 内部自行选最优融合粒度）。
- `cuda::available()` / 各 `*_enabled()` 保留（宏 + getenv 开关不变）。

---

## 5. Model 层：Layer 持有已解析权重

解决现状「forward 每层 `t(prefix+"...")` 现查字符串」的耦合与开销：

```cpp
struct LinearAttnWeights { TensorRef in_proj_qkv, in_proj_z, conv, a_log, ...; };
struct FullAttnWeights   { TensorRef q_proj, k_proj, v_proj, q_norm, ...; };
struct MlpWeights        { TensorRef gate, up, down; };
struct LayerWeights {
    std::string type;            // "linear_attention" / "full_attention"
    TensorRef input_norm, post_norm;
    LinearAttnWeights lin; FullAttnWeights full; MlpWeights mlp;
};
// weights.cpp: 加载时一次性把所有 TensorRef 解析进 vector<LayerWeights>
```

`QwenModel`（由 `NativeQwen` 重构）持有 `vector<LayerWeights>`，forward 时直接用已解析引用，不再拼字符串查 map。生成循环 `generate_next/decode_one/generate_sequence_device` 的编排保留。

---

## 6. 头文件组织

轻量沿用 train 的约定，但**不新建独立 include/ 根目录**（推理项目现用相对路径 `#include "../core/xxx.h"`，保持一致以减小改动）。各层头文件就近放在各自 `src/<层>/` 下，跨层用相对路径引用。CUDA 私有头（`cuda_ops.h`、`cuda_kernels.h`）不被 model 层直接引用，只经 `ops::`。

---

## 7. 分阶段实施计划（每阶段结束都保证本地编译通过）

| 阶段 | 内容 | 编译验证 |
|------|------|---------|
| **P0** | 建立本地构建基线：当前代码在 macOS 编译通过 + 跑一次小输入 | ✅ 必须先通过 |
| **P1** | 新增 `tensor/tensor.{h,cpp}`（Tensor + DType/Device + from_weight），不改调用方 | ✅ 编译通过 |
| **P2** | 拆分 `tensor_ops.cpp` 的 CPU 算子 → `kernels/cpu/*.cpp`（`namespace cpu`），签名不变，NativeQwen 改调 `cpu::` | ✅ 编译 + 小输入 |
| **P3** | 迁移 CUDA `cuda_*_layer` → `kernels/cuda/cuda_ops.cu`（`namespace cuda`）+ `cuda_stub.cpp`；行为逐字保留 | ✅ 编译（CUDA 跳过） |
| **P4** | 新增 `ops/ops.{h,cpp}` 分发层，把 NativeQwen 里的 try-fallback 收敛到 ops:: | ✅ 编译 + 小输入 |
| **P5** | 新增 `model/weights.*` + `layer.*`，`NativeQwen`→`QwenModel` 持有已解析权重 | ✅ 编译 + 小输入 |
| **P6** | 更新 CMakeLists 源文件列表；清理旧 `tensor_ops.*`、`NativeQwen.*` | ✅ 最终编译通过 |

> 每阶段独立可编译、可回退。P3 是最高风险（CUDA 迁移），因本地无法编译 CUDA，只能保证 CPU 路径编译通过 + 代码逐字搬迁不改逻辑，CUDA 数值验证需后续在 devbox_sg 上单独做。

---

## 8. 风险与注意

1. **CUDA 无法本地验证**：本地只能保证 CPU 路径编译 + 数值不变；CUDA 融合路径靠「逐字搬迁不改逻辑」保正确，需后续在 devbox_sg 用 Qwen3.5-4B 回归 [8160,579]。
2. **改动量大**：`tensor_ops.cpp` ~3600 行 + `NativeQwen.cpp`。分 6 阶段降低单次风险。
3. **性能不回退**：CUDA 融合粒度选择逻辑必须完整保留，不能因分层而丢失任何一级融合。
4. **不引入 autograd**：明确不照搬 train 的 TensorNode/backward，避免无谓复杂度和内存开销。
```