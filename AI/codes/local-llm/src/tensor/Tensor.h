//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_TENSOR_H
#define LOCAL_LLM_TENSOR_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// 统一的张量数据类型枚举，兼容 GGUF 量化类型与 safetensors 浮点类型。
// 数值特意与 ggml_type 保持一致，便于 GgmlType 与本枚举互转。
enum class DType : int32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I32 = 24,
    BF16 = 30,
    UNKNOWN = -1,
};

// DType -> 可读名（用于日志/错误信息）。
const char *dtype_name(DType dt);

// 当前推理链路实际支持的 dtype：浮点 F32/F16/BF16，以及已实现反量化 kernel 的
// 量化类型 Q4_K/Q5_0/Q6_K/Q8_0（见 Quant::dequantize_to_f16）。其余量化类型即便能被
// 识别，反量化时也会失败，因此在模型加载阶段就拒绝。
bool is_supported_dtype(DType dt);

// device 权重缓存池（CUDA backend），此处仅前向声明并持有裸指针，
// 不引入任何 CUDA 头，避免 tensor 层反向 include backend 头形成循环依赖
// （format -> tensor -> backend -> format）。指针指向进程内唯一的全局 pool
// （见 global_cuda_weight_pool），在权重解析阶段被填充。
class CudaWeightPool;
// device 端权重缓冲（CUDA backend），此处仅前向声明用于返回指针。
class CudaWeight;

// 一份张量数据可能同时驻留在多个存储位置上（位掩码，可按位组合）。
enum class TensorLocation : uint32_t {
    None = 0,
    // disk mmap：disk_data 指向模型文件 mmap 区域（不拥有内存）。
    DiskMmap = 1u << 0,
    // cpu mem：cpu_data 指向可写的 host 堆内存（当前仅预留字段，尚未实现搬运）。
    CpuMem = 1u << 1,
    // gpu mem：经 pool 惰性上传后驻留在 device（缓存条目见 cached_weight）。
    GpuMem = 1u << 2,
};

inline TensorLocation operator|(TensorLocation a, TensorLocation b) {
    return static_cast<TensorLocation>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

// 单个张量的统一视图，记录同一份值在 disk mmap / cpu mem / gpu mem 上的驻留情况，
// 同一份值可以同时存在于多个位置。本对象不拥有 disk mmap 内存（disk_data 指向
// 模型文件 mmap 区域）。成员保持 public：loader 直接写入这些字段，业务侧直接读取。
class Tensor {
public:
    std::string name;
    std::vector<int64_t> shape;
    DType dtype = DType::UNKNOWN;
    size_t nbytes = 0;

    // disk mmap 位置：指向模型文件 mmap 区域，不拥有内存。
    const uint8_t *disk_data = nullptr;
    // cpu mem 位置：可写的 host 堆内存。当前仅预留，尚未实现从 disk 搬运的逻辑。
    // 也用于承载 host 侧的输入视图（如 token id 序列，见 host_view）。
    void *cpu_data = nullptr;
    // gpu mem 位置（权重）：指向 device 权重缓存池，使得持有本 tensor 的 module 无需再单独传入 pool。
    CudaWeightPool *pool = nullptr;
    // gpu mem 位置（激活/非池化）：直接持有的 device 内存裸指针。权重经 pool 惰性上传
    // 并由 cached_weight() 返回；而运行时激活（scratch buffer）不进 pool，直接用本字段
    // 包住 scratch 指针构成一个 device 视图（见 gpu_activation）。本对象不拥有该内存。
    void *gpu_data = nullptr;

    // 当前值驻留的位置集合（位掩码）。cached_weight() 为 const 但会补标 GpuMem，故用 mutable。
    mutable uint32_t locations = static_cast<uint32_t>(TensorLocation::None);

    // 标记 / 查询某个位置是否驻留了本张量的值。
    void mark_location(TensorLocation loc) const { locations |= static_cast<uint32_t>(loc); }
    bool has_location(TensorLocation loc) const {
        return (locations & static_cast<uint32_t>(loc)) != 0;
    }

    // === 运行时激活视图工厂（不拥有内存，仅包住已有指针）===
    // 用一段已在 device 上的激活内存构造视图（不进 pool、不拥有）。
    static Tensor gpu_activation(void *device_ptr, std::vector<int64_t> shape, DType dt = DType::F32) {
        Tensor t;
        t.shape = std::move(shape);
        t.dtype = dt;
        t.gpu_data = device_ptr;
        t.mark_location(TensorLocation::GpuMem);
        return t;
    }
    // 用一段 host 内存构造视图（如 token id 序列），不拥有。
    static Tensor host_view(const void *host_ptr, std::vector<int64_t> shape, DType dt) {
        Tensor t;
        t.shape = std::move(shape);
        t.dtype = dt;
        t.cpu_data = const_cast<void *>(host_ptr);
        t.mark_location(TensorLocation::CpuMem);
        return t;
    }

    // === 便捷访问器 ===
    // device 激活指针（gpu_data，运行时激活路径）。
    float *gpu_f32() const { return static_cast<float *>(gpu_data); }
    // host 侧 int 视图指针（cpu_data，如 token id）。
    const int *host_i32() const { return static_cast<const int *>(cpu_data); }

    // 元素总数（shape 各维乘积；空 shape 视为 0）。
    int64_t numel() const {
        if (shape.empty()) { return 0; }
        int64_t n = 1;
        for (int64_t d : shape) { n *= d; }
        return n;
    }
    // 二维视角下的行数 / 列数（约定最后一维为列，其余维乘积为行）。
    int64_t rows() const {
        if (shape.empty()) { return 0; }
        int64_t r = 1;
        for (size_t i = 0; i + 1 < shape.size(); ++i) { r *= shape[i]; }
        return r;
    }
    int64_t cols() const { return shape.empty() ? 0 : shape.back(); }

    // 惰性把本 tensor 上传到 device 并返回缓存条目（等价于 pool->cached_weight(*this)），
    // 同时把 gpu mem 标记进 locations。实现见 CudaWeightPool.cpp，避免 tensor 层依赖
    // CUDA 头。pool 为空时行为未定义（调用方须保证已填充 pool）。
    CudaWeight *cached_weight() const;
};

#endif // LOCAL_LLM_TENSOR_H
