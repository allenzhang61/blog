//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_TENSOR_H
#define LOCAL_LLM_TENSOR_H

#include <cstddef>
#include <cstdint>
#include <memory>
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
class CudaScratch;

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

TensorLocation operator|(TensorLocation a, TensorLocation b);

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
    // gpu mem 位置（激活/非池化）：直接持有的 device 内存裸指针。权重经 to_gpu() 走 pool
    // 惰性上传；而运行时激活（scratch buffer）不进 pool，直接用本字段
    // 包住 scratch 指针构成一个 device 视图（见 gpu_view）。本对象不拥有该内存。
    void *gpu_data = nullptr;
    // gpu mem 位置（权重反量化/可计算 view）：普通 F16/BF16/F32 权重指向原始 GPU 权重，
    // 量化权重指向 dequant pool 中的 F16 view。本对象通过 weight_view_lease 持有 view 生命周期。
    mutable void *gpu_data_dequant = nullptr;
    mutable DType dtype_dequant = DType::UNKNOWN;
    mutable size_t nbytes_dequant = 0;

    // 当前值驻留的位置集合（位掩码）。权重 to_gpu() 为 const 但会补标 GpuMem，故用 mutable。
    mutable uint32_t locations = static_cast<uint32_t>(TensorLocation::None);

    // 标记 / 查询某个位置是否驻留了本张量的值。
    void mark_location(TensorLocation loc) const;
    bool has_location(TensorLocation loc) const;

    // === 运行时激活视图工厂（不拥有内存，仅包住已有指针）===
    // 从 scratch 中按 shape/dtype 申请一段 device 激活内存，并构造成 gpu view。
    static Tensor gpu_scratch(CudaScratch &scratch, const std::string &key,
                         std::vector<int64_t> shape, DType dt = DType::F32);
    // 用一段已在 device 上的激活内存构造视图（不进 pool、不拥有）。
    static Tensor gpu_view(void *device_ptr, std::vector<int64_t> shape, DType dt = DType::F32);
    // 用一段 host 内存构造视图（如 token id 序列），不拥有。
    static Tensor host_view(const void *host_ptr, std::vector<int64_t> shape, DType dt);

    // === 便捷访问器 ===
    // device 激活指针（gpu_data，运行时激活路径）。
    float *gpu_f32() const;
    // device 侧 int 视图指针（gpu_data，如 token id / top-k expert id）。
    int *gpu_i32() const;
    // host 侧 int 视图指针（cpu_data，如 token id）。
    const int *host_i32() const;

    // === host/device 搬运 ===
    // 把当前 host/disk view 拷贝到一段 scratch GPU buffer，并让本 Tensor 同时持有 GpuMem view。
    void to_gpu(CudaScratch &scratch, const std::string &key, const std::string &what);
    // 把当前 host/disk view 拷贝到调用方提供的 GPU buffer。
    void to_gpu(void *device_ptr, const std::string &what) const;
    // 权重路径：经由 CudaWeightPool 惰性上传到 GPU。
    void to_gpu() const;
    // 把当前 GPU view 拷贝到调用方提供的 host buffer。
    void to_host(void *host_ptr, const std::string &what) const;
    // 按 dtype/shape 推导出的逻辑字节数（不用于量化权重物理大小）。
    size_t byte_size() const;

    // 当前 Tensor 作为权重，返回可直接参与计算的 device Tensor view。
    // 调用前应由 module 显式调用 to_gpu()。
    Tensor try_dequant() const;
    // 返回当前权重的 device 指针；Tensor 内部持有必要的 dequant/cache view 生命周期。
    const void *weight_gpu_data() const;

    // 元素总数（shape 各维乘积；空 shape 视为 0）。
    int64_t numel() const;
    // 二维视角下的行数 / 列数（约定最后一维为列，其余维乘积为行）。
    int64_t rows() const;
    int64_t cols() const;

    // 最近一次 try_dequant()/weight_gpu_data() 的权重 view 生命周期保持器。
    mutable std::shared_ptr<void> weight_view_lease;
};

#endif // LOCAL_LLM_TENSOR_H
