//
// Created by zhangyoulun on 9/8/2026.
//

#include "GgufFile.h"

#include "backend/cuda/mem/CudaWeightPool.h"
#include "format/gguf/GGUFTokenizer.h"
#include "utils/log/Log.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {
constexpr uint32_t kGgufMagic = 0x46554747; // "GGUF" little-endian

// 每种 ggml 量化类型的 block：block 内元素数与字节数。
// 参考 ggml.h 的 type_traits。这里只列推理可能用到的类型。
struct BlockTrait {
    int64_t block_size;  // 一个 block 覆盖的元素数
    size_t type_size;    // 一个 block 占用的字节数
};

BlockTrait block_trait(GgmlType type) {
    switch (type) {
        case GgmlType::F32:  return {1, 4};
        case GgmlType::F16:  return {1, 2};
        case GgmlType::BF16: return {1, 2};
        case GgmlType::Q4_0: return {32, 18};   // d(2) + qs(16)
        case GgmlType::Q4_1: return {32, 20};   // d(2) + m(2) + qs(16)
        case GgmlType::Q5_0: return {32, 22};
        case GgmlType::Q5_1: return {32, 24};
        case GgmlType::Q8_0: return {32, 34};   // d(2) + qs(32)
        case GgmlType::Q8_1: return {32, 36};
        case GgmlType::Q2_K: return {256, 84};
        case GgmlType::Q3_K: return {256, 110};
        case GgmlType::Q4_K: return {256, 144};  // d(2)+dmin(2)+scales(12)+qs(128)
        case GgmlType::Q5_K: return {256, 176};
        case GgmlType::Q6_K: return {256, 210};
        case GgmlType::Q8_K: return {256, 292};
        default:
            throw std::runtime_error("未知 ggml 量化类型: " + std::to_string(static_cast<int>(type)));
    }
}

// GgufValueType -> 可读名。
const char *value_type_name(GgufValueType t) {
    switch (t) {
    case GgufValueType::UINT8: return "u8";
    case GgufValueType::INT8: return "i8";
    case GgufValueType::UINT16: return "u16";
    case GgufValueType::INT16: return "i16";
    case GgufValueType::UINT32: return "u32";
    case GgufValueType::INT32: return "i32";
    case GgufValueType::FLOAT32: return "f32";
    case GgufValueType::BOOL: return "bool";
    case GgufValueType::STRING: return "str";
    case GgufValueType::ARRAY: return "array";
    case GgufValueType::UINT64: return "u64";
    case GgufValueType::INT64: return "i64";
    case GgufValueType::FLOAT64: return "f64";
    default: return "?";
    }
}

// dims 向量 -> "[a, b, c]"。
std::string dims_to_string(const std::vector<int64_t> &dims) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) os << ", ";
        os << dims[i];
    }
    os << "]";
    return os.str();
}

int64_t num_elements(const std::vector<int64_t> &dims) {
    int64_t n = 1;
    for (int64_t d : dims) {
        n *= d;
    }
    return n;
}

// 单条标量/数组元数据值 -> 可读字符串（数组做预览截断，避免词表刷屏）。
std::string value_to_string(const GgufValue &v) {
    constexpr size_t kPreview = 8; // 数组最多预览的元素数
    std::ostringstream os;
    switch (v.type) {
    case GgufValueType::STRING: {
        std::string s = v.str;
        if (s.size() > 120) s = s.substr(0, 120) + "...";
        os << s;
        break;
    }
    case GgufValueType::BOOL:
        os << (v.boolean ? "true" : "false");
        break;
    case GgufValueType::FLOAT32:
    case GgufValueType::FLOAT64:
        os << v.f64;
        break;
    case GgufValueType::ARRAY: {
        os << "array<" << value_type_name(v.elem_type) << ">[";
        if (!v.arr_str.empty()) {
            os << v.arr_str.size() << "] = {";
            for (size_t i = 0; i < v.arr_str.size() && i < kPreview; ++i) {
                if (i) os << ", ";
                std::string s = v.arr_str[i];
                if (s.size() > 24) s = s.substr(0, 24) + "..";
                os << '"' << s << '"';
            }
            if (v.arr_str.size() > kPreview) os << ", ...";
        } else if (!v.arr_f64.empty()) {
            os << v.arr_f64.size() << "] = {";
            for (size_t i = 0; i < v.arr_f64.size() && i < kPreview; ++i) {
                if (i) os << ", ";
                os << v.arr_f64[i];
            }
            if (v.arr_f64.size() > kPreview) os << ", ...";
        } else {
            os << v.arr_i64.size() << "] = {";
            for (size_t i = 0; i < v.arr_i64.size() && i < kPreview; ++i) {
                if (i) os << ", ";
                os << v.arr_i64[i];
            }
            if (v.arr_i64.size() > kPreview) os << ", ...";
        }
        os << "}";
        break;
    }
    default: // 各种整型
        os << v.i64;
        break;
    }
    return os.str();
}
} // namespace

GgufFile::GgufFile(const std::string &path) : path_(path) {
    fd_ = ::open(path_.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("open 失败：" + path_.string() + "，" + std::strerror(errno));
    }
    struct stat st {};
    if (fstat(fd_, &st) != 0) {
        throw std::runtime_error("fstat 失败：" + path_.string() + "，" + std::strerror(errno));
    }
    size_ = static_cast<size_t>(st.st_size);
    if (size_ < 24) {
        throw std::runtime_error("GGUF 文件太小：" + path_.string());
    }
    void *ptr = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("mmap 失败：" + path_.string() + "，" + std::strerror(errno));
    }
    data_ = static_cast<const uint8_t *>(ptr);

    Cursor c(data_, size_);

    const uint32_t magic = c.read_u32();
    if (magic != kGgufMagic) {
        throw std::runtime_error("非 GGUF 文件（magic 不匹配）：" + path_.string());
    }
    version_ = c.read_u32();
    if (version_ != 2 && version_ != 3) {
        throw std::runtime_error("不支持的 GGUF 版本: " + std::to_string(version_));
    }

    const uint64_t tensor_count = c.read_u64();
    const uint64_t metadata_kv_count = c.read_u64();

    // 解析元数据 KV。
    for (uint64_t i = 0; i < metadata_kv_count; ++i) {
        const std::string key = c.read_string();
        const auto value_type = static_cast<GgufValueType>(c.read_u32());
        metadata_map_[key] = c.read_value(value_type);
    }

    // 对齐：general.alignment（默认 32）。
    if (metadata_map_.contains("general.alignment")) {
        alignment_ = static_cast<uint32_t>(metadata<int64_t>("general.alignment"));
        if (alignment_ == 0) {
            alignment_ = 32;
        }
    }

    struct ParsedTensor {
        DiskTensor view;
        uint64_t offset = 0;
    };

    // 解析张量信息表（name / n_dims / dims / type / offset）。
    std::vector<ParsedTensor> parsed_tensors;
    parsed_tensors.reserve(tensor_count);
    for (uint64_t i = 0; i < tensor_count; ++i) {
        ParsedTensor tensor;
        tensor.view.name = c.read_string();
        const uint32_t n_dims = c.read_u32();
        std::vector<int64_t> dims(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            dims[d] = static_cast<int64_t>(c.read_u64());
        }
        const GgmlType type = static_cast<GgmlType>(c.read_u32());
        tensor.offset = c.read_u64();
        tensor.view.shape.assign(dims.rbegin(), dims.rend());
        tensor.view.dtype = gguf_type_to_dtype(type);
        tensor.view.nbytes = type_nbytes(type, num_elements(dims));
        parsed_tensors.push_back(std::move(tensor));
    }

    // 张量数据段起始 = 张量表结束位置，向上对齐到 alignment_。
    size_t data_start = c.pos();
    if (data_start % alignment_ != 0) {
        data_start += alignment_ - (data_start % alignment_);
    }
    if (data_start > size_) {
        throw std::runtime_error("GGUF 数据段起始越界：" + path_.string());
    }
    const uint8_t *data_base = data_ + data_start;

    // 回填每个张量的数据指针。
    tensors_.reserve(parsed_tensors.size());
    for (ParsedTensor &tensor : parsed_tensors) {
        const uint8_t *tensor_data = data_base + tensor.offset;
        if (tensor_data + tensor.view.nbytes > data_ + size_) {
            throw std::runtime_error("张量数据越界: " + tensor.view.name);
        }
        tensor.view.data = tensor_data;
        tensors_.push_back(std::move(tensor.view));
    }

    tokenizer_ = std::make_unique<GGUFTokenizer>(*this);
}

GgufFile::~GgufFile() {
    close();
}

void GgufFile::close() {
    if (data_ != nullptr && size_ > 0) {
        munmap(const_cast<uint8_t *>(data_), size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
    data_ = nullptr;
    fd_ = -1;
    size_ = 0;
}

Cursor::Cursor(const uint8_t *base, size_t size) : base_(base), size_(size) {
}

uint8_t Cursor::read_u8() {
    if (pos_ + 1 > size_) {
        throw std::runtime_error("GGUF 解析越界（u8）");
    }
    return base_[pos_++];
}

uint32_t Cursor::read_u32() {
    if (pos_ + 4 > size_) {
        throw std::runtime_error("GGUF 解析越界（u32）");
    }
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
        v |= static_cast<uint32_t>(base_[pos_ + i]) << (8 * i);
    }
    pos_ += 4;
    return v;
}

uint64_t Cursor::read_u64() {
    if (pos_ + 8 > size_) {
        throw std::runtime_error("GGUF 解析越界（u64）");
    }
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(base_[pos_ + i]) << (8 * i);
    }
    pos_ += 8;
    return v;
}

double Cursor::read_f64_bits(GgufValueType t) {
    if (t == GgufValueType::FLOAT32) {
        const uint32_t bits = read_u32();
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return static_cast<double>(f);
    }
    // FLOAT64
    const uint64_t bits = read_u64();
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
}

std::string Cursor::read_string() {
    const uint64_t len = read_u64();
    if (pos_ + len > size_) {
        throw std::runtime_error("GGUF 解析越界（string）");
    }
    std::string s(reinterpret_cast<const char *>(base_ + pos_), len);
    pos_ += len;
    return s;
}

GgufValue Cursor::read_value(GgufValueType type) {
    GgufValue v;
    v.type = type;
    switch (type) {
        case GgufValueType::UINT8:  v.i64 = read_u8(); break;
        case GgufValueType::INT8:   v.i64 = static_cast<int8_t>(read_u8()); break;
        case GgufValueType::UINT16: { uint32_t lo = read_u8(), hi = read_u8(); v.i64 = lo | (hi << 8); break; }
        case GgufValueType::INT16:  { uint32_t lo = read_u8(), hi = read_u8(); v.i64 = static_cast<int16_t>(lo | (hi << 8)); break; }
        case GgufValueType::UINT32: v.i64 = read_u32(); break;
        case GgufValueType::INT32:  v.i64 = static_cast<int32_t>(read_u32()); break;
        case GgufValueType::UINT64: v.i64 = static_cast<int64_t>(read_u64()); break;
        case GgufValueType::INT64:  v.i64 = static_cast<int64_t>(read_u64()); break;
        case GgufValueType::FLOAT32: v.f64 = read_f64_bits(type); break;
        case GgufValueType::FLOAT64: v.f64 = read_f64_bits(type); break;
        case GgufValueType::BOOL:   v.boolean = (read_u8() != 0); break;
        case GgufValueType::STRING: v.str = read_string(); break;
        case GgufValueType::ARRAY: {
            v.elem_type = static_cast<GgufValueType>(read_u32());
            const uint64_t n = read_u64();
            for (uint64_t i = 0; i < n; ++i) {
                GgufValue e = read_value(v.elem_type);
                switch (v.elem_type) {
                    case GgufValueType::STRING:  v.arr_str.push_back(std::move(e.str)); break;
                    case GgufValueType::FLOAT32:
                    case GgufValueType::FLOAT64: v.arr_f64.push_back(e.f64); break;
                    case GgufValueType::BOOL:    v.arr_i64.push_back(e.boolean ? 1 : 0); break;
                    default:                     v.arr_i64.push_back(e.i64); break;
                }
            }
            break;
        }
        default:
            throw std::runtime_error("未知 GGUF 值类型: " + std::to_string(static_cast<uint32_t>(type)));
    }
    return v;
}

size_t GgufFile::type_nbytes(GgmlType type, int64_t num_elements) {
    const BlockTrait bt = block_trait(type);
    if (num_elements % bt.block_size != 0) {
        throw std::runtime_error("元素数不是 block_size 的整数倍");
    }
    return static_cast<size_t>(num_elements / bt.block_size) * bt.type_size;
}

bool GgufFile::contain_metadata(const std::string &key) const {
    return metadata_map_.contains(key);
}

Metadata GgufFile::metadata_value(const std::string &key) const {
    auto it = metadata_map_.find(key);
    if (it == metadata_map_.end()) {
        throw std::runtime_error("GGUF 缺少元数据 key: " + key);
    }
    const GgufValue &value = it->second;
    switch (value.type) {
        case GgufValueType::FLOAT32:
        case GgufValueType::FLOAT64:
            return static_cast<float>(value.f64);
        case GgufValueType::BOOL:
            return value.boolean;
        case GgufValueType::STRING:
            return value.str;
        case GgufValueType::ARRAY:
            if (value.elem_type == GgufValueType::STRING) {
                return value.arr_str;
            }
            return value.arr_i64;
        default:
            return value.i64;
    }
}

bool GgufFile::contain_tensor_view(const std::string &name) const {
    for (const DiskTensor &tensor : tensors_) {
        if (tensor.name == name) {
            return true;
        }
    }
    return false;
}

DType GgufFile::gguf_type_to_dtype(GgmlType t) {
    switch (t) {
    case GgmlType::F32: return DType::F32;
    case GgmlType::F16: return DType::F16;
    case GgmlType::Q4_0: return DType::Q4_0;
    case GgmlType::Q4_1: return DType::Q4_1;
    case GgmlType::Q5_0: return DType::Q5_0;
    case GgmlType::Q5_1: return DType::Q5_1;
    case GgmlType::Q8_0: return DType::Q8_0;
    case GgmlType::Q8_1: return DType::Q8_1;
    case GgmlType::Q2_K: return DType::Q2_K;
    case GgmlType::Q3_K: return DType::Q3_K;
    case GgmlType::Q4_K: return DType::Q4_K;
    case GgmlType::Q5_K: return DType::Q5_K;
    case GgmlType::Q6_K: return DType::Q6_K;
    case GgmlType::Q8_K: return DType::Q8_K;
    case GgmlType::BF16: return DType::BF16;
    default: return DType::UNKNOWN;
    }
}

const DiskTensor &GgufFile::get_tensor_view(const std::string &name) const {
    for (const DiskTensor &tensor : tensors_) {
        if (tensor.name == name) {
            // 权重解析发生在全局 pool 就绪之后，这里补写 pool 链接，
            // 使持有该 view 的 module 无需再单独传 pool。
            const_cast<DiskTensor &>(tensor).pool = &global_cuda_weight_pool();
            return tensor;
        }
    }
    throw std::runtime_error("GGUF 缺少张量: " + name);
}

std::vector<std::string> GgufFile::tensor_view_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const DiskTensor &tensor : tensors_) {
        names.push_back(tensor.name);
    }
    return names;
}

std::vector<int> GgufFile::tokenizer_encode(const std::string &text) const {
    if (!tokenizer_) {
        throw std::runtime_error("GGUF tokenizer 尚未初始化");
    }
    return tokenizer_->encode(text);
}

std::string GgufFile::tokenizer_decode(const std::vector<int> &ids) const {
    if (!tokenizer_) {
        throw std::runtime_error("GGUF tokenizer 尚未初始化");
    }
    return tokenizer_->decode(ids);
}

void GgufFile::debug_dump() const {
    Log::debug("GGUF: " + path_.string());
    Log::debug("  version: " + std::to_string(version_));
    Log::debug("  alignment: " + std::to_string(alignment_));
    Log::debug("  metadata_kv: " + std::to_string(metadata_map_.size()));
    Log::debug("  tensors: " + std::to_string(tensors_.size()));

    // 全部元数据 KV（按 key 字典序，std::map 已有序）。
    Log::debug("  === metadata (" + std::to_string(metadata_map_.size()) + ") ===");
    for (const auto &[key, val] : metadata_map_) {
        Log::debug("    " + key + " : " + value_type_name(val.type) + " = " + value_to_string(val));
    }

    // 张量类型直方图（先统计，放在张量列表之上做概览）。
    std::map<DType, size_t> type_hist;
    for (const DiskTensor &tensor : tensors_) {
        type_hist[tensor.dtype]++;
    }
    Log::debug("  === tensor type histogram ===");
    for (const auto &[type, count] : type_hist) {
        Log::debug(std::string("    ") + dtype_name(type) + " : " + std::to_string(count));
    }

    // 全部张量元信息（不含权重数据段），按文件出现顺序。
    Log::debug("  === tensors (" + std::to_string(tensors_.size()) + ") ===");
    for (const DiskTensor &tensor : tensors_) {
        std::ostringstream os;
        os << "    " << tensor.name << "  shape=" << dims_to_string(tensor.shape)
           << " dtype=" << dtype_name(tensor.dtype)
           << " nbytes=" << tensor.nbytes;
        Log::debug(os.str());
    }
}
