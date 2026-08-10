//
// Created by zhangyoulun on 9/8/2026.
//

#include "GgufFile.h"

#include "utils/log/Log.h"

#include <cerrno>
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

// GgmlType -> 可读名。
const char *ggml_type_name(GgmlType t) {
    switch (t) {
    case GgmlType::F32: return "F32";
    case GgmlType::F16: return "F16";
    case GgmlType::Q4_0: return "Q4_0";
    case GgmlType::Q4_1: return "Q4_1";
    case GgmlType::Q5_0: return "Q5_0";
    case GgmlType::Q5_1: return "Q5_1";
    case GgmlType::Q8_0: return "Q8_0";
    case GgmlType::Q8_1: return "Q8_1";
    case GgmlType::Q2_K: return "Q2_K";
    case GgmlType::Q3_K: return "Q3_K";
    case GgmlType::Q4_K: return "Q4_K";
    case GgmlType::Q5_K: return "Q5_K";
    case GgmlType::Q6_K: return "Q6_K";
    case GgmlType::Q8_K: return "Q8_K";
    case GgmlType::BF16: return "BF16";
    default: return "?";
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

int64_t GgufTensorInfo::num_elements() const {
    int64_t n = 1;
    for (int64_t d : dims) {
        n *= d;
    }
    return n;
}

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

    Cursor c{data_, size_, 0};

    const uint32_t magic = read_u32(c);
    if (magic != kGgufMagic) {
        throw std::runtime_error("非 GGUF 文件（magic 不匹配）：" + path_.string());
    }
    version_ = read_u32(c);
    if (version_ != 2 && version_ != 3) {
        throw std::runtime_error("不支持的 GGUF 版本: " + std::to_string(version_));
    }

    const uint64_t tensor_count = read_u64(c);
    const uint64_t metadata_kv_count = read_u64(c);

    // 解析元数据 KV。
    for (uint64_t i = 0; i < metadata_kv_count; ++i) {
        const std::string key = read_string(c);
        const auto value_type = static_cast<GgufValueType>(read_u32(c));
        metadata_[key] = read_value(c, value_type);
    }

    // 对齐：general.alignment（默认 32）。
    if (metadata_.count("general.alignment")) {
        alignment_ = static_cast<uint32_t>(metadata_i64("general.alignment"));
        if (alignment_ == 0) {
            alignment_ = 32;
        }
    }

    // 解析张量信息表（name / n_dims / dims / type / offset）。
    tensors_.reserve(tensor_count);
    for (uint64_t i = 0; i < tensor_count; ++i) {
        GgufTensorInfo info;
        info.name = read_string(c);
        const uint32_t n_dims = read_u32(c);
        info.dims.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            info.dims[d] = static_cast<int64_t>(read_u64(c));
        }
        info.type = static_cast<GgmlType>(read_u32(c));
        info.offset = read_u64(c);
        tensors_.push_back(std::move(info));
    }

    // 张量数据段起始 = 张量表结束位置，向上对齐到 alignment_。
    size_t data_start = c.pos;
    if (data_start % alignment_ != 0) {
        data_start += alignment_ - (data_start % alignment_);
    }
    if (data_start > size_) {
        throw std::runtime_error("GGUF 数据段起始越界：" + path_.string());
    }
    const uint8_t *data_base = data_ + data_start;

    // 回填每个张量的数据指针与字节数，并建立名称索引。
    for (size_t i = 0; i < tensors_.size(); ++i) {
        GgufTensorInfo &t = tensors_[i];
        t.nbytes = type_nbytes(t.type, t.num_elements());
        t.data = data_base + t.offset;
        if (t.data + t.nbytes > data_ + size_) {
            throw std::runtime_error("张量数据越界: " + t.name);
        }
        tensor_index_[t.name] = i;
    }
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

uint8_t GgufFile::read_u8(Cursor &c) {
    if (c.pos + 1 > c.size) {
        throw std::runtime_error("GGUF 解析越界（u8）");
    }
    return c.base[c.pos++];
}

uint32_t GgufFile::read_u32(Cursor &c) {
    if (c.pos + 4 > c.size) {
        throw std::runtime_error("GGUF 解析越界（u32）");
    }
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
        v |= static_cast<uint32_t>(c.base[c.pos + i]) << (8 * i);
    }
    c.pos += 4;
    return v;
}

uint64_t GgufFile::read_u64(Cursor &c) {
    if (c.pos + 8 > c.size) {
        throw std::runtime_error("GGUF 解析越界（u64）");
    }
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(c.base[c.pos + i]) << (8 * i);
    }
    c.pos += 8;
    return v;
}

double GgufFile::read_f64_bits(Cursor &c, GgufValueType t) {
    if (t == GgufValueType::FLOAT32) {
        const uint32_t bits = read_u32(c);
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return static_cast<double>(f);
    }
    // FLOAT64
    const uint64_t bits = read_u64(c);
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
}

std::string GgufFile::read_string(Cursor &c) {
    const uint64_t len = read_u64(c);
    if (c.pos + len > c.size) {
        throw std::runtime_error("GGUF 解析越界（string）");
    }
    std::string s(reinterpret_cast<const char *>(c.base + c.pos), len);
    c.pos += len;
    return s;
}

GgufValue GgufFile::read_value(Cursor &c, GgufValueType type) {
    GgufValue v;
    v.type = type;
    switch (type) {
        case GgufValueType::UINT8:  v.i64 = read_u8(c); break;
        case GgufValueType::INT8:   v.i64 = static_cast<int8_t>(read_u8(c)); break;
        case GgufValueType::UINT16: { uint32_t lo = read_u8(c), hi = read_u8(c); v.i64 = lo | (hi << 8); break; }
        case GgufValueType::INT16:  { uint32_t lo = read_u8(c), hi = read_u8(c); v.i64 = static_cast<int16_t>(lo | (hi << 8)); break; }
        case GgufValueType::UINT32: v.i64 = read_u32(c); break;
        case GgufValueType::INT32:  v.i64 = static_cast<int32_t>(read_u32(c)); break;
        case GgufValueType::UINT64: v.i64 = static_cast<int64_t>(read_u64(c)); break;
        case GgufValueType::INT64:  v.i64 = static_cast<int64_t>(read_u64(c)); break;
        case GgufValueType::FLOAT32: v.f64 = read_f64_bits(c, type); break;
        case GgufValueType::FLOAT64: v.f64 = read_f64_bits(c, type); break;
        case GgufValueType::BOOL:   v.boolean = (read_u8(c) != 0); break;
        case GgufValueType::STRING: v.str = read_string(c); break;
        case GgufValueType::ARRAY: {
            v.elem_type = static_cast<GgufValueType>(read_u32(c));
            const uint64_t n = read_u64(c);
            for (uint64_t i = 0; i < n; ++i) {
                GgufValue e = read_value(c, v.elem_type);
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

bool GgufFile::has_metadata(const std::string &key) const {
    return metadata_.count(key) > 0;
}

const GgufValue &GgufFile::metadata(const std::string &key) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) {
        throw std::runtime_error("GGUF 缺少元数据 key: " + key);
    }
    return it->second;
}

int64_t GgufFile::metadata_i64(const std::string &key) const {
    return metadata(key).i64;
}

float GgufFile::metadata_f32(const std::string &key) const {
    return static_cast<float>(metadata(key).f64);
}

std::string GgufFile::metadata_str(const std::string &key) const {
    return metadata(key).str;
}

bool GgufFile::metadata_bool(const std::string &key) const {
    return metadata(key).boolean;
}

bool GgufFile::has_tensor(const std::string &name) const {
    return tensor_index_.count(name) > 0;
}

const GgufTensorInfo &GgufFile::tensor_info(const std::string &name) const {
    auto it = tensor_index_.find(name);
    if (it == tensor_index_.end()) {
        throw std::runtime_error("GGUF 缺少张量: " + name);
    }
    return tensors_[it->second];
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

const TensorView &GgufFile::tensor(const std::string &name) const {
    auto cached = view_cache_.find(name);
    if (cached != view_cache_.end()) {
        return cached->second;
    }
    const GgufTensorInfo &info = tensor_info(name);
    TensorView view;
    view.name = info.name;
    // GGUF dims 以最内连续维在前；统一约定 shape 为行主序 [out, in]，故反转。
    view.shape.assign(info.dims.rbegin(), info.dims.rend());
    view.dtype = gguf_type_to_dtype(info.type);
    view.data = info.data;
    view.nbytes = info.nbytes;
    auto [it, _] = view_cache_.emplace(name, std::move(view));
    return it->second;
}

std::vector<std::string> GgufFile::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const GgufTensorInfo &t : tensors_) {
        names.push_back(t.name);
    }
    return names;
}

void GgufFile::DebugDump() const {
    Log::debug("GGUF: " + path_.string());
    Log::debug("  version: " + std::to_string(version_));
    Log::debug("  alignment: " + std::to_string(alignment_));
    Log::debug("  metadata_kv: " + std::to_string(metadata_.size()));
    Log::debug("  tensors: " + std::to_string(tensors_.size()));

    // 全部元数据 KV（按 key 字典序，std::map 已有序）。
    Log::debug("  === metadata (" + std::to_string(metadata_.size()) + ") ===");
    for (const auto &[key, val] : metadata_) {
        Log::debug("    " + key + " : " + value_type_name(val.type) + " = " + value_to_string(val));
    }

    // 张量量化类型直方图（先统计，放在张量列表之上做概览）。
    std::map<GgmlType, size_t> type_hist;
    for (const auto &t : tensors_) {
        type_hist[t.type]++;
    }
    Log::debug("  === tensor type histogram ===");
    for (const auto &[type, count] : type_hist) {
        Log::debug(std::string("    ") + ggml_type_name(type) + " : " + std::to_string(count));
    }

    // 全部张量元信息（不含权重数据段），按文件出现顺序。
    Log::debug("  === tensors (" + std::to_string(tensors_.size()) + ") ===");
    for (const auto &t : tensors_) {
        std::ostringstream os;
        os << "    " << t.name << "  dims=" << dims_to_string(t.dims)
           << " type=" << ggml_type_name(t.type) << " offset=" << t.offset
           << " nbytes=" << t.nbytes;
        Log::debug(os.str());
    }
}
