#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

constexpr const char * MODEL_ID = "Qwen/Qwen3.5-4B";
constexpr const char * DEFAULT_PROMPT = "介绍一下 TCP 三次握手";

// tokenizer.apply_chat_template(... enable_thinking=True) for DEFAULT_PROMPT.
const std::vector<int> DEFAULT_PROMPT_IDS = {
    248045, 846, 198, 113552, 25804, 220, 110114, 119587,
    248046, 198, 248045, 74455, 198, 248068, 198,
};

using Clock = std::chrono::steady_clock;

struct Args {
    std::string model_dir;
    std::string prompt = DEFAULT_PROMPT;
    std::vector<int> input_ids;
    int max_new_tokens = 1;
    int warmup_runs = 0;
    float temperature = 0.7f;
    bool greedy = false;
    bool disable_thinking = false;
    bool profile_timing = false;
    bool dump_tensors = false;
};

struct ModelConfig {
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int vocab_size = 0;
    int head_dim = 0;
    int linear_num_value_heads = 0;
    int linear_num_key_heads = 0;
    int linear_key_head_dim = 0;
    int linear_value_head_dim = 0;
    int linear_conv_kernel_dim = 4;
    int eos_token_id = -1;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000000.0f;
    float partial_rotary_factor = 0.25f;
    std::vector<std::string> layer_types;
};

struct MappedFile {
    fs::path path;
    int fd = -1;
    size_t size = 0;
    const uint8_t * data = nullptr;

    MappedFile() = default;
    MappedFile(const MappedFile &) = delete;
    MappedFile & operator=(const MappedFile &) = delete;

    MappedFile(MappedFile && other) noexcept {
        *this = std::move(other);
    }

    MappedFile & operator=(MappedFile && other) noexcept {
        if (this != &other) {
            close();
            path = std::move(other.path);
            fd = other.fd;
            size = other.size;
            data = other.data;
            other.fd = -1;
            other.size = 0;
            other.data = nullptr;
        }
        return *this;
    }

    ~MappedFile() {
        close();
    }

    void close() {
        if (data != nullptr && size > 0) {
            munmap(const_cast<uint8_t *>(data), size);
        }
        if (fd >= 0) {
            ::close(fd);
        }
        data = nullptr;
        fd = -1;
        size = 0;
    }
};

struct TensorInfo {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    size_t data_begin = 0;
    size_t data_end = 0;
    size_t file_index = 0;
};

struct TensorRef {
    const TensorInfo * info = nullptr;
    const uint8_t * data = nullptr;
};

struct ModelWeights {
    std::vector<MappedFile> files;
    std::map<std::string, TensorInfo> tensors;
};

struct LinearLayerState {
    std::vector<float> conv_state;
    std::vector<float> recurrent_state;
};

struct FullAttentionState {
    std::vector<float> key_cache;
    std::vector<float> value_cache;
};

struct RunState {
    int seq_len = 0;
    std::vector<LinearLayerState> linear;
    std::vector<FullAttentionState> full;
};

struct Timing {
    double load_config_s = 0.0;
    double load_weights_s = 0.0;
    double load_vocab_s = 0.0;
    double validate_s = 0.0;
    double prefill_s = 0.0;
    double decode_total_s = 0.0;
    double logits_s = 0.0;
    double warmup_s = 0.0;
    double infer_wall_s = 0.0;
    int input_tokens = 0;
    int generated_tokens = 0;
    std::vector<int> generated_ids;
};

double elapsed_s(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void log(const std::string & message) {
    std::cerr << message << std::endl;
}

[[noreturn]] void usage(const char * argv0, int code) {
    std::cerr
        << "用法:\n"
        << "  " << argv0 << " --model-dir DIR [--prompt TEXT] [options]\n\n"
        << "参数:\n"
        << "  --model-dir DIR          Hugging Face 模型 snapshot 目录，必填\n"
        << "  -p, --prompt TEXT        输入语句；当前仅内置默认 prompt tokenizer\n"
        << "  --input-ids IDS          逗号分隔 token ids，用于绕过 tokenizer\n"
        << "  --max-new-tokens N       最大生成 token 数；默认 1\n"
        << "  --temperature T          记录参数；当前完整路径只实现 greedy\n"
        << "  --greedy                 贪心解码\n"
        << "  --disable-thinking       当前仅影响日志；默认 prompt token ids 是 thinking=True 口径\n"
        << "  --warmup-runs N          正式统计前预热 N 次；默认 0\n"
        << "  --profile-timing         输出 PROFILE_TIMING_JSON\n"
        << "  --dump-tensors           打印 safetensors 中的 tensor 列表\n"
        << "  -h, --help               显示帮助\n";
    std::exit(code);
}

int parse_int(const char * value, const std::string & name) {
    try {
        return std::stoi(value);
    } catch (...) {
        throw std::runtime_error(name + " 需要整数。");
    }
}

float parse_float(const char * value, const std::string & name) {
    try {
        return std::stof(value);
    } catch (...) {
        throw std::runtime_error(name + " 需要数字。");
    }
}

std::vector<int> parse_input_ids(const std::string & text) {
    std::vector<int> ids;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            ids.push_back(std::stoi(item));
        }
    }
    return ids;
}

Args parse_args(int argc, char ** argv) {
    Args args;
    std::string positional_prompt;

    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&](const std::string & name) -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error(name + " 缺少参数值。");
            }
            return argv[++i];
        };

        if (key == "-h" || key == "--help") {
            usage(argv[0], 0);
        } else if (key == "--model-dir") {
            args.model_dir = need_value(key);
        } else if (key == "-p" || key == "--prompt") {
            args.prompt = need_value(key);
        } else if (key == "--input-ids") {
            args.input_ids = parse_input_ids(need_value(key));
        } else if (key == "--max-new-tokens") {
            args.max_new_tokens = parse_int(need_value(key), key);
        } else if (key == "--temperature") {
            args.temperature = parse_float(need_value(key), key);
        } else if (key == "--greedy") {
            args.greedy = true;
        } else if (key == "--disable-thinking") {
            args.disable_thinking = true;
        } else if (key == "--warmup-runs") {
            args.warmup_runs = parse_int(need_value(key), key);
        } else if (key == "--profile-timing") {
            args.profile_timing = true;
        } else if (key == "--dump-tensors") {
            args.dump_tensors = true;
        } else if (key == "--device" || key == "--dtype" || key == "--cache-dir" || key == "--revision" ||
                   key == "--torch-profiler") {
            (void) need_value(key);
            log("提示：" + key + " 是 Python 版本参数，当前原生 C++ CPU 实现先忽略。");
        } else if (key == "--fast-decode" || key == "--static-cache" || key == "--nvtx") {
            log("提示：" + key + " 当前原生 C++ 实现先忽略。");
        } else if (!key.empty() && key[0] == '-') {
            throw std::runtime_error("未知参数：" + key);
        } else {
            if (!positional_prompt.empty()) {
                positional_prompt += " ";
            }
            positional_prompt += key;
        }
    }

    if (!positional_prompt.empty() && args.prompt == DEFAULT_PROMPT) {
        args.prompt = positional_prompt;
    }
    if (args.model_dir.empty()) {
        usage(argv[0], 1);
    }
    if (args.max_new_tokens <= 0) {
        throw std::runtime_error("--max-new-tokens 必须大于 0。");
    }
    if (args.warmup_runs < 0) {
        args.warmup_runs = 0;
    }
    if (!args.greedy) {
        log("提示：当前原生 C++ 完整推理路径只实现 greedy，已按 greedy 执行。");
        args.greedy = true;
    }
    return args;
}

std::string read_text_file(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("无法读取文件：" + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int json_int(const std::string & json, const std::string & key, int default_value = 0) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return std::stoi(match[1].str());
    }
    return default_value;
}

float json_float(const std::string & json, const std::string & key, float default_value = 0.0f) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:e-?[0-9]+)?)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return std::stof(match[1].str());
    }
    return default_value;
}

std::vector<std::string> parse_layer_types(const std::string & json) {
    std::vector<std::string> values;
    const std::regex block("\"layer_types\"\\s*:\\s*\\[([^\\]]+)\\]");
    std::smatch match;
    if (!std::regex_search(json, match, block)) {
        return values;
    }
    const std::string body = match[1].str();
    const std::regex item("\"([^\"]+)\"");
    for (auto it = std::sregex_iterator(body.begin(), body.end(), item); it != std::sregex_iterator(); ++it) {
        values.push_back((*it)[1].str());
    }
    return values;
}

ModelConfig load_config(const fs::path & model_dir) {
    const std::string json = read_text_file(model_dir / "config.json");
    ModelConfig config;
    config.hidden_size = json_int(json, "hidden_size");
    config.intermediate_size = json_int(json, "intermediate_size");
    config.num_hidden_layers = json_int(json, "num_hidden_layers");
    config.num_attention_heads = json_int(json, "num_attention_heads");
    config.num_key_value_heads = json_int(json, "num_key_value_heads", config.num_attention_heads);
    config.vocab_size = json_int(json, "vocab_size");
    config.head_dim = json_int(json, "head_dim", config.hidden_size / std::max(config.num_attention_heads, 1));
    config.linear_num_value_heads = json_int(json, "linear_num_value_heads", 32);
    config.linear_num_key_heads = json_int(json, "linear_num_key_heads", 16);
    config.linear_key_head_dim = json_int(json, "linear_key_head_dim", 128);
    config.linear_value_head_dim = json_int(json, "linear_value_head_dim", 128);
    config.linear_conv_kernel_dim = json_int(json, "linear_conv_kernel_dim", 4);
    config.eos_token_id = json_int(json, "eos_token_id", -1);
    config.rms_norm_eps = json_float(json, "rms_norm_eps", 1e-6f);
    config.rope_theta = json_float(json, "rope_theta", 10000000.0f);
    config.partial_rotary_factor = json_float(json, "partial_rotary_factor", 0.25f);
    config.layer_types = parse_layer_types(json);

    if (config.hidden_size <= 0 || config.num_hidden_layers <= 0 || config.num_attention_heads <= 0) {
        throw std::runtime_error("config.json 缺少 Qwen 推理所需字段。");
    }
    if (static_cast<int>(config.layer_types.size()) != config.num_hidden_layers) {
        config.layer_types.clear();
        for (int i = 0; i < config.num_hidden_layers; ++i) {
            config.layer_types.push_back((i + 1) % 4 == 0 ? "full_attention" : "linear_attention");
        }
    }
    return config;
}

MappedFile mmap_file(const fs::path & path) {
    MappedFile file;
    file.path = path;
    file.fd = ::open(path.c_str(), O_RDONLY);
    if (file.fd < 0) {
        throw std::runtime_error("open 失败：" + path.string() + "，" + std::strerror(errno));
    }

    struct stat st {};
    if (fstat(file.fd, &st) != 0) {
        throw std::runtime_error("fstat 失败：" + path.string() + "，" + std::strerror(errno));
    }
    file.size = static_cast<size_t>(st.st_size);
    if (file.size < 8) {
        throw std::runtime_error("safetensors 文件太小：" + path.string());
    }

    void * ptr = mmap(nullptr, file.size, PROT_READ, MAP_PRIVATE, file.fd, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("mmap 失败：" + path.string() + "，" + std::strerror(errno));
    }
    file.data = static_cast<const uint8_t *>(ptr);
    return file;
}

uint64_t read_u64_le(const uint8_t * data) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(data[i]) << (8 * i);
    }
    return value;
}

std::vector<int64_t> parse_i64_array(const std::string & text) {
    std::vector<int64_t> values;
    const std::regex number("-?[0-9]+");
    for (auto it = std::sregex_iterator(text.begin(), text.end(), number); it != std::sregex_iterator(); ++it) {
        values.push_back(std::stoll((*it).str()));
    }
    return values;
}

void parse_safetensors_header(ModelWeights & weights, size_t file_index) {
    const MappedFile & file = weights.files[file_index];
    const uint64_t header_len = read_u64_le(file.data);
    if (8 + header_len > file.size) {
        throw std::runtime_error("safetensors header 长度异常：" + file.path.string());
    }

    const std::string header(reinterpret_cast<const char *>(file.data + 8), static_cast<size_t>(header_len));
    const size_t data_base = 8 + static_cast<size_t>(header_len);
    const std::regex tensor_pattern(
        R"REGEX("([^"]+)"\s*:\s*\{\s*"dtype"\s*:\s*"([^"]+)"\s*,\s*"shape"\s*:\s*\[([^\]]*)\]\s*,\s*"data_offsets"\s*:\s*\[\s*([0-9]+)\s*,\s*([0-9]+)\s*\])REGEX");

    for (auto it = std::sregex_iterator(header.begin(), header.end(), tensor_pattern);
         it != std::sregex_iterator();
         ++it) {
        TensorInfo info;
        info.name = (*it)[1].str();
        info.dtype = (*it)[2].str();
        info.shape = parse_i64_array((*it)[3].str());
        info.data_begin = data_base + static_cast<size_t>(std::stoull((*it)[4].str()));
        info.data_end = data_base + static_cast<size_t>(std::stoull((*it)[5].str()));
        info.file_index = file_index;
        if (info.data_begin > info.data_end || info.data_end > file.size) {
            throw std::runtime_error("tensor data_offsets 越界：" + info.name);
        }
        weights.tensors.emplace(info.name, std::move(info));
    }
}

std::vector<fs::path> find_safetensors_files(const fs::path & model_dir) {
    std::vector<fs::path> files;
    for (const auto & entry : fs::directory_iterator(model_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        throw std::runtime_error("模型目录下没有 .safetensors 文件：" + model_dir.string());
    }
    return files;
}

ModelWeights load_weights_mmap(const fs::path & model_dir) {
    ModelWeights weights;
    for (const fs::path & file_path : find_safetensors_files(model_dir)) {
        weights.files.push_back(mmap_file(file_path));
        parse_safetensors_header(weights, weights.files.size() - 1);
    }
    return weights;
}

TensorRef tensor_ref(const ModelWeights & weights, const std::string & name) {
    auto it = weights.tensors.find(name);
    if (it == weights.tensors.end()) {
        throw std::runtime_error("缺少 tensor：" + name);
    }
    const TensorInfo & info = it->second;
    return TensorRef{ &info, weights.files[info.file_index].data + info.data_begin };
}

bool has_tensor(const ModelWeights & weights, const std::string & name) {
    return weights.tensors.find(name) != weights.tensors.end();
}

float bf16_to_float(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float f16_to_float(uint16_t h) {
    const uint16_t h_exp = h & 0x7C00u;
    const uint16_t h_sig = h & 0x03FFu;
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t f;
    if (h_exp == 0) {
        if (h_sig == 0) {
            f = sign;
        } else {
            int exp = -1;
            uint16_t sig = h_sig;
            do {
                ++exp;
                sig <<= 1;
            } while ((sig & 0x0400u) == 0);
            sig &= 0x03FFu;
            f = sign | static_cast<uint32_t>(127 - 15 - exp) << 23 | static_cast<uint32_t>(sig) << 13;
        }
    } else if (h_exp == 0x7C00u) {
        f = sign | 0x7F800000u | (static_cast<uint32_t>(h_sig) << 13);
    } else {
        f = sign | (static_cast<uint32_t>((h_exp >> 10) + (127 - 15)) << 23) | (static_cast<uint32_t>(h_sig) << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

float tensor_value(const TensorRef & ref, size_t index) {
    if (ref.info->dtype == "BF16") {
        const auto * p = reinterpret_cast<const uint16_t *>(ref.data);
        return bf16_to_float(p[index]);
    }
    if (ref.info->dtype == "F16") {
        const auto * p = reinterpret_cast<const uint16_t *>(ref.data);
        return f16_to_float(p[index]);
    }
    if (ref.info->dtype == "F32") {
        const auto * p = reinterpret_cast<const float *>(ref.data);
        return p[index];
    }
    throw std::runtime_error("暂不支持 dtype：" + ref.info->dtype + " tensor=" + ref.info->name);
}

void matvec(const TensorRef & weight, const std::vector<float> & x, std::vector<float> & y) {
    if (weight.info->shape.size() != 2) {
        throw std::runtime_error("matvec 需要二维权重：" + weight.info->name);
    }
    const int out_dim = static_cast<int>(weight.info->shape[0]);
    const int in_dim = static_cast<int>(weight.info->shape[1]);
    if (static_cast<int>(x.size()) != in_dim) {
        throw std::runtime_error("matvec 输入维度不匹配：" + weight.info->name);
    }
    y.assign(out_dim, 0.0f);

#pragma omp parallel for schedule(static)
    for (int o = 0; o < out_dim; ++o) {
        double sum = 0.0;
        const size_t base = static_cast<size_t>(o) * static_cast<size_t>(in_dim);
        for (int i = 0; i < in_dim; ++i) {
            sum += static_cast<double>(tensor_value(weight, base + static_cast<size_t>(i))) * x[i];
        }
        y[o] = static_cast<float>(sum);
    }
}

void embedding_lookup(const TensorRef & emb, int token_id, std::vector<float> & y) {
    const int vocab = static_cast<int>(emb.info->shape[0]);
    const int hidden = static_cast<int>(emb.info->shape[1]);
    if (token_id < 0 || token_id >= vocab) {
        throw std::runtime_error("token id 越界：" + std::to_string(token_id));
    }
    y.resize(hidden);
    const size_t base = static_cast<size_t>(token_id) * static_cast<size_t>(hidden);
    for (int i = 0; i < hidden; ++i) {
        y[i] = tensor_value(emb, base + static_cast<size_t>(i));
    }
}

void rms_norm(const TensorRef & weight, const std::vector<float> & x, std::vector<float> & y, float eps, bool one_plus) {
    const int dim = static_cast<int>(x.size());
    double ss = 0.0;
    for (float v : x) {
        ss += static_cast<double>(v) * v;
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(ss / dim) + eps);
    y.resize(dim);
    for (int i = 0; i < dim; ++i) {
        const float w = tensor_value(weight, static_cast<size_t>(i));
        y[i] = x[i] * scale * (one_plus ? (1.0f + w) : w);
    }
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float silu(float x) {
    return x * sigmoid(x);
}

float softplus(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return std::exp(x);
    }
    return std::log1p(std::exp(x));
}

void l2_norm_inplace(float * x, int dim, float eps = 1e-6f) {
    double ss = 0.0;
    for (int i = 0; i < dim; ++i) {
        ss += static_cast<double>(x[i]) * x[i];
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(ss) + eps);
    for (int i = 0; i < dim; ++i) {
        x[i] *= scale;
    }
}

void gated_rms_norm_head(
    const TensorRef & weight,
    const float * x,
    const float * gate,
    float * y,
    int dim,
    float eps) {
    double ss = 0.0;
    for (int i = 0; i < dim; ++i) {
        ss += static_cast<double>(x[i]) * x[i];
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(ss / dim) + eps);
    for (int i = 0; i < dim; ++i) {
        y[i] = tensor_value(weight, static_cast<size_t>(i)) * x[i] * scale * silu(gate[i]);
    }
}

void add_inplace(std::vector<float> & x, const std::vector<float> & y) {
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] += y[i];
    }
}

void validate_qwen_tensors(const ModelWeights & weights, const ModelConfig & config) {
    const std::string root = "model.language_model.";
    for (const std::string & name : {
             root + "embed_tokens.weight",
             root + "norm.weight",
         }) {
        if (!has_tensor(weights, name)) {
            throw std::runtime_error("缺少 tensor：" + name);
        }
    }

    for (int layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = root + "layers." + std::to_string(layer) + ".";
        for (const std::string & name : {
                 prefix + "input_layernorm.weight",
                 prefix + "post_attention_layernorm.weight",
                 prefix + "mlp.gate_proj.weight",
                 prefix + "mlp.up_proj.weight",
                 prefix + "mlp.down_proj.weight",
             }) {
            if (!has_tensor(weights, name)) {
                throw std::runtime_error("缺少 tensor：" + name);
            }
        }
        if (config.layer_types[layer] == "linear_attention") {
            for (const std::string & name : {
                     prefix + "linear_attn.A_log",
                     prefix + "linear_attn.norm.weight",
                     prefix + "linear_attn.conv1d.weight",
                     prefix + "linear_attn.dt_bias",
                     prefix + "linear_attn.in_proj_a.weight",
                     prefix + "linear_attn.in_proj_b.weight",
                     prefix + "linear_attn.in_proj_qkv.weight",
                     prefix + "linear_attn.in_proj_z.weight",
                     prefix + "linear_attn.out_proj.weight",
                 }) {
                if (!has_tensor(weights, name)) {
                    throw std::runtime_error("缺少 tensor：" + name);
                }
            }
        } else {
            for (const std::string & name : {
                     prefix + "self_attn.q_proj.weight",
                     prefix + "self_attn.k_proj.weight",
                     prefix + "self_attn.v_proj.weight",
                     prefix + "self_attn.o_proj.weight",
                     prefix + "self_attn.q_norm.weight",
                     prefix + "self_attn.k_norm.weight",
                 }) {
                if (!has_tensor(weights, name)) {
                    throw std::runtime_error("缺少 tensor：" + name);
                }
            }
        }
    }
}

RunState make_run_state(const ModelConfig & config, int max_seq_len) {
    RunState state;
    state.linear.resize(config.num_hidden_layers);
    state.full.resize(config.num_hidden_layers);
    const int conv_dim = config.linear_key_head_dim * config.linear_num_key_heads * 2 +
                         config.linear_value_head_dim * config.linear_num_value_heads;
    for (int layer = 0; layer < config.num_hidden_layers; ++layer) {
        if (config.layer_types[layer] == "linear_attention") {
            state.linear[layer].conv_state.assign(static_cast<size_t>(conv_dim) * config.linear_conv_kernel_dim, 0.0f);
            state.linear[layer].recurrent_state.assign(
                static_cast<size_t>(config.linear_num_value_heads) *
                    config.linear_key_head_dim *
                    config.linear_value_head_dim,
                0.0f);
        } else {
            state.full[layer].key_cache.assign(
                static_cast<size_t>(max_seq_len) * config.num_key_value_heads * config.head_dim,
                0.0f);
            state.full[layer].value_cache.assign(
                static_cast<size_t>(max_seq_len) * config.num_key_value_heads * config.head_dim,
                0.0f);
        }
    }
    return state;
}

class NativeQwen {
public:
    NativeQwen(const ModelConfig & config, const ModelWeights & weights)
        : config_(config), weights_(weights) {}

    int generate_next(const std::vector<int> & prompt_ids, RunState & state, Timing & timing) const {
        std::vector<float> hidden;
        const auto prefill_start = Clock::now();
        for (int token : prompt_ids) {
            hidden = forward_token(token, state);
        }
        timing.prefill_s = elapsed_s(prefill_start);
        return argmax_logits(hidden, timing);
    }

    int decode_one(int token, RunState & state, Timing & timing) const {
        const auto decode_start = Clock::now();
        std::vector<float> hidden = forward_token(token, state);
        const int next = argmax_logits(hidden, timing);
        timing.decode_total_s += elapsed_s(decode_start);
        return next;
    }

private:
    TensorRef t(const std::string & name) const {
        return tensor_ref(weights_, name);
    }

    std::vector<float> forward_token(int token, RunState & state) const {
        std::vector<float> x;
        embedding_lookup(t("model.language_model.embed_tokens.weight"), token, x);
        const int pos = state.seq_len;

        for (int layer = 0; layer < config_.num_hidden_layers; ++layer) {
            const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
            std::vector<float> residual = x;
            std::vector<float> normed;
            rms_norm(t(prefix + "input_layernorm.weight"), x, normed, config_.rms_norm_eps, true);

            std::vector<float> mixer_out;
            if (config_.layer_types[layer] == "linear_attention") {
                linear_attention_layer(prefix, normed, state.linear[layer], mixer_out);
            } else {
                full_attention_layer(prefix, normed, state.full[layer], pos, mixer_out);
            }
            x = residual;
            add_inplace(x, mixer_out);

            residual = x;
            rms_norm(t(prefix + "post_attention_layernorm.weight"), x, normed, config_.rms_norm_eps, true);
            mlp_layer(prefix, normed, mixer_out);
            x = residual;
            add_inplace(x, mixer_out);
        }

        std::vector<float> normed;
        rms_norm(t("model.language_model.norm.weight"), x, normed, config_.rms_norm_eps, true);
        state.seq_len += 1;
        return normed;
    }

    void mlp_layer(const std::string & prefix, const std::vector<float> & x, std::vector<float> & out) const {
        std::vector<float> gate;
        std::vector<float> up;
        std::vector<float> prod;
        matvec(t(prefix + "mlp.gate_proj.weight"), x, gate);
        matvec(t(prefix + "mlp.up_proj.weight"), x, up);
        prod.resize(gate.size());
        for (size_t i = 0; i < gate.size(); ++i) {
            prod[i] = silu(gate[i]) * up[i];
        }
        matvec(t(prefix + "mlp.down_proj.weight"), prod, out);
    }

    void linear_attention_layer(
        const std::string & prefix,
        const std::vector<float> & x,
        LinearLayerState & state,
        std::vector<float> & out) const {
        const int key_heads = config_.linear_num_key_heads;
        const int value_heads = config_.linear_num_value_heads;
        const int k_dim = config_.linear_key_head_dim;
        const int v_dim = config_.linear_value_head_dim;
        const int key_total = key_heads * k_dim;
        const int value_total = value_heads * v_dim;
        const int conv_dim = key_total * 2 + value_total;
        const int kernel = config_.linear_conv_kernel_dim;

        std::vector<float> mixed;
        std::vector<float> z;
        std::vector<float> b;
        std::vector<float> a;
        matvec(t(prefix + "linear_attn.in_proj_qkv.weight"), x, mixed);
        matvec(t(prefix + "linear_attn.in_proj_z.weight"), x, z);
        matvec(t(prefix + "linear_attn.in_proj_b.weight"), x, b);
        matvec(t(prefix + "linear_attn.in_proj_a.weight"), x, a);

        const TensorRef conv_w = t(prefix + "linear_attn.conv1d.weight");
        std::vector<float> conv_out(conv_dim);
        for (int d = 0; d < conv_dim; ++d) {
            float * row = state.conv_state.data() + static_cast<size_t>(d) * kernel;
            for (int i = 0; i < kernel - 1; ++i) {
                row[i] = row[i + 1];
            }
            row[kernel - 1] = mixed[d];
            double sum = 0.0;
            for (int k = 0; k < kernel; ++k) {
                sum += static_cast<double>(tensor_value(conv_w, static_cast<size_t>(d) * kernel + k)) * row[k];
            }
            conv_out[d] = silu(static_cast<float>(sum));
        }

        const float * query_base = conv_out.data();
        const float * key_base = conv_out.data() + key_total;
        const float * value_base = conv_out.data() + key_total * 2;
        std::vector<float> core(value_total, 0.0f);
        const TensorRef a_log = t(prefix + "linear_attn.A_log");
        const TensorRef dt_bias = t(prefix + "linear_attn.dt_bias");

        const int repeat = value_heads / key_heads;
        const float q_scale = 1.0f / std::sqrt(static_cast<float>(k_dim));

        for (int vh = 0; vh < value_heads; ++vh) {
            const int kh = vh / repeat;
            std::array<float, 128> q {};
            std::array<float, 128> k {};
            for (int i = 0; i < k_dim; ++i) {
                q[static_cast<size_t>(i)] = query_base[kh * k_dim + i];
                k[static_cast<size_t>(i)] = key_base[kh * k_dim + i];
            }
            l2_norm_inplace(q.data(), k_dim);
            l2_norm_inplace(k.data(), k_dim);
            for (int i = 0; i < k_dim; ++i) {
                q[static_cast<size_t>(i)] *= q_scale;
            }

            const float beta = sigmoid(b[vh]);
            const float g = -std::exp(tensor_value(a_log, static_cast<size_t>(vh))) *
                            softplus(a[vh] + tensor_value(dt_bias, static_cast<size_t>(vh)));
            const float decay = std::exp(g);
            float * rec = state.recurrent_state.data() +
                          static_cast<size_t>(vh) * k_dim * v_dim;

            for (int i = 0; i < k_dim * v_dim; ++i) {
                rec[i] *= decay;
            }

            std::array<float, 128> kv_mem {};
            for (int kd = 0; kd < k_dim; ++kd) {
                const float kval = k[static_cast<size_t>(kd)];
                const float * rec_row = rec + static_cast<size_t>(kd) * v_dim;
                for (int vd = 0; vd < v_dim; ++vd) {
                    kv_mem[static_cast<size_t>(vd)] += rec_row[vd] * kval;
                }
            }

            std::array<float, 128> delta {};
            const float * value = value_base + static_cast<size_t>(vh) * v_dim;
            for (int vd = 0; vd < v_dim; ++vd) {
                delta[static_cast<size_t>(vd)] = (value[vd] - kv_mem[static_cast<size_t>(vd)]) * beta;
            }
            for (int kd = 0; kd < k_dim; ++kd) {
                float * rec_row = rec + static_cast<size_t>(kd) * v_dim;
                const float kval = k[static_cast<size_t>(kd)];
                for (int vd = 0; vd < v_dim; ++vd) {
                    rec_row[vd] += kval * delta[static_cast<size_t>(vd)];
                }
            }
            float * core_head = core.data() + static_cast<size_t>(vh) * v_dim;
            for (int kd = 0; kd < k_dim; ++kd) {
                const float qval = q[static_cast<size_t>(kd)];
                const float * rec_row = rec + static_cast<size_t>(kd) * v_dim;
                for (int vd = 0; vd < v_dim; ++vd) {
                    core_head[vd] += rec_row[vd] * qval;
                }
            }
        }

        const TensorRef norm_w = t(prefix + "linear_attn.norm.weight");
        std::vector<float> gated(value_total);
        for (int vh = 0; vh < value_heads; ++vh) {
            gated_rms_norm_head(
                norm_w,
                core.data() + static_cast<size_t>(vh) * v_dim,
                z.data() + static_cast<size_t>(vh) * v_dim,
                gated.data() + static_cast<size_t>(vh) * v_dim,
                v_dim,
                config_.rms_norm_eps);
        }
        matvec(t(prefix + "linear_attn.out_proj.weight"), gated, out);
    }

    void apply_rope(float * vec, int pos) const {
        const int rotary_dim = static_cast<int>(config_.head_dim * config_.partial_rotary_factor);
        const int half = rotary_dim / 2;
        for (int i = 0; i < half; ++i) {
            const float inv_freq = 1.0f / std::pow(config_.rope_theta, static_cast<float>(2 * i) / rotary_dim);
            const float angle = static_cast<float>(pos) * inv_freq;
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            const float x1 = vec[i];
            const float x2 = vec[i + half];
            vec[i] = x1 * c - x2 * s;
            vec[i + half] = x2 * c + x1 * s;
        }
    }

    void full_attention_layer(
        const std::string & prefix,
        const std::vector<float> & x,
        FullAttentionState & state,
        int pos,
        std::vector<float> & out) const {
        const int n_heads = config_.num_attention_heads;
        const int kv_heads = config_.num_key_value_heads;
        const int head_dim = config_.head_dim;
        const int q_total = n_heads * head_dim;
        const int kv_total = kv_heads * head_dim;

        std::vector<float> q_and_gate;
        std::vector<float> k;
        std::vector<float> v;
        matvec(t(prefix + "self_attn.q_proj.weight"), x, q_and_gate);
        matvec(t(prefix + "self_attn.k_proj.weight"), x, k);
        matvec(t(prefix + "self_attn.v_proj.weight"), x, v);

        std::vector<float> q(q_total);
        std::vector<float> gate(q_total);
        for (int h = 0; h < n_heads; ++h) {
            const int src = h * head_dim * 2;
            const int dst = h * head_dim;
            std::copy(q_and_gate.begin() + src, q_and_gate.begin() + src + head_dim, q.begin() + dst);
            std::copy(q_and_gate.begin() + src + head_dim, q_and_gate.begin() + src + head_dim * 2, gate.begin() + dst);
        }

        const TensorRef q_norm_w = t(prefix + "self_attn.q_norm.weight");
        const TensorRef k_norm_w = t(prefix + "self_attn.k_norm.weight");
        for (int h = 0; h < n_heads; ++h) {
            std::vector<float> tmp(q.begin() + h * head_dim, q.begin() + (h + 1) * head_dim);
            std::vector<float> normed;
            rms_norm(q_norm_w, tmp, normed, config_.rms_norm_eps, true);
            std::copy(normed.begin(), normed.end(), q.begin() + h * head_dim);
            apply_rope(q.data() + h * head_dim, pos);
        }
        for (int h = 0; h < kv_heads; ++h) {
            std::vector<float> tmp(k.begin() + h * head_dim, k.begin() + (h + 1) * head_dim);
            std::vector<float> normed;
            rms_norm(k_norm_w, tmp, normed, config_.rms_norm_eps, true);
            std::copy(normed.begin(), normed.end(), k.begin() + h * head_dim);
            apply_rope(k.data() + h * head_dim, pos);
        }

        std::copy(k.begin(), k.end(), state.key_cache.begin() + static_cast<size_t>(pos) * kv_total);
        std::copy(v.begin(), v.end(), state.value_cache.begin() + static_cast<size_t>(pos) * kv_total);

        std::vector<float> attn(q_total, 0.0f);
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        std::vector<float> scores(static_cast<size_t>(pos) + 1);

        for (int h = 0; h < n_heads; ++h) {
            const int kh = h / (n_heads / kv_heads);
            const float * qh = q.data() + h * head_dim;
            float max_score = -std::numeric_limits<float>::infinity();
            for (int tpos = 0; tpos <= pos; ++tpos) {
                const float * khp = state.key_cache.data() + static_cast<size_t>(tpos) * kv_total + kh * head_dim;
                double dot = 0.0;
                for (int d = 0; d < head_dim; ++d) {
                    dot += static_cast<double>(qh[d]) * khp[d];
                }
                scores[static_cast<size_t>(tpos)] = static_cast<float>(dot) * scale;
                max_score = std::max(max_score, scores[static_cast<size_t>(tpos)]);
            }
            double denom = 0.0;
            for (int tpos = 0; tpos <= pos; ++tpos) {
                const float e = std::exp(scores[static_cast<size_t>(tpos)] - max_score);
                scores[static_cast<size_t>(tpos)] = e;
                denom += e;
            }
            float * ah = attn.data() + h * head_dim;
            for (int tpos = 0; tpos <= pos; ++tpos) {
                const float prob = scores[static_cast<size_t>(tpos)] / static_cast<float>(denom);
                const float * vh = state.value_cache.data() + static_cast<size_t>(tpos) * kv_total + kh * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    ah[d] += prob * vh[d];
                }
            }
        }

        for (int i = 0; i < q_total; ++i) {
            attn[i] *= sigmoid(gate[i]);
        }
        matvec(t(prefix + "self_attn.o_proj.weight"), attn, out);
    }

    int argmax_logits(const std::vector<float> & hidden, Timing & timing) const {
        const TensorRef emb = t("model.language_model.embed_tokens.weight");
        const int vocab = static_cast<int>(emb.info->shape[0]);
        const int hidden_size = static_cast<int>(emb.info->shape[1]);
        if (static_cast<int>(hidden.size()) != hidden_size) {
            throw std::runtime_error("logits hidden size 不匹配。");
        }

        const auto start = Clock::now();
        int best_id = 0;
        float best = -std::numeric_limits<float>::infinity();

#pragma omp parallel
        {
            int local_id = 0;
            float local_best = -std::numeric_limits<float>::infinity();
#pragma omp for schedule(static)
            for (int token = 0; token < vocab; ++token) {
                double sum = 0.0;
                const size_t base = static_cast<size_t>(token) * hidden_size;
                for (int i = 0; i < hidden_size; ++i) {
                    sum += static_cast<double>(tensor_value(emb, base + static_cast<size_t>(i))) * hidden[i];
                }
                const float score = static_cast<float>(sum);
                if (score > local_best) {
                    local_best = score;
                    local_id = token;
                }
            }
#pragma omp critical
            {
                if (local_best > best) {
                    best = local_best;
                    best_id = local_id;
                }
            }
        }
        timing.logits_s += elapsed_s(start);
        return best_id;
    }

    const ModelConfig & config_;
    const ModelWeights & weights_;
};

std::string json_unescape_string(const std::string & s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\' || i + 1 >= s.size()) {
            out.push_back(s[i]);
            continue;
        }
        const char e = s[++i];
        switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u':
            // vocab.json for byte-level BPE mostly stores UTF-8 directly; keep unicode escapes readable enough for now.
            if (i + 4 < s.size()) {
                out += "\\u" + s.substr(i + 1, 4);
                i += 4;
            }
            break;
        default:
            out.push_back(e);
        }
    }
    return out;
}

std::unordered_map<int, std::string> load_vocab_reverse(const fs::path & model_dir, double & elapsed) {
    auto start = Clock::now();
    const std::string json = read_text_file(model_dir / "vocab.json");
    std::unordered_map<int, std::string> vocab;
    const std::regex item(R"REGEX("((?:\\.|[^"\\])*)"\s*:\s*([0-9]+))REGEX");
    for (auto it = std::sregex_iterator(json.begin(), json.end(), item); it != std::sregex_iterator(); ++it) {
        vocab.emplace(std::stoi((*it)[2].str()), json_unescape_string((*it)[1].str()));
    }
    elapsed = elapsed_s(start);
    return vocab;
}

std::unordered_map<uint32_t, uint8_t> bytes_to_unicode_inverse() {
    std::vector<int> bs;
    for (int i = static_cast<int>('!'); i <= static_cast<int>('~'); ++i) bs.push_back(i);
    for (int i = 0xA1; i <= 0xAC; ++i) bs.push_back(i);
    for (int i = 0xAE; i <= 0xFF; ++i) bs.push_back(i);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    std::unordered_map<uint32_t, uint8_t> inv;
    for (size_t i = 0; i < bs.size(); ++i) {
        inv[static_cast<uint32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
    }
    return inv;
}

bool next_utf8_codepoint(const std::string & s, size_t & i, uint32_t & cp) {
    if (i >= s.size()) {
        return false;
    }
    const unsigned char c = static_cast<unsigned char>(s[i++]);
    if (c < 0x80) {
        cp = c;
        return true;
    }
    int extra = 0;
    cp = 0;
    if ((c & 0xE0) == 0xC0) {
        cp = c & 0x1F;
        extra = 1;
    } else if ((c & 0xF0) == 0xE0) {
        cp = c & 0x0F;
        extra = 2;
    } else if ((c & 0xF8) == 0xF0) {
        cp = c & 0x07;
        extra = 3;
    } else {
        cp = c;
        return true;
    }
    for (int j = 0; j < extra && i < s.size(); ++j) {
        cp = (cp << 6) | (static_cast<unsigned char>(s[i++]) & 0x3F);
    }
    return true;
}

std::string decode_piece(const std::string & piece) {
    static const auto inv = bytes_to_unicode_inverse();
    std::string bytes;
    size_t i = 0;
    uint32_t cp = 0;
    while (next_utf8_codepoint(piece, i, cp)) {
        auto it = inv.find(cp);
        if (it != inv.end()) {
            bytes.push_back(static_cast<char>(it->second));
        } else if (cp < 128) {
            bytes.push_back(static_cast<char>(cp));
        }
    }
    return bytes;
}

std::string detokenize(const std::vector<int> & ids, const std::unordered_map<int, std::string> & vocab) {
    std::string out;
    for (int id : ids) {
        auto it = vocab.find(id);
        if (it == vocab.end()) {
            out += "<id:" + std::to_string(id) + ">";
            continue;
        }
        const std::string & piece = it->second;
        if (piece.rfind("<|", 0) == 0) {
            continue;
        }
        out += decode_piece(piece);
    }
    return out;
}

std::vector<int> resolve_input_ids(const Args & args) {
    if (!args.input_ids.empty()) {
        return args.input_ids;
    }
    if (args.prompt == DEFAULT_PROMPT && !args.disable_thinking) {
        return DEFAULT_PROMPT_IDS;
    }
    throw std::runtime_error("当前 C++ 版本尚未实现 tokenizer。请使用默认 prompt，或用 --input-ids 传入 token ids。");
}

std::string shape_to_string(const std::vector<int64_t> & shape) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << shape[i];
    }
    out << "]";
    return out.str();
}

void dump_tensors(const ModelWeights & weights) {
    for (const auto & [name, info] : weights.tensors) {
        std::cerr << name << " dtype=" << info.dtype
                  << " shape=" << shape_to_string(info.shape)
                  << " file=" << weights.files[info.file_index].path.filename().string()
                  << " bytes=" << (info.data_end - info.data_begin)
                  << "\n";
    }
}

std::string json_escape(const std::string & value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch);
            } else {
                out << ch;
            }
        }
    }
    return out.str();
}

std::string profile_json(
    const ModelConfig & config,
    const ModelWeights & weights,
    const Timing & timing,
    const Args & args) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"model_id\": \"" << MODEL_ID << "\",\n";
    out << "  \"model_dir\": \"" << json_escape(args.model_dir) << "\",\n";
    out << "  \"load_config_s\": " << timing.load_config_s << ",\n";
    out << "  \"load_weights_mmap_s\": " << timing.load_weights_s << ",\n";
    out << "  \"load_vocab_s\": " << timing.load_vocab_s << ",\n";
    out << "  \"validate_tensors_s\": " << timing.validate_s << ",\n";
    out << "  \"mapped_files\": " << weights.files.size() << ",\n";
    out << "  \"tensor_count\": " << weights.tensors.size() << ",\n";
    out << "  \"hidden_size\": " << config.hidden_size << ",\n";
    out << "  \"num_hidden_layers\": " << config.num_hidden_layers << ",\n";
    out << "  \"input_tokens\": " << timing.input_tokens << ",\n";
    out << "  \"generated_tokens\": " << timing.generated_tokens << ",\n";
    out << "  \"generated_ids\": [";
    for (size_t i = 0; i < timing.generated_ids.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << timing.generated_ids[i];
    }
    out << "],\n";
    out << "  \"prefill_s\": " << timing.prefill_s << ",\n";
    out << "  \"decode_total_s\": " << timing.decode_total_s << ",\n";
    out << "  \"logits_s\": " << timing.logits_s << ",\n";
    out << "  \"warmup_runs\": " << args.warmup_runs << ",\n";
    out << "  \"warmup_s\": " << timing.warmup_s << ",\n";
    out << "  \"infer_wall_s\": " << timing.infer_wall_s << ",\n";
    out << "  \"status\": \"native_cpu_forward_done\"\n";
    out << "}";
    return out.str();
}

} // namespace

int main(int argc, char ** argv) {
    try {
        Args args = parse_args(argc, argv);
        const fs::path model_dir(args.model_dir);
        Timing timing;

        log(std::string("开始加载 ") + MODEL_ID + " 原生 C++ 权重 ...");
        auto start = Clock::now();
        const ModelConfig config = load_config(model_dir);
        timing.load_config_s = elapsed_s(start);

        start = Clock::now();
        ModelWeights weights = load_weights_mmap(model_dir);
        timing.load_weights_s = elapsed_s(start);

        start = Clock::now();
        validate_qwen_tensors(weights, config);
        timing.validate_s = elapsed_s(start);

        double vocab_s = 0.0;
        auto vocab = load_vocab_reverse(model_dir, vocab_s);
        timing.load_vocab_s = vocab_s;

        if (args.dump_tensors) {
            dump_tensors(weights);
        }

        const std::vector<int> input_ids = resolve_input_ids(args);
        timing.input_tokens = static_cast<int>(input_ids.size());

        NativeQwen model(config, weights);

        if (args.warmup_runs > 0) {
            log("开始预热，次数 " + std::to_string(args.warmup_runs) + "，不计入正式推理耗时...");
            start = Clock::now();
            for (int i = 0; i < args.warmup_runs; ++i) {
                Timing warm_timing;
                RunState warm_state = make_run_state(config, timing.input_tokens + args.max_new_tokens + 4);
                (void) model.generate_next(input_ids, warm_state, warm_timing);
            }
            timing.warmup_s = elapsed_s(start);
            log("预热完成，耗时 " + std::to_string(timing.warmup_s) + "s");
        }

        log("开始推理...");
        start = Clock::now();
        RunState state = make_run_state(config, timing.input_tokens + args.max_new_tokens + 4);
        std::vector<int> generated;
        int next = model.generate_next(input_ids, state, timing);
        for (int i = 0; i < args.max_new_tokens; ++i) {
            generated.push_back(next);
            timing.generated_ids.push_back(next);
            timing.generated_tokens += 1;
            if (next == config.eos_token_id) {
                break;
            }
            if (i + 1 < args.max_new_tokens) {
                next = model.decode_one(next, state, timing);
            }
        }
        timing.infer_wall_s = elapsed_s(start);
        log("推理完成，耗时 " + std::to_string(timing.infer_wall_s) +
            "s，max_new_tokens=" + std::to_string(args.max_new_tokens));

        if (args.profile_timing) {
            log("PROFILE_TIMING_JSON:");
            log(profile_json(config, weights, timing, args));
        }

        std::cout << detokenize(generated, vocab) << std::endl;
        return 0;
    } catch (const std::exception & exc) {
        std::cerr << "推理失败：" << exc.what() << std::endl;
        return 1;
    }
}
