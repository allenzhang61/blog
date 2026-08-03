#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace llm_inference {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

extern const char * MODEL_ID;
extern const char * DEFAULT_PROMPT;
extern const std::vector<int> DEFAULT_PROMPT_IDS;

struct Args {
    std::string model_dir;
    std::string prompt;
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
    MappedFile(MappedFile && other) noexcept;
    MappedFile & operator=(MappedFile && other) noexcept;
    ~MappedFile();
    void close();
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

double elapsed_s(Clock::time_point start);
void log(const std::string & message);
Args parse_args(int argc, char ** argv);
std::string read_text_file(const fs::path & path);
std::string json_escape(const std::string & value);
std::string shape_to_string(const std::vector<int64_t> & shape);
std::string profile_json(const ModelConfig & config, const ModelWeights & weights, const Timing & timing, const Args & args);

ModelConfig load_config(const fs::path & model_dir);
ModelWeights load_weights_mmap(const fs::path & model_dir);
TensorRef tensor_ref(const ModelWeights & weights, const std::string & name);
bool has_tensor(const ModelWeights & weights, const std::string & name);
void dump_tensors(const ModelWeights & weights);

float tensor_value(const TensorRef & ref, size_t index);
float dot_row(const TensorRef & weight, int row, const std::vector<float> & x);
void matvec(const TensorRef & weight, const std::vector<float> & x, std::vector<float> & y);
void embedding_lookup(const TensorRef & emb, int token_id, std::vector<float> & y);
void rms_norm(const TensorRef & weight, const std::vector<float> & x, std::vector<float> & y, float eps, bool one_plus);
float sigmoid(float x);
float silu(float x);
float softplus(float x);
void l2_norm_inplace(float * x, int dim, float eps = 1e-6f);
void gated_rms_norm_head(const TensorRef & weight, const float * x, const float * gate, float * y, int dim, float eps);
void add_inplace(std::vector<float> & x, const std::vector<float> & y);
bool cuda_cublas_enabled();
bool cuda_fused_mlp_enabled();
bool cuda_rmsnorm_mlp_enabled();
bool cuda_mlp_layer(const TensorRef & gate_w, const TensorRef & up_w, const TensorRef & down_w, const std::vector<float> & x, std::vector<float> & out);
bool cuda_rmsnorm_mlp_layer(
    const TensorRef & norm_w,
    const TensorRef & gate_w,
    const TensorRef & up_w,
    const TensorRef & down_w,
    const std::vector<float> & x,
    float eps,
    bool one_plus,
    std::vector<float> & out);

std::unordered_map<int, std::string> load_vocab_reverse(const fs::path & model_dir, double & elapsed);
std::string detokenize(const std::vector<int> & ids, const std::unordered_map<int, std::string> & vocab);
std::vector<int> resolve_input_ids(const Args & args);

void validate_qwen_tensors(const ModelWeights & weights, const ModelConfig & config);
std::vector<int> run_generation(const ModelConfig & config, const ModelWeights & weights, const Args & args, const std::vector<int> & input_ids, Timing & timing);

} // namespace llm_inference
