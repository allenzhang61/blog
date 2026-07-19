#pragma once

#include "llm/backend.hpp"
#include "llm/ops.hpp"

namespace llm {

class Module {
public:
    virtual ~Module() = default;
    virtual std::vector<Tensor*> parameters();
    void zero_grad();
};

class Linear : public Module {
public:
    Tensor weight;
    Tensor bias;
    bool use_bias{true};

    Linear(int64_t in_features, int64_t out_features, bool bias_enabled = true);
    Tensor forward(const Tensor& x);
    std::vector<Tensor*> parameters() override;
};

class Embedding : public Module {
public:
    Tensor weight;

    Embedding(int64_t num_embeddings, int64_t embedding_dim);
    Tensor forward(const Tensor& ids);
    std::vector<Tensor*> parameters() override;
};

class LayerNorm : public Module {
public:
    Tensor scale;
    Tensor shift;
    double eps{1e-5};

    explicit LayerNorm(int64_t emb_dim);
    Tensor forward(const Tensor& x);
    std::vector<Tensor*> parameters() override;
};

class GELU : public Module {
public:
    Tensor forward(const Tensor& x);
};

struct GPTConfig {
    int64_t vocab_size{50257};
    int64_t context_length{256};
    int64_t emb_dim{768};
    int64_t n_heads{12};
    int64_t n_layers{12};
    double drop_rate{0.1};
    bool qkv_bias{false};
};

class MultiHeadAttention : public Module {
public:
    int64_t d_out;
    int64_t num_heads;
    int64_t head_dim;
    int64_t context_length;
    Linear W_query;
    Linear W_key;
    Linear W_value;
    Linear out_proj;

    MultiHeadAttention(int64_t d_in, int64_t d_out_, int64_t context, int64_t heads, bool qkv_bias = false);
    Tensor forward(const Tensor& x);
    std::vector<Tensor*> parameters() override;
};

class FeedForward : public Module {
public:
    Linear fc1;
    GELU gelu;
    Linear fc2;

    explicit FeedForward(const GPTConfig& cfg);
    Tensor forward(const Tensor& x);
    std::vector<Tensor*> parameters() override;
};

class TransformerBlock : public Module {
public:
    MultiHeadAttention att;
    FeedForward ff;
    LayerNorm norm1;
    LayerNorm norm2;

    explicit TransformerBlock(const GPTConfig& cfg);
    Tensor forward(const Tensor& x);
    std::vector<Tensor*> parameters() override;
};

class GPTModel : public Module {
public:
    GPTConfig cfg;
    Embedding tok_emb;
    Embedding pos_emb;
    std::vector<TransformerBlock> blocks;
    LayerNorm final_norm;
    Linear out_head;

    explicit GPTModel(GPTConfig cfg_);
    Tensor forward(const Tensor& ids);
    std::vector<Tensor*> parameters() override;
};

} // namespace llm
