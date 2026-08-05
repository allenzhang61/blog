#include "profile.h"

#include "../kernels/cuda/cuda_ops.h"

#include <iomanip>
#include <iostream>
#include <sstream>

namespace llm_inference {

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
    out << "  \"cuda_cublas_enabled\": " << (cuda_cublas_enabled() ? "true" : "false") << ",\n";
    out << "  \"cuda_fused_mlp_enabled\": " << (cuda_fused_mlp_enabled() ? "true" : "false") << ",\n";
    out << "  \"cuda_project_attention_enabled\": " << (cuda_project_attention_enabled() ? "true" : "false") << ",\n";
    out << "  \"cuda_full_layer_enabled\": " << (cuda_full_layer_enabled() ? "true" : "false") << ",\n";
    out << "  \"cuda_rmsnorm_mlp_enabled\": " << (cuda_rmsnorm_mlp_enabled() ? "true" : "false") << ",\n";
    out << "  \"status\": \""
        << (cuda_full_layer_enabled() ? "native_cuda_full_layer_forward_done" :
            (cuda_project_attention_enabled() ? "native_cuda_project_attention_fused_mlp_forward_done" :
            (cuda_fused_mlp_enabled() ? "native_cuda_matvec_fused_mlp_forward_done" :
            (cuda_cublas_enabled() ? "native_cuda_matvec_forward_done" : "native_cpu_forward_done"))
        ))
        << "\"\n";
    out << "}";
    return out.str();
}

} // namespace llm_inference
