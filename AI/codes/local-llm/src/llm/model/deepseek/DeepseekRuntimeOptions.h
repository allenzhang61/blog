//
// Created by zhangyoulun on 31/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_RUNTIME_OPTIONS_H
#define LOCAL_LLM_DEEPSEEK_RUNTIME_OPTIONS_H

#include <cstdlib>

struct DeepseekRuntimeOptions {
    bool quant_direct = true;
    bool experimental_moe_fused_swiglu = false;
    bool debug_device_indexed_moe = false;
    bool experimental_mla_absorb = false;
    bool experimental_mla_absorb_hybrid = false;
    bool debug_mla_absorb_compare = false;
    bool experimental_prefill_tiled_mmq = false;
};

inline DeepseekRuntimeOptions deepseek_runtime_options() {
    const auto flag_enabled = [](const char *key) {
        const char *env = std::getenv(key);
        return env != nullptr && std::atoi(env) > 0;
    };
    const auto flag_default_enabled = [](const char *key) {
        const char *env = std::getenv(key);
        return env == nullptr || std::atoi(env) > 0;
    };

    return {
        /* .quant_direct = */ flag_default_enabled("LOCAL_LLM_DEEPSEEK_QUANT_DIRECT"),
        /* .experimental_moe_fused_swiglu = */
        flag_enabled("LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_MOE_FUSED_SWIGLU"),
        /* .debug_device_indexed_moe = */ flag_enabled("LOCAL_LLM_DEBUG_DEVICE_INDEXED_MOE"),
        /* .experimental_mla_absorb = */ flag_enabled("LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_MLA_ABSORB"),
        /* .experimental_mla_absorb_hybrid = */
        flag_enabled("LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_MLA_ABSORB_HYBRID"),
        /* .debug_mla_absorb_compare = */ flag_enabled("LOCAL_LLM_DEBUG_DEEPSEEK_MLA_ABSORB_COMPARE"),
        /* .experimental_prefill_tiled_mmq = */
        flag_enabled("LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_PREFILL_TILED_MMQ"),
    };
}

#endif // LOCAL_LLM_DEEPSEEK_RUNTIME_OPTIONS_H
