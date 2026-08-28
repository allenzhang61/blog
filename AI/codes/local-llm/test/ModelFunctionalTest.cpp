#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <sys/wait.h>

#ifndef LOCAL_LLM_BINARY_PATH
#define LOCAL_LLM_BINARY_PATH ""
#endif

namespace {

struct CommandResult {
    int exit_code = -1;
    std::string output;
};

std::string env_or_default(const char *key, const char *fallback) {
    const char *value = std::getenv(key);
    return (value != nullptr && value[0] != '\0') ? value : fallback;
}

std::string shell_quote(const std::string &value) {
    std::string quoted = "'";
    for (const char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

CommandResult run_command(const std::string &command) {
    CommandResult result;
    std::array<char, 4096> buffer{};
    FILE *pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        throw std::runtime_error("popen failed");
    }

    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        result.output += buffer.data();
    }
    const int status = pclose(pipe);
    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    } else {
        result.exit_code = status;
    }
    return result;
}

void require_existing_path(const std::string &path, const char *env_name) {
    if (!std::filesystem::exists(path)) {
        GTEST_SKIP() << env_name << " is not set and default path does not exist: " << path;
    }
}

std::string local_llm_binary() {
    return env_or_default("LOCAL_LLM_TEST_BINARY", LOCAL_LLM_BINARY_PATH);
}

CommandResult run_model(const std::string &model, const std::string &model_dir,
                        const std::string &prompt, int max_output_tokens,
                        const std::string &extra_env = "") {
    const std::string command =
        extra_env +
        "PROMPT=" + shell_quote(prompt) + " " +
        shell_quote(local_llm_binary()) +
        " --model " + shell_quote(model) +
        " --model-dir " + shell_quote(model_dir) +
        " --max-output-tokens " + std::to_string(max_output_tokens) +
        " 2>&1";
    return run_command(command);
}

std::string generated_text(const CommandResult &result) {
    static const std::string kBegin = "生成结果：";
    static const std::string kEnd = "\n[decode]";
    const size_t begin = result.output.find(kBegin);
    if (begin == std::string::npos) {
        return "";
    }
    size_t content_begin = begin + kBegin.size();
    if (content_begin < result.output.size() && result.output[content_begin] == ' ') {
        ++content_begin;
    }
    const size_t end = result.output.find(kEnd, content_begin);
    if (end == std::string::npos) {
        return result.output.substr(content_begin);
    }
    return result.output.substr(content_begin, end - content_begin);
}

void expect_generated_text_eq(const CommandResult &result, const std::string &expected) {
    ASSERT_EQ(0, result.exit_code) << result.output;
    EXPECT_EQ(expected, generated_text(result)) << result.output;
}

} // namespace

TEST(ModelFunctionalTest, QwenStoryPromptOutputsCoherentContinuation) {
    const std::string binary = local_llm_binary();
    if (binary.empty()) {
        GTEST_SKIP() << "LOCAL_LLM_TEST_BINARY is not set and LOCAL_LLM_BINARY_PATH is empty";
    }
    require_existing_path(binary, "LOCAL_LLM_TEST_BINARY");
    const std::string model_dir = env_or_default("LOCAL_LLM_TEST_QWEN_MODEL_DIR",
                                                 "/home/zyl/models/Qwen3.5-4B-Base");
    require_existing_path(model_dir, "LOCAL_LLM_TEST_QWEN_MODEL_DIR");

    const CommandResult result = run_model("qwen", model_dir,
                                           "Once upon a time, in a small village", 48);

    expect_generated_text_eq(result,
                             R"(, there lived a young boy named Tom. Tom was very curious about everything around him, especially about the world of science.

One day, Tom's teacher told him about something called "the scientific method". Tom was really interested and wanted)");
}

TEST(ModelFunctionalTest, DeepseekStoryPromptOutputsCoherentContinuation) {
    const std::string binary = local_llm_binary();
    if (binary.empty()) {
        GTEST_SKIP() << "LOCAL_LLM_TEST_BINARY is not set and LOCAL_LLM_BINARY_PATH is empty";
    }
    require_existing_path(binary, "LOCAL_LLM_TEST_BINARY");
    const std::string model_dir = env_or_default("LOCAL_LLM_TEST_DEEPSEEK_MODEL_DIR",
                                                 "/home/zyl/models/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf");
    require_existing_path(model_dir, "LOCAL_LLM_TEST_DEEPSEEK_MODEL_DIR");

    const CommandResult result = run_model(
        "deepseek", model_dir,
        "User: What is the capital of France?\n\nAssistant:", 7,
        "LOCAL_LLM_CUDA_DEQUANT_POOL_GB=1 ");

    expect_generated_text_eq(result,
                             "The capital of France is Paris.");
}
