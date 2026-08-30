#include <fstream>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include <cuda_runtime.h>

#include "backend/cuda/common.h"
#include "utils/cli/Args.h"
#include "format/MF.h"
#include "format/MFFactory.h"
#include "llm/model/BaseModel.h"
#include "llm/model/ModelFactory.h"
#include "backend/cuda/mem/SessionBase.h"
#include "backend/cuda/mem/CudaWeightDequantPool.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "utils/stats/DeviceMonitor.h"
#include "utils/stats/MemoryReporter.h"
#include "utils/stats/MemoryUsageProvider.h"
#include "utils/stats/Profiler.h"
#include "utils/stats/ScopedTimer.h"
#include "utils/stats/WeightLoadTracker.h"

namespace {
class EmptyMemoryUsageProvider final : public MemoryUsageProvider {
public:
    size_t kv_state_bytes() const override { return 0; }
    size_t scratch_bytes() const override { return 0; }
};

bool env_flag_enabled(const char *key) {
    const char *env = std::getenv(key);
    return env != nullptr && std::atoi(env) > 0;
}
} // namespace

int main(int argc, char **argv) {
    Args args(argc, argv);
    args.debug_dump();

    CudaWeightDequantPool dequant_pool;
    set_global_cuda_weight_dequant_pool(&dequant_pool);
    CudaWeightPool weight_pool;
    set_global_cuda_weight_pool(&weight_pool);

    // 非阻塞流：让全部 kernel/cuBLAS/async memcpy 走同一条可捕获的流，为 CUDA Graph 铺路。
    cudaStream_t compute_stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking), "创建 compute stream 失败");
    set_current_cuda_stream(compute_stream);
    auto destroy_compute_stream = [&compute_stream]() {
        if (compute_stream != nullptr) {
            check_cuda(cudaStreamSynchronize(compute_stream), "同步 compute stream 失败");
            set_current_cuda_stream(nullptr);
            check_cuda(cudaStreamDestroy(compute_stream), "销毁 compute stream 失败");
            compute_stream = nullptr;
        }
    };

    // main 负责打开具体模型文件格式；模型本身只接收 MF 抽象。
    std::unique_ptr<MF> mf = open_mf(args.model_dir);
    // 经工厂按 --model 构造具体模型；主循环之后只依赖 BaseModel 接口。
    std::unique_ptr<BaseModel> model = create_model(args.model, std::move(mf),
                                                    args.max_output_tokens, args.sampling);
    CudaWeightPool &pool = global_cuda_weight_pool();

    const std::string text = std::getenv("PROMPT") ? std::getenv("PROMPT") : "法国的首都是";
    const int eos = model->eos_token_id();

    // warmup：profile 模式先完整生成一轮，触发权重惰性上传、CUDA context / cuBLAS
    // 初始化与 kernel 首次启动；之后才开启统计，让性能报告尽量只反映稳态推理。
    // 非 profile 模式仍只短 warmup，避免普通运行无端多跑一遍完整输出。
    {
        std::unique_ptr<SessionBase> warmup_session(model->create_session(text));
        int wnext = model->prefill(*warmup_session);
        const int warmup_decode_steps = args.profile
                                            ? args.max_output_tokens
                                            : (args.max_output_tokens < 4 ? args.max_output_tokens : 4);
        int warmup_tokens = 0;
        for (int i = 0; i < warmup_decode_steps && wnext != eos; ++i) {
            warmup_session->append_output(wnext);
            wnext = model->decode(*warmup_session);
            ++warmup_tokens;
        }
        check_cuda(cudaStreamSynchronize(get_current_cuda_stream()), "warmup sync 失败");
        if (args.profile) {
            std::cout << "[warmup] tokens=" << warmup_tokens
                      << " finished before profiling; cached_bytes=" << pool.cached_bytes()
                      << std::endl;
        }
    }

    // 性能采集：warmup 完成后才开启，所有埋点否则零开销。
    Profiler::instance().enable(args.profile);
    // 权重懒加载追踪：只统计正式推理阶段；如果仍出现事件，说明 warmup 未覆盖或 pool 发生淘汰。
    InMemoryWeightLoadTracker weight_tracker;
    if (args.profile) {
        pool.set_load_tracker(&weight_tracker);
    }
    // 设备侧时序监控：后台线程按间隔采 SM / 带宽 / 功耗 / 温度。
    DeviceMonitor device_monitor(0);
    // 显存分层时间线：prefill / 每个 decode step 采一条，看清 KV cache 增长曲线。
    MemoryReporter mem_reporter;
    EmptyMemoryUsageProvider empty_memory_usage;
    const bool profile_memory_stages = args.profile && env_flag_enabled("LOCAL_LLM_PROFILE_MEMORY_STAGES");
    if (profile_memory_stages) {
        mem_reporter.sample(pool, empty_memory_usage, "after_warmup");
    }

    if (args.profile) {
        device_monitor.start(100);
    }

    std::unique_ptr<SessionBase> session(model->create_session(text));
    if (profile_memory_stages) {
        mem_reporter.sample(pool, model->memory_usage(*session), "session_created");
    }
    const int64_t input_size = session->h_input_i32_.numel();

    // prefill：喂入整段输入 token，跑完各层，返回首个生成 token。
    int next;
    {
        ScopedCpuTimer t("prefill");
        next = model->prefill(*session);
    }
    if (args.profile) {
        mem_reporter.sample(pool, model->memory_usage(*session), "prefill");
    }

    // decode：从输入 token 之后的位置开始逐 token 生成。
    int decode_tokens = 0;
    // 基础墙钟计时：与 profile 埋点无关，无 CUDA 同步、无落盘，恒定开销可忽略。
    const auto decode_wall_start = std::chrono::steady_clock::now();
    {
        ScopedCpuTimer t("decode_total");
        for (int step = 0; step < args.max_output_tokens; ++step) {
            if (next == eos) break;
            session->append_output(next);
            // 采样式 profile：仅每隔 profile_sample_every 步开启一次全量 GPU event
            // 计时，其余步零 event 开销（timer 短路，不插 event 进流）。稳态吞吐因此
            // 接近无 profile；--profile-sample-every 1 时等价每步全量最细。
            // decode_token 是 CPU 计时，不受 capture 影响，逐步都记以保稳态吞吐口径。
            const bool sample_this_step =
                args.profile && (step % args.profile_sample_every == 0);
            Profiler::instance().set_capture(sample_this_step);
            {
                ScopedCpuTimer td("decode_token");
                next = model->decode(*session);
            }
            ++decode_tokens;
            if (sample_this_step) {
                // 本采样步结束即结算其 GPU event（此时该步各层 kernel 已随层内
                // cudaDeviceSynchronize 全部完成），把 event 归还池中复用。
                Profiler::instance().flush_gpu_events();
                mem_reporter.sample(pool, model->memory_usage(*session), "decode");
            }
        }
    }
    // 确保所有 decode kernel 落地后再停表，得到真实稳态墙钟。
    cudaDeviceSynchronize();
    const double decode_wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - decode_wall_start).count();

    if (args.profile) {
        device_monitor.stop();
        // 低扰动 GPU 计时：上面的 cudaDeviceSynchronize 已保证所有埋点 event 完成，
        // 此处统一结算排队中的 GPU event（cudaEventElapsedTime + record），落盘前必须做。
        Profiler::instance().flush_gpu_events();
    }

    std::cout << "生成结果：" << model->output(*session) << std::endl;

    // 基础耗时：无论是否 profile 都打印（仅墙钟，无逐算子埋点/同步/落盘）。
    if (decode_tokens > 0 && decode_wall_ms > 0.0) {
        const double avg_ms = decode_wall_ms / decode_tokens;
        const double tps = decode_tokens / decode_wall_ms * 1000.0;
        std::cout << "[decode] tokens=" << decode_tokens
                  << " wall_ms=" << decode_wall_ms
                  << " avg_ms_per_token=" << avg_ms
                  << " tokens_per_sec=" << tps << std::endl;
    }

    if (!args.profile) {
        destroy_compute_stream();
        return 0;
    }

    // 报告文件名加模型 name 区分，便于多模型对比：profile_<name>.{jsonl,_summary.json,_summary.md}。
    const std::string prefix = args.profile_dir + "/profile_" + model->name();

    // === 三类报告落盘 ===
    // 1) JSONL 原始日志：各采集器的明细时间线逐行追加到同一文件。
    {
        std::ofstream jsonl(prefix + ".jsonl");
        Profiler::instance().write_jsonl(jsonl);
        weight_tracker.write_jsonl(jsonl);
        device_monitor.write_jsonl(jsonl);
        mem_reporter.write_jsonl(jsonl);
    }
    // 2) JSON summary：各采集器聚合结论逐行写入（每行一个 JSON 对象）。
    {
        std::ofstream json(prefix + "_summary.json");
        Profiler::instance().write_json_summary(json);
        weight_tracker.write_json_summary(json);
        device_monitor.write_json_summary(json);
        mem_reporter.write_json_summary(json);
    }
    // 3) Markdown summary：人类可读汇总，可直接贴进 doc / MR。
    {
        std::ofstream md(prefix + "_summary.md");
        md << "# Profile summary (" << model->name() << ")\n\n";
        Profiler::instance().write_markdown_summary(md);
        mem_reporter.write_markdown_summary(md);
        device_monitor.write_markdown_summary(md);
        weight_tracker.write_markdown_summary(md);
    }

    // 同时在 stdout 打印稳态端到端指标，便于快速查看。
    std::cout << "PROFILE model=" << model->name()
              << " input_tokens=" << input_size
              << " decode_tokens=" << decode_tokens
              << " reports=" << prefix << ".{jsonl,_summary.json,_summary.md}"
              << std::endl;
    destroy_compute_stream();
    return 0;
}
