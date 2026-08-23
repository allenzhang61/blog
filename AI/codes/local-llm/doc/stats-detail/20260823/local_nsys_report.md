# local-llm nsys GPU kernel 汇总（qwen3_5，同 llama.cpp 口径）

采集命令（RTX 3080）：

```
nsys profile --trace=cuda --sample=none --cuda-flush-interval=100 -o local_nsys2 \
    ./build-release/local_llm --model-dir /home/zyl/models/Qwen3.5-4B-Base --max-output-tokens 64
# QdstrmImporter 导出 .nsys-rep 后：
nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum local_nsys2.nsys-rep
```

> 说明：`--cuda-flush-interval=100` 用于规避本机 nsys 2023.4 + QdstrmImporter(CUDA12.4) 组合下的「事件乱序」导入失败。
> 口径与 `llama_nsys_report.md` 完全一致：占比为各 kernel 占 **GPU kernel 忙时总和**（约 1,012.39M ns）的比例，已把 warmup / 权重加载 / prefill / decode 全过程的 GPU kernel 都算进去。

## CUDA GPU Kernel Summary（原始，按耗时降序）

| Time (%) | Total (ns) | Instances | Avg (ns) | Kernel |
| ---: | ---: | ---: | ---: | --- |
| 44.2 | 447,339,062 | 6,528 | 68,526 | ampere_s16816gemm_bf16_64x64（GEMM，tensor core） |
| 20.8 | 210,672,772 | 4,448 | 47,364 | cutlass wmma s161616gemm_bf16_16x16（GEMM） |
| 14.8 | 149,825,015 | 1,702 | 88,029 | gemv2T_kernel_val（GEMV） |
| 6.1 | 62,082,176 | 1,632 | 38,041 | linear_attention_recurrent_kernel |
| 2.7 | 27,172,327 | 4,550 | 5,972 | rms_norm_kernel |
| 2.3 | 23,323,097 | 17,430 | 1,338 | f32_to_bf16_copy_kernel |
| 1.5 | 14,937,844 | 544 | 27,459 | full_attention_attend_kernel |
| 1.4 | 14,056,513 | 7,648 | 1,838 | cublasLt splitKreduce_kernel |
| 0.9 | 9,441,797 | 1,088 | 8,678 | gemvx::kernel（GEMV） |
| 0.8 | 8,595,280 | 3,264 | 2,633 | gemvx::kernel（GEMV） |
| 0.8 | 7,832,562 | 128 | 61,192 | cutlass wmma 16x16_128x1（GEMM） |
| 0.7 | 7,055,308 | 4,480 | 1,575 | add_kernel |
| 0.5 | 5,138,364 | 48 | 107,049 | linear_attention_recurrent_batch_kernel（prefill 批量） |
| 0.5 | 4,759,263 | 2,240 | 2,125 | silu_mul_kernel |
| 0.4 | 4,260,920 | 64 | 66,577 | ampere_s16816gemm_bf16_128x64（GEMM） |
| 0.4 | 4,098,143 | 1,632 | 2,511 | linear_attention_conv_kernel |
| 0.4 | 3,887,732 | 112 | 34,712 | ampere_s16816gemm sliced 64x5（GEMM） |
| 0.4 | 3,719,666 | 64 | 58,120 | ampere_s16816gemm sliced 64x6（GEMM） |
| 0.2 | 1,611,467 | 544 | 2,962 | full_attention_kv_kernel |
| 0.2 | 1,575,729 | 544 | 2,897 | full_attention_q_kernel |
| 0.0 | 314,621 | 32 | 9,832 | cutlass tensorop 64x64（GEMM） |
| 0.0 | 313,989 | 70 | 4,486 | embedding_lookup_kernel |
| 0.0 | 222,691 | 48 | 4,639 | linear_attention_conv_batch_kernel |
| 0.0 | 145,762 | 16 | 9,110 | full_attention_attend_batch_kernel |
| 0.0 | 52,957 | 16 | 3,310 | full_attention_q_batch_kernel |
| 0.0 | 49,729 | 16 | 3,108 | full_attention_kv_batch_kernel |

## 按功能环节聚合（占 GPU kernel 忙时 ≈ 1,012.39M ns）

| 功能环节 | 归入的 kernel | 合计(ns) | 占比 |
| --- | --- | ---: | ---: |
| 矩阵乘（GEMM/GEMV） | ampere/cutlass GEMM + gemv2T/gemvx + splitKreduce | 859.95M | 84.9% |
| linear attention 递归 | recurrent + recurrent_batch | 67.22M | 6.6% |
| 归一化 | rms_norm + f32_to_bf16_copy | 50.49M | 5.0% |
| full attention | attend/kv/q(+batch) | 18.38M | 1.8% |
| 逐元素 / 激活 | add + silu_mul | 11.82M | 1.2% |
| linear attention conv1d | conv + conv_batch | 4.32M | 0.4% |
| embedding | embedding_lookup | 0.31M | 0.03% |

## CUDA GPU MemOps Summary（by Time）

| Time (%) | Total (ns) | Count | Operation |
| ---: | ---: | ---: | --- |
| 99.3 | 863,909,967 | 496 | Host-to-Device（权重加载，最大单次 135ms） |
| 0.4 | 3,145,581 | 6,688 | memset |
| 0.3 | 2,698,256 | 70 | Device-to-Host |
</content>
</invoke>
