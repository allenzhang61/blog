# debug-ds-gather-latent [CLOSED]

## Symptom

- DeepSeek trace runs emit useful prefill/decode trace, then abort with `ds.gather.latent：invalid argument`.
- The abort prevents longer generated-token trace comparison between local safe, local quant-direct, and llama.cpp.

## Reproduction

- Binary: `~/codes/local-llm-trace/build-release/local_llm` on `zyl@192.168.1.211`
- Model: `/home/zyl/models/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf`
- Example prompt: `Write a short story about a robot learning to paint.`
- Command shape:

```bash
env PROMPT="Write a short story about a robot learning to paint." \
    LOCAL_LLM_DEEPSEEK_TRACE=1 \
    ./build-release/local_llm \
      --model deepseek \
      --model-dir /home/zyl/models/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf \
      --max-output-tokens 1
```

## Hypotheses

1. `ds.gather.latent` receives an out-of-range source row index when decode starts after prefill.
2. The latent KV cache row count is smaller than the requested `pos` because cache append/update is off by one.
3. The gather index tensor dtype/shape does not match the kernel expectation for decode (`I32`, one row).
4. The trace path changes scratch reuse timing and overwrites the gather index/latent input before the kernel launches.
5. The failure is independent of trace and is a general decode bug in the current DeepSeek path.

## Evidence

- Existing runs show both safe and quant-direct trace jobs abort after emitting early decode traces.
- Temporary `LOCAL_LLM_DEBUG_GATHER_LATENT=1` instrumentation around `ds.gather.latent` showed the failing warmup decode reached `seq=14` while `cache_rows=13` when the run used `--max-output-tokens 1`.
- `DeepseekSession` allocates KV cache as `prompt_len + max_output_tokens`; the old warmup loop always attempted up to 4 decode steps, so small `max_output_tokens` runs could write/read past the allocated KV cache.
- After clamping warmup decode steps to `min(4, args.max_output_tokens)`, the same `--max-output-tokens 1` trace run finished without `ds.gather.latent：invalid argument`.
- A second issue caused no-trace quant-direct story generation to diverge while trace or `CUDA_LAUNCH_BLOCKING=1` looked correct. Root cause was the D2D latent gather using default-stream `cudaMemcpy2D`, which was not ordered with the non-blocking compute stream that writes the KV cache.
- Moving `cuda_memcpy2d_d2d` to `cudaMemcpy2DAsync(..., get_current_cuda_stream())` fixed the no-trace story smoke without adding per-layer synchronization.

## Status

- Fixed and verified on `zyl@192.168.1.211`.
- Temporary `LOCAL_LLM_DEBUG_GATHER_LATENT` instrumentation has been removed from `MLA.cpp`.
- Verification:
  - `cmake --build build-release -j 8`: passed.
  - `ctest --test-dir build-release --output-on-failure`: 7/7 passed.
  - Quant-direct story, `--max-output-tokens 16`, no trace: outputs `Once upon a time...` instead of repeated Chinese text.
  - Quant-direct story, `--max-output-tokens 128`, no trace: completes with coherent English continuation, about `21.2 tok/s`.
