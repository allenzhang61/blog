# Profile summary (qwen)

## Profiler summary

- decode 吞吐：43.0689 tokens/s（64 tokens，平均 23.2186 ms/token）

| name | count | total_ms | avg_ms | pct | bandwidth(GB/s) |
|---|---:|---:|---:|---:|---:|
| decode_total | 1 | 1487.62 | 1487.62 | 35.9365% | 0 |
| decode_token | 64 | 1485.99 | 23.2186 | 35.8971% | 0 |
| mlp.gate | 2080 | 159.87 | 0.0768608 | 3.86199% | 613.914 |
| mlp.up | 2080 | 154.813 | 0.0744294 | 3.73982% | 633.969 |
| mlp.down | 2080 | 151.985 | 0.0730698 | 3.67151% | 645.765 |
| lm_head | 65 | 93.8904 | 1.44447 | 2.26811% | 880.184 |
| linattn.in_proj_qkv | 1560 | 93.4899 | 0.0599294 | 2.25844% | 699.874 |
| linattn.out_proj | 1560 | 72.0001 | 0.0461539 | 1.73931% | 454.383 |
| linear_attention_recurrent | 1536 | 65.0528 | 0.0423521 | 1.57148% | 0 |
| linattn.in_proj_z | 1560 | 57.1882 | 0.0366591 | 1.3815% | 572.069 |
| float_to_lowp | 8385 | 50.2306 | 0.00599053 | 1.21342% | 0 |
| rms_norm | 4225 | 42.2655 | 0.0100037 | 1.02101% | 0 |
| prefill | 1 | 33.0134 | 33.0134 | 0.797505% | 0 |
| fullattn.q_proj | 520 | 31.0616 | 0.0597338 | 0.750355% | 702.165 |
| add | 4160 | 25.3467 | 0.00609295 | 0.6123% | 0 |
| linattn.in_proj_b | 1560 | 23.6922 | 0.0151873 | 0.572333% | 10.788 |
| linattn.in_proj_a | 1560 | 22.7344 | 0.0145734 | 0.549196% | 11.2424 |
| fullattn.o_proj | 520 | 22.5997 | 0.043461 | 0.545942% | 482.536 |
| full_attention_attend | 512 | 16.5594 | 0.0323425 | 0.400025% | 0 |
| silu_mul | 2080 | 11.9646 | 0.00575223 | 0.28903% | 0 |
| linear_attention_conv | 1536 | 10.1999 | 0.00664056 | 0.246399% | 0 |
| fullattn.k_proj | 520 | 8.53744 | 0.0164182 | 0.206239% | 319.334 |
| fullattn.v_proj | 520 | 8.27046 | 0.0159047 | 0.19979% | 329.643 |
| full_attention_q | 512 | 3.63072 | 0.00709125 | 0.0877073% | 0 |
| full_attention_kv | 512 | 3.58832 | 0.00700844 | 0.0866831% | 0 |
| linear_attention_recurrent_batch | 24 | 2.90701 | 0.121125 | 0.0702246% | 0 |
| embedding_lookup | 65 | 0.629056 | 0.00967778 | 0.0151961% | 0 |
| linear_attention_conv_batch | 24 | 0.217024 | 0.00904267 | 0.00524265% | 0 |
| full_attention_attend_batch | 8 | 0.109664 | 0.013708 | 0.00264915% | 0 |
| full_attention_q_batch | 8 | 0.06656 | 0.00832 | 0.00160789% | 0 |
| full_attention_kv_batch | 8 | 0.060416 | 0.007552 | 0.00145947% | 0 |

## Memory summary

采样数：65（下表为时间线峰值）

| layer | peak(MiB) | pct of device |
|---|---:|---:|
| weights | 8021.84 | 33.2523% |
| kv cache / state | 55.25 | 0.229023% |
| scratch (peak) | 2.43652 | 0.0100999% |
| device used | 8479.12 | 35.1478% |
| device total | 24124.2 | 100% |

### KV cache growth

| ts_ms | label | kv(MiB) | scratch(MiB) | device used(MiB) |
|---:|---|---:|---:|---:|
| 0 | prefill | 55.25 | 2.43652 | 8479.12 |
| 23.6374 | decode | 55.25 | 2.43652 | 8479.12 |
| 46.9861 | decode | 55.25 | 2.43652 | 8479.12 |
| 69.9941 | decode | 55.25 | 2.43652 | 8479.12 |
| 93.2068 | decode | 55.25 | 2.43652 | 8479.12 |
| 116.488 | decode | 55.25 | 2.43652 | 8479.12 |
| 139.975 | decode | 55.25 | 2.43652 | 8479.12 |
| 163.275 | decode | 55.25 | 2.43652 | 8479.12 |
| 186.339 | decode | 55.25 | 2.43652 | 8479.12 |
| 209.475 | decode | 55.25 | 2.43652 | 8479.12 |
| 232.532 | decode | 55.25 | 2.43652 | 8479.12 |
| 255.907 | decode | 55.25 | 2.43652 | 8479.12 |
| 279.078 | decode | 55.25 | 2.43652 | 8479.12 |
| 302.677 | decode | 55.25 | 2.43652 | 8479.12 |
| 325.912 | decode | 55.25 | 2.43652 | 8479.12 |
| 349.103 | decode | 55.25 | 2.43652 | 8479.12 |
| 372.748 | decode | 55.25 | 2.43652 | 8479.12 |
| 396.035 | decode | 55.25 | 2.43652 | 8479.12 |
| 419.311 | decode | 55.25 | 2.43652 | 8479.12 |
| 442.547 | decode | 55.25 | 2.43652 | 8479.12 |
| 465.924 | decode | 55.25 | 2.43652 | 8479.12 |
| 489.379 | decode | 55.25 | 2.43652 | 8479.12 |
| 512.777 | decode | 55.25 | 2.43652 | 8479.12 |
| 536.29 | decode | 55.25 | 2.43652 | 8479.12 |
| 559.81 | decode | 55.25 | 2.43652 | 8479.12 |
| 583.246 | decode | 55.25 | 2.43652 | 8479.12 |
| 607.258 | decode | 55.25 | 2.43652 | 8479.12 |
| 630.574 | decode | 55.25 | 2.43652 | 8479.12 |
| 653.87 | decode | 55.25 | 2.43652 | 8479.12 |
| 677.121 | decode | 55.25 | 2.43652 | 8479.12 |
| 700.405 | decode | 55.25 | 2.43652 | 8479.12 |
| 724.45 | decode | 55.25 | 2.43652 | 8479.12 |
| 748.255 | decode | 55.25 | 2.43652 | 8479.12 |
| 771.563 | decode | 55.25 | 2.43652 | 8479.12 |
| 795.008 | decode | 55.25 | 2.43652 | 8479.12 |
| 818.128 | decode | 55.25 | 2.43652 | 8479.12 |
| 841.132 | decode | 55.25 | 2.43652 | 8479.12 |
| 864.158 | decode | 55.25 | 2.43652 | 8479.12 |
| 887.199 | decode | 55.25 | 2.43652 | 8479.12 |
| 910.213 | decode | 55.25 | 2.43652 | 8479.12 |
| 933.252 | decode | 55.25 | 2.43652 | 8479.12 |
| 956.26 | decode | 55.25 | 2.43652 | 8479.12 |
| 979.264 | decode | 55.25 | 2.43652 | 8479.12 |
| 1002.34 | decode | 55.25 | 2.43652 | 8479.12 |
| 1025.32 | decode | 55.25 | 2.43652 | 8479.12 |
| 1048.36 | decode | 55.25 | 2.43652 | 8479.12 |
| 1071.39 | decode | 55.25 | 2.43652 | 8479.12 |
| 1094.44 | decode | 55.25 | 2.43652 | 8479.12 |
| 1117.47 | decode | 55.25 | 2.43652 | 8479.12 |
| 1140.53 | decode | 55.25 | 2.43652 | 8479.12 |
| 1163.59 | decode | 55.25 | 2.43652 | 8479.12 |
| 1186.62 | decode | 55.25 | 2.43652 | 8479.12 |
| 1209.65 | decode | 55.25 | 2.43652 | 8479.12 |
| 1233.92 | decode | 55.25 | 2.43652 | 8479.12 |
| 1256.89 | decode | 55.25 | 2.43652 | 8479.12 |
| 1279.96 | decode | 55.25 | 2.43652 | 8479.12 |
| 1303.01 | decode | 55.25 | 2.43652 | 8479.12 |
| 1326.12 | decode | 55.25 | 2.43652 | 8479.12 |
| 1349.19 | decode | 55.25 | 2.43652 | 8479.12 |
| 1372.24 | decode | 55.25 | 2.43652 | 8479.12 |
| 1395.36 | decode | 55.25 | 2.43652 | 8479.12 |
| 1418.46 | decode | 55.25 | 2.43652 | 8479.12 |
| 1441.53 | decode | 55.25 | 2.43652 | 8479.12 |
| 1464.57 | decode | 55.25 | 2.43652 | 8479.12 |
| 1487.63 | decode | 55.25 | 2.43652 | 8479.12 |

## Device summary (15 samples)

| metric | avg | max |
|---|---:|---:|
| SM util (%) | 53.4 | 71 |
| mem-bw util (%) | 36.6 | 52 |
| power (W) | 188.661 | 268.726 |
| temp (C) | - | 38 |
| mem used (MiB) | - | 8930.94 |

## Weight load summary

- 上传总量：8021.84 MiB
- 驻留峰值：8021.84 MiB
- 分配总耗时（cudaMalloc）：65.8193 ms
- 拷贝总耗时（H2D）：1458.73 ms

| ts_ms | event | name | bytes(MiB) | ms | resident(MiB) |
|---:|---|---|---:|---:|---:|
| 0 | alloc | model.language_model.embed_tokens.weight | 1212.5 | 0.207051 | 1212.5 |
| 0.0004 | upload | model.language_model.embed_tokens.weight | 1212.5 | 246.963 | 1212.5 |
| 0.965492 | alloc | model.language_model.layers.0.input_layernorm.weight | 0.00488281 | 0.00279 | 1212.5 |
| 0.965792 | upload | model.language_model.layers.0.input_layernorm.weight | 0.00488281 | 0.049152 | 1212.5 |
| 15.7708 | alloc | model.language_model.layers.0.linear_attn.in_proj_qkv.weight | 40 | 0.30746 | 1252.5 |
| 15.7719 | upload | model.language_model.layers.0.linear_attn.in_proj_qkv.weight | 40 | 14.4466 | 1252.5 |
| 22.7838 | alloc | model.language_model.layers.0.linear_attn.in_proj_z.weight | 20 | 0.283061 | 1272.5 |
| 22.7841 | upload | model.language_model.layers.0.linear_attn.in_proj_z.weight | 20 | 6.70707 | 1272.5 |
| 23.0933 | alloc | model.language_model.layers.0.linear_attn.in_proj_b.weight | 0.15625 | 0.0229 | 1272.66 |
| 23.0944 | upload | model.language_model.layers.0.linear_attn.in_proj_b.weight | 0.15625 | 0.272384 | 1272.66 |
| 23.1508 | alloc | model.language_model.layers.0.linear_attn.in_proj_a.weight | 0.15625 | 0.00436 | 1272.82 |
| 23.1509 | upload | model.language_model.layers.0.linear_attn.in_proj_a.weight | 0.15625 | 0.042688 | 1272.82 |
| 23.1844 | alloc | model.language_model.layers.0.linear_attn.conv1d.weight | 0.0625 | 0.00259 | 1272.88 |
| 23.1846 | upload | model.language_model.layers.0.linear_attn.conv1d.weight | 0.0625 | 0.023104 | 1272.88 |
| 23.2092 | alloc | model.language_model.layers.0.linear_attn.A_log | 0.00012207 | 0.00899 | 1272.88 |
| 23.2092 | upload | model.language_model.layers.0.linear_attn.A_log | 0.00012207 | 0.007168 | 1272.88 |
| 23.2284 | alloc | model.language_model.layers.0.linear_attn.dt_bias | 6.10352e-05 | 0.00393 | 1272.88 |
| 23.2293 | upload | model.language_model.layers.0.linear_attn.dt_bias | 6.10352e-05 | 0.007168 | 1272.88 |
| 23.2481 | alloc | model.language_model.layers.0.linear_attn.norm.weight | 0.000488281 | 0.00331 | 1272.88 |
| 23.2481 | upload | model.language_model.layers.0.linear_attn.norm.weight | 0.000488281 | 0.007168 | 1272.88 |
| 30.4782 | alloc | model.language_model.layers.0.linear_attn.out_proj.weight | 20 | 0.251971 | 1292.88 |
| 30.4786 | upload | model.language_model.layers.0.linear_attn.out_proj.weight | 20 | 6.9632 | 1292.88 |
| 105.531 | alloc | model.language_model.layers.0.post_attention_layernorm.weight | 0.00488281 | 0.00515 | 1292.89 |
| 105.531 | upload | model.language_model.layers.0.post_attention_layernorm.weight | 0.00488281 | 0.01536 | 1292.89 |
| 112.12 | alloc | model.language_model.layers.0.mlp.gate_proj.weight | 45 | 0.31651 | 1337.89 |
| 112.12 | upload | model.language_model.layers.0.mlp.gate_proj.weight | 45 | 6.22806 | 1337.89 |
| 117.996 | alloc | model.language_model.layers.0.mlp.up_proj.weight | 45 | 0.32039 | 1382.89 |
| 117.997 | upload | model.language_model.layers.0.mlp.up_proj.weight | 45 | 5.52032 | 1382.89 |
| 123.877 | alloc | model.language_model.layers.0.mlp.down_proj.weight | 45 | 0.325011 | 1427.89 |
| 123.877 | upload | model.language_model.layers.0.mlp.down_proj.weight | 45 | 5.52278 | 1427.89 |
| 124.453 | alloc | model.language_model.layers.1.input_layernorm.weight | 0.00488281 | 0.00699 | 1427.89 |
| 124.453 | upload | model.language_model.layers.1.input_layernorm.weight | 0.00488281 | 0.013312 | 1427.89 |
| 141.137 | alloc | model.language_model.layers.1.linear_attn.in_proj_qkv.weight | 40 | 0.302601 | 1467.89 |
| 141.145 | upload | model.language_model.layers.1.linear_attn.in_proj_qkv.weight | 40 | 16.3441 | 1467.89 |
| 149.278 | alloc | model.language_model.layers.1.linear_attn.in_proj_z.weight | 20 | 0.29273 | 1487.89 |
| 149.278 | upload | model.language_model.layers.1.linear_attn.in_proj_z.weight | 20 | 7.81466 | 1487.89 |
| 149.645 | alloc | model.language_model.layers.1.linear_attn.in_proj_b.weight | 0.15625 | 0.02139 | 1488.05 |
| 149.646 | upload | model.language_model.layers.1.linear_attn.in_proj_b.weight | 0.15625 | 0.332256 | 1488.05 |
| 149.701 | alloc | model.language_model.layers.1.linear_attn.in_proj_a.weight | 0.15625 | 0.00495 | 1488.2 |
| 149.701 | upload | model.language_model.layers.1.linear_attn.in_proj_a.weight | 0.15625 | 0.040064 | 1488.2 |
| 149.735 | alloc | model.language_model.layers.1.linear_attn.conv1d.weight | 0.0625 | 0.0034 | 1488.27 |
| 149.735 | upload | model.language_model.layers.1.linear_attn.conv1d.weight | 0.0625 | 0.019008 | 1488.27 |
| 149.761 | alloc | model.language_model.layers.1.linear_attn.A_log | 0.00012207 | 0.00933 | 1488.27 |
| 149.761 | upload | model.language_model.layers.1.linear_attn.A_log | 0.00012207 | 0.007232 | 1488.27 |
| 149.781 | alloc | model.language_model.layers.1.linear_attn.dt_bias | 6.10352e-05 | 0.00531 | 1488.27 |
| 149.781 | upload | model.language_model.layers.1.linear_attn.dt_bias | 6.10352e-05 | 0.007168 | 1488.27 |
| 149.804 | alloc | model.language_model.layers.1.linear_attn.norm.weight | 0.000488281 | 0.00677 | 1488.27 |
| 149.804 | upload | model.language_model.layers.1.linear_attn.norm.weight | 0.000488281 | 0.006368 | 1488.27 |
| 158.309 | alloc | model.language_model.layers.1.linear_attn.out_proj.weight | 20 | 0.27547 | 1508.27 |
| 158.31 | upload | model.language_model.layers.1.linear_attn.out_proj.weight | 20 | 8.21507 | 1508.27 |
| 158.888 | alloc | model.language_model.layers.1.post_attention_layernorm.weight | 0.00488281 | 0.01756 | 1508.27 |
| 158.888 | upload | model.language_model.layers.1.post_attention_layernorm.weight | 0.00488281 | 0.013312 | 1508.27 |
| 164.81 | alloc | model.language_model.layers.1.mlp.gate_proj.weight | 45 | 0.291901 | 1553.27 |
| 164.81 | upload | model.language_model.layers.1.mlp.gate_proj.weight | 45 | 5.59072 | 1553.27 |
| 170.701 | alloc | model.language_model.layers.1.mlp.up_proj.weight | 45 | 0.31699 | 1598.27 |
| 170.701 | upload | model.language_model.layers.1.mlp.up_proj.weight | 45 | 5.55213 | 1598.27 |
| 176.595 | alloc | model.language_model.layers.1.mlp.down_proj.weight | 45 | 0.326501 | 1643.27 |
| 176.595 | upload | model.language_model.layers.1.mlp.down_proj.weight | 45 | 5.54467 | 1643.27 |
| 177.345 | alloc | model.language_model.layers.2.input_layernorm.weight | 0.00488281 | 0.01958 | 1643.28 |
| 177.345 | upload | model.language_model.layers.2.input_layernorm.weight | 0.00488281 | 0.314368 | 1643.28 |
| 197.748 | alloc | model.language_model.layers.2.linear_attn.in_proj_qkv.weight | 40 | 0.333041 | 1683.28 |
| 197.748 | upload | model.language_model.layers.2.linear_attn.in_proj_qkv.weight | 40 | 20.0297 | 1683.28 |
| 207.922 | alloc | model.language_model.layers.2.linear_attn.in_proj_z.weight | 20 | 0.30323 | 1703.28 |
| 207.922 | upload | model.language_model.layers.2.linear_attn.in_proj_z.weight | 20 | 9.84986 | 1703.28 |
| 208.227 | alloc | model.language_model.layers.2.linear_attn.in_proj_b.weight | 0.15625 | 0.02121 | 1703.43 |
| 208.235 | upload | model.language_model.layers.2.linear_attn.in_proj_b.weight | 0.15625 | 0.270432 | 1703.43 |
| 208.5 | alloc | model.language_model.layers.2.linear_attn.in_proj_a.weight | 0.15625 | 0.00456 | 1703.59 |
| 208.5 | upload | model.language_model.layers.2.linear_attn.in_proj_a.weight | 0.15625 | 0.250272 | 1703.59 |
| 208.544 | alloc | model.language_model.layers.2.linear_attn.conv1d.weight | 0.0625 | 0.00473 | 1703.65 |
| 208.544 | upload | model.language_model.layers.2.linear_attn.conv1d.weight | 0.0625 | 0.030656 | 1703.65 |
| 208.57 | alloc | model.language_model.layers.2.linear_attn.A_log | 0.00012207 | 0.01039 | 1703.65 |
| 208.57 | upload | model.language_model.layers.2.linear_attn.A_log | 0.00012207 | 0.007264 | 1703.65 |
| 208.592 | alloc | model.language_model.layers.2.linear_attn.dt_bias | 6.10352e-05 | 0.0065 | 1703.65 |
| 208.593 | upload | model.language_model.layers.2.linear_attn.dt_bias | 6.10352e-05 | 0.006144 | 1703.65 |
| 208.616 | alloc | model.language_model.layers.2.linear_attn.norm.weight | 0.000488281 | 0.00707 | 1703.65 |
| 208.616 | upload | model.language_model.layers.2.linear_attn.norm.weight | 0.000488281 | 0.007168 | 1703.65 |
| 219.018 | alloc | model.language_model.layers.2.linear_attn.out_proj.weight | 20 | 0.286561 | 1723.65 |
| 219.018 | upload | model.language_model.layers.2.linear_attn.out_proj.weight | 20 | 10.0983 | 1723.65 |
| 219.594 | alloc | model.language_model.layers.2.post_attention_layernorm.weight | 0.00488281 | 0.01637 | 1723.66 |
| 219.594 | upload | model.language_model.layers.2.post_attention_layernorm.weight | 0.00488281 | 0.013504 | 1723.66 |
| 225.516 | alloc | model.language_model.layers.2.mlp.gate_proj.weight | 45 | 0.302381 | 1768.66 |
| 225.516 | upload | model.language_model.layers.2.mlp.gate_proj.weight | 45 | 5.58157 | 1768.66 |
| 231.424 | alloc | model.language_model.layers.2.mlp.up_proj.weight | 45 | 0.322491 | 1813.66 |
| 231.424 | upload | model.language_model.layers.2.mlp.up_proj.weight | 45 | 5.56192 | 1813.66 |
| 237.306 | alloc | model.language_model.layers.2.mlp.down_proj.weight | 45 | 0.323401 | 1858.66 |
| 237.306 | upload | model.language_model.layers.2.mlp.down_proj.weight | 45 | 5.52717 | 1858.66 |
| 237.761 | alloc | model.language_model.layers.3.input_layernorm.weight | 0.00488281 | 0.01888 | 1858.66 |
| 237.761 | upload | model.language_model.layers.3.input_layernorm.weight | 0.00488281 | 0.024608 | 1858.66 |
| 243.038 | alloc | model.language_model.layers.3.self_attn.q_proj.weight | 40 | 0.309771 | 1898.66 |
| 243.038 | upload | model.language_model.layers.3.self_attn.q_proj.weight | 40 | 4.92925 | 1898.66 |
| 244.01 | alloc | model.language_model.layers.3.self_attn.k_proj.weight | 5 | 0.2966 | 1903.66 |
| 244.01 | upload | model.language_model.layers.3.self_attn.k_proj.weight | 5 | 0.649888 | 1903.66 |
| 244.794 | alloc | model.language_model.layers.3.self_attn.v_proj.weight | 5 | 0.12135 | 1908.66 |
| 244.794 | upload | model.language_model.layers.3.self_attn.v_proj.weight | 5 | 0.644608 | 1908.66 |
| 244.822 | alloc | model.language_model.layers.3.self_attn.q_norm.weight | 0.000488281 | 0.01038 | 1908.66 |
| 244.822 | upload | model.language_model.layers.3.self_attn.q_norm.weight | 0.000488281 | 0.009088 | 1908.66 |
| 244.84 | alloc | model.language_model.layers.3.self_attn.k_norm.weight | 0.000488281 | 0.00288 | 1908.66 |
| 244.84 | upload | model.language_model.layers.3.self_attn.k_norm.weight | 0.000488281 | 0.007168 | 1908.66 |
| 247.49 | alloc | model.language_model.layers.3.self_attn.o_proj.weight | 20 | 0.16479 | 1928.66 |
| 247.49 | upload | model.language_model.layers.3.self_attn.o_proj.weight | 20 | 2.46688 | 1928.66 |
| 258.462 | alloc | model.language_model.layers.3.post_attention_layernorm.weight | 0.00488281 | 0.0074 | 1928.67 |
| 258.462 | upload | model.language_model.layers.3.post_attention_layernorm.weight | 0.00488281 | 0.01536 | 1928.67 |
| 264.49 | alloc | model.language_model.layers.3.mlp.gate_proj.weight | 45 | 0.22476 | 1973.67 |
| 264.49 | upload | model.language_model.layers.3.mlp.gate_proj.weight | 45 | 5.75882 | 1973.67 |
| 270.392 | alloc | model.language_model.layers.3.mlp.up_proj.weight | 45 | 0.32186 | 2018.67 |
| 270.392 | upload | model.language_model.layers.3.mlp.up_proj.weight | 45 | 5.54595 | 2018.67 |
| 276.302 | alloc | model.language_model.layers.3.mlp.down_proj.weight | 45 | 0.358831 | 2063.67 |
| 276.303 | upload | model.language_model.layers.3.mlp.down_proj.weight | 45 | 5.52682 | 2063.67 |
| 276.755 | alloc | model.language_model.layers.4.input_layernorm.weight | 0.00488281 | 0.02084 | 2063.67 |
| 276.756 | upload | model.language_model.layers.4.input_layernorm.weight | 0.00488281 | 0.024736 | 2063.67 |
| 282.018 | alloc | model.language_model.layers.4.linear_attn.in_proj_qkv.weight | 40 | 0.31612 | 2103.67 |
| 282.018 | upload | model.language_model.layers.4.linear_attn.in_proj_qkv.weight | 40 | 4.90925 | 2103.67 |
| 284.816 | alloc | model.language_model.layers.4.linear_attn.in_proj_z.weight | 20 | 0.29825 | 2123.67 |
| 284.816 | upload | model.language_model.layers.4.linear_attn.in_proj_z.weight | 20 | 2.47914 | 2123.67 |
| 284.886 | alloc | model.language_model.layers.4.linear_attn.in_proj_b.weight | 0.15625 | 0.0201 | 2123.83 |
| 284.886 | upload | model.language_model.layers.4.linear_attn.in_proj_b.weight | 0.15625 | 0.03792 | 2123.83 |
| 284.936 | alloc | model.language_model.layers.4.linear_attn.in_proj_a.weight | 0.15625 | 0.00327 | 2123.98 |
| 284.936 | upload | model.language_model.layers.4.linear_attn.in_proj_a.weight | 0.15625 | 0.029856 | 2123.98 |
| 284.97 | alloc | model.language_model.layers.4.linear_attn.conv1d.weight | 0.0625 | 0.00294 | 2124.05 |
| 284.97 | upload | model.language_model.layers.4.linear_attn.conv1d.weight | 0.0625 | 0.01648 | 2124.05 |
| 284.998 | alloc | model.language_model.layers.4.linear_attn.A_log | 0.00012207 | 0.01163 | 2124.05 |
| 284.998 | upload | model.language_model.layers.4.linear_attn.A_log | 0.00012207 | 0.007232 | 2124.05 |
| 285.019 | alloc | model.language_model.layers.4.linear_attn.dt_bias | 6.10352e-05 | 0.00519 | 2124.05 |
| 285.019 | upload | model.language_model.layers.4.linear_attn.dt_bias | 6.10352e-05 | 0.007168 | 2124.05 |
| 285.042 | alloc | model.language_model.layers.4.linear_attn.norm.weight | 0.000488281 | 0.00731 | 2124.05 |
| 285.042 | upload | model.language_model.layers.4.linear_attn.norm.weight | 0.000488281 | 0.007008 | 2124.05 |
| 287.82 | alloc | model.language_model.layers.4.linear_attn.out_proj.weight | 20 | 0.299011 | 2144.05 |
| 287.821 | upload | model.language_model.layers.4.linear_attn.out_proj.weight | 20 | 2.4641 | 2144.05 |
| 288.407 | alloc | model.language_model.layers.4.post_attention_layernorm.weight | 0.00488281 | 0.01638 | 2144.05 |
| 288.418 | upload | model.language_model.layers.4.post_attention_layernorm.weight | 0.00488281 | 0.017472 | 2144.05 |
| 294.302 | alloc | model.language_model.layers.4.mlp.gate_proj.weight | 45 | 0.295831 | 2189.05 |
| 294.302 | upload | model.language_model.layers.4.mlp.gate_proj.weight | 45 | 5.54899 | 2189.05 |
| 300.253 | alloc | model.language_model.layers.4.mlp.up_proj.weight | 45 | 0.31553 | 2234.05 |
| 300.254 | upload | model.language_model.layers.4.mlp.up_proj.weight | 45 | 5.61283 | 2234.05 |
| 306.141 | alloc | model.language_model.layers.4.mlp.down_proj.weight | 45 | 0.324001 | 2279.05 |
| 306.142 | upload | model.language_model.layers.4.mlp.down_proj.weight | 45 | 5.51859 | 2279.05 |
| 306.579 | alloc | model.language_model.layers.5.input_layernorm.weight | 0.00488281 | 0.02052 | 2279.06 |
| 306.58 | upload | model.language_model.layers.5.input_layernorm.weight | 0.00488281 | 0.01536 | 2279.06 |
| 311.871 | alloc | model.language_model.layers.5.linear_attn.in_proj_qkv.weight | 40 | 0.31637 | 2319.06 |
| 311.872 | upload | model.language_model.layers.5.linear_attn.in_proj_qkv.weight | 40 | 4.93888 | 2319.06 |
| 314.699 | alloc | model.language_model.layers.5.linear_attn.in_proj_z.weight | 20 | 0.29638 | 2339.06 |
| 314.7 | upload | model.language_model.layers.5.linear_attn.in_proj_z.weight | 20 | 2.5103 | 2339.06 |
| 314.773 | alloc | model.language_model.layers.5.linear_attn.in_proj_b.weight | 0.15625 | 0.02239 | 2339.21 |
| 314.773 | upload | model.language_model.layers.5.linear_attn.in_proj_b.weight | 0.15625 | 0.037376 | 2339.21 |
| 314.82 | alloc | model.language_model.layers.5.linear_attn.in_proj_a.weight | 0.15625 | 0.00377 | 2339.37 |
| 314.82 | upload | model.language_model.layers.5.linear_attn.in_proj_a.weight | 0.15625 | 0.033856 | 2339.37 |
| 314.856 | alloc | model.language_model.layers.5.linear_attn.conv1d.weight | 0.0625 | 0.01106 | 2339.43 |
| 314.856 | upload | model.language_model.layers.5.linear_attn.conv1d.weight | 0.0625 | 0.017056 | 2339.43 |
| 314.883 | alloc | model.language_model.layers.5.linear_attn.A_log | 0.00012207 | 0.01045 | 2339.43 |
| 314.883 | upload | model.language_model.layers.5.linear_attn.A_log | 0.00012207 | 0.007264 | 2339.43 |
| 314.902 | alloc | model.language_model.layers.5.linear_attn.dt_bias | 6.10352e-05 | 0.00402 | 2339.43 |
| 314.902 | upload | model.language_model.layers.5.linear_attn.dt_bias | 6.10352e-05 | 0.007168 | 2339.43 |
| 314.926 | alloc | model.language_model.layers.5.linear_attn.norm.weight | 0.000488281 | 0.00686 | 2339.43 |
| 314.926 | upload | model.language_model.layers.5.linear_attn.norm.weight | 0.000488281 | 0.007264 | 2339.43 |
| 317.708 | alloc | model.language_model.layers.5.linear_attn.out_proj.weight | 20 | 0.285561 | 2359.43 |
| 317.709 | upload | model.language_model.layers.5.linear_attn.out_proj.weight | 20 | 2.47962 | 2359.43 |
| 318.298 | alloc | model.language_model.layers.5.post_attention_layernorm.weight | 0.00488281 | 0.01641 | 2359.44 |
| 318.299 | upload | model.language_model.layers.5.post_attention_layernorm.weight | 0.00488281 | 0.019456 | 2359.44 |
| 324.172 | alloc | model.language_model.layers.5.mlp.gate_proj.weight | 45 | 0.300891 | 2404.44 |
| 324.172 | upload | model.language_model.layers.5.mlp.gate_proj.weight | 45 | 5.53331 | 2404.44 |
| 330.106 | alloc | model.language_model.layers.5.mlp.up_proj.weight | 45 | 0.36207 | 2449.44 |
| 330.107 | upload | model.language_model.layers.5.mlp.up_proj.weight | 45 | 5.54634 | 2449.44 |
| 335.985 | alloc | model.language_model.layers.5.mlp.down_proj.weight | 45 | 0.333441 | 2494.44 |
| 335.986 | upload | model.language_model.layers.5.mlp.down_proj.weight | 45 | 5.52278 | 2494.44 |
| 336.44 | alloc | model.language_model.layers.6.input_layernorm.weight | 0.00488281 | 0.02148 | 2494.44 |
| 336.44 | upload | model.language_model.layers.6.input_layernorm.weight | 0.00488281 | 0.01536 | 2494.44 |
| 341.684 | alloc | model.language_model.layers.6.linear_attn.in_proj_qkv.weight | 40 | 0.31092 | 2534.44 |
| 341.685 | upload | model.language_model.layers.6.linear_attn.in_proj_qkv.weight | 40 | 4.89837 | 2534.44 |
| 344.474 | alloc | model.language_model.layers.6.linear_attn.in_proj_z.weight | 20 | 0.291121 | 2554.44 |
| 344.475 | upload | model.language_model.layers.6.linear_attn.in_proj_z.weight | 20 | 2.47811 | 2554.44 |
| 344.545 | alloc | model.language_model.layers.6.linear_attn.in_proj_b.weight | 0.15625 | 0.02073 | 2554.6 |
| 344.546 | upload | model.language_model.layers.6.linear_attn.in_proj_b.weight | 0.15625 | 0.037888 | 2554.6 |
| 344.59 | alloc | model.language_model.layers.6.linear_attn.in_proj_a.weight | 0.15625 | 0.0032 | 2554.76 |
| 344.59 | upload | model.language_model.layers.6.linear_attn.in_proj_a.weight | 0.15625 | 0.032128 | 2554.76 |
| 344.626 | alloc | model.language_model.layers.6.linear_attn.conv1d.weight | 0.0625 | 0.00322 | 2554.82 |
| 344.626 | upload | model.language_model.layers.6.linear_attn.conv1d.weight | 0.0625 | 0.016192 | 2554.82 |
| 344.652 | alloc | model.language_model.layers.6.linear_attn.A_log | 0.00012207 | 0.0098 | 2554.82 |
| 344.652 | upload | model.language_model.layers.6.linear_attn.A_log | 0.00012207 | 0.007168 | 2554.82 |
| 344.671 | alloc | model.language_model.layers.6.linear_attn.dt_bias | 6.10352e-05 | 0.0036 | 2554.82 |
| 344.671 | upload | model.language_model.layers.6.linear_attn.dt_bias | 6.10352e-05 | 0.007168 | 2554.82 |
| 344.693 | alloc | model.language_model.layers.6.linear_attn.norm.weight | 0.000488281 | 0.00298 | 2554.82 |
| 344.693 | upload | model.language_model.layers.6.linear_attn.norm.weight | 0.000488281 | 0.007104 | 2554.82 |
| 347.486 | alloc | model.language_model.layers.6.linear_attn.out_proj.weight | 20 | 0.290081 | 2574.82 |
| 347.486 | upload | model.language_model.layers.6.linear_attn.out_proj.weight | 20 | 2.48262 | 2574.82 |
| 348.062 | alloc | model.language_model.layers.6.post_attention_layernorm.weight | 0.00488281 | 0.01436 | 2574.82 |
| 348.062 | upload | model.language_model.layers.6.post_attention_layernorm.weight | 0.00488281 | 0.012288 | 2574.82 |
| 353.943 | alloc | model.language_model.layers.6.mlp.gate_proj.weight | 45 | 0.310341 | 2619.82 |
| 353.943 | upload | model.language_model.layers.6.mlp.gate_proj.weight | 45 | 5.53085 | 2619.82 |
| 359.838 | alloc | model.language_model.layers.6.mlp.up_proj.weight | 45 | 0.338361 | 2664.82 |
| 359.838 | upload | model.language_model.layers.6.mlp.up_proj.weight | 45 | 5.53162 | 2664.82 |
| 365.758 | alloc | model.language_model.layers.6.mlp.down_proj.weight | 45 | 0.334711 | 2709.82 |
| 365.768 | upload | model.language_model.layers.6.mlp.down_proj.weight | 45 | 5.56384 | 2709.82 |
| 366.233 | alloc | model.language_model.layers.7.input_layernorm.weight | 0.00488281 | 0.02038 | 2709.83 |
| 366.233 | upload | model.language_model.layers.7.input_layernorm.weight | 0.00488281 | 0.01536 | 2709.83 |
| 371.51 | alloc | model.language_model.layers.7.self_attn.q_proj.weight | 40 | 0.310991 | 2749.83 |
| 371.511 | upload | model.language_model.layers.7.self_attn.q_proj.weight | 40 | 4.9271 | 2749.83 |
| 372.467 | alloc | model.language_model.layers.7.self_attn.k_proj.weight | 5 | 0.278931 | 2754.83 |
| 372.467 | upload | model.language_model.layers.7.self_attn.k_proj.weight | 5 | 0.660768 | 2754.83 |
| 373.261 | alloc | model.language_model.layers.7.self_attn.v_proj.weight | 5 | 0.13088 | 2759.83 |
| 373.261 | upload | model.language_model.layers.7.self_attn.v_proj.weight | 5 | 0.651456 | 2759.83 |
| 373.297 | alloc | model.language_model.layers.7.self_attn.q_norm.weight | 0.000488281 | 0.01759 | 2759.83 |
| 373.297 | upload | model.language_model.layers.7.self_attn.q_norm.weight | 0.000488281 | 0.009024 | 2759.83 |
| 373.321 | alloc | model.language_model.layers.7.self_attn.k_norm.weight | 0.000488281 | 0.003691 | 2759.83 |
| 373.321 | upload | model.language_model.layers.7.self_attn.k_norm.weight | 0.000488281 | 0.0072 | 2759.83 |
| 375.98 | alloc | model.language_model.layers.7.self_attn.o_proj.weight | 20 | 0.1723 | 2779.83 |
| 375.98 | upload | model.language_model.layers.7.self_attn.o_proj.weight | 20 | 2.47155 | 2779.83 |
| 376.414 | alloc | model.language_model.layers.7.post_attention_layernorm.weight | 0.00488281 | 0.01767 | 2779.83 |
| 376.415 | upload | model.language_model.layers.7.post_attention_layernorm.weight | 0.00488281 | 0.01216 | 2779.83 |
| 382.293 | alloc | model.language_model.layers.7.mlp.gate_proj.weight | 45 | 0.30613 | 2824.83 |
| 382.293 | upload | model.language_model.layers.7.mlp.gate_proj.weight | 45 | 5.53331 | 2824.83 |
| 388.187 | alloc | model.language_model.layers.7.mlp.up_proj.weight | 45 | 0.336401 | 2869.83 |
| 388.187 | upload | model.language_model.layers.7.mlp.up_proj.weight | 45 | 5.53456 | 2869.83 |
| 394.12 | alloc | model.language_model.layers.7.mlp.down_proj.weight | 45 | 0.344701 | 2914.83 |
| 394.12 | upload | model.language_model.layers.7.mlp.down_proj.weight | 45 | 5.56499 | 2914.83 |
| 394.574 | alloc | model.language_model.layers.8.input_layernorm.weight | 0.00488281 | 0.03118 | 2914.84 |
| 394.574 | upload | model.language_model.layers.8.input_layernorm.weight | 0.00488281 | 0.021504 | 2914.84 |
| 399.892 | alloc | model.language_model.layers.8.linear_attn.in_proj_qkv.weight | 40 | 0.322391 | 2954.84 |
| 399.893 | upload | model.language_model.layers.8.linear_attn.in_proj_qkv.weight | 40 | 4.95946 | 2954.84 |
| 402.69 | alloc | model.language_model.layers.8.linear_attn.in_proj_z.weight | 20 | 0.278711 | 2974.84 |
| 402.691 | upload | model.language_model.layers.8.linear_attn.in_proj_z.weight | 20 | 2.48726 | 2974.84 |
| 402.762 | alloc | model.language_model.layers.8.linear_attn.in_proj_b.weight | 0.15625 | 0.02044 | 2975 |
| 402.762 | upload | model.language_model.layers.8.linear_attn.in_proj_b.weight | 0.15625 | 0.039232 | 2975 |
| 402.804 | alloc | model.language_model.layers.8.linear_attn.in_proj_a.weight | 0.15625 | 0.00345 | 2975.15 |
| 402.805 | upload | model.language_model.layers.8.linear_attn.in_proj_a.weight | 0.15625 | 0.0296 | 2975.15 |
| 402.834 | alloc | model.language_model.layers.8.linear_attn.conv1d.weight | 0.0625 | 0.00323 | 2975.21 |
| 402.834 | upload | model.language_model.layers.8.linear_attn.conv1d.weight | 0.0625 | 0.017472 | 2975.21 |
| 402.867 | alloc | model.language_model.layers.8.linear_attn.A_log | 0.00012207 | 0.01645 | 2975.21 |
| 402.867 | upload | model.language_model.layers.8.linear_attn.A_log | 0.00012207 | 0.007168 | 2975.21 |
| 402.887 | alloc | model.language_model.layers.8.linear_attn.dt_bias | 6.10352e-05 | 0.0045 | 2975.21 |
| 402.887 | upload | model.language_model.layers.8.linear_attn.dt_bias | 6.10352e-05 | 0.0072 | 2975.21 |
| 402.911 | alloc | model.language_model.layers.8.linear_attn.norm.weight | 0.000488281 | 0.00478 | 2975.21 |
| 402.911 | upload | model.language_model.layers.8.linear_attn.norm.weight | 0.000488281 | 0.006272 | 2975.21 |
| 405.717 | alloc | model.language_model.layers.8.linear_attn.out_proj.weight | 20 | 0.296651 | 2995.21 |
| 405.717 | upload | model.language_model.layers.8.linear_attn.out_proj.weight | 20 | 2.48694 | 2995.21 |
| 406.319 | alloc | model.language_model.layers.8.post_attention_layernorm.weight | 0.00488281 | 0.016771 | 2995.22 |
| 406.319 | upload | model.language_model.layers.8.post_attention_layernorm.weight | 0.00488281 | 0.020448 | 2995.22 |
| 412.199 | alloc | model.language_model.layers.8.mlp.gate_proj.weight | 45 | 0.29795 | 3040.22 |
| 412.199 | upload | model.language_model.layers.8.mlp.gate_proj.weight | 45 | 5.54467 | 3040.22 |
| 418.117 | alloc | model.language_model.layers.8.mlp.up_proj.weight | 45 | 0.345091 | 3085.22 |
| 418.117 | upload | model.language_model.layers.8.mlp.up_proj.weight | 45 | 5.54829 | 3085.22 |
| 424.023 | alloc | model.language_model.layers.8.mlp.down_proj.weight | 45 | 0.347141 | 3130.22 |
| 424.024 | upload | model.language_model.layers.8.mlp.down_proj.weight | 45 | 5.53587 | 3130.22 |
| 424.483 | alloc | model.language_model.layers.9.input_layernorm.weight | 0.00488281 | 0.02112 | 3130.22 |
| 424.483 | upload | model.language_model.layers.9.input_layernorm.weight | 0.00488281 | 0.017376 | 3130.22 |
| 429.777 | alloc | model.language_model.layers.9.linear_attn.in_proj_qkv.weight | 40 | 0.320211 | 3170.22 |
| 429.777 | upload | model.language_model.layers.9.linear_attn.in_proj_qkv.weight | 40 | 4.93206 | 3170.22 |
| 432.559 | alloc | model.language_model.layers.9.linear_attn.in_proj_z.weight | 20 | 0.283721 | 3190.22 |
| 432.559 | upload | model.language_model.layers.9.linear_attn.in_proj_z.weight | 20 | 2.47968 | 3190.22 |
| 432.64 | alloc | model.language_model.layers.9.linear_attn.in_proj_b.weight | 0.15625 | 0.01977 | 3190.38 |
| 432.64 | upload | model.language_model.layers.9.linear_attn.in_proj_b.weight | 0.15625 | 0.039104 | 3190.38 |
| 432.682 | alloc | model.language_model.layers.9.linear_attn.in_proj_a.weight | 0.15625 | 0.00366 | 3190.54 |
| 432.682 | upload | model.language_model.layers.9.linear_attn.in_proj_a.weight | 0.15625 | 0.029312 | 3190.54 |
| 432.711 | alloc | model.language_model.layers.9.linear_attn.conv1d.weight | 0.0625 | 0.00315 | 3190.6 |
| 432.711 | upload | model.language_model.layers.9.linear_attn.conv1d.weight | 0.0625 | 0.01728 | 3190.6 |
| 432.756 | alloc | model.language_model.layers.9.linear_attn.A_log | 0.00012207 | 0.00921 | 3190.6 |
| 432.756 | upload | model.language_model.layers.9.linear_attn.A_log | 0.00012207 | 0.007168 | 3190.6 |
| 432.776 | alloc | model.language_model.layers.9.linear_attn.dt_bias | 6.10352e-05 | 0.00507 | 3190.6 |
| 432.794 | upload | model.language_model.layers.9.linear_attn.dt_bias | 6.10352e-05 | 0.00704 | 3190.6 |
| 432.816 | alloc | model.language_model.layers.9.linear_attn.norm.weight | 0.000488281 | 0.00528 | 3190.6 |
| 432.816 | upload | model.language_model.layers.9.linear_attn.norm.weight | 0.000488281 | 0.008128 | 3190.6 |
| 435.642 | alloc | model.language_model.layers.9.linear_attn.out_proj.weight | 20 | 0.329321 | 3210.6 |
| 435.642 | upload | model.language_model.layers.9.linear_attn.out_proj.weight | 20 | 2.48086 | 3210.6 |
| 436.217 | alloc | model.language_model.layers.9.post_attention_layernorm.weight | 0.00488281 | 0.01644 | 3210.6 |
| 436.217 | upload | model.language_model.layers.9.post_attention_layernorm.weight | 0.00488281 | 0.013312 | 3210.6 |
| 442.098 | alloc | model.language_model.layers.9.mlp.gate_proj.weight | 45 | 0.30512 | 3255.6 |
| 442.098 | upload | model.language_model.layers.9.mlp.gate_proj.weight | 45 | 5.5385 | 3255.6 |
| 447.997 | alloc | model.language_model.layers.9.mlp.up_proj.weight | 45 | 0.341301 | 3300.6 |
| 447.997 | upload | model.language_model.layers.9.mlp.up_proj.weight | 45 | 5.53331 | 3300.6 |
| 453.897 | alloc | model.language_model.layers.9.mlp.down_proj.weight | 45 | 0.34478 | 3345.6 |
| 453.897 | upload | model.language_model.layers.9.mlp.down_proj.weight | 45 | 5.53248 | 3345.6 |
| 454.343 | alloc | model.language_model.layers.10.input_layernorm.weight | 0.00488281 | 0.02111 | 3345.61 |
| 454.343 | upload | model.language_model.layers.10.input_layernorm.weight | 0.00488281 | 0.01456 | 3345.61 |
| 474.063 | alloc | model.language_model.layers.10.linear_attn.in_proj_qkv.weight | 40 | 0.328451 | 3385.61 |
| 474.063 | upload | model.language_model.layers.10.linear_attn.in_proj_qkv.weight | 40 | 19.3492 | 3385.61 |
| 476.858 | alloc | model.language_model.layers.10.linear_attn.in_proj_z.weight | 20 | 0.297871 | 3405.61 |
| 476.858 | upload | model.language_model.layers.10.linear_attn.in_proj_z.weight | 20 | 2.4784 | 3405.61 |
| 477.509 | alloc | model.language_model.layers.10.linear_attn.in_proj_b.weight | 0.15625 | 0.268941 | 3405.77 |
| 477.509 | upload | model.language_model.layers.10.linear_attn.in_proj_b.weight | 0.15625 | 0.366432 | 3405.77 |
| 477.562 | alloc | model.language_model.layers.10.linear_attn.in_proj_a.weight | 0.15625 | 0.00535 | 3405.92 |
| 477.562 | upload | model.language_model.layers.10.linear_attn.in_proj_a.weight | 0.15625 | 0.038112 | 3405.92 |
| 477.617 | alloc | model.language_model.layers.10.linear_attn.conv1d.weight | 0.0625 | 0.01857 | 3405.98 |
| 477.617 | upload | model.language_model.layers.10.linear_attn.conv1d.weight | 0.0625 | 0.028288 | 3405.98 |
| 477.642 | alloc | model.language_model.layers.10.linear_attn.A_log | 0.00012207 | 0.00926 | 3405.98 |
| 477.642 | upload | model.language_model.layers.10.linear_attn.A_log | 0.00012207 | 0.007168 | 3405.98 |
| 477.664 | alloc | model.language_model.layers.10.linear_attn.dt_bias | 6.10352e-05 | 0.00547 | 3405.99 |
| 477.664 | upload | model.language_model.layers.10.linear_attn.dt_bias | 6.10352e-05 | 0.007168 | 3405.99 |
| 477.688 | alloc | model.language_model.layers.10.linear_attn.norm.weight | 0.000488281 | 0.00829 | 3405.99 |
| 477.688 | upload | model.language_model.layers.10.linear_attn.norm.weight | 0.000488281 | 0.007168 | 3405.99 |
| 480.373 | alloc | model.language_model.layers.10.linear_attn.out_proj.weight | 20 | 0.168511 | 3425.99 |
| 480.379 | upload | model.language_model.layers.10.linear_attn.out_proj.weight | 20 | 2.50182 | 3425.99 |
| 480.953 | alloc | model.language_model.layers.10.post_attention_layernorm.weight | 0.00488281 | 0.01597 | 3425.99 |
| 480.954 | upload | model.language_model.layers.10.post_attention_layernorm.weight | 0.00488281 | 0.01136 | 3425.99 |
| 486.826 | alloc | model.language_model.layers.10.mlp.gate_proj.weight | 45 | 0.27187 | 3470.99 |
| 486.826 | upload | model.language_model.layers.10.mlp.gate_proj.weight | 45 | 5.56154 | 3470.99 |
| 492.736 | alloc | model.language_model.layers.10.mlp.up_proj.weight | 45 | 0.340781 | 3515.99 |
| 492.736 | upload | model.language_model.layers.10.mlp.up_proj.weight | 45 | 5.54614 | 3515.99 |
| 498.655 | alloc | model.language_model.layers.10.mlp.down_proj.weight | 45 | 0.340341 | 3560.99 |
| 498.655 | upload | model.language_model.layers.10.mlp.down_proj.weight | 45 | 5.55306 | 3560.99 |
| 499.096 | alloc | model.language_model.layers.11.input_layernorm.weight | 0.00488281 | 0.02179 | 3561 |
| 499.096 | upload | model.language_model.layers.11.input_layernorm.weight | 0.00488281 | 0.01536 | 3561 |
| 504.431 | alloc | model.language_model.layers.11.self_attn.q_proj.weight | 40 | 0.362151 | 3601 |
| 504.432 | upload | model.language_model.layers.11.self_attn.q_proj.weight | 40 | 4.93507 | 3601 |
| 505.399 | alloc | model.language_model.layers.11.self_attn.k_proj.weight | 5 | 0.298201 | 3606 |
| 505.399 | upload | model.language_model.layers.11.self_attn.k_proj.weight | 5 | 0.653152 | 3606 |
| 506.179 | alloc | model.language_model.layers.11.self_attn.v_proj.weight | 5 | 0.120111 | 3611 |
| 506.179 | upload | model.language_model.layers.11.self_attn.v_proj.weight | 5 | 0.64432 | 3611 |
| 506.208 | alloc | model.language_model.layers.11.self_attn.q_norm.weight | 0.000488281 | 0.01023 | 3611 |
| 506.208 | upload | model.language_model.layers.11.self_attn.q_norm.weight | 0.000488281 | 0.009728 | 3611 |
| 506.228 | alloc | model.language_model.layers.11.self_attn.k_norm.weight | 0.000488281 | 0.00292 | 3611 |
| 506.228 | upload | model.language_model.layers.11.self_attn.k_norm.weight | 0.000488281 | 0.007968 | 3611 |
| 508.884 | alloc | model.language_model.layers.11.self_attn.o_proj.weight | 20 | 0.184311 | 3631 |
| 508.884 | upload | model.language_model.layers.11.self_attn.o_proj.weight | 20 | 2.4561 | 3631 |
| 509.329 | alloc | model.language_model.layers.11.post_attention_layernorm.weight | 0.00488281 | 0.01755 | 3631 |
| 509.329 | upload | model.language_model.layers.11.post_attention_layernorm.weight | 0.00488281 | 0.015168 | 3631 |
| 515.188 | alloc | model.language_model.layers.11.mlp.gate_proj.weight | 45 | 0.285261 | 3676 |
| 515.188 | upload | model.language_model.layers.11.mlp.gate_proj.weight | 45 | 5.53546 | 3676 |
| 521.085 | alloc | model.language_model.layers.11.mlp.up_proj.weight | 45 | 0.35037 | 3721 |
| 521.086 | upload | model.language_model.layers.11.mlp.up_proj.weight | 45 | 5.52499 | 3721 |
| 526.989 | alloc | model.language_model.layers.11.mlp.down_proj.weight | 45 | 0.345561 | 3766 |
| 526.99 | upload | model.language_model.layers.11.mlp.down_proj.weight | 45 | 5.53488 | 3766 |
| 527.435 | alloc | model.language_model.layers.12.input_layernorm.weight | 0.00488281 | 0.02087 | 3766.01 |
| 527.436 | upload | model.language_model.layers.12.input_layernorm.weight | 0.00488281 | 0.014336 | 3766.01 |
| 532.746 | alloc | model.language_model.layers.12.linear_attn.in_proj_qkv.weight | 40 | 0.32212 | 3806.01 |
| 532.747 | upload | model.language_model.layers.12.linear_attn.in_proj_qkv.weight | 40 | 4.93603 | 3806.01 |
| 535.565 | alloc | model.language_model.layers.12.linear_attn.in_proj_z.weight | 20 | 0.28093 | 3826.01 |
| 535.565 | upload | model.language_model.layers.12.linear_attn.in_proj_z.weight | 20 | 2.5185 | 3826.01 |
| 535.648 | alloc | model.language_model.layers.12.linear_attn.in_proj_b.weight | 0.15625 | 0.02225 | 3826.16 |
| 535.648 | upload | model.language_model.layers.12.linear_attn.in_proj_b.weight | 0.15625 | 0.037376 | 3826.16 |
| 535.692 | alloc | model.language_model.layers.12.linear_attn.in_proj_a.weight | 0.15625 | 0.00379 | 3826.32 |
| 535.692 | upload | model.language_model.layers.12.linear_attn.in_proj_a.weight | 0.15625 | 0.031392 | 3826.32 |
| 535.72 | alloc | model.language_model.layers.12.linear_attn.conv1d.weight | 0.0625 | 0.00349 | 3826.38 |
| 535.72 | upload | model.language_model.layers.12.linear_attn.conv1d.weight | 0.0625 | 0.016064 | 3826.38 |
| 535.749 | alloc | model.language_model.layers.12.linear_attn.A_log | 0.00012207 | 0.00968 | 3826.38 |
| 535.749 | upload | model.language_model.layers.12.linear_attn.A_log | 0.00012207 | 0.008192 | 3826.38 |
| 535.77 | alloc | model.language_model.layers.12.linear_attn.dt_bias | 6.10352e-05 | 0.00535 | 3826.38 |
| 535.77 | upload | model.language_model.layers.12.linear_attn.dt_bias | 6.10352e-05 | 0.007168 | 3826.38 |
| 535.791 | alloc | model.language_model.layers.12.linear_attn.norm.weight | 0.000488281 | 0.00525 | 3826.38 |
| 535.791 | upload | model.language_model.layers.12.linear_attn.norm.weight | 0.000488281 | 0.006272 | 3826.38 |
| 538.589 | alloc | model.language_model.layers.12.linear_attn.out_proj.weight | 20 | 0.302591 | 3846.38 |
| 538.59 | upload | model.language_model.layers.12.linear_attn.out_proj.weight | 20 | 2.47891 | 3846.38 |
| 539.176 | alloc | model.language_model.layers.12.post_attention_layernorm.weight | 0.00488281 | 0.01631 | 3846.39 |
| 539.177 | upload | model.language_model.layers.12.post_attention_layernorm.weight | 0.00488281 | 0.013152 | 3846.39 |
| 545.045 | alloc | model.language_model.layers.12.mlp.gate_proj.weight | 45 | 0.307081 | 3891.39 |
| 545.045 | upload | model.language_model.layers.12.mlp.gate_proj.weight | 45 | 5.52358 | 3891.39 |
| 550.951 | alloc | model.language_model.layers.12.mlp.up_proj.weight | 45 | 0.35353 | 3936.39 |
| 550.952 | upload | model.language_model.layers.12.mlp.up_proj.weight | 45 | 5.52845 | 3936.39 |
| 556.882 | alloc | model.language_model.layers.12.mlp.down_proj.weight | 45 | 0.368231 | 3981.39 |
| 556.882 | upload | model.language_model.layers.12.mlp.down_proj.weight | 45 | 5.52675 | 3981.39 |
| 557.343 | alloc | model.language_model.layers.13.input_layernorm.weight | 0.00488281 | 0.02095 | 3981.39 |
| 557.343 | upload | model.language_model.layers.13.input_layernorm.weight | 0.00488281 | 0.02048 | 3981.39 |
| 562.618 | alloc | model.language_model.layers.13.linear_attn.in_proj_qkv.weight | 40 | 0.31874 | 4021.39 |
| 562.618 | upload | model.language_model.layers.13.linear_attn.in_proj_qkv.weight | 40 | 4.91402 | 4021.39 |
| 565.484 | alloc | model.language_model.layers.13.linear_attn.in_proj_z.weight | 20 | 0.2844 | 4041.39 |
| 565.484 | upload | model.language_model.layers.13.linear_attn.in_proj_z.weight | 20 | 2.56221 | 4041.39 |
| 565.568 | alloc | model.language_model.layers.13.linear_attn.in_proj_b.weight | 0.15625 | 0.02293 | 4041.55 |
| 565.568 | upload | model.language_model.layers.13.linear_attn.in_proj_b.weight | 0.15625 | 0.03888 | 4041.55 |
| 565.61 | alloc | model.language_model.layers.13.linear_attn.in_proj_a.weight | 0.15625 | 0.00392 | 4041.7 |
| 565.61 | upload | model.language_model.layers.13.linear_attn.in_proj_a.weight | 0.15625 | 0.029248 | 4041.7 |
| 565.638 | alloc | model.language_model.layers.13.linear_attn.conv1d.weight | 0.0625 | 0.00337 | 4041.77 |
| 565.639 | upload | model.language_model.layers.13.linear_attn.conv1d.weight | 0.0625 | 0.016032 | 4041.77 |
| 565.66 | alloc | model.language_model.layers.13.linear_attn.A_log | 0.00012207 | 0.00549 | 4041.77 |
| 565.66 | upload | model.language_model.layers.13.linear_attn.A_log | 0.00012207 | 0.007264 | 4041.77 |
| 565.685 | alloc | model.language_model.layers.13.linear_attn.dt_bias | 6.10352e-05 | 0.0083 | 4041.77 |
| 565.685 | upload | model.language_model.layers.13.linear_attn.dt_bias | 6.10352e-05 | 0.007168 | 4041.77 |
| 565.706 | alloc | model.language_model.layers.13.linear_attn.norm.weight | 0.000488281 | 0.00469 | 4041.77 |
| 565.706 | upload | model.language_model.layers.13.linear_attn.norm.weight | 0.000488281 | 0.007168 | 4041.77 |
| 568.541 | alloc | model.language_model.layers.13.linear_attn.out_proj.weight | 20 | 0.31536 | 4061.77 |
| 568.541 | upload | model.language_model.layers.13.linear_attn.out_proj.weight | 20 | 2.50381 | 4061.77 |
| 569.152 | alloc | model.language_model.layers.13.post_attention_layernorm.weight | 0.00488281 | 0.01685 | 4061.77 |
| 569.152 | upload | model.language_model.layers.13.post_attention_layernorm.weight | 0.00488281 | 0.012288 | 4061.77 |
| 575.05 | alloc | model.language_model.layers.13.mlp.gate_proj.weight | 45 | 0.310021 | 4106.77 |
| 575.051 | upload | model.language_model.layers.13.mlp.gate_proj.weight | 45 | 5.5495 | 4106.77 |
| 580.94 | alloc | model.language_model.layers.13.mlp.up_proj.weight | 45 | 0.335811 | 4151.77 |
| 580.94 | upload | model.language_model.layers.13.mlp.up_proj.weight | 45 | 5.53091 | 4151.77 |
| 586.859 | alloc | model.language_model.layers.13.mlp.down_proj.weight | 45 | 0.335961 | 4196.77 |
| 586.859 | upload | model.language_model.layers.13.mlp.down_proj.weight | 45 | 5.54938 | 4196.77 |
| 587.311 | alloc | model.language_model.layers.14.input_layernorm.weight | 0.00488281 | 0.03379 | 4196.78 |
| 587.311 | upload | model.language_model.layers.14.input_layernorm.weight | 0.00488281 | 0.016256 | 4196.78 |
| 592.634 | alloc | model.language_model.layers.14.linear_attn.in_proj_qkv.weight | 40 | 0.339501 | 4236.78 |
| 592.634 | upload | model.language_model.layers.14.linear_attn.in_proj_qkv.weight | 40 | 4.9456 | 4236.78 |
| 595.453 | alloc | model.language_model.layers.14.linear_attn.in_proj_z.weight | 20 | 0.304331 | 4256.78 |
| 595.454 | upload | model.language_model.layers.14.linear_attn.in_proj_z.weight | 20 | 2.49302 | 4256.78 |
| 595.525 | alloc | model.language_model.layers.14.linear_attn.in_proj_b.weight | 0.15625 | 0.02135 | 4256.93 |
| 595.525 | upload | model.language_model.layers.14.linear_attn.in_proj_b.weight | 0.15625 | 0.038816 | 4256.93 |
| 595.568 | alloc | model.language_model.layers.14.linear_attn.in_proj_a.weight | 0.15625 | 0.00338 | 4257.09 |
| 595.568 | upload | model.language_model.layers.14.linear_attn.in_proj_a.weight | 0.15625 | 0.029888 | 4257.09 |
| 595.604 | alloc | model.language_model.layers.14.linear_attn.conv1d.weight | 0.0625 | 0.00336 | 4257.15 |
| 595.604 | upload | model.language_model.layers.14.linear_attn.conv1d.weight | 0.0625 | 0.016736 | 4257.15 |
| 595.626 | alloc | model.language_model.layers.14.linear_attn.A_log | 0.00012207 | 0.00543 | 4257.15 |
| 595.627 | upload | model.language_model.layers.14.linear_attn.A_log | 0.00012207 | 0.008192 | 4257.15 |
| 595.647 | alloc | model.language_model.layers.14.linear_attn.dt_bias | 6.10352e-05 | 0.00504 | 4257.15 |
| 595.647 | upload | model.language_model.layers.14.linear_attn.dt_bias | 6.10352e-05 | 0.007168 | 4257.15 |
| 595.671 | alloc | model.language_model.layers.14.linear_attn.norm.weight | 0.000488281 | 0.0082 | 4257.15 |
| 595.671 | upload | model.language_model.layers.14.linear_attn.norm.weight | 0.000488281 | 0.00624 | 4257.15 |
| 598.498 | alloc | model.language_model.layers.14.linear_attn.out_proj.weight | 20 | 0.30995 | 4277.15 |
| 598.499 | upload | model.language_model.layers.14.linear_attn.out_proj.weight | 20 | 2.50054 | 4277.15 |
| 599.072 | alloc | model.language_model.layers.14.post_attention_layernorm.weight | 0.00488281 | 0.0156 | 4277.16 |
| 599.072 | upload | model.language_model.layers.14.post_attention_layernorm.weight | 0.00488281 | 0.01216 | 4277.16 |
| 604.989 | alloc | model.language_model.layers.14.mlp.gate_proj.weight | 45 | 0.31726 | 4322.16 |
| 604.989 | upload | model.language_model.layers.14.mlp.gate_proj.weight | 45 | 5.5623 | 4322.16 |
| 610.929 | alloc | model.language_model.layers.14.mlp.up_proj.weight | 45 | 0.383001 | 4367.16 |
| 610.93 | upload | model.language_model.layers.14.mlp.up_proj.weight | 45 | 5.53248 | 4367.16 |
| 616.842 | alloc | model.language_model.layers.14.mlp.down_proj.weight | 45 | 0.3465 | 4412.16 |
| 616.85 | upload | model.language_model.layers.14.mlp.down_proj.weight | 45 | 5.53949 | 4412.16 |
| 617.295 | alloc | model.language_model.layers.15.input_layernorm.weight | 0.00488281 | 0.02181 | 4412.16 |
| 617.295 | upload | model.language_model.layers.15.input_layernorm.weight | 0.00488281 | 0.01616 | 4412.16 |
| 622.586 | alloc | model.language_model.layers.15.self_attn.q_proj.weight | 40 | 0.337631 | 4452.16 |
| 622.586 | upload | model.language_model.layers.15.self_attn.q_proj.weight | 40 | 4.9151 | 4452.16 |
| 623.566 | alloc | model.language_model.layers.15.self_attn.k_proj.weight | 5 | 0.296621 | 4457.16 |
| 623.566 | upload | model.language_model.layers.15.self_attn.k_proj.weight | 5 | 0.667008 | 4457.16 |
| 624.358 | alloc | model.language_model.layers.15.self_attn.v_proj.weight | 5 | 0.134101 | 4462.16 |
| 624.358 | upload | model.language_model.layers.15.self_attn.v_proj.weight | 5 | 0.645728 | 4462.16 |
| 624.388 | alloc | model.language_model.layers.15.self_attn.q_norm.weight | 0.000488281 | 0.01209 | 4462.16 |
| 624.388 | upload | model.language_model.layers.15.self_attn.q_norm.weight | 0.000488281 | 0.009472 | 4462.16 |
| 624.407 | alloc | model.language_model.layers.15.self_attn.k_norm.weight | 0.000488281 | 0.003 | 4462.16 |
| 624.407 | upload | model.language_model.layers.15.self_attn.k_norm.weight | 0.000488281 | 0.007168 | 4462.16 |
| 627.072 | alloc | model.language_model.layers.15.self_attn.o_proj.weight | 20 | 0.170701 | 4482.16 |
| 627.072 | upload | model.language_model.layers.15.self_attn.o_proj.weight | 20 | 2.47974 | 4482.16 |
| 627.524 | alloc | model.language_model.layers.15.post_attention_layernorm.weight | 0.00488281 | 0.02067 | 4482.17 |
| 627.524 | upload | model.language_model.layers.15.post_attention_layernorm.weight | 0.00488281 | 0.014272 | 4482.17 |
| 633.392 | alloc | model.language_model.layers.15.mlp.gate_proj.weight | 45 | 0.279641 | 4527.17 |
| 633.392 | upload | model.language_model.layers.15.mlp.gate_proj.weight | 45 | 5.54528 | 4527.17 |
| 639.329 | alloc | model.language_model.layers.15.mlp.up_proj.weight | 45 | 0.33872 | 4572.17 |
| 639.329 | upload | model.language_model.layers.15.mlp.up_proj.weight | 45 | 5.57654 | 4572.17 |
| 645.223 | alloc | model.language_model.layers.15.mlp.down_proj.weight | 45 | 0.343291 | 4617.17 |
| 645.223 | upload | model.language_model.layers.15.mlp.down_proj.weight | 45 | 5.52755 | 4617.17 |
| 645.674 | alloc | model.language_model.layers.16.input_layernorm.weight | 0.00488281 | 0.02059 | 4617.17 |
| 645.674 | upload | model.language_model.layers.16.input_layernorm.weight | 0.00488281 | 0.016384 | 4617.17 |
| 650.928 | alloc | model.language_model.layers.16.linear_attn.in_proj_qkv.weight | 40 | 0.32748 | 4657.17 |
| 650.928 | upload | model.language_model.layers.16.linear_attn.in_proj_qkv.weight | 40 | 4.88432 | 4657.17 |
| 653.721 | alloc | model.language_model.layers.16.linear_attn.in_proj_z.weight | 20 | 0.28542 | 4677.17 |
| 653.721 | upload | model.language_model.layers.16.linear_attn.in_proj_z.weight | 20 | 2.48752 | 4677.17 |
| 653.8 | alloc | model.language_model.layers.16.linear_attn.in_proj_b.weight | 0.15625 | 0.02073 | 4677.33 |
| 653.8 | upload | model.language_model.layers.16.linear_attn.in_proj_b.weight | 0.15625 | 0.038176 | 4677.33 |
| 653.846 | alloc | model.language_model.layers.16.linear_attn.in_proj_a.weight | 0.15625 | 0.00342 | 4677.49 |
| 653.846 | upload | model.language_model.layers.16.linear_attn.in_proj_a.weight | 0.15625 | 0.032896 | 4677.49 |
| 653.875 | alloc | model.language_model.layers.16.linear_attn.conv1d.weight | 0.0625 | 0.00345 | 4677.55 |
| 653.875 | upload | model.language_model.layers.16.linear_attn.conv1d.weight | 0.0625 | 0.017248 | 4677.55 |
| 653.9 | alloc | model.language_model.layers.16.linear_attn.A_log | 0.00012207 | 0.00512 | 4677.55 |
| 653.9 | upload | model.language_model.layers.16.linear_attn.A_log | 0.00012207 | 0.0072 | 4677.55 |
| 653.921 | alloc | model.language_model.layers.16.linear_attn.dt_bias | 6.10352e-05 | 0.00554 | 4677.55 |
| 653.921 | upload | model.language_model.layers.16.linear_attn.dt_bias | 6.10352e-05 | 0.007168 | 4677.55 |
| 653.943 | alloc | model.language_model.layers.16.linear_attn.norm.weight | 0.000488281 | 0.00591 | 4677.55 |
| 653.943 | upload | model.language_model.layers.16.linear_attn.norm.weight | 0.000488281 | 0.007168 | 4677.55 |
| 656.742 | alloc | model.language_model.layers.16.linear_attn.out_proj.weight | 20 | 0.309801 | 4697.55 |
| 656.742 | upload | model.language_model.layers.16.linear_attn.out_proj.weight | 20 | 2.47453 | 4697.55 |
| 657.333 | alloc | model.language_model.layers.16.post_attention_layernorm.weight | 0.00488281 | 0.01676 | 4697.55 |
| 657.334 | upload | model.language_model.layers.16.post_attention_layernorm.weight | 0.00488281 | 0.01232 | 4697.55 |
| 663.241 | alloc | model.language_model.layers.16.mlp.gate_proj.weight | 45 | 0.342721 | 4742.55 |
| 663.241 | upload | model.language_model.layers.16.mlp.gate_proj.weight | 45 | 5.52701 | 4742.55 |
| 669.159 | alloc | model.language_model.layers.16.mlp.up_proj.weight | 45 | 0.35721 | 4787.55 |
| 669.159 | upload | model.language_model.layers.16.mlp.up_proj.weight | 45 | 5.53667 | 4787.55 |
| 675.076 | alloc | model.language_model.layers.16.mlp.down_proj.weight | 45 | 0.363761 | 4832.55 |
| 675.077 | upload | model.language_model.layers.16.mlp.down_proj.weight | 45 | 5.52794 | 4832.55 |
| 675.526 | alloc | model.language_model.layers.17.input_layernorm.weight | 0.00488281 | 0.021231 | 4832.56 |
| 675.526 | upload | model.language_model.layers.17.input_layernorm.weight | 0.00488281 | 0.016384 | 4832.56 |
| 680.783 | alloc | model.language_model.layers.17.linear_attn.in_proj_qkv.weight | 40 | 0.32213 | 4872.56 |
| 680.783 | upload | model.language_model.layers.17.linear_attn.in_proj_qkv.weight | 40 | 4.89386 | 4872.56 |
| 683.574 | alloc | model.language_model.layers.17.linear_attn.in_proj_z.weight | 20 | 0.28907 | 4892.56 |
| 683.575 | upload | model.language_model.layers.17.linear_attn.in_proj_z.weight | 20 | 2.48189 | 4892.56 |
| 683.901 | alloc | model.language_model.layers.17.linear_attn.in_proj_b.weight | 0.15625 | 0.269951 | 4892.71 |
| 683.901 | upload | model.language_model.layers.17.linear_attn.in_proj_b.weight | 0.15625 | 0.041504 | 4892.71 |
| 683.947 | alloc | model.language_model.layers.17.linear_attn.in_proj_a.weight | 0.15625 | 0.00415 | 4892.87 |
| 683.947 | upload | model.language_model.layers.17.linear_attn.in_proj_a.weight | 0.15625 | 0.032096 | 4892.87 |
| 683.984 | alloc | model.language_model.layers.17.linear_attn.conv1d.weight | 0.0625 | 0.01113 | 4892.93 |
| 683.984 | upload | model.language_model.layers.17.linear_attn.conv1d.weight | 0.0625 | 0.016608 | 4892.93 |
| 684.006 | alloc | model.language_model.layers.17.linear_attn.A_log | 0.00012207 | 0.00563 | 4892.93 |
| 684.006 | upload | model.language_model.layers.17.linear_attn.A_log | 0.00012207 | 0.007168 | 4892.93 |
| 684.027 | alloc | model.language_model.layers.17.linear_attn.dt_bias | 6.10352e-05 | 0.00578 | 4892.93 |
| 684.027 | upload | model.language_model.layers.17.linear_attn.dt_bias | 6.10352e-05 | 0.007136 | 4892.93 |
| 684.051 | alloc | model.language_model.layers.17.linear_attn.norm.weight | 0.000488281 | 0.00906 | 4892.93 |
| 684.052 | upload | model.language_model.layers.17.linear_attn.norm.weight | 0.000488281 | 0.007168 | 4892.93 |
| 686.725 | alloc | model.language_model.layers.17.linear_attn.out_proj.weight | 20 | 0.19068 | 4912.93 |
| 686.725 | upload | model.language_model.layers.17.linear_attn.out_proj.weight | 20 | 2.46826 | 4912.93 |
| 687.316 | alloc | model.language_model.layers.17.post_attention_layernorm.weight | 0.00488281 | 0.01513 | 4912.94 |
| 687.316 | upload | model.language_model.layers.17.post_attention_layernorm.weight | 0.00488281 | 0.013152 | 4912.94 |
| 693.156 | alloc | model.language_model.layers.17.mlp.gate_proj.weight | 45 | 0.26724 | 4957.94 |
| 693.156 | upload | model.language_model.layers.17.mlp.gate_proj.weight | 45 | 5.53629 | 4957.94 |
| 699.148 | alloc | model.language_model.layers.17.mlp.up_proj.weight | 45 | 0.428621 | 5002.94 |
| 699.149 | upload | model.language_model.layers.17.mlp.up_proj.weight | 45 | 5.53821 | 5002.94 |
| 705.118 | alloc | model.language_model.layers.17.mlp.down_proj.weight | 45 | 0.345481 | 5047.94 |
| 705.119 | upload | model.language_model.layers.17.mlp.down_proj.weight | 45 | 5.60106 | 5047.94 |
| 705.559 | alloc | model.language_model.layers.18.input_layernorm.weight | 0.00488281 | 0.0226 | 5047.94 |
| 705.559 | upload | model.language_model.layers.18.input_layernorm.weight | 0.00488281 | 0.01536 | 5047.94 |
| 725.927 | alloc | model.language_model.layers.18.linear_attn.in_proj_qkv.weight | 40 | 0.363071 | 5087.94 |
| 725.927 | upload | model.language_model.layers.18.linear_attn.in_proj_qkv.weight | 40 | 19.9671 | 5087.94 |
| 732.809 | alloc | model.language_model.layers.18.linear_attn.in_proj_z.weight | 20 | 0.298701 | 5107.94 |
| 732.809 | upload | model.language_model.layers.18.linear_attn.in_proj_z.weight | 20 | 6.56182 | 5107.94 |
| 732.89 | alloc | model.language_model.layers.18.linear_attn.in_proj_b.weight | 0.15625 | 0.02559 | 5108.1 |
| 732.89 | upload | model.language_model.layers.18.linear_attn.in_proj_b.weight | 0.15625 | 0.040768 | 5108.1 |
| 732.943 | alloc | model.language_model.layers.18.linear_attn.in_proj_a.weight | 0.15625 | 0.01097 | 5108.26 |
| 732.943 | upload | model.language_model.layers.18.linear_attn.in_proj_a.weight | 0.15625 | 0.032384 | 5108.26 |
| 732.972 | alloc | model.language_model.layers.18.linear_attn.conv1d.weight | 0.0625 | 0.00402 | 5108.32 |
| 732.972 | upload | model.language_model.layers.18.linear_attn.conv1d.weight | 0.0625 | 0.015648 | 5108.32 |
| 732.998 | alloc | model.language_model.layers.18.linear_attn.A_log | 0.00012207 | 0.00936 | 5108.32 |
| 732.998 | upload | model.language_model.layers.18.linear_attn.A_log | 0.00012207 | 0.007168 | 5108.32 |
| 733.023 | alloc | model.language_model.layers.18.linear_attn.dt_bias | 6.10352e-05 | 0.00981 | 5108.32 |
| 733.023 | upload | model.language_model.layers.18.linear_attn.dt_bias | 6.10352e-05 | 0.007168 | 5108.32 |
| 733.046 | alloc | model.language_model.layers.18.linear_attn.norm.weight | 0.000488281 | 0.00578 | 5108.32 |
| 733.046 | upload | model.language_model.layers.18.linear_attn.norm.weight | 0.000488281 | 0.007136 | 5108.32 |
| 739.873 | alloc | model.language_model.layers.18.linear_attn.out_proj.weight | 20 | 0.32902 | 5128.32 |
| 739.873 | upload | model.language_model.layers.18.linear_attn.out_proj.weight | 20 | 6.48339 | 5128.32 |
| 740.459 | alloc | model.language_model.layers.18.post_attention_layernorm.weight | 0.00488281 | 0.01594 | 5128.32 |
| 740.459 | upload | model.language_model.layers.18.post_attention_layernorm.weight | 0.00488281 | 0.013312 | 5128.32 |
| 746.339 | alloc | model.language_model.layers.18.mlp.gate_proj.weight | 45 | 0.29721 | 5173.32 |
| 746.34 | upload | model.language_model.layers.18.mlp.gate_proj.weight | 45 | 5.54662 | 5173.32 |
| 752.246 | alloc | model.language_model.layers.18.mlp.up_proj.weight | 45 | 0.352871 | 5218.32 |
| 752.246 | upload | model.language_model.layers.18.mlp.up_proj.weight | 45 | 5.53056 | 5218.32 |
| 758.139 | alloc | model.language_model.layers.18.mlp.down_proj.weight | 45 | 0.354821 | 5263.32 |
| 758.139 | upload | model.language_model.layers.18.mlp.down_proj.weight | 45 | 5.51632 | 5263.32 |
| 758.574 | alloc | model.language_model.layers.19.input_layernorm.weight | 0.00488281 | 0.02057 | 5263.33 |
| 758.574 | upload | model.language_model.layers.19.input_layernorm.weight | 0.00488281 | 0.01536 | 5263.33 |
| 778.464 | alloc | model.language_model.layers.19.self_attn.q_proj.weight | 40 | 0.329721 | 5303.33 |
| 778.491 | upload | model.language_model.layers.19.self_attn.q_proj.weight | 40 | 19.5114 | 5303.33 |
| 781.07 | alloc | model.language_model.layers.19.self_attn.k_proj.weight | 5 | 0.289021 | 5308.33 |
| 781.07 | upload | model.language_model.layers.19.self_attn.k_proj.weight | 5 | 2.23699 | 5308.33 |
| 783.918 | alloc | model.language_model.layers.19.self_attn.v_proj.weight | 5 | 0.14037 | 5313.33 |
| 783.918 | upload | model.language_model.layers.19.self_attn.v_proj.weight | 5 | 2.69533 | 5313.33 |
| 783.951 | alloc | model.language_model.layers.19.self_attn.q_norm.weight | 0.000488281 | 0.013841 | 5313.33 |
| 783.951 | upload | model.language_model.layers.19.self_attn.q_norm.weight | 0.000488281 | 0.009216 | 5313.33 |
| 783.972 | alloc | model.language_model.layers.19.self_attn.k_norm.weight | 0.000488281 | 0.00599 | 5313.33 |
| 783.976 | upload | model.language_model.layers.19.self_attn.k_norm.weight | 0.000488281 | 0.006272 | 5313.33 |
| 791.087 | alloc | model.language_model.layers.19.self_attn.o_proj.weight | 20 | 0.18013 | 5333.33 |
| 791.087 | upload | model.language_model.layers.19.self_attn.o_proj.weight | 20 | 6.9144 | 5333.33 |
| 791.557 | alloc | model.language_model.layers.19.post_attention_layernorm.weight | 0.00488281 | 0.01785 | 5333.34 |
| 791.557 | upload | model.language_model.layers.19.post_attention_layernorm.weight | 0.00488281 | 0.013312 | 5333.34 |
| 797.451 | alloc | model.language_model.layers.19.mlp.gate_proj.weight | 45 | 0.284141 | 5378.34 |
| 797.451 | upload | model.language_model.layers.19.mlp.gate_proj.weight | 45 | 5.57251 | 5378.34 |
| 803.346 | alloc | model.language_model.layers.19.mlp.up_proj.weight | 45 | 0.336301 | 5423.34 |
| 803.346 | upload | model.language_model.layers.19.mlp.up_proj.weight | 45 | 5.53606 | 5423.34 |
| 809.279 | alloc | model.language_model.layers.19.mlp.down_proj.weight | 45 | 0.368341 | 5468.34 |
| 809.28 | upload | model.language_model.layers.19.mlp.down_proj.weight | 45 | 5.54176 | 5468.34 |
| 809.72 | alloc | model.language_model.layers.20.input_layernorm.weight | 0.00488281 | 0.0213 | 5468.34 |
| 809.72 | upload | model.language_model.layers.20.input_layernorm.weight | 0.00488281 | 0.016256 | 5468.34 |
| 830.189 | alloc | model.language_model.layers.20.linear_attn.in_proj_qkv.weight | 40 | 0.343261 | 5508.34 |
| 830.189 | upload | model.language_model.layers.20.linear_attn.in_proj_qkv.weight | 40 | 20.0878 | 5508.34 |
| 840.24 | alloc | model.language_model.layers.20.linear_attn.in_proj_z.weight | 20 | 0.303381 | 5528.34 |
| 840.24 | upload | model.language_model.layers.20.linear_attn.in_proj_z.weight | 20 | 9.71891 | 5528.34 |
| 840.628 | alloc | model.language_model.layers.20.linear_attn.in_proj_b.weight | 0.15625 | 0.03042 | 5528.5 |
| 840.628 | upload | model.language_model.layers.20.linear_attn.in_proj_b.weight | 0.15625 | 0.344672 | 5528.5 |
| 840.676 | alloc | model.language_model.layers.20.linear_attn.in_proj_a.weight | 0.15625 | 0.00525 | 5528.65 |
| 840.677 | upload | model.language_model.layers.20.linear_attn.in_proj_a.weight | 0.15625 | 0.03344 | 5528.65 |
| 840.716 | alloc | model.language_model.layers.20.linear_attn.conv1d.weight | 0.0625 | 0.00383 | 5528.72 |
| 840.716 | upload | model.language_model.layers.20.linear_attn.conv1d.weight | 0.0625 | 0.027296 | 5528.72 |
| 840.742 | alloc | model.language_model.layers.20.linear_attn.A_log | 0.00012207 | 0.00985 | 5528.72 |
| 840.742 | upload | model.language_model.layers.20.linear_attn.A_log | 0.00012207 | 0.008 | 5528.72 |
| 840.763 | alloc | model.language_model.layers.20.linear_attn.dt_bias | 6.10352e-05 | 0.00515 | 5528.72 |
| 840.763 | upload | model.language_model.layers.20.linear_attn.dt_bias | 6.10352e-05 | 0.007168 | 5528.72 |
| 840.788 | alloc | model.language_model.layers.20.linear_attn.norm.weight | 0.000488281 | 0.00546 | 5528.72 |
| 840.788 | upload | model.language_model.layers.20.linear_attn.norm.weight | 0.000488281 | 0.007168 | 5528.72 |
| 851.133 | alloc | model.language_model.layers.20.linear_attn.out_proj.weight | 20 | 0.311071 | 5548.72 |
| 851.134 | upload | model.language_model.layers.20.linear_attn.out_proj.weight | 20 | 10.0136 | 5548.72 |
| 851.714 | alloc | model.language_model.layers.20.post_attention_layernorm.weight | 0.00488281 | 0.01694 | 5548.72 |
| 851.714 | upload | model.language_model.layers.20.post_attention_layernorm.weight | 0.00488281 | 0.013312 | 5548.72 |
| 857.628 | alloc | model.language_model.layers.20.mlp.gate_proj.weight | 45 | 0.319871 | 5593.72 |
| 857.629 | upload | model.language_model.layers.20.mlp.gate_proj.weight | 45 | 5.55741 | 5593.72 |
| 863.536 | alloc | model.language_model.layers.20.mlp.up_proj.weight | 45 | 0.350311 | 5638.72 |
| 863.536 | upload | model.language_model.layers.20.mlp.up_proj.weight | 45 | 5.53334 | 5638.72 |
| 869.473 | alloc | model.language_model.layers.20.mlp.down_proj.weight | 45 | 0.351611 | 5683.72 |
| 869.473 | upload | model.language_model.layers.20.mlp.down_proj.weight | 45 | 5.56115 | 5683.72 |
| 869.922 | alloc | model.language_model.layers.21.input_layernorm.weight | 0.00488281 | 0.03056 | 5683.73 |
| 869.922 | upload | model.language_model.layers.21.input_layernorm.weight | 0.00488281 | 0.017408 | 5683.73 |
| 885.505 | alloc | model.language_model.layers.21.linear_attn.in_proj_qkv.weight | 40 | 0.348311 | 5723.73 |
| 885.506 | upload | model.language_model.layers.21.linear_attn.in_proj_qkv.weight | 40 | 15.1977 | 5723.73 |
| 893.113 | alloc | model.language_model.layers.21.linear_attn.in_proj_z.weight | 20 | 0.297 | 5743.73 |
| 893.114 | upload | model.language_model.layers.21.linear_attn.in_proj_z.weight | 20 | 7.2903 | 5743.73 |
| 893.483 | alloc | model.language_model.layers.21.linear_attn.in_proj_b.weight | 0.15625 | 0.02926 | 5743.88 |
| 893.484 | upload | model.language_model.layers.21.linear_attn.in_proj_b.weight | 0.15625 | 0.326528 | 5743.88 |
| 893.532 | alloc | model.language_model.layers.21.linear_attn.in_proj_a.weight | 0.15625 | 0.00519 | 5744.04 |
| 893.533 | upload | model.language_model.layers.21.linear_attn.in_proj_a.weight | 0.15625 | 0.033824 | 5744.04 |
| 893.574 | alloc | model.language_model.layers.21.linear_attn.conv1d.weight | 0.0625 | 0.00392 | 5744.1 |
| 893.574 | upload | model.language_model.layers.21.linear_attn.conv1d.weight | 0.0625 | 0.029696 | 5744.1 |
| 893.598 | alloc | model.language_model.layers.21.linear_attn.A_log | 0.00012207 | 0.00795 | 5744.1 |
| 893.598 | upload | model.language_model.layers.21.linear_attn.A_log | 0.00012207 | 0.007968 | 5744.1 |
| 893.619 | alloc | model.language_model.layers.21.linear_attn.dt_bias | 6.10352e-05 | 0.00532 | 5744.1 |
| 893.619 | upload | model.language_model.layers.21.linear_attn.dt_bias | 6.10352e-05 | 0.007104 | 5744.1 |
| 893.641 | alloc | model.language_model.layers.21.linear_attn.norm.weight | 0.000488281 | 0.00609 | 5744.1 |
| 893.641 | upload | model.language_model.layers.21.linear_attn.norm.weight | 0.000488281 | 0.007168 | 5744.1 |
| 901.696 | alloc | model.language_model.layers.21.linear_attn.out_proj.weight | 20 | 0.317471 | 5764.1 |
| 901.696 | upload | model.language_model.layers.21.linear_attn.out_proj.weight | 20 | 7.71936 | 5764.1 |
| 902.287 | alloc | model.language_model.layers.21.post_attention_layernorm.weight | 0.00488281 | 0.01684 | 5764.11 |
| 902.287 | upload | model.language_model.layers.21.post_attention_layernorm.weight | 0.00488281 | 0.023552 | 5764.11 |
| 908.197 | alloc | model.language_model.layers.21.mlp.gate_proj.weight | 45 | 0.319001 | 5809.11 |
| 908.197 | upload | model.language_model.layers.21.mlp.gate_proj.weight | 45 | 5.55274 | 5809.11 |
| 914.167 | alloc | model.language_model.layers.21.mlp.up_proj.weight | 45 | 0.391391 | 5854.11 |
| 914.168 | upload | model.language_model.layers.21.mlp.up_proj.weight | 45 | 5.55434 | 5854.11 |
| 920.079 | alloc | model.language_model.layers.21.mlp.down_proj.weight | 45 | 0.358491 | 5899.11 |
| 920.079 | upload | model.language_model.layers.21.mlp.down_proj.weight | 45 | 5.52797 | 5899.11 |
| 920.528 | alloc | model.language_model.layers.22.input_layernorm.weight | 0.00488281 | 0.02104 | 5899.11 |
| 920.529 | upload | model.language_model.layers.22.input_layernorm.weight | 0.00488281 | 0.016352 | 5899.11 |
| 937.178 | alloc | model.language_model.layers.22.linear_attn.in_proj_qkv.weight | 40 | 0.3351 | 5939.11 |
| 937.178 | upload | model.language_model.layers.22.linear_attn.in_proj_qkv.weight | 40 | 16.2758 | 5939.11 |
| 944.772 | alloc | model.language_model.layers.22.linear_attn.in_proj_z.weight | 20 | 0.311541 | 5959.11 |
| 944.772 | upload | model.language_model.layers.22.linear_attn.in_proj_z.weight | 20 | 7.25987 | 5959.11 |
| 945.096 | alloc | model.language_model.layers.22.linear_attn.in_proj_b.weight | 0.15625 | 0.02321 | 5959.27 |
| 945.096 | upload | model.language_model.layers.22.linear_attn.in_proj_b.weight | 0.15625 | 0.287744 | 5959.27 |
| 945.156 | alloc | model.language_model.layers.22.linear_attn.in_proj_a.weight | 0.15625 | 0.00473 | 5959.42 |
| 945.156 | upload | model.language_model.layers.22.linear_attn.in_proj_a.weight | 0.15625 | 0.044416 | 5959.42 |
| 945.19 | alloc | model.language_model.layers.22.linear_attn.conv1d.weight | 0.0625 | 0.01009 | 5959.49 |
| 945.191 | upload | model.language_model.layers.22.linear_attn.conv1d.weight | 0.0625 | 0.016256 | 5959.49 |
| 945.211 | alloc | model.language_model.layers.22.linear_attn.A_log | 0.00012207 | 0.00512 | 5959.49 |
| 945.211 | upload | model.language_model.layers.22.linear_attn.A_log | 0.00012207 | 0.008064 | 5959.49 |
| 945.233 | alloc | model.language_model.layers.22.linear_attn.dt_bias | 6.10352e-05 | 0.00598 | 5959.49 |
| 945.233 | upload | model.language_model.layers.22.linear_attn.dt_bias | 6.10352e-05 | 0.006336 | 5959.49 |
| 945.258 | alloc | model.language_model.layers.22.linear_attn.norm.weight | 0.000488281 | 0.00948 | 5959.49 |
| 945.258 | upload | model.language_model.layers.22.linear_attn.norm.weight | 0.000488281 | 0.00704 | 5959.49 |
| 952.552 | alloc | model.language_model.layers.22.linear_attn.out_proj.weight | 20 | 0.316811 | 5979.49 |
| 952.552 | upload | model.language_model.layers.22.linear_attn.out_proj.weight | 20 | 6.9617 | 5979.49 |
| 953.139 | alloc | model.language_model.layers.22.post_attention_layernorm.weight | 0.00488281 | 0.0178 | 5979.49 |
| 953.139 | upload | model.language_model.layers.22.post_attention_layernorm.weight | 0.00488281 | 0.014336 | 5979.49 |
| 959.056 | alloc | model.language_model.layers.22.mlp.gate_proj.weight | 45 | 0.332951 | 6024.49 |
| 959.057 | upload | model.language_model.layers.22.mlp.gate_proj.weight | 45 | 5.54109 | 6024.49 |
| 965.068 | alloc | model.language_model.layers.22.mlp.up_proj.weight | 45 | 0.423011 | 6069.49 |
| 965.069 | upload | model.language_model.layers.22.mlp.up_proj.weight | 45 | 5.55421 | 6069.49 |
| 970.986 | alloc | model.language_model.layers.22.mlp.down_proj.weight | 45 | 0.362371 | 6114.49 |
| 970.986 | upload | model.language_model.layers.22.mlp.down_proj.weight | 45 | 5.53158 | 6114.49 |
| 971.434 | alloc | model.language_model.layers.23.input_layernorm.weight | 0.00488281 | 0.02113 | 6114.5 |
| 971.435 | upload | model.language_model.layers.23.input_layernorm.weight | 0.00488281 | 0.015456 | 6114.5 |
| 988.948 | alloc | model.language_model.layers.23.self_attn.q_proj.weight | 40 | 0.343831 | 6154.5 |
| 988.948 | upload | model.language_model.layers.23.self_attn.q_proj.weight | 40 | 17.1302 | 6154.5 |
| 991.449 | alloc | model.language_model.layers.23.self_attn.k_proj.weight | 5 | 0.303161 | 6159.5 |
| 991.449 | upload | model.language_model.layers.23.self_attn.k_proj.weight | 5 | 2.17984 | 6159.5 |
| 994.125 | alloc | model.language_model.layers.23.self_attn.v_proj.weight | 5 | 0.1416 | 6164.5 |
| 994.125 | upload | model.language_model.layers.23.self_attn.v_proj.weight | 5 | 2.52141 | 6164.5 |
| 994.157 | alloc | model.language_model.layers.23.self_attn.q_norm.weight | 0.000488281 | 0.01335 | 6164.5 |
| 994.158 | upload | model.language_model.layers.23.self_attn.q_norm.weight | 0.000488281 | 0.00944 | 6164.5 |
| 994.18 | alloc | model.language_model.layers.23.self_attn.k_norm.weight | 0.000488281 | 0.00338 | 6164.5 |
| 994.18 | upload | model.language_model.layers.23.self_attn.k_norm.weight | 0.000488281 | 0.00704 | 6164.5 |
| 1001.28 | alloc | model.language_model.layers.23.self_attn.o_proj.weight | 20 | 0.203061 | 6184.5 |
| 1001.28 | upload | model.language_model.layers.23.self_attn.o_proj.weight | 20 | 6.86941 | 6184.5 |
| 1001.71 | alloc | model.language_model.layers.23.post_attention_layernorm.weight | 0.00488281 | 0.01691 | 6184.5 |
| 1001.71 | upload | model.language_model.layers.23.post_attention_layernorm.weight | 0.00488281 | 0.012128 | 6184.5 |
| 1007.64 | alloc | model.language_model.layers.23.mlp.gate_proj.weight | 45 | 0.303201 | 6229.5 |
| 1007.64 | upload | model.language_model.layers.23.mlp.gate_proj.weight | 45 | 5.59181 | 6229.5 |
| 1013.58 | alloc | model.language_model.layers.23.mlp.up_proj.weight | 45 | 0.367481 | 6274.5 |
| 1013.59 | upload | model.language_model.layers.23.mlp.up_proj.weight | 45 | 5.54666 | 6274.5 |
| 1019.5 | alloc | model.language_model.layers.23.mlp.down_proj.weight | 45 | 0.345691 | 6319.5 |
| 1019.5 | upload | model.language_model.layers.23.mlp.down_proj.weight | 45 | 5.5391 | 6319.5 |
| 1019.93 | alloc | model.language_model.layers.24.input_layernorm.weight | 0.00488281 | 0.0212 | 6319.51 |
| 1019.93 | upload | model.language_model.layers.24.input_layernorm.weight | 0.00488281 | 0.01536 | 6319.51 |
| 1035.63 | alloc | model.language_model.layers.24.linear_attn.in_proj_qkv.weight | 40 | 0.357821 | 6359.51 |
| 1035.63 | upload | model.language_model.layers.24.linear_attn.in_proj_qkv.weight | 40 | 15.2989 | 6359.51 |
| 1043.29 | alloc | model.language_model.layers.24.linear_attn.in_proj_z.weight | 20 | 0.30148 | 6379.51 |
| 1043.29 | upload | model.language_model.layers.24.linear_attn.in_proj_z.weight | 20 | 7.33773 | 6379.51 |
| 1043.89 | alloc | model.language_model.layers.24.linear_attn.in_proj_b.weight | 0.15625 | 0.272991 | 6379.66 |
| 1043.89 | upload | model.language_model.layers.24.linear_attn.in_proj_b.weight | 0.15625 | 0.311328 | 6379.66 |
| 1043.94 | alloc | model.language_model.layers.24.linear_attn.in_proj_a.weight | 0.15625 | 0.00522 | 6379.82 |
| 1043.94 | upload | model.language_model.layers.24.linear_attn.in_proj_a.weight | 0.15625 | 0.03664 | 6379.82 |
| 1043.99 | alloc | model.language_model.layers.24.linear_attn.conv1d.weight | 0.0625 | 0.00829 | 6379.88 |
| 1043.99 | upload | model.language_model.layers.24.linear_attn.conv1d.weight | 0.0625 | 0.030688 | 6379.88 |
| 1044.01 | alloc | model.language_model.layers.24.linear_attn.A_log | 0.00012207 | 0.00906 | 6379.88 |
| 1044.01 | upload | model.language_model.layers.24.linear_attn.A_log | 0.00012207 | 0.007168 | 6379.88 |
| 1044.04 | alloc | model.language_model.layers.24.linear_attn.dt_bias | 6.10352e-05 | 0.00588 | 6379.88 |
| 1044.04 | upload | model.language_model.layers.24.linear_attn.dt_bias | 6.10352e-05 | 0.007168 | 6379.88 |
| 1044.06 | alloc | model.language_model.layers.24.linear_attn.norm.weight | 0.000488281 | 0.00547 | 6379.88 |
| 1044.06 | upload | model.language_model.layers.24.linear_attn.norm.weight | 0.000488281 | 0.007072 | 6379.88 |
| 1051.88 | alloc | model.language_model.layers.24.linear_attn.out_proj.weight | 20 | 0.1958 | 6399.88 |
| 1051.88 | upload | model.language_model.layers.24.linear_attn.out_proj.weight | 20 | 7.60896 | 6399.88 |
| 1052.47 | alloc | model.language_model.layers.24.post_attention_layernorm.weight | 0.00488281 | 0.01828 | 6399.89 |
| 1052.47 | upload | model.language_model.layers.24.post_attention_layernorm.weight | 0.00488281 | 0.018432 | 6399.89 |
| 1058.39 | alloc | model.language_model.layers.24.mlp.gate_proj.weight | 45 | 0.294291 | 6444.89 |
| 1058.39 | upload | model.language_model.layers.24.mlp.gate_proj.weight | 45 | 5.58803 | 6444.89 |
| 1064.28 | alloc | model.language_model.layers.24.mlp.up_proj.weight | 45 | 0.349611 | 6489.89 |
| 1064.28 | upload | model.language_model.layers.24.mlp.up_proj.weight | 45 | 5.51405 | 6489.89 |
| 1070.22 | alloc | model.language_model.layers.24.mlp.down_proj.weight | 45 | 0.355841 | 6534.89 |
| 1070.22 | upload | model.language_model.layers.24.mlp.down_proj.weight | 45 | 5.55443 | 6534.89 |
| 1070.67 | alloc | model.language_model.layers.25.input_layernorm.weight | 0.00488281 | 0.02148 | 6534.89 |
| 1070.67 | upload | model.language_model.layers.25.input_layernorm.weight | 0.00488281 | 0.01536 | 6534.89 |
| 1088.23 | alloc | model.language_model.layers.25.linear_attn.in_proj_qkv.weight | 40 | 0.342441 | 6574.89 |
| 1088.23 | upload | model.language_model.layers.25.linear_attn.in_proj_qkv.weight | 40 | 17.1825 | 6574.89 |
| 1096.82 | alloc | model.language_model.layers.25.linear_attn.in_proj_z.weight | 20 | 0.307921 | 6594.89 |
| 1096.82 | upload | model.language_model.layers.25.linear_attn.in_proj_z.weight | 20 | 8.26278 | 6594.89 |
| 1097.19 | alloc | model.language_model.layers.25.linear_attn.in_proj_b.weight | 0.15625 | 0.02219 | 6595.05 |
| 1097.19 | upload | model.language_model.layers.25.linear_attn.in_proj_b.weight | 0.15625 | 0.333344 | 6595.05 |
| 1097.25 | alloc | model.language_model.layers.25.linear_attn.in_proj_a.weight | 0.15625 | 0.00595 | 6595.21 |
| 1097.25 | upload | model.language_model.layers.25.linear_attn.in_proj_a.weight | 0.15625 | 0.040064 | 6595.21 |
| 1097.29 | alloc | model.language_model.layers.25.linear_attn.conv1d.weight | 0.0625 | 0.00383 | 6595.27 |
| 1097.29 | upload | model.language_model.layers.25.linear_attn.conv1d.weight | 0.0625 | 0.028288 | 6595.27 |
| 1097.32 | alloc | model.language_model.layers.25.linear_attn.A_log | 0.00012207 | 0.00945 | 6595.27 |
| 1097.32 | upload | model.language_model.layers.25.linear_attn.A_log | 0.00012207 | 0.007968 | 6595.27 |
| 1097.34 | alloc | model.language_model.layers.25.linear_attn.dt_bias | 6.10352e-05 | 0.00584 | 6595.27 |
| 1097.34 | upload | model.language_model.layers.25.linear_attn.dt_bias | 6.10352e-05 | 0.006176 | 6595.27 |
| 1097.37 | alloc | model.language_model.layers.25.linear_attn.norm.weight | 0.000488281 | 0.0063 | 6595.27 |
| 1097.37 | upload | model.language_model.layers.25.linear_attn.norm.weight | 0.000488281 | 0.007168 | 6595.27 |
| 1106.67 | alloc | model.language_model.layers.25.linear_attn.out_proj.weight | 20 | 0.350781 | 6615.27 |
| 1106.67 | upload | model.language_model.layers.25.linear_attn.out_proj.weight | 20 | 8.92672 | 6615.27 |
| 1107.29 | alloc | model.language_model.layers.25.post_attention_layernorm.weight | 0.00488281 | 0.02084 | 6615.27 |
| 1107.29 | upload | model.language_model.layers.25.post_attention_layernorm.weight | 0.00488281 | 0.02992 | 6615.27 |
| 1113.35 | alloc | model.language_model.layers.25.mlp.gate_proj.weight | 45 | 0.338291 | 6660.27 |
| 1113.35 | upload | model.language_model.layers.25.mlp.gate_proj.weight | 45 | 5.6815 | 6660.27 |
| 1119.39 | alloc | model.language_model.layers.25.mlp.up_proj.weight | 45 | 0.36059 | 6705.27 |
| 1119.39 | upload | model.language_model.layers.25.mlp.up_proj.weight | 45 | 5.65459 | 6705.27 |
| 1125.43 | alloc | model.language_model.layers.25.mlp.down_proj.weight | 45 | 0.361271 | 6750.27 |
| 1125.44 | upload | model.language_model.layers.25.mlp.down_proj.weight | 45 | 5.65744 | 6750.27 |
| 1125.88 | alloc | model.language_model.layers.26.input_layernorm.weight | 0.00488281 | 0.02206 | 6750.28 |
| 1125.88 | upload | model.language_model.layers.26.input_layernorm.weight | 0.00488281 | 0.01536 | 6750.28 |
| 1146.64 | alloc | model.language_model.layers.26.linear_attn.in_proj_qkv.weight | 40 | 0.336211 | 6790.28 |
| 1146.64 | upload | model.language_model.layers.26.linear_attn.in_proj_qkv.weight | 40 | 20.3752 | 6790.28 |
| 1156.93 | alloc | model.language_model.layers.26.linear_attn.in_proj_z.weight | 20 | 0.314891 | 6810.28 |
| 1156.94 | upload | model.language_model.layers.26.linear_attn.in_proj_z.weight | 20 | 9.95901 | 6810.28 |
| 1157.32 | alloc | model.language_model.layers.26.linear_attn.in_proj_b.weight | 0.15625 | 0.02277 | 6810.43 |
| 1157.32 | upload | model.language_model.layers.26.linear_attn.in_proj_b.weight | 0.15625 | 0.341056 | 6810.43 |
| 1157.37 | alloc | model.language_model.layers.26.linear_attn.in_proj_a.weight | 0.15625 | 0.00599 | 6810.59 |
| 1157.37 | upload | model.language_model.layers.26.linear_attn.in_proj_a.weight | 0.15625 | 0.036864 | 6810.59 |
| 1157.41 | alloc | model.language_model.layers.26.linear_attn.conv1d.weight | 0.0625 | 0.00488 | 6810.65 |
| 1157.41 | upload | model.language_model.layers.26.linear_attn.conv1d.weight | 0.0625 | 0.028768 | 6810.65 |
| 1157.44 | alloc | model.language_model.layers.26.linear_attn.A_log | 0.00012207 | 0.01107 | 6810.65 |
| 1157.44 | upload | model.language_model.layers.26.linear_attn.A_log | 0.00012207 | 0.007168 | 6810.65 |
| 1157.47 | alloc | model.language_model.layers.26.linear_attn.dt_bias | 6.10352e-05 | 0.00593 | 6810.65 |
| 1157.47 | upload | model.language_model.layers.26.linear_attn.dt_bias | 6.10352e-05 | 0.00624 | 6810.65 |
| 1157.49 | alloc | model.language_model.layers.26.linear_attn.norm.weight | 0.000488281 | 0.00598 | 6810.65 |
| 1157.49 | upload | model.language_model.layers.26.linear_attn.norm.weight | 0.000488281 | 0.007168 | 6810.65 |
| 1168.13 | alloc | model.language_model.layers.26.linear_attn.out_proj.weight | 20 | 0.338601 | 6830.65 |
| 1168.13 | upload | model.language_model.layers.26.linear_attn.out_proj.weight | 20 | 10.2846 | 6830.65 |
| 1168.73 | alloc | model.language_model.layers.26.post_attention_layernorm.weight | 0.00488281 | 0.02064 | 6830.66 |
| 1168.73 | upload | model.language_model.layers.26.post_attention_layernorm.weight | 0.00488281 | 0.016256 | 6830.66 |
| 1174.78 | alloc | model.language_model.layers.26.mlp.gate_proj.weight | 45 | 0.337391 | 6875.66 |
| 1174.78 | upload | model.language_model.layers.26.mlp.gate_proj.weight | 45 | 5.67568 | 6875.66 |
| 1180.78 | alloc | model.language_model.layers.26.mlp.up_proj.weight | 45 | 0.35717 | 6920.66 |
| 1180.78 | upload | model.language_model.layers.26.mlp.up_proj.weight | 45 | 5.62403 | 6920.66 |
| 1186.82 | alloc | model.language_model.layers.26.mlp.down_proj.weight | 45 | 0.354371 | 6965.66 |
| 1186.82 | upload | model.language_model.layers.26.mlp.down_proj.weight | 45 | 5.65923 | 6965.66 |
| 1187.32 | alloc | model.language_model.layers.27.input_layernorm.weight | 0.00488281 | 0.03515 | 6965.66 |
| 1187.32 | upload | model.language_model.layers.27.input_layernorm.weight | 0.00488281 | 0.016384 | 6965.66 |
| 1208.28 | alloc | model.language_model.layers.27.self_attn.q_proj.weight | 40 | 0.349771 | 7005.66 |
| 1208.28 | upload | model.language_model.layers.27.self_attn.q_proj.weight | 40 | 20.5607 | 7005.66 |
| 1211.53 | alloc | model.language_model.layers.27.self_attn.k_proj.weight | 5 | 0.30576 | 7010.66 |
| 1211.53 | upload | model.language_model.layers.27.self_attn.k_proj.weight | 5 | 2.92733 | 7010.66 |
| 1214.53 | alloc | model.language_model.layers.27.self_attn.v_proj.weight | 5 | 0.142641 | 7015.66 |
| 1214.53 | upload | model.language_model.layers.27.self_attn.v_proj.weight | 5 | 2.84717 | 7015.66 |
| 1214.56 | alloc | model.language_model.layers.27.self_attn.q_norm.weight | 0.000488281 | 0.01351 | 7015.66 |
| 1214.56 | upload | model.language_model.layers.27.self_attn.q_norm.weight | 0.000488281 | 0.010432 | 7015.66 |
| 1214.58 | alloc | model.language_model.layers.27.self_attn.k_norm.weight | 0.000488281 | 0.0037 | 7015.66 |
| 1214.58 | upload | model.language_model.layers.27.self_attn.k_norm.weight | 0.000488281 | 0.007168 | 7015.66 |
| 1225.02 | alloc | model.language_model.layers.27.self_attn.o_proj.weight | 20 | 0.193931 | 7035.66 |
| 1225.02 | upload | model.language_model.layers.27.self_attn.o_proj.weight | 20 | 10.2217 | 7035.66 |
| 1225.47 | alloc | model.language_model.layers.27.post_attention_layernorm.weight | 0.00488281 | 0.01905 | 7035.67 |
| 1225.47 | upload | model.language_model.layers.27.post_attention_layernorm.weight | 0.00488281 | 0.016384 | 7035.67 |
| 1231.47 | alloc | model.language_model.layers.27.mlp.gate_proj.weight | 45 | 0.305661 | 7080.67 |
| 1231.47 | upload | model.language_model.layers.27.mlp.gate_proj.weight | 45 | 5.65376 | 7080.67 |
| 1237.5 | alloc | model.language_model.layers.27.mlp.up_proj.weight | 45 | 0.35491 | 7125.67 |
| 1237.5 | upload | model.language_model.layers.27.mlp.up_proj.weight | 45 | 5.65363 | 7125.67 |
| 1243.49 | alloc | model.language_model.layers.27.mlp.down_proj.weight | 45 | 0.346591 | 7170.67 |
| 1243.49 | upload | model.language_model.layers.27.mlp.down_proj.weight | 45 | 5.61968 | 7170.67 |
| 1243.94 | alloc | model.language_model.layers.28.input_layernorm.weight | 0.00488281 | 0.02246 | 7170.67 |
| 1243.94 | upload | model.language_model.layers.28.input_layernorm.weight | 0.00488281 | 0.01536 | 7170.67 |
| 1261.86 | alloc | model.language_model.layers.28.linear_attn.in_proj_qkv.weight | 40 | 0.362481 | 7210.67 |
| 1261.86 | upload | model.language_model.layers.28.linear_attn.in_proj_qkv.weight | 40 | 17.5126 | 7210.67 |
| 1270.63 | alloc | model.language_model.layers.28.linear_attn.in_proj_z.weight | 20 | 0.30273 | 7230.67 |
| 1270.64 | upload | model.language_model.layers.28.linear_attn.in_proj_z.weight | 20 | 8.44685 | 7230.67 |
| 1271 | alloc | model.language_model.layers.28.linear_attn.in_proj_b.weight | 0.15625 | 0.02308 | 7230.83 |
| 1271 | upload | model.language_model.layers.28.linear_attn.in_proj_b.weight | 0.15625 | 0.32144 | 7230.83 |
| 1271.06 | alloc | model.language_model.layers.28.linear_attn.in_proj_a.weight | 0.15625 | 0.00599 | 7230.99 |
| 1271.06 | upload | model.language_model.layers.28.linear_attn.in_proj_a.weight | 0.15625 | 0.038048 | 7230.99 |
| 1271.1 | alloc | model.language_model.layers.28.linear_attn.conv1d.weight | 0.0625 | 0.00361 | 7231.05 |
| 1271.1 | upload | model.language_model.layers.28.linear_attn.conv1d.weight | 0.0625 | 0.02832 | 7231.05 |
| 1271.12 | alloc | model.language_model.layers.28.linear_attn.A_log | 0.00012207 | 0.00964 | 7231.05 |
| 1271.12 | upload | model.language_model.layers.28.linear_attn.A_log | 0.00012207 | 0.007168 | 7231.05 |
| 1271.15 | alloc | model.language_model.layers.28.linear_attn.dt_bias | 6.10352e-05 | 0.00631 | 7231.05 |
| 1271.15 | upload | model.language_model.layers.28.linear_attn.dt_bias | 6.10352e-05 | 0.006336 | 7231.05 |
| 1271.17 | alloc | model.language_model.layers.28.linear_attn.norm.weight | 0.000488281 | 0.00582 | 7231.05 |
| 1271.17 | upload | model.language_model.layers.28.linear_attn.norm.weight | 0.000488281 | 0.007168 | 7231.05 |
| 1280.34 | alloc | model.language_model.layers.28.linear_attn.out_proj.weight | 20 | 0.347901 | 7251.05 |
| 1280.34 | upload | model.language_model.layers.28.linear_attn.out_proj.weight | 20 | 8.80944 | 7251.05 |
| 1280.94 | alloc | model.language_model.layers.28.post_attention_layernorm.weight | 0.00488281 | 0.02458 | 7251.05 |
| 1280.94 | upload | model.language_model.layers.28.post_attention_layernorm.weight | 0.00488281 | 0.013312 | 7251.05 |
| 1286.94 | alloc | model.language_model.layers.28.mlp.gate_proj.weight | 45 | 0.333501 | 7296.05 |
| 1286.94 | upload | model.language_model.layers.28.mlp.gate_proj.weight | 45 | 5.62864 | 7296.05 |
| 1292.98 | alloc | model.language_model.layers.28.mlp.up_proj.weight | 45 | 0.374221 | 7341.05 |
| 1292.98 | upload | model.language_model.layers.28.mlp.up_proj.weight | 45 | 5.63738 | 7341.05 |
| 1299.02 | alloc | model.language_model.layers.28.mlp.down_proj.weight | 45 | 0.403231 | 7386.05 |
| 1299.02 | upload | model.language_model.layers.28.mlp.down_proj.weight | 45 | 5.61126 | 7386.05 |
| 1299.47 | alloc | model.language_model.layers.29.input_layernorm.weight | 0.00488281 | 0.02179 | 7386.06 |
| 1299.47 | upload | model.language_model.layers.29.input_layernorm.weight | 0.00488281 | 0.015584 | 7386.06 |
| 1313.48 | alloc | model.language_model.layers.29.linear_attn.in_proj_qkv.weight | 40 | 0.352121 | 7426.06 |
| 1313.48 | upload | model.language_model.layers.29.linear_attn.in_proj_qkv.weight | 40 | 13.6234 | 7426.06 |
| 1318.81 | alloc | model.language_model.layers.29.linear_attn.in_proj_z.weight | 20 | 0.299721 | 7446.06 |
| 1318.81 | upload | model.language_model.layers.29.linear_attn.in_proj_z.weight | 20 | 5.00282 | 7446.06 |
| 1319.12 | alloc | model.language_model.layers.29.linear_attn.in_proj_b.weight | 0.15625 | 0.027321 | 7446.22 |
| 1319.12 | upload | model.language_model.layers.29.linear_attn.in_proj_b.weight | 0.15625 | 0.276064 | 7446.22 |
| 1319.18 | alloc | model.language_model.layers.29.linear_attn.in_proj_a.weight | 0.15625 | 0.0053 | 7446.37 |
| 1319.18 | upload | model.language_model.layers.29.linear_attn.in_proj_a.weight | 0.15625 | 0.03696 | 7446.37 |
| 1319.22 | alloc | model.language_model.layers.29.linear_attn.conv1d.weight | 0.0625 | 0.00386 | 7446.43 |
| 1319.22 | upload | model.language_model.layers.29.linear_attn.conv1d.weight | 0.0625 | 0.0288 | 7446.43 |
| 1319.24 | alloc | model.language_model.layers.29.linear_attn.A_log | 0.00012207 | 0.00553 | 7446.43 |
| 1319.24 | upload | model.language_model.layers.29.linear_attn.A_log | 0.00012207 | 0.007168 | 7446.43 |
| 1319.27 | alloc | model.language_model.layers.29.linear_attn.dt_bias | 6.10352e-05 | 0.00628 | 7446.43 |
| 1319.27 | upload | model.language_model.layers.29.linear_attn.dt_bias | 6.10352e-05 | 0.007168 | 7446.43 |
| 1319.29 | alloc | model.language_model.layers.29.linear_attn.norm.weight | 0.000488281 | 0.00695 | 7446.44 |
| 1319.29 | upload | model.language_model.layers.29.linear_attn.norm.weight | 0.000488281 | 0.007168 | 7446.44 |
| 1322.19 | alloc | model.language_model.layers.29.linear_attn.out_proj.weight | 20 | 0.33008 | 7466.44 |
| 1322.19 | upload | model.language_model.layers.29.linear_attn.out_proj.weight | 20 | 2.55002 | 7466.44 |
| 1322.78 | alloc | model.language_model.layers.29.post_attention_layernorm.weight | 0.00488281 | 0.01657 | 7466.44 |
| 1322.79 | upload | model.language_model.layers.29.post_attention_layernorm.weight | 0.00488281 | 0.013184 | 7466.44 |
| 1328.73 | alloc | model.language_model.layers.29.mlp.gate_proj.weight | 45 | 0.32002 | 7511.44 |
| 1328.73 | upload | model.language_model.layers.29.mlp.gate_proj.weight | 45 | 5.59078 | 7511.44 |
| 1334.71 | alloc | model.language_model.layers.29.mlp.up_proj.weight | 45 | 0.363501 | 7556.44 |
| 1334.71 | upload | model.language_model.layers.29.mlp.up_proj.weight | 45 | 5.59542 | 7556.44 |
| 1340.72 | alloc | model.language_model.layers.29.mlp.down_proj.weight | 45 | 0.359271 | 7601.44 |
| 1340.72 | upload | model.language_model.layers.29.mlp.down_proj.weight | 45 | 5.62 | 7601.44 |
| 1341.18 | alloc | model.language_model.layers.30.input_layernorm.weight | 0.00488281 | 0.0218 | 7601.45 |
| 1341.18 | upload | model.language_model.layers.30.input_layernorm.weight | 0.00488281 | 0.027648 | 7601.45 |
| 1346.58 | alloc | model.language_model.layers.30.linear_attn.in_proj_qkv.weight | 40 | 0.356821 | 7641.45 |
| 1346.58 | upload | model.language_model.layers.30.linear_attn.in_proj_qkv.weight | 40 | 5.00528 | 7641.45 |
| 1349.42 | alloc | model.language_model.layers.30.linear_attn.in_proj_z.weight | 20 | 0.29192 | 7661.45 |
| 1349.42 | upload | model.language_model.layers.30.linear_attn.in_proj_z.weight | 20 | 2.51501 | 7661.45 |
| 1349.5 | alloc | model.language_model.layers.30.linear_attn.in_proj_b.weight | 0.15625 | 0.02312 | 7661.6 |
| 1349.5 | upload | model.language_model.layers.30.linear_attn.in_proj_b.weight | 0.15625 | 0.038688 | 7661.6 |
| 1349.8 | alloc | model.language_model.layers.30.linear_attn.in_proj_a.weight | 0.15625 | 0.258851 | 7661.76 |
| 1349.8 | upload | model.language_model.layers.30.linear_attn.in_proj_a.weight | 0.15625 | 0.033024 | 7661.76 |
| 1349.83 | alloc | model.language_model.layers.30.linear_attn.conv1d.weight | 0.0625 | 0.0044 | 7661.82 |
| 1349.83 | upload | model.language_model.layers.30.linear_attn.conv1d.weight | 0.0625 | 0.016768 | 7661.82 |
| 1349.85 | alloc | model.language_model.layers.30.linear_attn.A_log | 0.00012207 | 0.00583 | 7661.82 |
| 1349.85 | upload | model.language_model.layers.30.linear_attn.A_log | 0.00012207 | 0.007168 | 7661.82 |
| 1349.88 | alloc | model.language_model.layers.30.linear_attn.dt_bias | 6.10352e-05 | 0.00956 | 7661.82 |
| 1349.88 | upload | model.language_model.layers.30.linear_attn.dt_bias | 6.10352e-05 | 0.007168 | 7661.82 |
| 1349.9 | alloc | model.language_model.layers.30.linear_attn.norm.weight | 0.000488281 | 0.00647 | 7661.82 |
| 1349.9 | upload | model.language_model.layers.30.linear_attn.norm.weight | 0.000488281 | 0.007104 | 7661.82 |
| 1352.64 | alloc | model.language_model.layers.30.linear_attn.out_proj.weight | 20 | 0.195091 | 7681.82 |
| 1352.64 | upload | model.language_model.layers.30.linear_attn.out_proj.weight | 20 | 2.52522 | 7681.82 |
| 1353.22 | alloc | model.language_model.layers.30.post_attention_layernorm.weight | 0.00488281 | 0.02251 | 7681.83 |
| 1353.22 | upload | model.language_model.layers.30.post_attention_layernorm.weight | 0.00488281 | 0.01536 | 7681.83 |
| 1359.19 | alloc | model.language_model.layers.30.mlp.gate_proj.weight | 45 | 0.297291 | 7726.83 |
| 1359.19 | upload | model.language_model.layers.30.mlp.gate_proj.weight | 45 | 5.6312 | 7726.83 |
| 1365.21 | alloc | model.language_model.layers.30.mlp.up_proj.weight | 45 | 0.394341 | 7771.83 |
| 1365.21 | upload | model.language_model.layers.30.mlp.up_proj.weight | 45 | 5.59341 | 7771.83 |
| 1371.21 | alloc | model.language_model.layers.30.mlp.down_proj.weight | 45 | 0.376321 | 7816.83 |
| 1371.21 | upload | model.language_model.layers.30.mlp.down_proj.weight | 45 | 5.60742 | 7816.83 |
| 1371.67 | alloc | model.language_model.layers.31.input_layernorm.weight | 0.00488281 | 0.02175 | 7816.83 |
| 1371.67 | upload | model.language_model.layers.31.input_layernorm.weight | 0.00488281 | 0.015424 | 7816.83 |
| 1377.08 | alloc | model.language_model.layers.31.self_attn.q_proj.weight | 40 | 0.357281 | 7856.83 |
| 1377.08 | upload | model.language_model.layers.31.self_attn.q_proj.weight | 40 | 5.01613 | 7856.83 |
| 1378.08 | alloc | model.language_model.layers.31.self_attn.k_proj.weight | 5 | 0.316691 | 7861.83 |
| 1378.08 | upload | model.language_model.layers.31.self_attn.k_proj.weight | 5 | 0.663904 | 7861.83 |
| 1378.88 | alloc | model.language_model.layers.31.self_attn.v_proj.weight | 5 | 0.13064 | 7866.83 |
| 1378.88 | upload | model.language_model.layers.31.self_attn.v_proj.weight | 5 | 0.656896 | 7866.83 |
| 1378.91 | alloc | model.language_model.layers.31.self_attn.q_norm.weight | 0.000488281 | 0.01523 | 7866.83 |
| 1378.91 | upload | model.language_model.layers.31.self_attn.q_norm.weight | 0.000488281 | 0.008672 | 7866.83 |
| 1378.93 | alloc | model.language_model.layers.31.self_attn.k_norm.weight | 0.000488281 | 0.00361 | 7866.83 |
| 1378.93 | upload | model.language_model.layers.31.self_attn.k_norm.weight | 0.000488281 | 0.007168 | 7866.83 |
| 1381.68 | alloc | model.language_model.layers.31.self_attn.o_proj.weight | 20 | 0.200401 | 7886.83 |
| 1381.68 | upload | model.language_model.layers.31.self_attn.o_proj.weight | 20 | 2.53056 | 7886.83 |
| 1382.13 | alloc | model.language_model.layers.31.post_attention_layernorm.weight | 0.00488281 | 0.01704 | 7886.84 |
| 1382.13 | upload | model.language_model.layers.31.post_attention_layernorm.weight | 0.00488281 | 0.017408 | 7886.84 |
| 1388.14 | alloc | model.language_model.layers.31.mlp.gate_proj.weight | 45 | 0.30106 | 7931.84 |
| 1388.14 | upload | model.language_model.layers.31.mlp.gate_proj.weight | 45 | 5.66899 | 7931.84 |
| 1394.17 | alloc | model.language_model.layers.31.mlp.up_proj.weight | 45 | 0.369371 | 7976.84 |
| 1394.17 | upload | model.language_model.layers.31.mlp.up_proj.weight | 45 | 5.63398 | 7976.84 |
| 1400.2 | alloc | model.language_model.layers.31.mlp.down_proj.weight | 45 | 0.366891 | 8021.84 |
| 1400.2 | upload | model.language_model.layers.31.mlp.down_proj.weight | 45 | 5.63728 | 8021.84 |
| 1400.64 | alloc | model.language_model.norm.weight | 0.00488281 | 0.02144 | 8021.84 |
| 1400.64 | upload | model.language_model.norm.weight | 0.00488281 | 0.01536 | 8021.84 |

