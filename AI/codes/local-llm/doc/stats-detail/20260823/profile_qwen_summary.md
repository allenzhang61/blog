# Profile summary (qwen)

## Profiler summary

- decode 吞吐：47.3695 tokens/s（64 tokens，平均 21.1106 ms/token）

| name | count | total_ms | avg_ms | pct | bandwidth(GB/s) |
|---|---:|---:|---:|---:|---:|
| decode_total | 1 | 1352.04 | 1352.04 | 35.4794% | 0 |
| decode_token | 64 | 1351.08 | 21.1106 | 35.4541% | 0 |
| mlp.gate | 2080 | 161.82 | 0.077798 | 4.24636% | 606.519 |
| mlp.up | 2080 | 160.442 | 0.0771357 | 4.21022% | 611.726 |
| mlp.down | 2080 | 149.815 | 0.0720266 | 3.93135% | 655.118 |
| linattn.d_in_proj_qkv | 1560 | 104.007 | 0.0666711 | 2.72928% | 629.104 |
| lm_head | 65 | 95.9948 | 1.47684 | 2.51903% | 860.889 |
| linattn.d_out_proj | 1560 | 63.5779 | 0.0407551 | 1.66837% | 514.575 |
| linear_attention_recurrent | 1536 | 61.441 | 0.0400006 | 1.61229% | 0 |
| f32_to_bf16_copy | 16185 | 52.554 | 0.00324708 | 1.37909% | 0 |
| linattn.d_in_proj_z | 1560 | 50.9925 | 0.0326875 | 1.33811% | 641.576 |
| fullattn.d_q_proj | 520 | 34.6658 | 0.066665 | 0.909677% | 629.161 |
| rms_norm | 4225 | 33.3862 | 0.00790205 | 0.876097% | 0 |
| prefill | 1 | 26.2986 | 26.2986 | 0.69011% | 0 |
| fullattn.d_o_proj | 520 | 19.6604 | 0.0378084 | 0.515914% | 554.678 |
| full_attention_attend | 512 | 15.4334 | 0.0301434 | 0.404994% | 0 |
| linattn.d_in_proj_b | 1560 | 14.058 | 0.00901151 | 0.368899% | 18.1812 |
| add | 4160 | 13.8524 | 0.00332989 | 0.363504% | 0 |
| linattn.d_in_proj_a | 1560 | 13.3676 | 0.00856901 | 0.350785% | 19.1201 |
| silu_mul | 2080 | 7.63168 | 0.00366908 | 0.200265% | 0 |
| linear_attention_conv | 1536 | 6.8792 | 0.00447865 | 0.180519% | 0 |
| fullattn.d_k_proj | 520 | 6.80102 | 0.0130789 | 0.178468% | 400.866 |
| fullattn.d_v_proj | 520 | 6.47917 | 0.0124599 | 0.170022% | 420.779 |
| full_attention_kv | 512 | 2.55162 | 0.00498362 | 0.0669578% | 0 |
| linear_attention_recurrent_batch | 24 | 2.54966 | 0.106236 | 0.0669065% | 0 |
| full_attention_q | 512 | 2.54093 | 0.00496275 | 0.0666773% | 0 |
| embedding_lookup | 65 | 0.526432 | 0.00809895 | 0.0138143% | 0 |
| linear_attention_conv_batch | 24 | 0.158336 | 0.00659733 | 0.00415494% | 0 |
| full_attention_attend_batch | 8 | 0.089056 | 0.011132 | 0.00233695% | 0 |
| full_attention_q_batch | 8 | 0.043744 | 0.005468 | 0.0011479% | 0 |
| full_attention_kv_batch | 8 | 0.042976 | 0.005372 | 0.00112775% | 0 |

## Memory summary

采样数：65（下表为时间线峰值）

| layer | peak(MiB) | pct of device |
|---|---:|---:|
| weights | 8021.84 | 67.3503% |
| kv cache / state | 55.25 | 0.463872% |
| scratch (peak) | 2.44142 | 0.0204978% |
| device used | 8459.88 | 71.028% |
| device total | 11910.6 | 100% |

### KV cache growth

| ts_ms | label | kv(MiB) | scratch(MiB) | device used(MiB) |
|---:|---|---:|---:|---:|
| 0 | prefill | 55.25 | 2.44142 | 8459.88 |
| 20.9837 | decode | 55.25 | 2.44142 | 8459.88 |
| 41.9719 | decode | 55.25 | 2.44142 | 8459.88 |
| 62.9528 | decode | 55.25 | 2.44142 | 8459.88 |
| 83.9363 | decode | 55.25 | 2.44142 | 8459.88 |
| 105.05 | decode | 55.25 | 2.44142 | 8459.88 |
| 126.035 | decode | 55.25 | 2.44142 | 8459.88 |
| 147.049 | decode | 55.25 | 2.44142 | 8459.88 |
| 168.125 | decode | 55.25 | 2.44142 | 8459.88 |
| 189.142 | decode | 55.25 | 2.44142 | 8459.88 |
| 210.174 | decode | 55.25 | 2.44142 | 8459.88 |
| 231.382 | decode | 55.25 | 2.44142 | 8459.88 |
| 252.412 | decode | 55.25 | 2.44142 | 8459.88 |
| 273.46 | decode | 55.25 | 2.44142 | 8459.88 |
| 294.576 | decode | 55.25 | 2.44142 | 8459.88 |
| 315.581 | decode | 55.25 | 2.44142 | 8459.88 |
| 336.609 | decode | 55.25 | 2.44142 | 8459.88 |
| 357.705 | decode | 55.25 | 2.44142 | 8459.88 |
| 378.85 | decode | 55.25 | 2.44142 | 8459.88 |
| 400.004 | decode | 55.25 | 2.44142 | 8459.88 |
| 421.075 | decode | 55.25 | 2.44142 | 8459.88 |
| 442.142 | decode | 55.25 | 2.44142 | 8459.88 |
| 463.597 | decode | 55.25 | 2.44142 | 8459.88 |
| 484.664 | decode | 55.25 | 2.44142 | 8459.88 |
| 505.74 | decode | 55.25 | 2.44142 | 8459.88 |
| 526.793 | decode | 55.25 | 2.44142 | 8459.88 |
| 547.873 | decode | 55.25 | 2.44142 | 8459.88 |
| 568.985 | decode | 55.25 | 2.44142 | 8459.88 |
| 590.07 | decode | 55.25 | 2.44142 | 8459.88 |
| 611.177 | decode | 55.25 | 2.44142 | 8459.88 |
| 632.257 | decode | 55.25 | 2.44142 | 8459.88 |
| 653.368 | decode | 55.25 | 2.44142 | 8459.88 |
| 674.465 | decode | 55.25 | 2.44142 | 8459.88 |
| 695.575 | decode | 55.25 | 2.44142 | 8459.88 |
| 716.703 | decode | 55.25 | 2.44142 | 8459.88 |
| 737.842 | decode | 55.25 | 2.44142 | 8459.88 |
| 758.97 | decode | 55.25 | 2.44142 | 8459.88 |
| 780.118 | decode | 55.25 | 2.44142 | 8459.88 |
| 801.239 | decode | 55.25 | 2.44142 | 8459.88 |
| 822.361 | decode | 55.25 | 2.44142 | 8459.88 |
| 843.478 | decode | 55.25 | 2.44142 | 8459.88 |
| 864.613 | decode | 55.25 | 2.44142 | 8459.88 |
| 885.74 | decode | 55.25 | 2.44142 | 8459.88 |
| 906.869 | decode | 55.25 | 2.44142 | 8459.88 |
| 928.775 | decode | 55.25 | 2.44142 | 8459.88 |
| 949.894 | decode | 55.25 | 2.44142 | 8459.88 |
| 971.061 | decode | 55.25 | 2.44142 | 8459.88 |
| 992.184 | decode | 55.25 | 2.44142 | 8459.88 |
| 1013.33 | decode | 55.25 | 2.44142 | 8459.88 |
| 1034.49 | decode | 55.25 | 2.44142 | 8459.88 |
| 1055.61 | decode | 55.25 | 2.44142 | 8459.88 |
| 1076.78 | decode | 55.25 | 2.44142 | 8459.88 |
| 1097.91 | decode | 55.25 | 2.44142 | 8459.88 |
| 1119.09 | decode | 55.25 | 2.44142 | 8459.88 |
| 1140.25 | decode | 55.25 | 2.44142 | 8459.88 |
| 1161.43 | decode | 55.25 | 2.44142 | 8459.88 |
| 1182.63 | decode | 55.25 | 2.44142 | 8459.88 |
| 1203.81 | decode | 55.25 | 2.44142 | 8459.88 |
| 1224.97 | decode | 55.25 | 2.44142 | 8459.88 |
| 1246.16 | decode | 55.25 | 2.44142 | 8459.88 |
| 1267.34 | decode | 55.25 | 2.44142 | 8459.88 |
| 1288.5 | decode | 55.25 | 2.44142 | 8459.88 |
| 1309.7 | decode | 55.25 | 2.44142 | 8459.88 |
| 1330.88 | decode | 55.25 | 2.44142 | 8459.88 |
| 1352.05 | decode | 55.25 | 2.44142 | 8459.88 |

## Device summary (14 samples)

| metric | avg | max |
|---|---:|---:|
| SM util (%) | 74.3571 | 81 |
| mem-bw util (%) | 45.8571 | 58 |
| power (W) | 184.864 | 269.181 |
| temp (C) | - | 42 |
| mem used (MiB) | - | 8837.25 |

## Weight load summary

- 上传总量：8021.84 MiB
- 驻留峰值：8021.84 MiB
- 分配总耗时（cudaMalloc）：18.4392 ms
- 拷贝总耗时（H2D）：854.307 ms

| ts_ms | event | name | bytes(MiB) | ms | resident(MiB) |
|---:|---|---|---:|---:|---:|
| 0 | alloc | model.language_model.embed_tokens.weight | 1212.5 | 0.12615 | 1212.5 |
| 0.00012 | upload | model.language_model.embed_tokens.weight | 1212.5 | 128.897 | 1212.5 |
| 0.533995 | alloc | model.language_model.layers.0.input_layernorm.weight | 0.00488281 | 0.003417 | 1212.5 |
| 0.534115 | upload | model.language_model.layers.0.input_layernorm.weight | 0.00488281 | 0.008192 | 1212.5 |
| 4.97955 | alloc | model.language_model.layers.0.linear_attn.in_proj_qkv.weight | 40 | 0.155796 | 1252.5 |
| 4.97973 | upload | model.language_model.layers.0.linear_attn.in_proj_qkv.weight | 40 | 4.2392 | 1252.5 |
| 53.8766 | alloc | model.language_model.layers.0.linear_attn.in_proj_z.weight | 20 | 0.095251 | 1272.5 |
| 53.8767 | upload | model.language_model.layers.0.linear_attn.in_proj_z.weight | 20 | 2.71203 | 1272.5 |
| 54.0956 | alloc | model.language_model.layers.0.linear_attn.in_proj_b.weight | 0.15625 | 0.074121 | 1272.66 |
| 54.0985 | upload | model.language_model.layers.0.linear_attn.in_proj_b.weight | 0.15625 | 0.03808 | 1272.66 |
| 54.6103 | alloc | model.language_model.layers.0.linear_attn.in_proj_a.weight | 0.15625 | 0.002736 | 1272.82 |
| 54.6103 | upload | model.language_model.layers.0.linear_attn.in_proj_a.weight | 0.15625 | 0.032736 | 1272.82 |
| 54.6752 | alloc | model.language_model.layers.0.linear_attn.conv1d.weight | 0.0625 | 0.002315 | 1272.88 |
| 54.6753 | upload | model.language_model.layers.0.linear_attn.conv1d.weight | 0.0625 | 0.014336 | 1272.88 |
| 54.7358 | alloc | model.language_model.layers.0.linear_attn.A_log | 0.00012207 | 0.004098 | 1272.88 |
| 54.7359 | upload | model.language_model.layers.0.linear_attn.A_log | 0.00012207 | 0.005856 | 1272.88 |
| 54.7503 | alloc | model.language_model.layers.0.linear_attn.norm.weight | 0.000488281 | 0.002935 | 1272.88 |
| 54.7525 | upload | model.language_model.layers.0.linear_attn.norm.weight | 0.000488281 | 0.00512 | 1272.88 |
| 54.766 | alloc | model.language_model.layers.0.linear_attn.dt_bias | 6.10352e-05 | 0.001703 | 1272.88 |
| 54.766 | upload | model.language_model.layers.0.linear_attn.dt_bias | 6.10352e-05 | 0.006144 | 1272.88 |
| 57.368 | alloc | model.language_model.layers.0.linear_attn.out_proj.weight | 20 | 0.067759 | 1292.88 |
| 57.3681 | upload | model.language_model.layers.0.linear_attn.out_proj.weight | 20 | 2.3681 | 1292.88 |
| 57.5143 | alloc | model.language_model.layers.0.post_attention_layernorm.weight | 0.00488281 | 0.002766 | 1292.89 |
| 57.5144 | upload | model.language_model.layers.0.post_attention_layernorm.weight | 0.00488281 | 0.006144 | 1292.89 |
| 62.6514 | alloc | model.language_model.layers.0.mlp.gate_proj.weight | 45 | 0.093418 | 1337.89 |
| 62.6515 | upload | model.language_model.layers.0.mlp.gate_proj.weight | 45 | 5.01187 | 1337.89 |
| 67.6648 | alloc | model.language_model.layers.0.mlp.up_proj.weight | 45 | 0.084821 | 1382.89 |
| 67.6649 | upload | model.language_model.layers.0.mlp.up_proj.weight | 45 | 4.7631 | 1382.89 |
| 72.6495 | alloc | model.language_model.layers.0.mlp.down_proj.weight | 45 | 0.081696 | 1427.89 |
| 72.6497 | upload | model.language_model.layers.0.mlp.down_proj.weight | 45 | 4.76384 | 1427.89 |
| 72.8383 | alloc | model.language_model.layers.1.input_layernorm.weight | 0.00488281 | 0.00485 | 1427.89 |
| 72.8384 | upload | model.language_model.layers.1.input_layernorm.weight | 0.00488281 | 0.006144 | 1427.89 |
| 77.1569 | alloc | model.language_model.layers.1.linear_attn.in_proj_qkv.weight | 40 | 0.092997 | 1467.89 |
| 77.1604 | upload | model.language_model.layers.1.linear_attn.in_proj_qkv.weight | 40 | 4.19946 | 1467.89 |
| 79.4413 | alloc | model.language_model.layers.1.linear_attn.in_proj_z.weight | 20 | 0.072938 | 1487.89 |
| 79.4413 | upload | model.language_model.layers.1.linear_attn.in_proj_z.weight | 20 | 2.10243 | 1487.89 |
| 79.5458 | alloc | model.language_model.layers.1.linear_attn.in_proj_b.weight | 0.15625 | 0.005551 | 1488.05 |
| 79.5458 | upload | model.language_model.layers.1.linear_attn.in_proj_b.weight | 0.15625 | 0.031456 | 1488.05 |
| 79.637 | alloc | model.language_model.layers.1.linear_attn.in_proj_a.weight | 0.15625 | 0.002755 | 1488.2 |
| 79.637 | upload | model.language_model.layers.1.linear_attn.in_proj_a.weight | 0.15625 | 0.028352 | 1488.2 |
| 79.7041 | alloc | model.language_model.layers.1.linear_attn.conv1d.weight | 0.0625 | 0.002905 | 1488.27 |
| 79.7042 | upload | model.language_model.layers.1.linear_attn.conv1d.weight | 0.0625 | 0.016064 | 1488.27 |
| 79.7469 | alloc | model.language_model.layers.1.linear_attn.A_log | 0.00012207 | 0.015419 | 1488.27 |
| 79.747 | upload | model.language_model.layers.1.linear_attn.A_log | 0.00012207 | 0.006144 | 1488.27 |
| 79.7613 | alloc | model.language_model.layers.1.linear_attn.norm.weight | 0.000488281 | 0.002254 | 1488.27 |
| 79.7614 | upload | model.language_model.layers.1.linear_attn.norm.weight | 0.000488281 | 0.006144 | 1488.27 |
| 79.7743 | alloc | model.language_model.layers.1.linear_attn.dt_bias | 6.10352e-05 | 0.001613 | 1488.27 |
| 79.7743 | upload | model.language_model.layers.1.linear_attn.dt_bias | 6.10352e-05 | 0.00512 | 1488.27 |
| 82.1757 | alloc | model.language_model.layers.1.linear_attn.out_proj.weight | 20 | 0.076435 | 1508.27 |
| 82.1758 | upload | model.language_model.layers.1.linear_attn.out_proj.weight | 20 | 2.18691 | 1508.27 |
| 82.2853 | alloc | model.language_model.layers.1.post_attention_layernorm.weight | 0.00488281 | 0.00514 | 1508.27 |
| 82.2853 | upload | model.language_model.layers.1.post_attention_layernorm.weight | 0.00488281 | 0.006144 | 1508.27 |
| 87.2199 | alloc | model.language_model.layers.1.mlp.gate_proj.weight | 45 | 0.122053 | 1553.27 |
| 87.22 | upload | model.language_model.layers.1.mlp.gate_proj.weight | 45 | 4.78621 | 1553.27 |
| 92.1824 | alloc | model.language_model.layers.1.mlp.up_proj.weight | 45 | 0.091995 | 1598.27 |
| 92.1825 | upload | model.language_model.layers.1.mlp.up_proj.weight | 45 | 4.75894 | 1598.27 |
| 97.1311 | alloc | model.language_model.layers.1.mlp.down_proj.weight | 45 | 0.082316 | 1643.27 |
| 97.1312 | upload | model.language_model.layers.1.mlp.down_proj.weight | 45 | 4.74627 | 1643.27 |
| 97.2933 | alloc | model.language_model.layers.2.input_layernorm.weight | 0.00488281 | 0.008216 | 1643.28 |
| 97.2934 | upload | model.language_model.layers.2.input_layernorm.weight | 0.00488281 | 0.013312 | 1643.28 |
| 101.707 | alloc | model.language_model.layers.2.linear_attn.in_proj_qkv.weight | 40 | 0.082888 | 1683.28 |
| 101.707 | upload | model.language_model.layers.2.linear_attn.in_proj_qkv.weight | 40 | 4.30461 | 1683.28 |
| 104.01 | alloc | model.language_model.layers.2.linear_attn.in_proj_z.weight | 20 | 0.081816 | 1703.28 |
| 104.011 | upload | model.language_model.layers.2.linear_attn.in_proj_z.weight | 20 | 2.10982 | 1703.28 |
| 104.134 | alloc | model.language_model.layers.2.linear_attn.in_proj_b.weight | 0.15625 | 0.01057 | 1703.43 |
| 104.138 | upload | model.language_model.layers.2.linear_attn.in_proj_b.weight | 0.15625 | 0.03424 | 1703.43 |
| 104.232 | alloc | model.language_model.layers.2.linear_attn.in_proj_a.weight | 0.15625 | 0.002766 | 1703.59 |
| 104.232 | upload | model.language_model.layers.2.linear_attn.in_proj_a.weight | 0.15625 | 0.029504 | 1703.59 |
| 104.295 | alloc | model.language_model.layers.2.linear_attn.conv1d.weight | 0.0625 | 0.002575 | 1703.65 |
| 104.295 | upload | model.language_model.layers.2.linear_attn.conv1d.weight | 0.0625 | 0.01424 | 1703.65 |
| 104.33 | alloc | model.language_model.layers.2.linear_attn.A_log | 0.00012207 | 0.007223 | 1703.65 |
| 104.33 | upload | model.language_model.layers.2.linear_attn.A_log | 0.00012207 | 0.007104 | 1703.65 |
| 104.344 | alloc | model.language_model.layers.2.linear_attn.norm.weight | 0.000488281 | 0.001623 | 1703.65 |
| 104.344 | upload | model.language_model.layers.2.linear_attn.norm.weight | 0.000488281 | 0.00608 | 1703.65 |
| 104.362 | alloc | model.language_model.layers.2.linear_attn.dt_bias | 6.10352e-05 | 0.006483 | 1703.65 |
| 104.362 | upload | model.language_model.layers.2.linear_attn.dt_bias | 6.10352e-05 | 0.005984 | 1703.65 |
| 106.697 | alloc | model.language_model.layers.2.linear_attn.out_proj.weight | 20 | 0.082347 | 1723.65 |
| 106.697 | upload | model.language_model.layers.2.linear_attn.out_proj.weight | 20 | 2.11462 | 1723.65 |
| 106.802 | alloc | model.language_model.layers.2.post_attention_layernorm.weight | 0.00488281 | 0.004949 | 1723.66 |
| 106.802 | upload | model.language_model.layers.2.post_attention_layernorm.weight | 0.00488281 | 0.006144 | 1723.66 |
| 111.7 | alloc | model.language_model.layers.2.mlp.gate_proj.weight | 45 | 0.080604 | 1768.66 |
| 111.7 | upload | model.language_model.layers.2.mlp.gate_proj.weight | 45 | 4.79171 | 1768.66 |
| 116.648 | alloc | model.language_model.layers.2.mlp.up_proj.weight | 45 | 0.081615 | 1813.66 |
| 116.648 | upload | model.language_model.layers.2.mlp.up_proj.weight | 45 | 4.7577 | 1813.66 |
| 121.606 | alloc | model.language_model.layers.2.mlp.down_proj.weight | 45 | 0.079652 | 1858.66 |
| 121.606 | upload | model.language_model.layers.2.mlp.down_proj.weight | 45 | 4.7569 | 1858.66 |
| 121.764 | alloc | model.language_model.layers.3.input_layernorm.weight | 0.00488281 | 0.010209 | 1858.66 |
| 121.764 | upload | model.language_model.layers.3.input_layernorm.weight | 0.00488281 | 0.012288 | 1858.66 |
| 126.198 | alloc | model.language_model.layers.3.self_attn.q_proj.weight | 40 | 0.088368 | 1898.66 |
| 126.198 | upload | model.language_model.layers.3.self_attn.q_proj.weight | 40 | 4.2313 | 1898.66 |
| 126.915 | alloc | model.language_model.layers.3.self_attn.k_proj.weight | 5 | 0.069062 | 1903.66 |
| 126.915 | upload | model.language_model.layers.3.self_attn.k_proj.weight | 5 | 0.544064 | 1903.66 |
| 132.081 | alloc | model.language_model.layers.3.self_attn.v_proj.weight | 5 | 0.071917 | 1908.66 |
| 132.081 | upload | model.language_model.layers.3.self_attn.v_proj.weight | 5 | 0.55408 | 1908.66 |
| 132.145 | alloc | model.language_model.layers.3.self_attn.q_norm.weight | 0.000488281 | 0.007043 | 1908.66 |
| 132.145 | upload | model.language_model.layers.3.self_attn.q_norm.weight | 0.000488281 | 0.006336 | 1908.66 |
| 132.206 | alloc | model.language_model.layers.3.self_attn.k_norm.weight | 0.000488281 | 0.001623 | 1908.66 |
| 132.206 | upload | model.language_model.layers.3.self_attn.k_norm.weight | 0.000488281 | 0.005984 | 1908.66 |
| 134.687 | alloc | model.language_model.layers.3.self_attn.o_proj.weight | 20 | 0.081335 | 1928.66 |
| 134.687 | upload | model.language_model.layers.3.self_attn.o_proj.weight | 20 | 2.3129 | 1928.66 |
| 134.803 | alloc | model.language_model.layers.3.post_attention_layernorm.weight | 0.00488281 | 0.007113 | 1928.67 |
| 134.803 | upload | model.language_model.layers.3.post_attention_layernorm.weight | 0.00488281 | 0.006048 | 1928.67 |
| 139.788 | alloc | model.language_model.layers.3.mlp.gate_proj.weight | 45 | 0.076826 | 1973.67 |
| 139.788 | upload | model.language_model.layers.3.mlp.gate_proj.weight | 45 | 4.88358 | 1973.67 |
| 144.786 | alloc | model.language_model.layers.3.mlp.up_proj.weight | 45 | 0.081555 | 2018.67 |
| 144.786 | upload | model.language_model.layers.3.mlp.up_proj.weight | 45 | 4.80486 | 2018.67 |
| 149.741 | alloc | model.language_model.layers.3.mlp.down_proj.weight | 45 | 0.080262 | 2063.67 |
| 149.741 | upload | model.language_model.layers.3.mlp.down_proj.weight | 45 | 4.75715 | 2063.67 |
| 149.902 | alloc | model.language_model.layers.4.input_layernorm.weight | 0.00488281 | 0.008607 | 2063.67 |
| 149.902 | upload | model.language_model.layers.4.input_layernorm.weight | 0.00488281 | 0.012288 | 2063.67 |
| 154.213 | alloc | model.language_model.layers.4.linear_attn.in_proj_qkv.weight | 40 | 0.082597 | 2103.67 |
| 154.213 | upload | model.language_model.layers.4.linear_attn.in_proj_qkv.weight | 40 | 4.20138 | 2103.67 |
| 156.493 | alloc | model.language_model.layers.4.linear_attn.in_proj_z.weight | 20 | 0.071446 | 2123.67 |
| 156.493 | upload | model.language_model.layers.4.linear_attn.in_proj_z.weight | 20 | 2.10218 | 2123.67 |
| 156.601 | alloc | model.language_model.layers.4.linear_attn.in_proj_b.weight | 0.15625 | 0.008957 | 2123.83 |
| 156.601 | upload | model.language_model.layers.4.linear_attn.in_proj_b.weight | 0.15625 | 0.03056 | 2123.83 |
| 156.696 | alloc | model.language_model.layers.4.linear_attn.in_proj_a.weight | 0.15625 | 0.002655 | 2123.98 |
| 156.696 | upload | model.language_model.layers.4.linear_attn.in_proj_a.weight | 0.15625 | 0.032576 | 2123.98 |
| 156.765 | alloc | model.language_model.layers.4.linear_attn.conv1d.weight | 0.0625 | 0.002976 | 2124.05 |
| 156.765 | upload | model.language_model.layers.4.linear_attn.conv1d.weight | 0.0625 | 0.01536 | 2124.05 |
| 156.802 | alloc | model.language_model.layers.4.linear_attn.A_log | 0.00012207 | 0.008716 | 2124.05 |
| 156.802 | upload | model.language_model.layers.4.linear_attn.A_log | 0.00012207 | 0.005984 | 2124.05 |
| 156.817 | alloc | model.language_model.layers.4.linear_attn.norm.weight | 0.000488281 | 0.003837 | 2124.05 |
| 156.817 | upload | model.language_model.layers.4.linear_attn.norm.weight | 0.000488281 | 0.00512 | 2124.05 |
| 156.832 | alloc | model.language_model.layers.4.linear_attn.dt_bias | 6.10352e-05 | 0.001753 | 2124.05 |
| 156.832 | upload | model.language_model.layers.4.linear_attn.dt_bias | 6.10352e-05 | 0.00512 | 2124.05 |
| 159.172 | alloc | model.language_model.layers.4.linear_attn.out_proj.weight | 20 | 0.079 | 2144.05 |
| 159.172 | upload | model.language_model.layers.4.linear_attn.out_proj.weight | 20 | 2.12118 | 2144.05 |
| 159.282 | alloc | model.language_model.layers.4.post_attention_layernorm.weight | 0.00488281 | 0.005631 | 2144.05 |
| 159.288 | upload | model.language_model.layers.4.post_attention_layernorm.weight | 0.00488281 | 0.011264 | 2144.05 |
| 164.136 | alloc | model.language_model.layers.4.mlp.gate_proj.weight | 45 | 0.072608 | 2189.05 |
| 164.136 | upload | model.language_model.layers.4.mlp.gate_proj.weight | 45 | 4.75024 | 2189.05 |
| 169.043 | alloc | model.language_model.layers.4.mlp.up_proj.weight | 45 | 0.082186 | 2234.05 |
| 169.044 | upload | model.language_model.layers.4.mlp.up_proj.weight | 45 | 4.7192 | 2234.05 |
| 173.981 | alloc | model.language_model.layers.4.mlp.down_proj.weight | 45 | 0.101513 | 2279.05 |
| 173.981 | upload | model.language_model.layers.4.mlp.down_proj.weight | 45 | 4.71357 | 2279.05 |
| 174.14 | alloc | model.language_model.layers.5.input_layernorm.weight | 0.00488281 | 0.008486 | 2279.06 |
| 174.14 | upload | model.language_model.layers.5.input_layernorm.weight | 0.00488281 | 0.007168 | 2279.06 |
| 178.457 | alloc | model.language_model.layers.5.linear_attn.in_proj_qkv.weight | 40 | 0.078921 | 2319.06 |
| 178.457 | upload | model.language_model.layers.5.linear_attn.in_proj_qkv.weight | 40 | 4.21277 | 2319.06 |
| 180.749 | alloc | model.language_model.layers.5.linear_attn.in_proj_z.weight | 20 | 0.079331 | 2339.06 |
| 180.749 | upload | model.language_model.layers.5.linear_attn.in_proj_z.weight | 20 | 2.10976 | 2339.06 |
| 180.853 | alloc | model.language_model.layers.5.linear_attn.in_proj_b.weight | 0.15625 | 0.006893 | 2339.21 |
| 180.853 | upload | model.language_model.layers.5.linear_attn.in_proj_b.weight | 0.15625 | 0.031008 | 2339.21 |
| 180.95 | alloc | model.language_model.layers.5.linear_attn.in_proj_a.weight | 0.15625 | 0.002565 | 2339.37 |
| 180.95 | upload | model.language_model.layers.5.linear_attn.in_proj_a.weight | 0.15625 | 0.033792 | 2339.37 |
| 181.015 | alloc | model.language_model.layers.5.linear_attn.conv1d.weight | 0.0625 | 0.003296 | 2339.43 |
| 181.015 | upload | model.language_model.layers.5.linear_attn.conv1d.weight | 0.0625 | 0.014304 | 2339.43 |
| 181.05 | alloc | model.language_model.layers.5.linear_attn.A_log | 0.00012207 | 0.008556 | 2339.43 |
| 181.051 | upload | model.language_model.layers.5.linear_attn.A_log | 0.00012207 | 0.00512 | 2339.43 |
| 181.077 | alloc | model.language_model.layers.5.linear_attn.norm.weight | 0.000488281 | 0.009368 | 2339.43 |
| 181.077 | upload | model.language_model.layers.5.linear_attn.norm.weight | 0.000488281 | 0.006048 | 2339.43 |
| 181.091 | alloc | model.language_model.layers.5.linear_attn.dt_bias | 6.10352e-05 | 0.002194 | 2339.43 |
| 181.091 | upload | model.language_model.layers.5.linear_attn.dt_bias | 6.10352e-05 | 0.006016 | 2339.43 |
| 183.453 | alloc | model.language_model.layers.5.linear_attn.out_proj.weight | 20 | 0.086284 | 2359.43 |
| 183.453 | upload | model.language_model.layers.5.linear_attn.out_proj.weight | 20 | 2.13699 | 2359.43 |
| 183.57 | alloc | model.language_model.layers.5.post_attention_layernorm.weight | 0.00488281 | 0.006282 | 2359.44 |
| 183.57 | upload | model.language_model.layers.5.post_attention_layernorm.weight | 0.00488281 | 0.011264 | 2359.44 |
| 188.423 | alloc | model.language_model.layers.5.mlp.gate_proj.weight | 45 | 0.074281 | 2404.44 |
| 188.423 | upload | model.language_model.layers.5.mlp.gate_proj.weight | 45 | 4.75373 | 2404.44 |
| 193.339 | alloc | model.language_model.layers.5.mlp.up_proj.weight | 45 | 0.07893 | 2449.44 |
| 193.339 | upload | model.language_model.layers.5.mlp.up_proj.weight | 45 | 4.73101 | 2449.44 |
| 198.271 | alloc | model.language_model.layers.5.mlp.down_proj.weight | 45 | 0.080683 | 2494.44 |
| 198.271 | upload | model.language_model.layers.5.mlp.down_proj.weight | 45 | 4.73146 | 2494.44 |
| 198.426 | alloc | model.language_model.layers.6.input_layernorm.weight | 0.00488281 | 0.007704 | 2494.44 |
| 198.426 | upload | model.language_model.layers.6.input_layernorm.weight | 0.00488281 | 0.011264 | 2494.44 |
| 202.759 | alloc | model.language_model.layers.6.linear_attn.in_proj_qkv.weight | 40 | 0.085182 | 2534.44 |
| 202.759 | upload | model.language_model.layers.6.linear_attn.in_proj_qkv.weight | 40 | 4.2215 | 2534.44 |
| 205.056 | alloc | model.language_model.layers.6.linear_attn.in_proj_z.weight | 20 | 0.07365 | 2554.44 |
| 205.056 | upload | model.language_model.layers.6.linear_attn.in_proj_z.weight | 20 | 2.11872 | 2554.44 |
| 205.168 | alloc | model.language_model.layers.6.linear_attn.in_proj_b.weight | 0.15625 | 0.006823 | 2554.6 |
| 205.168 | upload | model.language_model.layers.6.linear_attn.in_proj_b.weight | 0.15625 | 0.033984 | 2554.6 |
| 205.264 | alloc | model.language_model.layers.6.linear_attn.in_proj_a.weight | 0.15625 | 0.005992 | 2554.76 |
| 205.264 | upload | model.language_model.layers.6.linear_attn.in_proj_a.weight | 0.15625 | 0.030016 | 2554.76 |
| 205.327 | alloc | model.language_model.layers.6.linear_attn.conv1d.weight | 0.0625 | 0.003005 | 2554.82 |
| 205.327 | upload | model.language_model.layers.6.linear_attn.conv1d.weight | 0.0625 | 0.014336 | 2554.82 |
| 205.363 | alloc | model.language_model.layers.6.linear_attn.A_log | 0.00012207 | 0.008176 | 2554.82 |
| 205.363 | upload | model.language_model.layers.6.linear_attn.A_log | 0.00012207 | 0.006144 | 2554.82 |
| 205.38 | alloc | model.language_model.layers.6.linear_attn.norm.weight | 0.000488281 | 0.004909 | 2554.82 |
| 205.38 | upload | model.language_model.layers.6.linear_attn.norm.weight | 0.000488281 | 0.00512 | 2554.82 |
| 205.396 | alloc | model.language_model.layers.6.linear_attn.dt_bias | 6.10352e-05 | 0.004098 | 2554.82 |
| 205.396 | upload | model.language_model.layers.6.linear_attn.dt_bias | 6.10352e-05 | 0.006016 | 2554.82 |
| 207.737 | alloc | model.language_model.layers.6.linear_attn.out_proj.weight | 20 | 0.078189 | 2574.82 |
| 207.737 | upload | model.language_model.layers.6.linear_attn.out_proj.weight | 20 | 2.12461 | 2574.82 |
| 207.839 | alloc | model.language_model.layers.6.post_attention_layernorm.weight | 0.00488281 | 0.003316 | 2574.82 |
| 207.839 | upload | model.language_model.layers.6.post_attention_layernorm.weight | 0.00488281 | 0.006048 | 2574.82 |
| 212.756 | alloc | model.language_model.layers.6.mlp.gate_proj.weight | 45 | 0.072669 | 2619.82 |
| 212.756 | upload | model.language_model.layers.6.mlp.gate_proj.weight | 45 | 4.8081 | 2619.82 |
| 217.718 | alloc | model.language_model.layers.6.mlp.up_proj.weight | 45 | 0.106723 | 2664.82 |
| 217.718 | upload | model.language_model.layers.6.mlp.up_proj.weight | 45 | 4.74806 | 2664.82 |
| 222.676 | alloc | model.language_model.layers.6.mlp.down_proj.weight | 45 | 0.079962 | 2709.82 |
| 222.676 | upload | model.language_model.layers.6.mlp.down_proj.weight | 45 | 4.75789 | 2709.82 |
| 222.829 | alloc | model.language_model.layers.7.input_layernorm.weight | 0.00488281 | 0.008746 | 2709.83 |
| 222.829 | upload | model.language_model.layers.7.input_layernorm.weight | 0.00488281 | 0.006144 | 2709.83 |
| 227.156 | alloc | model.language_model.layers.7.self_attn.q_proj.weight | 40 | 0.079861 | 2749.83 |
| 227.156 | upload | model.language_model.layers.7.self_attn.q_proj.weight | 40 | 4.2208 | 2749.83 |
| 227.888 | alloc | model.language_model.layers.7.self_attn.k_proj.weight | 5 | 0.07371 | 2754.83 |
| 227.888 | upload | model.language_model.layers.7.self_attn.k_proj.weight | 5 | 0.555008 | 2754.83 |
| 228.581 | alloc | model.language_model.layers.7.self_attn.v_proj.weight | 5 | 0.070184 | 2759.83 |
| 228.581 | upload | model.language_model.layers.7.self_attn.v_proj.weight | 5 | 0.5568 | 2759.83 |
| 228.645 | alloc | model.language_model.layers.7.self_attn.q_norm.weight | 0.000488281 | 0.009257 | 2759.83 |
| 228.645 | upload | model.language_model.layers.7.self_attn.q_norm.weight | 0.000488281 | 0.006144 | 2759.83 |
| 228.677 | alloc | model.language_model.layers.7.self_attn.k_norm.weight | 0.000488281 | 0.00529 | 2759.83 |
| 228.677 | upload | model.language_model.layers.7.self_attn.k_norm.weight | 0.000488281 | 0.00512 | 2759.83 |
| 230.911 | alloc | model.language_model.layers.7.self_attn.o_proj.weight | 20 | 0.07385 | 2779.83 |
| 230.911 | upload | model.language_model.layers.7.self_attn.o_proj.weight | 20 | 2.1193 | 2779.83 |
| 231.018 | alloc | model.language_model.layers.7.post_attention_layernorm.weight | 0.00488281 | 0.005911 | 2779.83 |
| 231.018 | upload | model.language_model.layers.7.post_attention_layernorm.weight | 0.00488281 | 0.006112 | 2779.83 |
| 235.887 | alloc | model.language_model.layers.7.mlp.gate_proj.weight | 45 | 0.082667 | 2824.83 |
| 235.887 | upload | model.language_model.layers.7.mlp.gate_proj.weight | 45 | 4.76122 | 2824.83 |
| 240.813 | alloc | model.language_model.layers.7.mlp.up_proj.weight | 45 | 0.077026 | 2869.83 |
| 240.813 | upload | model.language_model.layers.7.mlp.up_proj.weight | 45 | 4.74288 | 2869.83 |
| 245.763 | alloc | model.language_model.layers.7.mlp.down_proj.weight | 45 | 0.080613 | 2914.83 |
| 245.763 | upload | model.language_model.layers.7.mlp.down_proj.weight | 45 | 4.7471 | 2914.83 |
| 245.914 | alloc | model.language_model.layers.8.input_layernorm.weight | 0.00488281 | 0.008406 | 2914.84 |
| 245.914 | upload | model.language_model.layers.8.input_layernorm.weight | 0.00488281 | 0.006144 | 2914.84 |
| 250.256 | alloc | model.language_model.layers.8.linear_attn.in_proj_qkv.weight | 40 | 0.081705 | 2954.84 |
| 250.256 | upload | model.language_model.layers.8.linear_attn.in_proj_qkv.weight | 40 | 4.23558 | 2954.84 |
| 252.548 | alloc | model.language_model.layers.8.linear_attn.in_proj_z.weight | 20 | 0.071656 | 2974.84 |
| 252.548 | upload | model.language_model.layers.8.linear_attn.in_proj_z.weight | 20 | 2.11667 | 2974.84 |
| 252.656 | alloc | model.language_model.layers.8.linear_attn.in_proj_b.weight | 0.15625 | 0.007384 | 2975 |
| 252.656 | upload | model.language_model.layers.8.linear_attn.in_proj_b.weight | 0.15625 | 0.033312 | 2975 |
| 252.744 | alloc | model.language_model.layers.8.linear_attn.in_proj_a.weight | 0.15625 | 0.005461 | 2975.15 |
| 252.744 | upload | model.language_model.layers.8.linear_attn.in_proj_a.weight | 0.15625 | 0.023136 | 2975.15 |
| 252.807 | alloc | model.language_model.layers.8.linear_attn.conv1d.weight | 0.0625 | 0.002785 | 2975.21 |
| 252.807 | upload | model.language_model.layers.8.linear_attn.conv1d.weight | 0.0625 | 0.016288 | 2975.21 |
| 252.842 | alloc | model.language_model.layers.8.linear_attn.A_log | 0.00012207 | 0.006763 | 2975.21 |
| 252.842 | upload | model.language_model.layers.8.linear_attn.A_log | 0.00012207 | 0.006144 | 2975.21 |
| 252.861 | alloc | model.language_model.layers.8.linear_attn.norm.weight | 0.000488281 | 0.006142 | 2975.21 |
| 252.861 | upload | model.language_model.layers.8.linear_attn.norm.weight | 0.000488281 | 0.00512 | 2975.21 |
| 252.874 | alloc | model.language_model.layers.8.linear_attn.dt_bias | 6.10352e-05 | 0.001853 | 2975.21 |
| 252.874 | upload | model.language_model.layers.8.linear_attn.dt_bias | 6.10352e-05 | 0.00512 | 2975.21 |
| 255.227 | alloc | model.language_model.layers.8.linear_attn.out_proj.weight | 20 | 0.091974 | 2995.21 |
| 255.227 | upload | model.language_model.layers.8.linear_attn.out_proj.weight | 20 | 2.12115 | 2995.21 |
| 255.335 | alloc | model.language_model.layers.8.post_attention_layernorm.weight | 0.00488281 | 0.003677 | 2995.22 |
| 255.335 | upload | model.language_model.layers.8.post_attention_layernorm.weight | 0.00488281 | 0.011264 | 2995.22 |
| 260.222 | alloc | model.language_model.layers.8.mlp.gate_proj.weight | 45 | 0.102145 | 3040.22 |
| 260.222 | upload | model.language_model.layers.8.mlp.gate_proj.weight | 45 | 4.75302 | 3040.22 |
| 265.161 | alloc | model.language_model.layers.8.mlp.up_proj.weight | 45 | 0.079481 | 3085.22 |
| 265.161 | upload | model.language_model.layers.8.mlp.up_proj.weight | 45 | 4.75149 | 3085.22 |
| 270.134 | alloc | model.language_model.layers.8.mlp.down_proj.weight | 45 | 0.085452 | 3130.22 |
| 270.134 | upload | model.language_model.layers.8.mlp.down_proj.weight | 45 | 4.77062 | 3130.22 |
| 270.29 | alloc | model.language_model.layers.9.input_layernorm.weight | 0.00488281 | 0.008807 | 3130.22 |
| 270.29 | upload | model.language_model.layers.9.input_layernorm.weight | 0.00488281 | 0.006144 | 3130.22 |
| 274.616 | alloc | model.language_model.layers.9.linear_attn.in_proj_qkv.weight | 40 | 0.084361 | 3170.22 |
| 274.617 | upload | model.language_model.layers.9.linear_attn.in_proj_qkv.weight | 40 | 4.21606 | 3170.22 |
| 276.92 | alloc | model.language_model.layers.9.linear_attn.in_proj_z.weight | 20 | 0.075994 | 3190.22 |
| 276.921 | upload | model.language_model.layers.9.linear_attn.in_proj_z.weight | 20 | 2.12358 | 3190.22 |
| 277.027 | alloc | model.language_model.layers.9.linear_attn.in_proj_b.weight | 0.15625 | 0.008056 | 3190.38 |
| 277.027 | upload | model.language_model.layers.9.linear_attn.in_proj_b.weight | 0.15625 | 0.030432 | 3190.38 |
| 277.125 | alloc | model.language_model.layers.9.linear_attn.in_proj_a.weight | 0.15625 | 0.002936 | 3190.54 |
| 277.125 | upload | model.language_model.layers.9.linear_attn.in_proj_a.weight | 0.15625 | 0.033056 | 3190.54 |
| 277.188 | alloc | model.language_model.layers.9.linear_attn.conv1d.weight | 0.0625 | 0.003106 | 3190.6 |
| 277.188 | upload | model.language_model.layers.9.linear_attn.conv1d.weight | 0.0625 | 0.014336 | 3190.6 |
| 277.238 | alloc | model.language_model.layers.9.linear_attn.A_log | 0.00012207 | 0.011973 | 3190.6 |
| 277.238 | upload | model.language_model.layers.9.linear_attn.A_log | 0.00012207 | 0.006144 | 3190.6 |
| 277.253 | alloc | model.language_model.layers.9.linear_attn.norm.weight | 0.000488281 | 0.002725 | 3190.6 |
| 277.269 | upload | model.language_model.layers.9.linear_attn.norm.weight | 0.000488281 | 0.00512 | 3190.6 |
| 277.283 | alloc | model.language_model.layers.9.linear_attn.dt_bias | 6.10352e-05 | 0.002495 | 3190.6 |
| 277.283 | upload | model.language_model.layers.9.linear_attn.dt_bias | 6.10352e-05 | 0.006048 | 3190.6 |
| 279.621 | alloc | model.language_model.layers.9.linear_attn.out_proj.weight | 20 | 0.076084 | 3210.6 |
| 279.621 | upload | model.language_model.layers.9.linear_attn.out_proj.weight | 20 | 2.12493 | 3210.6 |
| 279.725 | alloc | model.language_model.layers.9.post_attention_layernorm.weight | 0.00488281 | 0.004528 | 3210.6 |
| 279.725 | upload | model.language_model.layers.9.post_attention_layernorm.weight | 0.00488281 | 0.006144 | 3210.6 |
| 284.583 | alloc | model.language_model.layers.9.mlp.gate_proj.weight | 45 | 0.074863 | 3255.6 |
| 284.583 | upload | model.language_model.layers.9.mlp.gate_proj.weight | 45 | 4.75933 | 3255.6 |
| 289.514 | alloc | model.language_model.layers.9.mlp.up_proj.weight | 45 | 0.078018 | 3300.6 |
| 289.514 | upload | model.language_model.layers.9.mlp.up_proj.weight | 45 | 4.7455 | 3300.6 |
| 294.492 | alloc | model.language_model.layers.9.mlp.down_proj.weight | 45 | 0.093067 | 3345.6 |
| 294.493 | upload | model.language_model.layers.9.mlp.down_proj.weight | 45 | 4.76794 | 3345.6 |
| 294.645 | alloc | model.language_model.layers.10.input_layernorm.weight | 0.00488281 | 0.008225 | 3345.61 |
| 294.645 | upload | model.language_model.layers.10.input_layernorm.weight | 0.00488281 | 0.006368 | 3345.61 |
| 298.952 | alloc | model.language_model.layers.10.linear_attn.in_proj_qkv.weight | 40 | 0.081785 | 3385.61 |
| 298.953 | upload | model.language_model.layers.10.linear_attn.in_proj_qkv.weight | 40 | 4.20093 | 3385.61 |
| 301.255 | alloc | model.language_model.layers.10.linear_attn.in_proj_z.weight | 20 | 0.071506 | 3405.61 |
| 301.255 | upload | model.language_model.layers.10.linear_attn.in_proj_z.weight | 20 | 2.12323 | 3405.61 |
| 301.425 | alloc | model.language_model.layers.10.linear_attn.in_proj_b.weight | 0.15625 | 0.065755 | 3405.77 |
| 301.425 | upload | model.language_model.layers.10.linear_attn.in_proj_b.weight | 0.15625 | 0.031392 | 3405.77 |
| 301.523 | alloc | model.language_model.layers.10.linear_attn.in_proj_a.weight | 0.15625 | 0.006152 | 3405.92 |
| 301.523 | upload | model.language_model.layers.10.linear_attn.in_proj_a.weight | 0.15625 | 0.02912 | 3405.92 |
| 301.588 | alloc | model.language_model.layers.10.linear_attn.conv1d.weight | 0.0625 | 0.002866 | 3405.98 |
| 301.588 | upload | model.language_model.layers.10.linear_attn.conv1d.weight | 0.0625 | 0.016384 | 3405.98 |
| 301.621 | alloc | model.language_model.layers.10.linear_attn.A_log | 0.00012207 | 0.00545 | 3405.98 |
| 301.621 | upload | model.language_model.layers.10.linear_attn.A_log | 0.00012207 | 0.006144 | 3405.98 |
| 301.639 | alloc | model.language_model.layers.10.linear_attn.norm.weight | 0.000488281 | 0.007154 | 3405.99 |
| 301.639 | upload | model.language_model.layers.10.linear_attn.norm.weight | 0.000488281 | 0.00512 | 3405.99 |
| 301.653 | alloc | model.language_model.layers.10.linear_attn.dt_bias | 6.10352e-05 | 0.001813 | 3405.99 |
| 301.653 | upload | model.language_model.layers.10.linear_attn.dt_bias | 6.10352e-05 | 0.00512 | 3405.99 |
| 304.025 | alloc | model.language_model.layers.10.linear_attn.out_proj.weight | 20 | 0.102665 | 3425.99 |
| 304.025 | upload | model.language_model.layers.10.linear_attn.out_proj.weight | 20 | 2.12886 | 3425.99 |
| 304.134 | alloc | model.language_model.layers.10.post_attention_layernorm.weight | 0.00488281 | 0.007774 | 3425.99 |
| 304.134 | upload | model.language_model.layers.10.post_attention_layernorm.weight | 0.00488281 | 0.006016 | 3425.99 |
| 308.981 | alloc | model.language_model.layers.10.mlp.gate_proj.weight | 45 | 0.075183 | 3470.99 |
| 308.981 | upload | model.language_model.layers.10.mlp.gate_proj.weight | 45 | 4.74698 | 3470.99 |
| 313.934 | alloc | model.language_model.layers.10.mlp.up_proj.weight | 45 | 0.07898 | 3515.99 |
| 313.934 | upload | model.language_model.layers.10.mlp.up_proj.weight | 45 | 4.76867 | 3515.99 |
| 318.957 | alloc | model.language_model.layers.10.mlp.down_proj.weight | 45 | 0.08916 | 3560.99 |
| 318.957 | upload | model.language_model.layers.10.mlp.down_proj.weight | 45 | 4.81667 | 3560.99 |
| 319.129 | alloc | model.language_model.layers.11.input_layernorm.weight | 0.00488281 | 0.011242 | 3561 |
| 319.129 | upload | model.language_model.layers.11.input_layernorm.weight | 0.00488281 | 0.00688 | 3561 |
| 323.638 | alloc | model.language_model.layers.11.self_attn.q_proj.weight | 40 | 0.088949 | 3601 |
| 323.638 | upload | model.language_model.layers.11.self_attn.q_proj.weight | 40 | 4.39203 | 3601 |
| 324.388 | alloc | model.language_model.layers.11.self_attn.k_proj.weight | 5 | 0.074442 | 3606 |
| 324.388 | upload | model.language_model.layers.11.self_attn.k_proj.weight | 5 | 0.568576 | 3606 |
| 325.086 | alloc | model.language_model.layers.11.self_attn.v_proj.weight | 5 | 0.067408 | 3611 |
| 325.086 | upload | model.language_model.layers.11.self_attn.v_proj.weight | 5 | 0.554208 | 3611 |
| 325.15 | alloc | model.language_model.layers.11.self_attn.q_norm.weight | 0.000488281 | 0.007094 | 3611 |
| 325.15 | upload | model.language_model.layers.11.self_attn.q_norm.weight | 0.000488281 | 0.00608 | 3611 |
| 325.178 | alloc | model.language_model.layers.11.self_attn.k_norm.weight | 0.000488281 | 0.002535 | 3611 |
| 325.179 | upload | model.language_model.layers.11.self_attn.k_norm.weight | 0.000488281 | 0.00512 | 3611 |
| 327.442 | alloc | model.language_model.layers.11.self_attn.o_proj.weight | 20 | 0.082186 | 3631 |
| 327.442 | upload | model.language_model.layers.11.self_attn.o_proj.weight | 20 | 2.1407 | 3631 |
| 327.548 | alloc | model.language_model.layers.11.post_attention_layernorm.weight | 0.00488281 | 0.004508 | 3631 |
| 327.548 | upload | model.language_model.layers.11.post_attention_layernorm.weight | 0.00488281 | 0.006944 | 3631 |
| 332.443 | alloc | model.language_model.layers.11.mlp.gate_proj.weight | 45 | 0.081655 | 3676 |
| 332.443 | upload | model.language_model.layers.11.mlp.gate_proj.weight | 45 | 4.78803 | 3676 |
| 337.403 | alloc | model.language_model.layers.11.mlp.up_proj.weight | 45 | 0.120159 | 3721 |
| 337.403 | upload | model.language_model.layers.11.mlp.up_proj.weight | 45 | 4.73274 | 3721 |
| 342.348 | alloc | model.language_model.layers.11.mlp.down_proj.weight | 45 | 0.081365 | 3766 |
| 342.348 | upload | model.language_model.layers.11.mlp.down_proj.weight | 45 | 4.74352 | 3766 |
| 342.5 | alloc | model.language_model.layers.12.input_layernorm.weight | 0.00488281 | 0.008306 | 3766.01 |
| 342.5 | upload | model.language_model.layers.12.input_layernorm.weight | 0.00488281 | 0.00512 | 3766.01 |
| 346.84 | alloc | model.language_model.layers.12.linear_attn.in_proj_qkv.weight | 40 | 0.098647 | 3806.01 |
| 346.84 | upload | model.language_model.layers.12.linear_attn.in_proj_qkv.weight | 40 | 4.21341 | 3806.01 |
| 349.144 | alloc | model.language_model.layers.12.linear_attn.in_proj_z.weight | 20 | 0.077277 | 3826.01 |
| 349.144 | upload | model.language_model.layers.12.linear_attn.in_proj_z.weight | 20 | 2.1225 | 3826.01 |
| 349.254 | alloc | model.language_model.layers.12.linear_attn.in_proj_b.weight | 0.15625 | 0.006733 | 3826.16 |
| 349.254 | upload | model.language_model.layers.12.linear_attn.in_proj_b.weight | 0.15625 | 0.035456 | 3826.16 |
| 349.35 | alloc | model.language_model.layers.12.linear_attn.in_proj_a.weight | 0.15625 | 0.003226 | 3826.32 |
| 349.35 | upload | model.language_model.layers.12.linear_attn.in_proj_a.weight | 0.15625 | 0.023744 | 3826.32 |
| 349.413 | alloc | model.language_model.layers.12.linear_attn.conv1d.weight | 0.0625 | 0.006362 | 3826.38 |
| 349.413 | upload | model.language_model.layers.12.linear_attn.conv1d.weight | 0.0625 | 0.012288 | 3826.38 |
| 349.447 | alloc | model.language_model.layers.12.linear_attn.A_log | 0.00012207 | 0.007434 | 3826.38 |
| 349.447 | upload | model.language_model.layers.12.linear_attn.A_log | 0.00012207 | 0.00512 | 3826.38 |
| 349.463 | alloc | model.language_model.layers.12.linear_attn.norm.weight | 0.000488281 | 0.003787 | 3826.38 |
| 349.463 | upload | model.language_model.layers.12.linear_attn.norm.weight | 0.000488281 | 0.00512 | 3826.38 |
| 349.478 | alloc | model.language_model.layers.12.linear_attn.dt_bias | 6.10352e-05 | 0.004117 | 3826.38 |
| 349.478 | upload | model.language_model.layers.12.linear_attn.dt_bias | 6.10352e-05 | 0.005088 | 3826.38 |
| 351.81 | alloc | model.language_model.layers.12.linear_attn.out_proj.weight | 20 | 0.082197 | 3846.38 |
| 351.81 | upload | model.language_model.layers.12.linear_attn.out_proj.weight | 20 | 2.12678 | 3846.38 |
| 351.907 | alloc | model.language_model.layers.12.post_attention_layernorm.weight | 0.00488281 | 0.00537 | 3846.39 |
| 351.907 | upload | model.language_model.layers.12.post_attention_layernorm.weight | 0.00488281 | 0.005792 | 3846.39 |
| 356.764 | alloc | model.language_model.layers.12.mlp.gate_proj.weight | 45 | 0.07879 | 3891.39 |
| 356.765 | upload | model.language_model.layers.12.mlp.gate_proj.weight | 45 | 4.75514 | 3891.39 |
| 361.707 | alloc | model.language_model.layers.12.mlp.up_proj.weight | 45 | 0.083058 | 3936.39 |
| 361.71 | upload | model.language_model.layers.12.mlp.up_proj.weight | 45 | 4.75453 | 3936.39 |
| 366.672 | alloc | model.language_model.layers.12.mlp.down_proj.weight | 45 | 0.08408 | 3981.39 |
| 366.672 | upload | model.language_model.layers.12.mlp.down_proj.weight | 45 | 4.76365 | 3981.39 |
| 366.828 | alloc | model.language_model.layers.13.input_layernorm.weight | 0.00488281 | 0.008566 | 3981.39 |
| 366.828 | upload | model.language_model.layers.13.input_layernorm.weight | 0.00488281 | 0.00704 | 3981.39 |
| 371.172 | alloc | model.language_model.layers.13.linear_attn.in_proj_qkv.weight | 40 | 0.081445 | 4021.39 |
| 371.173 | upload | model.language_model.layers.13.linear_attn.in_proj_qkv.weight | 40 | 4.2376 | 4021.39 |
| 373.466 | alloc | model.language_model.layers.13.linear_attn.in_proj_z.weight | 20 | 0.074201 | 4041.39 |
| 373.466 | upload | model.language_model.layers.13.linear_attn.in_proj_z.weight | 20 | 2.11837 | 4041.39 |
| 373.571 | alloc | model.language_model.layers.13.linear_attn.in_proj_b.weight | 0.15625 | 0.007233 | 4041.55 |
| 373.571 | upload | model.language_model.layers.13.linear_attn.in_proj_b.weight | 0.15625 | 0.033088 | 4041.55 |
| 373.656 | alloc | model.language_model.layers.13.linear_attn.in_proj_a.weight | 0.15625 | 0.002875 | 4041.7 |
| 373.656 | upload | model.language_model.layers.13.linear_attn.in_proj_a.weight | 0.15625 | 0.023168 | 4041.7 |
| 373.718 | alloc | model.language_model.layers.13.linear_attn.conv1d.weight | 0.0625 | 0.003136 | 4041.77 |
| 373.718 | upload | model.language_model.layers.13.linear_attn.conv1d.weight | 0.0625 | 0.012288 | 4041.77 |
| 373.749 | alloc | model.language_model.layers.13.linear_attn.A_log | 0.00012207 | 0.003537 | 4041.77 |
| 373.749 | upload | model.language_model.layers.13.linear_attn.A_log | 0.00012207 | 0.005952 | 4041.77 |
| 373.767 | alloc | model.language_model.layers.13.linear_attn.norm.weight | 0.000488281 | 0.006001 | 4041.77 |
| 373.767 | upload | model.language_model.layers.13.linear_attn.norm.weight | 0.000488281 | 0.005088 | 4041.77 |
| 373.78 | alloc | model.language_model.layers.13.linear_attn.dt_bias | 6.10352e-05 | 0.001924 | 4041.77 |
| 373.78 | upload | model.language_model.layers.13.linear_attn.dt_bias | 6.10352e-05 | 0.005216 | 4041.77 |
| 376.112 | alloc | model.language_model.layers.13.linear_attn.out_proj.weight | 20 | 0.081746 | 4061.77 |
| 376.112 | upload | model.language_model.layers.13.linear_attn.out_proj.weight | 20 | 2.12835 | 4061.77 |
| 376.211 | alloc | model.language_model.layers.13.post_attention_layernorm.weight | 0.00488281 | 0.00497 | 4061.77 |
| 376.211 | upload | model.language_model.layers.13.post_attention_layernorm.weight | 0.00488281 | 0.006048 | 4061.77 |
| 381.066 | alloc | model.language_model.layers.13.mlp.gate_proj.weight | 45 | 0.082677 | 4106.77 |
| 381.066 | upload | model.language_model.layers.13.mlp.gate_proj.weight | 45 | 4.74733 | 4106.77 |
| 386.015 | alloc | model.language_model.layers.13.mlp.up_proj.weight | 45 | 0.081545 | 4151.77 |
| 386.015 | upload | model.language_model.layers.13.mlp.up_proj.weight | 45 | 4.75965 | 4151.77 |
| 390.978 | alloc | model.language_model.layers.13.mlp.down_proj.weight | 45 | 0.100241 | 4196.77 |
| 390.978 | upload | model.language_model.layers.13.mlp.down_proj.weight | 45 | 4.74602 | 4196.77 |
| 391.13 | alloc | model.language_model.layers.14.input_layernorm.weight | 0.00488281 | 0.008796 | 4196.78 |
| 391.13 | upload | model.language_model.layers.14.input_layernorm.weight | 0.00488281 | 0.007168 | 4196.78 |
| 395.468 | alloc | model.language_model.layers.14.linear_attn.in_proj_qkv.weight | 40 | 0.079431 | 4236.78 |
| 395.468 | upload | model.language_model.layers.14.linear_attn.in_proj_qkv.weight | 40 | 4.2335 | 4236.78 |
| 397.766 | alloc | model.language_model.layers.14.linear_attn.in_proj_z.weight | 20 | 0.082587 | 4256.78 |
| 397.766 | upload | model.language_model.layers.14.linear_attn.in_proj_z.weight | 20 | 2.1145 | 4256.78 |
| 397.872 | alloc | model.language_model.layers.14.linear_attn.in_proj_b.weight | 0.15625 | 0.007755 | 4256.93 |
| 397.872 | upload | model.language_model.layers.14.linear_attn.in_proj_b.weight | 0.15625 | 0.032896 | 4256.93 |
| 397.955 | alloc | model.language_model.layers.14.linear_attn.in_proj_a.weight | 0.15625 | 0.002715 | 4257.09 |
| 397.955 | upload | model.language_model.layers.14.linear_attn.in_proj_a.weight | 0.15625 | 0.023264 | 4257.09 |
| 398.012 | alloc | model.language_model.layers.14.linear_attn.conv1d.weight | 0.0625 | 0.002876 | 4257.15 |
| 398.012 | upload | model.language_model.layers.14.linear_attn.conv1d.weight | 0.0625 | 0.013152 | 4257.15 |
| 398.045 | alloc | model.language_model.layers.14.linear_attn.A_log | 0.00012207 | 0.003386 | 4257.15 |
| 398.045 | upload | model.language_model.layers.14.linear_attn.A_log | 0.00012207 | 0.005472 | 4257.15 |
| 398.06 | alloc | model.language_model.layers.14.linear_attn.norm.weight | 0.000488281 | 0.004098 | 4257.15 |
| 398.06 | upload | model.language_model.layers.14.linear_attn.norm.weight | 0.000488281 | 0.00512 | 4257.15 |
| 398.077 | alloc | model.language_model.layers.14.linear_attn.dt_bias | 6.10352e-05 | 0.001483 | 4257.15 |
| 398.077 | upload | model.language_model.layers.14.linear_attn.dt_bias | 6.10352e-05 | 0.00512 | 4257.15 |
| 400.423 | alloc | model.language_model.layers.14.linear_attn.out_proj.weight | 20 | 0.080854 | 4277.15 |
| 400.423 | upload | model.language_model.layers.14.linear_attn.out_proj.weight | 20 | 2.14154 | 4277.15 |
| 400.529 | alloc | model.language_model.layers.14.post_attention_layernorm.weight | 0.00488281 | 0.00519 | 4277.16 |
| 400.529 | upload | model.language_model.layers.14.post_attention_layernorm.weight | 0.00488281 | 0.006144 | 4277.16 |
| 405.378 | alloc | model.language_model.layers.14.mlp.gate_proj.weight | 45 | 0.077407 | 4322.16 |
| 405.379 | upload | model.language_model.layers.14.mlp.gate_proj.weight | 45 | 4.74829 | 4322.16 |
| 410.31 | alloc | model.language_model.layers.14.mlp.up_proj.weight | 45 | 0.085663 | 4367.16 |
| 410.311 | upload | model.language_model.layers.14.mlp.up_proj.weight | 45 | 4.74333 | 4367.16 |
| 415.294 | alloc | model.language_model.layers.14.mlp.down_proj.weight | 45 | 0.083219 | 4412.16 |
| 415.294 | upload | model.language_model.layers.14.mlp.down_proj.weight | 45 | 4.78266 | 4412.16 |
| 415.45 | alloc | model.language_model.layers.15.input_layernorm.weight | 0.00488281 | 0.009759 | 4412.16 |
| 415.45 | upload | model.language_model.layers.15.input_layernorm.weight | 0.00488281 | 0.006144 | 4412.16 |
| 419.79 | alloc | model.language_model.layers.15.self_attn.q_proj.weight | 40 | 0.082757 | 4452.16 |
| 419.79 | upload | model.language_model.layers.15.self_attn.q_proj.weight | 40 | 4.23082 | 4452.16 |
| 420.518 | alloc | model.language_model.layers.15.self_attn.k_proj.weight | 5 | 0.072729 | 4457.16 |
| 420.518 | upload | model.language_model.layers.15.self_attn.k_proj.weight | 5 | 0.555264 | 4457.16 |
| 421.201 | alloc | model.language_model.layers.15.self_attn.v_proj.weight | 5 | 0.064382 | 4462.16 |
| 421.202 | upload | model.language_model.layers.15.self_attn.v_proj.weight | 5 | 0.555008 | 4462.16 |
| 421.263 | alloc | model.language_model.layers.15.self_attn.q_norm.weight | 0.000488281 | 0.009838 | 4462.16 |
| 421.264 | upload | model.language_model.layers.15.self_attn.q_norm.weight | 0.000488281 | 0.006144 | 4462.16 |
| 421.291 | alloc | model.language_model.layers.15.self_attn.k_norm.weight | 0.000488281 | 0.002685 | 4462.16 |
| 421.291 | upload | model.language_model.layers.15.self_attn.k_norm.weight | 0.000488281 | 0.00608 | 4462.16 |
| 423.545 | alloc | model.language_model.layers.15.self_attn.o_proj.weight | 20 | 0.081996 | 4482.16 |
| 423.545 | upload | model.language_model.layers.15.self_attn.o_proj.weight | 20 | 2.13267 | 4482.16 |
| 423.646 | alloc | model.language_model.layers.15.post_attention_layernorm.weight | 0.00488281 | 0.004618 | 4482.17 |
| 423.647 | upload | model.language_model.layers.15.post_attention_layernorm.weight | 0.00488281 | 0.006144 | 4482.17 |
| 428.503 | alloc | model.language_model.layers.15.mlp.gate_proj.weight | 45 | 0.07911 | 4527.17 |
| 428.503 | upload | model.language_model.layers.15.mlp.gate_proj.weight | 45 | 4.7536 | 4527.17 |
| 433.458 | alloc | model.language_model.layers.15.mlp.up_proj.weight | 45 | 0.095221 | 4572.17 |
| 433.458 | upload | model.language_model.layers.15.mlp.up_proj.weight | 45 | 4.75408 | 4572.17 |
| 438.397 | alloc | model.language_model.layers.15.mlp.down_proj.weight | 45 | 0.081264 | 4617.17 |
| 438.397 | upload | model.language_model.layers.15.mlp.down_proj.weight | 45 | 4.73882 | 4617.17 |
| 438.547 | alloc | model.language_model.layers.16.input_layernorm.weight | 0.00488281 | 0.009748 | 4617.17 |
| 438.548 | upload | model.language_model.layers.16.input_layernorm.weight | 0.00488281 | 0.006144 | 4617.17 |
| 442.876 | alloc | model.language_model.layers.16.linear_attn.in_proj_qkv.weight | 40 | 0.08977 | 4657.17 |
| 442.876 | upload | model.language_model.layers.16.linear_attn.in_proj_qkv.weight | 40 | 4.21472 | 4657.17 |
| 445.176 | alloc | model.language_model.layers.16.linear_attn.in_proj_z.weight | 20 | 0.070975 | 4677.17 |
| 445.176 | upload | model.language_model.layers.16.linear_attn.in_proj_z.weight | 20 | 2.12944 | 4677.17 |
| 445.282 | alloc | model.language_model.layers.16.linear_attn.in_proj_b.weight | 0.15625 | 0.007754 | 4677.33 |
| 445.282 | upload | model.language_model.layers.16.linear_attn.in_proj_b.weight | 0.15625 | 0.03056 | 4677.33 |
| 445.367 | alloc | model.language_model.layers.16.linear_attn.in_proj_a.weight | 0.15625 | 0.002555 | 4677.49 |
| 445.367 | upload | model.language_model.layers.16.linear_attn.in_proj_a.weight | 0.15625 | 0.025088 | 4677.49 |
| 445.424 | alloc | model.language_model.layers.16.linear_attn.conv1d.weight | 0.0625 | 0.003146 | 4677.55 |
| 445.424 | upload | model.language_model.layers.16.linear_attn.conv1d.weight | 0.0625 | 0.012288 | 4677.55 |
| 445.456 | alloc | model.language_model.layers.16.linear_attn.A_log | 0.00012207 | 0.003667 | 4677.55 |
| 445.456 | upload | model.language_model.layers.16.linear_attn.A_log | 0.00012207 | 0.006048 | 4677.55 |
| 445.472 | alloc | model.language_model.layers.16.linear_attn.norm.weight | 0.000488281 | 0.004439 | 4677.55 |
| 445.472 | upload | model.language_model.layers.16.linear_attn.norm.weight | 0.000488281 | 0.006048 | 4677.55 |
| 445.485 | alloc | model.language_model.layers.16.linear_attn.dt_bias | 6.10352e-05 | 0.001673 | 4677.55 |
| 445.485 | upload | model.language_model.layers.16.linear_attn.dt_bias | 6.10352e-05 | 0.00512 | 4677.55 |
| 447.817 | alloc | model.language_model.layers.16.linear_attn.out_proj.weight | 20 | 0.081885 | 4697.55 |
| 447.817 | upload | model.language_model.layers.16.linear_attn.out_proj.weight | 20 | 2.12845 | 4697.55 |
| 447.922 | alloc | model.language_model.layers.16.post_attention_layernorm.weight | 0.00488281 | 0.00529 | 4697.55 |
| 447.922 | upload | model.language_model.layers.16.post_attention_layernorm.weight | 0.00488281 | 0.00512 | 4697.55 |
| 452.791 | alloc | model.language_model.layers.16.mlp.gate_proj.weight | 45 | 0.07865 | 4742.55 |
| 452.791 | upload | model.language_model.layers.16.mlp.gate_proj.weight | 45 | 4.7665 | 4742.55 |
| 457.722 | alloc | model.language_model.layers.16.mlp.up_proj.weight | 45 | 0.086454 | 4787.55 |
| 457.722 | upload | model.language_model.layers.16.mlp.up_proj.weight | 45 | 4.73949 | 4787.55 |
| 462.681 | alloc | model.language_model.layers.16.mlp.down_proj.weight | 45 | 0.079992 | 4832.55 |
| 462.681 | upload | model.language_model.layers.16.mlp.down_proj.weight | 45 | 4.76365 | 4832.55 |
| 462.83 | alloc | model.language_model.layers.17.input_layernorm.weight | 0.00488281 | 0.008887 | 4832.56 |
| 462.83 | upload | model.language_model.layers.17.input_layernorm.weight | 0.00488281 | 0.005984 | 4832.56 |
| 467.209 | alloc | model.language_model.layers.17.linear_attn.in_proj_qkv.weight | 40 | 0.081504 | 4872.56 |
| 467.209 | upload | model.language_model.layers.17.linear_attn.in_proj_qkv.weight | 40 | 4.26819 | 4872.56 |
| 469.497 | alloc | model.language_model.layers.17.linear_attn.in_proj_z.weight | 20 | 0.071636 | 4892.56 |
| 469.497 | upload | model.language_model.layers.17.linear_attn.in_proj_z.weight | 20 | 2.11491 | 4892.56 |
| 469.668 | alloc | model.language_model.layers.17.linear_attn.in_proj_b.weight | 0.15625 | 0.069663 | 4892.71 |
| 469.668 | upload | model.language_model.layers.17.linear_attn.in_proj_b.weight | 0.15625 | 0.032704 | 4892.71 |
| 469.761 | alloc | model.language_model.layers.17.linear_attn.in_proj_a.weight | 0.15625 | 0.006292 | 4892.87 |
| 469.761 | upload | model.language_model.layers.17.linear_attn.in_proj_a.weight | 0.15625 | 0.026496 | 4892.87 |
| 469.821 | alloc | model.language_model.layers.17.linear_attn.conv1d.weight | 0.0625 | 0.005741 | 4892.93 |
| 469.821 | upload | model.language_model.layers.17.linear_attn.conv1d.weight | 0.0625 | 0.012288 | 4892.93 |
| 469.85 | alloc | model.language_model.layers.17.linear_attn.A_log | 0.00012207 | 0.003427 | 4892.93 |
| 469.85 | upload | model.language_model.layers.17.linear_attn.A_log | 0.00012207 | 0.00512 | 4892.93 |
| 469.869 | alloc | model.language_model.layers.17.linear_attn.norm.weight | 0.000488281 | 0.007073 | 4892.93 |
| 469.869 | upload | model.language_model.layers.17.linear_attn.norm.weight | 0.000488281 | 0.005344 | 4892.93 |
| 469.882 | alloc | model.language_model.layers.17.linear_attn.dt_bias | 6.10352e-05 | 0.002044 | 4892.93 |
| 469.882 | upload | model.language_model.layers.17.linear_attn.dt_bias | 6.10352e-05 | 0.005824 | 4892.93 |
| 472.225 | alloc | model.language_model.layers.17.linear_attn.out_proj.weight | 20 | 0.084912 | 4912.93 |
| 472.225 | upload | model.language_model.layers.17.linear_attn.out_proj.weight | 20 | 2.13542 | 4912.93 |
| 472.325 | alloc | model.language_model.layers.17.post_attention_layernorm.weight | 0.00488281 | 0.006823 | 4912.94 |
| 472.325 | upload | model.language_model.layers.17.post_attention_layernorm.weight | 0.00488281 | 0.006144 | 4912.94 |
| 477.324 | alloc | model.language_model.layers.17.mlp.gate_proj.weight | 45 | 0.119688 | 4957.94 |
| 477.324 | upload | model.language_model.layers.17.mlp.gate_proj.weight | 45 | 4.85146 | 4957.94 |
| 482.292 | alloc | model.language_model.layers.17.mlp.up_proj.weight | 45 | 0.095642 | 5002.94 |
| 482.292 | upload | model.language_model.layers.17.mlp.up_proj.weight | 45 | 4.76131 | 5002.94 |
| 487.272 | alloc | model.language_model.layers.17.mlp.down_proj.weight | 45 | 0.090432 | 5047.94 |
| 487.272 | upload | model.language_model.layers.17.mlp.down_proj.weight | 45 | 4.77306 | 5047.94 |
| 487.426 | alloc | model.language_model.layers.18.input_layernorm.weight | 0.00488281 | 0.009268 | 5047.94 |
| 487.426 | upload | model.language_model.layers.18.input_layernorm.weight | 0.00488281 | 0.006144 | 5047.94 |
| 491.768 | alloc | model.language_model.layers.18.linear_attn.in_proj_qkv.weight | 40 | 0.08435 | 5087.94 |
| 491.769 | upload | model.language_model.layers.18.linear_attn.in_proj_qkv.weight | 40 | 4.2329 | 5087.94 |
| 494.088 | alloc | model.language_model.layers.18.linear_attn.in_proj_z.weight | 20 | 0.071596 | 5107.94 |
| 494.088 | upload | model.language_model.layers.18.linear_attn.in_proj_z.weight | 20 | 2.14787 | 5107.94 |
| 494.199 | alloc | model.language_model.layers.18.linear_attn.in_proj_b.weight | 0.15625 | 0.009558 | 5108.1 |
| 494.199 | upload | model.language_model.layers.18.linear_attn.in_proj_b.weight | 0.15625 | 0.03568 | 5108.1 |
| 494.283 | alloc | model.language_model.layers.18.linear_attn.in_proj_a.weight | 0.15625 | 0.003496 | 5108.26 |
| 494.284 | upload | model.language_model.layers.18.linear_attn.in_proj_a.weight | 0.15625 | 0.02608 | 5108.26 |
| 494.345 | alloc | model.language_model.layers.18.linear_attn.conv1d.weight | 0.0625 | 0.003136 | 5108.32 |
| 494.345 | upload | model.language_model.layers.18.linear_attn.conv1d.weight | 0.0625 | 0.013312 | 5108.32 |
| 494.383 | alloc | model.language_model.layers.18.linear_attn.A_log | 0.00012207 | 0.011301 | 5108.32 |
| 494.383 | upload | model.language_model.layers.18.linear_attn.A_log | 0.00012207 | 0.00512 | 5108.32 |
| 494.4 | alloc | model.language_model.layers.18.linear_attn.norm.weight | 0.000488281 | 0.004268 | 5108.32 |
| 494.4 | upload | model.language_model.layers.18.linear_attn.norm.weight | 0.000488281 | 0.005216 | 5108.32 |
| 494.415 | alloc | model.language_model.layers.18.linear_attn.dt_bias | 6.10352e-05 | 0.001874 | 5108.32 |
| 494.415 | upload | model.language_model.layers.18.linear_attn.dt_bias | 6.10352e-05 | 0.00512 | 5108.32 |
| 496.744 | alloc | model.language_model.layers.18.linear_attn.out_proj.weight | 20 | 0.085773 | 5128.32 |
| 496.744 | upload | model.language_model.layers.18.linear_attn.out_proj.weight | 20 | 2.12022 | 5128.32 |
| 496.84 | alloc | model.language_model.layers.18.post_attention_layernorm.weight | 0.00488281 | 0.003827 | 5128.32 |
| 496.84 | upload | model.language_model.layers.18.post_attention_layernorm.weight | 0.00488281 | 0.006144 | 5128.32 |
| 501.703 | alloc | model.language_model.layers.18.mlp.gate_proj.weight | 45 | 0.085562 | 5173.32 |
| 501.703 | upload | model.language_model.layers.18.mlp.gate_proj.weight | 45 | 4.75334 | 5173.32 |
| 506.725 | alloc | model.language_model.layers.18.mlp.up_proj.weight | 45 | 0.082597 | 5218.32 |
| 506.725 | upload | model.language_model.layers.18.mlp.up_proj.weight | 45 | 4.83283 | 5218.32 |
| 511.714 | alloc | model.language_model.layers.18.mlp.down_proj.weight | 45 | 0.08424 | 5263.32 |
| 511.714 | upload | model.language_model.layers.18.mlp.down_proj.weight | 45 | 4.78928 | 5263.32 |
| 511.869 | alloc | model.language_model.layers.19.input_layernorm.weight | 0.00488281 | 0.013385 | 5263.33 |
| 511.869 | upload | model.language_model.layers.19.input_layernorm.weight | 0.00488281 | 0.006144 | 5263.33 |
| 516.255 | alloc | model.language_model.layers.19.self_attn.q_proj.weight | 40 | 0.106482 | 5303.33 |
| 516.277 | upload | model.language_model.layers.19.self_attn.q_proj.weight | 40 | 4.25376 | 5303.33 |
| 517.035 | alloc | model.language_model.layers.19.self_attn.k_proj.weight | 5 | 0.074772 | 5308.33 |
| 517.035 | upload | model.language_model.layers.19.self_attn.k_proj.weight | 5 | 0.555616 | 5308.33 |
| 517.724 | alloc | model.language_model.layers.19.self_attn.v_proj.weight | 5 | 0.068019 | 5313.33 |
| 517.724 | upload | model.language_model.layers.19.self_attn.v_proj.weight | 5 | 0.559104 | 5313.33 |
| 517.783 | alloc | model.language_model.layers.19.self_attn.q_norm.weight | 0.000488281 | 0.006763 | 5313.33 |
| 517.783 | upload | model.language_model.layers.19.self_attn.q_norm.weight | 0.000488281 | 0.006144 | 5313.33 |
| 517.81 | alloc | model.language_model.layers.19.self_attn.k_norm.weight | 0.000488281 | 0.002635 | 5313.33 |
| 517.81 | upload | model.language_model.layers.19.self_attn.k_norm.weight | 0.000488281 | 0.00512 | 5313.33 |
| 520.05 | alloc | model.language_model.layers.19.self_attn.o_proj.weight | 20 | 0.077667 | 5333.33 |
| 520.051 | upload | model.language_model.layers.19.self_attn.o_proj.weight | 20 | 2.12326 | 5333.33 |
| 520.162 | alloc | model.language_model.layers.19.post_attention_layernorm.weight | 0.00488281 | 0.004619 | 5333.34 |
| 520.162 | upload | model.language_model.layers.19.post_attention_layernorm.weight | 0.00488281 | 0.006016 | 5333.34 |
| 525.061 | alloc | model.language_model.layers.19.mlp.gate_proj.weight | 45 | 0.075444 | 5378.34 |
| 525.062 | upload | model.language_model.layers.19.mlp.gate_proj.weight | 45 | 4.80138 | 5378.34 |
| 530.068 | alloc | model.language_model.layers.19.mlp.up_proj.weight | 45 | 0.092446 | 5423.34 |
| 530.068 | upload | model.language_model.layers.19.mlp.up_proj.weight | 45 | 4.80806 | 5423.34 |
| 535.042 | alloc | model.language_model.layers.19.mlp.down_proj.weight | 45 | 0.079571 | 5468.34 |
| 535.042 | upload | model.language_model.layers.19.mlp.down_proj.weight | 45 | 4.77123 | 5468.34 |
| 535.199 | alloc | model.language_model.layers.20.input_layernorm.weight | 0.00488281 | 0.009569 | 5468.34 |
| 535.199 | upload | model.language_model.layers.20.input_layernorm.weight | 0.00488281 | 0.006144 | 5468.34 |
| 539.497 | alloc | model.language_model.layers.20.linear_attn.in_proj_qkv.weight | 40 | 0.081896 | 5508.34 |
| 539.497 | upload | model.language_model.layers.20.linear_attn.in_proj_qkv.weight | 40 | 4.19098 | 5508.34 |
| 541.767 | alloc | model.language_model.layers.20.linear_attn.in_proj_z.weight | 20 | 0.075504 | 5528.34 |
| 541.767 | upload | model.language_model.layers.20.linear_attn.in_proj_z.weight | 20 | 2.09418 | 5528.34 |
| 541.871 | alloc | model.language_model.layers.20.linear_attn.in_proj_b.weight | 0.15625 | 0.006693 | 5528.5 |
| 541.871 | upload | model.language_model.layers.20.linear_attn.in_proj_b.weight | 0.15625 | 0.032 | 5528.5 |
| 541.963 | alloc | model.language_model.layers.20.linear_attn.in_proj_a.weight | 0.15625 | 0.005982 | 5528.65 |
| 541.963 | upload | model.language_model.layers.20.linear_attn.in_proj_a.weight | 0.15625 | 0.028896 | 5528.65 |
| 542.021 | alloc | model.language_model.layers.20.linear_attn.conv1d.weight | 0.0625 | 0.003076 | 5528.72 |
| 542.021 | upload | model.language_model.layers.20.linear_attn.conv1d.weight | 0.0625 | 0.012288 | 5528.72 |
| 542.051 | alloc | model.language_model.layers.20.linear_attn.A_log | 0.00012207 | 0.003667 | 5528.72 |
| 542.051 | upload | model.language_model.layers.20.linear_attn.A_log | 0.00012207 | 0.00512 | 5528.72 |
| 542.07 | alloc | model.language_model.layers.20.linear_attn.norm.weight | 0.000488281 | 0.008015 | 5528.72 |
| 542.071 | upload | model.language_model.layers.20.linear_attn.norm.weight | 0.000488281 | 0.005024 | 5528.72 |
| 542.091 | alloc | model.language_model.layers.20.linear_attn.dt_bias | 6.10352e-05 | 0.002475 | 5528.72 |
| 542.091 | upload | model.language_model.layers.20.linear_attn.dt_bias | 6.10352e-05 | 0.005344 | 5528.72 |
| 544.408 | alloc | model.language_model.layers.20.linear_attn.out_proj.weight | 20 | 0.087406 | 5548.72 |
| 544.409 | upload | model.language_model.layers.20.linear_attn.out_proj.weight | 20 | 2.10701 | 5548.72 |
| 544.508 | alloc | model.language_model.layers.20.post_attention_layernorm.weight | 0.00488281 | 0.007985 | 5548.72 |
| 544.508 | upload | model.language_model.layers.20.post_attention_layernorm.weight | 0.00488281 | 0.006144 | 5548.72 |
| 549.357 | alloc | model.language_model.layers.20.mlp.gate_proj.weight | 45 | 0.077687 | 5593.72 |
| 549.357 | upload | model.language_model.layers.20.mlp.gate_proj.weight | 45 | 4.7481 | 5593.72 |
| 554.288 | alloc | model.language_model.layers.20.mlp.up_proj.weight | 45 | 0.088287 | 5638.72 |
| 554.288 | upload | model.language_model.layers.20.mlp.up_proj.weight | 45 | 4.73882 | 5638.72 |
| 559.256 | alloc | model.language_model.layers.20.mlp.down_proj.weight | 45 | 0.11021 | 5683.72 |
| 559.259 | upload | model.language_model.layers.20.mlp.down_proj.weight | 45 | 4.74278 | 5683.72 |
| 559.409 | alloc | model.language_model.layers.21.input_layernorm.weight | 0.00488281 | 0.008726 | 5683.73 |
| 559.409 | upload | model.language_model.layers.21.input_layernorm.weight | 0.00488281 | 0.006144 | 5683.73 |
| 563.717 | alloc | model.language_model.layers.21.linear_attn.in_proj_qkv.weight | 40 | 0.086254 | 5723.73 |
| 563.717 | upload | model.language_model.layers.21.linear_attn.in_proj_qkv.weight | 40 | 4.19638 | 5723.73 |
| 565.991 | alloc | model.language_model.layers.21.linear_attn.in_proj_z.weight | 20 | 0.073169 | 5743.73 |
| 565.991 | upload | model.language_model.layers.21.linear_attn.in_proj_z.weight | 20 | 2.10016 | 5743.73 |
| 566.103 | alloc | model.language_model.layers.21.linear_attn.in_proj_b.weight | 0.15625 | 0.006632 | 5743.88 |
| 566.103 | upload | model.language_model.layers.21.linear_attn.in_proj_b.weight | 0.15625 | 0.038048 | 5743.88 |
| 566.201 | alloc | model.language_model.layers.21.linear_attn.in_proj_a.weight | 0.15625 | 0.003146 | 5744.04 |
| 566.201 | upload | model.language_model.layers.21.linear_attn.in_proj_a.weight | 0.15625 | 0.03152 | 5744.04 |
| 566.26 | alloc | model.language_model.layers.21.linear_attn.conv1d.weight | 0.0625 | 0.003126 | 5744.1 |
| 566.26 | upload | model.language_model.layers.21.linear_attn.conv1d.weight | 0.0625 | 0.013312 | 5744.1 |
| 566.292 | alloc | model.language_model.layers.21.linear_attn.A_log | 0.00012207 | 0.006422 | 5744.1 |
| 566.293 | upload | model.language_model.layers.21.linear_attn.A_log | 0.00012207 | 0.005472 | 5744.1 |
| 566.308 | alloc | model.language_model.layers.21.linear_attn.norm.weight | 0.000488281 | 0.004148 | 5744.1 |
| 566.308 | upload | model.language_model.layers.21.linear_attn.norm.weight | 0.000488281 | 0.005792 | 5744.1 |
| 566.321 | alloc | model.language_model.layers.21.linear_attn.dt_bias | 6.10352e-05 | 0.002204 | 5744.1 |
| 566.321 | upload | model.language_model.layers.21.linear_attn.dt_bias | 6.10352e-05 | 0.005952 | 5744.1 |
| 568.64 | alloc | model.language_model.layers.21.linear_attn.out_proj.weight | 20 | 0.084631 | 5764.1 |
| 568.641 | upload | model.language_model.layers.21.linear_attn.out_proj.weight | 20 | 2.11213 | 5764.1 |
| 568.741 | alloc | model.language_model.layers.21.post_attention_layernorm.weight | 0.00488281 | 0.00538 | 5764.11 |
| 568.741 | upload | model.language_model.layers.21.post_attention_layernorm.weight | 0.00488281 | 0.00512 | 5764.11 |
| 573.596 | alloc | model.language_model.layers.21.mlp.gate_proj.weight | 45 | 0.078129 | 5809.11 |
| 573.596 | upload | model.language_model.layers.21.mlp.gate_proj.weight | 45 | 4.75414 | 5809.11 |
| 578.587 | alloc | model.language_model.layers.21.mlp.up_proj.weight | 45 | 0.087616 | 5854.11 |
| 578.587 | upload | model.language_model.layers.21.mlp.up_proj.weight | 45 | 4.79875 | 5854.11 |
| 583.541 | alloc | model.language_model.layers.21.mlp.down_proj.weight | 45 | 0.081044 | 5899.11 |
| 583.541 | upload | model.language_model.layers.21.mlp.down_proj.weight | 45 | 4.75814 | 5899.11 |
| 583.695 | alloc | model.language_model.layers.22.input_layernorm.weight | 0.00488281 | 0.009638 | 5899.11 |
| 583.695 | upload | model.language_model.layers.22.input_layernorm.weight | 0.00488281 | 0.00608 | 5899.11 |
| 588.012 | alloc | model.language_model.layers.22.linear_attn.in_proj_qkv.weight | 40 | 0.07866 | 5939.11 |
| 588.012 | upload | model.language_model.layers.22.linear_attn.in_proj_qkv.weight | 40 | 4.21155 | 5939.11 |
| 590.302 | alloc | model.language_model.layers.22.linear_attn.in_proj_z.weight | 20 | 0.072889 | 5959.11 |
| 590.302 | upload | model.language_model.layers.22.linear_attn.in_proj_z.weight | 20 | 2.11411 | 5959.11 |
| 590.409 | alloc | model.language_model.layers.22.linear_attn.in_proj_b.weight | 0.15625 | 0.006833 | 5959.27 |
| 590.409 | upload | model.language_model.layers.22.linear_attn.in_proj_b.weight | 0.15625 | 0.031296 | 5959.27 |
| 590.499 | alloc | model.language_model.layers.22.linear_attn.in_proj_a.weight | 0.15625 | 0.002946 | 5959.42 |
| 590.499 | upload | model.language_model.layers.22.linear_attn.in_proj_a.weight | 0.15625 | 0.029184 | 5959.42 |
| 590.56 | alloc | model.language_model.layers.22.linear_attn.conv1d.weight | 0.0625 | 0.006292 | 5959.49 |
| 590.56 | upload | model.language_model.layers.22.linear_attn.conv1d.weight | 0.0625 | 0.012288 | 5959.49 |
| 590.589 | alloc | model.language_model.layers.22.linear_attn.A_log | 0.00012207 | 0.003747 | 5959.49 |
| 590.589 | upload | model.language_model.layers.22.linear_attn.A_log | 0.00012207 | 0.006016 | 5959.49 |
| 590.606 | alloc | model.language_model.layers.22.linear_attn.norm.weight | 0.000488281 | 0.004809 | 5959.49 |
| 590.606 | upload | model.language_model.layers.22.linear_attn.norm.weight | 0.000488281 | 0.006144 | 5959.49 |
| 590.621 | alloc | model.language_model.layers.22.linear_attn.dt_bias | 6.10352e-05 | 0.003987 | 5959.49 |
| 590.621 | upload | model.language_model.layers.22.linear_attn.dt_bias | 6.10352e-05 | 0.005984 | 5959.49 |
| 592.99 | alloc | model.language_model.layers.22.linear_attn.out_proj.weight | 20 | 0.085662 | 5979.49 |
| 592.99 | upload | model.language_model.layers.22.linear_attn.out_proj.weight | 20 | 2.14234 | 5979.49 |
| 593.094 | alloc | model.language_model.layers.22.post_attention_layernorm.weight | 0.00488281 | 0.006332 | 5979.49 |
| 593.094 | upload | model.language_model.layers.22.post_attention_layernorm.weight | 0.00488281 | 0.00512 | 5979.49 |
| 597.945 | alloc | model.language_model.layers.22.mlp.gate_proj.weight | 45 | 0.075764 | 6024.49 |
| 597.945 | upload | model.language_model.layers.22.mlp.gate_proj.weight | 45 | 4.75104 | 6024.49 |
| 602.957 | alloc | model.language_model.layers.22.mlp.up_proj.weight | 45 | 0.109438 | 6069.49 |
| 602.957 | upload | model.language_model.layers.22.mlp.up_proj.weight | 45 | 4.79674 | 6069.49 |
| 607.912 | alloc | model.language_model.layers.22.mlp.down_proj.weight | 45 | 0.087686 | 6114.49 |
| 607.912 | upload | model.language_model.layers.22.mlp.down_proj.weight | 45 | 4.7503 | 6114.49 |
| 608.077 | alloc | model.language_model.layers.23.input_layernorm.weight | 0.00488281 | 0.009468 | 6114.5 |
| 608.077 | upload | model.language_model.layers.23.input_layernorm.weight | 0.00488281 | 0.01024 | 6114.5 |
| 612.443 | alloc | model.language_model.layers.23.self_attn.q_proj.weight | 40 | 0.094089 | 6154.5 |
| 612.443 | upload | model.language_model.layers.23.self_attn.q_proj.weight | 40 | 4.2473 | 6154.5 |
| 613.181 | alloc | model.language_model.layers.23.self_attn.k_proj.weight | 5 | 0.074722 | 6159.5 |
| 613.181 | upload | model.language_model.layers.23.self_attn.k_proj.weight | 5 | 0.556608 | 6159.5 |
| 613.862 | alloc | model.language_model.layers.23.self_attn.v_proj.weight | 5 | 0.068681 | 6164.5 |
| 613.862 | upload | model.language_model.layers.23.self_attn.v_proj.weight | 5 | 0.550464 | 6164.5 |
| 613.921 | alloc | model.language_model.layers.23.self_attn.q_norm.weight | 0.000488281 | 0.007093 | 6164.5 |
| 613.921 | upload | model.language_model.layers.23.self_attn.q_norm.weight | 0.000488281 | 0.005824 | 6164.5 |
| 613.947 | alloc | model.language_model.layers.23.self_attn.k_norm.weight | 0.000488281 | 0.001894 | 6164.5 |
| 613.947 | upload | model.language_model.layers.23.self_attn.k_norm.weight | 0.000488281 | 0.00512 | 6164.5 |
| 616.201 | alloc | model.language_model.layers.23.self_attn.o_proj.weight | 20 | 0.078319 | 6184.5 |
| 616.201 | upload | model.language_model.layers.23.self_attn.o_proj.weight | 20 | 2.13395 | 6184.5 |
| 616.303 | alloc | model.language_model.layers.23.post_attention_layernorm.weight | 0.00488281 | 0.004218 | 6184.5 |
| 616.303 | upload | model.language_model.layers.23.post_attention_layernorm.weight | 0.00488281 | 0.006144 | 6184.5 |
| 621.173 | alloc | model.language_model.layers.23.mlp.gate_proj.weight | 45 | 0.085443 | 6229.5 |
| 621.173 | upload | model.language_model.layers.23.mlp.gate_proj.weight | 45 | 4.76016 | 6229.5 |
| 626.106 | alloc | model.language_model.layers.23.mlp.up_proj.weight | 45 | 0.080473 | 6274.5 |
| 626.106 | upload | model.language_model.layers.23.mlp.up_proj.weight | 45 | 4.74851 | 6274.5 |
| 631.049 | alloc | model.language_model.layers.23.mlp.down_proj.weight | 45 | 0.081004 | 6319.5 |
| 631.049 | upload | model.language_model.layers.23.mlp.down_proj.weight | 45 | 4.7479 | 6319.5 |
| 631.209 | alloc | model.language_model.layers.24.input_layernorm.weight | 0.00488281 | 0.009388 | 6319.51 |
| 631.209 | upload | model.language_model.layers.24.input_layernorm.weight | 0.00488281 | 0.006336 | 6319.51 |
| 635.553 | alloc | model.language_model.layers.24.linear_attn.in_proj_qkv.weight | 40 | 0.087697 | 6359.51 |
| 635.553 | upload | model.language_model.layers.24.linear_attn.in_proj_qkv.weight | 40 | 4.23075 | 6359.51 |
| 637.863 | alloc | model.language_model.layers.24.linear_attn.in_proj_z.weight | 20 | 0.074151 | 6379.51 |
| 637.863 | upload | model.language_model.layers.24.linear_attn.in_proj_z.weight | 20 | 2.13178 | 6379.51 |
| 638.03 | alloc | model.language_model.layers.24.linear_attn.in_proj_b.weight | 0.15625 | 0.064593 | 6379.66 |
| 638.03 | upload | model.language_model.layers.24.linear_attn.in_proj_b.weight | 0.15625 | 0.035168 | 6379.66 |
| 638.12 | alloc | model.language_model.layers.24.linear_attn.in_proj_a.weight | 0.15625 | 0.003517 | 6379.82 |
| 638.12 | upload | model.language_model.layers.24.linear_attn.in_proj_a.weight | 0.15625 | 0.02336 | 6379.82 |
| 638.18 | alloc | model.language_model.layers.24.linear_attn.conv1d.weight | 0.0625 | 0.00534 | 6379.88 |
| 638.18 | upload | model.language_model.layers.24.linear_attn.conv1d.weight | 0.0625 | 0.012288 | 6379.88 |
| 638.21 | alloc | model.language_model.layers.24.linear_attn.A_log | 0.00012207 | 0.004017 | 6379.88 |
| 638.213 | upload | model.language_model.layers.24.linear_attn.A_log | 0.00012207 | 0.005952 | 6379.88 |
| 638.23 | alloc | model.language_model.layers.24.linear_attn.norm.weight | 0.000488281 | 0.00505 | 6379.88 |
| 638.23 | upload | model.language_model.layers.24.linear_attn.norm.weight | 0.000488281 | 0.00512 | 6379.88 |
| 638.243 | alloc | model.language_model.layers.24.linear_attn.dt_bias | 6.10352e-05 | 0.002134 | 6379.88 |
| 638.243 | upload | model.language_model.layers.24.linear_attn.dt_bias | 6.10352e-05 | 0.00512 | 6379.88 |
| 640.573 | alloc | model.language_model.layers.24.linear_attn.out_proj.weight | 20 | 0.083199 | 6399.88 |
| 640.573 | upload | model.language_model.layers.24.linear_attn.out_proj.weight | 20 | 2.12326 | 6399.88 |
| 640.673 | alloc | model.language_model.layers.24.post_attention_layernorm.weight | 0.00488281 | 0.008085 | 6399.89 |
| 640.673 | upload | model.language_model.layers.24.post_attention_layernorm.weight | 0.00488281 | 0.00592 | 6399.89 |
| 645.558 | alloc | model.language_model.layers.24.mlp.gate_proj.weight | 45 | 0.097896 | 6444.89 |
| 645.558 | upload | model.language_model.layers.24.mlp.gate_proj.weight | 45 | 4.76205 | 6444.89 |
| 650.507 | alloc | model.language_model.layers.24.mlp.up_proj.weight | 45 | 0.087747 | 6489.89 |
| 650.507 | upload | model.language_model.layers.24.mlp.up_proj.weight | 45 | 4.75443 | 6489.89 |
| 655.494 | alloc | model.language_model.layers.24.mlp.down_proj.weight | 45 | 0.085954 | 6534.89 |
| 655.494 | upload | model.language_model.layers.24.mlp.down_proj.weight | 45 | 4.78131 | 6534.89 |
| 655.649 | alloc | model.language_model.layers.25.input_layernorm.weight | 0.00488281 | 0.009879 | 6534.89 |
| 655.65 | upload | model.language_model.layers.25.input_layernorm.weight | 0.00488281 | 0.005888 | 6534.89 |
| 660.047 | alloc | model.language_model.layers.25.linear_attn.in_proj_qkv.weight | 40 | 0.080182 | 6574.89 |
| 660.047 | upload | model.language_model.layers.25.linear_attn.in_proj_qkv.weight | 40 | 4.29258 | 6574.89 |
| 662.38 | alloc | model.language_model.layers.25.linear_attn.in_proj_z.weight | 20 | 0.070955 | 6594.89 |
| 662.38 | upload | model.language_model.layers.25.linear_attn.in_proj_z.weight | 20 | 2.15347 | 6594.89 |
| 662.487 | alloc | model.language_model.layers.25.linear_attn.in_proj_b.weight | 0.15625 | 0.006753 | 6595.05 |
| 662.489 | upload | model.language_model.layers.25.linear_attn.in_proj_b.weight | 0.15625 | 0.03328 | 6595.05 |
| 662.569 | alloc | model.language_model.layers.25.linear_attn.in_proj_a.weight | 0.15625 | 0.003536 | 6595.21 |
| 662.569 | upload | model.language_model.layers.25.linear_attn.in_proj_a.weight | 0.15625 | 0.022336 | 6595.21 |
| 662.629 | alloc | model.language_model.layers.25.linear_attn.conv1d.weight | 0.0625 | 0.003016 | 6595.27 |
| 662.629 | upload | model.language_model.layers.25.linear_attn.conv1d.weight | 0.0625 | 0.012288 | 6595.27 |
| 662.664 | alloc | model.language_model.layers.25.linear_attn.A_log | 0.00012207 | 0.005811 | 6595.27 |
| 662.664 | upload | model.language_model.layers.25.linear_attn.A_log | 0.00012207 | 0.00512 | 6595.27 |
| 662.681 | alloc | model.language_model.layers.25.linear_attn.norm.weight | 0.000488281 | 0.0051 | 6595.27 |
| 662.681 | upload | model.language_model.layers.25.linear_attn.norm.weight | 0.000488281 | 0.00512 | 6595.27 |
| 662.696 | alloc | model.language_model.layers.25.linear_attn.dt_bias | 6.10352e-05 | 0.004288 | 6595.27 |
| 662.696 | upload | model.language_model.layers.25.linear_attn.dt_bias | 6.10352e-05 | 0.00512 | 6595.27 |
| 665.058 | alloc | model.language_model.layers.25.linear_attn.out_proj.weight | 20 | 0.092235 | 6615.27 |
| 665.058 | upload | model.language_model.layers.25.linear_attn.out_proj.weight | 20 | 2.1464 | 6615.27 |
| 665.161 | alloc | model.language_model.layers.25.post_attention_layernorm.weight | 0.00488281 | 0.005601 | 6615.27 |
| 665.161 | upload | model.language_model.layers.25.post_attention_layernorm.weight | 0.00488281 | 0.006144 | 6615.27 |
| 670.11 | alloc | model.language_model.layers.25.mlp.gate_proj.weight | 45 | 0.079351 | 6660.27 |
| 670.11 | upload | model.language_model.layers.25.mlp.gate_proj.weight | 45 | 4.84518 | 6660.27 |
| 675.134 | alloc | model.language_model.layers.25.mlp.up_proj.weight | 45 | 0.08427 | 6705.27 |
| 675.134 | upload | model.language_model.layers.25.mlp.up_proj.weight | 45 | 4.83482 | 6705.27 |
| 680.209 | alloc | model.language_model.layers.25.mlp.down_proj.weight | 45 | 0.088087 | 6750.27 |
| 680.209 | upload | model.language_model.layers.25.mlp.down_proj.weight | 45 | 4.87091 | 6750.27 |
| 680.364 | alloc | model.language_model.layers.26.input_layernorm.weight | 0.00488281 | 0.012824 | 6750.28 |
| 680.364 | upload | model.language_model.layers.26.input_layernorm.weight | 0.00488281 | 0.006144 | 6750.28 |
| 684.787 | alloc | model.language_model.layers.26.linear_attn.in_proj_qkv.weight | 40 | 0.088358 | 6790.28 |
| 684.787 | upload | model.language_model.layers.26.linear_attn.in_proj_qkv.weight | 40 | 4.3104 | 6790.28 |
| 687.114 | alloc | model.language_model.layers.26.linear_attn.in_proj_z.weight | 20 | 0.073089 | 6810.28 |
| 687.114 | upload | model.language_model.layers.26.linear_attn.in_proj_z.weight | 20 | 2.15251 | 6810.28 |
| 687.22 | alloc | model.language_model.layers.26.linear_attn.in_proj_b.weight | 0.15625 | 0.006743 | 6810.43 |
| 687.22 | upload | model.language_model.layers.26.linear_attn.in_proj_b.weight | 0.15625 | 0.03264 | 6810.43 |
| 687.306 | alloc | model.language_model.layers.26.linear_attn.in_proj_a.weight | 0.15625 | 0.003476 | 6810.59 |
| 687.306 | upload | model.language_model.layers.26.linear_attn.in_proj_a.weight | 0.15625 | 0.026592 | 6810.59 |
| 687.371 | alloc | model.language_model.layers.26.linear_attn.conv1d.weight | 0.0625 | 0.003717 | 6810.65 |
| 687.371 | upload | model.language_model.layers.26.linear_attn.conv1d.weight | 0.0625 | 0.012288 | 6810.65 |
| 687.404 | alloc | model.language_model.layers.26.linear_attn.A_log | 0.00012207 | 0.006923 | 6810.65 |
| 687.404 | upload | model.language_model.layers.26.linear_attn.A_log | 0.00012207 | 0.00608 | 6810.65 |
| 687.423 | alloc | model.language_model.layers.26.linear_attn.norm.weight | 0.000488281 | 0.00534 | 6810.65 |
| 687.423 | upload | model.language_model.layers.26.linear_attn.norm.weight | 0.000488281 | 0.00512 | 6810.65 |
| 687.436 | alloc | model.language_model.layers.26.linear_attn.dt_bias | 6.10352e-05 | 0.002154 | 6810.65 |
| 687.437 | upload | model.language_model.layers.26.linear_attn.dt_bias | 6.10352e-05 | 0.004992 | 6810.65 |
| 689.92 | alloc | model.language_model.layers.26.linear_attn.out_proj.weight | 20 | 0.117163 | 6830.65 |
| 689.92 | upload | model.language_model.layers.26.linear_attn.out_proj.weight | 20 | 2.24384 | 6830.65 |
| 690.024 | alloc | model.language_model.layers.26.post_attention_layernorm.weight | 0.00488281 | 0.006813 | 6830.66 |
| 690.024 | upload | model.language_model.layers.26.post_attention_layernorm.weight | 0.00488281 | 0.00592 | 6830.66 |
| 695.117 | alloc | model.language_model.layers.26.mlp.gate_proj.weight | 45 | 0.086024 | 6875.66 |
| 695.124 | upload | model.language_model.layers.26.mlp.gate_proj.weight | 45 | 4.98006 | 6875.66 |
| 700.131 | alloc | model.language_model.layers.26.mlp.up_proj.weight | 45 | 0.090642 | 6920.66 |
| 700.131 | upload | model.language_model.layers.26.mlp.up_proj.weight | 45 | 4.80909 | 6920.66 |
| 705.094 | alloc | model.language_model.layers.26.mlp.down_proj.weight | 45 | 0.087046 | 6965.66 |
| 705.094 | upload | model.language_model.layers.26.mlp.down_proj.weight | 45 | 4.75818 | 6965.66 |
| 705.245 | alloc | model.language_model.layers.27.input_layernorm.weight | 0.00488281 | 0.009578 | 6965.66 |
| 705.245 | upload | model.language_model.layers.27.input_layernorm.weight | 0.00488281 | 0.006144 | 6965.66 |
| 709.575 | alloc | model.language_model.layers.27.self_attn.q_proj.weight | 40 | 0.087557 | 7005.66 |
| 709.575 | upload | model.language_model.layers.27.self_attn.q_proj.weight | 40 | 4.21718 | 7005.66 |
| 710.311 | alloc | model.language_model.layers.27.self_attn.k_proj.weight | 5 | 0.076776 | 7010.66 |
| 710.311 | upload | model.language_model.layers.27.self_attn.k_proj.weight | 5 | 0.559232 | 7010.66 |
| 711.004 | alloc | model.language_model.layers.27.self_attn.v_proj.weight | 5 | 0.078339 | 7015.66 |
| 711.004 | upload | model.language_model.layers.27.self_attn.v_proj.weight | 5 | 0.553696 | 7015.66 |
| 711.067 | alloc | model.language_model.layers.27.self_attn.q_norm.weight | 0.000488281 | 0.007755 | 7015.66 |
| 711.067 | upload | model.language_model.layers.27.self_attn.q_norm.weight | 0.000488281 | 0.006144 | 7015.66 |
| 711.103 | alloc | model.language_model.layers.27.self_attn.k_norm.weight | 0.000488281 | 0.005381 | 7015.66 |
| 711.103 | upload | model.language_model.layers.27.self_attn.k_norm.weight | 0.000488281 | 0.005056 | 7015.66 |
| 713.403 | alloc | model.language_model.layers.27.self_attn.o_proj.weight | 20 | 0.081625 | 7035.66 |
| 713.404 | upload | model.language_model.layers.27.self_attn.o_proj.weight | 20 | 2.18035 | 7035.66 |
| 713.502 | alloc | model.language_model.layers.27.post_attention_layernorm.weight | 0.00488281 | 0.004799 | 7035.67 |
| 713.502 | upload | model.language_model.layers.27.post_attention_layernorm.weight | 0.00488281 | 0.00512 | 7035.67 |
| 718.381 | alloc | model.language_model.layers.27.mlp.gate_proj.weight | 45 | 0.082587 | 7080.67 |
| 718.381 | upload | model.language_model.layers.27.mlp.gate_proj.weight | 45 | 4.77386 | 7080.67 |
| 723.41 | alloc | model.language_model.layers.27.mlp.up_proj.weight | 45 | 0.083458 | 7125.67 |
| 723.41 | upload | model.language_model.layers.27.mlp.up_proj.weight | 45 | 4.83728 | 7125.67 |
| 728.367 | alloc | model.language_model.layers.27.mlp.down_proj.weight | 45 | 0.087547 | 7170.67 |
| 728.368 | upload | model.language_model.layers.27.mlp.down_proj.weight | 45 | 4.75424 | 7170.67 |
| 728.524 | alloc | model.language_model.layers.28.input_layernorm.weight | 0.00488281 | 0.013746 | 7170.67 |
| 728.524 | upload | model.language_model.layers.28.input_layernorm.weight | 0.00488281 | 0.005792 | 7170.67 |
| 732.855 | alloc | model.language_model.layers.28.linear_attn.in_proj_qkv.weight | 40 | 0.101503 | 7210.67 |
| 732.855 | upload | model.language_model.layers.28.linear_attn.in_proj_qkv.weight | 40 | 4.20419 | 7210.67 |
| 735.152 | alloc | model.language_model.layers.28.linear_attn.in_proj_z.weight | 20 | 0.076536 | 7230.67 |
| 735.152 | upload | model.language_model.layers.28.linear_attn.in_proj_z.weight | 20 | 2.11418 | 7230.67 |
| 735.259 | alloc | model.language_model.layers.28.linear_attn.in_proj_b.weight | 0.15625 | 0.007735 | 7230.83 |
| 735.259 | upload | model.language_model.layers.28.linear_attn.in_proj_b.weight | 0.15625 | 0.031072 | 7230.83 |
| 735.344 | alloc | model.language_model.layers.28.linear_attn.in_proj_a.weight | 0.15625 | 0.002805 | 7230.99 |
| 735.344 | upload | model.language_model.layers.28.linear_attn.in_proj_a.weight | 0.15625 | 0.026688 | 7230.99 |
| 735.406 | alloc | model.language_model.layers.28.linear_attn.conv1d.weight | 0.0625 | 0.003517 | 7231.05 |
| 735.406 | upload | model.language_model.layers.28.linear_attn.conv1d.weight | 0.0625 | 0.013088 | 7231.05 |
| 735.439 | alloc | model.language_model.layers.28.linear_attn.A_log | 0.00012207 | 0.006272 | 7231.05 |
| 735.439 | upload | model.language_model.layers.28.linear_attn.A_log | 0.00012207 | 0.00592 | 7231.05 |
| 735.458 | alloc | model.language_model.layers.28.linear_attn.norm.weight | 0.000488281 | 0.007564 | 7231.05 |
| 735.458 | upload | model.language_model.layers.28.linear_attn.norm.weight | 0.000488281 | 0.00512 | 7231.05 |
| 735.472 | alloc | model.language_model.layers.28.linear_attn.dt_bias | 6.10352e-05 | 0.002445 | 7231.05 |
| 735.472 | upload | model.language_model.layers.28.linear_attn.dt_bias | 6.10352e-05 | 0.004928 | 7231.05 |
| 737.794 | alloc | model.language_model.layers.28.linear_attn.out_proj.weight | 20 | 0.085393 | 7251.05 |
| 737.795 | upload | model.language_model.layers.28.linear_attn.out_proj.weight | 20 | 2.11283 | 7251.05 |
| 737.89 | alloc | model.language_model.layers.28.post_attention_layernorm.weight | 0.00488281 | 0.003807 | 7251.05 |
| 737.89 | upload | model.language_model.layers.28.post_attention_layernorm.weight | 0.00488281 | 0.00512 | 7251.05 |
| 742.745 | alloc | model.language_model.layers.28.mlp.gate_proj.weight | 45 | 0.083579 | 7296.05 |
| 742.746 | upload | model.language_model.layers.28.mlp.gate_proj.weight | 45 | 4.74822 | 7296.05 |
| 747.71 | alloc | model.language_model.layers.28.mlp.up_proj.weight | 45 | 0.082187 | 7341.05 |
| 747.711 | upload | model.language_model.layers.28.mlp.up_proj.weight | 45 | 4.77427 | 7341.05 |
| 752.662 | alloc | model.language_model.layers.28.mlp.down_proj.weight | 45 | 0.082908 | 7386.05 |
| 752.662 | upload | model.language_model.layers.28.mlp.down_proj.weight | 45 | 4.75456 | 7386.05 |
| 752.815 | alloc | model.language_model.layers.29.input_layernorm.weight | 0.00488281 | 0.010009 | 7386.06 |
| 752.815 | upload | model.language_model.layers.29.input_layernorm.weight | 0.00488281 | 0.007104 | 7386.06 |
| 757.166 | alloc | model.language_model.layers.29.linear_attn.in_proj_qkv.weight | 40 | 0.08413 | 7426.06 |
| 757.167 | upload | model.language_model.layers.29.linear_attn.in_proj_qkv.weight | 40 | 4.24243 | 7426.06 |
| 759.467 | alloc | model.language_model.layers.29.linear_attn.in_proj_z.weight | 20 | 0.080222 | 7446.06 |
| 759.467 | upload | model.language_model.layers.29.linear_attn.in_proj_z.weight | 20 | 2.11907 | 7446.06 |
| 759.574 | alloc | model.language_model.layers.29.linear_attn.in_proj_b.weight | 0.15625 | 0.007495 | 7446.22 |
| 759.574 | upload | model.language_model.layers.29.linear_attn.in_proj_b.weight | 0.15625 | 0.03248 | 7446.22 |
| 759.659 | alloc | model.language_model.layers.29.linear_attn.in_proj_a.weight | 0.15625 | 0.005721 | 7446.37 |
| 759.659 | upload | model.language_model.layers.29.linear_attn.in_proj_a.weight | 0.15625 | 0.02592 | 7446.37 |
| 759.72 | alloc | model.language_model.layers.29.linear_attn.conv1d.weight | 0.0625 | 0.003176 | 7446.43 |
| 759.72 | upload | model.language_model.layers.29.linear_attn.conv1d.weight | 0.0625 | 0.012288 | 7446.43 |
| 759.755 | alloc | model.language_model.layers.29.linear_attn.A_log | 0.00012207 | 0.009137 | 7446.43 |
| 759.755 | upload | model.language_model.layers.29.linear_attn.A_log | 0.00012207 | 0.005472 | 7446.43 |
| 759.773 | alloc | model.language_model.layers.29.linear_attn.norm.weight | 0.000488281 | 0.004509 | 7446.44 |
| 759.775 | upload | model.language_model.layers.29.linear_attn.norm.weight | 0.000488281 | 0.00512 | 7446.44 |
| 759.789 | alloc | model.language_model.layers.29.linear_attn.dt_bias | 6.10352e-05 | 0.002274 | 7446.44 |
| 759.789 | upload | model.language_model.layers.29.linear_attn.dt_bias | 6.10352e-05 | 0.00512 | 7446.44 |
| 762.144 | alloc | model.language_model.layers.29.linear_attn.out_proj.weight | 20 | 0.084661 | 7466.44 |
| 762.144 | upload | model.language_model.layers.29.linear_attn.out_proj.weight | 20 | 2.14717 | 7466.44 |
| 762.243 | alloc | model.language_model.layers.29.post_attention_layernorm.weight | 0.00488281 | 0.006142 | 7466.44 |
| 762.243 | upload | model.language_model.layers.29.post_attention_layernorm.weight | 0.00488281 | 0.006144 | 7466.44 |
| 767.087 | alloc | model.language_model.layers.29.mlp.gate_proj.weight | 45 | 0.079131 | 7511.44 |
| 767.087 | upload | model.language_model.layers.29.mlp.gate_proj.weight | 45 | 4.74256 | 7511.44 |
| 772.014 | alloc | model.language_model.layers.29.mlp.up_proj.weight | 45 | 0.087106 | 7556.44 |
| 772.014 | upload | model.language_model.layers.29.mlp.up_proj.weight | 45 | 4.7343 | 7556.44 |
| 776.997 | alloc | model.language_model.layers.29.mlp.down_proj.weight | 45 | 0.104709 | 7601.44 |
| 776.997 | upload | model.language_model.layers.29.mlp.down_proj.weight | 45 | 4.76214 | 7601.44 |
| 777.148 | alloc | model.language_model.layers.30.input_layernorm.weight | 0.00488281 | 0.009518 | 7601.45 |
| 777.148 | upload | model.language_model.layers.30.input_layernorm.weight | 0.00488281 | 0.006144 | 7601.45 |
| 781.487 | alloc | model.language_model.layers.30.linear_attn.in_proj_qkv.weight | 40 | 0.081164 | 7641.45 |
| 781.487 | upload | model.language_model.layers.30.linear_attn.in_proj_qkv.weight | 40 | 4.23363 | 7641.45 |
| 783.798 | alloc | model.language_model.layers.30.linear_attn.in_proj_z.weight | 20 | 0.079441 | 7661.45 |
| 783.798 | upload | model.language_model.layers.30.linear_attn.in_proj_z.weight | 20 | 2.13171 | 7661.45 |
| 783.907 | alloc | model.language_model.layers.30.linear_attn.in_proj_b.weight | 0.15625 | 0.010611 | 7661.6 |
| 783.907 | upload | model.language_model.layers.30.linear_attn.in_proj_b.weight | 0.15625 | 0.032576 | 7661.6 |
| 784.07 | alloc | model.language_model.layers.30.linear_attn.in_proj_a.weight | 0.15625 | 0.075403 | 7661.76 |
| 784.07 | upload | model.language_model.layers.30.linear_attn.in_proj_a.weight | 0.15625 | 0.026816 | 7661.76 |
| 784.164 | alloc | model.language_model.layers.30.linear_attn.conv1d.weight | 0.0625 | 0.004619 | 7661.82 |
| 784.164 | upload | model.language_model.layers.30.linear_attn.conv1d.weight | 0.0625 | 0.012224 | 7661.82 |
| 784.2 | alloc | model.language_model.layers.30.linear_attn.A_log | 0.00012207 | 0.009488 | 7661.82 |
| 784.2 | upload | model.language_model.layers.30.linear_attn.A_log | 0.00012207 | 0.00512 | 7661.82 |
| 784.219 | alloc | model.language_model.layers.30.linear_attn.norm.weight | 0.000488281 | 0.007805 | 7661.82 |
| 784.219 | upload | model.language_model.layers.30.linear_attn.norm.weight | 0.000488281 | 0.00512 | 7661.82 |
| 784.232 | alloc | model.language_model.layers.30.linear_attn.dt_bias | 6.10352e-05 | 0.001914 | 7661.82 |
| 784.232 | upload | model.language_model.layers.30.linear_attn.dt_bias | 6.10352e-05 | 0.00512 | 7661.82 |
| 786.555 | alloc | model.language_model.layers.30.linear_attn.out_proj.weight | 20 | 0.085923 | 7681.82 |
| 786.555 | upload | model.language_model.layers.30.linear_attn.out_proj.weight | 20 | 2.11446 | 7681.82 |
| 786.655 | alloc | model.language_model.layers.30.post_attention_layernorm.weight | 0.00488281 | 0.006693 | 7681.83 |
| 786.655 | upload | model.language_model.layers.30.post_attention_layernorm.weight | 0.00488281 | 0.005824 | 7681.83 |
| 791.507 | alloc | model.language_model.layers.30.mlp.gate_proj.weight | 45 | 0.080914 | 7726.83 |
| 791.508 | upload | model.language_model.layers.30.mlp.gate_proj.weight | 45 | 4.74864 | 7726.83 |
| 796.442 | alloc | model.language_model.layers.30.mlp.up_proj.weight | 45 | 0.081144 | 7771.83 |
| 796.442 | upload | model.language_model.layers.30.mlp.up_proj.weight | 45 | 4.75085 | 7771.83 |
| 801.406 | alloc | model.language_model.layers.30.mlp.down_proj.weight | 45 | 0.081324 | 7816.83 |
| 801.406 | upload | model.language_model.layers.30.mlp.down_proj.weight | 45 | 4.76778 | 7816.83 |
| 801.559 | alloc | model.language_model.layers.31.input_layernorm.weight | 0.00488281 | 0.009949 | 7816.83 |
| 801.559 | upload | model.language_model.layers.31.input_layernorm.weight | 0.00488281 | 0.00624 | 7816.83 |
| 805.866 | alloc | model.language_model.layers.31.self_attn.q_proj.weight | 40 | 0.090372 | 7856.83 |
| 805.866 | upload | model.language_model.layers.31.self_attn.q_proj.weight | 40 | 4.19267 | 7856.83 |
| 806.594 | alloc | model.language_model.layers.31.self_attn.k_proj.weight | 5 | 0.071246 | 7861.83 |
| 806.594 | upload | model.language_model.layers.31.self_attn.k_proj.weight | 5 | 0.555936 | 7861.83 |
| 807.274 | alloc | model.language_model.layers.31.self_attn.v_proj.weight | 5 | 0.067278 | 7866.83 |
| 807.274 | upload | model.language_model.layers.31.self_attn.v_proj.weight | 5 | 0.553056 | 7866.83 |
| 807.34 | alloc | model.language_model.layers.31.self_attn.q_norm.weight | 0.000488281 | 0.01026 | 7866.83 |
| 807.34 | upload | model.language_model.layers.31.self_attn.q_norm.weight | 0.000488281 | 0.006144 | 7866.83 |
| 807.37 | alloc | model.language_model.layers.31.self_attn.k_norm.weight | 0.000488281 | 0.002966 | 7866.83 |
| 807.37 | upload | model.language_model.layers.31.self_attn.k_norm.weight | 0.000488281 | 0.00512 | 7866.83 |
| 809.617 | alloc | model.language_model.layers.31.self_attn.o_proj.weight | 20 | 0.082637 | 7886.83 |
| 809.618 | upload | model.language_model.layers.31.self_attn.o_proj.weight | 20 | 2.1256 | 7886.83 |
| 809.717 | alloc | model.language_model.layers.31.post_attention_layernorm.weight | 0.00488281 | 0.006222 | 7886.84 |
| 809.719 | upload | model.language_model.layers.31.post_attention_layernorm.weight | 0.00488281 | 0.006144 | 7886.84 |
| 814.615 | alloc | model.language_model.layers.31.mlp.gate_proj.weight | 45 | 0.085292 | 7931.84 |
| 814.615 | upload | model.language_model.layers.31.mlp.gate_proj.weight | 45 | 4.7872 | 7931.84 |
| 819.578 | alloc | model.language_model.layers.31.mlp.up_proj.weight | 45 | 0.098557 | 7976.84 |
| 819.578 | upload | model.language_model.layers.31.mlp.up_proj.weight | 45 | 4.75267 | 7976.84 |
| 824.531 | alloc | model.language_model.layers.31.mlp.down_proj.weight | 45 | 0.085543 | 8021.84 |
| 824.531 | upload | model.language_model.layers.31.mlp.down_proj.weight | 45 | 4.75024 | 8021.84 |
| 824.68 | alloc | model.language_model.norm.weight | 0.00488281 | 0.009227 | 8021.84 |
| 824.68 | upload | model.language_model.norm.weight | 0.00488281 | 0.006112 | 8021.84 |

