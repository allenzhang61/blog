# llama.cpp nsys GPU 剖析 (llama-bench tg64 + pp15, -fa on -ngl 99, RTX 3080)

build: bb4caa7 | nsys 2023.4.4 | 本次 bench: pp15=716.08 t/s, tg64=85.98 t/s


NOTICE: Existing SQLite export found: llama_nsys2.sqlite
        It is assumed file was previously exported from: llama_nsys2.nsys-rep
        Consider using --force-export=true if needed.

Processing [llama_nsys2.sqlite] with [/usr/lib/x86_64-linux-gnu/nsight-systems/target-linux-x64/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

+----------+-----------------+-----------+----------+----------+----------+-----------+-------------+------------------------------------------------------------------------------------------------------+
| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns)  | StdDev (ns) |                                                 Name                                                 |
+----------+-----------------+-----------+----------+----------+----------+-----------+-------------+------------------------------------------------------------------------------------------------------+
|     30.0 |      11,294,113 |       144 | 78,431.3 | 58,352.5 |   27,072 |   112,065 |    30,989.7 | void mul_mat_vec_f<__nv_bfloat16, float, (int)1, (int)256, (bool)1, (bool)0>(const T1 *, const floa… |
|     28.9 |      10,889,462 |       291 | 37,420.8 | 26,528.0 |    2,176 | 1,469,512 |   147,563.9 | void mul_mat_vec_f<__nv_bfloat16, float, (int)1, (int)256, (bool)0, (bool)0>(const T1 *, const floa… |
|     19.4 |       7,292,551 |       184 | 39,633.4 | 54,128.5 |    9,408 |    64,768 |    23,258.7 | void mul_mat_f<__nv_bfloat162, (int)32, (int)15, (int)7, (bool)0>(const T1 *, const float *, const … |
|      8.9 |       3,363,666 |        64 | 52,557.3 | 52,976.5 |   32,704 |    72,481 |    18,580.3 | void mul_mat_f<__nv_bfloat162, (int)32, (int)15, (int)8, (bool)0>(const T1 *, const float *, const … |
|      2.0 |         749,921 |        72 | 10,415.6 |  3,872.0 |    3,807 |    23,680 |     9,338.1 | void gated_delta_net_cuda<(int)128, (bool)0, (bool)0>(const float *, const float *, const float *, … |
|      2.0 |         737,733 |       195 |  3,783.2 |  3,584.0 |    3,360 |     4,736 |       341.5 | void rms_norm_f32<(int)1024, (bool)1, (bool)0>(const float *, float *, int, long, long, long, float… |
|      1.0 |         371,300 |       184 |  2,017.9 |  2,128.0 |    1,504 |     2,432 |       288.9 | void k_bin_bcast<&op_add, float, float, float, const float *>(const T2 *, const T3 *, T4 *, unsigne… |
|      0.9 |         338,468 |        96 |  3,525.7 |  3,712.0 |    1,471 |     5,664 |     1,867.4 | scale_f32(const float *, float *, float, float, long)                                                |
|      0.8 |         304,065 |        24 | 12,669.4 | 12,736.0 |   12,224 |    13,184 |       268.7 | void concat_non_cont<unsigned int, (int)0>(const char *, const char *, char *, long, long, long, lo… |
|      0.7 |         267,623 |       120 |  2,230.2 |  1,792.0 |    1,696 |     3,264 |       585.8 | void rms_norm_f32<(int)256, (bool)1, (bool)0>(const float *, float *, int, long, long, long, float,… |
|      0.7 |         248,549 |        72 |  3,452.1 |  3,008.0 |    2,880 |     4,480 |       693.9 | void k_get_rows_float_vec<float>(const T1 *, const int *, T1 *, long, long, uint3, unsigned long, u… |
|      0.6 |         240,863 |       104 |  2,316.0 |  2,176.0 |    1,760 |     3,168 |       532.4 | void unary_gated_op_kernel<&op_silu, float>(const T2 *, const T2 *, T2 *, long, long, long, long)    |
|      0.6 |         217,889 |        96 |  2,269.7 |  2,287.5 |    1,888 |     2,944 |       207.1 | void cpy_scalar<&cpy_1_scalar<float, float>>(const char *, char *, long, long, long, long, long, lo… |
|      0.6 |         207,998 |       144 |  1,444.4 |  1,408.0 |    1,375 |     1,632 |        73.3 | void l2_norm_f32<(int)32>(const float *, float *, int, long, long, long, float)                      |
|      0.4 |         168,291 |        16 | 10,518.2 | 10,496.5 |   10,272 |    10,784 |       164.3 | void flash_attn_ext_f16<(int)256, (int)256, (int)2, (int)4, (bool)0, (bool)0>(const char *, const c… |
|      0.4 |         157,572 |        72 |  2,188.5 |  1,888.0 |    1,472 |     3,296 |       730.5 | void ssm_conv_f32<(bool)1, (unsigned long)128, (unsigned long)4>(const float *, const float *, cons… |
|      0.4 |         141,406 |        75 |  1,885.4 |  1,760.0 |    1,664 |     2,368 |       235.9 | void k_get_rows_float<float, float>(const T1 *, const int *, T2 *, long, long, uint3, unsigned long… |
|      0.3 |         116,929 |         8 | 14,616.1 | 14,544.5 |   14,368 |    15,264 |       302.5 | void flash_attn_ext_f16<(int)256, (int)256, (int)16, (int)4, (bool)0, (bool)0>(const char *, const … |
|      0.3 |         113,534 |        48 |  2,365.3 |  2,368.0 |    2,240 |     2,592 |        72.9 | void concat_cont<unsigned int, (int)0>(const T1 *, const T1 *, T1 *, long, long, long, long, long, … |
|      0.2 |          82,016 |        48 |  1,708.7 |  1,680.0 |    1,472 |     2,081 |       195.9 | void rope_multi<(bool)1, (bool)0, float>(const T3 *, T3 *, int, int, int, int, int, int, int, int, … |
|      0.2 |          78,594 |        72 |  1,091.6 |  1,088.0 |    1,056 |     1,121 |        16.6 | void unary_op_kernel<&op_sigmoid, float>(const T2 *, T2 *, int)                                      |
|      0.2 |          77,855 |        48 |  1,622.0 |  1,600.0 |    1,568 |     1,856 |        69.3 | void k_set_rows<float, long, __half>(const T1 *, const T2 *, T3 *, long, long, long, long, long, lo… |
|      0.2 |          65,986 |        48 |  1,374.7 |  1,376.0 |    1,312 |     1,409 |        26.4 | void unary_gated_op_kernel<&op_softplus, float>(const T2 *, const T2 *, T2 *, long, long, long, lon… |
|      0.1 |          39,200 |        24 |  1,633.3 |  1,632.0 |    1,568 |     1,696 |        43.8 | void k_bin_bcast<&op_mul, float, float, float, const float *>(const T2 *, const T3 *, T4 *, unsigne… |
|      0.1 |          32,512 |        24 |  1,354.7 |  1,248.0 |    1,248 |     1,568 |       151.5 | void unary_gated_op_kernel<&op_sigmoid, float>(const T2 *, const T2 *, T2 *, long, long, long, long) |
|      0.1 |          26,721 |        16 |  1,670.1 |  1,664.0 |    1,632 |     1,696 |        17.4 | void flash_attn_stream_k_fixup_uniform<(int)256, (int)2, (int)4>(float *, const float2 *, int, int,… |
|      0.1 |          26,496 |        24 |  1,104.0 |  1,104.0 |    1,087 |     1,121 |        16.3 | void unary_op_kernel<&op_softplus, float>(const T2 *, T2 *, int)                                     |
|      0.1 |          21,086 |         8 |  2,635.8 |  2,624.0 |    2,623 |     2,656 |        16.8 | void flash_attn_stream_k_fixup_uniform<(int)256, (int)16, (int)4>(float *, const float2 *, int, int… |
+----------+-----------------+-----------+----------+----------+----------+-----------+-------------+------------------------------------------------------------------------------------------------------+



NOTICE: Existing SQLite export found: llama_nsys2.sqlite
        It is assumed file was previously exported from: llama_nsys2.nsys-rep
        Consider using --force-export=true if needed.

Processing [llama_nsys2.sqlite] with [/usr/lib/x86_64-linux-gnu/nsight-systems/target-linux-x64/reports/cuda_gpu_mem_time_sum.py]... 

 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

+----------+-----------------+-------+-----------+----------+----------+-------------+-------------+------------------------------+
| Time (%) | Total Time (ns) | Count | Avg (ns)  | Med (ns) | Min (ns) |  Max (ns)   | StdDev (ns) |          Operation           |
+----------+-----------------+-------+-----------+----------+----------+-------------+-------------+------------------------------+
|     99.7 |     786,791,968 |   895 | 879,097.2 |    416.0 |      351 | 118,144,735 | 4,214,781.0 | [CUDA memcpy Host-to-Device] |
|      0.3 |       2,578,451 |    67 |  38,484.3 | 38,336.0 |   38,240 |      43,745 |       702.0 | [CUDA memcpy Device-to-Host] |
|      0.0 |         144,097 |     4 |  36,024.3 | 36,400.0 |    8,992 |      62,305 |    30,243.0 | [CUDA memset]                |
+----------+-----------------+-------+-----------+----------+----------+-------------+-------------+------------------------------+

