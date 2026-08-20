# GENERATED — the CUDA sources hoid emits for the model's builtin
# (non-custom) ops, with the Llama-3.1-8B-Instruct constants baked in. Do
# not edit the kernel bodies.
#
# These are the builtin kernels the winning stack launches besides cuBLAS
# matmuls: embedding, apply_rope (a 32-head Q variant and an 8-head KV
# variant; the two bodies differ only in the head count), and cache_write.
# Baked constants: HIDDEN=4096, N_Q_HEADS=32, N_KV_HEADS=8, HEAD_DIM=128,
# with the dynamic dims at the kernels' fixed dyn_dims indices: seq 's' =
# dyn_dims[18], offset 'o' = dyn_dims[14]. Entry names carry a _q/_kv
# suffix here because the port compiles each variant as its own extension
# module.
#
# {MAX_SEQ} is the one bake-in that depends on the port's cache size
# (cache_write's row stride, 9216 here).

EMBEDDING = r"""#include <cuda_bf16.h>
extern "C" __global__ void embedding_k(__nv_bfloat16* out, const __nv_bfloat16* in_0, const int* in_1, const int* dyn_dims) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int H = 4096;
    int n = dyn_dims[18] * H;
    if (idx >= n) return;
    int row = idx / H, col = idx % H;
    int id = in_1[row];
    out[idx] = in_0[(long)id * H + col];
}
"""

# apply_rope source for the 32-head Q projection
APPLY_ROPE_Q = r"""#include <cuda_bf16.h>
#define TO_F(x) __bfloat162float(x)
#define FROM_F(x) __float2bfloat16(x)
extern "C" __global__ void apply_rope_q_k(
    __nv_bfloat16* out,
    const __nv_bfloat16* x_in,
    const float* cos_table,
    const float* sin_table,
    const int* dyn_dims
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = ((dyn_dims[18] * 32) * 128);
    if (idx >= n) return;
    int d = idx % 128;
    int tmp = idx / 128;
    int h = tmp % 32;
    int tok = tmp / 32;
    int pair = d < 64 ? d : d - 64;
    int base = (tok * 32 + h) * 128;
    float a = TO_F(x_in[base + pair]);
    float b = TO_F(x_in[base + pair + 64]);
    int pos = dyn_dims[14] + tok;
    float c = cos_table[(long)pos * 64 + pair];
    float s = sin_table[(long)pos * 64 + pair];
    float y = d < 64 ? a * c - b * s : b * c + a * s;
    out[idx] = FROM_F(y);
}
"""

# apply_rope source for the 8-head K projection
APPLY_ROPE_KV = r"""#include <cuda_bf16.h>
#define TO_F(x) __bfloat162float(x)
#define FROM_F(x) __float2bfloat16(x)
extern "C" __global__ void apply_rope_kv_k(
    __nv_bfloat16* out,
    const __nv_bfloat16* x_in,
    const float* cos_table,
    const float* sin_table,
    const int* dyn_dims
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = ((dyn_dims[18] * 8) * 128);
    if (idx >= n) return;
    int d = idx % 128;
    int tmp = idx / 128;
    int h = tmp % 8;
    int tok = tmp / 8;
    int pair = d < 64 ? d : d - 64;
    int base = (tok * 8 + h) * 128;
    float a = TO_F(x_in[base + pair]);
    float b = TO_F(x_in[base + pair + 64]);
    int pos = dyn_dims[14] + tok;
    float c = cos_table[(long)pos * 64 + pair];
    float s = sin_table[(long)pos * 64 + pair];
    float y = d < 64 ? a * c - b * s : b * c + a * s;
    out[idx] = FROM_F(y);
}
"""

# cache_write source (8 KV heads; in-place on the cache — out aliases the
# cache buffer, the port passes the cache twice)
CACHE_WRITE = r"""#include <cuda_bf16.h>
extern "C" __global__ void cache_write_k(
    __nv_bfloat16* out,
    __nv_bfloat16* cache,
    const __nv_bfloat16* src,
    const int* dyn_dims
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int src_n = ((dyn_dims[18] * 8) * 128);
    int off = dyn_dims[14];
    if (idx < src_n) {
        int d = idx % 128;
        int tmp = idx / 128;
        int h = tmp % 8;
        int tok = tmp / 8;
        int cache_idx = ((h * {MAX_SEQ} + off + tok) * 128) + d;
        out[cache_idx] = src[idx];
    }
}
"""
