#include <cuda_bf16.h>
#define TO_F(x) __bfloat162float(x)
#define FROM_F(x) __float2bfloat16(x)

extern "C" __global__ void prepare_qkv_k(
    __nv_bfloat16* out_q,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const __nv_bfloat16* q_norm_w,
    const __nv_bfloat16* k_norm_w,
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    const float* cos_table,
    const float* sin_table,
    const int* dyn_dims
) {
    int slot = blockIdx.x;                 // one warp per (token, head)
    int s = dyn_dims[18];
    if (slot >= s * 40) return;
    int lane  = threadIdx.x & 31;
    int token = slot / 40;
    int h     = slot % 40;
    int pos   = dyn_dims[14] + token;

    if (h < 32 + 4) {
        const __nv_bfloat16* src;
        const __nv_bfloat16* nw;
        __nv_bfloat16* dst;
        if (h < 32) {
            src = q + (long)token * 4096 + (long)h * 128;
            nw  = q_norm_w;
            dst = out_q + (long)token * 4096 + (long)h * 128;
        } else {
            int kh = h - 32;
            src = k + (long)token * 512 + (long)kh * 128;
            nw  = k_norm_w;
            dst = k_cache + ((long)kh * 2048 + pos) * 128;
        }

        float lo[2], hi[2];
        float partial = 0.f;
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            int p = lane + t * 32;
            if (p < 64) {
                float a = TO_F(src[p]);
                float b = TO_F(src[p + 64]);
                lo[t] = a; hi[t] = b;
                partial += a * a + b * b;
            } else { lo[t] = 0.f; hi[t] = 0.f; }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            partial += __shfl_xor_sync(0xffffffffu, partial, o);
        float inv = rsqrtf(partial / (float)128 + 1e-6f);

        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            int p = lane + t * 32;
            if (p < 64) {
                float a = lo[t] * inv * TO_F(nw[p]);
                float b = hi[t] * inv * TO_F(nw[p + 64]);
                float c  = cos_table[(long)pos * 64 + p];
                float sn = sin_table[(long)pos * 64 + p];
                dst[p]          = FROM_F(a * c - b * sn);
                dst[p + 64] = FROM_F(b * c + a * sn);
            }
        }
    } else {
        int vh = h - 32 - 4;
        const __nv_bfloat16* src = v + (long)token * 512 + (long)vh * 128;
        __nv_bfloat16* dst = v_cache + ((long)vh * 2048 + pos) * 128;
        for (int d = lane; d < 128; d += 32) dst[d] = src[d];
    }
}
