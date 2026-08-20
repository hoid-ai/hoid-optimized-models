#include <cuda_bf16.h>
extern "C" __global__ void fullcta_add_rmsnorm_final_k(
    __nv_bfloat16* norm,
    const __nv_bfloat16* lhs, const __nv_bfloat16* rhs,
    const __nv_bfloat16* weight, const int* dyn_dims)
{
    constexpr int N = 2048;
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const long base = (long)row * N;
    __nv_bfloat16 rounded[2];
    float ss = 0.0f;
    #pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int i = tid + q * 1024;
        // Match add_k's BF16 store before rmsnorm_k promotes and squares.
        const __nv_bfloat16 s = lhs[base + i] + rhs[base + i];
        rounded[q] = s;
        const float v = __bfloat162float(s);
        ss += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_down_sync(0xffffffffu, ss, off);
    __shared__ float warp_ss[32];
    __shared__ float inv_sh;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    if (lane == 0) warp_ss[wid] = ss;
    __syncthreads();
    if (wid == 0) {
        float v = lane < 32 ? warp_ss[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) inv_sh = rsqrtf(v / (float)N + 1.0e-6f);
    }
    __syncthreads();
    const float inv = inv_sh;
    #pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int i = tid + q * 1024;
        const float v = __bfloat162float(rounded[q]);
        norm[base + i] = __float2bfloat16(v * inv * __bfloat162float(weight[i]));
    }
}
