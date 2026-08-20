#include <cuda_bf16.h>

// Preserve whisper_ln_k's source-level accumulation and shared-tree order.
// Volatile shared accesses plus converged __syncwarp calls make each final
// warp-tree landing point observable under independent thread scheduling.
extern "C" __global__ void whisper_ln_volatile_warptail_k(
    __nv_bfloat16* out,
    const __nv_bfloat16* in,
    const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta,
    const int* dyn_dims)
{
    const int D = 1280;
    const float eps = 0.00001f;
    int row = blockIdx.x;
    const __nv_bfloat16* x = in + (long)row * D;
    __nv_bfloat16* y = out + (long)row * D;
    __shared__ float red[256];
    int t = threadIdx.x;

    // Mean: retain the baseline's runtime blockDim stride and loop shape.
    float s = 0.f;
    for (int i = t; i < D; i += blockDim.x) s += __bfloat162float(x[i]);
    red[t] = s; __syncthreads();
    for (int o = blockDim.x/2; o >= 32; o >>= 1) {
        if (t < o) red[t] += red[t+o];
        __syncthreads();
    }
    if (t < 32) {
        volatile float* vr = red;
        if (t < 16) vr[t] = vr[t] + vr[t+16];
        __syncwarp(0xffffffffu);
        if (t < 8) vr[t] = vr[t] + vr[t+8];
        __syncwarp(0xffffffffu);
        if (t < 4) vr[t] = vr[t] + vr[t+4];
        __syncwarp(0xffffffffu);
        if (t < 2) vr[t] = vr[t] + vr[t+2];
        __syncwarp(0xffffffffu);
        if (t < 1) vr[t] = vr[t] + vr[t+1];
    }
    __syncthreads();
    float mean = red[0] / D; __syncthreads();

    // Variance: use the identical compiler-visible accumulation structure.
    float v = 0.f;
    for (int i = t; i < D; i += blockDim.x) {
        float d = __bfloat162float(x[i]) - mean;
        v += d*d;
    }
    red[t] = v; __syncthreads();
    for (int o = blockDim.x/2; o >= 32; o >>= 1) {
        if (t < o) red[t] += red[t+o];
        __syncthreads();
    }
    if (t < 32) {
        volatile float* vr = red;
        if (t < 16) vr[t] = vr[t] + vr[t+16];
        __syncwarp(0xffffffffu);
        if (t < 8) vr[t] = vr[t] + vr[t+8];
        __syncwarp(0xffffffffu);
        if (t < 4) vr[t] = vr[t] + vr[t+4];
        __syncwarp(0xffffffffu);
        if (t < 2) vr[t] = vr[t] + vr[t+2];
        __syncwarp(0xffffffffu);
        if (t < 1) vr[t] = vr[t] + vr[t+1];
    }
    __syncthreads();
    float inv = rsqrtf(red[0] / D + eps); __syncthreads();

    for (int i = t; i < D; i += blockDim.x) {
        float n = (__bfloat162float(x[i]) - mean) * inv;
        n = n * __bfloat162float(gamma[i]) + __bfloat162float(beta[i]);
        y[i] = __float2bfloat16(n);
    }
}
