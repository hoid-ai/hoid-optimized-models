#include <cuda_bf16.h>
extern "C" __global__ void moe_down_accum_k(
    __nv_bfloat16* out,           // [s, HID]
    const float* act,             // [s, K, INTER] (f32, already weight-scaled)
    const int* ids,               // [s, K]
    const __nv_bfloat16* down,    // [E, HID, INTER]
    const int* dyn_dims)
{
    const int HID = 2048, INTER = 768, K = 8;
    int t = blockIdx.x;
    int s = dyn_dims[18];
    if (t >= s) return;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wpb = blockDim.x >> 5;
    int d = blockIdx.y * wpb + warp;
    extern __shared__ float sm[];
    float* actsh = sm;                       // K*INTER
    int*   idsh  = (int*)(sm + K * INTER);   // K
    for (int i = threadIdx.x; i < K * INTER; i += blockDim.x) actsh[i] = act[(long)t * K * INTER + i];
    if (threadIdx.x < K) idsh[threadIdx.x] = ids[(long)t * K + threadIdx.x];
    __syncthreads();
    if (d >= HID) return;
    float acc = 0.f;
    for (int k = 0; k < K; k++) {
        int e = idsh[k];
        const __nv_bfloat16* drow = down + ((long)e * HID + d) * INTER;
        float dot = 0.f;
        for (int i = lane; i < INTER; i += 32) dot += actsh[k * INTER + i] * __bfloat162float(drow[i]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, o);
        acc += dot;   // routing weight already folded into act by gate_up
    }
    if (lane == 0) out[(long)t * HID + d] = __float2bfloat16(acc);
}
