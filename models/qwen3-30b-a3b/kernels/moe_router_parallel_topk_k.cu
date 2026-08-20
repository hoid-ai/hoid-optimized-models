#include <cuda_bf16.h>
extern "C" __global__ void moe_router_parallel_topk_k(
    int* topk_ids,
    float* topk_w,
    const __nv_bfloat16* logits,
    const int* dyn_dims)
{
    const int NE = 128;
    const int K = 8;
    const unsigned FULL = 0xffffffffu;
    int t = blockIdx.x;
    int s = dyn_dims[18];
    if (t >= s) return;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    __shared__ float p[NE];
    __shared__ float warp_v[4];
    __shared__ int warp_i[4];
    __shared__ float mx_shared;
    __shared__ float inv_shared;
    __shared__ int seli[K];
    __shared__ float selw[K];

    const __nv_bfloat16* lg = logits + (long)t * NE;
    float v = __bfloat162float(lg[tid]);
    p[tid] = v;

    // Exact maximum value, with the reference's lowest-index tie break.
    int ix = tid;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float cv = __shfl_down_sync(FULL, v, o);
        int ci = __shfl_down_sync(FULL, ix, o);
        if (cv > v || (cv == v && ci < ix)) { v = cv; ix = ci; }
    }
    if (lane == 0) { warp_v[warp] = v; warp_i[warp] = ix; }
    __syncthreads();
    if (warp == 0) {
        v = lane < 4 ? warp_v[lane] : -1e30f;
        ix = lane < 4 ? warp_i[lane] : 0x7fffffff;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            float cv = __shfl_down_sync(FULL, v, o);
            int ci = __shfl_down_sync(FULL, ix, o);
            if (cv > v || (cv == v && ci < ix)) { v = cv; ix = ci; }
        }
        if (lane == 0) mx_shared = v;
    }
    __syncthreads();

    // expf is elementwise.  Keep the reference's left-to-right fp32 sum so
    // the normalized probabilities and final bf16-rounded gates are bit exact.
    p[tid] = __expf(p[tid] - mx_shared);
    __syncthreads();
    if (tid == 0) {
        float sum = 0.f;
        #pragma unroll 1
        for (int e = 0; e < NE; ++e) sum += p[e];
        inv_shared = 1.f / sum;
    }
    __syncthreads();
    p[tid] *= inv_shared;
    __syncthreads();

    // Repeated block argmax preserves the original strict-'>' scan semantics,
    // but replaces 8*128 scalar comparisons with four parallel warp trees.
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        v = p[tid];
        ix = tid;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            float cv = __shfl_down_sync(FULL, v, o);
            int ci = __shfl_down_sync(FULL, ix, o);
            if (cv > v || (cv == v && ci < ix)) { v = cv; ix = ci; }
        }
        if (lane == 0) { warp_v[warp] = v; warp_i[warp] = ix; }
        __syncthreads();
        if (warp == 0) {
            v = lane < 4 ? warp_v[lane] : -1.f;
            ix = lane < 4 ? warp_i[lane] : 0x7fffffff;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                float cv = __shfl_down_sync(FULL, v, o);
                int ci = __shfl_down_sync(FULL, ix, o);
                if (cv > v || (cv == v && ci < ix)) { v = cv; ix = ci; }
            }
            if (lane == 0) {
                seli[k] = ix;
                selw[k] = v;
                p[ix] = -1.f;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        float wsum = 0.f;
        #pragma unroll
        for (int k = 0; k < K; ++k) wsum += selw[k];
        float winv = wsum > 0.f ? 1.f / wsum : 0.f;
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            topk_ids[(long)t * K + k] = seli[k];
            float w = selw[k] * winv;
            topk_w[(long)t * K + k] = __bfloat162float(__float2bfloat16(w));
        }
    }
}
