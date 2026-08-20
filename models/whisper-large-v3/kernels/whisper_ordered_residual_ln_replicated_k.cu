#include <cuda_bf16.h>

__device__ __forceinline__ float warp_finish_ordered_rep(float x) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    x += __shfl_down_sync(0xffffffffu, x, off);
  return x;
}

// Preserve the incumbent's leaf layout and arithmetic association, but let
// every warp reconstruct the same final value after the one required publish
// barrier.  Broadcasting lane zero inside each warp removes the incumbent's
// shared scalar, its second block-wide barrier, and the scalar load hot spot.
__device__ __forceinline__ float ordered_sum_256x2_replicated(
    float x0, float x1, float* red) {
  const int t = (int)threadIdx.x;
  red[2 * t] = x0;
  red[2 * t + 1] = x1;
  __syncthreads();

  const int u = t & 31;
  float a0 = red[u] + red[u + 128];
  float a1 = red[u + 64] + red[u + 192];
  float b0 = a0 + a1;
  float a2 = red[u + 32] + red[u + 160];
  float a3 = red[u + 96] + red[u + 224];
  float b1 = a2 + a3;
  float v = warp_finish_ordered_rep(b0 + b1);
  return __shfl_sync(0xffffffffu, v, 0);
}

extern "C" __global__ void whisper_ordered_residual_ln_replicated_k(
    __nv_bfloat16* __restrict__ residual_out,
    __nv_bfloat16* __restrict__ norm_out,
    const __nv_bfloat16* __restrict__ mat,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ skip,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    const int* dyn_dims) {
  const int PAIRS = 640;
  const float eps = 0.00001f;
  const int row = (int)blockIdx.x;
  const int t = (int)threadIdx.x;
  const long pair_base = (long)row * PAIRS;
  const __nv_bfloat162* __restrict__ mat2 = reinterpret_cast<const __nv_bfloat162*>(mat);
  const __nv_bfloat162* __restrict__ bias2 = reinterpret_cast<const __nv_bfloat162*>(bias);
  const __nv_bfloat162* __restrict__ skip2 = reinterpret_cast<const __nv_bfloat162*>(skip);
  const __nv_bfloat162* __restrict__ gamma2 = reinterpret_cast<const __nv_bfloat162*>(gamma);
  const __nv_bfloat162* __restrict__ beta2 = reinterpret_cast<const __nv_bfloat162*>(beta);
  __nv_bfloat162* __restrict__ residual2 = reinterpret_cast<__nv_bfloat162*>(residual_out);
  __nv_bfloat162* __restrict__ norm2 = reinterpret_cast<__nv_bfloat162*>(norm_out);
  __shared__ float red[256];

  __nv_bfloat162 mv[5], bv[5], sv[5], held[5];
  #pragma unroll
  for (int k = 0; k < 5; ++k) {
    int p = t + k * 128;
    mv[k] = mat2[pair_base + p];
    bv[k] = bias2[p];
    sv[k] = skip2[pair_base + p];
  }

  float s0 = 0.0f, s1 = 0.0f;
  #pragma unroll
  for (int k = 0; k < 5; ++k) {
    int p = t + k * 128;
    float2 m = __bfloat1622float2(mv[k]);
    float2 b = __bfloat1622float2(bv[k]);
    float2 sk = __bfloat1622float2(sv[k]);
    float z0 = __bfloat162float(__float2bfloat16_rn(m.x + b.x));
    float z1 = __bfloat162float(__float2bfloat16_rn(m.y + b.y));
    __nv_bfloat162 r = __floats2bfloat162_rn(sk.x + z0, sk.y + z1);
    held[k] = r;
    residual2[pair_base + p] = r;
    float2 rf = __bfloat1622float2(r);
    s0 += rf.x;
    s1 += rf.y;
  }

  float mean = ordered_sum_256x2_replicated(s0, s1, red) / 1280.0f;
  float v0 = 0.0f, v1 = 0.0f;
  #pragma unroll
  for (int k = 0; k < 5; ++k) {
    float2 r = __bfloat1622float2(held[k]);
    float d0 = r.x - mean;
    float d1 = r.y - mean;
    v0 += d0 * d0;
    v1 += d1 * d1;
  }
  float inv = rsqrtf(ordered_sum_256x2_replicated(v0, v1, red) / 1280.0f + eps);

  __nv_bfloat162 gv[5], betav[5];
  #pragma unroll
  for (int k = 0; k < 5; ++k) {
    int p = t + k * 128;
    gv[k] = gamma2[p];
    betav[k] = beta2[p];
  }
  #pragma unroll
  for (int k = 0; k < 5; ++k) {
    int p = t + k * 128;
    float2 r = __bfloat1622float2(held[k]);
    float2 g = __bfloat1622float2(gv[k]);
    float2 be = __bfloat1622float2(betav[k]);
    float n0 = (r.x - mean) * inv;
    float n1 = (r.y - mean) * inv;
    norm2[pair_base + p] = __floats2bfloat162_rn(
        n0 * g.x + be.x, n1 * g.y + be.y);
  }
}
