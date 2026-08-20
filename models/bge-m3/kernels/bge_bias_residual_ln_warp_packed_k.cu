#include <cuda_bf16.h>
extern "C" __global__ void bge_bias_residual_ln_warp_packed_k(
    __nv_bfloat16* out, const __nv_bfloat16* mm,
    const __nv_bfloat16* mm_bias, const __nv_bfloat16* residual,
    const __nv_bfloat16* weight, const __nv_bfloat16* ln_bias,
    const int* dyn_dims) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int r = blockIdx.x * 8 + warp;
  const long base = (long)r * 1024;
  __nv_bfloat162 x[4][4];
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int j0 = k * 256 + lane * 8;
    const uint4 mv = *reinterpret_cast<const uint4*>(mm + base + j0);
    const uint4 bv = *reinterpret_cast<const uint4*>(mm_bias + j0);
    const uint4 rv = *reinterpret_cast<const uint4*>(residual + base + j0);
    const __nv_bfloat162* m4 = reinterpret_cast<const __nv_bfloat162*>(&mv);
    const __nv_bfloat162* b4 = reinterpret_cast<const __nv_bfloat162*>(&bv);
    const __nv_bfloat162* r4 = reinterpret_cast<const __nv_bfloat162*>(&rv);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const __nv_bfloat162 biased = __hadd2(m4[e], b4[e]);
      const __nv_bfloat162 val = __hadd2(r4[e], biased);
      x[k][e] = val;
      sum += __bfloat162float(__low2bfloat16(val));
      sum += __bfloat162float(__high2bfloat16(val));
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);
  sum = __shfl_sync(0xffffffffu, sum, 0);
  const float mean = sum * (1.0f / 1024.0f);
  float var = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float v0 = __bfloat162float(__low2bfloat16(x[k][e])) - mean;
      const float v1 = __bfloat162float(__high2bfloat16(x[k][e])) - mean;
      var = fmaf(v0, v0, var);
      var = fmaf(v1, v1, var);
    }
#pragma unroll
  for (int o = 16; o; o >>= 1) var += __shfl_down_sync(0xffffffffu, var, o);
  var = __shfl_sync(0xffffffffu, var, 0);
  const float inv = rsqrtf(var * (1.0f / 1024.0f) + 0.00001f);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int j0 = k * 256 + lane * 8;
    const uint4 wv = *reinterpret_cast<const uint4*>(weight + j0);
    const uint4 lv = *reinterpret_cast<const uint4*>(ln_bias + j0);
    const __nv_bfloat16* w8 = reinterpret_cast<const __nv_bfloat16*>(&wv);
    const __nv_bfloat16* l8 = reinterpret_cast<const __nv_bfloat16*>(&lv);
    const __nv_bfloat16* xv = reinterpret_cast<const __nv_bfloat16*>(&x[k][0]);
    uint4 ov;
    __nv_bfloat16* o8 = reinterpret_cast<__nv_bfloat16*>(&ov);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float v = __bfloat162float(xv[e]);
      o8[e] = __float2bfloat16((v - mean) * inv * __bfloat162float(w8[e]) +
                               __bfloat162float(l8[e]));
    }
    *reinterpret_cast<uint4*>(out + base + j0) = ov;
  }
}