#include <cuda_bf16.h>
constexpr int WARPS = 64;
constexpr int NQ = 32;
constexpr int HD = 128;
constexpr int RECORD = HD + 2;

extern "C" __global__ void dense_cached_attention_residue_combine_k(
    __nv_bfloat16* out, const float* in_0, const int* dyn_dims) {
  int bh = blockIdx.x;
  int qi = bh / NQ;
  int h = bh % NQ;
  int lane = threadIdx.x;
  const float* base = in_0 + ((long)qi * NQ + h) * WARPS * RECORD;

  float M = -1.0e30f;
  #pragma unroll
  for (int w = 0; w < WARPS; ++w)
    M = fmaxf(M, base[(long)w * RECORD + HD]);
  float den = 0.f;
  #pragma unroll
  for (int w = 0; w < WARPS; ++w)
    den += base[(long)w * RECORD + HD + 1] *
           expf(base[(long)w * RECORD + HD] - M);

  __nv_bfloat16* dst = out + ((long)qi * NQ + h) * HD;
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    int d = lane + k * 32;
    float num = 0.f;
    #pragma unroll
    for (int w = 0; w < WARPS; ++w)
      num += base[(long)w * RECORD + d] *
             expf(base[(long)w * RECORD + HD] - M);
    dst[d] = __float2bfloat16(num / den);
  }
}
