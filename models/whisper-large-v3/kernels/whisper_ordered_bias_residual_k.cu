#include <cuda_bf16.h>
extern "C" __global__ void whisper_ordered_bias_residual_k(
    __nv_bfloat16* out, const __nv_bfloat16* in_0,
    const __nv_bfloat16* in_1, const __nv_bfloat16* in_2,
    const int* dyn_dims) {
  // One thread owns an aligned 16-byte group.  All graph buffers and
  // row starts are 16-byte aligned (1280 bf16 values per row).
  struct __align__(16) Bf16x8 { __nv_bfloat162 lane[4]; };
  long group = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= 240000L) return;
  long i = group * 8L;
  int b = (int)(i % 1280L);
  Bf16x8 m = *reinterpret_cast<const Bf16x8*>(in_0 + i);
  Bf16x8 bias = *reinterpret_cast<const Bf16x8*>(in_1 + b);
  Bf16x8 residual = *reinterpret_cast<const Bf16x8*>(in_2 + i);
  Bf16x8 result;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    float2 mv = __bfloat1622float2(m.lane[j]);
    float2 bv = __bfloat1622float2(bias.lane[j]);
    float2 rv = __bfloat1622float2(residual.lane[j]);
    // Preserve both materialization points exactly: bf16 round after
    // bias, widen, residual add, then bf16 round on the final store.
    float z0 = __bfloat162float(__float2bfloat16_rn(mv.x + bv.x));
    float z1 = __bfloat162float(__float2bfloat16_rn(mv.y + bv.y));
    result.lane[j] = __floats2bfloat162_rn(rv.x + z0, rv.y + z1);
  }
  *reinterpret_cast<Bf16x8*>(out + i) = result;
}
