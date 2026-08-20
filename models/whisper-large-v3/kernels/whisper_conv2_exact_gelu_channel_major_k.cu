#include <cuda_bf16.h>
extern "C" __global__ void whisper_conv2_exact_gelu_channel_major_k(
    __nv_bfloat16* out, const __nv_bfloat16* in_0,
    const int* dyn_dims) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long n = 1500L * 1280L;
  if (idx >= n) return;
  const int m = (int)(idx / 1280L);
  const int co = (int)(idx - (long)m * 1280L);
  const float x = __bfloat162float(in_0[idx]);
  out[(long)co * 1500L + m] = __float2bfloat16_rn(
      0.5f * x * (1.0f + erff(x * 0.7071067811865476f)));
}
