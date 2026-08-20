#include <cuda_bf16.h>
extern "C" __global__ void fused_final_residual_rmsnorm_k(
    __nv_bfloat16* out,
    const __nv_bfloat16* in_0,
    const __nv_bfloat16* in_1,
    const __nv_bfloat16* in_2,
    const int* dyn_dims) {
  constexpr int N = 2560;
  constexpr int THREADS = 256;
  constexpr int ITEMS = N / THREADS;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  long row_base = (long)row * N;

  float values[ITEMS];
  float ss = 0.0f;
  #pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    int i = tid + k * THREADS;
    __nv_bfloat16 summed = in_0[row_base + i] + in_1[row_base + i];
    float v = __bfloat162float(summed);
    values[k] = v;
    ss += v * v;
  }
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    ss += __shfl_down_sync(0xffffffffu, ss, offset);

  __shared__ float warp_sums[8];
  __shared__ float inv_shared;
  int lane = tid & 31;
  int warp = tid >> 5;
  if (lane == 0) warp_sums[warp] = ss;
  __syncthreads();
  if (warp == 0) {
    float v = lane < 8 ? warp_sums[lane] : 0.0f;
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1)
      v += __shfl_down_sync(0xffffffffu, v, offset);
    if (lane == 0)
      inv_shared = rsqrtf(v / (float)N + 1.000000e-6f);
  }
  __syncthreads();

  float inv = inv_shared;
  #pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    int i = tid + k * THREADS;
    out[row_base + i] = __float2bfloat16(
        values[k] * inv * __bfloat162float(in_2[i]));
  }
}
