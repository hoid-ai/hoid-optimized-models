#include <cuda_bf16.h>
#define TO_F(x) __bfloat162float(x)
#define FROM_F(x) __float2bfloat16(x)
extern "C" __global__ void rmsnorm_register_k(
    __nv_bfloat16* out, const __nv_bfloat16* in_0,
    const __nv_bfloat16* in_1, const int* dyn_dims) {
  constexpr int N = 2560;
  constexpr int THREADS = 256;
  constexpr int ITEMS = N / THREADS;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  const __nv_bfloat16* x = in_0 + (long)row * N;

  // Keep the ten values owned by this lane live across the reduction.  The
  // accumulation and reduction order match the built-in kernel exactly, but
  // normalization no longer issues a second load of x.
  float values[ITEMS];
  float ss = 0.0f;
  #pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    float v = TO_F(x[tid + k * THREADS]);
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
    if (lane == 0) inv_shared = rsqrtf(v / (float)N + 1.000000e-6f);
  }
  __syncthreads();

  float inv = inv_shared;
  __nv_bfloat16* dst = out + (long)row * N;
  #pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    int i = tid + k * THREADS;
    dst[i] = FROM_F(values[k] * inv * TO_F(in_1[i]));
  }
}
