#include <cuda_bf16.h>
// X is channel-major [1280,3000]. Coalesced reads along sequence are
// transposed through padded shared memory to GEMM A=[1500,3840].
extern "C" __global__ void whisper_conv2_im2col_tmajor_k(
    __nv_bfloat16* out, const __nv_bfloat16* in_0,
    const int* dyn_dims) {
  __shared__ __nv_bfloat16 tile[32][33];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int m0 = (int)blockIdx.x * 32;
  const int c0 = (int)blockIdx.y * 32;
  const int t = (int)blockIdx.z;
  #pragma unroll
  for (int j = 0; j < 32; j += 8) {
    const int c = c0 + ty + j;
    const int m = m0 + tx;
    const int src = 2 * m + t - 1;
    tile[ty + j][tx] =
        (c < 1280 && m < 1500 && src >= 0 && src < 3000)
        ? in_0[(long)c * 3000 + src]
        : __float2bfloat16_rn(0.0f);
  }
  __syncthreads();
  #pragma unroll
  for (int j = 0; j < 32; j += 8) {
    const int m = m0 + ty + j;
    const int c = c0 + tx;
    if (m < 1500 && c < 1280)
      out[((long)m * 3 + t) * 1280 + c] = tile[tx][ty + j];
  }
}
