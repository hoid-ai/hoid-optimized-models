#include <cuda_bf16.h>
// W[oc,ci,t] -> row-major GEMM B[oc,t,ci].
extern "C" __global__ void whisper_conv2_weight_tmajor_k(
    __nv_bfloat16* out, const __nv_bfloat16* in_0,
    const int* dyn_dims) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long n = 1280L * 3L * 1280L;
  if (idx >= n) return;
  const int ci = idx % 1280;
  const int t = (idx / 1280) % 3;
  const int oc = idx / (1280 * 3);
  out[idx] = in_0[((long)oc * 1280 + ci) * 3 + t];
}
