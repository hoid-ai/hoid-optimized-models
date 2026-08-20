#include <cuda_bf16.h>
extern "C" __global__ void copy_vec8_4096_k(__nv_bfloat16* out_0, const __nv_bfloat16* in_0, const int* dyn_dims) {
    const long vi = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long nv = ((long)dyn_dims[18] * 4096) / 8;
    if (vi < nv) reinterpret_cast<uint4*>(out_0)[vi] = reinterpret_cast<const uint4*>(in_0)[vi];
}
