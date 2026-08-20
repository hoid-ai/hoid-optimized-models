#include <cuda_bf16.h>
extern "C" __global__ void whisper_tpos_k(
    __nv_bfloat16* out,
    const __nv_bfloat16* conv,
    const __nv_bfloat16* pos,
    const int* dyn_dims)
{
    const int C = 1280;
    const int L = 1500;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)L * C) return;
    int li = idx / C;      // token position (0..L)
    int ci = idx % C;      // channel (0..C)
    float val = __bfloat162float(conv[(long)ci * L + li]) + __bfloat162float(pos[(long)li * C + ci]);
    out[idx] = __float2bfloat16(val);
}
