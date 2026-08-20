#include <cuda_bf16.h>
extern "C" __global__ void whisper_bias_gelu_k(
    __nv_bfloat16* out, const __nv_bfloat16* mm,
    const __nv_bfloat16* bias, const int* dyn_dims) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<5120) {
        __nv_bfloat16 z=__float2bfloat16(__bfloat162float(mm[i])+__bfloat162float(bias[i]));
        float x=__bfloat162float(z);
        out[i]=__float2bfloat16(0.5f*x*(1.f+erff(x*0.7071067811865476f)));
    }
}