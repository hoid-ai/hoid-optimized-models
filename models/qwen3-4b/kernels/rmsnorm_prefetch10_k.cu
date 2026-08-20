#include <cuda_bf16.h>
#define TO_F(x) __bfloat162float(x)
#define FROM_F(x) __float2bfloat16(x)
extern "C" __global__ void rmsnorm_prefetch10_k(__nv_bfloat16* out_0, const __nv_bfloat16* in_0, const __nv_bfloat16* in_1, const int* dyn_dims) {
    int row = blockIdx.x; int tid = threadIdx.x; const int N = 2560;
    const __nv_bfloat16* x = in_0 + (long)row*N;
    float vals[10];
    #pragma unroll
    for (int k=0;k<10;++k) vals[k]=TO_F(x[tid+k*256]);
    float ss=0.f;
    #pragma unroll
    for (int k=0;k<10;++k) ss += vals[k]*vals[k];
    for (int off=16;off>0;off>>=1) ss += __shfl_down_sync(0xffffffff,ss,off);
    __shared__ float sh[8]; int wid=tid/32, lane=tid%32;
    if (lane==0) sh[wid]=ss; __syncthreads();
    __shared__ float inv_sh;
    if (wid==0) { float v=(lane<8)?sh[lane]:0.f; for(int off=4;off>0;off>>=1) v+=__shfl_down_sync(0xffffffff,v,off); if(lane==0) inv_sh=rsqrtf(v/(float)N+1e-6f); }
    __syncthreads();
    float inv=inv_sh; __nv_bfloat16* o=out_0+(long)row*N;
    #pragma unroll
    for(int k=0;k<10;++k){ int i=tid+k*256; float v=TO_F(x[i]); o[i]=FROM_F(v*inv*TO_F(in_1[i])); }
}