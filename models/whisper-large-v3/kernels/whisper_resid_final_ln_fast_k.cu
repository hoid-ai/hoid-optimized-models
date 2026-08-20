#include <cuda_bf16.h>
extern "C" __global__ void whisper_resid_final_ln_fast_k(
    __nv_bfloat16* out_norm,
    const __nv_bfloat16* mm, const __nv_bfloat16* bias,
    const __nv_bfloat16* residual, const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta, const int* dyn_dims) {
    const int D=1280, ITEMS=5; const float eps=1.0e-5f;
    int t=threadIdx.x,lane=t&31,warp=t>>5;
    float vals[ITEMS]; float sum=0.f;
    #pragma unroll
    for(int k=0;k<ITEMS;++k){int i=t+k*256;__nv_bfloat16 z=__float2bfloat16(__bfloat162float(mm[i])+__bfloat162float(bias[i]));__nv_bfloat16 r=__float2bfloat16(__bfloat162float(residual[i])+__bfloat162float(z));vals[k]=__bfloat162float(r);sum+=vals[k];}
    #pragma unroll
    for(int o=16;o>0;o>>=1)sum+=__shfl_down_sync(0xffffffff,sum,o);
    __shared__ float ws[8]; __shared__ float stat;
    if(lane==0)ws[warp]=sum; __syncthreads();
    if(warp==0){float x=lane<8?ws[lane]:0.f;for(int o=4;o>0;o>>=1)x+=__shfl_down_sync(0xffffffff,x,o);if(lane==0)stat=x/D;} __syncthreads();
    float mean=stat,var=0.f;
    #pragma unroll
    for(int k=0;k<ITEMS;++k){float d=vals[k]-mean;var+=d*d;}
    #pragma unroll
    for(int o=16;o>0;o>>=1)var+=__shfl_down_sync(0xffffffff,var,o);
    if(lane==0)ws[warp]=var; __syncthreads();
    if(warp==0){float x=lane<8?ws[lane]:0.f;for(int o=4;o>0;o>>=1)x+=__shfl_down_sync(0xffffffff,x,o);if(lane==0)stat=rsqrtf(x/D+eps);} __syncthreads();
    float inv=stat;
    #pragma unroll
    for(int k=0;k<ITEMS;++k){int i=t+k*256;out_norm[i]=__float2bfloat16((vals[k]-mean)*inv*__bfloat162float(gamma[i])+__bfloat162float(beta[i]));}
}