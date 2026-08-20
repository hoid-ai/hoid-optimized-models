#include <cuda_bf16.h>
extern "C" __global__ void whisper_qkv_prefetch10_raw_b128_k(
    __nv_bfloat16* out_q, __nv_bfloat16* out_k, __nv_bfloat16* out_v,
    const __nv_bfloat16* x, const __nv_bfloat16* wq,
    const __nv_bfloat16* wk, const __nv_bfloat16* wv, const int* dyn_dims) {
  const int D=1280;int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  int oid=blockIdx.x*4+warp;if(oid>=3840)return;int g=oid/D,row=oid-g*D;
  const __nv_bfloat16* w=g==0?wq:(g==1?wk:wv);float total=0.f;
  #pragma unroll
  for(int group=0;group<2;++group){
    float xa[10],xb[10],wa[10],wb[10];
    #pragma unroll
    for(int u=0;u<10;++u){int kk=(group*10+u)*64+lane;xa[u]=__bfloat162float(x[kk]);xb[u]=__bfloat162float(x[kk+32]);wa[u]=__bfloat162float(w[(long)row*D+kk]);wb[u]=__bfloat162float(w[(long)row*D+kk+32]);}
    #pragma unroll
    for(int u=0;u<10;++u){float s=xa[u]*wa[u]+xb[u]*wb[u];
      #pragma unroll
      for(int o=16;o>0;o>>=1)s+=__shfl_down_sync(0xffffffff,s,o);
      if(lane==0)total+=s;}
  }
  if(lane==0){__nv_bfloat16 z=__float2bfloat16(total);if(g==0)out_q[row]=z;else if(g==1)out_k[row]=z;else out_v[row]=z;}
}