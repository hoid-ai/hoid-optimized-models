#include <cuda_bf16.h>
extern "C" __global__ void whisper_resid_ln_oneshot_vec2_k(
    __nv_bfloat16* out_res, __nv_bfloat16* out_norm,
    const __nv_bfloat16* mm, const __nv_bfloat16* bias,
    const __nv_bfloat16* residual, const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta, const int* dyn_dims) {
  const int D=1280, PAIRS=5; const float eps=1e-5f; int t=threadIdx.x,lane=t&31,warp=t>>5;
  __nv_bfloat162 ma[PAIRS],ba[PAIRS],ra[PAIRS],ga[PAIRS],ea[PAIRS],va[PAIRS]; float sum=0.f,sq=0.f;
  const __nv_bfloat162* m2=(const __nv_bfloat162*)mm;const __nv_bfloat162* b2=(const __nv_bfloat162*)bias;const __nv_bfloat162* r2=(const __nv_bfloat162*)residual;const __nv_bfloat162* g2=(const __nv_bfloat162*)gamma;const __nv_bfloat162* e2=(const __nv_bfloat162*)beta;
  #pragma unroll
  for(int k=0;k<PAIRS;++k){int i=t+k*128;ma[k]=m2[i];ba[k]=b2[i];ra[k]=r2[i];ga[k]=g2[i];ea[k]=e2[i];}
  #pragma unroll
  for(int k=0;k<PAIRS;++k){float2 m=__bfloat1622float2(ma[k]),b=__bfloat1622float2(ba[k]),r=__bfloat1622float2(ra[k]);__nv_bfloat16 z0=__float2bfloat16(m.x+b.x),z1=__float2bfloat16(m.y+b.y);__nv_bfloat16 v0=__float2bfloat16(r.x+__bfloat162float(z0)),v1=__float2bfloat16(r.y+__bfloat162float(z1));va[k]=__halves2bfloat162(v0,v1);float x=__bfloat162float(v0),y=__bfloat162float(v1);sum+=x;sum+=y;sq+=x*x;sq+=y*y;}
  ((__nv_bfloat162*)out_res)[t]=va[0];
  #pragma unroll
  for(int k=1;k<PAIRS;++k)((__nv_bfloat162*)out_res)[t+k*128]=va[k];
  #pragma unroll
  for(int o=16;o>0;o>>=1){sum+=__shfl_down_sync(0xffffffff,sum,o);sq+=__shfl_down_sync(0xffffffff,sq,o);}
  __shared__ float ss[4],qq[4],mean_s,inv_s;if(lane==0){ss[warp]=sum;qq[warp]=sq;}__syncthreads();
  if(warp==0){float x=lane<4?ss[lane]:0.f,y=lane<4?qq[lane]:0.f;for(int o=2;o>0;o>>=1){x+=__shfl_down_sync(0xffffffff,x,o);y+=__shfl_down_sync(0xffffffff,y,o);}if(lane==0){float mean=x/D;mean_s=mean;inv_s=rsqrtf(y/D-mean*mean+eps);}}__syncthreads();
  float mean=mean_s,inv=inv_s;
  #pragma unroll
  for(int k=0;k<PAIRS;++k){float2 v=__bfloat1622float2(va[k]),g=__bfloat1622float2(ga[k]),e=__bfloat1622float2(ea[k]);__nv_bfloat16 o0=__float2bfloat16((v.x-mean)*inv*g.x+e.x),o1=__float2bfloat16((v.y-mean)*inv*g.y+e.y);((__nv_bfloat162*)out_norm)[t+k*128]=__halves2bfloat162(o0,o1);}
}