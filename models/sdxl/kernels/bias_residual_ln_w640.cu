#include <cuda_bf16.h>
union P8 { uint4 u; __nv_bfloat162 h[4]; };
extern "C" __global__ void bias_residual_ln_w640_k(
    __nv_bfloat16* add_out, __nv_bfloat16* norm_out,
    const __nv_bfloat16* a, const __nv_bfloat16* mm,
    const __nv_bfloat16* proj_bias, const __nv_bfloat16* w,
    const __nv_bfloat16* bias, const int* dyn_dims) {
  int r=blockIdx.x, lane=threadIdx.x, base=r*640;
  __shared__ __align__(16) __nv_bfloat16 x[640];
  __shared__ float warp_s[4], warp_ss[4], mean_s, inv_s;
  float s=0.0f, ss=0.0f;
  for (int j=8*lane; j<640; j+=1024) {
    P8 av,mv,pv,ov; av.u=*(const uint4*)(a+base+j); mv.u=*(const uint4*)(mm+base+j); pv.u=*(const uint4*)(proj_bias+j);
#pragma unroll
    for(int t=0;t<4;t++) { ov.h[t]=__hadd2(av.h[t],__hadd2(mv.h[t],pv.h[t])); float2 f=__bfloat1622float2(ov.h[t]); s+=f.x; s+=f.y; ss+=f.x*f.x; ss+=f.y*f.y; }
    *(uint4*)(x+j)=ov.u; *(uint4*)(add_out+base+j)=ov.u;
  }
  unsigned mask=0xffffffffu;
  for(int d=16;d;d>>=1){ s+=__shfl_down_sync(mask,s,d); ss+=__shfl_down_sync(mask,ss,d); }
  int l=lane&31, wid=lane>>5;
  if(l==0){warp_s[wid]=s; warp_ss[wid]=ss;}
  __syncthreads();
  if(wid==0){ s=(l<4)?warp_s[l]:0.0f; ss=(l<4)?warp_ss[l]:0.0f; for(int d=16;d;d>>=1){s+=__shfl_down_sync(mask,s,d);ss+=__shfl_down_sync(mask,ss,d);} if(l==0){float m=s/640.0f; mean_s=m; inv_s=rsqrtf(fmaxf(ss/640.0f-m*m,0.0f)+1e-5f);} }
  __syncthreads();
  float mean=mean_s,inv=inv_s;
  for (int j=8*lane; j<640; j+=1024) {
    P8 xv,wv,bv,ov; xv.u=*(const uint4*)(x+j); wv.u=*(const uint4*)(w+j); bv.u=*(const uint4*)(bias+j);
#pragma unroll
    for(int t=0;t<4;t++){float2 xf=__bfloat1622float2(xv.h[t]), wf=__bfloat1622float2(wv.h[t]), bf=__bfloat1622float2(bv.h[t]); ov.h[t]=__floats2bfloat162_rn((xf.x-mean)*inv*wf.x+bf.x,(xf.y-mean)*inv*wf.y+bf.y);}
    *(uint4*)(norm_out+base+j)=ov.u;
  }
}
