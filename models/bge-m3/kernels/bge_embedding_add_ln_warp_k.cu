#include <cuda_bf16.h>
extern "C" __global__ void bge_embedding_add_ln_warp_k(
    __nv_bfloat16* out, const __nv_bfloat16* word, const int* ids,
    const __nv_bfloat16* pos, const __nv_bfloat16* tok,
    const __nv_bfloat16* w, const __nv_bfloat16* b, const int* dyn_dims) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int r = blockIdx.x * 8 + warp;
  const int id = ids[r], seq = r & 511;
  const long wbase = (long)id * 1024, pbase = (long)(seq + 2) * 1024;
  __nv_bfloat16 x[4][8];
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int j0 = k * 256 + lane * 8;
    const uint4 wv = *reinterpret_cast<const uint4*>(word + wbase + j0);
    const uint4 pv = *reinterpret_cast<const uint4*>(pos + pbase + j0);
    const uint4 tv = *reinterpret_cast<const uint4*>(tok + j0);
    const __nv_bfloat16* ww = reinterpret_cast<const __nv_bfloat16*>(&wv);
    const __nv_bfloat16* pp = reinterpret_cast<const __nv_bfloat16*>(&pv);
    const __nv_bfloat16* tt = reinterpret_cast<const __nv_bfloat16*>(&tv);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      x[k][e] = __float2bfloat16(__bfloat162float(ww[e]) +
                                  __bfloat162float(pp[e]) +
                                  __bfloat162float(tt[e]));
      sum += __bfloat162float(x[k][e]);
    }
  }
#pragma unroll
  for (int o=16;o;o>>=1) sum += __shfl_down_sync(0xffffffffu,sum,o);
  const float mean=__shfl_sync(0xffffffffu,sum,0)*(1.0f/1024.0f);
  float var=0.0f;
#pragma unroll
  for(int k=0;k<4;++k)
#pragma unroll
    for(int e=0;e<8;++e){float v=__bfloat162float(x[k][e])-mean;var=fmaf(v,v,var);}
#pragma unroll
  for(int o=16;o;o>>=1) var += __shfl_down_sync(0xffffffffu,var,o);
  const float inv=rsqrtf(__shfl_sync(0xffffffffu,var,0)*(1.0f/1024.0f)+0.00001f);
#pragma unroll
  for(int k=0;k<4;++k){
    const int j0=k*256+lane*8;
    const uint4 gw=*reinterpret_cast<const uint4*>(w+j0);
    const uint4 gb=*reinterpret_cast<const uint4*>(b+j0);
    const __nv_bfloat16* wp=reinterpret_cast<const __nv_bfloat16*>(&gw);
    const __nv_bfloat16* bp=reinterpret_cast<const __nv_bfloat16*>(&gb);
    uint4 ov; __nv_bfloat16* op=reinterpret_cast<__nv_bfloat16*>(&ov);
#pragma unroll
    for(int e=0;e<8;++e){float v=__bfloat162float(x[k][e]);op[e]=__float2bfloat16((v-mean)*inv*__bfloat162float(wp[e])+__bfloat162float(bp[e]));}
    *reinterpret_cast<uint4*>(out+(long)r*1024+j0)=ov;
  }
}