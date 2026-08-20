#include <cuda_bf16.h>
extern "C" __global__ void whisper_embed_add_ln_exact_k(__nv_bfloat16* out_res,__nv_bfloat16* out_norm,const __nv_bfloat16* tok_w,const int* tok_id,const __nv_bfloat16* pos_w,const int* pos_id,const __nv_bfloat16* gamma,const __nv_bfloat16* beta,const int* dyn_dims){
 const int D=1280;const float eps=0.00001f;int t=threadIdx.x;int tok=tok_id[0],pos=pos_id[0];float vals[5],s=0.f;
 #pragma unroll
 for(int k=0;k<5;++k){int i=t+k*256;__nv_bfloat16 r=__float2bfloat16(__bfloat162float(tok_w[(long)tok*D+i])+__bfloat162float(pos_w[(long)pos*D+i]));out_res[i]=r;vals[k]=__bfloat162float(r);s+=vals[k];}
 __shared__ float red[256];red[t]=s;__syncthreads();for(int o=128;o>0;o>>=1){if(t<o)red[t]+=red[t+o];__syncthreads();}float mean=red[0]/D;__syncthreads();float v=0.f;
 #pragma unroll
 for(int k=0;k<5;++k){float d=vals[k]-mean;v+=d*d;}red[t]=v;__syncthreads();for(int o=128;o>0;o>>=1){if(t<o)red[t]+=red[t+o];__syncthreads();}float inv=rsqrtf(red[0]/D+eps);__syncthreads();
 #pragma unroll
 for(int k=0;k<5;++k){int i=t+k*256;float n=(vals[k]-mean)*inv;n=n*__bfloat162float(gamma[i])+__bfloat162float(beta[i]);out_norm[i]=__float2bfloat16(n);}
}
