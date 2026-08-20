#include <cuda_bf16.h>
#define TO_F(x) __bfloat162float(x)
#define FROM_F(x) __float2bfloat16(x)
extern "C" __global__ void rmsnorm_k(__nv_bfloat16* out, const __nv_bfloat16* in_0, const __nv_bfloat16* in_1, const int* dyn_dims) {
    int row = blockIdx.x; int tid = threadIdx.x; int N = 2048;
    const __nv_bfloat16* x = in_0 + (long)row*N;
    float ss = 0.f;
    for (int i=tid;i<N;i+=256) { float v=TO_F(x[i]); ss += v*v; }
    for (int o=16;o>0;o>>=1) ss += __shfl_down_sync(0xffffffff, ss, o);
    __shared__ float sh[8]; int wid=tid/32, lane=tid%32;
    if (lane==0) sh[wid]=ss; __syncthreads();
    __shared__ float inv_sh;
    if (wid==0) { float v=(lane<8)?sh[lane]:0.f; for(int o=4;o>0;o>>=1) v+=__shfl_down_sync(0xffffffff,v,o);
        if (lane==0) inv_sh = rsqrtf(v/(float)N + 1e-6f); }
    __syncthreads();
    float inv = inv_sh;
    __nv_bfloat16* o = out + (long)row*N;
    for (int i=tid;i<N;i+=256) { float v=TO_F(x[i]); o[i]=FROM_F(v*inv*TO_F(in_1[i])); }
}
