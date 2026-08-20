#include <cuda_bf16.h>
extern "C" __global__ void attn_combine_k(
    __nv_bfloat16* out,           // [s, NQ, HD]
    const float* part,            // [s, NQ, KSPLIT, HD+2]
    const int* dyn_dims)
{
    const int NQ=32, HD=128, KSPLIT=8;
    int qi = blockIdx.x, h = blockIdx.y;
    int s = dyn_dims[18];
    if (qi >= s) return;
    const float* base = part + ((long)qi*NQ + h)*KSPLIT*(HD+2);
    __shared__ float sm_[KSPLIT], sl_[KSPLIT], se_[KSPLIT], Mden[2];
    int d = threadIdx.x;
    if (d < KSPLIT) { sm_[d]=base[d*(HD+2)+HD]; sl_[d]=base[d*(HD+2)+HD+1]; }
    __syncthreads();
    if (d == 0) {
        float M=-1e30f;
        for (int sp=0;sp<KSPLIT;sp++) M=fmaxf(M, sm_[sp]);
        float den=0.f;
        for (int sp=0;sp<KSPLIT;sp++) { float e=exp2f(sm_[sp]-M); se_[sp]=e; den += sl_[sp]*e; }
        Mden[0]=M; Mden[1]=(den>0.f)?den:1.f;
    }
    __syncthreads();
    if (d < HD) {
        float num=0.f;
        for (int sp=0;sp<KSPLIT;sp++) num += base[sp*(HD+2)+d]*se_[sp];
        out[((long)qi*NQ + h)*HD + d] = __float2bfloat16(num / Mden[1]);
    }
}
