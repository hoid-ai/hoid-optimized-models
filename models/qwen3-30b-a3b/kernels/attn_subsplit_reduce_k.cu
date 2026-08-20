extern "C" __global__ void attn_subsplit_reduce_k(
    float* part, const float* fine, const int* dyn_dims) {
  const int NQ=32, HD=128, OUTER=8, SUB=4, FINE=32;
  const int qi=blockIdx.x, h=blockIdx.y, os=blockIdx.z, d=threadIdx.x;
  if (qi>=dyn_dims[18]) return;
  const float* src=fine+((long)qi*NQ+h)*FINE*(HD+2);
  float* dst=part+(((long)qi*NQ+h)*OUTER+os)*(HD+2);
  __shared__ float weights[SUB], M, L;
  if (d==0) {
    float mm=-1e30f;
    #pragma unroll
    for (int t=0;t<SUB;++t) mm=fmaxf(mm,src[(os*SUB+t)*(HD+2)+HD]);
    float ll=0.f;
    #pragma unroll
    for (int t=0;t<SUB;++t) {
      const float w=exp2f(src[(os*SUB+t)*(HD+2)+HD]-mm);
      weights[t]=w; ll+=src[(os*SUB+t)*(HD+2)+HD+1]*w;
    }
    M=mm; L=ll;
  }
  __syncthreads();
  if (d<HD) {
    float a=0.f;
    #pragma unroll
    for (int t=0;t<SUB;++t) a+=src[(os*SUB+t)*(HD+2)+d]*weights[t];
    dst[d]=a;
  }
  if (d==0) { dst[HD]=M; dst[HD+1]=L; }
}
