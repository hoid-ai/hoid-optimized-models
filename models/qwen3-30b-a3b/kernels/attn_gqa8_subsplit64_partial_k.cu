#include <cuda_bf16.h>
extern "C" __global__ void __launch_bounds__(256, 4) attn_gqa8_subsplit64_partial_k(
    float* part, const __nv_bfloat16* q, const __nv_bfloat16* kc,
    const __nv_bfloat16* vc, const int* dyn_dims) {
  const int NQ=32, HD=128, GROUP=8, MAXSEQ=2048, FINE=64, OUTER=8, BASE_SUB=4, DPL=4;
  const int R=32;
  const float scale_log2=0.08838834764831843f*1.4426950408889634f;
  const int qi=blockIdx.x, kvh=blockIdx.y, fs=blockIdx.z;
  const int tid=threadIdx.x, warp=tid>>5, lane=tid&31, h=kvh*GROUP+warp;
  if (qi>=dyn_dims[18]) return;
  const int P=dyn_dims[14]+qi+1;
  // Bisect each incumbent (OUTER=8, BASE_SUB=4) interval rather
  // than repartitioning an outer interval into eighths.  For lengths
  // not divisible by eight those are different boundaries, and the
  // section gate observes this workspace before its consumer.
  const int base_fs=fs>>1, half=fs&1;
  const int outer_chunk=(P+OUTER-1)/OUTER;
  const int os=base_fs/BASE_SUB, ss=base_fs-os*BASE_SUB;
  const int outer0=os*outer_chunk;
  int outer1=outer0+outer_chunk; if (outer1>P) outer1=P;
  int outer_len=outer1-outer0; if (outer_len<0) outer_len=0;
  const int base_chunk=(outer_len+BASE_SUB-1)/BASE_SUB;
  const int base0=outer0+ss*base_chunk;
  int base1=base0+base_chunk; if (base1>outer1) base1=outer1;
  const int mid=base0+(base1-base0+1)/2;
  const int j0=half ? mid : base0;
  const int j1=half ? base1 : mid;
  const __nv_bfloat16* qv=q+((long)qi*NQ+h)*HD;
  float qr[DPL];
  #pragma unroll
  for (int k=0;k<DPL;++k) qr[k]=__bfloat162float(qv[lane+k*32]);
  float m=-1e30f, l=0.f, acc[DPL];
  #pragma unroll
  for (int k=0;k<DPL;++k) acc[k]=0.f;
  // Stage R rows per __syncthreads round trip instead of one: the serialized
  // 256 B-per-row walk pays a full memory latency per row, which is what made
  // the step scale at ~0.5 ms per 1k tokens of context. Same j order, same
  // per-row softmax updates -> bit-identical output.
  __shared__ __nv_bfloat16 ksh[R][HD], vsh[R][HD];
  for (int j=j0;j<j1;j+=R) {
    const int n=(j1-j<R)?(j1-j):R;
    for (int t=tid;t<n*HD;t+=256) {
      const int r=t>>7, cl=t&127;
      const long base=((long)kvh*MAXSEQ+(j+r))*HD;
      ksh[r][cl]=kc[base+cl];
      vsh[r][cl]=vc[base+cl];
    }
    __syncthreads();
    for (int r=0;r<n;++r) {
      float dot=0.f;
      #pragma unroll
      for (int k=0;k<DPL;++k) dot+=qr[k]*__bfloat162float(ksh[r][lane+k*32]);
      #pragma unroll
      for (int o=16;o>0;o>>=1) dot+=__shfl_xor_sync(0xffffffffu,dot,o);
      const float sc=dot*scale_log2, mn=fmaxf(m,sc);
      const float c=exp2f(m-mn), w=exp2f(sc-mn);
      l=l*c+w;
      #pragma unroll
      for (int k=0;k<DPL;++k) acc[k]=acc[k]*c+w*__bfloat162float(vsh[r][lane+k*32]);
      m=mn;
    }
    __syncthreads();
  }
  float* pb=part+(((long)qi*NQ+h)*FINE+fs)*(HD+2);
  #pragma unroll
  for (int k=0;k<DPL;++k) pb[lane+k*32]=acc[k];
  if (lane==0) { pb[HD]=m; pb[HD+1]=l; }
}
