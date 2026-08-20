extern "C" __global__ void attn_subsplit64_pair_reduce_k(
    float* coarse, const float* fine, const int* dyn_dims) {
  const int NQ=32, HD=128, FINE=64, COARSE=32;
  const int qi=blockIdx.x, h=blockIdx.y, g=blockIdx.z, d=threadIdx.x;
  if (qi>=dyn_dims[18]) return;
  const float* src=fine+((long)qi*NQ+h)*FINE*(HD+2);
  float* dst=coarse+(((long)qi*NQ+h)*COARSE+g)*(HD+2);
  const float* a=src+(2*g)*(HD+2);
  const float* b=a+(HD+2);
  __shared__ float wa, wb, mm, ll;
  if (d==0) {
    const float ma=a[HD], mb=b[HD];
    mm=fmaxf(ma,mb);
    wa=exp2f(ma-mm); wb=exp2f(mb-mm);
    ll=a[HD+1]*wa+b[HD+1]*wb;
  }
  __syncthreads();
  if (d<HD) dst[d]=a[d]*wa+b[d]*wb;
  if (d==0) { dst[HD]=mm; dst[HD+1]=ll; }
}
