#include <cuda_bf16.h>
union PairBits { unsigned u; __nv_bfloat162 b; };
__device__ __forceinline__ unsigned add_pair(unsigned ab, unsigned bb) {
    PairBits a, b, o; a.u = ab; b.u = bb;
    float2 af = __bfloat1622float2(a.b);
    float2 bf = __bfloat1622float2(b.b);
    o.b = __floats2bfloat162_rn(af.x + bf.x, af.y + bf.y);
    return o.u;
}
extern "C" __global__ void add_bf16_vec8_k(__nv_bfloat16* out_0, const __nv_bfloat16* in_0, const __nv_bfloat16* in_1, const int* dyn_dims) {
    const long vi = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long nv = ((long)dyn_dims[18] * 4096) / 8;
    if (vi < nv) {
        uint4 a = reinterpret_cast<const uint4*>(in_0)[vi];
        uint4 b = reinterpret_cast<const uint4*>(in_1)[vi];
        uint4 o;
        o.x = add_pair(a.x, b.x); o.y = add_pair(a.y, b.y);
        o.z = add_pair(a.z, b.z); o.w = add_pair(a.w, b.w);
        reinterpret_cast<uint4*>(out_0)[vi] = o;
    }
}