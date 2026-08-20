#include <cuda_bf16.h>
union PairBits { unsigned u; __nv_bfloat162 b; };
__device__ __forceinline__ unsigned silu_pair(unsigned gb, unsigned ub) {
    PairBits g, u, o; g.u = gb; u.u = ub;
    float2 gf = __bfloat1622float2(g.b);
    float2 uf = __bfloat1622float2(u.b);
    float sx = gf.x / (1.0f + __expf(-gf.x));
    float sy = gf.y / (1.0f + __expf(-gf.y));
    o.b = __floats2bfloat162_rn(sx * uf.x, sy * uf.y);
    return o.u;
}
extern "C" __global__ void silu_mul_vec8_fast_k(__nv_bfloat16* out_0, const __nv_bfloat16* in_0, const __nv_bfloat16* in_1, const int* dyn_dims) {
    const long vi = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long nv = ((long)dyn_dims[18] * 9728) / 8;
    if (vi < nv) {
        uint4 g = reinterpret_cast<const uint4*>(in_0)[vi];
        uint4 u = reinterpret_cast<const uint4*>(in_1)[vi];
        uint4 o;
        o.x = silu_pair(g.x, u.x); o.y = silu_pair(g.y, u.y);
        o.z = silu_pair(g.z, u.z); o.w = silu_pair(g.w, u.w);
        reinterpret_cast<uint4*>(out_0)[vi] = o;
    }
}