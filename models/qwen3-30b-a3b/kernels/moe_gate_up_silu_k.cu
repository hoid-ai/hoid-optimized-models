#include <cuda_bf16.h>
extern "C" __global__ void moe_gate_up_silu_k(
    float* act,                    // [s, K, INTER] (f32 intermediate)
    const __nv_bfloat16* x,        // [s, HID]
    const int* ids,                // [s, K]
    const float* w,                // [s, K] routing weights (folded in here)
    const __nv_bfloat16* gate_up,  // [E, 2*INTER, HID]
    const int* dyn_dims)
{
    const int HID = 2048, INTER = 768, K = 8;
    int t = blockIdx.x;
    int s = dyn_dims[18];
    if (t >= s) return;
    int kk = blockIdx.y;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wpb = blockDim.x >> 5;
    int j = blockIdx.z * wpb + warp;
    extern __shared__ float xsh[]; // HID
    for (int d = threadIdx.x; d < HID; d += blockDim.x) xsh[d] = __bfloat162float(x[(long)t * HID + d]);
    __syncthreads();
    if (j >= INTER) return;
    int e = ids[(long)t * K + kk];
    const __nv_bfloat16* grow = gate_up + ((long)e * (2 * INTER) + j) * HID;
    const __nv_bfloat16* urow = gate_up + ((long)e * (2 * INTER) + INTER + j) * HID;
    float g = 0.f, u = 0.f;
    for (int d = lane; d < HID; d += 32) { float xv = xsh[d]; g += xv * __bfloat162float(grow[d]); u += xv * __bfloat162float(urow[d]); }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) { g += __shfl_xor_sync(0xffffffff, g, o); u += __shfl_xor_sync(0xffffffff, u, o); }
    // Fold the routing weight into act here so down_accum no longer needs topk_w
    // (math-identical: w*(act.down) == (w*act).down); also lets down read the
    // router at slot 0 only, which the fuse path can address.
    if (lane == 0) { float sg = g / (1.f + __expf(-g)); act[((long)t * K + kk) * INTER + j] = w[(long)t * K + kk] * sg * u; }
}
