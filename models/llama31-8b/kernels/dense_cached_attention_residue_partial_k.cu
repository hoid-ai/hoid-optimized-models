#include <cuda_bf16.h>
#define TO_F(x) __bfloat162float(x)
constexpr int RESIDUES = 64;
constexpr int NQ = 32;
constexpr int NKV = 8;
constexpr int GROUP = 4;
constexpr int HD = 128;
constexpr int TILE_KEYS = 4;
constexpr int RECORD = HD + 2;
constexpr int MAX_SEQ = 9216;
constexpr float SCALE = 0.08838834764831845f;

__device__ __forceinline__ void cp_async_16(void* dst, const void* src, int valid_bytes) {
  unsigned smem = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::
               "r"(smem), "l"(src), "r"(valid_bytes));
}

extern "C" __global__ void dense_cached_attention_residue_partial_k(
    float* out, const __nv_bfloat16* in_0,
    const __nv_bfloat16* in_1, const __nv_bfloat16* in_2,
    const int* dyn_dims) {
  int residue = blockIdx.x % RESIDUES;
  int qkv = blockIdx.x / RESIDUES;
  int kvh = qkv % NKV;
  int qi = qkv / NKV;
  int lane = threadIdx.x & 31;
  bool consumer = threadIdx.x < 128;
  int qg = threadIdx.x >> 5;
  int qpos = dyn_dims[14] + qi;
  int residue_count = qpos >= residue ? (qpos - residue) / RESIDUES + 1 : 0;

  float q_reg[4] = {0.f, 0.f, 0.f, 0.f};
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  int h = kvh * GROUP + qg;
  if (consumer) {
    const __nv_bfloat16* qv = in_0 + ((long)qi * NQ + h) * HD;
    #pragma unroll
    for (int k = 0; k < 4; ++k)
      q_reg[k] = TO_F(qv[lane + k * 32]);
  }

  __shared__ __nv_bfloat16 sk[2][TILE_KEYS * HD];
  __shared__ __nv_bfloat16 sv[2][TILE_KEYS * HD];

  int producer_lane = threadIdx.x - 128;
  if (producer_lane >= 0) {
    int elem = producer_lane * 8;
    int t = elem / HD;
    int j = residue + t * RESIDUES;
    int bytes = t < residue_count ? 16 : 0;
    long off = ((long)kvh * MAX_SEQ + j) * HD + (elem % HD);
    cp_async_16(&sk[0][elem], in_1 + off, bytes);
    cp_async_16(&sv[0][elem], in_2 + off, bytes);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
  }
  __syncthreads();

  float m = -1.0e30f;
  float l = 0.f;
  for (int base = 0, stage = 0; base < residue_count; base += TILE_KEYS, stage ^= 1) {
    int next = base + TILE_KEYS;
    if (producer_lane >= 0 && next < residue_count) {
      int elem = producer_lane * 8;
      int t = elem / HD;
      int r = next + t;
      int j = residue + r * RESIDUES;
      int bytes = r < residue_count ? 16 : 0;
      long off = ((long)kvh * MAX_SEQ + j) * HD + (elem % HD);
      cp_async_16(&sk[stage ^ 1][elem], in_1 + off, bytes);
      cp_async_16(&sv[stage ^ 1][elem], in_2 + off, bytes);
      asm volatile("cp.async.commit_group;");
      asm volatile("cp.async.wait_group 0;");
    }

    if (consumer) {
      float scores[TILE_KEYS] = {0.f, 0.f, 0.f, 0.f};
      #pragma unroll
      for (int k = 0; k < 4; ++k) {
        float qk = q_reg[k];
        scores[0] += qk * TO_F(sk[stage][0 * HD + lane + k * 32]);
        scores[1] += qk * TO_F(sk[stage][1 * HD + lane + k * 32]);
        scores[2] += qk * TO_F(sk[stage][2 * HD + lane + k * 32]);
        scores[3] += qk * TO_F(sk[stage][3 * HD + lane + k * 32]);
      }
      #pragma unroll
      for (int delta = 16; delta > 0; delta >>= 1) {
        scores[0] += __shfl_xor_sync(0xffffffffu, scores[0], delta);
        scores[1] += __shfl_xor_sync(0xffffffffu, scores[1], delta);
        scores[2] += __shfl_xor_sync(0xffffffffu, scores[2], delta);
        scores[3] += __shfl_xor_sync(0xffffffffu, scores[3], delta);
      }

      #pragma unroll
      for (int t = 0; t < TILE_KEYS; ++t) {
        if (base + t < residue_count) {
          float score = scores[t] * SCALE;
          float m_new = fmaxf(m, score);
          float corr = expf(m - m_new);
          float p = expf(score - m_new);
          l = l * corr + p;
          #pragma unroll
          for (int k = 0; k < 4; ++k)
            acc[k] = acc[k] * corr +
                     p * TO_F(sv[stage][t * HD + lane + k * 32]);
          m = m_new;
        }
      }
    }
    __syncthreads();
  }

  if (consumer) {
    float* rec = out + (((long)qi * NQ + h) * RESIDUES + residue) * RECORD;
    #pragma unroll
    for (int k = 0; k < 4; ++k)
      rec[lane + k * 32] = acc[k];
    if (lane == 0) {
      rec[HD] = m;
      rec[HD + 1] = l;
    }
  }
}
