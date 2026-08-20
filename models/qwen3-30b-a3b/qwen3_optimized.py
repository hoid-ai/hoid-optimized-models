#!/usr/bin/env python
"""Qwen3-30B-A3B-Instruct-2507's decode step in PyTorch, on the hoid kernels.

The kernels come verbatim from the hoid decode campaign's winning stack,
**4.767 ms/step** on a B200 (the tuned-torch comparison at 1k and 8k prompts
lives in README.md):

  `prepare_qkv`      q/k RMSNorm + RoPE + the KV-cache write, one warp per
                     (token, head)                       [generated built-in]
  `attn_fine`        GQA-8 split-K flash-decode partials, 64 splits with
                     boundary-preserving bisection of the 8x4 base intervals
  `attn_pair`        pairwise 64 -> 32 partial merge
  `attn_reduce`      grouped 32 -> 8 partial merge
  `attn_combine`     final 8 -> 1 merge, bf16 out
  `moe_router`       exact softmax top-8 with the reference's tie-break and
                     bf16-rounded normalized weights
  `moe_gate_up`      per-(expert-slot, row) warp GEMV + SiLU, routing weight
                     folded into the f32 activation
  `moe_down`         per-output-dim warp accumulation over the 8 experts
  `fullcta_add_rmsnorm`        residual add + RMSNorm emitting both tensors
  `fullcta_add_rmsnorm_final`  the last residual add + model.norm, norm only
  `rmsnorm`          layer 0's input norm                [generated built-in]

Everything else is a cuBLAS matmul on the HF-layout weight (transposed view,
exactly the graph's `rhs_transpose: true` dispatch), except that q/k/v are
packed into one [5120, 2048] weight so the three skinny decode GEMVs become
one launch — same numbers per row, one cuBLAS call instead of three.

A decode step is ~4.8 ms over ~800 launches, so the only sane deployment is a
captured CUDA graph: every buffer is allocated up front, the cache write and
the attention extent are data-driven through `dyn_dims[14]` (the position), and
one captured graph serves every position.

Expert weights are NOT duplicated: transformers 5.x already stores each
layer's experts fused as `gate_up_proj` [128, 1536, 2048] (gate rows first)
and `down_proj` [128, 2048, 768] — byte-for-byte the layout `moe_gate_up` /
`moe_down` index — so the kernels read the stock model's own parameters.
"""

from __future__ import annotations

import hashlib
import math
import os

import torch

from qwen3_kernels_gen import KERNELS

try:
    from qwen3_elastic_kernels_gen import KERNELS as ELASTIC_KERNELS
except ImportError:  # decode-only checkout
    ELASTIC_KERNELS = {}

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, "kernels")

HID = 2048
NQ = 32
NKV = 4
HEAD_DIM = 128
LAYERS = 48
EXPERTS = 128
TOPK = 8
INTER = 768
VOCAB = 151936
MAX_SEQ = 2048          # the decode graph's cache window (memory bucket max)
ROPE_THETA = 1.0e7
QKV_OUT = NQ * HEAD_DIM + 2 * NKV * HEAD_DIM   # 5120


def _src(key: str) -> str:
    """The kernel source, hash-checked against the manifest."""
    meta = KERNELS[key]
    with open(os.path.join(KDIR, meta["file"])) as f:
        text = f.read()
    got = hashlib.sha256(text.encode()).hexdigest()
    if got != meta["sha256"]:
        raise RuntimeError(
            f"kernel {key} ({meta['file']}) hashes {got[:16]}, manifest says "
            f"{meta['sha256'][:16]} — source drifted from the manifest.")
    return text


# The cache window is a compile-time constant in exactly two kernels (every
# other 2048 is the hidden size); for longer contexts it is substituted AFTER
# the hash check, count-asserted so a kernel edit can't silently change what
# gets patched.
_WINDOW_SUBS = {
    "attn_gqa8_subsplit64_partial_k": ("MAXSEQ=2048", "MAXSEQ={w}", 1),
    "prepare_qkv_k": ("* 2048 + pos)", "* {w} + pos)", 2),
}


def _window_src(key: str, window: int) -> str:
    text = _src(key)
    if window == MAX_SEQ or key not in _WINDOW_SUBS:
        return text
    pat, repl, count = _WINDOW_SUBS[key]
    found = text.count(pat)
    if found != count:
        raise RuntimeError(
            f"kernel {key}: cache-window pattern {pat!r} occurs {found}x, "
            f"expected {count} — the source drifted; update _WINDOW_SUBS.")
    return text.replace(pat, repl.format(w=window))


# ---------------------------------------------------------------------------
# Launch wrappers: each kernel with the geometry the graph measured, at the
# decode shape s = 1 (`seq:dyn` in the manifest; the buffers are s=1 too).
# `dyn` is the runtime's 26-int dynamic-dims buffer: [18] = s, [14] = position.
# ---------------------------------------------------------------------------
_CUDA_WRAPPERS = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

using bf16 = __nv_bfloat16;
#define STREAM at::cuda::getCurrentCUDAStream().stream()
#define P(t) reinterpret_cast<bf16*>((t).data_ptr())
#define C(t) reinterpret_cast<const bf16*>((t).data_ptr())
#define PF(t) reinterpret_cast<float*>((t).data_ptr())
#define CF(t) reinterpret_cast<const float*>((t).data_ptr())
#define PI(t) reinterpret_cast<int*>((t).data_ptr())
#define CI(t) reinterpret_cast<const int*>((t).data_ptr())

// grid s*40 x block 32 — one warp per (token, head); h<32 q, 32..35 k, 36..39 v
void prepare_qkv(torch::Tensor out_q, torch::Tensor q, torch::Tensor k,
                 torch::Tensor v, torch::Tensor qn_w, torch::Tensor kn_w,
                 torch::Tensor kcache, torch::Tensor vcache,
                 torch::Tensor cos_t, torch::Tensor sin_t, torch::Tensor dyn) {
    prepare_qkv_k<<<40, 32, 0, STREAM>>>(
        P(out_q), C(q), C(k), C(v), C(qn_w), C(kn_w), P(kcache), P(vcache),
        CF(cos_t), CF(sin_t), CI(dyn));
}

// grid (s, 4, 64) x block 256 — 8 q-heads (one GQA group) per block
void attn_fine(torch::Tensor part, torch::Tensor q, torch::Tensor kc,
               torch::Tensor vc, torch::Tensor dyn) {
    attn_gqa8_subsplit64_partial_k<<<dim3(1, 4, 64), 256, 0, STREAM>>>(
        PF(part), C(q), C(kc), C(vc), CI(dyn));
}

// grid (s, 32, 32) x block 128
void attn_pair(torch::Tensor coarse, torch::Tensor fine, torch::Tensor dyn) {
    attn_subsplit64_pair_reduce_k<<<dim3(1, 32, 32), 128, 0, STREAM>>>(
        PF(coarse), CF(fine), CI(dyn));
}

// grid (s, 32, 8) x block 128
void attn_reduce(torch::Tensor part, torch::Tensor fine, torch::Tensor dyn) {
    attn_subsplit_reduce_k<<<dim3(1, 32, 8), 128, 0, STREAM>>>(
        PF(part), CF(fine), CI(dyn));
}

// grid (s, 32, 1) x block 128
void attn_combine(torch::Tensor out, torch::Tensor part, torch::Tensor dyn) {
    attn_combine_k<<<dim3(1, 32, 1), 128, 0, STREAM>>>(
        P(out), CF(part), CI(dyn));
}

// grid s x block 128 — one block per token over the 128 router logits
void moe_router(torch::Tensor ids, torch::Tensor w, torch::Tensor logits,
                torch::Tensor dyn) {
    moe_router_parallel_topk_k<<<1, 128, 0, STREAM>>>(
        PI(ids), PF(w), C(logits), CI(dyn));
}

// grid (s, 8, 96) x block 256, smem 8192 — warp per (expert slot, inter row)
void moe_gate_up(torch::Tensor act, torch::Tensor x, torch::Tensor ids,
                 torch::Tensor w, torch::Tensor gate_up, torch::Tensor dyn) {
    moe_gate_up_silu_k<<<dim3(1, 8, 96), 256, 8192, STREAM>>>(
        PF(act), C(x), CI(ids), CF(w), C(gate_up), CI(dyn));
}

// grid (s, 256) x block 256, smem 24608 — warp per output dim, 8-expert loop
void moe_down(torch::Tensor out, torch::Tensor act, torch::Tensor ids,
              torch::Tensor down, torch::Tensor dyn) {
    moe_down_accum_k<<<dim3(1, 256), 256, 24608, STREAM>>>(
        P(out), CF(act), CI(ids), C(down), CI(dyn));
}

// grid s x block 1024 — residual add + RMSNorm, both tensors out
void fullcta(torch::Tensor residual, torch::Tensor norm, torch::Tensor lhs,
             torch::Tensor rhs, torch::Tensor weight, torch::Tensor dyn) {
    fullcta_add_rmsnorm_k<<<1, 1024, 0, STREAM>>>(
        P(residual), P(norm), C(lhs), C(rhs), C(weight), CI(dyn));
}

// grid s x block 1024 — the last residual add + model.norm, norm only
void fullcta_final(torch::Tensor norm, torch::Tensor lhs, torch::Tensor rhs,
                   torch::Tensor weight, torch::Tensor dyn) {
    fullcta_add_rmsnorm_final_k<<<1, 1024, 0, STREAM>>>(
        P(norm), C(lhs), C(rhs), C(weight), CI(dyn));
}

// grid s x block 256 — layer 0's plain input RMSNorm
void rmsnorm(torch::Tensor out, torch::Tensor x, torch::Tensor w,
             torch::Tensor dyn) {
    rmsnorm_k<<<1, 256, 0, STREAM>>>(P(out), C(x), C(w), CI(dyn));
}

// prefill: grid n x block 1024 — fused residual add + RMSNorm over n rows
void fullcta_n(torch::Tensor residual, torch::Tensor norm, torch::Tensor lhs,
               torch::Tensor rhs, torch::Tensor weight, torch::Tensor dyn,
               int64_t n) {
    fullcta_add_rmsnorm_k<<<(unsigned)n, 1024, 0, STREAM>>>(
        P(residual), P(norm), C(lhs), C(rhs), C(weight), CI(dyn));
}

// prefill: grid n x block 1024 — the last residual add + model.norm
void fullcta_final_n(torch::Tensor norm, torch::Tensor lhs, torch::Tensor rhs,
                     torch::Tensor weight, torch::Tensor dyn, int64_t n) {
    fullcta_add_rmsnorm_final_k<<<(unsigned)n, 1024, 0, STREAM>>>(
        P(norm), C(lhs), C(rhs), C(weight), CI(dyn));
}

// prefill: grid n x block 256 — plain RMSNorm over n rows
void rmsnorm_n(torch::Tensor out, torch::Tensor x, torch::Tensor w,
               torch::Tensor dyn, int64_t n) {
    rmsnorm_k<<<(unsigned)n, 256, 0, STREAM>>>(P(out), C(x), C(w), CI(dyn));
}

// prefill: the same elastic kernel over n tokens (dyn[18] = n, dyn[14] = 0)
void prepare_qkv_n(torch::Tensor out_q, torch::Tensor q, torch::Tensor k,
                   torch::Tensor v, torch::Tensor qn_w, torch::Tensor kn_w,
                   torch::Tensor kcache, torch::Tensor vcache,
                   torch::Tensor cos_t, torch::Tensor sin_t, torch::Tensor dyn,
                   int64_t n_tokens) {
    prepare_qkv_k<<<(unsigned)(n_tokens * 40), 32, 0, STREAM>>>(
        P(out_q), C(q), C(k), C(v), C(qn_w), C(kn_w), P(kcache), P(vcache),
        CF(cos_t), CF(sin_t), CI(dyn));
}
"""

_CUDA_DECLS = """
void prepare_qkv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void attn_fine(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void attn_pair(torch::Tensor, torch::Tensor, torch::Tensor);
void attn_reduce(torch::Tensor, torch::Tensor, torch::Tensor);
void attn_combine(torch::Tensor, torch::Tensor, torch::Tensor);
void moe_router(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void moe_gate_up(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void moe_down(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void fullcta(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void fullcta_final(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void rmsnorm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void prepare_qkv_n(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
void fullcta_n(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
void fullcta_final_n(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
void rmsnorm_n(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
"""

_KERNEL_KEYS = (
    "prepare_qkv_k", "attn_gqa8_subsplit64_partial_k",
    "attn_subsplit64_pair_reduce_k", "attn_subsplit_reduce_k",
    "attn_combine_k", "moe_router_parallel_topk_k", "moe_gate_up_silu_k",
    "moe_down_accum_k", "fullcta_add_rmsnorm_k", "fullcta_add_rmsnorm_final_k",
    "rmsnorm_k",
)

_FUNCTIONS = [
    "prepare_qkv", "attn_fine", "attn_pair", "attn_reduce", "attn_combine",
    "moe_router", "moe_gate_up", "moe_down", "fullcta", "fullcta_final",
    "rmsnorm", "prepare_qkv_n", "fullcta_n", "fullcta_final_n", "rmsnorm_n",
]

KM = None
_BUILT_WINDOW = None


def build_kernels(verbose: bool = True, window: int = MAX_SEQ):
    global KM, _BUILT_WINDOW
    if _BUILT_WINDOW == window:
        return
    if _BUILT_WINDOW is not None:
        raise RuntimeError(
            f"kernels already built for window {_BUILT_WINDOW}; one cache "
            f"window per process.")
    from torch.utils.cpp_extension import load_inline

    if verbose:
        print(f"[build] nvcc: 11 decode kernels (window {window}) ...")
    KM = load_inline(
        name=f"qwen3_30b_hoid_kernels_w{window}",
        cpp_sources=[_CUDA_DECLS],
        cuda_sources=["\n".join(_window_src(k, window) for k in _KERNEL_KEYS)
                      + _CUDA_WRAPPERS],
        functions=_FUNCTIONS,
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    _BUILT_WINDOW = window


def rope_tables(device, window: int = MAX_SEQ) -> tuple[torch.Tensor, torch.Tensor]:
    """The graph's cos/sin tables, same construction: f64 inv_freq rounded to
    f32, f32 angles on device, cos as sin(x + pi/2)."""
    half = HEAD_DIM // 2
    inv = torch.tensor(
        [ROPE_THETA ** (-2.0 * j / HEAD_DIM) for j in range(half)],
        dtype=torch.float64,
    ).to(torch.float32).to(device)
    pos = torch.arange(window, device=device, dtype=torch.float32).unsqueeze(1)
    angles = pos * inv.unsqueeze(0)
    return torch.sin(angles + math.pi / 2).contiguous(), torch.sin(angles).contiguous()


_TMP_MODULES: list[str] = []  # keep tempdirs alive for the process lifetime


class TritonFMHA:
    """kernels/fmha_prefill_elastic.triton.py — one shape-generic FA2
    (two-phase causal loop, device-side TMA descriptors over the KV caches)
    whose grid and store masks read the prompt length from dyn_dims at
    launch. NQH/GROUP/HDIM/MAXSEQ are launch constexprs; the KV-head count
    is the one in-source constant and is substituted for this model after
    the byte-identical hash check, like the cache-window constants."""

    def __init__(self, meta: dict, window: int):
        import importlib.util
        import tempfile

        import triton

        src = open(os.path.join(HERE, meta["file"])).read()
        got = hashlib.sha256(src.encode()).hexdigest()
        assert got == meta["sha256"], f"{meta['file']}: sha mismatch — kernels/ edited?"
        pat, repl = "NKVH: tl.constexpr = 8", f"NKVH: tl.constexpr = {NKV}"
        assert src.count(pat) == 1, "FA2 KV-head bake-in moved"
        src = src.replace(pat, repl)
        d = tempfile.mkdtemp(prefix="q30b_tfmha_")
        _TMP_MODULES.append(d)
        path = os.path.join(d, "q30b_tfmha_module.py")
        with open(path, "w") as f:
            f.write("import triton\nimport triton.language as tl\n\n" + src)
        spec = importlib.util.spec_from_file_location("q30b_tfmha_module", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._kernel = getattr(mod, meta["entry"])
        cx = dict(meta["params"]["constexprs"])
        cx["GROUP"] = NQ // NKV
        cx["MAXSEQ"] = window
        self._cx = cx
        self._nw = int(meta["params"]["num_warps"])
        self._ns = int(meta["params"]["num_stages"])
        # tl.make_tensor_descriptor allocates its descriptors host-side
        triton.set_allocator(
            lambda size, align, stream: torch.empty(size, device="cuda",
                                                    dtype=torch.int8))

    def __call__(self, out, q, kc, vc, dyn, seq: int):
        bm = self._cx["BLOCK_M"]
        grid = ((seq + bm - 1) // bm, self._cx["NQH"], 1)
        self._kernel[grid](out, q, kc, vc, dyn,
                           num_warps=self._nw, num_stages=self._ns, **self._cx)


def expert_views(model) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """The stock model's fused expert parameters, which transformers 5.x
    stores in exactly the layout the kernels index: gate_up_proj
    [E, 2*INTER, HID] with gate rows first, down_proj [E, HID, INTER]."""
    gate_up, down = [], []
    for layer in model.model.layers:
        gu = layer.mlp.experts.gate_up_proj.detach()
        dn = layer.mlp.experts.down_proj.detach()
        assert gu.shape == (EXPERTS, 2 * INTER, HID) and gu.is_contiguous(), gu.shape
        assert dn.shape == (EXPERTS, HID, INTER) and dn.is_contiguous(), dn.shape
        gate_up.append(gu)
        down.append(dn)
    return gate_up, down


# ---------------------------------------------------------------------------
class OptimizedQwen3MoeDecoder:
    """One cached decode step on the hoid kernel stack.

    Weights are views into the HF `Qwen3MoeForCausalLM` (only q/k/v are
    copied, into one packed weight per layer); the caches and every
    intermediate are allocated once, up front, so the step can be captured
    into a CUDA graph.
    """

    def __init__(self, model, device: str = "cuda", cache_window: int = MAX_SEQ):
        build_kernels(verbose=False, window=cache_window)
        self.device = device
        self.window = cache_window
        m = model.model

        self.embed = m.embed_tokens.weight.detach()
        self.lm_head_t = model.lm_head.weight.detach().t()   # [2048, V] view

        self.qkv_t, self.o_t, self.gate_t = [], [], []
        # contiguous per-projection copies for the prefill GEMMs (the packed
        # qkv weight would force strided-view GEMMs at M = S)
        self.pq_t, self.pk_t, self.pv_t = [], [], []
        self.qn, self.kn, self.ln_in, self.post_ln = [], [], [], []
        for layer in m.layers:
            a = layer.self_attn
            # One [5120, 2048] weight: three skinny GEMVs become one launch.
            qkv = torch.cat([a.q_proj.weight.detach(), a.k_proj.weight.detach(),
                             a.v_proj.weight.detach()]).contiguous()
            self.qkv_t.append(qkv.t())
            self.pq_t.append(a.q_proj.weight.detach().t().contiguous())
            self.pk_t.append(a.k_proj.weight.detach().t().contiguous())
            self.pv_t.append(a.v_proj.weight.detach().t().contiguous())
            self.o_t.append(a.o_proj.weight.detach().t())
            self.gate_t.append(layer.mlp.gate.weight.detach().t())
            self.qn.append(a.q_norm.weight.detach().contiguous())
            self.kn.append(a.k_norm.weight.detach().contiguous())
            self.ln_in.append(layer.input_layernorm.weight.detach().contiguous())
            self.post_ln.append(layer.post_attention_layernorm.weight.detach().contiguous())
        self.final_ln = m.norm.weight.detach().contiguous()

        self.gate_up, self.down = expert_views(model)
        # grouped-GEMM operand views for the prefill: [E, K, N]
        self.gate_up_t = [w.transpose(1, 2) for w in self.gate_up]
        self.down_t = [w.transpose(1, 2) for w in self.down]
        self.cos_t, self.sin_t = rope_tables(device, cache_window)

        self.tfmha = (TritonFMHA(ELASTIC_KERNELS["fmha_prefill_elastic"],
                                 cache_window)
                      if ELASTIC_KERNELS else None)
        self._pbuf: dict[int, dict[str, torch.Tensor]] = {}
        self._pgraph: dict[int, torch.cuda.CUDAGraph] = {}

        self._alloc(device)
        self.graph = None

    def _alloc(self, dev):
        def z(*shape, dtype=torch.bfloat16):
            return torch.zeros(*shape, device=dev, dtype=dtype)

        self.tok_id = torch.zeros(1, device=dev, dtype=torch.int32)
        # The runtime's 26-int dynamic-dims buffer: [18] = 's' (new tokens,
        # always 1 here), [14] = 'o' (the token's absolute position).
        self.dyn = torch.zeros(26, device=dev, dtype=torch.int32)
        self.dyn[18] = 1
        self.dyn_pos = self.dyn[14:15]

        self.kcache = [z(NKV, self.window, HEAD_DIM) for _ in range(LAYERS)]
        self.vcache = [z(NKV, self.window, HEAD_DIM) for _ in range(LAYERS)]

        self.res_a = z(1, HID)          # residual stream (starts as the embedding row)
        self.res_b = z(1, HID)
        self.h = z(1, HID)              # the normed row every consumer reads
        self.qkv = z(1, QKV_OUT)
        self.q = z(1, NQ * HEAD_DIM)    # prepare_qkv's normed+roped q
        self.fine = z(NQ * 64 * (HEAD_DIM + 2), dtype=torch.float32)
        self.coarse = z(NQ * 32 * (HEAD_DIM + 2), dtype=torch.float32)
        self.part8 = z(NQ * 8 * (HEAD_DIM + 2), dtype=torch.float32)
        self.attn = z(1, NQ * HEAD_DIM)
        self.o = z(1, HID)
        self.router = z(1, EXPERTS)
        self.ids = z(TOPK, dtype=torch.int32)
        self.topw = z(TOPK, dtype=torch.float32)
        self.act = z(TOPK * INTER, dtype=torch.float32)
        self.moe = z(1, HID)
        self.logits = z(1, VOCAB)

    def reset_cache(self):
        for k, v in zip(self.kcache, self.vcache):
            k.zero_()
            v.zero_()

    # -- one step, on static buffers ---------------------------------------
    def step_static(self):
        """Run one decode step from `tok_id`/`dyn[14]` into `logits`."""
        torch.index_select(self.embed, 0, self.tok_id, out=self.res_a)
        KM.rmsnorm(self.h, self.res_a, self.ln_in[0], self.dyn)
        for i in range(LAYERS):
            torch.matmul(self.h, self.qkv_t[i], out=self.qkv)
            KM.prepare_qkv(self.q, self.qkv[:, :NQ * HEAD_DIM],
                           self.qkv[:, NQ * HEAD_DIM:(NQ + NKV) * HEAD_DIM],
                           self.qkv[:, (NQ + NKV) * HEAD_DIM:],
                           self.qn[i], self.kn[i], self.kcache[i], self.vcache[i],
                           self.cos_t, self.sin_t, self.dyn)
            KM.attn_fine(self.fine, self.q, self.kcache[i], self.vcache[i], self.dyn)
            KM.attn_pair(self.coarse, self.fine, self.dyn)
            KM.attn_reduce(self.part8, self.coarse, self.dyn)
            KM.attn_combine(self.attn, self.part8, self.dyn)
            torch.matmul(self.attn, self.o_t[i], out=self.o)
            KM.fullcta(self.res_b, self.h, self.res_a, self.o, self.post_ln[i], self.dyn)
            torch.matmul(self.h, self.gate_t[i], out=self.router)
            KM.moe_router(self.ids, self.topw, self.router, self.dyn)
            KM.moe_gate_up(self.act, self.h, self.ids, self.topw,
                           self.gate_up[i], self.dyn)
            KM.moe_down(self.moe, self.act, self.ids, self.down[i], self.dyn)
            if i + 1 < LAYERS:
                KM.fullcta(self.res_a, self.h, self.res_b, self.moe,
                           self.ln_in[i + 1], self.dyn)
            else:
                KM.fullcta_final(self.h, self.res_b, self.moe, self.final_ln, self.dyn)
        torch.matmul(self.h, self.lm_head_t, out=self.logits)
        return self.logits

    # -- CUDA graph ---------------------------------------------------------
    def capture(self, warmup: int = 3):
        """Capture one step. The cache write and the attention extent are
        data-driven through `dyn[14]`, not shape-driven, so a single captured
        graph serves every position."""
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(warmup):
                self.step_static()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.reset_cache()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.step_static()
        return self

    def step(self, tok: int, pos: int) -> torch.Tensor:
        """One decode step; returns the logits buffer (overwritten next call)."""
        self.tok_id.fill_(int(tok))
        self.dyn_pos.fill_(int(pos))
        if self.graph is not None:
            self.graph.replay()
        else:
            with torch.no_grad():
                self.step_static()
        return self.logits

    # -- prefill on the port's own stack ------------------------------------
    def _prefill_buffers(self, S: int) -> dict[str, torch.Tensor]:
        if S not in self._pbuf:
            z = lambda *sh, dt=torch.bfloat16: torch.zeros(*sh, device=self.device,
                                                           dtype=dt)
            self._pbuf[S] = {
                "ids": torch.zeros(S, device=self.device, dtype=torch.int32),
                "x": z(S, HID), "b": z(S, HID), "h": z(S, HID),
                "q3": z(S, NQ * HEAD_DIM), "k3": z(S, NKV * HEAD_DIM),
                "v3": z(S, NKV * HEAD_DIM), "q": z(S, NQ * HEAD_DIM),
                "att": z(S, NQ * HEAD_DIM), "o": z(S, HID), "moe": z(S, HID),
                "logits": z(1, VOCAB),
                "dnu": z(S * TOPK, HID),
                "ones8": torch.ones(S * TOPK, device=self.device,
                                    dtype=torch.int32),
                "counts": torch.zeros(EXPERTS, device=self.device,
                                      dtype=torch.int32),
                "pdyn": torch.zeros(26, device=self.device, dtype=torch.int32),
            }
            self._pbuf[S]["pdyn"][18] = S  # s; position stays 0 for prefill
        return self._pbuf[S]

    def _prefill_static(self, S: int) -> None:
        """The capturable prefill body — step_static's dataflow at M = S:
        fixed shapes and buffers throughout (the MoE sort/gather/grouped-GEMM
        chain is data-driven but shape-static), so the whole forward records
        into one CUDA graph."""
        e = self._pbuf[S]
        dyn = e["pdyn"]
        torch.index_select(self.embed, 0, e["ids"], out=e["x"])
        res_a, res_b = e["x"], e["b"]
        KM.rmsnorm_n(e["h"], res_a, self.ln_in[0], dyn, S)
        for i in range(LAYERS):
            torch.matmul(e["h"], self.pq_t[i], out=e["q3"])
            torch.matmul(e["h"], self.pk_t[i], out=e["k3"])
            torch.matmul(e["h"], self.pv_t[i], out=e["v3"])
            KM.prepare_qkv_n(e["q"], e["q3"], e["k3"], e["v3"],
                             self.qn[i], self.kn[i],
                             self.kcache[i], self.vcache[i],
                             self.cos_t, self.sin_t, dyn, S)
            self.tfmha(e["att"], e["q"], self.kcache[i], self.vcache[i], dyn, S)
            torch.matmul(e["att"], self.o_t[i], out=e["o"])
            KM.fullcta_n(res_b, e["h"], res_a, e["o"], self.post_ln[i], dyn, S)
            router = e["h"] @ self.gate_t[i]
            probs = torch.softmax(router.float(), dim=-1)
            topw, sel = torch.topk(probs, TOPK, dim=-1)
            topw = (topw / topw.sum(-1, keepdim=True)).to(torch.bfloat16)
            flat = sel.reshape(-1)
            # sort on int32 keys (half the radix passes of int64)
            order = torch.argsort(flat.to(torch.int32), stable=True)
            tok_of = order // TOPK
            # bincount sizes its output from a device max() — a host sync
            # that also breaks CUDA-graph capture; scatter_add is shape-static
            e["counts"].zero_()
            e["counts"].scatter_add_(0, flat, e["ones8"])
            offs = e["counts"].cumsum(0).to(torch.int32)
            xg = e["h"].index_select(0, tok_of)
            gu = torch._grouped_mm(xg, self.gate_up_t[i], offs=offs)
            act = torch.nn.functional.silu(gu[:, :INTER]) * gu[:, INTER:]
            dn = torch._grouped_mm(act, self.down_t[i], offs=offs)
            dn.mul_(topw.reshape(-1).index_select(0, order).unsqueeze(1))
            # every token has exactly TOPK pairs, so the weighted scatter-sum
            # is an unsort (permutation index_copy, no atomics) + a dense sum
            e["dnu"].index_copy_(0, order, dn)
            torch.sum(e["dnu"].view(S, TOPK, HID), dim=1, out=e["moe"])
            if i + 1 < LAYERS:
                KM.fullcta_n(res_a, e["h"], res_b, e["moe"],
                             self.ln_in[i + 1], dyn, S)
            else:
                KM.fullcta_final_n(e["h"], res_b, e["moe"], self.final_ln,
                                   dyn, S)
        torch.matmul(e["h"][S - 1:S], self.lm_head_t, out=e["logits"])

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """One forward over the whole prompt on the port's stack: the elastic
        prepare_qkv (writes the graph's cache layout directly — no HF-cache
        relayout afterwards), the shape-generic triton FA2 over the caches,
        cuBLAS projections, and the MoE as two grouped GEMMs over
        expert-sorted (token, expert) pairs instead of the stock per-expert
        loop. Router/norm math mirrors HF op for op (fp32 softmax, fp32
        variance). The whole forward replays as one CUDA graph per prompt
        length (a decode step's dispatch overhead, paid ~1500x per eager
        prefill, priced this in). Writes cache rows 0..S-1, returns
        last-position logits."""
        ids = input_ids.view(-1)
        S = ids.numel()
        assert S + 1 <= self.window, "prompt exceeds the cache window"
        assert self.tfmha is not None, "qwen3_elastic_kernels_gen.py missing"
        e = self._prefill_buffers(S)
        e["ids"].copy_(ids.to(torch.int32))
        if S not in self._pgraph:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._prefill_static(S)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._prefill_static(S)
            self._pgraph[S] = g
        self._pgraph[S].replay()
        return e["logits"]

