"""Llama-3.1-8B-Instruct on the hoid-winning kernels — standalone PyTorch.

No hoid runtime anywhere: the kernels under kernels/ are the byte-identical
sources of the optimized hoid 8k stack, compiled with
torch.utils.cpp_extension.load_inline and launched with the exact geometry
the manifests record. The builtin ops the graphs use besides
cuBLAS matmuls (embedding / apply_rope / cache_write) are hoid's own generated
sources, materialized for this model's constants in llama31_native_gen.py.

Design (the qwen3-4b / whisper port pattern):
- hoid ABI preserved: every kernel takes `(out..., in..., const int* dyn_dims)`.
  dyn_dims is a real int32[26] device buffer; seq 's' at index 18, offset 'o'
  at index 14 — the kernels' fixed dyn_dims layout. The decode
  step reads its position from dyn_dims inside the kernels, so ONE captured
  CUDA graph serves every decode position.
- Static buffers rule: the whole decode working set (KV caches included) is
  allocated up front; every kernel writes into caller-provided buffers;
  matmuls use `out=`.
- MAX_SEQ (the KV-cache row stride, baked into cache_write and the decode
  attention) and RESIDUES (the decode attention's context split factor) are
  compile-time bake-ins re-derived for the requested cache size. The shipped
  graph froze RESIDUES=64 for its 8k regime; see `pick_residues`.
- Prefill is one shape-generic path at every prompt length: the elastic
  prefill graph's stack — prefetch RMSNorm, cuBLAS GEMMs, hoid's rope and
  cache-write builtins, vec8 elementwise, and the elastic triton FA2
  (`kernels/fmha_prefill_elastic.triton.py`: grid and masks read the prompt
  length from dyn_dims at launch; MAXSEQ re-baked per engine build). It
  writes the cache layout the decode kernels read:
  [n_kv_heads, MAX_SEQ, head_dim] per layer. Unlike the hoid prefill graph
  (pinned to emit [s, vocab] logits), the port projects only the last
  position to logits.
- Llama vs the qwen port: no q/k RMSNorm (so no prepare_qkv — rope and cache
  write stay hoid's separate builtin kernels, as in the winning graphs),
  untied lm_head, eps 1e-5, and llama3-scaled RoPE tables built host-side in
  f64 exactly as hoid builds them.
"""

from __future__ import annotations

import hashlib
import math
import os

import torch
from torch.utils.cpp_extension import load_inline

import llama31_native_gen as NG
from llama31_decode_kernels_gen import KERNELS as DECODE_KERNELS
from llama31_prefill_kernels_gen import KERNELS as PREFILL_KERNELS
from llama31_elastic_kernels_gen import KERNELS as ELASTIC_KERNELS

HERE = os.path.dirname(os.path.abspath(__file__))

HIDDEN = 4096
LAYERS = 32
NQ = 32
NKV = 8
HD = 128
HALF = HD // 2
Q_HID = NQ * HD          # 4096
KV_HID = NKV * HD        # 1024
INTER = 14336
VOCAB = 128256
EPS = 1e-5
SCALE = HD ** -0.5

# llama3 rope scaling (rope_type "llama3" in the checkpoint's config.json)
ROPE_THETA = 5e5
ROPE_FACTOR = 8.0
ROPE_LOW_FREQ_FACTOR = 1.0
ROPE_HIGH_FREQ_FACTOR = 4.0
ROPE_ORIGINAL_MAX_POS = 8192.0

# ---------------------------------------------------------------------------
# kernel compilation: one extension module per kernel (per-module keeps every
# source byte-identical; their file-scope constexprs would collide otherwise)
# ---------------------------------------------------------------------------

# entry -> (wrapper arg spec, block.x). Arg spec: (kind:dtype) in ABI order.
_WRAPPER_SPECS = {
    "rmsnorm_prefetch16_k": (["o:bf16", "i:bf16", "i:bf16"], 256),
    "add_bf16_vec8_k": (["o:bf16", "i:bf16", "i:bf16"], 256),
    "silu_mul_vec8_fast_k": (["o:bf16", "i:bf16", "i:bf16"], 256),
    "dense_cached_attention_residue_partial_k": (["o:f32", "i:bf16", "i:bf16", "i:bf16"], 192),
    "dense_cached_attention_residue_combine_k": (["o:bf16", "i:f32"], 32),
    "embedding_k": (["o:bf16", "i:bf16", "i:i32"], 256),
    "apply_rope_q_k": (["o:bf16", "i:bf16", "i:f32", "i:f32"], 256),
    "apply_rope_kv_k": (["o:bf16", "i:bf16", "i:f32", "i:f32"], 256),
    "cache_write_k": (["o:bf16", "i:bf16", "i:bf16"], 256),
}

_CPP_PRELUDE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#define STREAM at::cuda::getCurrentCUDAStream().stream()
#define PB(t)  reinterpret_cast<__nv_bfloat16*>((t).data_ptr())
#define CB(t)  reinterpret_cast<const __nv_bfloat16*>((t).data_ptr())
#define PF(t)  (t).data_ptr<float>()
#define CF(t)  (t).data_ptr<float>()
#define CI(t)  (t).data_ptr<int>()
"""


def _wrapper(entry: str) -> tuple[str, str]:
    """(cuda wrapper source, cpp prototype) for one kernel entry."""
    spec, block = _WRAPPER_SPECS[entry]
    args, calls = [], []
    for i, s in enumerate(spec):
        kind, dt = s.split(":")
        args.append(f"at::Tensor t{i}")
        cast = {"bf16": ("PB", "CB"), "f32": ("PF", "CF"), "i32": ("CI", "CI")}[dt]
        calls.append(f"{cast[0] if kind == 'o' else cast[1]}(t{i})")
    # cache_write is in-place on the cache: hoid aliases out to the cache
    # buffer; the wrapper mirrors that by passing the cache tensor twice
    # (t0 = out alias, t1 = the unused cache param), src as t2.
    if entry == "cache_write_k":
        calls[1] = "PB(t1)"
    args.append("at::Tensor dyn")
    calls.append("CI(dyn)")
    args.append("int64_t grid_x")
    sig = ", ".join(args)
    fn = f"run_{entry}"
    src = (
        f"void {fn}({sig}) {{\n"
        f"    {entry}<<<(unsigned)grid_x, {block}, 0, STREAM>>>({', '.join(calls)});\n"
        f"}}\n"
    )
    return src, f"void {fn}({sig});"


def _kernel_sources() -> dict[str, str]:
    """entry -> verbatim source, provenance-checked against the manifests."""
    srcs = {}
    for manifest in (DECODE_KERNELS, PREFILL_KERNELS):
        for entry, meta in manifest.items():
            if meta["format"] != "cuda":
                continue  # the manifest's cutedsl FMHA is not part of
                # the elastic prefill; attention runs the triton kernel
            if entry == "copy_vec8_4096_k":
                continue  # the graph's q_pad/o_crop; the port's buffers are static
            path = os.path.join(HERE, meta["file"])
            src = open(path).read()
            got = hashlib.sha256(src.encode()).hexdigest()
            assert got == meta["sha256"], f"{meta['file']}: sha mismatch — kernels/ edited?"
            srcs[entry] = src
    srcs["embedding_k"] = NG.EMBEDDING
    srcs["apply_rope_q_k"] = NG.APPLY_ROPE_Q
    srcs["apply_rope_kv_k"] = NG.APPLY_ROPE_KV
    srcs["cache_write_k"] = NG.CACHE_WRITE
    return srcs


def pick_residues(max_seq: int) -> int:
    """The decode attention's context split factor.

    The partial kernel launches `NKV * RESIDUES` blocks whatever the depth,
    and each block strides the cache by RESIDUES — this one constant decides
    how much of the device the attention fills. Measured whole-step on B200:
    at offset 8192 R=16/32/64/128 give 7.25/5.97/5.56/5.98 ms, and at
    offset 1024 R=64 still edges R=16 (4.79 vs 4.89 on the hoid graph;
    4.62 vs 4.72 through this port, 64/64 token parity both) — for THIS kernel the bigger split's combine
    overhead never outweighs its occupancy, so 64 wins at every depth and
    is used unconditionally. Re-measure across depths before assuming that
    for a different kernel or GPU (qwen's pipeline-variant kernel loses
    with big R at 1k).
    """
    return 64


def _rebake_constants(entry: str, src: str, max_seq: int) -> str:
    """Re-bake the compile-time constants the shipped graphs froze at their
    own regime: the KV-cache row stride (MAX_SEQ=9216) and the decode
    attention split factor (RESIDUES=64, derived at 8k context)."""
    r = pick_residues(max_seq)
    if entry == "dense_cached_attention_residue_partial_k":
        old = "constexpr int MAX_SEQ = 9216;"
        assert src.count(old) == 1, "partial kernel MAX_SEQ bake-in moved"
        src = src.replace(old, f"constexpr int MAX_SEQ = {max_seq};")
        if r != 64:
            old = "constexpr int RESIDUES = 64;"
            assert src.count(old) == 1, "partial kernel RESIDUES bake-in moved"
            src = src.replace(old, f"constexpr int RESIDUES = {r};")
        return src
    if entry == "dense_cached_attention_residue_combine_k":
        if r == 64:
            return src  # shipped split — keep the source byte-identical
        old = "constexpr int WARPS = 64;"
        assert src.count(old) == 1, "combine kernel WARPS bake-in moved"
        return src.replace(old, f"constexpr int WARPS = {r};")
    if entry == "cache_write_k":
        return src.replace("{MAX_SEQ}", str(max_seq))
    return src


_MODULES: dict[str, object] = {}


def build_kernels(max_seq: int, verbose: bool = False):
    """Compile all kernels for one MAX_SEQ; returns {entry: callable}."""
    out = {}
    for entry, src in _kernel_sources().items():
        key = f"{entry}_{max_seq}"
        if key not in _MODULES:
            wrapper, proto = _wrapper(entry)
            cuda_src = _rebake_constants(entry, src, max_seq) + _CPP_PRELUDE + wrapper
            _MODULES[key] = load_inline(
                name=f"l31_{key}",
                cpp_sources=[proto],
                cuda_sources=[cuda_src],
                functions=[f"run_{entry}"],
                extra_cuda_cflags=["-O3"],
                verbose=verbose,
            )
        out[entry] = getattr(_MODULES[key], f"run_{entry}")
    return out


_TMP_MODULES: list[str] = []  # keep tempdirs alive for the process lifetime


class TritonFMHA:
    """kernels/fmha_prefill_elastic.triton.py — the elastic prefill graph's
    attention: one shape-generic FA2 (two-phase causal loop, device-side TMA
    descriptors over the KV caches) whose grid and store masks read the
    prompt length from dyn_dims at launch, so a single compiled kernel serves
    every prompt length."""

    def __init__(self, meta: dict, max_seq: int):
        import importlib.util
        import tempfile

        import triton

        src = open(os.path.join(HERE, meta["file"])).read()
        got = hashlib.sha256(src.encode()).hexdigest()
        assert got == meta["sha256"], f"{meta['file']}: sha mismatch — kernels/ edited?"
        d = tempfile.mkdtemp(prefix="l31_tfmha_")
        _TMP_MODULES.append(d)
        path = os.path.join(d, "l31_tfmha_module.py")
        with open(path, "w") as f:
            f.write("import triton\nimport triton.language as tl\n\n" + src)
        spec = importlib.util.spec_from_file_location("l31_tfmha_module", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._kernel = getattr(mod, meta["entry"])
        cx = dict(meta["params"]["constexprs"])
        cx["MAXSEQ"] = max_seq   # the cache row stride, re-baked like the CUDA kernels
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


# ---------------------------------------------------------------------------
# weights & tables
# ---------------------------------------------------------------------------

def llama3_inv_freq() -> torch.Tensor:
    """transformers' _compute_llama3_parameters, in f64 — hoid's
    exact table math (attention_scaling is 1.0 for this rope type)."""
    j = torch.arange(HALF, dtype=torch.float64)
    inv = torch.pow(torch.tensor(ROPE_THETA, dtype=torch.float64), -2.0 * j / HD)
    wavelen = 2.0 * math.pi / inv
    low_wavelen = ROPE_ORIGINAL_MAX_POS / ROPE_LOW_FREQ_FACTOR
    high_wavelen = ROPE_ORIGINAL_MAX_POS / ROPE_HIGH_FREQ_FACTOR
    smooth = (ROPE_ORIGINAL_MAX_POS / wavelen - ROPE_LOW_FREQ_FACTOR) / (
        ROPE_HIGH_FREQ_FACTOR - ROPE_LOW_FREQ_FACTOR
    )
    mid = (1.0 - smooth) * inv / ROPE_FACTOR + smooth * inv
    out = torch.where(wavelen > low_wavelen, inv / ROPE_FACTOR,
                      torch.where(wavelen < high_wavelen, inv, mid))
    return out


def rope_tables(max_seq: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 [max_seq, HALF] cos/sin from f64 angles — hoid's rope_tables."""
    inv_freq = llama3_inv_freq()
    pos = torch.arange(max_seq, dtype=torch.float64)
    ang = pos[:, None] * inv_freq[None, :]
    return (ang.cos().to(torch.float32).to(device),
            ang.sin().to(torch.float32).to(device))


def load_weights(weights_dir: str, device="cuda") -> dict[str, torch.Tensor]:
    """HF safetensors -> {name: bf16 tensor on device}. No transformers."""
    from safetensors import safe_open

    files = sorted(
        f for f in os.listdir(weights_dir) if f.endswith(".safetensors")
    )
    assert files, f"no safetensors under {weights_dir}"
    w = {}
    for fname in files:
        with safe_open(os.path.join(weights_dir, fname), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("model.") or k == "lm_head.weight":
                    w[k] = f.get_tensor(k).to(device=device, dtype=torch.bfloat16)
    assert "model.embed_tokens.weight" in w
    assert "lm_head.weight" in w, "llama-3.1-8b has untied embeddings"
    return w


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------

class Llama31Optimized:
    """Holds weights, KV caches, static buffers, and the two phases:
    prefill (hoid 8k kernels or plain torch, both writing the hoid cache
    layout) and the hoid-kernel decode step (CUDA-graph captured)."""

    def __init__(self, weights: dict[str, torch.Tensor], max_seq: int,
                 device="cuda", verbose: bool = False):
        self.device = device
        self.max_seq = max_seq
        self.residues = pick_residues(max_seq)
        self.K = build_kernels(max_seq, verbose=verbose)

        g = lambda n: weights[n]
        self.embed = g("model.embed_tokens.weight")
        self.final_norm_w = g("model.norm.weight")
        self.lm_head_t = g("lm_head.weight").t()   # untied

        L = lambda i, s: g(f"model.layers.{i}.{s}.weight")
        self.ln1 = [L(i, "input_layernorm") for i in range(LAYERS)]
        self.ln2 = [L(i, "post_attention_layernorm") for i in range(LAYERS)]
        self.qw_t = [L(i, "self_attn.q_proj").t() for i in range(LAYERS)]
        self.kw_t = [L(i, "self_attn.k_proj").t() for i in range(LAYERS)]
        self.vw_t = [L(i, "self_attn.v_proj").t() for i in range(LAYERS)]
        self.ow_t = [L(i, "self_attn.o_proj").t() for i in range(LAYERS)]
        self.gw_t = [L(i, "mlp.gate_proj").t() for i in range(LAYERS)]
        self.uw_t = [L(i, "mlp.up_proj").t() for i in range(LAYERS)]
        self.dw_t = [L(i, "mlp.down_proj").t() for i in range(LAYERS)]

        self.cos, self.sin = rope_tables(max_seq, device)

        bf = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=device)
        # per-layer KV caches, the layout cache_write/partial bake:
        # row = (kv_head * MAX_SEQ + pos) * HD
        self.kcache = [bf(NKV, max_seq, HD) for _ in range(LAYERS)]
        self.vcache = [bf(NKV, max_seq, HD) for _ in range(LAYERS)]

        # dyn_dims: the kernels' fixed 26-int device buffer ('a'..'z')
        self.dyn = torch.zeros(26, dtype=torch.int32, device=device)
        self.dyn[18] = 1  # 's' = 1 token per decode step
        self.tok = torch.zeros(1, dtype=torch.int32, device=device)

        # decode-step scratch (seq = 1), all preallocated for capture
        self.x = bf(1, HIDDEN)          # residual stream (ping)
        self.x2 = bf(1, HIDDEN)         # residual stream (pong)
        self.h = bf(1, HIDDEN)          # normed
        self.q = bf(1, Q_HID)
        self.k = bf(1, KV_HID)
        self.v = bf(1, KV_HID)
        self.qr = bf(1, Q_HID)          # roped q
        self.kr = bf(1, KV_HID)         # roped k
        self.partials = torch.zeros(1, NQ, self.residues, HD + 2,
                                    dtype=torch.float32, device=device)
        self.attn = bf(1, NQ, HD)
        self.o = bf(1, HIDDEN)
        self.gate = bf(1, INTER)
        self.up = bf(1, INTER)
        self.act = bf(1, INTER)
        self.down = bf(1, HIDDEN)
        self.final = bf(1, HIDDEN)
        self.logits = bf(1, VOCAB)

        self.tfmha = TritonFMHA(ELASTIC_KERNELS["fmha_prefill_elastic"],
                                max_seq)
        self._ebuf: dict[int, dict[str, torch.Tensor]] = {}

        self.graph = None

    # ---- the decode step: the winning graph's dataflow, node for node ----
    def step_static(self):
        K, dyn = self.K, self.dyn
        K["embedding_k"](self.x, self.embed, self.tok, dyn, (HIDDEN + 255) // 256)
        resid, resid_next = self.x, self.x2
        for i in range(LAYERS):
            K["rmsnorm_prefetch16_k"](self.h, resid, self.ln1[i], dyn, 1)
            torch.matmul(self.h, self.qw_t[i], out=self.q)
            torch.matmul(self.h, self.kw_t[i], out=self.k)
            torch.matmul(self.h, self.vw_t[i], out=self.v)
            K["apply_rope_q_k"](self.qr, self.q, self.cos, self.sin, dyn,
                                (Q_HID + 255) // 256)
            K["apply_rope_kv_k"](self.kr, self.k, self.cos, self.sin, dyn,
                                 (KV_HID + 255) // 256)
            K["cache_write_k"](self.kcache[i], self.kcache[i], self.kr, dyn,
                               (KV_HID + 255) // 256)
            K["cache_write_k"](self.vcache[i], self.vcache[i], self.v, dyn,
                               (KV_HID + 255) // 256)
            K["dense_cached_attention_residue_partial_k"](
                self.partials, self.qr, self.kcache[i], self.vcache[i],
                dyn, NKV * self.residues)
            K["dense_cached_attention_residue_combine_k"](
                self.attn, self.partials, dyn, NQ)
            torch.matmul(self.attn.view(1, Q_HID), self.ow_t[i], out=self.o)
            K["add_bf16_vec8_k"](resid_next, resid, self.o, dyn, 2)
            resid, resid_next = resid_next, resid
            K["rmsnorm_prefetch16_k"](self.h, resid, self.ln2[i], dyn, 1)
            torch.matmul(self.h, self.gw_t[i], out=self.gate)
            torch.matmul(self.h, self.uw_t[i], out=self.up)
            K["silu_mul_vec8_fast_k"](self.act, self.gate, self.up, dyn, 7)
            torch.matmul(self.act, self.dw_t[i], out=self.down)
            K["add_bf16_vec8_k"](resid_next, resid, self.down, dyn, 2)
            resid, resid_next = resid_next, resid
        K["rmsnorm_prefetch16_k"](self.final, resid, self.final_norm_w, dyn, 1)
        torch.matmul(self.final, self.lm_head_t, out=self.logits)
        return self.logits

    # ---- capture & replay -------------------------------------------------
    def capture(self, warmup: int = 3):
        """Warmup at pos 0 (its garbage cache rows are overwritten by every
        prefill), then capture. dyn_dims[14] is read inside the kernels, so
        the one graph replays at every position."""
        self.dyn[14] = 0
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(warmup):
                self.step_static()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.step_static()

    def step(self, tok: int, pos: int) -> torch.Tensor:
        self.tok.fill_(tok)
        self.dyn[14] = pos
        self.graph.replay()
        return self.logits

    # ---- prefill: the elastic graph, node for node ------------------------
    def _elastic_buffers(self, P: int) -> dict[str, torch.Tensor]:
        if P not in self._ebuf:
            bf = lambda *s: torch.zeros(*s, dtype=torch.bfloat16,
                                        device=self.device)
            self._ebuf[P] = {
                "ids": torch.zeros(P, dtype=torch.int32, device=self.device),
                "x": bf(P, HIDDEN), "b": bf(P, HIDDEN), "h": bf(P, HIDDEN),
                "q": bf(P, Q_HID), "k": bf(P, KV_HID), "v": bf(P, KV_HID),
                "qr": bf(P, Q_HID), "kr": bf(P, KV_HID),
                "att": bf(P, Q_HID), "o": bf(P, HIDDEN),
                "gate": bf(P, INTER), "up": bf(P, INTER), "act": bf(P, INTER),
                "down": bf(P, HIDDEN),
            }
        return self._ebuf[P]

    @torch.no_grad()
    def prefill_elastic(self, ids: torch.Tensor) -> torch.Tensor:
        """The elastic prefill graph's rewrite: the same prefetch RMSNorm /
        cuBLAS GEMM / rope / cache-write / vec8 stack as prefill_hoid, with
        the static pad -> CuTe FMHA -> crop chain replaced by the one
        shape-generic triton FA2 whose launch reads the prompt length from
        dyn_dims. Any prompt length up to the cache budget."""
        K, dyn = self.K, self.dyn
        P = ids.numel()
        assert P + 2 <= self.max_seq, "prompt too long for this cache"
        e = self._elastic_buffers(P)
        dyn[18] = P
        dyn[14] = 0
        e["ids"].copy_(ids.to(torch.int32))
        K["embedding_k"](e["x"], self.embed, e["ids"], dyn,
                         (P * HIDDEN + 255) // 256)
        a, b = e["x"], e["b"]
        for i in range(LAYERS):
            K["rmsnorm_prefetch16_k"](e["h"], a, self.ln1[i], dyn, P)
            torch.matmul(e["h"], self.qw_t[i], out=e["q"])
            torch.matmul(e["h"], self.kw_t[i], out=e["k"])
            torch.matmul(e["h"], self.vw_t[i], out=e["v"])
            K["apply_rope_q_k"](e["qr"], e["q"], self.cos, self.sin, dyn,
                                (P * Q_HID + 255) // 256)
            K["apply_rope_kv_k"](e["kr"], e["k"], self.cos, self.sin, dyn,
                                 (P * KV_HID + 255) // 256)
            K["cache_write_k"](self.kcache[i], self.kcache[i], e["kr"], dyn,
                               (P * KV_HID + 255) // 256)
            K["cache_write_k"](self.vcache[i], self.vcache[i], e["v"], dyn,
                               (P * KV_HID + 255) // 256)
            self.tfmha(e["att"], e["qr"], self.kcache[i], self.vcache[i],
                       dyn, P)
            torch.matmul(e["att"], self.ow_t[i], out=e["o"])
            K["add_bf16_vec8_k"](b, a, e["o"], dyn, (P * HIDDEN + 2047) // 2048)
            K["rmsnorm_prefetch16_k"](e["h"], b, self.ln2[i], dyn, P)
            torch.matmul(e["h"], self.gw_t[i], out=e["gate"])
            torch.matmul(e["h"], self.uw_t[i], out=e["up"])
            K["silu_mul_vec8_fast_k"](e["act"], e["gate"], e["up"], dyn,
                                      (P * INTER + 2047) // 2048)
            torch.matmul(e["act"], self.dw_t[i], out=e["down"])
            K["add_bf16_vec8_k"](a, b, e["down"], dyn,
                                 (P * HIDDEN + 2047) // 2048)
        K["rmsnorm_prefetch16_k"](e["h"], a, self.final_norm_w, dyn, P)
        dyn[18] = 1  # restore the decode invariant before any replay
        return e["h"][P - 1:P] @ self.lm_head_t

    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        return self.prefill_elastic(ids)

    # ---- generation -------------------------------------------------------
    @torch.no_grad()
    def generate(self, ids: torch.Tensor, n_new: int) -> list[int]:
        P = ids.numel()
        first = int(self.prefill(ids).argmax())
        out = [first]
        tok = first
        for j in range(n_new - 1):
            logits = self.step(tok, P + j)
            tok = int(logits.argmax())
            out.append(tok)
        return out
