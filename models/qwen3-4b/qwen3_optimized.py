"""Qwen3-4B-Instruct-2507 on the hoid-winning kernels — standalone PyTorch.

No hoid runtime anywhere: the kernels under kernels/ are the byte-identical
sources of the optimized hoid decode stack, compiled with
torch.utils.cpp_extension.load_inline and launched with the exact launch
geometry the manifest records. The builtin ops the graph uses besides cuBLAS
matmuls (prepare_qkv / silu_mul / embedding) are hoid's own generated sources,
materialized for this model's constants in qwen3_native_gen.py.

Design (the whisper-decoder port pattern):
- hoid ABI preserved: every kernel takes `(out..., in..., const int* dyn_dims)`.
  dyn_dims is a real int32[26] device buffer; seq 's' lives at index 18 and
  offset 'o' at index 14 — the kernels' fixed dyn_dims layout.
  The decode step reads the position from dyn_dims inside the kernels, so ONE
  captured CUDA graph serves every decode position.
- Static buffers rule: a tensor allocated inside a CUDA-graph capture and
  freed afterwards is NOT protected from later allocations. The whole decode
  working set (KV caches included) is allocated up front; every kernel writes
  into caller-provided buffers; matmuls use `out=`.
- MAX_SEQ is a compile-time bake-in in two kernels (the KV-cache row stride).
  The graph shipped it at 1040; the engine re-bakes it for the requested cache
  size and asserts the literal it replaces occurs exactly once.
- RESIDUES (the decode attention's context split factor) is the same kind of
  bake-in and is re-baked the same way. The graph shipped 16, which fills the
  device at ~1k context but leaves it idle at 8k, since the block count does
  not depend on depth. See `pick_residues`.
- Prefill is one shape-generic path at every prompt length: the 8k graph's
  prefetch RMSNorm / vec8 elementwise kernels and the decode graph's
  seq-elastic prepare_qkv at grid = S, with attention on the elastic triton
  FA2 (`kernels/fmha_prefill_elastic.triton.py` — grid and masks read the
  prompt length from dyn_dims at launch; MAXSEQ re-baked per engine build).
  It writes the cache layout the decode kernels read:
  [n_kv_heads, MAX_SEQ, head_dim] per layer. Unlike the hoid prefill graph
  (pinned to emit [s, vocab] logits), the port only projects the last
  position to logits.
"""

from __future__ import annotations

import hashlib
import os

import torch
from torch.utils.cpp_extension import load_inline

import qwen3_native_gen as NG
from qwen3_decode_kernels_gen import KERNELS as DECODE_KERNELS

try:
    from qwen3_prefill_kernels_gen import KERNELS as PREFILL_KERNELS
except ImportError:  # decode-only checkout
    PREFILL_KERNELS = {}

try:
    from qwen3_elastic_kernels_gen import KERNELS as ELASTIC_KERNELS
except ImportError:  # decode-only checkout
    ELASTIC_KERNELS = {}

HERE = os.path.dirname(os.path.abspath(__file__))

HIDDEN = 2560
LAYERS = 36
NQ = 32
NKV = 8
HD = 128
HALF = HD // 2
Q_HID = NQ * HD          # 4096
KV_HID = NKV * HD        # 1024
INTER = 9728
VOCAB = 151936
EPS = 1e-6
ROPE_THETA = 5e6
SCALE = HD ** -0.5

# ---------------------------------------------------------------------------
# kernel compilation: one extension module per kernel (their file-scope
# constexprs/macros collide, and per-module keeps every source byte-identical)
# ---------------------------------------------------------------------------

# entry -> (wrapper arg spec, block.x). Arg spec: (name, kind) in ABI order,
# kinds: bf16 out/in, f32 out/in, i32 in. Grids are computed by the caller.
_WRAPPER_SPECS = {
    "rmsnorm_register_k": (["o:bf16", "i:bf16", "i:bf16"], 256),
    "fused_residual_rmsnorm_multi_k": (["o:bf16", "o:bf16", "i:bf16", "i:bf16", "i:bf16"], 256),
    "fused_final_residual_rmsnorm_k": (["o:bf16", "i:bf16", "i:bf16", "i:bf16"], 256),
    "dense_cached_attention_residue_pipeline_k": (["o:f32", "i:bf16", "i:bf16", "i:bf16"], 192),
    "dense_cached_attention_residue_combine_k": (["o:bf16", "i:f32"], 32),
    "prepare_qkv_k": (
        ["o:bf16", "i:bf16", "i:bf16", "i:bf16", "i:bf16", "i:bf16",
         "o:bf16", "o:bf16", "i:f32", "i:f32"],  # k_cache/v_cache are written
        32,
    ),
    "silu_mul_k": (["o:bf16", "i:bf16", "i:bf16"], 256),
    "embedding_k": (["o:bf16", "i:bf16", "i:i32"], 256),
    # prefill kernels from the optimized 8k graph (all standard flat ABI)
    "rmsnorm_prefetch10_k": (["o:bf16", "i:bf16", "i:bf16"], 256),
    "add_bf16_vec8_k": (["o:bf16", "i:bf16", "i:bf16"], 256),
    "silu_mul_vec8_fast_k": (["o:bf16", "i:bf16", "i:bf16"], 256),
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
    """entry -> verbatim source, provenance-checked against the manifest."""
    srcs = {}
    for manifest in (DECODE_KERNELS, PREFILL_KERNELS):
        for entry, meta in manifest.items():
            if meta["format"] != "cuda":
                continue  # the manifest's cutedsl FMHA is not part of the
                # elastic prefill; attention runs the triton kernel
            if entry == "prepare_qkv_w4_direct_k":
                continue  # static-8k Q-layout variant; the elastic prefill
                # uses the seq-generic prepare_qkv_k
            path = os.path.join(HERE, meta["file"])
            src = open(path).read()
            got = hashlib.sha256(src.encode()).hexdigest()
            assert got == meta["sha256"], f"{meta['file']}: sha mismatch — kernels/ edited?"
            srcs[entry] = src
    srcs["prepare_qkv_k"] = NG.PREPARE_QKV
    srcs["silu_mul_k"] = NG.SILU_MUL
    srcs["embedding_k"] = NG.EMBEDDING
    return srcs


def pick_residues(max_seq: int) -> int:
    """The decode attention's context split factor.

    The pipeline kernel launches `NKV * RESIDUES` blocks whatever the depth,
    and each block strides the cache by RESIDUES — so this one constant
    decides how much of the device the attention fills, and a deeper cache
    buys no extra parallelism, only more serial work per block. A per-depth
    sweep (rebuild with each value, time a decode step) shows R=16's 128
    blocks saturate the device up to ~1.8k keys and lose past it, while R=64
    never wins — the partials buffer and the combine's merge both grow with R
    and outrun the extra parallelism.

    Chosen once from max_seq (one captured graph serves every position);
    max_seq is an upper bound on the depth a run reaches, so the threshold
    sits above the measured crossover rather than on it.
    """
    return 16 if max_seq < 3072 else 32


def _rebake_constants(entry: str, src: str, max_seq: int) -> str:
    """Re-bake the compile-time constants the shipped decode graph froze at
    its own regime: the KV-cache row stride (MAX_SEQ=1040) and the attention
    split factor (RESIDUES=16, derived at ~1k context). The 8k prefill kernels
    bake 9216 and are kept verbatim — the engine asserts max_seq == 9216
    whenever they are in play.
    """
    if entry == "dense_cached_attention_residue_pipeline_k":
        old = "constexpr int MAX_SEQ = 1040;"
        assert src.count(old) == 1, "pipeline kernel MAX_SEQ bake-in moved"
        src = src.replace(old, f"constexpr int MAX_SEQ = {max_seq};")
        old = "constexpr int RESIDUES = 16;"
        assert src.count(old) == 1, "pipeline kernel RESIDUES bake-in moved"
        return src.replace(old, f"constexpr int RESIDUES = {pick_residues(max_seq)};")
    if entry == "dense_cached_attention_residue_combine_k":
        # WARPS is the combine's name for the same split; they must agree.
        r = pick_residues(max_seq)
        if r == 16:
            return src  # shipped split — keep the source byte-identical
        old = "constexpr int WARPS = 16;"
        assert src.count(old) == 1, "combine kernel WARPS bake-in moved"
        src = src.replace(old, f"constexpr int WARPS = {r};")
        # full unroll over the records is right at 16, oversized past it —
        # codegen only, the merge order and arithmetic are unchanged
        return src.replace("#pragma unroll\n  for (int w",
                           "#pragma unroll 8\n  for (int w")
    if entry == "prepare_qkv_k":
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
                name=f"q4b_{key}",
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
    """kernels/fmha_prefill_elastic.triton.py — one shape-generic FA2
    (two-phase causal loop, device-side TMA descriptors over the KV caches)
    whose grid and store masks read the prompt length from dyn_dims at
    launch, so a single compiled kernel serves every prompt length."""

    def __init__(self, meta: dict, max_seq: int):
        import importlib.util
        import tempfile

        import triton

        src = open(os.path.join(HERE, meta["file"])).read()
        got = hashlib.sha256(src.encode()).hexdigest()
        assert got == meta["sha256"], f"{meta['file']}: sha mismatch — kernels/ edited?"
        d = tempfile.mkdtemp(prefix="q3_tfmha_")
        _TMP_MODULES.append(d)
        path = os.path.join(d, "q3_tfmha_module.py")
        with open(path, "w") as f:
            f.write("import triton\nimport triton.language as tl\n\n" + src)
        spec = importlib.util.spec_from_file_location("q3_tfmha_module", path)
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

def rope_tables(max_seq: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """hoid's rope sin/cos tables: fp32 [max_seq, HALF];
    cos computed as sin(x + pi/2), inv_freq[j] = theta^(-2j/HD)."""
    j = torch.arange(HALF, dtype=torch.float64)
    inv_freq = torch.pow(torch.tensor(ROPE_THETA, dtype=torch.float64), -2.0 * j / HD)
    pos = torch.arange(max_seq, dtype=torch.float64)
    ang = (pos[:, None] * inv_freq[None, :]).to(torch.float32)
    import math
    return torch.sin(ang + math.pi / 2).to(device), torch.sin(ang).to(device)


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
    return w


# ---------------------------------------------------------------------------
# the decode engine
# ---------------------------------------------------------------------------

class Qwen3Optimized:
    """Holds weights, KV caches, static buffers, and the two phases:
    torch-native prefill (writes the hoid cache layout) and the hoid-kernel
    decode step (CUDA-graph captured)."""

    def __init__(self, weights: dict[str, torch.Tensor], max_seq: int,
                 device="cuda", verbose: bool = False):
        assert PREFILL_KERNELS, "qwen3_prefill_kernels_gen.py missing"
        assert ELASTIC_KERNELS, "qwen3_elastic_kernels_gen.py missing"
        self.device = device
        self.max_seq = max_seq
        self.residues = pick_residues(max_seq)
        self.K = build_kernels(max_seq, verbose=verbose)

        g = lambda n: weights[n]
        self.embed = g("model.embed_tokens.weight")
        self.final_norm_w = g("model.norm.weight")
        # tied lm_head — matmul against the transposed view, exactly the
        # tied-embedding matmul the graph's `logits` node performs
        self.lm_head_t = self.embed.t()

        L = lambda i, s: g(f"model.layers.{i}.{s}.weight")
        self.ln1 = [L(i, "input_layernorm") for i in range(LAYERS)]
        self.ln2 = [L(i, "post_attention_layernorm") for i in range(LAYERS)]
        self.qw_t = [L(i, "self_attn.q_proj").t() for i in range(LAYERS)]
        self.kw_t = [L(i, "self_attn.k_proj").t() for i in range(LAYERS)]
        self.vw_t = [L(i, "self_attn.v_proj").t() for i in range(LAYERS)]
        self.ow_t = [L(i, "self_attn.o_proj").t() for i in range(LAYERS)]
        self.qnw = [L(i, "self_attn.q_norm") for i in range(LAYERS)]
        self.knw = [L(i, "self_attn.k_norm") for i in range(LAYERS)]
        self.gw_t = [L(i, "mlp.gate_proj").t() for i in range(LAYERS)]
        self.uw_t = [L(i, "mlp.up_proj").t() for i in range(LAYERS)]
        self.dw_t = [L(i, "mlp.down_proj").t() for i in range(LAYERS)]

        self.cos, self.sin = rope_tables(max_seq, device)

        bf = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=device)
        # per-layer KV caches, the layout prepare_qkv/pipeline bake:
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
        self.out_q = bf(1, Q_HID)
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
        K["rmsnorm_register_k"](self.h, self.x, self.ln1[0], dyn, 1)
        for i in range(LAYERS):
            torch.matmul(self.h, self.qw_t[i], out=self.q)
            torch.matmul(self.h, self.kw_t[i], out=self.k)
            torch.matmul(self.h, self.vw_t[i], out=self.v)
            K["prepare_qkv_k"](self.out_q, self.q, self.k, self.v,
                               self.qnw[i], self.knw[i],
                               self.kcache[i], self.vcache[i],
                               self.cos, self.sin, dyn, NQ + 2 * NKV)
            K["dense_cached_attention_residue_pipeline_k"](
                self.partials, self.out_q, self.kcache[i], self.vcache[i],
                dyn, NKV * self.residues)
            K["dense_cached_attention_residue_combine_k"](
                self.attn, self.partials, dyn, NQ)
            torch.matmul(self.attn.view(1, Q_HID), self.ow_t[i], out=self.o)
            K["fused_residual_rmsnorm_multi_k"](
                resid_next, self.h, resid, self.o, self.ln2[i], dyn, 1)
            resid, resid_next = resid_next, resid
            torch.matmul(self.h, self.gw_t[i], out=self.gate)
            torch.matmul(self.h, self.uw_t[i], out=self.up)
            K["silu_mul_k"](self.act, self.gate, self.up, dyn,
                            (INTER + 255) // 256)
            torch.matmul(self.act, self.dw_t[i], out=self.down)
            if i + 1 < LAYERS:
                K["fused_residual_rmsnorm_multi_k"](
                    resid_next, self.h, resid, self.down, self.ln1[i + 1], dyn, 1)
                resid, resid_next = resid_next, resid
            else:
                K["fused_final_residual_rmsnorm_k"](
                    self.final, resid, self.down, self.final_norm_w, dyn, 1)
        torch.matmul(self.final, self.lm_head_t, out=self.logits)
        return self.logits

    # ---- capture & replay -------------------------------------------------
    def capture(self, warmup: int = 3):
        """Warmup at pos 0 (its garbage cache row is overwritten by every
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

    # ---- prefill: elastic — the same stack at any prompt length -----------
    def _elastic_buffers(self, P: int) -> dict[str, torch.Tensor]:
        if P not in self._ebuf:
            bf = lambda *s: torch.zeros(*s, dtype=torch.bfloat16,
                                        device=self.device)
            self._ebuf[P] = {
                "ids": torch.zeros(P, dtype=torch.int32, device=self.device),
                "x": bf(P, HIDDEN), "b": bf(P, HIDDEN), "h": bf(P, HIDDEN),
                "q": bf(P, Q_HID), "k": bf(P, KV_HID), "v": bf(P, KV_HID),
                "qr": bf(P, Q_HID), "att": bf(P, Q_HID), "o": bf(P, HIDDEN),
                "gate": bf(P, INTER), "up": bf(P, INTER), "act": bf(P, INTER),
                "down": bf(P, HIDDEN),
            }
        return self._ebuf[P]

    @torch.no_grad()
    def prefill_elastic(self, ids: torch.Tensor) -> torch.Tensor:
        """The 8k prefill stack with its two shape-frozen pieces swapped for
        shape-generic ones: prepare_qkv (the decode graph's builtin — per-head
        q/k RMSNorm + RoPE + cache write, one warp per token/head) in place of
        the static-layout w4_direct variant, and the elastic triton FA2 in
        place of the static-SQ CuTe FMHA. Any prompt length up to the cache
        budget."""
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
            K["rmsnorm_prefetch10_k"](e["h"], a, self.ln1[i], dyn, P)
            torch.matmul(e["h"], self.qw_t[i], out=e["q"])
            torch.matmul(e["h"], self.kw_t[i], out=e["k"])
            torch.matmul(e["h"], self.vw_t[i], out=e["v"])
            K["prepare_qkv_k"](e["qr"], e["q"], e["k"], e["v"],
                               self.qnw[i], self.knw[i],
                               self.kcache[i], self.vcache[i],
                               self.cos, self.sin, dyn, P * 48)
            self.tfmha(e["att"], e["qr"], self.kcache[i], self.vcache[i],
                       dyn, P)
            torch.matmul(e["att"], self.ow_t[i], out=e["o"])
            K["add_bf16_vec8_k"](b, a, e["o"], dyn, (P * HIDDEN + 2047) // 2048)
            K["rmsnorm_prefetch10_k"](e["h"], b, self.ln2[i], dyn, P)
            torch.matmul(e["h"], self.gw_t[i], out=e["gate"])
            torch.matmul(e["h"], self.uw_t[i], out=e["up"])
            K["silu_mul_vec8_fast_k"](e["act"], e["gate"], e["up"], dyn,
                                      (P * INTER + 2047) // 2048)
            torch.matmul(e["act"], self.dw_t[i], out=e["down"])
            K["add_bf16_vec8_k"](a, b, e["down"], dyn,
                                 (P * HIDDEN + 2047) // 2048)
        K["rmsnorm_prefetch10_k"](e["h"], a, self.final_norm_w, dyn, P)
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
