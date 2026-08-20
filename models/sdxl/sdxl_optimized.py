#!/usr/bin/env python
"""SDXL's UNet in PyTorch, running the winning kernel stack.

The stock `UNet2DConditionModel` is kept whole — every convolution, sampler and
skip connection is the diffusers one — and the three places the tuned kernels
beat what a compiler can reach are swapped out:

  `geglu`       ONE persistent Blackwell dual-GEMM: the value and gate halves
                of the feed-forward projection share an A-operand TMA stream
                and two TMEM accumulators, and the epilogue adds both bias
                halves, evaluates a poly8 GELU on the gate and multiplies —
                so the 10240-wide intermediate is never written to HBM.
                70 sites.
  `cross-kv`    every cross-attention K and V projection at one hidden width
                is a slice of ONE GEMM against a pre-packed weight. The
                projections all read the same encoder states, so 120 GEMMs of
                M=154 become one; a compiler cannot see across the blocks to
                find this. The packed buffer is also exactly the layout the
                cross-attention kernel wants.
  `attention`   Triton flash-attention reading K and V straight out of that
                packed buffer at a column offset (cross), and CUTLASS
                Blackwell FMHA or flash-attn 4 (self) — chosen by measurement,
                see `SELF_ATTN`.

Optionally (`--fused-ln`) the attention output projection, its bias, the
residual add and the next LayerNorm collapse into one kernel over 199 sites.

The CUDA kernels are nvcc-JIT'd at import, the Triton kernel is compiled by
Triton, and the CuTe DSL kernels by `cute.compile` against the static shapes.

Everything here is a torch custom op with a fake implementation, so the whole
UNet still traces under `torch.compile` and captures into CUDA graphs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
import time
import urllib.request

import torch
import torch.nn.functional as F
from torch import nn

from sdxl_kernels_gen import KERNELS

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, "kernels")


# ---------------------------------------------------------------------------
# The pinned CUTLASS CuTe DSL example files the dual-GEMM and FMHA kernels
# subclass. The `nvidia-cutlass-dsl` wheel does not ship the example tree, so
# fetch exactly the files needed at the pinned tag, verify their sha256, and
# lay them out under the relative path the kernel sources expect — they locate
# the tree by looking for a sys.path entry ending in examples/python/CuTeDSL,
# which is how the CUTLASS examples import each other. Cached in
# .cutlass_examples/ next to this file; later runs are a hash check.
# ---------------------------------------------------------------------------
CUTLASS_TAG = "v4.6.1"
_RAW = f"https://raw.githubusercontent.com/NVIDIA/cutlass/{CUTLASS_TAG}/examples/python/CuTeDSL"

# path relative to examples/python/CuTeDSL -> sha256 at CUTLASS_TAG
CUTLASS_FILES = (
    "cute/blackwell/kernel/dense_gemm/dense_gemm_persistent.py",
    "cute/blackwell/kernel/attention/fmha/fmha.py",
    "cute/blackwell/kernel/attention/mixed_input_fmha/mixed_input_fmha_prefill_d512.py",
    "cute/blackwell/kernel/attention/mixed_input_fmha/prefill_helpers.py",
    "helpers/__init__.py",
    "helpers/fmha_helpers.py",
)

CUTLASS_ROOT = os.path.join(HERE, ".cutlass_examples", "examples", "python", "CuTeDSL")
_HASHES_PATH = os.path.join(HERE, "cutlass_pins.txt")


def _sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _load_pins() -> dict:
    pins = {}
    if os.path.exists(_HASHES_PATH):
        for line in open(_HASHES_PATH):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sha, rel = line.split(None, 1)
            pins[rel] = sha
    return pins


def _cutlass_ensure(verbose: bool = True) -> str:
    """Download-if-missing + hash-check, and return the CuTeDSL example root.

    A local CUTLASS checkout can be pointed at with CUTLASS_EXAMPLES_DIR (it
    must be the `examples/python/CuTeDSL` directory) — then nothing is fetched.
    """
    local = os.environ.get("CUTLASS_EXAMPLES_DIR")
    if local:
        return local
    pins = _load_pins()
    for rel in CUTLASS_FILES:
        dst = os.path.join(CUTLASS_ROOT, rel)
        want = pins.get(rel)
        if os.path.exists(dst) and (want is None or _sha256_file(dst) == want):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        url = f"{_RAW}/{rel}"
        if verbose:
            print(f"[cutlass] fetching {rel} @ {CUTLASS_TAG}")
        with urllib.request.urlopen(url) as r:
            blob = r.read()
        got = hashlib.sha256(blob).hexdigest()
        if want is not None and got != want:
            raise RuntimeError(
                f"cutlass example {rel} at {CUTLASS_TAG} hashes {got}, expected {want}. "
                "The pin moved or the download is corrupt; do not run with unverified sources.")
        with open(dst, "wb") as f:
            f.write(blob)
    return CUTLASS_ROOT


def cutlass_install(verbose: bool = True) -> str:
    """`ensure` + put the tree on sys.path the way the CUTLASS examples do."""
    root = _cutlass_ensure(verbose)
    for p in (root,
              os.path.join(root, "cute/blackwell/kernel/dense_gemm"),
              os.path.join(root, "cute/blackwell/kernel/attention/fmha"),
              os.path.join(root, "cute/blackwell/kernel/attention/mixed_input_fmha")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


# ---------------------------------------------------------------------------
# Loading authored kernel sources
#
# Both Triton and the CuTe DSL read a decorated function's source back with
# `inspect.getsourcelines`, so neither can be exec'd into a namespace — each
# source has to become a real file on disk. The preamble is the imports every
# source assumes plus the node's compile params as module-level constants.
# ---------------------------------------------------------------------------
_TMP_MODULES = []  # keeps the generated files alive for the life of the process
_EMPTY_DYN: dict = {}   # the graph ABI's trailing dynamic-dims argument, unused here


def _src(key: str) -> str:
    """The kernel source, hash-checked against the manifest."""
    meta = KERNELS[key]
    path = os.path.join(KDIR, meta["file"])
    with open(path) as f:
        src = f.read()
    got = hashlib.sha256(src.encode()).hexdigest()
    if got != meta["sha256"]:
        raise RuntimeError(
            f"{meta['file']} hashes {got}, manifest says {meta['sha256']} — the kernel "
            "source and the manifest disagree.")
    return src


def _load_kernel_module(name: str, key: str, preamble: list[str]):
    tmp = tempfile.mkdtemp(prefix="sdxl_kernel_")
    _TMP_MODULES.append(tmp)
    path = os.path.join(tmp, f"{name}.py")
    with open(path, "w") as f:
        f.write("\n".join(preamble) + "\n\n" + _src(key))
    sys.path.insert(0, tmp)
    spec = importlib.util.spec_from_file_location(f"sdxl_kernel_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cu_stream():
    import cuda.bindings.driver as cuda_driver

    return cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)


class CuteRunner:
    """One compiled CuTe DSL entry, called with torch tensors.

    Descriptors are rebuilt per call rather than cached by data pointer.
    Caching them looks tempting — building one is a few microseconds — but the
    cache holds a live reference to every output buffer it has seen, so the
    caching allocator can never reuse one and every call pays a real
    multi-megabyte cudaMalloc, which costs several times the kernel itself.
    """

    def __init__(self, key: str, params: dict, arg_shapes, verbose: bool = True):
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack

        self._from_dlpack = from_dlpack
        meta = KERNELS[key]
        tag = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
        preamble = ["import cutlass", "import cutlass.cute as cute",
                    "import cuda.bindings.driver as cuda"]
        preamble += [f"{k} = {v!r}" for k, v in params.items()]
        module = _load_kernel_module(f"{key}_{abs(hash(tag)) % (10 ** 8)}", key, preamble)
        entry = getattr(module, meta["entry"])
        t0 = time.perf_counter()
        fakes = [self._fake(shape) for shape in arg_shapes]
        self.compiled = cute.compile(entry, *fakes, _cu_stream(),
                                     options=meta.get("options") or "")
        if verbose:
            print(f"[cutedsl] {key} {tag} compiled in {time.perf_counter() - t0:.1f}s")

    def _fake(self, shape):
        import cutlass
        from cutlass.cute.runtime import make_fake_compact_tensor

        order = tuple(range(len(shape) - 1, -1, -1))  # row-major
        return make_fake_compact_tensor(cutlass.BFloat16, tuple(shape), stride_order=order)

    def tensor(self, t: torch.Tensor):
        t = t.detach()  # module parameters carry requires_grad; dlpack refuses those
        view = t.unsqueeze(-1) if t.dim() == 2 else t
        return self._from_dlpack(view, assumed_align=16)

    def __call__(self, *tensors: torch.Tensor):
        self.compiled(*[self.tensor(t) for t in tensors], _cu_stream())


# ---------------------------------------------------------------------------
# The kernel bundle
# ---------------------------------------------------------------------------
CUTE: dict = {}     # (kind, shape-key) -> CuteRunner
TRITON = None       # (triton kernel, meta)
KM = None           # the nvcc-built CUDA module, or None when --no-fused-ln
_FA4: dict = {}
_BUILT = False

# Which kernel runs SDXL's self-attention. The graph's own choice is the
# CUTLASS Blackwell FMHA ("fmha"); "fa4" is flash-attn 4, the substitution both
# reference ports made. torch's own SDPA measured faster than both at SDXL's
# two self-attention shapes, so it is the default — the README's attention
# table is the measurement that decided it.
SELF_ATTN = os.environ.get("SDXL_SELF_ATTN", "sdpa")


def _build_triton():
    meta = KERNELS["cross_attn"]
    module = _load_kernel_module("cross_attn", "cross_attn",
                                 ["import triton", "import triton.language as tl"])
    return getattr(module, meta["entry"]), meta


_CUDA_PREAMBLE = r"""
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
union P8 { uint4 u; __nv_bfloat162 h[4]; };
"""

def _cuda_wrappers() -> str:
    """The launch shims. Block sizes come from the manifest, not from here: the
    kernels bake their block width into a fixed-size shared array, so launching
    one at any other width is an out-of-bounds write, not a slowdown."""
    b1280 = KERNELS["bias_residual_ln_w1280"]["block"][0]
    b640 = KERNELS["bias_residual_ln_w640"]["block"][0]
    return f"""
void bias_residual_ln(torch::Tensor add_out, torch::Tensor norm_out,
                      torch::Tensor a, torch::Tensor mm, torch::Tensor proj_bias,
                      torch::Tensor w, torch::Tensor b) {{
  const int width = a.size(-1);
  const int rows = a.numel() / width;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int* dyn = nullptr;
  if (width == 1280) {{
    bias_residual_ln_w1280_k<<<rows, {b1280}, 0, stream>>>(
      (__nv_bfloat16*)add_out.data_ptr(), (__nv_bfloat16*)norm_out.data_ptr(),
      (const __nv_bfloat16*)a.data_ptr(), (const __nv_bfloat16*)mm.data_ptr(),
      (const __nv_bfloat16*)proj_bias.data_ptr(), (const __nv_bfloat16*)w.data_ptr(),
      (const __nv_bfloat16*)b.data_ptr(), dyn);
  }} else if (width == 640) {{
    bias_residual_ln_w640_k<<<rows, {b640}, 0, stream>>>(
      (__nv_bfloat16*)add_out.data_ptr(), (__nv_bfloat16*)norm_out.data_ptr(),
      (const __nv_bfloat16*)a.data_ptr(), (const __nv_bfloat16*)mm.data_ptr(),
      (const __nv_bfloat16*)proj_bias.data_ptr(), (const __nv_bfloat16*)w.data_ptr(),
      (const __nv_bfloat16*)b.data_ptr(), dyn);
  }} else {{
    TORCH_CHECK(false, "no fused bias+residual+LayerNorm kernel for width ", width);
  }}
}}
"""


def _build_cuda_module(verbose: bool = True):
    from torch.utils.cpp_extension import load_inline

    # Each extracted kernel carries its own `#include` lines and the vectorised
    # bf16 union the graph defines per source file; concatenating them would
    # redefine both, so the shared preamble is emitted once and stripped there.
    shared = ("#include", "union P8")
    body = "\n".join(line for k in ("bias_residual_ln_w1280", "bias_residual_ln_w640")
                      for line in _src(k).splitlines()
                      if not line.startswith(shared))
    return load_inline(
        name="sdxl_hoid_kernels",
        cpp_sources=["void bias_residual_ln(torch::Tensor, torch::Tensor, torch::Tensor,"
                     " torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"],
        cuda_sources=[_CUDA_PREAMBLE + body + _cuda_wrappers()],
        functions=["bias_residual_ln"],
        extra_cuda_cflags=["-O3"],
        verbose=verbose,
    )


def _geglu_params(M, K, O):
    # The persistent scheduler is sized in clusters of CLUSTER_M CTAs, so the
    # count follows the device rather than the constant baked in the graph.
    clusters = torch.cuda.get_device_properties(0).multi_processor_count // 2
    return dict(M=M, K=K, O=O, TILE_M=256, TILE_N=128, CLUSTER_M=2, CLUSTER_N=1,
                TWO_CTA=True, MAX_CLUSTERS=clusters)


def build(fused_ln: bool = True, verbose: bool = True):
    """Compile every kernel. Idempotent."""
    global TRITON, KM, _BUILT
    if _BUILT:
        return
    cutlass_install(verbose=verbose)
    if verbose:
        print("[build] triton: cross-attention ...")
    TRITON = _build_triton()
    _install_triton_allocator()
    if fused_ln:
        if verbose:
            print("[build] nvcc: fused bias+residual+LayerNorm ...")
        KM = _build_cuda_module(verbose=False)
    if SELF_ATTN == "fa4":
        from flash_attn.cute.interface import _flash_attn_fwd

        _FA4["fwd"] = _flash_attn_fwd
    _BUILT = True


def set_self_attn(mode: str):
    """Pick the self-attention implementation: sdpa | fa4 | fmha."""
    global SELF_ATTN
    SELF_ATTN = mode
    if mode == "fa4" and "fwd" not in _FA4:
        from flash_attn.cute.interface import _flash_attn_fwd

        _FA4["fwd"] = _flash_attn_fwd


def _geglu_runner(M, K, O, verbose=True):
    key = ("geglu", M, K, O)
    if key not in CUTE:
        CUTE[key] = CuteRunner("dual_geglu", _geglu_params(M, K, O),
                               [(M, O, 1), (M, K, 1), (2 * O, K, 1), (2 * O,)], verbose)
    return CUTE[key]


def _fmha_runner(B, H, D, SQ, SKV, verbose=True):
    key = ("fmha", B, H, D, SQ, SKV)
    if key not in CUTE:
        CUTE[key] = CuteRunner("fmha", dict(B=B, H=H, D=D, SQ=SQ, SKV=SKV),
                               [(B, SQ, H, 1, D), (B, SQ, H, 1, D),
                                (B, SKV, H, 1, D), (B, SKV, H, 1, D)], verbose)
    return CUTE[key]


# ---------------------------------------------------------------------------
# torch custom ops — allocating wrappers over the kernels, so the whole model
# still traces under torch.compile and captures into CUDA graphs.
# ---------------------------------------------------------------------------
_LIB = "sdxlturbo"


@torch.library.custom_op(f"{_LIB}::dual_geglu", mutates_args=())
def dual_geglu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """One persistent dual-GEMM: out = (x@Wv^T + bv) * gelu(x@Wg^T + bg).

    `weight` is the stock GEGLU projection [2*O, K] and `bias` its [2*O] — the
    value half first, exactly as `chunk(2, -1)` splits the stock output.
    """
    x = x.contiguous()
    M, K = x.shape
    O = weight.shape[0] // 2
    out = torch.empty(M, O, device=x.device, dtype=x.dtype)
    _geglu_runner(M, K, O, verbose=False)(out, x, weight, bias)
    return out


@dual_geglu.register_fake
def _(x, weight, bias):
    return x.new_empty(x.shape[0], weight.shape[0] // 2)


@torch.library.custom_op(f"{_LIB}::cross_attention", mutates_args=())
def cross_attention(q: torch.Tensor, kv_wide: torch.Tensor, scale: torch.Tensor,
                    heads: int, k_col: int, v_col: int) -> torch.Tensor:
    """Flash-attention whose K and V are columns of a shared packed buffer.

    `q` is [B, S, H*D]; `kv_wide` is [B, KV_LEN, ROW_STRIDE] holding every
    cross-attention K and V at this hidden width, and this head's K starts at
    column `k_col`, its V at `v_col`.
    """
    kernel, meta = TRITON
    q = q.contiguous()
    B, S, HD = q.shape
    D = HD // heads
    KV_LEN, ROW = kv_wide.shape[1], kv_wide.shape[2]
    cx = dict(meta["constexpr_variants"][0])
    BM, BN = cx["BLOCK_M"], cx["BLOCK_N"]
    out = torch.empty_like(q)
    grid = (-(-S // BM), B * heads, 1)
    dyn = _EMPTY_DYN.get("t")
    if dyn is None:
        dyn = _EMPTY_DYN["t"] = torch.empty(0, dtype=torch.int32, device=q.device)
    kernel[grid](out, q, kv_wide, scale, dyn,
                 Q_LEN=S, KV_LEN=KV_LEN, HEAD_DIM=D, NUM_HEADS=heads,
                 BLOCK_M=BM, BLOCK_N=BN, KV_ROW_STRIDE=ROW, K_COL=k_col, V_COL=v_col,
                 num_warps=meta["num_warps"], num_stages=meta["num_stages"])
    return out


@cross_attention.register_fake
def _(q, kv_wide, scale, heads, k_col, v_col):
    return torch.empty_like(q)


@torch.library.custom_op(f"{_LIB}::self_attention", mutates_args=())
def self_attention(qkv: torch.Tensor, heads: int) -> torch.Tensor:
    """Self-attention over ONE packed [B, S, 3*H*D] projection, no mask.

    Packed because the three projections read the same rows, and one wide GEMM
    reaches roughly twice the throughput of three narrow ones at these shapes.
    """
    B, S, HD3 = qkv.shape
    HD = HD3 // 3
    D = HD // heads
    q, k, v = qkv.split(HD, dim=-1)
    if SELF_ATTN == "sdpa":
        qh, kh, vh = (t.view(B, S, heads, D).transpose(1, 2) for t in (q, k, v))
        return (F.scaled_dot_product_attention(qh, kh, vh)
                .transpose(1, 2).reshape(B, S, HD))
    q, k, v = (t.contiguous() for t in (q, k, v))
    out = torch.empty_like(q)
    if SELF_ATTN == "fa4":
        _FA4["fwd"](q.view(B, S, heads, D), k.view(B, S, heads, D), v.view(B, S, heads, D),
                    softmax_scale=D ** -0.5, causal=False, out=out.view(B, S, heads, D))
    else:
        _fmha_runner(B, heads, D, S, S, verbose=False)(
            out.view(B, S, heads, 1, D), q.view(B, S, heads, 1, D),
            k.view(B, S, heads, 1, D), v.view(B, S, heads, 1, D))
    return out


@self_attention.register_fake
def _(qkv, heads):
    B, S, HD3 = qkv.shape
    return qkv.new_empty(B, S, HD3 // 3)


@torch.library.custom_op(f"{_LIB}::bias_residual_ln", mutates_args=())
def bias_residual_ln(mm: torch.Tensor, proj_bias: torch.Tensor, residual: torch.Tensor,
                     gamma: torch.Tensor, beta: torch.Tensor) -> list[torch.Tensor]:
    """(residual + mm + proj_bias) and its LayerNorm, in one pass."""
    # the kernels index rows by a flat offset, so a strided view would read
    # the wrong elements rather than run slower
    mm, residual = mm.contiguous(), residual.contiguous()
    add_out = torch.empty_like(residual)
    norm_out = torch.empty_like(residual)
    KM.bias_residual_ln(add_out, norm_out, residual, mm, proj_bias, gamma, beta)
    return [add_out, norm_out]


@bias_residual_ln.register_fake
def _(mm, proj_bias, residual, gamma, beta):
    return [torch.empty_like(residual), torch.empty_like(residual)]


# ---------------------------------------------------------------------------
# The packed cross-attention KV
#
# Every cross-attention at one hidden width projects the SAME encoder states,
# so their K and V weights concatenate into one matrix and their projections
# become one GEMM. The result is laid out [K_0 | V_0 | K_1 | V_1 | ...] along
# the row, which is exactly what the cross-attention kernel indexes into with
# a column offset — the packing is free, not a repack before the attention.
# ---------------------------------------------------------------------------
class CrossKVPack(nn.Module):
    def __init__(self, attns):
        super().__init__()
        dim = attns[0].to_k.out_features
        cols = []
        for a in attns:
            assert a.to_k.bias is None and a.to_v.bias is None, "SDXL cross-attn is bias-free"
            assert a.to_k.out_features == dim, "one pack per hidden width"
            cols += [a.to_k.weight.data.t(), a.to_v.weight.data.t()]
        # [ctx_dim, n*2*dim] so the projection is a plain row-major GEMM
        self.register_buffer("weight_t", torch.cat(cols, dim=1).contiguous())
        self.row_stride = self.weight_t.shape[1]
        self.dim = dim

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        B, S, _ = ctx.shape
        return (ctx.reshape(B * S, -1) @ self.weight_t).view(B, S, self.row_stride)


# ---------------------------------------------------------------------------
# The transformer block, rebuilt around the winning kernels
# ---------------------------------------------------------------------------
class Kernels:
    """Which kernels are switched on. Every one defaults to on; the flags exist
    so each kernel's own contribution can be measured, and the README's
    ablation table is exactly these rows."""

    def __init__(self, fused_ln=True, geglu=True, packed_kv=True):
        self.fused_ln, self.geglu, self.packed_kv = fused_ln, geglu, packed_kv

    def __repr__(self):
        on = [n for n in ("fused_ln", "geglu", "packed_kv") if getattr(self, n)]
        return f"kernels({', '.join(on) or 'none'})"


class HoidTransformerBlock(nn.Module):
    """Stands in for `BasicTransformerBlock` at SDXL's settings.

    Holds the stock submodules and runs SDXL's exact block algebra —
    LN -> self-attn -> residual -> LN -> cross-attn -> residual -> LN -> GEGLU
    feed-forward -> residual — with the kernels swapped in.
    """

    def __init__(self, block, pack, k_col: int, v_col: int, kernels: "Kernels"):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = block.norm1, block.norm2, block.norm3
        self.attn1, self.attn2 = block.attn1, block.attn2
        self.ff = block.ff
        self.pack = [pack]           # a plain list: not a child, so it is packed once
        self.k_col, self.v_col = k_col, v_col
        self.heads = block.attn1.heads
        self.k = kernels
        self.fused_ln = kernels.fused_ln
        a1 = block.attn1
        assert a1.to_q.bias is None, "SDXL self-attention is bias-free"
        self.register_buffer("qkv_weight", torch.cat(
            [a1.to_q.weight.data, a1.to_k.weight.data, a1.to_v.weight.data], dim=0).contiguous())
        self.register_buffer(
            "cross_scale",
            torch.tensor(block.attn2.scale, dtype=torch.bfloat16,
                         device=block.attn2.to_q.weight.device))
        # the originals are dead once packed; dropping them keeps the port's
        # memory in line with the stock model instead of 600 MB above it
        a1.to_q = a1.to_k = a1.to_v = None
        if kernels.packed_kv:
            block.attn2.to_k = block.attn2.to_v = None

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, timestep=None, cross_attention_kwargs=None,
                class_labels=None, added_cond_kwargs=None):
        return self.run(hidden_states, None, None, cross_attention_kwargs,
                        encoder_hidden_states)[0]

    def run(self, hidden_states, prenorm, next_norm, cross_attention_kwargs,
            encoder_hidden_states=None):
        """One block. `prenorm` is this block's `norm1(hidden_states)` when the
        previous block already produced it; `next_norm` is the following
        block's norm1, folded into this block's feed-forward residual add.
        Returns (hidden_states, that following block's pre-norm or None)."""
        h = hidden_states

        # -- 1. self-attention, q/k/v as one GEMM -----------------------------
        n = self.norm1(h) if prenorm is None else prenorm
        a = self_attention(F.linear(n, self.qkv_weight), self.heads)
        h, n = self._out_proj_residual_norm(a, self.attn1, h, self.norm2)

        # -- 2. cross-attention, K and V read out of the shared pack ----------
        q = self.attn2.to_q(n)
        if self.k.packed_kv:
            wide = (cross_attention_kwargs or {})["hoid_wide"][self.pack[0].dim]
            a = cross_attention(q, wide, self.cross_scale, self.heads,
                                self.k_col, self.v_col)
        else:
            a = self._stock_cross_attention(q, encoder_hidden_states)
        h, n = self._out_proj_residual_norm(a, self.attn2, h, self.norm3)

        # -- 3. feed-forward: one dual-GEMM for the whole GEGLU ---------------
        B, T, K = n.shape
        if self.k.geglu:
            proj = self.ff.net[0].proj
            g = dual_geglu(n.reshape(B * T, K), proj.weight, proj.bias).view(B, T, -1)
        else:
            g = self.ff.net[0](n)
        out = self.ff.net[2]
        if self.fused_ln and next_norm is not None:
            return bias_residual_ln(F.linear(g, out.weight), out.bias, h,
                                    next_norm.weight, next_norm.bias)
        return F.linear(g, out.weight, out.bias) + h, None

    def _stock_cross_attention(self, q, encoder_hidden_states):
        """The diffusers path: per-block K/V projections and SDPA."""
        a2 = self.attn2
        B, S, HD = q.shape
        D = HD // self.heads
        k = a2.to_k(encoder_hidden_states)
        v = a2.to_v(encoder_hidden_states)
        kv = encoder_hidden_states.shape[1]
        qh = q.view(B, S, self.heads, D).transpose(1, 2)
        kh = k.view(B, kv, self.heads, D).transpose(1, 2)
        vh = v.view(B, kv, self.heads, D).transpose(1, 2)
        return (F.scaled_dot_product_attention(qh, kh, vh)
                .transpose(1, 2).reshape(B, S, HD))

    def _out_proj_residual_norm(self, a, attn, h, norm):
        """attn output projection -> +bias -> +residual -> LayerNorm."""
        out = attn.to_out[0]
        if self.fused_ln:
            add_out, norm_out = bias_residual_ln(
                F.linear(a, out.weight), out.bias, h, norm.weight, norm.bias)
            return add_out, norm_out
        h = out(a) + h
        return h, norm(h)


class HoidBlockChain(nn.Module):
    """The transformer blocks of one `Transformer2DModel`, run as a chain.

    Standing in for the whole block list rather than one block is what lets a
    block's feed-forward residual add be fused with the NEXT block's
    `norm1` — the compiler cannot reach across the loop iteration to do it,
    and it is 59 of the 199 fused-LayerNorm sites.
    """

    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, timestep=None, cross_attention_kwargs=None,
                class_labels=None, added_cond_kwargs=None):
        n = None
        last = len(self.blocks) - 1
        for i, block in enumerate(self.blocks):
            nxt = self.blocks[i + 1].norm1 if i < last else None
            hidden_states, n = block.run(hidden_states, n, nxt, cross_attention_kwargs,
                                         encoder_hidden_states)
        return hidden_states


class OptimizedUNet(nn.Module):
    """The stock UNet with its transformer blocks replaced, plus the one
    pre-pass (the packed cross-attention KV) the blocks share."""

    def __init__(self, unet, kernels: "Kernels" = None):
        super().__init__()
        kernels = kernels or Kernels()
        self.kernels = kernels
        self.unet = unet
        self.config = unet.config
        self.dtype = unet.dtype
        blocks = [b for m in unet.modules()
                  for b in getattr(m, "transformer_blocks", [])]
        by_dim: dict[int, list] = {}
        for b in blocks:
            by_dim.setdefault(b.attn2.to_k.out_features, []).append(b)

        self.packs = nn.ModuleDict()
        replacement = {}
        for dim, group in by_dim.items():
            pack = CrossKVPack([b.attn2 for b in group]) if kernels.packed_kv else None
            if pack is not None:
                self.packs[str(dim)] = pack
            for i, b in enumerate(group):
                replacement[id(b)] = HoidTransformerBlock(
                    b, pack, 2 * i * dim, (2 * i + 1) * dim, kernels)
        # Each Transformer2DModel's whole block list becomes ONE chain module,
        # so `for block in self.transformer_blocks` runs it in a single call.
        for m in unet.modules():
            stock = getattr(m, "transformer_blocks", None)
            if stock is None or not len(stock):
                continue
            m.transformer_blocks = nn.ModuleList(
                [HoidBlockChain([replacement[id(b)] for b in stock])])

    def __getattr__(self, name):
        """Attribute reads the diffusers pipeline makes on a UNet — `config`,
        `add_embedding`, `sample_size` — fall through to the stock module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            stock = self._modules.get("unet")
            if stock is None:
                raise
            return getattr(stock, name)

    def forward(self, sample, timestep, encoder_hidden_states, added_cond_kwargs=None,
                cross_attention_kwargs=None, return_dict=True, **kw):
        wide = {p.dim: p(encoder_hidden_states) for p in self.packs.values()}
        cak = dict(cross_attention_kwargs or {})
        cak["hoid_wide"] = wide
        return self.unet(sample, timestep, encoder_hidden_states,
                         added_cond_kwargs=added_cond_kwargs,
                         cross_attention_kwargs=cak, return_dict=return_dict, **kw)


def optimize(unet, kernels: "Kernels" = None, verbose: bool = True) -> OptimizedUNet:
    """Compile the kernels and return the UNet with them wired in.

    The UNet is modified in place; hand it a copy if you need the stock one.
    """
    kernels = kernels or Kernels()
    build(fused_ln=kernels.fused_ln, verbose=verbose)
    return OptimizedUNet(unet, kernels=kernels)


_ALLOC = {}


def _install_triton_allocator():
    """The cross-attention kernel builds its TMA maps on the device, which
    Triton serves out of a caller-provided global scratch buffer."""
    import triton

    def alloc(size: int, align: int, stream):
        buf = _ALLOC.get(size)
        if buf is None:
            buf = torch.empty(size, dtype=torch.int8, device="cuda")
            _ALLOC[size] = buf
        return buf

    triton.set_allocator(alloc)


def bench(fn, iters: int = 20, warmup: int = 5):
    """mean/min ms over `iters` timed calls, each with a sync on both sides."""
    import statistics

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1e3)
    return statistics.mean(times), min(times)


def snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref, got = ref.float(), got.float()
    err = got - ref
    return (10 * torch.log10(ref.square().sum() / err.square().sum().clamp_min(1e-30))).item()


# ---------------------------------------------------------------------------
# The VAE decoder's mid-block attention
#
# It is a SINGLE head of dim 512 over all 16384 latent positions, which is past
# what torch's flash backend accepts, so SDPA falls back to a memory-efficient
# sm80 kernel and the one attention costs more than every GroupNorm in the
# decoder put together. The CUTLASS Blackwell d=512 prefill FMHA does the same
# maths on the tensor cores instead.
# ---------------------------------------------------------------------------
VAE_ATTN_TOKENS = 16384
VAE_ATTN_DIM = 512


def _fmha_d512_runner(S=VAE_ATTN_TOKENS, D=VAE_ATTN_DIM, verbose=True):
    key = ("fmha_d512", S, D)
    if key not in CUTE:
        shape = (1, S, 1, 1, D)
        CUTE[key] = CuteRunner("fmha_d512", {}, [shape] * 4, verbose)
    return CUTE[key]


@torch.library.custom_op(f"{_LIB}::vae_attention", mutates_args=())
def vae_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Single-head attention over [1, S, 512] operands, no mask."""
    q, k, v = (t.contiguous() for t in (q, k, v))
    B, S, D = q.shape
    out = torch.empty_like(q)
    shape = (B, S, 1, 1, D)
    _fmha_d512_runner(S, D, verbose=False)(
        out.view(shape), q.view(shape), k.view(shape), v.view(shape))
    return out


@vae_attention.register_fake
def _(q, k, v):
    return torch.empty_like(q)


class HoidVAEAttnProcessor:
    """Stands in for `AttnProcessor2_0` on the VAE mid-block attention.

    Same algebra as diffusers — group_norm, q/k/v, attention, out projection,
    residual, rescale — with the attention kernel swapped.
    """

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kwargs):
        residual = hidden_states
        B, C, H, W = hidden_states.shape
        x = attn.group_norm(hidden_states.view(B, C, H * W)).transpose(1, 2)
        q, k, v = attn.to_q(x), attn.to_k(x), attn.to_v(x)
        out = vae_attention(q, k, v)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return (out + residual) / attn.rescale_output_factor


def optimize_vae(vae, verbose: bool = True):
    """Wire the tuned attention kernel into a stock `AutoencoderKL`.

    Modifies the VAE in place and returns it. Only the decoder's mid-block
    attention changes; every convolution and GroupNorm stays stock diffusers.
    """
    cutlass_install(verbose=verbose)
    patched = 0
    for attn in getattr(vae.decoder.mid_block, "attentions", []):
        if attn.heads != 1 or attn.to_q.out_features != VAE_ATTN_DIM:
            continue
        attn.set_processor(HoidVAEAttnProcessor())
        patched += 1
    if verbose:
        print(f"[build] vae: {patched} mid-block attention(s) on the d=512 FMHA")
    return vae
