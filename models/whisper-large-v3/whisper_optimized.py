#!/usr/bin/env python
"""whisper-large-v3's encoder in PyTorch, running the hoid-winning kernels.

Every kernel here is from the stack hoid landed at 3.525 ms/forward on a
B200 (1.176x over torch's best config), lifted out verbatim and driven
from torch:

  - `conv1`   Triton implicit-GEMM stride-1 conv + bias + exact-erf GELU
  - `conv2`   im2col -> Blackwell EFC dense GEMM with a fused bias epilogue
              -> exact-erf GELU written back channel-major
  - `tpos`    transpose + positional-embedding add
  - `qkv`     ONE persistent Blackwell triple-GEMM (q, k, v; q/v biased)
  - `attn`    flash-attn-4 (CuTe DSL) in place of the graph's CUTLASS FMHA
  - `fc1`     persistent Blackwell GEMM with a fused bias + exact-erf GELU
  - `ln`      fused (bias -> residual -> LayerNorm) with the graph's exact
              two-pass ordered reduction, and the plain row LayerNorm
  - out_proj / fc2 stay cuBLAS GEMMs, as they are in the winning graph

The CUDA kernels are nvcc-JIT'd at import, the Triton kernel is compiled by
Triton, and the CuTe DSL kernels are compiled by `cute.compile` against the
static shapes — no hoid runtime is involved at any point.

Two forwards, same kernels:

  `forward`         each op allocates its output and is a torch custom op —
                    the form `torch.compile` can trace.
  `forward_static`  the same calls on buffers allocated once up front. This is
                    the fast path and the only capture-safe one: a tensor
                    allocated *inside* a CUDA-graph capture and freed
                    afterwards is not protected from later allocations, so a
                    graph built around allocating ops silently starts reading
                    other people's memory a few hundred megabytes later.
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

from whisper_kernels_gen import KERNELS

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, "kernels")


# ---------------------------------------------------------------------------
# The pinned CUTLASS CuTe DSL example files the packed-QKV, fc1 and conv2 GEMM
# kernels subclass. The `nvidia-cutlass-dsl` wheel does not ship the example
# tree, so fetch exactly three files at the pinned tag, verify their sha256, and
# lay them out under the relative path the kernel sources expect (they locate
# the tree by looking for a sys.path entry ending in examples/python/CuTeDSL,
# which is how the CUTLASS examples import each other). Cached in
# .cutlass_examples/ next to this file; later runs are a hash check.
# ---------------------------------------------------------------------------
CUTLASS_TAG = "v4.6.1"
_RAW = f"https://raw.githubusercontent.com/NVIDIA/cutlass/{CUTLASS_TAG}/examples/python/CuTeDSL"

# path relative to examples/python/CuTeDSL -> sha256 at CUTLASS_TAG
FILES = {
    "cute/blackwell/kernel/dense_gemm/dense_gemm_persistent.py":
        "d59344faf902cb215a2cee3f2ae6415a14589c6ad8f93e5e74e2612c1e6a0810",
    "cute/blackwell/efc/common_dense_gemm_efc.py":
        "be85b6566bd96d46df515842d45e75c8a93d8df41ecebb165ec0cdbb9b5efb3b",
    "cute/blackwell/efc/common_efc.py":
        "3e1a71f80d7e6f1ec8de277c7a97814d40c5a97739700f39c5866a99d0d372ed",
}

CUTLASS_ROOT = os.path.join(HERE, ".cutlass_examples", "examples", "python", "CuTeDSL")


def _cutlass_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _cutlass_ensure(verbose: bool = True) -> str:
    """Download-if-missing + hash-check, and return the CuTeDSL example root."""
    for rel, want in FILES.items():
        dst = os.path.join(CUTLASS_ROOT, rel)
        if os.path.exists(dst) and _cutlass_sha256(dst) == want:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        url = f"{_RAW}/{rel}"
        if verbose:
            print(f"[cutlass] fetching {rel} @ {CUTLASS_TAG}")
        with urllib.request.urlopen(url) as r:
            blob = r.read()
        got = hashlib.sha256(blob).hexdigest()
        if got != want:
            raise RuntimeError(
                f"cutlass example {rel} at {CUTLASS_TAG} hashes {got}, expected {want}. "
                "The pin moved or the download is corrupt; do not run with unverified sources."
            )
        with open(dst, "wb") as f:
            f.write(blob)
    return CUTLASS_ROOT


def cutlass_install(verbose: bool = True) -> str:
    """`ensure` + put the tree on sys.path the way the CUTLASS examples do."""
    root = _cutlass_ensure(verbose)
    for p in (root,
              os.path.join(root, "cute/blackwell/kernel/dense_gemm"),
              os.path.join(root, "cute/blackwell/efc")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


# whisper-large-v3 encoder, the static shape the kernels were tuned for.
MEL_BINS = 128
MEL_FRAMES = 3000
POS = 1500          # encoder positions after the stride-2 conv
DIM = 1280
HEADS = 20
HEAD_DIM = 64
FFN = 5120
LAYERS = 32
ATTN_SCALE = HEAD_DIM ** -0.5


def _src(key: str) -> str:
    """The kernel source, hash-checked against the manifest."""
    meta = KERNELS[key]
    with open(os.path.join(KDIR, meta["file"])) as f:
        text = f.read()
    got = hashlib.sha256(text.encode()).hexdigest()
    if got != meta["sha256"]:
        raise RuntimeError(
            f"kernel {key} ({meta['file']}) hashes {got[:16]}, manifest says "
            f"{meta['sha256'][:16]} — the source no longer matches the shipped "
            "artifact it was extracted from."
        )
    return text


# ---------------------------------------------------------------------------
# CUDA kernels (verbatim) + the launch geometry hoid measured them with
# ---------------------------------------------------------------------------
_CUDA_WRAPPERS = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

using bf16 = __nv_bfloat16;
#define STREAM at::cuda::getCurrentCUDAStream().stream()
#define P(t) reinterpret_cast<bf16*>((t).data_ptr())
#define C(t) reinterpret_cast<const bf16*>((t).data_ptr())

// Every entry point writes into caller-provided buffers: under CUDA-graph
// capture, a tensor allocated inside the capture and freed afterwards is not
// protected from later allocations, so the whole forward runs on buffers that
// were allocated before capture began.

// grid 1500 x block 128 — one block per encoder position, as in the graph.
void ordered_residual_ln(torch::Tensor residual, torch::Tensor norm, torch::Tensor mat,
                         torch::Tensor bias, torch::Tensor skip, torch::Tensor gamma,
                         torch::Tensor beta) {
    whisper_ordered_residual_ln_replicated_k<<<1500, 128, 0, STREAM>>>(
        P(residual), P(norm), C(mat), C(bias), C(skip), C(gamma), C(beta), nullptr);
}

// grid 1500 x block 256
void row_layer_norm(torch::Tensor out, torch::Tensor x, torch::Tensor gamma,
                    torch::Tensor beta) {
    whisper_ln_volatile_warptail_k<<<1500, 256, 0, STREAM>>>(
        P(out), C(x), C(gamma), C(beta), nullptr);
}

// grid 938 x block 256 (1500*1280 elements, 8 per thread)
void bias_residual(torch::Tensor out, torch::Tensor mat, torch::Tensor bias,
                   torch::Tensor skip) {
    whisper_ordered_bias_residual_k<<<938, 256, 0, STREAM>>>(
        P(out), C(mat), C(bias), C(skip), nullptr);
}

// grid 7500 x block 256
void transpose_pos_add(torch::Tensor out, torch::Tensor conv, torch::Tensor pos) {
    whisper_tpos_k<<<7500, 256, 0, STREAM>>>(P(out), C(conv), C(pos), nullptr);
}

// grid (47, 40, 3) x block (32, 8, 1)
void conv2_im2col(torch::Tensor out, torch::Tensor x) {
    whisper_conv2_im2col_tmajor_k<<<dim3(47, 40, 3), dim3(32, 8, 1), 0, STREAM>>>(
        P(out), C(x), nullptr);
}

// grid 19200 x block 256 — weight prep, run once at load
torch::Tensor conv2_weight_tmajor(torch::Tensor w) {
    auto out = torch::empty({1280, 3 * 1280}, w.options());
    whisper_conv2_weight_tmajor_k<<<19200, 256, 0, STREAM>>>(P(out), C(w), nullptr);
    return out;
}

// grid 7500 x block 256
void conv2_gelu_channel_major(torch::Tensor out, torch::Tensor x) {
    whisper_conv2_exact_gelu_channel_major_k<<<7500, 256, 0, STREAM>>>(
        P(out), C(x), nullptr);
}
"""

_CUDA_DECLS = """
void ordered_residual_ln(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void row_layer_norm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void bias_residual(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void transpose_pos_add(torch::Tensor, torch::Tensor, torch::Tensor);
void conv2_im2col(torch::Tensor, torch::Tensor);
torch::Tensor conv2_weight_tmajor(torch::Tensor);
void conv2_gelu_channel_major(torch::Tensor, torch::Tensor);
"""

_CUDA_KEYS = ("ordered_residual_ln", "ln", "ordered_bias_residual", "tpos",
              "conv2_im2col", "conv2_weight_tmajor", "conv2_gelu_cm")


def _build_cuda_module(verbose: bool = False):
    from torch.utils.cpp_extension import load_inline

    body = "\n".join(_src(k) for k in _CUDA_KEYS)
    return load_inline(
        name="whisper_hoid_kernels",
        cpp_sources=[_CUDA_DECLS],
        cuda_sources=[body + _CUDA_WRAPPERS],
        functions=["ordered_residual_ln", "row_layer_norm", "bias_residual",
                   "transpose_pos_add", "conv2_im2col", "conv2_weight_tmajor",
                   "conv2_gelu_channel_major"],
        extra_cuda_cflags=["-O3"],
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Loading authored kernel sources
#
# Both Triton and the CuTe DSL read a decorated function's source back with
# `inspect.getsourcelines`, so neither can be exec'd into a namespace — each
# source has to become a real file on disk. The preamble supplies the
# contract every source assumes: its imports plus the compile params as
# module-level constants.
# ---------------------------------------------------------------------------
_TMP_MODULES = []  # keeps the generated files alive for the life of the process


def _load_kernel_module(key: str, preamble: list[str]):
    tmp = tempfile.mkdtemp(prefix="whisper_kernel_")
    _TMP_MODULES.append(tmp)
    path = os.path.join(tmp, f"{key}_kernel.py")
    with open(path, "w") as f:
        f.write("\n".join(preamble) + "\n\n" + _src(key))
    sys.path.insert(0, tmp)
    spec = importlib.util.spec_from_file_location(f"whisper_kernel_{key}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_triton_conv1():
    meta = KERNELS["conv1"]
    module = _load_kernel_module("conv1", ["import triton", "import triton.language as tl"])
    return getattr(module, meta["entry"]), meta


def _load_cutedsl_entry(key: str):
    meta = KERNELS[key]
    preamble = ["import cutlass", "import cutlass.cute as cute",
                "import cuda.bindings.driver as cuda"]
    for name, value in (meta.get("compile") or {}).items():
        preamble.append(f"{name} = {value!r}")
    module = _load_kernel_module(key, preamble)
    return getattr(module, meta["entry"]), meta


class CuteRunner:
    """One compiled CuTe DSL entry, called with torch tensors.

    Tensor descriptors are cached by data pointer: building them costs a few
    microseconds each, which is invisible under CUDA-graph replay but not in a
    3.5 ms eager forward, and the buffers are static here anyway.
    """

    def __init__(self, key: str, arg_shapes, verbose: bool = True):
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack

        self._from_dlpack = from_dlpack
        self._cache: dict[int, object] = {}
        entry, meta = _load_cutedsl_entry(key)
        self.key = key
        self.meta = meta
        t0 = time.perf_counter()
        fakes = [self._fake(shape) for shape in arg_shapes]
        self.compiled = cute.compile(entry, *fakes, _cu_stream(),
                                     options=meta.get("options") or "")
        if verbose:
            print(f"[cutedsl] {key} compiled in {time.perf_counter() - t0:.1f}s")

    def _fake(self, shape):
        import cutlass
        from cutlass.cute.runtime import make_fake_compact_tensor

        # row-major, exactly as hoid binds them
        order = tuple(range(len(shape) - 1, -1, -1))
        return make_fake_compact_tensor(cutlass.BFloat16, tuple(shape), stride_order=order)

    def tensor(self, t: torch.Tensor):
        """A cached CuTe view of a torch tensor (2-D tensors get the
        trailing unit mode the kernels expect for GEMM operands)."""
        key = t.data_ptr()
        hit = self._cache.get(key)
        if hit is None:
            t = t.detach()  # HF parameters carry requires_grad; dlpack refuses those
            view = t.unsqueeze(-1) if t.dim() == 2 else t
            hit = self._from_dlpack(view, assumed_align=16)
            self._cache[key] = hit
        return hit

    def __call__(self, *tensors: torch.Tensor):
        self.compiled(*[self.tensor(t) for t in tensors], _cu_stream())


def _cu_stream():
    import cuda.bindings.driver as cuda_driver

    return cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)


# ---------------------------------------------------------------------------
# Low-level entry points: kernels writing into caller-owned buffers. The static
# forward and the CUDA-graph capture use these directly; the torch custom ops
# below are thin allocating wrappers over the same calls.
# ---------------------------------------------------------------------------
def k_conv1(out, mel, weight, bias, dyn):
    kernel, meta = CONV1
    cx = meta["constexprs"]
    grid = (-(-MEL_FRAMES // cx["BLOCK_L"]), -(-DIM // cx["BLOCK_CO"]), 1)  # ceil_div, as in the graph
    kernel[grid](out, mel, weight, bias, dyn,
                 BLOCK_CO=cx["BLOCK_CO"], BLOCK_L=cx["BLOCK_L"], BLOCK_K=cx["BLOCK_K"],
                 num_warps=meta["num_warps"], num_stages=meta["num_stages"])


def k_conv2(out, im2col, gemm, x, weight_t, bias):
    """im2col -> EFC GEMM(+bias) -> exact GELU, written back channel-major."""
    KM.conv2_im2col(im2col, x)
    CUTE["conv2_efc"](gemm, im2col, weight_t, bias)
    KM.conv2_gelu_channel_major(out, gemm)


def k_attn(out, q, k, v):
    """flash-attn 4, writing into a preallocated output."""
    _FA4["fwd"](q.view(1, POS, HEADS, HEAD_DIM), k.view(1, POS, HEADS, HEAD_DIM),
                v.view(1, POS, HEADS, HEAD_DIM), softmax_scale=ATTN_SCALE, causal=False,
                out=out.view(1, POS, HEADS, HEAD_DIM))


# ---------------------------------------------------------------------------
# the whole kernel bundle + torch custom-op registration
# ---------------------------------------------------------------------------
KM = None          # the nvcc-built CUDA module
CONV1 = None       # (triton kernel, meta)
CUTE = {}          # key -> CuteRunner
_FA4 = {}          # flash-attn 4 forward
_REGISTERED = False


def register_ops(verbose: bool = True, fa4: bool = True):
    """Compile every kernel and register it as a torch custom op."""
    global KM, CONV1, CUTE, _REGISTERED
    if _REGISTERED:
        return

    cutlass_install(verbose=verbose)
    if verbose:
        print("[build] nvcc: 7 CUDA kernels ...")
    KM = _build_cuda_module()
    if verbose:
        print("[build] triton: conv1 ...")
    CONV1 = _build_triton_conv1()
    if verbose:
        print("[build] cutedsl: qkv / fc1 / conv2 / fc2 (first compile is slow) ...")
    CUTE["triple_qkv"] = CuteRunner(
        "triple_qkv",
        [(POS, DIM, 1)] * 3 + [(POS, DIM, 1), (DIM, DIM, 1), (DIM,), (DIM, DIM, 1),
                               (DIM, DIM, 1), (DIM,)],
        verbose)
    CUTE["fc1_gelu"] = CuteRunner(
        "fc1_gelu", [(POS, FFN, 1), (POS, DIM, 1), (FFN, DIM, 1), (FFN,)], verbose)
    CUTE["conv2_efc"] = CuteRunner(
        "conv2_efc", [(POS, DIM, 1), (POS, 3 * DIM, 1), (DIM, 3 * DIM, 1), (DIM,)], verbose)
    CUTE["dense_residency"] = CuteRunner(
        "dense_residency", [(POS, DIM, 1), (POS, FFN, 1), (DIM, FFN, 1)], verbose)

    from flash_attn.cute.interface import _flash_attn_fwd

    _FA4["fwd"] = _flash_attn_fwd

    # -- torch custom ops: allocating wrappers over the same kernels, so the
    # -- stack also composes with torch.compile and eager autograd-free code.
    lib = torch.library

    @lib.custom_op("whisperturbo::residual_ln", mutates_args=())
    def residual_ln(mat: torch.Tensor, bias: torch.Tensor, skip: torch.Tensor,
                    gamma: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual, norm = torch.empty_like(skip), torch.empty_like(skip)
        KM.ordered_residual_ln(residual, norm, mat, bias, skip, gamma, beta)
        return residual, norm

    @residual_ln.register_fake
    def _(mat, bias, skip, gamma, beta):
        return torch.empty_like(skip), torch.empty_like(skip)

    @lib.custom_op("whisperturbo::layer_norm", mutates_args=())
    def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        KM.row_layer_norm(out, x, gamma, beta)
        return out

    @layer_norm.register_fake
    def _(x, gamma, beta):
        return torch.empty_like(x)

    @lib.custom_op("whisperturbo::bias_residual", mutates_args=())
    def bias_residual(mat: torch.Tensor, bias: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(skip)
        KM.bias_residual(out, mat, bias, skip)
        return out

    @bias_residual.register_fake
    def _(mat, bias, skip):
        return torch.empty_like(skip)

    @lib.custom_op("whisperturbo::conv1", mutates_args=())
    def conv1(mel: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        out = torch.empty(DIM, MEL_FRAMES, device=mel.device, dtype=mel.dtype)
        k_conv1(out, mel, weight, bias, torch.zeros(1, device=mel.device, dtype=torch.int32))
        return out

    @conv1.register_fake
    def _(mel, weight, bias):
        return mel.new_empty(DIM, MEL_FRAMES)

    @lib.custom_op("whisperturbo::conv2", mutates_args=())
    def conv2(x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        out = torch.empty(DIM, POS, device=x.device, dtype=x.dtype)
        im2col = torch.empty(POS, 3 * DIM, device=x.device, dtype=x.dtype)
        gemm = torch.empty(POS, DIM, device=x.device, dtype=x.dtype)
        k_conv2(out, im2col, gemm, x, weight_t, bias)
        return out

    @conv2.register_fake
    def _(x, weight_t, bias):
        return x.new_empty(DIM, POS)

    @lib.custom_op("whisperturbo::tpos", mutates_args=())
    def tpos(conv: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        out = torch.empty(POS, DIM, device=conv.device, dtype=conv.dtype)
        KM.transpose_pos_add(out, conv, pos)
        return out

    @tpos.register_fake
    def _(conv, pos):
        return conv.new_empty(POS, DIM)

    @lib.custom_op("whisperturbo::qkv", mutates_args=())
    def qkv(x: torch.Tensor, wq: torch.Tensor, bq: torch.Tensor, wk: torch.Tensor,
            wv: torch.Tensor, bv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = (torch.empty(POS, DIM, device=x.device, dtype=x.dtype) for _ in range(3))
        CUTE["triple_qkv"](q, k, v, x, wq, bq, wk, wv, bv)
        return q, k, v

    @qkv.register_fake
    def _(x, wq, bq, wk, wv, bv):
        return (torch.empty_like(x), torch.empty_like(x), torch.empty_like(x))

    @lib.custom_op("whisperturbo::fc1", mutates_args=())
    def fc1(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        out = torch.empty(POS, FFN, device=x.device, dtype=x.dtype)
        CUTE["fc1_gelu"](out, x, w, bias)
        return out

    @fc1.register_fake
    def _(x, w, bias):
        return x.new_empty(POS, FFN)

    @lib.custom_op("whisperturbo::fc2_cutedsl", mutates_args=())
    def fc2_cutedsl(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        out = torch.empty(POS, DIM, device=x.device, dtype=x.dtype)
        CUTE["dense_residency"](out, x, w)
        return out

    @fc2_cutedsl.register_fake
    def _(x, w):
        return x.new_empty(POS, DIM)

    @lib.custom_op("whisperturbo::attn", mutates_args=())
    def attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(q)
        k_attn(out, q, k, v)
        return out

    @attn.register_fake
    def _(q, k, v):
        return torch.empty_like(q)

    _REGISTERED = True


# ---------------------------------------------------------------------------
# the encoder
# ---------------------------------------------------------------------------
class OptimizedWhisperEncoder(torch.nn.Module):
    """Static-shape whisper-large-v3 encoder on the hoid kernel stack.

    Takes the HF `WhisperEncoder` (bf16, cuda) and reads its weights; the
    forward runs only the kernels above plus two cuBLAS GEMMs per layer.
    """

    def __init__(self, hf_encoder, fc2_mode: str = "hoid"):
        super().__init__()
        if fc2_mode not in ("hoid", "cublas", "cutedsl"):
            raise ValueError("fc2_mode must be hoid | cublas | cutedsl")
        self.fc2_mode = fc2_mode
        enc = hf_encoder
        dev = enc.conv1.weight.device

        def w(t):
            return t.detach().contiguous()

        self.conv1_w = w(enc.conv1.weight)                    # [1280, 128, 3]
        self.conv1_b = w(enc.conv1.bias)
        self.conv2_w_t = KM.conv2_weight_tmajor(w(enc.conv2.weight))
        self.conv2_b = w(enc.conv2.bias)
        self.pos = w(enc.embed_positions.weight.to(torch.bfloat16))

        self.wq, self.bq, self.wk, self.wv, self.bv = [], [], [], [], []
        self.wo, self.bo = [], []
        self.w1, self.b1, self.w2, self.b2 = [], [], [], []
        self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b = [], [], [], []
        for layer in enc.layers:
            a = layer.self_attn
            self.wq.append(w(a.q_proj.weight)); self.bq.append(w(a.q_proj.bias))
            self.wk.append(w(a.k_proj.weight))
            self.wv.append(w(a.v_proj.weight)); self.bv.append(w(a.v_proj.bias))
            self.wo.append(w(a.out_proj.weight.t()))
            self.bo.append(w(a.out_proj.bias))
            self.w1.append(w(layer.fc1.weight)); self.b1.append(w(layer.fc1.bias))
            self.w2.append(w(layer.fc2.weight.t()))
            self.b2.append(w(layer.fc2.bias))
            self.ln1_g.append(w(layer.self_attn_layer_norm.weight))
            self.ln1_b.append(w(layer.self_attn_layer_norm.bias))
            self.ln2_g.append(w(layer.final_layer_norm.weight))
            self.ln2_b.append(w(layer.final_layer_norm.bias))
        # the CuTe DSL fc2 wants B as (N, K), i.e. the untransposed HF weight
        self.w2_cute = [w(layer.fc2.weight) for layer in enc.layers]
        self.ln_g = w(enc.layer_norm.weight)
        self.ln_b = w(enc.layer_norm.bias)
        # whisper's q is pre-scaled by 1/sqrt(head_dim) inside HF attention;
        # here flash-attn applies the scale, so q_proj stays unscaled.
        self.device = dev
        self.buf = self._alloc_buffers(dev)

    def _alloc_buffers(self, dev):
        """Every buffer the static forward touches, allocated once.

        Under CUDA-graph capture, a tensor allocated inside the capture and
        freed afterwards is NOT protected from later allocations — the graph
        would go on reading memory that the rest of the process has since been
        handed. Allocating the whole working set up front removes the class of
        bug entirely (and removes the allocator from the hot path).
        """
        from types import SimpleNamespace

        def z(*shape):
            return torch.empty(*shape, device=dev, dtype=torch.bfloat16)

        return SimpleNamespace(
            mel=z(MEL_BINS, MEL_FRAMES), c1=z(DIM, MEL_FRAMES),
            im2col=z(POS, 3 * DIM), gemm=z(POS, DIM), c2=z(DIM, POS),
            x=(z(POS, DIM), z(POS, DIM)),     # residual stream, ping-pong
            h=z(POS, DIM), q=z(POS, DIM), k=z(POS, DIM), v=z(POS, DIM),
            attn=z(POS, DIM), o=z(POS, DIM), f=z(POS, FFN), y=z(POS, DIM),
            out=z(POS, DIM),
            dyn=torch.zeros(1, device=dev, dtype=torch.int32),
        )

    # -- pieces -------------------------------------------------------------
    def frontend(self, mel: torch.Tensor) -> torch.Tensor:
        """log-mel [128, 3000] -> positioned hidden states [1500, 1280]."""
        c1 = torch.ops.whisperturbo.conv1(mel, self.conv1_w, self.conv1_b)
        c2 = torch.ops.whisperturbo.conv2(c1, self.conv2_w_t, self.conv2_b)
        return torch.ops.whisperturbo.tpos(c2, self.pos)

    def layer(self, i: int, x: torch.Tensor, h: torch.Tensor):
        """One encoder layer. `x` is the residual stream, `h` its LayerNorm."""
        q, k, v = torch.ops.whisperturbo.qkv(h, self.wq[i], self.bq[i], self.wk[i],
                                             self.wv[i], self.bv[i])
        a = torch.ops.whisperturbo.attn(q, k, v)
        o = a @ self.wo[i]
        x, h = torch.ops.whisperturbo.residual_ln(o, self.bo[i], x, self.ln2_g[i], self.ln2_b[i])
        f = torch.ops.whisperturbo.fc1(h, self.w1[i], self.b1[i])
        if self.fc2_mode == "cutedsl" or (self.fc2_mode == "hoid" and i == 0):
            y = torch.ops.whisperturbo.fc2_cutedsl(f, self.w2_cute[i])
        else:
            y = f @ self.w2[i]
        return x, y

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """log-mel features [128, 3000] (or [1, 128, 3000]) -> [1500, 1280].

        The functional path: every op allocates its own output, which is what
        `torch.compile` wants to see. `forward_static` is the same computation
        on preallocated buffers.
        """
        if mel.dim() == 3:
            mel = mel[0]
        x = self.frontend(mel)
        h = torch.ops.whisperturbo.layer_norm(x, self.ln1_g[0], self.ln1_b[0])
        for i in range(LAYERS):
            x, y = self.layer(i, x, h)
            if i + 1 < LAYERS:
                x, h = torch.ops.whisperturbo.residual_ln(
                    y, self.b2[i], x, self.ln1_g[i + 1], self.ln1_b[i + 1])
            else:
                x = torch.ops.whisperturbo.bias_residual(y, self.b2[i], x)
        return torch.ops.whisperturbo.layer_norm(x, self.ln_g, self.ln_b)

    # -- the static path: no allocation, no dispatch, capture-safe ------------
    def forward_static(self, mel: torch.Tensor) -> torch.Tensor:
        """Same computation on preallocated buffers, calling the kernels
        directly. This is what gets captured into a CUDA graph, and it is also
        the fastest eager path — the custom-op dispatch above costs more than
        several of these kernels put together."""
        b = self.buf
        if mel.dim() == 3:
            mel = mel[0]
        if mel.data_ptr() != b.mel.data_ptr():
            b.mel.copy_(mel)
        k_conv1(b.c1, b.mel, self.conv1_w, self.conv1_b, b.dyn)
        k_conv2(b.c2, b.im2col, b.gemm, b.c1, self.conv2_w_t, self.conv2_b)
        KM.transpose_pos_add(b.x[0], b.c2, self.pos)
        KM.row_layer_norm(b.h, b.x[0], self.ln1_g[0], self.ln1_b[0])
        cur = 0
        for i in range(LAYERS):
            CUTE["triple_qkv"](b.q, b.k, b.v, b.h, self.wq[i], self.bq[i],
                               self.wk[i], self.wv[i], self.bv[i])
            k_attn(b.attn, b.q, b.k, b.v)
            torch.matmul(b.attn, self.wo[i], out=b.o)
            KM.ordered_residual_ln(b.x[1 - cur], b.h, b.o, self.bo[i], b.x[cur],
                                   self.ln2_g[i], self.ln2_b[i])
            cur = 1 - cur
            CUTE["fc1_gelu"](b.f, b.h, self.w1[i], self.b1[i])
            if self.fc2_mode == "cutedsl" or (self.fc2_mode == "hoid" and i == 0):
                CUTE["dense_residency"](b.y, b.f, self.w2_cute[i])
            else:
                torch.matmul(b.f, self.w2[i], out=b.y)
            if i + 1 < LAYERS:
                KM.ordered_residual_ln(b.x[1 - cur], b.h, b.y, self.b2[i], b.x[cur],
                                       self.ln1_g[i + 1], self.ln1_b[i + 1])
                cur = 1 - cur
            else:
                KM.bias_residual(b.x[1 - cur], b.y, self.b2[i], b.x[cur])
                cur = 1 - cur
        KM.row_layer_norm(b.out, b.x[cur], self.ln_g, self.ln_b)
        return b.out


class CapturedEncoder:
    """The encoder replayed from a CUDA graph: static input buffer, static
    output buffer, no launches from Python at all.

    The encoder is ~3.3 ms of GPU work over ~230 kernels, so per-launch host
    cost is a first-order term here, not a rounding error — and the shapes are
    completely static, so capture is the natural way to run it. torch's own
    fastest stock config (`reduce-overhead`) is CUDA-graphed for the same
    reason. Capture uses `forward_static`, whose whole working set is allocated
    before capture begins.
    """

    def __init__(self, encoder: OptimizedWhisperEncoder, warmup: int = 3):
        self.encoder = encoder
        self.mel = encoder.buf.mel
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(warmup):
                encoder.forward_static(self.mel)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.out = encoder.forward_static(self.mel)

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.dim() == 3:
            mel = mel[0]
        self.mel.copy_(mel)
        self.graph.replay()
        return self.out
