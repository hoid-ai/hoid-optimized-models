#!/usr/bin/env python
"""bge-m3's dense-embedding forward in PyTorch, running the hoid-winning kernels.

Every kernel here is lifted verbatim from the winning hoid stack and
driven from torch:

  - `fa2`               the graph's flash-attention-2 fuse, shipped as the PTX
                        artifact hoid promoted (Triton-compiled at
                        BM128/BN32/w4/stages4, 154 regs/thread): q/k/v bias
                        folded around the tiles, f32 online softmax — the
                        [64,16,512,512] score matrix never exists in HBM.
                        Loaded with the CUDA driver API and launched with the
                        manifest's geometry; its 48 KiB dynamic smem needs the
                        opt-in attribute.
  - `bias_residual_ln`  warp-per-row bias -> residual -> LayerNorm with packed
                        BF16x2 adds (48 sites)
  - `bias_gelu_lut`     bias + exact bf16-domain GELU via a 65536-entry table
                        baked into the source — bit-exact to erf by construction
  - `embedding_add_ln`  the whole embedding block in one warp-per-row kernel:
                        word gather + static RoBERTa positions (offset +2) +
                        token type 0 + LayerNorm
  - `pool`              CLS select + L2 normalize to fp32
  - all 6 projection GEMMs per layer stay cuBLAS, as in the winning graph
    (strict f32 K-reduction: reduced-precision reduction is disabled to match
    the discipline hoid tuned under; the campaign measured fast_accum neutral)

The CUDA kernels are nvcc-JIT'd at import with the launch geometry the manifest
records. No hoid runtime is involved at any point.

Three ways to run the same kernels:

  `forward_ops`     every kernel wrapped as a torch custom op that allocates
                    its output — the form `torch.compile` can trace.
  `forward_static`  the same calls on buffers allocated once up front. The
                    fast eager path and the only capture-safe one: a tensor
                    allocated *inside* a CUDA-graph capture and freed
                    afterwards is not protected from later allocations.
  `capture()`       `forward_static` captured into a CUDA graph; returns a
                    replay callable.
"""

from __future__ import annotations

import ctypes
import hashlib
import os

import torch

from bge_kernels_gen import KERNELS

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, "kernels")

# The static workload the kernels were tuned for.
BATCH = 64
SEQ = 512
R = BATCH * SEQ           # 32768 rows
DIM = 1024
HEADS = 16
HEAD_DIM = 64
FFN = 4096
LAYERS = 24


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


def _grid(key: str) -> tuple[int, int, int]:
    g = KERNELS[key]["launch"]["grid"]
    return tuple(int(v) for v in g) + (1,) * (3 - len(g))


def _block(key: str) -> tuple[int, int, int]:
    b = KERNELS[key]["launch"].get("block", ["256"])
    return tuple(int(v) for v in b) + (1,) * (3 - len(b))


# ---------------------------------------------------------------------------
# CUDA kernels (verbatim) + the launch geometry from the manifest. Every entry
# point writes into caller-provided buffers (capture safety, see module doc).
# ---------------------------------------------------------------------------
_CUDA_WRAPPER_TMPL = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

using bf16 = __nv_bfloat16;
#define STREAM at::cuda::getCurrentCUDAStream().stream()
#define P(t) reinterpret_cast<bf16*>((t).data_ptr())
#define C(t) reinterpret_cast<const bf16*>((t).data_ptr())

void embedding_add_ln(torch::Tensor out, torch::Tensor word, torch::Tensor ids,
                      torch::Tensor pos, torch::Tensor tok,
                      torch::Tensor w, torch::Tensor b) {{
    bge_embedding_add_ln_warp_k<<<{g_emb}, {b_emb}, 0, STREAM>>>(
        P(out), C(word), reinterpret_cast<const int*>(ids.data_ptr()),
        C(pos), C(tok), C(w), C(b), nullptr);
}}

void bias_residual_ln(torch::Tensor out, torch::Tensor mm, torch::Tensor mm_bias,
                      torch::Tensor residual, torch::Tensor w, torch::Tensor b) {{
    bge_bias_residual_ln_warp_packed_k<<<{g_brl}, {b_brl}, 0, STREAM>>>(
        P(out), C(mm), C(mm_bias), C(residual), C(w), C(b), nullptr);
}}

void bias_gelu_lut(torch::Tensor out, torch::Tensor mm, torch::Tensor bias) {{
    bge_bias_gelu_bf16_lut_direct_k<<<{g_gelu}, {b_gelu}, 0, STREAM>>>(
        P(out), C(mm), C(bias), nullptr);
}}

void pool(torch::Tensor out, torch::Tensor x) {{
    bge_pool_k<<<{g_pool}, {b_pool}, 0, STREAM>>>(
        reinterpret_cast<float*>(out.data_ptr()), C(x), nullptr);
}}
"""

_CUDA_DECLS = """
void embedding_add_ln(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void bias_residual_ln(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void bias_gelu_lut(torch::Tensor, torch::Tensor, torch::Tensor);
void pool(torch::Tensor, torch::Tensor);
"""

_CUDA_KEYS = ("embedding_add_ln", "bias_residual_ln", "bias_gelu_lut", "pool")


def _build_cuda_module(verbose: bool = False):
    from torch.utils.cpp_extension import load_inline

    wrappers = _CUDA_WRAPPER_TMPL.format(
        g_emb=_grid("embedding_add_ln")[0], b_emb=_block("embedding_add_ln")[0],
        g_brl=_grid("bias_residual_ln")[0], b_brl=_block("bias_residual_ln")[0],
        g_gelu=_grid("bias_gelu_lut")[0], b_gelu=_block("bias_gelu_lut")[0],
        g_pool=_grid("pool")[0], b_pool=_block("pool")[0],
    )
    body = "\n".join(_src(k) for k in _CUDA_KEYS)
    return load_inline(
        name="bge_hoid_kernels",
        cpp_sources=[_CUDA_DECLS],
        cuda_sources=[body + wrappers],
        functions=["embedding_add_ln", "bias_residual_ln", "bias_gelu_lut", "pool"],
        extra_cuda_cflags=["-O3"],
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# The FA2 PTX artifact, via the CUDA driver API. The port ships the exact
# compiled form hoid promoted, so there is nothing to recompile or re-tune —
# load, opt into its dynamic smem, launch with the manifest geometry. The
# trailing scratch param is the triton>=3.3 ABI; the manifest records its
# global_size as 0, so a null pointer satisfies it.
# ---------------------------------------------------------------------------
KM = None            # the nvcc-built CUDA module
_FA2_FN = None       # CUfunction of the PTX artifact
_FA2_MOD = None      # keeps the CUmodule alive
_FA2_SMEM = 0
_DYN = None          # dummy dyn_dims buffer (the graph runs fully static)


def _cu_check(err, what):
    from cuda.bindings import driver as cu

    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what}: {err}")


def _load_ptx_fa2():
    from cuda.bindings import driver as cu

    meta = KERNELS["fa2"]
    assert meta["format"] == "ptx", meta["format"]
    assert int(meta["scratch"]["global_size"]) == 0, meta["scratch"]
    assert int(meta["scratch"]["profile_size"]) == 0, meta["scratch"]
    err, mod = cu.cuModuleLoadData(_src("fa2").encode())
    _cu_check(err, "cuModuleLoadData(fa2)")
    err, fn = cu.cuModuleGetFunction(mod, meta["entry"].encode())
    _cu_check(err, "cuModuleGetFunction(fa2)")
    smem = int(meta["launch"]["shared_mem"])
    if smem > 48 * 1024:
        (err,) = cu.cuFuncSetAttribute(
            fn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem)
        _cu_check(err, "cuFuncSetAttribute(fa2 smem opt-in)")
    return mod, fn, smem


def _fa2_launch(out, q, qb, k, kb, v, vb):
    from cuda.bindings import driver as cu

    gx, gy, gz = _grid("fa2")
    bx, by, bz = _block("fa2")
    # 10 params: the flat (out, in..., dyn_dims) ABI plus the triton>=3.4
    # trailing global- and profile-scratch pointers, both sized 0 here.
    ptrs = (out.data_ptr(), q.data_ptr(), qb.data_ptr(), k.data_ptr(),
            kb.data_ptr(), v.data_ptr(), vb.data_ptr(), _DYN.data_ptr(), 0, 0)
    (err,) = cu.cuLaunchKernel(
        _FA2_FN, gx, gy, gz, bx, by, bz, _FA2_SMEM,
        cu.CUstream(torch.cuda.current_stream().cuda_stream),
        (ptrs, (ctypes.c_void_p,) * len(ptrs)), 0)
    _cu_check(err, "cuLaunchKernel(fa2)")


def register_ops(verbose: bool = True):
    global KM, _FA2_MOD, _FA2_FN, _FA2_SMEM, _DYN
    if KM is not None:
        return
    torch.zeros(1, device="cuda")  # materialize the context the driver calls use
    if verbose:
        print("[build] nvcc: 4 CUDA kernels (first build takes ~1 min; cached after)...")
    KM = _build_cuda_module()
    _FA2_MOD, _FA2_FN, _FA2_SMEM = _load_ptx_fa2()
    _DYN = torch.zeros(26, dtype=torch.int32, device="cuda")

    @torch.library.custom_op("bge_hoid::embedding_add_ln", mutates_args=())
    def op_embedding_add_ln(ids: torch.Tensor, word: torch.Tensor, pos: torch.Tensor,
                            tok: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = torch.empty(ids.numel(), DIM, dtype=torch.bfloat16, device=ids.device)
        KM.embedding_add_ln(out, word, ids, pos, tok, w, b)
        return out

    @op_embedding_add_ln.register_fake
    def _(ids, word, pos, tok, w, b):
        return torch.empty(ids.numel(), DIM, dtype=torch.bfloat16, device=ids.device)

    @torch.library.custom_op("bge_hoid::fa2", mutates_args=())
    def op_fa2(q: torch.Tensor, qb: torch.Tensor, k: torch.Tensor, kb: torch.Tensor,
               v: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(q)
        _fa2_launch(out, q, qb, k, kb, v, vb)
        return out

    @op_fa2.register_fake
    def _(q, qb, k, kb, v, vb):
        return torch.empty_like(q)

    @torch.library.custom_op("bge_hoid::bias_residual_ln", mutates_args=())
    def op_bias_residual_ln(mm: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor,
                            w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(mm)
        KM.bias_residual_ln(out, mm, bias, residual, w, b)
        return out

    @op_bias_residual_ln.register_fake
    def _(mm, bias, residual, w, b):
        return torch.empty_like(mm)

    @torch.library.custom_op("bge_hoid::bias_gelu_lut", mutates_args=())
    def op_bias_gelu_lut(mm: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(mm)
        KM.bias_gelu_lut(out, mm, bias)
        return out

    @op_bias_gelu_lut.register_fake
    def _(mm, bias):
        return torch.empty_like(mm)

    @torch.library.custom_op("bge_hoid::pool", mutates_args=())
    def op_pool(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty(BATCH, DIM, dtype=torch.float32, device=x.device)
        KM.pool(out, x)
        return out

    @op_pool.register_fake
    def _(x):
        return torch.empty(BATCH, DIM, dtype=torch.float32, device=x.device)


class OptimizedBgeM3:
    """The dense forward on the winning kernels. Weights come from the stock
    HF model; every projection weight is pre-transposed once so the hot loop is
    `mm(x, w_t, out=...)` — the same cuBLAS GEMM the graph runs."""

    def __init__(self, stock, verbose: bool = True):
        register_ops(verbose)
        # The matmuls keep hoid's strict discipline: full-f32 K-reduction.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        def t(x):
            return x.detach().to("cuda", torch.bfloat16).contiguous()

        e = stock.embeddings
        self.word_emb = t(e.word_embeddings.weight)
        self.pos_emb = t(e.position_embeddings.weight)
        self.tok_type = t(e.token_type_embeddings.weight)
        self.emb_ln = (t(e.LayerNorm.weight), t(e.LayerNorm.bias))

        self.layers = []
        for lyr in stock.encoder.layer:
            a, o = lyr.attention.self, lyr.attention.output
            self.layers.append({
                "wq_t": t(a.query.weight).T.contiguous(), "qb": t(a.query.bias),
                "wk_t": t(a.key.weight).T.contiguous(), "kb": t(a.key.bias),
                "wv_t": t(a.value.weight).T.contiguous(), "vb": t(a.value.bias),
                "wo_t": t(o.dense.weight).T.contiguous(), "ob": t(o.dense.bias),
                "ln1": (t(o.LayerNorm.weight), t(o.LayerNorm.bias)),
                "w1_t": t(lyr.intermediate.dense.weight).T.contiguous(),
                "b1": t(lyr.intermediate.dense.bias),
                "w2_t": t(lyr.output.dense.weight).T.contiguous(),
                "b2": t(lyr.output.dense.bias),
                "ln2": (t(lyr.output.LayerNorm.weight), t(lyr.output.LayerNorm.bias)),
            })

        opt = dict(dtype=torch.bfloat16, device="cuda")
        self.ids = torch.zeros(R, dtype=torch.int32, device="cuda")
        self.h = torch.empty(R, DIM, **opt)
        self.h2 = torch.empty(R, DIM, **opt)
        self.qmm = torch.empty(R, DIM, **opt)
        self.kmm = torch.empty(R, DIM, **opt)
        self.vmm = torch.empty(R, DIM, **opt)
        self.ctx = torch.empty(R, DIM, **opt)
        self.omm = torch.empty(R, DIM, **opt)
        self.upmm = torch.empty(R, FFN, **opt)
        self.act = torch.empty(R, FFN, **opt)
        self.pooled = torch.empty(BATCH, DIM, dtype=torch.float32, device="cuda")

    # -- the static-buffer path (eager fast path; the capture body) ----------

    def load_inputs(self, input_ids: torch.Tensor):
        self.ids.copy_(input_ids.reshape(-1))

    def forward_static(self):
        KM.embedding_add_ln(self.h, self.word_emb, self.ids, self.pos_emb,
                            self.tok_type, *self.emb_ln)
        h, h2 = self.h, self.h2
        for L in self.layers:
            torch.mm(h, L["wq_t"], out=self.qmm)
            torch.mm(h, L["wk_t"], out=self.kmm)
            torch.mm(h, L["wv_t"], out=self.vmm)
            _fa2_launch(self.ctx, self.qmm, L["qb"], self.kmm, L["kb"],
                        self.vmm, L["vb"])
            torch.mm(self.ctx, L["wo_t"], out=self.omm)
            KM.bias_residual_ln(h2, self.omm, L["ob"], h, *L["ln1"])
            torch.mm(h2, L["w1_t"], out=self.upmm)
            KM.bias_gelu_lut(self.act, self.upmm, L["b1"])
            torch.mm(self.act, L["w2_t"], out=self.omm)
            KM.bias_residual_ln(h, self.omm, L["b2"], h2, *L["ln2"])
        KM.pool(self.pooled, h)
        return self.pooled

    def last_hidden(self) -> torch.Tensor:
        """The final hidden state ([64,512,1024]) left in the ping-pong buffer
        by forward_static (24 layers = even number of swaps -> self.h)."""
        return self.h.view(BATCH, SEQ, DIM)

    # -- CUDA-graph capture --------------------------------------------------

    def capture(self):
        """Capture forward_static; returns replay(input_ids) -> pooled."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.forward_static()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.forward_static()

        def replay(input_ids: torch.Tensor) -> torch.Tensor:
            self.load_inputs(input_ids)
            graph.replay()
            return self.pooled

        return replay

    # -- the custom-op path (torch.compile-traceable) ------------------------

    def forward_ops(self, input_ids: torch.Tensor) -> torch.Tensor:
        ops = torch.ops.bge_hoid
        ids = input_ids.reshape(-1).to(torch.int32)
        h = ops.embedding_add_ln(ids, self.word_emb, self.pos_emb,
                                 self.tok_type, *self.emb_ln)
        for L in self.layers:
            ctx = ops.fa2(h @ L["wq_t"], L["qb"], h @ L["wk_t"], L["kb"],
                          h @ L["wv_t"], L["vb"])
            h2 = ops.bias_residual_ln(ctx @ L["wo_t"], L["ob"], h, *L["ln1"])
            act = ops.bias_gelu_lut(h2 @ L["w1_t"], L["b1"])
            h = ops.bias_residual_ln(act @ L["w2_t"], L["b2"], h2, *L["ln2"])
        return ops.pool(h)
