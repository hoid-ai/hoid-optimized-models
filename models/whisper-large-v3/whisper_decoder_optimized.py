#!/usr/bin/env python
"""whisper-large-v3's DECODER step in PyTorch, running the hoid-winning kernels.

The sibling of whisper_optimized.py. Where that module runs the encoder's
forward, this one runs a single cached decode step — one token in, logits over
51866 out, against a 448-slot self-attention KV cache and the cross-attention
K/V of a 30 s window computed once.

The kernels come verbatim from the hoid decoder run's winning stack,
**1.8219 ms/step** on a B200 (torch's best static-cache compiled config on
the same box: 2.7876 ms):

  `embed_add_ln`    token + position embedding and the first LayerNorm, fused
  `qkv_raw`         one kernel producing raw q, k and v (biases applied later)
  `self_partial`    split-K flash-decode over the 448-slot window; also writes
                    the step's k/v into the cache at `pos`  (Triton, 16 splits)
  `self_reduce`     combines the 16 partials                 (Triton)
  `cross_partial`   split-K flash-decode over the 1500 encoder positions
                                                             (Triton, 32 splits)
  `cross_reduce`    combines the 32 partials                 (Triton)
  `resid_ln`        bias -> residual -> LayerNorm in one pass, emitting both
  `bias_gelu`       fc1 bias + exact-erf GELU
  `resid_final_ln`  the last residual and the final LayerNorm

Everything else is a cuBLAS GEMV, exactly as in the graph.

Unlike the encoder module there is no functional/`torch.compile` path here: a
decode step is 1.8 ms over ~300 launches, so the only sane deployment is a
captured CUDA graph, and the whole working set is allocated before capture.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile

import torch

from whisper_decoder_kernels_gen import KERNELS

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, "kernels")

DIM = 1280
HEADS = 20
HEAD_DIM = 64
FFN = 5120
LAYERS = 32
VOCAB = 51866
DEC_POS = 448          # whisper's max_target_positions — the static cache window
ENC_POS = 1500
PROMPT_LEN = 4


def _src(key: str) -> str:
    """The kernel source, hash-checked against the manifest."""
    meta = KERNELS[key]
    with open(os.path.join(KDIR, meta["file"])) as f:
        text = f.read()
    got = hashlib.sha256(text.encode()).hexdigest()
    if got != meta["sha256"]:
        raise RuntimeError(
            f"decoder kernel {key} ({meta['file']}) hashes {got[:16]}, manifest says "
            f"{meta['sha256'][:16]} — source drifted from the manifest.")
    return text


_TMP_MODULES = []


def _load_kernel_module(key: str, preamble: list[str]):
    """Triton reads a decorated function's source back with
    `inspect.getsourcelines`, so it has to be a real file."""
    tmp = tempfile.mkdtemp(prefix="whisper_dec_kernel_")
    _TMP_MODULES.append(tmp)
    path = os.path.join(tmp, f"{key}_kernel.py")
    with open(path, "w") as f:
        f.write("\n".join(preamble) + "\n\n" + _src(key))
    sys.path.insert(0, tmp)
    spec = importlib.util.spec_from_file_location(f"whisper_dec_{key}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# CUDA half: five kernels, each launched with the geometry hoid measured
# ---------------------------------------------------------------------------
_CUDA_WRAPPERS = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

using bf16 = __nv_bfloat16;
#define STREAM at::cuda::getCurrentCUDAStream().stream()
#define P(t) reinterpret_cast<bf16*>((t).data_ptr())
#define C(t) reinterpret_cast<const bf16*>((t).data_ptr())
#define CI(t) reinterpret_cast<const int*>((t).data_ptr())

// grid 1 x block 256
void embed_add_ln(torch::Tensor res, torch::Tensor norm, torch::Tensor tok_w,
                  torch::Tensor tok_id, torch::Tensor pos_w, torch::Tensor pos_id,
                  torch::Tensor gamma, torch::Tensor beta) {
    whisper_embed_add_ln_exact_k<<<1, 256, 0, STREAM>>>(
        P(res), P(norm), C(tok_w), CI(tok_id), C(pos_w), CI(pos_id),
        C(gamma), C(beta), nullptr);
}

// grid 960 x block 128 — 3840 output rows, 4 warps per block
void qkv_raw(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor x,
             torch::Tensor wq, torch::Tensor wk, torch::Tensor wv) {
    whisper_qkv_prefetch10_raw_b128_k<<<960, 128, 0, STREAM>>>(
        P(q), P(k), P(v), C(x), C(wq), C(wk), C(wv), nullptr);
}

// grid 1 x block 128
void resid_ln(torch::Tensor res_out, torch::Tensor norm_out, torch::Tensor mm,
              torch::Tensor bias, torch::Tensor residual, torch::Tensor gamma,
              torch::Tensor beta) {
    whisper_resid_ln_oneshot_vec2_k<<<1, 128, 0, STREAM>>>(
        P(res_out), P(norm_out), C(mm), C(bias), C(residual), C(gamma), C(beta), nullptr);
}

// grid 20 x block 256 — 5120 elements
void bias_gelu(torch::Tensor out, torch::Tensor mm, torch::Tensor bias) {
    whisper_bias_gelu_k<<<20, 256, 0, STREAM>>>(P(out), C(mm), C(bias), nullptr);
}

// grid 1 x block 256
void resid_final_ln(torch::Tensor norm_out, torch::Tensor mm, torch::Tensor bias,
                    torch::Tensor residual, torch::Tensor gamma, torch::Tensor beta) {
    whisper_resid_final_ln_fast_k<<<1, 256, 0, STREAM>>>(
        P(norm_out), C(mm), C(bias), C(residual), C(gamma), C(beta), nullptr);
}
"""

_CUDA_DECLS = """
void embed_add_ln(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void qkv_raw(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void resid_ln(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void bias_gelu(torch::Tensor, torch::Tensor, torch::Tensor);
void resid_final_ln(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
"""

_CUDA_KEYS = ("embed_add_ln", "qkv_raw", "resid_ln", "bias_gelu", "resid_final_ln")

KM = None      # the nvcc-built CUDA module
TK = {}        # triton kernels, by manifest key
_BUILT = False


def build_kernels(verbose: bool = True):
    global KM, TK, _BUILT
    if _BUILT:
        return
    from torch.utils.cpp_extension import load_inline

    if verbose:
        print("[build] nvcc: 5 decoder CUDA kernels ...")
    KM = load_inline(
        name="whisper_hoid_decoder_kernels",
        cpp_sources=[_CUDA_DECLS],
        cuda_sources=["\n".join(_src(k) for k in _CUDA_KEYS) + _CUDA_WRAPPERS],
        functions=["embed_add_ln", "qkv_raw", "resid_ln", "bias_gelu", "resid_final_ln"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    if verbose:
        print("[build] triton: self/cross split-K attention ...")
    for key in ("self_partial", "self_reduce", "cross_partial", "cross_reduce"):
        mod = _load_kernel_module(key, ["import triton", "import triton.language as tl"])
        TK[key] = getattr(mod, KERNELS[key]["entry"])
    _BUILT = True


# ---------------------------------------------------------------------------
class OptimizedWhisperDecoder:
    """One cached decode step on the hoid kernel stack.

    Weights are read from the HF `WhisperForConditionalGeneration`; the caches
    and every intermediate are allocated once, up front, so the step can be
    captured into a CUDA graph.
    """

    def __init__(self, model, device: str = "cuda"):
        build_kernels(verbose=False)
        dec = model.model.decoder
        self.device = device

        def w(t):
            return t.detach().contiguous()

        def wt(t):                       # pre-transposed for the M=1 GEMV
            return t.detach().t().contiguous()

        self.embed_tok = w(dec.embed_tokens.weight)
        self.embed_pos = w(dec.embed_positions.weight)
        self.lm_head_t = wt(dec.embed_tokens.weight)      # tied head, [1280, 51866]

        self.q_w, self.q_b, self.k_w, self.v_w, self.v_b = [], [], [], [], []
        self.o_wt, self.o_b = [], []
        self.cq_wt, self.cq_b, self.ck_w, self.cv_w, self.cv_b = [], [], [], [], []
        self.co_wt, self.co_b = [], []
        self.fc1_wt, self.fc1_b, self.fc2_wt, self.fc2_b = [], [], [], []
        self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b, self.ln3_g, self.ln3_b = ([] for _ in range(6))
        for layer in dec.layers:
            a, c = layer.self_attn, layer.encoder_attn
            self.q_w.append(w(a.q_proj.weight)); self.q_b.append(w(a.q_proj.bias))
            self.k_w.append(w(a.k_proj.weight))
            self.v_w.append(w(a.v_proj.weight)); self.v_b.append(w(a.v_proj.bias))
            self.o_wt.append(wt(a.out_proj.weight)); self.o_b.append(w(a.out_proj.bias))
            self.cq_wt.append(wt(c.q_proj.weight)); self.cq_b.append(w(c.q_proj.bias))
            self.ck_w.append(w(c.k_proj.weight))
            self.cv_w.append(w(c.v_proj.weight)); self.cv_b.append(w(c.v_proj.bias))
            self.co_wt.append(wt(c.out_proj.weight)); self.co_b.append(w(c.out_proj.bias))
            self.fc1_wt.append(wt(layer.fc1.weight)); self.fc1_b.append(w(layer.fc1.bias))
            self.fc2_wt.append(wt(layer.fc2.weight)); self.fc2_b.append(w(layer.fc2.bias))
            self.ln1_g.append(w(layer.self_attn_layer_norm.weight))
            self.ln1_b.append(w(layer.self_attn_layer_norm.bias))
            self.ln2_g.append(w(layer.encoder_attn_layer_norm.weight))
            self.ln2_b.append(w(layer.encoder_attn_layer_norm.bias))
            self.ln3_g.append(w(layer.final_layer_norm.weight))
            self.ln3_b.append(w(layer.final_layer_norm.bias))
        self.lnf_g = w(dec.layer_norm.weight)
        self.lnf_b = w(dec.layer_norm.bias)

        # Prefill-only packing. The single-token path has a kernel that fuses
        # q/k/v into one pass; the M=4 prefill is plain cuBLAS, where three
        # skinny GEMMs per layer are three latency-bound launches. One [3840,
        # 1280] weight makes it one. Same for the window's cross K/V, which is
        # two [1280,1280] projections of the same 1500-row input.
        self.qkv_wt = [torch.cat([self.q_w[i], self.k_w[i], self.v_w[i]]).t().contiguous()
                       for i in range(LAYERS)]
        self.qkv_b = [torch.cat([self.q_b[i], torch.zeros_like(self.q_b[i]), self.v_b[i]])
                      for i in range(LAYERS)]
        self.ckv_wt = [torch.cat([self.ck_w[i], self.cv_w[i]]).t().contiguous()
                       for i in range(LAYERS)]

        self._alloc(device)
        self._alloc_prefill()
        self.graph = None
        self.prefill_graph = None

    def _alloc(self, dev):
        def z(*shape, dtype=torch.bfloat16):
            return torch.zeros(*shape, device=dev, dtype=dtype)

        self.tok_id = torch.zeros(1, device=dev, dtype=torch.int32)
        self.pos_id = torch.zeros(1, device=dev, dtype=torch.int32)
        self.dyn = torch.zeros(1, device=dev, dtype=torch.int32)
        # caches: per layer, heads-major, the full static window
        self.kcache = [z(HEADS, DEC_POS, HEAD_DIM) for _ in range(LAYERS)]
        self.vcache = [z(HEADS, DEC_POS, HEAD_DIM) for _ in range(LAYERS)]
        # cross-attention K/V for the current 30 s window, filled by set_encoder()
        self.cross_k = [z(ENC_POS, DIM) for _ in range(LAYERS)]
        self.cross_v = [z(ENC_POS, DIM) for _ in range(LAYERS)]
        # per-step scratch
        self.res = (z(1, DIM), z(1, DIM))          # residual stream, ping-pong
        self.h = z(1, DIM)
        self.qraw, self.kraw, self.vraw = z(1, DIM), z(1, DIM), z(1, DIM)
        self.sp = torch.zeros(HEADS, KERNELS["self_partial"]["constexprs"]["SPLITS"],
                              HEAD_DIM + 2, device=dev, dtype=torch.float32)
        self.cp = torch.zeros(HEADS, KERNELS["cross_partial"]["constexprs"]["SPLITS"],
                              HEAD_DIM + 2, device=dev, dtype=torch.float32)
        self.attn, self.o = z(1, DIM), z(1, DIM)
        self.cq, self.cattn, self.co = z(1, DIM), z(1, DIM), z(1, DIM)
        self.f, self.g, self.y = z(1, FFN), z(1, FFN), z(1, DIM)
        self.hidden = z(1, DIM)
        self.logits = z(1, VOCAB)

    def _alloc_prefill(self):
        """Buffers for the 4-token prefill, which shares nothing with the
        single-token step's scratch (it is captured as its own graph)."""
        dev = self.device
        n = PROMPT_LEN
        self.pf_tokens = torch.zeros(n, device=dev, dtype=torch.long)
        self.enc_hidden = torch.zeros(ENC_POS, DIM, device=dev, dtype=torch.bfloat16)
        self.pf_x = torch.zeros(n, DIM, device=dev, dtype=torch.bfloat16)
        self.pf_logits = torch.zeros(1, VOCAB, device=dev, dtype=torch.bfloat16)

    def _prefill_static(self):
        """The forced prefix in ONE pass, plus the window's cross-attention K/V.

        The hoid kernels are shaped for a single token: running a 4-token prefix
        through them means reading all 1.6 GB of decoder weights four times, and
        that is the entire time-to-first-token gap against torch, which prefills
        the prefix in one forward. This does the same — plain cuBLAS at M=4 and
        SDPA, writing k/v straight into the hoid cache layout — so the weights
        are read once. Everything after the first token still runs the hoid
        kernels.
        """
        import torch.nn.functional as F

        n = PROMPT_LEN
        x = (self.embed_tok[self.pf_tokens] + self.embed_pos[:n])
        for i in range(LAYERS):
            # the window's cross K/V, projected once (also read by every later step)
            ckv = self.enc_hidden @ self.ckv_wt[i]
            self.cross_k[i].copy_(ckv[:, :DIM])
            self.cross_v[i].copy_(ckv[:, DIM:] + self.cv_b[i])

            h = F.layer_norm(x, (DIM,), self.ln1_g[i], self.ln1_b[i], 1e-5)
            qkv = (h @ self.qkv_wt[i] + self.qkv_b[i]).view(n, 3, HEADS, HEAD_DIM)
            q = qkv[:, 0].transpose(0, 1)
            k = qkv[:, 1].transpose(0, 1)
            v = qkv[:, 2].transpose(0, 1)
            self.kcache[i][:, :n, :].copy_(k)
            self.vcache[i][:, :n, :].copy_(v)
            a = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                                               is_causal=True)[0]
            x = x + (a.transpose(0, 1).reshape(n, DIM) @ self.o_wt[i] + self.o_b[i])

            h = F.layer_norm(x, (DIM,), self.ln2_g[i], self.ln2_b[i], 1e-5)
            cq = (h @ self.cq_wt[i] + self.cq_b[i]).view(n, HEADS, HEAD_DIM).transpose(0, 1)
            ck = self.cross_k[i].view(ENC_POS, HEADS, HEAD_DIM).transpose(0, 1)
            cv = self.cross_v[i].view(ENC_POS, HEADS, HEAD_DIM).transpose(0, 1)
            ca = F.scaled_dot_product_attention(cq.unsqueeze(0), ck.unsqueeze(0),
                                                cv.unsqueeze(0))[0]
            x = x + (ca.transpose(0, 1).reshape(n, DIM) @ self.co_wt[i] + self.co_b[i])

            h = F.layer_norm(x, (DIM,), self.ln3_g[i], self.ln3_b[i], 1e-5)
            f = F.gelu(h @ self.fc1_wt[i] + self.fc1_b[i], approximate="none")
            x = x + (f @ self.fc2_wt[i] + self.fc2_b[i])

        hidden = F.layer_norm(x[-1:], (DIM,), self.lnf_g, self.lnf_b, 1e-5)
        torch.matmul(hidden, self.lm_head_t, out=self.pf_logits)
        return self.pf_logits

    def capture_prefill(self, prompt_ids, warmup: int = 3, compile_first: bool = True):
        """Capture cross-K/V + the 4-token prefill as one graph. The prefix is
        constant, so the only per-window input is the encoder output.

        The prefill is plain torch ops at M=4, which inductor fuses well —
        compiling before capture is worth ~1.2 ms of the ~5.3 ms pass, and the
        capture then removes what is left of the launch cost. `torch.compile`
        earns its keep here and not in the single-token step precisely because
        here there is something to fuse.
        """
        self.pf_tokens.copy_(torch.tensor(prompt_ids, device=self.device))
        fn = self._prefill_static
        if compile_first:
            fn = torch.compile(self._prefill_static, mode="max-autotune-no-cudagraphs",
                               dynamic=False)
            with torch.no_grad():
                fn()                       # pay the compile outside the capture
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.prefill_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.prefill_graph), torch.no_grad():
            fn()
        return self

    def prefill(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        """One replay: the window's cross K/V and the whole forced prefix.
        Returns the logits that choose the first generated token."""
        enc = encoder_hidden[0] if encoder_hidden.dim() == 3 else encoder_hidden
        self.enc_hidden.copy_(enc)
        if self.prefill_graph is not None:
            self.prefill_graph.replay()
        else:
            with torch.no_grad():
                self._prefill_static()
        return self.pf_logits

    # -- per-window setup ---------------------------------------------------
    def set_encoder(self, encoder_hidden: torch.Tensor):
        """Compute the 32 layers' cross-attention K/V for a 30 s window.

        Once per window, not per step — this is what the decode graph takes as
        an input, and its cost is outside the step metric (as it is for the
        torch baseline, which caches the same thing)."""
        enc = encoder_hidden.to(self.device, torch.bfloat16)
        if enc.dim() == 3:
            enc = enc[0]
        with torch.no_grad():
            for i in range(LAYERS):
                torch.matmul(enc, self.ck_w[i].t(), out=self.cross_k[i])
                torch.matmul(enc, self.cv_w[i].t(), out=self.cross_v[i])
                self.cross_v[i].add_(self.cv_b[i])

    def reset_cache(self):
        for k, v in zip(self.kcache, self.vcache):
            k.zero_()
            v.zero_()

    # -- one step, on static buffers ---------------------------------------
    def step_static(self):
        """Run one decode step from `tok_id`/`pos_id` into `logits`."""
        sp_meta, cp_meta = KERNELS["self_partial"], KERNELS["cross_partial"]
        KM.embed_add_ln(self.res[0], self.h, self.embed_tok, self.tok_id,
                        self.embed_pos, self.pos_id, self.ln1_g[0], self.ln1_b[0])
        cur = 0
        for i in range(LAYERS):
            KM.qkv_raw(self.qraw, self.kraw, self.vraw, self.h,
                       self.q_w[i], self.k_w[i], self.v_w[i])
            TK["self_partial"][(320,)](
                self.sp, self.qraw, self.q_b[i], self.kraw, self.vraw, self.v_b[i],
                self.pos_id, self.kcache[i], self.vcache[i], self.dyn,
                BLOCK_N=sp_meta["constexprs"]["BLOCK_N"], CHUNK=sp_meta["constexprs"]["CHUNK"],
                HEAD_DIM=HEAD_DIM, SPLITS=sp_meta["constexprs"]["SPLITS"],
                num_warps=sp_meta["num_warps"], num_stages=sp_meta["num_stages"])
            TK["self_reduce"][(HEADS,)](
                self.attn, self.sp, self.dyn, HEAD_DIM=HEAD_DIM,
                SPLITS=sp_meta["constexprs"]["SPLITS"],
                num_warps=KERNELS["self_reduce"]["num_warps"],
                num_stages=KERNELS["self_reduce"]["num_stages"])
            torch.matmul(self.attn, self.o_wt[i], out=self.o)
            KM.resid_ln(self.res[1 - cur], self.h, self.o, self.o_b[i], self.res[cur],
                        self.ln2_g[i], self.ln2_b[i])
            cur = 1 - cur

            torch.matmul(self.h, self.cq_wt[i], out=self.cq)
            TK["cross_partial"][(640,)](
                self.cp, self.cq, self.cq_b[i], self.cross_k[i], self.cross_v[i], self.dyn,
                BLOCK_N=cp_meta["constexprs"]["BLOCK_N"], CHUNK=cp_meta["constexprs"]["CHUNK"],
                HEAD_DIM=HEAD_DIM, SPLITS=cp_meta["constexprs"]["SPLITS"],
                num_warps=cp_meta["num_warps"], num_stages=cp_meta["num_stages"])
            TK["cross_reduce"][(HEADS,)](
                self.cattn, self.cp, self.dyn, HEAD_DIM=HEAD_DIM,
                SPLITS=cp_meta["constexprs"]["SPLITS"],
                num_warps=KERNELS["cross_reduce"]["num_warps"],
                num_stages=KERNELS["cross_reduce"]["num_stages"])
            torch.matmul(self.cattn, self.co_wt[i], out=self.co)
            KM.resid_ln(self.res[1 - cur], self.h, self.co, self.co_b[i], self.res[cur],
                        self.ln3_g[i], self.ln3_b[i])
            cur = 1 - cur

            torch.matmul(self.h, self.fc1_wt[i], out=self.f)
            KM.bias_gelu(self.g, self.f, self.fc1_b[i])
            torch.matmul(self.g, self.fc2_wt[i], out=self.y)
            if i + 1 < LAYERS:
                KM.resid_ln(self.res[1 - cur], self.h, self.y, self.fc2_b[i], self.res[cur],
                            self.ln1_g[i + 1], self.ln1_b[i + 1])
                cur = 1 - cur
            else:
                KM.resid_final_ln(self.hidden, self.y, self.fc2_b[i], self.res[cur],
                                  self.lnf_g, self.lnf_b)
        torch.matmul(self.hidden, self.lm_head_t, out=self.logits)
        return self.logits

    # -- CUDA graph ---------------------------------------------------------
    def capture(self, warmup: int = 3):
        """Capture one step. The cache write is data-driven through `pos_id`,
        not shape-driven, so a single captured graph serves every position."""
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
        self.pos_id.fill_(int(pos))
        if self.graph is not None:
            self.graph.replay()
        else:
            with torch.no_grad():
                self.step_static()
        return self.logits
