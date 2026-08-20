"""Shared harness for the Llama-3.1-8B torch port: model/prompt loading, per-phase
CUDA-event timing, and the report protocol both best_*.py scripts print.

The measurement protocol: prefill and
decode measured separately, never blended; coherent English prompt (tiled) so
greedy argmax is not flipping on numeric noise; static KV cache; p50 over
repetitions after warmup.
"""

from __future__ import annotations

import json
import os
import statistics
import time

import torch

DEFAULT_WEIGHTS = "NousResearch/Meta-Llama-3.1-8B-Instruct"


def resolve_weights(weights: str) -> str:
    """A local safetensors dir is used as-is; anything else is an HF hub
    repo id, downloaded to the hub cache on first use."""
    if os.path.isdir(weights):
        return weights
    from huggingface_hub import snapshot_download

    return snapshot_download(
        weights, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])

PARAGRAPH = (
    "The design of a fast inference engine begins with the memory system, "
    "because a single decode step at batch one streams every weight exactly "
    "once and does almost nothing else. The arithmetic is trivial; the "
    "bandwidth is not. A prefill over a long prompt inverts the balance: the "
    "same weights are reused across thousands of tokens, the GEMMs grow "
    "compute-bound, and the attention score matrix becomes the object whose "
    "handling separates a good engine from a naive one. "
)


def build_prompt_ids(tokenizer, n_tokens: int) -> torch.Tensor:
    ids = tokenizer(PARAGRAPH, return_tensors="pt").input_ids[0]
    reps = n_tokens // ids.numel() + 1
    return ids.repeat(reps)[:n_tokens]


def prompt_preview(tokenizer, ids, n_chars: int = 200) -> str:
    """The decoded head of the (tiled) prompt, for display."""
    return tokenizer.decode(ids[:64]).strip()[:n_chars] + " …"


def load_tokenizer(weights_dir: str = DEFAULT_WEIGHTS):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(weights_dir)


def load_hf_model(weights_dir: str = DEFAULT_WEIGHTS, device: str = "cuda"):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        weights_dir, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    m.eval()
    return m


def cuda_time(fn, reps: int, warmup: int) -> list[float]:
    """Per-call latencies in ms via CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b))
    return out


def p50(xs: list[float]) -> float:
    return statistics.median(xs)


def strict_matmul_precision() -> None:
    """Match the cuBLAS discipline hoid tuned under: full-fp32 K-reductions for bf16, and
    no TF32 surprises anywhere a float32 op sneaks in."""
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = True  # torch default for f32 GEMM
    torch.backends.cudnn.allow_tf32 = True


def gpu_banner() -> str:
    p = torch.cuda.get_device_properties(0)
    return (f"{p.name}, {p.total_memory / 2**30:.0f} GiB, "
            f"torch {torch.__version__}, sm{p.major}{p.minor}")


def report(tag: str, prompt_tokens: int, gen_tokens: int, m: dict) -> None:
    print(f"\n== {tag}: {prompt_tokens}-token prompt, {gen_tokens} greedy tokens, batch 1, bf16 ==")
    print(f"prefill (TTFT, model only)      {m['prefill_p50_ms']:9.2f} ms")
    print(f"ms per decode token (p50)       {m['decode_p50_ms']:9.3f} ms")
    print(f"decode throughput               {1000.0 / m['decode_p50_ms']:9.1f} tok/s")
    print(f"e2e generate (prefill + decode) {m['e2e_ms']:9.1f} ms")


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {path}")


def wall_ms(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3
