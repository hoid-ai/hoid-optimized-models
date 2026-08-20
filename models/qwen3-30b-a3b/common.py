#!/usr/bin/env python
"""Shared pieces of the Qwen3-30B-A3B torch benchmark: model load, the
prompt protocol, and CUDA-event timing.

The workload: batch 1, a 1024-token prompt,
greedy decode — and the prompt is the same coherent passage tiled to length, so
greedy decode is confident and a correctness gate measures real divergence, not
argmax noise on a flat distribution.
"""

from __future__ import annotations

import time

import torch

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PROMPT_TOKENS = 1024
DECODE_WARMUP = 8
MEASURED_DECODE_STEPS = 128
CORRECTNESS_TOKENS = 32
CORRECTNESS_MIN_RATIO = 0.95

# The same passage both harnesses tile, so the two measure the identical
# workload.
PROMPT_TEXT = (
    "The history of computing hardware spans a remarkable arc from mechanical "
    "calculators to the massively parallel accelerators that train modern neural "
    "networks. Early machines filled entire rooms and could perform only a few "
    "operations per second, yet they established the foundational ideas of stored "
    "programs, addressable memory, and sequential control flow. As transistors "
    "replaced vacuum tubes and integrated circuits packed ever more logic onto a "
    "single die, performance grew exponentially for decades. Today the frontier "
    "of performance lies in specialized processors that exploit data parallelism, "
    "high memory bandwidth, and carefully tuned software compilers to keep "
    "thousands of arithmetic units busy. Understanding where time is actually "
    "spent, and why, remains the central discipline of high-performance "
    "engineering, whether the workload is a physics simulation, a database join, "
    "or the autoregressive decoding of a large language model one token at a time. "
)


def load_model(device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_float32_matmul_precision("high")
    # device_map streams shards straight to the GPU instead of staging 61 GB on host.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map=device,
    ).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    return model, tok


def build_prompt(tok, device: str = "cuda",
                 n_tokens: int = PROMPT_TOKENS) -> torch.Tensor:
    ids = tok(PROMPT_TEXT, return_tensors="pt").input_ids[0]
    if ids.numel() < n_tokens:
        reps = (n_tokens // ids.numel()) + 1
        ids = ids.repeat(reps)
    return ids[:n_tokens].unsqueeze(0).to(device)


def results_name(base: str, n_tokens: int) -> str:
    """Result-file name for a prompt length: the default 1024 keeps the bare
    name, other lengths get an explicit suffix (best_torch_p8192.json)."""
    return f"{base}.json" if n_tokens == PROMPT_TOKENS else f"{base}_p{n_tokens}.json"


def static_cache(model, max_len: int, device, dtype=torch.bfloat16):
    from transformers import StaticCache

    cache = StaticCache(config=model.config, max_cache_len=max_len,
                        device=device, dtype=dtype)
    # Allocate outside any compiled region: lazy init under dynamo (we prefill
    # compiled) skips mark_static_address, which the cudagraph configs require.
    cache.early_initialization(1, model.config.num_key_value_heads,
                               model.config.head_dim, dtype,
                               torch.device(device))
    return cache


def prefill_once(forward, input_ids, cache, device):
    seq = input_ids.shape[1]
    out = forward(
        input_ids=input_ids, use_cache=True, past_key_values=cache,
        cache_position=torch.arange(seq, device=device),
    )
    return out.logits[:, -1:, :].argmax(-1), seq


def greedy_static(forward, model, input_ids, n_new: int, device) -> list[int]:
    """The correctness reference: manual greedy over a StaticCache."""
    with torch.inference_mode():
        cache = static_cache(model, input_ids.shape[1] + n_new + 1, device)
        cur, past = prefill_once(forward, input_ids, cache, device)
        gen = [int(cur.item())]
        for _ in range(n_new - 1):
            out = forward(
                input_ids=cur, use_cache=True, past_key_values=cache,
                cache_position=torch.tensor([past], device=device),
            )
            cur = out.logits[:, -1:, :].argmax(-1)
            gen.append(int(cur.item()))
            past += 1
    return gen


def greedy_match(ref: list[int], got: list[int]) -> tuple[bool, float, int | None]:
    n = min(len(ref), len(got))
    matches = sum(1 for a, b in zip(ref[:n], got[:n]) if a == b)
    ratio = matches / len(ref) if ref else 1.0
    first = next((i for i in range(n) if ref[i] != got[i]), None)
    return ratio >= CORRECTNESS_MIN_RATIO, ratio, first


def percentile(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


def sync_ms(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def cuda_event_steps(step, n_steps: int) -> list[float]:
    """Per-step latencies (ms) via CUDA events, one sync per step — the same
    shape as the baseline's decode-timing loop."""
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(n_steps):
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times
