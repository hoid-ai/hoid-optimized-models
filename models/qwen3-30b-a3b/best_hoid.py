#!/usr/bin/env python
"""Qwen3-30B-A3B-Instruct-2507 on the hoid kernel stack, under PyTorch.

Prefill runs the port's own stack at every prompt length: the elastic
prepare_qkv kernel writing the graph's cache layout directly, the
shape-generic triton FA2 over the caches, cuBLAS projections, and the MoE as
two grouped GEMMs over expert-sorted (token, expert) pairs — the whole
forward captured as one CUDA graph per prompt length. Every subsequent token
is one replay of the captured decode graph over the winning kernels.

Correctness first: 32 greedy tokens on the hoid path must match the eager
StaticCache reference at >= 0.95 before any timing runs.
A faster decoder that says something else is not a result.

    uv run best_hoid.py [--prompt-tokens 8192]

Reads best_torch.json (written by best_torch.py) for the comparison table.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

import common
from common import (CORRECTNESS_TOKENS, DECODE_WARMUP, MEASURED_DECODE_STEPS,
                    build_prompt, cuda_event_steps, greedy_match, greedy_static,
                    load_model, percentile, results_name, sync_ms)
from qwen3_optimized import MAX_SEQ, OptimizedQwen3MoeDecoder

HERE = os.path.dirname(os.path.abspath(__file__))


def hoid_greedy(opt, input_ids, n_new: int) -> list[int]:
    """Prefill on the port stack, then n_new greedy tokens on the hoid step."""
    opt.reset_cache()
    logits = opt.prefill(input_ids)
    cur, past = logits.argmax(), input_ids.shape[1]
    gen = [int(cur.item())]
    tok = gen[0]
    for i in range(n_new - 1):
        logits = opt.step(tok, past + i)
        tok = int(logits.argmax().item())
        gen.append(tok)
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=common.PROMPT_TOKENS,
                    help="prompt length; beyond the graph's 2048 window the "
                         "kernels rebuild with a larger cache window")
    args = ap.parse_args()
    device = "cuda"

    # Mirror the runtime's strict-accumulation cuBLAS dispatch: the graph's
    # bf16 matmuls run with reduced-precision reductions disallowed.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    print(f"loading {common.MODEL} ...")
    model, tok = load_model(device)
    input_ids = build_prompt(tok, device, args.prompt_tokens)
    n_prompt = input_ids.shape[1]
    needed = n_prompt + DECODE_WARMUP + MEASURED_DECODE_STEPS + 2
    window = MAX_SEQ if needed <= MAX_SEQ else -(-needed // 256) * 256

    print("eager reference: 32 greedy tokens on the stock StaticCache path ...")
    ref = greedy_static(model.forward, model, input_ids, CORRECTNESS_TOKENS, device)

    print(f"building the hoid decoder (expert repack + kernel build + capture, "
          f"cache window {window}) ...")
    opt = OptimizedQwen3MoeDecoder(model, device, cache_window=window)
    opt.capture()

    # ---- the gate, before any number is printed ---------------------------
    got = hoid_greedy(opt, input_ids, CORRECTNESS_TOKENS)
    ok, ratio, first = greedy_match(ref, got)
    print(f"correctness: {ratio:.3f}"
          + ("" if first is None else f" (first divergence at {first})"))
    if not ok:
        raise SystemExit(f"hoid path failed the greedy-token gate ({ratio:.3f} < 0.95); "
                         "no timing will be reported")
    print("  " + repr(tok.decode(got)))

    # ---- prefill / TTFT ---------------------------------------------------
    # The port prefill rewrites cache rows 0..S-1 in place every call.
    def one_prefill():
        opt.prefill(input_ids)

    for _ in range(3):
        one_prefill()
    prefill_ms = [sync_ms(one_prefill) for _ in range(10)]
    cur = opt.prefill(input_ids).argmax()
    past = n_prompt

    # ---- decode: warmup to steady state, then the 128-step timing window --
    tok_id = int(cur.item())
    pos = past
    state = {"tok": tok_id, "pos": pos}

    def step():
        logits = opt.step(state["tok"], state["pos"])
        state["tok"] = int(logits.argmax().item())
        state["pos"] += 1

    for _ in range(DECODE_WARMUP):
        step()
    torch.cuda.synchronize()
    decode_ms = cuda_event_steps(step, MEASURED_DECODE_STEPS)

    res = {
        "prompt_tokens": n_prompt,
        "cache_window": window,
        "prefill_p50_ms": percentile(prefill_ms, 0.5),
        "decode_p50_ms": percentile(decode_ms, 0.5),
        "decode_p95_ms": percentile(decode_ms, 0.95),
        "decode_tok_s": 1000.0 / percentile(decode_ms, 0.5),
        "correctness_ratio": ratio,
    }
    with open(os.path.join(HERE, results_name("best_hoid", n_prompt)), "w") as f:
        json.dump(res, f, indent=2)

    # ---- the table --------------------------------------------------------
    base = None
    base_path = os.path.join(HERE, results_name("best_torch", n_prompt))
    if os.path.exists(base_path):
        with open(base_path) as f:
            base = json.load(f)["best"]

    print(f"\nQwen3-30B-A3B decode — {torch.cuda.get_device_name(0)}, "
          f"torch {torch.__version__}, bf16, prompt {n_prompt}")
    if base:
        print(f"{'':24}{'tuned torch':>14}{'hoid':>14}{'':>9}")
        r = base["decode_p50_ms"] / res["decode_p50_ms"]
        print(f"{'decode step p50 (ms)':24}{base['decode_p50_ms']:>14.4f}"
              f"{res['decode_p50_ms']:>14.4f}{r:>8.2f}x")
        r = res["decode_tok_s"] / base["decode_tok_s"]
        print(f"{'decode tok/s':24}{base['decode_tok_s']:>14.1f}"
              f"{res['decode_tok_s']:>14.1f}{r:>8.2f}x")
        r = base["prefill_p50_ms"] / res["prefill_p50_ms"]
        print(f"{'prefill p50 (ms)':24}{base['prefill_p50_ms']:>14.2f}"
              f"{res['prefill_p50_ms']:>14.2f}{r:>8.2f}x")
        print(f"\n(torch column: {base['config']} from "
              f"{results_name('best_torch', n_prompt)})")
    else:
        print(f"decode step p50  {res['decode_p50_ms']:.4f} ms "
              f"({res['decode_tok_s']:.1f} tok/s)")
        print(f"prefill p50      {res['prefill_p50_ms']:.2f} ms")
        print("(run best_torch.py to fill the comparison column)")


if __name__ == "__main__":
    main()
