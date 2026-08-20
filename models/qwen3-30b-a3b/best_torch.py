#!/usr/bin/env python
"""The tuned plain-PyTorch baseline for Qwen3-30B-A3B-Instruct-2507 decode.

The protocol: a 1024-token prompt
into a StaticCache, then a manual greedy single-token loop, prefill and decode
timed separately with CUDA events. Two configs:

  out-of-box   torch.compile default
  tuned        the requested --mode, default max-autotune + fullgraph (C4,
               cudagraphs — viable because common.static_cache
               early-initializes the cache; a cudagraph failure still
               retries as max-autotune-no-cudagraphs)

Correctness: 32 greedy tokens on the compiled path must match the eager
StaticCache reference at >= 0.95 before a number is reported.

    uv run best_torch.py                 # tuned (max-autotune)
    uv run best_torch.py --config c2     # the out-of-box row
    uv run best_torch.py --config eager  # no compile

Writes best_torch.json (winning config + metrics) for best_hoid.py's table.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

import common
from common import (CORRECTNESS_TOKENS, DECODE_WARMUP, MEASURED_DECODE_STEPS,
                    build_prompt, cuda_event_steps, greedy_match, greedy_static,
                    load_model, percentile, prefill_once, results_name,
                    static_cache, sync_ms)

HERE = os.path.dirname(os.path.abspath(__file__))

CONFIGS = {
    "eager": None,
    "c2": {"mode": "default", "fullgraph": False, "dynamic": False},
    "c3": {"mode": "reduce-overhead", "fullgraph": True, "dynamic": False},
    "c4": {"mode": "max-autotune", "fullgraph": True, "dynamic": False},
    "c4-nocg": {"mode": "max-autotune-no-cudagraphs", "fullgraph": True,
                "dynamic": False},
}


def time_prefill(forward, model, input_ids, device, repeat: int = 10) -> list[float]:
    seq = input_ids.shape[1]
    # One reused cache, reset outside the timed region: a fresh one per call
    # times allocator traffic and forces a cudagraph re-record per prefill.
    cache = static_cache(model, seq + 2, device)

    def one():
        with torch.inference_mode():
            forward(input_ids=input_ids, use_cache=True, past_key_values=cache,
                    cache_position=torch.arange(seq, device=device))

    for _ in range(3):
        cache.reset()
        one()
    times = []
    for _ in range(repeat):
        cache.reset()
        times.append(sync_ms(one))
    return times


def time_decode(forward, model, input_ids, device) -> list[float]:
    with torch.inference_mode():
        cache = static_cache(
            model, input_ids.shape[1] + DECODE_WARMUP + MEASURED_DECODE_STEPS + 2,
            device)
        cur, past = prefill_once(forward, input_ids, cache, device)
        state = {"cur": cur, "past": past}

        def step():
            out = forward(
                input_ids=state["cur"], use_cache=True, past_key_values=cache,
                cache_position=torch.tensor([state["past"]], device=device),
            )
            state["cur"] = out.logits[:, -1:, :].argmax(-1)
            state["past"] += 1

        for _ in range(DECODE_WARMUP):
            step()
        torch.cuda.synchronize()
        torch.compiler.set_stance("fail_on_recompile")
        try:
            times = cuda_event_steps(step, MEASURED_DECODE_STEPS)
        finally:
            torch.compiler.set_stance("default")
    return times


def run_config(model, input_ids, name: str, device: str, ref: list[int]) -> dict:
    kwargs = CONFIGS[name]
    torch._dynamo.reset()
    forward = model.forward if kwargs is None else torch.compile(model, **kwargs)

    got = greedy_static(forward, model, input_ids, CORRECTNESS_TOKENS, device)
    ok, ratio, first = greedy_match(ref, got)
    print(f"[{name}] correctness: {ratio:.3f}"
          + ("" if first is None else f" (first divergence at {first})"))
    if not ok:
        raise SystemExit(f"[{name}] failed the greedy-token gate ({ratio:.3f} < 0.95)")

    prefill_ms = time_prefill(forward, model, input_ids, device)
    decode_ms = time_decode(forward, model, input_ids, device)
    res = {
        "config": name,
        "prefill_p50_ms": percentile(prefill_ms, 0.5),
        "decode_p50_ms": percentile(decode_ms, 0.5),
        "decode_p95_ms": percentile(decode_ms, 0.95),
        "decode_tok_s": 1000.0 / percentile(decode_ms, 0.5),
        "correctness_ratio": ratio,
    }
    print(f"[{name}] prefill p50 {res['prefill_p50_ms']:.2f} ms | "
          f"decode p50 {res['decode_p50_ms']:.4f} ms "
          f"({res['decode_tok_s']:.1f} tok/s)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="c4",
                    choices=list(CONFIGS) + ["sweep"],
                    help="config to run (default c4; cudagraph failure falls "
                         "back to c4-nocg)")
    ap.add_argument("--prompt-tokens", type=int, default=common.PROMPT_TOKENS,
                    help="prompt length; non-default lengths write "
                         "best_torch_p<N>.json")
    args = ap.parse_args()
    device = "cuda"

    print(f"loading {common.MODEL} ...")
    model, tok = load_model(device)
    input_ids = build_prompt(tok, device, args.prompt_tokens)
    print(f"prompt: {input_ids.shape[1]} tokens")

    ref = greedy_static(model.forward, model, input_ids, CORRECTNESS_TOKENS, device)
    print(f"eager reference: {ref[:8]} ...")

    names = (["eager", "c2", "c3", "c4", "c4-nocg"]
             if args.config == "sweep" else [args.config])
    results = []
    for name in names:
        try:
            results.append(run_config(model, input_ids, name, device, ref))
        except SystemExit:
            raise
        except Exception as e:
            if name == "c4":
                print(f"[c4] failed ({type(e).__name__}: {e}); "
                      "retrying as max-autotune-no-cudagraphs")
                results.append(run_config(model, input_ids, "c4-nocg", device, ref))
            else:
                print(f"[{name}] failed: {type(e).__name__}: {e}")

    best = min(results, key=lambda r: r["decode_p50_ms"])
    out = {"model": common.MODEL, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0),
           "prompt_tokens": input_ids.shape[1], "best": best, "all": results}
    out_name = results_name("best_torch", input_ids.shape[1])
    with open(os.path.join(HERE, out_name), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwinner: {best['config']} — decode p50 {best['decode_p50_ms']:.4f} ms, "
          f"prefill p50 {best['prefill_p50_ms']:.2f} ms -> {out_name}")


if __name__ == "__main__":
    main()
