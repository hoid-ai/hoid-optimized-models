#!/usr/bin/env python3
"""Llama-3.1-8B-Instruct generate on the hoid kernel stack — no hoid runtime.

Builds the engine from llama31_optimized.py (the winning decode kernels under
a CUDA-graph-captured step; prefill on the shape-generic elastic stack at any
prompt length), gates it on greedy-token parity against stock HF
transformers, then measures best_torch.py's protocol and prints the
head-to-head.

    uv run best_hoid.py [--prompt-tokens 8192] [--gen-tokens 128]
                        [--weights DIR] [--skip-reference]
"""

from __future__ import annotations

import argparse
import json
import os

import torch

import common as C
import llama31_optimized as Q


@torch.no_grad()
def hf_reference_tokens(weights: str, ids: torch.Tensor, n_new: int) -> list[int]:
    """Greedy continuation from stock HF transformers (eager SDPA), the
    correctness authority for this port."""
    model = C.load_hf_model(weights)
    out = model.generate(
        ids.unsqueeze(0), max_new_tokens=n_new, do_sample=False,
        num_beams=1, pad_token_id=0,
    )[0, ids.numel():].tolist()
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=8192)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--weights", default=C.DEFAULT_WEIGHTS)
    ap.add_argument("--skip-reference", action="store_true",
                    help="skip the HF parity gate (timing-only rerun)")
    args = ap.parse_args()
    args.weights = C.resolve_weights(args.weights)

    C.strict_matmul_precision()
    print(f"[{C.gpu_banner()}]")
    tokzr = C.load_tokenizer(args.weights)
    ids = C.build_prompt_ids(tokzr, args.prompt_tokens).to("cuda")
    print(f"[prompt] {args.prompt_tokens} tokens, this passage tiled: "
          f"{C.prompt_preview(tokzr, ids)!r}")
    P, G = args.prompt_tokens, args.gen_tokens

    hf_tokens = None
    if not args.skip_reference:
        print("[ref] stock HF transformers greedy reference ...")
        hf_tokens = hf_reference_tokens(args.weights, ids, G)

    print("[eng] building the hoid kernel stack (first build compiles the "
          "extensions) ...")
    weights = Q.load_weights(args.weights)
    eng = Q.Llama31Optimized(weights, max_seq=P + G + 2)
    eng.capture()

    # ---- correctness gate: this is not a result until it matches HF -------
    port_tokens = eng.generate(ids, G)
    if hf_tokens is not None:
        n = sum(a == b for a, b in zip(port_tokens, hf_tokens))
        first_div = next((i for i, (a, b) in enumerate(zip(port_tokens, hf_tokens))
                          if a != b), None)
        print(f"[gate] greedy parity vs HF: {n}/{G} (first divergence: {first_div})")
        ok = port_tokens[:16] == hf_tokens[:16] and n >= int(0.95 * G)
        assert ok, "token parity gate FAILED — not a result"

    output_text = tokzr.decode(port_tokens)
    print(f"[output] {G} greedy tokens:\n{output_text}")

    # ---- timing -----------------------------------------------------------
    prefill_reps = 3 if P >= 4096 else 7
    prefill_ms = C.cuda_time(lambda: eng.prefill(ids), reps=prefill_reps, warmup=1)

    first = int(eng.prefill(ids).argmax())
    tok, d = first, []
    for _ in range(8):
        tok = int(eng.step(tok, P).argmax())  # warmup replays at pos P
    steps = min(G, 64)
    for j in range(steps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        logits = eng.step(tok, P + j)
        am = logits.argmax()
        b.record()
        torch.cuda.synchronize()
        tok = int(am)
        d.append(a.elapsed_time(b))

    e2e_ms = C.wall_ms(lambda: eng.generate(ids, G))

    mine = {
        "prefill_p50_ms": round(C.p50(prefill_ms), 3),
        "decode_p50_ms": round(C.p50(d), 4),
        "e2e_ms": round(e2e_ms, 1),
    }
    C.report("hoid kernel stack (captured decode)", P, G, mine)

    # ---- head-to-head -----------------------------------------------------
    tj = "out/best_torch.json"
    if os.path.exists(tj):
        t = json.load(open(tj))
        if t["workload"] == {"prompt_tokens": P, "gen_tokens": G}:
            b = t["best"]
            print(f"\n== vs stock torch (best per phase, {t['torch']}) ==")
            print(f"{'':34s}{'torch':>10s}{'hoid':>10s}{'ratio':>8s}")
            for key, label in [("prefill_p50_ms", "prefill (TTFT) ms"),
                               ("decode_p50_ms", "ms per decode token"),
                               ("e2e_ms", "e2e generate ms")]:
                r = b[key] / mine[key]
                print(f"{label:34s}{b[key]:10.3f}{mine[key]:10.3f}{r:7.2f}x")
        else:
            print(f"({tj} is for a different workload — rerun best_torch.py)")

    C.write_json("out/best_hoid.json", {
        "workload": {"prompt_tokens": P, "gen_tokens": G},
        "torch": torch.__version__,
        "prompt_preview": C.prompt_preview(tokzr, ids),
        "output_text": output_text,
        "metrics": mine,
    })


if __name__ == "__main__":
    main()
