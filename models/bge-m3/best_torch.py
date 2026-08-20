#!/usr/bin/env python
"""Best stock-PyTorch configuration for bge-m3 dense embedding: the full
compile lattice on the campaign's frozen workload (batch 64 x seq 512, bf16,
fully occupied), winner promoted to the retrieval demo.

No hoid kernels anywhere — this is the strongest configuration reachable with
the unmodified HF model, as the fair baseline for best_hoid.py.

The lattice (eager-SDPA, compile default, reduce-overhead, max-autotune,
max-autotune-no-cudagraphs; fullgraph first with automatic fallback) re-derives
the winner on every box: same-GPU boxes still differ in driver and clocks.
Every config is gated on pooled-embedding cosine >= 0.99 against the eager
reference — the measured bf16-vs-fp32 floor for this model is ~0.9917, so a
tighter gate would reject configs bf16 itself cannot distinguish. The measured
window runs under `fail_on_recompile` so a silent eager fallback raises instead
of reporting a bogus "compiled" number.

Run (see README.md in this folder for the uv environment setup):
  uv run best_torch.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import harness as H  # noqa: E402

LATTICE = [
    ("c1-eager-sdpa", None),
    ("c2-compile-default", {}),
    ("c3-reduce-overhead", {"mode": "reduce-overhead"}),
    ("c4-max-autotune", {"mode": "max-autotune"}),
    ("c5-max-autotune-no-cudagraphs", {"mode": "max-autotune-no-cudagraphs"}),
]


def compile_with_fallback(model, kwargs, inputs, attempts=(True, False)):
    """torch.compile preferring fullgraph=True; returns the warmed callable."""
    for fullgraph in attempts:
        torch._dynamo.reset()
        cand = torch.compile(model, dynamic=False, fullgraph=fullgraph, **kwargs)
        try:
            with torch.inference_mode():
                cand(**inputs)
            return cand, fullgraph
        except Exception as e:  # dynamo raises on the first call, not at compile()
            print(f"    fullgraph={fullgraph} failed: {type(e).__name__}: {str(e)[:160]}")
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--compile-mode", default=None,
                    help="run ONE lattice config by name (e.g. c4-max-autotune)")
    ap.add_argument("--json", default="out/best_torch.json")
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    torch.set_float32_matmul_precision("high")
    print(f"[env] torch {torch.__version__}, {torch.cuda.get_device_name()}")

    tok = AutoTokenizer.from_pretrained(H.REPO)
    model = (AutoModel.from_pretrained(H.REPO, dtype=torch.bfloat16,
                                       attn_implementation="sdpa")
             .to("cuda").eval())
    inputs = H.frozen_inputs(tok)

    with torch.inference_mode():
        ref = H.pool_cls(model(**inputs).last_hidden_state).clone()

    lattice = [c for c in LATTICE if args.compile_mode in (None, c[0])]
    if not lattice:
        raise SystemExit(f"unknown config {args.compile_mode}")

    rows, best = {}, None
    for name, kwargs in lattice:
        if kwargs is None:
            fwd, fg = model, "-"
        else:
            print(f"[compile] {name} (autotuning can take a while)...")
            fwd, fg = compile_with_fallback(model, kwargs, inputs)
            if fwd is None:
                print(f"[bench] {name}: FAILED to compile in any configuration")
                continue

        with torch.inference_mode():
            emb = H.pool_cls(fwd(**inputs).last_hidden_state)
        lo = H.row_cosines(emb, ref).min().item()
        if lo < 0.99:
            print(f"[bench] {name}: REJECTED, min cosine {lo:.4f} < 0.99")
            continue

        def one():
            with torch.inference_mode():
                fwd(**inputs)

        torch.compiler.set_stance("fail_on_recompile" if kwargs is not None else "default")
        try:
            stats = H.bench(one, iters=args.iters, warmup=args.warmup)
        finally:
            torch.compiler.set_stance("default")
        rows[name] = (stats, lo, fg)
        print(f"[bench] {name:32} {H.fmt(stats)}  min_cos={lo:.4f} fullgraph={fg}")
        if best is None or stats["mean"] < rows[best][0]["mean"]:
            best, best_fwd = name, fwd

    if best is None:
        raise SystemExit("no lattice config passed the gate")
    print(f"[choose] winner: {best}")

    # -- e2e retrieval demo on the winner ------------------------------------
    # 8 cross-lingual query/passage pairs + 48 distractors in ONE batch-64
    # forward — the timed workload, doing its real job.
    demo_ids, q_rows, p_rows = H.demo_batch(tok)
    demo_inputs = {"input_ids": demo_ids, "attention_mask": torch.ones_like(demo_ids)}

    def demo_once():
        with torch.inference_mode():
            pooled = H.pool_cls(best_fwd(**demo_inputs).last_hidden_state)
        H.retrieval_top1(pooled, q_rows)
        return pooled

    demo_once()  # warm (values changed, shapes did not)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pooled = demo_once()
    torch.cuda.synchronize()
    demo_ms = (time.perf_counter() - t0) * 1e3

    print("\n== retrieval demo (winner config) ==")
    H.print_retrieval(pooled, q_rows, p_rows)

    print(f"\n== stock torch, bge-m3 dense, batch {H.BATCH} x seq {H.SEQ}, bf16, warm ==")
    for name, (stats, lo, fg) in rows.items():
        tag = "  <- winner" if name == best else ""
        print(f"{name:34} {H.fmt(stats)}  min_cos={lo:.4f}{tag}")
    print(f"e2e demo (embed 64 texts + rank)   {demo_ms:8.3f} ms")
    thr = H.BATCH / (rows[best][0]['p50'] / 1e3)
    print(f"throughput at winner p50           {thr:8.1f} seq/s")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"impl": f"stock torch {best}",
                       "gpu": torch.cuda.get_device_name(0),
                       "chosen": best, "batch": H.BATCH, "seq": H.SEQ,
                       "lattice_ms": {k: {"mean": round(s["mean"], 3),
                                          "min": round(s["min"], 3),
                                          "p50": round(s["p50"], 3),
                                          "min_cos": round(lo, 4)}
                                      for k, (s, lo, _) in rows.items()},
                       "forward_p50_ms": round(rows[best][0]["p50"], 3),
                       "forward_mean_ms": round(rows[best][0]["mean"], 3),
                       "throughput_seq_s": round(thr, 1),
                       "demo_ms": round(demo_ms, 3),
                       "retrieval": H.retrieval_rows(pooled, q_rows, p_rows)},
                      f, indent=2)
        print(f"[json] {args.json}")


if __name__ == "__main__":
    main()
