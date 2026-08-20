#!/usr/bin/env python
"""bge-m3 dense embedding on the hoid-winning kernel stack — pure PyTorch, no
hoid runtime. Kernels come verbatim from the winning hoid stack (see
bge_kernels_gen.py); the projection GEMMs stay cuBLAS, as hoid ran them.

Refuses to print performance until the correctness gates pass:
  1. last_hidden SNR vs an fp32 reference (TF32 off)  >= 15 dB   (campaign gate)
  2. pooled min row-cosine vs the fp32 reference      >= 0.95    (campaign oracle)
  3. the retrieval demo must retrieve the SAME passage as stock bf16
     for every query (task-level check)

Execution mode is empirical: all three ways of running the same kernels are
measured — eager on static buffers, torch.compile over the custom ops, and
CUDA-graph capture — and the winner runs the demo.

Run (see README.md in this folder for the uv environment setup):
  uv run best_hoid.py
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
from bge_optimized import OptimizedBgeM3, register_ops  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--skip-check", action="store_true",
                    help="skip the correctness gates (debugging only)")
    ap.add_argument("--json", default="out/best_hoid.json")
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    print(f"[env] torch {torch.__version__}, {torch.cuda.get_device_name()}")
    register_ops()

    tok = AutoTokenizer.from_pretrained(H.REPO)
    stock = (AutoModel.from_pretrained(H.REPO, dtype=torch.bfloat16,
                                       attn_implementation="sdpa")
             .to("cuda").eval())
    inputs = H.frozen_inputs(tok)

    optim = OptimizedBgeM3(stock)
    optim.load_inputs(inputs["input_ids"])

    # -- correctness gates (before any timing) -------------------------------
    if not args.skip_check:
        with torch.inference_mode():
            stock_hidden = stock(**inputs).last_hidden_state
            stock_pooled = H.pool_cls(stock_hidden)

        # fp32 reference with TF32 off — otherwise the reference itself moves
        # several dB between runs and every SNR is noise.
        print("[check] building fp32 reference (TF32 off)...")
        torch.backends.cuda.matmul.allow_tf32 = False
        ref32 = (AutoModel.from_pretrained(H.REPO, dtype=torch.float32,
                                           attn_implementation="sdpa")
                 .to("cuda").eval())
        with torch.inference_mode():
            out32 = ref32(**inputs).last_hidden_state
            hidden32 = out32.clone()
            pooled32 = H.pool_cls(out32).clone()
        del ref32, out32
        torch.cuda.empty_cache()

        pooled = optim.forward_static()
        hidden = optim.last_hidden()
        snr = H.snr_db(hidden32.float(), hidden.float())
        cos32 = H.row_cosines(pooled, pooled32).min().item()
        cos_bf16 = H.row_cosines(pooled, stock_pooled).min().item()
        snr_stock = H.snr_db(hidden32.float(), stock_hidden.float())
        print(f"[check] last_hidden SNR vs fp32: hoid {snr:.2f} dB "
              f"(stock bf16 scores {snr_stock:.2f} dB), gate >= 15")
        print(f"[check] pooled min-cosine: vs fp32 {cos32:.4f} (gate >= 0.95), "
              f"vs stock bf16 {cos_bf16:.4f}")
        if snr < 15.0 or cos32 < 0.95:
            raise SystemExit("FAIL: hoid stack diverges from the fp32 reference — "
                             "no performance numbers until this is fixed")
        del stock_hidden, hidden32, pooled32
        torch.cuda.empty_cache()

    # -- execution-mode lattice: same kernels, three drivers -----------------
    rows = {}

    optim.load_inputs(inputs["input_ids"])
    rows["eager-static-buffers"] = H.bench(optim.forward_static,
                                           iters=args.iters, warmup=args.warmup)
    print(f"[bench] eager-static-buffers        {H.fmt(rows['eager-static-buffers'])}")

    print("[compile] custom ops, mode=max-autotune-no-cudagraphs...")
    compiled = None
    for fullgraph in (True, False):
        torch._dynamo.reset()
        cand = torch.compile(optim.forward_ops, dynamic=False, fullgraph=fullgraph,
                             mode="max-autotune-no-cudagraphs")
        try:
            with torch.inference_mode():
                cand(inputs["input_ids"])
            compiled = cand
            break
        except Exception as e:
            print(f"    fullgraph={fullgraph} failed: {type(e).__name__}: {str(e)[:160]}")
    if compiled is not None:
        def one_compiled():
            with torch.inference_mode():
                compiled(inputs["input_ids"])
        torch.compiler.set_stance("fail_on_recompile")
        try:
            rows["compile-custom-ops"] = H.bench(one_compiled, iters=args.iters,
                                                 warmup=args.warmup)
        finally:
            torch.compiler.set_stance("default")
        print(f"[bench] compile-custom-ops          {H.fmt(rows['compile-custom-ops'])}")
    else:
        print("[bench] compile-custom-ops          FAILED to compile")

    replay = optim.capture()
    rows["cuda-graph"] = H.bench(lambda: replay(inputs["input_ids"]),
                                 iters=args.iters, warmup=args.warmup)
    print(f"[bench] cuda-graph                  {H.fmt(rows['cuda-graph'])}")

    best = min(rows, key=lambda k: rows[k]["mean"])
    print(f"[choose] winner: {best}")

    # -- retrieval demo on the winner + task-level gate ----------------------
    demo_ids, q_rows, p_rows = H.demo_batch(tok)

    if best == "cuda-graph":
        run = lambda ids: replay(ids).clone()
    elif best == "compile-custom-ops":
        def run(ids):
            with torch.inference_mode():
                return compiled(ids).clone()
    else:
        def run(ids):
            optim.load_inputs(ids)
            return optim.forward_static().clone()

    run(demo_ids)  # warm (values changed, shapes did not)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pooled_demo = run(demo_ids)
    torch.cuda.synchronize()
    demo_ms = (time.perf_counter() - t0) * 1e3

    with torch.inference_mode():
        stock_demo = H.pool_cls(
            stock(input_ids=demo_ids,
                  attention_mask=torch.ones_like(demo_ids)).last_hidden_state)
    hoid_top1 = H.retrieval_top1(pooled_demo, q_rows)
    stock_top1 = H.retrieval_top1(stock_demo, q_rows)
    if hoid_top1 != stock_top1 and not args.skip_check:
        print(f"hoid top1:  {hoid_top1}\nstock top1: {stock_top1}")
        raise SystemExit("FAIL: retrieval demo ranks differently than stock bf16 — "
                         "no performance numbers until this is fixed")

    print("\n== retrieval demo (hoid stack; stock retrieves identically) ==")
    H.print_retrieval(pooled_demo, q_rows, p_rows)

    print(f"\n== hoid kernel stack, bge-m3 dense, batch {H.BATCH} x seq {H.SEQ}, bf16, warm ==")
    for name, stats in rows.items():
        tag = "  <- winner" if name == best else ""
        print(f"{name:34} {H.fmt(stats)}{tag}")
    print(f"e2e demo (embed 64 texts + rank)   {demo_ms:8.3f} ms")
    thr = H.BATCH / (rows[best]["p50"] / 1e3)
    print(f"throughput at winner p50           {thr:8.1f} seq/s")

    if args.json:
        row = {"impl": f"hoid kernel stack ({best})",
               "gpu": torch.cuda.get_device_name(0),
               "chosen": best, "batch": H.BATCH, "seq": H.SEQ,
               "drivers_ms": {k: {"mean": round(s["mean"], 3),
                                  "min": round(s["min"], 3),
                                  "p50": round(s["p50"], 3)}
                              for k, s in rows.items()},
               "forward_p50_ms": round(rows[best]["p50"], 3),
               "forward_mean_ms": round(rows[best]["mean"], 3),
               "throughput_seq_s": round(thr, 1),
               "demo_ms": round(demo_ms, 3)}
        if not args.skip_check:
            row["gates"] = {"last_hidden_snr_db": round(snr, 2),
                            "pooled_cos_fp32": round(cos32, 4),
                            "pooled_cos_stock_bf16": round(cos_bf16, 4)}
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(row, f, indent=2)
        print(f"[json] {args.json}")


if __name__ == "__main__":
    main()
