#!/usr/bin/env python
"""The strongest stock-PyTorch configuration for SDXL base 1.0 at 1024x1024.

No custom kernels anywhere — this is the bar `best_hoid.py` has to clear, so it
is built to be as fast as unmodified diffusers can be made:

  1. It runs the whole configuration lattice on the hot component (one UNet
     denoise step at the CFG-doubled batch): eager-SDPA, compile default,
     reduce-overhead, max-autotune, max-autotune with the diffusion extras
     (channels_last + cudnn.benchmark), and max-autotune without inductor's
     CUDA graphs. `fullgraph=True` is tried first for every compiled mode,
     falling back to partial graphs. Every row is printed — the ratio in the
     README is only as honest as the losing rows next to it.
  2. The measured winner is promoted into the end-to-end phase, where the VAE
     and both text encoders are compiled too, so no stage is left eager.
  3. A real image is generated and saved.

Self-deriving the winner matters: two boxes with the same GPU still differ in
driver, clocks and inductor version, and the mode that wins moves with them.

Run (see README.md for the uv environment setup):
    uv run best_torch.py
    uv run best_torch.py --compile-mode c5-max-autotune-extras   # skip the lattice
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import common  # noqa: E402


def compile_unet(unet, cfg, inputs, fullgraph_attempts):
    """Apply one lattice row and return a warmed callable, or None."""
    sample, timestep, ctx, added = inputs
    if cfg["compile"] is None:
        with torch.no_grad():
            unet(sample, timestep, encoder_hidden_states=ctx, added_cond_kwargs=added)
        return unet
    for fullgraph in fullgraph_attempts:
        cand = torch.compile(unet, fullgraph=fullgraph, **cfg["compile"])
        try:
            with torch.no_grad():
                cand(sample, timestep, encoder_hidden_states=ctx, added_cond_kwargs=added)
            print(f"[compile] OK  fullgraph={fullgraph}")
            return cand
        except Exception as e:
            print(f"[compile] fullgraph={fullgraph} failed: {type(e).__name__}: {str(e)[:200]}")
            torch._dynamo.reset()
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default=common.DEFAULT_PROMPT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--compile-mode", default=None,
                    help="force ONE lattice row by name and skip the sweep")
    ap.add_argument("--no-fullgraph", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--e2e-warmup", type=int, default=1)
    ap.add_argument("--e2e-repeat", type=int, default=3)
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--out", default="out/best_torch.png")
    ap.add_argument("--json", default="out/best_torch.json")
    args = ap.parse_args()

    print(f"[env] {common.gpu_banner()}")
    pipe = common.load_pipeline()
    unet = pipe.unet

    rows = ({args.compile_mode: common.TORCH_CONFIGS[args.compile_mode]}
            if args.compile_mode else common.TORCH_CONFIGS)
    attempts = (False,) if args.no_fullgraph else (True, False)

    # -- 1. the lattice ------------------------------------------------------
    results: dict[str, tuple[float, float]] = {}
    extras_on = False
    for name, cfg in rows.items():
        torch._dynamo.reset()
        if cfg["extras"] and not extras_on:
            # channels_last is a weight-layout change: applied once, from here
            # on. The lattice is ordered so no later row needs it off again.
            unet.to(memory_format=torch.channels_last)
            torch.backends.cudnn.benchmark = True
            extras_on = True
        inputs = common.unet_inputs(channels_last=extras_on)
        print(f"\n[lattice] {name} (compiling/autotuning, this can take minutes)...")
        fn = compile_unet(unet, cfg, inputs, attempts)
        if fn is None:
            print(f"[lattice] {name}: no configuration ran — skipped")
            continue
        sample, timestep, ctx, added = inputs
        with torch.no_grad():
            mean, mn = common.bench_forward(
                lambda: fn(sample, timestep, encoder_hidden_states=ctx, added_cond_kwargs=added),
                iters=args.iters, warmup=args.warmup)
        results[name] = (mean, mn)
        print(f"[lattice] {name}: {mean:8.3f} ms mean / {mn:.3f} ms min")

    if not results:
        raise SystemExit("no lattice row completed")
    chosen = min(results, key=lambda k: results[k][0])
    print(f"\n[choose] winner: {chosen} at {results[chosen][0]:.3f} ms")

    if args.skip_e2e:
        _summary(results, chosen, None, None, None, args)
        return

    # -- 2. e2e with the winner, every other stage compiled too --------------
    torch._dynamo.reset()
    cfg = common.TORCH_CONFIGS[chosen]
    if cfg["extras"] and not extras_on:
        unet.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
        extras_on = True
    elif not cfg["extras"] and extras_on:
        # a later lattice row turned the extras on; the winner did not use
        # them, so e2e must not either
        unet.to(memory_format=torch.contiguous_format)
        torch.backends.cudnn.benchmark = False
        extras_on = False
    inputs = common.unet_inputs(channels_last=extras_on)
    fn = compile_unet(unet, cfg, inputs, attempts)
    pipe.unet = fn
    common.compile_side_stages(pipe)

    e2e_ms, stages, out_path, runs = common.e2e_phase(pipe, args, chosen)

    # -- 3. the same forward once more, everything maximally warm -----------
    sample, timestep, ctx, added = inputs
    with torch.no_grad():
        mean2, mn2 = common.bench_forward(
            lambda: fn(sample, timestep, encoder_hidden_states=ctx, added_cond_kwargs=added),
            iters=args.iters, warmup=args.warmup)
    print(f"[bench] ONE UNet forward after e2e: {mean2:.3f} ms mean / {mn2:.3f} ms min")

    _summary(results, chosen, (mean2, mn2), (e2e_ms, runs), (stages, out_path), args)


def _summary(results, chosen, after, e2e, stage_info, args):
    print(f"\n== stock PyTorch, SDXL base 1.0, 1024x1024, {args.steps} steps, warm ==")
    print(f"   {common.gpu_banner()}")
    print(f"   prompt: {args.prompt!r}")
    for name, (mean, mn) in results.items():
        tag = "  <- chosen" if name == chosen else ""
        print(f"ONE UNet forward  {name:32} {mean:9.3f} ms  (min {mn:.3f}){tag}")
    if after:
        print(f"ONE UNet forward  {'after e2e':32} {after[0]:9.3f} ms  (min {after[1]:.3f})")
    if stage_info:
        stages, out_path = stage_info
        for stage in ("text_encode", "denoise_step", "vae_decode"):
            if stage in stages:
                s = stages[stage]
                per_run = s['calls'] // args.e2e_repeat
                print(f"{stage + ' (in pipeline)':50} "
                      f"{s['median_ms'] * per_run:9.2f} ms/run"
                      f"  ({per_run} calls, median {s['median_ms']:.2f} ms/call)")
        med, runs = e2e
        print(f"{'e2e whole image (median)':50} {med:9.0f} ms   "
              f"(runs: {', '.join(f'{v:.0f}' for v in runs)})")
        print(f"image -> {out_path}")
    if args.json:
        row = {"impl": f"stock torch {chosen}", "gpu": common.gpu_banner(),
               "chosen": chosen, "prompt": args.prompt, "image": args.out,
               "lattice_ms": {k: {"mean": round(m, 3), "min": round(mn, 3)}
                              for k, (m, mn) in results.items()},
               "unet_forward_ms": round((after or results[chosen])[0], 3),
               "unet_forward_min_ms": round((after or results[chosen])[1], 3)}
        if stage_info:
            stages, _ = stage_info
            row["stages_ms"] = {
                s: {"per_call": round(st["median_ms"], 3),
                    "per_image": round(st["median_ms"] * (st["calls"] // args.e2e_repeat), 3)}
                for s, st in stages.items()}
            row["image_ms"], row["image_runs_ms"] = round(e2e[0], 1), \
                [round(v, 1) for v in e2e[1]]
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(row, f, indent=2)
        print(f"[json] {args.json}")


if __name__ == "__main__":
    main()
