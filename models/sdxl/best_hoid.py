#!/usr/bin/env python
"""SDXL base 1.0 at 1024x1024 with the tuned kernel stack, end to end.

The UNet's feed-forward, cross-attention projections and attention run the
kernels described in `sdxl_optimized.py`; every convolution, sampler, text
encoder and the VAE stay stock diffusers. Pure PyTorch — the kernels are
nvcc/Triton/CuTe-DSL compiled at import.

The script refuses to print performance until both correctness gates pass:

  1. numeric — one UNet forward on fixed inputs against the stock UNet on the
     same inputs, SNR in dB (floor 30 dB, the same bar the reference ports use),
  2. task-level — a full seed-fixed generation compared image-to-image against
     the same generation on the stock UNet, and both images saved.

Then it measures the three execution modes (eager over the custom ops,
torch.compile over them, and manual CUDA-graph capture), keeps the fastest,
and runs the whole pipeline with it.

Run (see README.md for the uv environment setup):
    uv run best_hoid.py
    uv run best_hoid.py --exec-mode compile      # skip the sweep
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common  # noqa: E402
import sdxl_optimized as SO  # noqa: E402

SNR_FLOOR_DB = 30.0
# A 30-step diffusion loop amplifies any difference in rounding, so an image
# gate needs a scale. The run derives one: it generates the same seed and
# prompt twice on the STOCK model, once contiguous and once channels_last —
# two legitimate stock configurations — and reports how far apart they land.
# The absolute floor below is what the gate enforces; the measured stock-vs-
# stock spread is printed next to it so the floor can be judged.
IMAGE_PSNR_FLOOR_DB = 25.0


def image_psnr(a, b) -> float:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    mse = ((x - y) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


class CapturedUNet:
    """The whole denoise step replayed from one CUDA graph.

    Inputs are copied into buffers owned before the capture, so nothing the
    graph reads can be reallocated underneath it between replays.
    """

    def __init__(self, unet, inputs):
        sample, timestep, ctx, added = inputs
        self.sample = sample.clone()
        self.ctx = ctx.clone()
        self.added = {k: v.clone() for k, v in added.items()}
        # the timestep changes every denoise step, so it is a copied-into
        # buffer like the rest — not the caller's tensor captured once
        self.timestep = timestep.clone()
        with torch.no_grad():
            for _ in range(3):  # warm every lazy allocation before capture
                unet(self.sample, self.timestep, encoder_hidden_states=self.ctx,
                     added_cond_kwargs=self.added)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = unet(self.sample, self.timestep, encoder_hidden_states=self.ctx,
                                added_cond_kwargs=self.added).sample

    def __call__(self, sample, timestep, encoder_hidden_states, added_cond_kwargs=None,
                 **kw):
        self.sample.copy_(sample)
        self.timestep.copy_(torch.as_tensor(timestep, device=self.timestep.device)
                            .reshape(self.timestep.shape))
        self.ctx.copy_(encoder_hidden_states)
        for k, v in (added_cond_kwargs or {}).items():
            self.added[k].copy_(v)
        self.graph.replay()
        return self.out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default=common.DEFAULT_PROMPT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--exec-mode", default=None, choices=["eager", "compile", "capture"],
                    help="force ONE execution mode and skip the sweep")
    ap.add_argument("--compile-mode", default="max-autotune")
    ap.add_argument("--no-fused-ln", action="store_true",
                    help="leave the attention output projection / residual / LayerNorm "
                         "to the compiler instead of the fused kernel")
    ap.add_argument("--no-geglu", action="store_true",
                    help="stock GEGLU instead of the dual-GEMM")
    ap.add_argument("--no-vae", action="store_true",
                    help="leave the VAE decoder's mid-block attention on torch's SDPA")
    ap.add_argument("--no-packed-kv", action="store_true",
                    help="per-block cross-attention K/V projections + SDPA instead of "
                         "one packed GEMM and the packed-KV flash kernel")
    ap.add_argument("--self-attn", default=None, choices=["sdpa", "fa4", "fmha"])
    ap.add_argument("--no-extras", action="store_true",
                    help="skip channels_last + cudnn.benchmark (they are what makes the "
                         "stock baseline's winning row win, so both sides get them)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--e2e-warmup", type=int, default=1)
    ap.add_argument("--e2e-repeat", type=int, default=3)
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--out", default="out/best_hoid.png")
    ap.add_argument("--ref-out", default="out/stock_reference.png")
    ap.add_argument("--json", default="out/best_hoid.json")
    args = ap.parse_args()
    if args.self_attn:
        SO.set_self_attn(args.self_attn)

    print(f"[env] {common.gpu_banner()}")
    pipe = common.load_pipeline()
    inputs = common.unet_inputs()
    sample, timestep, ctx, added = inputs

    # -- the references, taken from the stock model BEFORE it is modified ----
    with torch.no_grad():
        ref = pipe.unet(sample, timestep, encoder_hidden_states=ctx,
                        added_cond_kwargs=added).sample.clone()
    print("[ref] stock UNet forward captured")
    ref_image, stock_spread = None, None
    if not args.skip_e2e:
        print(f"[ref] stock {args.steps}-step generation, contiguous ...")
        contiguous_image, ref_ms = common.run_e2e(pipe, args.prompt, args.seed, args.steps)
        print(f"[ref] {ref_ms:.0f} ms eager")

    # -- the kernel stack ----------------------------------------------------
    # The same two diffusion knobs the stock baseline's winning row carries.
    # They only touch the convolutions, which stay stock diffusers here, so
    # leaving them off would be measuring a weaker model, not a fairer one.
    if not args.no_extras:
        pipe.unet.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
        inputs = common.unet_inputs(channels_last=True)
        sample, timestep, ctx, added = inputs

    if not args.skip_e2e:
        # The task-level reference: the stock model in the SAME memory format
        # the kernel stack runs in, so the comparison isolates the kernels.
        print(f"[ref] stock {args.steps}-step generation, "
              f"{'channels_last' if not args.no_extras else 'contiguous'} ...")
        ref_image, _ = common.run_e2e(pipe, args.prompt, args.seed, args.steps)
        os.makedirs(os.path.dirname(args.ref_out) or ".", exist_ok=True)
        ref_image.save(args.ref_out)
        stock_spread = image_psnr(ref_image, contiguous_image)
        print(f"[ref] -> {args.ref_out};  stock contiguous vs stock channels_last: "
              f"{stock_spread:.2f} dB (the scale this gate is read against)")
    print("[build] compiling the kernel stack ...")
    kernels = SO.Kernels(fused_ln=not args.no_fused_ln, geglu=not args.no_geglu,
                         packed_kv=not args.no_packed_kv)
    opt = SO.optimize(pipe.unet, kernels=kernels)
    if not args.no_vae:
        SO.optimize_vae(pipe.vae)
    print(f"[build] ready (self-attention: {SO.SELF_ATTN}, {kernels})")

    # -- gate 1: numeric -----------------------------------------------------
    with torch.no_grad():
        got = opt(sample, timestep, encoder_hidden_states=ctx, added_cond_kwargs=added).sample
    snr = common.snr_db(ref, got)
    max_abs = (got.float() - ref.float()).abs().max().item()
    print(f"[gate] UNet forward vs stock: SNR {snr:.2f} dB, max_abs {max_abs:.4f}")
    if snr < SNR_FLOOR_DB:
        raise SystemExit(f"correctness gate failed: {snr:.2f} dB < {SNR_FLOOR_DB} dB floor")

    # -- the execution-mode sweep -------------------------------------------
    modes = [args.exec_mode] if args.exec_mode else ["eager", "compile", "capture"]
    results: dict[str, tuple[float, float]] = {}
    callables: dict[str, object] = {}
    for mode in modes:
        torch._dynamo.reset()
        print(f"\n[exec] {mode} ...")
        try:
            if mode == "eager":
                fn = opt
            elif mode == "compile":
                fn = torch.compile(opt, mode=args.compile_mode, dynamic=False)
            else:
                fn = CapturedUNet(opt, inputs)
            with torch.no_grad():
                mean, mn = common.bench_forward(
                    lambda: fn(sample, timestep, encoder_hidden_states=ctx,
                               added_cond_kwargs=added),
                    iters=args.iters, warmup=args.warmup)
        except Exception as e:
            print(f"[exec] {mode} failed: {type(e).__name__}: {str(e)[:300]}")
            continue
        with torch.no_grad():
            out = fn(sample, timestep, encoder_hidden_states=ctx, added_cond_kwargs=added)
        out = out if isinstance(out, torch.Tensor) else out.sample
        s = common.snr_db(ref, out)
        results[mode] = (mean, mn)
        callables[mode] = fn
        print(f"[exec] {mode}: {mean:8.3f} ms mean / {mn:.3f} ms min   (SNR {s:.2f} dB)")
        if s < SNR_FLOOR_DB:
            print(f"[exec] {mode} dropped — it diverges ({s:.2f} dB)")
            results.pop(mode), callables.pop(mode)

    if not results:
        raise SystemExit("no execution mode ran")
    chosen = min(results, key=lambda k: results[k][0])
    print(f"\n[choose] winner: {chosen} at {results[chosen][0]:.3f} ms")

    if args.skip_e2e:
        _summary(results, chosen, None, None, None, args, snr, None)
        return

    # -- e2e with the winner, every other stage compiled too -----------------
    fn = callables[chosen]
    pipe.unet = fn if isinstance(fn, torch.nn.Module) else _AsModule(fn, opt)
    common.compile_side_stages(pipe)

    e2e_ms, stages, out_path, runs = common.e2e_phase(pipe, args, chosen)

    # -- gate 2: task level --------------------------------------------------
    from PIL import Image

    psnr = image_psnr(Image.open(out_path), ref_image)
    print(f"[gate] generated image vs the stock generation: PSNR {psnr:.2f} dB "
          f"(two stock configurations differ by {stock_spread:.2f} dB)")
    if psnr < IMAGE_PSNR_FLOOR_DB:
        raise SystemExit(f"task-level gate failed: {psnr:.2f} dB < {IMAGE_PSNR_FLOOR_DB} dB floor")

    with torch.no_grad():
        mean2, mn2 = common.bench_forward(
            lambda: fn(sample, timestep, encoder_hidden_states=ctx, added_cond_kwargs=added),
            iters=args.iters, warmup=args.warmup)
    print(f"[bench] ONE UNet forward after e2e: {mean2:.3f} ms mean / {mn2:.3f} ms min")

    _summary(results, chosen, (mean2, mn2), (e2e_ms, runs), (stages, out_path), args,
             snr, (psnr, stock_spread))


class _AsModule(torch.nn.Module):
    """Wraps a non-module callable (the captured step) so the diffusers
    pipeline can still read UNet attributes off it."""

    def __init__(self, fn, unet):
        super().__init__()
        self.unet = unet
        self._fn = [fn]      # a plain list: not a child module

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["unet"], name)

    def forward(self, *a, **kw):
        return self._fn[0](*a, **kw)


def _summary(results, chosen, after, e2e, stage_info, args, snr, psnr):
    print(f"\n== tuned kernel stack, SDXL base 1.0, 1024x1024, {args.steps} steps, warm ==")
    print(f"   {common.gpu_banner()}")
    print(f"   self-attention {SO.SELF_ATTN} | "
          f"{SO.Kernels(fused_ln=not args.no_fused_ln, geglu=not args.no_geglu, packed_kv=not args.no_packed_kv)}")
    for name, (mean, mn) in results.items():
        tag = "  <- chosen" if name == chosen else ""
        print(f"ONE UNet forward  {name:32} {mean:9.3f} ms  (min {mn:.3f}){tag}")
    if after:
        print(f"ONE UNet forward  {'after e2e':32} {after[0]:9.3f} ms  (min {after[1]:.3f})")
    print(f"{'numeric gate vs stock UNet':50} {snr:9.2f} dB")
    if psnr is not None:
        got, spread = psnr
        print(f"{'image gate vs stock generation':50} {got:9.2f} dB")
        print(f"{'  (two stock configurations differ by)':50} {spread:9.2f} dB")
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
        print(f"image -> {out_path}   (stock reference -> {args.ref_out})")
    if args.json:
        row = {"impl": f"hoid kernel stack ({chosen})", "gpu": common.gpu_banner(),
               "chosen": chosen,
               "exec_modes_ms": {k: {"mean": round(m, 3), "min": round(mn, 3)}
                                 for k, (m, mn) in results.items()},
               "unet_forward_ms": round((after or results[chosen])[0], 3),
               "unet_forward_min_ms": round((after or results[chosen])[1], 3),
               "unet_snr_db": round(snr, 2)}
        if psnr is not None:
            row["image_psnr_db"], row["stock_spread_db"] = \
                round(psnr[0], 2), round(psnr[1], 2)
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
