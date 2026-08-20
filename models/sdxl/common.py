"""Shared plumbing for the two scripts here: the model, the inputs, the clock.

The rule both scripts follow: they see the same pipeline, the same weights, the
same latents and the same timing loop — only the UNet implementation differs.

The hot component is the UNet denoise step. At 1024x1024 with classifier-free
guidance the pipeline calls it once per step on a CFG-doubled batch, so the
contract shape is fixed and static:

    sample                [2, 4, 128, 128]   bf16
    encoder_hidden_states [2, 77, 2048]      bf16
    text_embeds           [2, 1280]          bf16
    time_ids              [2, 6]             bf16

Both scripts bench exactly that call, then run the whole pipeline (text encode
-> 30 steps -> VAE decode) and save the image.
"""

from __future__ import annotations

import os
import statistics
import time

import torch

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

BATCH, LATENT, LATENT_CH = 2, 128, 4
CTX_TOKENS, CTX_DIM, POOLED_DIM = 77, 2048, 1280

DEFAULT_PROMPT = ("A cinematic photograph of an astronaut riding a horse across a "
                  "red desert at sunset, dust in the air, sharp focus, 50mm lens")

# The stock configuration lattice for the hot component. `extras` are the
# diffusion-specific knobs (channels_last + cudnn.benchmark) that the strongest
# stock configuration carries; the rest is the inductor mode.
TORCH_CONFIGS = {
    "c1-eager-sdpa": dict(compile=None, extras=False),
    "c2-compile-default": dict(compile={"dynamic": False}, extras=False),
    "c3-reduce-overhead": dict(compile={"mode": "reduce-overhead", "dynamic": False}, extras=False),
    "c4-max-autotune": dict(compile={"mode": "max-autotune", "dynamic": False}, extras=False),
    "c5-max-autotune-extras": dict(compile={"mode": "max-autotune", "dynamic": False}, extras=True),
    "c6-max-autotune-no-cudagraphs": dict(
        compile={"mode": "max-autotune-no-cudagraphs", "dynamic": False}, extras=True),
}


def load_pipeline(dtype=torch.bfloat16, device="cuda"):
    from diffusers import StableDiffusionXLPipeline

    path = os.environ.get("SDXL_PATH", MODEL)
    pipe = StableDiffusionXLPipeline.from_pretrained(path, torch_dtype=dtype,
                                                     use_safetensors=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def unet_inputs(device="cuda", dtype=torch.bfloat16, seed=0, channels_last=False):
    """The fixed denoise-step inputs, identical for every configuration."""
    g = torch.Generator(device).manual_seed(seed)
    sample = torch.randn(BATCH, LATENT_CH, LATENT, LATENT, generator=g,
                         device=device, dtype=dtype)
    if channels_last:
        sample = sample.to(memory_format=torch.channels_last)
    ctx = torch.randn(BATCH, CTX_TOKENS, CTX_DIM, generator=g, device=device, dtype=dtype)
    added = {
        "text_embeds": torch.randn(BATCH, POOLED_DIM, generator=g, device=device, dtype=dtype),
        "time_ids": torch.tensor([[1024.0, 1024, 0, 0, 1024, 1024]] * BATCH,
                                 device=device, dtype=dtype),
    }
    timestep = torch.tensor(981, device=device)
    return sample, timestep, ctx, added


def bench_forward(fn, iters: int = 20, warmup: int = 5):
    """mean/min ms over `iters` timed calls, each synchronized on both sides."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.mean(times), min(times)


def snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref, got = ref.float(), got.float()
    err = got - ref
    return (10 * torch.log10(ref.square().sum() / err.square().sum().clamp_min(1e-30))).item()


def gpu_banner() -> str:
    return (f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
            f"CUDA {torch.version.cuda}")


class StageTimer:
    """Wall-clock per-stage timing via module hooks + bound-method wrappers.

    Synchronizes at stage boundaries, so it is for the e2e breakdown only —
    unhook before any tight-loop bench.
    """

    def __init__(self) -> None:
        self.stages: dict[str, list[float]] = {}
        self._starts: dict[str, float] = {}
        self._handles: list = []

    def _begin(self, name: str) -> None:
        torch.cuda.synchronize()
        self._starts[name] = time.perf_counter()

    def _end(self, name: str) -> None:
        torch.cuda.synchronize()
        self.stages.setdefault(name, []).append((time.perf_counter() - self._starts[name]) * 1e3)

    def hook_module(self, module: torch.nn.Module, name: str) -> None:
        self._handles.append(module.register_forward_pre_hook(lambda *_: self._begin(name)))
        self._handles.append(module.register_forward_hook(lambda *_: self._end(name)))

    def wrap_method(self, obj, attr: str, name: str) -> None:
        inner = getattr(obj, attr)

        def timed(*a, **kw):
            self._begin(name)
            try:
                return inner(*a, **kw)
            finally:
                self._end(name)

        setattr(obj, attr, timed)

    def reset(self) -> None:
        self.stages = {}

    def unhook(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def summary(self) -> dict[str, dict]:
        """Per-stage totals, plus the MEDIAN per call.

        The median is what the tables quote: a stage that is CUDA-graph
        recorded on its first timed call (the VAE is, if the warmups did not
        already cover it) costs several hundred milliseconds once and ~30 ms
        after, and a mean over three runs turns that into a number that
        describes neither.
        """
        return {name: {"calls": len(ms), "total_ms": round(sum(ms), 2),
                       "mean_ms": round(statistics.mean(ms), 2),
                       "median_ms": round(statistics.median(ms), 2)}
                for name, ms in self.stages.items()}


def compile_side_stages(pipe, mode: str = "max-autotune"):
    """Compile the VAE and both text encoders.

    They are not what is being compared, but leaving one eager would bury the
    comparison in tens of unrelated milliseconds. A failure here is reported
    and the stage stays eager rather than ending the run — the number it would
    have changed is the e2e total, not the denoise step.

    The text encoders deliberately skip inductor's CUDA graphs. SDXL runs both
    of them and concatenates their hidden states, and a captured encoder hands
    back a tensor owned by the graph pool: the second encoder's run overwrites
    the first one's output before `torch.concat` reads it, which torch reports
    as "accessing tensor output of CUDAGraphs that has been overwritten by a
    subsequent run". They are ~6 ms of the generation, so the capture is not
    worth the hazard.
    """
    print(f"[compile] vae.decode mode={mode}, text encoders mode={mode}-no-cudagraphs "
          "(the first e2e warmup pays the autotune)...")
    enc_mode = "max-autotune-no-cudagraphs"
    try:
        pipe.vae.decode = torch.compile(pipe.vae.decode, mode=mode)
        pipe.text_encoder = torch.compile(pipe.text_encoder, mode=enc_mode)
        pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2, mode=enc_mode)
    except Exception as e:
        print(f"[compile] side stages stayed eager: {type(e).__name__}: {str(e)[:200]}")


def run_e2e(pipe, prompt: str, seed: int, steps: int, guidance: float = 5.0,
            size: int = 1024):
    """One whole-image generation; returns (PIL image, wall ms)."""
    gen = torch.Generator("cuda").manual_seed(seed)
    # tells inductor's CUDA-graph trees that a new iteration starts here, so
    # tensors the last generation left in the graph pool are not read by this one
    torch.compiler.cudagraph_mark_step_begin()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    image = pipe(prompt=prompt, height=size, width=size, num_inference_steps=steps,
                 guidance_scale=guidance, generator=gen).images[0]
    torch.cuda.synchronize()
    return image, (time.perf_counter() - t0) * 1e3


def e2e_phase(pipe, args, label: str):
    """Warm the pipeline, time `--e2e-repeat` generations, save the last image.

    Returns (median ms, per-stage summary, output path).
    """
    # One CUDA-graph iteration per denoise step. Without it the pipeline's
    # `noise_pred` is still live when the next step calls the UNet, so
    # inductor's cudagraph tree cannot reuse its recording and re-records the
    # whole graph every step — several times the cost of the step itself. The
    # scheduler consumes `noise_pred` into fresh memory before the next call,
    # so marking the boundary here is sound.
    mark = pipe.unet.register_forward_pre_hook(
        lambda *_: torch.compiler.cudagraph_mark_step_begin())

    timer = StageTimer()
    # SDXL runs BOTH text encoders per prompt; the stage is their sum
    timer.hook_module(pipe.text_encoder, "text_encode")
    timer.hook_module(pipe.text_encoder_2, "text_encode")
    timer.hook_module(pipe.unet, "denoise_step")
    timer.wrap_method(pipe.vae, "decode", "vae_decode")

    for i in range(args.e2e_warmup):
        _, ms = run_e2e(pipe, args.prompt, args.seed, args.steps)
        print(f"[e2e warmup {i + 1}/{args.e2e_warmup}] {ms:.0f} ms")
    timer.reset()

    runs, image = [], None
    for i in range(args.e2e_repeat):
        image, ms = run_e2e(pipe, args.prompt, args.seed, args.steps)
        runs.append(ms)
        print(f"[e2e run {i + 1}/{args.e2e_repeat}] {ms:.0f} ms")

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    image.save(out)
    timer.unhook()
    mark.remove()
    return statistics.median(runs), timer.summary(), out, runs
