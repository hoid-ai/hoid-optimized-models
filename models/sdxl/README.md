# SDXL base 1.0

[stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) generating a 1024×1024 image (30 steps, CFG, bf16) on the kernel stack hoid found for its UNet and VAE decoder: `best_torch.py` is the tuned stock baseline (its own `torch.compile` config lattice), `best_hoid.py` the hoid stack, gated on UNet SNR ≥ 30 dB and image PSNR ≥ 25 dB vs the stock generation before it reports — both scripts end by saving a real image to `out/`. Needs an NVIDIA Blackwell GPU (sm100), CUDA 13 with `nvcc` on PATH, and [uv](https://docs.astral.sh/uv/).

```bash
cd models/sdxl
uv sync
uv run best_torch.py && uv run best_hoid.py
```

## Measured — NVIDIA B200 (CUDA 13.0, torch 2.13, bf16)

| metric | best torch | best hoid | ratio |
|---|---:|---:|---:|
| one image, 30 steps (ms) | 709.8 | **618.4** | **1.15x** |
| UNet forward (ms) | 21.57 | **18.57** | **1.16x** |
| VAE decode (ms) | 21.32 | **18.43** | **1.16x** |
