# Qwen3-30B-A3B-Instruct-2507

[Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) (MoE, 3B active) served batch-1 on the kernel stack hoid found for it — the whole forward captured as one CUDA graph per prompt length: `best_torch.py` is the tuned stock baseline, `best_hoid.py` the hoid stack, gated at correctness 1.000 vs the eager StaticCache reference before it reports. Needs an NVIDIA Blackwell GPU (sm100), CUDA 13 with `nvcc` on PATH, and [uv](https://docs.astral.sh/uv/).

```bash
cd models/qwen3-30b-a3b
uv sync
uv run best_torch.py --config sweep && uv run best_hoid.py
```

Defaults measure the 1024-token prompt; add `--prompt-tokens 8192` to both for the 8k rows.

## Measured — NVIDIA B200 (CUDA 13.0, torch 2.13, bf16, bs=1)

| metric | best torch | best hoid | ratio |
|---|---:|---:|---:|
| TTFT @ 1k (ms) | 31.31 | **28.66** | **1.09x** |
| decode @ 1k (tok/s) | 171.6 | **241.1** | **1.41x** |
| TTFT @ 8k (ms) | 207.90 | **144.76** | **1.44x** |
| decode @ 8k (tok/s) | 127.6 | **188.6** | **1.48x** |
