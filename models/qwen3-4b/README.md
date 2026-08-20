# Qwen3-4B-Instruct-2507

[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) served batch-1 on the kernel stack hoid found for it, as pure PyTorch custom ops: `best_torch.py` is the tuned stock baseline (its own `torch.compile` config lattice), `best_hoid.py` the hoid stack, gated on 128/128 greedy-token parity with HF before it reports. Needs an NVIDIA Blackwell GPU (sm100), CUDA 13 with `nvcc` on PATH, and [uv](https://docs.astral.sh/uv/).

```bash
cd models/qwen3-4b
uv sync
uv run best_torch.py && uv run best_hoid.py
```

Defaults measure the 8192-token prompt; add `--prompt-tokens 1024` to both for the 1k rows.

## Measured — NVIDIA B200 (CUDA 13.0, torch 2.13, bf16, bs=1)

| metric | best torch | best hoid | ratio |
|---|---:|---:|---:|
| TTFT @ 1k (ms) | 15.24 | **9.58** | **1.59x** |
| decode @ 1k (tok/s) | 228.0 | **269.8** | **1.18x** |
| TTFT @ 8k (ms) | 120.00 | **83.33** | **1.44x** |
| decode @ 8k (tok/s) | 172.4 | **195.0** | **1.13x** |
