# Llama-3.1-8B-Instruct

[Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) served batch-1 on the kernel stack hoid found for it, as pure PyTorch custom ops: `best_torch.py` is the tuned stock baseline (its own `torch.compile` config lattice), `best_hoid.py` the hoid stack, gated on 128/128 greedy-token parity with HF before it reports. Needs an NVIDIA Blackwell GPU (sm100), CUDA 13 with `nvcc` on PATH, and [uv](https://docs.astral.sh/uv/).

```bash
cd models/llama31-8b
uv sync
uv run best_torch.py && uv run best_hoid.py
```

Defaults measure the 8192-token prompt; add `--prompt-tokens 1024` to both for the 1k rows. Weights default to the ungated `NousResearch/Meta-Llama-3.1-8B-Instruct` mirror (`--weights` overrides).

## Measured — NVIDIA B200 (CUDA 13.0, torch 2.13, bf16, bs=1)

| metric | best torch | best hoid | ratio |
|---|---:|---:|---:|
| TTFT @ 1k (ms) | 19.08 | **14.72** | **1.30x** |
| decode @ 1k (tok/s) | 209.3 | **215.7** | **1.03x** |
| TTFT @ 8k (ms) | 145.34 | **119.53** | **1.22x** |
| decode @ 8k (tok/s) | 159.2 | **185.6** | **1.17x** |
