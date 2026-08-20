# Hoid optimized models

This repo highlights models where hoid-generated kernels achieve superior performance compared to PyTorch. Every model is pure PyTorch - the winning kernels run as torch custom ops (CUDA / Triton / CuTe DSL, JIT-compiled at import), with no hoid runtime involved. Each folder is self-sufficient: its own pinned `uv` environment, a `best_torch.py` that derives the strongest stock `torch.compile` configuration on your box, and a `best_hoid.py` that refuses to report performance until its correctness gates pass.

All numbers below were measured on an NVIDIA B200 (CUDA 13.0, torch 2.13.0+cu130, bf16); the kernels target Blackwell (sm100). Prerequisites: CUDA 13 with `nvcc` on PATH and [uv](https://docs.astral.sh/uv/). Weights download from the HF Hub on first run.

| model | workload | vs tuned torch |
|---|---|---|
| [qwen3-4b](models/qwen3-4b) | batch-1 serving | TTFT **1.59x** @ 1k / **1.44x** @ 8k, decode **1.18x** / **1.13x** |
| [llama31-8b](models/llama31-8b) | batch-1 serving | TTFT **1.30x** @ 1k / **1.22x** @ 8k, decode **1.03x** / **1.17x** |
| [qwen3-30b-a3b](models/qwen3-30b-a3b) | batch-1 serving (MoE) | TTFT **1.09x** @ 1k / **1.44x** @ 8k, decode **1.41x** / **1.48x** |
| [sdxl](models/sdxl) | one 1024×1024 image, 30 steps | image latency 709.8 → 618.4 ms, **1.15x** |
| [whisper-large-v3](models/whisper-large-v3) | transcribe a real 30 s window | latency 267.2 → 190.3 ms, **1.40x** |
| [bge-m3](models/bge-m3) | embed batch 64 × seq 512 | throughput 2425 → 2721 seq/s, **1.12x** |

To run any model:

```bash
cd models/<model>
uv sync
uv run best_torch.py && uv run best_hoid.py
```

Each model's README has the exact commands and the full measured table. The stock baseline is never a strawman: `best_torch.py` sweeps a whole configuration lattice (eager SDPA, compile default, reduce-overhead, max-autotune, …) and the hoid stack is compared against the per-metric best of it.
