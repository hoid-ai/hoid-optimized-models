# bge-m3

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) dense embedding at batch 64 × seq 512, bf16, fully occupied, on the kernel stack hoid found for it: `best_torch.py` is the tuned stock baseline (its own `torch.compile` config lattice), `best_hoid.py` the hoid stack, gated on fp32-reference SNR, pooled cosine, and an identical retrieval ranking before it reports — both scripts end with a cross-lingual retrieval demo. Needs an NVIDIA GPU with CUDA 13 and `nvcc` on PATH (measured on a B200), and [uv](https://docs.astral.sh/uv/).

```bash
cd models/bge-m3
uv sync
uv run best_torch.py && uv run best_hoid.py
```

## Measured — NVIDIA B200 (CUDA 13.0, torch 2.13, bf16)

| metric | best torch | best hoid | ratio |
|---|---:|---:|---:|
| embedding throughput (seq/s) | 2424.9 | **2721.3** | **1.12x** |
| forward p50 (ms) | 26.39 | **23.52** | **1.12x** |
