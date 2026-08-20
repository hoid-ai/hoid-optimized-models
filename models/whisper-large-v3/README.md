# whisper-large-v3

[openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) transcribing one real 30 s speech window, batch 1, bf16 — encoder and decoder both on the kernel stack hoid found for them: `best_torch.py` is the tuned stock baseline (static KV cache, compiled), `best_hoid.py` the hoid stack, which refuses to report until its transcript is identical to stock HF `generate()` on the same audio. Needs an NVIDIA Blackwell GPU (sm100), CUDA 13 with `nvcc` on PATH, and [uv](https://docs.astral.sh/uv/).

```bash
cd models/whisper-large-v3
uv sync
uv run best_torch.py && uv run best_hoid.py
```

## Measured — NVIDIA B200 (CUDA 13.0, torch 2.13, bf16, bs=1)

| metric | best torch | best hoid | ratio |
|---|---:|---:|---:|
| transcribe the 30 s window (ms) | 267.2 | **190.3** | **1.40x** |
| TTFT incl. log-mel features (ms) | 11.44 | **8.14** | **1.41x** |
| decode throughput (tok/s) | 355.9 | **499.7** | **1.40x** |
