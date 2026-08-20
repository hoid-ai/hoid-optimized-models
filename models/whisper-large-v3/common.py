"""Shared plumbing for the scripts here: the model, the audio, and the clock.

The rule every script follows: torch and the hoid kernel stack see the same
model, the same weights, the same features and the same timing loop — only the
implementation under test differs.

Audio is one public LibriSpeech reading (`distil-whisper/librispeech_long`,
16 kHz mono, ~62 s), pulled from the HF Hub and cached there. The scripts
transcribe its first 30 s — real speech, so the transcript is a real transcript.
"""

from __future__ import annotations

import statistics
import time

import torch

# The accuracy column compares every bf16 config against the same model in
# float32, so that reference has to actually be float32: with TF32 left on,
# cuBLAS and cuDNN pick tensor-core paths for the fp32 encoder and the
# "reference" moves by dB between runs. bf16 paths are unaffected by these.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import functools
import io

import numpy as np

MODEL = "openai/whisper-large-v3"

# The stock configuration lattice for the baseline.
TORCH_CONFIGS = {
    "c1-eager-sdpa": None,
    "c2-compile-default": {"dynamic": False},
    "c3-reduce-overhead": {"mode": "reduce-overhead", "dynamic": False},
    "c4-max-autotune": {"mode": "max-autotune", "dynamic": False},
}


def load_model(dtype=torch.bfloat16, device="cuda"):
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL, dtype=dtype, attn_implementation="sdpa")
    return model.to(device).eval()


def hoid_encoder(model, fc2_mode: str = "hoid", mode: str = "capture", verbose: bool = True):
    """A callable mel[1,128,3000] -> hidden[1500,1280] on the hoid kernel stack.

    mode="capture"    CUDA-graph replay of the static-buffer forward (default)
    mode="static"     the same forward, launched from Python
    mode="functional" the torch custom ops, each allocating its own output —
                      the form `torch.compile` can trace

    Captured by default: the encoder is ~3.3 ms of GPU work over ~230 launches,
    so per-launch host cost is a first-order term, not a rounding error the way
    it is in a 50 ms diffusion step.
    """
    import whisper_optimized as WO

    WO.register_ops(verbose=verbose)
    opt = WO.OptimizedWhisperEncoder(model.model.encoder, fc2_mode=fc2_mode)
    if mode == "functional":
        return opt
    if mode == "static":
        return opt.forward_static
    if mode != "capture":
        raise ValueError("mode must be capture | static | functional")
    captured = WO.CapturedEncoder(opt)

    def run(mel):
        return captured(mel)

    return run


def gpu_banner() -> str:
    return (f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
            f"CUDA {torch.version.cuda}")


# ---------------------------------------------------------------------------
# Audio: one real reading, its log-mel windows
# ---------------------------------------------------------------------------
AUDIO_REPO = "distil-whisper/librispeech_long"
AUDIO_FILE = "clean/validation-00000-of-00001-913508124a40cb97.parquet"
SAMPLE_RATE = 16000
WINDOW_SECONDS = 30


@functools.lru_cache(maxsize=1)
def load_speech() -> np.ndarray:
    """The source clip as float32 mono at 16 kHz."""
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(AUDIO_REPO, AUDIO_FILE, repo_type="dataset")
    row = pq.read_table(path).to_pylist()[0]
    data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"expected {SAMPLE_RATE} Hz source audio, got {sr}")
    return data


@functools.lru_cache(maxsize=1)
def feature_extractor():
    from transformers import AutoFeatureExtractor

    return AutoFeatureExtractor.from_pretrained(MODEL)


def mel_features(audio: np.ndarray, device: str = "cuda"):
    """log-mel features [1, 128, 3000] for one 30 s window (padded/truncated).

    Runs the STFT on the GPU when one is available — the numpy path costs
    ~800 ms per window on this box, which would dwarf every number here.
    """
    import torch

    fe = feature_extractor()
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"
    out = fe(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", device=device)
    return out.input_features.to(torch.float32)


