#!/usr/bin/env python
"""whisper-large-v3 on the hoid kernel stack, transcribing one 30 s window.

The counterpart to best_torch.py: same audio, same greedy decoding, same forced
prefix and suppression, same reported numbers — but the model runs the kernels
the hoid optimization runs found, in three captured CUDA graphs:

  1. ENCODER — whisper_optimized.py, 3.3 ms/window. Triton conv frontend,
     Blackwell CuTe DSL packed-QKV and fc1 GEMMs, flash-attn 4, fused
     residual+LayerNorm. 1.23x over the best stock encoder config.

  2. PREFILL — the window's cross-attention K/V for all 32 layers, plus the
     4-token forced prefix in ONE pass, plus the first token's logits. 3.9 ms.
     This part is deliberately NOT the hoid decode kernels: they are shaped for
     a single token, so running a 4-token prefix through them would read all
     1.6 GB of decoder weights four times. It is plain cuBLAS at M=4 and SDPA,
     writing k/v straight into the hoid cache layout, `torch.compile`d before
     capture (at M=4 inductor has something to fuse) with q/k/v and cross-K/V
     packed into one GEMM per layer.

  3. DECODE STEP — whisper_decoder_optimized.py, 1.74 ms/token captured. Five
     CUDA kernels and four Triton split-K flash-decode kernels over a 448-slot
     KV cache and the 1500 cached encoder positions. 1.60x over torch's tuned
     static-cache step.

Before reporting anything, the transcript is checked against stock HF
`generate()` on the same audio. A faster transcriber that says something else
is not a result.

  uv run best_torch.py     # writes out/best_torch.json
  uv run best_hoid.py      # prints its own numbers and the comparison
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import common as C  # noqa: E402
import whisper_decoder_optimized as WD  # noqa: E402
from common import SAMPLE_RATE, WINDOW_SECONDS, load_speech, mel_features  # noqa: E402
from best_torch import MAX_NEW_TOKENS, report, sync_ms  # noqa: E402

VOCAB = WD.VOCAB


class HoidWhisper:
    """Encoder and decoder on the hoid stack, both CUDA-graphed."""

    def __init__(self, model, processor, verbose: bool = True):
        gc = model.generation_config
        self.processor = processor
        self.prompt = [int(gc.decoder_start_token_id), int(gc.lang_to_id["<|en|>"]),
                       int(gc.task_to_id["transcribe"]), int(gc.no_timestamps_token_id)]
        self.eos = int(gc.eos_token_id)
        # HF applies these inside generate(); applying them here is what makes
        # the two transcripts comparable. Additive masks keep the per-token host
        # work to one add and one argmax.
        self.mask = torch.zeros(VOCAB, device="cuda", dtype=torch.float32)
        if gc.suppress_tokens:
            self.mask[torch.tensor(list(gc.suppress_tokens), device="cuda")] = float("-inf")
        self.begin_mask = self.mask.clone()
        if gc.begin_suppress_tokens:
            self.begin_mask[torch.tensor(list(gc.begin_suppress_tokens), device="cuda")] = float("-inf")
        self.encoder = C.hoid_encoder(model, verbose=verbose)
        self.decoder = WD.OptimizedWhisperDecoder(model)
        self.decoder.capture()                      # the single-token step
        self.decoder.capture_prefill(self.prompt)   # cross-K/V + the whole prefix
        self.stages = {}

    def prefill(self, mel: torch.Tensor):
        """Everything up to and including the first token: the encoder, then one
        replay that projects the window's cross-attention K/V and runs the whole
        4-token forced prefix in a single pass."""
        d = self.decoder
        t0 = time.perf_counter()
        hidden = self.encoder(mel)
        self.stages["encoder"] = sync_ms(t0)
        t0 = time.perf_counter()
        logits = d.prefill(hidden)
        first = int((logits[0] + self.begin_mask).argmax())
        self.stages["cross-K/V + prefix + 1st token"] = sync_ms(t0)
        return first

    def transcribe(self, mel: torch.Tensor):
        """Greedy to EOS. Returns (token ids, text)."""
        d = self.decoder
        tok = self.prefill(mel)
        ids = []
        for i in range(MAX_NEW_TOKENS):
            if tok == self.eos:
                break
            ids.append(tok)
            logits = d.step(tok, len(self.prompt) + i)
            tok = int((logits[0] + self.mask).argmax())
        return ids, self.processor.tokenizer.decode(ids, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", type=int, default=5, help="timed passes (median kept)")
    ap.add_argument("--skip-check", action="store_true",
                    help="skip the transcript check against stock HF generate()")
    ap.add_argument("--json", default="out/best_hoid.json")
    ap.add_argument("--compare", default="out/best_torch.json",
                    help="best_torch.py's json, for the comparison table")
    args = ap.parse_args()

    from transformers import AutoProcessor

    print(f"[gpu] {C.gpu_banner()}")
    audio = load_speech()[:WINDOW_SECONDS * SAMPLE_RATE]
    print(f"[audio] {len(audio) / SAMPLE_RATE:.1f} s of LibriSpeech, 16 kHz mono")
    processor = AutoProcessor.from_pretrained(C.MODEL)

    model = C.load_model()
    model.generation_config.forced_decoder_ids = None
    model.generation_config.max_length = None

    mel_features(audio)                                        # warm
    fs = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        mel_cpu = mel_features(audio)
        fs.append(sync_ms(t0))
    feat_ms = statistics.median(fs)
    mel = mel_cpu.to("cuda", torch.bfloat16)[0].contiguous()

    # -- the reference transcript, before anything is timed --------------------
    ref_text = None
    if not args.skip_check:
        with torch.no_grad():
            ids = model.generate(input_features=mel.unsqueeze(0), do_sample=False, num_beams=1,
                                 max_new_tokens=MAX_NEW_TOKENS, language="en", task="transcribe")
        ref_text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    print("[build] compiling the hoid kernel stack (nvcc, triton, cute) ...")
    hoid = HoidWhisper(model, processor)

    ids, text = hoid.transcribe(mel)                           # warm
    if ref_text is not None:
        ok = text == ref_text
        print(f"[check] transcript vs stock HF generate(): {'identical' if ok else 'DIFFERS'}")
        if not ok:
            import difflib
            a, b = ref_text.split(), text.split()
            sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
            agree = sum(bl.size for bl in sm.get_matching_blocks()) / max(len(a), len(b), 1)
            print(f"        word agreement {agree * 100:.2f}% ({len(a)} vs {len(b)} words)")
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag != "equal":
                    print(f"          {tag}: {' '.join(a[i1:i2])!r} vs {' '.join(b[j1:j2])!r}")
        assert ok, "hoid transcript differs from stock HF — not a result"

    ttfts, totals = [], []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        hoid.prefill(mel)
        ttfts.append(sync_ms(t0))
    stages = dict(hoid.stages)
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        ids, text = hoid.transcribe(mel)
        totals.append(sync_ms(t0))

    row = report("hoid kernel stack", feat_ms, statistics.median(ttfts),
                 statistics.median(totals), len(ids), text, stages=stages)

    # -- the comparison, if best_torch.py has been run ------------------------
    if args.compare and os.path.exists(args.compare):
        with open(args.compare) as f:
            t = json.load(f)
        print(f"\n== vs {t['impl']} ==")
        print(f"{'':34s}{'torch':>12s}{'hoid':>12s}{'':4s}{'ratio':>8s}")
        for key, label, lower_better in (
            ("ttft_ms", "TTFT (model only) ms", True),
            ("ttft_with_features_ms", "TTFT incl. features ms", True),
            ("tokens_per_s", "decode throughput tok/s", False),
            ("window_latency_ms", "window latency ms", True),
            ("rtfx", "RTFx", False),
        ):
            a, b = t[key], row[key]
            ratio = (a / b) if lower_better else (b / a)
            print(f"{label:34s}{a:12.1f}{b:12.1f}{'':4s}{ratio:7.2f}x")
        per_t = (t["window_latency_ms"] - t["ttft_with_features_ms"]) / max(t["tokens"] - 1, 1)
        per_h = (row["window_latency_ms"] - row["ttft_with_features_ms"]) / max(row["tokens"] - 1, 1)
        print(f"{'ms per decode token':34s}{per_t:12.2f}{per_h:12.2f}{'':4s}{per_t / per_h:7.2f}x")
        print(f"{'tokens generated':34s}{t['tokens']:12d}{row['tokens']:12d}"
              f"{'':4s}{'same' if t['tokens'] == row['tokens'] else 'DIFFER':>8s}")
        if t.get("stages_ms"):
            print(f"\n  torch TTFT: " + ", ".join(f"{k} {v:.2f} ms" for k, v in t["stages_ms"].items()))
        print(f"  hoid  TTFT: " + ", ".join(f"{k} {v:.2f} ms" for k, v in stages.items()))
        print(f"  transcripts identical: {t['transcript'] == row['transcript']}")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"impl": "hoid kernel stack", "gpu": torch.cuda.get_device_name(0),
                       "stages_ms": {k: round(v, 3) for k, v in stages.items()},
                       "transcript_matches_hf": ref_text is None or text == ref_text,
                       **row}, f, indent=2)
        print(f"\n[json] {args.json}")


if __name__ == "__main__":
    main()
