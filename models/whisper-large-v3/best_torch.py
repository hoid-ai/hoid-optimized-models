#!/usr/bin/env python
"""Stock PyTorch whisper-large-v3, transcribing one 30 s window — the baseline.

Two modes, because they answer different questions:

  --mode static-cache   (default) the strongest torch configuration: a manual
      greedy loop over a StaticCache for self-attention plus a static
      cross-attention cache sized to the encoder output, with both encoder and
      decoder `torch.compile`d — constant shapes, so CUDA graphs capture. This
      is the protocol the 2.7876 ms decode
      step came from. It is the like-for-like bar for the hoid stack, which is
      also a manual loop over a static cache.
  --mode generate       `model.generate()`, which is what a user actually
      writes. It keeps a *dynamic* KV cache and pays HF's generation-loop
      overhead on every token, so it is slower — informative, but not the bar
      to claim a kernel win against.

Reports the four numbers that matter when one stream is being transcribed for a
person to read:

  TTFT                 features -> first token. What a live caption waits for.
  decode throughput    tokens/s in steady state. How fast the text then flows.
  window latency       features -> last token, for the whole 30 s window.
  RTFx                 audio seconds transcribed per wall second.

TTFT is reported twice: model-only, and including log-mel feature extraction —
the same work on any implementation, but not free, and a user's stopwatch sees
it.

  uv run best_torch.py
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
from common import SAMPLE_RATE, WINDOW_SECONDS, load_speech, mel_features  # noqa: E402

MAX_NEW_TOKENS = 224


def sync_ms(t0: float) -> float:
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def report(tag, feat_ms, ttft_ms, total_ms, n_tokens, transcript, stages=None, extra=None):
    """The shared result block — best_hoid.py prints the identical shape."""
    decode_ms = total_ms - ttft_ms
    tps = (n_tokens - 1) / (decode_ms / 1000) if n_tokens > 1 else float("nan")
    print(f"\n== {tag}: one 30 s window of real speech, batch 1, bf16 ==")
    print(f"{'TTFT (model only)':32s} {ttft_ms:9.1f} ms")
    print(f"{'TTFT (incl. log-mel features)':32s} {ttft_ms + feat_ms:9.1f} ms")
    print(f"{'decode throughput':32s} {tps:9.1f} tok/s   ({n_tokens} tokens)")
    print(f"{'window latency (feat -> last tok)':32s} {feat_ms + total_ms:9.1f} ms")
    print(f"{'RTFx (audio s / wall s)':32s} {WINDOW_SECONDS / ((feat_ms + total_ms) / 1000):9.1f}")
    print(f"{'  log-mel features':32s} {feat_ms:9.1f} ms")
    for k, v in (stages or {}).items():
        print(f"{'  ' + k:32s} {v:9.1f} ms")
    for k, v in (extra or {}).items():
        print(f"{k:32s} {v}")
    print(f"\ntranscript: {transcript}")
    return {"ttft_ms": round(ttft_ms, 2), "ttft_with_features_ms": round(ttft_ms + feat_ms, 2),
            "tokens_per_s": round(tps, 2), "tokens": n_tokens,
            "window_latency_ms": round(feat_ms + total_ms, 2),
            "rtfx": round(WINDOW_SECONDS / ((feat_ms + total_ms) / 1000), 2),
            "features_ms": round(feat_ms, 2), "transcript": transcript}


def build_static_cache(model):
    """StaticCache for self-attention + a cross-attention cache sized exactly to
    the encoder output. Constant shapes are what let compile capture one decode
    graph and cudagraphs record it; the geometry comes from the top-level config
    because whisper's decoder sub-config leaves the head fields at defaults."""
    from transformers import EncoderDecoderCache, StaticCache

    text_cfg = model.config.get_text_config(decoder=True)
    self_cache = StaticCache(config=text_cfg, max_cache_len=448)
    cross_cache = StaticCache(config=text_cfg, max_cache_len=model.config.max_source_positions)
    n_heads = model.config.decoder_attention_heads
    head_dim = model.config.d_model // n_heads
    for cache in (self_cache, cross_cache):
        cache.early_initialization(1, n_heads, head_dim, model.dtype, model.device)
    return EncoderDecoderCache(self_cache, cross_cache)


class StaticCacheGreedy:
    """The decode protocol: encode once, prefill the forced prefix,
    then one compiled step per token over the static cache."""

    def __init__(self, model, processor, config: str):
        from transformers.modeling_outputs import BaseModelOutput

        self._BMO = BaseModelOutput
        gc = model.generation_config
        self.model, self.processor = model, processor
        self.prompt = torch.tensor([[int(gc.decoder_start_token_id),
                                     int(gc.lang_to_id["<|en|>"]),
                                     int(gc.task_to_id["transcribe"]),
                                     int(gc.no_timestamps_token_id)]], device="cuda")
        self.eos = int(gc.eos_token_id)
        kwargs = C.TORCH_CONFIGS[config]
        torch._dynamo.reset()
        self.enc = model.model.encoder if kwargs is None else torch.compile(model.model.encoder, **kwargs)
        self.dec = model if kwargs is None else torch.compile(model, **kwargs)
        self.cache = build_static_cache(model)
        self.stages = {}

    def prefill(self, mel):
        """Encoder + the forced prefix + the first token."""
        self.cache.reset()
        t0 = time.perf_counter()
        enc_out = self._BMO(last_hidden_state=self.enc(mel).last_hidden_state.clone())
        self.stages["encoder"] = sync_ms(t0)
        t0 = time.perf_counter()
        out = self.dec(encoder_outputs=enc_out, decoder_input_ids=self.prompt,
                       past_key_values=self.cache, use_cache=True)
        cur = out.logits[:, -1:, :].argmax(-1)
        self.stages["prefix + first token"] = sync_ms(t0)
        return enc_out, cur

    def transcribe(self, mel):
        with torch.inference_mode():
            enc_out, cur = self.prefill(mel)
            ids = []
            for _ in range(MAX_NEW_TOKENS - 1):
                tok = int(cur.item())
                if tok == self.eos:
                    break
                ids.append(tok)
                out = self.dec(encoder_outputs=enc_out, decoder_input_ids=cur,
                               past_key_values=self.cache, use_cache=True)
                cur = out.logits[:, -1:, :].argmax(-1)
        return ids, self.processor.tokenizer.decode(ids, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="static-cache", choices=("static-cache", "generate"))
    ap.add_argument("--config", default="c4-max-autotune", choices=list(C.TORCH_CONFIGS))
    ap.add_argument("--repeat", type=int, default=5, help="timed passes (median kept)")
    ap.add_argument("--json", default="out/best_torch.json")
    args = ap.parse_args()

    print(f"[gpu] {C.gpu_banner()}")
    audio = load_speech()[:WINDOW_SECONDS * SAMPLE_RATE]      # real speech, first 30 s
    print(f"[audio] {len(audio) / SAMPLE_RATE:.1f} s of LibriSpeech, 16 kHz mono")

    model = C.load_model()
    model.generation_config.forced_decoder_ids = None
    model.generation_config.max_length = None
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(C.MODEL)
    if args.mode == "generate":
        kwargs = C.TORCH_CONFIGS[args.config]
        if kwargs is not None:
            print(f"[compile] encoder, {args.config} (first pass pays the autotune)")
            model.model.encoder.forward = torch.compile(model.model.encoder.forward, **kwargs)

    # log-mel: same work for any implementation, timed separately
    mel_features(audio)                                        # warm
    fs = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        mel = mel_features(audio)
        fs.append(sync_ms(t0))
    feat_ms = statistics.median(fs)
    mel = mel.to("cuda", torch.bfloat16)

    # ---- the strong bar: manual greedy loop over a static cache -------------
    if args.mode == "static-cache":
        print(f"[compile] encoder + decoder, {args.config}, static KV cache "
              f"(first passes pay the autotune)")
        runner = StaticCacheGreedy(model, processor, args.config)
        for _ in range(2):
            ids_list, text = runner.transcribe(mel)                 # warm
        ttfts, totals = [], []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            with torch.inference_mode():
                runner.prefill(mel)
            ttfts.append(sync_ms(t0))
        stages = dict(runner.stages)
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            ids_list, text = runner.transcribe(mel)
            totals.append(sync_ms(t0))
        row = report(f"torch {args.config}, static cache + compile", feat_ms,
                     statistics.median(ttfts), statistics.median(totals),
                     len(ids_list), text, stages=stages)
        if args.json:
            os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
            with open(args.json, "w") as f:
                json.dump({"impl": f"torch {args.config} static-cache",
                           "gpu": torch.cuda.get_device_name(0),
                           "stages_ms": {k: round(v, 3) for k, v in stages.items()},
                           **row}, f, indent=2)
            print(f"\n[json] {args.json}")
        return

    def gen(max_new):
        return model.generate(input_features=mel, do_sample=False, num_beams=1,
                              max_new_tokens=max_new, language="en", task="transcribe")

    with torch.no_grad():
        for _ in range(2):                                     # warm compile + cudagraphs
            gen(MAX_NEW_TOKENS)

        ttfts, totals = [], []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            gen(1)
            ttfts.append(sync_ms(t0))
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            ids = gen(MAX_NEW_TOKENS)
            totals.append(sync_ms(t0))

    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    # Count exactly what best_hoid.py counts: the tokens that become text.
    # transformers already strips the forced prefix from generate()'s output,
    # so this is a filter on special ids, not an offset.
    special = set(processor.tokenizer.all_special_ids)
    n_tokens = sum(1 for t in ids[0].tolist() if t not in special)

    row = report(f"torch {args.config}", feat_ms, statistics.median(ttfts),
                 statistics.median(totals), n_tokens, text)
    print("\nnote: this is `generate()`, which is what a user calls — its per-token cost\n"
          "      carries HF's framework overhead on top of the model. The isolated\n"
          "      static-cache decode step on this box measures 2.7876 ms\n"
          "      (c4-max-autotune).")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"impl": f"torch {args.config} generate()", "gpu": torch.cuda.get_device_name(0),
                       **row}, f, indent=2)
        print(f"\n[json] {args.json}")


if __name__ == "__main__":
    main()
