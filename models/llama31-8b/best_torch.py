#!/usr/bin/env python3
"""The strong plain-PyTorch baseline for Llama-3.1-8B-Instruct generate.

Sweeps the stock torch.compile config lattice (minus the two
cudagraph modes, which fail to compile on this torch/triton — recorded, not
hidden), measures prefill and decode separately with a static KV cache, and
reports the per-phase best. On B200 at an 8k prompt the per-phase winners
differ: eager cuDNN SDPA wins prefill while max-autotune wins decode — the
JSON records both, and `best` picks per phase.

    uv run best_torch.py [--prompt-tokens 8192] [--gen-tokens 128]
                         [--weights DIR] [--configs c1,c2,c4nc]
"""

from __future__ import annotations

import argparse

import torch

import common as C


def make_forward(model, mode: str | None):
    if mode is None:
        return model
    return torch.compile(model, mode=mode, fullgraph=True, dynamic=False)


CONFIGS = {
    "c1-eager-sdpa": None,
    "c2-compile-default": "default",
    "c4-max-autotune-no-cudagraphs": "max-autotune-no-cudagraphs",
}


def fresh_cache(model, max_len: int):
    from transformers import StaticCache

    return StaticCache(
        config=model.config, max_batch_size=1, max_cache_len=max_len,
        device="cuda", dtype=torch.bfloat16,
    )


@torch.no_grad()
def prefill_once(forward, model, ids, cache):
    """One prompt forward. logits_to_keep=1 (what HF generate() itself passes)
    projects only the last position — the strongest honest TTFT for torch."""
    cache_pos = torch.arange(ids.numel(), device="cuda")
    kw = dict(
        input_ids=ids.unsqueeze(0), past_key_values=cache,
        cache_position=cache_pos, use_cache=True,
    )
    try:
        out = forward(**kw, logits_to_keep=1)
    except TypeError:
        out = forward(**kw)
    return out.logits[0, -1].argmax()


@torch.no_grad()
def decode_step(forward, tok, cache, pos_scalar):
    out = forward(
        input_ids=tok.view(1, 1), past_key_values=cache,
        cache_position=pos_scalar, use_cache=True,
    )
    return out.logits[0, -1].argmax()


@torch.no_grad()
def measure_config(name, mode, model, ids, gen_tokens, prefill_reps):
    P = ids.numel()
    max_len = P + gen_tokens + 2
    forward = make_forward(model, mode)

    # correctness + compile warmup: 32 greedy tokens
    cache = fresh_cache(model, max_len)
    tok = prefill_once(forward, model, ids, cache)
    toks = [int(tok)]
    for j in range(31):
        pos = torch.tensor([P + j], device="cuda")
        tok = decode_step(forward, tok, cache, pos)
        toks.append(int(tok))

    # prefill: fresh static cache per call
    def one_prefill():
        c = fresh_cache(model, max_len)
        prefill_once(forward, model, ids, c)

    prefill_ms = C.cuda_time(one_prefill, reps=prefill_reps, warmup=1)

    # decode: steady state at ~P context (cache sized for warmup + timed steps,
    # which can exceed gen_tokens)
    steps = min(gen_tokens, 64)
    cache = fresh_cache(model, P + 8 + steps + 2)
    tok = prefill_once(forward, model, ids, cache)
    pos = torch.tensor([0], device="cuda")
    state = {"j": 0, "tok": tok}

    def one_step():
        pos.fill_(P + state["j"])
        state["tok"] = decode_step(forward, state["tok"], cache, pos)
        state["j"] += 1

    for _ in range(8):
        one_step()
    d = []
    for _ in range(steps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record(); one_step(); b.record()
        torch.cuda.synchronize()
        d.append(a.elapsed_time(b))

    # e2e: one full generate
    def e2e():
        c = fresh_cache(model, max_len)
        t = prefill_once(forward, model, ids, c)
        for j in range(gen_tokens - 1):
            p = torch.tensor([P + j], device="cuda")
            t = decode_step(forward, t, c, p)

    e2e_ms = C.wall_ms(e2e)

    return {
        "prefill_p50_ms": round(C.p50(prefill_ms), 3),
        "decode_p50_ms": round(C.p50(d), 4),
        "e2e_ms": round(e2e_ms, 1),
        "greedy32": toks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", type=int, default=8192)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--weights", default=C.DEFAULT_WEIGHTS)
    ap.add_argument("--configs", default="c1-eager-sdpa,c2-compile-default,c4-max-autotune-no-cudagraphs")
    args = ap.parse_args()

    C.strict_matmul_precision()
    print(f"[{C.gpu_banner()}]")
    tokzr = C.load_tokenizer(args.weights)
    ids = C.build_prompt_ids(tokzr, args.prompt_tokens).to("cuda")
    print(f"[prompt] {args.prompt_tokens} tokens, this passage tiled: "
          f"{C.prompt_preview(tokzr, ids)!r}")
    model = C.load_hf_model(args.weights)

    results, ref32 = {}, None
    prefill_reps = 3 if args.prompt_tokens >= 4096 else 7
    for name in args.configs.split(","):
        mode = CONFIGS[name]
        torch.compiler.reset()
        print(f"[cfg] {name} ...")
        m = measure_config(name, mode, model, ids, args.gen_tokens, prefill_reps)
        if ref32 is None:
            ref32 = m["greedy32"]
        m["tokens_match_eager"] = m["greedy32"] == ref32
        results[name] = m
        print(f"      prefill {m['prefill_p50_ms']} ms | decode {m['decode_p50_ms']} ms"
              f" | e2e {m['e2e_ms']} ms | tokens match: {m['tokens_match_eager']}")
        if not m["tokens_match_eager"]:
            print("      NOTE: greedy tokens differ from eager — excluded from best")

    ok = {k: v for k, v in results.items() if v["tokens_match_eager"]}
    best = {
        "prefill_p50_ms": min(v["prefill_p50_ms"] for v in ok.values()),
        "prefill_config": min(ok, key=lambda k: ok[k]["prefill_p50_ms"]),
        "decode_p50_ms": min(v["decode_p50_ms"] for v in ok.values()),
        "decode_config": min(ok, key=lambda k: ok[k]["decode_p50_ms"]),
        "e2e_ms": min(v["e2e_ms"] for v in ok.values()),
        "e2e_config": min(ok, key=lambda k: ok[k]["e2e_ms"]),
    }
    C.report("stock torch, best per phase", args.prompt_tokens, args.gen_tokens, best)
    print(f"  prefill winner: {best['prefill_config']}, decode winner: {best['decode_config']},"
          f" e2e winner: {best['e2e_config']}")

    output_text = tokzr.decode(ref32)
    print(f"[output] 32 greedy tokens (every config in `best` matches these): "
          f"{output_text!r}")
    for v in results.values():
        v.pop("greedy32")
    C.write_json("out/best_torch.json", {
        "workload": {"prompt_tokens": args.prompt_tokens, "gen_tokens": args.gen_tokens},
        "torch": torch.__version__,
        "prompt_preview": C.prompt_preview(tokzr, ids),
        "output_text": output_text,
        "configs": results,
        "best": best,
    })


if __name__ == "__main__":
    main()
