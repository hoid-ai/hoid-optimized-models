"""Shared fixtures and measurement for the bge-m3 head-to-head.

The frozen workload is the one the whole campaign was measured on: batch 64 x
seq 512, fully occupied (no padding, all-ones mask), bf16. `DOC_TEXT` and the
per-row roll are frozen — the hoid kernels were timed on exactly this
batch, or the comparison is void.
"""

from __future__ import annotations

import statistics
import time

import torch

REPO = "BAAI/bge-m3"
BATCH = 64
SEQ = 512

DOC_TEXT = (
    "Dense retrieval systems encode queries and passages into a shared vector "
    "space where semantic similarity becomes geometric proximity. A multilingual "
    "encoder must place a sentence and its translation near each other while "
    "keeping unrelated passages far apart, which requires both a large subword "
    "vocabulary and a long context window. Training such a model combines "
    "contrastive objectives over mined hard negatives with knowledge distillation "
    "from a stronger teacher, and evaluation spans passage ranking, cross-lingual "
    "retrieval, and long-document search. At inference the cost is dominated by a "
    "single bidirectional forward pass over the full sequence, so throughput is "
    "governed by attention and feed-forward matrix multiplications rather than by "
    "any autoregressive loop. "
)


def tile_ids(tok, text: str) -> torch.Tensor:
    """One text as a fully-occupied SEQ-token row: <s> + the text repeated to
    fill + </s>. The workload (and the hoid kernels) assume no padding, so
    shorter texts are tiled rather than padded — and the <s> matters, because
    bge-m3's dense vector IS the CLS (<s>) position."""
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    body = ids.repeat((SEQ - 2) // ids.numel() + 1)[:SEQ - 2]
    return torch.cat([torch.tensor([tok.bos_token_id]), body,
                      torch.tensor([tok.eos_token_id])])


def frozen_inputs(tok, device="cuda") -> dict[str, torch.Tensor]:
    """The frozen batch: DOC_TEXT tiled, row r rolled by 7*r."""
    ids = tok(DOC_TEXT, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if ids.numel() < SEQ:
        ids = ids.repeat((SEQ // ids.numel()) + 1)
    rows = [ids.roll(int(r * 7)).clone()[:SEQ] for r in range(BATCH)]
    input_ids = torch.stack(rows).to(device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


# ---------------------------------------------------------------------------
# Retrieval demo: 8 cross-lingual query/passage pairs + 48 distractor rows,
# embedded in ONE batch-64 forward (exactly the timed workload). The meaningful
# output is the ranking; the task-level gate is that both stacks retrieve the
# same passage for every query.
# ---------------------------------------------------------------------------
DEMO_PAIRS = [
    ("How do solar panels convert sunlight into electricity?",
     "Photovoltaic cells absorb photons and release electrons, generating a "
     "direct current that an inverter converts to alternating current."),
    ("Wie funktioniert der Wasserkreislauf der Erde?",
     "Wasser verdunstet aus Ozeanen, kondensiert zu Wolken und kehrt als "
     "Niederschlag zurueck, der Fluesse und Grundwasser speist."),
    ("Quels sont les avantages du vaccin contre la grippe ?",
     "La vaccination annuelle reduit le risque d'infection grippale et attenue "
     "la gravite des symptomes chez les personnes vaccinees."),
    ("What causes inflation in an economy?",
     "When aggregate demand outpaces supply or the money supply grows faster "
     "than output, the general price level rises across the economy."),
    ("Como funciona la fotosintesis en las plantas?",
     "Los cloroplastos capturan la luz solar y transforman dioxido de carbono "
     "y agua en glucosa, liberando oxigeno en el proceso."),
    ("What is the role of the hippocampus in memory?",
     "The hippocampus consolidates short-term experiences into long-term "
     "declarative memories and supports spatial navigation."),
    ("Warum ist der Himmel blau?",
     "Kurzwelliges blaues Licht wird an den Molekuelen der Atmosphaere staerker "
     "gestreut als langwelliges rotes Licht, daher erscheint der Himmel blau."),
    ("How does a lithium-ion battery store energy?",
     "Lithium ions shuttle between anode and cathode through the electrolyte, "
     "storing energy during charging and releasing it on discharge."),
]


def demo_batch(tok, device="cuda"):
    """(input_ids [64,512], query_rows, passage_rows). Rows 0..7 queries,
    8..15 their passages, 16..63 distractors (rolled DOC_TEXT)."""
    rows = [tile_ids(tok, q) for q, _ in DEMO_PAIRS]
    rows += [tile_ids(tok, p) for _, p in DEMO_PAIRS]
    base = tok(DOC_TEXT, add_special_tokens=False, return_tensors="pt").input_ids[0]
    base = base.repeat((SEQ // base.numel()) + 2)
    bos = torch.tensor([tok.bos_token_id])
    eos = torch.tensor([tok.eos_token_id])
    rows += [torch.cat([bos, base.roll(11 * (r + 3))[:SEQ - 2], eos])
             for r in range(BATCH - len(rows))]
    ids = torch.stack(rows).to(device)
    n = len(DEMO_PAIRS)
    return ids, list(range(n)), list(range(n, 2 * n))


def retrieval_top1(pooled: torch.Tensor, query_rows: list[int]) -> list[int]:
    """For each query row, the best-scoring non-query row (cosine on the
    already-normalized embeddings)."""
    sims = pooled @ pooled.T
    top1 = []
    for q in query_rows:
        s = sims[q].clone()
        s[query_rows] = -2.0
        top1.append(int(s.argmax()))
    return top1


def retrieval_rows(pooled: torch.Tensor, query_rows, passage_rows) -> list[dict]:
    """The demo's evidence, JSON-shaped: each query's text, the text it
    retrieved, the cosine, and whether that is its paired passage."""
    top1 = retrieval_top1(pooled, query_rows)
    sims = pooled @ pooled.T
    n = len(query_rows)
    rows = []
    for i, q in enumerate(query_rows):
        got = top1[i]
        text = (DEMO_PAIRS[got - n][1] if n <= got < 2 * n
                else "(a distractor row: rolled corpus text)")
        rows.append({"query": DEMO_PAIRS[i][0],
                     "retrieved": text,
                     "cosine": round(float(sims[q, got]), 4),
                     "is_paired_passage": got == passage_rows[i]})
    return rows


def print_retrieval(pooled: torch.Tensor, query_rows, passage_rows) -> None:
    rows = retrieval_rows(pooled, query_rows, passage_rows)
    hits = 0
    for i, r in enumerate(rows):
        hits += r["is_paired_passage"]
        print(f"  {'OK ' if r['is_paired_passage'] else 'MISS'} q{i}: "
              f"{r['query'][:52]!r}\n"
              f"       -> (cos {r['cosine']:.4f}) {r['retrieved'][:90]!r}")
    print(f"  {hits}/{len(rows)} queries retrieved their paired passage")


# ---------------------------------------------------------------------------
# Scoring and timing
# ---------------------------------------------------------------------------

def pool_cls(hidden: torch.Tensor) -> torch.Tensor:
    """bge-m3 dense vector: CLS token, L2-normalized, fp32."""
    return torch.nn.functional.normalize(hidden[:, 0].float(), p=2, dim=-1)


def row_cosines(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1)


def snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    import math

    r, g = ref.double(), got.double()
    den = ((r - g) ** 2).sum().item()
    if den == 0:
        return math.inf
    return 10.0 * math.log10((r * r).sum().item() / den)


def bench(fn, iters=50, warmup=10) -> dict[str, float]:
    """Wall-clock stats over synchronized iterations: mean/min/p50 in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return {"mean": statistics.mean(times), "min": min(times),
            "p50": statistics.median(times)}


def fmt(stats: dict[str, float]) -> str:
    return f"{stats['mean']:8.3f} ms mean / {stats['min']:.3f} min / {stats['p50']:.3f} p50"
