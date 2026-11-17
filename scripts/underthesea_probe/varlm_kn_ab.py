#!/usr/bin/env python3
"""TK_031 bước 2 — A/B: λ-interpolation (scores) vs Kneser-Ney (scores_kn).

Standalone probe (whitespace tokenize, KHÔNG cần underthesea), protocol log2-rank giống
TK_031. Đo trên history_vi + statements để có mật độ ctx. Mục tiêu: xem KN có hạ bits held-out
so với nội-suy λ cố định không — cổng đo TRƯỚC khi wire vào runtime (measure-first, Luật 0 sạch).

Chạy: python scripts/underthesea_probe/varlm_kn_ab.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.varlm import VarOrderLM  # noqa: E402
from nn_aivie.teaching.align import tokenize  # noqa: E402

CORPORA = [
    _ROOT / "data/real_vi/statements.jsonl",
    _ROOT / "data/real_vi/statements_doc.jsonl",
    _ROOT / "data/real_vi/history_vi.jsonl",
]
OUT = _ROOT / "out/tk031_kn_ab.log"
MAX_CTX = 4
LAM = 0.45
HELD_FRAC = 0.1
SEED = 42
DISCOUNTS = [0.5, 0.75, 0.9]

_t0 = time.time()


def _log(msgs: list[str], m: str) -> None:
    line = f"[{time.time() - _t0:6.1f}s] {m}"
    print(line)
    msgs.append(line)


def _load_sentences() -> list[str]:
    texts: list[str] = []
    for cp in CORPORA:
        if not cp.exists():
            continue
        for line in cp.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = (obj.get("text") or obj.get("statement") or "").strip()
            if text:
                texts.append(text)
    return texts


def _build_vocab(seqs: list[list[str]]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for seq in seqs:
        for t in seq:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def _encode(seq: list[str], vocab: dict[str, int]) -> list[int]:
    return [vocab[t] for t in seq if t in vocab]


def _rank(scores: dict[int, float], gold: int) -> int | None:
    if not scores or gold not in scores:
        return None
    gs = scores[gold]
    return 1 + sum(1 for v in scores.values() if v > gs)


def _eval(lm: VarOrderLM, held: list[list[int]], mode: str, d: float) -> dict:
    sum_bits = sum_rr = 0.0
    t1 = t5 = 0
    n_pos = n_skip = 0
    for ids in held:
        for i in range(1, len(ids)):
            gold = ids[i]
            ctx = ids[max(0, i - lm.max_ctx) : i]
            scores = lm.scores(ctx) if mode == "lam" else lm.scores_kn(ctx, d=d)
            rank = _rank(scores, gold)
            if rank is None:
                n_skip += 1
                continue
            sum_bits += math.log2(rank)
            sum_rr += 1.0 / rank
            t1 += rank <= 1
            t5 += rank <= 5
            n_pos += 1
    if n_pos == 0:
        return {"bits": 0.0, "mrr": 0.0, "top1": 0.0, "top5": 0.0, "n_pos": 0, "n_skip": n_skip}
    return {
        "bits": sum_bits / n_pos,
        "mrr": sum_rr / n_pos,
        "top1": t1 / n_pos,
        "top5": t5 / n_pos,
        "n_pos": n_pos,
        "n_skip": n_skip,
    }


def main() -> None:
    import random

    msgs: list[str] = []
    sents = _load_sentences()
    random.Random(SEED).shuffle(sents)
    n_held = max(1, int(len(sents) * HELD_FRAC))
    train_sents, held_sents = sents[:-n_held], sents[-n_held:]

    ws_train = [tokenize(s) for s in train_sents]
    ws_held = [tokenize(s) for s in held_sents]
    vocab = _build_vocab(ws_train)
    _log(msgs, f"sentences={len(sents)} train={len(train_sents)} held={len(held_sents)} vocab={len(vocab)}")

    lm = VarOrderLM(max_ctx=MAX_CTX, lam=LAM)
    for s in ws_train:
        ids = _encode(s, vocab)
        if len(ids) >= 2:
            lm.add_sequence(ids)
    lm.prune(min_total=2, from_ctx=3)
    _log(msgs, f"lm built: {lm.stats()}")

    held_ids = [_encode(s, vocab) for s in ws_held]
    held_ids = [x for x in held_ids if len(x) >= 2]

    base = _eval(lm, held_ids, "lam", 0.0)
    _log(msgs, f"λ-interp (lam={LAM}): bits={base['bits']:.4f} MRR={base['mrr']:.3f} "
               f"top1={100*base['top1']:.1f}% top5={100*base['top5']:.1f}% n={base['n_pos']} skip={base['n_skip']}")

    best = None
    for d in DISCOUNTS:
        r = _eval(lm, held_ids, "kn", d)
        delta = r["bits"] - base["bits"]
        _log(msgs, f"KN d={d}: bits={r['bits']:.4f} ({delta:+.4f}) MRR={r['mrr']:.3f} "
                   f"top1={100*r['top1']:.1f}% top5={100*r['top5']:.1f}% n={r['n_pos']} skip={r['n_skip']}")
        if best is None or r["bits"] < best[1]["bits"]:
            best = (d, r)

    dbits = best[1]["bits"] - base["bits"]
    _log(msgs, f"BEST KN d={best[0]}: Δbits={dbits:+.4f} (âm = KN tốt hơn λ-interp)")
    _log(msgs, "XONG")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(msgs) + "\n", "utf-8")
    print(f"✓ wrote {OUT}")


if __name__ == "__main__":
    main()
