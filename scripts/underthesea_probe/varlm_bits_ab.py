#!/usr/bin/env python3
"""TK_032 — A/B VarOrderLM: whitespace vs underthesea tokens trên history_vi.jsonl."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.varlm import VarOrderLM  # noqa: E402
from nn_aivie.teaching.align import tokenize  # noqa: E402

CORPUS = _ROOT / "data/real_vi/history_vi.jsonl"
OUT = _ROOT / "out/tk032_varlm.log"
MAX_CTX = 4
LAM = 0.45
HELD_FRAC = 0.2


def _load_sentences() -> list[str]:
    texts: list[str] = []
    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = (obj.get("text") or obj.get("statement") or "").strip()
            if text:
                texts.append(text)
    return texts


def _tokenize_uts(text: str, uts_fn) -> list[str]:
    return uts_fn(text)


def _build_vocab(seqs: list[list[str]]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for seq in seqs:
        for t in seq:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def _encode(seq: list[str], vocab: dict[str, int]) -> list[int]:
    return [vocab[t] for t in seq if t in vocab]


def _train_lm(
    token_seqs: list[list[str]], max_ctx: int = MAX_CTX, lam: float = LAM
) -> tuple[VarOrderLM, dict[str, int]]:
    vocab = _build_vocab(token_seqs)
    lm = VarOrderLM(max_ctx=max_ctx, lam=lam)
    id_seqs = [_encode(s, vocab) for s in token_seqs]
    for ids in id_seqs:
        if len(ids) >= 2:
            lm.add_sequence(ids)
    lm.prune(min_total=2, from_ctx=3)
    return lm, vocab


def _eval_lm(
    lm: VarOrderLM,
    vocab: dict[str, int],
    held_seqs: list[list[str]],
) -> dict[str, float]:
    sum_bits = sum_rr = 0.0
    t1 = t5 = 0
    n_pos = 0
    for seq in held_seqs:
        ids = _encode(seq, vocab)
        for i in range(1, len(ids)):
            gold = ids[i]
            ctx = ids[:i]
            scores = lm.scores(ctx[-lm.max_ctx :])
            if not scores:
                continue
            ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
            rank = 1
            for tid, sc in ranked:
                if tid == gold:
                    break
                rank += 1
            else:
                rank = len(ranked) + 1
            sum_bits += math.log2(rank)
            sum_rr += 1.0 / rank
            t1 += rank <= 1
            t5 += rank <= 5
            n_pos += 1
    if n_pos == 0:
        return {"bits": 0.0, "mrr": 0.0, "top1": 0.0, "top5": 0.0, "n_pos": 0}
    return {
        "bits": sum_bits / n_pos,
        "mrr": sum_rr / n_pos,
        "top1": t1 / n_pos,
        "top5": t5 / n_pos,
        "n_pos": n_pos,
    }


def main() -> None:
    try:
        from underthesea import word_tokenize as uts_tokenize
    except ImportError as e:
        print(f"SKIP: underthesea not installed ({e})")
        sys.exit(1)

    sentences = _load_sentences()
    n_held = max(1, int(len(sentences) * HELD_FRAC))
    train_sents = sentences[:-n_held]
    held_sents = sentences[-n_held:]

    ws_train = [tokenize(s) for s in train_sents]
    uts_train = [_tokenize_uts(s, uts_tokenize) for s in train_sents]
    ws_held = [tokenize(s) for s in held_sents]
    uts_held = [_tokenize_uts(s, uts_tokenize) for s in held_sents]

    lm_ws, vocab_ws = _train_lm(ws_train)
    lm_uts, vocab_uts = _train_lm(uts_train)

    m_ws = _eval_lm(lm_ws, vocab_ws, ws_held)
    m_uts = _eval_lm(lm_uts, vocab_uts, uts_held)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as log:
        def w(msg: str = "") -> None:
            print(msg)
            log.write(msg + "\n")

        w(f"corpus: {CORPUS.relative_to(_ROOT)}")
        w(f"train sentences: {len(train_sents)} | held-out: {len(held_sents)}")
        w(f"VarOrderLM max_ctx={MAX_CTX} lam={LAM} (protocol: log2-rank, giống TK_031)")
        w(f"lm_ws tokens counted: {lm_ws.n_tokens} vocab={len(vocab_ws)} | {lm_ws.stats()}")
        w(f"lm_uts tokens: {lm_uts.n_tokens} vocab={len(vocab_uts)} | {lm_uts.stats()}")
        w("")
        w("=== HELD-OUT (whitespace tokenize) ===")
        w(
            f"n={int(m_ws['n_pos'])} bits={m_ws['bits']:.3f} "
            f"MRR={m_ws['mrr']:.3f} top1={100 * m_ws['top1']:.1f}% top5={100 * m_ws['top5']:.1f}%"
        )
        w("")
        w("=== HELD-OUT (underthesea word_tokenize) ===")
        w(
            f"n={int(m_uts['n_pos'])} bits={m_uts['bits']:.3f} "
            f"MRR={m_uts['mrr']:.3f} top1={100 * m_uts['top1']:.1f}% top5={100 * m_uts['top5']:.1f}%"
        )
        w("")
        delta = m_uts["bits"] - m_ws["bits"]
        w(f"Δbits (uts − ws) = {delta:+.3f} (âm = underthesea tốt hơn)")
        w("XONG")

    print(f"✓ wrote {OUT}")


if __name__ == "__main__":
    main()
