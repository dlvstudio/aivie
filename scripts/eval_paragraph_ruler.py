#!/usr/bin/env python3
"""
TK_035 / roadmap — LM ruler trên held-out WIKI/LỊCH SỬ (đúng domain VarLM đoạn).

Giống eval_lm_ruler.py nhưng corpus = đơn vị 2–4 câu từ history_wiki_para, split theo doc_id
(20% doc held-out, seed cố định). So sánh base-LM vs VarLM paragraph sidecar.

Chạy:
  PYTHONPATH=src python scripts/eval_paragraph_ruler.py
  PYTHONPATH=src python scripts/eval_paragraph_ruler.py --varlm-weights 0 0.2 0.35 0.5
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.data.paragraph_units import iter_units_from_doc_jsonl
from nn_aivie.eval.metrics import next_token_lm

_ROOT = Path(__file__).resolve().parents[1]
_HELDOUT = _ROOT / "data/eval_v4/heldout_paragraph_lm.jsonl"
_DEFAULT_CORPUS = _ROOT / "data/real_vi/history_wiki_para.jsonl"


def _doc_ids(path: Path) -> list[str]:
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            did = str(row.get("doc_id", ""))
            if did:
                seen.add(did)
    return sorted(seen)


def _build_heldout(
    corpus: Path,
    *,
    holdout_frac: float,
    seed: int,
    max_units: int,
) -> list[str]:
    if _HELDOUT.exists():
        return [json.loads(l)["text"] for l in _HELDOUT.read_text("utf-8").splitlines() if l.strip()]

    doc_ids = _doc_ids(corpus)
    rng = random.Random(seed)
    rng.shuffle(doc_ids)
    n_hold = max(1, int(len(doc_ids) * holdout_frac))
    hold_docs = set(doc_ids[:n_hold])

    # Lọc jsonl tạm chỉ doc held-out
    units: list[str] = []
    by_para: dict[tuple[str, int], list[str]] = {}
    with corpus.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            did = str(row.get("doc_id", ""))
            if did not in hold_docs:
                continue
            key = (did, int(row.get("para", 0)))
            by_para.setdefault(key, []).append(row.get("text", "") or "")

    tmp = _ROOT / "out/_heldout_para_filter.jsonl"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as out:
        for (did, para), sents in by_para.items():
            for s in sents:
                out.write(json.dumps({"text": s, "doc_id": did, "para": para}, ensure_ascii=False) + "\n")

    for unit in iter_units_from_doc_jsonl(tmp, min_sent=2, max_sent=4, dedup=True):
        units.append(unit)
        if len(units) >= max_units:
            break
    tmp.unlink(missing_ok=True)

    _HELDOUT.parent.mkdir(parents=True, exist_ok=True)
    _HELDOUT.write_text(
        "\n".join(json.dumps({"text": u}, ensure_ascii=False) for u in units),
        encoding="utf-8",
    )
    return units


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v9_histwiki_paragraph.pkl")
    ap.add_argument("--corpus", default=str(_DEFAULT_CORPUS))
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--max-units", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--varlm-weights", nargs="+", type=float, default=[0.0, 0.35, 0.5])
    args = ap.parse_args()

    corpus = Path(args.corpus)
    texts = _build_heldout(corpus, holdout_frac=args.holdout_frac, seed=args.seed, max_units=args.max_units)
    print(f"Held-out paragraph units: {len(texts)} ({_HELDOUT.relative_to(_ROOT)})")
    print(f"Checkpoint: {args.checkpoint}  (đang load…)")
    state, _ = load_checkpoint(_ROOT / args.checkpoint)
    has_varlm = getattr(state, "varlm", None) is not None
    print(f"vocab={len(state.vocab)}  parents={len(state.parents.parents)}  varlm={has_varlm}\n")

    print("=== LM ruler — wiki/lịch sử đoạn (base-LM, không agenda) ===")
    print(f"{'w':>6}  {'bits':>7}  {'top-1':>7}  {'MRR':>7}  n_pos")
    print("-" * 42)
    rows: list[str] = []
    for w in args.varlm_weights:
        if w > 0 and not has_varlm:
            print(f"{w:6.2f}  (skip — no varlm sidecar)")
            continue
        r = next_token_lm(state, texts, varlm_weight=w)
        line = f"{w:6.2f}  {r.bits:7.3f}  {r.top1:6.1%}  {r.mrr:7.3f}  {r.n_pos}"
        print(line)
        rows.append(line)

    out = _ROOT / "out/tk035_paragraph_ruler.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join([
        f"checkpoint={args.checkpoint}",
        f"heldout={_HELDOUT.name} n={len(texts)}",
        f"varlm={has_varlm}",
        "",
        *rows,
    ])
    out.write_text(body + "\n", encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
