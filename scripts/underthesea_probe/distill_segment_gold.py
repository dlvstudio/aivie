#!/usr/bin/env python3
"""TK_032 / doc 10 — ĐO khoảng cách segment: ISG-tần-suất vs gold từ underthesea.

Câu hỏi đo:
  "Trên corpus thật, có bao nhiêu TỪ tiếng Việt đúng (cụm đa âm tiết) mà đường ISG hiện tại
   (promote cặp khi bigram_count >= min_frequency) KHÔNG BAO GIỜ phát hiện, nhưng
   dictionary-seed / gold-distill sẽ bắt được ngay?"

Không sửa checkpoint. Chỉ đọc corpus + chạy underthesea word_tokenize (distill 1 lần lúc build).
Kết quả → out/tk032_segment.log để viết docs/study/10_SEGMENT_NEURON_FROM_UTS.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.teaching.align import tokenize  # noqa: E402  (whitespace = nguyên tử ISG)

CORPUS = _ROOT / "data/real_vi/history_vi.jsonl"
OUT = _ROOT / "out/tk032_segment.log"
MIN_FREQ = 3  # seg_graph.SegmentGraph.min_frequency mặc định
MAX_EXAMPLES = 30


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


def main() -> None:
    ap = argparse.ArgumentParser(description="TK_032/TK_033: segment gap probe (underthesea gold)")
    ap.add_argument("--checkpoint", default=None, help="Nếu có: đo coverage theo cluster_of_pair trong ckpt (TK_033 S1)")
    ap.add_argument("--out", default=None, help="Output log path (mặc định tk032_segment.log hoặc tk033_segment_seed.log)")
    ap.add_argument("--min-freq", type=int, default=MIN_FREQ)
    args = ap.parse_args()

    try:
        from underthesea import word_tokenize as uts_tokenize
    except ImportError as e:
        print(f"SKIP: underthesea not installed ({e})")
        sys.exit(1)

    sentences = _load_sentences()
    min_freq = int(args.min_freq)

    seg_pairs: set[tuple[int, int]] | None = None
    vocab = None
    if args.checkpoint:
        from nn_aivie.core.checkpoint_io import load_checkpoint  # local import (test env)

        st, _cache = load_checkpoint(str(Path(args.checkpoint)))
        if st.parents and st.parents.seg:
            seg_pairs = set(st.parents.seg.cluster_of_pair.keys())
        vocab = st.vocab

    out_path = Path(args.out) if args.out else (
        _ROOT / "out" / ("tk033_segment_seed.log" if args.checkpoint else "tk032_segment.log")
    )

    # 1) Đếm bigram âm-tiết trên toàn corpus (mô phỏng đúng cách ISG đếm: token liền kề).
    syl_bigram: Counter[tuple[str, str]] = Counter()
    for s in sentences:
        syls = tokenize(s)
        for a, b in zip(syls, syls[1:]):
            syl_bigram[(a, b)] += 1

    # 2) Gom mọi TỪ gold (cụm đa âm-tiết) từ underthesea trên corpus.
    gold_compounds: Counter[str] = Counter()
    for s in sentences:
        for w in uts_tokenize(s):
            if " " in w:  # cụm đa âm tiết
                gold_compounds[w] += 1

    # 3) Với mỗi cụm gold: các cặp âm-tiết BÊN TRONG có đủ tần suất để ISG promote không?
    isg_discoverable = 0  # mọi cặp trong cụm đều >= MIN_FREQ
    isg_missed = 0  # có ít nhất một cặp < MIN_FREQ (ISG bỏ lỡ trên corpus này)
    seed_discoverable = 0  # (count>=min_freq) OR (pair đã có cluster trong checkpoint)
    seed_missed = 0
    missed_examples: list[str] = []
    discoverable_examples: list[str] = []
    seed_examples: list[str] = []
    n_merge_decisions = 0  # tổng cạnh MERGE distill được (cặp âm tiết trong cụm)

    for compound, freq in gold_compounds.items():
        syls = compound.split(" ")
        pairs = list(zip(syls, syls[1:]))
        n_merge_decisions += len(pairs) * freq
        min_pair = min(syl_bigram.get(p, 0) for p in pairs)
        if min_pair >= min_freq:
            isg_discoverable += 1
            if len(discoverable_examples) < MAX_EXAMPLES:
                discoverable_examples.append(
                    f"  {compound!r} (freq={freq}, min_pair_count={min_pair})"
                )
        else:
            isg_missed += 1
            if len(missed_examples) < MAX_EXAMPLES:
                weakest = min(pairs, key=lambda p: syl_bigram.get(p, 0))
                missed_examples.append(
                    f"  {compound!r} (freq={freq}, weakest {weakest}={syl_bigram.get(weakest, 0)})"
                )

        if seg_pairs is not None and vocab is not None:
            ok = True
            for a, b in pairs:
                if syl_bigram.get((a, b), 0) >= min_freq:
                    continue
                ai = vocab._token_to_id.get(a) or vocab._token_to_id.get(a.lower())
                bi = vocab._token_to_id.get(b) or vocab._token_to_id.get(b.lower())
                if ai is None or bi is None:
                    ok = False
                    break
                if (int(ai), int(bi)) not in seg_pairs:
                    ok = False
                    break
            if ok:
                seed_discoverable += 1
                if len(seed_examples) < MAX_EXAMPLES and min_pair < min_freq:
                    seed_examples.append(
                        f"  {compound!r} (freq={freq}, min_pair_count={min_pair})"
                    )
            else:
                seed_missed += 1

    n_distinct = len(gold_compounds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as log:
        def w(msg: str = "") -> None:
            print(msg)
            log.write(msg + "\n")

        w(f"corpus: {CORPUS.relative_to(_ROOT)}")
        if args.checkpoint:
            w(f"checkpoint: {args.checkpoint}")
        w(f"sentences: {len(sentences)}")
        w(f"ISG min_frequency (promote threshold): {min_freq}")
        w("")
        w("=== GOLD COMPOUNDS (underthesea word_tokenize, cụm đa âm tiết) ===")
        w(f"distinct_compounds: {n_distinct}")
        w(f"total_compound_occurrences: {sum(gold_compounds.values())}")
        w(f"distilled_merge_decisions (cạnh MERGE có hướng, đếm cả lặp): {n_merge_decisions}")
        w("")
        w("=== ISG COVERAGE (dựa TẦN SUẤT bigram âm-tiết corpus) ===")
        if n_distinct:
            w(
                f"ISG-discoverable (mọi cặp >= {min_freq}): "
                f"{isg_discoverable} ({100 * isg_discoverable / n_distinct:.1f}%)"
            )
            w(
                f"ISG-MISSED  (>=1 cặp < {min_freq}): "
                f"{isg_missed} ({100 * isg_missed / n_distinct:.1f}%)"
            )
        if seg_pairs is not None:
            w("")
            w("=== SEED COVERAGE (count>=min_freq OR cluster exists in checkpoint) ===")
            if n_distinct:
                w(
                    f"Seed-discoverable: {seed_discoverable} ({100 * seed_discoverable / n_distinct:.1f}%)"
                )
                w(
                    f"Seed-missed     : {seed_missed} ({100 * seed_missed / n_distinct:.1f}%)"
                )
        w("")
        w("→ Cụm ISG BỎ LỠ = giá trị dictionary-seed / gold-distill (bắt ngay, không chờ tần suất).")
        w("")
        w(f"=== ISG-MISSED EXAMPLES (max {MAX_EXAMPLES}) ===")
        for ex in missed_examples:
            w(ex)
        if seg_pairs is not None:
            w("")
            w(f"=== SEED-RESCUED EXAMPLES (max {MAX_EXAMPLES}, only those ISG missed) ===")
            for ex in seed_examples:
                w(ex)
        w("")
        w(f"=== ISG-DISCOVERABLE EXAMPLES (max {MAX_EXAMPLES}) ===")
        for ex in discoverable_examples:
            w(ex)
        w("")
        w("XONG")

    print(f"✓ wrote {out_path}")


if __name__ == "__main__":
    main()
