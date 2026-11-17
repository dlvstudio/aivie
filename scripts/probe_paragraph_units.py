#!/usr/bin/env python3
"""TK_035 D0 — probe thống kê đơn vị đoạn 2–4 câu (dấu phẩy/chấm/than/hỏi)."""

from __future__ import annotations

import argparse
from pathlib import Path

from nn_aivie.data.paragraph_units import (
    count_punctuation_stats,
    iter_units_from_conv_jsonl,
    iter_units_from_doc_jsonl,
)
from nn_aivie.util.progress import get_run_logger

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "out/tk035_paragraph_probe.log"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-sent", type=int, default=2)
    ap.add_argument("--max-sent", type=int, default=4)
    ap.add_argument("--max-per-source", type=int, default=50000)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    log = get_run_logger("probe_paragraph_units")

    sources = [
        ("statements_doc", _ROOT / "data/real_vi/statements_doc.jsonl"),
        ("history_wiki", _ROOT / "data/real_vi/history_wiki_para.jsonl"),
        ("stories", _ROOT / "data/conv/stories_raw.jsonl"),
    ]
    lines: list[str] = []
    all_units: list[str] = []

    for name, path in sources:
        units: list[str] = []
        if not path.exists():
            lines.append(f"{name}: MISSING {path}")
            continue
        if "conv" in str(path):
            it = iter_units_from_conv_jsonl(path, min_sent=args.min_sent, max_sent=args.max_sent)
        else:
            it = iter_units_from_doc_jsonl(path, min_sent=args.min_sent, max_sent=args.max_sent)
        for i, u in enumerate(it):
            units.append(u)
            if i + 1 >= args.max_per_source:
                break
        st = count_punctuation_stats(units)
        lines.append(
            f"{name}: units={st['units']} multi_sent={st['multi_sent']} "
            f"comma={st['comma']} period={st['period']} !={st['exclaim']} ?={st['question']}"
        )
        if units:
            lines.append(f"  sample: {units[0][:120]}…")
        all_units.extend(units)

    total = count_punctuation_stats(all_units)
    lines.append(
        f"TOTAL: units={total['units']} multi_sent={total['multi_sent']} "
        f"comma={total['comma']} period={total['period']} !={total['exclaim']} ?={total['question']}"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    out_path.write_text(text + "\n", encoding="utf-8")
    for ln in lines:
        log.log(ln)
    log.log(f"→ {out_path}")


if __name__ == "__main__":
    main()
