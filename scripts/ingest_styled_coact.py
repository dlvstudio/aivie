#!/usr/bin/env python3
"""
INGEST styled → CO_ACT có điều kiện label↔thân (cho M4).

    ⚠️ CHẠY M4.
    PYTHONPATH=src python -u scripts/ingest_styled_coact.py \
        --checkpoint checkpoints/network_v11.pkl \
        --styled data/text_corpus/styled_sft_1k.jsonl \
        --extra data/text_corpus/styled_sft.jsonl \
        --mode conditional \
        --out checkpoints/network_v11_styled_1k.pkl \
        --log-file out/ingest_styled_coact_1k.log

## Modes
  full         — g1_passive cả câu (cũ; tạo artifact nhãn↔topic)
  conditional  — CHỈ bồi CO_ACT giữa token HOW (nhãn) ↔ token thân (gidf≥min)
                 + cửa sổ thân nội bộ (eta thấp). Không sentence_bind toàn bài.

Hỗ trợ format:
  "Viết hài hước về X: BODY…"
  "Hỏi: … hài hước … Đáp: BODY…"
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.edge_kinds import EdgeKind  # noqa: E402
from nn_aivie.core.hebb import update_from_coactivation  # noqa: E402
from nn_aivie.data.types import CorpusRecord  # noqa: E402
from nn_aivie.hca.style_readout import style_seeds_from_prompt  # noqa: E402
from nn_aivie.teaching.align import tokenize  # noqa: E402
from nn_aivie.training.g1_passive import run_g1_passive  # noqa: E402

# 1k: "Hỏi: …? Đáp: …" cùng một dòng — không neo ^/\n.
_DAP = re.compile(r"Đáp\s*:\s*", re.IGNORECASE)
_COLON_LABEL = re.compile(
    r"^(?:Viết|Kể|Hãy viết|Hãy kể)[^:]{0,80}:\s*",
    re.IGNORECASE,
)


def _split_label_body(text: str) -> tuple[str, str]:
    """(label_span, body). Fallback: cả câu = body, label=cả câu."""
    m = _DAP.search(text)
    if m:
        return text[: m.start()], text[m.end() :]
    m2 = _COLON_LABEL.match(text)
    if m2:
        return text[: m2.end()], text[m2.end() :]
    # Hỏi: … không có Đáp rõ
    if text.lower().startswith("hỏi:"):
        return text, text
    return text, text


def _conditional_ingest(
    state,
    texts: list[str],
    *,
    eta: float,
    body_eta_scale: float,
    gidf_min: float,
    max_seeds: int,
    repeat: int,
    log,
) -> dict[str, int]:
    vocab, ps = state.vocab, state.parents
    stats = {
        "records": 0,
        "pairs": 0,
        "body_windows": 0,
        "skip_no_seed": 0,
        "skip_no_body": 0,
    }
    for _ in range(repeat):
        for text in texts:
            stats["records"] += 1
            label_txt, body_txt = _split_label_body(text)
            label_ids = list(vocab.encode_prompt(label_txt))
            body_ids = list(vocab.encode_prompt(body_txt))
            seeds = style_seeds_from_prompt(
                label_ids, vocab, ps, gidf_min=gidf_min, max_seeds=max_seeds,
            )
            if not seeds:
                # fallback: mọi token thường gidf đủ trong label
                seeds = style_seeds_from_prompt(
                    label_ids, vocab, ps, gidf_min=max(3.0, gidf_min - 1.0), max_seeds=4,
                )
            if not seeds:
                stats["skip_no_seed"] += 1
                continue
            body_keep = [
                t for t in body_ids
                if vocab.is_surface(t) and ps._gidf(t) >= gidf_min  # noqa: SLF001
            ]
            if not body_keep:
                stats["skip_no_body"] += 1
                continue
            # label ↔ thân (cặp chéo — tín hiệu style chính)
            for s in seeds:
                for b in body_keep:
                    if s == b:
                        continue
                    state.adjacency.add(s, b, eta, EdgeKind.CO_ACTIVATION)
                    state.adjacency.add(b, s, eta, EdgeKind.CO_ACTIVATION)
                    stats["pairs"] += 1
            # thân nội bộ: cửa sổ 4, eta thấp — giữ cụm sắc-thái dính nhau
            body_eta = eta * body_eta_scale
            for i in range(len(body_keep)):
                win = body_keep[i : i + 4]
                if len(win) >= 2:
                    update_from_coactivation(state.adjacency, win, body_eta)
                    stats["body_windows"] += 1
    log(
        f"conditional: records={stats['records']} pairs={stats['pairs']} "
        f"body_win={stats['body_windows']} skip_seed={stats['skip_no_seed']} "
        f"skip_body={stats['skip_no_body']}"
    )
    return stats


def _load_texts(paths: list[Path]) -> list[str]:
    out: list[str] = []
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            t = json.loads(line).get("text", "")
            if t:
                out.append(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11.pkl")
    ap.add_argument("--styled", default="data/text_corpus/styled_sft_1k.jsonl")
    ap.add_argument("--extra", default="data/text_corpus/styled_sft.jsonl",
                    help="ghép thêm seed tone-dày; '' = tắt")
    ap.add_argument("--history", default="data/text_corpus/styled_humor_history.jsonl",
                    help="hài gắn entity lịch sử; '' = tắt")
    ap.add_argument("--out", default="checkpoints/network_v11_styled_1k.pkl")
    ap.add_argument("--mode", choices=("conditional", "full"), default="conditional")
    ap.add_argument("--eta", type=float, default=0.03)
    ap.add_argument("--body-eta-scale", type=float, default=0.35)
    ap.add_argument("--gidf-min", type=float, default=4.0)
    ap.add_argument("--max-seeds", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--log-file", default="out/ingest_styled_coact_1k.log")
    args = ap.parse_args()

    logf = open(_ROOT / args.log_file, "w")
    t0 = time.time()

    def log(m=""):
        line = f"[{time.time()-t0:6.0f}s] {m}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    paths = [_ROOT / args.styled]
    if args.history:
        paths.append(_ROOT / args.history)
    if args.extra:
        paths.append(_ROOT / args.extra)

    log(f"load {args.checkpoint} (gc.disable) …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    log(f"✓ loaded · parents {len(state.parents):,}")

    texts = _load_texts(paths)
    log(f"styled texts: {len(texts)} từ {[str(p.relative_to(_ROOT)) for p in paths if p.exists()]}")

    if args.mode == "full":
        records = [
            CorpusRecord(text=t, grade=0, dataset_id="styled_sft", domain="style")
            for _ in range(args.repeat) for t in texts
        ]
        log(f"full g1: {len(texts)} × {args.repeat} = {len(records)}")
        stats = run_g1_passive(state, records, eta=args.eta, enable_neurogenesis=False)
        log(f"g1_passive stats: {stats}")
    else:
        stats = _conditional_ingest(
            state, texts,
            eta=args.eta,
            body_eta_scale=args.body_eta_scale,
            gidf_min=args.gidf_min,
            max_seeds=args.max_seeds,
            repeat=args.repeat,
            log=log,
        )

    log(f"lưu → {args.out}")
    state.save(str(_ROOT / args.out))
    log("✓ xong. Kế: probe_style_coact.py --checkpoint " + args.out)
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
