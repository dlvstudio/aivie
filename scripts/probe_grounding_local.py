#!/usr/bin/env python3
"""
GROUNDING LOCAL — build graph NHỎ (history + grade) tại chỗ để iterate fix (TK_042).

    PYTHONPATH=src python -u scripts/probe_grounding_local.py

## Vì sao

probe_grounding trên network_v11 (20GB) TREO trên M4 (swap). Nhưng cơ chế drift đo được LOCAL:
build graph nhỏ từ history_articles (158 bài, có 'Chiến thắng Bạch Đằng' + 'Trần Hưng Đạo' +
'Nhà Trần') + grade → tái hiện ranking, test facts_about theme=0 vs 1.0, iterate subject-boost
KHÔNG cần M4. (Scale nhỏ nên power-bias yếu hơn v11 — nếu vẫn tái hiện thì fix chắc; nếu không,
ghi rõ là hiệu ứng scale.)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint  # noqa: E402
from nn_aivie.core.state import GraphState  # noqa: E402
from nn_aivie.core.tokenizer import BPETokenizer  # noqa: E402
from nn_aivie.core.vocab import Vocabulary  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import (  # noqa: E402
    _build_segment_plan, _completion_from_parent, _content_ids,
)
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.training.dataset_spec import DatasetSpec  # noqa: E402
from nn_aivie.training.pipeline import finalize_graph, ingest_text_file  # noqa: E402

CACHE = _ROOT / "checkpoints/mini_grounding.pkl"

FILES = [  # BỎ qa_supervised (660 dòng = chậm nhất); history đủ chứa các ca drift
    ("data/text_corpus/history_articles.jsonl", 0.88),
    ("data/text_corpus/subject_articles.jsonl", 0.92),
    ("data/text_corpus/grade_lessons.jsonl", 0.95),
]
PROMPTS = [
    ("Hỏi: Bạch Đằng là gì", "drift→Trần Hưng Đạo?"),
    ("Hỏi: Ngô Quyền là ai", "drift→Trần Hưng Đạo?"),
    ("Hỏi: Nhà Trần là gì", "drift→Hồ Quý Ly?"),
    ("Hỏi: Thăng Long là gì", "surface Long"),
    ("Hỏi: Trần Hưng Đạo là ai", "control"),
]


def main() -> int:
    t0 = time.time()

    def log(m):
        print(f"[{time.time()-t0:5.1f}s] {m}", flush=True)

    if CACHE.exists():
        state, _ = load_checkpoint(str(CACHE))
        log(f"✓ load cache {CACHE.name} · {len(state.parents)} parents")
    else:
        tok = BPETokenizer.load(_ROOT / "checkpoints/bpe_v1.json")
        state = GraphState(vocab=Vocabulary(tok))
        for path, pri in FILES:
            p = _ROOT / path
            if not p.exists():
                log(f"  ⚠ bỏ (không có): {path}")
                continue
            ingest_text_file(state, p, DatasetSpec(path=path, priority=pri))
            log(f"  nạp {Path(path).name} → parents={len(state.parents)}")
        finalize_graph(state, dim=96)
        save_checkpoint(state, CACHE)
        log(f"✓ build + cache {CACHE.name} · {len(state.parents)} parents")
    ps, vocab = state.parents, state.vocab
    cfg = HCAConfig()

    def show(pi):
        return decode_surface(state, list(_completion_from_parent(ps, pi)))[:90]

    for prompt, note in PROMPTS:
        log(f"\n═══ {prompt}  [{note}] ═══")
        qids = list(vocab.encode_prompt(prompt))
        qc = list(dict.fromkeys(_content_ids(qids, vocab, skip=set())))
        focus = ps.top_focus_tokens(tuple(qc), k=2, vocab=vocab)
        log("  FOCUS: " + " ".join(f"{vocab.token_of(t)!r}" for t in focus))
        # tham chiếu: facts_about-theme (đường ĐÚNG)
        facts = ps.facts_about(tuple(focus), top_k=3, vocab=vocab, theme_weight=1.0)
        log(f"  facts_about theme=1.0 top: {show(facts[0])!r}" if facts else "  facts_about: ∅")
        # AGENDA THẬT (có fix _build_segment_plan) — decode xem nó neo entity NÀO
        plan = _build_segment_plan(state, qids, cfg, max_facts=3)
        ag = decode_surface(state, list(plan.flat_agenda))[:120] if plan.flat_agenda else "∅"
        log(f"  → _build_segment_plan AGENDA: {ag!r}")

    log("\n→ AGENDA (có fix) giờ KHỚP facts_about-theme (entity ĐÚNG)? control vẫn đúng?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
