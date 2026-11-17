#!/usr/bin/env python3
"""Dựng MINI **có episode** cho máy dev — TK_045 §12.21.

    PYTHONPATH=src python -u scripts/build_mini_ep.py

`checkpoints/mini_sent.pkl` (dựng bởi `diag_wrong_anchor_local.py`) truyền `episode_groups` vào
`ingest_text_file` nhưng **quên gọi `build_episodes_from_provenance`** — bước mà pipeline thật
làm ở `training/pipeline.py:333`. Hệ quả: mini có 0 episode ⇒ mọi đường handoff đa-câu
(`_episode_next_span`, `agenda_handoff_span`) KHÔNG đo được trên dev, phải đẩy hết sang M4/v12.

Script này bù đúng bước đó → `checkpoints/mini_sent_ep.pkl`. Giữ NGUYÊN `mini_sent.pkl` để các
số đã đo ở §12.13–§12.14 còn tái lập được.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import save_checkpoint  # noqa: E402
from nn_aivie.core.state import GraphState  # noqa: E402
from nn_aivie.core.tokenizer import BPETokenizer  # noqa: E402
from nn_aivie.core.vocab import Vocabulary  # noqa: E402
from nn_aivie.training.dataset_spec import DatasetSpec  # noqa: E402
from nn_aivie.training.pipeline import finalize_graph, ingest_text_file  # noqa: E402

OUT = _ROOT / "checkpoints/mini_sent_ep.pkl"
OUT_QA = _ROOT / "checkpoints/mini_sent_ep_qa.pkl"
CORPUS = [("data/text_corpus/history_articles.jsonl", 1.0),
          ("data/text_corpus/subject_articles.jsonl", 1.0)]
# §12.27: `--with-qa` thêm corpus QA ⇒ mini có cha `is_qa` ⇒ gỡ được bug "QA sinh 1 token"
# (§12.22/§12.27) ngay trên dev. Không có nó thì mini `is_qa=0`, mọi ca QA phải đẩy sang v12.
QA_CORPUS = [("data/text_corpus/qa_supervised.jsonl", 1.0)]


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--with-qa", action="store_true", help="nạp thêm qa_supervised.jsonl")
    args = ap.parse_args()
    out_path = OUT_QA if args.with_qa else OUT
    corpus = CORPUS + (QA_CORPUS if args.with_qa else [])
    t0 = time.time()

    def log(m=""):
        print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

    gc.disable()
    tok = BPETokenizer.load(_ROOT / "checkpoints/bpe_v1.json")
    state = GraphState(vocab=Vocabulary(tok))
    eps: list = []
    for path, pri in corpus:
        p = _ROOT / path
        if not p.exists():
            log(f"  ! thiếu {path}")
            continue
        ingest_text_file(state, p, DatasetSpec(path=path, priority=pri),
                         sentence_parents=True, episode_groups=eps)
        log(f"  + {path} · parents={len(state.parents):,} · groups={len(eps):,}")

    made, matched = state.parents.build_episodes_from_provenance(eps, min_members=2)
    log(f"✓ episodes: made={made:,} matched={matched:,} · tổng={len(state.parents.episodes):,}")

    finalize_graph(state, dim=96)
    save_checkpoint(state, out_path)
    gc.enable()

    # Kiểm chỉ mục ngược — `episodes_of` rỗng = handoff câm (đúng lỗi làm 2 test đỏ ở §12.4).
    n_linked = sum(1 for pi in range(len(state.parents.parents))
                   if state.parents.episodes_of(pi))
    n_qa = sum(1 for r in state.parents.parents if getattr(r, "is_qa", False))
    log(f"✓ {out_path.name}: {len(state.parents.parents):,} cha · "
        f"{len(state.parents.episodes):,} episode · {n_linked:,} cha có episodes_of ≠ () · "
        f"is_qa={n_qa:,}")
    if not n_linked:
        log("⚠️ KHÔNG cha nào liên kết episode — handoff sẽ câm, kiểm lại provenance.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
