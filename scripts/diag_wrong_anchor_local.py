#!/usr/bin/env python3
"""
DIAG wrong-anchor trên MINI checkpoint LOCAL (TK_045 Tier-1.2) — chạy được trên máy dev 8GB.

    PYTHONPATH=src python -u scripts/diag_wrong_anchor_local.py

## Vì sao mini

`network_v12_sent.pkl` (430k parents) → ~20GB RAM: KHÔNG load nổi trên dev 8GB. Nhưng câu hỏi
H1/H2 (path/succ có fire ở "…dời đô về" không?) là câu hỏi **cơ chế**, tái lập được trên mini:
`history_articles.jsonl` (1.6MB) CÓ 12 dòng "Ngô Quyền" + 12 dòng "dời đô" — đúng entity đang gãy.
Build 1 lần → cache `checkpoints/mini_sent.pkl` (theo precedent `probe_grounding_local.py`).

Mini KHÔNG thay v12 để chốt số cuối (ít parent → act khác), nhưng đủ để trả lời:
**path/succ có fire không, cha đúng có được kích không, cổng bag-midsent có đổi argmax không.**
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint  # noqa: E402
from nn_aivie.core.state import GraphState  # noqa: E402
from nn_aivie.core.tokenizer import BPETokenizer  # noqa: E402
from nn_aivie.core.vocab import Vocabulary  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.training.dataset_spec import DatasetSpec  # noqa: E402
from nn_aivie.training.pipeline import finalize_graph, ingest_text_file  # noqa: E402

CACHE = _ROOT / "checkpoints/mini_sent.pkl"
CORPUS = [("data/text_corpus/history_articles.jsonl", 1.0),
          ("data/text_corpus/subject_articles.jsonl", 1.0)]
PROMPTS = ["Nhà Lý dời đô về", "Trận Bạch Đằng năm",
           "Ngô Quyền là vị vua sáng lập nhà Ngô, được"]


def main() -> int:
    t0 = time.time()

    def log(m=""):
        print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

    gc.disable()
    if CACHE.exists():
        state, _ = load_checkpoint(str(CACHE))
        log(f"✓ cache {CACHE.name} · {len(state.parents):,} parents")
    else:
        log("build mini (sentence_parents=True) …")
        tok = BPETokenizer.load(_ROOT / "checkpoints/bpe_v1.json")
        state = GraphState(vocab=Vocabulary(tok))
        eps: list = []
        for path, pri in CORPUS:
            p = _ROOT / path
            if not p.exists():
                continue
            ingest_text_file(state, p, DatasetSpec(path=path, priority=pri),
                             sentence_parents=True, episode_groups=eps)
            log(f"  + {path} · parents={len(state.parents):,}")
        finalize_graph(state, dim=96)
        save_checkpoint(state, CACHE)
        log(f"✓ build + cache · {len(state.parents):,} parents · episodes={len(eps):,}")
    gc.enable()

    from nn_aivie.hca.matrix_core import (
        ensure_parent_bipartite, matrix_core_logits, parent_coactivation_logits,
    )
    vocab, ps = state.vocab, state.parents
    sur = vocab.token_of
    n = len(vocab)

    for p in PROMPTS:
        log(f"\n═══ {p!r} ═══")
        hist = list(vocab.encode_prompt(p))
        cfg = HCAConfig()
        bag, succ, conf, act = parent_coactivation_logits(
            state, hist, vocab_size=n, lam=0.85, ctx_len=0, gamma=1.5,
            succ_boost=3.0, top_k_parents=48, gidf_hi=4.0,
        )
        log(f"  conf={conf:.3f} · bag.max={bag.max():.4g} · succ+path.max={succ.max():.4g}"
            f"   ← H1 nếu succ≈0")
        bi = ensure_parent_bipartite(state)
        if act.size:
            for pi in np.argsort(-act)[:4]:
                if act[pi] <= 0:
                    continue
                em = bi.emits[int(pi)][:12] if int(pi) < len(bi.emits) else []
                log(f"    act={act[pi]:.4g} pi={int(pi)} {' '.join(sur(int(t)) for t in em)!r}")
        # A/B cổng bag-midsent
        for pen in (1.0, 0.35):
            cfg.ar_matrix_bag_midsent_penalty = pen
            lg = matrix_core_logits(state, hist, config=cfg)
            top = np.argsort(-lg)[:5]
            tag = "TẮT " if pen >= 1.0 else "BẬT "
            log(f"  cổng {tag}(pen={pen}): " +
                "  ".join(f"{sur(int(j))!r}({lg[int(j)]:.2f})" for j in top))
    log("\n→ succ≈0 ⇒ H1 (path không fire) → sửa act/act_floor/best_k, KHÔNG phải cổng bag.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
