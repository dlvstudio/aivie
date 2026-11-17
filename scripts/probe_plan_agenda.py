#!/usr/bin/env python3
"""
TK_044 P3.2 diagnostic — plan-agenda có bám thực-thể MỚI hay TRÔI sang train-entity? (cho M4).

    ⚠️ CHẠY M4. PYTHONPATH=src python -u scripts/probe_plan_agenda.py \
        --checkpoint checkpoints/network_v11_coref.pkl --probe data/sft/bind_probe_real.jsonl \
        --log-file out/probe_plan_agenda.log

## Vì sao (RUNBOOK_TK_044_P3.2 §11)

Plan B/A đều 0/11: sinh TRÔI ("Lữ Bằng Sơn…" → "Nguyễn Lữ/Tây Sơn"). Câu hỏi CHỐT: drift ở tầng
PLAN (agenda đã route sang train-entity TỪ ĐẦU) hay tầng SCORE (agenda đúng, chỉ token lệch)?
Kênh copy sửa `scores` (tầng token) → nếu drift ở PLAN thì boost vô ích, phải xử retrieval/plan.

## Đọc

Với mỗi prompt in:
  * entity_register  = thực-thể prompt (kênh copy định boost cái này)
  * plan.entity_ids  = thực-thể plan giữ
  * plan.agenda      = Ý plan sẽ diễn đạt  ← MẤU CHỐT
  * facts_about#1    = fact top của focus

→ agenda/facts chứa **train-entity** (Nguyễn Lữ…) thay vì thực-thể-mới ⇒ **drift PLAN-level**
  ⇒ copy-channel (score) không cứu được; phải xử tầng plan/retrieval. Ngược lại = score-level.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import (  # noqa: E402
    _build_segment_plan, _completion_from_parent, _content_ids,
)
from nn_aivie.hca.readout import decode_surface  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11_coref.pkl")
    ap.add_argument("--probe", default="data/sft/bind_probe_real.jsonl")
    ap.add_argument("--log-file", default="out/probe_plan_agenda.log")
    args = ap.parse_args()

    logf = open(_ROOT / args.log_file, "w")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True); logf.write(str(m) + "\n"); logf.flush()

    log(f"[{time.time()-t0:.0f}s] load {args.checkpoint} (gc.disable) …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    ps, vocab = state.parents, state.vocab
    sur = vocab.token_of
    cfg = HCAConfig()
    log(f"[{time.time()-t0:.0f}s] ✓ loaded · parents {len(ps):,}")

    items = [json.loads(l) for l in (_ROOT / args.probe).read_text().splitlines() if l.strip()]

    def dec(ids, n=90):
        return decode_surface(state, list(ids))[:n] if ids else "∅"

    log("\n═══ PLAN-AGENDA diagnostic — agenda bám thực-thể MỚI hay train? ═══")
    for it in items:
        p = it["prompt"]
        qids = list(vocab.encode_prompt(p))
        qc = list(dict.fromkeys(_content_ids(qids, vocab, skip=set())))
        reg = ps.entity_register_ids(tuple(qc))
        focus = ps.top_focus_tokens(tuple(qc), k=2, vocab=vocab)
        fa = ps.facts_about(tuple(focus), top_k=1, vocab=vocab,
                            theme_weight=cfg.compose_theme_weight)
        fatop = dec(_completion_from_parent(ps, fa[0]), 70) if fa else "∅"
        plan = _build_segment_plan(state, qids, cfg, max_facts=3)
        log(f"\n[{it['type']}] {p!r}")
        log(f"    entity_register : {dec(reg, 60)!r}")
        log(f"    focus           : {' '.join(f'{sur(t)!r}({ps._gidf(t):.1f})' for t in focus)}")
        log(f"    plan.entity_ids : {dec(plan.entity_ids, 60)!r}")
        log(f"    plan.AGENDA     : {dec(plan.flat_agenda)!r}   ← mấu chốt")
        log(f"    facts_about#1   : {fatop!r}")
    log(f"\n[{time.time()-t0:.0f}s] → agenda/facts = train-entity (Nguyễn Lữ…) ⇒ drift PLAN-level "
        "(copy-score không cứu); = thực-thể-mới/rỗng ⇒ score-level.")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
