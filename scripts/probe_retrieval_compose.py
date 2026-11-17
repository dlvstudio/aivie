#!/usr/bin/env python3
"""
PROBE RETRIEVAL + COMPOSITION — đồ thị chọn ĐÚNG fact + ghép NHIỀU fact không? (TK_042).

    ⚠️ CHẠY M4 — checkpoint đồ thị 2GB → ~20GB RAM (TK_040).
    PYTHONPATH=src python -u scripts/probe_retrieval_compose.py --checkpoint checkpoints/network_v11.pkl

## Vì sao (reframe tác giả 2026-07-20: "văn phong tự nhiên là ĐỦ, không cần đúng ngữ pháp")

Chấp nhận đồ thị CHÉP = engine fluency. Chất lượng "tự nhiên + ĐÚNG NGỮ CẢNH" giờ do 2 lever
(đã đo là lever thật, không phải ngữ pháp): (1) RETRIEVAL chọn đúng fact; (2) COMPOSITION ghép
nhiều fact mạch lạc. Probe này đo cả hai + trả lời thẳng câu hỏi tác giả:

  "lúc copy fact này lúc copy fact khác được không, miễn đúng ngữ cảnh?"

→ ĐÃ CÓ: `generate(multi=True)` gom nhiều fact về thực-thể trọng-tâm. Đo `n_facts` = số ĐƠN VỊ
CORPUS khác nhau mà output ghép từ (chia câu → mỗi câu khớp verbatim đơn vị nào). n_facts>1 =
đang copy-fact-A-rồi-fact-B thật. Câu hỏi mở KHÔNG phải "có ghép được không" (được) mà "ghép có
ĐÚNG NGỮ CẢNH không" = ent_keep (đúng thực thể) + ans_recall (trúng đáp) + đọc tay.

## Đo (mỗi prompt, generate multi=True)

* `n_facts`   số đơn vị corpus KHÁC NHAU các câu output chép từ (≥50% LCS) — BỀ RỘNG composition
* `copy`      LCS tối đa với 1 đơn vị — mức verbatim (reuse _copy_frac)
* `ent_keep`  thực thể trong câu hỏi có mặt ở đáp — ĐÚNG THỰC THỂ (chống drift)
* `ans_recall` (grade có gold) token nội-dung gold có trong đáp — RETRIEVAL trúng
* toàn văn — đọc tay: fact ghép có LẠC chủ đề không (QFocus bleed / drift đo trước)

## Điều kiện đọc (GHI TRƯỚC)

n_facts>1 + ent_keep cao + đọc tay đúng chủ đề ⇒ đồ thị đã copy-đa-fact đúng ngữ cảnh — lever
cần đẩy = SCALE coverage. n_facts>1 nhưng ent_keep thấp / đọc tay lạc ⇒ ghép SAI ngữ cảnh
(drift/bleed) ⇒ lever = RETRIEVAL (chọn-đúng-fact, TK_021/024). n_facts≈1 ⇒ chỉ chép 1 khối,
composition chưa bật ⇒ xem agenda/multi.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from eval_discourse_prompts import (  # noqa: E402 — reuse harness đồ thị
    COPY_TOPK, CorpusIndex, _copy_frac, _entity_spans, _lcs_len, build_corpus_index,
)
from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.data.quality import normalize_ws  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.teaching.align import tokenize_punct  # noqa: E402

ENTITY_PROMPTS = [  # thực thể wiki giàu fact (corpus index phủ history/statements_doc)
    "Hỏi: Nhà Lý là gì", "Hỏi: Nhà Trần là gì", "Hỏi: Hà Nội là gì",
    "Hỏi: Sông Hồng là gì", "Hỏi: Trần Hưng Đạo là ai", "Hỏi: Thăng Long là gì",
]


def source_units(gen_only: str, index: CorpusIndex, min_seg=4, thresh=0.5):
    """Mỗi câu output → đơn vị corpus khớp nhất (LCS). Trả list unit-id của câu chép ≥thresh.
    Số unit-id KHÁC NHAU = bao nhiêu FACT khác nhau được ghép (bề rộng composition)."""
    hits = []
    for seg in re.split(r"[.!?\n]", gen_only):
        g = tokenize_punct(normalize_ws(seg))
        if len(g) < min_seg:
            continue
        cnt: dict[int, int] = defaultdict(int)
        for tk in set(g):
            for ui in index.inv.get(tk, ()):
                cnt[ui] += 1
        if not cnt:
            continue
        cands = sorted(cnt, key=cnt.__getitem__, reverse=True)[:COPY_TOPK]
        ui, lcs = max(((u, _lcs_len(g, index.units[u])) for u in cands), key=lambda x: x[1])
        if lcs / len(g) >= thresh:
            hits.append(ui)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11.pkl")
    ap.add_argument("--holdout", default="data/sft/qa_heldout.jsonl")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--corpus-limit", type=int, default=20000)
    ap.add_argument("--n-grade", type=int, default=12, help="số QA grade closed-book (có gold)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()

    def log(m):
        print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

    log(f"load {args.checkpoint} (gc.disable) …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    index = build_corpus_index(limit_units=args.corpus_limit)
    log(f"✓ loaded · corpus index {len(index.units):,} đơn vị")
    cfg = HCAConfig()

    # grade closed-book (có gold) từ qa_heldout
    grade = []
    for line in (_ROOT / args.holdout).open(encoding="utf-8"):
        o = json.loads(line)
        if o.get("source") == "grade_val" and " Đáp: " in o["text"]:
            q, gold = o["text"].split(" Đáp: ", 1)
            grade.append((q, gold))
    grade = grade[: args.n_grade]

    def run(prompt, gold=None):
        ids = generate(state, prompt, max_len=args.max_len, config=cfg, multi=True, seed=args.seed)
        out = decode_surface(state, ids)
        gen = out[len(prompt):].strip() if out.startswith(prompt) else out
        srcs = source_units(gen, index)
        ents = _entity_spans(prompt)
        ent_keep = sum(1 for e in ents if e in gen) / len(ents) if ents else float("nan")
        rec = float("nan")
        if gold:
            gt = [w for w in tokenize_punct(normalize_ws(gold)) if len(w) > 1]
            got = set(tokenize_punct(normalize_ws(gen)))
            rec = sum(1 for w in gt if w in got) / len(gt) if gt else float("nan")
        return gen, len(set(srcs)), _copy_frac(gen, index), ent_keep, rec

    # DIAGNOSTIC (2026-07-21): fix _build_segment_plan không nhích metric ở v11 dù code đúng.
    # "Thăng Long→Gia Long" BẤT KHẢ nếu primary='Thăng' → nghi ở 65k+viwiki, focus/primary chọn
    # 'Long' (va chạm) hoặc facts_about-theme rỗng → fallback rank_matches. In focus+gidf +
    # facts_about-theme top để định VỊ gãy: (a) focus sai token? (b) theme trôi? (c) plan bỏ theme?
    from nn_aivie.hca.generate_hebbian import _build_segment_plan, _completion_from_parent, _content_ids
    ps = state.parents

    def diag(prompt):
        qids = list(state.vocab.encode_prompt(prompt))
        qc = list(dict.fromkeys(_content_ids(qids, state.vocab, skip=set())))
        focus = ps.top_focus_tokens(tuple(qc), k=2, vocab=state.vocab)
        fstr = " ".join(f"{state.vocab.token_of(t)!r}({ps._gidf(t):.2f})" for t in focus)
        fa = ps.facts_about(tuple(focus), top_k=2, vocab=state.vocab, theme_weight=cfg.compose_theme_weight)
        fatop = decode_surface(state, list(_completion_from_parent(ps, fa[0])))[:70] if fa else "∅"
        plan = _build_segment_plan(state, qids, cfg, max_facts=3)
        pag = decode_surface(state, list(plan.flat_agenda))[:70] if plan.flat_agenda else "∅"
        log(f"    [diag] focus={fstr} · facts_about-theme#1={fatop!r} · plan.agenda={pag!r}")

    log("═══ ENTITY (multi-fact composition) — n_facts = số fact KHÁC NHAU ghép ═══")
    nf, ck, ek = [], [], []
    for p in ENTITY_PROMPTS:
        diag(p)
        gen, n_facts, copy, ent_keep, _ = run(p)
        nf.append(n_facts); ck.append(copy); ek.append(ent_keep)
        log(f"  {p}\n    n_facts={n_facts} copy={copy:.2f} ent_keep={ent_keep:.2f}\n    → {gen}")

    log("═══ GRADE closed-book — ans_recall = retrieval trúng đáp ═══")
    rc = []
    for q, gold in grade:
        gen, n_facts, copy, _ent, rec = run(q, gold)
        rc.append(rec)
        log(f"  {q}  [gold: {gold}]\n    ans_recall={rec:.2f} n_facts={n_facts}\n    → {gen}")

    import statistics as st
    def avg(xs): xs = [x for x in xs if x == x]; return st.mean(xs) if xs else float("nan")
    log("═══════════ TÓM TẮT ═══════════")
    log(f"  ENTITY: n_facts TB {avg(nf):.2f} · copy TB {avg(ck):.2f} · ent_keep TB {avg(ek):.2f}")
    log(f"  GRADE : ans_recall TB {avg(rc):.2f}")
    log("→ ĐỌC: n_facts>1 = ghép đa-fact THẬT (trả lời câu tác giả). ent_keep/ans_recall + đọc tay "
        "= có ĐÚNG NGỮ CẢNH không. Thấp ⇒ lever RETRIEVAL; cao ⇒ lever SCALE coverage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
