#!/usr/bin/env python3
"""ĐO W_skip + giao đề xuất theo khoảng cách — ý tác giả (không phải pos_rel).

    PYTHONPATH=src python -u scripts/probe_w_skip_intersect.py \
        --checkpoint checkpoints/mini_sent_ep_qa.pkl \
        --log-file out/w_skip_mini.log

## Cổng (chốt trước)

  G1  hard∩ không rỗng ≥ 30% ca (prompt ≥2 token nội dung, gold = token kế trong câu)
  G2  khi hard∩ ≠ ∅: gold ∈ hard∩ ≥ 40%
  G3  soft top-10 chứa gold ≥ 50%
  G4  so baseline chỉ d=1 (W_seq-like): G3_skip − G3_d1 ≥ +0.05
      (nếu không hơn d=1 thì ràng buộc xa không thêm gì)

Trượt G4 ⇒ kênh xa không đáng cắm. Trượt G1/G2 vẫn có thể sống nhờ soft (G3).
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.hca.sent_intra import _content_positions, iter_sentence_spans  # noqa: E402
from nn_aivie.hca.w_skip import (  # noqa: E402
    WSkip,
    build_w_skip,
    clear_w_skip_cache,
    constrain_next,
)


def _surface(vocab, tid: int) -> str:
    return vocab.token_of(int(tid)).lstrip("Ġ")


def _collect_cases(state, *, max_cases: int, seed: int, min_hist: int = 2):
    """Mỗi ca: hist = token nội dung đầu..k, gold = token k+1 trong cùng câu."""
    vocab = state.vocab
    n = len(vocab)
    struct = [is_structural_token(vocab, i) for i in range(n)]
    cases = []
    for rec in state.parents.parents:
        seq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
        for span in iter_sentence_spans(seq, vocab):
            toks = _content_positions(
                span, vocab, content_only=True, struct_mask=struct
            )
            if len(toks) < min_hist + 1:
                continue
            # lấy vài điểm cắt trong câu
            for k in range(min_hist, min(len(toks), min_hist + 4)):
                cases.append((toks[:k], toks[k]))
    rng = random.Random(seed)
    rng.shuffle(cases)
    return cases[:max_cases]


def _eval(wskip: WSkip, cases, vocab, *, topk: int, ctx_max: int, d1_only: bool,
          lam: float = 0.85, gidf_fn=None, gidf_min: float = 0.0):
    """d1_only: che mats[d>1] — baseline chỉ bigram."""
    if d1_only:
        mats = list(wskip.mats)
        empty = mats[0]
        for d in range(2, wskip.d_max + 1):
            mats[d] = empty
        w = WSkip(mats, d_max=wskip.d_max, min_count=wskip.min_count,
                  n_spans=wskip.n_spans, n_edges=wskip.n_edges)
    else:
        w = wskip

    n = len(cases)
    hard_nonempty = hard_hit = soft10 = multi = soft10_multi = 0
    demos = []
    for hist, gold in cases:
        res = constrain_next(
            w, hist, vocab, topk=topk, ctx_max=ctx_max, lam=lam,
            gidf_fn=gidf_fn, gidf_min=gidf_min,
        )
        hard = res["hard"]
        soft = res["soft"]
        if hard:
            hard_nonempty += 1
            if gold in hard:
                hard_hit += 1
        ranked = sorted(soft, key=lambda t: -soft[t])[:10]
        if gold in ranked:
            soft10 += 1
        if res.get("n_active", 0) >= 2:
            multi += 1
            if gold in ranked:
                soft10_multi += 1
        if len(demos) < 8 and len(hist) >= 2:
            demos.append((hist, gold, res, ranked[:5]))
    return {
        "n": n,
        "hard_nonempty": hard_nonempty / max(n, 1),
        "hard_hit_given": hard_hit / max(hard_nonempty, 1),
        "hard_hit_all": hard_hit / max(n, 1),
        "soft10": soft10 / max(n, 1),
        "soft10_multi": soft10_multi / max(multi, 1),
        "n_multi": multi,
        "demos": demos,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--log-file", default="out/w_skip_mini.log")
    ap.add_argument("--d-max", type=int, default=12)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--ctx-max", type=int, default=8)
    ap.add_argument("--max-cases", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gidf-min", type=float, default=4.0,
                    help="lọc hub khỏi đề xuất (0 = tắt); mặc định 4.0 như cột lạc")
    args = ap.parse_args()

    logf = (_ROOT / args.log_file).open("w", encoding="utf-8")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    log(f"[{time.time()-t0:.0f}s] load {args.checkpoint} …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    log(f"[{time.time()-t0:.0f}s] ✓ {len(state.parents):,} cha")

    clear_w_skip_cache()
    log(f"[{time.time()-t0:.0f}s] dựng W_skip d_max={args.d_max} min_count={args.min_count} …")
    wskip = build_w_skip(
        state, d_max=args.d_max, min_count=args.min_count, content_only=True
    )
    nnz = [int(wskip.mats[d].nnz) for d in range(1, args.d_max + 1)]
    log(f"[{time.time()-t0:.0f}s] ✓ spans={wskip.n_spans:,} edge_adds={wskip.n_edges:,}")
    log(f"  nnz theo d: {nnz}")

    cases = _collect_cases(state, max_cases=args.max_cases, seed=args.seed)
    log(f"[{time.time()-t0:.0f}s] {len(cases)} ca đo (hist→gold trong câu)")
    log(f"  gidf_min={args.gidf_min}")

    ps = state.parents
    gidf_fn = (lambda tid, _ps=ps: float(_ps._gidf(int(tid)))) if args.gidf_min > 0 else None  # noqa: SLF001
    kw = dict(topk=args.topk, gidf_fn=gidf_fn, gidf_min=args.gidf_min)

    full = _eval(
        wskip, cases, state.vocab, ctx_max=args.ctx_max, d1_only=False, **kw
    )
    d1 = _eval(
        wskip, cases, state.vocab, ctx_max=args.ctx_max, d1_only=True, **kw
    )
    sweep = {}
    for cm in (2, 3, 4, 8):
        sweep[cm] = _eval(
            wskip, cases, state.vocab, ctx_max=cm, d1_only=False, **kw
        )

    log("\n" + "=" * 72)
    log(f"{'thước':<28}{'W_skip d≤D':>14}{'chỉ d=1':>14}")
    log("=" * 72)
    log(f"{'hard∩ ≠ ∅':<28}{full['hard_nonempty']:>14.3f}{d1['hard_nonempty']:>14.3f}")
    log(f"{'gold∈hard | hard≠∅':<28}{full['hard_hit_given']:>14.3f}{d1['hard_hit_given']:>14.3f}")
    log(f"{'gold∈hard (mọi ca)':<28}{full['hard_hit_all']:>14.3f}{d1['hard_hit_all']:>14.3f}")
    log(f"{'gold∈soft top-10':<28}{full['soft10']:>14.3f}{d1['soft10']:>14.3f}")
    log(f"{'soft10 | ≥2 proposer':<28}{full['soft10_multi']:>14.3f}{d1['soft10_multi']:>14.3f}")
    log("=" * 72)
    log("quét ctx_max (W_skip đầy):")
    for cm, r in sweep.items():
        log(f"  ctx={cm}: hard≠∅={r['hard_nonempty']:.3f}  "
            f"gold|hard={r['hard_hit_given']:.3f}  soft10={r['soft10']:.3f}  "
            f"soft10|multi={r['soft10_multi']:.3f} (n_multi={r['n_multi']})")

    g1 = full["hard_nonempty"] >= 0.30
    g2 = full["hard_hit_given"] >= 0.40
    g3 = full["soft10"] >= 0.50
    g4 = (full["soft10"] - d1["soft10"]) >= 0.05
    # G4b: trên ca ≥2 proposer, soft có hơn d=1 không?
    g4b = (full["soft10_multi"] - d1["soft10_multi"]) >= 0.05
    log(f"G1 hard∩≠∅ ≥0.30          : {'ĐẠT' if g1 else 'TRƯỢT'} ({full['hard_nonempty']:.3f})")
    log(f"G2 gold∈hard|≠∅ ≥0.40     : {'ĐẠT' if g2 else 'TRƯỢT'} ({full['hard_hit_given']:.3f})")
    log(f"G3 soft top-10 ≥0.50      : {'ĐẠT' if g3 else 'TRƯỢT'} ({full['soft10']:.3f})")
    log(f"G4 soft10 − d1 ≥+0.05     : {'ĐẠT' if g4 else 'TRƯỢT'} "
        f"({full['soft10']-d1['soft10']:+.3f})")
    log(f"G4b soft10|multi − d1 ≥+0.05: {'ĐẠT' if g4b else 'TRƯỢT'} "
        f"({full['soft10_multi']-d1['soft10_multi']:+.3f})")
    best_cm = max(sweep, key=lambda c: sweep[c]["soft10_multi"])
    log(f"ctx_max tốt nhất theo soft10|multi: {best_cm} "
        f"({sweep[best_cm]['soft10_multi']:.3f})")
    if (g3 and g4) or g4b:
        log("⇒ ĐÁNG thử cắm kênh W_skip vào sinh (A/B, mặc định vẫn OFF)")
    elif g1 and g2:
        log("⇒ hard∩ có precision khi không rỗng; soft chưa thắng d=1 — A/B nhẹ hoặc chỉnh topk")
    else:
        log("⇒ chưa đủ để cắm mặc định; xem demo / chỉnh topk·min_count")

    log("\n── demo (tối đa 8) ──")
    for hist, gold, res, top5 in full["demos"]:
        hs = " ".join(_surface(state.vocab, t) for t in hist)
        gs = _surface(state.vocab, gold)
        props = ", ".join(
            f"{_surface(state.vocab, p['src'])}@d{p['d']}→{p['n_prop']}"
            for p in res["proposers"]
        )
        hard_s = " ".join(_surface(state.vocab, t) for t in list(res["hard"])[:8])
        top_s = " ".join(_surface(state.vocab, t) for t in top5)
        hit_h = "✓" if gold in res["hard"] else "✗"
        hit_s = "✓" if gold in top5 or gold in sorted(res["soft"], key=lambda t: -res["soft"][t])[:10] else "✗"
        log(f"  [{hs}] → gold={gs}  hard{hit_h} soft10{hit_s}")
        log(f"     proposers: {props}")
        log(f"     hard∩: {{{hard_s}}}{'…' if len(res['hard']) > 8 else ''}")
        log(f"     soft top: {top_s}")

    # ví dụ tay nếu encode được
    for prompt, expect in (
        ("Việt Nam", None),
        ("Nhà Lý", None),
        ("Ngô Quyền", None),
    ):
        try:
            ids = state.vocab.encode_prompt(prompt)
        except Exception:
            continue
        res = constrain_next(
            wskip, ids, state.vocab, topk=args.topk, ctx_max=args.ctx_max,
            gidf_fn=gidf_fn, gidf_min=args.gidf_min,
        )
        hard_s = " ".join(_surface(state.vocab, t) for t in list(res["hard"])[:12])
        soft_ranked = sorted(res["soft"], key=lambda t: -res["soft"][t])[:8]
        soft_s = " ".join(_surface(state.vocab, t) for t in soft_ranked)
        log(f"\n  tay '{prompt}' ids={len(ids)} hard∩={{{hard_s}}} soft={{ {soft_s} }}")

    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
