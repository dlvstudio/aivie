"""Đo `via_pivot` phủ được bao nhiêu CẠNH LỚP có thật — thước cho `head_rule`.

Nút thắt đo 2026-08-12: dữ liệu có 797 cạnh `thành viên → lớp` (164 lớp ≥3
thành viên) nhưng `via_pivot` chỉ thấy 18,1%, vì max-gidf nhặt BỔ NGỮ thay vì
ĐẦU LỚP (`Cuba là quốc gia duy nhất` → `duy nhất`).

Thước nằm TRONG đồ thị: so cạnh cơ chế dựng được với cạnh đọc thẳng từ câu cha,
KHÔNG qua chấm người. Chạy mọi `head_rule` trong MỘT lần nạp checkpoint.

Chạy::

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/measure_pivot_coverage.py \
        --checkpoint checkpoints/classrich.pkl \
        --ws-spans checkpoints/classrich.ws_spans.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.tokenizer import MARK_SPACE  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402

_DET = frozenset({"một", "các", "những", "mọi", "mỗi"})
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:9.1f}s] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ws-spans", default="")
    ap.add_argument("--out", default="out/pivot_coverage.json")
    ap.add_argument("--rules", default="max_gidf,trai_nhat,trai_du_dac_thu,dau_lop")
    ap.add_argument("--indeg-q", default="0.0", help="phân vị lọc BẬC-VÀO, phẩy ngăn")
    ap.add_argument("--head-rule-trai", default="max_gidf")
    ap.add_argument("--gidf-q", default="0.20", help="phân vị lọc gidf nhãn, phẩy ngăn")
    ap.add_argument("--lookahead", default="3", help="cửa sổ sau pivot, phẩy ngăn")
    ap.add_argument("--d-max", type=int, default=3)
    ap.add_argument("--min-count", type=float, default=2.0)
    args = ap.parse_args()

    log(f"nạp {args.checkpoint} …")
    state, _ = load_checkpoint(args.checkpoint)
    vocab, ps = state.vocab, state.parents
    n = len(vocab)
    raw = [str(vocab.token_of(i)) for i in range(n)]
    S = [r.replace(MARK_SPACE, "").strip() for r in raw]
    BAD = {i for i in range(n) if re.match(r"^<.*>$", S[i]) or S[i].endswith(":")}
    LA = {i for i in range(n) if S[i] == "là"}
    DET = {i for i in range(n) if S[i].lower() in _DET}
    FUNC = {i for i in range(n)
            if S[i].lower() in H._FUNC or len(S[i]) <= 1
            or MARK_SPACE not in raw[i]} | BAD

    # ── chân lý: cạnh lớp đọc THẲNG từ câu cha ────────────────────────────
    log("đọc cạnh lớp từ câu cha …")
    cls: dict[str, set[str]] = defaultdict(set)
    pairs: set[tuple[str, str, str]] = set()
    for rec in ps.parents:
        seq = list(rec.emit_ids)
        for k, t in enumerate(seq):
            if t not in LA or k == 0 or k + 1 >= len(seq):
                continue
            mem = seq[k - 1]
            if mem in FUNC or mem in DET:
                continue
            j = k + 1
            while j < len(seq) and seq[j] in DET:
                j += 1
            if j >= len(seq) or seq[j] in FUNC:
                continue
            head = S[seq[j]]
            full = head + (" " + S[seq[j + 1]]
                           if j + 1 < len(seq) and seq[j + 1] not in FUNC else "")
            cls[full].add(S[mem])
            pairs.add((S[mem], full, head))
    big = {c for c, m in cls.items() if len(m) >= 3}
    want = {(a, c, h) for (a, c, h) in pairs if c in big}
    log(f"  lớp ≥3 thành viên = {len(big)} · cạnh lớp = {len(want)}")

    wsp = None
    if args.ws_spans and Path(args.ws_spans).exists():
        wsp = H.load_ws_spans(args.ws_spans)
        log(f"  ws_spans = {len(wsp) if wsp else 0}")

    res: dict[str, dict] = {}
    combos = [(r.strip(), float(q), float(g), int(la))
              for r in args.rules.split(",") if r.strip()
              for q in args.indeg_q.split(",") if q.strip()
              for g in args.gidf_q.split(",") if g.strip()
              for la in args.lookahead.split(",") if la.strip()]
    for rule, iq, gq, la in combos:
        t0 = time.time()
        ten = f"indeg={iq} gidf_q={gq} look={la}"
        log(f"── {ten} · dựng index …")
        idx = H.build_hub_spread_index(
            state, d_max=args.d_max, min_count=args.min_count,
            head_rule=rule, head_rule_trai=args.head_rule_trai,
            label_indeg_q=iq, label_gidf_q=gq, head_lookahead=la, ws_spans=wsp)
        vp: dict[str, set[str]] = defaultdict(set)
        for src, dsts in idx.via_pivot.items():
            a = H._surf_ix(idx, vocab, src)
            for dst in dsts:
                vp[a].add(H._surf_ix(idx, vocab, dst))
        hit = sum(1 for a, c, h in want
                  if c in vp.get(a, ()) or h in vp.get(a, ()))
        n_edge = sum(len(d) for d in idx.via_pivot.values())
        pct = 100.0 * hit / max(len(want), 1)
        res[ten] = {"phu": hit, "tong_can": len(want), "pct": round(pct, 1),
                     "canh_typed": n_edge, "giay": round(time.time() - t0, 1)}
        log(f"   PHỦ {hit}/{len(want)} = {pct:.1f}%  · cạnh typed={n_edge}"
            f"  ({time.time() - t0:.0f}s)")
        # vài ca để đọc tay
        ex = [(a, c, sorted(vp.get(a, ()))[:4]) for a, c, h in sorted(want)
              if not (c in vp.get(a, ()) or h in vp.get(a, ()))][:5]
        res[ten]["mat_mau"] = [{"cap": f"{a} là {c}", "pivot_thay": v}
                                for a, c, v in ex]
        for a, c, v in ex:
            log(f"     mất: «{a} là {c}» → thấy {v}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    log("═══ TỔNG ═══")
    for r, d in sorted(res.items(), key=lambda kv: -kv[1]["pct"]):
        log(f"  {r:18} phủ {d['pct']:5.1f}%  ({d['phu']}/{d['tong_can']})"
            f"  cạnh typed={d['canh_typed']}")
    log(f"✓ → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
