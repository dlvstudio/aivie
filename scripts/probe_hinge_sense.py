#!/usr/bin/env python
"""P73 (TK_045 §13.39) — **BỆNH BẢN LỀ ĐỒNG TỰ**: đo trực tiếp, không đoán.

**Bệnh (§13.25).** Mối nối bắc cầu qua **một token bản lề**: `last_content(A) → first_content(B)`
qua `W_skip`. Tiếng Việt dùng **chung một id** cho đồng tự ⇒ `bệnh nhân` nối sang `2 nhân 3`,
`tỉ lệ` → `lệ quỷ`, `ví dụ` → `dụ quân địch` (§13.37 có ví dụ dày).

**Đã đóng một lối chữa:** §13.31 `S6` bỏ hẳn bản lề rồi chấm ngữ cảnh lân cận — AUC **.404**,
dưới ngẫu nhiên. Ngữ cảnh lân cận trong `W_skip` **không** mang thông tin nối.

**Chưa ai đo:** bệnh này **to cỡ nào**? Suốt tám vòng nó chỉ được thấy bằng **ví dụ**, chưa
bằng **tỷ lệ**. Không có số thì không biết chữa nó đáng bao nhiêu.

**Cách đo.** Nghĩa của một âm tiết đồng tự do **chữ ĐỨNG TRƯỚC nó** quyết định: `bệnh` trong
`bệnh nhân`, `tỉ` trong `tỉ lệ`. Với mỗi mối nối `(w_left, t, w_right)`:

    hop_le = P( w_left đứng ngay trước t │ t đồng hiện với w_right trong khoảng d )

Nếu `W_skip[t, w_right]` được chống đỡ **toàn bằng** những lần `t` mang nghĩa KHÁC (tức
`w_left` gần như không bao giờ đứng trước), thì mối nối đó **vay** đồng hiện của nghĩa khác —
đúng định nghĩa của bệnh, và đếm được.

**Đối chứng nằm trong mạng** (như §13.38): mối nối THẬT giữa hai mệnh đề **liên tiếp của cùng
một cha**. Đó là chỗ người viết thật nối câu. Không có nó thì mọi con số `hop_le` đều "trông
thấp".

Chỉ đếm ⇒ Luật 0 sạch. Script không sửa cơ chế.

Chạy:
  PYTHONPATH=src python scripts/probe_hinge_sense.py --n-prompt 400
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.clause_layer import (  # noqa: E402
    CoverageGate,
    build_clause_layer,
    compose_chain,
    content_folds,
    ensure_boundary_signal,
    word_pivot_strength,
)
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.hca.w_skip import ensure_w_skip  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

# --- CỔNG G-hinge, nêu TRƯỚC khi chạy (§13.39) -------------------------------------------
# Bệnh chỉ đáng chữa nếu mối nối SINH mang nghĩa sai **rõ rệt hơn** mối nối THẬT.
GATE_GAP_MIN = 0.15       # (1) hợp-lệ trung bình: THẬT − SINH ≥ 0,15
GATE_ZERO_GAP_MIN = 0.10  # (2) tỷ lệ nối "vay hẳn nghĩa khác" (hop_le = 0): SINH − THẬT ≥ 0,10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n-prompt", type=int, default=400)
    ap.add_argument("--max-clauses", type=int, default=5)
    ap.add_argument("--d", type=int, default=8, help="khoảng đồng hiện, khớp d_max của W_skip")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/probe_hinge_sense.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_hinge_sense")
    t0 = time.time()
    logger.log(f"P73 bản lề đồng tự · {args.checkpoint} · {args.n_prompt} đề")
    logger.log(f"cổng: khoảng cách hợp-lệ ≥{GATE_GAP_MIN} · chênh 'vay hẳn' ≥{GATE_ZERO_GAP_MIN}")

    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    layer = build_clause_layer(state, pmi_theta=0.0, logger=logger)
    ws = ensure_w_skip(state, d_max=8, min_count=1, content_only=False)
    logger.log(f"tầng mệnh đề {len(layer):,} ({time.time()-t0:.0f}s)")

    # --- kho văn: ghép mệnh đề theo thứ tự trong từng cha ---------------------------------
    seqs: list[list[int]] = []
    si_of_parent: dict[int, int] = {}
    off_of_parent: dict[int, list[int]] = {}   # cha → offset đầu mỗi mệnh đề trong chuỗi
    for _p, ids in layer.by_parent.items():
        s: list[int] = []
        offs: list[int] = []
        for ci in ids:
            offs.append(len(s))
            s.extend(int(t) for t in layer.clauses[ci].tokens)
        if len(s) >= 3:
            si_of_parent[_p] = len(seqs)
            off_of_parent[_p] = offs
            seqs.append(s)
    # vị trí của từng token: (chỉ số câu, vị trí)
    pos: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for si, s in enumerate(seqs):
        for i, t in enumerate(s):
            pos[t].append((si, i))
    logger.log(f"kho văn {len(seqs):,} chuỗi · {sum(len(s) for s in seqs):,} token "
               f"({time.time()-t0:.0f}s)")

    # --- đếm bigram: để đo "bản lề ĐỔI VAI TRONG TỪ" ------------------------------------
    # Chữ ký thật của bệnh, đọc ra từ chính danh sách ví dụ: `quốc gia` ⇒ `gia vị`,
    # `Ban Nha` ⇒ `Nha biển`, `khai sinh` ⇒ `sinh ra`. Bản lề `gia` là âm tiết **CUỐI** của
    # từ trong A, rồi mối nối đọc nó thành âm tiết **ĐẦU** của một từ khác. Đo được bằng
    # bigram: t vừa bị buộc chặt sang TRÁI (w_left·t) vừa buộc chặt sang PHẢI (t·w_right).
    uni: Counter[int] = Counter()
    big: Counter[tuple[int, int]] = Counter()
    for s_ in seqs:
        uni.update(s_)
        big.update(zip(s_, s_[1:]))

    # ⚠️ KHÔNG định nghĩa lại công thức ở đây. `clause_layer` ghi rõ nguyên tắc "MỘT định
    # nghĩa DUY NHẤT" — probe tự tính lại thì probe và cơ chế lệch nhau âm thầm, đúng lớp
    # lỗi dự án này đã dính nhiều lần. Dùng thẳng hàm production.
    _bound = ensure_boundary_signal(state)

    def pivot(w_left: int, t: int, w_right: int, loo: bool = False) -> tuple[float, float]:
        """(sức buộc TRÁI, sức buộc PHẢI). `loo` chỉ để KIỂM hiện vật, xem ghi chú dưới."""
        pl = _bound.p_before(w_left, t)
        pr_ = _bound.p_after(t, w_right)
        if loo:
            # trừ chính lần xuất hiện này khỏi tử số (mẫu số lớn ⇒ bỏ qua chênh 1)
            nb = float(_bound._colmass[t]) or 1.0   # noqa: SLF001
            na = float(_bound._rowsum[t]) or 1.0    # noqa: SLF001
            pl = max(0.0, pl - 1.0 / nb)
            pr_ = max(0.0, pr_ - 1.0 / na)
        return pl, pr_

    d = args.d
    MAX_OCC = 4000   # token siêu phổ biến: lấy mẫu để khỏi quét cả kho

    def hop_le(w_left: int, t: int, w_right: int, skip: tuple[int, int] | None = None
               ) -> float | None:
        """P(w_left đứng ngay trước t | t đồng hiện w_right trong khoảng d).

        ⚠️ `skip` là **bắt buộc** cho đối chứng, không phải tuỳ chọn. Mối nối THẬT nằm ngay
        trong kho văn ⇒ nó **tự đếm chính nó**, và lần đó luôn hợp lệ (w_left thật sự đứng
        trước t ở đúng chỗ đó). Không bỏ ra thì đối chứng có `hop_le = 0` ở **0,0%** một cách
        máy móc — và bản đầu của probe này đã ra đúng con số đó, trông như bằng chứng đẹp mà
        thật ra là hiện vật. Mối nối SINH không có lần tự đếm ấy ⇒ so hai bên là so lệch.
        """
        occ = pos.get(t)
        if not occ:
            return None
        if len(occ) > MAX_OCC:
            occ = occ[::max(1, len(occ) // MAX_OCC)]
        n_co = n_ok = 0
        for si, i in occ:
            if skip is not None and si == skip[0] and i == skip[1]:
                continue
            s = seqs[si]
            lo, hi = max(0, i - d), min(len(s), i + d + 1)
            if w_right in s[lo:i] or w_right in s[i + 1:hi]:
                n_co += 1
                if i > 0 and s[i - 1] == w_left:
                    n_ok += 1
        return (n_ok / n_co) if n_co else None

    def joins_of(clauses) -> list[tuple[int, int, int, int]]:
        """(w_left, bản lề, chữ mở đầu B, vị trí bản lề TRONG A) cho từng mối nối."""
        out = []
        for a, b in zip(clauses, clauses[1:]):
            ta, tb = list(a.tokens), list(b.tokens)
            if len(ta) < 2 or not tb:
                continue
            hinge = int(a.last_content)
            try:
                k = len(ta) - 1 - ta[::-1].index(hinge)   # vị trí bản lề trong A
            except ValueError:
                continue
            if k == 0:
                continue
            out.append((int(ta[k - 1]), hinge, int(b.first_content), k))
        return out

    rng = random.Random(args.seed)
    openers = [c for c in layer.clauses if c.opens_parent and len(c.tokens) >= 4]
    rng.shuffle(openers)
    prompts = openers[: args.n_prompt]

    gen_scores: list[float] = []
    real_scores: list[float] = []
    gen_piv: list[float] = []
    real_piv: list[float] = []
    real_piv_noloo: list[float] = []
    ex_bad: list[str] = []
    ex_piv: list[str] = []

    for i, p in enumerate(prompts):
        if i and i % 100 == 0:
            logger.log(f"  {i}/{len(prompts)} ({time.time()-t0:.0f}s)")
        gate = CoverageGate(
            theme_folds=frozenset(content_folds(state, p.tokens)), anchor_min_folds=1)
        ch, _ = compose_chain(
            state, layer, ws, p.tokens, max_clauses=args.max_clauses,
            gate=gate, rng=random.Random(args.seed),
        )
        for wl, t, wr, _k in joins_of(ch):
            v = hop_le(wl, t, wr)
            if v is None:
                continue
            gen_scores.append(v)
            pl, pr_ = pivot(wl, t, wr)
            gen_piv.append(min(pl, pr_))
            if min(pl, pr_) >= 0.05 and len(ex_piv) < 10:
                ex_piv.append(
                    f"{decode_surface(state, [wl, t])!r} ⇒ {decode_surface(state, [t, wr])!r}"
                    f"  (trái {pl:.0%} · phải {pr_:.0%})")
            if v == 0.0 and len(ex_bad) < 12:
                ex_bad.append(
                    f"{decode_surface(state, [wl, t])!r} ⇒ {decode_surface(state, [t, wr])!r}")

    # đối chứng: mối nối THẬT giữa mệnh đề liên tiếp cùng cha
    crng = random.Random(args.seed + 7)
    multi = [p_ for p_, ids in layer.by_parent.items() if len(ids) >= 2 and p_ in si_of_parent]
    crng.shuffle(multi)
    for p_ in multi[: max(200, args.n_prompt)]:
        ids = layer.by_parent[p_]
        cls = [layer.clauses[j] for j in ids]
        si, offs = si_of_parent[p_], off_of_parent[p_]
        for j, (wl, t, wr, k) in enumerate(joins_of(cls)):
            # bỏ CHÍNH lần xuất hiện này ra khỏi mẫu số — xem docstring `hop_le`
            v = hop_le(wl, t, wr, skip=(si, offs[j] + k))
            if v is not None:
                real_scores.append(v)
            pl, pr_ = pivot(wl, t, wr, loo=True)   # bỏ chính mối nối này khỏi bigram
            real_piv.append(min(pl, pr_))
            pl0, pr0 = pivot(wl, t, wr, loo=False)  # nhạy cảm: KHÔNG trừ, để kiểm hiện vật
            real_piv_noloo.append(min(pl0, pr0))

    def summ(xs: list[float]) -> dict:
        if not xs:
            return {}
        return {"n": len(xs), "trung_binh": statistics.mean(xs),
                "trung_vi": statistics.median(xs),
                "ty_le_bang_0": sum(1 for x in xs if x == 0.0) / len(xs),
                "ty_le_tren_50pc": sum(1 for x in xs if x >= 0.5) / len(xs)}

    g, r = summ(gen_scores), summ(real_scores)
    print(f"\n{'':<24}{'SINH':>10}{'THẬT':>12}")
    for k, lab in (("n", "số mối nối"), ("trung_binh", "hợp-lệ trung bình"),
                   ("trung_vi", "hợp-lệ trung vị"), ("ty_le_bang_0", "vay HẲN nghĩa khác"),
                   ("ty_le_tren_50pc", "hợp-lệ ≥50%")):
        a, b = g.get(k, 0), r.get(k, 0)
        f = (lambda v: f"{v:,}") if k == "n" else (lambda v: f"{v:.1%}")
        print(f"{lab:<24}{f(a):>10}{f(b):>12}")

    def prate(xs):
        return sum(1 for x in xs if x >= 0.05) / len(xs) if xs else 0.0
    print(f"\nBẢN LỀ ĐỔI VAI TRONG TỪ (buộc chặt CẢ HAI phía ≥5%):")
    print(f"  THẬT không trừ self (kiểm hiện vật): {prate(real_piv_noloo):.1%}")
    print(f"  SINH {prate(gen_piv):.1%}  ·  THẬT {prate(real_piv):.1%}"
          f"  ⇒ chênh {prate(gen_piv) - prate(real_piv):+.1%}")
    if ex_piv:
        print("  ví dụ (SINH):")
        for e in ex_piv:
            print(f"    {e}")

    gap = r.get("trung_binh", 0) - g.get("trung_binh", 0)
    zgap = g.get("ty_le_bang_0", 0) - r.get("ty_le_bang_0", 0)
    g1, g2 = gap >= GATE_GAP_MIN, zgap >= GATE_ZERO_GAP_MIN
    print(f"\nkhoảng cách hợp-lệ (THẬT−SINH) = {gap:+.1%} · chênh 'vay hẳn' = {zgap:+.1%}")
    print(f"CỔNG G-hinge: (1) {'✅' if g1 else '❌'} · (2) {'✅' if g2 else '❌'}"
          f"  ⇒ {'ĐẠT' if (g1 and g2) else 'TRƯỢT'}")

    if ex_bad:
        print("\nMối nối VAY HẲN nghĩa khác (hợp-lệ = 0):")
        for e in ex_bad:
            print(f"  {e}")

    out = {"checkpoint": args.checkpoint, "n_prompt": len(prompts), "d": d,
           "gate": {"gap_min": GATE_GAP_MIN, "zero_gap_min": GATE_ZERO_GAP_MIN},
           "sinh": g, "that": r, "gap": gap, "zero_gap": zgap,
           "gate_pass": bool(g1 and g2), "vi_du_vay_han": ex_bad,
           "doi_vai_sinh": prate(gen_piv), "doi_vai_that": prate(real_piv),
           "doi_vai_that_khong_tru": prate(real_piv_noloo),
           "vi_du_doi_vai": ex_piv,
           "seconds": round(time.time() - t0, 1)}
    op = _ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"→ {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
