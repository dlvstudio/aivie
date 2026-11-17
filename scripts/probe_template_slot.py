#!/usr/bin/env python
"""P99 (TK_047 §8) — **CHA CỤ THỂ LÀM KHUÔN, THAY KHE**: có phải chép trá hình không?

§7 giết tầng K: khung = **chuỗi nhãn** không phải đơn vị đúng (phủ nhãn lên, chia sẻ xuống).
Ứng viên còn lại cho "vật thể cam kết toàn cục": **một cha CỤ THỂ**. Khuôn khi đó bảo đảm đúng
ngữ pháp vì nó **là** một câu người viết; cái mới đến từ **phép thay khe**.

Điểm mấu chốt: không cần cơ chế mới nào. "Được phép thay" định nghĩa **thuần đếm**, và bộ xếp
hạng ứng viên chính là `tach_lat` đã thắng `E` **1,60×** (§13.67) — nó xếp hạng *ứng viên trong
một khung*, mà một khe **chính là** một khung.

## Định nghĩa (chặt nhất có thể, không tham số)

Vị trí `i` của cha `p` có bối cảnh `(a, x, b)`. Token `c` **được phép** thay `x` nếu bộ ba
`(a, c, b)` **có thật ở một cha khác**. Không ngưỡng, không trọng số, không tham số điều chỉnh.

## Ba câu hỏi, ba thước

| | |
|---|---|
| **có thay được không** | bao nhiêu vị trí có ≥1 ứng viên khác bản gốc |
| **có mới không** | `log10(∏ ứng viên mỗi vị trí)` — số câu khuôn cho phép so với **1** câu đã thấy |
| **thay có ĐÚNG không** | suy phép thay trên nửa **A**, xác nhận trên nửa **B** *(như P93)* |

Thước thứ ba có hai mức:
- **hẹp**: bộ ba `(a, c, b)` xuất hiện ở **B** — bối cảnh cục bộ có thật ở nơi khác;
- **chủ đề**: `c` đồng hiện ở **B** với ≥2 token nội dung khác **của chính câu đó** — phép thay
  không phá chủ đề. §13.41 đo hỏng **phẳng theo vị trí** ⇒ lệch chủ đề mới là lỗi thật.

⚠️ **Đối chứng bắt buộc**: `ngau_nhien_cung_loai` — bốc `c` ngẫu nhiên **cùng từ loại**, cùng
ngân sách. Không có cột này thì "được phép" có thể chẳng nghĩa gì (§13.20: token phổ biến đồng
hiện với mọi thứ).

## CỔNG G-tmpl (nêu TRƯỚC)

1. **≥300 cha** có ≥1 vị trí thay được.
2. **Tổ hợp mới**: trung vị `log10(∏ ứng viên)` ≥ **1,0** — cùng thanh với cổng (3) của P98 để
   so được. Dưới đó thì "khuôn + thay" **là chép có trang trí**.
3. **Được-phép phải hơn ngẫu nhiên**: thước **chủ đề** ≥ **2,0×** `ngau_nhien_cung_loai`.

## Dự đoán ghi TRƯỚC (Claude)

- (1) **đạt** dễ.
- (2) **thật sự không chắc** — bộ ba là ràng buộc chặt trên kho 5.664 cha. Đoán **0,7–1,5**,
  tức **borderline**, có thể trượt sát.
- (3) đoán **đạt, 2–4×**. Nhưng cảnh báo trước: nếu **đối chứng cũng cao** thì đó là **bão hoà**
  (§13.20), và tôi ghi **chưa phân giải được** chứ không đọc là thắng.
- Nếu (2) trượt: kết luận là **kho quá nhỏ cho phép thay theo bộ ba**, và câu hỏi kế là đo lại
  trên v12 (76×) chứ không phải nới định nghĩa "được phép" cho lỏng ra.
- Chạy **hai lần** trước khi đọc.

Chạy: `PYTHONPATH=src python scripts/probe_template_slot.py`
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

GATE_MIN_PARENT = 300
GATE_NOVEL_MIN = 1.0
GATE_VS_RANDOM = 2.0
# TK_047 §13 — CỔNG TUYỆT ĐỐI, thêm sau khi vòng ép thay cho thấy tỷ số một mình ĐÁNH LỪA.
# Trên mini: hẹp 23,4% = "17× sàn ngẫu nhiên" ⇒ tôi đọc là tốt; thực chất 76,6% phép thay
# KHÔNG tái xuất ở nửa kia, và ép thay tụt −42,5đ. Phép thay chỉ có cơ hội không-gây-hại khi
# ĐA SỐ phép thay được chứng thực ⇒ ngưỡng tự nhiên là 50%.
GATE_NARROW_MIN = 0.50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa_pos.pkl")
    ap.add_argument("--map", default="out/token_labels_mini.json")
    ap.add_argument("--n-parent", type=int, default=600)
    ap.add_argument("--max-index-parent", type=int, default=0,
                    help="giới hạn số cha dùng dựng chỉ mục mỗi nửa (0 = không giới hạn). "
                         "Dùng khi RAM không đủ cho v12.")
    ap.add_argument("--gidf-min", type=float, default=4.0)
    ap.add_argument("--licence", choices=("bo_ba", "lop", "giao"), default="bo_ba",
                    help="bo_ba = (a,c,b) có ở cha khác (TK_047) · "
                         "lop = c là BẠN CÙNG LỚP THAY THẾ của x (TK_048) · "
                         "giao = GIAO cả hai — khớp bối cảnh cục bộ VÀ cùng lớp")
    ap.add_argument("--min-share", type=int, default=3)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/probe_template_slot.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_template_slot")
    t0 = time.time()
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v, ps = state.vocab, state.parents
    n = len(v)

    keep = np.zeros(n, dtype=bool)
    for t in range(n):
        if is_structural_token(v, t) or not v.is_surface(t):
            continue
        try:
            if float(ps._gidf(t)) >= args.gidf_min:  # noqa: SLF001
                keep[t] = True
        except Exception:
            continue

    # Bản đồ nhãn chỉ dùng cho ĐỐI CHỨNG cùng-từ-loại (cổng 3). Cổng 4 — độ chính xác TUYỆT
    # ĐỐI, cột quyết định — KHÔNG cần nhãn. Nên thiếu bản đồ thì chạy tiếp với đối chứng yếu
    # hơn và NÓI RÕ, thay vì chặn cả phép đo.
    _mf = _ROOT / args.map
    mp = json.loads(_mf.read_text(encoding="utf-8"))["tu_loai"] if _mf.exists() else {}
    if not mp:
        logger.log(f"⚠️ KHÔNG có bản đồ nhãn ({args.map}) ⇒ đối chứng bốc trong TOÀN BỘ token "
                   f"nội dung, KHÔNG cùng từ loại. Cổng (3) yếu đi; cổng (4) không ảnh hưởng.")
    lab: dict[int, str] = {}
    for ts, d in mp.items():
        if not d:
            continue
        best = max(d.values())
        top = sorted(k for k, x in d.items() if x == best)
        if len(top) == 1:
            lab[int(ts)] = top[0]
    by_lab: dict[str, list[int]] = defaultdict(list)
    for t, g in lab.items():
        if 0 <= t < n and keep[t]:
            by_lab[g].append(t)
    _all_keep = [int(t) for t in np.nonzero(keep)[0]]
    if not by_lab:
        by_lab["(khong_nhan)"] = _all_keep

    rng = random.Random(args.seed)
    idx = list(range(len(ps.parents)))
    rng.shuffle(idx)
    half = len(idx) // 2
    A, B = idx[:half], idx[half:]

    def tri_index(par: list[int]) -> dict[tuple[int, int], set[int]]:
        d: dict[tuple[int, int], set[int]] = defaultdict(set)
        for k_, pi in enumerate(par):
            if k_ and k_ % 50000 == 0:
                logger.log(f"  chỉ mục A {k_:,}/{len(par):,} · bối cảnh {len(d):,} "
                           f"({time.time()-t0:.0f}s)")
            e = [int(t) for t in ps.parents[pi].emit_ids]
            for i in range(1, len(e) - 1):
                if 0 <= e[i] < n and keep[e[i]]:
                    d[(e[i - 1], e[i + 1])].add(e[i])
        return d

    if args.max_index_parent:
        A = A[: args.max_index_parent]
        B = B[: args.max_index_parent]
        logger.log(f"giới hạn chỉ mục: {args.max_index_parent:,} cha mỗi nửa")
    triA = tri_index(A)

    # --- TK_048 → TK_047: CẤP PHÉP BẰNG LỚP THAY THẾ -------------------------------------
    # ⚠️ So sánh 84,5% (lớp) với 43,8% (bộ ba) là KHÔNG cùng thước: thước lớp hỏi "y có điền
    # được MỘT bối cảnh nào đó của x ở B không", còn thước bộ ba hỏi "ĐÚNG bộ ba (a,c,b) có ở
    # B không" — chặt hơn hẳn. Vòng này bắt lớp thay thế chịu ĐÚNG thước của bộ ba.
    lop_nb: dict[int, list[int]] = {}
    if args.licence in ("lop", "giao"):
        ctxA: dict[int, set] = defaultdict(set)
        for pi in A:
            e = [int(t) for t in ps.parents[pi].emit_ids]
            for i in range(1, len(e) - 1):
                if 0 <= e[i] < n and keep[e[i]]:
                    ctxA[e[i]].add((e[i - 1], e[i + 1]))
        ctxA = {t: c for t, c in ctxA.items() if len(c) >= 3}
        invA: dict[tuple, list[int]] = defaultdict(list)
        for t, cs in ctxA.items():
            for c in cs:
                invA[c].append(t)
        for x, cs in ctxA.items():
            sh: dict[int, int] = defaultdict(int)
            for c in cs:
                for y in invA[c]:
                    if y != x:
                        sh[y] += 1
            nx = len(cs) ** 0.5
            sc = sorted(((m / (nx * len(ctxA[y]) ** 0.5), y)
                         for y, m in sh.items() if m >= args.min_share), reverse=True)
            if sc:
                lop_nb[x] = [y for _s, y in sc[: args.topk]]
        logger.log(f"lớp thay thế (từ nửa A): {len(lop_nb):,} đơn vị có bạn cùng lớp")
    triB_set: set[tuple[int, int, int]] = set()
    coB: dict[int, set[int]] = defaultdict(set)
    for pi in B:
        e = [int(t) for t in ps.parents[pi].emit_ids]
        for i in range(1, len(e) - 1):
            triB_set.add((e[i - 1], e[i], e[i + 1]))
        u = {t for t in e if 0 <= t < n and keep[t]}
        for t in u:
            coB[t] |= u
    logger.log(f"A {len(A):,} cha · B {len(B):,} cha · bối cảnh A {len(triA):,} "
               f"({time.time()-t0:.0f}s)")

    rng2 = random.Random(args.seed + 1)
    pool_A = [pi for pi in A]
    rng2.shuffle(pool_A)

    n_par_ok = 0
    novel: list[float] = []
    n_slot = n_slot_sub = 0
    n_cand: list[int] = []
    stat = {k: {"n": 0, "hep": 0, "chu_de": 0}
            for k in ("duoc_phep", "ngau_nhien_cung_loai")}
    ex: list[str] = []

    from nn_aivie.hca.readout import decode_surface  # noqa: PLC0415

    for pi in pool_A[: args.n_parent]:
        e = [int(t) for t in ps.parents[pi].emit_ids]
        others = {t for t in e if 0 <= t < n and keep[t]}
        logs = 0.0
        got = False
        for i in range(1, len(e) - 1):
            x = e[i]
            if not (0 <= x < n) or not keep[x]:
                continue
            n_slot += 1
            _tri_c = triA.get((e[i - 1], e[i + 1]), set()) - {x}
            _lop_c = set(lop_nb.get(x, ()))
            cands = (_lop_c if args.licence == "lop"
                     else (_tri_c & _lop_c) if args.licence == "giao" else _tri_c)
            if not cands:
                continue
            n_slot_sub += 1
            got = True
            n_cand.append(len(cands))
            logs += math.log10(len(cands) + 1)          # +1 = giữ nguyên bản gốc

            c = rng2.choice(sorted(cands))
            g = lab.get(x)
            pool = by_lab.get(g or "", []) or by_lab.get("(khong_nhan)", _all_keep)
            rc = rng2.choice(pool) if pool else None
            for name, cc in (("duoc_phep", c), ("ngau_nhien_cung_loai", rc)):
                if cc is None:
                    continue
                stat[name]["n"] += 1
                if (e[i - 1], cc, e[i + 1]) in triB_set:
                    stat[name]["hep"] += 1
                if len((coB.get(cc, set()) & others) - {cc}) >= 2:
                    stat[name]["chu_de"] += 1
            if len(ex) < 12 and len(cands) >= 2:
                ex.append(f"{decode_surface(state, [e[i-1], x, e[i+1]])!r}"
                          f"  →  thay {decode_surface(state, [x])!r} bằng "
                          f"{decode_surface(state, [c])!r}")
        if got:
            n_par_ok += 1
            novel.append(logs)

    med_novel = statistics.median(novel) if novel else 0.0
    print(f"\ncha có ≥1 vị trí thay được: {n_par_ok}/{min(args.n_parent, len(pool_A))}")
    print(f"khe nội dung: {n_slot:,} · thay được {n_slot_sub:,} = "
          f"{n_slot_sub/max(n_slot,1):.1%}")
    print(f"ứng viên/khe — trung vị {statistics.median(n_cand) if n_cand else 0:.0f}"
          f" · trung bình {statistics.mean(n_cand) if n_cand else 0:.1f}")
    print(f"TỔ HỢP MỚI (log10, trung vị) = {med_novel:.2f}  ⇒ ~{10**med_novel:,.0f}× câu")

    print(f"\n{'bộ thay':<24}{'n':>7}{'hẹp':>9}{'chủ đề':>10}")
    res = {}
    for k, s in stat.items():
        nn = max(s["n"], 1)
        res[k] = {"n": s["n"], "hep": s["hep"] / nn, "chu_de": s["chu_de"] / nn}
        print(f"{k:<24}{s['n']:>7}{s['hep']/nn:>8.1%}{s['chu_de']/nn:>10.1%}")

    r = res["duoc_phep"]["chu_de"] / max(res["ngau_nhien_cung_loai"]["chu_de"], 1e-9)
    g1 = n_par_ok >= GATE_MIN_PARENT
    g2 = med_novel >= GATE_NOVEL_MIN
    g3 = r >= GATE_VS_RANDOM
    g4 = res["duoc_phep"]["hep"] >= GATE_NARROW_MIN
    print(f"\nduoc_phep/ngau_nhien (chủ đề) = {r:.2f}×")
    print(f"ĐỘ CHÍNH XÁC TUYỆT ĐỐI (hẹp)  = {res['duoc_phep']['hep']:.1%}"
          f"   (mini: 23,4% · cổng ≥{GATE_NARROW_MIN:.0%})   ← CỘT QUYẾT ĐỊNH")
    print(f"CỔNG G-tmpl: (1) {'✅' if g1 else '❌'} · (2) tổ hợp mới "
          f"{'✅' if g2 else '❌'} · (3) hơn ngẫu nhiên {'✅' if g3 else '❌'} · "
          f"(4) TUYỆT ĐỐI {'✅' if g4 else '❌'} ⇒ "
          f"{'ĐẠT' if (g1 and g2 and g3 and g4) else 'TRƯỢT'}")
    if res["ngau_nhien_cung_loai"]["chu_de"] > 0.5:
        print("⚠️ đối chứng ngẫu nhiên >50% ⇒ BÃO HOÀ (§13.20), thước chủ đề chưa phân giải")
    if ex:
        print("\nví dụ phép thay:")
        for e_ in ex:
            print(f"  {e_}")

    (_ROOT / args.out).write_text(json.dumps(
        {"checkpoint": args.checkpoint, "n_parent": n_par_ok,
         "khe": n_slot, "khe_thay_duoc": n_slot_sub,
         "to_hop_moi_log10": med_novel, "ket_qua": res, "ty_so_chu_de": r,
         "gate": {"min_parent": GATE_MIN_PARENT, "novel_min": GATE_NOVEL_MIN,
                  "vs_random": GATE_VS_RANDOM, "narrow_min": GATE_NARROW_MIN},
         "gate_pass": bool(g1 and g2 and g3 and g4), "vi_du": ex,
         "seconds": round(time.time() - t0, 1)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    logger.log(f"→ {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
