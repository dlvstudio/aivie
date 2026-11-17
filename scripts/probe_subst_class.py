#!/usr/bin/env python
"""P104 (TK_048 §1) — **NƠ-RON LỚP THAY THẾ**: trục *thay-thế-được*, khác hẳn trục của `E`.

## Chỗ trống được đo, không phải đoán

| có gì | |
|---|---|
| `E` | **bạn cùng NGỮ CẢNH** — `vua → Tông, ngôi, Lý, Thánh` (§16: syntagmatic) |
| nhãn từ loại | **8 lớp thô** — `khỉ` và `bàn` cùng là *danh từ* (§15: giúp 1,03×) |
| 🔴 **thiếu** | biểu diễn *"hai token này **thay thế được** cho nhau"* |

Ý tưởng gốc của tác giả (`khỉ` ~ `chó` ~ `mèo` ⇒ suy ra `là động vật`) đòi **đúng tầng giữa**
đó. TK_046 đo được nó **thắng `E` 1,60×** ở xếp hạng, **tái lập chính xác hai lần** — nhưng nó
**chưa bao giờ được đông cứng thành nơ-ron**: mỗi lần tính lại tại chỗ rồi vứt đi.

## Định nghĩa (thuần đếm, không tham số học)

**Bối cảnh** của token `x` = tập cặp `(a, b)` sao cho `a x b` xuất hiện trong corpus.

    sim(x, y) = |ctx(x) ∩ ctx(y)| / sqrt(|ctx(x)| · |ctx(y)|)

⚠️ **Phải chuẩn hoá theo tần suất.** Không chuẩn hoá thì token phổ biến chia bối cảnh với mọi
thứ và mọi cụm gom về một khối — đúng bệnh §13.20 đã giết hai đối chứng ở TK_046 §16.

## Vì sao khác cái ĐÃ HỎNG

Cấp phép của TK_047 nói *"`(a,c,b)` tồn tại ở đâu đó"* — **MỘT** mẩu bằng chứng, và P100 đo
được nó chỉ đúng **43,8%**. Lớp thay thế nói *"`c` và `x` thay nhau được qua **nhiều** bối
cảnh"* — một **mẫu hình nhất quán**. Khác nhau về **chất**, không phải về ngưỡng.

Chạy: `PYTHONPATH=src python scripts/probe_subst_class.py`
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

PROBE = ("vua", "sông", "đánh", "Nguyễn", "lớn", "quân", "năm", "thành", "nước", "người",
         "Việt Nam", "nhà Nguyễn", "triều Nguyễn", "Hà Nội", "quân Tống", "nhà Trần")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--gidf-min", type=float, default=4.0)
    ap.add_argument("--min-ctx", type=int, default=3, help="token phải có ≥ ngần này bối cảnh")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--min-share", type=int, default=3,
                    help="số BỐI CẢNH CHUNG tối thiểu — cốt lõi của ý tưởng")
    ap.add_argument("--unit", choices=("token", "cum", "thuc_the"), default="token",
                    help="token = âm tiết/BPE · cum = TỪ + ghép PMI · "
                         "thuc_the = TỪ + ghép chuỗi TÊN RIÊNG liền nhau (ranh giới THẬT)")
    ap.add_argument("--label-map", default="out/token_labels_mini.json")
    ap.add_argument("--ws-model", default="out/models/ws_crf_uts_full")
    ap.add_argument("--merge-pmi", type=float, default=3.0,
                    help="ghép hai đơn vị liền nhau thành CỤM khi PMI ≥ ngưỡng")
    ap.add_argument("--holdout", action="store_true",
                    help="chia đôi corpus: suy lớp trên A, XÁC NHẬN trên B (thước đếm được)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not args.out:
        args.out = f"out/probe_subst_class_{args.unit}.json"

    logger = get_run_logger("probe_subst_class")
    t0 = time.time()
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v, ps = state.vocab, state.parents
    n = len(v)
    _tn = lambda t: (v.token_of(t) if hasattr(v, "token_of") else str(t))
    name = lambda u: "".join(_tn(t) for t in u).replace("Ġ", " ").strip() \
        if isinstance(u, tuple) else _tn(u).replace("Ġ", "")

    keep = np.zeros(n, dtype=bool)
    for t in range(n):
        if is_structural_token(v, t) or not v.is_surface(t):
            continue
        try:
            if float(ps._gidf(t)) >= args.gidf_min:  # noqa: SLF001
                keep[t] = True
        except Exception:
            continue

    # --- dựng CHUỖI ĐƠN VỊ cho từng cha --------------------------------------------------
    # Tác giả chỉ ra: mọi phân tích tới giờ là TOKEN ĐƠN LẺ. `Nguyễn` tách khỏi `triều Nguyễn`,
    # `Việt` tách khỏi `Việt Nam`. Hạ tầng nơ-ron cụm (`entity_id`, `concept_id`) có trong code
    # nhưng checkpoint mini có **0** cái. Nhánh `--unit cum` dựng đơn vị CỤM: tách TỪ bằng
    # WS-CRF rồi ghép tiếp hai đơn vị liền nhau khi PMI đủ cao (`triều Nguyễn`, `Việt Nam`).
    seqs: list[list[tuple[int, ...]]] = []
    if args.unit == "token":
        for rec in ps.parents:
            seqs.append([(int(t),) for t in rec.emit_ids])
    else:  # "cum" hoặc "thuc_the" — cùng dùng WS-CRF tách TỪ
        from nn_aivie.hca.readout import decode_surface  # noqa: PLC0415
        from nn_aivie.ws_crf.ws_model import WSCRFModel  # noqa: PLC0415
        wsm = WSCRFModel.load(_ROOT / args.ws_model)
        for pi, rec in enumerate(ps.parents):
            if pi and pi % 1500 == 0:
                logger.log(f"  tách từ {pi:,}/{len(ps.parents):,} ({time.time()-t0:.0f}s)")
            ids = [int(t) for t in rec.emit_ids]
            whole = decode_surface(state, ids)
            syl = whole.split()
            if not syl:
                continue
            tags = wsm.tag(syl)
            if len(tags) != len(syl):
                continue
            # token → chỉ số âm tiết
            si_of: list[int | None] = []
            for k in range(len(ids)):
                pre = decode_surface(state, ids[: k + 1])
                si_of.append(len(pre.split()) - 1
                             if whole.startswith(pre) and not whole[len(pre):][:1].isalnum()
                             else None)
            # gộp token theo TỪ (âm tiết `I-W` nối vào từ trước)
            units: list[list[int]] = []
            w_of_syl: list[int] = []
            for j, tg in enumerate(tags):
                if tg == "I-W" and w_of_syl:
                    w_of_syl.append(w_of_syl[-1])
                else:
                    w_of_syl.append(len(units))
                    units.append([])
                w_of_syl[-1] = w_of_syl[j] if j < len(w_of_syl) else len(units) - 1
            w_of_syl = []
            cur = -1
            for tg in tags:
                if tg != "I-W" or cur < 0:
                    cur += 1
                w_of_syl.append(cur)
            units = [[] for _ in range(cur + 1)]
            for k, t in enumerate(ids):
                si = si_of[k]
                if si is not None and 0 <= si < len(w_of_syl):
                    units[w_of_syl[si]].append(t)
            seqs.append([tuple(u) for u in units if u])

    # --- ghép THỰC THỂ: chuỗi TÊN RIÊNG liền nhau thành MỘT đơn vị ------------------------
    # Khác PMI ở chỗ quyết định: ranh giới đến từ **lớp ký tự** (§7 — thống kê viết hoa GIỮA
    # câu, không danh sách tên nào), nên nó là ranh giới THẬT của một thực thể, không phải
    # ngưỡng thống kê. Trên v12, ghép PMI tạo 191.031 cặp — quá nhiều và quá thô, và lớp thu
    # được thì nhiễu (§6). Ghép thực thể chỉ đụng vào `Lý Thái Tổ`, `Việt Nam`, `Bạch Đằng`.
    if args.unit == "thuc_the":
        mp_ = _ROOT / args.label_map
        pn: set[int] = set()
        if mp_.exists():
            for ts, d in (json.loads(mp_.read_text(encoding="utf-8")).get("tu_loai") or {}).items():
                if d and max(d, key=d.get) == "tên riêng":
                    pn.add(int(ts))
        if not pn:
            raise SystemExit(f"⛔ {args.label_map} không có nhãn `tên riêng` ⇒ không ghép được")
        logger.log(f"token mang nhãn `tên riêng`: {len(pn):,}")
        is_pn = lambda u: bool(u) and all(t in pn for t in u)
        out2 = []
        n_merge = 0
        for sq in seqs:
            r, i = [], 0
            while i < len(sq):
                if is_pn(sq[i]):
                    j = i + 1
                    while j < len(sq) and is_pn(sq[j]):
                        j += 1
                    if j > i + 1:
                        n_merge += 1
                    r.append(tuple(t for u in sq[i:j] for t in u))
                    i = j
                else:
                    r.append(sq[i]); i += 1
            out2.append(r)
        seqs = out2
        logger.log(f"chuỗi tên riêng được ghép: {n_merge:,}")

    # --- ghép CỤM theo PMI: hai đơn vị liền nhau dính chặt thì thành MỘT ------------------
    if args.unit == "cum":
        from collections import Counter as _C
        uni: _C = _C()
        big: _C = _C()
        for sq in seqs:
            uni.update(sq)
            big.update(zip(sq, sq[1:]))
        tot = max(sum(big.values()), 1)
        tu = max(sum(uni.values()), 1)
        import math as _m
        merge = {ab for ab, c in big.items() if c >= 3
                 and _m.log2((c / tot) / ((uni[ab[0]] / tu) * (uni[ab[1]] / tu) + 1e-30))
                 >= args.merge_pmi}
        logger.log(f"cặp được ghép thành cụm (PMI≥{args.merge_pmi}): {len(merge):,}")
        out2 = []
        for sq in seqs:
            r, i = [], 0
            while i < len(sq):
                if i + 1 < len(sq) and (sq[i], sq[i + 1]) in merge:
                    r.append(sq[i] + sq[i + 1]); i += 2
                else:
                    r.append(sq[i]); i += 1
            out2.append(r)
        seqs = out2

    # --- bối cảnh: đơn vị → tập (trước, sau) ---------------------------------------------
    def ok(u: tuple[int, ...]) -> bool:
        return all(0 <= t < n and keep[t] for t in u)

    import random as _rd
    idxs = list(range(len(seqs)))
    if args.holdout:
        _rd.Random(args.seed).shuffle(idxs)
        A_i, B_i = idxs[: len(idxs) // 2], idxs[len(idxs) // 2:]
    else:
        A_i, B_i = idxs, []
    ctxB: dict[tuple[int, ...], set] = defaultdict(set)
    for k_ in B_i:
        sq = seqs[k_]
        for i in range(1, len(sq) - 1):
            if ok(sq[i]):
                ctxB[sq[i]].add((sq[i - 1], sq[i + 1]))

    ctx: dict[tuple[int, ...], set] = defaultdict(set)
    for k_ in A_i:
        sq = seqs[k_]
        for i in range(1, len(sq) - 1):
            if ok(sq[i]):
                ctx[sq[i]].add((sq[i - 1], sq[i + 1]))
    ctx = {t: c for t, c in ctx.items() if len(c) >= args.min_ctx}
    logger.log(f"đơn vị có ≥{args.min_ctx} bối cảnh: {len(ctx):,} ({time.time()-t0:.0f}s)")

    # chỉ mục ngược: bối cảnh → token, để không phải quét cặp
    inv: dict[tuple[int, int], list[int]] = defaultdict(list)
    for t, cs in ctx.items():
        for c in cs:
            inv[c].append(t)
    logger.log(f"bối cảnh phân biệt: {len(inv):,} ({time.time()-t0:.0f}s)")

    def neighbors(x: int, k: int) -> list[tuple[float, int]]:
        """Top-k token THAY THẾ ĐƯỢC cho `x`, cosine trên tập bối cảnh."""
        share: dict[int, int] = defaultdict(int)
        for c in ctx[x]:
            for y in inv[c]:
                if y != x:
                    share[y] += 1
        # ⚠️ ĐÒI ≥ min_share BỐI CẢNH CHUNG. Bản đầu nhận m=1 và bị token HIẾM thổi lên:
        # 1/sqrt(|ctx(y)|) rất lớn khi |ctx(y)| nhỏ ⇒ `vua → kali, xoáy, modem`. Và một bối
        # cảnh chung chính là bằng chứng yếu mà P100 đo được chỉ đúng 43,8%. Cả ý tưởng nằm
        # ở chữ "qua NHIỀU bối cảnh".
        nx = len(ctx[x]) ** 0.5
        sc = [(m / (nx * len(ctx[y]) ** 0.5), y)
              for y, m in share.items() if m >= args.min_share]
        sc.sort(reverse=True)
        return sc[:k]

    E = np.asarray(state.embedding.matrix, dtype=np.float32)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    ids = np.array(sorted({u[0] for u in ctx if len(u) == 1}), dtype=np.int64)

    def e_neighbors(x, k: int) -> list[str]:
        if len(x) != 1:
            return ["(cụm — E không có vector)"]
        sc_ = En[ids] @ En[x[0]]
        return [name((int(ids[j]),)) for j in np.argsort(-sc_)[: k + 1]
                if int(ids[j]) != x[0]][:k]

    out = {}
    print(f"\n{'':<10}{'LỚP THAY THẾ (trục mới)':<52}{'E (bạn cùng ngữ cảnh)'}")
    print("-" * 118)
    # ⚠️ LỖI ĐÃ SỬA: `{name(u): u for u in ctx}` bị VA CHẠM — `ĠNguyễn` (đầu từ) và `Nguyễn`
    # (mảnh giữa từ) đều rút gọn về "Nguyễn", dict giữ cái ĐẾN SAU. Trên mini tình cờ đúng;
    # trên v12 nó chọn biến thể hiếm (18 bối cảnh thay vì hàng trăm) ⇒ hàng "từ dò" của v12
    # KHÔNG đọc được. Sửa: khi trùng tên, giữ đơn vị có NHIỀU BỐI CẢNH NHẤT.
    by_name: dict = {}
    for u in ctx:
        k_ = name(u)
        if k_ not in by_name or len(ctx[u]) > len(ctx[by_name[k_]]):
            by_name[k_] = u
    for w in PROBE:
        t = by_name.get(w)
        if t is None:
            continue
        sub = [name(y) for _s, y in neighbors(t, args.topk)]
        emb = e_neighbors(t, args.topk)
        out[w] = {"lop_thay_the": sub, "embedding": emb, "n_ctx": len(ctx[t])}
        print(f"{w:<10}{', '.join(sub):<52}{', '.join(emb)}")

    # --- KHÔNG lấy bao đóng bắc cầu ------------------------------------------------------
    # Bản đầu dùng union-find trên cạnh top-k HAI CHIỀU ⇒ **thẩm thấu**: một cụm khổng lồ
    # 2.061/2.680 token (77%). Đó là hiện tượng percolation của đồ thị bậc ~6, không phải
    # tính chất của độ tương đồng. `A~B` và `B~C` KHÔNG kéo theo `A~C` ở quan hệ thay-thế.
    # ⇒ Nơ-ron lớp thay thế là **vùng láng giềng của từng token**, không phải phân hoạch.
    cov = [x for x in ctx if neighbors(x, 1)]
    logger.log(f"token có ≥1 láng giềng (≥{args.min_share} bối cảnh chung): "
               f"{len(cov):,}/{len(ctx):,} = {len(cov)/max(len(ctx),1):.1%}")
    print(f"\n\nphủ: {len(cov):,}/{len(ctx):,} token có ≥1 láng giềng thay-thế-được "
          f"(≥{args.min_share} bối cảnh chung)")
    ex = []
    import random as _r
    smp = sorted(cov, key=lambda t: -len(ctx[t]))[:400]
    _r.Random(7).shuffle(smp)
    print(f"\n12 vùng láng giềng bốc ngẫu nhiên:")
    for x in smp[:12]:
        ys = [name(y) for _s, y in neighbors(x, 6)]
        ex.append({"token": name(x), "lop": ys})
        print(f"  {name(x):<14} → {', '.join(ys)}")

    # --- THƯỚC ĐẾM ĐƯỢC: lớp suy trên A có ĐÚNG trên B không? ---------------------------
    # Không ngắm ví dụ nữa. Với `x` và bạn cùng lớp `y` (suy từ A), hỏi: ở nửa B có bối cảnh
    # nào của `x` mà `y` CŨNG điền được không? Đối chứng: `y` bốc ngẫu nhiên trong các đơn vị
    # có cỡ bối cảnh tương đương — thiếu cột này thì tỷ lệ nào cũng "trông cao" (§13.20).
    xac_nhan = {}
    if args.holdout:
        rr = _rd.Random(args.seed + 5)
        pool_by = sorted(ctx, key=lambda u: len(ctx[u]))
        hit = {"lop": [0, 0], "ngau_nhien": [0, 0]}
        for x in cov[:1500]:
            ys = [y for _s, y in neighbors(x, 3)]
            if not ys or x not in ctxB:
                continue
            bx = ctxB[x]
            for nm2, cand in (("lop", ys),
                              ("ngau_nhien", [pool_by[rr.randrange(len(pool_by))]
                                              for _ in ys])):
                for y in cand:
                    hit[nm2][1] += 1
                    if bx & ctxB.get(y, set()):
                        hit[nm2][0] += 1
        for nm2, (a_, b_) in hit.items():
            xac_nhan[nm2] = {"n": b_, "ty_le": a_ / max(b_, 1)}
        _rn = xac_nhan["ngau_nhien"]["ty_le"]
        r_ = (xac_nhan["lop"]["ty_le"] / _rn) if _rn > 0 else float("inf")
        print(f"\nXÁC NHẬN TRÊN NỬA B (lớp suy từ nửa A):")
        print(f"  lớp        n={xac_nhan['lop']['n']:<6} {xac_nhan['lop']['ty_le']:>6.1%}")
        print(f"  ngẫu nhiên n={xac_nhan['ngau_nhien']['n']:<6} "
              f"{xac_nhan['ngau_nhien']['ty_le']:>6.1%}")
        print(f"  tỷ số {'∞ (đối chứng = 0)' if r_ == float('inf') else f'{r_:.2f}×'}"
              f"   ·   mức TUYỆT ĐỐI {xac_nhan['lop']['ty_le']:.1%}  ← đọc cột này TRƯỚC")
        xac_nhan["ty_so"] = None if r_ == float("inf") else r_

    (_ROOT / args.out).write_text(json.dumps(
        {"checkpoint": args.checkpoint, "n_token": len(ctx), "n_boi_canh": len(inv),
         "min_share": args.min_share, "n_co_lang_gieng": len(cov),
         "vi_du_lang_gieng": out, "vi_du_vung": ex, "xac_nhan_B": xac_nhan,
         "seconds": round(time.time() - t0, 1)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    logger.log(f"→ {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
