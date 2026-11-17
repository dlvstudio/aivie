#!/usr/bin/env python3
"""ĐO theme prior của `facts_about` — `1/(1+vị_trí)` có phải thang tốt không? TK_045 §12.53.
**Chỉ đo, không đụng đường sinh.**

    PYTHONPATH=src python -u scripts/probe_theme_prior.py \
        --checkpoint checkpoints/mini_sent_ep_qa.pkl \
        --log-file out/probe_theme_prior_mini.log

## Vì sao

§12.52d đo được: `facts_about` nhân `cov *= 0.5 + theme_weight × w_theme` với
`w_theme = 1/(1+vị_trí_token_đầu_tiên)`. Chỉ 6,13% cặp đạt 1.0, trung vị 0.100 ⇒ **94% fact bị
nén vào dải phẳng 0.6–0.75**: prior thực chất NHỊ PHÂN (đứng đầu câu / không), không phải thang.
Và nó phạt oan trạng ngữ mở đầu — `Năm 1010, nhà Lý dời đô…` (0.125) bị coi ngang câu có thực thể
nằm trong mệnh đề nhúng — `Sử cũ không ghi chép … nhà Lý dời đô…` (0.111).

## Bộ nhãn — vì sao KHÔNG chấm tay, và vì sao thiết kế §12.52e phải bỏ

§12.52e định lấy dương = `w_theme = 1.0`, âm = có theme-primary khác đứng trước. **Thoái hoá**:
dương định nghĩa bằng `pos == 0`, mà "không có gì đứng trước" thì đúng hiển nhiên khi `pos == 0`
⇒ đặc trưng đoán được nhãn bằng chính định nghĩa. Đã tìm nhãn cấu trúc độc lập trong đồ thị và
**không có**: `match_span[0] == 0` ở CẢ 5004 cha câu (nó là cả câu, không phải chủ ngữ), còn
`phrase_ids` là rác (`ḅ` · `ى` · `城`). Chấm tay thì lại rơi vào bẫy nhóm-đo-tự-chọn (§12.52c:
29 prompt tôi chọn cho 2,4×, 660 câu hỏi thật cho 0,2× — dấu NGƯỢC).

Nên dùng **gold suy ra từ đồ thị**: cha QA nào có cha CÂU phủ ≥0.9 token của `answer_ids` thì cặp
(câu hỏi ấy → cha câu ấy) là một ca truy hồi có đáp án đúng. Được **169/660** (v12 **274/670**).
Ngưỡng phải là 0.9:
ở 0.7 gold đã hỏng (`Một ngày có bao nhiêu giờ?` → `Một tuần có bảy ngày`).

## Cổng — CHỐT TRƯỚC KHI CHẠY

  G1 bench lành   prior hiện tại phải đạt **top-10 ≥ 0.40**. (Bản đầu dùng top-1 ≥0.50 và ĐÓ LÀ
                  LỖI — ngưỡng tuyệt đối hiệu chỉnh trên bể 5.004 cha, áp lên bể 429.711 cha thì
                  nó đo CỠ BỂ chứ không đo sức khoẻ bench; §12.54 trượt vì lẽ đó, xem §12.56.)
  G2 prior có tác dụng gì không  so `theme_weight=0` với `=1`. Không phải cổng đậu/rớt — là SỰ
                  THẬT phải ghi: nếu bằng nhau thì prior đang bật mà vô dụng.
  G3 thay thế     prior mới phải hơn prior hiện tại **≥ +5 điểm %** top-1 mới đáng viết vào mã.

Chẩn đoán kèm (không phải cổng): với focus `Lý`, prior mới có kéo `Năm 1010, nhà Lý dời đô…` lên
TRÊN `Sử cũ không ghi chép…` không.

## Các prior đem so — đều Luật 0 sạch (chỉ dùng `gidf` đã học + vị trí)

  none     1.0                          (tắt prior)
  pos      1/(1+vị_trí_token)           HIỆN TẠI
  content  1/(1+số token gidf≥hi đứng trước)   trạng ngữ rẻ, thực thể cạnh tranh đắt
  clear    1.0 nếu không có token gidf≥hi đứng trước, 0.0 nếu có   (nhị phân, giả thuyết §12.52e)
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.embedding import truncated_svd  # noqa: E402
from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.hca.matrix_core import _path_fold, wh_slot_ids  # noqa: E402
from nn_aivie.hca.sent_intra import _content_positions, iter_sentence_spans  # noqa: E402

PRIORS = ("none", "pos", "content", "clear")

# §12.76 — THƯỚC CẶP CÓ THỨ TỰ. `A` của `ParentBipartite` là TÚI TỪ thuần
# (`A[t,pi] = √power × gidf`), còn `E`/`succ` chỉ giữ thứ tự ở d=1. Đo trên mini bằng chính luật
# `accumulate_span_pairs`: cặp phân biệt d=1 là 50.184 ‖ d≤12 là 386.307 ⇒ **87% số cặp có thứ
# tự đã học bị vứt**. Nên tiêu chí chọn hiện nay về mặt toán học KHÔNG phân biệt nổi
# "Nhà Lý dời đô về đâu" với "đâu dời đô về nhà Lý". Thước này hỏi: ứng viên có TÔN TRỌNG các
# quan hệ thứ tự của câu hỏi không — tỉ lệ cặp (a trước b) của câu hỏi được giữ đúng chiều.
PAIR_LAM = 0.85
PAIR_DMAX = 12


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--log-file", default="out/probe_theme_prior.log")
    ap.add_argument("--gold-cover", type=float, default=0.9)
    ap.add_argument("--gidf-hi", type=float, default=4.0)
    ap.add_argument("--theme-weight", type=float, default=1.0)
    ap.add_argument("--alt-cover", type=float, default=0.7,
                    help="nền#1 phủ >= ngưỡng này thì coi là đáp ĐÚNG KHÁC (nhiễu nhãn), không "
                         "phải lỗi xếp hạng (§12.68b)")
    ap.add_argument("--emb", action="store_true",
                    help="§12.88: dựng embedding từ W_skip (câu-giới-hạn, giảm λ^d, PPMI+SVD) "
                         "rồi so cos với embedding hiện tại trên cùng bench")
    ap.add_argument("--emb-dim", type=int, default=96)
    ap.add_argument("--emb-dmax", type=int, default=8)
    ap.add_argument("--emb-lam", type=float, default=0.85)
    ap.add_argument("--dump-c", type=int, default=20,
                    help="in N ca nhóm C để ĐỌC — thiết kế tín hiệu nội nhóm phải nhìn dữ liệu "
                         "trước, không suy từ trực giác (§12.64)")
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
    v, ps = state.vocab, state.parents
    wh = set(wh_slot_ids(state, ask_form_min=0.8))
    log(f"[{time.time()-t0:.0f}s] ✓ {len(ps):,} cha")

    def surf(ids) -> str:
        return "".join(v.token_of(int(x)) for x in ids).replace("Ġ", " ")

    # ---- cha câu: emit + fold + số token nội dung đứng trước mỗi vị trí ---------------------
    sents: list[list[int]] = []
    s_idx: list[int] = []
    for i, rec in enumerate(ps.parents):
        if getattr(rec, "is_qa", False):
            continue
        e = [int(x) for x in (getattr(rec, "emit_ids", ()) or ())]
        if len(e) >= 5:
            sents.append(e)
            s_idx.append(i)
    s_folds = [{_path_fold(v, t) for t in e} for e in sents]
    # §12.68 — CÙNG MỘT CÂU khác nhau mỗi khoảng trắng đầu vẫn là hai chuỗi id KHÁC NHAU
    # (`Sông` 7630 ‖ `ĠSông` 6559), nên lệch 1 token trên câu 8 token duy nhất là phủ 0.875 < 0.9
    # ⇒ bản kia RỚT ngưỡng gold và bị tính là "chọn nhầm" dù chữ y hệt. Vá §12.66 gom gold theo
    # id nên KHÔNG bắt được; phải gom theo VĂN BẢN đã chuẩn hoá.
    s_norm = [surf(e).replace("<EOS>", "").strip() for e in sents]
    norm_inv: dict[str, list[int]] = defaultdict(list)
    for k, t_ in enumerate(s_norm):
        if t_:
            norm_inv[t_].append(k)
    # `n_content_before[k][j]` = số token gidf≥hi ở vị trí < j của câu k
    n_before: list[list[int]] = []
    for e in sents:
        acc, c = [], 0
        for t in e:
            acc.append(c)
            if float(ps._gidf(t)) >= args.gidf_hi:  # noqa: SLF001
                c += 1
        n_before.append(acc)
    log(f"  {len(sents):,} cha câu ứng viên")

    # ---- gold: cha QA có cha câu phủ ≥ gold_cover token của answer_ids ----------------------
    inv: dict[int, list[int]] = defaultdict(list)
    u_len: list[int] = []
    for k, e in enumerate(sents):
        u = set(e)
        u_len.append(len(u))
        for t in u:
            inv[t].append(k)
    cases = []
    for rec in ps.parents:
        if not (getattr(rec, "is_qa", False) and getattr(rec, "answer_ids", ())):
            continue
        A = {int(x) for x in rec.answer_ids}
        cnt: dict[int, int] = defaultdict(int)
        for t in A:
            for k in inv.get(t, ()):
                cnt[k] += 1
        # §12.56 LỖI THƯỚC 1: bản cũ chỉ chấm 80 ứng viên chia sẻ nhiều token nhất ⇒ trên 430k
        # cha, gold thật thường rơi ngoài top-80 (v12 ra 74 gold ‖ mini 149 — nhiều cha hơn mà ÍT
        # gold hơn, vô lý). `cnt[k]` đã là số token DUY NHẤT của câu k nằm trong đáp, nên
        # `cnt[k]/|unique(k)|` CHÍNH LÀ độ phủ: tính đủ mọi ứng viên, không cắt, không xấp xỉ.
        # §12.66 LỖI NHÃN 3: gold là MỘT cha (argmax). Trên v12 CÙNG MỘT CÂU tồn tại thành
        # nhiều cha (mini: 5.004 cha / 5.004 văn bản duy nhất ⇒ 0 bản sao, nên phát triển trên
        # mini KHÔNG BAO GIỜ thấy). Các bản sao có điểm nền BẰNG NHAU TUYỆT ĐỐI, nên tie-break
        # theo chỉ số chọn bản khác ⇒ bị tính là "chọn nhầm" dù văn bản y hệt. Đó là phần lớn
        # nhóm C ở §12.65. Vá: gold là TẬP mọi cha phủ ≥ ngưỡng; đúng khi BẤT KỲ cái nào hạng 1.
        bc = 0.0
        golds: set[int] = set()
        for k, c_ in cnt.items():
            c = c_ / u_len[k]
            if c > bc:
                bc = c
            if c >= args.gold_cover:
                golds.add(k)
        if bc < args.gold_cover or not golds:
            continue
        for g in list(golds):                       # mở rộng sang mọi bản CÙNG VĂN BẢN
            golds.update(norm_inv.get(s_norm[g], ()))
        qc: dict[str, float] = {}
        foc: list[tuple[float, int]] = []
        for x in rec.match_ids:
            t = int(x)
            f = _path_fold(v, t)
            g = float(ps._gidf(t))  # noqa: SLF001
            if f and g >= args.gidf_hi and t not in wh:
                qc[f] = max(qc.get(f, 0.0), g)
                foc.append((g, t))
        if qc and foc:
            cases.append((qc, max(foc)[1], frozenset(golds), rec, frozenset(A)))
    log(f"  {len(cases)} ca gold (phủ ≥{args.gold_cover}) — nhãn suy từ đồ thị, không chấm tay")
    if len(cases) < 50:
        log("⚠️ quá ít ca gold ⇒ không kết luận.")
        return 1

    def theme_w(kind: str, k: int, variants: set[int]) -> float:
        e = sents[k]
        pos = next((j for j, t in enumerate(e) if t in variants), None)
        if pos is None:
            return 0.0
        if kind == "none":
            return 1.0
        if kind == "pos":
            return 1.0 / (1.0 + pos)
        nb = n_before[k][pos]
        if kind == "content":
            return 1.0 / (1.0 + nb)
        return 1.0 if nb == 0 else 0.0  # clear

    RATIOS = (3.0, 6.0, 20.0)   # 3.0 = dải [0.5,1.5] hiện dùng; 6/20 = prior mạnh hơn nhiều
    flippable = {r: 0 for r in RATIOS}
    reachable = {r: 0 for r in RATIOS}
    tie: dict[str, list[int]] = {p: [] for p in PRIORS}
    pool_sz: list[int] = []
    part: dict[str, int] = defaultdict(int)
    extra: dict[str, list[int]] = defaultdict(list)
    dom: dict[str, list[float]] = defaultdict(list)
    dump_c: list = []
    # vị trí fold nội dung đầu tiên trong mỗi câu — dựng LƯỜI, chỉ cho ứng viên thật sự cần
    _fpos: dict[int, dict[str, int]] = {}

    def fpos(k: int) -> dict[str, int]:
        hit = _fpos.get(k)
        if hit is None:
            hit = {}
            for i, t in enumerate(sents[k]):
                f = _path_fold(v, t)
                if f and f not in hit and float(ps._gidf(t)) >= args.gidf_hi:  # noqa: SLF001
                    hit[f] = i
            _fpos[k] = hit
        return hit

    def qpairs(rec) -> list[tuple[str, str, float]]:
        """Cặp (fold_trước, fold_sau, λ^d) của CÂU HỎI, chỉ fold nội dung, bỏ WH."""
        seq = []
        for x in rec.match_ids:
            t = int(x)
            f = _path_fold(v, t)
            if f and t not in wh and float(ps._gidf(t)) >= args.gidf_hi:  # noqa: SLF001
                if f not in [z for z, _ in seq]:
                    seq.append((f, len(seq)))
        out = []
        for i in range(len(seq)):
            for j in range(i + 1, min(len(seq), i + 1 + PAIR_DMAX)):
                out.append((seq[i][0], seq[j][0], PAIR_LAM ** (j - i)))
        return out

    # ---- §12.88: embedding từ W_skip -------------------------------------------------------
    E_old = E_new = None
    if args.emb:
        import math as _m

        import numpy as _np
        from scipy.sparse import coo_matrix as _coo

        ge = getattr(state, "embedding", None)
        E_old = getattr(ge, "matrix", None) if ge is not None else None  # §12.88: `matrix`, không phải `vectors`
        _struct = [is_structural_token(v, i) for i in range(len(v))]
        _c: dict[tuple[int, int], float] = defaultdict(float)
        _u: dict[int, float] = defaultdict(float)
        _ut = 0.0
        for rec in ps.parents:
            _sq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
            for _sp in iter_sentence_spans(_sq, v):
                _tk = [int(t) for t in _content_positions(
                    _sp, v, content_only=True, struct_mask=_struct)]
                for t in _tk:
                    _u[t] += 1.0
                    _ut += 1.0
                for i in range(len(_tk)):
                    for j in range(i + 1, min(len(_tk), i + 1 + args.emb_dmax)):
                        w_ = args.emb_lam ** (j - i - 1)
                        # ĐỐI XỨNG hoá: embedding lo TƯƠNG ĐỒNG, không lo thứ tự (§12.88a).
                        _c[(_tk[i], _tk[j])] += w_
                        _c[(_tk[j], _tk[i])] += w_
        _tot = sum(_c.values()) or 1.0
        _r, _cc, _vv = [], [], []
        for (a, b), w_ in _c.items():
            q = (_u[a] / _ut) * (_u[b] / _ut)
            if q <= 0:
                continue
            x = _m.log((w_ / _tot) / q)
            if x > 0:
                _r.append(a)
                _cc.append(b)
                _vv.append(x)
        _n = len(v)
        _ppmi = _coo((_vv, (_r, _cc)), shape=(_n, _n)).tocsr()
        log(f"[{time.time()-t0:.0f}s] PPMI từ W_skip: {_ppmi.nnz:,} ô ≠0 → SVD {args.emb_dim}d")
        E_new = truncated_svd(_ppmi, args.emb_dim)
        log(f"[{time.time()-t0:.0f}s] ✓ E_new {E_new.shape} · E_old "
            f"{None if E_old is None else E_old.shape}")

        def _emb(M, ids):
            if M is None:
                return None
            ii = [int(t) for t in ids if 0 <= int(t) < M.shape[0]]
            if not ii:
                return None
            x = M[ii].mean(0)
            nr = float(_np.linalg.norm(x))
            return x / nr if nr > 1e-9 else None

    tw = float(args.theme_weight)
    rank1: dict[str, list[int]] = {p: [] for p in PRIORS}
    hit = {p: 0 for p in PRIORS}
    hit0 = 0                     # theme_weight = 0 (G2)
    mrr = {p: 0.0 for p in PRIORS}
    top10 = {p: 0 for p in PRIORS}
    # Thu bể ứng viên bằng chỉ mục ngược THEO FOLD. CHÍNH XÁC chứ không xấp xỉ: cha không chia
    # sẻ fold nội dung nào có `base = 0` nên không bao giờ là rank-1 (gold luôn nằm trong bể).
    # Bắt buộc để chạy nổi v12: vòng phẳng là O(430k) MỖI ca MỖI prior.
    f_inv: dict[str, list[int]] = defaultdict(list)
    for k, fs in enumerate(s_folds):
        for f in fs:
            f_inv[f].append(k)

    for qc, focus, golds, _rec, A_ in cases:
        _qp = qpairs(_rec)
        _qw = sum(w for _a, _b, w in _qp)
        var = set(ps.case_equiv_ids(focus, v)) or {focus}
        pool = set()
        for f in qc:
            pool.update(f_inv.get(f, ()))
        pool.update(golds)
        pool_l = sorted(pool)
        base = {k: sum(g for f, g in qc.items() if f in s_folds[k]) / math.sqrt(len(s_folds[k]))
                for k in pool_l}

        def pair_score(k: int) -> float:
            if not _qp or _qw <= 0:
                return 0.0
            fs = s_folds[k]
            # rẻ: chỉ dựng vị trí khi ứng viên có ≥2 fold của câu hỏi (điều kiện cần để có cặp)
            if sum(1 for f in qc if f in fs) < 2:
                return 0.0
            pm = fpos(k)
            tot = 0.0
            for a, b, w in _qp:
                ia, ib = pm.get(a), pm.get(b)
                if ia is not None and ib is not None and ia < ib:
                    tot += w
            return tot / _qw

        pair = {k: pair_score(k) for k in pool_l}
        def _rank(order):
            for i, k in enumerate(order):
                if k in golds:
                    return i + 1
            return len(order) + 1

        r0 = _rank(sorted(pool_l, key=lambda k: (-base[k], k)))
        hit0 += int(r0 == 1)
        # TRẦN CỦA MỌI PRIOR NHÂN (§12.58). Hệ số nằm trong [lo,hi] ⇒ ứng viên j chỉ lật nổi
        # quán quân i khi base_j·hi > base_i·lo, tức base_j > base_i/(hi/lo). Đếm xem có ứng
        # viên nào trong dải đó không: KHÔNG có ⇒ top-1 BẤT KHẢ XÂM PHẠM với mọi prior nhân,
        # bất kể prior thông minh cỡ nào. Đây là trần, không phải kết quả của một prior cụ thể.
        b1 = max(base.values()) if base else 0.0
        for rat in RATIOS:
            if b1 > 0 and sum(1 for x in base.values() if x > b1 / rat) >= 2:
                flippable[rat] += 1
            # Trần THẬT SỰ đáng quan tâm: prior có kéo nổi GOLD lên hạng 1 không. "Đổi được
            # top-1" mà gold nằm ngoài tầm thì đổi cũng vô ích.
            if b1 > 0 and max((base.get(g, 0.0) for g in golds), default=0.0) > b1 / rat:
                reachable[rat] += 1
        # BÃO HOÀ (§12.60): prior chỉ phân biệt được khi ít ứng viên cùng chạm giá trị CAO NHẤT.
        # Bể càng lớn càng nhiều câu mở đầu bằng thực thể ⇒ cùng w_theme=1.0 ⇒ prior nhân đều,
        # hết tác dụng phân biệt. Đây là ứng viên giải thích cho việc `pos` khai thác 68.8% trần
        # trên mini nhưng chỉ 8.7% trên v12.
        pool_sz.append(len(pool_l))
        for p in PRIORS:
            if p != "none":
                wv = [theme_w(p, k, var) for k in pool_l]
                mx = max(wv) if wv else 0.0
                tie[p].append(sum(1 for x in wv if x >= mx - 1e-9))
                if p == "pos":
                    # §12.62 — PHÂN HOẠCH ca theo chỗ gold nằm so với NHÓM HOÀ ở đỉnh prior.
                    # Nó quyết định hướng sửa: tín hiệu phân biệt NỘI NHÓM chỉ cứu được nhóm C.
                    grp = [k for k, x in zip(pool_l, wv) if x >= mx - 1e-9]
                    if golds & set(grp):
                        best_in = max(grp, key=lambda k: base[k])
                        gold = max(golds & set(grp), key=lambda k: base[k])
                        if best_in in golds:
                            part["B_hoà_và_nền_đã_nhất"] += 1
                        else:
                            # NGUỒN NHIỄU CÒN LẠI (§12.68b): câu thắng có thể là một đáp ĐÚNG
                            # KHÁC, chỉ không trùng câu được chọn làm gold — `Thủ đô Việt Nam là
                            # Hà Nội` ‖ `Thủ đô Hà Nội ở Việt Nam`. Không vá tự động được, nhưng
                            # ĐO được: nó phủ bao nhiêu phần câu đáp. Tách để khỏi phải hỏi lại.
                            cov_w = (sum(1 for t in set(sents[best_in]) if t in A_)
                                     / max(1, u_len[best_in]))
                            part["C1_nền#1_có_thể_CŨNG_ĐÚNG" if cov_w >= args.alt_cover
                                 else "C2_lỗi_xếp_hạng_THẬT"] += 1
                            part["C_hoà_nhưng_nền_thua"] += 1
                        if (best_in not in golds and len(dump_c) < args.dump_c
                                and cov_w < args.alt_cover):
                            top3 = sorted(grp, key=lambda k: -base[k])[:3]
                            dump_c.append((
                                "".join(v.token_of(int(x)) for x in _rec.match_ids)
                                .replace("Ġ", " ")[:96],
                                surf(sents[gold])[:112], base[gold],
                                [(surf(sents[k])[:112], base[k]) for k in top3],
                            ))
                    else:
                        part["D_gold_NGOÀI_nhóm_hoà"] += 1
        def _z(d):
            xs = list(d.values())
            m = sum(xs) / len(xs)
            sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
            return {k: 0.0 for k in d} if sd < 1e-12 else {k: (d[k] - m) / sd for k in d}

        zb, zp = _z(base), _z(pair)
        # CỘNG HƯỞNG hay TRÙNG? B+PAIR trên mini ra đúng 0.615 = mức theme prior đạt được.
        # Nếu chồng hai thứ lên nhau mà không hơn thì chúng đang đo CÙNG một thứ.
        _bp = {k: base[k] * (0.5 + tw * theme_w("pos", k, var)) for k in pool_l}
        zbp = _z(_bp)
        for nm, sc_ in (("PAIR", pair),
                        ("pos+PAIR", {k: zbp[k] + zp[k] for k in pool_l}),
                        ("B+PAIR", {k: zb[k] + zp[k] for k in pool_l}),
                        ("B+.5PAIR", {k: zb[k] + 0.5 * zp[k] for k in pool_l})):
            r_ = _rank(sorted(pool_l, key=lambda k: (-sc_[k], k)))
            extra[nm].append(int(r_ == 1))
        extra["B"].append(int(_rank(sorted(pool_l, key=lambda k: (-base[k], k))) == 1))
        if args.emb:
            import numpy as _np2

            qv = {"cosE_old": _emb(E_old, _rec.match_ids),
                  "cosE_skip": _emb(E_new, _rec.match_ids)}
            _cs: dict[str, dict[int, float]] = {}
            for nm_, M_ in (("cosE_old", E_old), ("cosE_skip", E_new)):
                q_ = qv[nm_]
                if q_ is None:
                    continue
                sc_ = {}
                for kk in pool_l:
                    cv = _emb(M_, sents[kk])
                    sc_[kk] = float(_np2.dot(q_, cv)) if cv is not None else -1e9
                _cs[nm_] = sc_
                extra[nm_].append(
                    int(_rank(sorted(pool_l, key=lambda kk: (-sc_[kk], kk))) == 1))
            # §12.92 — HÌNH DẠNG CÔNG THỨC. `multizone` xếp bằng `cos × (0.25 + 0.75·cov)`:
            # `cos` là THỪA SỐ CHÍNH, `cov` chỉ điều biến biên độ 4×. Trên v12 `cos` một mình
            # xếp ở 0.004 (gần vô thông tin) ⇒ nhân nó vào `cov` có thể chỉ THÊM NHIỄU.
            # Đo thẳng: B đóng vai `cov`, so ba hình dạng.
            # §12.93 — CHO `cos` ĐÚNG VIỆC CỦA NÓ. `pool` đã lọc theo trùng-fold, tức đã
            # topically coherent TRƯỚC KHI cos lên tiếng ⇒ `cosE_old = 0.004` chỉ nói "cos không
            # thêm gì SAU lọc từ vựng", KHÔNG nói cos vô dụng. Tác giả nêu đúng: cos(Lý,Trần)=0.93
            # có thể là tín hiệu MIỀN hợp lệ. Đo: cos tới ứng viên TRONG bể (cùng miền) so với
            # ứng viên NGOÀI bể (nhiều khả năng khác miền).
            co = _cs.get("cosE_old")
            if co is not None and len(sents) > len(pool_l) + 50:
                import random as _rd
                _r = _rd.Random(1234 + len(pool_l))
                out_ = []
                for _ in range(40):
                    kk = _r.randrange(len(sents))
                    if kk not in pool:
                        out_.append(kk)
                if out_:
                    q0 = qv.get("cosE_old")
                    if q0 is not None:
                        vin = [co[kk] for kk in pool_l]
                        vout = []
                        for kk in out_:
                            cv = _emb(E_old, sents[kk])
                            if cv is not None:
                                vout.append(float(_np2.dot(q0, cv)))
                        if vin and vout:
                            dom["in"].append(sum(vin) / len(vin))
                            dom["out"].append(sum(vout) / len(vout))
                            dom["gold"].append(max(co[g] for g in golds if g in co))
            if co is not None:
                mx = max(base.values()) or 1.0
                nb_ = {kk: base[kk] / mx for kk in pool_l}          # cov chuẩn hoá [0,1]
                nc_ = {kk: (co[kk] + 1.0) / 2.0 for kk in pool_l}   # cos → [0,1]
                # §12.96 — PHÉP ĐO QUYẾT ĐỊNH cho câu "có dùng embedding W_skip không":
                # đúng công thức production, chỉ thay embedding. Chưa ai chạy. §12.95 vừa dạy
                # "tốt hơn khi đứng MỘT MÌNH không suy ra tốt hơn khi KẾT HỢP" (cos_old 0.004
                # một mình mà +0.018 khi nhân vào cov), nên KHÔNG được suy — phải đo.
                ck = _cs.get("cosE_skip")
                shapes = [
                    ("cosxcov(mz)", lambda kk: nc_[kk] * (0.25 + 0.75 * nb_[kk])),
                    ("covxcos(đảo)", lambda kk: nb_[kk] * (0.25 + 0.75 * nc_[kk])),
                ]
                if ck is not None:
                    nk_ = {kk: (ck[kk] + 1.0) / 2.0 for kk in pool_l}
                    shapes.append(
                        ("skipxcov(mz)", lambda kk: nk_[kk] * (0.25 + 0.75 * nb_[kk])))
                for nm_, f_ in shapes:
                    extra[nm_].append(
                        int(_rank(sorted(pool_l, key=lambda kk: (-f_(kk), kk))) == 1))
        extra["pos"].append(int(_rank(sorted(pool_l, key=lambda k: (-_bp[k], k))) == 1))

        for p in PRIORS:
            sc = {k: (base[k] if p == "none" else base[k] * (0.5 + tw * theme_w(p, k, var)))
                  for k in pool_l}
            r = _rank(sorted(pool_l, key=lambda k: (-sc[k], k)))
            hit[p] += int(r == 1)
            mrr[p] += 1.0 / r
            top10[p] += int(r <= 10)
            rank1[p].append(int(r == 1))

    n = len(cases)
    log("\n" + "=" * 66)
    log(f"{'prior':<10}{'top1':>9}{'top10':>9}{'MRR':>9}{'Δ so pos':>12}   (n={n})")
    log("=" * 66)
    log(f"{'(tw=0)':<10}{hit0/n:>9.3f}{'—':>9}{'—':>9}{'—':>12}")
    for p in PRIORS:
        log(f"{p:<10}{hit[p]/n:>9.3f}{top10[p]/n:>9.3f}{mrr[p]/n:>9.3f}"
            f"{hit[p]/n-hit['pos']/n:>+12.3f}")
    log("=" * 66)

    # CỘT BẮT BUỘC: hai prior bất đồng bao nhiêu ca? Δ nhỏ có thể là "hoà" chứ không phải "thua"
    # — bench không chứa hiện tượng thì nó không PHÁN được, và đó là kết luận khác hẳn.
    log("\nBẤT ĐỒNG so với `pos` (ca mà top-1 đổi kết quả):")
    for p in PRIORS:
        if p == "pos":
            continue
        w = sum(1 for a, b in zip(rank1[p], rank1["pos"]) if a and not b)
        l = sum(1 for a, b in zip(rank1[p], rank1["pos"]) if b and not a)
        log(f"  {p:<8} thắng {w:>3} · thua {l:>3} · bất đồng {w+l:>3}/{n} = {(w+l)/n:.1%}")

    # §12.56 LỖI THƯỚC 2: G1 cũ là top-1 ≥0.50 — con số ấy hiệu chỉnh trên bể 5.004 cha rồi đem
    # áp lên bể 429.711 cha, nên nó đo CỠ BỂ chứ không đo sức khoẻ bench. Thay bằng top-10: câu
    # hỏi hợp lệ thì gold phải nằm trong 10 ứng viên đầu, bất kể bể to nhỏ.
    def _med(xs):
        return sorted(xs)[len(xs) // 2] if xs else 0

    if extra["B"]:
        nb = len(extra["B"])
        log("\nTHƯỚC CẶP CÓ THỨ TỰ (§12.76) — `A` là túi từ nên không phân biệt nổi trật tự:")
        log(f"  {'thước':<12}{'top1':>9}{'Δ so B':>10}{'thắng':>8}{'thua':>7}   (n={nb})")
        b0 = sum(extra["B"]) / nb
        log(f"  {'B (túi từ)':<12}{b0:>9.3f}{'—':>10}{'—':>8}{'—':>7}")
        for nm in ("PAIR", "B+PAIR", "B+.5PAIR", "pos", "pos+PAIR", "cosE_old", "cosE_skip",
                   "cosxcov(mz)", "covxcos(đảo)", "skipxcov(mz)"):
            if not extra[nm]:
                continue
            a = sum(extra[nm]) / nb
            w = sum(1 for x, y in zip(extra[nm], extra["B"]) if x and not y)
            l = sum(1 for x, y in zip(extra[nm], extra["B"]) if y and not x)
            log(f"  {nm:<12}{a:>9.3f}{a-b0:>+10.3f}{w:>8}{l:>7}")
        # CỔNG chấm so TIÊU CHÍ HIỆN HÀNH (`pos` = base × theme prior), KHÔNG so túi từ.
        p0 = sum(extra["pos"]) / nb
        pp = sum(extra["pos+PAIR"]) / nb
        w = sum(1 for x, y in zip(extra["pos+PAIR"], extra["pos"]) if x and not y)
        l = sum(1 for x, y in zip(extra["pos+PAIR"], extra["pos"]) if y and not x)
        ok = (pp - p0) >= 0.05 and (w - l) >= 5
        log(f"\n  CỔNG (§12.56c) so TIÊU CHÍ HIỆN HÀNH `pos` {p0:.3f} → `pos+PAIR` {pp:.3f}"
            f"  (Δ {pp-p0:+.3f} · thắng {w} · thua {l})")
        log(f"  ⇒ {'ĐẠT — đáng viết vào mã' if ok else 'TRƯỢT'}"
            "  (cần Δ ≥ +0.05 VÀ thắng−thua ≥ 5)")

    if dom["in"]:
        import statistics as _st
        mi, mo = _st.mean(dom["in"]), _st.mean(dom["out"])
        mg = _st.mean(dom["gold"])
        log("\n`cos` CÓ ĐO MIỀN KHÔNG (§12.93) — cho nó việc mà bench đã lọc mất:")
        log(f"  cos tới ứng viên TRONG bể (đã trùng fold) : {mi:+.3f}")
        log(f"  cos tới ứng viên NGOÀI bể (khác miền)      : {mo:+.3f}")
        log(f"  cos tới GOLD                                : {mg:+.3f}")
        log(f"  ⇒ tách miền = {mi - mo:+.3f}   ·   tách gold-trong-bể = {mg - mi:+.3f}")
        # KHÔNG viết sẵn kết luận — hai cột này phải được SO với nhau rồi mới đọc.
        if mg - mi > (mi - mo) * 1.3:
            log("  ⇒ tách GOLD > tách MIỀN ⇒ `cos` vẫn mang tín hiệu DANH TÍNH (dù yếu).")
        elif (mi - mo) > (mg - mi) * 1.3:
            log("  ⇒ tách MIỀN > tách GOLD ⇒ `cos` chủ yếu lo MIỀN, không lo danh tính.")
        else:
            log("  ⇒ hai tách XẤP XỈ nhau ⇒ `cos` mang cả hai, không nghiêng hẳn bên nào.")
        log("  Lưu ý: tách gold đo so TRUNG BÌNH bể. Cao hơn trung bình vẫn có thể xa hạng 1 khi")
        log("  bể tới hàng chục nghìn — đó là vì sao `cosE_old` xếp 0.004 mà tách gold vẫn dương.")

    log("\nBÃO HOÀ — bao nhiêu ứng viên CÙNG chạm giá trị prior cao nhất (trung vị):")
    log(f"  bể ứng viên (trung vị) = {_med(pool_sz):,}")
    for p in PRIORS:
        if p == "none":
            continue
        m = _med(tie[p])
        log(f"  {p:<8} {m:>7,} ứng viên cùng đỉnh = {m/max(1,_med(pool_sz)):>6.2%} bể"
            f"   ⇒ prior {'KHÔNG phân biệt nổi' if m > 20 else 'còn phân biệt được'}")

    log("\nGOLD nằm đâu so với NHÓM HOÀ ở đỉnh prior `pos` — quyết định HƯỚNG SỬA:")
    for k_ in ("B_hoà_và_nền_đã_nhất", "C_hoà_nhưng_nền_thua",
               "  C1_nền#1_có_thể_CŨNG_ĐÚNG", "  C2_lỗi_xếp_hạng_THẬT",
               "D_gold_NGOÀI_nhóm_hoà"):
        k_ = k_.strip()
        log(f"  {k_:<26}{part[k_]:>6} = {part[k_]/n:>6.1%}")
    log("  B = đã đúng sẵn · C = gold trong nhóm nhưng nền thua · D = cần cách khác.")
    log("  **C2 mới là phần tín hiệu nội nhóm thật sự cứu được** — C1 là nhiễu nhãn (đáp đúng khác).")

    log("\nTRẦN của MỌI prior nhân — ca mà top-1 CÓ THỂ đổi, theo dải hệ số:")
    for r in RATIOS:
        lo = 1.0 / (1.0 + (r - 1.0) / 2.0) * 1.0
        log(f"  dải {r:>4.0f}× (vd [0.5,{0.5*r:.1f}]): đổi được top-1 {flippable[r]/n:>6.1%}"
            f"  ·  **kéo được GOLD lên hạng 1 {reachable[r]/n:>6.1%}** ({reachable[r]}/{n})")
    log("  Cột thứ hai mới là trần thật: đổi được top-1 mà gold ngoài tầm thì đổi cũng vô ích.")
    log("  Trần thấp ⇒ điểm NỀN đã quyết xong; mọi prior NHÂN đều không với tới, dù thông minh cỡ nào.")

    g1 = top10["pos"] / n >= 0.40
    best = max((hit[p] / n, p) for p in PRIORS if p != "pos")
    g3 = best[0] - hit["pos"] / n >= 0.05
    log(f"G1 bench lành (pos top-10 ≥0.40): {'ĐẠT' if g1 else 'TRƯỢT'} "
        f"(top10 {top10['pos']/n:.3f} · top1 {hit['pos']/n:.3f})")
    log(f"G2 prior có tác dụng?            : tw=0 → {hit0/n:.3f} ‖ pos → {hit['pos']/n:.3f} "
        f"(Δ {hit['pos']/n-hit0/n:+.3f})")
    log(f"G3 thay thế hơn ≥+0.05          : {'ĐẠT' if g3 else 'TRƯỢT'} "
        f"(tốt nhất `{best[1]}` {best[0]:.3f}, Δ {best[0]-hit['pos']/n:+.3f})")
    log(f"\n⇒ {'ĐI TIẾP' if (g1 and g3) else 'CHỐT ÂM TÍNH — không sửa prior theo hướng này'}")

    # ---- chẩn đoán ca 0 --------------------------------------------------------------------
    log("\nCHẨN ĐOÁN ca 0 (focus `Lý`) — prior nào xếp câu ĐÚNG trên câu XẤU:")
    fid = v._token_to_id.get("ĠLý") or v._token_to_id.get("Lý")  # noqa: SLF001
    if fid is not None:
        var = set(ps.case_equiv_ids(fid, v)) or {fid}
        want = {}
        for k, e in enumerate(sents):
            s_ = surf(e)
            if s_.startswith("Năm 1010, nhà Lý dời đô"):
                want["ĐÚNG"] = k
            elif "Sử cũ không ghi chép" in s_:
                want["XẤU"] = k
        for p in PRIORS:
            vals = {lbl: 0.5 + tw * theme_w(p, k, var) for lbl, k in want.items()}
            ok = vals.get("ĐÚNG", 0) > vals.get("XẤU", 0)
            log(f"  {p:<8} ĐÚNG={vals.get('ĐÚNG', 0):.3f}  XẤU={vals.get('XẤU', 0):.3f}  "
                f"{'✓ tách' if ok else '✗ không tách'}")
    if dump_c:
        log("\n" + "=" * 84)
        log(f"ĐỌC {len(dump_c)} CA NHÓM **C2** — đã lọc bỏ C1 (nền#1 cũng là đáp đúng).")
        log("Đây là đám mà tín hiệu phân biệt NỘI NHÓM phải sửa. `nền` = trùng-từ có trọng gidf.")
        log("=" * 84)
        for i, (q, g, gb, top3) in enumerate(dump_c):
            log(f"\n[{i}] HỎI: {q}")
            log(f"    GOLD (nền {gb:.3f}): {g}")
            for j, (t, b_) in enumerate(top3):
                log(f"    nền#{j+1} ({b_:.3f}): {t}")
    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
