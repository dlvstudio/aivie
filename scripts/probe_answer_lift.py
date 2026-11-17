#!/usr/bin/env python3
"""ĐO LIFT `P(fold ∈ đáp | WH-fold ∈ hỏi)` — tín hiệu "hợp KIỂU câu hỏi" có TÁCH được gì không?
TK_045 §12.51. **Chỉ đo, không đụng đường sinh.**

    PYTHONPATH=src python -u scripts/probe_answer_lift.py \
        --checkpoint checkpoints/mini_sent_ep_qa.pkl \
        --log-file out/probe_answer_lift_mini.log

## Vì sao script này tồn tại

§12.50 chốt: ba cơ chế độc lập (chép span đáp · WH-slot · nới vùng PMI) đều cải thiện **cái gì
được đưa vào xét** mà không cái nào chạm **tiêu chí chọn**. Tiêu chí hiện tại — `cos(zone, fact)`
và `act` — đo "nói về cùng thực thể", không đo "trả lời đúng câu hỏi". Bằng chứng sạch nhất:
hỏi *về đâu* → đáp *"Sử cũ không ghi chép chi tiết nhà Lý dời đô bằng đường nào"*: giống hệt về
thực thể, **sai** về kiểu hỏi.

Đề xuất §12.50d: từ cha QA học phân bố token phần ĐÁP theo **loại câu hỏi**, rồi dùng **lift**
so nền để xếp hạng lại. Nhưng 670 cha QA là MỎNG — đúng lý do `ai` chết ở §12.48d. Nên đo trước,
viết đường sinh sau. Hai lần liền (§12.46 PMI · §12.38 cổng n=4) cái hỏng là THƯỚC, không phải mã.

## Đo trên mini có hợp lệ không? — CÓ, và theo chiều ÂM

mini 660 cha QA ‖ v12 670. Cùng một nền học. Bể ứng viên ở đây (vài trăm đáp QA) **nhỏ hơn
nhiều** bể thật của đường sinh (430k cha câu). Nên:

    tách được trên mini  ⇒ CHƯA kết luận, phải M4 xác nhận trên v12
    KHÔNG tách trên mini ⇒ CHỐT ÂM TÍNH (dễ hơn mà đã thua thì khó hơn không thể thắng)

Ngược hẳn ca PMI (§12.46/§12.49) — ở đó mini DƯƠNG rồi v12 âm, vì mini ít cha nên nới vùng là đủ
đổi thứ hạng. Ở đây mini được dùng đúng chiều nó chứng minh được.

## Ba cổng — CHỐT TRƯỚC KHI CHẠY

  G1 phủ  ≥40% cha QA có WH-fold dùng được. Thấp hơn ⇒ dù tín hiệu hoàn hảo cũng chạm rất ít ca.
  G2 tách  top1(B+L) − top1(B) ≥ **+10 điểm %** toàn cục VÀ ≥0 ở ≥4/5 fold (không cho MỘT fold
           may mắn kéo cả cột — đúng lỗi §12.38).
  G3 chuyển  hạn lift về **chỉ các fold CÓ trong corpus câu** vẫn giữ ≥ nửa mức lợi (≥+5 điểm %).
           Cổng này bắt buộc vì đo được: đáp QA gần như KHÔNG nằm trong corpus câu (9/40 ca có
           cha câu phủ ≥0.9 đáp). Lift dồn hết vào fold NỘI DUNG riêng của corpus QA thì đúng
           trên bảng này mà vô dụng khi xếp hạng cha câu — tức dương giả.

Ba cổng phải ĐẠT CẢ BA mới viết đường sinh. Trượt bất kỳ cổng nào ⇒ ghi âm tính vào ticket.

## Thiết kế phép đo — LIỆT KÊ LOẠI ĐẦU VÀO TRƯỚC

Loại câu hỏi (không bỏ sót loại nào, kể cả loại dễ quên):

  W-đâu/nào/gì/ai/mấy/sao …  mỗi WH-fold học được là một nhóm, báo cáo RIÊNG
  W-none                     câu hỏi KHÔNG có WH-fold nào — loại dễ quên nhất, và nếu nó chiếm
                             đa số thì trần của cả cơ chế thấp bất kể lift mạnh cỡ nào (G1)

**Chọn WH-fold phải lọc theo `gidf`.** Lần chạy đầu (§12.51a) `ask_form` cao nhất lại là `ġ`
(khoảng trắng, 1.00) và `?` (0.99) — có ở MỌI câu hỏi, không bao giờ ở đáp, nên chúng "hoàn hảo"
theo `ask_form` mà rỗng thông tin. Hậu quả: 606/659 ca gom vào một nhóm `ġ` ≈ nền, và L "thắng"
0.938 chỉ vì tách được chủ đề toán (`mấy`) khỏi phần còn lại. Cột ĐỐI CHỨNG lộ ra ngay:
`rnd/L = 0.112` ≈ chance 0.125 ⇒ L không có tín hiệu độc lập nào. Lọc `gidf ≥ gidf_hi` loại token
cấu trúc bằng chính thống kê tần suất đã học — không nhìn mặt chữ, Luật 0 sạch.
(Ghi chú: `wh_slot_ids` production cũng cấm `''`/`?` vì lý do này. Ở đó vô hại — cấm token cấu
trúc khỏi ô neo không mất gì. Ở đây thì chí mạng, vì ta cần PHÂN LOẠI câu hỏi.)

Ứng viên nhiễu định nghĩa theo đúng thứ truy hồi thật đưa lên: **`topB` = đáp mà chính tiêu chí
B chấm CAO NHẤT**. Lần chạy đầu tôi lấy "trùng thực thể câu hỏi ↔ câu hỏi" nên 113/132 ca không
đủ nhiễu, phải pad ngẫu nhiên ⇒ cột "khó" gần bằng cột ngẫu nhiên (B 0.885 ‖ 0.907) và phép đo
mất hết ý nghĩa. Định nghĩa mới có thiên vị CHỐNG B — cố ý: câu hỏi cần trả lời chính là
"L có cứu được những ca B bị lừa không". Ba bộ nhiễu:

    topB   B chấm cao nhất, MỌI loại câu hỏi     ← THỰC TẾ, cổng G2 chấm ở đây
    topBx  B chấm cao nhất, KHÁC WH-fold          ← nơi L ĐÁNG LẼ phải toả sáng (ca 0 ngoài đời)
    rnd    ngẫu nhiên                             ← ĐỐI CHỨNG: L một mình phải hơn chance

Thước:
  B  = trùng thực thể, có trọng gidf — XẤP XỈ tiêu chí hiện tại ("nói về cùng thực thể")
  L  = lift theo WH-fold — tiêu chí ĐỀ XUẤT, chạy MỘT MÌNH để biết nó có tín hiệu độc lập không
  B+L= chuẩn hoá z rồi cộng — thứ sẽ thật sự dùng (không thay thế cos-thực-thể, mà cộng vào)

Chống rò: `ask/ans` và tập WH-fold đều đếm **chỉ trên fold TRAIN**. Đếm trên cả 660 rồi chấm
trên chính 660 sẽ ra bảng rất đẹp và vô giá trị.
"""

from __future__ import annotations

import argparse
import gc
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.matrix_core import _path_fold  # noqa: E402


class QA:
    __slots__ = ("qf", "af", "qcontent", "wh")

    def __init__(self, qf, af, qcontent):
        self.qf = qf            # set[str] fold câu hỏi
        self.af = af            # set[str] fold câu đáp
        self.qcontent = qcontent  # dict[str,float] fold nội dung câu hỏi → gidf
        self.wh = ""            # gán trong mỗi fold CV (phụ thuộc thống kê train)


def _overlap(a: dict, b: set) -> float:
    return sum(g for f, g in a.items() if f in b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--log-file", default="out/probe_answer_lift.log")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-cand", type=int, default=8, help="1 gold + (n-1) nhiễu")
    ap.add_argument("--min-ask", type=int, default=5)
    ap.add_argument("--ask-form-min", type=float, default=0.8)
    ap.add_argument("--min-group", type=int, default=10, help="cỡ nhóm WH tối thiểu trong train")
    ap.add_argument("--gidf-hi", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
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
    log(f"[{time.time()-t0:.0f}s] ✓ {len(ps):,} cha")

    # ---- gom cha QA + fold của corpus CÂU (cho G3) -----------------------------------------
    items: list[QA] = []
    sent_folds: set[str] = set()
    gidf_f: dict[str, float] = {}
    for rec in ps.parents:
        if getattr(rec, "is_qa", False) and getattr(rec, "answer_ids", ()):
            qf, qc = set(), {}
            for x in rec.match_ids:
                f = _path_fold(v, int(x))
                if not f:
                    continue
                qf.add(f)
                g = float(ps._gidf(int(x)))  # noqa: SLF001
                gidf_f[f] = max(gidf_f.get(f, 0.0), g)
                if g >= args.gidf_hi:
                    qc[f] = max(qc.get(f, 0.0), g)
            af = {f for f in (_path_fold(v, int(x)) for x in rec.answer_ids) if f}
            if qf and af:
                items.append(QA(qf, af, qc))
        else:
            for x in (getattr(rec, "emit_ids", ()) or ()):
                f = _path_fold(v, int(x))
                if f:
                    sent_folds.add(f)
    n = len(items)
    log(f"cha QA dùng được = {n:,} · fold có trong corpus CÂU = {len(sent_folds):,}")
    if n < 100:
        log("⚠️ quá ít cha QA ⇒ không kết luận được.")
        return 1

    LAMS = (0.1, 0.2, 0.35, 0.5, 0.75, 1.5, 2.5)
    VETOS = (0.5, 0.8, 1.0, 1.3, 1.6)

    rng = random.Random(args.seed)
    order = list(range(n))
    rng.shuffle(order)
    folds = [order[i::args.folds] for i in range(args.folds)]

    tot = defaultdict(float)          # thước → tổng
    per_fold: list[dict] = []
    wh_hist: dict[str, int] = defaultdict(int)
    cov_num = 0
    lift_dump: dict[str, list] = {}

    for fi in range(args.folds):
        te_idx = set(folds[fi])
        tr = [items[i] for i in range(n) if i not in te_idx]
        te = [items[i] for i in te_idx]

        # --- WH-fold HỌC TỪ TRAIN (chống rò) -------------------------------------------------
        ask: dict[str, int] = defaultdict(int)
        ans: dict[str, int] = defaultdict(int)
        for it in tr:
            for f in it.qf:
                ask[f] += 1
            for f in it.af:
                ans[f] += 1
        # `gidf_f` lọc token CẤU TRÚC (`ġ` 3.13 · `?` 3.21) — xem docstring §12.51a.
        wh_set = {
            f for f, q in ask.items()
            if q >= args.min_ask and gidf_f.get(f, 0.0) >= args.gidf_hi
            and q / float(q + ans.get(f, 0)) >= args.ask_form_min
        }

        def wh_of(it: QA) -> str:
            c = [f for f in it.qf if f in wh_set]
            if not c:
                return ""
            return max(c, key=lambda f: ask[f] / float(ask[f] + ans.get(f, 0)))

        for it in tr + te:
            it.wh = wh_of(it)

        # --- lift theo nhóm WH ---------------------------------------------------------------
        grp: dict[str, list[QA]] = defaultdict(list)
        for it in tr:
            if it.wh:
                grp[it.wh].append(it)
        N = len(tr)
        base: dict[str, int] = defaultdict(int)
        for it in tr:
            for f in it.af:
                base[f] += 1
        alpha = 1.0 / max(1, N)
        lift: dict[str, dict[str, float]] = {}
        for w, g in grp.items():
            if len(g) < args.min_group:
                continue
            c: dict[str, int] = defaultdict(int)
            for it in g:
                for f in it.af:
                    c[f] += 1
            nw = len(g)
            lift[w] = {
                f: math.log((cw / nw + alpha) / (base.get(f, 0) / N + alpha))
                for f, cw in c.items()
            }
            if fi == 0:
                rows = sorted(lift[w].items(), key=lambda kv: -kv[1])[:14]
                lift_dump[w] = [(f, r, c[f], nw, f in sent_folds) for f, r in rows]

        def score_L(af: set, w: str, only_sent: bool) -> float:
            tab = lift.get(w) or {}
            xs = [tab.get(f, 0.0) for f in af if (not only_sent or f in sent_folds)]
            return sum(xs) / len(xs) if xs else 0.0

        def z(xs: list[float]) -> list[float]:
            m = sum(xs) / len(xs)
            s = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
            return [0.0] * len(xs) if s < 1e-12 else [(x - m) / s for x in xs]

        # --- chấm ---------------------------------------------------------------------------
        acc = defaultdict(float)
        cnt = 0
        n_pad = 0
        for it in te:
            if not it.wh or it.wh not in lift:
                continue
            wh_hist[it.wh] += 1
            # `qc` KHÔNG chứa chính WH-fold: B không được cộng điểm nhờ trùng kiểu hỏi, nếu không
            # là B đã lén dùng tín hiệu mà ta đang định đo riêng cho L.
            qc = {f: g for f, g in it.qcontent.items() if f != it.wh}
            # nhiễu KHÓ = đáp mà chính B chấm cao nhất (§12.51b) — đúng thứ truy hồi đưa lên
            scored = sorted(
                ((_overlap(qc, o.af) / math.sqrt(max(1, len(o.af))), k) for k, o in enumerate(tr)),
                key=lambda x: -x[0],
            )
            topb = [tr[k] for _s, k in scored[: args.n_cand - 1]]
            topbx = [tr[k] for _s, k in scored if tr[k].wh and tr[k].wh != it.wh][
                : args.n_cand - 1]
            if len(topbx) < args.n_cand - 1:
                n_pad += 1
            rnd = [o for o in tr if o is not it]
            rng.shuffle(rnd)
            rnd = rnd[: args.n_cand - 1]

            for tag, dis in (("topB", topb), ("topBx", topbx), ("rnd", rnd)):
                cands = [it] + list(dis)
                idx = list(range(len(cands)))
                rng.shuffle(idx)
                cands = [cands[i] for i in idx]
                gold = idx.index(0)
                B = [_overlap(qc, c.af) / math.sqrt(max(1, len(c.af))) for c in cands]
                L = [score_L(c.af, it.wh, False) for c in cands]
                Ls = [score_L(c.af, it.wh, True) for c in cands]
                zB, zL, zLs = z(B), z(L), z(Ls)
                pairs = [("B", B), ("L", L), ("BL", [a + b for a, b in zip(zB, zL)]),
                         ("BLs", [a + b for a, b in zip(zB, zLs)])]
                # QUÉT λ: kết luận âm mà chỉ thử λ=1 là ẩu — phải cho giả thuyết trọng số TỐT
                # NHẤT của nó. λ chọn trên chính test �⇒ con số này là CẬN TRÊN (oracle), thua
                # cận trên thì thua thật.
                pairs += [(f"lam{lam}", [a + lam * b for a, b in zip(zB, zL)]) for lam in LAMS]
                # PHỦ QUYẾT ĐÁY: `topBx` +0.087 ‖ `topB` −0.077 ⇒ L biết loại nhưng khi nhiễu
                # CÙNG loại thì nó nâng đều gold lẫn nhiễu, thành nhiễu xô lệch B. Toán tử đúng
                # với hình dạng đó là chỉ DÌM đuôi (khác loại rõ), không đụng thứ tự phần còn lại.
                pairs += [(f"veto{th}", [a - (9.9 if b < -th else 0.0) for a, b in zip(zB, zL)])
                          for th in VETOS]
                for name, sc in pairs:
                    rank = sorted(range(len(sc)), key=lambda i: (-sc[i], i)).index(gold) + 1
                    acc[f"{tag}/{name}/top1"] += float(rank == 1)
                    acc[f"{tag}/{name}/mrr"] += 1.0 / rank
                    if name in ("B", "BL"):
                        acc[f"{tag}/{name}/top1/{it.wh}"] += float(rank == 1)
            acc[f"n/{it.wh}"] += 1.0
            cnt += 1
        if not cnt:
            continue
        cov_num += cnt
        row = {k: val / cnt for k, val in acc.items()}
        row["_n"] = cnt
        row["_pad"] = n_pad
        per_fold.append(row)
        for k, val in acc.items():
            tot[k] += val
        log(f"[{time.time()-t0:.0f}s] fold {fi}: n_test={cnt:<4} thiếu_topBx={n_pad:<3} "
            f"topB top1  B={row['topB/B/top1']:.3f} L={row['topB/L/top1']:.3f} "
            f"BL={row['topB/BL/top1']:.3f} BLs={row['topB/BLs/top1']:.3f}")

    n_scored = cov_num
    cov = n_scored / float(n)
    chance = 1.0 / args.n_cand

    log("\n" + "=" * 84)
    log(f"PHỦ (G1): {n_scored}/{n} = **{cov:.1%}** cha QA có WH-fold dùng được "
        f"(nhóm ≥{args.min_group}, ask_form ≥{args.ask_form_min})")
    log(f"  phân bố WH-fold: "
        + ", ".join(f"{w}={c}" for w, c in sorted(wh_hist.items(), key=lambda kv: -kv[1])))
    log(f"  chance top1 = {chance:.3f} ({args.n_cand} ứng viên)")
    log("=" * 84)
    log(f"{'bộ nhiễu':<10}{'thước':<6}{'top1':>9}{'MRR':>9}   (n={n_scored})")
    for tag in ("topB", "topBx", "rnd"):
        for name in ("B", "L", "BL", "BLs"):
            log(f"{tag:<10}{name:<6}{tot[f'{tag}/{name}/top1']/n_scored:>9.3f}"
                f"{tot[f'{tag}/{name}/mrr']/n_scored:>9.3f}")
    log("=" * 84)

    d_bl = (tot["topB/BL/top1"] - tot["topB/B/top1"]) / n_scored
    d_bls = (tot["topB/BLs/top1"] - tot["topB/B/top1"]) / n_scored
    d_bx = (tot["topBx/BL/top1"] - tot["topBx/B/top1"]) / n_scored
    d_rl = tot["rnd/L/top1"] / n_scored
    signs = [r["topB/BL/top1"] - r["topB/B/top1"] for r in per_fold]
    n_pos = sum(1 for s_ in signs if s_ >= 0)
    log(f"Δtop1 (B+L − B) trên `topB` = **{d_bl:+.3f}**  · theo fold: "
        + " ".join(f"{s_:+.3f}" for s_ in signs))
    log(f"Δtop1 (B+L hạn ở fold CÓ trong corpus câu) = **{d_bls:+.3f}**")
    log(f"Δtop1 trên `topBx` (nhiễu KHÁC loại hỏi — nơi L đáng lẽ toả sáng) = **{d_bx:+.3f}**")
    log(f"ĐỐI CHỨNG: L một mình trên nhiễu ngẫu nhiên top1 = {d_rl:.3f} (chance {chance:.3f}) "
        f"⇒ {'CÓ' if d_rl > chance * 1.5 else 'KHÔNG'} tín hiệu độc lập")

    log("\nQUÉT λ trên `topB` (BL = z(B) + λ·z(L)) — CẬN TRÊN vì λ chọn trên test:")
    best = max([(tot[f"topB/lam{lam}/top1"] / n_scored, lam) for lam in LAMS]
               + [(tot["topB/BL/top1"] / n_scored, 1.0)])
    log("  " + "  ".join(
        f"λ={lam}:{tot[f'topB/lam{lam}/top1']/n_scored:.3f}" for lam in LAMS)
        + f"  λ=1.0:{tot['topB/BL/top1']/n_scored:.3f}")
    log(f"  λ tốt nhất = {best[1]} → top1 {best[0]:.3f} ‖ B thuần {tot['topB/B/top1']/n_scored:.3f}"
        f"  ⇒ Δ CẬN TRÊN = **{best[0]-tot['topB/B/top1']/n_scored:+.3f}**")

    for tag in ("topB", "topBx"):
        b_ = tot[f"{tag}/B/top1"] / n_scored
        log(f"\nPHỦ QUYẾT ĐÁY trên `{tag}` (dìm ứng viên có z(L) < −th) ‖ B thuần {b_:.3f}:")
        log("  " + "  ".join(
            f"th={th}:{tot[f'{tag}/veto{th}/top1']/n_scored:.3f}"
            f"({tot[f'{tag}/veto{th}/top1']/n_scored-b_:+.3f})" for th in VETOS))

    log("\nTHEO LOẠI HỎI (top1 trên `topB`) — cơ chế có thể ĐÚNG ở vài loại mà chìm ở toàn cục:")
    log(f"  {'WH':<8}{'n':>5}{'B':>8}{'B+L':>8}{'Δ':>8}")
    for w, c in sorted(wh_hist.items(), key=lambda kv: -kv[1]):
        nb = tot.get(f"n/{w}", 0.0) or 1.0
        b_, bl_ = tot[f"topB/B/top1/{w}"] / nb, tot[f"topB/BL/top1/{w}"] / nb
        log(f"  {w:<8}{int(nb):>5}{b_:>8.3f}{bl_:>8.3f}{bl_-b_:>+8.3f}")

    g1 = cov >= 0.40
    g2 = d_bl >= 0.10 and n_pos >= max(1, args.folds - 1)
    g3 = d_bls >= 0.05 and d_bl > 0 and d_bls >= 0.5 * d_bl
    log("")
    log(f"G1 phủ ≥40%             : {'ĐẠT' if g1 else 'TRƯỢT'}  ({cov:.1%})")
    log(f"G2 Δtop1 ≥+0.10 & ≥{max(1,args.folds-1)}/{args.folds} fold: "
        f"{'ĐẠT' if g2 else 'TRƯỢT'}  ({d_bl:+.3f}, {n_pos}/{len(per_fold)} fold ≥0)")
    log(f"G3 chuyển được ≥ nửa lợi: {'ĐẠT' if g3 else 'TRƯỢT'}  ({d_bls:+.3f})")
    log(f"\n⇒ {'ĐI TIẾP: viết bước chọn câu đáp' if (g1 and g2 and g3) else 'CHỐT ÂM TÍNH — không viết đường sinh'}")

    log("\n" + "=" * 84)
    log("TOP LIFT mỗi nhóm WH (fold 0) — cột `câu?` = fold này CÓ trong corpus câu không.")
    log("Lift dồn vào fold `câu?=-` ⇒ tín hiệu riêng của corpus QA, không xếp hạng nổi cha câu.")
    for w, rows in sorted(lift_dump.items()):
        insent = sum(1 for *_, s_ in rows if s_)
        log(f"\n  WH={w!r}  (n_nhóm={rows[0][3]})  {insent}/{len(rows)} top-fold có trong corpus câu")
        log("    " + "  ".join(f"{f}[{r:.2f}{'·' if s_ else '-'}]" for f, r, _c, _nw, s_ in rows))
    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
