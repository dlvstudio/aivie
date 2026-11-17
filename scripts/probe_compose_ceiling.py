#!/usr/bin/env python3
"""TRẦN CỦA TỔ HỢP — có VẬT LIỆU để soạn câu không? TK_045 §12.100. **Chỉ đo.**

    PYTHONPATH=src python -u scripts/probe_compose_ceiling.py \
        --checkpoint checkpoints/mini_sent_ep_qa.pkl \
        --log-file out/compose_mini.log

## Vì sao đây là phép đo ĐẦU TIÊN của hướng soạn câu

§12.99d: bốn phần, và phần này rẻ nhất mà giết được nhiều nhất. Nó hỏi **"có vật liệu không"**
trước khi hỏi **"lắp thế nào"**. Nếu một câu giữ lại không lắp nổi từ mảnh của các câu khác thì
mọi cơ chế lắp ghép đều vô nghĩa, và ta tiết kiệm nhiều tuần.

Lý do nghi ngờ có cơ sở (§12.99c): `slot_chain` chẻ câu thành slot thật, nhưng **77% slot là token
đơn**, và cluster **dùng lại nhiều** thì lại toàn khuôn mẫu (`.<EOS>` 1642 lần · `? Đáp:` 609) còn
mảnh **nội dung** chỉ 2–3 lượt.

## Thước — và bẫy thoái hoá phải chặn

Bẫy: nếu tính cả **token đơn** thì mọi câu phủ 100% (token nào chẳng xuất hiện đâu đó) ⇒ số vô
nghĩa. Mà lắp từng token chính là thứ đường sinh hiện tại đang làm, và nó ra văn què nếu không có
emit của cha để bám. Nên chỉ đếm **mảnh ≥2 token**.

Hai mức vật liệu:

  `cluster`  chỉ dùng cluster ĐÃ ĐĂNG KÝ trong `slot_chain` của nửa A — đo hạ tầng HIỆN CÓ.
  `ngram`    dùng MỌI đoạn liền ≥2 token từng xuất hiện ở nửA A — **trần thật** của việc
             "lắp từ mảnh đã nhớ", không phụ thuộc chất lượng chẻ slot.

`ngram` ≫ `cluster` ⇒ vật liệu có, chỉ hạ tầng chẻ slot chưa khai thác.
Cả hai thấp ⇒ **không có vật liệu**, đóng hướng.

**ĐỐI CHỨNG bắt buộc:** phủ chính câu ấy sau khi **xáo trật tự token**. Nếu câu xáo cũng phủ
tương đương thì mảnh không mã hoá cấu trúc cục bộ nào — nó chỉ là hiện vật tần suất.

## Cổng — CHỐT TRƯỚC (theo §12.99d)

  G1  ≥ **40%** câu giữ lại phủ được ≥ **80%** bằng `ngram` ⇒ **CÓ vật liệu**, đi tiếp bước (0).
  G2  `ngram` phải hơn đối chứng xáo ≥ **0,15** (phủ trung bình) ⇒ mảnh mã hoá cấu trúc thật.
  G3  số mảnh cần cho một câu ≤ **60%** số token của câu ⇒ thật sự đang GHÉP MẢNH, không phải
      lắp gần-như-từng-token đội tên.

Trượt G1 ⇒ đóng hướng soạn câu, ghi âm tính, khỏi dựng thước trôi chảy.
"""

from __future__ import annotations

import argparse
import gc
import random
import statistics as stx
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--log-file", default="out/compose.log")
    ap.add_argument("--max-piece", type=int, default=8, help="độ dài mảnh tối đa khi tiling")
    ap.add_argument("--n-test", type=int, default=1500)
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

    # ---- tách đôi theo CHA (không theo câu: cluster đăng ký ở mức cha) ----------------------
    seqs_a: list[list[int]] = []
    seqs_b: list[list[int]] = []
    chains_a = []
    for i, rec in enumerate(ps.parents):
        e = [int(x) for x in (getattr(rec, "emit_ids", ()) or ())]
        if len(e) < 6 or getattr(rec, "is_qa", False):
            continue
        if (i & 1) == 0:
            seqs_a.append(e)
            chains_a.append(getattr(rec, "slot_chain", ()) or ())
        else:
            seqs_b.append(e)
    log(f"  cha câu: nửa A {len(seqs_a):,} · nửa B {len(seqs_b):,}")

    # ---- vật liệu 1: cluster ĐÃ ĐĂNG KÝ ở nửa A -------------------------------------------
    inv_cluster: set[tuple[int, ...]] = set()
    for ch in chains_a:
        for s_ in ch:
            if getattr(s_, "kind", None) != "cluster":
                continue
            try:
                ids = tuple(int(x) for x in ps.seg.expand_slot(s_))
            except Exception:
                continue
            if len(ids) >= 2:
                inv_cluster.add(ids)
    log(f"  vật liệu `cluster`: {len(inv_cluster):,} mảnh ≥2 token")

    # ---- vật liệu 2: MỌI n-gram ≥2 từ nửa A (trần thật) ------------------------------------
    inv_ngram: set[tuple[int, ...]] = set()
    for e in seqs_a:
        L = len(e)
        for i in range(L):
            for n in range(2, min(args.max_piece, L - i) + 1):
                inv_ngram.add(tuple(e[i:i + n]))
    log(f"  vật liệu `ngram`  : {len(inv_ngram):,} mảnh ≥2 token")

    def tile(seq: list[int], inv: set) -> tuple[float, int]:
        """Ghép tham lam mảnh DÀI NHẤT có sẵn; trả (tỉ lệ token được phủ bởi mảnh ≥2, số mảnh)."""
        i = 0
        covered = 0
        pieces = 0
        L = len(seq)
        while i < L:
            best = 0
            for n in range(min(args.max_piece, L - i), 1, -1):
                if tuple(seq[i:i + n]) in inv:
                    best = n
                    break
            if best >= 2:
                covered += best
                pieces += 1
                i += best
            else:
                i += 1          # token đơn: KHÔNG tính là phủ (chặn bẫy thoái hoá)
        return covered / L, pieces

    rng = random.Random(args.seed)
    tgt = seqs_b[:]
    rng.shuffle(tgt)
    tgt = tgt[: args.n_test]
    log(f"  chấm {len(tgt):,} câu giữ lại\n")

    res: dict[str, list[float]] = {"cluster": [], "ngram": [], "ngram_xáo": []}
    frac_piece: list[float] = []
    for e in tgt:
        c1, _ = tile(e, inv_cluster)
        c2, p2 = tile(e, inv_ngram)
        sh = e[:]
        rng.shuffle(sh)
        c3, _ = tile(sh, inv_ngram)
        res["cluster"].append(c1)
        res["ngram"].append(c2)
        res["ngram_xáo"].append(c3)
        frac_piece.append(p2 / len(e))

    log("=" * 70)
    log(f"{'vật liệu':<14}{'phủ TB':>10}{'phủ ≥0.8':>11}{'phủ ≥0.5':>11}")
    log("=" * 70)
    for k in ("cluster", "ngram", "ngram_xáo"):
        xs = res[k]
        log(f"{k:<14}{stx.mean(xs):>10.3f}{sum(1 for x in xs if x >= 0.8)/len(xs):>11.1%}"
            f"{sum(1 for x in xs if x >= 0.5)/len(xs):>11.1%}")
    log("=" * 70)
    log(f"số mảnh / số token (TB) = {stx.mean(frac_piece):.3f}   "
        f"(thấp = ghép mảnh DÀI; gần 0.5 = gần như từng-token)")

    # §12.104 — ATTESTATION Ở MỐI NỐI. §12.103 chốt: xác suất n-gram không phân biệt "què" với
    # "hiếm". Đường còn lại: hỏi mọi ĐOẠN ≥k token có THẬT SỰ xuất hiện trong emit không. Nhưng
    # có nghi ngại đối xứng: câu GIỮ LẠI cũng có n-gram MỚI dù đúng ngữ pháp. Nên đo, đừng đoán.
    # Thước: với mỗi BIÊN j (giữa token j và j+1), k-gram BẮC QUA biên ấy có được chứng thực không.
    log("\nATTESTATION Ở MỐI NỐI — tỉ lệ biên có k-gram bắc qua ĐƯỢC CHỨNG THỰC:")
    log(f"  {'k':>3}{'corpus(B)':>12}{'ghép nửa':>11}{'xáo':>9}{'TÁCH ghép':>12}")
    for k in (2, 3, 4, 5, 6):
        iv: set[tuple[int, ...]] = set()
        for e in seqs_a:
            L = len(e)
            for i in range(L - k + 1):
                iv.add(tuple(e[i:i + k]))

        def att(seq):
            n = ok = 0
            h = k // 2
            for j in range(len(seq) - 1):
                lo = max(0, j - h + 1)
                if lo + k > len(seq):
                    continue
                n += 1
                ok += tuple(seq[lo:lo + k]) in iv
            return ok / n if n else float("nan")

        ac, ag, ax = [], [], []
        for e in tgt[:600]:
            if len(e) < 2 * k:
                continue
            ac.append(att(e))
            o = tgt[rng.randrange(len(tgt))]
            if len(o) >= 2 * k:
                ag.append(att(e[: len(e) // 2] + o[len(o) // 2:]))
            sh2 = e[:]
            rng.shuffle(sh2)
            ax.append(att(sh2))
        d_ = stx.mean(ac) - stx.mean(ag)
        log(f"  {k:>3}{stx.mean(ac):>12.3f}{stx.mean(ag):>11.3f}{stx.mean(ax):>9.3f}{d_:>+12.3f}")
    log("  `TÁCH ghép` lớn dần theo k ⇒ attestation là tín hiệu nghiệm thu. Phẳng/nhỏ ⇒ nó vướng")
    log("  đúng bẫy §12.103: không phân biệt \"mới mà đúng\" với \"ghép mà què\".")

    g1 = sum(1 for x in res["ngram"] if x >= 0.8) / len(res["ngram"]) >= 0.40
    d = stx.mean(res["ngram"]) - stx.mean(res["ngram_xáo"])
    g2 = d >= 0.15
    g3 = stx.mean(frac_piece) <= 0.60
    # §12.100c — PHỦ THEO CỠ KHO: kho n-gram lớn theo corpus, nên "v12 sẽ khác" phải là số đo
    # được, không phải lập luận. Dựng kho từ 25%/50%/100% nửa A rồi chấm lại cùng bộ câu.
    log("\nPHỦ THEO CỠ KHO VẬT LIỆU (cùng bộ câu chấm):")
    log(f"  {'% nửa A':>9}{'#mảnh':>12}{'phủ TB':>10}{'phủ ≥0.8':>11}")
    trend = []
    for frac in (0.25, 0.5, 1.0):
        sub = seqs_a[: max(1, int(len(seqs_a) * frac))]
        iv: set[tuple[int, ...]] = set()
        for e in sub:
            L = len(e)
            for i in range(L):
                for n in range(2, min(args.max_piece, L - i) + 1):
                    iv.add(tuple(e[i:i + n]))
        cs = [tile(e, iv)[0] for e in tgt]
        hi = sum(1 for x in cs if x >= 0.8) / len(cs)
        trend.append((frac, len(iv), stx.mean(cs), hi))
        log(f"  {frac:>8.0%}{len(iv):>12,}{stx.mean(cs):>10.3f}{hi:>11.1%}")
    if len(trend) >= 2:
        d_hi = trend[-1][3] - trend[-2][3]
        log(f"  ⇒ gấp đôi kho (50%→100%) thêm {d_hi:+.1%} ở cột `phủ ≥0.8`."
            f"  v12 có ~85× cha so mini.")

    log("")
    log(f"G1 ≥40% câu phủ ≥0.8 bằng `ngram` : {'ĐẠT' if g1 else 'TRƯỢT'} "
        f"({sum(1 for x in res['ngram'] if x >= 0.8)/len(res['ngram']):.1%})  ← quyết định")
    log(f"G2 `ngram` hơn xáo ≥0.15          : {'ĐẠT' if g2 else 'TRƯỢT'} ({d:+.3f})")
    log(f"G3 số mảnh ≤60% số token          : {'ĐẠT' if g3 else 'TRƯỢT'} "
        f"({stx.mean(frac_piece):.3f})")
    verdict = ("CÓ vật liệu ⇒ đi tiếp bước (0) dựng thước trôi chảy"
               if g1 else
               "KHÔNG có vật liệu ⇒ ĐÓNG hướng soạn câu, khỏi dựng thước")
    log(f"\n⇒ {verdict}")
    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
