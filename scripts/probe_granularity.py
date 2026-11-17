#!/usr/bin/env python3
"""
TK_042 §7.16 — cha-ĐOẠN có giúp hay hại? So HAI checkpoint CHỈ khác granularity.

    PYTHONPATH=src python -u scripts/probe_granularity.py <checkpoint.pkl> [log.txt]

## Câu hỏi (tác giả, 2026-07-17)

> *"vẫn lấy cha đoạn văn (dài, nhiều câu) là một nơ-ron, nhưng chia đoạn đó ra nhiều câu
> và có nơ-ron cho câu, nơ-ron này nối tới cha, rồi các token nối tới nơ-ron CÂU thay vì
> nối tới nơ-ron ĐOẠN DÀI — liệu có giúp gì cho mạng không?"*

**Hai checkpoint đã có sẵn CHÍNH LÀ hai nhánh của thí nghiệm đó** — không cần đổi kiến trúc:

    network_v10_histwiki.pkl            151 430 cha  = CHỈ cha-CÂU
    network_v10_histwiki_paragraph.pkl  176 430 cha  = + 25 000 cha-ĐOẠN

Biến độc lập = ĐÚNG MỘT: có cha-đoạn hay không.

## Đọc kết quả

* `_paragraph` cho đáp **DÀI hơn + copy% CAO hơn** ⇒ cha-đoạn **kéo dài đoạn bị chép**
  ⇒ đề xuất tầng thứ bậc **có lý** (cắt nhỏ = cắt ngắn cái bị chép).
* Hai nhánh **NGANG NHAU** ⇒ cha-đoạn **không bao giờ thắng truy hồi** ⇒ xây tầng thứ bậc
  là **công dã tràng**: mạng đã de-facto chạy ở mức câu rồi.
* `_paragraph` **TỐT hơn** ⇒ đề xuất đi **ngược** hướng.

`copy%` ≈ 100% sẽ xác nhận §7.13c (*"verbatim LÀ đặc tả, không phải lỗi"*) **trên checkpoint
THẬT** lần đầu — §7.11–7.13 đều đo trên mini (1 294 cha), thứ mà sổ tay dự án cảnh báo
*"ckpt nhỏ đánh lừa NGƯỢC DẤU"*.

## ⚠️ Hai caveat PHẢI nhớ khi kết luận

1. **`FRAME_BRACKET` gate TẮT** trên cả hai (151k cha > cap 5000, checkpoint chưa bake).
   Phép so vẫn có kiểm soát (cùng tắt cả hai nhánh), nhưng đây là mạng **thiếu một cổng**.
   Muốn bật: `python scripts/bake_checkpoint_indices.py <ckpt> --in-place`
2. **Vocab 113 572 = từ-mức CŨ, không phải BPE 32k.** Hai checkpoint này có TRƯỚC TK_041
   ⇒ kết luận nói về mạng **tiền-BPE**.

## Vì sao `gc.disable()` (đo được, không phải mê tín)

Unpickle dựng hàng chục triệu object Python nhỏ. GC gen-2 bị trigger liên tục và **mỗi lần
duyệt LẠI toàn bộ heap đã dựng** — trong khi chúng chắc chắn còn sống, không có gì để thu
⇒ tổng chi phí O(n²).

    KHÔNG gc.disable() → 73+ PHÚT vẫn chưa qua nổi bước `unpickle`
    CÓ  gc.disable() → unpickle 3.4s · load_checkpoint TRỌN VẸN 60.2s

Chữ ký nhận bệnh: **RSS đứng im** (không dựng thêm) + **CPU 99%** (GC đi bộ trên heap) +
stack vẫn nằm trong `_pickle`. Đo trên MacBook Intel 8GB.
"""

from __future__ import annotations

import gc
import sys
import time

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.hca import HCAConfig, generate
from nn_aivie.util.progress import FileProgressLog

PROMPTS = [
    "Nhà Trần là",
    "Khúc Thừa Dụ",
    "Ngô Quyền",
    "Lý Thái Tổ",
    "Hai Bà Trưng",
    "Nhà Lý",
    "Trần Hưng Đạo",
    "Lê Lợi",
]


def lcs(a: list[int], b: list[int]) -> int:
    """Chuỗi con LIÊN TIẾP chung dài nhất — thước `chép nguyên văn`."""
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    best = 0
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else 0
            best = max(best, dp[j])
            prev = cur
    return best


def build_ngram_index(ps, n: int, log) -> set[tuple[int, ...]]:
    """Mọi n-gram của TOÀN corpus (mọi cha) → set.

    Vì sao CẦN, và vì sao LCS-với-cha-hạng-1 KHÔNG đủ (lỗi bản đầu):
    bản đầu đo LCS với ĐÚNG MỘT cha — cha mà `rank_matches` xếp hạng 1 cho *prompt*.
    Nhưng bộ sinh dùng agenda/spreading, có thể lấy từ cha KHÁC HẲN. Nếu mạng **KHẢM**
    (ghép mảnh 3-5 token từ hàng chục cha), LCS với bất kỳ cha ĐƠN LẺ nào cũng ngắn
    ⇒ hiện `copy 5%` **y hệt** như khi mạng sinh THẬT. Hai giả thuyết → cùng một số
    ⇒ phép đo KHÔNG phân biệt được. Đó là lỗi im lặng: số ra đẹp, kết luận sai.

    Độ mới n-gram không quan tâm mảnh đến từ cha nào — nó hỏi thẳng: chuỗi này đã
    TỪNG xuất hiện ở ĐÂU ĐÓ trong corpus chưa?
    """
    idx: set[tuple[int, ...]] = set()
    for j, rec in enumerate(ps.parents):
        s = rec.source_ids
        for i in range(len(s) - n + 1):
            idx.add(tuple(s[i : i + n]))
        if j % 20000 == 0 and j:
            log.log(f"  n-gram index {j:,}/{len(ps.parents):,} · {len(idx):,} {n}-gram")
    return idx


def novelty(out: list[int], idx: set[tuple[int, ...]], n: int) -> tuple[int, int]:
    """(số n-gram MỚI, tổng n-gram) — 'mới' = chưa từng có ở BẤT KỲ cha nào."""
    grams = [tuple(out[i : i + n]) for i in range(len(out) - n + 1)]
    return sum(1 for g in grams if g not in idx), len(grams)


def main() -> int:
    ck = sys.argv[1]
    log = FileProgressLog(sys.argv[2] if len(sys.argv) > 2 else "out/probe_gran.log")
    N = 5  # 5-gram: đủ dài để trùng ngẫu nhiên là hiếm, đủ ngắn để cụm thật vẫn khớp

    gc.disable()  # ← 73 phút → 3.4s (xem docstring)
    state, _ = load_checkpoint(ck, logger=log)
    gc.enable()

    v, ps = state.vocab, state.parents
    lens = [len(p.source_ids) for p in ps.parents]
    n_long = sum(1 for L in lens if L > 60)
    print(f"\n{ck.split('/')[-1]}")
    print(
        f"  cha={len(ps.parents):,} · dài>60tok={n_long:,} ({100*n_long/len(lens):.1f}%)"
        f" · median={sorted(lens)[len(lens)//2]}"
    )

    log.log(f"dựng index {N}-gram toàn corpus …")
    t0 = time.time()
    idx = build_ngram_index(ps, N, log)
    log.log(f"✓ index: {len(idx):,} {N}-gram phân biệt ({time.time()-t0:.0f}s)")
    print(f"  index: {len(idx):,} {N}-gram phân biệt trong toàn corpus")

    cfg = HCAConfig()
    tot_out = tot_copy = tot_new = tot_gram = 0
    rows = []
    for i, p in enumerate(PROMPTS, 1):
        t0 = time.time()
        log.log(f"sinh {i}/{len(PROMPTS)}: {p!r} …")
        ids = generate(state, p, config=cfg, multi=True, max_len=40)
        out = ids[len(v.encode_prompt(p)) :]

        best_l = best_c = 0
        for pi, _sc, *_ in ps.rank_matches(list(v.encode_prompt(p)), v, top_n=1):
            rec = ps.parents[pi]
            best_l = len(rec.source_ids)
            best_c = lcs(out, list(rec.source_ids))

        n_new, n_gram = novelty(out, idx, N)
        tot_out += len(out)
        tot_copy += best_c
        tot_new += n_new
        tot_gram += n_gram
        rows.append((p, best_l, len(out), best_c, n_new, n_gram, v.detokenize(ids)[len(p) :][:56]))
        log.log(
            f"  ✓ {i}/{len(PROMPTS)} ({time.time()-t0:.0f}s) cha={best_l}tok "
            f"đáp={len(out)}tok copy={100*best_c/max(len(out),1):.0f}% "
            f"MỚI={n_new}/{n_gram} ({100*n_new/max(n_gram,1):.0f}%)"
        )

    print(
        f"\n{'prompt':>16} {'cha(tok)':>9} {'đáp(tok)':>9} {'LCS':>5} {'copy%':>6}"
        f" {'mới/'+str(N)+'gram':>11} {'MỚI%':>5}  text"
    )
    for p, bl, lo, bc, nn, ng, t in rows:
        print(
            f"{p:>16} {bl:>9} {lo:>9} {bc:>5} {100*bc/max(lo,1):>5.0f}%"
            f" {str(nn)+'/'+str(ng):>11} {100*nn/max(ng,1):>4.0f}%  {t!r}"
        )
    print(
        f"\n  TỔNG: đáp TB {tot_out/len(PROMPTS):.1f} tok · copy TB {100*tot_copy/max(tot_out,1):.0f}%"
        f" · {N}-gram MỚI {tot_new}/{tot_gram} = {100*tot_new/max(tot_gram,1):.0f}%"
    )
    print(f"\n  ⇒ MỚI ≈ 0%  = mọi cụm 5 token đều CHÉP từ corpus (khảm) — không sinh gì mới")
    print(f"    MỚI cao   = mạng ghép ra chuỗi CHƯA TỪNG CÓ ⇒ sinh thật (phải đọc tay xem có nghĩa)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
