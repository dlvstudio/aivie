"""§85.2 — vì sao khối HOÀ ĐIỂM không vỡ? Cặp có hướng `(X → là)` có bị lọc?

§85.1 đo được, với cấu hình y hệt production (dir_bonus=1.5, dmax=4, accum=sum):
`chó má là`, `chó săn là`, `chó là`, `chó ghẻ là`, `chó đẻ là` HOÀ đúng 2.154.
Cha đúng chôn ở hạng 3 (chó), 4 (lúa), 2 (sắt), 18 (hổ).

`rank_matches` chỉ thưởng cặp có hướng khi `min(idf(a), idf(b)) >= min_idf`
(mặc định 2.0) — «bỏ cặp hư-từ generic khỏi thưởng-hướng». Nhưng với KHUNG ĐỊNH
NGHĨA thì `là` KHÔNG phải hư từ nhiễu: nó chính là dấu hiệu khuôn. Nếu idf(`là`)
< 2.0 thì mọi cặp `(X → là)` bị vứt, và không còn gì phân biệt `chó là` với
`chó má là` — đúng khối hoà quan sát được.

╔═ CỔNG CHỐT TRƯỚC KHI CHẠY ═══════════════════════════════════════════════╗
║ (a) idf(`là`) < min_idf ⇒ XÁC NHẬN cặp khuôn bị lọc. Hướng vá: cho khớp  ║
║     TIỀN TỐ (cha bắt đầu đúng bằng truy vấn) một tín hiệu riêng, thay vì  ║
║     nới `min_idf` toàn cục (nới toàn cục sẽ bơm lại mọi cặp hư từ).      ║
║ (b) idf(`là`) >= min_idf ⇒ giả thuyết SAI, khối hoà do nguyên nhân khác;  ║
║     DỪNG, đo tiếp, không vá.                                             ║
║ Kèm: đếm xem TIỀN TỐ có phân biệt được không — trong khối hoà, có đúng    ║
║ một cha bắt đầu bằng truy vấn hay nhiều?                                  ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()
PROMPT = ["hổ là", "lúa là", "sắt là", "chó là", "sóc là"]


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/network_v12_sent_dn.pkl"
    state, _ = load_checkpoint(ck)
    v, ps = state.vocab, state.parents
    cfg = HCAConfig()
    ps._vocab_ref = v
    ps.focus_pos_alpha = cfg.focus_pos_alpha
    ps.directed_bigram_bonus = cfg.directed_bigram_bonus
    ps.directed_bigram_min_idf = cfg.directed_bigram_min_idf
    ps.directed_bigram_dmax = int(cfg.directed_bigram_dmax)
    ps.directed_bigram_lam = float(cfg.directed_bigram_lam)
    ps.directed_bigram_accum = str(cfg.directed_bigram_accum)
    print(f"\n### min_idf = {ps.directed_bigram_min_idf}\n")

    for p in PROMPT:
        qs = [int(t) for t in v.encode_prompt(p)]
        idfs = [(v.detokenize([t]), ps._idf(int(t))) for t in qs]  # noqa: SLF001
        print(f"── «{p}»")
        for s, x in idfs:
            print(f"     idf({s!r:>8}) = {x:6.3f}"
                  f"{'   ← DƯỚI ngưỡng' if x < ps.directed_bigram_min_idf else ''}")
        cap_qua = min(x for _s, x in idfs) >= ps.directed_bigram_min_idf
        print(f"     ⇒ cặp khuôn qua ngưỡng: {cap_qua}")

        # TIỀN TỐ có phân biệt được trong khối hoà không?
        ranked = ps.rank_matches(qs, v, None, fuzzy_topk=0, top_n=8)
        if ranked:
            dinh = ranked[0][1]
            hoa = [pi for pi, sc in ranked if abs(sc - dinh) < 1e-9]
            tien_to = []
            for pi in hoa:
                e = list(ps.parents[int(pi)].emit_ids)
                if e[:len(qs)] == qs:
                    tien_to.append(v.detokenize(e)[:60])
            print(f"     khối HOÀ đỉnh: {len(hoa)} cha"
                  f" · trong đó khớp TIỀN TỐ: {len(tien_to)}")
            for t in tien_to:
                print(f"        ✓ {t}")
        print()

    print(f"[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
