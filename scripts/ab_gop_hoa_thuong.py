"""§86 A/B — gộp HOA/THƯỜNG lúc truy hồi có kéo cha đúng lên không, và có hại?

§85.3 đo được `hổ` và `Hổ` là HAI token khác nhau: từ điển viết thường,
Wikipedia viết hoa, prompt viết thường ⇒ nửa kho không với tới được.

    «hổ là» → cha ĐÚNG hạng 18   (đỉnh: hổ cốt, hổ phù, hổ phách — mục từ điển)
    «Hổ là» → cha ĐÚNG hạng  7   (đỉnh: Hổ là loài vật có lực cắn mạnh thứ hai)

Cách gộp đo ở đây là NỞ TRUY VẤN: chạy `rank_matches` cho cả hai dạng viết rồi
trộn theo điểm cao nhất. Không đụng vào bộ xếp hạng, không dựng lại chỉ mục —
nên vừa là phép đo vừa là một bản cài đặt khả dĩ.

MỐC ĐÚNG — sửa lỗi §85.4. Bản trước nhận cha đúng bằng mẩu chuỗi lỏng lẻo nên
`lúa` bắt nhầm `Các trang trại…`, `sóc` bắt nhầm `chuột là thú gặm nhấm…`. Nay:

    cha ĐÚNG = cha có văn bản (hạ thường) BẮT ĐẦU bằng prompt (hạ thường)

Định nghĩa này độc nhất, mù hoa/thường, và không cần mốc viết tay.

╔═ CỔNG CHỐT TRƯỚC KHI CHẠY ═══════════════════════════════════════════════╗
║ (a) Trên prompt ĐỊNH NGHĨA: tỉ lệ cha đúng vào top-20 TĂNG, và hạng       ║
║     trung vị GIẢM.                                                        ║
║ (b) PHỦ QUYẾT — chiều ngược: trên prompt DANH TỪ RIÊNG, hạng cha đúng     ║
║     KHÔNG được tụt. Gộp hoa/thường có thể kéo danh từ riêng lẫn vào danh  ║
║     từ chung; nếu (b) hỏng thì (a) đẹp mấy cũng bỏ.                       ║
║ (c) In đỉnh bảng vài ca cho MẮT đọc (bài học 124).                        ║
╚═══════════════════════════════════════════════════════════════════════════╝

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/ab_gop_hoa_thuong.py \
        checkpoints/network_v12_sent_dn.pkl
"""
from __future__ import annotations

import gc
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()
TOP = 200

DINH_NGHIA = ["hổ là", "chó là", "mèo là", "lúa là", "sắt là", "đồng là",
              "sóc là", "muối là", "nước là", "mặt trời là", "chim sẻ là",
              "cá là", "gạo là", "vàng là", "than là"]
# chiều NGƯỢC — danh từ riêng, nơi gộp hoa/thường dễ kéo nhầm nhất
RIENG = ["Hà Nội là", "Việt Nam là", "Sao Thổ là", "Trái Đất là",
         "Thái Lan là", "Man City là"]


def _cfg(ps, v):
    c = HCAConfig()
    ps._vocab_ref = v
    ps.focus_pos_alpha = c.focus_pos_alpha
    ps.directed_bigram_bonus = c.directed_bigram_bonus
    ps.directed_bigram_min_idf = c.directed_bigram_min_idf
    ps.directed_bigram_dmax = int(c.directed_bigram_dmax)
    ps.directed_bigram_lam = float(c.directed_bigram_lam)
    ps.directed_bigram_accum = str(c.directed_bigram_accum)


def _bien_the(p: str) -> list[str]:
    """Prompt gốc + dạng viết hoa/thường ngược lại của chữ đầu."""
    ra = [p]
    if p[:1].islower():
        ra.append(p[0].upper() + p[1:])
    elif p[:1].isupper():
        ra.append(p[0].lower() + p[1:])
    return ra


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/network_v12_sent_dn.pkl"
    state, _ = load_checkpoint(ck)
    v, ps = state.vocab, state.parents
    _cfg(ps, v)

    # MỐC ĐÚNG mù hoa/thường, dựng một lần: prompt → tập chỉ số cha hợp lệ
    print("\n### dựng mốc đúng (văn bản cha hạ thường bắt đầu bằng prompt)…",
          flush=True)
    moc: dict[str, set[int]] = {p: set() for p in DINH_NGHIA + RIENG}
    for i, r in enumerate(ps.parents):
        t = v.detokenize(list(r.emit_ids)).lower().lstrip()
        for p in moc:
            if t.startswith(p.lower()):
                moc[p].add(i)
    for nhom, ten in ((DINH_NGHIA, "ĐỊNH NGHĨA"), (RIENG, "DANH TỪ RIÊNG")):
        thieu = [p for p in nhom if not moc[p]]
        print(f"###   {ten}: {sum(1 for p in nhom if moc[p])}/{len(nhom)} prompt"
              f" có cha đúng" + (f" · BỎ (không có): {thieu}" if thieu else ""))

    def hang(p: str, gop: bool) -> int | None:
        """Hạng của cha ĐÚNG tốt nhất; None nếu ngoài top-TOP."""
        ds = _bien_the(p) if gop else [p]
        diem: dict[int, float] = {}
        for q in ds:
            qs = [int(t) for t in v.encode_prompt(q)]
            for pi, sc in ps.rank_matches(qs, v, None, fuzzy_topk=0, top_n=TOP):
                pi = int(pi)
                if sc > diem.get(pi, -1e18):
                    diem[pi] = float(sc)
        xep = sorted(diem.items(), key=lambda kv: -kv[1])
        for i, (pi, _s) in enumerate(xep, 1):
            if pi in moc[p]:
                return i
        return None

    for ten, nhom in (("ĐỊNH NGHĨA", DINH_NGHIA), ("DANH TỪ RIÊNG (phủ quyết)", RIENG)):
        print(f"\n═══ {ten}")
        for gop in (False, True):
            hs, vao20 = [], 0
            n = 0
            for p in nhom:
                if not moc[p]:
                    continue
                n += 1
                h = hang(p, gop)
                hs.append(h if h else TOP + 1)
                vao20 += int(h is not None and h <= 20)
            print(f"   {'GỘP hoa/thường' if gop else 'gốc (mốc)      '}"
                  f" · vào top-20 {vao20}/{n}"
                  f" · hạng trung vị {statistics.median(hs):.0f}", flush=True)
        # cho MẮT đọc
        for p in nhom[:3]:
            if moc[p]:
                print(f"     «{p}» hạng: gốc {hang(p, False)} → gộp {hang(p, True)}")

    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
