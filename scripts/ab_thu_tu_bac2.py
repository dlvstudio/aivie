"""§83 A/B — nối tín hiệu thứ tự BẬC-2 của cha vào đường sinh thật.

§83.1 đo được `ParentStore.seq_next` = **0 lời gọi** trong `generate()` thật, dù
vòng production chạy đủ 4 lần: kênh bậc-2 chỉ nối vào `generate_autoregressive`,
đường mà production không bao giờ vào. Lần thứ TƯ cùng kiểu lỗi trong hai phiên.

THƯỚC MỚI — §82.3 đã vạch ra «KẾT CÂU» đo sai thứ (`chim sẻ là loài cái kia có
không thể…` dừng sớm nên vẫn được tính là kết câu). Thước chính của vòng này đo
THẲNG trật tự:

    TRẬT TỰ = tỉ lệ cặp token LIỀN KỀ sinh ra có thật trong bigram của cha

Đó đúng là đại lượng §82 nói đang hỏng: từ thì đúng câu nguồn, cặp liền kề thì
sai. `LẶP` và `KẾT CÂU` giữ lại làm biến canh, không làm biến quyết định.

╔═ CỔNG CHỐT TRƯỚC KHI CHẠY ═══════════════════════════════════════════════╗
║ Một mức ĐÚNG HƯỚNG chỉ khi ĐỒNG THỜI:                                    ║
║   (a) TRẬT TỰ tăng so với mức 0 (tắt), và                                ║
║   (b) LẶP KHÔNG tăng, và                                                  ║
║   (c) 4 ca vàng không hỏng (in kèm để đối chiếu bằng mắt).               ║
║ Không mức nào qua ⇒ ÂM TÍNH, để mặc định 0, ghi lại, không vá thêm.      ║
╚═══════════════════════════════════════════════════════════════════════════╝

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/ab_thu_tu_bac2.py \
        checkpoints/network_v12_sent_dn.pkl
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca import generate_hebbian as G  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()
MAX = 24

PROMPT = [
    "chó là", "mèo là", "sóc là", "hổ là", "sắt là", "đồng là",
    "chim sẻ là", "lúa là", "muối là",
    "cây bàng là", "con tê tê là", "nước là", "mặt trời là",
    "nghịch ngợm như loài", "chó và mèo đều là",
    "chó nghịch ngợm, mèo nghịch ngợm, còn", "hổ giống mèo vì",
    "thủ đô của Việt Nam là", "vì sao trời mưa?", "một năm có bao nhiêu",
]
VANG = ["chó là", "sắt là", "hổ là", "lúa là"]


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/network_v12_sent_dn.pkl"
    state, _ = load_checkpoint(ck)
    v = state.vocab
    ps = state.parents
    ps._ensure_seq_index()          # noqa: SLF001 — cần `_seq_bi` làm thước
    bi = ps._seq_bi                 # noqa: SLF001
    print(f"\n### {ck} · {len(ps.parents)} cha · bigram-cha {len(bi)} hàng")

    for w in (0.0, 0.5, 1.0, 2.0):
        cfg = HCAConfig()
        cfg.hub_chain_boost = 1.0
        cfg.ar_w_skip_order2_weight = w
        # MỖI LẦN CHẠY TỰ IN DÒNG CỜ CỦA CHÍNH NÓ.
        print(f"\n═══ order2_weight={cfg.ar_w_skip_order2_weight}"
              f" · order2_min={cfg.ar_seq_order2_min}"
              f" · boost={cfg.hub_chain_boost}", flush=True)
        tt_co = tt_mau = 0
        lap_tu = lap_mau = 0
        n_ket = 0
        for p in PROMPT:
            n_pr = len(v.encode_prompt(p))
            ids = [int(t) for t in G.generate(state, p, max_len=MAX,
                                              config=cfg, seed=0)]
            sinh = ids[n_pr:]
            n_ket += int(len(sinh) < MAX)
            if sinh:
                lap_tu += len(sinh) - len(set(sinh))
                lap_mau += len(sinh)
            # cặp liền kề tính từ token cuối của prompt trở đi
            for a, b in zip(ids[max(0, n_pr - 1):], ids[max(0, n_pr - 1) + 1:]):
                tt_mau += 1
                tt_co += int(b in bi.get(a, ()))
            if p in VANG:
                print(f"   VÀNG «{p}» ▸ {v.detokenize(ids)}", flush=True)
        print(f"   ── TRẬT TỰ {100 * tt_co / max(1, tt_mau):5.1f}%"
              f"  · LẶP {100 * lap_tu / max(1, lap_mau):5.1f}%"
              f"  · KẾT CÂU {n_ket}/{len(PROMPT)} (biến canh)", flush=True)

    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
