"""§83 ĐO — đường sinh THẬT có dùng tín hiệu thứ tự bậc-2 của cha không?

§82 kết luận: mạng truy đúng câu nguồn nhưng MẤT TRẬT TỰ sau ~8–12 token. Đi
tìm `d_max` thì gặp thứ codebase đã tự ghi:

    parent_store.py:3228  «Global W_seq chỉ là bigram (bậc-1) → word-salad;
                           cha lưu trật tự đầy đủ nên successor bậc-2 …»
    varlm.py:4            «W_seq là order-1 → mất trí nhớ ngữ cảnh sau mỗi từ»

Tức `d_max` KHÔNG phải cửa sổ ngữ cảnh — nới nó chỉ nới khoảng cách cặp, không
thêm trí nhớ. Lối thoát đã có sẵn: `ParentStore.seq_next(prev, cur)`, phân bố kế
tiếp bậc-2 dựng từ chuỗi cha.

Đọc nguồn nói `seq_next` chỉ được gọi ở `generate_autoregressive` (dòng 3282) —
đường mà production KHÔNG BAO GIỜ vào (§79). Nhưng đọc nguồn đã lừa ba lần ở
TK_052 (vòng 5/6/7), nên ĐO THẲNG bằng cách đếm lời gọi thật.

╔═ CỔNG CHỐT TRƯỚC KHI CHẠY ═══════════════════════════════════════════════╗
║ Đếm lời gọi `ParentStore.seq_next` trong MỘT lần `generate()` thật:       ║
║   = 0  ⇒ tín hiệu bậc-2 CHẾT trên đường production (lần thứ TƯ cùng kiểu) ║
║          ⇒ đáng nối vào.                                                  ║
║   > 0  ⇒ đọc nguồn SAI, tín hiệu đã được dùng ⇒ word-salad có nguyên nhân ║
║          khác ⇒ DỪNG, không vá, báo lại.                                  ║
║ Kèm chứng: phải xác nhận đã đi qua `_w_skip_primary_loop` (đường thật).   ║
╚═══════════════════════════════════════════════════════════════════════════╝

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/probe_tin_hieu_thu_tu.py \
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
from nn_aivie.core.parent_store import ParentStore  # noqa: E402
from nn_aivie.hca import generate_hebbian as G  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()
PROMPT = ["hổ là", "sóc là", "lúa là", "chó là"]


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/network_v12_sent_dn.pkl"
    cfg = HCAConfig()
    cfg.hub_chain_boost = 1.0
    state, _ = load_checkpoint(ck)
    v = state.vocab

    dem = {"seq_next": 0, "vong_that": 0}

    _sn = ParentStore.seq_next

    def seq_next_dem(self, prev, cur, **kw):
        dem["seq_next"] += 1
        return _sn(self, prev, cur, **kw)

    ParentStore.seq_next = seq_next_dem

    _loop = G._w_skip_primary_loop

    def loop_dem(*a, **kw):
        dem["vong_that"] += 1
        return _loop(*a, **kw)

    G._w_skip_primary_loop = loop_dem

    print(f"\n### {ck} · ar_w_skip_primary={getattr(cfg, 'ar_w_skip_primary', None)}"
          f" · decode_dmax={getattr(cfg, 'ar_matrix_w_skip_decode_dmax', None)}\n")
    for p in PROMPT:
        truoc = dict(dem)
        ids = G.generate(state, p, max_len=24, config=cfg, seed=0)
        print(f"   «{p}» ▸ {v.detokenize(ids)}")
        print(f"        seq_next +{dem['seq_next'] - truoc['seq_next']}"
              f" · _w_skip_primary_loop +{dem['vong_that'] - truoc['vong_that']}")

    print(f"\n  TỔNG: seq_next {dem['seq_next']} lời gọi"
          f" · vòng production {dem['vong_that']} lần")
    if dem["vong_that"] == 0:
        print("  ⚠️ KHÔNG đi qua vòng production ⇒ phép đo VÔ NGHĨA, sửa rồi đo lại")
    elif dem["seq_next"] == 0:
        print("  CỔNG: ĐẠT ⇒ tín hiệu bậc-2 CHẾT trên đường thật, đáng nối vào")
    else:
        print("  CỔNG: TRƯỢT ⇒ tín hiệu ĐÃ được dùng, word-salad do nguyên nhân khác")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
