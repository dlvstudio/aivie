"""§82 A/B — cơ chế TIÊU THỤ CẠNH có phải thủ phạm làm mất trật tự từ?

§81 để lại: 16/20 prompt không hề phát dấu chấm. Đối chiếu với câu nguồn thật
(`data/text_corpus/dinh_nghia.jsonl`) cho thấy mạng lấy ĐÚNG câu nguồn — mọi từ
sinh ra đều nằm trong câu đó, không có từ lạ — nhưng MẤT TRẬT TỰ sau ~8–12 token
rồi quay vòng trong túi từ:

    NGUỒN  Hổ là một loài động vật có lãnh thổ hoạt động rất rộng, và các cá thể…
    SINH   hổ là một loài động vật có lãnh thổ hoạt  rất động  thể và các để đến
    NGUỒN  sóc là thú gặm nhấm lớn hơn chuột, sống trên cây, mõm tròn…
    SINH   sóc là thú gặm  một quả lớn nhấm hơn  và hạt cây ăn trên búp sống mõm

Nghi phạm: §12.144/145 `mark_w_skip_consume` — cạnh dùng rồi bị chặn CỨNG
(`ar_w_skip_consume_edge_scale = 0.0`). Câu định nghĩa hay lặp từ (`hoa`, `cây`,
`thể`, `động`), nên cạnh đúng bị khoá và bước đi buộc phải vòng.

╔═ CỔNG CHỐT TRƯỚC KHI CHẠY ═══════════════════════════════════════════════╗
║ So với mặc định (giết cứng), một mức được coi là ĐÚNG HƯỚNG chỉ khi ĐỒNG  ║
║ THỜI:                                                                     ║
║   (a) số prompt KẾT ĐƯỢC CÂU (EOS nổ trước max_len) TĂNG, và              ║
║   (b) tỉ lệ token LẶP GIẢM, và                                            ║
║   (c) không prompt nào đang đúng bị hỏng (đối chiếu 4 ca vàng in kèm).    ║
║ Nếu (a) và (b) không cùng cải thiện ⇒ nghi phạm SAI, KHÔNG đổi mặc định.  ║
╚═══════════════════════════════════════════════════════════════════════════╝

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/ab_tieu_thu_canh.py \
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
VANG = {"chó là", "sắt là", "đồng là", "thủ đô của Việt Nam là"}


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/network_v12_sent_dn.pkl"
    state, _ = load_checkpoint(ck)
    v = state.vocab

    # MỖI LẦN CHẠY TỰ IN DÒNG CỜ CỦA CHÍNH NÓ — bài học vụ awk cắt cột.
    muc = [("TẮT hẳn", dict(ar_w_skip_consume_edges=False)),
           ("giết CỨNG (mặc định)", dict(ar_w_skip_consume_edge_scale=0.0)),
           ("mềm 0,3", dict(ar_w_skip_consume_edge_scale=0.3)),
           ("mềm 0,6", dict(ar_w_skip_consume_edge_scale=0.6))]

    for ten, co in muc:
        cfg = HCAConfig()
        cfg.hub_chain_boost = 1.0
        for k, x in co.items():
            setattr(cfg, k, x)
        print(f"\n═══ {ten}  ·  consume_edges="
              f"{getattr(cfg, 'ar_w_skip_consume_edges')}"
              f" · edge_scale={getattr(cfg, 'ar_w_skip_consume_edge_scale')}"
              f" · boost={cfg.hub_chain_boost}", flush=True)
        n_ket = 0
        lap_tu = lap_mau = 0
        for p in PROMPT:
            n_pr = len(v.encode_prompt(p))
            ids = G.generate(state, p, max_len=MAX, config=cfg, seed=0)
            sinh = [int(t) for t in ids[n_pr:]]
            ket = len(sinh) < MAX          # EOS nổ trước trần ⇒ tự đóng câu
            n_ket += int(ket)
            if sinh:
                lap_tu += len(sinh) - len(set(sinh))
                lap_mau += len(sinh)
            if p in VANG:
                print(f"   VÀNG «{p}» ▸ {v.detokenize(ids)}", flush=True)
        print(f"   ── KẾT CÂU {n_ket}/{len(PROMPT)}"
              f" · LẶP {100 * lap_tu / max(1, lap_mau):.1f}%", flush=True)

    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
