#!/usr/bin/env python
"""Sinh mẫu cho CHẤM MÙ (bằng AI + đọc tay) — thay khung khớp-agenda cứng.

Quyết định tác giả (2026-07-30): bỏ đo verbatim-copy/agenda vì mục tiêu là SINH SÁNG TẠO,
"khớp agenda không phải mục tiêu". Thước mới: chấm bằng AI + người đọc, KHÔNG so với câu nhớ.

Script này CHỈ sinh + giấu nhãn + trộn. Không tự chấm (kẻo thiên lệch theo kỳ vọng). Xuất:
  out/blind_samples.txt  — mỗi mục một id ngẫu nhiên, KHÔNG lộ cấu hình, thứ tự đã trộn → để chấm
  out/blind_key.json     — bản đồ id → (cấu hình, prompt) → chỉ mở SAU khi chấm xong
  out/blind_readable.txt  — nhóm theo cấu hình, cho người đọc

Chạy:
  PYTHONPATH=src python scripts/gen_samples_blind.py --checkpoint checkpoints/mini_sent_ep_qa.pkl
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402

# Prompt đa dạng — dùng như HẠT GIỐNG, KHÔNG có đáp án kỳ vọng để so. Mở rộng §12.112 (n=12→30).
PROMPTS = [
    # sự kiện lịch sử — mở
    "Nhà Lý dời đô về",
    "Ngô Quyền đánh bại quân Nam Hán năm",
    "Trần Hưng Đạo tiêu diệt quân Nguyên Mông trên sông Bạch Đằng năm",
    "Kinh đô đầu tiên của nhà nước Đại Cồ Việt là",
    "Lê Lợi khởi nghĩa ở Lam Sơn năm 1418, sau nhiều năm",
    "Đinh Bộ Lĩnh dẹp loạn mười hai sứ quân rồi",
    "Hai Bà Trưng phất cờ khởi nghĩa năm 40 để",
    "Quang Trung đại phá quân Thanh vào",
    "Nhà Trần ba lần đánh bại quân Nguyên Mông nhờ",
    "Chiến thắng Điện Biên Phủ năm 1954 đã",
    # QA
    "Hỏi: Hoa Lư nằm ở tỉnh nào? Đáp:",
    "Hỏi: Ai đánh bại quân Nam Hán trên sông Bạch Đằng? Đáp:",
    "Hỏi: Nhà Lý dời đô về đâu? Đáp:",
    "Hỏi: Kinh đô của nhà Nguyễn ở đâu? Đáp:",
    "Hỏi: Lê Lợi khởi nghĩa ở đâu? Đáp:",
    # so sánh / đa thực thể
    "Cả Hoa Lư và Thăng Long đều từng là",
    "Hỏi: so sánh nhà Lý và nhà Trần Đáp:",
    "Ngô Quyền và Lê Hoàn giống nhau ở chỗ",
    # định nghĩa / thường thức — mở, không phải fact lịch sử
    "Phở là món ăn",
    "Việt Nam là một quốc gia nằm ở",
    "Áo dài là trang phục truyền thống",
    "Sông Hồng bồi đắp nên",
    "Tết Nguyên Đán là dịp",
    "Cà phê Việt Nam nổi tiếng vì",
    # ngữ cảnh dài
    "Trong thời kỳ Bắc thuộc, nhân dân ta",
    "Sau khi giành được độc lập, các triều đại phong kiến Việt Nam",
    "Nền văn minh lúa nước ở đồng bằng sông Hồng đã",
    "Trải qua hàng nghìn năm dựng nước và giữ nước, dân tộc ta",
    "Thủ đô Hà Nội có lịch sử hơn nghìn năm, từ khi",
    "Vịnh Hạ Long với hàng nghìn hòn đảo đá vôi là",
]

# Cấu hình A/B — nhãn ngắn, GIẤU khi chấm. §12.112: w10 thua rõ ⇒ bỏ; giữ OFF vs w05 cho sạch.
CONFIGS = {
    "OFF": dict(),
    "vote_w05": dict(ar_matrix_w_skip_decode_weight=0.5, hybrid_temperature=0.3,
                     ar_sharp_gate=False),
}

# ar_no_agenda=True: TẮT truy hồi/multizone ⇒ ép ĐƯỜNG SINH AR matrix-core, nơi W_skip sống.
# Không có cờ này thì generate() chạy đường CHÉP câu đã nhớ và kênh phiếu không bao giờ tới lượt
# (đo được: OFF/w05/w10 ra byte-identical). Đây mới là đường "sáng tạo" cần chấm.
BASE = dict(ar_matrix_core=True, ar_style_payoff=False, ar_matrix_stack_weight=0.0,
            ar_no_agenda=True, ar_sent_agenda_handoff=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    print(f"[{time.time()-t0:.0f}s] loaded {len(state.parents):,} cha", flush=True)

    rows = []  # (cfg_name, prompt, text, prompt_ids, cont_ids)
    for cfg_name, over in CONFIGS.items():
        cfg = HCAConfig(**{**BASE, **over})
        for p in PROMPTS:
            out = generate(state, p, max_len=args.max_len, config=cfg, seed=args.seed)
            pids = list(state.vocab.encode_prompt(p))
            cids = [int(t) for t in out[len(pids):]]
            cont = decode_surface(state, out[len(pids):]).strip()
            rows.append((cfg_name, p, cont, [int(t) for t in pids], cids))
        print(f"[{time.time()-t0:.0f}s] xong {cfg_name}", flush=True)

    # gán id ngẫu nhiên + trộn để chấm mù
    rng = random.Random(20260730)
    ids = [f"S{n:03d}" for n in range(len(rows))]
    rng.shuffle(ids)
    key = {}
    blind = []
    for sid, (cfg_name, p, cont, pids, cids) in zip(ids, rows):
        key[sid] = {"config": cfg_name, "prompt": p, "prompt_ids": pids, "cont_ids": cids}
        blind.append((sid, p, cont))
    blind.sort(key=lambda r: r[0])

    outdir = _ROOT / "out"
    with open(outdir / "blind_samples.txt", "w") as f:
        f.write("# CHẤM MÙ — không có cấu hình nào lộ ở đây. Rubric ở blind_rubric.md.\n\n")
        for sid, p, cont in blind:
            f.write(f"[{sid}]  PROMPT: {p}\n        SINH  : {cont}\n\n")
    with open(outdir / "blind_key.json", "w") as f:
        json.dump(key, f, ensure_ascii=False, indent=1)
    with open(outdir / "blind_readable.txt", "w") as f:
        for cfg_name in CONFIGS:
            f.write(f"\n===== {cfg_name} =====\n")
            for c, p, cont, _pi, _ci in rows:
                if c == cfg_name:
                    f.write(f"  {p}\n    → {cont}\n")

    print(f"[{time.time()-t0:.0f}s] {len(rows)} mẫu → out/blind_samples.txt "
          f"· key giấu ở blind_key.json", flush=True)


if __name__ == "__main__":
    main()
