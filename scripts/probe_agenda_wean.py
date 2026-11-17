#!/usr/bin/env python3
"""
TK_042 G6 — CAI NGHIỆN AGENDA: tìm vùng mà kênh `micro` THẬT SỰ có tiếng nói.

    PYTHONPATH=src python -u scripts/probe_agenda_wean.py --checkpoint checkpoints/network_mini_bpe.pkl

## Vì sao (§7.11 — phát hiện quan trọng nhất TK_042)

```
GIẾT HẲN kênh micro (mọi ứng viên điểm BẰNG NHAU) → token vẫn TRÙNG 78/78 = 100%
```

`agenda`/`macro` quyết định TOÀN BỘ đầu ra. Kênh trôi chảy **không nối với đầu ra**. Nên:
* thay `W_seq`→LM (tốt hơn **1.494 bits/token**) đổi **0 token** (§7.11a)
* **cả 8 cơ chế trung tính đều cắm vào `micro`** ⇒ chúng bị **NGẮT MẠCH**, không phải yếu

⇒ **G6 phải làm TRƯỚC G10.** Mọi cải tiến kênh micro sẽ đo ra trung tính MÃI MÃI cho tới khi
micro có tiếng nói.

## Thước: MICRO-INFLUENCE

    micro_influence = % token ĐỔI khi giết hẳn micro

* `0%`  = micro vô dụng (**tình trạng HIỆN TẠI**) ⇒ mọi cơ chế cắm vào nó đều vô nghĩa
* `100%` = micro quyết tất (agenda mất tác dụng ⇒ nội dung sẽ trôi → word-salad)

Đây là **điều kiện CẦN**, không phải mục tiêu. Cai nghiện làm chất lượng **TỤT TRƯỚC KHI LÊN**
— agenda đang là thứ DUY NHẤT giữ nội dung đúng (v4: bỏ agenda → word-salad). Phải đọc tay.

## Vì sao sweep `ar_macro_floor` / `ar_spread_weight`

    scores = (ar_micro_floor + micro/mmax) · (ar_macro_floor + ar_spread_weight·macro/gmax)

Dải động: micro `0.15→1.15` (**7.7×**) · macro `0.10→1.10` (**11×**) ⇒ **macro thắng**.
Nâng `ar_macro_floor` **nén dải động của macro**: floor=1.0 ⇒ macro `1.0→2.0` (**2×**) ⇒ micro
(7.7×) giành lại quyền. Đó là núm trực tiếp nhất.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca import HCAConfig, generate  # noqa: E402
import nn_aivie.hca.ar_candidates as AC  # noqa: E402
import nn_aivie.hca.generate_hebbian as G  # noqa: E402

PROMPTS = [
    "Hỏi: Thủ đô Việt Nam là gì ? Đáp:",
    "Nhà Trần là",
    "Khúc Thừa Dụ",
    "Tiếng Việt",
]

_REAL = AC.build_micro_dict


def _kill_micro(w0, out, st, seq, V, cfg):
    """Mọi ứng viên ĐIỂM BẰNG NHAU ⇒ micro không mang thông tin gì."""
    return {k: 1.0 for k in _REAL(w0, out, st, seq, V, cfg)}


def _set_micro(fn):
    AC.build_micro_dict = fn
    G.build_micro_dict = fn  # generate_hebbian import TRỰC TIẾP → phải patch CẢ HAI


def run(state, cfg, max_len):
    return {p: generate(state, p, config=cfg, multi=True, max_len=max_len) for p in PROMPTS}


def influence(state, cfg, max_len) -> tuple[float, dict]:
    _set_micro(_REAL)
    base = run(state, cfg, max_len)
    _set_micro(_kill_micro)
    dead = run(state, cfg, max_len)
    _set_micro(_REAL)
    tot = chg = 0
    for p in PROMPTS:
        a, b = base[p], dead[p]
        n = max(len(a), len(b))
        tot += n
        chg += n - sum(1 for x, y in zip(a, b) if x == y)
    return (100.0 * chg / max(tot, 1)), base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max-len", type=int, default=24)
    ap.add_argument("--show", action="store_true", help="in text để ĐỌC TAY")
    args = ap.parse_args()

    print(f"load {args.checkpoint} …", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        state, _ = load_checkpoint(args.checkpoint)
    v = state.vocab

    # (nhãn, kwargs) — cai dần: nén dải động macro + gỡ backbone + hạ agenda_boost
    configs = [
        ("HIỆN TẠI (mặc định)", {}),
        ("−backbone", dict(ar_index_backbone=False)),
        ("−backbone −boost", dict(ar_index_backbone=False, ar_agenda_boost=0.0)),
        ("macro_floor 0.3", dict(ar_index_backbone=False, ar_agenda_boost=0.0, ar_macro_floor=0.3)),
        ("macro_floor 1.0", dict(ar_index_backbone=False, ar_agenda_boost=0.0, ar_macro_floor=1.0)),
        ("macro_floor 3.0", dict(ar_index_backbone=False, ar_agenda_boost=0.0, ar_macro_floor=3.0)),
        ("spread 0.2", dict(ar_index_backbone=False, ar_agenda_boost=0.0, ar_macro_floor=1.0,
                            ar_spread_weight=0.2)),
        ("agenda TẮT HẲN", dict(ar_index_backbone=False, ar_agenda_boost=0.0, ar_macro_floor=1.0,
                                ar_spread_weight=0.0)),
    ]
    print(f"\n{'cấu hình':>22} {'micro-influence':>16}  ← % token ĐỔI khi giết micro")
    print(f"{'':>22} {'0% = micro VÔ DỤNG':>16}")
    outs = {}
    for name, kw in configs:
        cfg = HCAConfig(**kw)
        inf, base = influence(state, cfg, args.max_len)
        outs[name] = base
        flag = " ← micro CÓ tiếng nói" if inf > 5 else ""
        print(f"{name:>22} {inf:>15.1f}%{flag}")

    if args.show:
        print(f"\n{'='*66}\n  ĐỌC TAY — cai nghiện làm chất lượng TỤT TRƯỚC KHI LÊN\n{'='*66}")
        for name, _ in configs:
            print(f"\n▶ {name}")
            for p in PROMPTS[:3]:
                t = v.detokenize(outs[name][p])[len(p):]
                print(f"    {t[:78]!r}")
    print("\n⚠ micro-influence là ĐIỀU KIỆN CẦN, KHÔNG phải mục tiêu. Agenda đang là thứ DUY")
    print("  NHẤT giữ nội dung đúng — buông nó thì phải có LM đủ tốt đỡ ngay. ĐỌC TAY bắt buộc.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
