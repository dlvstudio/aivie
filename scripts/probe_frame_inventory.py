#!/usr/bin/env python
"""P98 (TK_047 §5) — **KHO KHUNG CÂU có substrate không?**

TK_047 đề xuất tách LẬP KHUNG khỏi ĐIỀN KHE. Toàn bộ kiến trúc đó đứng trên một giả định chưa
kiểm: *chuỗi nhãn của các câu **lặp lại** đủ nhiều để khái quát*. Nếu mỗi câu có một chuỗi nhãn
riêng thì không có nơ-ron khung nào tồn tại, và kiến trúc chết ngay tại đây — biết trước khi
viết một dòng cơ chế nào.

## Khung = chuỗi nhãn từ loại của đơn vị

`con chó màu đỏ` → `DT DT TT TT`. Token không có nhãn → `?`. Thuần đếm ⇒ Luật 0 sạch.

## Ba mức hạt trong CÙNG một lần chạy

| mức | vì sao có mặt |
|---|---|
| `ca_cha` | đơn vị tự nhiên của đường chép (54,4%) |
| `menh_de` | đơn vị TK_045 đã dựng — **câu hỏi thật** |
| `cua_so_4` | **TRẦN**, không phải đáp án: cửa sổ quá nhỏ để là khung câu |

⚠️ Ba mức nằm cùng một lần chạy **có chủ ý**. Tôi đoán trước mức `ca_cha` sẽ trượt; đưa sẵn mức
thay thế vào cùng phép đo là để **không** rơi vào vòng "đo → hỏng → nới → đo lại", thứ đã làm
hỏng ba vòng đối chứng ở §16.

## CỔNG G-frame (nêu TRƯỚC)

1. **Chia sẻ** ≥ **30%** đơn vị có chuỗi nhãn trùng ≥1 đơn vị khác.
2. **Tập trung**: top-200 khung phủ ≥ **20%** đơn vị.
3. **Tổ hợp mới**: trung vị `log10(∏ filler khác nhau mỗi khe) − log10(số lần thấy)` ≥ **1,0**
   — khung phải cho phép ≥10× số câu nó đã được thấy. Dưới đó, "sinh" là chép có trang trí.

## Dự đoán ghi TRƯỚC (Claude)

- `ca_cha`: (1) **trượt**, câu dài ⇒ chuỗi nhãn gần như luôn duy nhất.
- `menh_de`: **borderline 25–45%** — đây là câu hỏi thật.
- `cua_so_4`: (1)(2) đạt dễ, nhưng là **TRẦN**, không đọc như đáp án.
- (3) phải đọc **kèm ĐỘ DÀI khung**: khung chia sẻ được có thể toàn loại ngắn tầm thường
  (`DT DT`), lúc đó tổ hợp mới cao **một cách vô nghĩa**.

Chạy: `PYTHONPATH=src python scripts/probe_frame_inventory.py`
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.clause_layer import build_clause_layer  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

GATE_SHARE_MIN = 0.30
GATE_TOP200_MIN = 0.20
GATE_NOVEL_MIN = 1.0

_SHORT = {"danh từ": "N", "động từ": "V", "tính từ": "A", "đại từ": "P",
          "kết từ": "C", "phụ từ": "R", "trợ từ": "T", "cảm từ": "I",
          "tên riêng": "E", "số": "M"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--map", default="out/token_labels_mini.json")
    ap.add_argument("--win", type=int, default=4)
    ap.add_argument("--out", default="out/probe_frame_inventory.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_frame_inventory")
    t0 = time.time()
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()

    mp = json.loads((_ROOT / args.map).read_text(encoding="utf-8"))["tu_loai"]
    lab: dict[int, str] = {}
    for ts, d in mp.items():
        if not d:
            continue
        best = max(d.values())
        top = sorted(k for k, x in d.items() if x == best)
        if len(top) == 1:                       # đồng hạng ⇒ bỏ, đừng phá vỡ bừa
            lab[int(ts)] = _SHORT.get(top[0], "X")
    logger.log(f"token có nhãn trội: {len(lab):,}")

    def skel(ids: list[int]) -> tuple[str, ...]:
        return tuple(lab.get(int(t), "?") for t in ids)

    units: dict[str, list[list[int]]] = {}
    units["ca_cha"] = [[int(t) for t in r.emit_ids] for r in state.parents.parents]

    layer = build_clause_layer(state, pmi_theta=0.0, logger=logger)
    units["menh_de"] = [[int(t) for t in c.tokens] for c in layer.clauses]

    w = args.win
    units["cua_so_%d" % w] = [
        u[i:i + w] for u in units["ca_cha"] for i in range(max(0, len(u) - w + 1))
    ]
    # CHẨN ĐOÁN (không phải cổng): nhãn phủ 73% lượt, mệnh đề trung vị dài 5 ⇒ chỉ ~21% mệnh
    # đề có ĐỦ nhãn. Ký tự `?` làm hai mệnh đề cùng ngữ pháp trông thành hai khung khác nhau
    # ⇒ ép `chia sẻ` xuống. Dòng này đo trên nhóm ĐỦ NHÃN để tách "substrate không có" khỏi
    # "nhãn thiếu". Đây là NHÓM CON có định nghĩa rõ, không phải cổng nới ra.
    units["menh_de_du_nhan"] = [u for u in units["menh_de"]
                                if u and all(int(t) in lab for t in u)]

    out: dict[str, dict] = {}
    print(f"\n{'mức hạt':<12}{'đơn vị':>9}{'khung':>9}{'chia sẻ':>10}"
          f"{'top200':>9}{'dài TV':>8}{'tổ hợp mới':>12}")
    for name, us in units.items():
        by: dict[tuple[str, ...], list[list[int]]] = defaultdict(list)
        for u in us:
            s = skel(u)
            if len(s) >= 2:
                by[s].append(u)
        n_u = sum(len(v) for v in by.values())
        if not n_u:
            continue
        shared = {s: v for s, v in by.items() if len(v) >= 2}
        share = sum(len(v) for v in shared.values()) / n_u
        top200 = sum(c for _s, c in Counter({s: len(v) for s, v in by.items()}
                                            ).most_common(200)) / n_u

        novel: list[float] = []
        lens: list[int] = []
        import math
        for s, v in shared.items():
            slots = [len({u[i] for u in v}) for i in range(len(s))]
            lg = sum(math.log10(max(x, 1)) for x in slots)
            novel.append(lg - math.log10(len(v)))
            lens.append(len(s))
        med_novel = statistics.median(novel) if novel else 0.0
        med_len = statistics.median(lens) if lens else 0

        out[name] = {"n_don_vi": n_u, "n_khung": len(by), "n_khung_chia_se": len(shared),
                     "chia_se": share, "top200": top200,
                     "dai_khung_trung_vi": med_len, "to_hop_moi_log10": med_novel}
        print(f"{name:<12}{n_u:>9,}{len(by):>9,}{share:>10.1%}{top200:>9.1%}"
              f"{med_len:>8.0f}{med_novel:>11.2f}×")

    print(f"\ncổng: chia sẻ ≥{GATE_SHARE_MIN:.0%} · top200 ≥{GATE_TOP200_MIN:.0%} "
          f"· tổ hợp mới ≥{GATE_NOVEL_MIN:.1f} (log10)")
    for name, o in out.items():
        g1 = o["chia_se"] >= GATE_SHARE_MIN
        g2 = o["top200"] >= GATE_TOP200_MIN
        g3 = o["to_hop_moi_log10"] >= GATE_NOVEL_MIN
        o["gate_pass"] = bool(g1 and g2 and g3)
        tag = (" (TRẦN — không đọc như đáp án)" if name.startswith("cua_so")
               else " (CHẨN ĐOÁN — nhóm con, không phải cổng)"
               if name.endswith("du_nhan") else "")
        print(f"  {name:<12}(1) {'✅' if g1 else '❌'} · (2) {'✅' if g2 else '❌'} · "
              f"(3) {'✅' if g3 else '❌'} ⇒ {'ĐẠT' if o['gate_pass'] else 'TRƯỢT'}{tag}")

    (_ROOT / args.out).write_text(json.dumps(
        {"checkpoint": args.checkpoint, "ket_qua": out,
         "gate": {"share_min": GATE_SHARE_MIN, "top200_min": GATE_TOP200_MIN,
                  "novel_min": GATE_NOVEL_MIN},
         "seconds": round(time.time() - t0, 1)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    logger.log(f"→ {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
