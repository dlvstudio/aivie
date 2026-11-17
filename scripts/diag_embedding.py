#!/usr/bin/env python3
"""CHẨN embedding `E` — lành hay thoái hoá? TK_045 §12.90d. **Chỉ đọc, không sửa gì.**

    PYTHONPATH=src python -u scripts/diag_embedding.py \
        --checkpoint checkpoints/network_v12_sent.pkl \
        --log-file out/diag_emb_v12.log

## Vì sao

P20 (§12.89) đo trên bench §12.53: `cosE_old` = **0,004** trên v12 nhưng **0,491** trên mini,
trong khi embedding dựng từ W_skip đạt **0,135** trên chính v12 — chênh **34×** giữa hai ma trận
**cùng 96 chiều, cùng bể ứng viên, cùng phép chấm**.

Chênh cỡ đó không giải thích được bằng "96 chiều không đủ tách 37.908 ứng viên" (lý do ấy áp cho
cả hai như nhau). Nên nghi `E` trên v12 **thoái hoá** — và nếu đúng thì `cos(zone-emb, fact-emb)`
trong `multizone`, tiêu chí xếp hạng ở đúng chỗ nút thắt §12.50c, **đang chạy trên rác**.

Đây là phân biệt **bug** ‖ **giới hạn**, và hai thứ đó dẫn tới hai hành động khác hẳn.

## Đọc

  `hàng chuẩn ≈0`   token KHÔNG xuất hiện trong corpus thì đương nhiên chuẩn 0 — mini ra
                    **69,26%** mà embedding vẫn rất lành. Nên con số này MỘT MÌNH không kết
                    luận được; nó chỉ đáng ngờ khi cao BẤT THƯỜNG so với tỉ lệ token thật sự
                    dùng. Cột phán là LÁNG GIỀNG.
  `láng giềng`      `Lý` phải gần `Trần`/`Thái`/`triều`… Nếu ra token ngẫu nhiên/rác ⇒ HỎNG.
                    Đây là phép thử đọc-tay, và nó phân định chắc hơn mọi con số tổng hợp.
  `cos trung bình`  giữa cặp token ngẫu nhiên. Gần 1 ⇒ mọi vector chụm một hướng (thoái hoá
                    kiểu "sụp hạng"); gần 0 ⇒ phân tán lành.
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402

PROBE = ("ĠLý", "ĠThăng", "Ġsông", "Ġvua", "ĠViệt", "ĠTrần", "Ġnăm", "Ġđô")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--log-file", default="out/diag_emb.log")
    ap.add_argument("--topk", type=int, default=8)
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
    v = state.vocab
    ge = getattr(state, "embedding", None)
    M = getattr(ge, "matrix", None) if ge is not None else None
    if M is None:
        log("⚠️ checkpoint KHÔNG có embedding ⇒ `cos(zone,fact)` chạy bằng gì? Báo lại ngay.")
        return 1
    M = np.asarray(M, dtype=np.float64)
    log(f"[{time.time()-t0:.0f}s] ✓ E {M.shape} · vocab {len(v):,}")

    nr = np.linalg.norm(M, axis=1)
    zero = float((nr < 1e-9).mean())
    log("\n" + "=" * 70)
    log(f"hàng chuẩn ≈ 0        : {zero:.2%}  ({int((nr < 1e-9).sum()):,}/{M.shape[0]:,})")
    log(f"chuẩn  min/trung vị/max: {nr.min():.4g} / {np.median(nr):.4g} / {nr.max():.4g}")

    Mn = M / np.maximum(nr[:, None], 1e-9)
    rng = random.Random(args.seed)
    live = [i for i in range(M.shape[0]) if nr[i] >= 1e-9]
    if len(live) >= 200:
        pa = [rng.choice(live) for _ in range(2000)]
        pb = [rng.choice(live) for _ in range(2000)]
        cs = np.einsum("ij,ij->i", Mn[pa], Mn[pb])
        log(f"cos cặp NGẪU NHIÊN    : trung bình {cs.mean():+.3f} · |trung bình| "
            f"{np.abs(cs).mean():.3f}   (gần 1 ⇒ sụp hạng)")
    log("=" * 70)

    log(f"\nLÁNG GIỀNG gần nhất (top {args.topk}) — phép thử đọc-tay, phân định chắc nhất:")
    for w in PROBE:
        t = v._token_to_id.get(w)  # noqa: SLF001
        if t is None or t >= M.shape[0] or nr[t] < 1e-9:
            log(f"  {w:>10} → (không có vector)")
            continue
        sim = Mn @ Mn[t]
        sim[t] = -9.0
        top = np.argsort(-sim)[: args.topk]
        log(f"  {w:>10} → " + " · ".join(
            f"{v.token_of(int(j))}({sim[j]:.2f})" for j in top))

    log("\nHỎNG nếu: nhiều hàng chuẩn ≈0, hoặc láng giềng vô nghĩa, hoặc cos ngẫu nhiên gần 1.")
    log("LÀNH nếu: `Lý` gần `Trần`/`Thái`/`triều`… và cos ngẫu nhiên gần 0.")
    log("Chuẩn MINI (embedding lành): hàng ≈0 69.26% · cos ngẫu nhiên +0.051")
    log("  Lý → Thánh·Tông·vua·Thái·Lê | sông → chảy·Hồng·bồi·ngòi | đô → Lư·dời·Hoa·Kinh")
    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
