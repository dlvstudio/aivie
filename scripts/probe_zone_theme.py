#!/usr/bin/env python3
"""ĐO theme prior có chống lại zone NHIỀU TOKEN không — TK_045 §12.54. **Chỉ đo.**

    PYTHONPATH=src python -u scripts/probe_zone_theme.py \
        --checkpoint checkpoints/network_v12_sent.pkl \
        --log-file out/probe_zone_theme_v12.log

## Vì sao

§12.49b chốt PMI nới-zone là âm trên v12 — nhưng ghi rõ **vùng ĐÃ ĐÚNG** (`['nhà','Lý']`) mà fact
vẫn là gia phả. Nghi phạm (§12.54a): `facts_about` rút zone về MỘT `pick_primary` để tra theme
neuron, rồi nhân `cov *= 0.5 + w_theme` với `w_theme = 1/(1+vị_trí)`. Mọi câu mở đầu bằng đúng
token `Lý` được ×1.5 — chính là lớp câu gia phả/tên người mà việc nới zone định loại bỏ.

Đo mini, 10 zone, thước khách quan: bật prior làm độ phủ cụm tụt **0.66 → 0.20**, 10/10 zone tệ
đi. Script này chạy đúng phép đo ấy trên v12, nơi lớp câu `Lý <tên người>` đông hơn nhiều bậc.

## Zone lấy từ đâu — TỰ SINH, không phải cụm người viết tự chọn

§12.52c: 29 prompt tôi tự chọn cho 2,4×, 660 câu hỏi thật cho 0,2× — **dấu ngược**. Nên zone ở đây
sinh bằng chính cơ chế đang xét: lấy `head` có gidf cao, rồi `left_pmi(head)` chọn token trái gắn
chặt nhất (≥ `--pmi-min`) — đúng luật `_extend_left_by_pmi`. Không ai chọn tay cụm nào.

## Thước — KHÁCH QUAN, đếm được

Trong top-5 `facts_about(zone)`, bao nhiêu fact chứa **ĐỦ** cụm zone (cả `prev` lẫn `head`), chứ
không chỉ `head`. Không đọc tay, không chấm chủ quan — khác cổng §12.49b (đọc tay 10 mẫu).

## Cổng

  ĐẠT   `tw=0` hơn `tw=1` với biên **≥ +0.15** trung bình ⇒ theme prior thật sự chống zone nhiều
        token ở quy mô v12 ⇒ đáng viết bản `w_theme` chấm theo CẢ CỤM (§12.54d).
  TRƯỢT biên < +0.15 ⇒ hiện tượng chỉ có ở mini ⇒ chốt âm tính, giữ nguyên `facts_about`.

Cổng này KHÔNG cho phép đổi knob nào. Nó chỉ quyết định có viết `w_theme` phiên bản mới hay không.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.matrix_core import ensure_parent_bipartite, left_pmi  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--log-file", default="out/probe_zone_theme.log")
    ap.add_argument("--n-zones", type=int, default=40)
    ap.add_argument("--pmi-min", type=float, default=3.0)
    ap.add_argument("--gidf-hi", type=float, default=5.0)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--head-cap", action="store_true",
                    help="chỉ lấy head VIẾT HOA (danh từ riêng) — quần thể mà cơ chế §12.54a "
                         "tuyên bố áp dụng; vẫn TỰ SINH, không chọn tay")
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
    v, ps = state.vocab, state.parents
    log(f"[{time.time()-t0:.0f}s] ✓ {len(ps):,} cha")

    bi = ensure_parent_bipartite(state)
    log(f"[{time.time()-t0:.0f}s] ✓ bipartite")

    # ---- head ứng viên: token gidf cao, được PHÁT ra nhiều (đủ dữ liệu cho PMI) -------------
    ec = bi.E.tocsc()
    import numpy as np

    e_cnt = np.diff(ec.indptr)
    def _cap(t: int) -> bool:
        s_ = v.token_of(int(t)).lstrip("ĠĊ")
        return bool(s_) and s_[:1].isupper()

    heads = [int(t) for t in np.argsort(-e_cnt)[: args.n_zones * 200]
             if float(ps._gidf(int(t))) >= args.gidf_hi  # noqa: SLF001
             and (not args.head_cap or _cap(int(t)))]
    log(f"  {len(heads)} head ứng viên (gidf ≥ {args.gidf_hi})")

    zones: list[tuple[int, int]] = []
    for h in heads:
        if len(zones) >= args.n_zones:
            break
        tab = left_pmi(state, h)
        if not tab:
            continue
        prev, sc = max(tab.items(), key=lambda kv: kv[1])
        if sc >= args.pmi_min and prev != h:
            zones.append((int(prev), int(h)))
    log(f"[{time.time()-t0:.0f}s] {len(zones)} zone TỰ SINH qua left_pmi ≥ {args.pmi_min}")
    if len(zones) < 8:
        log("⚠️ quá ít zone ⇒ hạ --pmi-min hoặc --gidf-hi rồi chạy lại.")
        return 1

    def sur(t: int) -> str:
        return v.token_of(int(t)).lstrip("ĠĊ")

    acc = {0.0: 0.0, 1.0: 0.0}
    log("")
    log(f"{'zone':<26}{'tw=0':>8}{'tw=1':>8}")
    log("-" * 42)
    for prev, h in zones:
        row = []
        for tw in (0.0, 1.0):
            pis = ps.facts_about((prev, h), top_k=args.top_k, vocab=v, theme_weight=tw)
            if not pis:
                row.append(0.0)
                continue
            full = sum(1 for pi in pis
                       if prev in set(ps.parents[pi].emit_ids)
                       and h in set(ps.parents[pi].emit_ids))
            row.append(full / len(pis))
            acc[tw] += row[-1]
        log(f"{sur(prev)+' '+sur(h):<26}{row[0]:>8.2f}{row[1]:>8.2f}")
    n = len(zones)
    a0, a1 = acc[0.0] / n, acc[1.0] / n
    log("-" * 42)
    log(f"{'TRUNG BÌNH':<26}{a0:>8.2f}{a1:>8.2f}   (n={n} zone)")
    log(f"\nbiên (tw=0 − tw=1) = **{a0-a1:+.3f}**")
    verdict = (
        "ĐẠT — theme prior CHỐNG zone nhiều token, đáng viết w_theme theo cả cụm (§12.54d)"
        if a0 - a1 >= 0.15
        else "TRƯỢT — hiện tượng chỉ có ở mini; chốt âm tính, giữ nguyên facts_about"
    )
    log(f"⇒ {verdict}")
    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
