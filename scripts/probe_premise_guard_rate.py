#!/usr/bin/env python3
"""ĐO TỈ LỆ NỔ của cổng tiền-đề trên v12 — TK_045 §12.16 (⚠️ CHẠY M4, dev không load nổi v12).

    PYTHONPATH=src python -u scripts/probe_premise_guard_rate.py \
        --checkpoint checkpoints/network_v12_sent.pkl \
        --log-file out/premise_guard_rate.log

## Câu hỏi

`ar_matrix_premise_guard` (§12.15) DỪNG sinh khi giao các cha của `last_k` token gidf-cao cuối
prompt = rỗng. Trên mini nó tách sạch bịa/thật với prompt NGẮN, nhưng với prompt DÀI + QA thì
**3/4 prompt HỢP LỆ cũng nổ** ⇒ default đã hạ về `False`. Nghi là ảo giác của mini (mini dựng từ
`history_articles`+`subject_articles`, **không có cha QA nào**, nên `đâu`/`nào`/`ai` không bao giờ
đồng hiện với content). v12 CÓ cha QA ⇒ phải đo lại.

## Cổng để bật default True

Trên prompt HỢP LỆ (QA thật, lấy từ chính corpus đã nạp): **tỉ lệ nổ ≤ 5%**,
ĐỒNG THỜI nhóm BỊA (hoán thực thể) nổ ≥ 80%. Đạt ở `last_k` nào thì đặt
`ar_matrix_premise_last_k` = giá trị đó và `ar_matrix_premise_guard=True`.
Không đạt ở mọi `last_k` ⇒ **giữ False** và ghi số vào ticket — đừng bật mù.

Script chỉ gọi `unattested_premise` (không sinh) nên chạy nhanh sau khi load xong.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.matrix_core import unattested_premise  # noqa: E402

# Đối chứng ÂM: tiền đề hoán thực thể — mạng PHẢI im (đo mini: sinh ra năm sai, rất trôi chảy).
FABRICATED = [
    "Nhà Trần dời đô về Thăng Long năm",
    "Ngô Quyền đánh bại quân Tống năm",
    "Lê Lợi dời đô về Thăng Long năm",
    "Nhà Lý đánh bại quân Nguyên Mông năm",
    "Đinh Bộ Lĩnh đánh bại quân Nam Hán trên sông Bạch Đằng năm",
    "Trần Hưng Đạo khởi nghĩa ở Lam Sơn năm",
    "Lê Hoàn dời đô về Thăng Long năm",
    "Ngô Quyền tiêu diệt quân Nguyên trên sông Bạch Đằng năm",
]

# THƯỚC là `qa_supervised` (ĐÃ train ⇒ nội dung CÓ trong đồ thị ⇒ nổ = báo động giả thật).
# `qa_heldout` báo cáo RIÊNG: nội dung chưa train nên nổ ở đó KHÔNG phải lỗi cổng — chính là
# việc nó phải làm. Trộn hai nguồn = kết luận sai (đo mini: 100% nổ chỉ vì prompt ngoài miền).
QA_FILES = {
    "qa_supervised (ĐÃ train)": "data/text_corpus/qa_supervised.jsonl",
    "qa_heldout (CHƯA train — tham khảo)": "data/sft/qa_heldout.jsonl",
}
# THƯỚC CHÍNH: prefix cắt từ chính EMIT CỦA CHA trong checkpoint. Nối tiếp một câu mạng đã
# THUỘC mà cổng vẫn nổ = báo động giả không cãi được.
#
# Bản đầu cắt prefix từ file text rồi tách câu theo `.` — SAI: `5.000` vỡ thành "000 lính và
# voi sang giúp, tháng" ⇒ thước báo 71% nổ trên mini, trong khi 5/5 prefix verbatim sạch KHÔNG
# nổ. Thước bẩn còn nguy hơn không đo. Đọc thẳng `bi.emits` thì không có khâu parse để hỏng.
HOI, DAP = "Hỏi:", "Đáp:"


def _qa_prompts(rel: str, limit: int) -> list[str]:
    """Prompt = phần "…Hỏi: <câu hỏi> Đáp:" của QA thật (bỏ phần đáp)."""
    p = _ROOT / rel
    if not p.exists():
        return []
    out: list[str] = []
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        # qa_supervised = chuỗi JSON TRẦN; qa_heldout = object có field `text`.
        txt = rec if isinstance(rec, str) else (rec.get("text", "") if isinstance(rec, dict) else "")
        if HOI not in txt or DAP not in txt:
            continue
        out.append(txt[: txt.rindex(DAP) + len(DAP)])
    random.Random(42).shuffle(out)
    return out[:limit]


def _parent_prefixes(state, limit: int) -> list[list[int]]:
    """Prefix ~70% emit của cha — đã là id, KHÔNG mã hoá lại (khỏi lệch tokenizer)."""
    from nn_aivie.hca.matrix_core import ensure_parent_bipartite

    bi = ensure_parent_bipartite(state)
    idx = list(range(len(bi.emits)))
    random.Random(42).shuffle(idx)
    out: list[list[int]] = []
    for pi in idx:
        seq = [int(t) for t in bi.emits[pi]]
        if not (9 <= len(seq) <= 40):
            continue
        out.append(seq[: max(6, int(len(seq) * 0.7))])
        if len(out) >= limit:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_sent.pkl")
    ap.add_argument("--log-file", default="out/premise_guard_rate.log")
    ap.add_argument("--limit", type=int, default=300, help="số prompt QA thật lấy mẫu")
    ap.add_argument("--last-k", default="0,4,6,8,10", help="danh sách last_k cần quét")
    args = ap.parse_args()

    logf = (_ROOT / args.log_file).open("w", encoding="utf-8")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    txt_groups = {k: _qa_prompts(rel, args.limit) for k, rel in QA_FILES.items()}
    txt_groups["BỊA (hoán thực thể)"] = FABRICATED

    log(f"[{time.time()-t0:.0f}s] load {args.checkpoint} (gc.disable) …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v = state.vocab
    log(f"[{time.time()-t0:.0f}s] ✓ parents {len(state.parents):,}")

    groups = {"emit CỦA CHA (THƯỚC)": None, **txt_groups}
    enc = {"emit CỦA CHA (THƯỚC)": _parent_prefixes(state, args.limit)}
    enc.update({k: [list(v.encode_prompt(p)) for p in g] for k, g in txt_groups.items()})
    if not enc["emit CỦA CHA (THƯỚC)"]:
        log("⚠️ checkpoint không có emit dùng được — dừng.")
        return 1
    for k in enc:
        log(f"  {k}: {len(enc[k])} prompt")
    ks = [int(x) for x in args.last_k.split(",")]

    log("\nnhóm: " + " | ".join(groups))
    log("last_k".rjust(7) + "  (nổ / tổng = tỉ lệ) theo thứ tự nhóm trên")
    rates: dict[int, dict[str, float]] = {}
    for k in ks:
        row, cells = {}, ""
        for gname, ids_list in enc.items():
            n = sum(unattested_premise(state, ids, last_k=k) for ids in ids_list)
            r = n / max(1, len(ids_list))
            row[gname] = r
            cells += f"{n:>6}/{len(ids_list):<5}={r:6.1%}      "
        rates[k] = row
        log(f"{k:>7}{cells}")

    ruler = "emit CỦA CHA (THƯỚC)"
    fake = "BỊA (hoán thực thể)"
    qa = "qa_supervised (ĐÃ train)"
    log("\nVerdict (thước ≤5% VÀ qa_supervised ≤5% VÀ bịa ≥80%):")
    best = None
    for k in ks:
        ok = (rates[k][ruler] <= 0.05 and rates[k].get(qa, 1.0) <= 0.05
              and rates[k][fake] >= 0.80)
        log(f"  last_k={k:<3} thước={rates[k][ruler]:6.1%}  qa={rates[k].get(qa, float('nan')):6.1%}"
            f"  bịa={rates[k][fake]:6.1%}  {'✅ ĐẠT' if ok else '❌'}")
        if ok and best is None:
            best = k
    # §12.18: bản trước CHỈ xét thước+bịa nên in "ĐẶT guard=True" trong khi cột qa_supervised
    # là 100% — verdict TỰ MÂU THUẪN với quyết định đúng. qa cao = hoặc corpus đó không có
    # trong checkpoint, hoặc cổng hỏng ở đường QA; phải phân định trước, đừng bật.
    if min(rates[k].get(qa, 1.0) for k in ks) > 0.20:
        log("\n⛔ CHẶN: qa_supervised nổ >20% ở MỌI last_k ⇒ chưa được bật guard.")
        log("   Chạy `scripts/diag_premise_qa.py` để phân định: (a) corpus QA không có trong")
        log("   checkpoint  vs  (b) cổng hỏng ở đường QA. Xong rồi mới quay lại bảng này.")
        best = None
    log(f"→ {'ĐẶT ar_matrix_premise_last_k=' + str(best) + ' + guard=True' if best is not None else 'GIỮ guard=False'}")

    k0 = ks[min(2, len(ks) - 1)]
    log(f"\nVí dụ THƯỚC bị nổ tại last_k={k0} (tối đa 8) — đọc tay xem cổng nhầm KIỂU gì:")
    shown = 0
    for ids in enc[ruler]:
        if shown >= 8:
            break
        if unattested_premise(state, ids, last_k=k0):
            log(f"  ⛔ {' '.join(v.token_of(int(t)) for t in ids)[-140:]!r}")
            shown += 1
    if not shown:
        log("  (không có)")

    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    log("Dán NGUYÊN bảng + verdict vào ticket §12.16 (kể cả khi không đạt).")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
