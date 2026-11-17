"""CHÂN LÝ v2 cho cạnh lớp — vá lỗi nền phát hiện ở TK_046 §30.

Chân lý v1 lấy «token NGAY TRƯỚC `là`» nên nhặt:
    đuôi từ ghép : «triều Hán sụp ĐỔ là…» → `đổ`;  «Thụy ĐIỂN là…» → `Điển`
    trạng từ     : «phổ biến NHẤT là…», «người Hán ĐỀU là…»
    đại từ       : «CHÚNG là những sinh vật…»
Không cạnh nào nên tồn tại — và cơ chế ĐÚNG khi không dựng chúng, nên v1 chấm
SAI đúng ở chỗ cơ chế làm ĐÚNG.

v2 sửa bằng HAI cơ chế, cả hai ĐỘC LẬP với luật `head_rule*` của `hub_spread`
(nếu dùng chính luật đó thì phép chấm thành vòng tròn):

1. **Đơn vị TỪ, không phải token** — dùng span WS-CRF đã bake (TK_034, F1
   0,9844). `sụp đổ` · `Thụy Điển` · `hy vọng` thành MỘT từ ⇒ hết đuôi ghép.
2. **Loại KẾT HỢP CỐ ĐỊNH theo tỉ lệ** — `đều`/`chính`/`cũng`/`nhất` đứng ngay
   trước `là` ở một TỈ LỆ RẤT CAO trong tổng số lần xuất hiện, còn thành viên
   lớp thật thì chỉ thỉnh thoảng. Ngưỡng theo PHÂN VỊ, không phải danh sách từ.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/build_chan_ly_v2.py \
        --checkpoint checkpoints/classrich.pkl \
        --ws-spans checkpoints/classrich.ws_spans.json
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.tokenizer import MARK_SPACE  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402

_DET = frozenset({"một", "các", "những", "mọi", "mỗi"})
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:9.1f}s] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ws-spans", required=True)
    ap.add_argument("--out", default="out/chan_ly_v2.json")
    ap.add_argument("--min-dem", type=int, default=20,
                    help="chỉ xét kết-hợp-cố-định cho từ đủ phổ biến — `Đốc giáo` "
                         "xuất hiện 1 lần và 100%% trước `là` thì tỉ lệ vô nghĩa")
    ap.add_argument("--ket-hop-q", type=float, default=0.75,
                    help="phân vị tỉ-lệ-đứng-trước-`là`; trên ngưỡng = kết hợp cố định")
    args = ap.parse_args()

    log(f"nạp {args.checkpoint} …")
    state, _ = load_checkpoint(args.checkpoint)
    v, ps = state.vocab, state.parents
    n = len(v)
    raw = [str(v.token_of(i)) for i in range(n)]
    S = [r.replace(MARK_SPACE, "").strip() for r in raw]
    BAD = {i for i in range(n) if re.match(r"^<.*>$", S[i]) or S[i].endswith(":")}
    LA = {i for i in range(n) if S[i] == "là"}
    DET = {i for i in range(n) if S[i].lower() in _DET}
    FUNC = {i for i in range(n)
            if S[i].lower() in H._FUNC or len(S[i]) <= 1
            or MARK_SPACE not in raw[i]} | BAD

    def ghep(sp) -> str:
        """Ghép mảnh BPE thành BỀ MẶT THẬT — §66.

        `S` đã bỏ MARK_SPACE nên `" ".join` biến `c`+`áo` thành `c áo`, và
        `resolve_surface_ids` không tra ngược được ⇒ ca bị bỏ LẶNG LẼ (§65.4).
        Chỉ chèn khoảng trắng khi token THẬT SỰ mở đầu một từ.
        """
        ra = ""
        for t in sp:
            ra += (" " if (MARK_SPACE in raw[int(t)] and ra) else "") + S[int(t)]
        return ra.strip()

    spans = H.load_ws_spans(args.ws_spans) or set()
    log(f"  ws_spans = {len(spans)}")
    by_first: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for sp in spans:
        by_first[sp[0]].append(sp)
    for k in by_first:
        by_first[k].sort(key=len, reverse=True)

    def tu_ket_thuc(seq: list[int], k: int) -> tuple[str, int]:
        """TỪ kết thúc tại vị trí k (quét ngược tìm span dài nhất phủ k)."""
        for start in range(max(0, k - 4), k + 1):
            for sp in by_first.get(seq[start], ()):
                if len(sp) >= 1 and start + len(sp) - 1 == k \
                        and tuple(seq[start:start + len(sp)]) == sp:
                    return ghep(sp), start
        return S[seq[k]], k

    def tu_bat_dau(seq: list[int], k: int) -> tuple[str, int]:
        """TỪ bắt đầu tại vị trí k."""
        for sp in by_first.get(seq[k], ()):
            if tuple(seq[k:k + len(sp)]) == sp:
                return ghep(sp), len(sp)
        return S[seq[k]], 1

    # PASS 1 — tần suất từ, và số lần từ đó đứng NGAY TRƯỚC `là`
    log("PASS 1 — tỉ lệ kết hợp cố định với `là` …")
    dem: dict[str, int] = defaultdict(int)
    truoc_la: dict[str, int] = defaultdict(int)
    tho: list[tuple[str, str, str]] = []
    for rec in ps.parents:
        seq = list(rec.emit_ids)
        i = 0
        while i < len(seq):
            w, ln = tu_bat_dau(seq, i)
            dem[w] += 1
            i += ln
        for k, t in enumerate(seq):
            if t not in LA or k == 0 or k + 1 >= len(seq):
                continue
            mem, _st = tu_ket_thuc(seq, k - 1)
            truoc_la[mem] += 1
            j = k + 1
            while j < len(seq) and seq[j] in DET:
                j += 1
            if j >= len(seq) or seq[j] in FUNC:
                continue
            lop, _ln = tu_bat_dau(seq, j)
            cau = " ".join(S[x] for x in seq[max(0, k - 8):k + 6])
            tho.append((mem, lop, cau))
    # Chỉ xét từ ĐỦ PHỔ BIẾN: với từ hiếm, tỉ lệ 1,0 là ngẫu nhiên chứ không
    # phải kết hợp cố định (v2 lần đầu lấy phân vị trên TẤT CẢ ⇒ ngưỡng = 1,000,
    # chỉ loại rác hiếm còn `đều`/`chủ yếu`/`có thể` lọt hết).
    ty = {w: truoc_la[w] / max(dem.get(w, 1), 1)
          for w in truoc_la if dem.get(w, 0) >= args.min_dem}
    xs = sorted(ty.values())
    cut = xs[min(len(xs) - 1, int(len(xs) * args.ket_hop_q))] if xs else 1.0
    ket_hop = {w for w, r in ty.items() if r >= cut}
    log(f"  ngưỡng tỉ lệ = {cut:.3f} · loại {len(ket_hop)} từ kết-hợp-cố-định")
    log(f"  ví dụ loại: {sorted(ket_hop, key=lambda w: -ty[w])[:12]}")

    # PASS 2 — lọc
    cls: dict[str, set[str]] = defaultdict(set)
    cau_of: dict[tuple[str, str], str] = {}
    bo = defaultdict(int)
    def hop_le(w: str) -> bool:
        if not any(ch.isalnum() for ch in w):
            return False                      # dấu câu: `,` `)` `"`
        if all(ch.isdigit() or not ch.isalnum() for ch in w):
            return False                      # số thuần: `0` `2` `8`
        return True

    for mem, lop, cau in tho:
        if not hop_le(mem) or not hop_le(lop):
            bo["dấu câu / số"] += 1
            continue
        if mem in ket_hop:
            bo["kết hợp cố định"] += 1
            continue
        if any(t in H._FUNC for t in mem.split()) and len(mem.split()) == 1:
            bo["hư từ"] += 1
            continue
        cls[lop].add(mem)
        cau_of.setdefault((mem, lop), cau)
    big = {c: m for c, m in cls.items() if len(m) >= 3}
    canh = {(a, c) for c, ms in big.items() for a in ms}
    log(f"  loại: {dict(bo)}")
    log(f"  lớp ≥3 thành viên = {len(big)} · cạnh lớp = {len(canh)}")

    log("\n=== ĐỌC TAY 12 CẠNH (bài học §30.4 số 26) ===")
    random.seed(0)
    for a, c in random.sample(sorted(canh), min(12, len(canh))):
        log(f"  «{a} là {c}»   ← …{cau_of.get((a, c), '?')}…")

    out = {"n_lop": len(big), "n_canh": len(canh),
           "n_ket_hop_loai": len(ket_hop),
           "ket_hop_mau": sorted(ket_hop, key=lambda w: -ty[w])[:40],
           "lop": {c: sorted(m) for c, m in
                   sorted(big.items(), key=lambda kv: -len(kv[1]))},
           "canh": sorted(f"{a}||{c}" for a, c in canh)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    log(f"\n  20 lớp lớn nhất:")
    for c, m in sorted(big.items(), key=lambda kv: -len(kv[1]))[:20]:
        log(f"    {c:22} {len(m):3} tv : {sorted(m)[:5]}")
    log(f"✓ → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
