"""ĐƯỜNG 3 — đếm xem corpus CÓ BAO NHIÊU cấu trúc đủ hình dạng cho chuỗi ba bước.

Chín vòng trước bốc thuộc tính NGẪU NHIÊN rồi hy vọng cấu trúc có sẵn. Đây làm
ngược lại: quét corpus, đếm bộ ba đủ hình dạng. Phép đếm THUẦN DỮ LIỆU — không
gọi `hub_spread`, nên kết quả không thể bị cơ chế hỏng làm nhiễu.

Hình dạng cần (đúng ví dụ chó/mèo/khỉ của tác giả)::

    ① P  → ≥2 thành viên      (P đồng hiện với ≥2 phần tử của lớp C)
    ② thành viên → lớp C      (câu `X là C`)
    ③ C  → ≥3 thành viên      (≥1 thành viên CHƯA đồng hiện với P = «con khỉ»)

Chạy::

    PYTHONPATH=src python scripts/mine_chain3_structures.py \
        --checkpoint checkpoints/classrich.pkl --out out/chain3_structures.json
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.tokenizer import MARK_SPACE  # noqa: E402
from nn_aivie.hca.hub_spread import _FUNC  # noqa: E402

# Lượng từ đứng trước danh từ lớp: `X là MỘT loài chim` → lớp vẫn là `loài chim`.
_DET = frozenset({"một", "các", "những", "mọi", "mỗi", "cái", "con", "người"})
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:9.1f}s] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="out/chain3_structures.json")
    ap.add_argument("--min-members", type=int, default=3,
                    help="lớp phải có ≥ ngần này thành viên (bước ③)")
    ap.add_argument("--min-seeds", type=int, default=2,
                    help="thuộc tính phải chạm ≥ ngần này thành viên (bước ①)")
    ap.add_argument("--max-parents-per-member", type=int, default=300)
    ap.add_argument("--max-classes", type=int, default=0, help="0 = không giới hạn")
    ap.add_argument("--max-cover", type=float, default=0.5,
                    help="thuộc tính chạm QUÁ tỉ lệ này thành viên = hư từ, loại. "
                         "`nhất`/`tại`/`năm` chạm 65/69 ⇒ không phải thuộc tính.")
    ap.add_argument("--allow-subword", action="store_true",
                    help="giữ cả mảnh BPE (mặc định chỉ token MỞ ĐẦU TỪ)")
    args = ap.parse_args()

    log(f"nạp {args.checkpoint} …")
    state, _ = load_checkpoint(args.checkpoint)
    vocab, ps = state.vocab, state.parents
    n_vocab = len(vocab)

    def surf(t: int) -> str:
        return str(vocab.token_of(int(t))).replace(MARK_SPACE, "").strip()

    log("dựng bảng bề mặt …")
    S = [surf(i) for i in range(n_vocab)]
    LA = {i for i in range(n_vocab) if S[i] == "là"}
    DET = {i for i in range(n_vocab) if S[i].lower() in _DET}
    FUNC = {i for i in range(n_vocab) if S[i].lower() in _FUNC or len(S[i]) <= 1}
    # Token KHÔNG mở đầu từ = mảnh BPE (`éal` của `Montréal`) — không phải ý niệm.
    if not args.allow_subword:
        sub = {i for i in range(n_vocab)
               if MARK_SPACE not in str(vocab.token_of(i))}
        FUNC |= sub
        log(f"  mảnh BPE loại = {len(sub)}")
    log(f"  |là|={len(LA)} |det|={len(DET)} |func|={len(FUNC)}")

    # ── PASS 1: câu `X là C` → lớp có những thành viên nào ────────────────
    log("PASS 1 — quét câu `X là C` …")
    cls_members: dict[str, set[str]] = defaultdict(set)
    cls_examples: dict[str, list[str]] = defaultdict(list)
    n_la = 0
    parents = ps.parents
    for pi, rec in enumerate(parents):
        if pi % 5000 == 0:
            log(f"  cha {pi}/{len(parents)} · lớp={len(cls_members)}")
        seq = list(rec.emit_ids)
        for k, t in enumerate(seq):
            if t not in LA or k == 0 or k + 1 >= len(seq):
                continue
            mem = seq[k - 1]
            if mem in FUNC or mem in DET:
                continue
            j = k + 1
            while j < len(seq) and seq[j] in DET:
                j += 1
            if j >= len(seq):
                continue
            head = seq[j]
            if head in FUNC:
                continue
            n_la += 1
            cls_members[S[head]].add(S[mem])
            if len(cls_examples[S[head]]) < 3:
                cls_examples[S[head]].append(
                    " ".join(S[x] for x in seq[max(0, k - 6):k + 6]))
    log(f"  câu `X là C` = {n_la} · lớp riêng = {len(cls_members)}")

    big = {c: m for c, m in cls_members.items() if len(m) >= args.min_members}
    log(f"  lớp có ≥{args.min_members} thành viên = {len(big)}")

    # ── PASS 2: token nào xuất hiện trong cha nào ─────────────────────────
    log("PASS 2 — bảng token → cha …")
    tok_parents: dict[str, set[int]] = defaultdict(set)
    parent_toks: list[frozenset[str]] = []
    for pi, rec in enumerate(parents):
        if pi % 5000 == 0:
            log(f"  cha {pi}/{len(parents)}")
        ts = {S[t] for t in rec.emit_ids if t not in FUNC}
        parent_toks.append(frozenset(ts))
        for s in ts:
            tok_parents[s].add(pi)

    # ── PASS 3: với mỗi lớp, thuộc tính nào chạm ≥min_seeds thành viên ────
    log(f"PASS 3 — dò thuộc tính trên {len(big)} lớp …")
    hits: list[dict] = []
    items = sorted(big.items(), key=lambda kv: -len(kv[1]))
    if args.max_classes:
        items = items[: args.max_classes]
    for ci, (cls, members) in enumerate(items):
        if ci % 200 == 0:
            log(f"  lớp {ci}/{len(items)} · bộ ba tìm được={len(hits)}")
        members = set(members)
        # thuộc tính p → tập thành viên mà p đồng hiện
        prop_hit: dict[str, set[str]] = defaultdict(set)
        for m in members:
            pids = tok_parents.get(m, ())
            if len(pids) > args.max_parents_per_member:
                pids = list(pids)[: args.max_parents_per_member]
            seen: set[str] = set()
            for pi in pids:
                seen |= parent_toks[pi]
            seen.discard(m)
            seen.discard(cls)
            for p in seen:
                prop_hit[p].add(m)
        for p, touched in prop_hit.items():
            if p in members or len(touched) < args.min_seeds:
                continue
            # Thuộc tính phải CHỌN LỌC. `nghịch ngợm` chạm 2/3 loài, nhưng `tại`
            # chạm 64/69 thành phố — cái sau là hư từ, không phân biệt được gì.
            if len(touched) > args.max_cover * len(members):
                continue
            unseen = members - touched
            if not unseen:  # không có «con khỉ» ⇒ không có gì để sáng tạo
                continue
            hits.append({
                "thuoc_tinh": p,
                "lop": cls,
                "hat_giong": sorted(touched)[:8],
                "n_hat_giong": len(touched),
                "thanh_vien_moi": sorted(unseen)[:8],
                "n_moi": len(unseen),
                "n_thanh_vien": len(members),
                "vi_du_cau": cls_examples.get(cls, [])[:2],
            })

    hits.sort(key=lambda h: (-h["n_hat_giong"], -h["n_moi"]))
    out = {
        "checkpoint": args.checkpoint,
        "n_parents": len(parents),
        "n_cau_la": n_la,
        "n_lop": len(cls_members),
        f"n_lop_>={args.min_members}_thanh_vien": len(big),
        "n_bo_ba": len(hits),
        "top": hits[:200],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    log(f"✓ bộ ba đủ hình dạng = {len(hits)} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
