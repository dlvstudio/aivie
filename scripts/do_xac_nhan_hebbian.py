"""§160 (TK_054 A) — ĐO xác nhận Hebbian trễ của cạnh ANALOGY bằng viwiki.

Cạnh `c→Y` (§158) là DỰ ĐOÁN: c và Y KHÔNG đồng hiện ≤4 trong corpus train.
viwiki (285 935 bài, KHÔNG trong manifest) là dữ liệu CHƯA THẤY. Nếu c và Y
đồng hiện trong câu viwiki ⇒ thực tế XÁC NHẬN dự đoán.

ĐO TRƯỚC KHI NÂNG: bao nhiêu % cạnh được xác nhận, và cạnh xác nhận có THIÊN về
ca SẠCH (kỹ thuật) hơn ca RÁC (đa nghĩa) không.

    PYTHONPATH=src python scripts/do_xac_nhan_hebbian.py --mau 40000
"""
from __future__ import annotations
import argparse, gc, json, re, sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()
from nn_aivie.core.checkpoint_io import load_checkpoint          # noqa: E402
from nn_aivie.core.edge_kinds import EdgeKind                    # noqa: E402
from nn_aivie.hca import hub_spread as H                         # noqa: E402
T0 = time.time()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/net_ls_analogy.pkl")
    ap.add_argument("--mau", type=int, default=40000, help="số bài viwiki")
    a = ap.parse_args()
    st, _ = load_checkpoint(a.ckpt); v = st.vocab
    AN = st.adjacency._layers[EdgeKind.ANALOGY]
    # dict c -> set(Y)
    an: dict[int, set[int]] = defaultdict(set)
    tong = 0
    for i, j, w in AN.items():
        an[int(i)].add(int(j)); tong += 1
    src_ids = set(an)
    print(f"[{time.time()-T0:.0f}s] {tong:,} cạnh ANALOGY · {len(an):,} nguồn", flush=True)
    # duyệt viwiki: mỗi CÂU, tập token; đếm xác nhận cặp (c,Y) đồng hiện
    xn: dict[tuple[int, int], int] = defaultdict(int)   # (c,Y) -> số câu xác nhận
    n_cau = 0
    q = Path("data/text_corpus/viwiki.jsonl")
    for k, ln in enumerate(q.open(encoding="utf-8")):
        if k >= a.mau: break
        try: t = json.loads(ln)
        except json.JSONDecodeError: continue
        if not isinstance(t, str): continue
        for cau in re.split(r"[.!?\n]", t):
            ids = [int(x) for x in v.encode_prompt(cau.strip())] if cau.strip() else []
            if len(ids) < 2: continue
            n_cau += 1
            sset = set(ids)
            hit_src = sset & src_ids
            for c in hit_src:
                for y in an[c] & sset:
                    xn[(c, y)] += 1
        if k % 10000 == 0 and k:
            print(f"[{time.time()-T0:.0f}s] {k} bài · {n_cau:,} câu · {len(xn):,} cạnh đã xác nhận", flush=True)
    n_xn = len(xn)
    print(f"\n### {n_cau:,} câu viwiki · {n_xn:,}/{tong:,} cạnh ANALOGY được xác nhận"
          f" ({100*n_xn/max(1,tong):.1f}%)", flush=True)
    # thiên lệch: ca sạch vs rác
    def surf(t): return H._surf(v, int(t)).strip()
    print("\n### cạnh XÁC NHẬN theo từ nguồn (sạch vs rác):")
    for w in ("sắt", "kẽm", "chì", "natri", "đồng", "chó", "lúa"):
        ids = H.resolve_surface_ids(v, w)
        if not ids or int(ids[0]) not in an:
            print(f"   «{w}» (không có cạnh)"); continue
        c = int(ids[0])
        co = sorted(((y, xn.get((c, y), 0)) for y in an[c]), key=lambda t: -t[1])
        n_co = sum(1 for _, k in co if k > 0)
        top = [(surf(y), k) for y, k in co[:6] if k > 0]
        print(f"   «{w}»  xác nhận {n_co}/{len(co)} cạnh  ·  top: {top}")
    print(f"[{time.time()-T0:.0f}s] xong", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
