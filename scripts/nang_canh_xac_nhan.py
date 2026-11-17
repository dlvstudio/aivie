"""§161 (TK_054 A) — NÂNG cạnh ANALOGY được viwiki XÁC NHẬN ≥ ngưỡng thành
REINFORCED (priority 0,50, THAM GIA sinh).

§160 đo: 20,2% cạnh ANALOGY được viwiki (giữ riêng) xác nhận, và xác nhận thiên
về ca sạch. Tần suất phân biệt tri-thức-thật (sắt→silic 32×) khỏi nhiễu lọt
(đồng→SWAT 1×). Ngưỡng ≥T lần đồng hiện ⇒ NÂNG.

Cạnh chưa xác nhận GIỮ NGUYÊN ở ANALOGY 0,05 (thụ động, chờ xác nhận sau).

    PYTHONPATH=src python scripts/nang_canh_xac_nhan.py --nguong 3 --ra CKPT_MOI
"""
from __future__ import annotations
import argparse, gc, json, math, re, sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()
from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint  # noqa: E402
from nn_aivie.core.edge_kinds import EdgeKind                             # noqa: E402
from nn_aivie.hca import hub_spread as H                                  # noqa: E402
T0 = time.time()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/net_ls_analogy.pkl")
    ap.add_argument("--ra", required=True)
    ap.add_argument("--mau", type=int, default=40000)
    ap.add_argument("--nguong", type=int, default=3, help="số lần đồng hiện tối thiểu")
    ap.add_argument("--w-max", type=float, default=0.5, help="trọng số REINFORCED trần")
    a = ap.parse_args()
    st, _ = load_checkpoint(a.ckpt); v = st.vocab
    AN = st.adjacency._layers[EdgeKind.ANALOGY]
    an: dict[int, set[int]] = defaultdict(set)
    for i, j, w in AN.items(): an[int(i)].add(int(j))
    src = set(an)
    print(f"[{time.time()-T0:.0f}s] {sum(len(x) for x in an.values()):,} cạnh · {len(an):,} nguồn", flush=True)
    # đếm xác nhận qua viwiki
    xn: dict[tuple[int,int], int] = defaultdict(int)
    for k, ln in enumerate(Path("data/text_corpus/viwiki.jsonl").open(encoding="utf-8")):
        if k >= a.mau: break
        try: t = json.loads(ln)
        except json.JSONDecodeError: continue
        if not isinstance(t, str): continue
        for cau in re.split(r"[.!?\n]", t):
            s = cau.strip()
            if not s: continue
            ids = set(int(x) for x in v.encode_prompt(s))
            for c in ids & src:
                for y in an[c] & ids: xn[(c, y)] += 1
        if k % 20000 == 0 and k: print(f"[{time.time()-T0:.0f}s] {k} bài", flush=True)
    # NÂNG cạnh ≥ ngưỡng
    n_nang = 0
    for (c, y), cnt in xn.items():
        if cnt >= a.nguong:
            w = min(a.w_max, 0.15 + 0.05 * math.log1p(cnt))   # thang theo log tần suất
            st.adjacency.add(int(c), int(y), float(w), EdgeKind.REINFORCED)
            n_nang += 1
    print(f"[{time.time()-T0:.0f}s] NÂNG {n_nang:,} cạnh (xác nhận ≥{a.nguong}) → REINFORCED", flush=True)
    save_checkpoint(st, a.ra)
    print(f"### lưu → {a.ra}", flush=True)
    # xác minh
    RF = st.adjacency._layers[EdgeKind.REINFORCED]
    rf: dict[int, list] = defaultdict(list)
    for i, j, w in RF.items(): rf[int(i)].append((int(j), w))
    print("\n### cạnh NÂNG (REINFORCED) theo từ nguồn:", flush=True)
    for w in ("sắt","kẽm","natri","đồng","chó","lúa"):
        ids = H.resolve_surface_ids(v, w)
        if not ids: continue
        c = int(ids[0])
        top = sorted(rf.get(c, []), key=lambda t:-t[1])[:6]
        print(f"   «{w}» → {[(H._surf(v,int(j)).strip(), round(x,2)) for j,x in top]}")
    print(f"[{time.time()-T0:.0f}s] xong", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
