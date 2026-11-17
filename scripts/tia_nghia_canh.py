"""§168 (TK_054) — TÁCH NGHĨA nhẹ: tỉa đích REINFORCED không nhất quán CỤM.

`đồng` đa nghĩa: đích {paladi(kim loại), Gregory(tên), mẹo} trộn nghĩa. Nhưng
đích CÙNG NGHĨA LIÊN KẾT NHAU qua cạnh ANALOGY (paladi chia sẻ đích với kim loại
khác), đích lạc nghĩa đứng LẺ. Giữ cụm liên kết chặt nhất (nghĩa trội), tỉa lẻ.

Nhẹ: chỉ dùng cấu trúc cạnh ANALOGY đã có, KHÔNG dựng lại W_skip.

Nhất quán(Y1,Y2) = Y1,Y2 chia sẻ ≥1 đích ANALOGY chung, HOẶC một trỏ tới kia.
Giữ Y nếu nhất quán với ≥1 đích khác của cùng nguồn.

    PYTHONPATH=src python scripts/tia_nghia_canh.py --ra CKPT_MOI
"""
from __future__ import annotations
import argparse, gc, sys, time
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
    ap.add_argument("--ckpt", default="checkpoints/net_ls_reinforced.pkl")
    ap.add_argument("--ra", required=True)
    a = ap.parse_args()
    st, _ = load_checkpoint(a.ckpt); v = st.vocab
    AN = st.adjacency._layers[EdgeKind.ANALOGY]
    RF = st.adjacency._layers[EdgeKind.REINFORCED]
    # đích ANALOGY của mỗi từ (để tính nhất quán)
    an_dst: dict[int, set[int]] = defaultdict(set)
    for i, j, w in AN.items(): an_dst[int(i)].add(int(j))
    print(f"[{time.time()-T0:.0f}s] ANALOGY {sum(len(x) for x in an_dst.values()):,}", flush=True)
    # REINFORCED theo nguồn
    rf_src: dict[int, list] = defaultdict(list)
    for i, j, w in RF.items(): rf_src[int(i)].append((int(j), float(w)))
    def nhat_quan(y1, y2):
        # chia sẻ đích chung, hoặc một trỏ kia
        if y2 in an_dst.get(y1, ()) or y1 in an_dst.get(y2, ()): return True
        return bool(an_dst.get(y1, set()) & an_dst.get(y2, set()))
    giu = 0; bo = 0
    bo_canh = []
    for c, tgts in rf_src.items():
        ys = [y for y, _ in tgts]
        keep = set()
        for y in ys:
            if any(y2 != y and nhat_quan(y, y2) for y2 in ys):
                keep.add(y)
        for y, w in tgts:
            if y in keep: giu += 1
            else:
                bo += 1
                RF.set(int(c), int(y), 0.0)     # xoá cạnh lạc nghĩa
                bo_canh.append((c, y))
    print(f"[{time.time()-T0:.0f}s] giữ {giu:,} · tỉa {bo:,} cạnh lạc nghĩa", flush=True)
    save_checkpoint(st, a.ra)
    print(f"### lưu → {a.ra}", flush=True)
    # xác minh
    rf2: dict[int, list] = defaultdict(list)
    for i, j, w in RF.items():
        if w > 0: rf2[int(i)].append((int(j), w))
    print("\n### REINFORCED sau tỉa:")
    for w in ("sắt", "kẽm", "natri", "đồng", "chó", "lúa"):
        ids = H.resolve_surface_ids(v, w)
        if not ids: continue
        top = sorted(rf2.get(int(ids[0]), []), key=lambda t:-t[1])[:6]
        print(f"   «{w}» → {[H._surf(v,int(j)).strip() for j,_ in top]}")
    print(f"[{time.time()-T0:.0f}s] xong", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
