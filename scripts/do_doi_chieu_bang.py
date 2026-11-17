"""§162 (TK_054 B) — ĐO đối chiếu cạnh ANALOGY với BẢNG NHÃN LỚP (§128).

Cạnh `c→Y` là «tri thức thật» (curated) nếu bảng `nhan_lop.json` gán c VÀ Y CÙNG
MỘT LỚP: `sắt→natri` đúng vì cả hai là `kim loại`; `đồng→Gregory` sai vì Gregory
không thuộc lớp nào. Độc lập với A (viwiki), neo vào từ điển.

    PYTHONPATH=src python scripts/do_doi_chieu_bang.py
"""
from __future__ import annotations
import gc, json, sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()
from nn_aivie.core.checkpoint_io import load_checkpoint   # noqa: E402
from nn_aivie.core.edge_kinds import EdgeKind             # noqa: E402
from nn_aivie.hca import hub_spread as H                  # noqa: E402
T0 = time.time()

def main() -> int:
    st, _ = load_checkpoint("checkpoints/net_ls_analogy.pkl"); v = st.vocab
    AN = st.adjacency._layers[EdgeKind.ANALOGY]
    # bảng: bề mặt từ -> tập lớp (nhãn ghép 2 từ)
    bang = json.loads(Path("data/dict/nhan_lop.json").read_text("utf-8"))
    w2lop: dict[str, set[str]] = {}
    for w, ls in bang.items():
        s = set()
        for x in ls:
            p = x.split()
            if len(p) >= 2:
                s.add(" ".join(p[:2]))
        if s:
            w2lop[w.lower()] = s
    print(f"[{time.time()-T0:.0f}s] bảng {len(w2lop):,} từ có lớp", flush=True)

    def surf(t): return H._surf(v, int(t)).strip().lower()
    n_tong = n_cung_lop = 0
    per_src: dict[int, list] = defaultdict(list)
    for i, j, w in AN.items():
        n_tong += 1
        cs = w2lop.get(surf(i)); ys = w2lop.get(surf(j))
        ok = bool(cs and ys and (cs & ys))
        if ok:
            n_cung_lop += 1
        per_src[int(i)].append((int(j), ok))
    print(f"\n### {n_cung_lop:,}/{n_tong:,} cạnh cùng lớp bảng ({100*n_cung_lop/max(1,n_tong):.1f}%)", flush=True)
    print("\n### theo từ nguồn (sạch vs rác):")
    for w in ("sắt", "kẽm", "chì", "natri", "đồng", "chó", "lúa"):
        ids = H.resolve_surface_ids(v, w)
        if not ids or int(ids[0]) not in per_src:
            print(f"   «{w}» (không cạnh)"); continue
        c = int(ids[0])
        rows = per_src[c]
        n_ok = sum(1 for _, ok in rows if ok)
        top = [surf(j) for j, ok in rows if ok][:6]
        print(f"   «{w}»  cùng lớp {n_ok}/{len(rows)}  ·  {top}")
    print(f"[{time.time()-T0:.0f}s] xong", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
