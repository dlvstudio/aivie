"""§163 (TK_054 C) — ĐO đồng thuận anh em cho cạnh ANALOGY.

Cạnh `c→Y`: đếm số ANH EM PHÂN BIỆT của c cùng nối tới Y = (S>0)@(W>0). Nhiều
⇒ thuộc tính LỚP thật; một ⇒ trùng hợp. Đúng K≥2 đặc tả gốc, tín hiệu NỘI TẠI
đồ thị (không nguồn ngoài ⇒ không mắc lỗi độ phủ như B §162).

    PYTHONPATH=src python scripts/do_dong_thuan.py
"""
from __future__ import annotations
import gc, sys, time
import numpy as np, scipy.sparse as sp
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()
from nn_aivie.core.checkpoint_io import load_checkpoint   # noqa: E402
from nn_aivie.core.edge_kinds import EdgeKind             # noqa: E402
from nn_aivie.hca import hub_spread as H                  # noqa: E402
from nn_aivie.hca.w_skip import build_w_skip              # noqa: E402
T0 = time.time()

def _topk(M, k):
    M = M.tocsr(); r=[];c=[];d=[]
    for i in range(M.shape[0]):
        s,e = M.indptr[i], M.indptr[i+1]
        if e<=s: continue
        dd=M.data[s:e]; cc=M.indices[s:e]
        idx=np.argsort(-dd)[:k]
        r+=[i]*len(idx); c+=list(cc[idx]); d+=list(dd[idx])
    return sp.coo_matrix((d,(r,c)),shape=M.shape).tocsr()

def main() -> int:
    st,_ = load_checkpoint("checkpoints/net_ls_analogy.pkl"); v=st.vocab
    ws = build_w_skip(st, d_max=6, min_count=2)
    W = ws.mats[1]
    for d in range(2,5): W = W + ws.mats[d]
    W = W.tocsr(); W.eliminate_zeros()
    indeg = np.asarray((W>0).sum(0)).ravel()
    q = np.quantile(indeg[indeg>0], 0.90)
    B = (W @ sp.diags((indeg<=q).astype(np.float64))).tocsr(); B.eliminate_zeros(); B.data[:]=1.0
    nrm = np.sqrt(np.asarray(B.sum(1)).ravel()); nrm[nrm==0]=1.0
    Bn = (sp.diags(1.0/nrm) @ B).tocsr()
    S = (Bn@Bn.T).tocsr(); S.setdiag(0); S.data[S.data<0.05]=0.0; S.eliminate_zeros()
    Sk = _topk(S, 10)
    print(f"[{time.time()-T0:.0f}s] Sk nnz={Sk.nnz:,}", flush=True)
    # ĐỒNG THUẬN = (Sk>0) @ (W>0) — số anh em nối tới Y
    Sb = Sk.copy(); Sb.data[:]=1.0
    Wb = W.copy(); Wb.data[:]=1.0
    C = (Sb @ Wb).tocsr()          # C[c,Y] = số anh em của c có cạnh tới Y
    print(f"[{time.time()-T0:.0f}s] C (đồng thuận) nnz={C.nnz:,}", flush=True)
    # tra đồng thuận cho từng cạnh ANALOGY
    AN = st.adjacency._layers[EdgeKind.ANALOGY]
    from collections import Counter
    dist = Counter()
    per_src = defaultdict(list)
    for i,j,w in AN.items():
        i,j = int(i),int(j)
        cc = int(C[i,j])
        dist[min(cc,6)] += 1
        per_src[i].append((j,cc))
    tong = sum(dist.values())
    print(f"\n### phân bố ĐỒNG THUẬN của {tong:,} cạnh ANALOGY:")
    for k in range(1,7):
        lbl = f"{k}" if k<6 else "≥6"
        print(f"   {lbl} anh em: {dist[k]:,} ({100*dist[k]/tong:.1f}%)")
    for K in (2,3,4):
        n = sum(dist[k] for k in range(K,7))
        print(f"   ⇒ giữ ≥{K}: {n:,} ({100*n/tong:.1f}%)")
    def surf(t): return H._surf(v,int(t)).strip()
    print("\n### đồng thuận theo từ nguồn (sạch vs rác):")
    for w in ("sắt","kẽm","natri","đồng","chó","lúa"):
        ids=H.resolve_surface_ids(v,w)
        if not ids or int(ids[0]) not in per_src: print(f"   «{w}» (không cạnh)"); continue
        rows = sorted(per_src[int(ids[0])], key=lambda t:-t[1])
        print(f"   «{w}» → {[(surf(j),cc) for j,cc in rows[:6]]}")
    print(f"[{time.time()-T0:.0f}s] xong", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
