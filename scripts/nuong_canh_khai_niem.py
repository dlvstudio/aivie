"""§173 — CLOSURE LOẠI SUY CẤP-TỪ (hạt khái niệm trọn vẹn).

Chạy S=Bn·Bnᵀ, E=S·W trên NÚT TỪ (dung_tu_khai_niem.py) thay vì token. Đích c→Y
là TỪ trọn vẹn (nhôm, thiếc, cobalt) — không còn mảnh (magnesi, sulf). Chứng minh
cơ chế §158 đúng ở cấp khái niệm; chỉ cần đổi HẠT. Hub đồng nghĩa (đồng df cao) còn
nhiễu → cần tách nghĩa §168.
"""
import numpy as np, pickle, time, scipy.sparse as sp
T0=time.time()
d=pickle.load(open("/tmp/wvocab.pkl","rb"))
w2id=d["w2id"]; wid_span=d["wid_span"]; wid_df=d["wid_df"]
id2w={i:w for w,i in w2id.items()}
PW=np.load("/tmp/parent_wids.npy",allow_pickle=True)
n=len(wid_span)
print(f"[{time.time()-T0:.0f}s] words={n} parents={len(PW)}",flush=True)
# ── W = skip co-occur d=1..4 (có hướng, đếm)
rows=[];cols=[];W_win=4
for wids in PW:
    L=len(wids)
    for a in range(L):
        wa=wids[a]
        for b in range(a+1,min(a+1+W_win,L)):
            rows.append(wa);cols.append(wids[b])
rows=np.asarray(rows,np.int32);cols=np.asarray(cols,np.int32)
print(f"[{time.time()-T0:.0f}s] pairs={len(rows):,}",flush=True)
W=sp.coo_matrix((np.ones(len(rows),np.float32),(rows,cols)),shape=(n,n)).tocsr()
W.sum_duplicates()
# min_count 2
W.data[W.data<2]=0; W.eliminate_zeros()
print(f"[{time.time()-T0:.0f}s] W nnz={W.nnz:,}",flush=True)
del rows,cols
# ── S = tương tự qua ngữ cảnh chung, cắt CỘT hub, cosine (§158)
indeg=np.asarray((W>0).sum(0)).ravel()
q=np.quantile(indeg[indeg>0],0.90)
B=(W@sp.diags((indeg<=q).astype(np.float32))).tocsr(); B.eliminate_zeros(); B.data[:]=1.0
nrm=np.sqrt(np.asarray(B.sum(1)).ravel()); nrm[nrm==0]=1.0
Bn=(sp.diags((1.0/nrm).astype(np.float32))@B).tocsr()
print(f"[{time.time()-T0:.0f}s] S=Bn@BnT…",flush=True)
S=(Bn@Bn.T).tocsr(); S.setdiag(0)
S.data[S.data<0.05]=0.0; S.eliminate_zeros()
print(f"[{time.time()-T0:.0f}s] S nnz={S.nnz:,}",flush=True)
def topk_row(M,k):
    M=M.tocsr();r=[];c=[];dd=[]
    for i in range(M.shape[0]):
        s,e=M.indptr[i],M.indptr[i+1]
        if e<=s:continue
        da=M.data[s:e];ca=M.indices[s:e]
        idx=np.argsort(-da)[:k]
        r+=[i]*len(idx);c+=list(ca[idx]);dd+=list(da[idx])
    return sp.coo_matrix((dd,(r,c)),shape=M.shape).tocsr()
Sk=topk_row(S,10)
print(f"[{time.time()-T0:.0f}s] Sk nnz={Sk.nnz:,}",flush=True)
# ── E = S·W, bỏ cạnh cũ, chuẩn hoá đích, top-k
E=(Sk@W).tocsr(); E=E-E.multiply(W>0); E.eliminate_zeros()
E=(E@sp.diags((1.0/np.maximum(1.0,indeg)).astype(np.float32))).tocsr()
Ek=topk_row(E,15)
print(f"[{time.time()-T0:.0f}s] Ek (c→Y mới) nnz={Ek.nnz:,}",flush=True)
sp.save_npz("/tmp/Ek_word.npz",Ek)
sp.save_npz("/tmp/Sk_word.npz",Sk)
# ── báo cáo
def nbrs(M,w,k=10):
    wid=w2id.get(w)
    if wid is None: return f"{w}: —"
    row=M.getrow(wid).tocoo()
    o=sorted(zip(row.data,row.col),reverse=True)[:k]
    return f"{w}(df{wid_df[wid]}) → "+", ".join(f"{id2w[j]}·{v:.2f}" for v,j in o)
print("\n### S (anh em) cấp-từ:",flush=True)
for w in ["sắt","kẽm","đồng","chó","khỉ","natri","mangan"]:
    print("  ",nbrs(Sk,w),flush=True)
print("\n### E (loại suy c→Y MỚI) cấp-từ:",flush=True)
for w in ["sắt","kẽm","đồng","chó","khỉ","natri","mangan","mèo"]:
    print("  ",nbrs(Ek,w),flush=True)
print("### hết",flush=True)
