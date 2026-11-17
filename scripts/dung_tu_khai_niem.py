"""§173 — DỰNG NÚT TỪ (khái niệm trọn vẹn) từ ranh giới Ġ.

Gộp token thành TỪ theo marker khoảng trắng Ġ (mỗi dấu câu = 1 từ). Ra word-vocab
(surface→wid, wid→span token) + chuỗi wid mỗi cha. Nền cho closure cấp-từ: khái
niệm đa-token (kẽm=k+ẽm) thành MỘT nút, hết cảnh đích là mảnh subword (§172).
"""
import sys; sys.path.insert(0,"src")
import gc; gc.disable()
import numpy as np, pickle, time
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.tokenizer import MARK_SPACE, MARK_NL, MARK_TAB, MARK_CR
T0=time.time()
st,_=load_checkpoint("checkpoints/network_v12_sent_ls.pkl"); v=st.vocab
ps=st.parents
print(f"[{time.time()-T0:.0f}s] load",flush=True)
N=len(v)
MARKS="".join((MARK_SPACE,MARK_NL,MARK_TAB,MARK_CR))
raw=[v.token_of(t) for t in range(N)]
low=[None]*N; boundary=np.zeros(N,bool); alnum=np.zeros(N,bool)
for t in range(N):
    s=raw[t]; core=s.lstrip(MARKS) or s
    low[t]=core.lower()
    boundary[t]= (s[:1] in MARKS)
    a=any(c.isalnum() for c in core)
    alnum[t]=a
    if core and not a:  # dấu câu = biên riêng
        boundary[t]=True
bnd=boundary  # local
print(f"[{time.time()-T0:.0f}s] precompute surfaces",flush=True)
# một lượt: word vocab + parent_wids
w2id={}; wid_span=[]; wid_df=[]
parent_wids=[]
nP=len(ps.parents)
for i in range(nP):
    emit=ps.parents[i].emit_ids
    wids=[]; cur=[]
    def flush(cur):
        if not cur: return
        if len(cur)==1: surf=low[cur[0]]
        else: surf="".join(low[t] for t in cur)
        surf=surf.strip()
        if not surf or not any(c.isalnum() for c in surf): return
        wid=w2id.get(surf)
        if wid is None:
            wid=len(wid_span); w2id[surf]=wid; wid_span.append(tuple(cur)); wid_df.append(0)
        wids.append(wid)
    for t in emit:
        if t>=N:
            flush(cur); cur=[]; continue
        if bnd[t] and cur:
            flush(cur); cur=[t]
        else:
            cur.append(t)
    flush(cur)
    parent_wids.append(np.asarray(wids,dtype=np.int32))
    # df
    for wd in set(wids): wid_df[wd]+=1
    if i%100000==0: print(f"[{time.time()-T0:.0f}s]  {i}/{nP}",flush=True)
print(f"[{time.time()-T0:.0f}s] distinct words={len(wid_span)}",flush=True)
pickle.dump({"w2id":w2id,"wid_span":wid_span,"wid_df":np.asarray(wid_df,np.int32)},
            open("/tmp/wvocab.pkl","wb"))
np.save("/tmp/parent_wids.npy", np.array(parent_wids,dtype=object), allow_pickle=True)
print(f"[{time.time()-T0:.0f}s] saved wvocab+parent_wids",flush=True)
for w in ["sắt","đồng","natri","kẽm","mangan","chó","khỉ"]:
    wid=w2id.get(w)
    print(f"   {w!r}: wid={wid} df={wid_df[wid] if wid is not None else '-'} span={wid_span[wid] if wid is not None else '-'}",flush=True)
print("### hết",flush=True)
