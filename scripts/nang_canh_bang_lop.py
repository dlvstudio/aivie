"""import sys; sys.path.insert(0,"src")
import gc; gc.disable()
import numpy as np, pickle, json, time, scipy.sparse as sp
from collections import defaultdict
T0=time.time()
from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint
from nn_aivie.core.edge_kinds import EdgeKind
d=pickle.load(open("/tmp/wvocab.pkl","rb"))
w2id=d["w2id"]; wid_span=d["wid_span"]; id2w={i:w for w,i in w2id.items()}
Ek=sp.load_npz("/tmp/Ek_word.npz").tocsr()
NL=json.load(open("data/dict/nhan_lop.json"))
def genus(w):
    labs=NL.get(w) or NL.get(w.lower())
    return {p for p in labs if len(p.split())>=2} if labs else set()
def bad(s): return ("<eos>" in s) or s.isdigit() or (not any(c.isalnum() for c in s))
# cạnh plan B: c≠Y, chung genus ≥2token
edges=[]  # (c_wid, Y_wid, label)
for c in range(Ek.shape[0]):
    cs=id2w.get(c,"")
    if bad(cs): continue
    gc_=genus(cs)
    if not gc_: continue
    r=Ek.getrow(c).tocoo()
    for k in np.argsort(-r.data)[:10]:
        Y=int(r.col[k]); ys=id2w.get(Y,"")
        if bad(ys) or ys==cs: continue          # bỏ self-loop bề mặt
        if wid_span[Y][0]==wid_span[c][0]: continue
        if gc_ & genus(ys):
            edges.append((c,Y,sorted(gc_&genus(ys),key=len)[0]))
print(f"[{time.time()-T0:.0f}s] plan B: {len(edges)} cạnh (bỏ self-loop)",flush=True)
st,_=load_checkpoint("checkpoints/net_ls_concept_conf.pkl"); v=st.vocab
N=len(v)
deltas={}
for c,Y,lab in edges:
    src=set(int(x) for x in wid_span[c])
    try: src|=set(int(x) for x in v.encode_prompt(id2w[c]))
    except: pass
    t0=int(wid_span[Y][0])
    for s_ in src:
        if 0<=s_<N and s_!=t0: deltas[(s_,t0)]=0.35   # curated = tin cao
st.adjacency.add_many(deltas,EdgeKind.REINFORCED)
save_checkpoint(st,"checkpoints/net_ls_concept_conf.pkl")
print(f"[{time.time()-T0:.0f}s] union {len(deltas)} cạnh token vào REINFORCED (A∪B)",flush=True)
print("### cạnh plan B mẫu:",flush=True)
seen=set()
for c,Y,lab in edges:
    key=id2w[c]
    if key in seen: continue
    seen.add(key)
    if len(seen)<=18: print(f"   {id2w[c]} →[{lab}] {id2w[Y]}",flush=True)
print("### hết",flush=True)
"""
import sys; sys.path.insert(0,"src")
import gc; gc.disable()
import numpy as np, pickle, json, time, scipy.sparse as sp
from collections import defaultdict
T0=time.time()
from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint
from nn_aivie.core.edge_kinds import EdgeKind
d=pickle.load(open("/tmp/wvocab.pkl","rb"))
w2id=d["w2id"]; wid_span=d["wid_span"]; id2w={i:w for w,i in w2id.items()}
Ek=sp.load_npz("/tmp/Ek_word.npz").tocsr()
NL=json.load(open("data/dict/nhan_lop.json"))
def genus(w):
    labs=NL.get(w) or NL.get(w.lower())
    return {p for p in labs if len(p.split())>=2} if labs else set()
def bad(s): return ("<eos>" in s) or s.isdigit() or (not any(c.isalnum() for c in s))
def safe(s):
    s=(s or "").strip()
    if not s or "<eos>" in s or any(c.isdigit() for c in s): return True
    return not all(c.isalpha() or c=="-" for c in s) or len(s)<2
# cạnh plan B: c≠Y, chung genus ≥2token
edges=[]  # (c_wid, Y_wid, label)
for c in range(Ek.shape[0]):
    cs=id2w.get(c,"")
    if bad(cs): continue
    gc_=genus(cs)
    if not gc_: continue
    r=Ek.getrow(c).tocoo()
    for k in np.argsort(-r.data)[:10]:
        Y=int(r.col[k]); ys=id2w.get(Y,"")
        if bad(ys) or safe(ys) or ys==cs: continue          # bỏ self-loop bề mặt
        if wid_span[Y][0]==wid_span[c][0]: continue
        if gc_ & genus(ys):
            edges.append((c,Y,sorted(gc_&genus(ys),key=len)[0]))
print(f"[{time.time()-T0:.0f}s] plan B: {len(edges)} cạnh (bỏ self-loop)",flush=True)
st,_=load_checkpoint("checkpoints/net_ls_concept_conf.pkl"); v=st.vocab
N=len(v)
deltas={}
for c,Y,lab in edges:
    src=set(int(x) for x in wid_span[c])
    try: src|=set(int(x) for x in v.encode_prompt(id2w[c]))
    except: pass
    t0=int(wid_span[Y][0])
    for s_ in src:
        if 0<=s_<N and s_!=t0: deltas[(s_,t0)]=0.35   # curated = tin cao
st.adjacency.add_many(deltas,EdgeKind.REINFORCED)
save_checkpoint(st,"checkpoints/net_ls_concept_conf.pkl")
print(f"[{time.time()-T0:.0f}s] union {len(deltas)} cạnh token vào REINFORCED (A∪B)",flush=True)
print("### cạnh plan B mẫu:",flush=True)
seen=set()
for c,Y,lab in edges:
    key=id2w[c]
    if key in seen: continue
    seen.add(key)
    if len(seen)<=18: print(f"   {id2w[c]} →[{lab}] {id2w[Y]}",flush=True)
print("### hết",flush=True)
