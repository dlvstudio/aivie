import sys; sys.path.insert(0,"src")
import gc; gc.disable()
import numpy as np, pickle, time, json, re, math, scipy.sparse as sp
from collections import defaultdict
T0=time.time()
from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint
from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.tokenizer import MARK_SPACE, MARK_NL, MARK_TAB, MARK_CR
MAU=int(sys.argv[1]) if len(sys.argv)>1 else 30000
d=pickle.load(open("/tmp/wvocab.pkl","rb"))
w2id=d["w2id"]; wid_span=d["wid_span"]; wid_df=d["wid_df"]; id2w={i:w for w,i in w2id.items()}
Ek=sp.load_npz("/tmp/Ek_word.npz").tocsr()
def bad(s): return ("<eos>" in s) or s.isdigit() or (not any(c.isalnum() for c in s))
def safe(s):  # §179 làm sạch đích: bỏ công thức/kí hiệu (chữ-số), phi-chữ-cái, <eos>
    s=(s or "").strip()
    if not s or "<eos>" in s or any(c.isdigit() for c in s): return True
    return not all(c.isalpha() or c=="-" for c in s) or len(s)<2
from collections import Counter
tc=Counter()
for c in range(Ek.shape[0]):
    r=Ek.getrow(c).tocoo()
    if r.nnz==0: continue
    for k in np.argsort(-r.data)[:8]: tc[int(r.col[k])]+=1
# candidates cấp-từ: c_wid -> set(Y_wid)
cand=defaultdict(set)
for c in range(Ek.shape[0]):
    cs=id2w.get(c,"")
    if bad(cs) or not (20<=wid_df[c]<=2000): continue  # §176: nguồn = TỪ NỘI DUNG, loại hư từ df cao
    r=Ek.getrow(c).tocoo()
    if r.nnz==0: continue
    tg=[]
    for k in np.argsort(-r.data):
        Y=int(r.col[k])
        if not (5<=wid_df[Y]<=3000) or safe(id2w.get(Y,"")) or tc[Y]>40: continue  # §179 đích SẠCH (bỏ công thức) + loại hư từ
        tg.append(Y)
        if len(tg)>=8: break
    if len(tg)>=2: cand[c]=set(tg)
cand_src=set(cand); cand_all=set(cand)|{y for s in cand.values() for y in s}
print(f"[{time.time()-T0:.0f}s] candidates: {len(cand_src)} nguồn, {sum(len(s) for s in cand.values())} cạnh",flush=True)
st,_=load_checkpoint("checkpoints/net_ls_concept.pkl"); v=st.vocab
print(f"[{time.time()-T0:.0f}s] load ckpt",flush=True)
N=len(v)
MARKS="".join((MARK_SPACE,MARK_NL,MARK_TAB,MARK_CR))
raw=[v.token_of(t) for t in range(N)]
low=[None]*N; bnd=np.zeros(N,bool); is_p=np.zeros(N,bool)
for t in range(N):
    s=raw[t]; core=s.lstrip(MARKS) or s
    low[t]=core.lower(); a=any(ch.isalnum() for ch in core)
    p= len(core)>0 and not a
    bnd[t]= (s[:1] in MARKS) or p; is_p[t]=p
def seg_wids(emit):
    out=[]; cur=[]
    def fl():
        if not cur: return
        surf=(low[cur[0]] if len(cur)==1 else "".join(low[t] for t in cur)).strip()
        if surf: 
            wid=w2id.get(surf)
            if wid is not None: out.append(wid)
    for t in emit:
        if t>=N: fl(); cur.clear(); continue
        if (bnd[t] or (cur and is_p[cur[-1]])) and cur:
            fl(); cur.clear(); cur.append(t)
        else: cur.append(t)
    fl(); return out
# duyệt viwiki
xn=defaultdict(int); ncau=0
for k,ln in enumerate(open("data/text_corpus/viwiki.jsonl",encoding="utf-8")):
    if k>=MAU: break
    try: txt=json.loads(ln)
    except: continue
    if not isinstance(txt,str): continue
    for cau in re.split(r"[.!?\n]",txt):
        cau=cau.strip()
        if not cau: continue
        ids=[int(x) for x in v.encode_prompt(cau)]
        if len(ids)<2: continue
        ncau+=1
        wset=set(seg_wids(ids))
        hs=wset & cand_src
        for c in hs:
            for y in cand[c] & wset:
                xn[(c,y)]+=1
    if k and k%10000==0: print(f"[{time.time()-T0:.0f}s] {k} bài · {ncau:,} câu · {len(xn):,} cạnh xn",flush=True)
print(f"[{time.time()-T0:.0f}s] {ncau:,} câu · {len(xn):,} cạnh XÁC NHẬN / {sum(len(s) for s in cand.values())}",flush=True)
# promote → REINFORCED
deltas={}; src_conf=defaultdict(list)
for (c,y),cnt in xn.items():
    if cnt<=0: continue
    w=min(0.5,0.15+0.05*math.log1p(cnt))
    src_toks=set(int(x) for x in wid_span[c])
    try: src_toks|=set(int(x) for x in v.encode_prompt(id2w[c]))
    except: pass
    t0=int(wid_span[y][0])
    for s_ in src_toks:
        if 0<=s_<N and s_!=t0: deltas[(s_,t0)]=w
    src_conf[id2w[c]].append((id2w[y],cnt))
st.adjacency.add_many(deltas,EdgeKind.REINFORCED)
save_checkpoint(st,"checkpoints/net_ls_concept_conf.pkl")
print(f"[{time.time()-T0:.0f}s] promote {len(deltas):,} cạnh REINFORCED → net_ls_concept_conf.pkl",flush=True)
# báo cáo concept vs stopword
print("\n### xác nhận theo nguồn (concept nên nhiều, stopword nên ít):",flush=True)
for w in ["sắt","kẽm","natri","mangan","nhôm","thiếc","chì","đồng","mèo","chó","số","học","toán","lớp","một","có"]:
    c=w2id.get(w)
    if c is None or c not in cand: print(f"  {w}: (không candidate)"); continue
    got=sorted(src_conf.get(w,[]),key=lambda t:-t[1])
    print(f"  {w:>8}: xn {len(got)}/{len(cand[c])} → {got[:6]}",flush=True)
print("### hết",flush=True)
