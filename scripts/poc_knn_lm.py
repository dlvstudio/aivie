"""§TK_056 POC — kNN-LM: encoder LEFT-context h có dự đoán next-token tốt hơn
mô-hình-đếm không? Đo bits/token held-out: kNN(h) vs unigram vs bigram.

Nhẹ: chỉ cần enc_data.npz (câu) + ctx_encoder.pt (encoder). Không cần checkpoint.
"""
import sys, time
import numpy as np, torch, torch.nn as nn
SP = sys.argv[1] if len(sys.argv)>1 else "."
T0=time.time()
dat=np.load(f"{SP}/enc_data.npz"); X=dat["X"]; L=dat["L"]; E0=dat["E"]
V=E0.shape[0]; ML=X.shape[1]; print(f"[{time.time()-T0:.0f}s] {len(X)} câu, V={V}",flush=True)

# encoder (d96clean: format cũ, d=96/2/4). CAUSAL forward.
class Enc(nn.Module):
    def __init__(s,V,d,ml,nl=2,nh=4):
        super().__init__(); s.emb=nn.Embedding(V+1,d); s.pos=nn.Embedding(ml,d)
        l=nn.TransformerEncoderLayer(d,nh,4*d,batch_first=True,dropout=0.0)
        s.enc=nn.TransformerEncoder(l,nl,enable_nested_tensor=False); s.norm=nn.LayerNorm(d)
    def causal_h(s,x,lens):
        # h[i] chỉ thấy token[0..i] (causal mask) → LEFT-context
        dv=x.device; p=torch.arange(x.shape[1],device=dv); hh=s.emb(x)+s.pos(p)[None]
        cm=torch.triu(torch.ones(x.shape[1],x.shape[1],dtype=torch.bool,device=dv),1)
        pad=(torch.arange(x.shape[1],device=dv)[None]>=lens[:,None])
        return s.norm(s.enc(hh,mask=cm,src_key_padding_mask=pad))
dev="mps" if torch.backends.mps.is_available() else "cpu"
m=Enc(V,96,ML).to(dev)
obj=torch.load("checkpoints/ctx_encoder.pt",map_location="cpu")
m.load_state_dict(obj["state"] if isinstance(obj,dict) and "state" in obj else obj); m.eval()
print(f"[{time.time()-T0:.0f}s] encoder nạp, dev={dev}",flush=True)

rng=np.random.default_rng(0)
idx=rng.permutation(len(X))
ds_sent=idx[:20000]; ho_sent=idx[20000:20800]   # datastore vs held-out

@torch.no_grad()
def collect(sent_ids, cap=None):
    keys=[]; nxt=[]; prev=[]
    for i in range(0,len(sent_ids),128):
        b=sent_ids[i:i+128]
        x=torch.tensor(X[b],dtype=torch.long).to(dev); lens=torch.tensor(L[b]).to(dev)
        h=m.causal_h(x,lens).float().cpu().numpy()
        for j,si in enumerate(b):
            ln=int(L[si])
            for t in range(ln-1):
                keys.append(h[j,t]); nxt.append(int(X[si,t+1])); prev.append(int(X[si,t]))
        if cap and len(keys)>=cap: break
    K=np.asarray(keys,np.float32); K/=np.linalg.norm(K,axis=1,keepdims=True)+1e-9
    return K, np.asarray(nxt,np.int32), np.asarray(prev,np.int32)

@torch.no_grad()
def collect_static(sent_ids):
    # túi tĩnh: trung bình E của LEFT-context (như PPMI-bag TK_042)
    keys=[]; nxt=[]
    for si in sent_ids:
        ln=int(L[si])
        for t in range(ln-1):
            ctx=[int(X[si,k]) for k in range(t+1) if int(X[si,k])<V]
            if not ctx: continue
            v=E0[np.asarray(ctx)].mean(0); keys.append(v); nxt.append(int(X[si,t+1]))
    K=np.asarray(keys,np.float32); K/=np.linalg.norm(K,axis=1,keepdims=True)+1e-9
    return K, np.asarray(nxt,np.int32)

Kd,Nd,Pd=collect(ds_sent); print(f"[{time.time()-T0:.0f}s] datastore {len(Kd)} vị trí",flush=True)
Kh,Nh,Ph=collect(ho_sent); print(f"[{time.time()-T0:.0f}s] held-out {len(Kh)} vị trí",flush=True)

# baselines: unigram, bigram (từ datastore)
from collections import Counter, defaultdict
uni=Counter(Nd.tolist()); Ntot=len(Nd)
big=defaultdict(Counter)
for p,n in zip(Pd.tolist(),Nd.tolist()): big[p][n]+=1
def bits_uni(tgt): return -np.log2((uni.get(tgt,0)+1)/(Ntot+V))
def bits_big(prev,tgt):
    c=big.get(prev)
    if not c: return bits_uni(tgt)
    tot=sum(c.values()); return -np.log2((c.get(tgt,0)+0.1)/(tot+0.1*V))

# kNN-LM: cho mỗi held-out, top-k trong datastore, softmax(sim/T) → P(next)
Kd_t=torch.tensor(Kd).to(dev)   # (Nd, d)
def knn_bits(batch=256,k=64,T=0.1,lam=1.0):
    tot=0.0; n=0
    for i in range(0,len(Kh),batch):
        q=torch.tensor(Kh[i:i+batch]).to(dev)          # (b,d)
        sim=q@Kd_t.T                                    # (b,Nd)
        tk=torch.topk(sim,k,dim=1)
        for r in range(q.shape[0]):
            w=torch.softmax(tk.values[r]/T,0).cpu().numpy()
            nb=Nd[tk.indices[r].cpu().numpy()]
            p=defaultdict(float)
            for ww,nn_ in zip(w,nb): p[int(nn_)]+=float(ww)
            tgt=int(Nh[i+r]); pk=p.get(tgt,0.0)
            # nội suy với unigram (làm mịn, tránh 0)
            pu=(uni.get(tgt,0)+1)/(Ntot+V)
            pf=lam*pk+(1-lam)*pu
            if pf<=0: pf=pu*1e-3
            tot+=-np.log2(pf); n+=1
    return tot/n
print(f"\n### bits/token held-out ({len(Kh)} vị trí):",flush=True)
print(f"   unigram          : {np.mean([bits_uni(int(t)) for t in Nh]):.3f}",flush=True)
print(f"   bigram (W_seq)    : {np.mean([bits_big(int(p),int(t)) for p,t in zip(Ph,Nh)]):.3f}",flush=True)
for lam in (0.2,0.3,0.5):
    print(f"   kNN-LM(h) λ={lam}   : {knn_bits(lam=lam):.3f}",flush=True)
# SO SÁNH: kNN túi-tĩnh (TK_042: h transformer 22× vs PPMI 10×)
Ks_d,Ns_d=collect_static(ds_sent); Ks_h,Ns_h=collect_static(ho_sent)
Ksd_t=torch.tensor(Ks_d).to(dev)
def knn_bits_static(batch=256,k=64,T=0.1,lam=0.3):
    tot=0.0;n=0
    for i in range(0,len(Ks_h),batch):
        q=torch.tensor(Ks_h[i:i+batch]).to(dev); sim=q@Ksd_t.T; tk=torch.topk(sim,k,1)
        for r in range(q.shape[0]):
            w=torch.softmax(tk.values[r]/T,0).cpu().numpy(); nb=Ns_d[tk.indices[r].cpu().numpy()]
            p=defaultdict(float)
            for ww,nn_ in zip(w,nb): p[int(nn_)]+=float(ww)
            tgt=int(Ns_h[i+r]); pu=(uni.get(tgt,0)+1)/(Ntot+V); pf=lam*p.get(tgt,0.0)+(1-lam)*pu
            tot+=-np.log2(max(pf,pu*1e-3)); n+=1
    return tot/n
print(f"   kNN túi-TĨNH λ=0.3: {knn_bits_static():.3f}  (so h)",flush=True)
print("### hết",flush=True)
