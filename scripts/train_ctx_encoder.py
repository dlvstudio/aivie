"""§190 — ENCODER NGỮ CẢNH NÔNG (Mức 2, TK_055). Self-attention 2 tầng trên
embedding tĩnh, huấn luyện masked-LM → vector-mỗi-lần-xuất-hiện. Cổng nghiệm thu:
silhouette gom cụm nghĩa của token ĐA-nghĩa (hổ) phải vượt hẳn ĐƠN-nghĩa (chó).

    PYTHONPATH=src python scripts/train_ctx_encoder.py --steps 3000
"""
import argparse, pickle, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
T0=time.time()

def kmeans_sil(X, K, seed=0):
    Xt=torch.tensor(X); 
    g=torch.Generator().manual_seed(seed)
    C=Xt[torch.randperm(len(Xt),generator=g)[:K]].clone()
    for _ in range(30):
        a=(Xt@C.T).argmax(1)
        for k in range(K):
            m=Xt[a==k]
            if len(m): c=m.mean(0); C[k]=c/(c.norm()+1e-9)
    a=(Xt@C.T).argmax(1).numpy()
    D=1-(Xt@Xt.T).numpy()
    s=[]
    for i in range(len(X)):
        same=[j for j in range(len(X)) if a[j]==a[i] and j!=i]
        if not same: continue
        ai=np.mean([D[i,j] for j in same])
        bs=[np.mean([D[i,j] for j in range(len(X)) if a[j]==k]) for k in range(K) if k!=a[i] and (a==k).any()]
        if not bs: continue
        bi=min(bs); s.append((bi-ai)/max(ai,bi,1e-9))
    return float(np.mean(s)) if s else 0.0

class Encoder(nn.Module):
    def __init__(self, V, d, E0, maxlen, nlayer=2, nhead=4):
        super().__init__()
        self.emb=nn.Embedding(V+1, d)          # +1 = MASK
        if E0 is not None:
            with torch.no_grad():
                k=min(d, E0.shape[1])
                self.emb.weight[:V, :k]=torch.tensor(E0[:, :k])  # giữ cấu trúc E ở k chiều đầu
        self.pos=nn.Embedding(maxlen, d)
        L=nn.TransformerEncoderLayer(d, nhead, dim_feedforward=4*d, batch_first=True, dropout=0.0)
        self.enc=nn.TransformerEncoder(L, nlayer, enable_nested_tensor=False)
        self.V=V; self.d=d; self.norm=nn.LayerNorm(d)
    def hidden(self, x, padmask):
        p=torch.arange(x.shape[1], device=x.device)
        h=self.emb(x)+self.pos(p)[None]
        return self.norm(self.enc(h, src_key_padding_mask=padmask))
    def logits(self, h):
        return h @ self.emb.weight[:self.V].T   # weight-tied

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clean-mask", default="")
    a=ap.parse_args()
    dev="mps" if torch.backends.mps.is_available() else "cpu"
    dat=np.load("/private/tmp/claude-501/-Users-user-Documents-learn-nn-aivie/8b631617-f26d-4163-b668-de1da8b57546/scratchpad/enc_data.npz"); X=dat["X"]; L=dat["L"]; E0=dat["E"]
    pool=np.arange(len(X))
    if a.clean_mask:
        cm=np.load(a.clean_mask); pool=np.where(cm)[0]
        print(f"[{time.time()-T0:.0f}s] CORPUS SẠCH: train trên {len(pool)}/{len(X)} câu",flush=True)
    ev=pickle.load(open("/private/tmp/claude-501/-Users-user-Documents-learn-nn-aivie/8b631617-f26d-4163-b668-de1da8b57546/scratchpad/enc_eval.pkl","rb"))
    V=E0.shape[0]; MAXLEN=X.shape[1]; MASK=V
    print(f"[{time.time()-T0:.0f}s] dev={dev} V={V} câu={len(X)} d={a.d}",flush=True)
    Xt=torch.tensor(X, dtype=torch.long); Lt=torch.tensor(L, dtype=torch.long)
    m=Encoder(V, a.d, E0, MAXLEN, a.layers, a.heads).to(dev)
    print(f"[{time.time()-T0:.0f}s] model d={a.d} layers={a.layers} heads={a.heads} init_E=partial({min(a.d,E0.shape[1])}d)",flush=True)
    opt=torch.optim.AdamW(m.parameters(), lr=a.lr)
    # eval occurrences: (sent_idx, pos) cho hổ/chó
    def occ_positions(tok_ids, cap=300):
        tset=set(tok_ids); out=[]
        for si in range(len(X)):
            row=X[si]; ln=L[si]
            for pos in range(ln):
                if int(row[pos]) in tset: out.append((si,pos)); break
        return out[:cap] if len(out)<=cap else [out[i] for i in np.random.default_rng(0).choice(len(out),cap,replace=False)]
    ev_occ={w:occ_positions(ev[w]) for w in ["hổ","chó","đồng","sắt"]}
    print(f"[{time.time()-T0:.0f}s] eval occ: "+", ".join(f"{w}={len(ev_occ[w])}" for w in ev_occ),flush=True)
    @torch.no_grad()
    def ctx_vecs(occ):
        m.eval(); vs=[]
        for i in range(0,len(occ),64):
            chunk=occ[i:i+64]
            xi=Xt[[o[0] for o in chunk]].to(dev)
            pad=(torch.arange(MAXLEN,device=dev)[None]>=Lt[[o[0] for o in chunk]].to(dev)[:,None])
            h=m.hidden(xi, pad)
            for j,(si,pos) in enumerate(chunk): vs.append(h[j,pos].float().cpu().numpy())
        m.train()
        V=np.stack(vs); V/=np.linalg.norm(V,axis=1,keepdims=True)+1e-9; return V
    def gate():
        r={}
        for w in ev_occ:
            Vv=ctx_vecs(ev_occ[w]); r[w]=np.mean([kmeans_sil(Vv,3,seed=s) for s in (0,1,2)])
        return r
    print(f"[{time.time()-T0:.0f}s] silhouette TRƯỚC train: "+", ".join(f"{w}={s:.3f}" for w,s in gate().items()),flush=True)
    # train masked-LM
    rng=np.random.default_rng(0); m.train()
    for step in range(1,a.steps+1):
        idx=pool[rng.integers(0,len(pool),a.bs)]
        x=Xt[idx].to(dev); ln=Lt[idx].to(dev)
        pad=(torch.arange(MAXLEN,device=dev)[None]>=ln[:,None])
        real=~pad
        mprob=torch.rand(x.shape,device=dev)<0.15
        mpos=mprob&real
        if mpos.sum()==0: continue
        xin=x.clone(); xin[mpos]=MASK
        h=m.hidden(xin,pad); lo=m.logits(h)
        loss=F.cross_entropy(lo[mpos], x[mpos])
        opt.zero_grad(); loss.backward(); opt.step()
        if step%1000==0:
            g=gate()
            print(f"[{time.time()-T0:.0f}s] step {step} loss {loss.item():.3f} | sil hổ={g['hổ']:.3f} chó={g['chó']:.3f} đồng={g['đồng']:.3f} sắt={g['sắt']:.3f}",flush=True)
    # DUMP cụm hổ cuối (xem xấu-hổ có tách không)
    from nn_aivie.core.checkpoint_io import load_checkpoint
    Vv=ctx_vecs(ev_occ["hổ"]); Xt2=torch.tensor(Vv)
    g=torch.Generator().manual_seed(0); C=Xt2[torch.randperm(len(Xt2),generator=g)[:3]].clone()
    for _ in range(30):
        aa=(Xt2@C.T).argmax(1)
        for k in range(3):
            mm=Xt2[aa==k]
            if len(mm): c=mm.mean(0); C[k]=c/(c.norm()+1e-9)
    aa=(Xt2@C.T).argmax(1).numpy()
    st,_=load_checkpoint("checkpoints/net_ls_concept_conf.pkl"); vv=st.vocab
    for k in range(3):
        idxk=[i for i in range(len(aa)) if aa[i]==k][:4]
        print(f"### hổ CỤM {k} ({int((aa==k).sum())}):",flush=True)
        for i in idxk:
            si=ev_occ["hổ"][i][0]
            print("   ",vv.detokenize([int(t) for t in X[si] if t>0])[:56],flush=True)
    torch.save({"cfg":{"d":a.d,"layers":a.layers,"heads":a.heads,"maxlen":MAXLEN},"state":m.state_dict()}, "checkpoints/ctx_encoder.pt")
    print(f"[{time.time()-T0:.0f}s] LƯU checkpoints/ctx_encoder.pt",flush=True)
    print("### hết",flush=True)

if __name__=="__main__":
    main()
