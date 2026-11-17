import sys; sys.path.insert(0,"src")
import gc; gc.disable()
import numpy as np, torch, torch.nn as nn
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.parent_store import _content_ids
st,_=load_checkpoint("checkpoints/net_ls_concept_conf.pkl"); v=st.vocab; ps=st.parents; E=st.embedding.matrix.astype(np.float32)
print("### load",flush=True)
# tái dựng encoder + nạp trọng số
class Encoder(nn.Module):
    def __init__(s,V,d,E0,ml,nl=2,nh=4):
        super().__init__(); s.emb=nn.Embedding(V+1,d)
        with torch.no_grad(): s.emb.weight[:V]=torch.tensor(E0)
        s.pos=nn.Embedding(ml,d); L=nn.TransformerEncoderLayer(d,nh,4*d,batch_first=True,dropout=0.0)
        s.enc=nn.TransformerEncoder(L,nl,enable_nested_tensor=False); s.norm=nn.LayerNorm(d); s.V=V
    def hidden(s,x,pm):
        p=torch.arange(x.shape[1]); return s.norm(s.enc(s.emb(x)+s.pos(p)[None],src_key_padding_mask=pm))
ML=48; m=Encoder(32002,96,E,ML); m.load_state_dict(torch.load("checkpoints/ctx_encoder.pt",map_location="cpu")); m.eval()
print("### encoder nạp",flush=True)
def ctx_of(emit, entset):
    e=[t for t in emit if 0<=t<32002][:ML]
    pos=next((k for k,t in enumerate(e) if t in entset), None)
    if pos is None: return None
    x=torch.zeros(1,ML,dtype=torch.long); x[0,:len(e)]=torch.tensor(e)
    pm=torch.arange(ML)[None]>=len(e)
    with torch.no_grad(): h=m.hidden(x,pm)[0,pos].numpy()
    n=np.linalg.norm(h); return h/n if n>0 else None
def static_of(emit, entset):
    cs=[t for t in _content_ids(list(emit),v,skip=set()) if t not in entset and 0<=t<32002]
    if not cs: return None
    mm=E[np.asarray(cs)].mean(0); n=np.linalg.norm(mm); return mm/n if n>0 else None
q="Hổ sống ở đâu?"
qc=list(dict.fromkeys(_content_ids(list(v.encode_prompt(q)),v,skip=set())))
ent=qc[0]; entset=set(ps.case_equiv_ids(ent,v))
d=ps.define_entity(v,v.token_of(ent).lstrip("ĠĊ"))
# vector định nghĩa (ngữ cảnh + tĩnh)
dvec_c=ctx_of(v.encode_prompt(d[0]),entset); dvec_s=static_of(v.encode_prompt(d[0]),entset)
idxs=ps.facts_about(tuple(qc),top_k=10,vocab=v,theme_weight=1.0,min_overlap=0.6)
print(f"«{q}»  (def: {d[0][:30]!r})",flush=True)
print(f"{'cos_NGỮCẢNH':>12} {'cos_TĨNH':>9}  câu",flush=True)
for pi in idxs:
    rec=ps.parents[pi]
    if rec.is_qa: continue
    cc=ctx_of(rec.emit_ids,entset); cs=static_of(rec.emit_ids,entset)
    a=float(cc@dvec_c) if cc is not None else -1
    b=float(cs@dvec_s) if cs is not None else -1
    print(f"{a:>12.2f} {b:>9.2f}  {v.detokenize(list(rec.emit_ids))[:46]!r}",flush=True)
print("### hết",flush=True)
