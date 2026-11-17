import sys; sys.path.insert(0,"src")
import gc; gc.disable()
import numpy as np
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.parent_store import _content_ids
st,_=load_checkpoint("checkpoints/net_ls_concept_conf.pkl"); v=st.vocab
ps=st.parents; E=st.embedding.matrix
print("### load",flush=True)
def ctx_vec(ids, ent_ids):
    cs=[t for t in _content_ids(list(ids),v,skip=set()) if t not in ent_ids]
    if len(cs)<1: return None
    m=E[np.array(cs)].mean(0); n=np.linalg.norm(m)
    return m/n if n>0 else None
def analyze(q):
    qc=list(dict.fromkeys(_content_ids(list(v.encode_prompt(q)),v,skip=set())))
    ent=qc[0]; ent_ids=set(ps.case_equiv_ids(ent,v))
    d=ps.define_entity(v, v.token_of(ent).lstrip("ĠĊ"))
    if not d: print(f"«{q}» (no def)"); return
    dvec=ctx_vec(v.encode_prompt(d[0]), ent_ids)
    idxs=ps.facts_about(tuple(qc),top_k=8,vocab=v,theme_weight=1.0,min_overlap=0.6)
    print(f"\n### «{q}»  (def: {d[0][:35]!r})",flush=True)
    for pi in idxs:
        rec=ps.parents[pi]
        if rec.is_qa: continue
        fv=ctx_vec(rec.emit_ids, ent_ids)
        cos=float(fv@dvec) if fv is not None else -1
        print(f"   cos={cos:+.2f} {v.detokenize(list(rec.emit_ids))[:52]!r}",flush=True)
for q in ["Hổ sống ở đâu?","Chó ăn gì?","Nước sôi ở nhiệt độ nào?","Sắt được dùng để làm gì?"]:
    analyze(q)
print("### hết",flush=True)
