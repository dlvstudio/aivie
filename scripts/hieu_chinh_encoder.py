import sys; sys.path.insert(0,"src")
import gc; gc.disable()
import numpy as np
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.parent_store import _content_ids
from nn_aivie.hca.ctx_encoder import load_ctx_encoder
st,_=load_checkpoint("checkpoints/net_ls_concept_conf.pkl"); v=st.vocab; ps=st.parents
enc=load_ctx_encoder(); print("### encoder:", "OK" if enc else "None",flush=True)
Qs=["Hổ sống ở đâu?","Chó ăn gì?","Nước sôi ở nhiệt độ nào?","Sắt được dùng để làm gì?",
    "Đồng dẫn điện không?","Lúa trồng ở đâu?","Khỉ ăn gì?"]
for q in Qs:
    qc=list(dict.fromkeys(_content_ids(list(v.encode_prompt(q)),v,skip=set())))
    ent=qc[0]; entset=set(ps.case_equiv_ids(ent,v))
    pair=(len(qc)>=2 and (qc[0],qc[1]) in ps._phrase_lookup)
    d=ps.define_entity(v,v.token_of(ent).lstrip("ĠĊ"))
    if pair or not d: print(f"«{q}» (gate cụm ghép / no def) → giữ aspect-top"); continue
    dvec=enc.vec(v.encode_prompt(d[0]),entset)
    idxs=ps.facts_about(tuple(qc),top_k=10,vocab=v,theme_weight=1.0,min_overlap=0.6)
    cands=[]
    for pi in idxs:
        rec=ps.parents[pi]
        if rec.is_qa: continue
        txt=v.detokenize(list(rec.emit_ids))
        if txt.rstrip().endswith("?"): continue
        cv=enc.vec(rec.emit_ids,entset)
        cos=float(cv@dvec) if (cv is not None and dvec is not None) else None
        cands.append((txt,pi,cos))
    if not cands: print(f"«{q}» rỗng"); continue
    top=cands[0]
    best=max((c for c in cands if c[2] is not None), key=lambda c:c[2], default=top)
    print(f"«{q}»",flush=True)
    print(f"   aspect-top cos={top[2]} :: {top[0][:44]!r}",flush=True)
    print(f"   max-cos    cos={round(best[2],2) if best[2] else None} :: {best[0][:44]!r}",flush=True)
print("### hết",flush=True)
