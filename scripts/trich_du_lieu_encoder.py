"""§190 — TRÍCH DỮ LIỆU cho encoder ngữ cảnh: câu (token id) + embedding tĩnh E.

Tách khỏi checkpoint nặng → train encoder không cần nạp 450k cha.

    PYTHONPATH=src python scripts/trich_du_lieu_encoder.py
"""
import sys; sys.path.insert(0, "src")
import gc; gc.disable()
import numpy as np, time, pickle
from nn_aivie.core.checkpoint_io import load_checkpoint
T0 = time.time()
st, _ = load_checkpoint("checkpoints/net_ls_concept_conf.pkl"); v = st.vocab; ps = st.parents
E = st.embedding.matrix.astype(np.float32)
print(f"[{time.time()-T0:.0f}s] load; E {E.shape} vocab {len(v)}", flush=True)
MAXLEN = 48; N = len(ps.parents)
X = np.zeros((N, MAXLEN), np.int32); L = np.zeros(N, np.int32)
for i, rec in enumerate(ps.parents):
    e = [t for t in rec.emit_ids if 0 <= t < len(v)][:MAXLEN]
    X[i, :len(e)] = e; L[i] = len(e)
ev = {w: [int(t) for t in ps.case_equiv_ids(v.encode_prompt(w)[-1], v)]
      for w in ["hổ", "chó", "nước", "đồng", "sắt", "khỉ", "mèo"]}
np.savez("/private/tmp/claude-501/-Users-user-Documents-learn-nn-aivie/8b631617-f26d-4163-b668-de1da8b57546/scratchpad/enc_data.npz", X=X, L=L, E=E)
pickle.dump(ev, open("/private/tmp/claude-501/-Users-user-Documents-learn-nn-aivie/8b631617-f26d-4163-b668-de1da8b57546/scratchpad/enc_eval.pkl", "wb"))
print(f"[{time.time()-T0:.0f}s] saved /private/tmp/claude-501/-Users-user-Documents-learn-nn-aivie/8b631617-f26d-4163-b668-de1da8b57546/scratchpad/enc_data.npz ({N} câu) + enc_eval.pkl", flush=True)
