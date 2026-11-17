"""§TK_059 — DỰNG LẠI dữ liệu train encoder vào checkpoints/ (BỀN, scratchpad hay
bị xoá). Gộp trich_du_lieu_encoder + loc_tu_tien: 1 lần load checkpoint →
enc_data.npz (X,L,E) + clean_mask.npy.

    PYTHONPATH=src python scripts/rebuild_enc_data.py
"""
import sys, time
sys.path.insert(0, "src")
import gc; gc.disable()
import numpy as np
from nn_aivie.core.checkpoint_io import load_checkpoint
T0 = time.time()
MARKERS = [
    "tu tiên","tu sĩ","tu vi","tu luyện","cảnh giới","linh khí","linh lực",
    "linh thạch","công pháp","kim đan","nguyên anh","trúc cơ","luyện khí",
    "hóa thần","độ kiếp","pháp bảo","yêu thú","yêu hoàng","đại yêu","ma tu",
    "kiếm khí","đạo hữu","sư huynh","sư đệ","sư tổ","chưởng môn","tông môn",
    "thần thức","đan điền","tiên nhân","đạo nhân","thượng cổ","hồng hoang",
    "nghịch thiên","chí tôn","yêu tộc","ma tộc","thần tộc","huyết mạch",
    "hắn","nàng","ngươi","muội","ca ca","đệ tử","sư phụ","lão phu","bổn tọa",
    "tại hạ","các hạ","đại nhân","thiếu gia","tiểu thư","công tử","lão nhân gia",
]
st, _ = load_checkpoint("checkpoints/net_ls_concept_conf.pkl"); v = st.vocab; ps = st.parents
E = st.embedding.matrix.astype(np.float32)
MAXLEN = 48; N = len(ps.parents)
print(f"[{time.time()-T0:.0f}s] load; E {E.shape} vocab {len(v)} câu {N}", flush=True)
X = np.zeros((N, MAXLEN), np.int32); L = np.zeros(N, np.int32)
for i, rec in enumerate(ps.parents):
    e = [t for t in rec.emit_ids if 0 <= t < len(v)][:MAXLEN]
    X[i, :len(e)] = e; L[i] = len(e)
np.savez("checkpoints/enc_data.npz", X=X, L=L, E=E)
print(f"[{time.time()-T0:.0f}s] LƯU checkpoints/enc_data.npz", flush=True)
# clean_mask
mark_ids = []
for m in MARKERS:
    ids = tuple(int(t) for t in v.encode_prompt(m))
    if ids: mark_ids.append(tuple(v.fold_group_id(t) for t in ids))
def has_marker(emit):
    fg = [v.fold_group_id(int(t)) for t in emit]; n = 0
    for mi in mark_ids:
        Lm = len(mi)
        for i in range(len(fg) - Lm + 1):
            if tuple(fg[i:i+Lm]) == mi: n += 1; break
    return n
clean = np.ones(N, bool); nflag = 0
for i, rec in enumerate(ps.parents):
    if has_marker(rec.emit_ids) >= 1: clean[i] = False; nflag += 1
    if i % 100000 == 0: print(f"[{time.time()-T0:.0f}s]  clean {i}/{N} · flagged {nflag}", flush=True)
np.save("checkpoints/clean_mask.npy", clean)
print(f"[{time.time()-T0:.0f}s] LƯU checkpoints/clean_mask.npy — sạch {int(clean.sum())}/{N} ({100*nflag/N:.1f}% loại)", flush=True)
print("### hết", flush=True)
