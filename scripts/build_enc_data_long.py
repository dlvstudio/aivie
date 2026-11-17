"""§TK_064 — DỰNG DỮ LIỆU NGỮ-CẢNH DÀI (đoạn văn) cho causal-LM.

Vấn đề (đo ở TK_064): enc_data.npz là CÂU ĐƠN (trung vị 22 token, maxlen 48) →
causal-LM không bao giờ thấy chuỗi >1 câu → không học được mạch-lạc-tầm-xa
(sinh 1 câu tốt rồi trôi). Parent records GIỮ THỨ TỰ NGUỒN (câu i, i+1 liên tiếp
cùng tài liệu) → ĐÓNG GÓI câu-sạch liên tiếp thành chuỗi tới MAXLEN token = đoạn
văn thật để train.

Gói tham lam: cộng dồn emit_ids câu-sạch liên tiếp; vượt MAXLEN → xả 1 hàng, mở
gói mới. Gặp câu DƠ (clean_mask=False, spam truyện tu tiên) → xả gói hiện tại +
bỏ qua tới khi sạch lại (không bắc cầu qua đoạn bị loại). Lưu enc_data_long.npz
(X,L,E) — MỌI hàng đã sạch nên KHÔNG cần clean_mask lúc train.

    PYTHONPATH=src python scripts/build_enc_data_long.py [--maxlen 128]
"""
import sys, time, argparse
sys.path.insert(0, "src")
import gc; gc.disable()
import numpy as np
from nn_aivie.core.checkpoint_io import load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--maxlen", type=int, default=128)
ap.add_argument("--ckpt", default="checkpoints/net_ls_concept_conf.pkl")
ap.add_argument("--clean-mask", default="checkpoints/clean_mask.npy")
ap.add_argument("--out", default="checkpoints/enc_data_long.npz")
a = ap.parse_args()
T0 = time.time()
ML = a.maxlen

st, _ = load_checkpoint(a.ckpt)
v = st.vocab; ps = st.parents
E = st.embedding.matrix.astype(np.float32)
V = len(v); N = len(ps.parents)
try:
    clean = np.load(a.clean_mask)
except Exception:
    clean = np.ones(N, bool)
print(f"[{time.time()-T0:.0f}s] load; E {E.shape} vocab {V} câu {N} sạch {int(clean.sum())}", flush=True)

rows = []            # list các list token (mỗi cái ≤ ML)
pack = []            # gói đang cộng dồn
for i in range(N):
    if not clean[i]:
        if pack:                      # xả gói tại biên câu-dơ
            rows.append(pack); pack = []
        continue
    e = [int(t) for t in ps.parents[i].emit_ids if 0 <= int(t) < V]
    if not e:
        continue
    if len(e) > ML:                   # câu đơn quá dài → cắt, thành 1 hàng riêng
        if pack: rows.append(pack); pack = []
        rows.append(e[:ML]); continue
    if len(pack) + len(e) > ML:       # gói đầy → xả, mở gói mới
        rows.append(pack); pack = list(e)
    else:
        pack.extend(e)
    if i % 50000 == 0:
        print(f"[{time.time()-T0:.0f}s]  {i}/{N} · {len(rows)} đoạn", flush=True)
if pack:
    rows.append(pack)

n = len(rows)
X = np.zeros((n, ML), np.int32); L = np.zeros(n, np.int32)
for j, r in enumerate(rows):
    X[j, :len(r)] = r; L[j] = len(r)
np.savez(a.out, X=X, L=L, E=E)
print(f"[{time.time()-T0:.0f}s] LƯU {a.out}: {n} đoạn, ML={ML}, "
      f"độ dài trung bình {L.mean():.1f}, %đầy(≥{int(ML*0.9)}) {100*(L>=ML*0.9).mean():.1f}", flush=True)
print("### hết", flush=True)
