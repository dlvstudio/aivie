"""§177 — ĐO plan B: xác nhận cạnh loại suy cấp-từ qua BẢNG NHÃN-LỚP curated.

c→Y xác nhận nếu chung nhãn lớp (genus ≥2 token) trong nhan_lop.json. Bắt loại
suy CHÉO-DANH-MỤC (món ăn, hàng dệt, kí hiệu) mà plan A (đồng-hiện) mù. Phủ thấp
(23% đích có genus) nhưng ĐỘ CHÍNH XÁC cao — neo bù cho A.
"""
import numpy as np, pickle, json, scipy.sparse as sp
from collections import defaultdict
d=pickle.load(open("/tmp/wvocab.pkl","rb"))
w2id=d["w2id"]; wid_span=d["wid_span"]; wid_df=d["wid_df"]; id2w={i:w for w,i in w2id.items()}
Ek=sp.load_npz("/tmp/Ek_word.npz").tocsr()
NL=json.load(open("data/dict/nhan_lop.json"))
# nhãn genus của từ = tập phrase ≥2 token trong list (bỏ 1-gram nhiễu)
def genus(w):
    labs=NL.get(w) or NL.get(w.lower())
    if not labs: return set()
    return {p for p in labs if len(p.split())>=2}
def bad(s): return ("<eos>" in s) or s.isdigit() or (not any(c.isalnum() for c in s))
# đo: candidate c→Y chung genus?
conf=defaultdict(list); ncand=0; ncoverY=0
for c in range(Ek.shape[0]):
    cs=id2w.get(c,"")
    if bad(cs): continue
    gc=genus(cs)
    if not gc: continue
    r=Ek.getrow(c).tocoo()
    for k in np.argsort(-r.data)[:10]:
        Y=int(r.col[k]); ys=id2w.get(Y,"")
        if bad(ys): continue
        ncand+=1
        gy=genus(ys)
        if gy: ncoverY+=1
        share=gc & gy
        if share:
            conf[cs].append((ys,sorted(share,key=len)[0]))
print(f"candidate có genus nguồn: xét {ncand} cạnh; đích có genus: {ncoverY}",flush=True)
print(f"cạnh xác nhận CHUNG-LỚP: {sum(len(v) for v in conf.values())} ({len(conf)} nguồn)",flush=True)
print("\n### mẫu xác nhận plan B (chung nhãn lớp):",flush=True)
for w in ["sắt","kẽm","đồng","chì","nhôm","chó","mèo","khỉ","hổ","bò","lúa","ngô"]:
    if w in conf:
        print(f"  {w} → {[(y,l) for y,l in conf[w][:6]]}",flush=True)
    else:
        g=genus(w); print(f"  {w}: (0 xác nhận; genus={sorted(g)[:2]})",flush=True)
print("\n### TẤT CẢ cạnh plan B xác nhận (chung nhãn lớp):",flush=True)
allc=[(cs,y,l) for cs,lst in conf.items() for y,l in lst]
from collections import Counter
bylab=Counter(l for _,_,l in allc)
print("theo nhãn:",bylab.most_common(15),flush=True)
for cs,lst in sorted(conf.items()):
    print(f"  {cs} → {[(y,l) for y,l in lst]}",flush=True)
