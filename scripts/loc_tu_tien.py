"""§193 — LỌC CORPUS TU-TIÊN. Nhiễu truyện tiên hiệp lấn occurrence (Bạch Hổ,
Hắc Hổ Kiếm…), dìm nghĩa sạch (§190/§191). Dò bằng MARKER MẠNH gần-như-chỉ-
tu-tiên; câu chứa ≥ngưỡng marker → gắn cờ, LOẠI khỏi train encoder. Marker ở
BAKE (như từ điển/pos_map) — không phải luật runtime.

    PYTHONPATH=src python scripts/loc_tu_tien.py --ra /private/tmp/claude-501/-Users-user-Documents-learn-nn-aivie/8b631617-f26d-4163-b668-de1da8b57546/scratchpad/clean_mask.npy
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
import gc; gc.disable()
from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.parent_store import _content_ids
T0=time.time()
MARKERS = [
    "tu tiên","tu sĩ","tu vi","tu luyện","cảnh giới","linh khí","linh lực",
    "linh thạch","công pháp","kim đan","nguyên anh","trúc cơ","luyện khí",
    "hóa thần","độ kiếp","pháp bảo","yêu thú","yêu hoàng","đại yêu","ma tu",
    "kiếm khí","đạo hữu","sư huynh","sư đệ","sư tổ","chưởng môn","tông môn",
    "thần thức","đan điền","tiên nhân","đạo nhân","thượng cổ","hồng hoang",
    "nghịch thiên","chí tôn","yêu tộc","ma tộc","thần tộc","huyết mạch",
    # xưng hô/đại từ tiểu thuyết (hiếm trong văn factual)
    "hắn","nàng","ngươi","muội","ca ca","đệ tử","sư phụ","lão phu","bổn tọa",
    "tại hạ","các hạ","đại nhân","thiếu gia","tiểu thư","công tử","lão nhân gia",
]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/net_ls_concept_conf.pkl")
    ap.add_argument("--ra", default="/private/tmp/claude-501/-Users-user-Documents-learn-nn-aivie/8b631617-f26d-4163-b668-de1da8b57546/scratchpad/clean_mask.npy")
    ap.add_argument("--thresh", type=int, default=1, help="≥ n marker → tu-tiên")
    a=ap.parse_args()
    st,_=load_checkpoint(a.ckpt); v=st.vocab; ps=st.parents
    print(f"[{time.time()-T0:.0f}s] load; {len(ps.parents)} câu",flush=True)
    # token-id của mỗi marker (chuỗi con); dò theo fold để bắt biến thể
    mark_ids=[]
    for m in MARKERS:
        ids=tuple(int(t) for t in v.encode_prompt(m))
        if ids: mark_ids.append(tuple(v.fold_group_id(t) for t in ids))
    def has_marker(emit):
        fg=[v.fold_group_id(int(t)) for t in emit]
        n=0
        for mi in mark_ids:
            L=len(mi)
            for i in range(len(fg)-L+1):
                if tuple(fg[i:i+L])==mi: n+=1; break
        return n
    N=len(ps.parents); clean=np.ones(N,bool); nflag=0
    for i,rec in enumerate(ps.parents):
        if has_marker(rec.emit_ids)>=a.thresh:
            clean[i]=False; nflag+=1
        if i%100000==0: print(f"[{time.time()-T0:.0f}s]  {i}/{N} · flagged {nflag}",flush=True)
    np.save(a.ra, clean)
    print(f"[{time.time()-T0:.0f}s] tu-tiên: {nflag} ({100*nflag/N:.1f}%) → LOẠI. Còn {int(clean.sum())} câu sạch.",flush=True)
    # kiểm occurrence hổ: bao nhiêu tu-tiên
    ent=set(ps.case_equiv_ids(v.encode_prompt("hổ")[-1],v))
    ho=[i for i in range(N) if any(int(t) in ent for t in ps.parents[i].emit_ids)]
    ho_dirty=[i for i in ho if not clean[i]]
    print(f"[{time.time()-T0:.0f}s] hổ: {len(ho)} câu, tu-tiên {len(ho_dirty)} ({100*len(ho_dirty)/max(1,len(ho)):.0f}%)",flush=True)
    print("### mẫu hổ tu-tiên (loại):",flush=True)
    for i in ho_dirty[:5]: print("   ",v.detokenize([int(t) for t in ps.parents[i].emit_ids if t>0])[:56],flush=True)
    print("### mẫu hổ SẠCH (giữ):",flush=True)
    for i in [x for x in ho if clean[x]][:5]: print("   ",v.detokenize([int(t) for t in ps.parents[i].emit_ids if t>0])[:56],flush=True)
    print("### hết",flush=True)

if __name__=="__main__": main()
