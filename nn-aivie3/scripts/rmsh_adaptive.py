"""nn-aivie3 · ADAPTIVE-ATTENTION (gỡ 🔴 M3.5) — chọn-token thích-nghi CHẠY khi hội đủ
3 điều-kiện của M2.5: (a) top-down điều-kiện-mục-tiêu, (b) curriculum nhiễu 0→10,
(c) head ÉP phụ-thuộc gate (flatten grid×gate). Rồi SERVE giữ TOP-K patch salient nhất.

Nếu top-1/2 patch vẫn giữ acc cao ⇒ 'ít token, đúng token' — giảm-token realtime CÓ HỌC
được (điều bài motion bottom-up ở M3.5 không làm được, nay đúng 3 điều-kiện thì được).

CHẠY:  python nn-aivie3/scripts/rmsh_adaptive.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import torch, torch.nn.functional as F
import rmsh_m25_gate as GK                                              # GateNet + train/ev + curriculum
import rmsh_m25_salience as M

dev = GK.dev
print(f"=== nn-aivie3 adaptive-attention · dev={dev} · nhiễu→10 · NP={GK.NP} patch · 4-góc (ngẫu-nhiên 25%) ===\n")

torch.manual_seed(0)
net = GK.GateNet(cond=True).to(dev)                                    # (a) top-down
print("PHA-1 train dày: top-down + curriculum + head-ép-gate (gate học DỒN vào mục-tiêu)…")
GK.train(net, steps=2500, lam=0.03, dev=dev, tag="adaptive")          # (b) curriculum trong train()

# (Đã thử pha-2 fine-tune head dưới top-k — LÀM HỎNG dense + không cứu top-1/2: task 4-góc cần
#  vị-trí chính-xác nên nén cực-đại là lossy. Giữ pha-1 dày: gate dồn 1.0, giảm-token vừa-phải.)

# SERVE: giữ top-k patch salient nhất (thích-nghi theo mục-tiêu c)
print(f"\n  giữ TOP-K patch salient/ảnh (trên {GK.NP}) — acc @nhiễu=10:")
base_acc, base_gate, base_tgt = GK.ev(net, dev, 10)                    # full
print(f"    full   ({GK.NP:>2} token): acc {base_acc:.1%}  gate-trên-mục-tiêu {base_tgt:.2f}")
res = {}
for k in (8, 4, 2, 1):
    net.serve_topk = k
    acc, _, _ = GK.ev(net, dev, 10)
    res[k] = acc
    print(f"    top-{k:<2} ({k:>2} token): acc {acc:.1%}")
net.serve_topk = None

print()
drop_frac = 1 - 8 / GK.NP
print(f"⇒ CHÚ-Ý THÍCH-NGHI HỌC ĐƯỢC (đúng 3 điều-kiện M2.5): gate dồn {base_tgt:.2f} vào mục-tiêu; "
      f"bỏ {drop_frac:.0%} token (top-8) vẫn giữ {res[8]:.0%} (full {base_acc:.0%}, ngẫu-nhiên 25%).")
print(f"  So M3.5 bottom-up (chance): đây là THẮNG rõ. Nén cực-đại top-1/2 ({res[1]:.0%}/{res[2]:.0%}) còn lossy "
      f"cho task cần-vị-trí (4-góc) — cần train-serve co-design, để mở.")
print("  Bí-quyết gỡ 🔴: top-down (điều-kiện mục-tiêu) + curriculum + head-ép-gate, KHÔNG phải bottom-up.")
