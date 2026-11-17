"""nn-aivie3 · M2 VPROBE — mắt (VFE) có trích được MÀU không?

Linear-probe: VFE(frame) → mean k token → Linear → màu. Train chung ~200 bước.
Nếu acc thấp ⇒ VFE nuốt mất màu (thủ-phạm làm M2 smoke kẹt ngẫu-nhiên).
So 3 biến-thể để khoanh nguyên-nhân:
  base   : VFE hiện-tại (LayerNorm khắp nơi).
  no_ln  : bỏ LayerNorm cuối encoder (giả-thuyết LN giặt thành-phần DC màu).
  meanstd: pool mean+std thay mean (giữ magnitude màu).

Chạy:  python nn-aivie3/scripts/rmsh_m2_vprobe.py
"""
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src"))
from nn_aivie3.vision import VisionFrontEnd                              # noqa: E402

N_COLORS, IMG = 6, 32
_PALETTE = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]], dtype=torch.float32)[:N_COLORS]


def dev_():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def frames(b, gen):
    c = torch.randint(0, N_COLORS, (b,), generator=gen)
    base = _PALETTE[c][:, :, None, None].expand(b, 3, IMG, IMG).clone()
    return (base + 0.05 * torch.randn(b, 3, IMG, IMG, generator=gen)), c


def probe(mode, dev, steps=250, bs=128, lr=2e-3):
    torch.manual_seed(0)
    vfe = VisionFrontEnd(d_model=128, d_v=128, n_query=4, patch=16, img=IMG, heads=4).to(dev)
    if mode == "no_ln":
        vfe.encoder.norm = nn.Identity().to(dev)
    feat_dim = 128 * (2 if mode == "meanstd" else 1)
    head = nn.Linear(feat_dim, N_COLORS).to(dev)
    opt = torch.optim.AdamW(list(vfe.parameters()) + list(head.parameters()), lr=lr)
    gen = torch.Generator().manual_seed(1)
    t0 = time.time()
    for it in range(steps):
        img, c = frames(bs, gen); img, c = img.to(dev), c.to(dev)
        tok = vfe(img)                                   # (b,k,d)
        feat = torch.cat([tok.mean(1), tok.std(1)], -1) if mode == "meanstd" else tok.mean(1)
        loss = F.cross_entropy(head(feat), c)
        opt.zero_grad(); loss.backward(); opt.step()
    g2 = torch.Generator().manual_seed(999)
    img, c = frames(512, g2); img, c = img.to(dev), c.to(dev)
    tok = vfe(img)
    feat = torch.cat([tok.mean(1), tok.std(1)], -1) if mode == "meanstd" else tok.mean(1)
    acc = (head(feat).argmax(-1) == c).float().mean().item()
    print(f"  {mode:8s}: loss {loss.item():.3f} · acc {acc:.1%} · {time.time()-t0:.0f}s")
    return acc


def main():
    dev = dev_()
    print(f"=== M2 vprobe · mắt trích màu? · dev={dev} · ngẫu-nhiên {1/N_COLORS:.1%} ===")
    r = {m: probe(m, dev) for m in ("base", "no_ln", "meanstd")}
    print()
    best = max(r, key=r.get)
    print(f"→ tốt nhất: {best}={r[best]:.1%}")
    if r["base"] > 0.9:
        print("KẾT: mắt vốn trích màu ĐƯỢC khi có head riêng → lỗi M2 là tín-hiệu màu YẾU "
              "so với temp/mod-emb trong luồng chung (cần cân lại tỉ-lệ / bơm màu).")
    elif r["no_ln"] > 0.9 or r["meanstd"] > 0.9:
        print(f"KẾT: sửa được bằng '{best}' → LayerNorm/mean-pool đang giặt màu.")
    else:
        print("KẾT: mắt KHÔNG trích nổi 6 màu đặc → sửa kiến-trúc VFE (skip màu thô / bỏ LN sớm).")


if __name__ == "__main__":
    main()
