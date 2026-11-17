"""M2.5 probe — resampler output có BIẾN theo ảnh không? (loss đứng im ln4 ⇒ nghi tok hằng-số)"""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import rmsh_m25_salience as M                                           # noqa: E402
M.N_DIST = 3
dev = M.dev_(); torch.manual_seed(0)
net = M.SalienceNet(m=4, cond=True, enc_layers=2).to(dev).eval()
gen = torch.Generator().manual_seed(5)
img, c, quad = M.render(8, gen, dev)
with torch.no_grad():
    grid = net.enc(img)                         # (8,36,d)
    cvec = net.tgt(c)                           # (8,d)
    tok = net.res(grid, cvec)                   # (8,m,d)
    out = net(img, c)                           # (8,4)
print(f"grid.std qua-batch (ảnh khác nhau): {grid.std(0).mean().item():.4f}  (nên >0)")
print(f"tok.std  qua-batch (resampler out): {tok.std(0).mean().item():.4f}  (=0 ⇒ HẰNG-SỐ, bug)")
print(f"out.std  qua-batch (logits):        {out.std(0).mean().item():.4f}")
# grid có biến theo VỊ-TRÍ không? (token khác nhau trong 1 ảnh)
print(f"grid.std qua-token (trong 1 ảnh):   {grid[0].std(0).mean().item():.4f}")
# thử: đổi grid ngẫu-nhiên xem tok có đổi không (resampler có ĐỌC grid không)
with torch.no_grad():
    tok2 = net.res(torch.randn_like(grid), cvec)
print(f"|tok(grid) - tok(grid_ngẫu)|:        {(tok-tok2).abs().mean().item():.4f}  (=0 ⇒ resampler PHỚT grid)")
