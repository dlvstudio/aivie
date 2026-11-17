"""nn-aivie3 · M2.5 — CHÚ-Ý HỌC-ĐƯỢC: top-down salience rẻ hơn mà đúng hơn.

Bài "phản-xạ giữa nhiễu": lưới 6×6 ô, đặt 1 MỤC-TIÊU màu c (đổi mỗi mẫu) + nhiều
vật NHIỄU màu khác ở ô ngẫu-nhiên. Hỏi: mục-tiêu ở GÓC-PHẦN-TƯ nào? (4 hành-động).
Phải ĐỊNH-VỊ đúng ô màu c giữa nhiễu → cần CHÚ-Ý theo ngữ-cảnh (c cho lúc chạy).

So hai perception-bottleneck ở nhiều NGÂN-SÁCH token m ∈ {1,2,4}:
  • uniform : m query CỐ-ĐỊNH (không biết c lúc chạy) → phải nén vị-trí MỌI màu.
  • topdown : m query ĐIỀU-BIẾN bởi target-emb(c) → "tìm đúng màu c" → định-vị bằng ít token.
Cả hai đều được target-emb(c) ở ĐẦU RA (head) — khác biệt DUY-NHẤT là bottleneck có
biết c hay không. Kỳ-vọng: topdown đạt độ-chính-xác cao ở m NHỎ (⇒ ít token ⇒ nhanh).

CHẠY:  python nn-aivie3/scripts/rmsh_m25_salience.py
"""
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src"))
from nn_aivie3.vision import VisionEncoder                               # noqa: E402

N_COLORS, G, CELL = 6, 6, 8
IMG = G * CELL                                  # 48
N_DIST = 10                                     # số vật nhiễu
_PAL = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]], dtype=torch.float32)


def dev_():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def render(b, gen, dev):
    perm = torch.argsort(torch.rand(b, G * G, generator=gen), dim=1)     # ô ngẫu-nhiên
    occ = perm[:, :1 + N_DIST]
    tgt_cell = occ[:, 0]                                                 # ô mục-tiêu
    dist_cells = occ[:, 1:]
    c = torch.randint(0, N_COLORS, (b,), generator=gen)                  # màu mục-tiêu (đổi mỗi mẫu)
    dc = torch.randint(0, N_COLORS - 1, (b, N_DIST), generator=gen)
    dc = dc + (dc >= c[:, None]).long()                                  # màu nhiễu != c
    grid = torch.full((b, G * G), -1, dtype=torch.long)
    grid.scatter_(1, tgt_cell[:, None], c[:, None])
    grid.scatter_(1, dist_cells, dc)
    pal = torch.cat([torch.zeros(1, 3), _PAL], 0)                        # 0=trống(đen), màu k→k+1
    rgb = pal[grid + 1].view(b, G, G, 3).permute(0, 3, 1, 2)            # (b,3,G,G)
    img = rgb.repeat_interleave(CELL, 2).repeat_interleave(CELL, 3)      # (b,3,48,48)
    img = img + 0.03 * torch.randn(b, 3, IMG, IMG, generator=gen)
    quad = (tgt_cell // G >= G // 2).long() * 2 + (tgt_cell % G >= G // 2).long()
    return img.to(dev), c.to(dev), quad.to(dev)


class CondResampler(nn.Module):
    def __init__(self, d, m, heads=4, cond=True, layers=2):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, m, d)); nn.init.normal_(self.q, std=0.02)
        self.cond = cond
        self.L = nn.ModuleList([nn.ModuleDict({
            "nq": nn.LayerNorm(d), "nk": nn.LayerNorm(d),
            "attn": nn.MultiheadAttention(d, heads, batch_first=True),
            "nm": nn.LayerNorm(d),
            "mlp": nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)),
        }) for _ in range(layers)])

    def forward(self, grid, cvec):                       # grid:(b,N,d) cvec:(b,d)
        q = self.q.expand(grid.shape[0], -1, -1).clone()
        if self.cond:
            q = q + cvec[:, None]                         # TOP-DOWN: query biết cần tìm màu gì
        for L in self.L:
            a, _ = L["attn"](L["nq"](q), L["nk"](grid), L["nk"](grid), need_weights=False)
            q = q + a
            q = q + L["mlp"](L["nm"](q))
        return q                                          # (b,m,d) — GIỮ vị-trí per-token


class SalienceNet(nn.Module):
    def __init__(self, m, cond, d=128, enc_layers=4):
        super().__init__()
        self.m, self.d = m, d
        self.enc = VisionEncoder(d_v=d, patch=CELL, img=IMG, heads=4, layers=enc_layers)
        self.tgt = nn.Embedding(N_COLORS, d); nn.init.normal_(self.tgt.weight, std=0.2)
        self.res = CondResampler(d, m, cond=cond)
        self.head = nn.Sequential(nn.Linear(m * d + d, d), nn.GELU(), nn.Linear(d, 4))

    def forward(self, img, c):
        grid = self.enc(img)                             # (b,36,d)
        cvec = self.tgt(c)                               # (b,d)
        tok = self.res(grid, cvec)                       # (b,m,d)
        return self.head(torch.cat([tok.reshape(tok.shape[0], -1), cvec], -1))  # FLATTEN (giữ vị-trí)


def run(m, cond, dev, steps=500, bs=128, lr=1.5e-3):
    torch.manual_seed(0)
    net = SalienceNet(m, cond).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(1)
    net.train()
    for it in range(steps):
        img, c, quad = render(bs, gen, dev)
        loss = F.cross_entropy(net(img, c), quad)
        opt.zero_grad(); loss.backward(); opt.step()
    g2 = torch.Generator().manual_seed(999); net.eval()
    img, c, quad = render(512, g2, dev)
    with torch.no_grad():
        acc = net(img, c).argmax(-1).eq(quad).float().mean().item()
    return acc


def main():
    dev = dev_()
    print(f"=== nn-aivie3 M2.5 salience · dev={dev} · nhiễu={N_DIST} · ngẫu-nhiên 25.0% ===")
    print("định-vị mục-tiêu (màu đổi mỗi mẫu) giữa nhiễu → 4 góc-phần-tư\n")
    print(f"{'ngân-sách m':>12} | {'uniform':>9} | {'topdown':>9} | lợi top-down")
    print("-" * 52)
    for m in (1, 2, 4):
        au = run(m, cond=False, dev=dev)
        at = run(m, cond=True, dev=dev)
        print(f"{m:>12} | {au:>8.1%} | {at:>8.1%} | {at-au:+.1%}")
    print("\n→ topdown đạt độ-chính-xác cao ở m NHỎ (ít token = nhanh hơn) trong khi uniform")
    print("  cần nhiều token mới bằng ⇒ 'chú-ý học-được' là ĐÒN GIẢM-TRỄ, không chỉ tăng-điểm.")


if __name__ == "__main__":
    main()
