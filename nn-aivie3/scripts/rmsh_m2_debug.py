"""nn-aivie3 · M2 DEBUG — cô-lập vì sao recall KHÔNG học (M2 smoke kẹt ở ngẫu-nhiên).

Hai câu hỏi tách bạch:
  (A) BACKBONE có làm nổi hồi-tưởng-theo-chỉ-số KHÔNG? (bỏ mắt: frame = token-màu
      + temp_emb; query = <q> + temp_emb[qi]). Nếu A hỏng → lỗi ở backbone/binding.
  (B) Nếu query dùng CHUNG temp_emb (thay vì bảng POS riêng) thì có học không?
      → kiểm giả-thuyết "hai bảng embedding không tự căn nhau".

Chạy nhanh:  python nn-aivie3/scripts/rmsh_m2_debug.py
"""
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from nn_aivie3.model import HybridBackbone                               # noqa: E402

N_COLORS, T = 6, 6
QUERY_TOK = N_COLORS
VOCAB = N_COLORS + 1


def dev_():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class BackboneRecall(nn.Module):
    """Bỏ mắt: frame = tok_emb(color) + temp_emb[t]. query = <q> + temp_emb[qi]."""
    def __init__(self, d=128, share_query_temp=True):
        super().__init__()
        self.d = d; self.share = share_query_temp
        self.tok = nn.Embedding(VOCAB, d); nn.init.normal_(self.tok.weight, std=0.02)
        self.temp = nn.Embedding(T, d); nn.init.normal_(self.temp.weight, std=0.02)
        self.qpos = nn.Embedding(T, d); nn.init.normal_(self.qpos.weight, std=0.02)  # bảng RIÊNG
        self.bb = HybridBackbone(d, n_layer=6, attn_layers=(2, 4), d_state=16, heads=4)

    def forward(self, colors, qi):                      # colors:(b,T) · qi:(b,)
        b = colors.shape[0]
        fr = self.tok(colors) + self.temp.weight[:T][None]         # (b,T,d)
        q = self.tok.weight[QUERY_TOK][None].expand(b, -1).clone() # (b,d)
        q = q + (self.temp(qi) if self.share else self.qpos(qi))   # gắn chỉ-số hỏi
        x = torch.cat([fr, q[:, None]], 1)                         # (b,T+1,d)
        h = self.bb(x)
        return h[:, -1] @ self.tok.weight.T                        # (b,V)


def run(share, dev, steps=250, bs=128, lr=2e-3):
    torch.manual_seed(0)
    m = BackboneRecall(share_query_temp=share).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(1)
    t0 = time.time()
    for it in range(steps):
        colors = torch.randint(0, N_COLORS, (bs, T), generator=gen).to(dev)
        qi = torch.randint(0, T, (bs,), generator=gen).to(dev)
        ans = colors[torch.arange(bs), qi]
        loss = F.cross_entropy(m(colors, qi), ans)
        opt.zero_grad(); loss.backward(); opt.step()
    # eval trên seed khác
    g2 = torch.Generator().manual_seed(999)
    colors = torch.randint(0, N_COLORS, (512, T), generator=g2).to(dev)
    qi = torch.randint(0, T, (512,), generator=g2).to(dev)
    ans = colors[torch.arange(512), qi]
    acc = (m(colors, qi).argmax(-1) == ans).float().mean().item()
    print(f"  share_temp={share}: loss cuối {loss.item():.3f} · acc {acc:.1%} "
          f"(ngẫu-nhiên {1/N_COLORS:.1%}) · {time.time()-t0:.0f}s")
    return acc


def main():
    dev = dev_()
    print(f"=== M2 debug · backbone-only recall · dev={dev} ===")
    print("(A/B) query dùng CHUNG temp_emb với frame:")
    a_share = run(True, dev)
    print("(control) query dùng bảng POS RIÊNG (như smoke gốc):")
    a_sep = run(False, dev)
    print()
    print(f"→ CHUNG={a_share:.1%} vs RIÊNG={a_sep:.1%}")
    if a_share > 0.6 and a_sep < 0.4:
        print("KẾT-LUẬN: backbone recall ĐƯỢC; lỗi smoke = HAI BẢNG embedding không tự căn.")
    elif a_share > 0.6:
        print("KẾT-LUẬN: backbone recall ĐƯỢC cả hai; lỗi smoke nằm ở MẮT (vision không truyền màu).")
    else:
        print("KẾT-LUẬN: backbone CHƯA recall được → cần thêm attn/bước/độ-rộng.")


if __name__ == "__main__":
    main()
