"""nn-aivie3 · M2 DEBUG2 — khoanh lỗi ghép: cấu-trúc luồng hay đặc-trưng vfe?

Dùng CHÍNH backbone/embedding của MultimodalRMSH, chỉ hoán 2 trục:
  content ∈ {clean = tok_emb(màu) lặp k lần , vfe = mắt thật}
  query   ∈ {two = [<Q>, POS(qi)] đọc ở token cuối , one = [<Q>+temp(qi)] đọc tại đó}

Suy-luận:
  • clean+two HỌC, vfe+two KẸT  → lỗi ở ĐẶC-TRƯNG VFE (màu có nhưng backbone khó
    vừa-khớp-temp vừa-đọc-màu từ token vfe).
  • clean+two cũng KẸT          → lỗi ở CẤU-TRÚC (k token/frame hoặc query 2-token).
  • vfe+one HỌC còn vfe+two KẸT → lỗi ở KIỂU QUERY (đọc-ở-đâu / bắc-cầu 2 bảng).

Chạy:  python nn-aivie3/scripts/rmsh_m2_debug2.py
"""
import os, sys, time
import torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import rmsh_m2_smoke as S                                               # noqa: E402
from nn_aivie3.rmsh import MOD_TEXT, MOD_VISION                          # noqa: E402

T = 5


def _colors_of(fr, dev):                                # fr:(b,T,3,H,W) → (b,T) nhãn màu
    mean = fr.mean((3, 4))                              # (b,T,3)
    pal = S._PALETTE.to(dev)                            # (6,3)
    d = ((mean[:, :, None, :] - pal[None, None]) ** 2).sum(-1)   # (b,T,6)
    return d.argmin(-1)


def embed(m, frames, colors, qi, content, query):
    qi = qi.to(m.tok_emb.weight.device)
    b = colors.shape[0]; k, d = m.k, m.d
    if content == "clean":
        vis = m.tok_emb(colors)[:, :, None, :].expand(b, T, k, d).clone()      # k bản màu-sạch
    else:
        vis = m.vis_ln(m.vfe(frames.flatten(0, 1))).view(b, T, k, d)
    vis = vis + m.mod_emb.weight[MOD_VISION] + m.temp_emb.weight[:T][None, :, None, :]
    vis = vis.reshape(b, T * k, d)
    q = m.tok_emb.weight[S.QUERY_TOK][None].expand(b, -1).clone() + m.mod_emb.weight[MOD_TEXT]
    if query == "two":
        pos = m.tok_emb(S.POS0 + qi) + m.mod_emb.weight[MOD_TEXT]
        txt = torch.stack([q, pos], 1)                                          # đọc token cuối
    else:
        txt = (q + m.temp_emb(qi))[:, None]                                     # 1 token, temp CHUNG
    return torch.cat([vis, txt], 1)


def run(content, query, dev, steps=250, bs=128, lr=2e-3):
    torch.manual_seed(0)
    m = S.build(dev).to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(1); t0 = time.time()
    for it in range(steps):
        fr, _, ans, qi = S.make_batch(bs, T, dev, gen)
        colors = _colors_of(fr, dev)
        x = embed(m, fr, colors, qi, content, query)
        loss = F.cross_entropy(m.backbone(x)[:, -1] @ m.tok_emb.weight.T, ans)
        opt.zero_grad(); loss.backward(); opt.step()
    # eval seed khác
    g2 = torch.Generator().manual_seed(999)
    fr, _, ans, qi = S.make_batch(512, T, dev, g2)
    colors = _colors_of(fr, dev)
    with torch.no_grad():
        x = embed(m, fr, colors, qi, content, query)
        acc = (m.backbone(x)[:, -1] @ m.tok_emb.weight.T).argmax(-1).eq(ans).float().mean().item()
    print(f"  {content:5s}+{query:3s}: loss {loss.item():.3f} · acc {acc:.1%} · {time.time()-t0:.0f}s "
          f"{'HỌC' if acc > 0.6 else 'kẹt'}")
    return acc


def main():
    dev = S.pick_device()
    print(f"=== M2 debug2 · dev={dev} · T={T} · ngẫu-nhiên {1/S.N_COLORS:.1%} ===")
    r = {}
    for content, query in [("clean", "two"), ("vfe", "two"), ("vfe", "one")]:
        r[(content, query)] = run(content, query, dev)
    print()
    ct, vt, vo = r[("clean", "two")], r[("vfe", "two")], r[("vfe", "one")]
    if ct > 0.6 and vt < 0.4:
        print("→ lỗi ở ĐẶC-TRƯNG VFE (cấu-trúc & query ổn).")
    elif ct < 0.4:
        print("→ lỗi ở CẤU-TRÚC luồng (k token/frame hoặc query 2-token).")
    if vo > 0.6 and vt < 0.4:
        print("→ và/hoặc lỗi ở KIỂU QUERY (one-token shared-temp cứu được).")


if __name__ == "__main__":
    main()
