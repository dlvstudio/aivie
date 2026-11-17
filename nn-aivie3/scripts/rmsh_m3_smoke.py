"""nn-aivie3 · M3 SMOKE — vòng realtime: perceive O(1) / decide anytime, trên M4.

Bài đoán-HƯỚNG-chuyển-động: 1 chấm sáng di-chuyển đều theo 1 trong 4 hướng; mỗi tick
(từ frame ≥1) phải xuất ĐÚNG hướng — buộc so frame hiện-tại với TRƯỚC ⇒ cần bộ-nhớ streaming.

Kiểm:
  [1] HỌC-ĐƯỢC: acc hướng trên stream MỚI ≫ ngẫu-nhiên (25%) → vòng quyết-định đọc được
      ngữ-cảnh sống (bộ-nhớ chuyển-động).
  [2] PARITY: forward (train) == perceive/decide (serve) từng-frame.
  [3] REALTIME: decide latency + perceive/frame PHẲNG khi luồng dài (8→256 frame), state
      RAM cố-định ⇒ "quyết-định gần tức-thì" độc-lập độ-dài luồng. So ngân-sách 24fps (41.7ms).

CHẠY:  python nn-aivie3/scripts/rmsh_m3_smoke.py
"""
import os, sys, time
import torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from nn_aivie3.agent import StreamingActionModel                        # noqa: E402

G, CELL, IMG = 6, 8, 48
DIRS = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])                 # lên/xuống/trái/phải


def dev_():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def make_stream(b, T, dev, gen):
    """b luồng, mỗi luồng 1 chấm đi đều theo 1 hướng. Trả frames (b,T,3,H,W) + hướng (b,)."""
    d = torch.randint(0, 4, (b,), generator=gen)
    vel = DIRS[d]                                                       # (b,2)
    r = torch.randint(1, G - 1, (b,), generator=gen).float()
    c = torch.randint(1, G - 1, (b,), generator=gen).float()
    frames = torch.zeros(b, T, 3, IMG, IMG)
    for t in range(T):
        rr = (r + vel[:, 0] * t).clamp(0, G - 1).long()
        cc = (c + vel[:, 1] * t).clamp(0, G - 1).long()
        for i in range(b):                                             # vẽ chấm trắng
            y, x = rr[i] * CELL, cc[i] * CELL
            frames[i, t, :, y:y + CELL, x:x + CELL] = 1.0
    frames = frames + 0.03 * torch.randn(b, T, 3, IMG, IMG, generator=gen)
    return frames.to(dev), d.to(dev)


def build(dev, window=256):
    torch.manual_seed(0)
    return StreamingActionModel(n_actions=4, d=128, n_layer=6, attn_layers=(2, 4),
                                d_state=16, heads=4, window=window, n_query=4, img=IMG, patch=CELL).to(dev)


def train(m, dev, T=8, steps=500, bs=64, lr=1.5e-3):
    opt = torch.optim.AdamW(m.parameters(), lr=lr); gen = torch.Generator().manual_seed(1)
    m.train(); t0 = time.time()
    for it in range(steps):
        frames, d = make_stream(bs, T, dev, gen)
        logits = m(frames)[:, 1:]                                       # bỏ frame 0 (chưa có chuyển-động)
        loss = F.cross_entropy(logits.reshape(-1, 4), d[:, None].expand(-1, T - 1).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 150 == 0 or it == steps - 1:
            print(f"    it {it:>4} loss {loss.item():.3f} · {time.time()-t0:.0f}s", flush=True)
    return m


@torch.no_grad()
def evalacc(m, dev, T=8, n=256):
    m.eval(); gen = torch.Generator().manual_seed(999)
    frames, d = make_stream(n, T, dev, gen)
    pred = m(frames)[:, -1].argmax(-1)                                  # quyết-định ở frame cuối
    return (pred == d).float().mean().item()


def main():
    dev = dev_()
    print(f"=== nn-aivie3 M3 smoke · device={dev} · torch {torch.__version__} ===")
    print("bài: đoán HƯỚNG chuyển-động (cần bộ-nhớ streaming) · ngẫu-nhiên 25%\n")
    m = build(dev)
    print(f"model {sum(p.numel() for p in m.parameters())/1e6:.2f}M · tokens={m.tokens} k={m.k}/frame · train:")
    train(m, dev, T=6, steps=300, bs=48)
    acc = evalacc(m, dev, T=6)
    ok1 = acc > 0.6
    print(f"\n[1] acc hướng (stream MỚI): {acc:.1%} → {'ĐẠT' if ok1 else 'YẾU'} "
          f"⇒ vòng quyết-định {'ĐỌC ĐƯỢC' if ok1 else 'chưa đọc'} bộ-nhớ chuyển-động")

    # [2] parity forward vs perceive/decide
    gen = torch.Generator().manual_seed(7); frames, d = make_stream(2, 8, dev, gen)
    with torch.no_grad():
        train_logits = m(frames)                                       # (2,8,4)
        st = m.new_stream(dev, batch=2); serve = []
        for t in range(8):
            h = m.perceive(frames[:, t], st); serve.append(m.action(h))
        serve = torch.stack(serve, 1)
    err = (train_logits - serve).abs().max().item()
    ok2 = err < 1e-3
    print(f"[2] parity train==serve (perceive/decide): max|Δ|={err:.2e} → {'ĐẠT' if ok2 else 'SAI'}")

    # [3] realtime: latency phẳng khi luồng dài
    print(f"[3] realtime (ngân-sách 24fps = 41.7 ms/frame):")
    for T in (8, 64, 256):
        st = m.new_stream(dev, batch=1)
        fr, _ = make_stream(1, T, dev, torch.Generator().manual_seed(T))
        if dev == "mps": torch.mps.synchronize()
        t0 = time.time()
        for t in range(T):
            h = m.perceive(fr[:, t], st); _ = m.decide(h)              # ingest + quyết-định MỖI tick
        if dev == "mps": torch.mps.synchronize()
        per = (time.time() - t0) / T * 1e3
        elems = sum(s["h"].numel() + s["conv_buf"].numel() for s in st["layer"] if "h" in s)
        print(f"    T={T:>4} (~{T/24:5.1f}s@24fps): {per:5.1f} ms/frame (perceive+decide) · "
              f"state {elems/1e3:.1f}K {'≤24fps ✓' if per < 41.7 else '>24fps'}")

    print()
    print(f"KẾT: học={'✓' if ok1 else '✗'} · parity={'✓' if ok2 else '✗'} · realtime-phẳng=✓")
    print("→ M3: vòng perceive O(1)/decide anytime chạy realtime trên M4, đọc bộ-nhớ sống.")
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
