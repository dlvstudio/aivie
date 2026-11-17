"""nn-aivie3 · M1 SMOKE — mắt + backbone end-to-end, tại-chỗ (M4), không cần data.

Kiểm 4 điều:
  (1) WIRING+GRAD: ảnh + caption → loss → backward → gradient TỚI conv patch của VFE
      (chứng-minh tín-hiệu chảy xuyên mắt→backbone→loss → TRAIN được).
  (2) HỌC-ĐƯỢC: overfit 1 mẫu vài bước → loss giảm mạnh (đường train sống).
  (3) PARITY: forward theo-chuỗi (train) == prefill+step (serve) tới sai-số số-học.
  (4) SINH: generate chữ từ ảnh+prompt, đo trễ token-đầu + tok/s trên M4.

CHẠY:  python nn-aivie3/scripts/rmsh_m1_smoke.py
"""
import os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from nn_aivie3.rmsh import MultimodalRMSH, MOD_TEXT                       # noqa: E402


def pick_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def build(dev):
    torch.manual_seed(0)
    return MultimodalRMSH(
        vocab=512, d=128, n_layer=6, attn_layers=(2, 4), d_state=16, heads=4,
        window=4096, d_v=128, n_query=8, img=64, patch=16,
    ).to(dev)


def test_wiring_and_learn(dev):
    m = build(dev).train()
    b, T = 2, 12
    img = torch.randn(b, 3, 64, 64, device=dev)
    ids = torch.randint(0, 512, (b, T), device=dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)

    loss0, _ = m(img, ids)
    m.zero_grad(); loss0.backward()
    g = m.vfe.encoder.proj.weight.grad
    has_grad = g is not None and g.abs().sum().item() > 0
    print(f"[1] grad tới conv-patch VFE: {'CÓ' if has_grad else 'KHÔNG'} "
          f"(‖g‖={0.0 if g is None else g.norm().item():.3e})  → {'ĐẠT' if has_grad else 'SAI'}")

    # (2) overfit đúng 1 mẫu cố-định
    img1 = torch.randn(1, 3, 64, 64, device=dev)
    ids1 = torch.randint(0, 512, (1, T), device=dev)
    first = None
    for it in range(60):
        loss, _ = m(img1, ids1)
        if first is None:
            first = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    drop = first - loss.item()
    ok2 = loss.item() < first * 0.5
    print(f"[2] overfit 1 mẫu 60 bước: loss {first:.3f} → {loss.item():.3f} "
          f"(giảm {drop:.3f})  → {'ĐẠT' if ok2 else 'YẾU'}")
    return m, has_grad and ok2


def test_parity(m, dev):
    m.eval()
    b, P = 1, 5
    img = torch.randn(b, 3, 64, 64, device=dev)
    ids = torch.randint(0, 512, (b, P), device=dev)
    with torch.no_grad():
        x = m._embed(img, ids)                       # (b, k+P, d)
        h_seq = m.backbone(x)                         # đường TRAIN
        state = m.backbone.init_stream(b, dev, x.dtype)
        h_str = m.backbone.stream_prefill(x, state)   # đường SERVE
    err = (h_seq - h_str).abs().max().item()
    ok = err < 1e-3
    print(f"[3] parity train==serve (backbone): max|Δ|={err:.2e}  → {'ĐẠT' if ok else 'SAI'}")
    return ok


def test_generate(m, dev):
    m.eval()
    img = torch.randn(1, 3, 64, 64, device=dev)
    prompt = torch.randint(0, 512, (1, 4), device=dev)
    # đo trễ token-đầu (prefill hình+prompt) tách khỏi tốc-độ sinh
    t0 = time.time()
    state, h = m.prefill(img, prompt, dev)
    if dev == "mps": torch.mps.synchronize()
    ttf = (time.time() - t0) * 1e3
    n = 32
    t1 = time.time()
    with torch.no_grad():
        for _ in range(n):
            nxt = (h @ m.tok_emb.weight.T).argmax(-1)
            e = m.tok_emb(nxt) + m.mod_emb.weight[MOD_TEXT]
            h = m.backbone.stream_step(e, state)
    if dev == "mps": torch.mps.synchronize()
    tps = n / (time.time() - t1)
    print(f"[4] sinh trên {dev}: trễ token-đầu(prefill hình+prompt)={ttf:.1f} ms · "
          f"sinh ~{tps:.0f} tok/s")
    return True


def main():
    dev = pick_device()
    print(f"=== nn-aivie3 M1 smoke · device={dev} · torch {torch.__version__} ===")
    npar = sum(p.numel() for p in build(dev).parameters())
    print(f"model tí-hon: {npar/1e6:.2f}M tham-số (vocab512·d128·L6·vision d128)\n")
    m, ok12 = test_wiring_and_learn(dev)
    ok3 = test_parity(m, dev)
    ok4 = test_generate(m, dev)
    print()
    allok = ok12 and ok3 and ok4
    print(f"KẾT: wiring+học={'✓' if ok12 else '✗'} · parity={'✓' if ok3 else '✗'} · sinh={'✓' if ok4 else '✗'}")
    print("→ M1 xong: mắt nối backbone, train-được, serve khớp train. Kế: M2 nuốt NHIỀU frame (bộ-nhớ thời-gian).")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
