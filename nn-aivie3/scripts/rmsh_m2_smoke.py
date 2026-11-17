"""nn-aivie3 · M2 SMOKE — BỘ-NHỚ THỜI-GIAN: state có nhớ frame quá-khứ không?

Bài hồi-tưởng-theo-chỉ-số trên "video" màu tổng-hợp (màu ngẫu-nhiên mỗi video ⇒
không học vẹt): nuốt T frame rồi hỏi "frame thứ qi màu gì?". Câu hỏi NEO bằng chính
temp_emb[qi] (cùng mã vị-trí-thời-gian mà frame đeo) — bài học M2 debug2.

Kiểm:
  [1] HỌC-ĐƯỢC (hybrid, cửa-sổ rộng): acc video MỚI ≫ ngẫu-nhiên.
  [2] PARITY: đường serve (nuốt frame-rồi-chữ từng token) == đường train.
  [3] RAM PHẲNG: luồng ×4 (T=8→32) → state SSM Y NGUYÊN cỡ.
  [4] LÕI (đo, không gate): ÉP cửa-sổ attention NHỎ → frame cũ NẰM NGOÀI tầm
      attention ⇒ recall CHỈ có thể đến từ MAMBA-STATE. Đo acc trên frame-sớm.
      Đây mới là phép thử thật cho luận-điểm "SSM-state = bộ-nhớ tri-giác O(1)".

CHẠY:  python nn-aivie3/scripts/rmsh_m2_smoke.py
"""
import os, sys, time
import torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from nn_aivie3.rmsh import MultimodalRMSH                                # noqa: E402

N_COLORS = 6
MAXT = 40
IMG = 32
QUERY_TOK = N_COLORS
VOCAB = N_COLORS + 1
_PALETTE = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]], dtype=torch.float32)[:N_COLORS]


def pick_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def make_batch(b, T, dev, gen, qi_lo=0, qi_hi=None):
    qi_hi = T if qi_hi is None else qi_hi
    colors = torch.randint(0, N_COLORS, (b, T), generator=gen)
    base = _PALETTE[colors][:, :, :, None, None].expand(b, T, 3, IMG, IMG).clone()
    frames = base + 0.05 * torch.randn(b, T, 3, IMG, IMG, generator=gen)
    qi = torch.randint(qi_lo, qi_hi, (b,), generator=gen)
    ans = colors[torch.arange(b), qi]
    text = torch.full((b, 1), QUERY_TOK)                        # 1 token hỏi (neo bằng temp_emb[qi])
    return frames.to(dev), text.to(dev), ans.to(dev), qi.to(dev)


def build(dev, window=4096):
    torch.manual_seed(0)
    return MultimodalRMSH(
        vocab=VOCAB, d=128, n_layer=6, attn_layers=(2, 4), d_state=16, heads=4,
        window=window, d_v=128, n_query=4, img=IMG, patch=16, max_frames=MAXT,
    ).to(dev)


def train_recall(m, dev, T=8, steps=400, bs=64, qi_lo=0, qi_hi=None, lr=2e-3):
    gen = torch.Generator().manual_seed(1)
    opt = torch.optim.AdamW(m.parameters(), lr=lr); m.train(); t0 = time.time()
    for it in range(steps):
        fr, tx, ans, qi = make_batch(bs, T, dev, gen, qi_lo, qi_hi)
        loss = F.cross_entropy(m.logits_multi(fr, tx, query_frame=qi)[:, -1], ans)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0 or it == steps - 1:
            print(f"    step {it:>4} · loss {loss.item():.3f} · {time.time()-t0:.0f}s")
    return m


@torch.no_grad()
def eval_recall(m, dev, T=8, n=512, qi_lo=0, qi_hi=None):
    gen = torch.Generator().manual_seed(999); m.eval()
    fr, tx, ans, qi = make_batch(n, T, dev, gen, qi_lo, qi_hi)
    pred = m.logits_multi(fr, tx, query_frame=qi)[:, -1].argmax(-1)
    return (pred == ans).float().mean().item()


def main():
    dev = pick_device()
    print(f"=== nn-aivie3 M2 smoke · device={dev} · torch {torch.__version__} ===")
    print(f"bài: nhớ màu frame quá-khứ · {N_COLORS} màu · ngẫu-nhiên={1/N_COLORS:.1%}\n")

    # [1] hybrid cửa-sổ rộng
    m = build(dev, window=4096)
    print(f"model {sum(p.numel() for p in m.parameters())/1e6:.2f}M · train hybrid (cửa-sổ rộng) T=8:")
    train_recall(m, dev, T=8, steps=400, bs=64)
    acc = eval_recall(m, dev, T=8)
    ok1 = acc > 0.60
    print(f"\n[1] acc video MỚI (T=8, hybrid): {acc:.1%} (ngẫu-nhiên {1/N_COLORS:.1%}) "
          f"→ {'ĐẠT' if ok1 else 'YẾU'}")

    # [2] parity
    gen = torch.Generator().manual_seed(7)
    fr, tx, ans, qi = make_batch(4, 8, dev, gen)
    with torch.no_grad():
        lb = m.logits_multi(fr, tx, query_frame=qi)[:, -1]
        ls, _ = m.stream_answer(fr, tx, dev, query_frame=qi)
    err = (lb - ls).abs().max().item()
    same = (lb.argmax(-1) == ls.argmax(-1)).all().item()
    ok2 = err < 1e-3 and same
    print(f"[2] parity train==serve: max|Δ|={err:.2e} · cùng đáp-án={same} → {'ĐẠT' if ok2 else 'SAI'}")

    # [3] RAM phẳng
    def state_elems(T):
        fr, tx, _, qi = make_batch(1, T, dev, torch.Generator().manual_seed(T))
        _, st = m.stream_answer(fr, tx, dev, query_frame=qi)
        return sum(s["h"].numel() + s["conv_buf"].numel() for s in st["layer"] if "h" in s)
    e8, e32 = state_elems(8), state_elems(32)
    ok3 = e8 == e32
    print(f"[3] RAM state SSM: T=8→{e8/1e3:.1f}K · T=32→{e32/1e3:.1f}K phần-tử "
          f"→ {'PHẲNG (chặn-trên)' if ok3 else 'PHÌNH (!)'} (luồng ×4, state không đổi)")

    # [4] LÕI: cửa-sổ NHỎ → frame cũ ngoài tầm attn → chỉ Mamba gánh
    W = 6                                                        # < k=4 ×1.5; frame<6 nằm ngoài tầm ở token cuối
    print(f"\n[4] LÕI — ép cửa-sổ attn W={W} (frame-sớm NGOÀI tầm attention), train T=8 hỏi frame 0..5:")
    ms = build(dev, window=W)
    train_recall(ms, dev, T=8, steps=400, bs=64, qi_lo=0, qi_hi=6)
    acc_ssm = eval_recall(ms, dev, T=8, qi_lo=0, qi_hi=6)
    # đối-chứng: cùng model, hỏi frame-mới-nhất (trong tầm attn)
    acc_recent = eval_recall(ms, dev, T=8, qi_lo=7, qi_hi=8)
    print(f"    acc frame-SỚM (chỉ Mamba-state): {acc_ssm:.1%} · frame-mới (trong tầm attn): {acc_recent:.1%} "
          f"· ngẫu-nhiên {1/N_COLORS:.1%}")
    verdict = ("MAMBA-STATE GÁNH ĐƯỢC bộ-nhớ thời-gian" if acc_ssm > 0.5
               else "Mamba-state gánh YẾU → cần tăng attn/d_state hoặc cửa-sổ (phát-hiện thật)")
    print(f"    ⇒ {verdict}")

    print()
    print(f"KẾT: học={'✓' if ok1 else '✗'} · parity={'✓' if ok2 else '✗'} · RAM-phẳng={'✓' if ok3 else '✗'} "
          f"· lõi-SSM-recall={acc_ssm:.0%}")
    print("→ M2: bộ-nhớ thời-gian chạy; [4] đo đúng năng-lực SSM-state. Kế: M3 vòng realtime + screen 24fps.")
    sys.exit(0 if (ok1 and ok2 and ok3) else 1)


if __name__ == "__main__":
    main()
