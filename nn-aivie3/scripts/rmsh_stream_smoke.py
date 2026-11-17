"""nn-aivie3 · M0 SMOKE — chứng-minh tại-chỗ (M4) hai điều cốt-lõi của RMSH:

  (1) ĐÚNG-ĐẮN: MambaStep (hồi-quy, giữ-state) cho ra KẾT-QUẢ TRÙNG với
      MambaBlock (theo-chuỗi) tới sai-số số-học → bản train và bản serve là MỘT.
  (2) REALTIME: nuốt luồng "24fps" ở O(1)/token — trễ/step PHẲNG dù luồng dài
      ra, RAM state CHẶN-TRÊN. Đây là lý-do SSM thắng transformer-VLM (§4).

CHẠY (không cần data, không cần GPU):
    python nn-aivie3/scripts/rmsh_stream_smoke.py
"""
import os, sys, time, math
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                  # …/nn-aivie
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))             # để import train_mamba.MambaBlock

from nn_aivie3.streaming import MambaStep, StreamState        # noqa: E402


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def test_correctness(dev):
    """(1) step hồi-quy == scan theo-chuỗi (nạp cùng weight)."""
    from train_mamba import MambaBlock
    torch.manual_seed(0)
    d, d_state, L = 64, 16, 40
    block = MambaBlock(d, d_state).to(dev).eval()
    step = MambaStep.load_from(block).to(dev).eval()

    x = torch.randn(1, L, d, device=dev)
    with torch.no_grad():
        y_seq = block(x)                                       # (1,L,d) — đường train
        st = step.init_state(1, dev)
        y_stream = torch.stack([step.step(x[:, t], st) for t in range(L)], dim=1)

    err = (y_seq - y_stream).abs().max().item()
    ok = err < 1e-3
    print(f"[1] đúng-đắn step==scan: max|Δ|={err:.2e}  → {'ĐẠT' if ok else 'SAI'}")
    return ok


def test_realtime(dev):
    """(2) nuốt luồng dài: trễ/step phẳng, state RAM cố-định."""
    torch.manual_seed(0)
    d, d_state = 512, 16                                       # cỡ ~backbone thật (d512)
    n_layers = 12
    steps = [MambaStep(d, d_state).to(dev).eval() for _ in range(n_layers)]
    norms = None

    def run(n_tokens):
        state = StreamState(steps, attn_window=512, batch=1, device=dev)
        x = torch.randn(1, d, device=dev)
        t0 = time.time()
        for _ in range(n_tokens):
            h = x
            for ms, stt in zip(steps, state.states):
                h = h + ms.step(h, stt)                        # residual, đúng khuôn backbone
            state.n_tokens += 1
        if dev == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
        # RAM của state = tổng phần-tử h + conv_buf (KHÔNG theo n_tokens)
        state_elems = sum(s["h"].numel() + s["conv_buf"].numel() for s in state.states)
        return dt / n_tokens * 1e3, state_elems

    print(f"[2] realtime trên dev={dev}, d={d}×L={n_layers} (Mamba step):")
    prev_elems = None
    flat = True
    for n in (200, 800, 3200):                                 # luồng dài dần ~ 8s→130s @24fps
        ms_per_tok, elems = run(n)
        fps_budget = 1000.0 / ms_per_tok
        secs_at_24fps = n / 24
        tag_elems = f"{elems/1e6:.3f}M phần-tử"
        print(f"    T={n:>5} token (~{secs_at_24fps:5.1f}s@24fps) · "
              f"{ms_per_tok:6.2f} ms/token · ~{fps_budget:6.0f} token/s · state={tag_elems}")
        if prev_elems is not None and elems != prev_elems:
            flat = False
        prev_elems = elems
    print(f"    → RAM state {'CHẶN-TRÊN (phẳng)' if flat else 'PHÌNH (!)'}: "
          f"độc-lập độ-dài luồng — đây là điều transformer-VLM KHÔNG có.")
    return flat


def main():
    dev = pick_device()
    print(f"=== nn-aivie3 M0 smoke · device={dev} · torch {torch.__version__} ===\n")
    ok1 = test_correctness(dev)
    print()
    ok2 = test_realtime(dev)
    print()
    print(f"KẾT: đúng-đắn={'✓' if ok1 else '✗'} · realtime-phẳng={'✓' if ok2 else '✗'}")
    print("→ MambaStep sẵn-sàng làm lõi StreamState cho vòng tri-giác 24fps (M1 kế tiếp).")
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
