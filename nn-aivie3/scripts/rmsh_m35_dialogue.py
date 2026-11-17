"""nn-aivie3 · M3.5(1) — KÊNH CHỮ/ĐỐI-THOẠI trên luồng nhìn (text trước).

Dùng cơ-chế M2 đã kiểm-chứng (MultimodalRMSH.stream_answer, recall 100%, có parity
streaming): agent XEM một luồng "màn-hình" (chuỗi frame màu = sự-kiện), rồi trả lời
CÂU HỎI BẰNG CHỮ về nội-dung đã thấy — chạy qua đúng runtime streaming (state ấm).
Đây là "đối-thoại realtime nhìn-màn-hình, đáp text" — text trước, đúng thứ-tự đã chọn.

CHẠY:  python nn-aivie3/scripts/rmsh_m35_dialogue.py
"""
import os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import rmsh_m2_smoke as S                                               # tái-dùng model+bài đã kiểm-chứng

MAU = ["đỏ", "lục", "lam", "vàng", "tím", "lơ"]                         # tên màu (đọc cho người)
STT = ["thứ 1", "thứ 2", "thứ 3", "thứ 4", "thứ 5", "thứ 6", "thứ 7", "thứ 8"]


def main():
    dev = S.pick_device()
    print(f"=== nn-aivie3 M3.5(1) · kênh chữ/đối-thoại trên luồng nhìn · dev={dev} ===")
    print("agent XEM luồng frame màu (màn-hình) → HỎI bằng chữ → ĐÁP bằng chữ (qua state streaming)\n")

    m = S.build(dev)
    print("huấn-luyện nhanh (cơ-chế M2 đã 100%)…")
    S.train_recall(m, dev, T=8, steps=400, bs=64)
    acc = S.eval_recall(m, dev, T=8)
    print(f"độ-chính-xác trả-lời (held-out): {acc:.1%}\n")

    # transcript đối-thoại đọc-được: streaming perceive rồi hỏi bằng chữ
    m.eval(); gen = torch.Generator().manual_seed(2025)
    frames, text, ans, qi = S.make_batch(4, 8, dev, gen)
    with torch.no_grad():
        logits, _ = m.stream_answer(frames, text, dev, query_frame=qi)  # ĐƯỜNG STREAMING (state ấm)
        pred = logits.argmax(-1)
    # màu từng frame (từ ảnh) để in "màn-hình đã thấy"
    seen = (frames.mean((3, 4))[..., None, :] - S._PALETTE.to(dev)[None, None]).pow(2).sum(-1).argmin(-1)
    for i in range(4):
        canh = " → ".join(MAU[seen[i, t].item()] for t in range(8))
        q = qi[i].item(); a = MAU[pred[i].item()]; g = MAU[ans[i].item()]
        ok = "✓" if pred[i].item() == ans[i].item() else "✗"
        print(f"  [màn-hình] {canh}")
        print(f"  [người]   Frame {STT[q]} màu gì?")
        print(f"  [agent]   Màu {a}.  {ok} (đúng: {g})\n")
    print("→ M3.5(1): kênh chữ chạy trên runtime streaming (xem→hỏi-chữ→đáp-chữ). "
          "Đây là trợ-lý nhìn-màn-hình trả-lời text — text trước, tiếng ở M4.")


if __name__ == "__main__":
    main()
