# nn-aivie3 — RMSH (Realtime Multimodal Streaming Hybrid)

Lõi tri-giác **thời-gian-thực, đa-phương-thức** mọc trên backbone **Hybrid Mamba+Attn** của nn-aivie.
Mục-tiêu: agent ↔ user **đối-đáp gần tức-thì**, model **nhìn webcam / màn-hình 24fps** để hiểu & trả-lời, cắm-được vào **robot** (mắt = cam).

- **Giải-pháp đầy-đủ:** [TK3_000_KIEN_TRUC.md](TK3_000_KIEN_TRUC.md)
- **Luận-điểm lõi:** *trạng-thái hồi-quy của SSM = bộ-nhớ-làm-việc tri-giác* → nuốt luồng cảm-biến ở **O(1)/token**, RAM chặn-trên, chạy realtime **mãi** trên M4 (transformer-VLM phình KV-cache theo frame → nổ).

## Chạy smoke (không cần data/GPU)
```bash
python nn-aivie3/scripts/rmsh_stream_smoke.py   # M0: streaming state O(1)
python nn-aivie3/scripts/rmsh_m1_smoke.py       # M1: mắt + backbone end-to-end
python nn-aivie3/scripts/rmsh_m2_smoke.py       # M2: bộ-nhớ thời-gian + đính-chính lõi
python nn-aivie3/scripts/rmsh_m25_gate.py       # M2.5: chú-ý học-được (top-down vs uniform)
python nn-aivie3/scripts/rmsh_m3_smoke.py       # M3: vòng realtime perceive/decide
```
- **M0** chứng-minh: `MambaStep` hồi-quy **trùng** `MambaBlock` theo-chuỗi (cùng weight → train và serve là một); trễ/step **phẳng** + state **cố-định** khi luồng dài ra.
- **M1** chứng-minh: mắt (VFE từ-scratch) nối backbone, **train-được** (grad tới conv-patch, overfit 6.18→0.006), **serve khớp train** (Δ=4.8e-7), sinh chữ từ ảnh+prompt (~120 tok/s trên M4).

## Cấu-trúc
```
nn-aivie3/
  TK3_000_KIEN_TRUC.md         # kế-hoạch + kiến-trúc (đọc cái này)
  src/nn_aivie3/
    streaming.py               # MambaStep + StreamState (M0)
    model.py                   # StreamAttn + HybridBackbone (train↔serve khớp)
    vision.py                  # VisionEncoder + Resampler + Projector (M1)
    rmsh.py                    # MultimodalRMSH: mắt + backbone + LM head (M1/M2)
    agent.py                   # RUNTIME realtime: perceive O(1)/decide anytime (M3)
  scripts/rmsh_*.py            # smoke M0..M3 + chẩn-đoán M2.5
```

## Trạng-thái
- **M0 ✅** streaming state O(1) · **M1 ✅** mắt+backbone end-to-end (train-được, serve khớp).
- **M2 ✅** bộ-nhớ thời-gian: hybrid recall 100%, parity, RAM phẳng — **kèm đính-chính lõi** (§4.1): recall-chính-xác frame ngoài cửa-sổ attention ≈ ngẫu-nhiên ⇒ cửa-sổ attn phải phủ chân-trời hồi-tưởng; Mamba lo bộ-nhớ-nền nén.
- **M2.5 ✅** chú-ý học-được (cổng salience): top-down 100% vs uniform 80.5%, bản-đồ chú-ý dồn 0.99 đúng mục-tiêu (curriculum, from-scratch).
- **M3 ✅** vòng realtime `agent.py` (perceive O(1)/decide anytime): parity Δ=4.5e-8, state RAM phẳng, ~10.7ms/frame (<24fps); quyết-định đọc bộ-nhớ chuyển-động 98.8% (token giữ-vị-trí).
- **M3.5** ✅ kênh chữ/đối-thoại (xem→hỏi-chữ→đáp-chữ 100%); **adaptive-attention** ✅ (`rmsh_adaptive.py`: gate dồn 1.0 vào mục-tiêu, top-8 giữ 88% — top-down+curriculum+head-ép-gate; bottom-up fail).
- **M4** ✅ tiếng 2 chiều: audio-IN (chirp 100%, parity Δ=9.5e-7) + **TTS-out** (`rmsh_m4_tts.py`: sinh mel, hướng-pitch khớp 100%).
- **M5** ✅ robot VLA vòng-kín: imitation 87.9%, chạm đích 72.7%/10 bước qua perceive O(1)/decide.

**Kiến-trúc:** mỗi phương-thức = front-end mỏng riêng → **CHUNG một backbone + StreamState** (early-fusion, không phải mạng riêng mỗi kênh). Expert (nếu có) chuyên theo MIỀN, không theo giác-quan.
