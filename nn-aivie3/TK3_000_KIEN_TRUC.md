# nn-aivie3 — Lõi đa-phương-thức streaming realtime (RMSH)

> Soạn: 2026-09-05 · Nối tiếp nn-aivie (HybridLM Mamba+Attn, TK_086→TK_100)
> Trạng thái: **KẾ HOẠCH + PoC chạy-được** (`scripts/rmsh_stream_smoke.py` smoke-test trên M4)
> Quyết định người dùng: text trước→voice sau · train GPU / **suy-luận trên M4 (hoặc thấp/cao hơn chút)** · **mở-rộng from-scratch backbone Hybrid của ta**

---

## 0. TL;DR — chốt nhanh

- **Tên kiến-trúc:** **RMSH** = *Realtime Multimodal Streaming Hybrid*. Một backbone **Mamba(SSM) + sliding-window Attention** (chính là HybridLM của ta, [[mamba-ssm-tren-m4]]) nuốt **một luồng token đa-phương-thức duy nhất** (hình + chữ + sau này tiếng).
- **Luận-điểm lõi (khác biệt thật):** **trạng-thái hồi-quy của SSM = bộ-nhớ-làm-việc tri-giác.** Toàn bộ quá-khứ (mọi frame webcam/màn-hình, mọi lời) được **nén vào state kích-thước cố-định**, cập-nhật **O(1)/token**. → Chạy **realtime mãi mãi** với chi-phí *chặn-trên*, nơi VLM transformer thuần phình KV-cache theo từng frame rồi **nổ RAM + chậm dần** sau vài phút. Đây là lý-do kỹ-thuật để chọn nền SSM cho webcam/robot/screen-24fps.
- **3 bài-toán khó & lời-giải:**
  1. *Nuốt 24fps không nổ tính-toán* → (a) **step hồi-quy giữ-state** (không tính lại lịch-sử), (b) **cổng keyframe/chuyển-động** (bỏ frame tĩnh), (c) **nén token/frame** bằng resampler (k token/frame).
  2. *Trễ đối-đáp <300ms* → state thế-giới **luôn ấm** trong SSM (không prefill lịch-sử lúc trả-lời) + backbone nhỏ + sliding-window attn + **spec-decode** (tái-dùng TK_058) + cắt-khúc TTS sớm.
  3. *Chạy trên M4* → cỡ nhỏ + **int4/int8**; SSM thân-thiện băng-thông-nhớ; **không có KV-cache phình**.
- **Ba cỡ đề-xuất (đích M4 24GB):** `nano` ~0.4B · **`mini` ~1.3B (mặc-định)** · `plus` ~3B (là bản "cao hơn chút" — có đề-xuất riêng ở §6).
- **Điểm trung-thực nhất:** nút-thắt vẫn là **DỮ-LIỆU** ([[nut-that-long-form-la-du-lieu]]), lần này là **ảnh-văn + video-văn tiếng Việt**. Vision-encoder from-scratch là chỗ **ROI thấp nhất** → giữ **ablation warm-start** làm baseline (đúng văn-hoá "luôn có baseline đã-chứng-minh" của ta).
- **Mảnh code đã dựng hôm nay:** `MambaStep` — bản step hồi-quy của `MambaBlock`, **nạp thẳng weight** từ block đã train, chứng-minh nuốt luồng ở O(1)/frame trên M4. Chạy: `python nn-aivie3/scripts/rmsh_stream_smoke.py`.

---

## 1. Vì sao KHÔNG dùng lại thẳng dòng LM-text

nn-aivie tới giờ là **LM văn-bản** (sinh chữ theo lượt, offline-ish). nn-aivie3 đổi trục sang **hệ tri-giác thời-gian-thực**: lõi khó không phải *kiến-thức* mà là **độ-trễ** và **nuốt luồng cảm-biến liên-tục**. Ba ràng-buộc mới:

| Ràng-buộc | LM-text (cũ) | RMSH (mới) |
|---|---|---|
| Thời-gian | theo lượt, tính lại prompt mỗi lần | **liên-tục**, state ấm sẵn, trả-lời tức-thì |
| Đầu-vào | chuỗi token rời-rạc | **luồng** hình 24fps + chữ (+ tiếng) đồng-thời |
| Bộ-nhớ | KV-cache theo ctx (chấp-nhận-được offline) | phải **chặn-trên** dù chạy hàng giờ → SSM state cố-định |
| Đích chạy | train/serve GPU | **suy-luận trên M4/edge/robot** |

→ Nền **SSM-hybrid của ta lại chính là lựa-chọn đúng**: nó vốn thắng ở *ctx dài* và *state hồi-quy* trên M4 ([[mamba-ssm-tren-m4]]). RMSH chỉ **mở đường step + thêm mắt**.

---

## 2. Kiến-trúc RMSH (tổng-thể)

```
                         ┌──────────────── LUỒNG THẾ-GIỚI (liên-tục) ────────────────┐
 webcam/screen 24fps ──▶ │ FrameSampler → CổngChuyểnĐộng → VisionEncoder(nhỏ)         │
                         │                       → Resampler(m query → k token/frame) │
                         │                       → proj(d) + pos không-gian/thời-gian │
 mic (pha 2) ──────────▶ │ AudioEncoder streaming → (token tiếng | ASR→chữ)          │
                         └───────────────┬───────────────────────────────────────────┘
                                         │  (đẩy token vào StreamState, O(1)/token)
 user chữ / ASR ───────────────────────▶ │
                                         ▼
                    ┌─────────────── BACKBONE HYBRID (StreamState) ──────────────┐
                    │  L tầng:  Mamba(step, giữ h + conv-ring)  ×N                │
                    │           SlidingWindowAttn(KV cửa-sổ W)   mỗi `attn_every` │
                    │  + type/modality-embedding · pos hợp-nhất                   │
                    └───────────────┬───────────────────────────────────────────┘
                                    ▼
        ┌─── LM head (weight-tied) → chữ ──▶ stream ra ──▶ (pha 2) TTS cắt-khúc ──▶ loa
        └─── (pha 3) Action head → token hành-động ──▶ robot
```

### 2.1 Front-end thị-giác (VFE)
- **FrameSampler:** capture 24fps nhưng **không encode mọi frame**. `CổngChuyểnĐộng` = diff khung rẻ (L1 trên ảnh thu-nhỏ / hash tri-giác). Frame ~tĩnh → đẩy **1 token "no-change"** (state tự giữ ngữ-cảnh); frame đổi → encode đầy-đủ. Đây là đòn giảm-tải lớn nhất cho 24fps.
- **VisionEncoder (nhỏ, train-được):** ConvNeXt-T / ViT-S → lưới patch. *From-scratch được* ở cỡ nhỏ, **nhưng** giữ **ablation warm-start** (nạp encoder mở đông-băng) làm baseline — đây là chỗ ROI thấp nhất khi from-scratch (§7 rủi-ro).
- **Resampler:** m learned-query (Perceiver/attention-pool) nén mỗi frame → **k token cố-định** (k≈8–32). Chốt-trên chi-phí/frame ở LM.
- **Pos:** 2D không-gian + tag thời-gian (frame index / mốc giây).

### 2.2 Front-end chữ
Dùng lại **BPE vocab của ta** (`src/nn_aivie/core/vocab.py`). Thêm token đặc-biệt: `<img>`, `<frame>`, `<user>`, `<assistant>`, `<screen>`, `<cam>`, `<no_change>`.

### 2.3 Front-end tiếng (PHA 2)
Hai lối, ablate: (a) **native speech token** (encoder âm-thanh → token rời, đẩy thẳng vào backbone như hình) — trễ thấp nhất, đúng "đối-đáp"; (b) **ASR→chữ** (rẻ, tách bạch). Đầu ra: **TTS streaming cắt-khúc** (phát ngay câu đầu trong khi sinh tiếp). Barge-in = cổng phát-hiện-giọng ngắt sinh.

### 2.4 Backbone Hybrid + `StreamState`
- **Tầng:** Mamba step (giữ `h` + ring-buffer conv) xen `SlidingWindowAttn` mỗi `attn_every` tầng. Attention **cửa-sổ W** (không full) → chi-phí chặn-trên dù luồng chạy hàng giờ; Mamba lo bộ-nhớ-dài.
- **StreamState (mảnh lõi mới):** với mỗi tầng Mamba giữ `(h: b×d_inner×n, conv_buf: b×d_inner×(d_conv-1))`; với mỗi tầng Attn giữ KV **cửa-sổ trượt** W token gần nhất. `step(token)` cập-nhật tại-chỗ, O(1) theo độ-dài luồng.
- **Một luồng, nhiều loại:** hình/chữ/tiếng chung một chuỗi + **modality-embedding** để backbone biết loại token.

### 2.5 Đầu ra
- **LM head** weight-tied → chữ (stream ra).
- **Action head (PHA 3, robot):** token hành-động rời-rạc (kiểu RT-2/OpenVLA) hoặc đầu hồi-quy nhỏ → khớp/bánh. VLA nối thẳng vào state tri-giác đang-ấm.

---

## 3. Runtime realtime — hai vòng-lặp

- **Vòng tri-giác (chạy nền, 24fps):** grab → cổng chuyển-động → encode keyframe → `state.push(vision_tokens)`. Mỗi keyframe vài token → vài **step O(1)** → rẻ. **State luôn phản-ánh thế-giới hiện-tại.**
- **Vòng đối-thoại:** user gõ / ASR phát chữ → `state.push(text)` → **decode trả-lời từ state đã-ấm** → stream ra → (pha 2) TTS. Vì tri-giác giữ state ấm, **trễ token-đầu ≈ chỉ decode vài token**, không prefill lại lịch-sử hình.

> Chính vì tách 2 vòng mà "đối-đáp như 2 người" khả-thi trên M4: lúc user nói xong, model **đã nhìn thấy hết** rồi.

---

## 4. Vì sao SSM thắng transformer cho bài-toán này (định-lượng)

Cho luồng chạy `T` frame, mỗi frame `k` token:

| | Transformer-VLM thuần | **RMSH (SSM-hybrid)** |
|---|---|---|
| Bộ-nhớ theo thời-gian | KV-cache `O(T·k)` → **phình vô-hạn** | state Mamba **`O(1)`** + KV cửa-sổ `O(W)` |
| Tính/token-mới | attention `O(T·k)` (dài dần) | **`O(1)`** (Mamba) + `O(W)` (attn cửa-sổ) |
| Chạy 1 giờ @24fps | OOM / chậm thảm | **tốc-độ + RAM phẳng** |

→ Đây không phải tinh-chỉnh; là **khác-biệt tiệm-cận (asymptotic)**. PoC `rmsh_stream_smoke.py` in ra **trễ/step phẳng** khi luồng dài ra → chứng-minh tại-chỗ.

### 4.1 ĐÍNH-CHÍNH quan-trọng (đo ở M2, 2026-09-06) — không nói quá

Phép thử lõi `rmsh_m2_smoke.py` §[4] cho kết-quả **trung-thực và phải ghi rõ**:

| Điều | Đo được | Kết-luận |
|---|---|---|
| RAM state **chặn-trên** khi luồng dài | ✅ T=8→32 state Y NGUYÊN 19.5K | thật — đây là win vững |
| Hồi-tưởng-chính-xác frame quá-khứ **khi attention thấy** (hybrid, cửa-sổ rộng) | ✅ **100%** | thật — **attention làm việc recall** |
| Hồi-tưởng-chính-xác frame **NGOÀI cửa-sổ attention** (chỉ Mamba-state gánh) | ❌ **~18% ≈ ngẫu-nhiên** | **SSM thuần YẾU ở precise recall** |

→ **Luận-điểm đúng đã hiệu-chỉnh:** state SSM cho **RAM phẳng + chi-phí/token O(1)** (điều transformer-VLM không có) — nhưng **hồi-tưởng-CHÍNH-XÁC một frame cụ-thể ở xa là việc của TẦNG ATTENTION**, không phải Mamba-state (khớp tài-liệu: SSM thuần kém "mò kim"). Vậy **cửa-sổ attention phải đủ phủ *chân-trời hồi-tưởng* ta cần**; Mamba-state lo **bộ-nhớ-nền nén, lossy** ngoài chân-trời đó. Chính đây là **lý-do tồn-tại của HYBRID** (Mamba rẻ + vài tầng attention sắc) — M2 vừa tái-hiện định-lượng.

→ **Hệ-quả thiết-kế (đưa vào M3+):** (a) chọn **cửa-sổ attention theo chân-trời hồi-tưởng** (vd vài giây gần nhất @24fps sau nén token) — chi-phí vẫn **chặn-trên theo cửa-sổ**, không theo độ-dài luồng; (b) cho ký-ức xa: **token "mốc" (landmark)/tóm-tắt** định-kỳ đẩy vào luồng, hoặc **RAG-kNN** truy-hồi ngoài (đòn bẩy đã có); (c) **đã quét `d_state` 16/64/128 (cửa-sổ W=6): đều ≈ ngẫu-nhiên** (`rmsh_m2_knob.py`) — tăng state Mamba KHÔNG cứu precise recall → đòn-bẩy đúng là **cửa-sổ/tỉ-lệ-attention + token-mốc/RAG**, không phải d_state.

---

## 5. Kế-hoạch train (GPU) → distill xuống M4

| Pha | Tên | Mục-tiêu | Data | Đóng-băng |
|---|---|---|---|---|
| **S0** | Backbone-text | LM nền mạch-lạc | pipeline TK_100 (`build_pretrain_pipeline`) | — |
| **S1** | Căn ảnh-văn | nối VFE↔backbone (1 frame) | ảnh-caption vi + dịch COCO/LLaVA→vi | đông backbone, train VFE+proj |
| **S2** | Streaming thời-gian | dạy SSM giữ bộ-nhớ-video + khuôn-lượt | video-caption, **screen-record tự-nhãn** (distill VLM-teacher), egocentric | mở dần |
| **S3** | Hội-thoại đa-p.thức SFT | đối-đáp realtime, barge-in | hội-thoại có hình/màn-hình | full |
| **S4** (pha 2) | Tiếng | ASR/TTS tier | audio-vi | tách module |
| **S5** (pha 3) | Robot VLA | token hành-động | teleop/sim | action head |

Xuyên suốt: **distill teacher→student** (văn-hoá distill của ta) + **quantize int4/int8** cho M4. **RAG-FAISS giữ nguyên** làm đòn-bẩy grounding sự-thật độc-lập ([[tro-ly-cua-ta-hoan-chinh]]).

---

## 6. Cỡ model & ngân-sách suy-luận M4

| Cỡ | Backbone | Vision | int4 ~ | M4 (ước) | Dùng cho |
|---|---|---|---|---|---|
| **nano** | ~0.4B | ~40M | ~0.4 GB | nhanh nhất, 30+ tok/s | robot/edge, Jetson |
| **mini** ⭐ | ~1.3B | ~100M | ~1.0 GB | ~15–25 tok/s | **mặc-định M4** |
| **plus** | ~3B | ~300M | ~2.3 GB | ~8–15 tok/s | chất-lượng cao nhất |

**Đề-xuất cho bản "cao hơn chút" (`plus` 3B):** vẫn **vừa M4 24GB** thoải-mái ở int4 (~2.3GB trọng-số + activation nhỏ vì không KV phình). Bù-lại chất-lượng hiểu-cảnh nhảy bậc. Nếu muốn cao hơn nữa (7B) → khuyến-nghị **train/serve bản đó trên GPU thuê**, còn edge/robot giữ `mini`/`nano` distill từ nó (teacher→student cùng họ). Chốt mặc-định: **`mini` cho M4, `nano` cho robot, `plus` khi cần đẹp.**

---

## 7. Rủi-ro & đánh-đổi trung-thực

1. **Vision-encoder from-scratch = ROI thấp nhất.** Ảnh-văn cần cực nhiều data. → **Bắt-buộc giữ ablation warm-start** (encoder mở đông-băng) làm baseline; chỉ theo đuổi from-scratch encoder nếu nó **không thua** baseline (đúng kỷ-luật đo-được của ta). Đây là chỗ duy-nhất tôi khuyên cân-nhắc lại chữ "from-scratch".
2. **Data video/temporal tiếng Việt khan.** → tự-nhãn screen-record bằng teacher-VLM (distill), sinh tổng-hợp, dịch.
3. **Realtime turn-taking + barge-in nặng phần hệ-thống** (đồng-bộ 2 vòng, hàng-đợi). → làm text trước (đúng lựa-chọn), voice sau.
4. **Quantize SSM trên MPS** cần kiểm kernel (int4 depthwise-conv + scan). → M1 smoke đo sớm.

---

## 8. Mốc (milestones)

- **M0 — Scaffold + `StreamState`** ✅ *(2026-09-05)*: `MambaStep` nạp-được weight từ `MambaBlock`, nuốt luồng giả 24fps O(1)/frame, smoke trên M4 (step≡scan Δ=6.7e-8; ~1.25ms/tok; state phẳng).
- **M1 — Ảnh tĩnh QA** ✅ *(2026-09-05)*: VFE từ-scratch (ViT-S + PerceiverResampler + projector) nối backbone; `MultimodalRMSH`. Smoke M4: grad tới conv-patch ✓, overfit 1 mẫu 6.18→0.006 ✓, **parity train==serve Δ=4.8e-7** ✓, sinh trễ token-đầu 41ms/~120 tok/s. Code: `src/nn_aivie3/{vision,model,rmsh}.py`, `scripts/rmsh_m1_smoke.py`.
- **M2 — Streaming video state** ✅ *(2026-09-06)*: bộ-nhớ thời-gian đa-frame chạy (hybrid recall 100%, parity Δ=1e-6, RAM phẳng T=8=T=32). **ĐÍNH-CHÍNH lõi (§4.1):** hồi-tưởng-chính-xác frame NGOÀI cửa-sổ attention (chỉ Mamba) ≈ ngẫu-nhiên → cửa-sổ attention phải phủ chân-trời hồi-tưởng. Bài học query: câu hỏi neo bằng `temp_emb` (không token chỉ-số rời). Code: `scripts/rmsh_m2_smoke.py` (+`_debug`,`_debug2`,`_vprobe`,`_knob`).
- **M2.5 — Chú-ý học-được** (§9, §4.1-5) ✅ *(2026-09-06)*: **KIỂM-CHỨNG from-scratch tại-chỗ** — cổng salience per-patch + curriculum: top-down 100% vs uniform 80.5% @nhiễu=10, bản-đồ chú-ý dồn 0.99 đúng mục-tiêu. Curriculum phá rào from-scratch (ca tốt-nhất 78%); cơ-chế đúng = gate giữ-vị-trí (resampler-nén kẹt bootstrapping). Đầu hành-động O(1) + halting: đã thiết-kế, dựng ở M3. Code: `rmsh_m25_{gate,curriculum,salience,diag,diag2,smear,crux,probe}.py`.
- **M3 — Vòng realtime perceive/decide** ✅ *(2026-09-06)*: runtime `agent.py` (perceive O(1) / decide anytime) chạy trên M4. **Parity train==serve Δ=4.5e-8**; **state RAM phẳng 19.5K** bất-kể T; latency steady-state **~10.7 ms/frame < 41.7 ms (24fps)**. Bài đoán-hướng-chuyển-động (cần bộ-nhớ): **grid-token giữ-vị-trí HỌC 98.8%** vs resampler-nén 22.7% (kẹt) → xác-nhận bài học M2.5 end-to-end. **Đánh-đổi lộ ra:** token giữ-vị-trí (16–36/frame) học tốt nhưng nặng; resampler (4/frame) rẻ nhưng mất vị-trí ⇒ lời-giải = **gate salience giữ-vị-trí (M2.5): ít token, đúng token**. Code: `src/nn_aivie3/agent.py`, `scripts/rmsh_m3_{smoke,learn}.py`. Còn lại: nối chữ/đối-thoại + screen thật (M3.5), tiếng (M4).
- **M3.5 — Giảm token giữ-vị-trí + kênh chữ** 🟡 *(2026-09-06, một phần)*: mục-tiêu ít token mà giữ vị-trí (realtime). **Trung-thực:**
  - ✅ **Đường thực-dụng đã có sẵn:** token spatial THÔ (lưới ít patch) = ít token + giữ vị-trí → M3 đã chứng-minh học được (grid 98.8%); lưới thô hơn = rẻ hơn, vẫn học miễn đủ độ-phân-giải → câu trả lời realtime NGAY.
  - 🟡→✅ **Chọn-token THÍCH-NGHI: GỠ ĐƯỢC bằng đúng 3 điều-kiện M2.5** (`rmsh_adaptive.py`). Bottom-up (motion) thử **4** cách (L1/softmax/hard-topk/Gumbel `rmsh_m35_*`) đều bootstrapping-fail. Nhưng với **(a) top-down điều-kiện-mục-tiêu + (b) curriculum + (c) head-ép-gate**: gate **dồn 1.00 vào mục-tiêu**, và **serve giữ top-8 (bỏ 78% token) vẫn 88%** (full 100%, ngẫu-nhiên 25%) → **giảm-token thích-nghi CÓ HỌC**. Nén cực-đại top-1/2 (49/52%) còn lossy cho task cần-vị-trí (4-góc); fine-tune head dưới top-k làm hỏng dense → cần **train-serve co-design**, để mở. **Chốt: adaptive-attention phải top-down, KHÔNG bottom-up.**
  - ✅ **Kênh chữ/đối-thoại CHẠY (`rmsh_m35_dialogue.py`):** agent XEM luồng "màn-hình" (chuỗi frame) → HỎI bằng chữ → ĐÁP bằng chữ qua runtime streaming (state ấm) — **100% held-out**, transcript đọc-được. Trợ-lý nhìn-màn-hình trả-lời text (text trước, đúng thứ-tự đã chọn); dùng cơ-chế M2 đã kiểm-chứng.
- **M4 — Tier tiếng** ✅ *(2026-09-06, 2 chiều)*: ✅ **audio-IN** (`rmsh_m4_audio.py`): waveform→STFT→32 khung×32 bin→token→StreamState; chirp lên/xuống/đều **acc 100%**, **parity Δ=9.5e-7**. ✅ **TTS-out** (`rmsh_m4_tts.py`): sinh chuỗi mel TỰ-HỒI-QUY có điều-kiện → **hướng-pitch khớp yêu-cầu 100%** (lên +0.08/xuống −0.08/đều 0.00), MSE 0.003 → đường "nói ra" khép **VÒNG TIẾNG 2 CHIỀU**. Còn: ASR→chữ + vocoder waveform (ISTFT/Griffin-Lim) ở quy-mô thật.
- **M5 — Robot VLA** ✅ *(2026-09-06)*: vision→action VÒNG KÍN (`rmsh_m5_vla.py`) trên runtime streaming — agent nhìn khung (mình+đích)→xuất hướng→di-chuyển→lặp. Imitation 87.9%; **vòng-kín chạm đích 72.7%/10 bước (tb 2.8 bước)** qua perceive O(1)/decide. Nền robot "mắt=cam→hiểu→hành-động". Nâng-cấp: action liên-tục (khớp/bánh), RL phần-thưởng, sim thật.

**Đo-được:** trễ token-đầu (ms) · frame/s nuốt-được · tok/s trên M4 · **trần RAM phẳng qua nhiều giờ** (chứng-minh chặn-trên) · chất-lượng: vi-image-QA, video-QA, hiểu-màn-hình.

---

## 9. Chú-ý HỌC-ĐƯỢC + quyết-định REALTIME (M2.5) — "biết nhìn cái quan-trọng"

Bối-cảnh yêu-cầu: đá bóng / đi đường → nhận luồng realtime và **quyết-định gần tức-thì**,
phải **chú-ý đúng thứ quan-trọng** theo ngữ-cảnh. Nối thẳng đính-chính §4.1: attention
là nơi ra quyết-định sắc-bén **nhưng ngân-sách CÓ HẠN (cửa-sổ/tính-toán)**. Vậy bài-toán
thật gồm 2 nửa:

> **(a) Rót đúng thứ quan-trọng vào ngân-sách attention có hạn** (chú-ý chọn-lọc, học-được).
> **(b) Ra hành-động ở O(1) từ state luôn-ấm, trong hạn-định (anytime).**

### 9.1 Bốn tầng chú-ý (rẻ → giàu), chi-phí đều CHẶN-TRÊN

| Tầng | Cơ-chế | Vai | Học thế nào |
|---|---|---|---|
| **L0 · cổng bottom-up** | diff chuyển-động / **bất-ngờ** (prediction-error) trên ảnh thu-nhỏ | phản-xạ: frame nào/đâu đáng encode | không cần nhãn (tự-giám-sát) |
| **L1 · resampler TOP-DOWN** | m query của Perceiver **được điều-biến bởi state LM/mục-tiêu** → trích token vùng **liên-quan nhiệm-vụ** (quả bóng, làn đường) | "nhìn theo ý-định" | end-to-end qua loss hành-động |
| **L2 · ngân-sách token thích-nghi + halting** | chấm điểm salience mỗi patch → giữ **top-m**; **dừng sớm** khi đủ tự-tin | chốt-trên độ-trễ (anytime) | ACT/early-exit, học kèm |
| **L3 · đầu HÀNH-ĐỘNG anytime** | policy-head nhỏ đọc `StreamState` **mỗi tick** → action ở O(1) | quyết-định tức-thì, luôn sẵn | imitation → RL (M5 VLA) |

Điểm mấu-chốt: **salience quyết-định token NÀO lọt vào cửa-sổ attention có hạn** (bài học
§4.1). Không phải "attention lâu hơn" mà "attention vào đúng chỗ".

### 9.2 Tín-hiệu học "cái gì quan-trọng" (không cần nhãn salience)

- **Task-driven (chính, khuyến-nghị):** salience **nổi lên từ loss hành-động/dự-báo**. Nếu
  nhìn quả bóng giúp chọn đúng hành-động → gradient tự đẩy chú-ý về quả bóng. Không cần
  gán nhãn "chỗ nào quan-trọng".
- **Surprise/prediction-error (phụ, bottom-up):** model dự-báo token-nén frame kế; vùng
  **sai dự-báo cao = bất-ngờ = đáng chú-ý** (xe tạt đầu, bóng đổi hướng). Nhãn-tự-do,
  rất hợp đá bóng/lái xe. Nối mạch "dự-báo/trôi" ta đã quen.
- **Reward/hậu-quả (M5):** RL cho hành-động ảnh-hưởng kết-cục.

### 9.3 Hai tốc-độ: phản-xạ vs cân-nhắc (bảo-đảm hạn-định)

- **Cung phản-xạ (fast):** L0→L3 bỏ qua xử-lý sâu → hành-động khẩn (thấy vật cản → phanh)
  trong 1 bước head, không chờ deliberation. 
- **Cân-nhắc (slow):** khi còn thời-gian, L1/L2 rót token giàu vào cửa-sổ attention để
  quyết-định tinh. **Ngân-sách tính-toán/tick** + early-exit ⇒ **trễ chặn-trên** dù cảnh phức-tạp.
- Ăn khớp kiến-trúc **2 vòng** (§3): vòng tri-giác giữ state ấm; vòng quyết-định đọc state
  bất-cứ lúc nào → "gần như realtime" là **thuộc-tính hệ-thống**, không phải may-rủi.

### 9.4 PoC đề-xuất (M2.5) — chứng-minh chú-ý học-được rẻ hơn mà đúng hơn

Bài "phản-xạ giữa nhiễu": khung có **1 mục-tiêu (bóng) + nhiều vật nhiễu**; mục-tiêu di-chuyển;
mỗi tick model phải xuất **hành-động đúng theo mục-tiêu** (hướng bóng / "sút"), **bỏ qua nhiễu**.
So **resampler đồng-đều** vs **resampler top-down (điều-biến bởi state)** trên đường cong
**độ-chính-xác vs ngân-sách-token/độ-trễ**. Kỳ-vọng: top-down đạt cùng độ-chính-xác với
**ÍT token hơn** (⇒ nhanh hơn) và trội rõ khi siết ngân-sách — chứng-minh "biết nhìn cái
quan-trọng" là đòn giảm-trễ, không chỉ tăng-điểm. Đo-được, chạy tại-chỗ M4.

**Kết-quả PoC (2026-09-06) — trung-thực:** đã chẩn kỹ (`rmsh_m25_*.py`). (1) mean-pool token
resampler **XOÁ vị-trí** → phải flatten/giữ-token (lỗi đã sửa); (2) encoder GIỮ vị-trí tốt
(flatten 1 vật = 100%); **(3) NHƯNG bài "định-vị màu-c giữa 10 nhiễu" KHÔNG học nổi trong
ngân-sách from-scratch tại-chỗ** — ngay *ca tốt-nhất* (flatten toàn grid + điều-kiện) cũng
~36% (≈chance 25%) ở nhiễu=10, chỉ bắt đầu học ở nhiễu=3 (51%). **Nhưng curriculum (nhiễu 0→10 + cosine-LR + 2500 bước)
PHÁ được rào:** ca tốt-nhất đạt **78.3% @nhiễu=10** ⇒ **thị-giác from-scratch HỌC ĐƯỢC
định-vị-giữa-nhiễu** — cam-kết from-scratch khả-thi cho tri-giác.

**(4) Blocker còn lại — bootstrapping của resampler nén:** với curriculum y-hệt, đường
**resampler (cả uniform lẫn top-down) loss đứng im ln(4), 23%≈chance** — KHÔNG học. Probe
(`rmsh_m25_probe.py`) cho thấy KHÔNG phải dead-path (tok biến theo ảnh, có đọc grid) mà là
**động-học tối-ưu**: init attention khuếch-tán → tok≈trung-bình-grid → vị-trí bình-quân mất →
head mù → gradient làm-sắc attention quá yếu → sụp hằng-số. Ca tốt-nhất học được vì head
**thấy thẳng** 36 token vị-trí.

**(5) LỜI-GIẢI + KIỂM-CHỨNG (2026-09-06):** thay resampler-nén bằng **CỔNG salience per-patch**
`g_p=σ(MLP([grid_p, target_emb(c)]))` (GIỮ vị-trí → kế-thừa học-được của flatten) + phạt thưa
L1·mean(g) = ngân-sách token. `rmsh_m25_gate.py` + curriculum, @nhiễu=10:

| | acc | mean-gate (ngân-sách) | **gate TRÊN patch mục-tiêu** |
|---|---|---|---|
| uniform (g không biết c) | 80.5% | 0.145 | 0.43 (dàn trải) |
| **top-down (g biết c)** | **100.0%** | 0.171 | **0.99 (dồn đúng mục-tiêu)** |

→ **CHÚ-Ý HỌC-ĐƯỢC được chứng-minh from-scratch, tại-chỗ M4:** top-down đạt 100% vs uniform
80.5%, **bản-đồ chú-ý dồn 0.99 vào đúng patch mục-tiêu** (uniform 0.43) — model **thật sự học
"nhìn cái quan-trọng theo ngữ-cảnh"**. Cơ-chế đúng = **gate salience giữ-vị-trí** (không phải
resampler-nén, vốn kẹt bootstrapping từ-scratch). Chốt: chú-ý cho thị-giác from-scratch nên
**gate token giữ-vị-trí + curriculum**, và top-down (điều-kiện mục-tiêu) là đòn tăng-đúng.

---

## 9b. Bộ-nhớ 2 CẤP + Tư-duy 2 HỆ (hướng nghiên-cứu, ánh-xạ nhận-thức người)

RMSH đã có sẵn đủ mảnh — chỉ cần tổ-chức thành 2 cấp / 2 hệ.

### Bộ-nhớ 2 cấp
| Cấp | Người | RMSH (đã có) |
|---|---|---|
| **Ngắn-hạn** | vài giây, nhỏ | **StreamState O(1)** (Mamba) + cửa-sổ attention |
| **Dài-hạn** | bền, lớn, truy-hồi | **RAG/kNN datastore** (FAISS 8M) + **token-mốc** ghi định-kỳ |

M2 §4.1 đã chứng-minh ranh-giới: state ngắn-hạn không giữ chi-tiết xa → phải truy-hồi dài-hạn (RAG).
"Củng-cố khi ngủ" = ghi episode nổi-bật (salience cao) từ state → datastore.

### Tư-duy 2 hệ
| Hệ | Tính-chất | RMSH |
|---|---|---|
| **Hệ 1 — phản-xạ** | O(1), tức-thì, muscle-memory | đầu hành-động đọc StreamState (M5); **đi/chạy/robot ở đây** |
| **Hệ 2 — suy sâu** | chậm, nhiều bước, có kế-hoạch | sinh chuỗi suy-luận qua LM + truy-hồi RAG + lặp |

**Cổng phân-xử:** mặc-định Hệ 1; leo Hệ 2 khi bất-ngờ/không-chắc/mới-lạ (tín-hiệu prediction-error §9.2).
Khớp 2 tốc-độ §9.3.

### "Luyện nhiều → phản-xạ" = distill Hệ 2 → Hệ 1 (mấu-chốt)
Tình-huống lặp nhiều: Hệ 2 giải (chậm) → **dạy Hệ 1 bắt-chước** (policy distillation / amortized inference)
→ lần sau O(1). Đi-lại/robot vốn cần kế-hoạch né vật-cản **quy về phản-xạ sau luyện**.
**PoC `rmsh_dual.py`:** mê-cung có tường (cần kế-hoạch = BFS = Hệ-2) → distill vào reflex-head (Hệ-1).
**Kết-quả trung-thực (2026-09-06):** BC ("chỉ xem thầy") reflex đoán ~50%/bước nhưng vòng-kín ~6-8%
(lỗi CỘNG-DỒN: sai 1 bước → lạc state lạ → kẹt). **DAgger** (luyện trên chính state reflex đi qua)
**cũng không cứu** ở cỡ đồ-chơi: vòng-kín kẹt ~8% << tham-lam 60%, vì **trần độ-chính-xác/bước ~46%**
(model tí-hon không học nổi *lập-kế-hoạch mê-cung* trong forward — đúng tường "spatial-reasoning
from-scratch khó" lặp lại).

**TINH-CHỈNH ý-tưởng (điều PoC dạy):** KHÔNG phải mọi di-chuyển quy về phản-xạ. **Locomotion/né-vật-cản-tức-thì
= phản-xạ** (học được, muscle-memory); **lập-kế-hoạch đường MỚI = Hệ 2** (đến khi luyện quen đúng tuyến đó).
Mê-cung-mới-mỗi-lần là bài NOVEL-PLANNING → không rút về phản-xạ thuần được — chính vì thế PoC fail đúng chỗ.
"Luyện → phản-xạ" áp cho tình-huống LẶP LẠI (quen), như người xây muscle-memory cho động-tác/tuyến quen,
không phải mê-cung bất-kỳ. Đường mở: (i) scale dung-lượng, (ii) cho reflex DÙNG StreamState xuyên-bước
(có bộ-nhớ, không re-derive kế-hoạch mỗi frame), (iii) phân-vai đúng: reflex lo locomotion, Hệ-2 lo tuyến.

---

## 9c. WORLD-MODEL — "trình giả-lập trong đầu" (keystone, đã kiểm dương-tính)

Thêm **đầu dự-báo (forward model)** cạnh StreamState: học **đoán quan-sát/state kế-tiếp**
(tự-giám-sát). Đây là mảnh THỐNG-NHẤT — bốn năng-lực mọc từ một world-model:

| Năng-lực | Dùng world-model | Ví-dụ |
|---|---|---|
| **Đoán trước (anticipation)** | ngoại-suy vài bước → hành-động theo TƯƠNG-LAI | thủ-môn căn đường bóng |
| **Tưởng-tượng lập kế-hoạch (Hệ 2)** | tua model tới trong "đầu" với hành-động ứng-viên → chọn tốt nhất (Dreamer/MuZero-lite) | **VÁ dual-process §9b**: reflex kẹt vì phản-ứng; imagination-rollout = suy-nghĩ-sâu |
| **Bất-ngờ = chú-ý** | **sai-số dự-báo** = mới-lạ → dồn chú-ý + leo Hệ 2 | §9.2 salience + cổng phân-xử |
| **Bộ-nhớ dự-báo** | state = đủ để dự-báo → nén đúng cái quan-trọng | nối §9b bộ-nhớ ngắn-hạn |

Phôi đã có: M2 (đoán token kế), **M4 TTS-out** (đoán khung phổ kế) = forward-model thu nhỏ.

**PoC `rmsh_worldmodel.py` (2026-09-06) — DƯƠNG-TÍNH:** "thủ-môn" — bóng bay ngang bị CHE-KHUẤT
đoạn cuối; muốn bắt phải ĐOÁN điểm rơi. Kết-quả: sai-số điểm-rơi phản-ứng 0.99 ô vs **world-model
0.04 ô**; **tỉ-lệ bắt 75% → 100%**. Mạng TỰ ngoại-suy quỹ-đạo xuyên che-khuất → hành-động theo
tương-lai dự-đoán. (Chạy được vì ngoại-suy tuyến-tính *tractable* — khác novel-planning §9b.)

**PoC `rmsh_imagine.py` (2026-09-06) — DƯƠNG-TÍNH MẠNH, VÁ ĐƯỢC §9b:** world-model tường-minh
học động-lực CỤC-BỘ (ô,tường,hành-động)→ô-kế = **100%** (bài dễ), rồi **IMAGINATION-PLANNING**
(BFS TRONG world-model, re-plan mỗi bước) giải mê-cung:

| Cách | Tới đích |
|---|---|
| reflex nguyên-khối (§9b) | ~8% (FAIL) |
| tham-lam | 64% |
| **imagination-planning (WM+BFS-trong-đầu)** | **100%** |
| BFS thật | 100% |

→ **Bài học chốt:** planning = **MÔ-HÌNH cục-bộ dễ-học + TÌM-KIẾM trong tưởng-tượng**, KHÔNG phải
phản-xạ nguyên-khối. Tách đôi (learn easy dynamics / search) là lý-do Dreamer/MuZero thắng, và
là cách Hệ-2 (mô-phỏng trong đầu) VÁ đúng thất-bại reflex §9b. Đây là trục hợp-nhất anticipation
(§9c thủ-môn) + planning (đây) + prediction-error-salience (§9.2) + escalation Hệ1↔Hệ2.
**PoC `rmsh_dreamer.py` (2026-09-06) — DREAMER GROUNDED TỪ THỊ-GIÁC, DƯƠNG-TÍNH:** encoder đọc
frame → latent CÓ CẤU-TRÚC mỗi-ô → giải-mã **tường/agent/đích ngay từ PIXEL** (tự-học): cảm-nhận
**tường 100%/ô · agent 100% · đích 100%**. Rồi imagination-BFS trên mê-cung ĐỌC-TỪ-MẮT (re-cảm-nhận
mỗi bước): **reflex 8% → Dreamer 100%** (= BFS thật; tham-lam 62%).

**BÀI HỌC HỢP-NHẤT (chốt cả chuỗi):** thứ giết monolithic-reflex là *bắt một forward tính đáp-số
TOÀN-CỤC*. Phá bằng **phân-rã**: perception = per-cell grounding DỄ (supervise từng ô) · world-model =
động-lực 1-bước DỄ · planning = TÌM-KIẾM trong tưởng-tượng. Mỗi mảnh DỄ → cả hệ giải được. Đây đúng
lý-do Dreamer/MuZero (và não) thắng: **tri-giác + mô-hình + tìm-kiếm, KHÔNG phải chính-sách phản-xạ nguyên-khối.**
Kế thật (khi lên GPU): latent dynamics tua NHIỀU BƯỚC trong latent (không re-render) + reward-head → full Dreamer/MuZero.

---

## 9d. CURIOSITY — động-cơ tự-thân (từ ghi-chú não 2016 của anh, đã kiểm dương-tính)

Hiện-thực ghi-chú "tò mò → khám phá → thoả-mãn" + "quy-luật mức-độ-biến-đổi" (đáp-ứng theo
thay-đổi/mới-lạ): **phần-thưởng nội-sinh = độ-bất-định/sai-số của WORLD-MODEL**. Agent thích đi
tới chỗ nó **chưa đoán được** → tự khám-phá; world-model càng học thì chỗ cũ hết "lạ" → dồn ra
biên chưa biết. Đây là mảnh **động-cơ** mà mọi PoC trước thiếu (đều học có-giám-sát).

**PoC `rmsh_curiosity.py` (2026-09-06) — DƯƠNG-TÍNH:** lưới 8×8, agent học world-model online;
chọn hành-động world-model KÉM TỰ-TIN nhất (tò mò) vs đi bừa. Kết-quả: world-model đạt 90% đúng
ở **tò mò @800 bước vs ngẫu-nhiên @1500** (~2× nhanh); phủ bản-đồ @800: 92% vs 80%. ⇒ tò mò
(bất-định world-model) khám-phá & học NHANH hơn → agent chuyển từ *được dạy* sang *TỰ khám-phá*.

**REWARD + BẪY (`rmsh_reward.py`, 2026-09-06):** trục GIÁ-TRỊ/AN-TOÀN từ "Bug Find Food" 2016 —
đích +thưởng (KTTT), bẫy -phạt (KTNH), bước -1. Imagination-planning chấm-điểm-thưởng (Dijkstra
có chi-phí bẫy): tới đích 100% · dẫm bẫy 0.41→**0.05**/tập · thưởng +2.7→**+5.3** so planner
chỉ-tới-đích. Ghép thẳng lên world-model. (Bước sau: LEARN bản-đồ bẫy từ trải-nghiệm.)

→ Nối trục: **world-model** (§9c) sinh ra cả (a) planning-tưởng-tượng (Hệ 2), (b) salience/bất-ngờ
(§9.2), (c) **curiosity** (§9d, động-cơ khám-phá), (d) **reward/an-toàn** (trên).

**AGENT TỰ-HỌC TÍCH-HỢP — CAPSTONE (`rmsh_agent.py`, 2026-09-06):** ghép TẤT-CẢ thành một agent
"Bug Find Food" sống. Curriculum: world-model học 'vật-lý bẩm-sinh' (tường 0→0.20) trước. Rồi THẢ
vào thế-giới LẠ (đích + 4 bẫy ẨN — không biết ở đâu). Mỗi bước: tri-giác → curiosity đi tìm đích →
KTTT/KTNH nhớ thưởng/bẫy từ trải-nghiệm → imagination value-iteration (tối-đa thưởng-đã-học +
curiosity, trong world-model) → hành-động. **Đường-cong tự-học 40 tập:** thưởng −9→**+4.8** · dẫm
bẫy 1.0→**0.0** · tới đích 100% (vs đi bừa ~−45). **KHÔNG được dạy gì** — tự khám-phá, tự nhớ nguy,
tự lập-kế-hoạch né. Đây là hiện-thân đầy-đủ của cả tài-liệu 2011-2018 + trục world-model. Kho ý-tưởng khác từ tài-liệu 2011-2018 chờ áp:
Âm/Dương = predictive-coding tường-minh · "nơ ron nhẹ" = habituation/IDF · Ping-Pong-cửa-sổ =
STDP+consolidation dual-memory · self-play/tự-vấn · neuroevolution.

---

## 10. Ánh-xạ tài-sản tái-dùng

| Cần cho RMSH | Đã có trong nn-aivie |
|---|---|
| Khối Mamba + RMSNorm | `scripts/train_mamba.py` (`MambaBlock`) → nn-aivie3 thêm `.step()` |
| Hybrid Mamba+Attn + RoPE/GQA/SDPA | `scripts/train_hybrid_scale.py` (`HybridLM`, `ScaledAttn`) |
| Vocab/BPE, checkpoint I/O | `src/nn_aivie/core/{vocab,checkpoint_io}.py` |
| Pipeline data pretrain, SFT-mix | `scripts/build_pretrain_pipeline.py`, `build_sft_mix.py` |
| Spec-decode (giảm trễ) | TK_058 |
| RAG-FAISS grounding | `checkpoints/knn_faiss_8m_d512.*`, combiner |

→ nn-aivie3 **không vứt gì**; nó **mọc mắt + đường-step** trên đúng backbone đã-thắng.

---

## §11 · Demo robot web: MẠNG THẬT lái (VIN chưng-cất BFS) — 2026-09-08

`scripts/rmsh_navpolicy.py` + `scratchpad/rmsh_robot.html` (artifact c5dce527). Thay lõi tìm-food
BFS(thuật-toán) → **mạng nơ-ron thật** để chứng-minh "net lái", chốt trục **luyện→phản-xạ** (Hệ2→Hệ1).

- **Kiến-trúc: Value Iteration Network** (conv học reward-map R rồi LẶP value-iteration K=40 vòng trong mạng;
  đọc-ra tại ô robot). = "lượng-giá N cấp" trong ghi-chú 2016 của user. **Chưng-cất từ BFS** (BC + DAgger) trên M4.
  67KB trọng-số nhúng JSON, chạy JS ~5ms/plan, **KHÔNG tìm-kiếm runtime**. Đạt **~66-71% hiệu-quả BFS**, mượt, 0/12 kẹt.
- **Kết-quả ÂM quan-trọng:** MLP feedforward toàn-map chỉ ~35% val (≈chance) — lập-đường toàn-cục cần LẶP
  truyền-giá-trị, không phải hàm nông ⇒ VIN. Reactive-net memoryless GIẬT/KẸT (đứng-im 60%, vòng-lặp-2-ô,
  hố-hút-V-giả) ⇒ sửa bằng: cam-kết-theo-ô + leo-gradient-V (thay pi) + quán-tính-hướng + novelty/habituation +
  chặn-quay-đầu. Sim-train PHẢI khớp raycast sim-chạy (NR=72/step0.05) nếu không lệch phân-bố.
- **Test:** chạy chính script HTML trong Node + DOM giả (headless food/moved%/mapped); parity JS↔Python Δ=0.
  Toggle 🧠VIN/⚙BFS so trực-tiếp. Chi-tiết: memory [[vin-robot-web-net-lai]].

### §11b · Mạng HỌC né bọ động (2026-09-08)
Nâng-cấp: input mạng **6 kênh** (+kênh "bọ" trong tầm cảm-nhận BSENSE=6). Thầy BFS né ô-bọ+lề (fallback).
VIN học **cả tìm-food lẫn né-bọ-động** → demo **bỏ vector-đẩy viết-tay**; né bọ giờ nằm TRONG quyết-định mạng.
Bọ gần → tái-quyết mỗi khung (né kịp). Đo trên HTML thật: **va-bọ ~0.2/1500 bước (gần sạch), food ~76% BFS,
mapped 95%, 0/14 kẹt**; parity JS↔Python 6-kênh Δ≈1e-4. ⇒ "gặp bọ→né vì mạng học" là THẬT (không còn hard-code).
