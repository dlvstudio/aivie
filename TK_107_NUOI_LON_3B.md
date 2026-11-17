# TK_107 — Nuôi lớn thân 1.3B → ~2,6B (tổng ~3,4B) · ngữ cảnh 2K → 8K · lộ trình WSD 20 → 50 → 100 tỷ token

Quyết định của user (2026-10-06): lên ~3B + ngữ cảnh dài; **phương án A — chạy theo bậc** (mốc 20 tỷ token, đo, rồi mới
quyết định kéo dài lên 50/100 tỷ). V5 đọc ảnh tạm dừng (phải train lại trên thân mới; dữ liệu V5 giữ để dùng lại).

## 1. Kiến trúc

| | Hiện tại (v5) | Mục tiêu |
|---|---|---|
| Thân | 1,277B · d=2048 · 48 khối (42 Mamba + 6 attention @7,15,…,47) | **2,488B** · d=2048 · **96 khối (84 Mamba + 12 attention @7,14,23,30,…,94)** |
| Ngữ cảnh | 2 048 · RoPE base 1e4 | **8 192** · RoPE base **1e6** (ABF) |
| Tokenizer | bpe_v1 (32 002) | giữ nguyên |

**Chèn khối đồng nhất** (`scripts/expand_depth.py`, kiểu LLaMA-Pro/SOLAR): khối cũ i → vị trí 2i; khối mới ở 2i+1 chép trọng
số từ khối cũ cùng loại gần nhất rồi **đặt lớp ra = 0** (Mamba `out_proj`, attention `wo`) → `x + khối(x) = x`.
Khối mới là attention khi i ≡ 3 (mod 8) (chèn giữa 2 attention cũ). **Kiểm trên thân thật (chat_v2): sai lệch logit 0,00 ·
argmax 48/48** → bắt đầu train tiếp không sốc loss. cfg mới: `layers=96, attn_layers=[…], rope_base, new_blocks=[1,3,…,95]`.

## 2. Trình train (`scripts/train_hybrid_scale.py`, đã nâng)

- **Nhiều GPU**: `torchrun --standalone --nproc_per_node N` (DDP, `no_sync` khi tích luỹ grad, rank 0 ghi log/lưu/đo).
- **Lịch WSD** (`--sched wsd --decay-frac 0.1`): ấm → giữ lr → hạ tuyến tính ở phần cuối; lưu `*_stable.pt` (kèm optimizer)
  ngay trước khi hạ → **kéo dài** = `--init-from *_stable.pt --tokens <lớn hơn>`.
- `--train-new-only` (chỉ khối mới học), `--reset-step` (pha mới / đổi tập tham số → optimizer mới), `--rope-base`,
  `--data "dirA:0.5,dirB:0.3,…"` (trộn nhóm theo trọng số), dựng model đúng cfg của ckpt (`build_from_cfg`).
- **Lỗi đã bắt khi thử**: cfg lưu `rope_base=0` (tham số mặc định đè giá trị thật → RoPE ra inf/nan) — đã sửa.
- Thử trên M4 (model tí hon): train từ đầu → chèn khối (khớp 0,00) → DDP 2 tiến trình chỉ khối mới (khối cũ đổi 0 tham số)
  → bản ổn định @ đúng bước → kéo dài từ bản ổn định → mở toàn bộ → trộn nhóm ở ngữ cảnh 1K: **đều chạy đúng**.
- `hybrid_infer.py`, `vlm_1p3b.py` nạp được cả cấu hình mới (rope_base theo khối). ~~Còn lại (P3): expert/LoRA/đọc ảnh gắn
  cứng `POS`~~ → đã sửa (§6).

## 3. Dữ liệu (P1)

- **Bộ chấm giá trị giáo dục** (`scripts/edu_classifier.py`): Gemma-31B chấm 40K đoạn web Việt (thang cộng điểm 0–5 kiểu
  FineWeb-Edu, Việt hoá) → model kiểu fastText (túi từ + cặp từ băm, PyTorch thuần) → giữ điểm ≥ 2,5.
- **Dựng shard** (`scripts/build_cpt_data.py`): đọc luồng HF → chuẩn hoá **giữ xuống dòng** → lọc heuristic (bản cũ gộp mọi
  dòng nên lọc "lặp dòng" vô tác dụng) → chấm giáo dục → khử trùng lặp chính xác → tokenize song song (~1M token/s/nhân,
  đã đo) → văn bản ≥4K token sang nhóm `long`; mỗi nguồn 1 tiến trình (shard `shard_<nguồn>_<số>.npy`).

| Nhóm | Nguồn (đã kiểm truy cập) | Mốc 1 |
|---|---|---|
| vi_web | FineWeb-2 vie (ODC-BY) · HPLT 2.0 vie (CC0) · [CulturaX: gated, cần user đăng nhập] | 9 tỷ |
| long | văn bản ≥4K token từ web/wiki | 2,4 tỷ |
| vi_wiki | Wikipedia vi · Wikisource vi | 0,8 tỷ |
| en_edu | FineWeb-Edu sample-100BT | 3 tỷ |
| math | FineMath-4+ | 1 tỷ |
| code | github-code-clean (giấy phép thoáng; Stack-Edu chỉ có mã blob, không có nội dung) | 1,5 tỷ |
| ôn lại | pretrain_vi cũ (18,8 tỷ, HF `nn-aivie-pretrain-vi`) | trộn 9–12% |

## 4. Lộ trình mốc 1 (~20 tỷ token) — `scripts/box_run_p2_cpt.sh` (8×H100)

| Pha | Học | Ngữ cảnh / RoPE | Token | Lịch |
|---|---|---|---|---|
| A1 | chỉ khối mới | 2K / 1e4 | 4 tỷ | lr giữ 3e-4 |
| A2 | chỉ khối mới | 8K / **1e6** | 6 tỷ | lr giữ 3e-4 |
| B | toàn bộ | 8K / 1e6 | 10 tỷ | 1,5e-4, hạ 10% cuối → `cpt_b_stable.pt` để kéo dài |

Lô toàn cục ~1M token/bước; lưu mỗi 500 bước, **sao lưu HF mỗi 1 000 bước** (`--hf-repo`). Đầu kịch bản: kiểm chèn khối
khớp, đo nhanh tốc độ/bộ nhớ ở 2K và 8K. Cuối: `eval_longctx.py` (loss theo vị trí + "kim" tiếng Việt 1K–8K) so 1.3B vs 2,6B.

Sau mốc 1 (P3, 1×H100 ~1,5–2 ngày): SFT chat, train lại 3 expert, vòng thầy, đọc ảnh (công thức V4 + dữ liệu V5, 12 ô gốc).

## 6. P3 — code cho thân 96 khối (2026-10-11, làm trên Mac trước khi thuê box)

- **Điểm cắm expert = 12 khối attention** `[7,14,23,30,39,46,55,62,71,78,87,94]` (user chọn 12, không phải 6), mỗi FFN
  **nửa cỡ** → tổng tham số expert giữ nguyên (3 bộ = 755M). `expert_ffn_1p3b.set_pos(cfg)` sửa `POS` TẠI CHỖ (mọi
  `from … import POS` thấy cùng list); `hid_eff()` quy `--hid` (tính theo 6 điểm) ra cỡ thật. Thân 1.3B cũ vẫn ra 6 điểm,
  cỡ cũ → v5 / vlm_v4 vẫn nạp được UI.
- Mọi nơi nạp thân đi qua `build_from_cfg` (expert, VLM, SFT). **Lỗi nguy hiểm đã bắt**: `train_hybrid_sft.py` dựng
  `HybridLM` không truyền `rope_base` → SFT thân 8K chạy RoPE 1e4, âm thầm hỏng ngữ cảnh dài; và đọc `attn_layers`
  dạng chuỗi (cfg mới là list) → sập.
- SFT: `--tok-budget` (lô vi mô ≤ ngân sách token sau đệm, giữ mẫu dài nhất), `--optim adamw8bit`, `--device`.
- Dữ liệu `build_sft_p3.py` (bản 2 — thân mới CHƯA qua SFT nào nên gộp ĐỦ mọi đợt SFT của 1.3B): sft_mix 85.174 +
  phần riêng sft_v2 15.168 (bám nguồn, không biết, toán) + DailyDialog Việt 10.323 + multiturn 8.665 = 119.330 ngắn
  (50,9M token, p50 256) + **6.000 mẫu dài 2,6K–8,2K** (4,8%, 33,7M token) → ~85M token/epoch, SFT 2 epoch. Mẫu dài tự sinh từ viwiki:
  bó nhiều bài, hỏi "X là gì?" (→ "Theo tài liệu k, …"), "Tài liệu nào nói về X?", và X không có trong bó
  (→ "không có thông tin", dạy nhận không biết).
- `smoke_p3.py` trên M4 với `hybrid_2p6b_a1_bf16.pt`: điểm cắm đúng · 3 bộ expert mới khớp thân 0,00 · học chỉ expert
  thân không đổi · Runner từng bước có expert khớp forward (fp32, 7e-7, 39/39) · SFT tí hon (RoPE 1e6, mẫu dài) loss giảm.
  **Bẫy Mac**: Mamba thuần PyTorch + chuỗi 8K mất cả phút/bước; MPS hết bộ nhớ KHÔNG ném lỗi mà cho loss 0,000 giả.
- `box_run_p3.sh` (1×H100): smoke → SFT (OOM thì tự chuyển AdamW 8-bit) → 3 expert mới (`trainGEN`, 3.000 bước) → chấm
  (eval_v1, easy20, văn phòng, SGK, toán, ngữ cảnh dài) so với v5. Dấu `.done` từng bước, sao lưu HF `nn-aivie-2p6b-p3`.
  `pack_p3_data.sh` đóng gói dữ liệu trên Mac → HF dataset (box tải). Vòng thầy (P3.3) và đọc ảnh (P3.4) chạy sau.
- Dữ liệu cũ đã mất trên Mac: `data/sft_chat/chat_mix.jsonl` (prep giờ lấy chat từ sft_v2), `easy20.jsonl` (dựng lại từ
  kết quả chấm v5), `data/vlm` (dựng lại bằng `gen_vlm_data.py` trên box khi tới P3.4).

## 5. Việc của user trước khi chạy

1. **P1**: thuê box **1×H100, ổ ≥500GB, ≥32 nhân CPU**; `hf auth login` (token WRITE, tự dán trên box) → `scripts/box_run_p1_data.sh`.
2. **P2**: thuê **8×H100 SXM, ổ ≥500GB** (kiểm không bị bóp công suất: `nvidia-smi -q -d POWER`); `hf auth login` →
   `scripts/box_run_p2_cpt.sh`. Ước tính mốc 1: ~3,5 ngày, ~$1 700–2 200 (giá thuê thay đổi).
3. (Tuỳ chọn) chấp nhận điều khoản CulturaX trên HF để thêm nguồn tiếng Việt.

## 7. Mốc 20 tỷ + đổi thứ tự P3 (2026-10-12)

- **P2 XONG** 2026-10-11 23:39 UTC. Val cố định (bits, 2K): thân chèn khối 3,47 → A1 2,73 → A2 2,67 → **B 2,31** (mọi nhóm giảm; code
  3,69→0,98, toán 3,77→1,88, web Việt 3,05→2,21). 8K: A2 2,60 → B 2,22. Kim tiếng Việt 4K **20/24**, 8K **14/24** (1.3B: 0/24, 0/24).
  Trắc nghiệm 44 câu: 1.3B 36 → 2,5B 38. User: dừng ở 20 tỷ (kéo dài sau từ `cpt_b_stable.pt` nếu cần). Đã xoá A1/A2 trên box.
- **Dữ liệu đọc ảnh mất** cùng box TK_106 (kịch bản cũ không sao lưu dữ liệu) → dựng lại trên box P2 (`box_vlm_data.sh`: A = CPU
  trong lúc train, B = 3 thầy Gemma GPU 1–3, nhìn ảnh chia 3 phần theo crc32) + `box_vlm_code.sh` (ảnh code). Đẩy HF công khai
  `nn-aivie-vlm-data` (trừ ảnh gốc bộ ngoài; trang báo nhúng ảnh COCO → chỉ khi user đồng ý).
- **Ảnh code (user yêu cầu)**: `gen_screens.py` thêm code cài 1 lỗi logic (`mutate`), trình soạn + TERMINAL có traceback trỏ ĐÚNG dòng
  (lỗi hợp với nội dung dòng: chia cho biến → ZeroDivisionError, khoá chuỗi → KeyError…), ảnh chụp màn hình (`--photo-frac`, có viền
  để không cắt chữ), lưu nguyên văn code (`ref`) cho giám khảo. Thử trên M4: V4 1.3B đọc "dòng 3" ra dòng 2, bịa ValueError.
- **Đổi thứ tự (user)**: P3.1 SFT → P3.2 expert → P3.5 chấm → **P3.4 đọc ảnh** → **P3.3' vòng thầy CHUNG chữ + ảnh** (`vlm_loop.py`:
  ~30% hội thoại có ảnh, trong đó 35% code; thầy nhìn ảnh + ghi chú nội dung chính xác, chấm khắt khe chữ/số/code, viết bản sửa)
  → buildV5/trainGEN/DPO cho expert chữ + `fix_<tag>.jsonl` train tiếp phần đọc ảnh. Lý do: vòng thầy sửa expert SAU khi có phần
  đọc ảnh → không lệch, không phải train lại; và đọc ảnh cũng được thầy chấm–sửa (trước đây chỉ có dữ liệu thầy viết + chấm đo).
