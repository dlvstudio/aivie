# TK_031 — VarOrderLM: mô-hình-chuỗi BIẾN-BẬC (PPM), nâng trần văn-mượt

| Field | Value |
|-------|-------|
| **ID** | TK_031 |
| **Status** | 🟢 DONE — guard 86/88/67 GIỮ; cluster-aware VarLM trên v10 (2026-07-09). |
| **Nguồn** | Tác giả duyệt hướng "composition-từ-trường / văn mượt tự nhiên" → bước nền: nâng W_seq bigram lên biến-bậc (nhớ cụm dài). |

## 1. Vì sao (trần trôi-chảy)

W_seq là order-1: sinh token kế chỉ nhìn 1 token trước → "mất trí nhớ ngữ cảnh sau mỗi từ" →
văn rời/lạc ("dời đô về" → quên "dời đô", đi lạc). VarOrderLM = PPM-style: dùng **chuỗi-khớp-
DÀI-NHẤT còn dữ liệu** (ctx 1..4), backoff mềm λ^(max−k). Thuần ĐẾM (Hebbian tầm xa hơn bigram),
không backprop, không rule — Luật 0 sạch.

## 2. Probe standalone (measure-first) — DƯƠNG mạnh nhất từng đo

`out/tk031_ppm_probe*.log` — corpus 167k câu/3.03M token, held-out 3097 vị trí, protocol log2-rank:

| Model | bits ↓ | MRR | top1 | top5 |
| :--- | :---: | :---: | :---: | :---: |
| bigram (ctx=1, ~W_seq) | 4.369 | 0.249 | 16.4% | 33.3% |
| **PPM ctx=1..4** | **4.155** | **0.288** | **20.6%** | **36.9%** |

Δ **−0.21 bits / +4pp top1** (so: gap-prior TK_020 −0.10; q·k −0.10). λ vô cảm (0.3–0.75 → 0.45).
**Prune ctx singleton k≥3 MIỄN PHÍ** (bits không đổi, bỏ ~58% context). Mật độ: 76% vị trí held-out
có ctx≥2 trong corpus → trần data không chặn.

## 3. Code

- `core/varlm.py`: `VarOrderLM` (tables ctx-len 1..4 → {next: count}; `scores()` trộn λ-backoff;
  prune; save/load pickle). `varlm_sidecar_path` = `<ckpt>.varlm.pkl`.
- `scripts/build_varlm.py`: đếm trên corpus (statements+doc+add+history), encode vocab checkpoint
  (OOV cắt chuỗi), prune → **network_v7_history.varlm.pkl 38MB** (1.72M ctx).
- `checkpoint_io.load_checkpoint`: attach `state.varlm` nếu sidecar có (log).
- Wire kênh micro (pattern giống order-2 bonus): `generate_autoregressive` A3 loop +
  `_wseq_fallback` (continuation). Cờ `ar_varlm_weight=1.0`; thiếu sidecar = no-op.
- `eval/metrics.next_token_lm(varlm_weight=)`: đo in-harness (ctx = prefix đầy đủ).

## 4. ĐO

- **Guard QA 86 / Para 88 / Syn 67 — GIỮ** với varlm ON + short-tail gate (`out/tk031_varlm.log`,
  `out/tk031c.log`). QA/sử prompts không đổi.
- **Ruler in-harness lần 1 VÔ HIỆU** (bug đo, honest): lượt OFF chạy trước làm `encode_tokens` GROW
  vocab với OOV held-out → lượt VARLM chấm thêm 277 vị trí OOV rank-tệ (n 2860 vs 3137) → so sánh
  không cùng tập. Fix: warm vocab trước → **đo SẠCH cùng n=3137** (`out/tk031b.log`):
  OFF 4.950 → w=0.3: 4.927 → **w=1.0: 4.915** (−0.035 bits, +1pp top5, đơn điệu theo w). Nhỏ hơn
  probe standalone vì in-harness readout ⊙ micro — readout gánh phần lớn tín hiệu; varlm chỉ điều
  chỉnh kênh micro. Dương, guard-safe.
- **Kênh varlm ĐÚNG**: varlm("Nhà Lý dời đô về") top-1 = "Thăng" 0.651; `_wseq_fallback` (có varlm)
  ra "Nhà Lý dời đô về **Thăng Long**" (đuôi sau mục ", và các" — attractor hư-từ khi ra khỏi vùng
  n-gram, vấn đề riêng của fallback không lưới dừng).

## 4b. HAI BUG ROUTING tìm được nhờ trace (fix trong TK_031)

1. **min_suf từ chối đuôi đúng-mà-ngắn**: R4 đòi `min_suffix = len(q_content)` (=5) → đuôi chuẩn
   "Thăng Long" (2 tok, pattern-completion TÌM ĐƯỢC) bị loại. Fix: chấp nhận đuôi ngắn KHI varlm
   (đếm-đã-học, ctx dài) xác nhận token đầu = argmax — gate bằng tín-hiệu-học, không rule tay.
2. **Nhánh opener-continuation (no-frame) thiếu entity-guard**: "Nhà Lý dời đô về" — "Nhà" trong
   lớp FB nhiễu → bị chộp vào route mệnh-đề-mở → conv-pool "phía trước", KHÔNG BAO GIỜ tới R4/varlm.
   Fix: đồng bộ entity-guard bước-2 TK_027 sang nhánh `q_ids == generated` (opener là thực-thể của
   prompt → không phải mệnh-đề mở). "nếu trời mưa" giữ nguyên lối cũ (entity_register demote FB-opener).

(Đo cuối sau 2 fix: `out/tk031d.log` — continuation + guard.)

## 5. Ghi chú kiến trúc

varlm subsume `ar_seq_order2` (k=2 table = order-2); giữ cả hai (đo). Kênh varlm chỉ là MỘT nhân
tố trong tích readout ⊙ micro ⊙ macro — không thay thế retrieval/agenda; nó nâng nền MƯỢT cho
mọi bước A3. Bước sau (đích "composition-từ-trường"): khung-slot (template lattice) + điền từ
trường — varlm là tầng duyệt-mượt của nó.

## 6. Review code (2026-07-08) — soát trước khi làm TK_033

Đọc lại toàn bộ code TK_031. **Kết luận: GIỮ nguyên, không sửa.** Thêm 1 test. Chi tiết:

### 6.1 P0 "train/serve tokenizer skew" — BÁC BỎ (chẩn đoán sai)

Nghi vấn ban đầu: `build_varlm.py` dùng `tokenize_punct()` còn ruler `next_token_lm` dùng
`tokenize()` → lệch. **Đào pipeline thì ngược lại — `tokenize_punct` MỚI ĐÚNG:**

- Substrate (`network_v7_history`) build bằng `ingest_statements(split_punct=True)` →
  `tokenize_punct` (xem `scripts/build_history_checkpoint.py`). Vocab có `.`/`,` là **token
  riêng**; W_seq nối token đã-tách-dấu.
- Lúc sinh, `varlm.scores(out[-max_ctx:])` nhận `out` = id sinh từ đồ thị = **đã tách dấu**.
- ⇒ VarLM PHẢI đếm trên chuỗi tách-dấu (`tokenize_punct`) để KHỚP kênh W_seq nó bồi + khớp
  chuỗi runtime. Đúng như code hiện tại.
- Bằng chứng: thử build lại bằng `tokenize()` → token 3.03M↓2.48M, ctx k=2 527k↓395k (dấu dính
  từ thành OOV → cắt chuỗi) = **MẤT dữ liệu**, không phải sửa skew.
- Ruler dùng `tokenize()` chỉ là artifact ĐO (bỏ qua vị trí OOV dấu-dính), KHÔNG phải đường serve.

### 6.2 P1 "scaling `_wseq_fallback` ≠ A3" — GIỮ nguyên (không đổi hành-vi-đã-đo)

Fallback cộng varlm trên micro ĐÃ normalize (peak 1, không ×`bmax`); A3 cộng trước rồi ×`bmax`.
Bất đối xứng có thật nhưng guard 86/88/67 (`out/tk031d.log`) đo trên code NÀY. Sửa thành ×`bmax`
= đổi cân bằng kênh qk/varlm ở fallback → **vô hiệu số guard đã đo**. Không sửa nếu chưa đo lại;
ghi làm nợ ablation nếu sau này muốn tinh chỉnh fallback.

### 6.3 Có làm

- Thêm `tests/test_varlm.py`: backoff ưu tiên ctx dài; prune chỉ đụng bậc ≥3 (không đổi core).

## 7. Nợ đã biết (ngoài phạm vi TK_031)

`test_best_match_open_query_requires_focus_entity` **đã xóa** (2026-07-09, TK_019 §7) — test QA
routing legacy, không thuộc substrate A3 multi.

## 8. Cluster-aware VarLM (2026-07-09, TK_035)

- `seg_graph.encode_tokens_with_clusters()` — ctx VarLM gồm token + cluster neuron đã kích hoạt
- `build_varlm.py --use-clusters`; `VarOrderLM(use_clusters=True)`
- Wire: `generate_autoregressive`, `eval/metrics.next_token_lm`
- v10 sidecar: `network_v10_histwiki_paragraph.varlm.pkl` (~80 MB)
- Ruler paragraph held-out @ w=0.50: **4.730 bits** (base 4.849)
