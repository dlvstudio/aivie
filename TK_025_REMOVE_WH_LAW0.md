# TK_025 — GỠ TOÀN BỘ WH-DETECTOR + rule đồng-tự (tuân Luật 0/0.0)

| Field | Value |
|-------|-------|
| **ID** | TK_025 |
| **Status** | 🟢 DONE (Claude 2026-07-06) — gỡ sạch, đo cost, ghi nợ. Chờ Phần 2 (sense neuron) + lớp frame-hlink sạch để phục hồi routing đúng-luật. |
| **Nguồn** | Tác giả chỉ 3 vi phạm Luật 0/0.0: (1) A1 quét CHUỖI completion ở inference + rule danh-từ; (2) WH rule; (3) rule đếm. "Mạng chỉ có token+edge+weight+kind; label chỉ để tìm & kích hoạt nơ-ron — KHÔNG xử-lý-chuỗi/nội-dung ở inference." |

## 1. Vi phạm đã gỡ

| # | Chỗ | Vi phạm | Xử lý |
|---|-----|---------|-------|
| 1 | `generate_hebbian._primary_in_name_context` + `len(entity)>=2` + `need=2 if…` | Quét chuỗi token của `completion` ở inference + **rule danh-từ** (proper vs common theo số token) + **rule đếm** | **Gỡ toàn bộ** — chỉ giữ kích-hoạt-bằng-token (fact chứa primary) |
| 2 | `parent_store.wh_slot_consistency_mode` | `if vocab.fold(cl.right) == "gì"` — **lexical hardcode** Luật 0 cấm thẳng | Gỡ hàm |
| 3 | `parent_store.wh_anchor_clusters` (+ `_frame_token_ids_in_query`) | Bộ **phân-loại câu-hỏi** (filler-diversity). Thống-kê-đồ-thị nhưng vẫn là rule phạm-trù "câu hỏi WH" | Gỡ hàm + mọi caller |
| 4 | `_readout_mode` + `ReadoutMode.KNOWLEDGE` + `_best_match_knowledge` + `retrieval_theme_weight` + `_theme_weight_map` | Nhánh if đổi hành-vi mạng theo phạm-trù WH (theme-first retrieval) | Gỡ enum + hàm + nhánh trong `best_match`/`rank_matches` |
| 5 | `best_match` wh_mode/wh_topic + `_pick_entity_candidate(wh_mode)` topic-first + `_entity_cands_filtered(wh_mode)` slot-consistency | Rẽ nhánh theo WH | Gỡ param, route 1 đường (curated→entity_cands→allp) |
| 6 | `_knowledge_extra_facts_eligible` | Cổng bồi-fact theo `wh_anchor_clusters` | Thay = **có thực-thể phân biệt** (`entity_register_ids`, idf/vị-trí — Luật 0) |
| 7 | `_prompt_is_open_continuation` | Phân-loại câu-hỏi vs mở-lửng qua `wh_anchor_clusters` | Rút gọn = chuỗi ≥2 token; việc "có phải câu hỏi" giờ do RETRIEVAL quyết (best_match trúng→đáp, trượt→pattern/W_seq) |

**Triết lý mới:** không phân-loại câu-hỏi ở inference. 1 đường: retrieval (`best_match`) chạy trên
KÍCH HOẠT token; trúng cha → đáp (+bồi fact nếu có thực thể); trượt → pattern-completion/W_seq.

Test: gỡ 4 test kiểm-thử cơ chế WH (`wh_anchor_clusters`, `topic_first`, `ReadoutMode.KNOWLEDGE`);
giữ `test_slot_consistency` (dùng `_parent_has_distinct_primary` — hàm còn sống). **146 pass.**

## 2. ĐO cost (network_v4_full, `out/tk025_nowh.log`) — trung thực

| Guard | Có WH (TK_024) | Gỡ WH (TK_025) | Δ |
| :--- | :---: | :---: | :---: |
| QA | 86 | **71** | −15 |
| Para | 88 | **62** | −26 |
| Syn | 67 | **50** | −17 |

Định tính (6 prompt):
- ✅ giữ: "Nhà Lý có gì"→Phạm Tu/Lý Nhật Tôn; "Nguyễn Huệ"→đại phá quân Thanh; "Mây"→định nghĩa đúng.
- ❌ **drift chéo thực-thể quay lại**: "Nhà Trần là gì"→"Nhà Lý thành lập 1009" (slot-consistency từng chặn — TK_021 ca 4).
- ❌ **degenerate**: "Lý Thái Tổ là ai"→", và các . , và…" (best_match trượt → W_seq rác).
- ❌ đồng-tự "Lý thuyết" lọt (A1-gate đã gỡ; đây là việc Phần 2 sense-neuron).

## 3. Sổ nợ TK_025 (Luật 3.5 — không bịa lại rule, ghi giá)

- **−15/−26/−17 guard** là giá của việc bỏ WH-routing. Chấp nhận theo lệnh tác giả (Luật 0 > metric).
- Routing "câu hỏi cần tri-thức" hiện **không có cơ chế thay thế** — probe cho thấy lớp hlink
  (FRAME_BRACKET) hiện là PMI nhiễu (3235 opener, "Hỏi:" không cạnh), CHƯA đủ sạch để route.
- **Đường phục hồi đúng-luật (không bịa rule):**
  1. **Phần 2 (kế tiếp):** sense-neuron tách "Lý"(triều)≠"Lý"(lý-thuyết) → trường kích thích không
     chạm fact nghĩa-khác → đồng-tự "Lý thuyết" tự biến mất KHÔNG cần rule.
  2. **Lớp frame-hlink SẠCH** (mine mức ĐOẠN: Hỏi:→Đáp:, khung định-nghĩa thật) → routing tri-thức
     nổi từ KÍCH HOẠT nơ-ron ngang, không từ rule phạm-trù. Trùng một phần nhánh B (TK_024).
- Config flags chết (giữ, harmless): `retrieval_theme_weight`, `wh_min_filler_diversity`,
  `knowledge_extra_wh_min_filler`, `slot_consistency_all_frames`, `topic_first_*`.

---

## P2 — Cộng-hưởng CO_ACTIVATION cho đồng-tự (2026-07-07) — ĐO TRỘN, default OFF

Tác giả chọn hướng "cộng-hưởng co-activation" (thay sense-neuron — probe cho thấy **không có
sense cho "Lý"**: tên riêng, nghĩa lý-thuyết nằm ở cụm ghép "lý thuyết"; cụm này tồn tại nhưng
segmentation không áp ổn định → "Lý" trong fact lý-thuyết ra token trần).

**Cơ chế (Luật 0 sạch, không rule/chuỗi/sense):** xếp hạng pool gia-đình theo CỘNG HƯỞNG với
TRƯỜNG co-activation của query, không phải chứa-token.
- `query_coactivation_field`: lân cận CO_ACTIVATION của token query, gộp (idf-weight, chuẩn hoá
  per-seed). LỌC: bỏ chính token query (tránh vòng fact-đồng-tự chứa "Lý" cộng hưởng giả); bỏ
  dấu-câu; bỏ hư-từ (gidf < 3.5 — `,`/`là`/`của` đồng-kích-hoạt mọi thứ → nhấn chìm ngữ-cảnh).
- `parent_coactivation_resonance`: trung bình idf-weighted field trên token fact.
- Xếp: **CỔNG resonance ≥ 0.7×max RỒI xếp stimulus** (stimulus range áp đảo: "Lý thuyết điện từ"
  lặp "Lý" → stimulus 650 >> dynasty 392; nhân multiplier không cứu → phải gate).

**ĐO (`out/tk025_p2gate.log`):**
- ✅ "Nhà Lý có gì" → "Lý Thường Kiệt rút quân . **Lý Thái Tổ dời đô Thăng Long 1010**" — physics
  "Lý thuyết điện từ" **biến mất**, dynasty thật lên (đúng ý TK_024).
- ❌ "Mây là gì" → **mất** định nghĩa "Mây bao gồm giọt nước" (gate giữ Mây-địa-lý theo nghĩa TRỘI
  corpus, bỏ Mây-khí-tượng).
- ❌ "Triều Lý đã làm gì" → nhiễu "**Triều Tiên**/Trung Quốc" (đồng-tự "Triều" khác lọt gate).
- Guard **71/62/50 — KHÔNG đổi** (P2 chỉ đụng đường bồi-fact, trung tính guard).

**Chẩn đoán trung thực (chống ảo tưởng):** field co-activation phản ánh nghĩa **TRỘI trong
corpus**, KHÔNG phải ý query → không phân biệt được Mây-mây vs Mây-núi. **Không có gate_frac nào
sửa cả Lý lẫn Mây**: hạ gate giữ Mây thì physics-Lý sống lại (resonance physics 0.030 vs dynasty
0.047, max 0.052); nâng gate bỏ physics thì bỏ luôn Mây-định-nghĩa. Đây là **giới hạn cùng loại
P0/B0** — đồng-hiện/đồng-kích-hoạt mức này quá nhiễu để tách nghĩa sạch.

**Quyết định:** `knowledge_coact_resonance = False` (default OFF) — không ship regression "Mây".
Cơ chế + flag + doc giữ làm **nợ đo-được**; bật nếu chấp nhận đánh đổi (sửa Lý, hỏng Mây/Triều).
Fix sạch thật cần: (a) per-fact sense-binding (bind_sense_resonance — nhưng "Lý" vô-nghĩa-từ-điển),
hoặc (b) phrase-precise retrieval (bỏ mở token-trần — mất recall vua/tướng, TK_024 tension).
