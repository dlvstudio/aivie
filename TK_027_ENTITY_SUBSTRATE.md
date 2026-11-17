# TK_027 — SUBSTRATE THỰC-THỂ tên-riêng (title bài) — giải đồng-tự "Lý"

| Field | Value |
|-------|-------|
| **ID** | TK_027 |
| **Status** | 🟢 BUILT + ĐO net-DƯƠNG — default ON. Sửa flagship "Lý thuyết điện từ" + Syn 50→67. Còn dư (trích-dẫn "Nhà xuất bản", "Mây" theo primary). Ghi nợ. |
| **Nguồn** | Tác giả: "tên riêng/lịch sử cần DATA — sao không bổ sung?" → đúng: đồng-tự "Lý" bị chặn ở tầng fold (TK_026), cần **nhãn thực-thể**. `statements_doc` đã có `title` = thực thể → dùng làm substrate. |

## 1. Vì sao (chốt chuỗi TK_024→027)

- Đồng-tự "Lý"(triều) vs "lý"(thuyết) KHÔNG tách được ở token/fold/sense: "Lý" case-fold về "lý"
  → gọi mọi nghĩa lý* (TK_026 net-âm). Cần biểu diễn NGHĨA cho tên-riêng — từ-điển không có.
- `statements_doc.jsonl` có **`title`** = tên bài wiki = THỰC THỂ. Gom câu theo title → nơ-ron
  `Entity:title` hồ-sơ SẠCH, **key theo cụm title (không fold)** → thoát va-chạm.

## 2. Probe raw (measure-first) — tách sạch

Hồ-sơ theo title, cosine: `Lý Thái Tổ ~ Lý thuyết dây = 0.085` (gần vuông) vs `~ Lý Thường Kiệt
= 0.385`. Fact "Lý thuyết điện từ" khoá **Lý-thuyết-dây=2.28** vs dynasty ≤0.14 (~16×).

## 3. Code

- `training/ingest_entities.py`: mỗi title (≥5 câu) → `Entity:title` + cạnh CO_ACTIVATION tới top-60
  token đặc-thù. **Trọng số = sublinear-TF (1+log count) × fold-idf, L2-normalize** — mấu chốt:
  fold-idf hạ token TÊN ("Lý" ở nhiều bài) + sublinear-TF cân token tần-cao vs đặc-thù ("Uẩn"/"dây").
- `core/vocab.py`: `entity_id` + prefix `Entity:` (non-surface).
- `scripts/build_entity_checkpoint.py`: network_v4_full → **network_v5_entity.pkl** (786 entity, 43k link).
- `core/parent_store.py`: `entity_anchor_field` (neo tokens → top thực-thể cộng hưởng → gộp hồ-sơ) +
  `parent_field_dot` (dot-product, không chia độ dài).
- `generate_hebbian._rank_extra_fact_candidates`: **CỔNG NEO** = neo từ câu đáp CHÍNH (base_out) →
  trường thực-thể → loại fact dot ≤ 0.1×max (khác domain), số sống xếp stimulus. Cờ
  `knowledge_entity_resonance` (default ON; checkpoint không có Entity: → no-op an toàn).

## 4. Probe trên MẠNG (network_v5_entity) — tách sạch, sau 3 lần sửa

Sau khi sửa: (v1) lưu count thô → "Lý" áp đảo; (v2) idf-weight vẫn count "Lý" 80× lấn; (v3)
**sublinear-TF + fold-idf** → "Lý thuyết điện từ" → Lý-thuyết=1.18, **dynasty=0.00**; "Lý Nhật Tôn"
→ dynasty 1.6-1.7, **physics=0.00**. Tách hoàn toàn.

**Bẫy đo được:** query "Nhà Lý" tự-neo KHÔNG được (down-weight token tên → query mất tín hiệu:
top toàn "Tiên đề/Thuật toán" như "Mây"). "Nguyễn Huệ"→Quang Trung=2.70 (đúng, khi tên đặc thù).
→ Neo bằng **câu đáp chính** (base_out), không bằng query.

## 5. ĐO end-to-end (`out/tk027_entity.log`)

| Prompt | Kết quả |
| :--- | :--- |
| **Nhà Lý có gì** | ✅ "Nhà Trần thay nhà Lý . **Lý Nhật Tôn kế vị→Lý Thánh Tông** . Đặng Huy Ruận Lý thuyết đồ thị…" — physics "điện từ Maxwell" **BIẾN MẤT**, dynasty lên. Dư: trích-dẫn sách "Nhà xuất bản" lọt. |
| **Triều Lý** | ✅ "Phạm Tu khai quốc nhà Tiền Lý…" — physics gone; dư "Lý thuyết chọn mẫu". |
| **Nguyễn Huệ** | ✅ đúng (Quang Trung context). |
| **Mây** | ❌ mất định nghĩa "Mây bao gồm…" — neo=primary "Đỉnh Mây Bạc" (địa-lý) → theo primary sai. |

**Guard: QA 71 / Para 62 / Syn 67** — Syn **50→67** (lên lại mức WH-era), QA/Para giữ. **Net-DƯƠNG.**

## 6. Đọc trung thực (chống ảo tưởng)

- ✅ **Đây là cách DUY NHẤT tách được flagship "Lý thuyết"** (P2/TK_026 đều net-âm) — vì nhãn
  thực-thể key-cụm thoát va-chạm fold. Substrate title đã có, chỉ cần dùng.
- ⚠️ Dư 1: fact trích-dẫn sách ("… Nhà xuất bản Khoa học Hà Nội") lọt — chứa token nhà-xuất-bản
  cộng hưởng nhẹ trường neo. Siết gate risk bỏ fact dynasty tốt. Ghi nợ.
- ⚠️ Dư 2: "Mây" theo primary — **neo=câu-đáp-chính, primary sai thì gate khuếch đại sai**. Đây
  thật ra là lỗi PRIMARY retrieval (best_match chọn "Đỉnh Mây Bạc" thay định nghĩa) — nợ WH-removal
  (TK_025), không phải gate. Gate trung thành với primary.

## 7b. BỔ SUNG DATA sử VN + kiến thức (tác giả duyệt "bổ sung data sử đi, cả kiến thức khác")

`ingest_hf_vi.py` +CLI `--start-offset/--max-articles` → kéo **80k câu / 2680 bài MỚI** Wikipedia
VI (offset 1039+) vào `statements_doc_add.jsonl`. **Trúng đích**: "Nhà Lý" (273 câu), "Văn Miếu –
Quốc Tử Giám" (142), "Nhà Trần" (261), "Hoàng thành Thăng Long" (207) — chính các bài B0/§KQ1 đo
là VẮNG.

**Fix hạ tầng:** `ingest_statements` thiếu `defer_index` → rebuild index O(#parents) MỖI câu (80k×144k
= treo). Sửa dùng `seg.begin_batch()`+`defer_index=True`+`end_batch` (lợi cả `train_v3`).

`scripts/build_knowledge_checkpoint.py`: network_v4_full → **ingest_statements**(80k, priority 0.6 →
PARENT đáp được) + **ingest_entities**(base+add, 2925 entity) → **network_v6_knowledge.pkl** (parents
64.5k→142.6k, 352MB).

**ĐO (`out/tk027_v6.log`):**
- 🎯 **"Văn Miếu là gì" → "Văn Miếu là trường đại học đầu tiên Việt Nam . thờ Khổng Tử . …1076"** —
  ĐÁP ĐƯỢC (ca flagship TK_024 trước = data-gap, nay lấp).
- "Mây" giàu hơn ("Mây ti… castellanus và floccus"); "Triều Lý" → "hoàng tộc Lý sang Cao Ly lập nghiệp".
- **Guard QA 71 / Para 62 / Syn 50** — Syn **67(v5)→50** (dilution 78k parent wiki chen synonym; = mức
  baseline no-WH, KHÔNG thủng dưới). QA/Para giữ (two-tier bảo vệ curated).
- Dư: "Lý thuyết đồ thị"(sách) vẫn lọt "Nhà Lý"; "Triều"→Triều Tiên; "Nhà Lý xây gì"/"Lý Thái Tổ ai"
  degenerate (nợ PRIMARY best_match hậu-WH — trực giao).

**Đọc trung thực:** data-addition là **thắng lớn về KIẾN THỨC** (Văn Miếu + sử + mây answerable, đúng
mục tiêu tác giả) với **giá dilution Syn 67→50** (về baseline). Xác nhận: nhãn+data thực-thể là enabler
đúng. Cân dilution: hạ priority wiki hoặc nạp chọn lọc hơn — nếu tác giả muốn giữ Syn 67.

## 7c. BƯỚC 2+3 (2026-07-07) — sửa rác PRIMARY + siết precision + KHÔI PHỤC GUARD

**Bước 2 — PRIMARY ra rác (", và các"):** chẩn — best_match KHÔNG trượt (tìm đúng "Lý Thái Tổ dời
đô"). Rác do `_is_frame_opener_continuation` over-fire: "Lý" trong lớp FB nhiễu + sau gỡ WH
`_prompt_is_open_continuation` luôn True → route nhầm continuation → W_seq rác. FIX: chặn khi
`content[0]` ∈ `entity_register_ids` (demote FB-opener → "nếu" không phải entity, "Lý"/"Nhà" có).

→ **KHÔI PHỤC GUARD: 71/62/50 → 86/88/67** (WH-era gốc) trên v6! Nhiều ca QA/Para/Syn cũng bị bug
này (80k token wiki thành FB-opener-nhiễu). **Bước 1 (dilution) tự giải** — không phải dilution.

**Bước 3 — siết dư (citation/Triều-Tiên):** chẩn dot-neo v6 — cổng base_out KHÔNG phân biệt (dot nén
86-100%; "Triều Lý" primary nhiễm Hàn-Quốc "sang Cao Ly"). FIX: `query_entity_field` — neo bằng
`Entity:title` CỦA QUERY (title-overlap ≥2 token, KHÔNG resonance → tránh bẫy down-weight tên).
- ✅ "Nhà Lý"→Entity:Nhà Lý (khớp 2 token)→trường dynasty SẠCH→**physics "Lý thuyết đồ thị" HẾT**.
- ⚠️ dư: "Lý Bạch" (thơ Đường, đồng-tự "Lý") lọt qua token generic (vua/Đường).
- ❌ "Triều Lý": overlap chỉ "lý" <2 → fallback base_out → VẪN Hàn-Quốc (ca khó: triều≠nhà title +
  Triều/Korea homograph).

**Infra fix kèm:** `entity_anchor_field`/`query_entity_field` chỉ xét entity lân-cận-token (perf, thay
quét 3711); viz `uvicorn reload=True`→restart load v6 giữa chừng→tắt (env `AIVIE_RELOAD`);
`viz_default_checkpoint`→v6; `ingest_statements` batch defer_index (§7b).

**Kết luận:** v6 = kiến thức mới (Văn Miếu answerable) + **guard đầy đủ 86/88/67** + đồng-tự physics
sạch. Dư nhỏ ghi nợ: "Lý Bạch"(thơ), "Triều Lý"→Korea, primary "Mây"/"Nhà Lý xây"→lệch (best_match).

## 7. Đường tiếp (ghi)

1. **Bổ sung data sử VN** (ý tác giả): coverage mỏng (24 đoạn "Lý", 2 "Văn Miếu"). Nạp thêm bài
   Nhà Lý/Văn Miếu/các vua mức đoạn → entity dày hơn + liên kết (TK_024 nhánh C). Pipeline
   `ingest_entities` sẵn sàng nhận.
2. **Neo tốt hơn primary**: khi primary sai (Mây), neo lung lay. Cân nhắc neo = union(primary +
   pool-consensus-entity) hoặc gate mềm (giảm điểm thay vì loại).
3. Sửa PRIMARY retrieval hậu-WH (best_match "Mây"/"Lý Thái Tổ" — nợ TK_025) — trực giao TK_027.
