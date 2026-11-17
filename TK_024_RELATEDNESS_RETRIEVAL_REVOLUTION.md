# TK_024 — CÁCH MẠNG: TRUY HỒI THEO LIÊN QUAN (không chỉ chứa token)

| Field | Value |
|-------|-------|
| **ID** | TK_024 |
| **Status** | 🟢 A1 DONE + B0 DONE + **C scale v10** (2026-07-09) — 160 bài / 19k câu; checkpoint `network_v10_histwiki.pkl`; guard 86/88/67 giữ. Sinh hợp nhất A3 multi (TK_019 §7). |
| **Nguồn** | Tác giả: fact chỉ được chọn vì CHỨA token/cụm input ("Nhà Lý" → chỉ fact có "Nhà Lý"); fact liên quan không chứa cụm (vua nhà Lý, Văn Miếu 1070) bị bỏ lỡ. Ý tác giả: dùng **kết nối ngang** ("Nhà Lý"→"Thăng Long") hoặc **embedding** để tìm fact liên quan không cần token chung. |

## 0. REVIEW staged Cursor (TK_021→023) — ĐẠT

`ReadoutMode` (EXPRESSION/KNOWLEDGE) + `theme_neuron` pool + `parent_stimulus_score` (lan xuôi
token→cụm ISG→cha, idf×power) + bồi fact đoạn. **Luật 0 sạch** (idf/power/vị-trí, không rule
từ). 4 ca lật đúng; guard **QA 86 / Para 88 / Syn 67; 150 test pass**. Code tốt.

**NHƯNG xác nhận trần tác giả chỉ:** cả `theme_neuron(e)` (fact = câu CHỨA e) lẫn
`parent_stimulus_score` (lan qua cụm CHỨA token) đều **token-bound**. Fact không chứa token
query → vô hình. Đây là **trần bag-of-words retrieval**.

## 1. PROBE RẺ (chặn build sai) — 3 phát hiện đo được

Chạy `scripts/probe_horizontal_links.py`-style trên `network_v4_full` (`out/probe_entity_
relatedness.log`, `out/probe_relate2.log`):

| Nguồn liên-quan | Kết quả đo | Kết luận |
| :--- | :--- | :--- |
| **Co-occurrence CÂU** (Lý ↔ ?) | top PMI = {1009, 1104, Vượng, Kiệt, Nghiêu} — mảnh tên + năm, KHÔNG phải Thăng Long/Văn Miếu | Câu quá NGẮN + rời → đồng-hiện thưa → nhiễu. **Build layer này = nhiễu, không cứu.** |
| **Embedding** nearest("Lý") | {Lê, Trần, Nguyễn, Tông, Phúc…} — **triều đại ANH EM** (cùng slot), KHÔNG phải associate | E là PARADIGMATIC (cùng ngữ cảnh) không ASSOCIATIVE. "phở"→{xào,nấu,mắm} associative TỐT — nhưng "Lý"→siblings = **nguy hiểm** (chính là drift chéo triều đại) |
| **Episode co-occurrence** Lý ↔ Văn Miếu | **0 episode** chứa cả 2 | Fact curated là câu-lẻ lớp học, không phải đoạn wiki → "Văn Miếu thuộc Lý" **KHÔNG có trong data dưới BẤT KỲ dạng nào** |

**Chẩn đoán lại (thay giả định ban đầu):** vấn đề tách làm HAI, tầng khác nhau:

- **(A) Trần RANKING (kiến trúc — sửa được NGAY):** fact CÓ token "Lý" (Lý Công Uẩn dời đô, Lý
  Thái Tổ, Lý Thường Kiệt) bị fact cụm "Nhà Lý" (power cao) chôn. `theme_neuron(Lý)` ĐÃ chứa
  chúng — chỉ xếp hạng thấp. Đây là cái tác giả liệt kê muốn ("vua nhà Lý").
- **(B) Trần DATA (không phải kiến trúc):** fact KHÔNG token chung ("Văn Miếu 1070") — association
  "Văn Miếu thuộc Lý" là TRI THỨC THẾ GIỚI **vắng mặt** trong corpus (0 token, 0 co-oc, 0 episode,
  embedding cho sibling). **Không cơ chế đồ thị nào tạo ra tri thức không có trong data.**

## 2. CÁCH MẠNG — 2 nhánh, đúng tầng (không tự lừa)

### Nhánh A — EXPAND-then-RANK (kiến trúc, làm được ngay; giải (A))

Truy hồi 2 pha, thay "1 pha chứa-token":

1. **EXPAND** seed: query "Nhà Lý" → thực thể-đầu **Lý** (đã có `_distinct_focus_primary`) →
   **gia đình thực thể** = mọi fact chứa biến-thể-tên cùng head "Lý *" (Lý Công Uẩn, Lý Thái Tổ,
   Lý Thường Kiệt) qua `theme_neuron(Lý)` + `case_equiv`. KHÔNG mở sang embedding-sibling (Trần/Lê
   — probe chứng minh = drift).
2. **RANK theo CỘNG HƯỞNG, không chứa-cụm:** mỗi fact ứng viên chấm = Σ `stimulus(token ∈ fact)`
   với trường kích thích gieo từ **cả gia đình** (không riêng cụm "Nhà Lý"). "Lý Công Uẩn dời đô"
   cộng hưởng qua "Lý" + "dời đô" (nếu query mở rộng) → lên top. Tái dùng `parent_stimulus_score`
   (TK_023) + gieo seed = family, không phrase.
3. **Đa dạng hoá (MMR):** bồi K fact phủ NHIỀU khía cạnh (thành lập / vua / văn hoá / chống giặc)
   thay K fact gần trùng — phạt trùng theme-token giữa các fact bồi (đã có Jaccard, nâng thành
   MMR: chọn fact xa nhất các fact đã chọn).

→ Cho ra ĐÚNG list tác giả muốn (Lý Công Uẩn, Lý Thái Tổ, các vua) mà không đụng data.

### Nhánh B — CONTENT HORIZONTAL LINK trên DATA ĐÚNG (ý tác giả, nhưng nguồn khác)

Ý "Nhà Lý → Thăng Long" **đúng kiến trúc** — chỉ sai NGUỒN: câu-lẻ quá thưa. Mine trên **đoạn
wiki đầy đủ** (`data/real_vi/statements_doc.jsonl` — 45k đoạn có doc_id+para; hoặc re-ingest bài
Nhà Lý nguyên vẹn nơi "Văn Miếu" đứng cùng đoạn với "Lý"):

1. **Lớp cạnh `ENTITY_LINK`** (tái dùng bộ 4-tham-số TK_020: PMI + power + **provenance** +
   không-gap): entity↔entity (gidf cao) đồng-hiện **cùng ĐOẠN** (không cùng câu), PMI-lọc. Provenance
   = đoạn/episode nối chúng → đúng ý tác giả 07-06 (đoạn kích hoạt dội ngược cạnh).
2. **Retrieval qua spreading** (máy `spreading_activation_read` CÓ SẴN): seed = entity query →
   lan 1-2 hop qua `ENTITY_LINK` → trường thực-thể-liên-quan (Lý → Thăng Long → Văn Miếu) →
   pool fact = union `theme_neuron` trên trường → rank theo cộng hưởng. Đây là **attention thật
   cho retrieval**: trường query = Q, token fact = K, cộng hưởng = Q·K.
3. **Embedding = CỔNG loại, không mở rộng:** dùng E để BIẾT Lý≠Trần là siblings (nén drift chéo),
   KHÔNG để expand. Đảo đúng vai (probe: E cho sibling).

### Nhánh C (thành thật về DATA — điều kiện để B thắng)

Fact "Văn Miếu thuộc Lý" chỉ xuất hiện nếu **nạp bài wiki nguyên đoạn** (không câu-lẻ). Đề xuất
re-ingest wiki mức ĐOẠN cho các chủ đề lịch sử → ENTITY_LINK có nguyên liệu thật. Không nạp =
nhánh B trung tính trên các ca này (ghi sổ, không ảo tưởng).

## 3. LỘ TRÌNH ĐO ĐƯỢC (thí nghiệm rẻ chặn công lớn)

| Phase | Nội dung | Đo | Điều kiện mở |
| :--- | :--- | :--- | :--- |
| **P0** ✅ | probe co-oc/embedding/episode | đã chạy → chẩn đoán lại (§1) | — |
| **A1** ✅ | EXPAND family + stimulus rerank gieo-family + MMR + cổng đồng-tự | ĐẠT (§6): vua/tướng nhà Lý lên top, "Lý thuyết" hết lọt, "Mây" giữ định nghĩa; guard 86/88/67 giữ | DONE |
| **B0** ✅ | probe ENTITY_LINK mức ĐOẠN trên statements_doc: Lý↔{Thăng Long,Văn Miếu} có nổi PMI? | DƯƠNG YẾU (§7): associate thật (Đại La/Hoa Lư/Thường Kiệt/Tống) nổi ở đoạn ≠ câu/E; NHƯNG mỏng (co2–5) + Văn Miếu=0 (data gap) + drift đồng-tự | DONE → chờ tác giả quyết B1 vs C |
| **B1** | lớp `ENTITY_LINK` + spreading-retrieval (nếu B0 dương) | recall fact-liên-quan-không-token trên held-out mới; guard giữ | B0 dương |
| **C** | re-ingest wiki mức đoạn (nếu B0 âm vì data) | B0 chạy lại | quyết định tác giả |

## 4. Rủi ro ghi trước (chống ảo tưởng)

- **Spreading drift**: 2 hop kéo rác (Thăng Long→Hà Nội→fact hiện đại). Gate hop-decay + PMI +
  provenance-strength; đo precision, không chỉ recall.
- **Embedding-sibling drift**: KHÔNG dùng E để expand (probe cảnh báo) — chỉ để nén chéo-sibling.
- **Data honesty**: (B) association vắng trong curated → không tạo được; (C) là data move.
- **Ranking vs Relevance**: (A) đã đủ cho phần lớn ca "X có gì" mà không cần (B). Làm A trước,
  đo, rồi mới quyết B/C.

## 5. Đề xuất thứ tự

**A1 trước** (rẻ, giải đúng phần tác giả liệt kê muốn, không đụng data) → **B0 probe** (quyết
nhánh B đáng làm không) → B1/C theo kết quả. Không build ENTITY_LINK trước khi B0 xác nhận
data có tín hiệu (probe P0 đã chặn 1 lần build sai ở mức câu).

---

## 6. A1 IMPLEMENTED — EXPAND-then-RANK (Claude 2026-07-06)

### Code (`hca/generate_hebbian.py`)

| Phần | Cơ chế |
|------|--------|
| **EXPAND** | pool ứng viên = `facts_about(entity)` ∪ `theme_neuron(primary).facts` (cờ `knowledge_family_expand`). Gieo seed = **toàn entity** (`frozenset(entity)`), rank qua `parent_stimulus_score` (TK_023) → fact có "Lý" nhưng KHÔNG cụm "Nhà Lý" (Lý Nhật Tôn, Phạm Tu khai quốc nhà Tiền Lý) cộng hưởng lên top. KHÔNG mở sang embedding-sibling (probe: Trần/Lê = drift). |
| **RANK/MMR** | dedup Jaccard>0.6 + MMR (`knowledge_mmr_lambda=0.5`) phạt trùng theme-token → K fact phủ nhiều khía cạnh. |
| **~~Cổng đồng-tự~~ ĐÃ GỠ** | ⛔ `_primary_in_name_context` + `len(entity)>=2` + `need` **VI PHẠM LUẬT 0/0.0** (tác giả chỉ 2026-07-06): quét CHUỖI token của completion ở inference + rule danh-từ (proper vs common theo số token) + rule đếm. Đã **gỡ toàn bộ** — mạng chỉ kích-hoạt-bằng-token, không xử-lý-chuỗi. Hệ quả: đồng-tự "Lý thuyết" **lọt lại** (nợ thật, xem §8). |

### ĐO (`network_v4_full`, `out/a1_v4.log`)

**Recall (đọc tay, cái tác giả liệt kê muốn):**
- `Nhà Lý có gì` → "Nhà Trần thay nhà Lý 1226 . **Phạm Tu danh tướng công thần khai quốc nhà Tiền Lý** . **Lý Nhật Tôn kế vị sử gọi là Lý Thánh Tông**" — vua/tướng nhà Lý ĐÃ lên top (trước bị "Nhà Lý…" power cao chôn).
- `Triều Lý đã làm gì` → "Lý Thái Tổ dời đô Thăng Long . Phạm Tu… khai quốc nhà Tiền Lý . Lý Nhật Tôn kế vị…" — đa dạng.
- `Mây là gì` → "Đỉnh Mây Bạc… . **Mây bao gồm những giọt nước và băng lơ lửng**…" — định nghĩa đúng GIỮ (cổng bỏ qua 1-token).

**Precision (đồng-tự đã chặn):** "Lý thuyết Lý thuyết điện từ của Maxwell" (theory) **HẾT lọt** vào cả hai prompt Lý (trước leak ở a1_v3).

**Guard:** QA **86** / Para **88** / Syn **67** — GIỮ NGUYÊN; 150 test pass.

**Đọc trung thực:** A1 giải đúng tầng (A) — RANKING của fact CÓ token chung. Tầng (B) (fact
KHÔNG token chung: "Văn Miếu 1070 thuộc Lý") A1 **không** đụng tới (không có nguyên liệu) — đó
là việc của B0/B1. Cổng đồng-tự đánh đổi: entity 1-token là tên riêng hiếm (VD hỏi trống)
có thể lọt đồng-tự — chấp nhận vì query "X có gì/là ai" hầu hết entity ≥2 token; ghi sổ.

### Sổ nợ A1
- Cổng bỏ qua entity 1-token → tên riêng 1-token (hiếm trong bộ query) mất lớp chống đồng-tự.
- Family pool = `theme_neuron(primary)` vẫn token-bound (primary phải là token) — tầng (B) chờ B0.

---

## 7. B0 PROBE — đồng-hiện mức ĐOẠN (Claude 2026-07-06)

**Câu hỏi:** ý tác giả "Nhà Lý → Thăng Long/Văn Miếu" có tín hiệu ở mức ĐOẠN không (P0 đã bác
mức CÂU = nhiễu)? Nếu có → nhánh B (`ENTITY_LINK` + spreading) đáng build; nếu không → chờ data.

**Data:** `statements_doc.jsonl` = **24 864 đoạn** (gộp theo `doc_id`+`para`). Thực thể = cụm
từ-viết-hoa liên tiếp (proper-name span), bỏ stop-cap. PMI đồng-đoạn với biến thể "Lý".

### KQ1 — ví dụ điển hình tác giả: DATA GAP (vẫn 0)
| Cặp | Đồng-đoạn |
| :--- | ---: |
| Lý ↔ **Văn Miếu** | **0** ("Văn Miếu" chỉ có 2 đoạn / toàn 24.8k) |
| Lý ↔ **Thăng Long** | 8 đoạn (token-level) |

→ "Văn Miếu 1070 thuộc Lý" **vắng cả mức đoạn** — xác nhận §1(B): association này KHÔNG có
trong corpus dưới bất kỳ dạng nào. B **không** tạo được ví dụ này (cần nhánh C re-ingest).

### KQ2 — nhưng đồng-đoạn CÓ associate lịch sử THẬT (≠ probe câu/E)
Top PMI (co≥2, df≥3): **Đại La** (co4, "dời đô từ thành Đại La"), **Hoa Lư** (co5, kinh đô cũ),
**Thái Tổ, Lý Thường Kiệt, Thuận Thiên** (niên hiệu Lý), **Dực Thánh** (hoàng tử Lý), **Tiền Lê,
Đại Cồ Việt, Tống/Bắc Tống** (chiến Lý Thường Kiệt), **Chế Mân** (vua Chăm). Đây ĐÚNG loại
"liên-quan không cần cụm chung" tác giả muốn — mà mức CÂU (P0) không cho, embedding (P0) cho
sibling (Trần/Lê = drift).

**Đối chiếu 3 nguồn (cùng entity Lý):**
| Nguồn | Cho ra | Verdict |
| :--- | :--- | :--- |
| Co-oc CÂU (P0) | mảnh tên + năm (1009, Vượng) | nhiễu — bác |
| Embedding (P0) | Trần/Lê/Nguyễn (sibling) | drift chéo triều — chỉ dùng làm CỔNG |
| **Co-oc ĐOẠN (B0)** | **Đại La/Hoa Lư/Thường Kiệt/Tống** | **associate thật** ✅ |

### Đọc trung thực (chống ảo tưởng)
- **Dương nhưng MỎNG:** associate thật co=2–5 (chỉ Tống co=12). PMI ở count thấp KHÔNG ổn định
  → B1 phải có sàn count + shrinkage; kỳ vọng recall-gain KHIÊM TỐN, không ngoạn mục.
- **Drift lẫn vào:** "Hán Hiến Đế / Lạc Dương / Hà Nam" (sử Trung Quốc, người họ Lý TQ) lọt top —
  đồng-tự entity-level. Embedding-gate (chống sibling) KHÔNG cứu (đây không phải sibling) → cần
  sàn PMI + provenance-strength; đo PRECISION không chỉ recall.
- **Ví dụ cờ hiệu vẫn là data gap:** Văn Miếu=0 → B không tạo được ca đó; cần nhánh C.

### Verdict B0 → quyết định cho tác giả
Tín hiệu đoạn **dương yếu**: đủ để một **B1 spike có kiểm soát** (ENTITY_LINK sàn-count + spreading
1-hop + gate PMI/provenance, đo precision), nhưng KHÔNG đủ mạnh để kỳ vọng nhảy vọt, và ca "Văn
Miếu" là **data gap** chỉ nhánh **C** (re-ingest wiki nguyên đoạn) mới lấp. Hai lựa chọn, giá khác nhau:
- **B1** (kiến trúc, ~1–2 buổi): recall-gain khiêm tốn trên associate đã-có-trong-đoạn (Đại La, Hoa Lư…), rủi ro drift entity-đồng-tự — đo được.
- **C** (data, quyết định tác giả): re-ingest chủ đề lịch sử mức đoạn → ENTITY_LINK có nguyên liệu dày (Văn Miếu mới xuất hiện). Không nạp = B1 trung tính trên các ca thiếu data.

Đề xuất: **A1 đã giải phần lớn ca "X có gì" tác giả liệt kê**; B1 là nâng cấp biên (marginal) trên
tín hiệu mỏng. Nên **chốt A1 + để tác giả quyết B1-spike vs C-data** trước khi bỏ công lớn (đúng
luật measure-first: P0 đã chặn 1 build sai mức câu; B0 cho thấy mức đoạn dương-yếu-có-drift).

---

## 8. NHÁNH C IMPLEMENTED — C-targeted data + gate bồi-drift (2026-07-09)

> Tác giả chọn **C-targeted** (nạp bài lịch sử theo TÊN, nguyên đoạn) sau khi 4 phép đo phiên
> hội tụ: KN âm / order-2 trung tính / gen-ar=gen-hy ×3 / drift TĂNG theo scale → bottleneck là
> **retrieval** không phải VarLM/decode. Và probe (§7) xác nhận đoạn hiện có QUÁ MỎNG (co=2–4).

### 8.1 Data — fetch targeted (Luật 0 sạch: chọn chủ đề = curriculum)

| Thành phần | File |
| :--- | :--- |
| Fetch Wikipedia VI theo tên, nguyên đoạn (MediaWiki extracts API, quality gate) | `scripts/fetch_history_wiki.py` |
| Mine ENTITY_LINK candidates (đo nguyên liệu, KHÔNG wire runtime) | `scripts/mine_entity_links.py` |
| Probe generate entity-answer + gidf (A/B checkpoint) | `scripts/probe_entity_answers.py` |
| Data | `data/real_vi/history_wiki_para.jsonl` (104 bài → 13.5k câu đoạn) |

**Re-probe density (mine_entity_links, `out/tk024_entity_links.log`):** density↑ (co→5–7 vs 2–4),
associate ĐÚNG (`Lý Thường Kiệt`→Như Nguyệt/Quân Tống/Quách Quỳ; `Thăng Long`→Thoát Hoan/Quân Nguyên),
**data-gap "Văn Miếu" LẤP** (df 2→11, →Khổng Tử). C-targeted > C-broad đúng như dự đoán.

### 8.2 Ingest → checkpoint + đo (KHÔNG cần ENTITY_LINK/B1)

`build_history_checkpoint.py --history history_wiki_para.jsonl --priority 0.8` (incremental):
`network_v7_history_segseed` → **`network_v9_histwiki.pkl`** (149 627 parents).

**Probe entity (`probe_entity_answers.py`):** đáp MẠCH LẠC ĐÚNG qua **A1 `theme_neuron` ĐÃ CÓ**:
- `Nhà Lý có gì` → "Nhà Lý là triều đại phong kiến… 1009–1225. Lý Thường Kiệt đánh Chiêm Thành… cản quân Tống đắp đê…"
- `Lý Thường Kiệt là ai` → "viết Nam quốc sơn hà… chiến lũy Như Nguyệt chặn quân Tống… chiếm thành Ung"
- `Văn Miếu là gì` → "trường ĐH đầu tiên VN… Quốc Tử Giám 1076… thờ Khổng Tử" (gap lấp)

**Guard `eval_v4`: QA 86 / Para 88 / Syn 67 GIỮ.**

> **Kết luận measure-first quan trọng:** **B1 (ENTITY_LINK + spreading) KHÔNG cần** — data dày
> đúng miền + A1 sẵn có đã đủ cho xô entity-knowledge. Tránh được build kiến trúc rủi-ro-guard.
> Đúng kỷ luật §5 "làm A trước, đo, rồi mới quyết B".

### 8.3 Vá bồi-drift — cap-gate primary (`knowledge_extra_require_cap_primary`)

Đo lộ bồi-drift trên role/synonym query: `Người chữa bệnh` → "Bác sĩ chữa bệnh." ✓ + bồi lạc
"Bệnh viêm gan C…". Calibrate: **gidf KHÔNG tách** (`bệnh` 5.94 > `Lý` 5.84). Tách sạch bằng
**cap_class của primary** (attr precompute TK_012 — Luật 0.0): tên riêng viết hoa (Lý/Kiệt/Miếu)
→ bồi; danh-từ-thường (`bệnh`) → chỉ crisp. Cờ `knowledge_extra_require_cap_primary` (default ON).

| | Trước | Sau |
| :--- | :--- | :--- |
| Người chữa bệnh | "Bác sĩ chữa bệnh. Bệnh viêm gan C…" | **"Bác sĩ chữa bệnh."** |
| Đoàn kết giúp em | drift thống kê dân tộc | **"Đoàn kết giúp em mạnh mẽ."** |
| Nhà Lý/Văn Miếu (viết hoa) | multi-fact | **giữ** |

Guard **86/88/67 giữ**; **151 test pass** (chỉ fail pre-existing `test_best_match_open_query_requires_focus_entity`, TK_031 §7).

**Sổ nợ (trung thực):** cap-gate là proxy thô — chủ đề **danh-từ-thường hợp lệ** (Mùa hè, phở)
cũng mất bồi (chỉ crisp). Cần discriminator tách "role/synonym query" khỏi "topic danh-từ-thường"
tốt hơn cap (vd: crisp-answer chứa entity KHÁC query = role query). Flag tắt được;
`test_knowledge_extra_facts_cap_primary_gate` tài liệu hóa trade-off.

### 8.4 Còn treo (nếu muốn đẩy tiếp nhánh C)

- **B1 ENTITY_LINK** — ưu tiên thấp (A1+data đủ); span sạch hơn qua TK_034 WS-seed.
- **Nối underthesea NER** cho `mine_entity_links.py` (phase sau).

### 8.5 Scale C-targeted (2026-07-09)

| | Trước | Sau expand |
| :--- | ---: | ---: |
| bài wiki | 103 | **160** (+57) |
| câu `history_wiki_para` | 13 564 | **19 314** (+5 750) |
| parents v9 | 149 627 | — |
| parents **v10** | — | **151 430** (+1 803 delta ingest) |
| output | `network_v9_histwiki.pkl` | **`network_v10_histwiki.pkl`** |

**Pipeline:**
```bash
python scripts/fetch_history_wiki.py --titles-file data/history_titles_expand.txt --append
python scripts/build_histwiki_incremental.py --titles-file data/history_titles_expand.txt \\
  --checkpoint checkpoints/network_v9_histwiki.pkl --out checkpoints/network_v10_histwiki.pkl
```

Script mới: `scripts/build_histwiki_incremental.py` (delta theo title/slug).
Seed mở rộng: `data/history_titles_expand.txt` (địa lý VN, văn hóa, KH, role-query control).

**Probe entity v10:** `out/tk024_probe_v10.log`
