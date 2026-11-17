# TK_044 — ATTENTION-HEBBIAN: variable binding / induction head KHÔNG backprop

| Field | Value |
|-------|-------|
| **ID** | TK_044 |
| **Status** | ⚫ **CHỐT — research detour HOÀN CHỈNH (âm tính, 2026-07-22).** POC 100% (P0/P1/P2, cơ chế ĐÚNG trong cô lập) NHƯNG không chuyển được sang graph free-generation (P3.2: copy 0/11; test cô lập continuation cũng 0/11); coref-cho-known-entity KHÔNG có target (sinh đã giữ coreference tốt). Verdict §7. Kế: xoay về mục tiêu gốc "chất lượng sinh" qua focus-bug (TK_042). |
| **Nguồn** | Ý tưởng tác giả 2026-07-21: "đồ thị = ma trận, LLM = phép toán ma trận đó → nhào đồ thị thành tầng tri thức/attention; Luật 0 chỉ quan ngại BACKPROP, muốn trọng số CÓ NGAY lúc train". |
| **Liên quan** | TK_042 §7 (grounding vá xong — đồ thị chọn đúng fact); [[variable-binding-poc]]; `v4.md` Trụ 2 (R-slot HOÃN — test SAI = ordering, use-case thật là binding); TK_037 (SlotRef/slot_chain); `core/embedding.py` (E); `core/parent_store.py` (induce_slot_hubs, entity_register). |

---

## 1. Bài toán

Train `[X] … ăn kẹo, [X] … đồ ngọt` (X=Mai/Tùng). Prompt `"Linh mê ăn kẹo,"` → next = **"Linh"**?
(Linh CHƯA đồng-hiện với "ăn kẹo/đồ ngọt"). = **variable binding / induction head** — điền biến MỚI
vào khuôn. LLM làm bẩm sinh qua attention+backprop; đồ thị chép/mosaic KHÔNG làm được (phát "Mai").

Tác giả: giải được → mở nhiều thứ, gồm **văn phong** (style = bind biến style rồi áp).

## 2. Nền lý thuyết — vì sao Luật-0-native

- **Hopfield = attention** (Ramsauer 2020 "Hopfield Networks is All You Need"): cập nhật Hopfield
  = softmax(Q·Kᵀ)·V = attention.
- **Trọng số Hopfield = TỔNG TÍCH NGOÀI của mẫu lưu = Hebbian thuần**, set lúc lưu, KHÔNG backprop.
- ⇒ **attention-Hebbian là Luật-0-native.** Luật 0 chỉ cấm BACKPROP-learned attention. Đồ thị ta ĐÃ
  là ma trận Hebbian (CO_ACT, W_seq = tích ngoài đồng-hiện) → **đã có một nửa attention.**

## 3. Cơ chế (mọi cạnh đếm-1-lần lúc ingest, KHÔNG backprop) — `scripts/poc_hebbian_binding.py`

1. **category** = cluster E (PPMI+SVD ngữ cảnh) → TÊN cùng cụm. *(= E của đồ thị, có sẵn)*
2. **slot-key** = ANCHOR TRÁI (token liền trước tên) — vị trí CẤU TRÚC (`<S>`=chủ ngữ, `tặng`=tân
   ngữ, `,`=tên sau phẩy…), không phải index tuyệt đối.
3. **Nhận tên MỚI chưa-thấy** qua **reverse-bigram**: "Linh" đứng trước "mê"; train "mê" toàn có
   TÊN đứng trước → Linh = filler slot-TÊN (bind từ NGỮ CẢNH, không cần từng thấy Linh).
4. **CẠNH COREFERENCE** (mảnh đa-referent): mỗi CẶP slot-key, đếm Hebbian "cùng filler trong-câu?"
   → key nào chia filler với key nào = corefer. Học được crossed: `,↔tặng`(Y), `<S>↔ơn`(X).
5. **Sinh (COPY):** slot-biến (bigram dự đoán category-nhiều-filler) → anchor trái hiện tại → theo
   CẠNH COREFERENCE → copy filler của slot corefer.

→ attention = {Q,K = category(E), V = token identity (copy), routing = cạnh coreference}.

## 4. Kết quả POC (numpy standalone, không torch/graph)

| Test | Kết quả |
|---|---|
| **single-referent** "[X] ăn kẹo, [X]…" → tên MỚI | **3/3** (Linh→Linh, Quân→Quân, Thảo→Thảo). Baseline bigram → chép tên TRAIN (Trang) SAI |
| **đa-referent CROSSED** "X tặng Y, Y cảm ơn X" — slot-fill | **4/4** (Y sau ','→Quân; X sau 'ơn'→Linh; chéo đúng). Heuristic "tên đầu/gần" SAI cả hai |

⇒ **induction head bằng Hebbian (category+bigram+coreference+copy), KHÔNG backprop — CHẠY.**

## 5. Giới hạn POC (thành thật)

- Synthetic sạch, **whitespace token (không BPE)**, khuôn cố định.
- **Continuation qua tên-MỚI chưa giải**: sau khi copy "Quân" (không trong vocab), bigram("Quân")
  rỗng → cần **role-tracking / category-backoff** (token sau tên tuỳ VAI: sau chủ ngữ→"tặng", sau
  tân ngữ→"cảm"). POC test slot-fill từng vị trí để cô lập binding. → Phase 2.
- slot-key = anchor-trái đơn (1 token) — thực tế cần signature ngữ cảnh rộng hơn.

## 5.5 P2 — stress-test (2026-07-22, `scripts/poc_binding_stress.py`)

Leo thang 3 nấc (mỗi nấc 6 seed), tên MỚI, không backprop:

| Nấc | Điều kiện | single | crossed |
|---|---|---|---|
| 1 baseline | 1 khuôn, tên 1-token, sạch | 24/24 | 48/48 |
| 2 variety+noise | khuôn đơn+chéo TRỘN + câu nhiễu chen giữa | 24/24 | 48/48 |
| 3 multitoken | tên ĐA-TOKEN ("Thanh Mai","Bảo Ngọc") + đa dạng + nhiễu | 18/18 | 36/36 |

**3 bug thật, mỗi cái tìm qua ĐO** (P0 sạch che giấu cả ba):

1. **Nhận tên-MỚI qua NGỮ CẢNH, không qua category.** Tên mới là OOV → không có category → span-detect
   phải dựa anchor trái (forward-bigram của anchor dự đoán NAME_CAT) rồi bind token dù OOV/đa-token.
   (P0 "gian lận" bằng cách hardcode `NAMES_TEST` vào bộ nhận tên.)
2. **Purity NAME_CAT — "," lọt cụm tên.** Dấu phẩy kề tên khắp nơi → PPMI+SVD+kmeans (K=8, vocab nhỏ)
   gộp "," vào NAME_CAT → "," bị nhận là filler + bị trừu-tượng-hoá `<N>` làm hỏng signature. Vá bằng
   **TẦN SUẤT (Luật-0 sạch, không nghĩa tiếng Việt):** mảnh-tên hiếm (vài % câu), "," phủ ~mọi câu →
   bỏ member phủ >50% câu. (Lọc "xuất hiện sau `<S>`" SAI: loại luôn mảnh-tên-thứ-2 "Mai","Châu".)
3. **Signature slot phải rộng (SIG=5) để bắt VERB-FRAME.** Anchor 1-token ("," đơn) đụng độ: "ăn kẹo,"
   (đơn→coref chủ ngữ) vs "tặng Y kẹo,"(chéo→coref tân ngữ) — cùng đuôi "kẹo ," trong 2 token. Cần
   signature đủ rộng chạm ĐỘNG TỪ khung, + trừu-tượng tên trong signature thành `<N>` (bất biến tên).

⇒ **induction-head Hebbian BỀN với token-vụn + khuôn đa dạng + nhiễu** — sau khi vá 3 chỗ trên. Cả
ba là bài học cho P3 (graph thật): category-purity + nhận-OOV-qua-ngữ-cảnh + slot-signature đủ rộng.

## 5.6 P1 — continuation (role-tracking) + đại từ (2026-07-22, `scripts/poc_binding_continuation.py`)

| Mảnh | Kết quả (6 seed) |
|---|---|
| **A. role-tracking** sinh FULL câu qua tên-MỚI | single-word 24/24+24/24 · **multitoken 18/18+18/18** — "Linh tặng Quân kẹo," → "…Quân cảm ơn Linh"; "Bảo Ngọc…"→"…Tuấn Kiệt cảm ơn Bảo Ngọc" |
| **B. đại từ** "… , nó …" → nó trỏ ai | nó=chủ-ngữ **24/24** · nó=tân-ngữ **24/24** (Linh→Linh, Quân→Quân/Thảo) |

**A — vá continuation:** P0/P2 chỉ điền 1 token rồi tắc (copy tên OOV "Quân" → `bigram(Quân)` RỖNG).
Vá: token sau một tên do **VAI** (slot-signature) quyết định, không do danh-tính → back-off
`after_role[sig]` (đếm Hebbian: sau tên ở slot-sig thường là token cấu-trúc gì). Sinh xen 3 kênh:
copy-slot (coref) → role-backoff (khi prev OOV) → bigram cấu-trúc.

**B — đại từ (giới hạn thật + cách vá):** "nó" phân-bố Y HỆT tên (điền đúng vị-trí-tên) → cơ chế gốc
KHÔNG tách được, lại làm **nhiễu category** (B0: giải 0/4). Vá: (1) phát hiện đại-từ KHÔNG cần
category — slot-biến = anchor nhiều follower HIẾM, đại-từ = OUTLIER df-cao lạc giữa; (2) MASK khỏi
co-occurrence + bỏ khỏi mass slot; (3) **GIẢI 'nó' bằng cách MƯỢN cạnh coref của TÊN** (sig chỗ nó
đứng → antecedent slot → filler). **Giới hạn:** phải có câu tên-LẶP để học slot-coref; corpus TOÀN
đại-từ sẽ không học được → P3 (graph thật có cả hai).

**2 bug bền-vững vá luôn (P2 "100%" cũ có phần may):** (a) nhận-tên theo **nhãn-cụm k-means** thiếu
ổn định (đổi seed → tách tên khác) → chuyển **centroid-cosine** tới tâm tên-train (ổn định); (b)
dựng coref **lặp qua set chuỗi** → `PYTHONHASHSEED` random hoá thứ-tự → kết quả **chập chờn** → tie-break
TẤT ĐỊNH theo (ratio, #đồng-hiện, key). Sau vá: P2 & P1 xanh ổn định qua mọi hash-seed.

## 6. Roadmap (measure-first, mỗi phase một số đo)

| Phase | Nội dung | Trạng thái |
|---|---|---|
| **P0** | POC single + đa-referent crossed (numpy) | ✅ **XONG** (3/3, 4/4) |
| **P1** | Continuation role-tracking (sinh full "Quân cảm ơn Linh") + coreference qua ĐẠI TỪ ("Mai…, nó…") | ✅ **XONG** (role 100%, đại từ 100%/6-seed · `scripts/poc_binding_continuation.py`) |
| **P2** | Token-vụn (tên đa-token) + khuôn đa dạng + nhiễu: binding có sống? | ✅ **XONG** (100%/6-seed, `scripts/poc_binding_stress.py`) |
| **P3** | Tích hợp graph — kênh COPY. P3.0 nền 0/17 ✓; P3.1 coref-pass 70–77% ✓ (500k cạnh); **P3.2 copy 0/11 — CHỐT ÂM TÍNH** (plan-drift + no-content-entity-mới + coref-known-entity không target). RUNBOOK P3/P3.1/P3.2 | ⚫ **DEAD-END (đo đủ)** |
| **P4** | Văn phong: bind biến STYLE (phân bố, không chỉ copy-token — khó hơn) | ⏳ (research) |

## 7. Trần trung thực (ghi trước)

- Binding token-copy ≠ style. Style là PHÂN BỐ/biến-đổi, không phải copy 1 token — P4 khó hơn hẳn,
  có thể chạm trần khác. P0-P3 là binding-token (nền), không tự động cho style.
- Đồ thị thật (BPE, nhiễu, đa nghĩa) có thể làm category/slot/coreference nhiễu — P2/P3 sẽ đo.
- Đây là **research thật**, không hứa. Nhưng P0 dương tính = đáng đào, và Luật-0 sạch.

**⚠️ Trần THẬT gặp ở P3.2 (2026-07-22, đo — không đoán):** copy-channel A/B = **0/11** trên graph
thật, KHÔNG vì cơ chế sai mà vì **PLAN-LEVEL token-collision drift** (`out/probe_plan_agenda.log`):
`_build_segment_plan` lấy token đặc-thù-nhất của tên-MỚI làm focus → `facts_about` kéo **bài TRAIN
chia token đó** ("Ngân" Hoài Lạc→Dải "Ngân" Hà; "Đằng"→Bạch "Đằng"; "Lữ"→Nguyễn "Lữ"). Agenda về sai
entity NGAY bước dựng plan → copy (tầng score) vô ích. **Gốc kiến trúc:** POC chạy vì KHUNG mang nội
dung (X chỉ là slot); graph thật nội dung đến từ FACT truy-hồi về entity → thực-thể MỚI không có fact
→ hoặc trôi (token-collision) hoặc sinh rỗng. ⇒ **binding-copy thực-thể-mới trong SINH-TỰ-DO là bài
toán sai-đề trên kiến trúc fact-retrieval.** Cơ chế binding (P0/P1/P2 = 100%) VẪN đúng trong cô-lập;
blocker là plan-drift. Test cô lập cuối (§14 runbook, continuation-mode): cơ chế có sống khi TẮT plan.

**VERDICT CUỐI (2026-07-22) — CHỐT ÂM TÍNH:**
1. **Test cô lập continuation-mode = 0/11** (copy off VÀ hopfield). Gỡ plan-drift rồi nhưng sinh **echo
   token đầu entity ("Lữ","Ngân") rồi suy biến ". ."** — graph KHÔNG có khung-sinh mạnh mang nội dung
   cho entity không-fact. Copy-boost sửa score, KHÔNG sửa thứ-bậc ứng-viên lẫn KHÔNG tạo nội dung.
2. **Coref-cho-known-entity KHÔNG có target** (đo `out/probe_retrieval_compose_v11_fix4.txt`): sinh về
   entity ĐÃ BIẾT đã giữ coreference tốt ("Triều Lý","Dòng sông","nước sông") — không có drift để chữa.
   `ent_keep=0.00` là metric-hỏng exact-case, không phải drift thật.
⇒ **500k cạnh coref + kênh copy KHÔNG có đường dùng productive.** Binding = kết quả research thật (cơ
chế Hebbian induction-head chạy 100% cô lập, Luật-0 sạch) nhưng **không chuyển sang hệ fact-retrieval**.
Đóng nhánh. Lever chất-lượng-sinh thật (cùng log): **focus-bug** (từ-để-hỏi "ai/gì" giành focus →
"Trần Hưng Đạo là ai"→bài AI) + số nát (tường TK_041).

## 8. Nhật ký

| Ngày | Việc | Kết quả |
|---|---|---|
| 2026-07-21 | P0 POC (tác giả gợi Hopfield=attention/không-backprop) | single 3/3 + crossed 4/4 · `scripts/poc_hebbian_binding.py` · commit chuỗi TK_044 |
| 2026-07-22 | P2 stress (đa-token + đa dạng + nhiễu, 6 seed) | 3 nấc 100% sau khi vá 3 bug (nhận-OOV, purity-tần-suất, SIG=5) · `scripts/poc_binding_stress.py` |
| 2026-07-22 | P1 role-tracking + đại từ (6 seed) | role sinh-full-câu 100% (gồm multitoken) + đại từ giải tiền-ngữ 100%; vá 2 bug bền (centroid thay k-means, tie-break tất-định) · `scripts/poc_binding_continuation.py` |
| 2026-07-22 | P3.0 baseline graph v11 (M4) | copy **0/17** (single 0/9 · crossed 0/4 · đại từ 0/4) — entity MỚI → TRÔI wiki lạc. Novelty-detector đúng (Thăng Long skip+copy✓). MỐC cho A/B · `scripts/probe_binding_graph.py`, `out/probe_bind_P30*.log` |
| 2026-07-22 | P3.1 dump hub (M4) | graph v11 chỉ **5 hub, TẤT CẢ khung câu-hỏi** ("X là gì?"), không frame đồng-chỉ → slot-hub KHÔNG phải nền binding (kill-criteria đúng) · `out/slot_hubs_dump_step1_*.log` |
| 2026-07-22 | P3.1 pivot: scan corpus (dev) | corpus THÔ tín hiệu binding DÀY: lặp-tên ≥2× 83% viwiki/92% history/89% conv; đại-từ 72–86% (nó/họ/ông/bà). ⇒ `run_coref_pass` quét CORPUS (POC-port), name-repeat single TRƯỚC · `scripts/scan_coref_signal.py`, `out/coref_signal_scan.txt` |
| 2026-07-22 | P3.1 pass v1 CỔNG FAIL (M4) | 648 cạnh nhưng occurrence=`ps.phrases`+gidf → bắt collocation BẢNG/GAME (ma biến*600, H2O→Fe2O3, năm-sinh) = lặp cấu-trúc, không đồng-chỉ; records_with_occ chỉ 6% · `out/coref_edges_P31_step2*` |
| 2026-07-22 | P3.1 vá detector v2 (dev) | occurrence = span HOA `vocab.cap_class≥1` + gidf (danh-từ-riêng, khớp scan), bỏ `ps.phrases`. Self-test pass. + serialize coref/after_role (state.py, 245 pytest) · `coref_pass.py`, `state.py` |
| 2026-07-22 | P3.1 smoke v2 CỔNG FAIL (M4) | 10,763 cạnh, records_with_occ 68% (tín hiệu đúng) nhưng ~73% nhiễu BPE-fragment/từ-Hoa-đầu-câu ('[ENT] g','[ENT] ước') · `out/coref_edges_smoke.log` |
| 2026-07-22 | P3.1 vá detector v3 (dev) | + word-start (`Ġ`/pos0) loại continuation + đa-token/`gidf_solo` loại chữ-cái/từ-đầu-câu. Self-test giữ Nguyễn Du/Hà Nội, loại Ước/G/ảnh · `coref_pass.py` |
| 2026-07-22 | Luật-0 audit + Hopfield audit (dev) | Luật 0 phân tuyến INGEST(string OK)/RUNTIME(cấm) — coref_pass=ingest → SẠCH; ràng buộc P3.2 runtime chỉ-đọc-weight. Hopfield: ta dùng nửa TRỌNG-SỐ (Hebbian đếm) + argmax CỨNG, CHƯA dùng nửa ĐỘNG-LỰC (softmax retrieval) → P3.2 phương án A = copy MỀM Hopfield (robust BPE) · `RUNBOOK_TK_044_P3.2.md` |
| 2026-07-22 | **P3.1 CỔNG ĐẠT** (M4, smoke) | vá bug thứ 2: `encode_tokens(tokenize())` mất Ġ → chỉ pos-0 qua gidf (SIG left=∅); `encode_text` sửa. 11,673 cạnh, ~70–77% hợp lý (Chích Quái, địa danh lịch sử, nhân vật). → full pass rồi P3.2 · `out/coref_edges_smoke_v3.log`, [[tokenizer-silent-bugs]] #5 |
| 2026-07-22 | **P3.2 copy-channel: TRẦN plan-drift** (M4+dev) | Plan B/A đều 0/11; diagnostic chốt PLAN-LEVEL token-collision (Ngân→Ngân Hà, Đằng→Bạch Đằng). Review-vá 2 lỗi (cosine-thresh, gidf-gate). Cơ chế đúng, blocker=plan. Kế: test cô lập continuation-mode · `out/probe_plan_agenda.log`, `RUNBOOK_TK_044_P3.2.md` §11-14 |
| 2026-07-22 | **CHỐT ÂM TÍNH** (dev) | test cô lập continuation 0/11 (echo token-đầu rồi suy biến); coref-known-entity KHÔNG target (sinh đã giữ coreference). Binding=research detour hoàn chỉnh. Xoay → focus-bug · `out/probe_bind_P32_continuation*.log`, `out/probe_retrieval_compose_v11_fix4.txt` |
