# RUNBOOK — VĂN PHONG (phrase-Hopfield?) — measure-first (cho M4)

Bàn 2026-07-22: "Hopfield có giải văn-phong không?" Tách bạch:
- **Binding-biến-style** (TK_044 P4) = ~CHẾT cùng lý do binding (plan-drift + no-content).
- **Hopfield-retrieval-cho-fluency** = KHÁC, còn sống. Hopfield-as-retrieval = content-addressable
  memory = CHÍNH cơ chế fluency (attention/LM).

Graph ĐÃ thử **Hopfield token-level** (`attention_qk.py` TK_017: Q/K từ W_seq, `qᵢ·kⱼ≈PPMI(i→j)`) —
đo Δ−0.103 (đỡ) nhưng tác-giả ghi "q·k một mình KHÔNG cứu, thiếu NÓI GÌ"; salad `. , và` vẫn còn.
**Lever chưa thử:** Hopfield **PHRASE-level** — truy hồi cả CỤM tự nhiên (parents/phrases/templates
đã lưu) thay chuyển-tiếp-token. Khác binding (không copy biến mới, chọn cách-nói từ kho đã có).

## Cổng measure-first (LÀM TRƯỚC khi xây phrase-Hopfield)

Câu hỏi chưa số: **style ở đường COMPOSED (A3, không chép verbatim) có THẬT tệ không?** Known-entity
nhìn mượt nhưng phần lớn là CHÉP. Phải đo đường composed.

```bash
PYTHONPATH=src python -u scripts/probe_style_composed.py \
    --checkpoint checkpoints/network_v11.pkl --log-file out/probe_style_composed.log
```
So 2 chế độ mỗi prompt (đa-fact + mở):
- **VERBATIM** (lưới `ar_verbatim_fallback` BẬT) = chép mượt (baseline "đẹp").
- **COMPOSED** (tắt verbatim + `essay_ar_min_cov=0`) = ép A3 token-by-token → phơi salad.

**Đọc tay (G15, không metric tự động cho style):**
- COMPOSED **salad rõ** (`. , và` / lặp / cụt) so VERBATIM ⇒ **có mục tiêu** → xây phrase-Hopfield.
- COMPOSED **cũng trôi/đọc được** ⇒ style ĐÃ ỔN → không phải bài toán (dừng, honest-negative).

Dán log về → quyết có đào phrase-Hopfield không.

## Nếu cổng dương → phrase-Hopfield (phác, chưa cam kết)

- Kho mẫu: `ps.completion(parent)` / `ps.phrases` / `pattern_templates` = cụm text tự nhiên.
- Retrieval: mỗi ranh-giới-cụm, q = emb(ngữ-cảnh) → softmax trên emb(cụm đã lưu) → phát cụm gần nhất.
- = attention {Q=ctx, K=emb-cụm, V=cụm-text}, Hebbian, không backprop → Luật-0-native.
- Đo A/B đọc-tay vs A3-token hiện tại (composed). Ràng buộc: KHÔNG string-rule runtime (Luật 0).

## Trần thành thật (ghi trước)

- Graph đã fluent nhờ CHÉP; lợi phrase-Hopfield chỉ ở đường composed (khi không chép).
- Truy hồi = mẫu GẦN NHẤT đã lưu → style mới-hoàn-toàn vẫn giới hạn (nhưng use-case "văn phong tự
  nhiên là đủ" KHÔNG cần cái đó).
- Nếu composed đã ổn (cổng âm) → đừng xây; đây là honest-negative có giá.

## v2 (2026-07-22) — probe v1 HỤT, sửa prompt phức tạp

**v1 SAI:** prompt single-fact ("Nhà Trần là gì") → gen **chép 1 parent verbatim**, COMPOSED byte-identical
VERBATIM → đường A3/ghép KHÔNG chạm. "No salad" là kết luận NON. **Phát hiện thật:** gen hiện tại
**retrieve-1-parent + copy, KHÔNG ghép fact** — "ghép fact" (multi-zone) genuinely CHƯA có.

**v2:** prompt đa-nội-dung / style (So sánh Nhà Lý–Trần; Bạch Đằng+Ngô Quyền; "hài hước… Trần Quốc
Toản… gen Z") — KHÔNG parent nào phủ trọn → buộc ghép. Chạy lại:
```bash
PYTHONPATH=src python -u scripts/probe_style_composed.py \
    --checkpoint checkpoints/network_v11.pkl --log-file out/probe_style_composed_v2.log
```
**Đọc tay điểm-xuất-phát composition (4 khả năng):**
- chép-1-parent rồi cụt (bỏ zone khác) ⇒ composition CHƯA có → build multi-zone.
- ghép nhiều fact nhưng SALAD ⇒ có ghép thô → phrase-Hopfield.
- trôi entity (Bạch Đằng→Nhà Trần) ⇒ grounding drift đa-thực-thể.
- style ('hài hước') bị bỏ qua ⇒ HOW không retrieve (xác nhận aspect probe) → style thuộc LM.
Dán log v2 về → quyết build multi-zone / phrase-Hopfield / RAG-LM.

## Kết quả v2 (2026-07-22, M4) — COMPOSED ≡ VERBATIM trên MỌI prompt kể cả phức tạp

**Chạy:** 7 prompts (So sánh Nhà Lý–Trần; Bạch Đằng+Ngô Quyền; hài hước; buồn mùa thu; Nhà Trần...)

**Pattern nhất quán (measurement):**
- COMPOSED ≡ VERBATIM byte-for-byte (ngoại trừ 1 dấu `.` trong prompt gen-Z)
- Gen **chép 1 parent duy nhất** rồi dừng, kể cả "So sánh Nhà Lý và Nhà Trần" (2 thực-thể)
- Style 'hài hước'/'buồn'/'gen Z' **bị bỏ qua hoàn toàn** — HOW không ảnh hưởng runtime
- Grounding: "hài hước trận Bạch Đằng" → gen bài Văn Miếu (drift xa)

→ Xác nhận: **composition (ghép fact) CHƯA tồn tại** — gen = retrieve-1-parent + verbatim copy;
**style KHÔNG execute** (graph không HOW-retrieval → style thuộc RAG-LM).

**Hướng (tác giả chỉ đạo 2026-07-22): BUILD multi-zone** (dưới) — "ghép fact" chưa có chính là chỗ
cần xây, và zone = vùng chú ý (multi-head attention). Focus-bug đã kiểm KHÔNG tái hiện
(RUNBOOK_GROUNDING). Style/blend để riêng cho LM (RAG). (Ghi chú: M4 từng đề xuất "no multi-zone" —
đã override theo chỉ đạo tác giả.)

## MULTI-ZONE retrieval = multi-head attention (2026-07-22, dev build)

Ý tác giả: zone = vùng chú ý → **multi-head attention** (mỗi zone 1 head). Build:
`src/nn_aivie/hca/multizone.py` — (1) zone hoá query (E-cosine union-find); (2) mỗi zone Hopfield-
attention retrieve: candidate qua `facts_about` (index) → RE-RANK `softmax(β·zq·K)` (zq=emb-zone,
K=emb-fact) → top; (3) zone-type theo top-cos (CONTENT cao / style-chức-năng thấp → bỏ). Luật-0 sạch
(E + facts_about học sẵn). Đây là nửa ĐỘNG-LỰC Hopfield (softmax retrieval) lần đầu dùng thật.

CHƯA wire vào generate — ĐO trước phần graph:
```bash
PYTHONPATH=src python -u scripts/probe_multizone.py \
    --checkpoint checkpoints/network_v11.pkl --log-file out/probe_multizone.log
```
Đọc tay: (a) zone tách đúng vùng nội-dung? (Trần Quốc Toản | gen Z); (b) Hopfield re-rank chọn fact
KHỚP zone (cos cao)?; (c) zone style ('hài hước') cos THẤP/rỗng → xác nhận HOW≠retrieve (thuộc LM);
(d) GHÉP-NAIVE (nối top-fact content zones) = input thô cho compose/RAG-LM. Self-test numpy core PASS.
Dán log → quyết: Hopfield-rerank có hơn facts_about thô? content-zone đủ grounded để wire vào generate?

## Kết quả probe_multizone (2026-07-22) — zone tách PHẦN, rerank có miss

**Chạy:** 5 prompts phức tạp (So sánh Nhà Lý–Trần; Trần Hưng Đạo+Bạch Đằng; hài hước Ngô Quyền; hài hước Trần Quốc Toản+gen Z; buồn mùa thu Hà Nội)

**Đọc tay:**

(a) **Zone tách entity đúng:** `Hà Nội` cos=0.83 → "Thủ đô Hà Nội" ✓; `Trần Hưng Đạo+Bạch Đằng` cos=0.67 → "Trần Hưng Đạo" ✓; `Nhà` → "Nhà Lý" ✓; `buồn câu chuyện` cos=0.77 → "cuộc sống không toàn niềm vui" ✓

(b) **Rerank miss rõ:** `gen` → genetics/ADN (không phải gen-Z); `Toản` → Tây Sơn/Nguyễn Nhạc (không phải Trần Quốc Toản); `mùa thu` → Muông thú (sai); `So sánh` → Biểu đồ cột (sai ngữ cảnh). Nguyên nhân: zone token đơn không đủ ngữ cảnh → rerank chọn surface match.

(c) **Style zone xác nhận HOW≠retrieve:** `hài hước` cos=0.78 → dict definition "vui đùa, gây cười" (không phải template style). Zone style tồn tại nhưng retrieve định nghĩa, không phải cách nói. Xác nhận: style thuộc LM, không thuộc graph-retrieval.

(d) **Function-word leakage:** `về` (cos=0.76) → "Đọc sách" (sai — function word bị classify CONTENT vì gidf cao); `So sánh và` → Biểu đồ cột (zone "so sánh" nhận bài so sánh số liệu thay entity). Cần lọc thêm hoặc tăng ngưỡng cos.

(e) **GHÉP-NAIVE usable:** Prompt "buồn mùa thu Hà Nội" → GHÉP = "Hà Nội" + "cuộc sống buồn" + "Muông thú" — 2/3 zone đúng. Entity zone đủ grounded; style/function zone nhiễu.

**Quyết định:**
- ✅ **Entity-zone rerank ĐÚNG hướng** — zones tách entity tốt, facts_about Hopfield-rerank hơn thô khi token đa (Bạch+Đằng+Hưng+Đạo+Trần grouping)
- ✗ **Token đơn zone unreliable** (`gen`, `Toản`, `mùa thu` mỗi mình → miss)
- → **Fix: require zone size ≥ 2 tokens** HOẶC **zone-context expand** (dùng neighbors từ coref/parents để bổ sung context khi token đơn)
- → **Next: wire content zones vào generate** (pilot: thay facts_about bằng multizone top-facts cho prompt phức), đo A/B đọc tay output relevance trước khi commit vào main gen path

## Kết quả probe_multizone v2 (2026-07-22) — adjacency fix PHẦN

**Fix v2:** adjacency-zoning (chỉ nối liền kề) thay union-find MỌI cặp → giữ thứ tự prompt, tách "Trần Hưng Đạo" ≠ "Bạch Đằng"

**Cải thiện:**
- ✅ "Trần Hưng Đạo và trận Bạch Đằng" → `Trần Hưng` (cos=0.70, "Trần Hưng Đạo" ✓) + `Bạch` (cos=0.51, "Chiến thắng Bạch Đằng" ✓) — 2 zone riêng đúng
- ✅ "hài hước Ngô Quyền" → `hài hước` cos=0.82 tăng từ 0.78

**Vấn đề mới (do adjacency):**
- ❌ "Viết một đoạn văn hài hước về Trần Quốc Toản..." → 7 token đầu gộp 1 zone (`Viết một đoạn văn hài hước về`) — mất `Toản`, `cam`, `gen Z`; only `Trần` zone còn lại → 2 zone thay 6 zone (v1)
- ❌ Threshold similarity cao quá → instruction phrase tokens (gidf thấp, E gần nhau về ngữ nghĩa) liên tục liên kết

**Vẫn còn:**
- ❌ `về`, `và` function word leakage CONTENT (cos=0.76/0.63 → "Đọc sách", "Đồng nhôm")
- ❌ `mùa thu` → Muông thú (embedding miss — "thu" = autumn vs "thú" phonetic neighbor?)
- ❌ `Trần` token đơn → "Trần Văn Thức/Tây Sơn" (sai — cần `Trần Quốc Toản`)

**Chẩn đoán root-cause:**
- Adjacency threshold quá rộng gộp instruction tokens liên tiếp → cần **split tại gidf-drop** (nối liền kề chỉ khi BOTH tokens gidf ≥ threshold, không nối function-word)
- Hoặc: **skip function-words** (lọc gidf < 3.0) trước zoning → entity tokens chỉ còn lại

**Quyết định next:**
- Lọc function-word (gidf < 3.0) TRƯỚC adjacency-zoning → "Trần Quốc Toản bóp nát quả cam gen Z" chỉ còn entity tokens → zone đúng hơn
- Dán log v3 về khi có fix

## Kết quả probe_multizone v3 (2026-07-22) — phrase-unit tốt hơn, threshold drop gây regression

**Fix v3:** phrase-unit + entity-gate (bỏ E-cosine, dùng ps.phrases + cap_class+gidf gate)

**Cải thiện:**
- ✅ "Trần Quốc Toản bóp nát quả cam gen Z" → 6 content zones (v2: 2 zone) — `nát quả cam` cos=0.67 ✓, `hài hước` cos=0.82 ✓
- ✅ "Ngô Quyền" → `của Ngô Quyền` (cos=0.74) → "Trận Bạch Đằng (938)" ✓ (v1/v2 miss)

**Regression:**
- ❌ "buồn mùa thu Hà Nội" → chỉ 1 zone (`Hà Nội`); `buồn`, `mùa thu`, `câu chuyện` bị drop (dưới gidf-gate?) — v2 có 3 content zones cho prompt này
- ❌ `Bạch Đằng` vẫn drop (cos=0.47 < 0.5 threshold) trong "Trần Hưng Đạo và trận Bạch Đằng"

**Vẫn còn:**
- ❌ Function word pairs (`về Trần`, `và trận`, `, hãy`) leak vào CONTENT — v3 ghép function word với entity liền kề → zone nhiễu
- ❌ `Toản` → Tây Sơn (ngữ nghĩa miss, không phải "Trần Quốc Toản")
- ❌ `Quốc` → Liên Hợp Quốc (đơn token ambiguous)

**Root-cause threshold:**
- cos ≥ 0.5 cắt `Bạch Đằng` (0.47) và nhiều semantic zones hợp lý
- Hạ threshold → nhiều noise; giữ threshold → miss entities có idf trung bình
- "buồn/mùa/thu" bị gidf-gate bỏ (thấp hơn entity threshold) — style/context words cần separate treatment

**Hướng tiếp theo (tác giả quyết):**
- Option A: Hạ threshold cos → 0.4 + filter `Bạch Đằng` in
- Option B: Dual-track zoning: entity tokens (gidf≥4, cap) dùng phrase-unit; context/style tokens (buồn/mùa thu/hài hước) dùng semantic threshold riêng thấp hơn (≥0.4)
- Option C: Dừng iterate zoning → wire v3 entity zones vào generate (A/B đọc tay), biết rằng style-zone miss là acceptable (style = LM boundary)

## Kết quả probe_multizone v4 (2026-07-22) — cap-span ĐỘT PHÁ, threshold là vấn đề duy nhất

**Fix v4:** zone = run cap_class≥1 liền kề (danh từ riêng viết hoa) + word-start/gidf gate; bỏ ps.phrases + E-cosine

**Đột phá:**
- ✅ Zero function word leakage — hài hước/về/và/So sánh ALL dropped
- ✅ `Nhà Trần` + `Nhà Lý` tách thành 2 zone entity đúng → "Nhà Trần" (cos=0.77) + "Nhà Lý" (cos=0.72) ✓
- ✅ `Trần Hưng Đạo` (3-token span) → "Trần Hưng Đạo tên thật là Trần Quốc Tuấn" cos=0.72 ✓
- ✅ `Ngô Quyền` → "Ngô Quyền là vị vua sáng lập nhà Ngô" cos=0.74 ✓
- ✅ `Trần Quốc Toản` span (3-token) — gom đúng

**Vẫn còn:**
- ❌ `Bạch Đằng` vẫn dropped (cos=0.47 < 0.5 threshold) — đây là vấn đề DUY NHẤT còn lại
- ⚠️ `Trần Quốc Toản` rerank → "Phạm hậu/Nguyễn Huệ" (embedding không biết Toản là ai → fact miss)
- ✅ "buồn mùa thu" chỉ có `Hà Nội` zone — acceptable (common noun bị drop by design, style=LM boundary)

**Chốt kỹ thuật:**
- Cap-span = đúng approach. Vấn đề còn lại: `Bạch Đằng` threshold (0.47 vs 0.5)
- Fix đơn giản: hạ cos_min → 0.4 hoặc dùng gidf-weighted threshold (entity gidf cao → cho phép cos thấp hơn)
- Rerank miss (`Toản`→Nguyễn Huệ) = training data issue, không phải zoning — NGOÀI scope

**Quyết định (tác giả chỉ đạo):**
- Wire v4 (cap-span) vào generate → test A/B output trên prompt phức
- Hạ threshold 0.5→0.4 để `Bạch Đằng` vào, chấp nhận thêm 1-2 zone noise
- Rerank miss để nguyên (training data limit, out of scope)
## §SFT — dạy VarLM văn phong (local-style, Luật-0, KHÔNG backprop) — 2026-07-22

Bàn "học văn phong không cần LM": style = mô-hình-chuỗi; graph ĐÃ có VarOrderLM (PPM ctx1..4) = "LM
Hebbian nội-tại". SFT-ingest styled = train chính VarLM đó bằng ĐẾM n-gram (Luật-0). 2 tầng đo:

**Tầng 0 (rẻ, KHÔNG data mới) — VarLM có mượt hơn bigram?** (cần varlm sidecar; thiếu → build trước)
```bash
PYTHONPATH=src python -u scripts/probe_style_varlm.py \
    --checkpoint checkpoints/network_v11.pkl --log-file out/probe_style_varlm.log
```
Đọc tay: cột VARLM ít salad/cụm-dài tự-nhiên hơn BIGRAM? Có ⇒ higher-order = local-style → tầng 1.

**Tầng 1 — SFT-ingest styled → VarLM học phrasing → đo TRANSFER:**
```bash
# 1) rebuild VarLM sidecar CÓ corpus styled (đếm phrasing hài/tự-nhiên):
PYTHONPATH=src python -u scripts/build_varlm.py --checkpoint checkpoints/network_v11.pkl \
    --extra-corpus data/text_corpus/styled_sft.jsonl
# 2) chạy lại probe → so VARLM (styled) vs BIGRAM:
PYTHONPATH=src python -u scripts/probe_style_varlm.py --checkpoint checkpoints/network_v11.pkl \
    --log-file out/probe_style_varlm_styled.log
```
`data/text_corpus/styled_sft.jsonl` = seed 15 đoạn **CÓ NHÃN định hướng** ("Viết hài hước về X: …",
"Kể tự nhiên về X: …") — v2 2026-07-22 (tác giả chỉ ra seed v1 thiếu nhãn → VarLM học phrasing hài
NHƯNG không gắn với từ "hài hước" → không trigger được). Nhãn cho 2 cơ chế: (1) **retrieval styled-
parent** (prompt "hài hước X" → chép parent hài, chủ-đề X ĐÃ thấy ✓); (2) **VarLM cục bộ** ("hài
hước →[từ hài]" vài token đầu). M4 mở rộng vài trăm nếu tầng 0 dương.

**Đọc tay (G15):** prompt chủ-đề-MỚI có nhặt giọng tự-nhiên không = TRANSFER local-style.
**⚠️ Ranh giới CỨNG (ghi trước):** VarLM ctx 1-4 → "hài hước" RỚT khỏi cửa sổ sau ~4 token → chỉ điều-
kiện-hoá VÀI TOKEN ĐẦU, không giữ giọng suốt đoạn. Conditional-GLOBAL-style (giữ hài cả đoạn về
chủ-đề MỚI) = long-range → **cần attention/backprop = LM**. Nhãn giúp seen-topic (retrieval) + local,
KHÔNG giúp novel-global. Đây là số cần đo để biết ranh giới local (VarLM làm được) vs global (cần LM).

## §CO_ACT-style (a_style) — tác giả sửa 2026-07-22: Hopfield/CO_ACT ≠ giới hạn ctx

Tôi over-claim "conditional-global cần backprop". SAI: **Hopfield=attention, trọng số CO_ACT (Hebbian
đồng-hiện) — KHÔNG giới hạn ctx như VarLM.** Cơ chế style Luật-0:
1. nhãn "hài hước" → **CO_ACT** → từ-SẮC-THÁI ("kiểu như/hơi lố/chứ/phết"). *(build_varlm KHÔNG đủ —
   liên kết ở CO_ACT, cần ingest g1_passive.)*
2. "hài hước" = topic-seed → **spreading-field** giữ marker kích-hoạt TOÀN cục (không rớt ctx-4).
3. kênh **a_style** ⊙ **a_discourse** (chèn marker tại ranh-giới mệnh-đề, không bừa = né salad).

Cấp: liên-kết(CO_ACT ✅) → giữ-toàn-cục(field ✅) → đặt-chỗ(tune ✅khó) → wit-ngữ-nghĩa(❌ cần LM).
"Văn phong tự nhiên là đủ" = cấp TONE → khả thi Luật-0.

**Measure-first (mắt-xích đầu — liên kết có hình thành không):**

v2 probe (2026-07-23): xếp **RAW / W×IDF / PPMI\*** — nơ-ron «hài hước» yếu không tranh hub;
cột đặc thù (W×IDF=`C×gidf`, PPMI\*=ego-PPMI) mới cho thấy marker.

v3 probe (2026-07-23): thêm **PPMI·C** (`PPMI×log(1+C)`, sàn C) + **ΔPPMI** (PPMI−mean control
`là`/`nghĩa`/`Trần Hưng Đạo`) + lọc surface `:`/punct — tránh one-shot BPE và artifact nhãn.
Đọc **PPMI·C / ΔPPMI**, không chỉ RAW/PPMI\*.

```bash
# 1) baseline v11 — rank v3 (CHẠY M4):
PYTHONPATH=src python -u scripts/probe_style_coact.py \
    --checkpoint checkpoints/network_v11.pkl --log-file out/probe_style_coact_rank.log
# 2) (tuỳ) ingest styled rồi đo lại — so PPMI·C / ΔPPMI before≈after?
PYTHONPATH=src python -u scripts/ingest_styled_coact.py --checkpoint checkpoints/network_v11.pkl \
    --out checkpoints/network_v11_styled.pkl
PYTHONPATH=src python -u scripts/probe_style_coact.py \
    --checkpoint checkpoints/network_v11_styled.pkl --log-file out/probe_style_coact_rank_after.log
```
Đọc tay «hài hước»: PPMI·C / ΔPPMI có marker sắc-thái không (RAW kỳ vọng vẫn `,` `.`).
Có ⇒ liên kết ĐẶC THÙ sống → xây a_style. Vẫn dict ⇒ Style hub / label↔body ingest, không eta mù.

## §HEADS — FACT (multizone) + STYLE (a_style) wire opt-in (2026-07-23)

Hai head tách (giống multi-head LLM, nhưng từ tầng ma trận thưa):

| Head | Lớp / máy | Config (default OFF) |
|---|---|---|
| FACT | `zone_retrieve` cap-span → multi-segment agenda | `ar_multizone=True`, `ar_multizone_cos_min=0.4` |
| STYLE | CO_ACT neighbors của token HOW (cap=0) → Hadamard | `ar_style_weight>0` (vd 0.5), `ar_style_floor=0.2` |

```bash
# A/B 4 chế độ OFF / MZ / STY / BOTH (CHẠY M4):
PYTHONPATH=src python -u scripts/probe_style_heads.py \
    --checkpoint checkpoints/network_v11.pkl --log-file out/probe_style_heads.log
```
Đọc tay: MZ ghép ≥2 entity đúng? STY có `khiếu/biếm/cười`? BOTH không salad/hại fact?
Module: `hca/style_readout.py`, `hca/multizone.py` ← `_plan_from_multizone` / `generate_autoregressive`.

### Kết quả probe_style_heads v1 (2026-07-23, network_v11)

**MZ (FACT) — tín hiệu dương từng phần:**
- «hài hước … Ngô Quyền»: OFF trôi Chăm Pa; **MZ đúng Ngô Quyền + Bạch Đằng** ✓
- «Nhà Lý / Nhà Trần»: 2 segment nhưng seg Trần lệch (Quý Ly); seg Lý đúng dời đô
- «Trần Hưng Đạo + Bạch Đằng»: zone Bạch Đằng OK; zone Trần Hưng Đạo → fact giáo khoa nhiễu
- Rác zone từ token viết-hoa đầu câu (`Viết`)

**STY (STYLE) — seed quá rộng → gần no-op/hại nhẹ:**
- Seed lấy mọi token thường → `đoạn/văn/trận` át `hài hước`; top boost ≠ sắc-thái
- → siết `ar_style_gidf_min=4.5`, `ar_style_max_seeds=2`, bỏ EOS khỏi boost (v1.1)

**Chốt:** bật `ar_multizone` đo tiếp A/B entity; `ar_style_weight` giữ 0 đến khi style_top ra `khiếu/biếm/cười`.

### v5 multizone (2026-07-23) — sau probe heads + token-rewrite âm

- Bỏ single-token == đầu prompt (`Viết`/`Kể`/`So`)
- Giữ thứ tự zone theo prompt (không sort cos)
- `cover_min=0.5`: fact đa-token phải phủ token zone
- `probe_token_rewrite`: rewrite cứng N→N = salad → **không wire**; residual mềm / span thôi

```bash
PYTHONPATH=src python -u scripts/probe_style_heads.py \
    --checkpoint checkpoints/network_v11.pkl --log-file out/probe_style_heads_v5.log
```

**Đọc tay v5:**
- ✅ «Ngô Quyền»: hết zone `Viết`/ChatGPT; 1 seg đúng Ngô Quyền
- ✅ «Hà Nội»: hết seg Sài Gòn; chỉ fact Hà Nội
- ✅ «Lý/Trần»: thứ tự Lý→Trần; seg Lý đúng dời đô; seg Trần vẫn citation (Trọng Kim) — cover chưa đủ
- ❌ «Trần Hưng Đạo»: vẫn fact giáo khoa nhiễu (training/retrieve limit)
- STY: không đổi kết luận — giữ weight=0

### v6 (2026-07-23) — A retrieve + B ingest conditional

**A — cover source∪match∪completion + rank cos×cover + pool×3 entity dài:**
- «Lý»: seg đúng «Nhà Lý tồn tại…» (hơn dời đô đơn)
- «Trần» / «Trần Hưng Đạo»: vẫn citation/giáo khoa — **giới hạn data/theme**, không chỉ cover

**B — ingest conditional label↔thân** (`styled_sft_1k`+seed15 → `network_v11_styled_1k.pkl`):
- 1533 records · 98k cặp CO_ACT · skip_seed=0
- «hài hước» C_seed 1543→2420; ΔPPMI: `biếm`↑#1 + `hệt`/`Thói` mới; vẫn nghiêng tu-từ, chưa `phết`/`hơi lố`
- RAW lộ `Hỏi:`/`Đáp:` — chấp nhận nhẹ (frame protocol)

```bash
# A
PYTHONPATH=src python -u scripts/probe_style_heads.py \
  --checkpoint checkpoints/network_v11.pkl --log-file out/probe_style_heads_v6.log
# B
PYTHONPATH=src python -u scripts/ingest_styled_coact.py --mode conditional \
  --out checkpoints/network_v11_styled_1k.pkl
PYTHONPATH=src python -u scripts/probe_style_coact.py \
  --checkpoint checkpoints/network_v11_styled_1k.pkl --log-file out/probe_style_coact_rank_1k.log
```

### Generate A/B trên styled_1k (2026-07-23) — a_style Hadamard **không xuyên agenda**

```bash
PYTHONPATH=src python -u scripts/probe_style_heads.py \
  --checkpoint checkpoints/network_v11_styled_1k.pkl --style-weight 0.5 \
  --log-file out/probe_style_heads_styled1k.log
PYTHONPATH=src python -u scripts/probe_style_weight_sweep.py   # w=0/0.5/1.5/3.0 + MZ
```

**Đọc tay:**
- MZ vẫn cứu Ngô Quyền / Hà Nội; STY đơn vẫn trôi (Chăm Pa / Hoa7)
- style_top có `biếm`/`khiếu`/`Bạn` (ingest có ăn) nhưng **output MZ+STY ≡ MZ** ở mọi weight 0→3
- Marker sắc-thái **hits=[]** — index-backbone/agenda fact thắng mọi boost soft

**Chốt kỹ thuật:** liên kết CO_ACT ĐỦ để *soi*; kênh Hadamard `a_style` **CHƯA ĐỦ** để *chèn giọng* khi agenda verbatim-fact. Cần cơ chế đặt chỗ (boundary / mixture agenda↔style / tắt backbone tại ranh giới), không chỉ tăng `ar_style_weight`.

### Boundary inject + inventory (2026-07-23)

```bash
PYTHONPATH=src python -u scripts/probe_style_inventory.py \
  --checkpoint checkpoints/network_v11_styled_1k.pkl
PYTHONPATH=src python -u scripts/probe_style_inventory_gen.py
```

**Đọc tay:**
- Boundary inject **chèn được** token — cơ chế đặt chỗ OK
- Raw top CO_ACT: `cốt/kịch/phim` (domain giải trí), không phải giọng
- Graph inventory (bf×style/leak): `trông mong` / `uống cà` — hết domain thô, **vẫn không phải tone hài**
- `khiếu/biếm` có trong raw nhưng bf thấp → dễ bị lọc boundary

**Chốt:** inventory graph từ neighbor CO_ACT **sai nguồn**. Marker giọng nằm trong **thân Đáp**, không phải neighbor nhãn.

### Corpus inventory label-conditioned (2026-07-23)

```bash
PYTHONPATH=src python -u scripts/probe_style_corpus_inv.py
PYTHONPATH=src python -u scripts/probe_style_inventory_gen.py  # graph vs corpus
```

**Fix split:** regex `Đáp:` phải khớp giữa dòng (`Hỏi:… Đáp:…` 1k), không neo `^|\\n` — trước đó chỉ index 15 body seed15.

**Rank:** tf_style × IDF_thân × gidf_nền; bỏ mảnh BPE; phrase cần df≥2.
- humor → `hệt`, `lướt`, `ngơ ngác` (không còn `cốt`); vẫn lẫn topic đời thường (`mua`)
- buồn → `lãng mạn`, `ấm`, `tơi bời`
- Config: `ar_style_corpus_inventory=True` (khi inventory on)

**Generate A/B** (`out/probe_style_corpus_gen.log` → sau siết gidf + re-ingest):
| Variant | hits | chèn |
|---|---|---|
| MZ only | [] | — |
| Graph inv | [] | `trông mong` / `uống cà` |
| Corpus (trước siết) | **`hệt`** | + `mua`/`lãng mạn` |
| Corpus singles gidf≥5.5 | **`hệt`** | `hệt` / `lướt` / `kiệm` — hết `mua` |

Re-ingest Đáp-fix: `records=1533 pairs=60300` → `network_v11_styled_1k.pkl`.

**Next:** mixture agenda↔style (chèn ranh giới vẫn lạc lõng giữa fact); corpus tone dày hơn nếu muốn `phết`/`hơi lố`; default-on inject **chưa**.

### Mixture cửa cosine (2026-07-23) — đo mượt bằng E, không ghép cứng

```bash
PYTHONPATH=src python -u scripts/probe_style_mixture.py \
  --checkpoint checkpoints/network_v11_styled_1k.pkl \
  --cos-thresh 0.12 --log-file out/probe_style_mixture.log
```

**Cơ chế:** `ar_style_mixture=True` → tại boundary chỉ chèn unit có `cos(E[w0], E[head]) ≥ thresh`; boost scale theo cosine. Dưới thresh → bỏ qua (không ép).

**Metric (Luật 0):**
| Variant | mean_bigram_cos | inject_fit | n_style |
|---|---:|---:|---:|
| MZ only | 0.383 | — | 0 |
| Hard boundary | 0.358 (−0.025) | 0.034 | 3 (`hệt/lướt/kiệm`) |
| Mixture cosine | 0.378 (−0.005) | 0.094 | 1 (`kiệm`, cos`.`→0.20) |

**Chốt:** ép chèn **gãy** mượt trên E; cửa cosine **hạn chế đúng** token lệch. Marker còn lại (`kiệm`) vẫn là mảnh BPE/topic — chất inventory, không phải metric. Default-on **chưa**.

**Next:** cải inventory (bỏ fragment BPE / ưu tiên marker tái xuất hiện trong thân); hoặc mixture mid-clause (không chỉ sau `.`) nếu calib cho thấy `,`/`Ċ` khớp style tốt hơn.

### Mixture every-position + bridge cosine (2026-07-23)

```bash
PYTHONPATH=src python -u scripts/probe_style_mixture.py \
  --cos-thresh 0.12 --log-file out/probe_style_mixture_every2.log
```

**Slot-scan (MZ draft):** chỗ giữa câu khớp E hơn punct rất nhiều:
- `hệt` best `là ▷ ▷ người` bridge=0.233
- `lướt` best `nhọn ▷ ▷ xuống` bridge=0.492
- `kiệm` best `cho ▷ ▷ đóng` bridge=0.466
(so với `.`→style ≈ 0.05–0.20)

| Variant | bigram_cos | inject_fit | text note |
|---|---:|---:|---|
| MZ | 0.383 | — | sạch |
| Hard cuối câu | 0.358 | 0.034 | `…chủ. hệt` — lạc |
| Mix every-pos | 0.365 | **0.184** | `Ngô hệt Tổ…` — fit↑ nhưng salad tên |
| Mix + margin (bridge≥agenda+0.05) | **0.380** (−0.003) | **0.177** | `Quyền hệt là` / `vua lướt sáng` / `dân kiệm đánh` — hết `Ngô hệt Tổ`, bigram ≈ MZ |

**Chốt:** user đúng — phải thử mọi vị trí; metric xác nhận mid-clause. Nhưng inventory hiện (`hệt/lướt/kiệm`) không phải particle xen fact lịch sử → E-cosine cao ≠ giọng hợp lý. Cần inventory discourse thật **hoặc** margin/gate chặt (every2).

### Multi-prompt «hài hước» (2026-07-23)

```bash
PYTHONPATH=src python -u scripts/probe_style_mixture.py \
  --variants mz,mix --log-file out/probe_style_mixture_humor.log
```

5 prompt: Ngô Quyền / Trần Hưng Đạo / Bạch Đằng / gen Z / việc học.

| Aggregate | mean_bigram | mean_fit | mean_n_style |
|---|---:|---:|---:|
| MZ | 0.400 | — | 0 |
| Mix every-pos | 0.397 (−0.003) | 0.27 | 2.2 |

Plan hầu hết vẫn `hệt|lướt|kiệm` (gen Z: `đống`). Mix chèn được nhưng: history → salad (`kiệt kiệm xuất`); gen Z/học → retrieve dictionary «hài hước» rồi nhét marker vào định nghĩa. **Không** ra giọng hài kiểu styled_sft.

### Discourse n-gram (2026-07-23)

```bash
PYTHONPATH=src python -u scripts/probe_style_discourse.py
```

**Vì sao uni = hệt/lướt/kiệm:** rank tf×gidf trên thân hài đời thường (topic), không phải cụm tu từ.

**Discourse rank:** bigram/trigram × W_seq × gidf × **điệp âm** ÷ content_leak.
- old_uni: `hệt | lướt | kiệm`
- discourse: `lộng lẫy`, `ngơ ngác`, `xôn xao`, `dọn dẹp`, `vĩnh viễn` (+ vẫn lẫn `cà phê`/`lãng mạn` nếu W_seq mạnh)
- Mix+discourse Ngô Quyền: chèn `lộng lẫy`, **inject_fit=0.63** (trước uni ~0.18); text `vua lộng lẫy sáng lập`

Config: `ar_style_discourse_ngram=True`. Vẫn chưa = “câu hài”; nhưng đã là **cụm sắc thái** thay vì mảnh BPE topic.

### Fact-safe mix + lexicon eval (2026-07-23)

```bash
PYTHONPATH=src python -u scripts/probe_style_fact_mix.py
```

**Inventory (eval-aligned):** glue-frame = df≥15 ∧ min(gidf)≤3.5 ∧ max<5.2 → `thế nhưng`, `rốt cuộc`; redup mạnh; demote `tiết kiệm`/`điện thoại`.  
(`hệt như` BPE = `h`+`ệt`+`như` — chưa bắt được như bigram word-start.)

**Mix:** `ar_style_fact_safe=True` → chỉ xen sau dấu câu thật (`.!?`) + cửa cosine; không xé giữa fact.

| Variant (Trần Hưng Đạo) | fact | bigram | style | note |
|---|---:|---:|---|---|
| MZ | 1.00 | 0.338 | — | fact sạch |
| Mix unsafe | 1.00 | 0.360 | thế nhưng | salad đầu câu |
| **Mix fact-safe** | **1.00** | 0.332 | **thế nhưng** | `…nhà Trần.` → `thế nhưng` → `Ông là tổng…` |

Ngô Quyền fact-safe ≡ MZ (cosine không mở slot) — fact nguyên. Đúng hướng: **fact liền + frame discourse giữa câu**.

### Setup → Pivot → Payoff (2026-07-23)

Humor ≠ bag of words. Compose: **FACT segments** (MZ) → **pivot** (glue discourse) → **payoff** (snippet đời thường).

```bash
PYTHONPATH=src python -u scripts/probe_style_payoff.py
```

**Soft decode (mặc định):** `ar_style_payoff_hard_copy=False`.
- **Pivot lock:** cụm glue ngắn (`thế nhưng` / `rốt cuộc`) luôn emit liền sau fact.
- **Lead lock:** ~8 token đầu thân payoff copy (tránh salad ngay seam).
- **Soft phần còn:** peaky W_seq vs hub+ctx; `fluency_weight·cos(w0,v)`; không slip agenda mặc định.
- Legacy full hard: `ar_style_payoff_hard_copy=True`.

| Variant (Trần) | fact | bigram | seam(nhưng→next) | note |
|---|---:|---:|---:|---|
| MZ | 1.00 | 0.343 | — | fact sạch |
| **MZ+soft** (pivot+lead lock) | **1.00** | 0.281 | 0.046 | `…Trần. thế nhưng Sáng…` — pivot đủ; punch nguyên |
| MZ+hard | 1.00 | 0.281 | 0.046 | tương đương trên snippet ngắn |

Ngô soft: `…tự chủ. rốt cuộc Mùa đông…` (hết mất `rốt` / salad `cuộc đông`).

### Corpus hài gắn entity (2026-07-23)

Nhà văn thêm `data/text_corpus/styled_humor_history.jsonl` (~74 dòng): Setup fact thật → `Thế nhưng`/`Rốt cuộc` → payoff đời thường **neo entity** (Ngô Quyền, Trần Hưng Đạo, Bạch Đằng, Lê Lợi, Quang Trung…).

```bash
python scripts/build_styled_humor_history.py
# retrieve payoff đọc file này trước (không cần ingest)
PYTHONPATH=src python -u scripts/probe_style_payoff.py
# (tuỳ) CO_ACT ingest thêm history:
PYTHONPATH=src python -u scripts/ingest_styled_coact.py --mode conditional \
  --checkpoint checkpoints/network_v11_styled_1k.pkl \
  --history data/text_corpus/styled_humor_history.jsonl \
  --out checkpoints/network_v11_styled_hist.pkl
```

Rank payoff: thưởng neo entity nhẹ + overlap query; không chỉ “xa entity”.

### Spectral stack 4 tầng — không gán nghề tầng (phase stack)

Máy đồng cấu `Attn(Wa)→Amp(Wffn)→top-n` ×4; `Wa/Wffn` từ residual SVD PPMI(`W_seq`).
**Không** gán L1=cú pháp / L2=style trước — chuyên hóa đọc sau probe (ortho, rank, khuôn).

```bash
PYTHONPATH=src python -u scripts/probe_spectral_stack.py \
  --checkpoint checkpoints/network_v11_styled_hist.pkl \
  --log-file out/probe_spectral_stack.log
```

Cờ gen: `ar_spectral_stack=True` (mặc định False). Khi bật: blend `a_out`; **tắt** pivot/lead/hard-copy.

**Forward tranche-1 (2026-07-24 đo):** query=`w0`; S←`W_seq` top-k (+ pending);
attn += β·log1p(W_seq); combine=`residual`; không `succ_L0` inject. Kết quả: Hα
~4.2→~0.7–2; `gold_in_S`↑; chỗ bigram dễ (`.`→`Ċ`, `thật`→`là`) ổn — content
hiếm (`danh` sau `là`) vẫn cần agenda prior.
