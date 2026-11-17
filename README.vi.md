> 🇬🇧 English overview: [`README.md`](README.md)

# ĐỀ ÁN NGHIÊN CỨU: MẠNG NƠ-RON HEBBIAN ĐỘNG (BIOLOGICALLY INSPIRED SLM)
## HƯỚNG TIẾP CẬN: HUẤN LUYỆN THEO QUY TRÌNH TIẾP NHẬN NGÔN NGỮ TỰ NHIÊN (EM BÉ HỌC TIẾNG)

> **Trạng thái stack (2026-07-17):** một đường sinh `generate(multi=True)` → A3 autoregressive;
> **đáy = BPE 32k** (TK_041 — §0.4.3); **245 test pass**.
> Chi tiết: §0.4.1, §0.4.2 (dấu câu = token + EOS), §0.4.3 (BPE), §0.6.9,
> [`docs/NN-AIVIE v4.md`](docs/NN-AIVIE%20v4.md).
> **Train from scratch:** [`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md) ·
> **Kiểm checkpoint sau train (BẮT BUỘC):** [`VERIFY_CHECKPOINT.md`](VERIFY_CHECKPOINT.md).
>
> 🔴 **HAI CẢNH BÁO (2026-07-17) — đọc trước khi tin bất kỳ số nào dưới đây:**
>
> 1. **Checkpoint mặc định `network_v10.pkl` KHÔNG load nổi trên máy dev** (1.85GB đĩa →
>    ~20GB RAM; máy = MacBook Pro **Intel 8GB**, không phải "24GB" như TK_040 M0 ghi). Mọi
>    checkpoint v10-scale mới (kể cả TK_041 B5) sẽ **cùng cảnh**. → **TK_040 (CSR) nay là
>    BLOCKER**, không phải tối ưu. Số đo gần đây chạy trên `network_v10_histwiki.pkl` (349MB).
> 2. **Thước LM cũ đo NHẦM đại lượng** (`eval/metrics.py:116`: `bits = log2(RANK)`, không phải
>    cross-entropy) và **bỏ qua đúng những vị trí mô hình thất bại**. → **Mọi con số "bits"
>    trong tài liệu này trước 2026-07-17 KHÔNG dùng được.** Thước thật:
>    `scripts/eval_bits.py`. Chi tiết + hệ quả: **`TK_042_GENERATIVE_CORE.md` §4**.

---

## 0. LUẬT KIẾN TRÚC (BẮT BUỘC)

**Nghiêm cấm hardcode** bất kỳ từ ngữ, nơ-ron hay ý niệm ngữ nghĩa nào vào **runtime inference** của mạng — kể cả ở giai đoạn POC.

| Được phép | Cấm |
| :--- | :--- |
| Cải tiến **kiến trúc** (HCA, matcher graph-only, fingerprint, zone…) | `if token == "màu"` / danh sách WH / loại câu hỏi / động từ tiếng Việt |
| Cải tiến **training** (teach, G1–G4, curriculum, imprint) | Hardcode tên rule (`vi_QA_hoi_dap`) lúc generate — phải **đọc từ đồ thị** |
| **Frame protocol** (`Hỏi:`/`Đáp:`, `<EOS>`, `Var:`, `Rule:`, `QFocus:`) | Hardcode ý niệm (`AnswerFocus.color`, compositional if/else) |
| **Dữ liệu huấn luyện** (lesson, jsonl, corpus) | Che kết quả yếu bằng heuristic ngôn ngữ → **ảo tưởng sức mạnh** |

**Vì sao:** Ngữ cảnh và câu thoại **muôn hình vạn trạng**. Hardcode chỉ đúng vài case cụ thể; muốn biết **sức mạnh thật** của mạng thì mọi disambiguation phải đến từ cạnh đã học (`rule_colocation`, `REINFORCED`, teach/G2/G3) — runtime chỉ **đọc đồ thị** và **cấu trúc frame**.

**Hướng xử lý khi sai:** thêm example + teach imprint, tăng G2 trên gold path, điều chỉnh kiến trúc readout — **không** thêm rule if/else tiếng Việt.

### 0.0. LUẬT TÁCH NỘI DUNG — MẠNG THUẦN NƠ-RON (BẮT BUỘC, 2026-07-03)

**Mạng nơ-ron KHÔNG chứa text/label/content.** Đồ thị chỉ gồm: **nơ-ron (id)**, **cạnh**
(kiểu + trọng số), **kiểu nơ-ron**, và **thành phần cấu trúc** (chuỗi token-id + index/power).
Mọi **nhãn, tiêu đề, nguồn, và text/content bề mặt** (câu, đoạn, cả episode) — **KHÔNG lưu
trong mạng**, mà để trong **file cache riêng** map `neuron_id → {text, tag, title, source…}`.

| Được phép trong mạng | CẤM trong mạng (→ đưa vào content-cache) |
| :--- | :--- |
| Token-neuron ↔ surface (tầng nguyên tử — ngoại lệ duy nhất) | `tag`="alpaca:0", `key`="wiki:4#2", `title`, `source` trên nơ-ron cha/episode |
| Chuỗi token-id (cấu trúc phân rã: nơ-ron này gồm token-neuron nào, thứ tự nào) | Chuỗi **string** đoạn văn / câu / episode dưới dạng text |
| Cạnh, trọng số, kiểu cạnh, kiểu nơ-ron | Dict khóa bằng **surface-string** (vd `phrase_assoc[("mì","kẹp")]`) |

**Inference & training CHỈ đọc và xử lý trên mạng thuần** (id + kết nối + trọng số + kiểu) —
KHÔNG decode string rồi thao tác chuỗi. Thuộc tính bề mặt của token (dấu câu, viết hoa,
lớp case-fold) phải là **thuộc tính-neuron tính sẵn lúc build vocab** (cờ trên token-neuron),
KHÔNG re-parse string mỗi bước sinh.

**Content-cache CHỈ dùng cho:** render UI, tracing, debug. KHÔNG một đường inference/eval nào
được quyết định đầu ra dựa trên đọc cache.

**Vì sao:** đảm bảo mọi năng lực đến THẬT từ đồ thị nơ-ron — không một AI agent hay dev nào
có thể (vô tình hay cố ý) "ăn gian" bằng cách xử lý trên text/label thay vì trên mạng. Là
Luật 0 siết chặt: Luật 0 cấm hardcode *từ vựng*; luật này cấm cả *đọc content* lúc suy luận.

**Trạng thái tuân thủ (rà soát lại 2026-07-16 — quét toàn repo):**

| nhóm | trạng thái |
| :--- | :--- |
| **Từ vựng tiếng Việt hardcode** | **0** — không WH-list, không danh sách động từ, không phân loại câu hỏi |
| (A) nhãn `tag/key/title/source` | ✅ dọn (TK_012 Phase 1 — content-cache sidecar) |
| (B) ~48 chỗ inference re-parse token-string | ✅ **XONG 2026-07-17 (TK_041)** — `w0_tok.endswith(":")` (chỗ cuối) **chết**: `Hỏi:`/`Đáp:` là special nguyên tử có id cố định → so INT. `is_surface_token(str)` → `vocab.is_surface(id)` (17 call-site). `surface_mask` → **slice** trên dải id, không quét string |
| (C) `phrase_assoc` khóa surface-string | ✅ dọn (TK_012 Phase 3 — nay khóa **fold-group-id** int) |
| **Danh sách dấu câu** (`is_punct`/`PUNCT_TOKENS`) | ✅ **xoá hết 2026-07-15** — dấu câu là token bình thường, IDF tự dìm (§0.4.2) |
| `FOLD_STRIP` + `align._PUNCT_CHARS` | ✅ **XONG 2026-07-17 (TK_041)** — đúng như TK_012 §9.5 tiên đoán, gỡ được **chỉ khi đổi vocab**; TK_041 đổi vocab nên trả luôn nợ. Chết trên đường BPE; **còn ở legacy — CỐ Ý** (cần checkpoint cũ tái lập baseline, xem `vocab.fold_of`) |

Lộ trình dọn: **`TK_012_PURE_GRAPH_CONTENT_CACHE.md`** (§9 = đợt dấu câu 2026-07-15/16) →
**`TK_041_BPE_TOKENIZATION.md`** (đợt tokenizer 2026-07-17 — đóng nốt (B) + `FOLD_STRIP`).

→ **Bảng này nay SẠCH trên đường BPE.** Không còn dòng code nào ở inference hỏi "token này
trông như thế nào".

Chi tiết QA: `docs/QA_CONTEXT_SPEC.md`.  
**Answer closure (thoả mãn câu trả lời):** `docs/ANSWER_CLOSURE_SPEC.md`.

---


## 0.1. ĐIỂM YẾU HIỆN TẠI (POC — cập nhật 2026-07)

> Tổng hợp từ eval harness, Graph UI trace, và train grade 3–7. **Không** che bằng hardcode runtime — xử lý bằng kiến trúc + curriculum + train (Mục 0).  
> **Lưu ý v3:** Khớp nơ-ron cha (`ParentStore.rank_matches`) trên `grade_v3.pkl` đã đạt **QA val 100%**, **paraphrase 100%** (Mục 0.5.12). Các hạn chế dưới đây chủ yếu còn đúng với **chuỗi sinh HCA đầy đủ** (next-token, closure, anaphora) — không nhầm với matching cha v3.

### A. Chất lượng sinh & metric

| Hiện tượng | Mức độ | Ghi chú quan sát |
| :--- | :--- | :--- |
| **Next-token accuracy** | Rất thấp | ~2–5% trên corpus train/val (POC metric trong `run_train_eval.py`) — **đường HCA cũ** |
| **QA val (v3 parent matching)** | **Tốt** | `grade_v3.pkl`: **100%** (7/7) sau matching phân tầng (0.5.12) |
| **Paraphrase held-out (v3)** | **Tốt** | **100%** (8/8) trên `grade_v3.pkl` |
| **Synonym probe (v3)** | Trung bình | **50%** (3/6) — trần E nhỏ + từ đồng nghĩa khác mặt chữ |
| **Câu trả lời đúng một số prompt dạy thuộc** | Khá | Vd. `Hỏi: Thủ đô Việt Nam là gì ?` → đáp đúng fact (CLI `generate` / v3 completion) |
| **Bits-per-byte proxy** | Chỉ smoke | So sánh xu hướng, chưa thay perplexity Transformer |

### B. QA & ngữ cảnh (đồ thị)

| Vấn đề | Nguyên nhân kiến trúc | Ví dụ |
| :--- | :--- | :--- |
| **QFocus bleed** | Cùng anchor (`Con mèo`, `Em`) → nhiều cặp QA học colocation chung; fingerprint gate chưa đủ sau dấu câu | `… thú cưng.` rồi `ngủ.` / `màu trắng.` (từ QA *làm gì* / *màu gì*) |
| **QFocus OOV** | Câu hỏi lạ không có node teach → fingerprint yếu, fallback colocation mơ hồ | `Em thích con gì` (chưa có trong lesson gốc) |
| **Dừng sớm (cụt)** | `qa_answer_complete` đóng khi `has_cont=False` sau 1 token neo | `Em học gì` → `Em học` |
| **Mảnh vị ngữ (cụt vế sau)** | Chain readout từ token cuối; **phạt lặp chủ ngữ**; không có slot đại từ / vế mới sau `,` | `ngủ.` thay vì `, nó thích ngủ` *(v2 HCA; v3 A2 đã có trên `grade_v3_big_a2` — xem 0.6.7)* |
| **Thiếu hồi chỉ (anaphora)** | v3 A2 (2026-07): đại từ học từ W_seq sau `,` + entity attention — **chưa** đủ `nó` gắn thực thể vô tri trên HF tin tức (hay ra `họ`) | v2: chưa `Var:P`; v3 compose: `Huế là cố đô , họ là kinh đô…` |

### C. Trace / generate / matcher

| Vấn đề | Chi tiết |
| :--- | :--- |
| **Graph UI v3 Parent Explorer** | Tab **Parent Explorer**: `/api/v3/complete` — **A3 token-by-token (multi)**; trace A3 trong response; mặc định `network_v10_histwiki_paragraph.pkl` |
| **Graph UI Inference Lab** | `viz/trace.py` (HCA v2) **đã xóa** (TK_019); trace A3 nằm trong `autoregressive.steps` |
| **Rule arbitration (HCA)** | Slot đóng, directed colocation, EOS sau frame — xem B.9 (một phần đã làm) |
| **G3 dopamine** | Thường **bỏ qua** trong train mặc định (`--g3` tắt) — LTD/reward chưa củng cố đủ |
| **Decode** | Chưa có top-k / top-p / temperature calibrated như `sample.py` GPT-2 |

### D. Quy mô & hiệu năng

| Vấn đề | Chi tiết |
| :--- | :--- |
| **Vocab & graph nhỏ** | ~2–3k token, ~10⁴–10⁵ cạnh/checkpoint — xa SLM production |
| **$K_{\max}=16$** | Coreference / mệnh đề dài (>16 token) thiếu ngữ cảnh (README Mục III.1) |
| **Retroactive Wave** | Thiết kế có (README III.6), **chưa triển khai** — đại từ / ambiguity khó |
| **GPU batch Hebbian** | Ma trận hóa $H^TH$ ở quy mô lớn — roadmap, chưa có trong POC |
| **Trace API chậm** | `/api/trace` có thể ~60s+ (load checkpoint + scoring từng bước) |

### E. Dữ liệu & curriculum

| Vấn đề | Chi tiết |
| :--- | :--- |
| **Corpus vẫn mỏng** | Rất tốt | ~250 cặp QA thực tế / ~660 câu kiến thức các cấp lớp (Cập nhật 2026-06-29) |
| **HF general** | Stream/lọc câu Việt — chất lượng không đồng đều |
| **Eval** | QA val đọc từ jsonl; chưa đo văn phong / anaphora / subject QA riêng |

### F. Hướng xử lý (không hardcode)

> **Roadmap chi tiết:** Mục **0.2** và [`docs/ANSWER_CLOSURE_SPEC.md`](docs/ANSWER_CLOSURE_SPEC.md).

1. **Answer closure** — `assess_answer_closure()`: SATISFIED chỉ khi agenda + coverage + không fp cont. + closure cuối  
2. **Sync trace ↔ generate** + query readout + early stop theo closure (không chỉ `.`)  
3. **Teach compound + Var:P** + G2 gold path nhiều vế  
4. **Siết colocation theo QFocus** sau ranh giới câu / SECTION_PAUSE  
5. **Retroactive / query readout** mở rộng (HCA v2)  
6. **Distillation trace** từ LLM mã nguồn mở → G2 (xem `docs/study/01_OPEN_LLM_TO_HEBBIAN_TRANSFER.md`)  
7. **Bật G3** có chọn lọc trên QA mask  

---

## 0.2. BÀI TOÁN & ROADMAP — ANSWER CLOSURE (2026-06-28)

> ⚠️ **LỖI THỜI (v2-era) — xem Mục 0.6.** Mục này + `docs/ANSWER_CLOSURE_SPEC.md` viết thời **v2
> dạy ngữ pháp** (QFocus/fingerprint/Rule/Var trên đường HCA `hca/generate.py`). v3 (nơ-ron cha
> + slot hub) **bỏ toàn bộ scaffolding đó**. Giữ lại **ý niệm** hữu ích — agenda ý cần phủ,
> đóng closure, sinh phân đoạn — nhưng **tái đúc trên khái niệm v3** (parent/hub/entity-focus),
> KHÔNG dùng lại QFocus/Rule. Lộ trình mới ở **Mục 0.6**.

> Ghi lại đề bài, trao đổi thiết kế, và kế hoạch implement.  
> Spec chi tiết (v2, tham khảo lịch sử): [`docs/ANSWER_CLOSURE_SPEC.md`](docs/ANSWER_CLOSURE_SPEC.md).

### 0.2.1. Bài toán

Mạng sinh câu trả lời trong frame `Hỏi: … Đáp: …`. Cần cơ chế **per-token** trả lời câu hỏi:

> *“Câu trả lời đã thoả mãn ý người hỏi chưa? Khi nào được phép sinh `<EOS>`?”*

**Hai lỗi đối lập quan sát được:**

| Lỗi | Ví dụ | Hậu quả |
|:---|:---|:---|
| **Cụt sớm** | `Em học gì` → `Em học` | Thiếu nội dung |
| **Thừa / mảnh vị ngữ** | `Con mèo là thú cưng.` rồi `ngủ.` `màu trắng.` | Bleed QA khác; không phải câu hoàn chỉnh |

**Mục tiêu văn phong (grade 4+):** câu ghép có đại từ, không lặp chủ ngữ:

```
Con mèo màu trắng, nó thích ngủ, là thú cưng.
```

**Mục tiêu câu dài (grade 6–7):** essay nhiều ý (vd. mô tả hổ, sách đỏ VN) — **không** fit một `QFocus` hash full-text.

### 0.2.2. Yêu cầu (tóm tắt)

1. Runtime **đánh giá closure mỗi bước token** — không hardcode từ/loại câu hỏi (Mục 0).
2. **INCOMPLETE** → suppress `<EOS>`, tiếp tục sinh.
3. **SATISFIED** → boost `<EOS>`, dừng generate.
4. Dấu `.` `,` `\n` là **discourse** học trên đồ thị — **`.` ≠ tự động dừng**.
5. Graph UI **trace** ≡ CLI **`generate()`** (cùng query readout + closure).
6. Essay: **agenda segment** + **coverage salient Q** — spec trong ANSWER_CLOSURE_SPEC.

### 0.2.3. Kết luận trao đổi thiết kế

#### QFocus

- Node **cấu trúc** `QFocus:{signature}`; teach imprint path câu trả lời đúng cho Q đó.
- **Không** hardcode ngữ nghĩa — đọc cạnh đồ thị lúc runtime.
- **Giới hạn:** một signature ≈ một câu Q cố định; paraphrase/essay cần hub + segment (không scale hash cả đoạn).

#### Bleed `Con mèo`

- Đồ thị **có** kiến thức `ngủ`, `màu trắng` (từ QA khác cùng anchor).
- Sinh ra **mảnh vị ngữ** vì: (a) trace không dừng closure; (b) thiếu `Var:P` / section restart sau `,`.

#### Điều kiện SATISFIED (AND — chưa code)

```
SATISFIED ⇔ agenda_rỗng
         ∧ ¬fingerprint_continuation
         ∧ coverage_salient_Q == 100%
         ∧ closure_token_cuối (. ! ? trên segment cuối)
```

Trạng thái trung gian **SECTION_PAUSE**: vế/section xong, còn segment khác — cho phép `.`/`,` nhưng **không** EOS.

#### Câu hỏi essay

- Gold nhiều section `[1]…[5]` trong lesson.
- Distill từ LLM mã nguồn mở → jsonl → G2 (`docs/study/01_OPEN_LLM_TO_HEBBIAN_TRANSFER.md`).

### 0.2.4. Giải pháp đề xuất (implementation)

| Phase | Nội dung | Ưu tiên |
|:---|:---|:---:|
| **P1** | `assess_answer_closure()`, bỏ “`.` ⇒ dừng”, sync `trace.py`, tests | ⏳ chưa xong |
| **P2** | Lesson compound, `Var:P`, `must_cover`, anaphor sau `,` | Tuần 1 |
| **P3** | Essay segment, distill gold, G4 `[Verify]` | Tuần 2+ |

**File chính:** `hca/qa_context.py` (closure), `hca/generate.py`, `viz/trace.py`, `teaching/compile.py`, `tests/test_answer_closure.py` (mới).

**Acceptance Phase 1:**

- `Con mèo là gì` — CLI = trace, không token thừa sau `.`
- `Em học gì` — không dừng sau một token neo nếu chain teach còn
- Inference Lab hiển thị `closure_status`

### 0.2.5. Tài liệu liên quan

| File | Nội dung |
|:---|:---|
| [`docs/ANSWER_CLOSURE_SPEC.md`](docs/ANSWER_CLOSURE_SPEC.md) | Spec đầy đủ: states, bảng quyết định, curriculum, test cases, checklist P1–P3 |
| [`docs/QA_CONTEXT_SPEC.md`](docs/QA_CONTEXT_SPEC.md) | QFocus, case studies bleed/cụt, essay |
| [`docs/study/01_OPEN_LLM_TO_HEBBIAN_TRANSFER.md`](docs/study/01_OPEN_LLM_TO_HEBBIAN_TRANSFER.md) | Distill essay gold từ open LLM |
| [`docs/study/00_LLM_TO_HEBBIAN_MAP.md`](docs/study/00_LLM_TO_HEBBIAN_MAP.md) | Bản đồ khái niệm LLM → Hebbian |

---

## 0.3. COMPOSITION — SINH CÂU BẰNG KẾT HỢP (2026-06-29)

> Spec đầy đủ: [`docs/HCA_COMPOSITION_SPEC.md`](docs/HCA_COMPOSITION_SPEC.md).

### 0.3.1. Nguyên lý

LLM **không phát minh từ hư vô** — nó **kết hợp / nội suy / khái quát hóa / ngoại suy** từ thống kê đã học. nn-aivie theo **cùng nguyên lý**, nhưng bằng **cạnh đồ thị + chồng chập kích hoạt**, học từ training.

**Bắt buộc:** Năng lực này phải đến từ **cải tiến kiến trúc mạng + training**, **không** từ code rule / routing từ vựng (Mục 0). Composition là **hành vi mặc định của readout**, không phải "nếu compound thì chọn rule compound".

### 0.3.2. Khiếm khuyết kiến trúc hiện tại

| | Khiếm khuyết | Hệ quả |
|:---|:---|:---|
| **K1** | `QFocus` *localist* (một node / cả câu) | Không khái quát hóa paraphrase |
| **K2** | `query_ids=()` — nhánh distributed **bị tắt** | G2 đã train `Q→A` nhưng infer không đọc |
| **K3** | Readout 1-hop + argmax winner-take-all | Chỉ nhớ một path, không ghép tổ hợp |

### 0.3.3. Bốn cơ chế (tổng quát, không rule)

1. **Distributed query** — câu hỏi = chồng chập kích hoạt token nội dung; fingerprint hạ xuống *một nguồn*. → khái quát hóa.
2. **Spreading activation đa hop** — activation lan 2–3 hop có decay; nội suy nổi lên ở hub chung. → kết hợp.
3. **Emergent consolidation** — hai ngữ cảnh cùng kích hoạt một đích lặp lại → tự sinh cạnh `EMERGENT`. → Hebb thật.
4. **G4 Verify** — token không có đường dẫn học được → ức chế. → ngoại suy có kiểm soát (không bịa fact).

### 0.3.4. Thứ tự + đo lường

| Bước | Cơ chế | Trạng thái |
|:---|:---|:---:|
| **#1** | Distributed query | ✅ Đã áp dụng IDF để neo từ khóa |
| **#2** | Spreading activation | ✅ Đã implement |
| **#3** | Emergent consolidation | ✅ Đã implement |
| **#4** | G4 Verify | ✅ Đã implement |

Mỗi bước: **implement → train → đo → giữ nếu cải thiện thật**. Phép đo khái quát hóa: **paraphrase chưa dạy** (`Mèo là gì`, `Con mèo là con gì`) phải sinh đáp án hợp lý.

---

## 0.4. CẢI TIẾN GẦN NHẤT (Cập nhật 2026-06-29)

> Các mục **0.5.8–0.5.12** ghi tiếp scale HF, vá matching, phân tầng, và Embedding UI (2026-06-30 → 2026-07-01).

Sau khi hoàn thiện 4 cơ chế Composition, hệ thống đã được nâng cấp toàn diện về dữ liệu và kiến trúc Neo câu trả lời:

1. **Loại bỏ Toy Data, Thay bằng Kiến thức thực tế (Real Knowledge)**:
   - Các ví dụ đồ chơi (`"Con mèo là thú cưng"`, `"Hỏi: gì Đáp: toán"`) đã bị loại bỏ hoàn toàn khỏi các lesson và corpus.
   - Tập dữ liệu môn học lớp 3–7 được mở rộng lên ~250 cặp QA và ~660 câu kiến thức thật.
   - Các corpus ngôn ngữ (`curated`, `copula`, `compound`) nay được **dẫn xuất tự động** từ chính các fact này, giúp mô hình học ngữ pháp từ kiến thức thật.
2. **Khắc phục tràn kiến thức chéo lớp (Cross-grade Bleed)**:
   - Lesson `vi_QA_hoi_dap` không còn dùng chung. Build script giờ đây tự động sinh `vi_QA_hoi_dap.gradeXX.lesson.yaml` riêng cho từng lớp dựa trên tập QA của riêng lớp đó.
   - Giảm `eta_teach` xuống `0.25` để tránh imprint quá tay, cho phép dataset `qa_corpus` (G1/G2) chiến thắng tần suất. Lớp 4 không còn bị trả lời nhầm kiến thức lớp 3.
3. **Refactor Answer Anchor theo IDF**:
   - Hàm `answer_anchor_ids` (chọn token từ câu hỏi để neo đáp án) trước đây chỉ lấy token salient đầu tiên (ví dụ: chữ `Hai` trong `Hai Bà Trưng`), khiến mô hình lan man.
   - Nay đã đổi sang tính **IDF (Nghịch đảo Out-degree trên đồ thị)**: Chọn token đặc thù nhất, phân biệt nhất (ví dụ `giặc Hán`, `năm 1954`, `sắt`). Nhờ đó mô hình trả lời thẳng vào trọng tâm (đối tượng trực tiếp) ngay ở các token đầu tiên.
4. **Kiểm thử**: full pass **131** tests (2026-07-09, sau hợp nhất A3); lịch sử eval v3 trên `grade_v3.pkl`; stack hiện tại v10 — xem §0.4.1, 0.6.9.

---

## 0.4.1. HỢP NHẤT ĐƯỜNG SINH A3 MULTI + STACK v10 (Cập nhật 2026-07-09)

> **Chỉ đạo tác giả:** chỉ giữ **một** substrate sinh — A3 token-by-token, **multi=True** (gom
> nhiều fact về thực thể). Bỏ song song hybrid/crisp/essay/composed/pool-OFF và eval tương ứng.

### Đường sinh duy nhất

```
prompt → generate(multi=True) → generate_autoregressive()
         ↑ agenda từ retrieval (best_match / facts_about / discourse pool)
         ↑ paragraph pool LUÔN BẬT (TK_035)
         ↑ VarLM cluster-aware (TK_031 + ISG ctx)
```

| Thành phần | File | Ghi chú |
| :--- | :--- | :--- |
| **API sinh** | `hca/generate_hebbian.py::generate` | Điểm vào duy nhất; `multi=True` mặc định cho viz/eval |
| **Vòng A3** | `generate_autoregressive` | 4 kênh Hadamard + agenda + EOS |
| **Paragraph pool** | `_discourse_pool_indices` | Gộp conv ∪ paragraph; không còn cờ `ar_paragraph_pool` |
| **VarLM** | `core/varlm.py` + sidecar `.varlm.pkl` | `--use-clusters` khi build; encode ctx qua ISG cluster |
| **Export** | `hca/__init__.py` | Chỉ export `generate`, `HCAConfig` |

### Đã xóa (code + scripts + tests)

| Nhóm | Chi tiết |
| :--- | :--- |
| **Mode sinh cũ** | `generate_hybrid`, `generate_composed`, `generate_essay` |
| **Eval/scripts** | `scripts/eval_v4.py`, `scripts/verify.py`, `scripts/eval_paragraph_ab.py`, `scripts/essay_ab_t3.py` |
| **Tests** | `tests/test_eval_v4.py`, `tests/test_hf_corpus.py`; essay/episode/SFT/tk010 legacy trong `test_parent_store.py`; ablation legacy trong `test_generate_hebbian.py` |
| **Graph UI** | Dropdown Hybrid/Compose/Essay → chỉ **A3 multi**; `mode` API chỉ `autoregressive` |

### Checkpoint mặc định (cập nhật 2026-07-16)

| Mục đích | File | Cha |
| :--- | :--- | ---: |
| **CLI/eval + Graph UI** (`core/paths.py`) | **`network_v10.pkl`** — train 2026-07-16, có EOS + dấu câu trong thống kê | **65 098** (vocab 126 518) |
| VarLM sidecar | `network_v10.varlm.pkl` — cluster-aware, `n_tokens` 9.4M | build **SAU** train |

Train 7.7 giờ / 8 dataset; bake: FRAME_BRACKET **1 812 272** cạnh, **endpoints 357 138**
(token/cluster/phrase = 47793/259095/50250 — đây là lần đầu `phrase_endpoints` TK_037 **thật sự
sống** trên production; các bản `network_v10_histwiki*.pkl` cũ có `endpoints=0` và **kẹt cứng**,
xem TK_037 §16), skip_spans 30 000.

**Bản cũ (lịch sử):** `network_v10_histwiki.pkl` (151 430 cha) / `_paragraph.pkl` (176 430) —
scale TK_024 C, guard QA/Para/Syn 86/88/67. **Không còn là mặc định**; chúng thiếu
`hlink_endpoints` và train trước đợt dấu câu/EOS (§0.4.2).

### Đo discourse (thay eval đa-mode)

`scripts/eval_discourse_prompts.py` — prompt từ corpus (history_wiki_para, statements_doc, conv,
common_noun, frame); sweep `ar_varlm_weight` 0 / 0.35 / 0.5; **chỉ** `generate(multi=True)`.
Log: `out/tk035_discourse_prompts_*.log`.

### Kiểm thử

**131 test pass** lúc hợp nhất A3 (2026-07-09) → **184** (2026-07-16, sau đợt dấu câu/EOS §0.4.2).

> ⚠️ Chạy `pytest -m "not slow"` — **4 giây** thay vì >10 phút. Hai test `@pytest.mark.slow`
> load checkpoint 126 MB; guard của chúng là `skipif(not exists)` nên **skip trên CI nhưng CHẠY
> trên máy có checkpoint**. Quy ước ở `pyproject.toml:34`.

Chi tiết ticket: `TK_019_LEGACY_CLEANUP.md` §7, `TK_035_PARAGRAPH_UNIT_TRAINING.md` §7i–7j.

---

## 0.4.2. DẤU CÂU LÀ TOKEN BÌNH THƯỜNG + EOS CUỐI ĐƠN VỊ (2026-07-15/16)

> **Chỉ đạo tác giả:** *"không phân biệt text hay dấu câu, dấu câu cũng là token như bình
> thường — bỏ tất cả chỗ nào đang hard code dấu câu"*, và *"nên có một token EOS ở cuối câu
> train"*. Chi tiết + số đo: **`TK_012_PURE_GRAPH_CONTENT_CACHE.md` §9**.

### Vòng luẩn quẩn đã gỡ

```
_is_match_token():  if vocab.is_punct(tid): return False   ← dấu câu bị loại khỏi thống kê
        ↓
_gdf['.'] không bao giờ được cộng  →  _gidf('.') = 7.168 = CAO NHẤT corpus (gấp 6× 'Việt')
        ↓
~24 filter is_punct rải khắp inference tồn tại để CHE hậu quả đó
```

Đếm dấu câu vào `_gdf` → `.` có df≈n → **gidf ≈ 1.1 = thấp nhất** → **tự chìm bằng thống kê**,
không cần danh sách nào nói hộ. Đo trên `network_v10` (2026-07-16): `'.'` gdf=38 976 **gidf=1.513**
(đối chiếu `Việt` 3.781, `là` 1.894).

**Đã xoá:** `vocab.PUNCT_TOKENS` / `is_punct()` / `_is_punct`, `parent_store._is_punct` (code
chết), ~24 call site (`generate_hebbian` / `skip_span` / `frame_bracket` / `g2_imitation` /
`consolidate` / `ingest_*` / `export_embedding`), `_strip_leading_punct_ids` (chống `. .` = tri
thức chính tả viết tay), `verify_v3` miễn-trừ G4 (A/B sinh **7/7 giống hệt** — miễn trừ thừa
suốt: grounding của `.` = 227 vs ngưỡng 0.04), `eval/v4.py` (422 dòng mồ côi từ TK_019).
**KHÔNG còn dòng code nào hỏi "token này có phải dấu câu không".**

### Mốc ranh giới: từ danh sách → HỌC ĐƯỢC

`at_boundary` (TK_009) cần biết *"w0 có kết thúc đơn vị không"* → trước đây trả lời bằng
`is_punct(w0)`. Thay bằng tín hiệu học từ **EOS nối cuối mỗi đơn vị train** (`_with_eos`):

**LIFT** = `P(EOS|w0) / P(EOS) nền` ≥ `ar_eos_boundary_min_lift` (2.0). Dùng lift chứ không
phải P tuyệt đối vì **P phụ thuộc độ dài đơn vị** — corpus 1 câu/đơn vị cho P=1.000, corpus
bài dài cho P=0.057; **lift gần như không đổi** (6.0 vs 7.0).

Hai tín hiệu dẫn xuất khác đã thử và **thất bại** — chúng không tách nổi ranh giới khỏi nội dung:
entropy W_seq (`.`=5.93 nhưng `là`=5.45), tỉ lệ đứng-cuối-chuỗi (`,`=`:`=`Đáp:`=**0.000**, ngang
từ nội dung). **Đồ thị không mã hoá "kết thúc đơn vị" cho tới khi dữ liệu train đánh dấu nó.**

### BA đường nạp — EOS phải có ở cả ba (đã vấp 2 lần)

| lệnh | đường |
| :--- | :--- |
| `scripts/train.py --manifest` | `pipeline.run_manifest` → `pipeline.ingest_text_file` |
| `scripts/train_v3.py --extra-dir` | `ingest_statements` / `ingest_qa` |
| `scripts/build_varlm.py` | tự tokenize → `lm.add_sequence` |

Sót `pipeline.py` → suýt bake 7.7 giờ ra checkpoint không mốc kết. Sót `build_varlm.py` → verify
FAIL `varlm học mốc kết = 0 ctx có EOS`. **→ Sau mỗi lần train PHẢI chạy
[`VERIFY_CHECKPOINT.md`](VERIFY_CHECKPOINT.md)** (`scripts/verify_checkpoint.py`, 10 mục, exit 1
nếu trượt) — log train **không** phát hiện được: nó không nhắc EOS, không in gidf.

### Kết quả sinh (eval discourse, `network_v10` 2026-07-16)

Dấu câu **quay lại và đặt đúng chỗ** — trước đây agenda bị lọc sạch nên A3 không thể sinh:

```
CŨ (còn is_punct): 'phở là món ăn rất phổ biến ở Việt Nam  nhưng phở Hà Nội có những cách…'
MỚI:               'phở là món ăn rất phổ biến ở Việt Nam , nhưng phở Hà Nội có những cách … .'
```

`multi%` tăng 50→90% (`common_noun`), 60→100% (`statements_doc`) — nhưng **phần lớn là
tautology**: `multi` = `sents>=2 or ("," in out …)`, tức nó đo *"có dấu câu không"*. Thước thật
là **`copy`** (LCS với corpus, thay `overlap` vốn **luôn = 1.00** ở 129/129 dòng): `conv`
copy=0.22 ≈ **sàn** (sinh thật), `history_wiki_para` 0.57 (chép nửa).

### Sổ nợ

~~`FOLD_STRIP` + `align._PUNCT_CHARS`~~ + ~~`w0_tok.endswith(":")`~~ → **TRẢ XONG 2026-07-17
(TK_041, §0.4.3)** — đúng như dự đoán, gỡ được chỉ khi đổi vocab. Còn: `resync_match_ids` chạy
vô điều kiện mỗi lần load. Xem TK_012 §9.5.

---

## 0.4.3. BPE TOKENIZATION — ĐỔI ĐÁY CỦA THÁP (2026-07-17)

> **Chỉ đạo tác giả:** *"tách token giống mạng LLM, dùng BPE — như vậy không bỏ qua bất kỳ cái
> gì, kể cả dấu cách"* + *"cải tiến để training tốt cho cả các số"*. **Chốt:** BPE **tự train
> trên corpus nhà** (Luật 0 sạch — KHÔNG mượn tokenizer LLM); **đáy = token BPE, mọi tầng trên
> GIỮ NGUYÊN** (ISG/cụm/cha/hub/theme không đụng một dòng). Chi tiết: **`TK_041_BPE_TOKENIZATION.md`**.

### Số đo chốt (toàn corpus 99.2 MB / 65 160 đơn vị)

| | trước | **sau** |
| :--- | ---: | ---: |
| Token/corpus | 18 211 343 | **19 631 500 (+7.8%)** |
| Vocab | **126 516 — MỞ vô hạn** | **32 002 — ĐÓNG** |
| Fertility | — | **1.195 tok/từ** · 5.05 bytes/token |
| Roundtrip byte-exact | **bất khả** | **65 160/65 160 đơn vị** |

**Nỗi sợ "chuỗi nổ 2–3× → RAM chết" là SAI.** Tiếng Việt là ngôn ngữ âm tiết → BPE phủ gần trọn
âm tiết ngay ở vocab 8k → chuỗi chỉ **+7.8%**, RAM CO_ACT ~+8% ⇒ TK_040 **không** chặn TK_041.

**Đối chứng ngoài** (tiktoken, cùng 60 bài history): ta **1.256** tok/từ · GPT-4o (o200k, vocab
200k) **1.482** · GPT-4 (cl100k, vocab 100k) **2.290**. → Tokenizer 32k tự train **tách tiếng
Việt tốt hơn GPT-4 ~1.8×** với vocab nhỏ hơn 3×. Tự train **thắng cả hai mặt** — không phải đánh
đổi purity lấy chất lượng.

### Hai phát hiện định hình thiết kế

1. **53% vocab cũ là hapax** (31 539/59 480) cho chỉ **1.05%** lượt token — quá nửa token-neuron
   là vật thể một-lần-dùng, `df=1`, **không bao giờ được củng cố Hebbian**. Mỗi lỗi chính tả
   trong corpus để lại một xác nơ-ron. Vocab đóng ⇒ hết.
2. **Số học đang BẤT KHẢ về kiến trúc, không phải "học kém"**: token chứa chữ số = **8 590 loại
   = 14.4% vocab** cho 2.5% lượt ⇒ `1288` và `1287` là **hai atom vô can**. Mạng **không có cơ
   quan** để học số. Digit-split (cưỡng chế ở **pre-token regex**, merge table không lách được)
   → **10 nơ-ron chữ số**, giá **+0.1%** độ dài.

### Bốn bất biến (mỗi cái một test — `tests/test_tokenizer.py`)

1. **VÔ-TỔN-HAO**: `decode(encode(x)) == NFC(x)` **từng byte** — kể cả khoảng trắng, `\n`, emoji,
   chữ Hán. Đây là định nghĩa vận hành của *"không bỏ qua bất kỳ cái gì"*; tokenizer cũ **không
   thể** pass.
2. **VOCAB ĐÓNG** · 3. **CHỮ SỐ RỜI** · 4. **SPECIAL NGUYÊN TỬ** (`<EOS>`/`Hỏi:`/`Đáp:`).

### Hệ quả kiến trúc

- **`\n` nay là token** (`normalize_ws_keep_newlines`) — trước đây `pipeline` **đốt nó ngay lúc
  nạp** rồi TK_035 phải dựng lại "paragraph unit" ở tầng trên. Đo: ISG học được cụm chứa `Ċ`
  (`'Tiếng Việt:Ċ'`, `'ĊNăm 1'`).
- **BPE ≠ thay ISG.** Chúng là **cùng thuật toán, khác vai**: BPE gộp **trong từ** (`n`+`g`→`ng`,
  offline, đóng băng); ISG gộp **qua dấu cách** (`Tiếng`+`ĠViệt` → ClusterNeuron power=375,
  online, rewire được, **mạng đọc được**). BPE **không** gộp qua dấu cách — pre-token regex cấm,
  y hệt GPT-4/Llama (đo: `of the` vẫn là 2 token trong mọi tokenizer GPT).
- **Mốc kết SỐNG SÓT** qua đổi tokenizer: `'.'` P(→EOS)=0.054 **lift 18.1** (ngưỡng 2.0);
  `'.'` gidf **1.641** (tự chìm bằng thống kê — §0.4.2 giữ nguyên hiệu lực).
- **Verify 17/17** (10 cũ + 7 mới); **245 test**; legacy `grade_v3.pkl` load/sinh nguyên vẹn.

### ⚠️ Lỗi chính tả — CONTAINMENT, không phải ROBUSTNESS

`Lập` → `['Lập','Ġtrình']` · `Lậb` → `['L','ậ','b',…]` → **chia 0 nơ-ron**. BPE cho: không
`<UNK>`, không mọc vocab (0 nơ-ron ma). BPE **KHÔNG** cho: hiểu typo. Đo: typo lẻ **sống** (token
khác giữ neo IDF), sai **hệ thống** (mất dấu: `Thu do Viet Nam la gi`) → **lạc hẳn**. Cái cứu ca
đầu là **IDF**, không phải BPE. Trần thật, cùng họ với trần synonym. TK_041 §9.1.

### Bài học phương pháp: lỗi tokenizer là lỗi IM LẶNG

Bốn bug thật, **không cái nào crash / làm test đỏ** — id vẫn hợp lệ, chỉ trỏ sai. Mỗi cái thoát
một lớp phòng thủ khác nhau (TK_041 §5.2):

| Lớp | Bắt được | MÙ với |
| :--- | :--- | :--- |
| Test roundtrip | marker `Ġ` trong text gốc | chữ số rớt alphabet (**roundtrip VẪN đúng**) |
| Test bất-biến-riêng | `52` → `['5','<0x32>']` | prompt còn `.split()` |
| **Đọc tay đầu ra** | `'Hỏi: Thủ đô'` → `'Hỏi:Thủđô'` | prompt không chuẩn hoá như train |
| **Người đọc CODE hỏi "tại sao"** | `.strip()` (tác giả tìm ra) | — |

→ Bất biến **TRAIN ≡ PROMPT** nay khoá bằng test. Không lớp nào thay được lớp nào.

### Còn lại

**B5** train full v11 (~8h — ⚠️ xem cảnh báo RAM đầu README) · **B6** retune (`varlm.max_ctx`
4→5 vì fertility 1.195; `max_len`; **ngưỡng promote ISG** — đáy vụn hơn ⇒ phân bố đếm đổi) ·
**B7** số học (N1 phân cách số, N2 curriculum, eval `arith`).
⚠️ **TK_042 §3 đã đo: B5 sẽ KHÔNG sửa được chuyện verbatim** — đó là lỗ hổng **mô hình**, không
phải tokenizer.

---

## 0.5. KIẾN TRÚC V3 — NƠ-RON CHA + PATTERN COMPLETION (Cập nhật 2026-06-30)

> Spec thiết kế: [`docs/study/02_HYBRID_HEBBIAN_MATRIX.md`](docs/study/02_HYBRID_HEBBIAN_MATRIX.md). Nguồn ý tưởng: `docs/NN-AIVIE v3.docx`.

### 0.5.1. Tinh thần cốt lõi

> Khi đưa **một câu / cụm / cả đoạn** vào lúc train, v3 **coi nó như MỘT nơ-ron cha**, rồi **mới phân rã** xuống nơ-ron trung gian + token (index + power). Sinh = **lan xuôi** (prompt → cha khớp nhiều con nhất) + **lan ngược pattern completion** (đọc con của cha theo index, dội ra phần thiếu = đáp sau `Đáp:`).

**Khác biệt mấu chốt với hướng trước:** KHÔNG dạy ngữ pháp (bỏ NVL/Rule/Var); ngữ pháp & cấu trúc đáp **tự nổi** từ dữ liệu thật. Embedding **dẫn xuất từ chính đồ thị** (PPMI+SVD), không pretrain — tôn trọng Luật 0.

### 0.5.2. Đã hoàn thiện

| Thành phần | File | Trạng thái |
| :--- | :--- | :--- |
| **Nơ-ron cha** (1 ví dụ = 1 nơ-ron; phân rã Re-Pair; lan xuôi `rank_matches` phân tầng IDF; lan ngược `completion`) | `core/parent_store.py` | ✅ (15+ test) |
| **Embedding suy-từ-đồ-thị E** (PPMI → SVD; mean-pool concept; cosine fuzzy — mặc định tắt fuzzy) | `core/embedding.py` | ✅ (7 test) |
| **W_seq** — chuyển dịch vị trí bigram (kênh micro/fluency) | `EdgeKind.SEQUENTIAL`, populate G1 | ✅ |
| **Lõi sinh A3 multi** (retrieval → agenda → sinh token-by-token) | `hca/generate_hebbian.py::generate` | ✅ |
| **Train THUẦN** (không teach pack; dữ liệu thật lớp 3–7; `--extra-dir` HF) | `scripts/train_v3.py` → `checkpoints/grade_v3.pkl` | ✅ (legacy nhỏ) |
| **Eval discourse** (prompt corpus-derived, varlm sweep) | `scripts/eval_discourse_prompts.py` | ✅ |
| **Graph UI** — Parent Explorer **A3 multi** | `viz/export_graph_v3.py`, `viz/server/main.py`, `viz/web/` | ✅ |
| **A2 discourse + entity attention** | `hca/discourse.py`, `hca/verify_v3.py`, `generate_hebbian` | 🟡 (0.6.7) |
| **R0 tokenize dấu câu (HF)** | `teaching/align.py::tokenize_punct`, `ingest(split_punct=True)` | ✅ |
| **Trace HCA** (Inference Lab; đường matcher cũ) | `viz/trace.py` | ✅ một phần |

**Mặc định runtime (2026-07-09):** CLI/viz dùng `generate(multi=True)` trên checkpoint **v10**
(`core/paths.py`). Checkpoint `grade_v3*.pkl` giữ cho so sánh lịch sử / guard cũ.

**Lịch sử v3 (tham chiếu):** `generate_hybrid` (verbatim), eval `grade_v3.pkl` đạt QA 100% —
xem 0.5.3; đã **gỡ** khỏi codebase (TK_019 §7).

### 0.5.3. Kết quả đo (A/B cùng thước đo gold-overlap)

| Bộ | baseline cũ (có dạy ngữ pháp) | **v3 nơ-ron cha (KHÔNG dạy)** |
| :--- | :---: | :---: |
| QA val (7) | 86% | **100%** *(grade_v3.pkl, 2026-07)* |
| Paraphrase giữ kín (8) | 62% | **100%** |
| Synonym probe (6) | — | **50%** *(trần E nhỏ)* |

Checkpoint `grade_v3_big.pkl` / `grade_v3_big_a2.pkl` (42k wiki + anaphora): QA **~86%**, Paraphrase **100%**, Synonym **17–33%** — xem 0.5.8–0.5.10, **0.6.7**.

→ **Ngang/vượt baseline ở câu đã thuộc + paraphrase**, không một dòng dạy ngữ pháp. Test: **269 passed**, 1 skipped (2026-07-02).

### 0.5.4. Phân rã BPE/Re-Pair → nơ-ron trung gian (Cập nhật 2026-06-30)

Đã hiện thực `ParentStore.decompose` (`core/parent_store.py`): khai phá **cụm con liền kề lặp**
≥ `min_frequency` cha → **nơ-ron trung gian** (2597 cụm trên `grade_v3`), gán cho từng cha,
và `rank_matches` cộng **bonus cụm phân biệt** (gate theo document-frequency để cụm phổ biến như
`Việt Nam`/`là gì` không nhiễu). Trực quan hóa tầng trung gian trong Graph UI (`/api/v3/complete`).
3 test đơn vị chứng minh cụm chung **phá thế hoà** khi token rời mơ hồ.

- **Đo:** trên eval hiện tại (15 ca, đã chạm trần) decomposition **trung tính** (vẫn 86%/100%,
  không tụt) — giá trị thật ở tổng quát hóa liên-mẫu trên bộ lớn/khó hơn (cần eval synonym riêng).

### 0.5.5. Markov hóa văn phong câu đáp (Cập nhật 2026-06-30)

Đã hiện thực `_markov_answer` (`hca/generate_hebbian.py`) + cờ `hybrid_markov_answer`: sau khi
chọn cha, thay vì dội đáp **verbatim**, sinh đáp bằng **W_seq Markov trên agenda token của cha**
+ temperature → mềm trật tự từ (docx §2.3/§4). **Đo:** accuracy giữ nguyên (QA 86% / Para 100%
ở cả T=0 và T=0.7). Vì với đáp đã trôi chảy sẵn, Markov có thể **xáo trật tự** mà không tăng
điểm, nên **mặc định TẮT (verbatim)**; bật khi cần biến hóa văn phong / kết hợp nhiều nguồn.

### 0.5.6. Giới hạn đã đo & còn lại

**Synonym probe** (`data/grade_03/qa/synonym_probe.jsonl`, từ đồng nghĩa khác mặt chữ): **50%** trên `grade_v3.pkl` (3/6); **17–33%** trên `grade_v3_big` — E ON/OFF như nhau khi `hybrid_fuzzy_topk=0` → embedding suy-từ-đồ-thị **không cứu synonym** ở corpus nhỏ. Là **trần thật** của Luật 0 (không pretrain) — chi tiết [`docs/study/02`](docs/study/02_HYBRID_HEBBIAN_MATRIX.md) §5c. Paraphrase *chia sẻ mặt chữ* vẫn 100%.

| Cơ chế (docx) | Trạng thái |
| :--- | :---: |
| Phân rã BPE/Re-Pair → nơ-ron trung gian (§1, §2.1) | ✅ (trung tính trên eval nhỏ) |
| Markov hóa văn phong câu đáp (§2.3/§4) | ✅ (toggle, accuracy-neutral; mặc định verbatim) |
| Index cứng tường minh (node index) | 🟡 mới là thứ tự list |
| Tổng quát hóa **synonym khác mặt chữ** | ⛔ ~50% curated / 17% big — cần thêm data, không pretrain |
| **Fuzzy E trong matching** | 🟡 có API; **mặc định tắt** (`hybrid_fuzzy_topk=0`) |
| Dual-mode + AST + [SETUP]/[CONFLICT] cho truyện/code (§ bổ sung) | ⏸️ hoãn |

### 0.5.7. Mở rộng dữ liệu thật (HF tiếng Việt) — pipeline sẵn sàng (2026-06-30)

Để vượt trần synonym + phủ kiến thức rộng hơn, đã dựng **pipeline nạp dữ liệu thật**:

| Thành phần | File | Ghi chú |
| :--- | :--- | :--- |
| **E thưa (scale)** | `core/embedding.py` | PPMI thưa + `scipy.svds` — vocab 1998: **22s→0.7s**, scale tới 10k–50k |
| **Cổng chất lượng** | `data/quality.py` | lọc câu fact tiếng Việt (độ dài, tỉ lệ dấu, kết câu, bỏ markup/số/URL) — chặn nhiễu kiểu "Champions/di sản" |
| **Ingest HF** | `scripts/ingest_hf_vi.py` | `--source wiki\|sentences\|viquad` → `data/real_vi/{statements,qa}.jsonl`; **`--anaphora-only`** → `anaphora.jsonl` (lọc `, nó/chúng/họ`) |
| **Nạp vào v3** | `training/ingest.py` + `train_v3 --extra-dir` | mỗi câu/QA → co-occurrence + W_seq + **nơ-ron cha**; HF/extra: **`split_punct=True`** (R0) |

**Đã kiểm thử end-to-end** (mẫu nhỏ tại chỗ): nạp fact mới → trả lời đúng QA mới, và **case con
kiến** (`Cá voi là gì` chưa từng dạy QA → hoàn thành từ statement parent). **218 test pass.**

**Đã chạy thật: ingest 12k câu Wikipedia VI** (qua **datasets-server REST `/rows`** — nhanh,
1.5s/10 bài; tránh streaming builder nặng). Lệnh:
```bash
python scripts/ingest_hf_vi.py --source wiki --max 12000   # → data/real_vi/statements.jsonl
python scripts/ingest_hf_vi.py --source sentences --anaphora-only --max 2000 \
  --out-name anaphora.jsonl   # → data/real_vi/anaphora.jsonl (A2)
python scripts/train_v3.py --extra-dir data/real_vi --out checkpoints/grade_v3_big_a2.pkl
python scripts/eval_suite.py --checkpoint checkpoints/grade_v3.pkl   # eval nhanh vẫn dùng bản nhỏ
```

### 0.5.8. Kết quả scale + bài học IDF (2026-06-30)

Checkpoint `grade_v3_big`: **vocab 18 532, 13 012 nơ-ron cha** (12k Wikipedia + 1k curated),
**1.5M cạnh** — sparse E xử lý gọn (dense SVD bất khả thi ở quy mô này).

| | naive scale | + provenance + curated-IDF | curated nhỏ |
| :--- | :---: | :---: | :---: |
| QA val | 43% | **71%** | 86% |
| Paraphrase | 88% | **100%** | 100% |

**Bài học (quan trọng):** dump data ngoài miền **thô** làm **TỤT** benchmark trong miền (86→43%).
Hai nguyên nhân + cách vá:
1. **Cha ngoài miền cạnh tranh** → thêm `priority` nguồn (curated=1.0, wiki=0.6), nhân vào score.
2. **Pha loãng IDF**: wiki dùng chung từ ("trồng","lúa") → df tăng → token trong miền mất độ phân
   biệt → xếp hạng curated lật. Vá: **IDF chỉ tính trên cha curated** (`priority≥0.9`); wiki vẫn
   truy hồi được nhưng không pha loãng. → phục hồi paraphrase 100%, QA 71%.

→ Scale **có hiệu quả** nhưng cần **routing theo miền**, không dump phẳng.

### 0.5.9. Debug matching + 2 vá quan trọng (2026-06-30)

Rà soát nhiều ca trên `grade_v3_big` phát hiện matching hỏng ở câu **định nghĩa / từ chung**:
`Cao Bằng là gì` → `Huế là cố đô`, `Lao động có ích gì` → `Chim có lông`. Nguyên nhân + vá:

1. **Token frame là "nội dung":** `Hỏi:`/`Đáp:` được đếm khi khớp → mọi câu `X là gì` khớp nhau
   qua `Hỏi:`+`là`+`gì`, nhấn chìm chủ thể phân biệt (`Cao Bằng`). **Vá:** loại `_FRAME_TOKENS`
   khỏi `_content_ids` (`core/parent_store.py`).
2. **Khớp qua từ chung:** cha chỉ phủ `là`/`gì` vẫn thắng. **Vá:** phạt theo **distinct-ratio**
   = (IDF token đặc thù nhất được phủ) / (IDF token đặc thù nhất của câu hỏi); <0.6 → hạ điểm.

**Kết quả sau vá (grade_v3_big, 13k cha gồm 12k wiki):**

| | trước vá | **sau vá** |
| :--- | :---: | :---: |
| QA val | 71% | **86%** |
| Paraphrase | 100% | **100%** |
| Synonym probe | 17% | **33%** |

Debug ca đều đúng: `Cao Bằng là gì` → **"Cao Bằng là một tỉnh thuộc vùng Đông Bắc Bộ, Việt Nam."**
(tri thức Wikipedia), `Lao động có ích gì` → "Lao động là vinh quang".

### 0.5.10. Scale 42k + hai cơ chế ổn định (2026-06-30)

Nâng lên **42k câu Wikipedia** → `grade_v3_big`: **43 012 nơ-ron cha**, **30 000 nơ-ron trung
gian** (phân rã, scale theo số cha), **vocab 42 538**, **3.9M cạnh** (sparse E build gọn).

Data lớn hơn lại **mất ổn định** matching (`Cao Bằng` → `Huế là cố đô`, synonym tụt). Hai vá:

1. **Two-tier retrieval** (`best_match`): thử pool **curated** trước; chỉ rơi xuống pool đầy đủ
   (gồm wiki) khi curated không đạt ngưỡng → câu trong miền **ổn định bất kể wiki lớn cỡ nào**,
   wiki chỉ trả lời câu ngoài miền.
2. **Phạt distinct-ratio bình phương**: cha chỉ phủ token kém đặc thù (`là/gì`, hoặc fuzzy nhiễu)
   bị hạ điểm mạnh (`ratio²`) → cha định nghĩa generic không chặn được câu hỏi chủ thể khác.
   Đúng ý "ưu tiên từ tần suất thấp" = IDF.

**Kết quả 42k (ổn định):** QA **86%**, Paraphrase **100%**, Synonym 17%. `Cao Bằng`/`Nhà Lý` trả
lời đúng từ Wikipedia; curated + paraphrase giữ nguyên. **218 test pass.**

→ **Scale tri thức thật hoạt động ổn định** với two-tier + IDF-distinct. Synonym (17%) là trade-off
của phạt distinct mạnh + trần data (từ đồng nghĩa chưa đồng-hiện đủ). Bản lớn giờ **phủ ~42k chủ
đề Wikipedia** mà không hại benchmark trong miền.

### 0.5.11. Phân rã Re-Pair ĐỆ QUY nhiều tầng (2026-06-30)

Nâng phân rã từ **n-gram phẳng (len 2–4)** lên **Re-Pair đệ quy thật** (docx §2.1): gộp cặp symbol
lặp nhiều nhất → symbol trung gian mới → lặp → **cụm-của-cụm nhiều tầng**.
- `core/parent_store.py::_repair_grammar` — linked-list + heap lazy → **O(số lần xuất hiện)**;
  chạy **7.5s** trên 43k cha (bản recount ~40 phút).
- `decompose_repair` → **21 666 nơ-ron trung gian phân cấp** (23 455 rule). Cụm dài mang nghĩa thật:
  *"Vương quốc Liên hiệp Anh và Bắc Ireland"*, *"thiết lập quan hệ ngoại giao với"*.
- **Đo:** metric **trung tính** (86/100/17 giữ nguyên) — đúng kỳ vọng: giá trị của phân rã là
  **biểu diễn đúng + tổng quát hóa liên-mẫu + mở đường chuỗi dài/composition**, không phải QA hẹp.
  Giữ vì đúng tinh thần v3 + hiệu quả + không hại (Luật 0). **218 test pass.**

### 0.5.12. Matching phân tầng + Embedding Explorer (2026-07-01)

Rà soát Graph UI và eval phát hiện matching **nhảy thẳng token→cha**, bỏ qua cụm trung gian; một số ca sai do filler phrase và thiếu QA val trong nơ-ron cha. Các cải tiến (không hardcode từ vựng):

| Thành phần | File | Nội dung |
| :--- | :--- | :--- |
| **Đơn vị match phân tầng** | `parent_store.query_match_units()` | Cụm Re-Pair (df≤8) greedy longest-first; token thuộc cụm **không** còn bare token |
| **Index phân cấp** | `rebuild_hierarchical_index()` | Token trong cụm cha → chỉ truy hồi qua `cụm→cha`, không lọc df trên flat `_index` |
| **Xếp hạng cha** | `rank_matches()` | Coverage qua cụm OR bare; cha phủ cụm cha trừ IDF token con đã phủ (fix `Thủ đô của Việt Nam là gì`) |
| **QA completion** | `qa_completion_mode` | Sau `Đáp:` — hạ statement parent ×0.01 khi tranh với QA |
| **Train parents** | `train_v3.build_parents()` | `qa_corpus` nạp cả shard **val** (fact QA vào bộ nhớ cha) |
| **Fuzzy embedding** | `hca/config.py` | `hybrid_fuzzy_topk=0` mặc định (ablation: fuzzy làm tụt trên corpus nhỏ) |
| **Graph viz** | `export_graph_v3.completion_graph()` | Cạnh token→cụm→cha; prefix `↳` trên node completion |
| **Embedding UI** | `export_embedding.py`, `EmbeddingExplorer.tsx` | Tab **Embedding E**: meta N×d, tra cosine, hàng xóm, so cặp, tìm vocab |

**Đã gỡ:** `_MATCH_FUNCTION_WORDS` / anchor penalty — mọi disambiguation từ IDF + cấu trúc cụm + priority nguồn (Luật 0).

**Kết quả `grade_v3.pkl`** (rebuild parents + decompose, ~1043 cha, vocab ~2k):

| Bộ | Trước | Sau |
| :--- | :---: | :---: |
| QA val | 86% | **100%** (7/7) |
| Paraphrase | 100% | **100%** (8/8) |
| Synonym probe | 17% | **50%** (3/6) |

Ca debug đã đúng: `Huế ở đâu`, `Thủ đô của Việt Nam là gì`, phân tầng `token→cụm→cha` trên Graph UI.

### 0.5.14. ISG — phân rã tăng dần qua link ngang (TK_013, 2026-07-03)

Thay (hoặc bổ sung) Re-Pair batch bằng **Incremental Segment Graph**: mỗi cha giữ `slot_chain` +
cạnh ngang `(parent_idx, slot_index)`; bigram token-token tích lũy khi ingest → đủ ngưỡng →
`ClusterNeuron` + rewire toàn cục. Index trên **link**, không trên neuron — đúng tinh thần
docx §1 (ý tưởng tác giả).

| Thành phần | File |
| :--- | :--- |
| Lõi ISG | `core/seg_graph.py` |
| Hook ingest + rebuild | `parent_store.add_example`, `rebuild_seg_graph()` |
| Migration | `scripts/rebuild_seg_graph.py` (`--log-file`, `--max-promotions`) |
| Train mặc định ISG | `train_v3.py`, `build_t1_unified.py` (`--legacy-repair` = Re-Pair cũ) |
| Viz | `export_graph_v3` — node `cl:*`, cạnh `seg_h` |
| Log tiến trình | `util/progress.FileProgressLog` |

**Checkpoint ISG:** `checkpoints/grade_v3_big_a2_t3_isg.pkl` (rebuild từ `_t3.pkl`, ~3 phút).
Cluster `ở→đâu` **power=19** (Re-Pair không từng tạo bigram bare này). Eval fast: QA **100/100**,
Para 88%, Syn 67%. **296 test pass.** Chi tiết: `TK_013_INCREMENTAL_SEG_GRAPH.md`.

```bash
# Rebuild ISG trên checkpoint có sẵn (không cần full train G1–G7)
python scripts/rebuild_seg_graph.py \
  --checkpoint checkpoints/grade_v3_big_a2_t3.pkl \
  --out checkpoints/grade_v3_big_a2_t3_isg.pkl \
  --log-file out/rebuild_isg_t3.log

# Graph UI
AIVIE_CHECKPOINT=checkpoints/grade_v3_big_a2_t3_isg.pkl python viz/server/main.py
```

**API Embedding E:** `GET /api/embedding/info`, `/lookup`, `/compare`, `/search` — xem **B.6**.

### 0.5.13. Fix regression case con kiến — `qa_completion_mode` dìm nhầm statement (2026-07-01)

Vá `qa_completion_mode` (0.5.12) phát hiện qua kiểm chứng trực tiếp (không chỉ tin test cũ):
`×0.01` áp **vô điều kiện** lên mọi cha statement bất cứ khi nào `qa_completion_mode=True` —
mà `generate_hebbian.generate_hybrid` luôn truyền `True` ngay từ bước chọn cha đầu tiên. Hệ quả:
một fact **chỉ tồn tại dưới dạng câu phát biểu** (chưa từng dạy QA — đúng case "con kiến",
README 0.5.7) bị dìm điểm xuống dưới `min_score` → `best_match` trả `None` → rơi vào fallback
W_seq (sinh lặp/vô nghĩa). Test `test_ingest_statement_enables_pattern_completion` vẫn pass
chỉ vì đồ thị đồ chơi 1 câu khiến fallback **tình cờ** nhại đúng từ — che mất việc cơ chế thật
đã chết (đúng kiểu "ảo tưởng sức mạnh" — Luật 0).

**Vá:** chỉ hạ statement khi có cha QA **thật sự cạnh tranh** — QA tốt nhất đạt ≥85% điểm của
ứng viên tốt nhất trong toàn bộ candidate (near-tie theo raw score), không phải "có tồn tại
bất kỳ cha QA nào, dù khớp mờ nhạt, trong top_n" (tiêu chí cũ gần như luôn đúng trên đồ thị
lớn → case con kiến gần như luôn thua). `core/parent_store.py::rank_matches`.

**Đo trên `grade_v3_big.pkl`** (43k cha, nơi bug lộ rõ vì luôn có ≥1 QA khớp lỏng lẻo):

| Bộ | Trước vá (bug) | **Sau vá** |
| :--- | :---: | :---: |
| QA val | 0% (0/7) | **71%** (5/7) |
| Paraphrase | 50% (4/8) | **88%** (7/8) |

`grade_v3.pkl` giữ nguyên **100%/100%/50%** (0.5.12) — vá không đụng case QA cạnh tranh thật.
Case "Đèo Hải Vân là gì" (statement, có 1 cha QA khớp mờ qua "Đà Nẵng") nay dội đúng
`"Đèo Hải Vân ngăn Huế và Đà Nẵng"` thay vì rơi fallback. Test mới:
`test_qa_completion_mode_does_not_crush_clearly_better_statement`. **219 test pass.**

### 0.5.14. Nơ-ron mẫu chung (slot hub) + nối filler LÚC TRAIN (2026-07-01)

> Trả lời trực tiếp đề bài: *"Thủ đô của Mỹ/Pháp/Anh… → một nơ-ron chung `n_quốc_gia_thủ_đô`;
> gặp `Thủ đô của Ý` (Ý không trong nơ-ron đó) mà có context khác nói về Thủ đô của Ý → mạng
> nối Ý vào hub để nhớ, và biết chú ý vào Ý khi trả lời."* Cơ chế theo docx §"Cross-sample
> Phase" / `W_struct` (README v3 phần bổ sung) — **không hardcode danh sách quốc gia**.

**Vì sao cần:** `rank_matches` chấm theo coverage token nội dung trùng khớp → `Thủ đô của Ý`
với cha `Thủ đô của Mỹ…` chia sẻ 0 token đặc thù (Ý≠Mỹ) → cha bị loại. Parent store cũ coi mỗi
câu độc lập tuyệt đối, **không có khái niệm "họ mẫu"**.

**Cơ chế (`core/parent_store.py`):**

| Thành phần | Method | Nội dung |
| :--- | :--- | :--- |
| **Induce hub** | `induce_slot_hubs()` | ≥2 cha cùng skeleton, lệch **đúng 1 vị trí** → hub `(left, right, fillers)`. Gate Luật 0: filler phải **phân biệt hơn khung** (min IDF filler ≥ IDF trung bình token khung) → slot là thực thể, khung là phần chung |
| **Quét chéo bind** | `bind_slot_fillers()` | Quét mọi câu train: gặp `left X …right…` với X phân biệt → thêm cạnh `hub→X`. Câu phát biểu `"Thủ đô của Ý là Rome"` đủ để **Ý gia nhập hub ngay lúc train** (không cần QA về Ý) |
| **Khớp lúc sinh** | `hub_for_query()` | Câu hỏi → (hub, filler X). X đã nối → *known*; X lạ (Đức) → vẫn nhận slot để chú ý + route case con kiến |

**Nối train-time vs inference-time (EMERGENT):** hub-bind chạy **offline, tất định** trong
train (quét trước toàn corpus) — khác `EMERGENT` củng cố *sau* khi đã trả lời. Hai cái bổ sung:
train-bind lo filler có context sẵn (Ý), EMERGENT lo filler chỉ gặp lúc chạy.

**Kết quả `grade_v3.pkl`** (dữ liệu thật lớp 3–7, không synthetic): **22 hub tự nổi**, 6 filler
nối lúc train. Các họ mẫu thật nổi lên đúng:

```
Nhà ___ thành lập năm nào         → {Lý, Đinh}
Việt Nam gia nhập ___ năm nào     → {WTO, ASEAN, Liên hợp quốc}
___ giúp ta làm gì                → {Phổi, Mắt, Tai}
2 cộng 3 bằng ___                 → {5, mấy}
```

Eval **giữ nguyên 100%/100%/50%** (hub additive, chưa đọc vào readout mặc định). Test:
`test_slot_hub_*`, `test_cross_sample_binds_new_filler_during_train`. **224 test pass.**

### 0.5.15. Hub guard — chống hallucination chéo filler + route novel filler (2026-07-01)

> Tận dụng nơ-ron hub lúc **sinh** (không chỉ quan sát). Phát hiện qua đo trực tiếp:
> `Nhà Trần thành lập năm nào` (Trần không có QA) → mạng **bịa** `Nhà Lý thành lập năm 1009`
> (fact của filler khác cùng họ). Đây là hallucination chéo filler — hub biết slot="Trần"
> nhưng đáp nói về "Lý".

**Cơ chế (`hca/generate_hebbian.py::_hub_guarded_completion`, cờ `hybrid_hub_guard`):** khi câu
hỏi khớp một hub với **filler LẠ** (chưa nối lúc train), đáp hợp lệ **bắt buộc phải chứa chính
slot_x**. Quét top-N ứng viên:
- có statement/QA **nhắc đúng thực thể** (vd "Thủ đô của Ý là Rome") → route tới nó (case con
  kiến qua hub, đúng thực thể);
- chỉ có fact filler khác ("Nhà Lý… 1009") / câu khác khung ("Đảng… 1930") → **loại hết →
  KHÔNG bịa** (đáp rỗng). Đúng G4 Verify: "token không có đường dẫn học được → ức chế".

Filler **đã biết** (có QA) → guard không đụng; luồng thường xử lý.

**Đo trên `grade_v3.pkl`:**

| Câu hỏi (novel filler) | Trước (bịa) | **Sau guard** |
| :--- | :--- | :--- |
| Nhà Trần thành lập năm nào | Nhà **Lý** thành lập năm 1009 ✗ | Nhà Trần đánh thắng quân Nguyên ba lần ✓ (fact đúng Trần) |
| Nhà Nguyễn thành lập năm nào | Nhà **Lý** thành lập năm 1009 ✗ | Nhà Nguyễn thống nhất đất nước năm 1802 ✓ |
| Thủ đô của Ý là gì (+ statement Ý) | — | Thủ đô của Ý là Rome ✓ (route case con kiến) |

Benchmark **giữ nguyên 100%/100%/50%** (guard chỉ tác động filler lạ, không đụng QA đã thuộc).
Graph UI: filler lạ hiện badge **route → đúng thực thể** / **chặn bịa**, node winner = None (hub
can thiệp), ↳ completion = đáp thật. Test: `test_hub_routes_novel_filler_to_entity_statement`,
`test_hub_suppresses_cross_filler_hallucination`, ablation `hybrid_hub_guard=False`. **229 test pass.**

**Về "nhớ" filler mới (hub→X):** việc nối filler là **train-time**, do `bind_slot_fillers()` lo
— hễ có statement mới về thực thể trong dữ liệu, chạy lại `induce_slot_hubs`+`bind_slot_fillers`
là filler nối **vĩnh viễn** vào hub. **KHÔNG** củng cố lúc inference: `generate_hybrid` thuần
read-only theo thiết kế (train đổi `A`/`hub.fillers`, sinh chỉ đọc) — đổi trọng số lúc sinh vừa
vi phạm ranh giới train/inference vừa không persist. Do đó bỏ hướng "EMERGENT lúc inference".

### 0.5.17. Kích thích cụm WH/frame + anchor gom (TK_014, 2026-07-04)

> Debug `Hỏi: Thủ đô Hà Nội ở đâu ?`: nơ-ron cụm `ở đâu` (`cl:5780`, power=19) **không kích
> hoạt**. Chỉ đạo tác giả: cụm WH KHÔNG được loại bỏ, phải được kích thích + sau này **gom** mọi
> query "nơi chốn" (`X→"ở đâu"`, `Y→"ở đâu"` ⇒ gom X,Y). Nối doc 03 (attention Q/K) + doc 05.

**Gốc (đo):** [`query_match_units`](src/nn_aivie/core/parent_store.py) chỉ nạp cụm làm đơn vị
match nếu `phrase_df ≤ phrase_max_df` (8). Cụm phổ biến (`ở đâu` df19, `là gì` 35, `Hà Nội` 286)
bị loại → `phrase_units=[]`, matching thoái hoá về token rời; UI lại chỉ vẽ cụm của cha THẮNG.

**Hướng 1 — kích thích:** cờ `HCAConfig.match_common_clusters` (mặc định BẬT) → bỏ hard-gate df,
nạp CẢ cụm phổ biến làm unit; cân bằng bằng IDF thấp + phạt distinct-ratio² (0.5.10) nên **không
nhấn chìm thực thể**. **Hướng 2 — anchor gom:** dùng `seg._cluster_index` (đã có), thêm
`anchor_members` / `anchor_left_fillers` (thực thể X trước anchor) / `query_anchor_clusters`;
viz hiện node cụm query `activated` + `anchor_groups`.

**Đo (2026-07-04):** A/B top-1 df=8↔∞ **0 flip** (grade_v3 21 ca + ISG QA 7). Full `--fast`:
grade_v3 QA **100/100**/Para 62/Syn 50/Comp 80 — y baseline; ISG QA **100/100**/Para 88/Syn 67 —
= baseline TK_013. `anchor_left_fillers("ở đâu")` = {Việt Nam, Cà Mau, Amazon, Hồ Tây, nhà
Nguyễn,…} — thực thể nơi chốn nổi lên đúng. **298 test pass.** Sổ nợ: overlap `xảy ra ở đâu`
(filler "xảy ra"); answer-type biasing (doc 03 Wq/Wk) = phase sau. Chi tiết `TK_014_WH_ANCHOR_ACTIVATION.md`.

---

### 0.5.18. BẢN ĐỒ MA TRẬN — TOÀN BỘ MA TRẬN, TẦNG LỚP, VAI TRÒ (TK_045 §12.75, 2026-07-29)

> Rà toàn bộ mã lúc trả lời câu hỏi của tác giả "mạng có bao nhiêu ma trận". Mọi con số ablation
> dưới đây đo trên **v12** (430.447 cha), không phải ước lượng.

**Nguyên tắc bao trùm:** không ma trận nào được huấn luyện bằng backprop. Tất cả là **view chiết
ra từ đồ thị Hebbian**, dựng lười lúc chạy rồi cache (Luật 0).

#### Tầng 0 — bộ nhớ thật (thứ duy nhất thực sự "học")

| | là gì | vai trò |
|---|---|---|
| `adjacency` (TypedAdjacency) | cạnh Hebbian có kiểu giữa token, gồm **`W_seq`** = bigram có hướng a→b | "chữ nào đi sau chữ nào" |
| `ParentStore` | **nơ-ron cha** — mỗi câu đã học là một nơ-ron: `emit_ids`, `match_ids`, `power`, `episodes`, theme neuron | "mạng biết những câu gì" |

Mọi ma trận bên dưới chỉ là cách **đọc lại** hai thứ này.

#### Tầng 1 — view của `W_seq` (lo *CÁCH nói*)

| ma trận | chiết bằng | vai trò |
|---|---|---|
| `embedding` E (N×96) | PPMI đối xứng → SVD | "hai chữ giống nhau" |
| `DirectedQK` (Wq, Wk) — `attention_qk.py` | PPMI **có hướng** → SVD; `Wq=U√Σ`, `Wk=V√Σ` | lấp cặp chuyển tiếp **CHƯA từng gặp**: `q·k ≈ PPMI(i→j)` kể cả bigram = 0 |
| Spectral stack (Wa, Wffn ×4 tầng) — `spectral_stack.py` | residual SVD deflation của PPMI có hướng | Attn → Amp → top-n, đồng cấu; không gán trước "tầng nào lo gì" |
| `varlm` — `context_positional.py` | LM bậc thay đổi (sidecar) | vá điểm yếu order-1 của `W_seq` |

#### Tầng 2 — view của `ParentStore` (lo *NÓI GÌ*)

| ma trận | hình dạng | vai trò |
|---|---|---|
| **`ParentBipartite.A`** | (N token, P cha) | token → cha: `match × √power × gidf`. **TRUY HỒI** |
| **`ParentBipartite.E`** | (P cha, N token) | cha → token: khối lượng emit. **PHÁT LẠI** |
| `succ`, `emits` | dict / list | successor trong cha; khớp hậu tố toàn lịch sử (chống trôi) |
| `end_after`, `cont_after` | (N,) | token thường đứng trước biên câu / trước chữ tiếp |
| `W_know` — `knowledge_seq.py` | (N,N) | chuyển tiếp **chỉ trong chuỗi cha** + kìm hub kiểu Oja — ít hub `để/là` hơn `W_seq` |
| `KnowFactors` (know_qk) | dim 64 | bản phân tích thừa số của `W_know` |
| theme neuron | entity → [(cha, w)] | `w = 1/(1+vị_trí)` — thực thể đứng đầu câu |

Cặp **A/E** là lõi kiến trúc: `A` hỏi "câu hỏi này đánh thức cha nào", `E` trả lời "cha ấy phát ra
chữ gì". Hành vi `copy ≈ 1.0` (TK_045 §12.72) đến thẳng từ đây.

#### Tầng 3 — liên câu / đoạn

`sent_intra.py`: **L2R** (trái→phải trong câu) · **R2L** (dùng để *neo và xếp hạng cha*, KHÔNG
phải phân phối next) · **SENT_SUCC → `W_agenda`** (câu nào theo câu nào) · `sent_qk`.
Cộng `episodes` trong ParentStore = thứ tự câu trong tài liệu gốc.

#### Chỗ trộn — `matrix_core_logits`: 11 kênh, nhưng chỉ ~6 kênh THẬT SỰ chạy

| kênh | β mặc định | ablation v12 (TK_045 §12.39) |
|---|---|---|
| `succ` + `path` | **5.0** + **4.0** | **gánh chính** — tắt thì `digit⎵` +0.09, `loop4` +0.09, `n_facts` −0.13 |
| `l2r` | 2.0 | nhỏ |
| `know` / `know_lr` | 1.2 / 0.5 | nhỏ (`n_tok` −2.0 / +1.3) |
| `eos` | 1.2 | nhỏ (`loop4` +0.04) |
| `bag(parent)` | 0.9 | nhỏ |
| `h` (ngữ cảnh dài) | 0.45 | rất nhỏ |
| `stack` | 0.35 | **bộ đo đang set = 0** ⇒ CHƯA TỪNG ĐO |
| `r2l_ffn` | 0.35 | **mọi Δ = 0 — kênh chết thật** |
| `qk` · `readout` · `agenda` | **0.0** | Δ = 0 vì **vốn đã tắt sẵn**, KHÔNG phải vì vô dụng |

**Ba điều phải nhớ khi đọc bảng này:**

1. **Kiến trúc mỏng hơn con số 11.** Ba kênh β=0 từ đầu, một kênh bị bộ đo tắt, một kênh đo được
   là chết ⇒ còn ~6 kênh chạy, và sức nặng dồn gần hết vào `succ+path` — kênh **khớp hậu tố emit
   của cha**. Đó là lời giải cơ học cho `copy ≈ 1.0`: đường sinh đang bị chi phối bởi kênh CHÉP.
2. **`qk` và spectral stack — hai thứ tinh vi nhất — chưa từng được bật để đo.** Chúng sinh ra để
   lấp chuyển tiếp chưa gặp, đúng thứ cần cho việc *soạn* thay vì *chép*.
3. **Không có ma trận nào cho tầng ĐOẠN VĂN.** `W_agenda` có nhưng β=0; mạch đoạn hiện chạy bằng
   `episodes` (thứ tự tài liệu gốc) qua handoff, không qua ma trận học được. Đó chính là chỗ
   **trôi đề** xảy ra (TK_045 §12.73: `lạc` v12 T1 0,12 ở cap 1 → 0,27 ở cap 3).

---

## 0.6. LỘ TRÌNH v4 — TIẾN TỚI LLM: COMPOSITION (CÂU GHÉP NHIỀU Ý) + CODE-GEN (AST)

> Mục tiêu: từ **máy truy hồi + điền mẫu** (v3 hiện tại: chọn 1 cha, dội đáp verbatim) tiến tới
> **template-aware compositional generator** — viết đoạn dài, ghép nhiều ý/câu, sinh code đúng
> văn phạm. **Trần thực tế (docx thừa nhận + Luật 0):** tái tổ hợp khuôn mẫu đã học, KHÔNG phát
> minh thuật toán/cốt truyện mới. Mọi năng lực từ **kiến trúc + train**, không hardcode (Mục 0).

### 0.6.1. Chẩn đoán (đã soi code 2026-07-01)

| Hiện trạng | Chi tiết |
| :--- | :--- |
| **Sinh = một-lần** | `completion()` dội **cả đáp verbatim**; không ghép nhiều mảnh → không viết đoạn dài |
| **Fluency yếu** | Fallback chỉ **W_seq bigram** (Markov bậc 1); không mạch lạc qua đoạn |
| **Readout giàu đang NGỦ** | `hca/forward.py` (`dual_context_read`, `spreading_activation_read`) — cơ chế "attention không softmax" — **không được gọi** trong `generate_hybrid`; kẹt ở đường v2 (`hca/generate.py`) kèm scaffolding fingerprint/Rule/QA-session (v2, bỏ) |
| **Trừu tượng cấu trúc** | Re-Pair chỉ bắt **lặp bề mặt**; mù với **vai trò chức năng** (SETUP/CONFLICT, `FunctionDef`) — liên-mẫu, cần W_struct/AST |
| **Slot hub = primitive** | 0.5.14–15 đã có "khung + filler" — chính là mầm của W_struct (khung code `def ___`) |

**Đánh giá dual-context/spreading:** *toán readout không lỗi thời* (là động cơ sinh token-level cho
composition); chỉ **vỏ v2** (fingerprint/Rule/session) là bỏ. Sẽ **tái sinh lõi**, seed lại từ
khái niệm v3 (parent/hub/entity-focus), không QFocus/Rule.

### 0.6.2. Track A — COMPOSITION (câu ghép nhiều ý)

| Phase | Nội dung | Trạng thái |
| :--- | :--- | :---: |
| **A1** | **Đa-fact composition:** entity-focus (IDF) → gom nhiều statement-parent về thực thể → dedup → ghép nhiều câu. Cho ra "nhiều ý" đầu tiên. | ✅ (0.6.5) |
| **A2** | **Connective + anaphora (v3 Hebbian):** discourse read + entity attention + G4 verify; ghép vế bằng `,` + đại từ học từ corpus (không hardcode) | 🟡 phase-1 (0.6.7) |
| **A3** | **Readout token-level v3:** sinh câu từng token bám agenda (4 kênh Hadamard + index backbone). Không verbatim | 🟡 phase-1+2+A2 (0.6.6–0.6.7) |
| **A4** | **Agenda/closure:** danh sách ý cần phủ + dừng khi phủ đủ (tái đúc từ 0.2 trên nền v3) → essay nhiều đoạn | ⬜ |

### 0.6.5. A1 — Đa-fact composition (2026-07-01)

Hiện thực "trả lời nhiều ý, nhiều câu ghép" bước đầu (KHÔNG hardcode từ):

| Thành phần | File | Nội dung |
| :--- | :--- | :--- |
| **Entity focus** | `parent_store.top_focus_tokens()` | Token thực thể phân biệt nhất (IDF cao) của câu hỏi; **loại token df=0** (từ chỉ có trong câu hỏi như "thế/nào" có IDF giả tạo, nhấn chìm thực thể thật) |
| **Gom fact** | `parent_store.facts_about()` | Statement-parent chứa thực thể trọng tâm, xếp theo IDF-coverage + power, **khử trùng** (Jaccard nội dung) |
| **Ghép** | `generate_hebbian.generate_composed()` | ≥2 fact → ghép nhiều câu (chèn ranh giới câu nếu vocab có dấu); <2 → nhường `generate_hybrid` (câu QA cần đáp gọn) |

**Kết quả (`grade_v3.pkl`, opt-in — KHÔNG đụng `generate_hybrid` mặc định, eval giữ 100/100/50):**

```
Nhà Trần thế nào → Nhà Trần đánh thắng quân Nguyên ba lần · Nhà Trần thay nhà Lý năm 1226
                   · Trần Hưng Đạo là Hưng Đạo Vương
Nhà Lý thế nào   → Nhà Lý thành lập năm 1009 · Nhà Trần thay nhà Lý năm 1226
                   · Nhà Lý mở nền văn hóa Thăng Long
```

**Giới hạn đã đo (→ động lực A2):** entity-focus một token bị **trôi khi đa nghĩa/generic**
(`Hồ Tây` → dính "Đông **Tây**"; `Sông` Hồng → dính "Sông Cửu Long"). Cần focus **cụm đa token**
(Re-Pair phrase) + liên từ/đại từ (A2). Test: `test_compose_chains_multiple_facts_about_entity`,
`test_facts_about_dedups_and_ranks_by_entity`. **232 test pass.**

**An toàn:** `generate_composed` KHÔNG đụng `generate_hybrid` mặc định (eval giữ 100/100/50);
composition là **luồng phụ song song**, không thay đáp chính.

**Graph UI (2026-07-09):** `completion_graph()` luôn `generate(..., multi=True)`; panel agenda +
trace token A3; không còn mode Hybrid/Compose/Essay. Mặc định `network_v10_histwiki_paragraph.pkl`.
Files: `viz/export_graph_v3.py`, `viz/web/src/{api.ts,GraphExplorer.tsx}`, `viz/server/main.py`.

### 0.6.6. A3 — Sinh AUTOREGRESSIVE token-level (2026-07-01)

> **Đề bài (bạn nêu):** A1 chỉ *truy hồi khối + ghép verbatim* — **không giống LLM**. LLM sinh
> **từng token**, điều kiện theo mọi token trước, có thể tạo câu **chưa từng lưu**. A3 dựng đúng
> cơ chế đó — chính là **vòng inference docx §4** (`a_final = a_macro ⊙ a_micro`) chưa từng code.

**Giải pháp (`generate_hebbian.generate_autoregressive`, `_build_agenda`):** vòng lặp mỗi bước:

```
w0 = token vừa sinh
a_micro = W_seq(w0→·)                    # token thường theo sau w0 (mượt, Markov)
a_macro = spreading_activation(seed = agenda CÒN THIẾU)   # kéo về ý chưa diễn đạt
a_final = (sàn + a_micro) ⊙ (sàn + a_macro)  + prior_thứ_tự(agenda)  # docx §4 + §2.3
chọn 1 token (argmax/nhiệt độ) → nối → lặp; EOS khi agenda phủ đủ
```

- **Agenda = "ý cần phủ"** (do A1 lập kế hoạch: đáp cha khớp gọn, hoặc đa-fact khi câu mở) —
  KHÔNG phải chuỗi verbatim. Mạng **tự dựng token** bám agenda.
- **Prior thứ tự mềm** (docx §2.3): boost token agenda kế tiếp theo trật tự nguồn, giảm dần —
  "index" mềm, W_seq vẫn quyết fluency. Xử lý token lặp ("Công **an** giữ **an** ninh") bằng
  đánh dấu theo **vị trí** (không set).
- Bỏ vỏ v2 (fingerprint/Rule/QA-session); chỉ dùng W_seq + spreading trên đồ thị v3.

**Kết quả (`grade_v3.pkl`, đo phủ agenda):** vòng autoregressive **chạy thật, từng token**. Tái
dựng **đúng verbatim** trên đường W_seq rõ — chứng minh engine hoạt động:

```
Ai chữa bệnh              → Bác sĩ chữa bệnh.          (100%, token-by-token)
Nhà Đinh thành lập năm nào → Nhà Đinh thành lập năm 968 (100%)
Một tuần có mấy ngày      → Một tuần có bảy ngày.       (100%)
Thủ đô Việt Nam là gì     → Việt Nam là Thủ đô Hà Nội  (100% phủ, ĐẢO NGỮ — câu chưa lưu!)
```

Phủ đủ agenda **4/6** ca thử; ca đảo ngữ ("Việt Nam là Thủ đô Hà Nội" thay vì "Thủ đô Việt Nam
là Hà Nội") **chính là bằng chứng sinh ≠ copy** (LLM-like).

**Phase-1 giới hạn:** W_seq bigram + prior thứ tự cộng quá yếu → trật tự trôi thành word-salad
(`Ai giữ an ninh` 25%), đặc biệt khi bigram phổ biến áp đảo ("Đáp:→Việt").

#### Phase-2 — Index backbone (docx §1 "index cứng") + no-progress guard

**Kiểm chứng quan trọng (đo trên `grade_v3`):** v3 có **0 cạnh position/R-slot** → hướng ban
đầu (hồi sinh `dual_context_read` kênh C_frame) **BẤT KHẢ** (C_frame cần `A[R_-k, w0]` rỗng).
Tín hiệu trật tự duy nhất v3 có là **W_seq bigram (4.2k)**. Nên phase-2 quay lại **"index cứng"
docx §1** — thứ tự con trong nơ-ron cha — làm **xương sống**:

- Token agenda kế tiếp theo trật tự nguồn, nếu **ĐẠT ĐƯỢC** từ w0 (W_seq/co-act) → **cho vượt
  trội** (`ar_index_margin`); không đạt → W_seq bắc cầu (mềm — docx §2.3).
- **No-progress guard** (`ar_patience`): nhiều bước không phủ thêm agenda → dừng, tránh trôi dài.

**Kết quả (`grade_v3`, phủ agenda **7/9**), trật tự ĐÚNG (word-salad phase-1 đã sửa):**

```
Thủ đô Việt Nam là gì      → Thủ đô Việt Nam là Hà Nội.   (ĐÚNG trật tự — phase-1 đảo "Việt Nam là Thủ đô")
Tam giác có bao nhiêu cạnh → Tam giác có tổng ba góc bằng 180 độ.
Nhà Lý thành lập năm nào   → Nhà Lý thành lập năm 1009
Ai dạy học                → Thầy cô dạy em học.            (đáp đồng nghĩa, không lặp từ hỏi)
```

**Giới hạn còn lại (data-level):** (1) **R0** (`tokenize_punct`) đã vá cho ingest HF — lesson alignment vẫn dùng `tokenize()` thường; (2) đồ thị thưa trên `grade_v3.pkl` nhỏ. Tiếp: corpus `, nó` gắn thực thể + **W_seq bậc cao**. **Opt-in A2/A3, KHÔNG đụng `generate_hybrid` → eval giữ 100/100/50.** Test:
`test_autoregressive_*` (gồm `_index_backbone_orders_words`), `test_discourse.py`. **246 test pass** (2026-07-01).

**Cập nhật A3 + A2 (2026-07-01):** readout mỗi bước mở rộng thành **4 kênh Hadamard**:

```
a_final = (sàn + a_micro) ⊙ (sàn + a_macro) ⊙ (sàn + a_entity) ⊙ (sàn + a_discourse)
```

- `a_micro` = W_seq; `a_macro` = spreading agenda; `a_entity` = `entity_attention_read`; `a_discourse` = W_seq sau ranh giới `,`/`.`
- G4 verify v3 (`hca/verify_v3.py`); repetition penalty sau dấu phẩy (`hca/discourse.py`).
- Bật/tắt: `HCAConfig.ar_anaphor_enabled` (mặc định `True`; chỉ ảnh hưởng `generate_composed` / `generate_autoregressive`, **không** đụng `generate_hybrid`).

### 0.6.7. A2 — Discourse + entity attention (2026-07-01)

> **Đề bài:** Nối nhiều vế bằng liên kết học từ corpus (`và`, `,`) và thay chủ ngữ lặp bằng đại từ
> (`nó`/`họ`/`chúng`) — **không** hardcode danh sách từ, không `Var:P` teach v2. Ánh xạ từ
> **attention LLM**: query = vị trí sinh (`w0`), keys = thực thể đăng ký (entity register), value =
> token ứng viên qua readout đồ thị.

| Thành phần | File | Nội dung |
| :--- | :--- | :--- |
| **Entity register** | `parent_store.entity_register_ids()` | "Keys" attention: cụm Re-Pair + token IDF cao của câu hỏi |
| **Entity attention** | `hca/discourse.py::entity_attention_read` | `readout(e→v)·readout(w0→v)` + retroactive-lite từ entity trong history |
| **Discourse read** | `hca/discourse.py::discourse_read` | W_seq sau `,`/`.` khi còn segment agenda |
| **Chọn đại từ** | `hca/discourse.py::best_anaphor_for_entity` | Sau `,`: ứng viên có **chữ ký cú pháp giống chủ ngữ** (cosine successor profile) — phân biệt đại từ vs liên từ `và` |
| **Ghép A1+A2** | `generate_hebbian.generate_composed` | ≥2 fact → `fact1 , <đại từ> fact2…`; lọc fact ghép dài trùng fact ngắn |
| **A3 + A2** | `generate_hebbian.generate_autoregressive` | Segment plan + bridge `,` giữa vế; 4 kênh readout |
| **G4 v3** | `hca/verify_v3.py` | Anchor = entity + agenda + w0 (không cần QASession v2) |
| **R0 tokenize** | `teaching/align.py::tokenize_punct` | Tách `,`/`.` khỏi từ (`ra,`→`ra`+`,`); opt-in `ingest(split_punct=True)` cho HF |
| **HF anaphora** | `ingest_hf_vi.py --anaphora-only` | Stream `thanhdath/vietnamese-sentences`, lọc `, nó/chúng/họ` → `anaphora.jsonl` |
| **Checkpoint** | `grade_v3_big_a2.pkl` | 44k cha (42k wiki + 1092 anaphora + curated); vocab ~31k; token `,`/`nó`/`họ`/`chúng` |

**Kết quả (`grade_v3_big_a2.pkl`, mode Compose):**

```
Hỏi: Huế là gì ? Đáp:  →  Huế là cố đô , họ là kinh đô nhà Nguyễn , Kinh thành Huế là di sản văn hóa .
```

Hybrid vẫn một câu: `Huế là cố đô.` — eval `grade_v3.pkl` giữ **100/100/50**.

**Giới hạn đã đo (data-level):**

1. **Corpus compound trong `data/grade_XX/compound/`** dùng `và`, không có `, nó` — A2 cần HF anaphora hoặc generator `build_subject_knowledge._compound_lesson_examples`.
2. **HF tin tức** (`sentences`): đại từ hay là `họ`/`chúng tôi` (người), ít `nó` gắn thực thể vô tri (`Huế`→`nó`).
3. **`grade_v3.pkl` nhỏ** không có token `,`/`nó` tách rời → ghép A1 dính chữ; cần `grade_v3_big_a2` + R0.

**Graph UI:** dropdown **Hybrid / Compose / A3**; panel composition + hint khi Hybrid; mặc định `viz_default_checkpoint()` → `grade_v3_big_a2.pkl`. Test: `tests/test_discourse.py`, `tests/test_export_graph_v3.py::test_completion_graph_autoregressive_mode`.

### 0.6.3. Track B — CODE-GEN (AST + dual-mode + self-verify)

| Phase | Nội dung | Trạng thái |
| :--- | :--- | :---: |
| **B1** | **Parent-từ-AST:** parse Python `ast` → subtree = parent; node cấu trúc (`FunctionDef`,`For`) = skeleton, identifier/literal = **filler slot** (tái dùng slot hub) | ⬜ |
| **B2** | **Symbol table cứng:** tên biến/hàm chính xác tuyệt đối; ràng buộc sinh chỉ dùng tên hợp lệ (dual-mode: soft cấu trúc + hard tên) | ⬜ |
| **B3** | **Vòng self-verify:** compiler/linter/unit-test làm reward khách quan (mở rộng G4 verify / hub guard) → generate-and-verify | ⬜ |

→ **Track B là "sân thử lý tưởng"** (docx): cấu trúc AST cho sẵn (hữu hạn, 0 nhiễu) + thước đo
đúng/sai miễn phí. Kế thừa thẳng slot hub + hub guard.

### 0.6.4. Thứ tự khởi động

**A1 trước** (đòn bẩy cao, tái dùng parent/hub, trả lời trực tiếp "nhiều ý nhiều câu"), rồi A2.
A3 (readout) là nền chung cho cả sinh câu mới lẫn code. Track B chạy song song khi A1–A2 ổn.
Mỗi phase: **implement → đo (giữ nếu cải thiện thật) → không hardcode**.

### 0.6.8. v4 — thước đo trung thực + focus/anaphora, và QUYẾT ĐỊNH TREO: TRẦN SYNONYM (2026-07-01)

> Chi tiết: [`docs/NN-AIVIE v4.md`](docs/NN-AIVIE%20v4.md). Tóm tắt để README tự đủ.

**Đã làm (đo được; cập nhật 2026-07-09):**
- **Thước đo v4** (`eval/v4.py` module — **CLI `scripts/eval_v4.py` đã gỡ**): tách retrieval vs
  generation + metric IDF + verbatim_rate. Thay runtime eval: **`scripts/eval_discourse_prompts.py`**
  (prompt corpus-derived, varlm sweep, `generate(multi=True)` only).
- **Hợp nhất sinh (TK_019 §7):** xóa `generate_hybrid`/`generate_composed`/`generate_essay`; một
  đường `generate(multi=True)` → A3. **131 test pass.**
- **Focus W3** (`parent_store`): `_gdf` (df toàn cục) + xếp hạng đơn-vị theo `IDF÷(1+vị_trí)` +
  `primary` = token phân biệt nhất → sửa drift (Cao Bằng, Nhà Trần "thế nào").
- **Anaphora A2** (`discourse`): neo đại từ vào token phân biệt + gate định-ngữ-vs-đại-từ (chính
  tả) + giữ chủ ngữ khi không chắc → hết "nhà mở nền", ra "họ mở nền".
- **Composition vào eval** (`data/grade_03/qa/composition.jsonl`): coverage 100% / redundancy 0%.
- Trụ 2 (successor bậc-2 từ chuỗi cha): đo **trung tính** → default TẮT (giữ ablation).

**QUYẾT ĐỊNH TREO — Trần synonym (chờ tác giả chốt).** Chẩn đoán (bằng chứng đồ thị): synonym
phân bố (chữa≈trị) trên corpus curated là **trần DỮ LIỆU/PRETRAIN, KHÔNG phải gap kiến trúc**:
`cấy`/`khâu` **OOV**; `trị` **sai nghĩa** (chỉ có "chính trị"); E bậc-1 **và** bậc-2 đều ra
**collocate** (chữa→bệnh), không tách được paradigmatic (thay-thế-được) vì mỗi từ chỉ 1–2 ngữ cảnh.
Có **2 loại "giống nhau"**: *syntagmatic* (đồng hiện, E làm được) vs *paradigmatic* (thay thế = synonym,
cần nhiều ngữ cảnh HOẶC pretrain). Ba đường ra — **quyết định sau cải tiến này**:

| | Hướng | Được | Giá |
| :--- | :--- | :--- | :--- |
| **A** | **Hub paradigm** (Luật-0 thuần): filler cùng slot ({Lý,Đinh,Trần}) → nơ-ron KIỂU; truyền đáp type-transfer (case con kiến ở quy mô) | Precision cao, chạy data nhỏ, đúng docx Cross-sample | KHÔNG fix synonym-probe (động từ tự do); metric ~trung tính → cần eval type-transfer riêng |
| **B** | **Pretrained E** — nới Luật 0 CHỈ tầng embedding (fuzzy match), phần còn lại graph-only | **Cách DUY NHẤT** phá trần chữa≈trị thật | Vi phạm Luật 0 (no-pretrain) — **quyết định triết lý nền tảng** |
| **C** | **Scale data đồng hiện** — thêm corpus để synonym tự nổi | Không đụng Luật 0 | Train lại lớn; tiếng Việt vẫn có thể thưa ngữ cảnh |

**Khuyến nghị Claude:** nếu Luật 0 bất khả xâm phạm → **A** (+ ghi synonym-probe là trần đã-biết,
đo bằng type-transfer). Nếu chấp nhận nới cho riêng tầng E → **B** là đòn phá trần thật, vẫn giữ
"mọi suy luận khác từ đồ thị". Câu hỏi chốt: **Luật 0 (cấm pretrain) là bất khả xâm phạm hay nới
được cho riêng embedding E?**

**Cập nhật cuối phiên 2026-07-01 — HANDOFF (chi tiết: [`docs/NN-AIVIE v4.md`](docs/NN-AIVIE%20v4.md) §7):**
- Thêm **Trụ 6** (Theme neuron + anaphor học + SFT) từ 2 ý tưởng tác giả — xem `docs/NN-AIVIE v4.md`.
- **P0 xong** (đo được): eval_anaphor + composition on_topic + gold `composition.jsonl`/`anaphor.jsonl`.
  Baseline big_a2: composition **73/0/92**, anaphor **50%** (đúng cho triều đại/người, sai animacy vô tri).
- **P2 anaphor-học-PMI: ĐO ÂM TÍNH (33%<50%) → REVERT** (không tách được đại-từ khỏi từ-nội-dung
  đồng-hiện nếu thiếu POS). Method giữ lại nhưng KHÔNG wire mặc định.
- **⚠️ Đầu phiên sau (lịch sử 07-01):** ~~`scripts/eval_v4.py`~~ — **đã gỡ 2026-07-09**; dùng
  `scripts/eval_discourse_prompts.py` + `pytest -q` (131 pass).
- **Việc tiếp (07-02 sáng):** ~~P1 Theme neuron (chưa làm)~~ → **P1 xong** (xem block dưới) →
  (chờ tác giả) quyết Trụ 3 synonym.

**Cập nhật 2026-07-02 — P0+P1 XONG (chi tiết: `docs/NN-AIVIE v4.md` mục 6 "Trụ 6 P0+P1" + §7):**
- Đầu phiên xác nhận xanh: 253 pass, eval khớp baseline (revert P2 sạch).
- **P0:** gold composition big_a2 mở rộng 5→**13 ca** (`composition.grade_v3_big_a2.jsonl`, gold theo
  checkpoint qua `_gold_for`) — mọi fact kiểm chứng tồn tại trước khi viết. Baseline lộ drift hệ
  thống: thực thể *có mặt* ≠ *là chủ đề câu* (Napoléon/"chiến tranh chống Napoléon", Biển Đông≈Baltic).
- **P1 — THEME NEURON `Concept:entity`** (`parent_store.theme_neuron`): vật chất hóa "mọi fact VỀ
  thực thể", trọng số theme = 1/(1+vị_trí) (topic-first, thuần vị trí — Luật 0); lazy trên checkpoint
  có sẵn. `facts_about` đọc từ theme; cờ `compose_theme_weight=1.0` (ON). Đo: big_a2 composition
  **67→69 cov / 82→84 top**; grade_v3 93 (−7 = artifact gold khóa-selection); retrieval nguyên vẹn
  từng số. + Vá `strip_leading_entity` (hết "họ **Đạo** đánh thắng"). **258 test pass** (cuối phiên:
  **269** sau T1 + bugfix).
- **3 trần lộ ra** (07-02): đại từ giả "các" (**đã vá** gate vị ngữ), token case "Trái Đất"≠"Trái
  đất" (**vá đọc** `case_equiv_ids`; re-ingest nếu muốn một token), câu wiki kiểu từ điển (**quality
  gate mới** — cần re-ingest) → việc tiếp §7.5.

**Cập nhật 2026-07-02 (chiều) — TRỤ 7: nơ-ron trừu tượng t1 (ý tưởng gốc tác giả) + T0 (chi tiết:
`docs/NN-AIVIE v4.md` Trụ 7):**
- **Thiết kế t1**: gold SFT dài ("bài văn về mèo") → một nơ-ron episode; kích hoạt toàn bộ token
  → cha nào cộng hưởng (np1, np2) → nối vào t1 (giữ THỨ TỰ); phân rã Re-Pair trên t1; thang t2+.
  3 vai trò: essay agenda (A4), coherence episode khi ghép câu, enrichment nghĩa (mắt sáng↔tinh anh).
- **T0 đo (thí nghiệm rẻ)**: E-episode (PPMI term×đoạn wiki, 24.8k đoạn) — **intrinsic DƯƠNG**
  (cấy/trồng rank 75→3; hàng xóm `chữa`={bệnh,thuốc,sẹo,tật}) nhưng **extrinsic ÂM TÍNH** (Syn 33%
  bất động: đa nghĩa "trị/giá trị", "bảo/bảo tàng" + 2 ca fail là bug ranking "Người X là ai").
  → synonym-qua-episode hoãn sang cạnh CỤM↔CỤM (T3); vai trò chính t1 (T1/T2) không phụ thuộc.
- **Phụ**: fuzzy+E-window trên big_a2 Syn 33→50 (QA/Para giữ); grade_v3 hại Para 75→62 → default
  giữ OFF, bật theo-checkpoint (chờ tác giả). Data mới: `statements_doc.jsonl` (45k câu, doc_id+
  para+title). Gold dài T2: `bkai-foundation-models/vi-alpaca`.

**⚠️ ĐÍNH CHÍNH t1 (tác giả, 2026-07-02):** episode/bài dài = MỘT NƠ-RON CHA THẬT (không chia
câu/đoạn; chuỗi token đầy đủ) + tầng link phụ (t1→fact có thứ tự, fact↔fact ngang, trung
gian→fact). Hiện thực TK_004 (chia câu + AbstractNeuron trỏ member) sẽ migrate theo
`TK_010_T1_UNIFIED_PARENT.md` — một kiểu train thống nhất top-down cho mọi quy mô.

**Cập nhật 2026-07-02 (cuối phiên) — TRỤ 7 T1 TRIỂN KHAI + BUGFIX RANKING (chi tiết:
`docs/NN-AIVIE v4.md` mục 6 + §7):**
- **T1 episode layer** — `AbstractNeuron` + `scripts/build_episodes.py` (wiki `statements_doc.jsonl`,
  nhóm theo `doc_id#para`). Checkpoint: `grade_v3_big_a2_t1.pkl` — **9,241 episode**, 38,563/45k câu
  match cha (86%), 27,008/44,133 cha thuộc ≥1 episode. `pick_facts_coherent` + cờ
  `compose_episode_weight=1.0` (ưu tiên fact cùng episode khi ghép).
- **Bug ranking (2 lớp)** — (a) `rebuild_hierarchical_index` không prune token khỏi `_index` (lệch
  phân đoạn cụm, vd "Người trị bệnh"); (b) `best_match` gate **primary entity** — curated QA khớp
  khung `là … ai` nhưng không chứa thực thể hỏi (Napoléon→Trung thu) → index fallback qua
  `_index[primary]`. Hybrid: Napoléon/ Người dạy học đúng entity; QA curated trong miền giữ nguyên.
- **Bug triage** — `case_equiv_ids` (Trái Đất≡Trái đất trong `facts_about`); gate định từ `các`
  (successor overlap vị ngữ); `quality.py` loại dòng từ điển/chữ Nho.
- **Eval v4** mở rộng: `_gold_for`, eval_anaphor, verbatim test. **269 test pass** (1 skipped).
  Graph UI: `AIVIE_CHECKPOINT=checkpoints/grade_v3_big_a2_t1.pkl python scripts/run_viz.py`.
- **Việc tiếp:** eval đầy đủ gold n=13 trên `_t1`; T2 essay (`vi-alpaca`); re-ingest checkpoint mới;
  Trụ 3 synonym (chờ quyết định); P2/P3 anaphor+SFT.

### 0.6.9. HỢP NHẤT SINH A3 MULTI — hoàn tất (2026-07-09)

> Tiếp nối P1-α (2026-07-03): **đã gỡ hẳn** shim hybrid/compose/essay và eval đa-mode. Chi tiết:
> README §0.4.1, `TK_019` §7, `docs/NN-AIVIE v4.md` nhật ký 2026-07-09.

`hca/generate_hebbian.generate(multi=True)` = **điểm vào duy nhất** → `generate_autoregressive`.
Retrieval (`best_match` / `facts_about` / discourse pool / hub-guard) chỉ **lập agenda**; token
surface luôn sinh token-by-token. Paragraph pool **luôn bật** (TK_035). VarLM cluster-aware (TK_031).
Checkpoint mặc định: **v10 paragraph** (`network_v10_histwiki_paragraph.pkl`). Graph UI: chỉ A3 multi.
**131 test pass.**

### 0.5.16. Bug ranking "X là ai" + episode t1 (2026-07-02)

Hai lớp bug retrieval lộ qua eval/UI (Hybrid/A3), **không** ảnh hưởng Compose (đi `facts_about`):

1. **Index prune (cụm Re-Pair):** token thuộc cụm cha bị loại khỏi `_index` → cha đúng biến mất
   khỏi candidate khi query lệch phân đoạn (vd "Người trị bệnh" vs cụm "chữa bệnh"). **Vá:**
   `rebuild_hierarchical_index` giữ mọi token trong `_index`; ưu tiên cụm là việc của scoring.
2. **Curated two-tier + khung `là … ai`:** QA curated "Trung thu là tết của ai" thắng vì khớp
   `là`/`ai` dù không chứa `Napoléon`; wiki về Napoléon bị loại pool curated (priority 0.6). **Vá:**
   `best_match` yêu cầu curated chứa **primary** (token phân biệt nhất); nếu không → tìm trong
   pool đầy đủ hoặc `_entity_parent_indices` (index fallback, hoạt động cả khi `fuzzy_topk=0`).

**T1 episode:** `scripts/build_episodes.py` → `grade_v3_big_a2_t1.pkl`; compose dùng
`pick_facts_coherent` khi checkpoint có episodes.

---

## I. TỔNG QUAN Ý TƯỞNG CỐT LÕI

Dự án hướng tới việc xây dựng một **Mô hình Ngôn ngữ Nhỏ (Small Language Model - SLM)** nhưng không dựa trên kiến trúc Transformer toán học truyền thống. Thay vào đó, mô hình mô phỏng cơ chế sinh học của não bộ thực tế dựa trên **Quy luật Hebb (Hebbian Learning)**: *"Neurons that fire together, wire together"* (Các nơ-ron cùng kích thích sẽ tạo liên kết và sinh nơ-ron chung).

### Các điểm mấu chốt của kiến trúc:
1. **Parallelized Hebbian Learning:** Ép cơ chế học sinh học chạy được trên phần cứng hiện đại (GPU) với **Batch size khổng lồ (hàng triệu token)** bằng cách ma trận hóa đồ thị động.
2. **Curriculum Learning (Giáo trình 12 lớp):** Huấn luyện theo lộ trình giáo dục phổ thông — từ lớp 1 đến lớp 12 — mỗi cấp tăng dần độ phức tạp từ vựng, ngữ pháp và tri thức chuyên ngành, thay vì nạp toàn bộ dữ liệu cùng lúc.
3. **Hebbian Context Attention (HCA):** Thay Self-Attention của Transformer bằng cơ chế co-activation trên ma trận kề $A$, với ba vai trò Query / Key / Value được chuyển ngữ sang $H$, $R_{-k}$ và Composite nodes — không dùng ma trận $c_q, c_k, c_v$ học bằng backprop. **Dual Context Readout** ($C_{\text{frame}}$ + $C_{\text{bind}}$): mọi token quá khứ đều đóng góp ngữ cảnh với trọng số khác nhau (Mục III.8).
4. **Cửa sổ ngữ cảnh tương đối & K động:** Quét ngược các **token quá khứ** (từ đã xuất hiện trước token hiện tại) trong phạm vi $K_{\max}$, với $K_{\text{eff}}$ do **Scope Neurons** điều khiển — mạng tự quyết nhìn gần hay nhìn xa tùy ngữ cảnh.
5. **Autoregressive Token Generation:** Đầu ra trả về các token đơn lẻ nối tiếp nhau cho đến khi kích hoạt nơ-ron kết thúc `[EOS]`.
6. **Chế độ Dạy Trực Tiếp (Direct Teaching / NVL):** Lesson `.nvl`, `.lesson.yaml`, hoặc `.lesson.json` — cùng Lesson AST, imprint vào $A$ bất kỳ lúc train (Phụ lục A).

---

## II. TOÁN HỌC HÓA KIẾN TRÚC MẠNG TRÊN GPU

Để tối ưu hóa cho GPU, cấu trúc đồ thị nơ-ron động được chuyển đổi thành các phép tính ma trận tĩnh với không gian nơ-ron dự phòng khổng lồ (Latent Nodes).

### 1. Thành phần cấu trúc
* **Ma trận Kề Động ($A$):** Kích thước $(N, N)$, biểu diễn độ mạnh liên kết giữa $N$ nơ-ron (bao gồm nơ-ron từ vựng cố định và nơ-ron chung cấp cao).
* **Ma trận Kích hoạt ($H$):** Kích thước $(B, N)$, biểu diễn trạng thái kích hoạt / năng lượng của các nơ-ron trong một Batch có kích thước $B$.
* **Buffer Decay ($D$):** Kích thước $(B, K_{\max})$, lưu vết năng lượng phân rã của các token quá khứ — tương đương KV cache nhưng dùng hàm mũ thay vì lưu vector Q/K/V.
* **Ma trận Thưởng Phạt ($R$):** Kích thước $(N, N)$, dùng ở Giai đoạn 3 (Dopamine Feedback) để củng cố hoặc suy yếu liên kết.

### 2. Công thức Cập nhật Trọng số hàng loạt (Batch Hebbian Update)
$$\Delta A = \eta \cdot (H^T \times H)$$
*(Trong đó $\eta$ là tốc độ học).*

### 3. Sơ đồ phân tầng hệ thống
Mạng được tổ chức theo các tầng nơ-ron để tránh bùng nổ tổ hợp:

| Tầng nơ-ron | Loại Nơ-ron | Mô tả chức năng |
| :--- | :--- | :--- |
| **Tầng 1** | Token Nodes | Cố định (Ví dụ: 50,000 từ vựng cơ bản) |
| **Tầng 2** | Composite Nodes | Các nơ-ron chung cấp 1 (đại diện cho cặp từ ghép) |
| **Tầng 3** | Structural Nodes | Ngữ pháp, Rule nodes (NVL), Scope (`narrow`/`medium`/`wide`), gate $G_{-k}$, `[Ambiguity]`, `[Self]`/`[Other]` |

---

## III. HEBBIAN CONTEXT ATTENTION (HCA) — THAY THẾ SELF-ATTENTION

Mô hình thay thế cơ chế Self-Attention của Transformer (Q·Kᵀ → softmax → V) bằng **Hebbian Context Attention (HCA)** — attention dựa trên co-activation trên ma trận kề $A$, kết hợp Decay và Retroactive Wave. Thiết kế tham chiếu sliding window attention trong các mô hình GPT hiện đại (ví dụ: pattern `SSSL` per-layer trong nanochat), nhưng trọng số đến từ Hebbian learning thay vì backprop.

### 1. Cửa sổ ngữ cảnh tương đối (Relative Context Window)

#### Định nghĩa: token quá khứ

**Token quá khứ** = các token **đã xuất hiện trước token hiện tại** trong chuỗi (đọc / sinh từ trái sang phải). Token tại **vị trí $0$** là token đang được xử lý hoặc dự đoán; token tại **vị trí $-k$** là token thứ $k$ đứng ngay trước nó.

*Ví dụ:* *"Em bé yêu chó, bởi vì nó rất đáng yêu"* — khi xử lý `"nó"`:

| Slot | Token quá khứ |
| :--- | :--- |
| $R_{-1}$ | `"vì"` |
| $R_{-2}$ | `"bởi"` |
| $R_{-3}$ | `"chó"` |
| $R_{-4}$ | `"yêu"` |
| $R_{-5}$ | `"bé"` |

Các token **đứng sau** `"nó"` (`"rất"`, `"đáng yêu"`) không nằm trong cửa sổ forward — vì khi sinh autoregressive, mô hình chưa "nhìn thấy" tương lai. Chúng chỉ ảnh hưởng qua **Retroactive Wave** (Mục III.6) khi mô hình đã xử lý đến đó.

#### $K_{\max}$ theo cấp lớp (đa ngôn ngữ)

$K$ được tính theo **token** (sau tokenizer), không theo từ thô — phù hợp mọi ngôn ngữ (Việt, Anh, Trung, Nhật, Đức, Ả Rập…). $K = 5$ chỉ phù hợp lớp sơ cấp; đa ngôn ngữ phổ biến cần $K_{\max}$ lớn hơn ở lớp cao:

| Lớp | $K_{\max}$ (token) | Scope khả dụng | Mục tiêu |
| :--- | :--- | :--- | :--- |
| **1–2** | 8 | `narrow` | Câu 2–5 token |
| **3–5** | 16 | `narrow` + `medium` | Câu đơn, hội thoại ngắn |
| **6–8** | 32 | cả ba Scope | Coreference, câu ghép — **~20 token** là mức cân bằng tốt |
| **9–12** | 64 | full + Ambiguity | Văn bản dài, thuật ngữ, câu Đức/Thổ |

| Hiện tượng ngôn ngữ | Khoảng cách thường gặp | $K \approx 5$ | $K \approx 20$ |
| :--- | :--- | :--- | :--- |
| Collocation / bigram | 1–3 token | Đủ | Dư |
| Đại từ → chủ thể | 3–8 token | Hay thiếu | Đủ hầu hết |
| Mệnh đề phụ | 5–15 token | Thiếu | Khá ổn |
| Câu dài (Đức, học thuật) | 15–30+ token | Rất thiếu | Cần $K_{\max}$ 32–64 + Decay |

**Lợi ích thiết kế:**
* **Sinh học:** Mô phỏng working memory — tập trung vào token gần, mở rộng khi mơ hồ.
* **Tính toán:** Độ phức tạp $O(N \cdot K_{\text{eff}})$ thay vì $O(N^2)$; $K_{\text{eff}} \ll K_{\max}$ khi Scope hẹp.
* **Đa ngôn ngữ:** Một cơ chế $R_{-k}$ + Decay $D[k] = e^{-\lambda k}$; có thể tinh chỉnh $\lambda$ theo ngôn ngữ nếu cần, không cần thiết kế lại kiến trúc.

### 2. Thuật toán HCA — bốn bước mỗi token

```
Token Layer (K₁)  →  Composite Layer (K₂)  →  Structural Layer (K₃ + Decay + Retroactive)
```

**Bước 1 — Kích hoạt:** $H[w_0] \leftarrow 1$; slot $R_0$ bật.

**Bước 2 — Context Read (forward):** **mọi** token quá khứ $w_{-k}$ trong cửa sổ đều đóng góp vào ngữ cảnh — kể cả `con` đầu câu, `kêu`, `mà` — không loại trừ token nào. Sự khác biệt nằm ở **trọng số** $\alpha_{-k}$ (khoảng cách, liên kết Hebbian, slot $R_{-k}$, gate). Chi tiết **Dual Context Readout** tại Mục III.8.

Hai kênh đọc ngữ cảnh song song trên **cùng một lịch sử** $w_{-1}, \ldots, w_{-K_{\text{eff}}}$:

$$\alpha_{-k}^{\text{frame}} = A[w_{-k}, w_0] \cdot A[R_{-k}, w_0] \cdot D_{\text{frame}}[k] \cdot g_{-k}$$

$$\alpha_{-k}^{\text{bind}} = A[w_{-k}, w_0] \cdot D_{\text{bind}}[k] \cdot g_{-k}$$

$$V(w_{-k}) = H[w_{-k}] + H[\text{composite}_k]$$

$$C_{\text{frame}} = \sum_{k=1}^{K_{\text{eff}}} \alpha_{-k}^{\text{frame}} \cdot V(w_{-k}), \quad C_{\text{bind}} = \sum_{k=1}^{K_{\text{eff}}} \alpha_{-k}^{\text{bind}} \cdot V(w_{-k})$$

* $C_{\text{frame}}$ — kênh **khung / vị trí**: nhấn mạnh $A[R_{-k}, w_0]$ (token gần, vai trò cấu trúc).
* $C_{\text{bind}}$ — kênh **nội dung / gắn nghĩa**: decay chậm hơn ($\lambda_{\text{bind}} < \lambda_{\text{frame}}$), kéo anchor xa như âm thanh → loài.

**Bước 3 — Read-out (dự đoán):**

$$\text{score}[v] = A[w_0, v] \cdot \Big( \langle C_{\text{frame}}, H[v] \rangle + \langle C_{\text{bind}}, H[v] \rangle \Big) \cdot \text{fluency}(v)$$

Tùy chọn **candidate narrowing**: nếu $A[w_0, :]$ có top-$k$ láng giềng mạnh (ví dụ sau `con` → chó/mèo/lợn), chỉ tính score trên tập ứng viên đó rồi rerank bằng $C_{\text{bind}}$.

**Bước 4 — Retroactive Wave (nếu entropy cao):** mở rộng $K_{\text{eff}}$, dội tín hiệu ngược qua $A$; tính lại score (Mục III.6).

### 3. Chuyển ngữ Q / K / V sang Hebbian

Transformer tách embedding thành ba projection $c_q, c_k, c_v$ học bằng backprop. AIVIE **không có** ba ma trận đó — ba vai trò nằm sẵn trong đồ thị:

| Transformer | AIVIE | Thành phần |
| :--- | :--- | :--- |
| **Q** = $c_q(x_t)$ — "tôi cần gì?" | $H[w_0] \times A[w_0, :]$ | Token hiện tại + hàng liên kết outgoing |
| **K** = $c_k(x_i)$ — "tôi liên quan không?" | $A[w_{-k}, w_0] \cdot A[R_{-k}, w_0] \cdot D[k]$ | Token quá khứ + slot vị trí + decay |
| **V** = $c_v(x_i)$ — "tôi mang thông tin gì?" | $H[w_{-k}] + H[\text{composite}]$ | Nội dung ngữ nghĩa thực |
| $\text{softmax}(Q \cdot K^T)$ | $\alpha_{-k}$ (năng lượng vật lý, không softmax) | Co-activation × decay × gate |
| $\sum \alpha \cdot V$ | $C = \sum_k \alpha_{-k} \cdot V(w_{-k})$ | Context vector |
| RoPE trên Q, K | Nơ-ron $R_{-1} \ldots R_{-K_{\max}}$ | Mã vị trí tương đối rời rạc |
| Value Embeddings | Token node + Composite node | Identity từ gắn trực tiếp vào V |
| Multi-head | Nhóm Composite nodes | Mỗi "head" = một cụm nơ-ron chung |
| Backprop $\partial L / \partial c_q$ | $\Delta A \mathrel{+}= \eta \cdot H[w_0]^T H[w_{-k}]$ | Q và K cùng fire → wire together |

**Công thức tương đương:**

$$\text{Transformer:}\quad O_t = \sum_i \text{softmax}(Q_t \cdot K_i)\, V_i$$

$$\text{AIVIE:}\quad C = C_{\text{frame}} + C_{\text{bind}}, \quad \text{score}[v] = A[w_0, v] \cdot \langle C, H[v] \rangle$$

**Triển khai GPU (ma trận ảo mỗi bước):**

```python
# history: (B, K_max) — MỌI token quá khứ; A: (N,N); H: (B,N)

V = H[history] + H[composites]                  # (B, K_max, N)

# Kênh frame — nhấn vị trí tương đối
a_frame = A[history, w0] * A[R_slots, w0] * D_frame * g   # (B, K_max)
C_frame = (a_frame.unsqueeze(-1) * V).sum(dim=1)          # (B, N)

# Kênh bind — nhấn nội dung, decay chậm hơn
a_bind  = A[history, w0] * D_bind * g                     # (B, K_max)
C_bind  = (a_bind.unsqueeze(-1) * V).sum(dim=1)           # (B, N)

C = C_frame + C_bind
score = A[w0, :] * (C @ H.T) * fluency                   # read-out
```

### 4. K động — Scope Neurons

Thay vì cố định $K$ cho mọi câu, **độ rộng cửa sổ hiệu dụng** $K_{\text{eff}}$ do mạng tự quyết qua Structural Nodes:

| Node | Ý nghĩa | $K_{\text{eff}}$ điển hình |
| :--- | :--- | :--- |
| `[Scope:narrow]` | Ngữ cảnh cục bộ | 3–5 token |
| `[Scope:medium]` | Câu đơn / hội thoại | 8–16 token |
| `[Scope:wide]` | Coreference, mệnh đề phụ | 20–32 token |
| `[Ambiguity]` | Entropy cao → kích mở rộng | Trigger Retroactive + wide |

**Gate trên từng slot** $G_{-k}$ (Tầng 3):

$$g_{-k} = \sigma\!\left( \sum_s H[\text{Scope}_s] \cdot A[\text{Scope}_s, R_{-k}] \right)$$

$$K_{\text{eff}} = \max\,\{ k : g_{-k} > \tau \} \quad \text{hoặc} \quad K_{\text{eff}} = \sum_k g_{-k} \text{ (dạng mềm)}$$

**Luồng 2 pha (kết hợp Retroactive):**
1. Pha 1: $K_{\min}$ — chỉ $R_{-1} \ldots R_{-K_{\min}}$; tính score.
2. Pha 2: nếu $\text{entropy}(\text{score}) > \tau$ → kích `[Scope:wide]` / mở $G_{-k}$ xa hơn → tính lại.

**Học Hebbian trên gate:**
* Dự đoán đúng với K hẹp → củng cố `Scope:narrow`, LTD trên slot xa.
* Cần mở rộng K mới đúng → dopamine củng cố `Scope:wide` + các $G_{-k}$ vừa bật.

Trên GPU: cấp phát buffer $K_{\max}$, mask sparse theo $g_{-k}$ — chi phí $\propto K_{\text{eff}}$, không phải $K_{\max}$.

### 5. Trí nhớ ngắn hạn — Decay

Hai profile decay trên **cùng** cửa sổ lịch sử (không tách tập token):

$$D_{\text{frame}}[k] = e^{-\lambda_{\text{frame}} k}, \quad D_{\text{bind}}[k] = e^{-\lambda_{\text{bind}} k} \quad (\lambda_{\text{bind}} < \lambda_{\text{frame}})$$

* $\lambda_{\text{frame}}$ lớn hơn → token **gần** $w_0$ chi phối khung (ví dụ `là` → `con`).
* $\lambda_{\text{bind}}$ nhỏ hơn → token **xa** vẫn đóng góp nội dung (ví dụ `ụt ịt` → `lợn`).

Khi token xuất hiện, nơ-ron không tắt ngay mà để lại vết năng lượng phân rã. Token **ngoài cửa sổ** ($k > K_{\max}$) không được quét trực tiếp; ảnh hưởng chỉ còn qua $D$ và Composite / Structural nodes.

* **Ứng dụng:** *"Bông hoa này nó rất đẹp"* → `"nó"` mượn năng lượng còn sót của `"Bông hoa"` (Relay Node).

### 6. Retroactive Wave — dội ngược có điều kiện

Khác Transformer (chỉ causal forward), AIVIE bổ sung **dội ngược** khi Q mơ hồ (đại từ, từ đồng âm):

$$\text{if } H[\text{Ambiguity}] > \tau: \quad H[w_{-k}] \mathrel{+}= \beta \cdot A[w_{-k}, w_0] \cdot C \quad \text{(rescore)}$$

* *"Tôi thích đường, tôi thích ăn ngọt"* → `"ăn ngọt"` dội về → nghĩa **đường ăn**.
* *"Em bé yêu chó, bởi vì nó rất đáng yêu"* → `"đáng yêu"` + vết `"chó"` (gần hơn `"em bé"`) → `"nó"` = chó.

### 7. So sánh với Transformer (tham chiếu nanochat)

| | Transformer / nanochat | AIVIE (HCA) |
| :--- | :--- | :--- |
| Trọng số attention | $\text{softmax}(QK^T)$ học backprop | $\alpha_{-k}^{\text{frame}}$, $\alpha_{-k}^{\text{bind}}$ trên mọi $w_{-k}$ |
| Cửa sổ | `window_pattern="SSSL"` cố định theo layer | $K_{\text{eff}}$ động qua Scope Neurons |
| Học | $\partial L / \partial c_q, c_k, c_v$ | $\Delta A = \eta \cdot H^T R H$ |
| Nhìn tương lai | Không (causal) | Không forward; Retroactive khi đã đọc tiếp |
| Chi phí | $O(T \cdot W)$ với window $W$ | $O(T \cdot K_{\text{eff}})$ |
| Ngữ cảnh | Mọi token trong window đều qua attention | Mọi $w_{-k}$ đều đóng góp; trọng số khác qua $\alpha^{\text{frame}}$ / $\alpha^{\text{bind}}$ |

### 8. Dual Context Readout — Ngữ cảnh toàn câu có trọng số

#### Nguyên tắc cốt lõi

Khi dự đoán tại $w_0$, **không loại trừ** bất kỳ token quá khứ nào. `con` đầu câu, `mà`, `kêu`, `ụt ịt`, `là`, `con` cuối — **tất cả** đều vào tổng ngữ cảnh; chỉ **trọng số** $\alpha_{-k}$ thay đổi theo:

| Yếu tố | Ảnh hưởng |
| :--- | :--- |
| **Khoảng cách $k$** | $D_{\text{frame}}[k]$, $D_{\text{bind}}[k]$ — gần vs xa |
| **Nội dung $w_{-k}$** | $A[w_{-k}, w_0]$ — token đó liên quan đến vị trí dự đoán không |
| **Vai trò vị trí $R_{-k}$** | $A[R_{-k}, w_0]$ — chỉ kênh frame; phân biệt `con₁` vs `con₂` |
| **Gate $g_{-k}$** | Scope Neurons mở/đóng slot |
| **Composite** | $H[\text{composite}_k]$ gom cụm (`kêu ụt ịt`, `là con`) |

Hai kênh $C_{\text{frame}}$ và $C_{\text{bind}}$ là **hai cách cộng trọng số** trên cùng danh sách $w_{-k}$, không phải hai tập token riêng.

#### Ví dụ: khung câu động vật — tiếng kêu

**Dataset huấn luyện (rút gọn):**

```
chó kêu gâu gâu, con mà kêu gâu gâu là con chó
mèo kêu meo meo, con mà kêu meo meo là con mèo
lợn kêu ụt ịt, con mà kêu ụt ịt là con lợn
```

**Truy vấn:** `con mà kêu ụt ịt là con` (sinh tiếp 2 token)

Lịch sử khi dự đoán (đọc từ phải sang trái từ $w_0$):

| $k$ | $w_{-k}$ | Vai trò trong câu |
| :---: | :--- | :--- |
| 1 | `con` | `con₂` — slot loài (gần $w_0$) |
| 2 | `là` | Liên từ kết |
| 3 | `ịt` | Phần âm thanh |
| 4 | `ụt` | Phần âm thanh |
| 5 | `kêu` | Động từ |
| 6 | `mà` | Liên từ |
| 7 | `con` | `con₁` — mở khung |

---

**Dự đoán 1:** `con mà kêu ụt ịt là [?]` — kỳ vọng `con`

| $w_{-k}$ | $\alpha^{\text{frame}}$ | $\alpha^{\text{bind}}$ | Giải thích |
| :--- | :---: | :---: | :--- |
| `là` ($k{=}1$) | **cao** | trung bình | $R_{-1}$ + decay frame → đẩy mạnh `con` sau `là` |
| `ụt`, `ịt` | thấp | trung bình | Xa hơn; bind còn gợi `lợn` nhưng **chưa thắng** ở slot này |
| `kêu` | thấp | thấp–TB | Góp khung “câu kêu”, không chọn loài |
| `con₁`, `mà` | rất thấp | thấp | Xa; vẫn đóng góp nhẹ vào frame tổng thể |

→ $\text{score}[\text{con}] \gg \text{score}[\text{lợn}]$ vì $C_{\text{frame}}$ chi phối tại $k{=}1$.

---

**Dự đoán 2:** `con mà kêu ụt ịt là con [?]` — kỳ vọng `lợn`

| $w_{-k}$ | $\alpha^{\text{frame}}$ | $\alpha^{\text{bind}}$ | Giải thích |
| :--- | :---: | :---: | :--- |
| `con` ($k{=}1$, `con₂`) | **cao** | trung bình | Mở **tập ứng viên** chó/mèo/lợn qua $A[\text{con}, v]$ |
| `là` | trung bình | thấp | Đã qua; ít ảnh hưởng trực tiếp |
| `ụt`, `ịt` | thấp | **cao** | $D_{\text{bind}}$ chậm + $A[\text{ụt ịt}, \text{lợn}]$ → kéo `lợn` |
| `kêu` | thấp | thấp | Khung động từ; không kích hoạt gâu/meo vì chúng **không có trong câu** |
| `con₁`, `mà` | rất thấp | thấp | Vẫn góp ngữ cảnh khung; không quyết định loài |

→ $\text{score}[\text{chó}] \approx \text{score}[\text{mèo}]$ từ frame; $\text{score}[\text{lợn}]$ thêm mạnh từ $C_{\text{bind}}$ → **chọn `lợn`**.

#### Luồng suy luận (tóm tắt)

```
Mọi w_{-k} ──► α_frame(k) ──► C_frame ──┐
                 α_bind(k)  ──► C_bind  ──┼──► score[v] ──► token tiếp theo
Mỗi bước autoregressive: w_0 thay đổi → toàn bộ α_{-k} tính lại
```

Không có “chỉ `là` quan trọng” hay “chỉ `ụt ịt` quan trọng” — **cùng một cơ chế**, trọng số tự điều chỉnh theo $w_0$ và $k$.

#### Composite tự sinh (Tầng 2) hỗ trợ binding

Qua $\Delta A = \eta H^T H$, các cụm lặp lại có thể spawn composite **trong training**, không tạo sẵn:

* `kêu` + âm thanh → composite âm thanh
* `là` + `con` → composite slot loài
* Toàn khung → composite frame (promote lên Structural khi đủ mạnh)

Composite tham gia $V(w_{-k})$ — tăng trọng số hiệu dụng mà không cần node “ý niệm” định nghĩa trước.

#### Học Hebbian (Giai đoạn 3 — Dopamine)

| Kết quả | Củng cố |
| :--- | :--- |
| Sau `là` → đúng `con` | $A[\text{là}, \text{con}]$, $A[R_{-1}, \text{con}]$ |
| Sau `con` → đúng `lợn` khi có `ụt ịt` | $A[\text{con}, \text{lợn}]$, $A[\text{ụt ịt}, \text{lợn}]$, composite liên quan |
| Sau `con` → sai `chó` khi âm thanh là `ụt ịt` | LTD trên liên kết sai |

#### Tham số gợi ý (POC)

| Tham số | Gợi ý | Vai trò |
| :--- | :--- | :--- |
| $\lambda_{\text{frame}}$ | 0.3–0.5 | Ưu tiên token gần cho khung |
| $\lambda_{\text{bind}}$ | 0.1–0.2 | Giữ anchor nội dung xa |
| $K_{\min}$ / $K_{\text{eff}}$ | 4 / 8–16 (lớp thấp) | Pha 1 hẹp; pha 2 mở khi entropy cao |
| top-$k$ candidates | 8–32 | Giới hạn read-out sau slot mở (`con` → loài) |

---

## IV. GIÁO TRÌNH HUẤN LUYỆN LUỸ TIẾN

Giáo trình huấn luyện được thiết kế theo **ba trục** song song:
* **Trục dọc — 12 Lớp (Grade 1 → 12):** Quyết định *nội dung* và *độ khó* của dữ liệu, tương tự lộ trình giáo dục phổ thông Việt Nam.
* **Trục ngang — 4 Giai đoạn học tập (G1–G4):** Quyết định *phương pháp* huấn luyện tại mỗi lớp (nghe thụ động → bắt chước → thưởng phạt → tự nói).
* **Trục chéo — Chế độ Dạy Trực Tiếp (NVL):** Có thể **bật bất kỳ lúc nào** — trước, giữa, hoặc sau G1–G4 — khi cần imprint rule ngữ pháp / cấu trúc mà không đợi thống kê corpus.

```
Lớp 1 ──► Lớp 2 ──► ... ──► Lớp 12
  │          │                  │
  ▼          ▼                  ▼
[G1][G2][G3][G4]  ...  [G1][G2][G3][G4]
  ↑ Giai đoạn G1–G4 lặp lại tại mỗi lớp

  NVL lesson ═══════════════════════════►  (bất kỳ thời điểm: xen kẽ G1–G4)
  ví dụ: G1 ──► [NVL: copula] ──► G2 ──► G3 ──► [NVL: câu ghép] ──► G4
```

### A. Hệ thống 12 Lớp huấn luyện

Mỗi "lớp" tương ứng một cấp độ năng lực ngôn ngữ và tri thức. AI chỉ được thăng lớp khi đạt ngưỡng ổn định về độ mượt tín hiệu (Signal Fluency) và độ chính xác dự đoán token.

| Lớp | Độ phức tạp ngôn ngữ | Ví dụ nội dung |
| :--- | :--- | :--- |
| **Lớp 1–2** | Từ đơn, câu 2–3 từ, ghép vần | *"Con mèo"*, *"Mẹ ơi"* |
| **Lớp 3–4** | Câu đơn giản, mô tả sự vật quen thuộc | *"Con mèo màu trắng đang ngủ"* |
| **Lớp 5–6** | Câu ghép, liên từ cơ bản | *"Em thích chơi vì ngoài trời đẹp"* |
| **Lớp 7–8** | Văn bản mô tả, tóm tắt, hội thoại phức tạp | Bài văn ngắn, đối thoại đa lượt |
| **Lớp 9–10** | Lập luận, phân tích, văn phong trung học | Tiểu luận, tranh biện |
| **Lớp 11–12** | Tri thức chuyên sâu, thuật ngữ học thuật | Bài báo khoa học, luận văn |

#### Dataset chuyên biệt theo từng lớp

Tại mỗi lớp, ngoài dữ liệu ngôn ngữ tổng quát, AI còn được huấn luyện trên các **bộ dữ liệu chuyên ngành** — tương tự cách học sinh vừa học Tiếng Việt vừa học các môn chuyên biệt ở trường:

| Môn / Lĩnh vực | Lớp bắt đầu | Mục tiêu huấn luyện |
| :--- | :--- | :--- |
| **Toán học** | Lớp 1 | Nhận diện số, phép tính, bài toán có lời văn |
| **Khoa học tự nhiên** | Lớp 3 | Mô tả hiện tượng tự nhiên, quan sát – giải thích |
| **Hóa học** | Lớp 8 | Thuật ngữ hóa học, phản ứng, công thức |
| **Vật lý** | Lớp 6 | Khái niệm lực, năng lượng, hiện tượng vật lý |
| **Sinh học** | Lớp 6 | Cấu trúc sinh vật, quá trình sống |
| **Lịch sử – Địa lý** | Lớp 4 | Sự kiện, địa danh, mối quan hệ không – thời gian |
| **Văn học** | Lớp 5 | Thể loại văn bản, hình ảnh, cảm xúc tác giả |
| **Ngữ pháp (NVL)** | Lớp 1+ | `.nvl`, `.lesson.yaml`, `.lesson.json` — pattern, bind, examples (Phụ lục A) |

Các dataset chuyên biệt giúp mạng Hebbian hình thành **nơ-ron chung cấp cao** (Composite / Structural Nodes) mang tính lĩnh vực — ví dụ cụm *"phản ứng oxi hóa khử"* liên kết chặt với nhau trong tầng Hóa học, độc lập với ngữ cảnh văn xuôi thông thường.

#### Tiêu chí thăng lớp

AI chỉ chuyển sang lớp tiếp theo khi thỏa mãn đồng thời:
1. **Signal Fluency** ≥ ngưỡng tối thiểu trên toàn bộ giai đoạn G1–G4 của lớp hiện tại.
2. **Độ chính xác dự đoán token** ≥ ngưỡng trên dataset ngôn ngữ tổng quát *và* ít nhất một dataset chuyên biệt của lớp đó.
3. **Ma trận kề $A$ ổn định:** Tốc độ thay đổi $\|\Delta A\|$ giảm dưới ngưỡng, tránh thăng lớp khi mạng còn dao động.
4. **Rule retention (sau lesson NVL):** Matcher đúng ≥ ngưỡng trên lesson pack kiểm tra của lớp (áp dụng sau mỗi đợt Direct Teaching, không phụ thuộc giai đoạn G1–G4).
5. **Generalize (sau G3):** Rule + corpus tự nhiên cùng đúng — tránh chỉ nhớ meta-lesson, không áp dụng được lên văn bản thật.

---

### B. 4 Giai đoạn học tập (lặp lại tại mỗi lớp)

Mỗi lớp trong 12 lớp trải qua đủ 4 giai đoạn G1–G4 dưới đây. **Chế độ Dạy Trực Tiếp (NVL)** có thể xen vào bất kỳ điểm nào trong chuỗi này (Mục IV.C). Dữ liệu đầu vào thay đổi theo cấp lớp, nhưng cơ chế học G1–G4 giữ nguyên.

#### Giai đoạn 1: Tắm ngôn ngữ (Passive Listening)
* **Phương pháp:** Nạp Batch khổng lồ dữ liệu văn bản cực kỳ đơn giản (câu 2-3 từ). AI chỉ nghe, không Predict.
* **Mục tiêu:** Cho GPU chạy phép nhân $H^T \times H$ một chiều để tạo ra các con đường mòn thô sơ nhất trong ma trận kề $A$.

#### Giai đoạn 2: Bắt chước xã hội (Social Imitation)
* **Phương pháp:** Nạp các cấu trúc hội thoại đa bên (Bố nói - Mẹ đáp - AI quan sát). 
* **Mục tiêu:** Kích hoạt các **Nơ-ron phản chiếu (Mirror Neurons)** để AI học cách tương tác từ cộng đồng, tạo ra các "liên kết mềm" tự nhiên.

#### Giai đoạn 3: Tương tác Nói leo & Thưởng Phạt (Dopamine Feedback)
AI bắt đầu sinh từng token đơn lẻ (Autoregressive). Hệ thống tạo ra Ma trận Thưởng Phạt ($R$) dựa trên nhãn chuẩn để can thiệp vào cấu trúc mạng:
* **Khi AI nói ĐÚNG (Thưởng):** Bơm Dopamine bằng hệ số nhân $\alpha > 1$ — củng cố liên kết Q↔K ($A[w_0, w_{-k}]$) và Scope gate đã dùng.
* **Khi AI nói SAI (Phạt):** Anti-Hebbian / LTD — suy yếu liên kết sai; nếu cần mở rộng $K_{\text{eff}}$ mới đúng, thưởng thêm `Scope:wide`.
$$\Delta A = \eta \cdot (H^T \times R \times H)$$

#### Giai đoạn 4: Tự trò chuyện (Self-Talk / Inner Monologue)
* **Phương pháp:** Ngắt hoàn toàn nhãn chuẩn. Cho AI mồi 1 từ và tự chạy vòng lặp sinh câu cho đến khi gặp token thỏa mãn `[EOS]`.
* **Mục tiêu:** Mạng tự đánh giá độ mượt mà của dòng điện nội bộ (Signal Fluency) để tự củng cố tư duy ngôn ngữ độc lập.

---

### C. Chế độ Dạy Trực Tiếp (Direct Teaching) — không phải giai đoạn cố định

Direct Teaching mô phỏng **giáo viên giải thích quy tắc** đúng lúc cần — truyền mẫu ngữ pháp khai báo (NVL) vào đồ thị thay vì đợi hàng nghìn/triệu lượt co-occurrence. Đây là **chế độ training** có thể kích hoạt **bất kỳ lúc nào**, độc lập với việc đang ở G1, G2, G3 hay G4.

#### Mục đích và cách dùng trong curriculum

| Khía cạnh | Mô tả |
| :--- | :--- |
| **Khi chạy** | **Bất kỳ lúc nào** — đầu lớp, giữa G1–G2, sau G3, khi phát hiện pattern yếu, hoặc khi thêm môn/ngữ pháp mới |
| **Đầu vào** | `.nvl` / `.lesson.yaml` / `.lesson.json` (Phụ lục A) |
| **Đầu ra** | Rule nodes (Tầng 3) + cạnh cấu trúc trong $A$ với $w_{\text{init}} > \theta_{\text{create}}$ |
| **Quan hệ G1–G4** | **Bổ sung**, không thay thế — G1–G4 củng cố và kiểm chứng bằng corpus tự nhiên |

**Ví dụ lịch xen kẽ (lớp 3):**

```
G1 (passive corpus) → [NVL: vi_copula] → G2 (hội thoại) → G3 (dopamine)
                              ↓
                    G1 tiếp (corpus có pattern copula) → [NVL: vi_SV] → G4
```

#### Khác biệt Direct Teaching vs G1 (passive)

| | Direct Teaching (NVL) | G1 Passive Listening |
| :--- | :--- | :--- |
| Thời điểm | Bất kỳ lúc nào trong train | Giai đoạn G1 trong curriculum |
| Nguồn | Lesson NVL do giáo viên/soạn | Corpus văn bản tự nhiên |
| Kích hoạt $H$ | **Ép fire** đồng thời các node trong mẫu (teacher forcing trên đồ thị) | Co-occurrence thụ động từ chuỗi token |
| Tốc độ pattern | Vài lượt `repeat` với $\eta_{\text{teach}}$ cao | Cần nhiều lượt lặp thống kê |
| Node mới | Rule, Var, Label (Tầng 3) | Composite emergent (Tầng 2) |

#### Công thức imprint (vẫn Hebbian, không backprop)

Compiler NVL sinh tập cặp node $(i, j)$ cần củng cố. Mỗi lượt dạy:

$$\Delta A \mathrel{+}= \eta_{\text{teach}} \cdot \sum_{(i,j) \in \text{pairs}(\text{lesson})} w_{ij} \cdot e_i e_j^T$$

Trong đó $e_i$ là kích hoạt sparse tại node $i$ (variable, constant, label, $R_{-k}$, Rule). Có thể lặp `repeat: N` với $\eta_{\text{teach}}$ giảm dần theo lượt.

**Tham số gợi ý:**

| Tham số | Gợi ý | Vai trò |
| :--- | :--- | :--- |
| $\eta_{\text{teach}}$ | 0.3–0.8 | Mạnh hơn $\eta$ thụ động (G1) |
| $w_{\text{init}}$ | 1.5–3.0 | Trọng số ban đầu cạnh rule |
| `repeat` | 1–5 | Số lần imprint mỗi lesson |
| $\theta_{\text{create}}$ | (emergent) | Rule imprint vượt ngưỡng này ngay khi dạy |

#### Luồng Direct Teaching (spec)

```
NVL lesson ──► Parser ──► Compiler ──► Imprint ──► A + Rule nodes
       ▲                                          │
       │         (có thể gọi lại bất kỳ lúc nào)  │
Corpus (G1–G4) ◄─────────────────────────────────┘ củng cố / sửa qua dopamine G3
```

#### Xung đột rule vs corpus

Khi rule NVL mâu thuẫn thực tế ngôn ngữ, **Giai đoạn 3 (dopamine)** ưu tiên corpus: LTD trên cạnh rule sai, củng cố liên kết đúng từ văn bản thật. Direct Teaching không làm rule “bất biến”.

---

## Phụ lục A: NVL — Neural Variable Language

NVL là DSL khai báo mẫu ngữ pháp cho **Chế độ Dạy Trực Tiếp**. Meta-syntax **không** xuất hiện trong corpus sinh văn bản — chỉ dùng lúc training để imprint cấu trúc.

**Ba định dạng lesson được hỗ trợ ngang hàng** — cùng semantics, cùng compiler đích (Lesson AST → imprint $A$):

| Định dạng | Extension gợi ý | Ai soạn | Ghi chú |
| :--- | :--- | :--- | :--- |
| **Tag NVL** | `.nvl` | Giáo viên, soạn tay | `<n.start>` … `<n.end>` |
| **YAML structured** | `.lesson.yaml` | Dev, curriculum | `slots:` dạng mảng — dễ đọc |
| **JSON structured** | `.lesson.json` | Tool, API, CI | JSON Schema validate |

```text
  .nvl          .lesson.yaml       .lesson.json
    │                │                  │
    └─ tag parser ───┴─ yaml loader ────┴─ json loader
                           │
                     Lesson AST (canonical)
                           │
                     Compiler → Imprint A + Rule nodes
```

*Transpile (tùy chọn):* `nvl ↔ yaml ↔ json` qua Lesson AST — không mất thông tin.

### A.1. Lesson AST (mô hình chung)

Mọi định dạng biên dịch về cùng cấu trúc logic trước khi imprint:

| Trường AST | Kiểu | Bắt buộc | Mô tả |
| :--- | :--- | :---: | :--- |
| `lesson_id` | string | ✓ | ID file lesson |
| `pattern_id` | string | ✓ | Tên rule (`vi_copula_S_la_NA`) |
| `grade` | int | | Lớp curriculum |
| `repeat` | int | | Số lần imprint (mặc định 1) |
| `eta_teach` | float | | Tốc độ học dạy trực tiếp |
| `w_init` | float | | Trọng số ban đầu cạnh rule |
| `slots` | Slot[] | ✓ | Thân pattern (thứ tự quan trọng) |
| `bind` | map Var→label | | `S` → `chủ ngữ`, … |
| `examples` | string[] | ✓ | Câu ví dụ để align + imprint token↔Var |

**Kiểu `Slot` (JSON/YAML structured):**

| Slot | Trường | Ví dụ | Tương đương tag NVL |
| :--- | :--- | :--- | :--- |
| Biến | `{ "var": "S" }` | `S` | `<n.variable:S>` |
| Hằng | `{ "const": "là" }` | `là` | `<n.constant:"là">` |
| Khoảng cách | `{ "gap": "s+", "min": 1 }` | ≥1 token | `<n.function:"s+">` |
| Lựa chọn | `{ "one_of": ["N", "A"] }` | N hoặc A | `<n.variable:N \|\| n.variable:A>` |

### A.2. BNF — Tag NVL (định dạng 1)

```bnf
lesson       ::= lesson_block | { legacy_statement }
lesson_block ::= "<n.start>" block_body "<n.end>"
block_body   ::= pattern_tag bind_tag? example_tag?
pattern_tag  ::= "<n.pattern:" RULE_ID ">" pattern_body
bind_tag     ::= "<n.bind>" { bind_line }           (* hoặc legacy bind_decl *)
example_tag  ::= "<n.example:" example_list ">"
example_list ::= STRING { "||" STRING }

legacy_statement ::= pattern_decl | bind_decl
pattern_decl     ::= "pattern" RULE_ID ":" pattern_body
bind_decl        ::= "bind" RULE_ID ":" { bind_line }

pattern_body ::= slot { slot }
slot         ::= variable | constant | function | choice
variable     ::= "<n.variable:" VAR_ID ">"
constant     ::= "<n.constant:" STRING ">"
function     ::= "<n.function:" STRING ">"
choice       ::= "<" slot ("||" slot)+ ">"
bind_line    ::= variable "<n.function:" "=" ">" label
label        ::= "<n.label:" STRING ">"
RULE_ID      ::= identifier
VAR_ID       ::= identifier
STRING       ::= quoted string
```

**Tag NVL — bảng construct:**

| Construct | Ý nghĩa | → AST |
| :--- | :--- | :--- |
| `<n.start>` … `<n.end>` | Một lesson block | toàn bộ Lesson AST |
| `<n.pattern:RULE_ID>` | Tên pattern | `pattern_id` |
| `<n.variable:S>` | Slot biến | `{ "var": "S" }` |
| `<n.constant:"là">` | Token cố định | `{ "const": "là" }` |
| `<n.function:"s+">` | Flex gap | `{ "gap": "s+", "min": 1 }` |
| `<n.variable:N \|\| n.variable:A>` | OR | `{ "one_of": ["N", "A"] }` |
| `<n.bind>` + `<n.function:"=">` | Gán nhãn | `bind` map |
| `<n.example:"…" \|\| "…">` | Ví dụ (một hoặc nhiều dòng tag) | `examples[]` |

*Nhiều dòng* `<n.example:"câu">` trong cùng block ≡ một `examples:` trong YAML/JSON.

### A.3. Định dạng 1 — Tag NVL (`.nvl`)

```nvl
<n.start>
<n.pattern:vi_copula_S_la_NA>
  <n.variable:S> <n.function:"s+"> <n.constant:"là"> <n.function:"s+"> <n.variable:N || n.variable:A>

<n.bind>
  <n.variable:S> <n.function:"="> <n.label:"chủ ngữ">
  <n.variable:N> <n.function:"="> <n.label:"danh từ">
  <n.variable:A> <n.function:"="> <n.label:"tính từ">

<n.example: "mẹ tôi là giáo viên" || "anh ấy là học sinh" || "anh ta là công nhân" || "cái bút là màu đỏ">
<n.end>
```

**Cú pháp legacy (tương đương)** — vẫn hợp lệ cho script/JSON ngắn:

```nvl
pattern vi_copula_S_la_NA:
  <n.variable:S> <n.function:"s+"> <n.constant:"là"> <n.function:"s+"> <n.variable:N || n.variable:A>

bind vi_copula_S_la_NA:
  <n.variable:S> <n.function:"="> <n.label:"chủ ngữ">
  <n.variable:N> <n.function:"="> <n.label:"danh từ">
  <n.variable:A> <n.function:"="> <n.label:"tính từ">
```

*Chi tiết compile → Mục A.8. Bảng align ví dụ copula:*

| Ví dụ | S | `là` | Vị trí 3 (N hoặc A) |
| :--- | :--- | :--- | :--- |
| *"mẹ tôi là giáo viên"* | `mẹ tôi` | `là` | `giáo viên` → N |
| *"cái bút là màu đỏ"* | `cái bút` | `là` | `màu đỏ` → A |

### A.4. Định dạng 2 — YAML structured (`.lesson.yaml`)

Cùng lesson `vi_copula_S_la_NA`:

```yaml
lesson_id: vi_grade3_copula
grade: 3
repeat: 3
eta_teach: 0.5
w_init: 2.0

pattern_id: vi_copula_S_la_NA
slots:
  - var: S
  - gap: s+
    min: 1
  - const: "là"
  - gap: s+
    min: 1
  - one_of: [N, A]

bind:
  S: chủ ngữ
  N: danh từ
  A: tính từ

examples:
  - "mẹ tôi là giáo viên"
  - "anh ấy là học sinh"
  - "anh ta là công nhân"
  - "cái bút là màu đỏ"
```

**Loader:** `yaml.safe_load` → map sang Lesson AST → compiler (giống tag NVL).

### A.5. Định dạng 3 — JSON structured (`.lesson.json`)

Cùng nội dung — tốt cho API và JSON Schema:

```json
{
  "lesson_id": "vi_grade3_copula",
  "grade": 3,
  "repeat": 3,
  "eta_teach": 0.5,
  "w_init": 2.0,
  "pattern_id": "vi_copula_S_la_NA",
  "slots": [
    { "var": "S" },
    { "gap": "s+", "min": 1 },
    { "const": "là" },
    { "gap": "s+", "min": 1 },
    { "one_of": ["N", "A"] }
  ],
  "bind": {
    "S": "chủ ngữ",
    "N": "danh từ",
    "A": "tính từ"
  },
  "examples": [
    "mẹ tôi là giáo viên",
    "anh ấy là học sinh",
    "anh ta là công nhân",
    "cái bút là màu đỏ"
  ]
}
```

**JSON Schema (tóm tắt):** `lesson_id`, `pattern_id`, `slots`, `examples` bắt buộc; mỗi phần tử `slots` đúng **một** trong `var` | `const` | `gap` | `one_of`.

**JSON nhúng tag NVL (tương thích ngược):** nếu file có field `"nvl": "<n.start>…"` thay vì `slots`/`examples`, parser tag chạy trước — kết quả AST giống structured. **Không** trộn `slots` và `nvl` trong cùng lesson (lỗi validate).

```json
{
  "lesson_id": "vi_grade3_copula",
  "grade": 3,
  "repeat": 3,
  "eta_teach": 0.5,
  "w_init": 2.0,
  "nvl": "<n.start>\n<n.pattern:vi_copula_S_la_NA>\n..."
}
```

### A.6. Transpile & chọn định dạng

| Tác vụ | Gợi ý |
| :--- | :--- |
| Soạn tay / giáo viên | `.nvl` |
| Repo curriculum, nhiều lesson | `.lesson.yaml` |
| CI, API upload, validate | `.lesson.json` |
| `nvl → yaml` | Tag parser → AST → dump YAML |
| `yaml → json` | Load YAML → dump JSON (AST giữ nguyên) |

**CLI (spec, Phase 1):**

```bash
# Một trong ba đầu vào — cùng imprint
aivie teach lessons/vi_copula.nvl
aivie teach lessons/vi_copula.lesson.yaml
aivie teach lessons/vi_copula.lesson.json

# Transpile
aivie nvl convert vi_copula.nvl -o vi_copula.lesson.yaml
aivie nvl convert vi_copula.lesson.yaml -o vi_copula.lesson.json
```

### A.7. Lesson pack bổ sung (cả 3 định dạng)

**`vi_AN` — Tính từ + danh từ (lớp 1–2):**

```nvl
<n.start>
<n.pattern:vi_AN>
  <n.variable:A> <n.function:"s+"> <n.variable:N>
<n.bind>
  <n.variable:A> <n.function:"="> <n.label:"tính từ">
  <n.variable:N> <n.function:"="> <n.label:"danh từ">
<n.example: "đẹp quá" || "xanh lá" || "cao to">
<n.end>
```

**`vi_SV` — Chủ ngữ + vị ngữ (lớp 1–2):**

```nvl
<n.start>
<n.pattern:vi_SV>
  <n.variable:S> <n.function:"s+"> <n.variable:V>
<n.bind>
  <n.variable:S> <n.function:"="> <n.label:"chủ ngữ">
  <n.variable:V> <n.function:"="> <n.label:"động từ">
<n.example: "Mèo ngủ" || "Bé khóc" || "Chó sủa">
<n.end>
```

**YAML tương đương (`vi_SV.lesson.yaml`):**

```yaml
lesson_id: vi_SV
pattern_id: vi_SV
slots:
  - var: S
  - gap: s+
  - var: V
bind:
  S: chủ ngữ
  V: động từ
examples:
  - "Mèo ngủ"
  - "Bé khóc"
  - "Chó sủa"
```

### A.8. Semantics compiler (chung cho 3 định dạng)

**Pattern + bind** (từ `slots` / tag body):

* Tạo `Rule:{pattern_id}`.
* Mỗi `gap` → wildcard $R_{-k}$; imprint $A[R_{-k}, \text{Rule}]$ với $w_{\text{init}}$.
* `bind` → $A[\text{Var}, \text{Lbl}]$.

**Examples** (từ `examples[]` / `<n.example>`):

1. Tokenize từng câu.
2. Align token → Var theo `slots`.
3. Imprint $A[\text{token}, \text{Var}]$ với $\eta_{\text{teach}}$.
4. Co-fire pattern → củng cố Rule.

*Ví dụ lỗi align → báo lỗi lesson, không imprint mù.*

### A.9. Ánh xạ NVL → HCA (Dual Context)

Khi infer câu tự nhiên *"An là học sinh"* với rule `vi_copula_S_la_NA` đã imprint:

1. **Matcher (runtime):** Căn token thực vào slot — `An`→S, `là`→constant, `học sinh`→N.
2. **$C_{\text{frame}}$:** Rule + $R_{-k}$ boost score token hợp khung (sau S kỳ vọng `là`; sau `là` kỳ vọng N hoặc A). Rule nodes tham gia như node trong $V(w_{-k})$.
3. **$C_{\text{bind}}$:** Nội dung token thực (`An`, `học sinh`) — bổ sung, không thay rule.

**Matcher không phải if/else cứng** — chỉ tăng $\text{score}[v]$ qua $A$ và Rule đã học; mọi $w_{-k}$ trong history vẫn đóng góp với trọng số khác nhau (Mục III.8).

| Vị trí $w_0$ | Frame (rule) | Bind (token thực) |
| :--- | :--- | :--- |
| Sau `An` | Kỳ vọng `là` | — |
| Sau `là` | Kỳ vọng N hoặc A | — |
| Đang sinh `học sinh` | Slot N active | Nội dung từ `học`, `sinh` |

**Imprint token thực ↔ Var (sau G1/G3):** Khi corpus có *"An là học sinh"*, dopamine củng cố $A[\text{An}, \text{Var:S}]$, $A[\text{học sinh}, \text{Var:N}]$ — gắn từ vựng mở vào slot rule.

#### Học Hebbian sau lesson NVL

| Kết quả | Củng cố |
| :--- | :--- |
| Sau S → đúng `là` | $A[\text{Var:S}, \text{là}]$, $A[R_{-1}, \text{là}]$ qua rule |
| Sau `là` → đúng N/A | $A[\text{là}, \text{Var:N}]$ / $A[\text{là}, \text{Var:A}]$ |
| Rule khớp nhưng token sai ngữ pháp thực tế | LTD (G3); corpus thắng |

---

## V. CÁC ĐẠI TỪ VÀ KHÁI NIỆM TRỪU TƯỢNG ĐỘNG

### 1. Đại từ nhân xưng (Tôi / Bạn)
Mạng được trang bị hai nơ-ron điều hướng góc nhìn đặc biệt:
* **Node [Self] (Bản thân AI)** và **Node [Other] (Đối tác)**.
* Khi trạng thái đổi từ "Nghe" sang "Nói", dòng tín hiệu buộc phải chạy qua hai node này để đảo ngược bản đồ liên kết tương ứng, giúp AI không bị nhại lại đại từ của người dùng.

### 2. Cấu trúc Ngữ pháp và Nơ-ron vị trí tương đối

Ngữ pháp hình thành qua **hai nguồn song song** — không lập trình cứng bằng parser, nhưng có thể **dạy trực tiếp** (NVL, bất kỳ lúc train) hoặc **tự nổi** (G1–G4):

| Nguồn | Cơ chế | Tốc độ | Khi dùng |
| :--- | :--- | :--- | :--- |
| **Instructed** | NVL imprint → Rule / Var / Label nodes → $A$ | Nhanh; vài lượt `repeat` | Bất kỳ lúc nào (Direct Teaching mode) |
| **Emergent** | Composite + $R_{-k}$ từ thống kê corpus | Chậm; cần nhiều data | G1–G4 |

Cả hai cùng tồn tại trong $A$. Xung đột được xử lý ở **G3 (dopamine)**: corpus thực tế thắng nếu rule NVL sai hoặc quá hẹp.

Các **Nơ-ron vị trí tương đối ($R_{-1}, R_{-2}, \ldots, R_{-K_{\max}}$)** — tương ứng Cửa sổ ngữ cảnh (Mục III.1) và vai trò **Key** trong HCA (Mục III.3) — dùng chung cho cả hai nguồn. NVL imprint liên kết $A[R_{-k}, \text{Rule}]$; G1–G4 củng cố $A[R_{-k}, \text{token}]$ từ thống kê.

Thay vì gán mỗi token vào slot tuyệt đối ($P_1, P_2, P_3\ldots$), mạng học quy luật *"từ loại nào thường xuất hiện cách tôi bao nhiêu bước"*. Qua NVL (rule) và/hoặc hàng triệu lần co-occurrence (G1+), các nhóm từ (Danh từ, Động từ, Tính từ…) liên kết với $R_{-k}$ và Rule nodes, tạo "lòng sông" nắn dòng token theo trật tự ngữ pháp.

**Ví dụ:** Khi dự đoán token tại vị trí $0$, $R_{-1}$ mang thông tin loại từ vừa xuất hiện; $R_{-2}$ góp cấu trúc cụm đang hình thành. Scope Neurons (Mục III.4) quyết định mở rộng đến $R_{-k}$ nào. Rule NVL (ví dụ `vi_copula_S_la_NA`) boost $C_{\text{frame}}$ tại các slot đã dạy.

---

## VI. ĐÁNH GIÁ KHẢ THI

### Mức khả thi theo thành phần

| Thành phần | Khả thi | Lý do | Rủi ro |
| :--- | :--- | :--- | :--- |
| **Ma trận hóa Hebbian ($H^T H$) trên GPU** | **Cao** | Phép nhân sparse/dense batch đã được chứng minh; tương tự co-occurrence matrix trong NLP cổ điển | Ma trận $A$ size $(N,N)$ với $N$ lớn tốn RAM — cần phân tầng / block sparse |
| **HCA — Dual Context Readout** | **Cao** | Hai kênh $C_{\text{frame}}$/$C_{\text{bind}}$ trên cùng lịch sử; pseudocode III.2–III.3 | Cần tune $\lambda_{\text{frame}}$ vs $\lambda_{\text{bind}}$; ablation trên câu template + binding |
| **HCA — Context Read banded $O(T \cdot K)$** | **Cao** | Cùng pattern sliding window như nanochat FA3/SDPA; triển khai trực tiếp bằng PyTorch | Cần benchmark so với baseline Transformer cùng size |
| **Chuyển ngữ Q/K/V (không $c_q,c_k,c_v$)** | **Cao** | Vai trò tách biệt rõ; không cần backprop qua attention | Biểu diễn $H$ sparse có thể kém expressiveness hơn dense embedding — cần Composite nodes bù |
| **Decay buffer $D$** | **Cao** | O(1) cập nhật mỗi token; tương đương KV cache đơn giản | Chọn $\lambda$ ảnh hưởng coreference xa |
| **$K_{\max}$ theo lớp (8→64 token)** | **Cao** | Hyperparameter curriculum; chi phí tuyến tính theo $K$ | Không cần K riêng từng ngôn ngữ ở giai đoạn đầu |
| **Scope Neurons — K động** | **Trung bình–Cao** | Khớp triết lý sinh học; gate sparse trên GPU khả thi | Cần Giai đoạn 3 (dopamine) để học mở/đóng K; cold-start có thể mở quá rộng hoặc quá hẹp |
| **Retroactive Wave** | **Trung bình** | Ý tưởng rõ, ví dụ đại từ thuyết phục | Khó hơn forward-only; cần định nghĩa rõ *khi nào* trigger (entropy, `[Ambiguity]`) và tránh loop không ổn định |
| **Composite / Structural nodes tự sinh** | **Trung bình** | Hebbian tự tạo bigram nodes qua $H^TH$ | Bùng nổ tổ hợp nếu không giới hạn — cần pruning / ngưỡng kích hoạt |
| **Direct Teaching + NVL (3 định dạng)** | **Cao** (spec) | `.nvl` / `.lesson.yaml` / `.lesson.json` → AST → imprint; không backprop | Cần 3 parser + transpile; JSON Schema; generalize ngoài lesson cần G1+G3 |
| **12 lớp + dataset chuyên biệt** | **Cao** (quy trình) | Curriculum learning đã được chứng minh (GPT, nanochat pretrain→SFT) | Cần xây dataset và metric thăng lớp — công sức dữ liệu, không phải blocker kỹ thuật |
| **Thay thế hoàn toàn Transformer SOTA** | **Thấp–Trung bình** (dài hạn) | Hướng nghiên cứu mới; chưa có POC công khai | Cần prototype + ablation từng thành phần trước khi kỳ vọng GPT-2 grade |

### Lộ trình triển khai đề xuất

```
Phase 0 — POC tối thiểu (khả thi cao, ~vài tuần)
  ├── Token nodes + A sparse + ΔA = η·HᵀH
  ├── HCA Dual Context (C_frame + C_bind) với K_max cố định (16)
  ├── Autoregressive generate + perplexity cơ bản
  ├── Ablation: câu template (là→con) + binding (ụt ịt→lợn)
  ├── Spec NVL: 3 định dạng lesson (.nvl / .yaml / .json) + 1 lesson vi_copula
  ├── [✅ 2026-06] EdgeKind + TypedAdjacency (grammar / collocation / co_activation / …)
  ├── [✅ 2026-06] Rule-scoped colocation teach + matcher gating
  ├── [✅ 2026-06] Graph UI (Explorer + Inference Lab) + export subgraph
  ├── [✅ 2026-06] QA context grounding + answer-zone isolation (`hca/qa_context.py`)
  ├── [✅ 2026-06] Neuron-first QA chain (teach compile + G2 rule colocation + graph readout)
  └── [⏳] Directed colocation, slot close, rule arbitration nâng cao → xem **B.9**

Phase 1 — Curriculum (khả thi cao)
  ├── G1–G4 × lớp 1–4
  ├── Direct Teaching: loader 3 định dạng + `aivie nvl convert`
  ├── NVL lesson xen kẽ khi cần (không cố định đầu lớp)
  ├── K_max tăng dần 8 → 16
  ├── Dopamine R matrix (Giai đoạn 3)
  ├── Validator matcher: rule retention + generalize trên corpus
  └── [✅ 2026-06] Full train grade 3–7 local; curriculum `train_curriculum.sh`

Phase 2 — K động & đa ngôn ngữ (khả thi trung bình)
  ├── Scope Neurons + G_{-k} gates
  ├── K_max → 32–64, đa ngôn ngữ tokenized
  └── Retroactive Wave + [Ambiguity]

Phase 3 — Scale & chuyên ngành (dài hạn)
  ├── Composite / Structural nodes tự sinh + pruning
  ├── Dataset chuyên biệt (Toán, Hóa…)
  └── So sánh benchmark với nanochat cùng quy mô
```

### Những gì **không** nên làm ở giai đoạn đầu

* Không loại trừ token quá khứ khỏi ngữ cảnh — mọi $w_{-k}$ đều đóng góp; chỉ khác trọng số qua $\alpha^{\text{frame}}$ / $\alpha^{\text{bind}}$.
* Không nhân bản $c_q, c_k, c_v$ + backprop — phá triết lý Hebbian.
* Không cố định $K = 5$ cho toàn bộ 12 lớp — quá hẹp cho đa ngôn ngữ.
* Không bật Retroactive Wave trước khi forward HCA ổn định.
* Không allocate $A$ full dense $(N,N)$ với $N = 10^6$ — dùng block sparse hoặc phân tầng ngay từ POC.
* **Direct Teaching:** Không biến NVL thành if/else cứng lúc infer — matcher chỉ **boost** score qua $A$ / Rule nodes (Luật kiến trúc Mục 0).
* **Direct Teaching:** Không thay thế G1–G4 — NVL không đủ học từ vựng mở và ngoại lệ ngôn ngữ.
* **Direct Teaching:** Không imprint với $\eta_{\text{teach}}$ vô hạn hoặc `repeat` quá lớn — tránh rule cứng không sửa được; dopamine/LTD (G3) vẫn áp dụng.

---

## Phụ lục B: Triển khai POC & vận hành (`nn-aivie`)

Phần này ghi **những gì đã làm trong repo**, cùng quy trình chạy **train**, **inference** và **Graph UI** (TK_001). Kiến trúc lý thuyết xem Mục I–VI; đây là hướng dẫn thực tế trên code hiện tại.

### B.1. Trạng thái triển khai (Phase 0 → 1 + v3)

| Hạng mục | Trạng thái | Ghi chú |
| :--- | :--- | :--- |
| **v3 — Nơ-ron cha + hybrid generate** | ✅ (lịch sử) | `train_v3.py`, `parent_store.py` — sinh **2026-07-09** chỉ `generate(multi=True)` |
| **v3 — Matching phân tầng** | ✅ | `query_match_units`, `rebuild_hierarchical_index`, `rank_matches` (0.5.12) |
| **v3 — Embedding E + UI** | ✅ | PPMI+SVD; tab Embedding E; `/api/embedding/*` |
| Lesson NVL / YAML / JSON + AST | ✅ | `lessons/`, `aivie teach`, `aivie teach-pack` — **đường curriculum cũ** |
| Imprint → `GraphState` (vocab + sparse $A$) | ✅ | `checkpoints/grade_03.pkl` … `grade_07.pkl` |
| **EdgeKind + TypedAdjacency** | ✅ | `grammar`, `anchor`, `secondary`, `position`, `collocation`, `co_activation`, `reinforced` |
| **Matcher HCA** (rule-scoped colocation) | ✅ | Đường cũ — Inference Lab trace |
| **QA context grounding** | ✅ | `hca/qa_context.py` — đường HCA |
| G1 passive / G2 imitation | ✅ | `run_train_eval.py`, `train_curriculum.sh` |
| G3 dopamine | ✅ (tuỳ chọn) | `--g3`; mặc định bỏ qua |
| **HF ingest + quality gate** | ✅ | `ingest_hf_vi.py`, `data/quality.py`, `train_v3 --extra-dir` |
| **Graph UI (TK_001)** | ✅ | 3 tab: Parent Explorer, Inference Lab, **Embedding E** |
| G4 self-talk, GPU batch | ⏳ | Roadmap |

**Checkpoint mặc định (2026-07-09):** `checkpoints/network_v10_histwiki.pkl` (`default_checkpoint`).  
**Checkpoint Graph UI + discourse:** `network_v10_histwiki_paragraph.pkl` + `.varlm.pkl` (`viz_default_checkpoint`).  
**Checkpoint v3 (lịch sử / guard nhỏ):** `grade_v3.pkl`, `grade_v3_big_a2.pkl` — xem §0.5.x.  
**Checkpoint curriculum cũ:** `grade_03.pkl` … `grade_07.pkl` — xem **B.4**.

**Khi nào cần train lại?** Xem **B.4** (curriculum) và **B.4.1** (train_v3). Sửa `parent_store` matching / `export_graph_v3` → rebuild parents trên checkpoint hiện có (không cần full G1/G2 nếu chỉ đổi logic cha).

### B.2. Cài đặt

```bash
cd nn-aivie
pip install -e ".[dev]"          # pytest, datasets
pip install -e ".[viz]"          # Graph UI (FastAPI)
cd viz/web && npm install        # frontend React (chỉ cần khi sửa/build UI)
```

Yêu cầu: Python ≥ 3.10. Checkpoint mặc định: `checkpoints/network_v10_histwiki.pkl` (train bằng
`scripts/build_histwiki_incremental.py` + `build_paragraph_pool.py`). Curriculum cũ: `grade_03.pkl` …

### B.3. Cấu trúc repo (tóm tắt)

| Path | Vai trò |
| :--- | :--- |
| `lessons/` | NVL / YAML / JSON + `grade_XX_pack.yaml` (curriculum cũ) |
| `data/grade_XX/` | JSONL → parquet (`manifest.yaml`) |
| `data/real_vi/` | HF ingest + **`history_wiki_para.jsonl`** (TK_024/035) |
| `checkpoints/` | **v10** `network_v10_histwiki*.pkl` + `.varlm.pkl`; lịch sử `grade_v3*.pkl` |
| `src/nn_aivie/core/` | `parent_store.py`, `embedding.py`, `varlm.py`, `state.py` |
| `src/nn_aivie/teaching/` | Parse lesson, compile, imprint |
| `src/nn_aivie/training/` | G1/G2/G3, `ingest.py` |
| `src/nn_aivie/hca/` | `generate_hebbian.py` (`generate` → A3), **`discourse.py`**, matcher, qa_context |
| `src/nn_aivie/viz/` | `export_graph_v3.py`, `export_embedding.py`, `trace.py` |
| `viz/web/` | React UI (3 tab) |
| `viz/server/` | FastAPI |
| `scripts/` | `train.py`, `normalize_corpus.py`, `eval_discourse_prompts.py`, `bake_checkpoint_indices.py`, `run_viz.py` |

### B.4. Quy trình training

> **Tài liệu đầy đủ (dataset, 15 bước, script, resume):** [`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md)

#### B.4.0. Train (một script — text jsonl)

```bash
cd nn-aivie
# Chuẩn hóa corpus (mỗi dòng = một JSON string)
PYTHONPATH=src python scripts/normalize_corpus.py

# Train full stack v10
PYTHONPATH=src python scripts/train.py \
  --manifest config/train_v10.yaml \
  --out checkpoints/network_v10.pkl \
  --bake

# Train tiếp thêm dataset
PYTHONPATH=src python scripts/train.py \
  --checkpoint checkpoints/network_v10.pkl \
  --out checkpoints/network_v10.pkl \
  --dataset data/text_corpus/histwiki_para.jsonl \
  --priority 0.85
```

Chi tiết schema + manifest: [`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md).

#### B.4.1. Train v10 (khuyên dùng — history wiki + paragraph units)

Pipeline **mặc định** cho stack v10 (TK_024 C + TK_035):

```bash
cd nn-aivie

# Fetch + ingest history wiki (nếu chưa có data/real_vi/history_wiki_para.jsonl)
python scripts/fetch_history_wiki.py

# Build checkpoint v10 + paragraph parents
python scripts/build_histwiki_incremental.py --out checkpoints/network_v10_histwiki.pkl
python scripts/build_paragraph_pool.py \
  --base checkpoints/network_v10_histwiki.pkl \
  --out checkpoints/network_v10_histwiki_paragraph.pkl

# VarLM cluster-aware sidecar
python scripts/build_varlm.py --use-clusters \
  --checkpoint checkpoints/network_v10_histwiki_paragraph.pkl

# Discourse eval (prompt corpus-derived)
PYTHONPATH=src python scripts/eval_discourse_prompts.py \
  --checkpoint checkpoints/network_v10_histwiki_paragraph.pkl

pytest -q   # 131 pass
```

**Train v3 (lịch sử — curated lớp 3–7 + HF):**

Pipeline cho nơ-ron cha + embedding E (không còn `generate_hybrid`):

```bash
cd nn-aivie

# Full curated lớp 3–7 → grade_v3.pkl
python scripts/train_v3.py --out checkpoints/grade_v3.pkl

# + dữ liệu HF đã ingest (+ anaphora A2)
python scripts/ingest_hf_vi.py --source wiki --max 42000          # → statements.jsonl
python scripts/ingest_hf_vi.py --source sentences --anaphora-only \
  --max 2000 --out-name anaphora.jsonl                             # → anaphora.jsonl
python scripts/train_v3.py --extra-dir data/real_vi --out checkpoints/grade_v3_big_a2.pkl

# Eval (giữ bản nhỏ cho benchmark 100%)
python scripts/eval_suite.py --checkpoint checkpoints/grade_v3.pkl
pytest -q
```

**Trong `train_v3.py`:** G1 → G2 → consolidate → **build parents** (ISG online tại `add_example`)
→ **`--extra-dir`** → **ISG** (mặc định; `--legacy-repair` = Re-Pair batch) → **slot hubs**
→ **build embedding E**.

**Chỉ rebuild phân rã (sau sửa `parent_store` / TK_013, không đổi G1/G2):**

```bash
python scripts/rebuild_seg_graph.py \
  --checkpoint checkpoints/grade_v3_big_a2_t3.pkl \
  --out checkpoints/grade_v3_big_a2_t3_isg.pkl \
  --log-file out/rebuild_isg.log
```

Hoặc load checkpoint + `rebuild_seg_graph(vocab)` trong Python. Không cần full train nếu
adjacency không đổi. Re-Pair cũ: thêm `--legacy-repair`.

#### Trình tự làm việc (curriculum cũ + Graph UI)

Khi cần **train lại từ đầu** hoặc lần đầu setup:

```
1. Chuẩn bị dữ liệu     augment_grade03 (lớp 3) + build_grade_curriculum (4–7)
                        → nn_aivie.data grade --grade X   (jsonl → parquet)

2. Train                ./scripts/train_curriculum.sh     (lớp 3→7, ~5 phút/lớp)
   hoặc một lớp         python scripts/run_train_eval.py --grade N --full \
                           --out checkpoints/grade_NN.pkl

   Trong mỗi lớp:       Teach pack → G1 → G2 → eval/smoke
                        (G3 bỏ qua; thêm --g3 nếu cần dopamine)

3. Kiểm tra nhanh       pytest -q
                        python scripts/prompt_one.py "Hỏi: Em học gì ? Đáp:"

4. Graph UI             pip install -e ".[viz]"
                        python scripts/run_viz.py
                        → http://127.0.0.1:8765  (mặc định network_v10_histwiki_paragraph.pkl)

   Checkpoint khác:     AIVIE_CHECKPOINT=checkpoints/network_v10_histwiki.pkl python scripts/run_viz.py
```

**Một lệnh train toàn curriculum** (sau bước 1):

```bash
chmod +x scripts/train_curriculum.sh
./scripts/train_curriculum.sh          # lớp 3 → 7
./scripts/train_curriculum.sh --from 4 # chỉ lớp 4 → 7
./scripts/train_curriculum.sh --grade 5
./scripts/train_curriculum.sh --g3     # bật G3 (chậm, tuỳ chọn)
```

Log: `/tmp/nn_train_curriculum.log`, `/tmp/nn_train_gradeXX.log`.

#### Khái niệm nhanh

| Thuật ngữ | Ý nghĩa |
| :--- | :--- |
| **Lớp (grade 3–7)** | Curriculum nội dung — lesson pack + corpus theo độ khó |
| **G1–G4** | Giai đoạn học — passive → imitation → dopamine → self-talk |
| **`--full`** | Train trên toàn bộ parquet local: curated + compound + copula + general + qa |

Pipeline chuẩn (mỗi lớp):

```
Teach pack → G1 (corpus) → G2 (QA/hội thoại) → eval + generate smoke
```

**G3 dopamine** (thưởng/LTD từng token — chậm ~30–60 phút) **bỏ qua mặc định**; bật bằng `--g3` khi cần tinh chỉnh trọng số cạnh sau matcher ổn.

Checkpoint mỗi lớp: `checkpoints/grade_XX.pkl` (ví dụ `grade_03.pkl`, `grade_07.pkl`).

#### Bảng curriculum hiện có (lớp 3–7)

| Lớp | Lessons | Corpus `--full` | Lesson mới (so với lớp trước) |
| :---: | :---: | :---: | :--- |
| **3** | 18 | **892** câu | Copula, S-V, câu ghép, QA (110 ví dụ) |
| **4** | 23 | 314 câu | So sánh (`…hơn…`, `…nhất`), bị/được, khi…, để… |
| **5** | 28 | 371 câu | Nếu…thì, mặc dù…vẫn… |
| **6** | 32 | 425 câu | Vừa…vừa…, mà, vì…nên…, càng…càng…, hay… |
| **7** | 38 | 781 câu | Tuy…nhưng, bằng, mà quan hệ, Trong…, Do… + QA |

Corpus lớp 3 lớn nhất vì `general/` đã merge HF (`augment_grade03_data.py`). Lớp 7 có thêm `qa/` (copy từ lớp 3).

#### QA context grounding & neuron-first chain

Spec chi tiết: `docs/QA_CONTEXT_SPEC.md`.

**Kiến trúc 3 lớp (runtime):**

1. **QA frame** — `Hỏi:` / `Đáp:` (train mask, zone isolation)
2. **QASession** — `QFocus:{q_signature}`, modality, anchor, post_ids
3. **Graph readout** — `rule_colocation` + `REINFORCED`; rule QA resolve từ đồ thị (`resolve_qa_rule_id`)

**Học trên đồ thị (ưu tiên so với heuristic runtime):**

| Lớp | Module | Nội dung |
| :--- | :--- | :--- |
| **Teach** | `teaching/compile.py` | Fingerprint `QFocus:Chó_màu_gì`; anchor→A chain; entity match |
| **G2** | `training/g2_imitation.py` | `add_rule_collocation` trên gold QA path |
| **Matcher** | `hca/qa_context.py`, `matcher.py` | Đọc cạnh; suppress Var:A hub; fingerprint gate |

**Smoke QA sau train (teach → G1 → G2):**

| Prompt | Kỳ vọng hiện tại |
| :--- | :--- |
| `Hỏi: 2 cộng 3 bằng mấy ? Đáp:` | Cấu trúc toán học — kỳ vọng `cộng 3 bằng 5.` |
| `Hỏi: Điện Biên Phủ thắng lợi năm nào ? Đáp:` | Lấy đúng fact lớp 4 — kỳ vọng `năm 1954.` |
| `Hỏi: Kinh đô nhà Nguyễn ở đâu ? Đáp:` | Lấy đúng fact lớp 5 — kỳ vọng bắt đầu `Huế là …` |
| `Hỏi: Em học gì ? Đáp:` | Bắt đầu `Em …` (cross-example bleed `Em có …` — cần G3 hoặc Q fingerprint) |
| `con chó` | Kiểm tra không kích hoạt copula compound sai |

**Chỉ sửa matcher / qa_context (zone, anchor):** không train lại.  
**Sửa teach compile hoặc G2 QA bind:** `teach-pack` + G1 + G2.  
**Thêm lesson QA examples:** teach (+ G2 tuỳ chọn).

```bash
python scripts/prompt_one.py "Hỏi: Mẹ làm gì ? Đáp:"
pytest tests/test_qa_context.py tests/test_qa_neuron.py -q
```

#### Train một lớp (khuyên dùng)

```bash
cd nn-aivie

# Lớp 3 — full local (892 câu, vài phút — không G3)
python scripts/run_train_eval.py --grade 3 --full --out checkpoints/grade_03.pkl

# Bật G3 nếu cần (chậm)
python scripts/run_train_eval.py --grade 3 --full --g3 --out checkpoints/grade_03.pkl

# Lớp 4–7 — tương tự
python scripts/run_train_eval.py --grade 4 --full --out checkpoints/grade_04.pkl
python scripts/run_train_eval.py --grade 5 --full --out checkpoints/grade_05.pkl
python scripts/run_train_eval.py --grade 6 --full --out checkpoints/grade_06.pkl
python scripts/run_train_eval.py --grade 7 --full --out checkpoints/grade_07.pkl
```

Script trên chạy **từ đầu**: teach pack → G1 → G2 → eval (G3 chỉ khi thêm `--g3`).

#### Train tuần tự nhiều lớp

```bash
chmod +x scripts/train_curriculum.sh

# Lớp 3 → 7 (xóa checkpoint cũ, train lại từ đầu mỗi lớp)
./scripts/train_curriculum.sh

# Chỉ lớp 4 → 7 (tự chờ nếu lớp 3 đang chạy)
./scripts/train_curriculum.sh --from 4

# Một lớp
./scripts/train_curriculum.sh --grade 5
```

Log mỗi lớp: `/tmp/nn_train_gradeXX.log`.

#### Chuẩn bị dữ liệu (chạy trước khi train lần đầu)

**Lớp 3 — merge HuggingFace vào general (một lần):**

```bash
python scripts/augment_grade03_data.py   # HF → general/ + copula/ jsonl
python -m nn_aivie.data grade --grade 3  # jsonl → parquet
python scripts/build_qa_lesson.py        # (tuỳ chọn) cập nhật vi_QA_hoi_dap từ qa/train.jsonl
```

**Lớp 4–7 — sinh corpus curated/compound/copula từ lesson pack:**

```bash
python scripts/build_grade_curriculum.py --grade 4 5 6 7
python -m nn_aivie.data grade --grade 4
python -m nn_aivie.data grade --grade 5
python -m nn_aivie.data grade --grade 6
python -m nn_aivie.data grade --grade 7
```

Sau khi sửa `data/grade_XX/*/train.jsonl`, luôn chạy lại `nn_aivie.data grade --grade X` trước train.

#### Train từng bước (CLI thủ công)

Dùng khi chỉ cần chạy lại một phase:

**Bước 1 — Teach (Direct Teaching)**

```bash
python -m nn_aivie.cli teach-pack lessons/grade_03_pack.yaml \
  -o checkpoints/grade_03.pkl
```

**Bước 2 — G1 Passive**

```bash
python -m nn_aivie.cli train --grade 3 --phase G1 \
  --checkpoint checkpoints/grade_03.pkl -o checkpoints/grade_03.pkl \
  --dataset curated_simple --dataset compound_corpus --dataset copula_corpus \
  --dataset general --dataset qa_corpus
```

**Bước 3 — G2 Imitation (QA)**

```bash
python -m nn_aivie.cli train --grade 3 --phase G2 \
  --checkpoint checkpoints/grade_03.pkl -o checkpoints/grade_03.pkl \
  --dataset qa_corpus --dataset curated_simple --dataset compound_corpus
```

**Bước 4 — G3 Dopamine (tuỳ chọn, `--g3`)**

```bash
python -m nn_aivie.cli train --grade 3 --phase G3 \
  --checkpoint checkpoints/grade_03.pkl -o checkpoints/grade_03.pkl \
  --dataset curated_simple --dataset compound_corpus --dataset copula_corpus \
  --dataset general --dataset qa_corpus \
  --alpha 1.5 --beta 1.0
```

Với `qa_corpus`, G3 chỉ cập nhật token **sau** `Đáp:` (`train_mask`). **POC hiện bỏ qua G3** — matcher + teach + G1/G2 đủ cho smoke QA.

#### Biến thể `run_train_eval.py`

```bash
# Mặc định lớp 3, curated + compound + copula (~375 câu)
python scripts/run_train_eval.py --out checkpoints/grade_03.pkl

# + general parquet (HF đã merge local)
python scripts/run_train_eval.py --grade 3 --with-hf --out checkpoints/grade_03.pkl

# Full local — khuyên dùng cho lớp 3–7
python scripts/run_train_eval.py --grade 3 --full --out checkpoints/grade_03.pkl
```

**Dataset trong manifest (`data/grade_XX/manifest.yaml`):** `curated_simple`, `compound_corpus`, `copula_corpus`, `general`, `qa_corpus`. Flag `--full` dùng parquet local; **không** stream Hub (`vi_sentences_hf`).

#### Khi nào cần train lại?

| Thay đổi | Teach | G1 | G2 | G3 |
| :--- | :---: | :---: | :---: | :---: |
| Sửa **matcher / qa_context** (zone, anchor, graph readout) | — | — | — | — |
| Sửa **`compile._add_qa_bindings`** hoặc **G2 QA colocation** | **Có** | **Có** | **Có** | — |
| Sửa **compile/imprint** hoặc lesson examples (khác QA) | **Có** | Tuỳ | Tuỳ | Tuỳ |
| Thêm/sửa **corpus** jsonl/parquet | — | **Có** | **Có** | **Có** |
| Đổi **G1/G2/G3 logic** | — | G1 | G2 | G3 |
| Thêm **lesson mới** vào pack | **Có** | **Có** | Tuỳ | **Có** |

Quy tắc nhanh: **sửa đồ thị teach / QA chain → `teach-pack` + G1 + G2; sửa matcher thuần → reload checkpoint; sửa corpus → G1+G2 (+G3 tuỳ chọn).**

#### Kiểm tra sau train

```bash
pytest -q
python scripts/eval_suite.py --checkpoint checkpoints/grade_v3.pkl
python scripts/prompt_one.py "Hỏi: Em học gì ? Đáp:"
python scripts/run_viz.py   # Graph UI → http://127.0.0.1:8765
```

Kỳ vọng `grade_v3.pkl`: QA val **100%**, paraphrase **100%** (`eval_suite.py`). Curriculum cũ lớp 3 không G3: ~**64k** edges.

### B.5. Quy trình inference (generate)

#### CLI (v10 — mặc định)

```bash
aivie generate --prompt "Hỏi: Ai trồng lúa ? Đáp:" \
  --checkpoint checkpoints/network_v10_histwiki_paragraph.pkl --max-len 32
```

Pipeline: `generate(multi=True)` → **agenda** từ retrieval → `generate_autoregressive` sinh từng token
(paragraph pool + VarLM cluster-aware). Không còn nhánh verbatim hybrid.

#### CLI (curriculum cũ — HCA matcher)

```bash
aivie generate --prompt "An là" --checkpoint checkpoints/grade_03.pkl --max-len 8
```

#### Debug một bước (top-k scores)

```bash
python scripts/prompt_one.py "Hỏi: Em học gì ? Đáp:"
```

Pipeline mỗi bước generate:

1. `dual_context_read` — $C_{\text{frame}} + C_{\text{bind}}$ qua $w_0$
2. `apply_matcher_boost` — rule arbitration, eligible Var, answer-zone sau `Đáp:`
3. `mask_non_surface` → repetition penalty → frame closure → **argmax**

#### QA prompt mẫu

| Prompt | Mục đích |
| :--- | :--- |
| `Hỏi: Mẹ làm gì ? Đáp:` | QA action chain — kỳ vọng `Mẹ nấu cơm.` |
| `Hỏi: Bố đi đâu ? Đáp:` | QA where — kỳ vọng `Bố đi làm.` |
| `Hỏi: mẹ tôi làm nghề gì ? Đáp:` | Nghề nghiệp |
| `Hỏi: Em học gì ? Đáp:` | Câu trả lời ngắn (còn bleed cross-example) |
| `Hỏi: Con chó màu gì ? Đáp:` | QA color — bắt đầu `Con … màu …` |
| `con chó` | Kiểm tra không kích hoạt copula compound sai |

### B.6. Graph UI (TK_001)

Web minh họa **matching nơ-ron cha v3**, **trace HCA từng bước**, và **ma trận embedding E**. Chi tiết spec cũ (rule subgraph): `TK_001_GRAPH_UI.md`.

#### Chạy nhanh (một lệnh)

```bash
pip install -e ".[viz]"
python scripts/run_viz.py
```

Mở **http://127.0.0.1:8765** (tự build React nếu chưa có `viz/web/dist/`).

Mặc định load `checkpoints/network_v10_histwiki_paragraph.pkl` (`viz_default_checkpoint`).
Đổi checkpoint:

```bash
AIVIE_CHECKPOINT=checkpoints/network_v10_histwiki.pkl python scripts/run_viz.py
```

#### Parent Explorer — chế độ sinh (2026-07-09)

Chỉ **A3 multi** — sinh từng token theo agenda (retrieval lập agenda, không dội verbatim).
Trace agenda/steps trong panel completion. Cần checkpoint paragraph + VarLM sidecar cho discourse tốt nhất.

Prompt thử (multi, v10 paragraph):

```text
Hỏi: Huế là gì ? Đáp:
Hỏi: Thủ đô Việt Nam là gì Đáp:
Hỏi: Ai chữa bệnh Đáp:
```

Sau khi sửa `viz/web/src/*.tsx`: `cd viz/web && npm run build` rồi refresh trang.

#### Chạy dev (hot reload)

```bash
# Terminal 1 — API (reload Python)
python -m viz.server.main

# Terminal 2 — Vite (sửa giao diện)
cd viz/web && npm run dev
# → http://127.0.0.1:5173 (proxy /api → :8765)
```

#### Tính năng UI (3 tab)

| Tab | Chức năng |
| :--- | :--- |
| **Parent Explorer** | Prompt `Hỏi: … Đáp:` → chế độ **Hybrid / Compose / A3**; đồ thị token→cụm→cha; panel composition + trace A3. `/api/v3/complete` body: `mode`, `multi`, `max_facts`. |
| **Inference Lab** | Trace autoregressive **đường HCA cũ**: $w_0$, decay, active rules, eligible Var, `qa_context`, bar chart base/matcher/final. |
| **Embedding E** | Meta ma trận **N×d** (PPMI+SVD); tra token → vector + hàng xóm cosine; so sánh cặp; tìm substring trong vocab. *Lưu ý: E theo **token vocab**, không phải cụm.* |

#### API REST

| Endpoint | Mô tả |
| :--- | :--- |
| `GET /api/health` | Checkpoint path, trạng thái UI |
| `POST /api/v3/complete` | Body: `{"prompt":"…","mode":"hybrid\|composed\|autoregressive","multi":false,"max_facts":3}` |
| `GET /api/v3/parents?query=&limit=` | Duyệt nơ-ron cha |
| `GET /api/embedding/info` | Kích thước E, norm, sample tokens |
| `GET /api/embedding/lookup?token=&topk=` | Vector + cosine neighbors |
| `GET /api/embedding/compare?a=&b=` | Cosine hai token |
| `GET /api/embedding/search?q=` | Tìm token theo substring |
| `GET /api/rules` | Danh sách rule (Inference Lab / export cũ) |
| `GET /api/graph?rules=…` | Subgraph rule-based (legacy) |
| `POST /api/trace` | Body: `{"prompt":"…","max_len":8,"top_k":12}` — trace HCA |

Module Python: `export_graph_v3.py`, `export_embedding.py`, `export_graph.py`, `trace.py`.

### B.7. Kiểm thử

```bash
pytest -q
```

Gồm parent_store, export_graph_v3, **discourse/A2**, export_embedding, teach, G1/G2/G3, matcher, generate_hebbian, viz. Hiện **269 passed**, 1 skipped (2026-07-02).

### B.8. Lệnh CLI tóm tắt

```bash
# v3 (mặc định)
python scripts/train_v3.py --out checkpoints/grade_v3.pkl
python scripts/eval_suite.py --checkpoint checkpoints/grade_v3.pkl
aivie generate --prompt "Hỏi: Ai trồng lúa ? Đáp:" --checkpoint checkpoints/grade_v3.pkl
python scripts/run_viz.py

# Curriculum cũ (NVL/teach)
aivie teach-pack lessons/grade_03_pack.yaml -o checkpoints/grade_03.pkl
aivie train --grade 3 --phase G1 --checkpoint checkpoints/grade_03.pkl -o checkpoints/grade_03.pkl --dataset qa_corpus
aivie export-viz --checkpoint checkpoints/grade_03.pkl --rules vi_QA_hoi_dap
aivie convert lessons/vi_copula_S_la_NA.nvl -o lessons/vi_copula_S_la_NA.lesson.yaml
```

### B.9. Kế hoạch Phase 1.5 — Matcher & đồ thị có hướng (chi tiết)

> **Mục tiêu:** Sinh câu đúng khung (vd. `Em rất vui` dừng sau một tính từ; không `đỏ hoa` ngược pattern).  
> **Nguyên tắc:** (1) rule-scoped — biết thuộc rule nào; (2) **directed** — biết thứ tự slot; (3) slot đóng khi đã fill; (4) arbitration — rule neo const thắng rule lỏng.

#### Gói 1 — Matcher + teach compile (**ưu tiên mai**; **không bắt buộc G1/G3** nếu chỉ sửa matcher)

| ID | Hạng mục | Mô tả kỹ thuật | File / module | Test gợi ý | Train lại? |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **A2** | **Directed collocation** | Teach chỉ `span_trước → span_sau` (vd. `hoa→đỏ`, `Em→vui`); **bỏ** chiều ngược. Matcher chỉ boost `s_tok → candidate` khi `s_tok` fill slot trước candidate trong cùng rule. | `teaching/compile.py`, `hca/matcher.py` | `Em rất` → `vui` rồi EOS; `hoa đỏ` forward | **Teach** |
| **B2** | **Đóng slot sau anchor** | `_eligible_var_ids`: sau const neo (`rất`, `là`, …), nếu token post-anchor đã fill Var (vd. `vui` fill `Var:A`) → **loại Var khỏi eligible**. | `hca/matcher.py` | `Em rất vui` không sinh `ngoan` | Không |
| **B4** | **Rule arbitration** | Rule có **const neo** đang active (score ≥ θ) **ắt** rule không-neo (`vi_N_gap_A`) không được mở slot / không boost candidate. Tránh `đỏ` kích hoạt `hoa` giữa câu `Em rất …`. | `hca/matcher.py` | `Em rất vui ngoan đỏ` không ra `hoa` | Không |
| **B3** | **EOS sau frame A** | `is_frame_complete = True` khi post-anchor đã fill mọi Var sau neo của rule active (không chỉ dựa rule S–V khác). | `hca/matcher.py` | `Lan rất vui` → EOS | Không |

**Thứ tự implement đề xuất (mai):** B2 → A2 (teach) → B4 → B3 → `pytest` → `teach-pack` (nếu A2) → smoke Inference Lab.

**Smoke prompts sau Gói 1:**

| Prompt | Kỳ vọng |
| :--- | :--- |
| `Em rất` | `Em rất vui` (hoặc một tính từ), **không** `ngoan đỏ hoa` |
| `Con chó rất` | `… nhanh` / `… vui`, không `con ngủ` |
| `hoa` (generate) | `hoa đỏ` (forward N→A), không `đỏ hoa` |
| `Em học và Bé ngủ` | Dừng đúng, không tail junk |

#### Gói 2 — Train & curriculum (sau Gói 1 ổn)

| ID | Hạng mục | Mô tả | Train? |
| :---: | :--- | :--- | :---: |
| **C3** | G1 align-aware | Co_activation gắn metadata rule/slot hoặc π thấp hơn khi structural active | G1 |
| **C4** | G3 slot violation LTD | Phạt mạnh token sinh ra khi slot đã đóng (ngoan sau vui) | G3 |
| **C1+** | Curriculum | Rà soát lesson trùng pattern (`Em rất vui` vs `Em rất ngoan` — collocation directed giảm xung đột); thêm ví dụ thiếu | Teach |

```bash
python scripts/run_train_eval.py --full --out checkpoints/grade_03.pkl
```

#### Gói 3 — Graph UI & quan sát (song song / sau)

| ID | Hạng mục | Mô tả |
| :---: | :--- | :--- |
| **D2** | Filter EdgeKind | Checkbox ẩn/hiện grammar / collocation / co_activation trên graph |
| **D3** | Trace EOS | Hiển thị `<EOS>` score + lý do frame closure trong Inference Lab |
| **D4** | Viz rule colocation | Tooltip/hover: cạnh xanh `Em→vui` kèm `Rule:vi_S_rat_A` |

#### Gói 4 — Phase 2 kiến trúc (dài hơn)

| ID | Hạng mục | Mô tả |
| :---: | :--- | :--- |
| **B7** | Dual-channel readout | `get()` structural vs `get_readout()` activation; matcher π theo EdgeKind |
| **B5** | Co_activation prefix | Tiếp tục tinh chỉnh khi G1 vẫn bleed (đã có `skip_co` một phần) |
| — | G2 / G4 / GPU batch | Roadmap Phase 1 gốc |

#### Case study (đã phân tích — lưu cho debug)

| Case | Triệu chứng | Nguyên nhân gốc | Fix map |
| :--- | :--- | :--- | :--- |
| `Hỏi: 2 cộng 3 bằng mấy ?` → `cộng bằng 5` | Tokenize và IDF hoạt động | IDF anchor focus đúng vào object cần tìm | Neuron-first QA chain ✅ |
| `Hỏi: Điện Biên Phủ thắng lợi năm nào` → `năm 1954` | Không bị bleed Hà Nội từ lớp 3 | Tách QA lesson theo từng lớp | Per-grade curriculum ✅ |
| `Hỏi: Em học gì ?` → `Em có bút` | Cross-example bleed | Var:A hub + aggregate `Em→có` từ nhiều QA | G3 LTD / Q fingerprint ⏳ |
| `Hỏi: Con chó màu gì ?` → tail junk | Nhiều example QA trộn | Frame chưa EOS sớm; compositional bleed | EOS color + rule gate ⏳ |
| `Bé học rất` → EOS | `<EOS>` dù matcher có `vui` | `vi_S_V` frame complete nhầm; EOS boost 12 | B3 ✅ một phần |
| `Bé học rất` → `ngủ` | Collocation `Bé→ngủ` từ S–V corpus | Global colocation + G1 | A1 ✅ |
| `Con chó rất` → `con ngủ chạy` | Collocation S–V bleed | Rule-scoped + curriculum | A1 ✅ + C1 |
| `Em rất` → `vui ngoan đỏ hoa` | Var:A không đóng; `vi_N_gap_A` trên `đỏ` | B2 + B4 + A2 | ⏳ |
| `đỏ` → `hoa` | Cạnh **ngược** `hoa đỏ` | Directed colocation | A2 ⏳ |

#### Checklist buổi làm việc (cập nhật 2026-07-09)

- [x] v3 nơ-ron cha + matching phân tầng (lịch sử)
- [x] **Hợp nhất sinh A3 multi** — xóa hybrid/compose/essay (TK_019 §7)
- [x] **Stack v10** — history wiki + paragraph units + VarLM cluster-aware (TK_024 C, TK_035)
- [x] Graph UI chỉ A3 multi; checkpoint mặc định v10 paragraph
- [x] Discourse eval corpus-derived (`eval_discourse_prompts.py`)
- [x] `pytest` **131 passed** (legacy tests gỡ)
- [ ] Cải thiện discourse proxy trên v10 (guard 86/88/67 giữ; discourse eval mở rộng)
- [ ] Synonym khác mặt chữ >50% (cần data / E lớn hơn, không pretrain)
- [ ] Sync Inference Lab trace ↔ `generate_autoregressive` (trace.py v2 desync còn lại)
- [ ] Answer closure P1 (`assess_answer_closure`) — Mục 0.2

---

### B.10. Tầm nhìn NN-AIVIE v2 — Đồ thị học khái niệm trừu tượng tự phát (Dynamic Latent Abstraction)

> **Ý tưởng đột phá (Cách mạng hóa cấu trúc Hebbian):** Thay vì sử dụng heuristic thống kê tĩnh hay gán nhãn thủ công để phân nhóm hoặc kế thừa (như Jaccard similarity tĩnh ở Phase 1), mạng v2 giới thiệu cơ chế **Unsupervised Neurogenesis & Hierarchical Association** (Tự sinh nơ-ron và liên kết phân cấp ẩn).
> 
> *   **Cơ chế hoạt động:** Khi một nhóm nơ-ron bề mặt (vd: `bố`, `mẹ`, `anh`, `chị`) liên tục đồng kích hoạt (co-fire) trong các ngữ cảnh tương đồng, một nơ-ron khái niệm ẩn mới (`Concept:001` - đại diện cho "phụ huynh/gia đình") sẽ được tự động sinh ra và liên kết chặt chẽ với chúng.
> *   **Liên kết đa tầng phân cấp (Multi-tier Abstraction):** Các nơ-ron khái niệm này tiếp tục co-fire với các nơ-ron khái niệm khác ở cấp độ rộng hơn (vd: `Concept:001` - gia đình, `Concept:002` - đồng nghiệp) để kích hoạt và tuyển chọn một nơ-ron khái niệm bậc cao hơn (`Concept:010` - "con người"). Quá trình này lặp lại liên tục lên đến các mức độ trừu tượng hóa cực độ (vật lý, sinh học, xã hội, công việc...), xây dựng nên một mạng DAG (Directed Acyclic Graph) tri thức ẩn thuần khiết Hebbian.
> *   **Lan truyền thông tin & Khái quát hóa (Information Spreading):** Nhờ đồ thị phân cấp tự phát này, khi mạng học được một thuộc tính mới ở mức trừu tượng cao (vd: "con người cần thở"), kích hoạt Hebbian tự động lan truyền ngược xuôi qua các tầng khái niệm bậc trung (`Concept:010` -> `Concept:001`) để suy luận zero-shot cho các thực thể cụ thể (`bố`, `mẹ` cũng cần thở), tạo ra khả năng suy luận logic đột phá mà không có bất kỳ dòng code logic hay luật ngôn ngữ hardcode nào.

Kế hoạch chi tiết, kiến trúc thiết kế và lộ trình triển khai cụ thể được trình bày tại: **[`nn-aivie-plan-v2.md`](nn-aivie-plan-v2.md)**.

### B.11. Chuyển đổi tài liệu sang Markdown (MarkItDown)

Dự án hỗ trợ chuyển đổi các tài liệu nghiên cứu dạng Word (`.docx`), PDF, hoặc các định dạng văn phòng khác sang Markdown bằng công cụ **MarkItDown** (phát triển bởi Microsoft), phục vụ cho việc tích hợp tri thức vào LLM hoặc hệ thống phân tích.

#### 1. Cài đặt thư viện
Cài đặt thư viện `markitdown` với các extension hỗ trợ Word (`docx`):
```bash
pip install "markitdown[docx]"
```

#### 2. Sử dụng qua dòng lệnh CLI
Để chuyển đổi một file tài liệu sang Markdown, sử dụng lệnh:
```bash
markitdown docs/study_nn.docx -o docs/study_nn.md
```

#### 3. Sử dụng trong Python API
```python
from markitdown import MarkItDown

md = MarkItDown()
result = md.convert("docs/study_nn.docx")
print(result.text_content)
```

---
*Hết nội dung tài liệu.*