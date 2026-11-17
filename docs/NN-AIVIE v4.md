# NN-AIVIE v4 — TỪ MÁY TRUY HỒI SANG MÁY SINH ĐO ĐƯỢC

> Xương sống v4: **đóng khoảng cách giữa lý thuyết (README Mục III) và code đang chạy**, và
> **biến SINH — không phải TRA — thành mặc định được đo trung thực**. Mọi năng lực đến từ
> kiến trúc + train (Luật 0, README Mục 0). Không hardcode từ vựng ở runtime inference.

Ngày mở: 2026-07-01. Tài liệu này chốt chẩn đoán + lộ trình; cập nhật trạng thái theo phase.

**Cập nhật 2026-07-09 — Hợp nhất substrate:** W4 (hai stack phân mảnh) **đã xử lý** — chỉ còn
`generate(multi=True)` → `generate_autoregressive`. Đã gỡ `generate_hybrid` / `generate_composed` /
`generate_essay`, eval đa-mode (`eval_v4` CLI), Graph UI multi-mode. Thước đo discourse chuyển sang
`eval_discourse_prompts.py` (prompt corpus-derived). Chi tiết README §0.4.1, TK_019 §7, TK_035 §7i.

---

## 1. Chẩn đoán: v3 là gì (soi code + đo trực tiếp)

Bỏ lớp vỏ ("nơ-ron cha", "Hebbian", "lan xuôi/ngược"), **đường sinh mặc định `generate_hybrid`
— cái đạt 100/100/50 — là TRUY HỒI + DỘI VERBATIM**, không phải sinh:

- `ParentStore.rank_matches` = **BM25/TF-IDF trá hình** (coverage token nội dung × IDF + cụm Re-Pair).
- `completion()` trả `answer_ids` **nguyên văn** (`generate_hebbian.py`: `emit = list(completion)`).
- "Nơ-ron cha" = một `ParentRecord` trong `list` lưu chuỗi token gốc. Không kích hoạt, không Hebbian.

**Hệ quả metric:** QA 100% = "lưu đáp rồi tra lại đáp" trên **n=7**. Paraphrase 100% vì **chia sẻ
mặt chữ**. Synonym 50% vì **khác mặt chữ** → retrieval không bắc cầu được.

**Nửa "generative" thật đang chết lâm sàng:**
- Toàn bộ Dual Context Readout (README III.8: C_frame/C_bind, Q/K/V qua Hebbian, Retroactive)
  **không được train v3 nạp dữ liệu**: `dual_context_read` cần `A[R_-k, w0]` nhưng v3 có **0 cạnh
  R-slot** (README 0.6.6 tự xác nhận) → kênh frame luôn = 0.
- `matcher.py` (1420 dòng), `apply_v2_dynamics`, `query_conditioned_read` = **dead code** với v3.
- Tín hiệu trật tự duy nhất = **W_seq bigram (Markov bậc 1)** → A3 chỉ tái dựng verbatim khi
  bigram rõ, word-salad khi không.

**Đo trực tiếp trên `grade_v3_big_a2.pkl` (44k cha, 2026-07-01):** QA 100%, Paraphrase **88%**,
Synonym **33%** — bản BIG **tệ hơn** bản nhỏ (100/100/50) do wiki pha loãng. Thêm: metric
gold-overlap 0.5 **cho điểm sai** (đáp "Một ngày có hai mươi tư giờ" cho hỏi "bao nhiêu ngày"
vẫn PASS vì trùng từ chức năng "có"). Composition (chế độ mới) sinh word-salad ở nhiều ca và
**hoàn toàn không được đo**.

---

## 1.5. BẢN ĐỒ MA TRẬN — v4 đang đứng trên cái gì (TK_045 §12.75, 2026-07-29)

§1 chẩn v3 là "truy hồi + dội verbatim". Sau TK_017/TK_042/TK_045, phải nói lại cho chính xác
**hiện tại** đứng ở đâu — bằng bản đồ ma trận, vì mọi trụ của §3 đều xây trên chúng.
Bảng đầy đủ ở `README.md` §0.5.18; đây là phần ảnh hưởng tới lộ trình v4.

**Bốn tầng, không tầng nào backprop:**

| tầng | ma trận | lo gì |
|---|---|---|
| 0 — bộ nhớ thật | `adjacency`/`W_seq` · `ParentStore` | chữ nào theo chữ nào · mạng biết câu gì |
| 1 — view của `W_seq` | `embedding` E · `DirectedQK` (Wq/Wk) · spectral stack (Wa/Wffn ×4) · `varlm` | **CÁCH nói** |
| 2 — view của `ParentStore` | **`ParentBipartite.A` / `.E`** · `succ`/`emits` · `W_know` · `know_qk` · theme neuron | **NÓI GÌ** |
| 3 — liên câu | `sent_intra` L2R/R2L · `W_agenda` · `sent_qk` · `episodes` | mạch đoạn |

**Ba sự thật đo được, ảnh hưởng thẳng tới §3 (5 trụ) và §4 (trần cứng):**

1. **Đường sinh hiện bị chi phối bởi kênh CHÉP.** `matrix_core_logits` có 11 kênh nhưng ~6 kênh
   thật sự chạy, và β dồn vào `succ+path` (5.0+4.0) = khớp hậu tố emit của cha. Đo v12:
   `copy` 0,93–1,00 ở T1/T2/T4 (TK_045 §12.72). Tức **Trụ 4 "hợp nhất truy hồi + sinh" chưa xảy
   ra** — vẫn là truy hồi, chỉ chép khéo hơn.
2. **Hai cơ chế dành cho việc SOẠN chưa từng được bật.** `qk` (β=0) lấp chuyển tiếp chưa gặp và
   spectral stack (β=0.35 nhưng **bộ đo set về 0**) là đúng thứ Trụ 2/Trụ 3 cần. Chưa có số nào
   về chúng ở β>0 ⇒ đây là **khoảng trắng đo lường**, không phải kết luận âm.
3. **Tầng ĐOẠN không có ma trận học được.** `W_agenda` (Trụ 6 diễn ngôn) có nhưng β=0; mạch đoạn
   chạy bằng `episodes` = thứ tự tài liệu gốc, qua handoff. Hệ quả đo được: **trôi đề** — trên
   v12, `lạc` (tỉ lệ câu từ câu 2 không chia fold nội dung nào với prompt) T1 = 0,12 ở cap 1 và
   **0,27** ở cap 3 (TK_045 §12.74). Cơ chế DUY NHẤT miễn nhiễm là chép-đáp QA (`lạc = 0,00` ở
   mọi cap) — vì nó là cơ chế duy nhất có **cổng phủ nội dung**.

**Hệ quả cho lộ trình:** Trụ 7 (episode) đã có substrate nhưng thiếu **cổng liên quan**; Trụ 6
(diễn ngôn) có ma trận nhưng chưa bật; và trần thật của "đoạn dài đúng nghĩa LLM" **không nằm ở
độ dài** mà ở trôi đề — thứ chỉ mới đo được lần đầu ngày 2026-07-29.

---

## 2. Bốn điểm yếu GỐC (không phải triệu chứng)

| # | Gốc rễ | Bằng chứng | Vì sao là gốc |
| :--- | :--- | :--- | :--- |
| **W1** | **Khủng hoảng thước đo** | eval n=7/8/6 bão hòa; metric cho điểm sai; mọi cải tiến "trung tính 86/100/17" | Không thể cách mạng khi thước có 7 vạch đều max. Đang **bay mù**. |
| **W2** | **Substrate cấu trúc rỗng** | 0 cạnh R-slot; W_seq chỉ bigram; `dual_context_read` trả 0 | Mọi năng lực sinh > verbatim phụ thuộc tín hiệu trật tự — v3 gần như không có. |
| **W3** | **Trần khái quát hóa** | PPMI+SVD corpus nhỏ = 50%; `_df` **chỉ đếm curated** → thực thể chỉ-có-wiki (Cao Bằng) bị lọc df>0, focus trôi về "gì/là" | PPMI bậc-1: "chó≈cún" chỉ nổi nếu đồng hiện. Corpus nhỏ + no-pretrain = trần cứng. |
| **W4** | **Hai stack sinh phân mảnh** | `generate_hybrid` (verbatim, đo) vs `generate_autoregressive` (sinh, opt-in, KHÔNG đo); trace.py v2 desync | **2026-07-09: ĐÃ HỢP NHẤT** — một đường `generate(multi=True)`; hybrid/essay/composed **xóa**; eval discourse mới. Vấn đề "đo ≠ sinh" còn ở mức metric (guard 86/88/67 vs discourse proxy), không còn ở code path song song. |

---

## 3. Cuộc cách mạng: 5 trụ (theo thứ tự đòn bẩy)

### Trụ 1 — DỰNG LẠI CÂY THƯỚC TRƯỚC (nền của mọi thứ)

> 🔴 **ĐÍNH CHÍNH 2026-07-17 (TK_042 G0) — TRỤ NÀY CHƯA BAO GIỜ XONG, VÀ ĐÓ LÀ THỨ ĐÃ GIAM v4.**
>
> Thước LM (`eval/metrics.py:116`) làm `sum_bits += math.log2(rank)`. **`bits` = log2(THỨ HẠNG),
> KHÔNG phải cross-entropy.** *"4.155 bits"* = *"token đúng ở hạng ~17.8"* — không liên quan
> lượng tin. Ba lỗi độc lập:
>
> 1. Gọi `log2(rank)` là "bits" → **mời gọi lỗi phạm trù** (tài liệu này đã mắc khi so với
>    perplexity Transformer).
> 2. `if not isfinite(scores[gold]): n_oov+=1; continue` → **BỎ QUA đúng những vị trí mô hình
>    thất bại hoàn toàn** → thiên lệch chọn mẫu → số đẹp giả tạo. (Đo: **9.0%** vị trí held-out
>    là OOV trên vocab legacy — thước cũ vứt sạch chúng.)
> 3. **HẠNG ⊥ CALIBRATION.** → Giải thích một âm tính bị hiểu sai suốt: *"KN smoothing âm +1.26
>    bits"* (§6, 2026-07-09). **KN sinh ra để sửa CALIBRATION**, nó đánh đổi top-1 lấy đuôi phân
>    bố. Thước hạng **chỉ thấy cái KN hy sinh, mù với cái KN sửa**. Ghi chú *"ruler leaky"* là
>    **nghi đúng, sai nguyên nhân** — không phải leakage, mà **đo nhầm đại lượng**.
>
> ⇒ **Mọi con số "bits" trong tài liệu này KHÔNG DÙNG ĐƯỢC.** Thước thật: `scripts/eval_bits.py`
> (cross-entropy, **MỌI vị trí**, bits/token + bits/**byte**). Chi tiết: **TK_042 §4**.

Không đụng kiến trúc cho đến khi thước phân biệt được **truy hồi** với **sinh**:

1. **Tách metric:** `retrieval_accuracy` (best_match chọn đúng cha) vs `generation_quality` (đáp
   sinh token-level, chấm nghiêm) — hiện `qa_v3` gộp cả hai.
2. **Metric nghiêm (IDF-weighted):** phủ token gold **đặc thù** có trọng số IDF + bắt buộc token
   gold phân biệt nhất có mặt → chặn false-pass "có/là/một". (Eval code — không thuộc runtime
   inference nên không ràng Luật 0, nhưng vẫn dùng IDF thay vì danh sách stopword hardcode.)
3. **`verbatim_rate`:** % đáp trùng nguyên văn một cha. =100% ⇒ máy tra cứu. Mục tiêu cách mạng:
   **giữ chất lượng nhưng GIẢM verbatim_rate** trên held-out.
4. **Composition coverage:** #fact phân biệt được phủ + phạt lạc đề (off-topic) — đưa chế độ ghép
   fact vào vòng đo.
5. **bits-per-byte trên corpus thật** làm metric liên tục không bão hòa (hiện chỉ smoke).

**Deliverable (lịch sử):** `src/nn_aivie/eval/v4.py` (module giữ cho essay metric nội bộ nếu cần).
**Đã gỡ CLI:** `scripts/eval_v4.py`, `tests/test_eval_v4.py` (2026-07-09). Thước đo runtime:
`scripts/eval_discourse_prompts.py` + guard spot-check trên subset QA/Para/Syn khi cần.

### Trụ 2 — HỒI SINH SUBSTRATE CẤU TRÚC (làm engine sinh có gì để đọc)

Lý thuyết HCA đã đúng, đã code, chỉ **train không nạp**. Sửa `training/ingest.py`:
1. **Nạp cạnh vị trí tương đối R_-k** → `dual_context_read` sống lại → A3 có kênh frame thật.
2. **W_seq bậc cao + backoff** (trigram/skip-gram kiểu Kneser-Ney trên đồ thị) thay bigram thuần.
3. Đưa A3 thành mặc định cho câu **mở/đa-ý**, verbatim chỉ cho QA-fact ngắn — đo bằng Trụ 1.

### Trụ 3 — LƯỢNG TỬ HÓA PARADIGM TỪ SLOT HUB (đánh trần synonym từ phía CẤU TRÚC)

`SlotHub.fillers` **theo định nghĩa** là lớp thay thế được (paradigm class) — "đồng nghĩa vị trí".
PPMI bậc-1 mù "Lý≈Trần" (không đồng hiện) nhưng hub thấy ngay (cùng slot).
1. Thăng filler-set thành **nơ-ron kiểu** hạng nhất: truyền đáp giữa filler type-compatible
   (case con kiến ở quy mô), cung cấp **tương đồng bậc-2 graph-native** thay fuzzy-E đang phải tắt.
2. Sửa gốc W3: `_df`/`entity_register` phân biệt "nhiễu chỉ-trong-câu-hỏi" với "thực thể wiki"
   (df=0) — để focus không trôi về "gì/là" trên thực thể chỉ-có-wiki.
3. Iterate PPMI sang **bậc 2** (giống nhau nếu hàng xóm giống nhau).

### Trụ 4 — HỢP NHẤT TRUY HỒI + SINH thành một lượt decode có điều kiện

Cha truy hồi → **agenda mềm + khung cấu trúc**; readout token-level (có cấu trúc từ Trụ 2) là bộ
decode, được G4 verify + hub-guard bảo vệ. Mục tiêu đo được: **bằng verbatim trên QA đã thuộc,
VƯỢT verbatim trên composition/held-out** → khi đạt, gỡ nhánh verbatim làm default.

### Trụ 5 — ĐẦU CẦU CODE-GEN (Track B) làm "cưỡng bức trung thực"

AST cho cấu trúc miễn phí (hữu hạn, 0 nhiễu) + compiler/test cho reward khách quan → phơi bày
ngay "trừu tượng cấu trúc W_struct" là thật hay ảo tưởng (prose cho phép tự lừa). B1 (parent-từ-AST,
tái dùng slot hub cho identifier slot) là sân thử cho Trụ 3–4 trước khi tin trên prose.

### Trụ 6 — TẦNG DIỄN NGÔN: THEME NEURON + ANAPHOR HỌC + SFT (2 ý tưởng tác giả, 2026-07-01)

> Hai ý tưởng của tác giả **hội tụ**: (1) đánh trọng số liên kết đại-từ↔chủ-thể, gate theo ngữ
> cảnh nơ-ron cha; (2) pha SFT tạo **nơ-ron trừu tượng** nối tới cha + gia giảm trọng số ghép
> câu/fact/hub. Cả hai gặp nhau ở một tầng mới: **Nơ-ron Chủ đề (Theme/Concept)**.

**Tầng phân cấp mới:** `token → cụm Re-Pair → cha(fact) → hub(ngang: 1 khung, nhiều filler)`
**`+ Theme(dọc: 1 thực thể, nhiều fact) → domain(tương lai)`**. Theme = `facts_about` hiện tính
on-the-fly, nay **vật chất hóa thành nơ-ron** (`Concept:entity`, tái dùng hạ tầng concept sẵn có)
+ gắn **thuộc tính diễn ngôn**: đại-từ ưu tiên, thứ tự trình bày, tập coverage.

**Ý 1 — Anaphor học 3 tầng (thay heuristic successor-cosine mong manh):**
| Tầng | Cơ chế | Ghi chú |
| :--- | :--- | :--- |
| (a) Cạnh trực tiếp | Hebbian trong-cha: chủ thể X + đại từ P đồng hiện → `A[X,P]↑` | chỉ có cho thực thể XUẤT HIỆN trong corpus anaphora |
| (b) Cạnh LỚP | X chưa có → mượn của lớp paradigm (hub filler class / theme cùng loại) | **nối Trụ 3** |
| (c) Heuristic | fallback cap%+cosine hiện tại | — |
Treo trên **Theme** → gate ngữ cảnh (đúng ý tác giả). Coi `,+đại_từ` là **một cụm** (Re-Pair).

**Ý 2 — SFT = pha CỦNG CỐ có giám sát (Hebbian G3-style, KHÔNG backprop):**
- 2a **Theme neuron** (cấu trúc) — làm TRƯỚC: xương sống composition (ordering định-nghĩa-trước,
  dedup, coverage) → context mạnh hơn; host anaphor.
- 2b **SFT reweight** — làm SAU: sinh trên gold (QA + `composition.jsonl`) → chấm eval_v4 → **củng
  cố** (LTP) cạnh Theme/hub/fact-join tạo đáp đúng+phủ; **ức chế** (LTD) cạnh lạc đề/trùng/sai đại
  từ. Tái mục đích `g3_dopamine` (đang tắt v3). **CHỐT AN TOÀN: đóng băng đường retrieval crisp-QA
  (100%)** — SFT chỉ chạm composition/theme/anaphor.

**Trần trung thực (phải nói thẳng — chống "ảo tưởng"):**
- **Animacy = trần DATA.** Corpus anaphora là tin tức (antecedent = người/đội → họ 391/chúng 533/
  nó 169/các 4). Thực thể composition (Huế, Nhà Trần, Sông Hồng) **không có mặt** → cạnh trực tiếp
  (a) **rỗng cho đúng thực thể ta cần**; lớp (b) cũng rỗng (Lý/Đinh/Trần không trong corpus). → **Ý 1
  học-cạnh sẽ TRUNG TÍNH trên data hiện tại cho các thực thể này** (đúng bẫy Trụ 2). "Nhà Trần→họ"
  hiện tại **vốn đã đúng** (triều đại = tập thể người → "họ"); chỉ địa danh vô tri (Huế→nên "nó")
  là sai, và đó là trần data (chọn (i): chấp nhận "họ" tạm).
- **Metric composition đang bão hòa** (100/0 trên grade_v3) → phải **mở rộng gold** trước, nếu không
  Theme/SFT lại "trung tính".
- **SFT dễ overfit gold nhỏ + rủi ro phá 100% QA** → cô lập tuyệt đối + mở gold trước.

**Lộ trình đo được (đúng kỷ luật Trụ 1):**
| Phase | Nội dung | Đo bằng | Rủi ro |
| :--- | :--- | :--- | :--- |
| **P0** | Eval anaphor (thực thể→đại từ chấp nhận được) + mở rộng `composition.jsonl` + metric **ordering** | thước đo mới | thấp — BẮT BUỘC trước |
| **P1** | Theme neuron: vật chất hóa + xương sống composition (ordering/coverage) | composition ordering | thấp (additive) |
| **P2** | Anaphor học 3 tầng, host trên Theme | eval anaphor | **kỳ vọng trung tính trên data hiện tại** → chỉ giữ nếu đo có lợi |
| **P3** | SFT củng cố (G3 tái mục đích), đóng băng QA | toàn bộ eval_v4 | cao — cuối |

**Quyết định thực dụng:** P0 (đo) + P1 (Theme structure) làm trước — rẻ, mở khóa. P2 chỉ commit
NẾU eval anaphor cho thấy lợi thật (nếu không, ghi âm tính như Trụ 2). P3 sau cùng khi gold đủ lớn.

### Trụ 7 — NƠ-RON TRỪU TƯỢNG t1 (EPISODE) — ý tưởng GỐC của tác giả (2026-07-02)

> **Ý tưởng gốc (tác giả, nguyên văn ý):** train SFT với gold DÀI (vd "Hỏi: cho tôi một đoạn văn
> dài về loài mèo Đáp: <bài văn nhiều đoạn>") → tạo **nơ-ron t1** (trừu tượng bậc 1) từ đoạn văn.
> t1 **kích hoạt toàn bộ token** của đoạn → quét xem **nơ-ron cha nào cộng hưởng** (np1 "Mèo rất
> nhanh nhẹn và có đôi mắt rất sáng", np2 "Mèo rất thích ngủ"…) → **nối np1, np2 → t1**. t1 vẫn
> phân rã Re-Pair ra cụm trung gian ("quấn quýt", "trung thành", "tinh anh"). Prompt khác về mèo
> có thể **đi lên trừu tượng hơn rồi phân rã xuống sâu hơn** (t2+). Và **làm giàu nghĩa**: "mắt
> sáng" ↔ "tinh anh" nối nhau qua trọng số/embedding vì cùng ngữ cảnh trừu tượng.

**Phân biệt với Theme neuron (Trụ 6 P1):** Theme = chỉ mục *thực thể→fact* (dọc, suy từ vị trí);
t1 = nơ-ron *tình tiết/episode* — một khối diễn ngôn hoàn chỉnh là một nơ-ron, thành viên nối bằng
**cộng hưởng kích hoạt** (không cần chung thực thể), giữ **thứ tự trình bày**. Sinh học: concept
cell / episodic memory. Hai cái bổ sung: theme trả "biết gì về X", t1 trả "một bài hoàn chỉnh về
X trông thế nào".

**Tầng kiến trúc:**
```
token → cụm Re-Pair → cha (1 câu) ──ngang── hub (khung+filler)
                         │          ──dọc──  theme (thực thể→fact)
                         ↓ member (THỨ TỰ + w_cộng_hưởng)
                      t1 (đoạn/episode/bài gold)  →  t2 (chủ đề)  [chỉ khi đủ t1]
```

**Ba cách dùng khi sinh:**
1. **Essay prompt** → khớp query lên t1 → agenda = members theo thứ tự → closure khi phủ đủ
   (= xương sống thật cho A4 agenda/closure).
2. **Coherence episode khi ghép câu:** ưu tiên fact cùng-t1 với fact trước → chặn drift đồng-âm
   khác-chủ-đề (Biển Đông VN vs "Biển Đông (Thụy Điển)" nằm t1 khác nhau).
3. **Làm giàu nghĩa (synonym paradigmatic):** cạnh bậc-2 giữa cụm đồng-thành-viên nhiều t1;
   E mức episode (term×episode PPMI+SVD — LSA-style). Mở **phương án D** cho quyết định treo
   Trụ 3: phá trần synonym KHÔNG cần pretrain (Luật 0 sạch).

**Lộ trình đo được:** T0 (thí nghiệm rẻ: E-episode intrinsic synonym) → T1 (AbstractNeuron + nối
xuất xứ wiki-doc + nối cộng hưởng + Re-Pair trên t1) → T2 (essay agenda; gold từ HF) → T3 (cạnh
enrichment) → T4 (SFT củng cố G3). Mỗi bước giữ nếp: đo dương mới giữ.

**Nguồn gold dài (khảo sát HF 2026-07-02):** `bkai-foundation-models/vi-alpaca` — instruction
tiếng Việt → đáp luận NHIỀU ĐOẠN ("Hãy viết một bài blog ngắn về lợi ích của việc đọc sách"),
không gated, đọc qua datasets-server /rows như wiki. Phụ: WikiLingua vi (how-to). Loại:
`tungkho178/VanMau` (gated, không docs).

**Hạ tầng đã có cho t1:** re-ingest wiki giữ **doc_id + title + para** (`ingest_hf_vi.py`,
`data/real_vi/statements_doc.jsonl`, 45k câu) — câu cùng đoạn = cùng episode; máy cộng hưởng
= `rank_matches`/spreading sẵn có; phân rã = **ISG online (TK_013)** hoặc `decompose_repair`
(`--legacy-repair`).

**⚠️ ĐÍNH CHÍNH THIẾT KẾ t1 (tác giả, 2026-07-02 — TK_010):** cách hiện thực TK_004 (chia bài
thành câu → AbstractNeuron trỏ member) SAI nguyên lý train thống nhất. Đúng: **episode/bài dài/
Hỏi-Đáp dài = MỘT NƠ-RON CHA THẬT** (chuỗi token đầy đủ, KHÔNG chia câu/đoạn), đối xử y hệt
nơ-ron fact; khác chỉ ở TẦNG LINK PHỤ: (a) link t1→fact tìm bằng cộng hưởng từ list token,
CÓ THỨ TỰ theo vị trí xuất hiện; (b) fact↔fact nối ngang theo thứ tự; (c) trung gian (Re-Pair
trên chuỗi t1) nối tới fact nếu là thành phần, nối ngang theo index. Một kiểu train duy nhất
top-down cho mọi quy mô: nơ-ron → token → trung gian (ngang theo index) → tìm fact → sắp thứ
tự → nối vào t1. Spec migration: `TK_010_T1_UNIFIED_PARENT.md` (chờ duyệt).

**T0 — KẾT QUẢ ĐO (2026-07-02): intrinsic DƯƠNG, extrinsic ÂM TÍNH trên synonym probe.**

*Setup:* E-episode = PPMI term×episode → SVD d=96, trên 45k câu wiki (`statements_doc.jsonl`,
có doc_id+para). Lần 1 (episode=cả bài, 1118 bài): **âm tính** — bài đầu wiki quá dài, trộn
trăm chủ đề. Lần 2 (episode=ĐOẠN VĂN, 24 864 đoạn): tín hiệu thật.

| Cặp synonym | rank E-window | rank E-episode |
| :--- | :---: | :---: |
| cấy/trồng | 75 | **3** |
| khâu/may | 1215 | **154** |
| hôm/ngày | 1118 | **256** |
| trị/chữa | 634 | 596 (đa nghĩa "giá trị" nuốt) |
| bảo/giữ · vệ/giữ | 93 · 708 | mất (-1) — "bảo" đa nghĩa + tách âm tiết từ ghép |

Đối chứng không-liên-quan vẫn thấp/xa (không phải "mọi thứ gần nhau hơn"). Hàng xóm định tính
đẹp: `chữa`→{bệnh, sẹo, thuốc, tật, AIDS, lây, mắc} — paradigmatic y khoa mà E-window không có.

*Extrinsic (fuzzy match, retrieval big_a2, sweep topk 3–5 / min_sim .20–.35):* Syn **33% bất
động** với E-episode. Vì: (a) cặp E-ep giỏi (cấy/trồng) rơi vào ca đã pass nhờ mặt chữ ("lúa");
(b) ca cần cầu nối (bảo vệ→giữ) đúng chỗ E-ep mù (đa nghĩa "bảo"); (c) 2 ca fail còn lại là
**bug ranking pattern "Người X là ai"** (→"Trung thu là tết trẻ em") — kể cả "dạy học" trùng
mặt chữ 100% vẫn fail → không phải trần synonym, là việc riêng phải sửa.

*Phát hiện phụ:* fuzzy + **E-window** topk3 sim.35: big_a2 Syn **33→50%**, QA/Para giữ nguyên;
nhưng grade_v3 (nhỏ) Para **75→62%** — ablation 0.5.12 "fuzzy hại" chỉ đúng corpus nhỏ.
→ default giữ OFF; bật là quyết định theo-checkpoint (corpus lớn: lợi không kèm hại đã đo).

*Kết luận T0:* đường "E-episode cứu synonym-probe" **KHÔNG đạt** (đa nghĩa một-vector + probe
nhỏ bão hòa bởi bug khác). Tín hiệu paradigmatic mức đoạn là THẬT → giữ làm nguyên liệu cho
**cạnh enrichment cụm↔cụm (T3)** — cụm đa-token ("bảo vệ", "mắt sáng") ít đa nghĩa hơn token
đơn lẻ, đúng ý tưởng gốc của tác giả (nối "mắt sáng"↔"tinh anh", không phải "bảo"↔"giữ").
Vai trò chính của t1 (essay agenda T2 + coherence episode T1) KHÔNG phụ thuộc kết quả này.

---

## 3.7. VÌ SAO TÁM CƠ CHẾ ĐỀU "TRUNG TÍNH" — MỘT BỆNH, KHÔNG PHẢI TÁM LẦN XUI (2026-07-17)

> Ghi ở đây vì nó đọc lại toàn bộ nhật ký §6 và Trụ 2/3/7. Chi tiết: **`TK_042_GENERATIVE_CORE.md`**.

| Cơ chế | Kết quả |
| :--- | :--- |
| Re-Pair · Re-Pair đệ quy · order-2 · compose_episode | trung tính |
| R-slot · E-episode (extrinsic) · KN smoothing | **âm tính** |
| Theme neuron | +2pp |

**Cả tám đều là "đếm thêm thứ gì đó". Mà bảng đếm KHÔNG điều-kiện-hoá được ngữ cảnh dài** —
thêm ngữ cảnh → đếm **thưa** → backoff → **về lại bigram**. Bằng chứng ngay trong repo: VarLM/PPM
nhìn **4 token** thay vì 1 chỉ mua **0.21 bit**.

**Đo (2026-07-17):** `H(next|w0)` = **6.42 bits** trên 151k cha = **86 lựa chọn ngang ngửa**,
mode giữ **16.7%** — và **entropy TĂNG theo scale** (mini 5.34 → 6.42). Tám cơ chế đang tinh
chỉnh một phân bố cần đi từ ~6 bits xuống ~2. **0.21 bit là nhiễu so với khe đó.**

**Ba hệ quả đảo cách đọc v4:**

1. **Agenda KHÔNG phải "scaffolding tạm" — nó là thứ DUY NHẤT cấp ngữ cảnh.** Bỏ nó → sinh từ
   phân bố 86-lựa-chọn → word salad. Giữ nó → có ngữ cảnh, **nhưng ngữ cảnh đó LÀ một đoạn văn
   đã lưu**. ⇒ **Truy hồi đang làm THAY việc của điều-kiện-hoá-ngữ-cảnh.** Đó là toàn bộ lý do
   đầu ra "verbatim copy" — **lỗ hổng MÔ HÌNH, không phải bug decode**.
2. **`ar_index_backbone` KHÔNG phải thủ phạm** — A/B: tắt nó → **99% token y hệt** (4 kênh còn
   lại đã đồng thanh chỉ về đúng đoạn đó). Backbone **thừa**, không phải nguyên nhân.
3. **`chó`/`cún` không nổi được** vì với entropy 6.42 mạng **không "chọn giữa chó và cún" — nó
   không biết gì cả**, agenda quyết hộ. ⇒ Bật `ar_temperature` **sẽ KHÔNG** cho văn phong: lấy
   mẫu từ phân bố dốt = nhiễu. **Phân bố phải giỏi TRƯỚC.**

**Và đính chính một câu SAI của Trụ 3:** *"PPMI bậc-1: 'chó≈cún' chỉ nổi nếu đồng hiện"* — **sai
kỹ thuật**. Tương đồng paradigmatic đến từ **hàng vector giống nhau** (cùng hàng xóm), không từ
đồng hiện trực tiếp — đó là cơ chế word2vec/GloVe. **Chính T0 đã chứng minh trên data này**:
`cấy/trồng` 75→**3**; `chữa`→{bệnh, sẹo, thuốc, tật}. ⇒ **Dự án ĐÃ CÓ word2vec** (`embedding.py`
= PPMI+SVD = bản SVD của word2vec, Levy & Goldberg 2014). Cơ chế **đã chạy**; cái thiếu là nó
**không được dùng để SINH** — và **embedding ≠ mô hình ngôn ngữ**: word2vec cho *"từ nào giống
từ nào"*, không cho `P(token kế | ngữ cảnh)`.

**Kết luận:** R1 (sinh thật) chỉ đạt bằng **ngữ cảnh dài** ⇒ buộc **nén ngữ cảnh thành vector**
⇒ buộc **HỌC một mô hình thật**. Đây là bức tường NLP đâm 30 năm (Kneser-Ney trên **hàng tỉ**
token vẫn ~1.7 bits/char vs neural ~1.1); **Bengio 2003 (NNLM) ra đời chính xác vì lý do này**.
→ Quyết định A/B/C: **TK_042 §5**. Việc kế: **G7** (`scripts/train_neural_lm.py`) — đo cái gap
cho ra số thật.

---

## 3.5. ĐỔI LUẬT 2026-07-02 (tác giả): KHÔNG TẮT CƠ CHẾ — SỔ NỢ TÍNH NĂNG

> **Chỉ đạo tác giả:** cơ chế đo yếu/trung tính KHÔNG tắt nữa — mặc định BẬT, chấp nhận số
> thấp hơn, ghi chú để fix sau (bug hoặc cần data lớn hơn). Cơ chế tắt thì không thể cải tiến
> mạng theo hướng cách mạng. (Thay tiền lệ "trung tính → default OFF" của order-2/episode.)
> Kỷ luật ĐO giữ nguyên: mọi giá phải được đo và ghi — chỉ thay đổi hành xử SAU khi đo.

**SỔ NỢ (flag đang BẬT dù chưa thắng — giá đã đo + nghi vấn gốc + hướng fix):**

| Flag (BẬT 07-02) | Giá đã đo | Nghi vấn gốc | Hướng fix |
| :--- | :--- | :--- | :--- |
| `hybrid_fuzzy_topk=3` (min_sim .35) | **đo 07-02 default mới: big_a2/`_t2` Syn 50→67 ✓✓ KHÔNG hại QA/Para/comp; grade_v3 Para 75→62 ✗** | corpus nhỏ thiếu đồng hiện — KHÔNG phải bug | scale data / E-episode cụm (T3) |
| `ar_seq_order2_weight=0.5` | trung tính từng-token (07-01) | index-backbone chi phối, micro chỉ phá thế hòa | sống chờ Trụ 4 (decode dựa micro hơn) |
| `compose_episode_weight=1.0` | trung tính (ON=OFF 64/2/83) | member chưa cạnh tranh thật; episode wiki mỏng | T2/T3 essay + episode dày hơn |
| `anaphor_pref_weight=1.0` (P2 PMI wire lại) | **đo 07-02: anaphor GIỮ 50% (không phá — filter min_pron_docs=20 + lowercase-gate đã chặn "nãi/chủ")** | thiếu bộ lọc đại-từ POS-like; "Việt Nam→đó" chưa lật được base "họ" | inventory đại từ sạch (TK_003) / data anaphora vô tri |
| `hybrid_markov_answer` | (vẫn False) accuracy-neutral nhưng xáo trật tự đáp | không phải "đo yếu" — đổi bản chất verbatim-first của v3, vỡ test-suite nền | ticket riêng khi Trụ 4 (đổi cả suite + đo lại) |

Số "giá" trong bảng cập nhật lại sau mỗi lần eval đầy đủ (xem nhật ký mục 6).

---

## 3.55. LUẬT 0.0 — MẠNG THUẦN NƠ-RON (2026-07-03)

Xem README Mục 0.0. Mạng chỉ chứa id+cạnh+trọng số+kiểu; text/label/content → content-cache
riêng (UI/trace/debug). Rà soát: vi phạm A(nhãn) B(~48 re-parse token-string) C(phrase_assoc
surface-key). Lộ trình dọn: `TK_012_PURE_GRAPH_CONTENT_CACHE.md`.

## 3.6. Ý TƯỞNG ĐANG Ủ (trao đổi tác giả 07-02/03 — spec trọn vẹn trong docs/study)

| Ý tưởng | File | Trạng thái |
| :--- | :--- | :--- |
| **Attention Q/K/V chiết từ đồ thị** — Wq/Wk = U√Σ/V√Σ từ SVD lớp cạnh CÓ HƯỚNG; multi-head theo loại cạnh; V = trường axon; G3 = tính dẻo | [`docs/study/03_ATTENTION_QKV_FROM_GRAPH.md`](study/03_ATTENTION_QKV_FROM_GRAPH.md) | Chờ A0 (thí nghiệm rẻ held-out bigram) → TK_011 |
| **Văn phong tùy biến** — khung = slot-hub ĐẢO; style cluster từ đồng-episode; kênh a_style thứ 5 + persona priming/mirroring | [`docs/study/04_STYLE_REGISTER_CONTROL.md`](study/04_STYLE_REGISTER_CONTROL.md) | Chờ S0 (mine khung + đọc tay tách cực) → TK_012 |

| **Học mồi (concept-first)** — gieo nơ-ron khái niệm theo curriculum (L3→TS); tự đấu dây bằng resonance (khởi tạo) + G3 (trọng số đúng); concept làm neo H_slow khi sinh | [`docs/study/06_PRIMED_CONCEPT_LEARNING.md`](study/06_PRIMED_CONCEPT_LEARNING.md) | Chờ C0 (gieo 5-10 concept + đo attach) → TK_014 |
| **Tầng neo — nơ-ron = BẤT BIẾN** — máy chiếu-bất-biến sinh THỂ LOẠI nơ-ron (topic, emotion, voice…); đồng hồ đa tầng H_fast/mid/slow khi sinh; neo thay dần gate mặt-chữ | [`docs/study/05_ANCHOR_NEURONS.md`](study/05_ANCHOR_NEURONS.md) | Chờ N0a/N0b (cluster topic + probe emotion) — ticket riêng sau TK_013 ISG |
| **Từ điển làm substrate CÓ NHÃN** — nạp từ điển VN (POS/nghĩa/đồng-trái nghĩa/ví dụ) làm data → neo POS/sense/attr (nhiều nơ-ron neo thay fact); cạnh SYN/ANT tường minh (hạ tường synonym+đa-nghĩa xuống độ-phủ); link ngang XA `FRAME_BRACKET` (`nếu→thì`,`Hỏi:→Đáp:`); hoàn thành Q/K doc 03 + neo doc 05. Luật 0 SẠCH (data, không rule) | [`docs/study/08_DICTIONARY_LABELED_SUBSTRATE.md`](study/08_DICTIONARY_LABELED_SUBSTRATE.md) → `TK_015` | Chờ bước 0 (held-out) + C0 (nạp tay 15 mục, đo tách sense) |

Cả ba đứng trên nền vừa xong: TK_009 (sinh token-level), TK_010 (t1 nguyên vẹn),
TK_008 (phrase_assoc). Điều kiện chung để bùng: **scale essay/corpus** (+ data review
phân cực cho emotion).

## 4. Trần cứng — nói thẳng (chống "ảo tưởng sức mạnh", Luật 0)

1. **Luật 0 + no-pretrain = trần synonym/tri-thức không kiến trúc nào phá hết.** "chó≈cún" chỉ
   đến từ (a) thêm data đồng hiện, hoặc (b) **nới Luật 0 cho RIÊNG tầng E** (embedding tiếng Việt
   pretrain, giữ mọi thứ khác graph-only) — **quyết định chính sách của tác giả**, không phải bug.
   Trụ 3 đẩy trần lên đáng kể nhưng không xóa.
2. Hệ này **tái tổ hợp khuôn mẫu, không phát minh** cốt truyện/thuật toán mới (docx tự thừa nhận).
   Định vị đúng: "lõi sinh chuẩn văn phạm + màng lọc ngữ cảnh", không phải LLM tổng quát.

---

## 5. Nợ kỹ thuật (dọn song song)

- Dead code v2: phần lớn `matcher.py`, `apply_v2_dynamics`, `query_conditioned_read` — hồi sinh
  (Trụ 2) hoặc xóa.
- `.pyc` bị track trong git → `.gitignore` + gỡ index.
- `viz/trace.py` (v2) **đã xóa** (TK_019). Graph UI trace A3 qua `autoregressive.steps` trong `/api/v3/complete`.

---

## 6. Nhật ký tiến độ

| Ngày | Việc | Kết quả |
| :--- | :--- | :--- |
| 2026-07-01 | Chẩn đoán v4 + đo big_a2 | QA 100 / Para 88 / Syn 33; phát hiện metric false-pass |
| 2026-07-01 | Vá `facts_about` (priority + cụm liền kề + dedup bỏ dấu) | Xóa rác citation (Nhà Trần), hết trùng (Bác sĩ); 247 pass |
| 2026-07-01 | **Trụ 1** — eval v4 (retrieval/generation split + IDF metric + verbatim_rate) | ✅ `eval/v4.py`, `scripts/eval_v4.py`, `tests/test_eval_v4.py`; 252 pass |
| 2026-07-02 | Xác nhận đầu phiên (handoff 7.4): pytest + eval sau revert P2 | ✅ 253 pass; big_a2 comp 73/0/92, anaphor 50% — revert sạch |
| 2026-07-02 | **P0/P1** — mở gold big_a2 (n=13) + **Theme neuron** + vá strip tên riêng | comp big_a2 67→69 cov, 82→84 top; grade_v3 93 (artifact); 258 pass |
| 2026-07-02 | **Trụ 7 T0** — E-episode (đoạn wiki, doc_id+para) intrinsic/extrinsic | Intrinsic DƯƠNG (cấy/trồng 75→3); extrinsic Syn 33% bất động → ÂM TÍNH; phụ: fuzzy+E-window big_a2 Syn 33→50 (nhỏ: hại Para) |
| 2026-07-02 | Trụ 7 khảo sát gold dài HF | `bkai-foundation-models/vi-alpaca` (instruction→luận nhiều đoạn, /rows OK); loại VanMau (gated) |
| 2026-07-02 | **Bug ranking "Người X là ai"** — `rebuild_hierarchical_index` không prune token khỏi `_index` | Cha đúng không biến mất khỏi candidate khi query lệch phân đoạn cụm; Syn probe có thể cải thiện |
| 2026-07-02 | **Trụ 7 T1** — `AbstractNeuron` + `build_episodes.py` + episode coherence trong compose | `pick_facts_coherent`; `compose_episode_weight`; 9,241 episode; 269 pass |
| 2026-07-02 | **Bug triage** — case-equiv token (`Trái Đất`≡`Trái đất`), gate định từ `các`, quality từ điển | `case_equiv_ids`; successor overlap gate; `quality.py` chữ Nho |
| 2026-07-02 | **Bug ranking "X là ai" (curated)** — `best_match` gate primary entity + index fallback | Napoléon/Hybrid không còn Trung thu; Người dạy học→Thầy cô; 269 pass |
| 2026-07-02 | **ĐO LẠI SAU cursor2 (Claude review — eval đầy đủ 2 checkpoint)** | grade_v3 giữ nguyên (100/75/50; comp 93/0/100). big_a2: QA 100 ✓, **Para 62→88 (+26)** ✓, **Syn 33→17 (−16) ✗** — gate primary-entity loại cha synonym ("Ai cấy lúa"→wiki "Họ làm ruộng cấy…", "Ai khâu quần áo"→rác "sân khâu"); comp 69→64; anaphor 50 ✓. **Cần sửa gate: chỉ override khi curated khớp KHUNG-thuần** (§7.5 #1) |
| 2026-07-02 | **TK_002 IMPLEMENTED (Claude)** — `_match_has_focus` nhận MỌI focus đủ phân biệt (gidf≥75% primary, không chỉ primary); `compose_episode_weight`→0.0 (đo trung tính); ứng viên đại từ phải viết thường (hết "Pháp") | big_a2: QA 100, Para 88, **Syn 17→50** ✓✓, comp 64/2/**83** (cov 64 = artifact gold: case-equiv tìm curated tốt hơn cho Trái Đất/Mặt Trời, top 100%), anaphor 50; grade_v3 nguyên vẹn; **270 pass**. Còn lại → **TK_003** (anchor họ-người + refresh gold 2 ca) |
| 2026-07-02 | **ĐỔI LUẬT (tác giả)** — không tắt cơ chế, SỔ NỢ (mục 3.5); bật fuzzy/order2/episode/pref-PMI | Giá đo lại: big_a2 **Syn 50→67** ✓✓ (fuzzy), QA/Para/comp/anaphor giữ; grade_v3 Para 75→**62** (giá corpus nhỏ — sổ nợ), còn lại giữ. **274 pass** |
| 2026-07-02 | **TK_004 IMPLEMENTED (Claude)** — T2 essay agenda: 500 essay vi-alpaca → `_t2.pkl` (+3680 cha, 500 t1 SFT, 2080 member cộng hưởng TB 4.2/bài); `match_episode` + `generate_essay` + `eval_essay` (ordering) | **Essay lần đầu: cov 67% / ordering 75%** (5/8 ca hoàn hảo — "đi lên t1, phân rã xuống" chạy end-to-end); 3 fail: match nhầm qua token instruction phổ biến (cần CỤM), đa nghĩa "căng thẳng", gold khóa-bài. Chi tiết TK_004 mục 5 |
| 2026-07-02 | **TK_007 IMPLEMENTED (Claude)** — match_episode: idf-của-chính-bigram + case-fold + distinct-guard mức bigram; gold đa-đích | Essay `_t2`: **match 88 (7/8) · cov 79 · ord 88** — "phở bò"/"thể dục" ✓; ca còn lại (bánh mì kẹp→cookies) = câu hỏi trùng khung hiếm bài khác, cần NGHĨA cụm → TK_008. 277 pass. TK_008 (gate chủ đề + cụm↔cụm) & TK_009 (Trụ 4 pilot) đã có spec |
| 2026-07-02 | **TK_009 VÒNG 2-4 (Claude) — SINH TOKEN-LEVEL THÀNH DEFAULT CHO ESSAY** — bóc 3 lớp bug đầu-segment (backbone "đạt được"×G4 −inf×is_surface("Đáp:")=True) | vb_seg **100→34% THẬT**, đọc tay sạch xáo cả 8 ca ("Việc đọc sách" ✓), cov/ord/match giữ 79/88/88 → **flip `essay_segment_mode="ar"`**. Lần đầu đường mặc định một chế độ sinh là TOKEN-LEVEL. 279 pass. Giá: chậm hơn (sổ nợ). Nấc sau: agenda mức Ý + trộn resonance vào agenda |
| 2026-07-02 | **TK_009 Trụ 4 PILOT (Claude)** — `essay_segment_mode=ar`: segment sinh token-level bằng A3 (member=agenda, fallback verbatim); metric `vb_seg` | A/B: vb_seg **100→4%**, cov/ord/match giữ 79/88/88 — số đạt acceptance NHƯNG đọc tay = **verbatim-xáo-đầu-segment** ("Việc đọc"→"đọc Việc", hệ thống; index-backbone bỏ agenda[0] không reachable từ w0). KHÔNG nhận là "máy sinh" — default verbatim; việc kế: vá trật tự đầu-segment A3 + ablation order2/markov. 279 pass |
| 2026-07-02 | **TK_008 (2) phrase_assoc (Claude)** — cạnh cụm↔cụm PMI đồng-episode, 3 vòng (band df, filter nối-chuỗi giữ paradigm cùng-vị-trí) | **108k cặp trên `_t2`, semantic thật: "thể dục↔tập luyện" 5.7, "đọc sách↔nghe nhạc"** (đúng "mắt sáng↔tinh anh"); chưa flip ca thật: "mì kẹp" df<3 (trần DATA — sổ nợ scale), "tập trung" đa nghĩa. Essay giữ 79/88/88; 278 pass. T3 có nền |
| 2026-07-02 | **TK_008 (1) gate chủ đề resonance (Claude)** — member phải chia ≥1 bigram MỚI ngoài câu hỏi với thân bài | mix 62→38; chặn 2/4 rác (sandwich Đức, Israel); giữ mix tốt (dinh dưỡng); **1 lọt** ("tập trung quân đội" lách qua bigram chức năng) → trần mặt-chữ, cần phần (2) phrase_assoc. cov/ord/match giữ 79/88/88; 277 pass |
| 2026-07-02 | **TK_005+006 (Cursor code, Claude review+đo)** — match bigram liền kề; phát resonance khi sinh. Review sửa 3 lỗi: eval min_score lệch generate; `mixed` proxy sai (che mix=0%); cap segment nuốt slot resonance → đặt chỗ + min 0.10→0.07 | Essay `_t2`: **cov 79 / ord 88 / match 75 / mix 62**; 2 ca fail cũ PASS; guard giữ; 276 pass. Regression mới: "phở bò"→bài bí đỏ (bigram generic "hướng dẫn nấu" — việc tiếp: df bigram); trộn 3/5 lạc chủ đề bề mặt ("tập trung QUÂN ĐỘI") → sổ nợ TK_006 §6, fix = gate chủ đề lúc bind (T3) |
| 2026-07-03 | **TK_010 IMPLEMENTED+ĐO (Claude build, Cursor đo+vá)** — t1 = cha THẬT (`fact_links`, `t1_agenda`, `build_t1_unified`→`_t3`); `match_t1_parent` (bigram-df TK_007); vá `eval_essay` unified | `_t3`: QA **100** ✓; essay **75/88/88** (match/ord = baseline; cov −4pp); vb_seg 96% (agenda verbatim — sổ nợ metric); guard big_a2 **100/88/67/comp64-2-83** + grade_v3 giữ; **282 pass**. Ca cookies giữ (TK_008). Chi tiết `TK_010_T1_UNIFIED_PARENT.md` §5 |
| 2026-07-03 | **TK_012 IMPLEMENTED (Luật 0.0 — mạng thuần nơ-ron)** — tách text/label sang `content_cache` sidecar (`checkpoint_io`); gỡ `tag/key/title/source` khỏi graph; Vocab precompute `is_punct/cap_class/fold_group_id`; `phrase_assoc` khóa fold-group-id | **Inference + training blind với cache** (test chứng minh graph/output bất biến); guard 3 checkpoint GIỮ NGUYÊN (`_t3` 100/75/67/comp80-0-90/ana50; `_t2` 100/88/67); **292 pass**. Auto-migrate checkpoint cũ |
| 2026-07-03 | **TK_011 Phase A (observability)** — `util/progress.py` (`stage`/`PerfSummary`), `eval_v4` flags `--verbose/--report/--fast/--skip-section` + fix triple-pass hybrid, `scripts/verify.py`, pytest `slow` marker | verify có log section/case + JSON timing; `-m "not slow"` 34s vs 56s. Perf summary lộ điểm nóng: load 108MB (~15s) + essay AR (~51 phút/run) → Phase B (G4 candidate-only) nhắm vào đây |
| 2026-07-03 | **A/B vb_seg `_t3`** (`essay_ab_t3.py`, trả lời câu hỏi TK_010B của Claude) | verbatim vb_seg **100%** vs ar **96%** (Δ+4%, cov/ord/match giữ 75/88/88) → vb_seg cao KHÔNG phải artifact verbatim; **`_t3` đang TÁI TẠO nguyên văn, chưa SINH thật** — món nợ điều tra A3 grounding. `out/essay_ab_t3.json` |
| 2026-07-03 | **TK_013 IMPLEMENTED (ISG — phân rã tăng dần qua link ngang)** — `seg_graph.py`; online bigram→cluster+rewire; `rebuild_seg_graph.py` + `FileProgressLog`; checkpoint `_t3_isg.pkl` (40k clusters, `ở→đâu` power=19) | eval fast ISG: QA **100/100**, Para **88%**, Syn **67%** (ngang guard); Graph UI `cl:*`/`seg_h`; **296 pass**. `TK_013_INCREMENTAL_SEG_GRAPH.md` |
| 2026-07-04 | **TK_014 IMPLEMENTED (Claude) — kích thích cụm WH + anchor gom** — debug "ở đâu" không kích hoạt: gate `phrase_df≤8` loại cụm phổ biến. Cờ `match_common_clusters` (bỏ hard-gate, IDF cân bằng + distinct-ratio² đã có); `anchor_members`/`anchor_left_fillers`/`query_anchor_clusters`; viz node cụm `activated` + `anchor_groups` | A/B **0 flip** (grade_v3 21 + ISG QA 7); full `--fast` grade_v3 **100/100/62/50** + ISG **100/100/88/67** = baseline (trung tính); `ở đâu` gom 17 cha → filler {Việt Nam,Cà Mau,Amazon,Hồ Tây,nhà Nguyễn}; **298 pass**. Sổ nợ: overlap "xảy ra ở đâu"; answer-type Wq/Wk (doc03) = phase sau. `TK_014_WH_ANCHOR_ACTIVATION.md` |
| 2026-07-09 | **Phiên đo hội tụ (Claude): lever KHÔNG phải VarLM/decode mà là RETRIEVAL** — 4 phép đo: (1) KN smoothing **âm** (+1.26 bits, ruler leaky); (2) order-2 trung tính (cũ); (3) **gen-ar = gen-hy ×3 checkpoint** (grade_v3/big_a2/v7, vb-hy=0%) → chẩn đoán "W4 verbatim máy-tra-cứu" KHÔNG còn đúng, decode-flip lợi ích thấp; (4) **drift TĂNG theo scale** (v7 142k cha lạc nặng hơn). `out/tk031_kn_ab.log` | Kết luận: micro-fluency/decode capped; đòn bẩy = **chọn-đúng-fact** (TK_021/024/W3-entity) |
| 2026-07-09 | **TK_024 C scale + TK_035 v10 paragraph stack** — 160 bài wiki, 19k câu; +25k parents đoạn; VarLM cluster-aware; guard 86/88/67 | `network_v10_histwiki_paragraph.pkl` + `.varlm.pkl`; `paths.py` default v10 |
| 2026-07-09 | **HỢP NHẤT sinh A3 multi (TK_019 §7)** — xóa hybrid/essay/composed; paragraph pool always-on; UI/eval một mode | `generate()` duy nhất; 131 test; `eval_discourse_prompts.py`; xóa `eval_v4` CLI, `eval_paragraph_ab`, `essay_ab_t3` |
| 2026-07-09 | **Discourse eval corpus-derived** — bucket history/doc/conv/common_noun/frame; varlm sweep | `out/tk035_discourse_prompts_*.log` |
| 2026-07-09 | **Dọn test legacy** — episode/SFT/tk010, migration checkpoint cũ, hf_corpus skip | 131 pass; `out/pytest_collect.log` |
| 2026-07-09 | **TK_024 nhánh C** — fetch history wiki nguyên đoạn → `network_v9/v10_histwiki`; theme_neuron A1; guard 86/88/67 | `TK_024 §8` |
| 2026-07-17 | **TK_041 BPE — đổi ĐÁY** (tự train, Luật 0 sạch; tầng trên giữ nguyên) | Chuỗi **+7.8%** (KHÔNG ×2-3 — tiếng Việt âm tiết); vocab **126 516 MỞ → 32 002 ĐÓNG**; fertility **1.195**; roundtrip **byte-exact 65 160/65 160**. Tách tiếng Việt **tốt hơn GPT-4 1.8×** (1.256 vs 2.290 tok/từ) với vocab nhỏ hơn 3×. **53% vocab cũ là hapax**; **14.4% vocab là số** (`1288`⊥`1287` ⇒ mạng KHÔNG có cơ quan học số). `\n` nay là token. **Trả xong nợ Luật 0.0** (`FOLD_STRIP`, `endswith(":")`). Verify 17/17; 245 test |
| 2026-07-17 | **TK_042 G0 — THƯỚC ĐO NHẦM ĐẠI LƯỢNG** (xem đính chính Trụ 1) | `bits`=log2(rank), không phải CE; bỏ qua ca thất bại (**9.0%** OOV); **hạng ⊥ calibration** ⇒ mù với thứ việc SINH cần. → `scripts/eval_bits.py`. **Baseline THẬT** (base-LM, không agenda): `v10_histwiki` **8.173 bits/token · 1.288 bits/byte** |
| 2026-07-17 | **TK_042 G7 — TRANSFORMER vs ĐẾM (cùng corpus/tokenizer/held-out)** | Transformer 11.4M params, **30,000 steps**, MPS: **1.007 bits/byte** vs đếm **1.276** → Δ=**−0.269 >> 0.1** (21% tốt hơn) → **DƯƠNG TÍNH. GAP THẬT. B/C CÓ NGHĨA.** Train/eval gap +0.811 bits/token > 0.5 threshold ✓. Transformer học được ngữ cảnh dài; đếm không. Commit SHA 15e3cc5. Log: `out/g7_nohup.out` |

### Trụ 1 — số liệu THẬT (thước đo nghiêm, min_cov=0.6, 2026-07-01)

`retr` = best_match chọn đúng cha · `gen-hy` = đường mặc định (dội verbatim) · `gen-ar` = A3
token-level · `vb` = verbatim_rate.

**`grade_v3.pkl`** (1043 cha):

| Bộ | retr | gen-hy | vb-hy | gen-ar | vb-ar |
| :--- | :---: | :---: | :---: | :---: | :---: |
| QA val | 100% | 100% | **100%** | 71% | 14% |
| Paraphrase | 75% | 75% | **100%** | 62% | 62% |
| Synonym | 50% | 50% | 67% | 33% | 17% |

**`grade_v3_big_a2.pkl`** (44133 cha):

| Bộ | retr | gen-hy | vb-hy | gen-ar | vb-ar |
| :--- | :---: | :---: | :---: | :---: | :---: |
| QA val | 100% | 100% | **100%** | 86% | 14% |
| Paraphrase | 62% | 62% | **100%** | 62% | 62% |
| Synonym | 33% | 33% | **100%** | 33% | 67% |

**Kết luận Trụ 1 (chứng minh chẩn đoán):**
1. **"100/100/50" là thổi phồng.** Nghiêm IDF: Paraphrase tụt 100→75% (nhỏ) / 88→62% (big) — 2–3
   false-pass bị chặn (vd "Một ngày có 24 giờ" cho hỏi "bao nhiêu ngày").
2. **gen-hy ≡ retr, verbatim ~100%** ⇒ đường mặc định **là máy tra cứu**, không sinh.
3. **gen-ar (A3) verbatim thấp hơn nhiều** (14% QA) ⇒ nó **thật sự sinh** token-level; độ đúng
   thấp hơn nhưng đó là chất lượng SINH thật, chưa bị verbatim che.
4. Mục tiêu Trụ 2–4 giờ **đo được**: kéo gen-ar lên ngang/vượt gen-hy trong khi verbatim thấp.

### Trụ 2 — successor bậc-2 từ chuỗi cha: KẾT QUẢ ÂM TÍNH (2026-07-01)

Thêm kênh `seq_next` (bậc-2 suy từ chuỗi con của cha, backoff bigram — `parent_store.seq_next`,
cờ `ar_seq_order2_weight`). **Đo:** A/B on/off trên 5 prompt + eval — **giống hệt từng token**,
metric **trung tính**. Nguyên nhân: A3 bị **index-backbone + agenda + spreading chi phối**; `micro`
chỉ phá thế hòa, mà các ca này không có thế hòa. Word-salad README mô tả đã được index-backbone
xử lý từ trước. → **Mặc định TẮT** (không bật cơ chế chưa chứng minh — chống "ảo tưởng"); giữ code
cho ablation + Trụ 4. **Bài học:** lever của gen-ar KHÔNG phải micro-fluency — mà là chất lượng
**agenda** (nội dung) + focus. → chuyển hướng sang W3.

### Trụ 2 — R-slot: THÍ NGHIỆM RẺ → KẾT QUẢ ÂM TÍNH, HOÃN (2026-07-01)

Trước khi train lại (đắt), test giả thuyết "vị trí giúp sắp trật tự tốt hơn bigram" bằng bài toán
reorder (xáo túi token đáp → sắp lại), suy stats từ chuỗi cha, KHÔNG train lại:

| Bộ | bigram-only | + prior vị trí |
| :--- | :--- | :--- |
| Câu ngắn QA/compose (n=15) | exact 87%, pair **98%** | +0 |
| Câu dài ≥8 (wiki, n=400) | exact 16%, pair 61% | **tệ hơn (11%/56%)** |
| Câu token-lặp (n=400) | exact 5%, pair 49% | **tệ hơn (4%/46%)** |

**Kết luận:** (1) câu ngắn — thứ hệ thật sự sinh — bigram đã giải ordering (98%), R-slot vô ích;
(2) câu dài có khoảng trống ordering thật nhưng prior vị trí + trigram bậc-2 **không lấp, còn hại**;
(3) cơ chế R-slot thật (rel-pos × content, C_frame) chỉ kiểm chứng được sau **train lại** — chi phí
cao, lợi ích suy đoán; (4) hệ hiện KHÔNG sinh câu dài token-by-token (composition ghép nguyên khối
fact đã đúng trật tự) → khoảng trống câu-dài **không cắn**. → **HOÃN R-slot; chỉ hồi sinh khi cam
kết Trụ 4 (sinh token-level mặc định) + nhắm văn dài/essay.** Thí nghiệm rẻ chặn 1 lần train vô ích.

### Trụ 3 (phần W3) — df TOÀN CỤC cho focus: THẮNG THẬT (2026-07-01)

Gốc Cao Bằng: `_df` chỉ đếm curated → thực thể chỉ-có-wiki có df=0 → bị lọc `df>0` khi chọn focus
→ focus trôi về "gì/là". **Vá:** thêm `_gdf` (df mọi nguồn) + `_gidf`, chỉ dùng cho `top_focus_tokens`
/`entity_register_ids` (KHÔNG đụng `_df` của retrieval — README 0.5.8). Thực thể wiki (gdf>0) giữ
lại, nhiễu chỉ-trong-câu-hỏi (gdf=0) vẫn loại, từ chức năng (gdf cao → gidf thấp) tự xuống hạng.

**Đo (`grade_v3_big_a2`, compose):**

| Câu | Trước | Sau |
| :--- | :--- | :--- |
| Cao Bằng là gì | *Tuy nhiên, nó có thể dùng cho bất kỳ tác phẩm…* (lạc) | **Cao Bằng là một tỉnh thuộc vùng Đông Bắc Bộ, Việt Nam … cách Hà Nội 279 km.** ✓ |

Retrieval giữ nguyên (dùng `_df`). Còn lại (v4 tiếp): anaphora "nhà/họ" hỏng ngữ pháp.

### Trụ 3 (W3) — chọn focus: 2 vá tiếp + composition VÀO EVAL (2026-07-01)

Đo trực tiếp lộ thêm 2 lỗi focus (không phải regression — có sẵn):
1. **Cụm câu-hỏi cướp focus:** `entity_register_ids` trả phrase_units **trước vô điều kiện** →
   "thế nào" (Re-Pair phrase, IDF cao vì hiếm) thắng "Nhà Trần" → drift sang "thế hệ/thế giới".
   **Vá:** xếp hạng MỌI đơn vị (cụm+token) CHUNG theo `IDF ÷ (1+vị_trí_đầu)` — câu hỏi tiếng
   Việt đặt thực thể ở đầu (topic), cụm câu-hỏi ở cuối. Thuần trật tự từ (Luật 0).
2. **Bleed qua token generic:** `primary=focus[0]` = token dẫn đầu (có thể generic "Nhà" chung mọi
   triều) → "Nhà Lý…" lọt khi hỏi "Nhà Trần". **Vá:** `primary` = focus token **phân biệt nhất**
   (gidf cao), không phải dẫn đầu → bắt buộc token đặc thù (Trần/Lý) có mặt.

**Composition vào eval_v4** (`eval/v4.py::eval_composition`, `data/.../composition.jsonl`,
`scripts/eval_v4.py`): đo **coverage** (group ý được phủ) + **redundancy** (segment trùng). Chế độ
ghép fact giờ **đo được** (trước đây chỉ ở Graph UI).

**Kết quả (`grade_v3`, mode Compose):**

| Câu | Trước (W3 chưa vá) | Sau |
| :--- | :--- | :--- |
| Nhà Trần thế nào | *Tiến hóa qua nhiều thế hệ… Dân số thế giới…* (lạc) | **Nhà Trần đánh thắng quân Nguyên… thay nhà Lý 1226… Trần Hưng Đạo là Hưng Đạo Vương** ✓ |
| Nhà Lý thế nào | *…Trần đánh thắng quân Nguyên* (bleed) | **Nhà Lý… 1009… Thăng Long là trung tâm quyền lực nhà Lý** ✓ |
| Composition (5 ca) | — | **coverage 100%, redundancy 0%** |

Retrieval (gen-hy/retr) **giữ nguyên** 100/75/50 — vá chỉ đụng đường composition/focus. **253 pass.**

### A2 — Anaphora: 2 bug ngữ pháp đã vá (2026-07-01)

Đo trên big_a2 lộ đại từ hỏng: `Nhà Lý… , **nhà** mở nền…` (định ngữ, không phải đại từ);
`… , mở nền văn hóa` (cắt cụt vô chủ ngữ). Ba vá (thuần chính tả/thống kê, Luật 0):

1. **Neo anchor đúng:** `best_anaphor_for_entity` neo vào token thực thể **phân biệt nhất** (Lý),
   không token dẫn đầu generic (Nhà) — nếu neo "Nhà" thì đại từ giống nhất là "nhà". (`generate_composed`)
2. **Gate định-ngữ-vs-đại-từ:** loại ứng viên có successor phần lớn **viết hoa / danh từ riêng**
   (nhà→Lý/Trần) — đo big_a2: pronoun nó/họ/chúng cap%≤0.07, "nhà"=0.16, "và"=0.21 → ngưỡng 0.12. (`discourse.py`)
3. **Không cắt cụt:** khi không có đại từ tin cậy → **giữ nguyên chủ ngữ** (lặp còn hơn mảnh cụt).

**Kết quả (`grade_v3_big_a2`):**

| Câu | Trước | Sau |
| :--- | :--- | :--- |
| Nhà Lý thế nào | *…, **nhà** mở nền văn hóa…* | *…, **họ** mở nền văn hóa Thăng Long, **họ** cho…* ✓ |
| Sông Hồng là gì | *…, **nhà** nuôi dưỡng…* | *…, **họ** nuôi dưỡng đồng bằng Bắc Bộ…* ✓ |

**Trần còn lại (data):** animacy — "họ" (người) cho thực thể vô tri (Huế/Sông) thay vì "nó".
Corpus anaphora HF là tin tức (đại từ người); thiếu antecedent "nó" gắn vật vô tri (README 0.6.7).
253 pass; grade_v3 composition giữ coverage 100% / redundancy 0%.

### Trụ 6 P0+P1 — Mở gold + THEME NEURON `Concept:entity` (2026-07-02)

**P0 (đo trước):** gold composition big_a2 mở rộng 5→**13 ca** (`composition.grade_v3_big_a2.jsonl`
— gold theo checkpoint qua `scripts/eval_v4.py::_gold_for`, giữ gold grade_v3 sạch). Mọi fact gold
đã **kiểm chứng tồn tại** trong checkpoint (dump `facts_about` trước khi viết — công bằng). Baseline
lộ đúng bệnh: **67% cov / 82% top**, drift hệ thống ở ca thực thể *có mặt nhưng không là chủ đề câu*
(Napoléon trong "chiến tranh chống Napoléon của vua Gustav…"; Mặt Trời làm tân ngữ; Biển Đông đồng
âm biển Baltic; Ai Cập không bao giờ là chủ ngữ).

**P1 (cơ chế):** `ParentStore.theme_neuron(entity)` (`core/parent_store.py::ThemeNeuron`) — vật chất
hóa "mọi fact VỀ thực thể" thành một nơ-ron: fact list + **trọng số theme = 1/(1+vị_trí_xuất_hiện)**
(topic-first tiếng Việt — thuần vị trí, cùng nguyên tắc Luật 0 với xếp hạng focus W3). Build **lazy
+ cache** trên checkpoint đã có (pattern `_ensure_seq_index` — không train lại). `facts_about`
đọc ứng viên từ theme neuron + nhân `(0.5 + w·w_theme)` vào điểm; cờ `HCAConfig.compose_theme_weight`
(0 = quét phẳng cũ). Chỉ đụng đường composed/AR-multi — **retrieval/gen-hy giữ nguyên từng số**.

**Vá kèm (đo lộ ra):** `discourse.strip_leading_entity` cắt cụt tên riêng khi register chỉ mang
một phần tên (cụm Re-Pair "Trần Hưng" cho "Trần Hưng Đạo") → "họ **Đạo** đánh thắng…". Vá: strip
nốt đuôi token viết-hoa liền kề; nếu cả fact là tên riêng → giữ nguyên chủ ngữ. (Thuần chính tả,
cùng lớp cap-gate.)

**Đo A/B (theme ON vs OFF, sau vá strip):**

| Gold | OFF | **ON** |
| :--- | :---: | :---: |
| big_a2 (n=13) cov/red/top | 67/0/82 | **69/2/84** |
| grade_v3 (n=5) | 100/0/100 | 93/0/100 |

Định tính ON sửa đúng lớp drift chẩn đoán: Napoléon → "trốn khỏi Elba… trị vì trăm ngày" (hết
"chiến tranh chống Napoléon của vua Gustav"); Mặt Trời → "họ là ngôi sao ở trung tâm… Hệ Mặt Trời";
Quang Trung/Lê Lợi/Sông Hồng đuôi wiki sạch hơn; Nhà Lý thêm "Lý Thái Tổ dời đô". grade_v3 −7pp là
**artifact chọn-gold**: ca Trái Đất chọn "Trục nghiêng bốn mùa" (đúng chủ đề, hợp lệ) thay "nhà kính"
(gold n=5 khóa theo selection cũ). **Giữ ON mặc định** (tiền lệ decompose 0.5.11: đúng kiến trúc +
hướng dương + không hại retrieval); lật lại = `compose_theme_weight=0`.

**Định lượng mỏng (+2pp) vì phần fail còn lại bị chặn bởi trần KHÁC (không phải ranking):**
1. **Đại từ giả "các"** (Nhà Nguyễn/Biển Đông, top 33–67%) — lọt cap-gate; cần inventory đại-từ
   sạch = đúng tường POS của P2 (§7.3 cũ).
2. **Token case:** "Trái Đất" (query/wiki) ≠ "Trái đất" (curated) → fact curated **không truy được**
   trên big_a2 qua focus "Đất". Trần tokenization (chưa quyết normalize — đụng Luật 0 nhẹ, thuần
   chính tả).
3. **Câu wiki rác kiểu từ điển** ("(chữ Nho : 陳興道 ; ?") lọt quality gate từ lúc ingest — theme
   boost câu định nghĩa đầu-câu nên lộ rõ hơn. Vá ở `data/quality.py` + re-ingest, không ở readout.

Test mới: `test_theme_neuron_*` (3), `test_strip_leading_entity_*` (2). **258 pass** (1 skipped).

### Trụ 7 T1 — Episode layer + bugfix ranking (2026-07-02, cuối phiên)

**T1 triển khai:**
- `AbstractNeuron` / `ParentStore.build_episodes_from_provenance` / `bind_episode_by_resonance` /
  `episodes_of` — serialize trong checkpoint.
- `scripts/build_episodes.py` — đọc `data/real_vi/statements_doc.jsonl`, nhóm `(doc_id, para)`,
  map câu → cha qua `_key_to_idx`. Output: `checkpoints/grade_v3_big_a2_t1.pkl`.
- **Số liệu build:** 9,241 episode (≥2 member), 38,563/45,000 câu match cha (86%), 27,008/44,133
  cha thuộc ≥1 episode.
- **Compose:** `pick_facts_coherent` + `HCAConfig.compose_episode_weight=1.0` — ưu tiên fact cùng
  episode khi ghép (chống drift đồng-âm khác chủ đề).

**Bug ranking (bổ sung sau T0):**
| Bug | Triệu chứng | Vá |
| :--- | :--- | :--- |
| Index prune cụm | "Người trị bệnh" → cha lạc | `rebuild_hierarchical_index` giữ flat `_index` |
| Curated khung `là ai` | "Napoléon là ai" → Trung thu (Hybrid) | `best_match` gate primary + index fallback |
| Token case | "Trái Đất"≠"Trái đất" | `case_equiv_ids` trong `facts_about` / `theme_neuron` |
| Đại từ giả `các` | định từ lọt cap-gate | successor overlap vị ngữ (`discourse.py`) |
| Wiki từ điển | câu `(chữ Nho : …` lọt ingest | `quality.py` — cần **re-ingest** để có hiệu lực |

**Tests:** 269 pass (1 skipped). Eval v4 mở rộng (`_gold_for`, anaphor, verbatim).

### Trụ 4 — HỢP NHẤT SINH VỀ MỘT MODE A3 (P1-α, 2026-07-03)

> Chỉ đạo tác giả: bỏ 3 mode (essay / ghép-fact / hybrid), quy về **MỘT chế độ A3** sinh từng
> token như LLM (agenda nổi từ đồ thị, sinh tiếp đến khi tự đóng/`max_len`). Kế hoạch duyệt:
> `/Users/dlv/.claude/plans/flickering-tickling-locket.md`. Đúng mục tiêu Trụ 4 (§3): "Cha truy
> hồi → agenda; readout token-level là bộ decode, G4+hub-guard bảo vệ; **bằng verbatim trên QA
> đã thuộc → gỡ nhánh verbatim làm default**".

**Làm (P1-α):** `hca/generate_hebbian.generate()` = điểm vào duy nhất — truy hồi (`best_match` /
`facts_about` / hub-guard) → `_SegmentPlan` (agenda) → `generate_autoregressive` sinh token
(KHÔNG dội verbatim). `generate_hybrid`/`generate_composed` = **shim** gọi `generate` (giữ chữ ký);
CLI repoint. `generate_autoregressive` nhận `plan` sẵn (segments↔agenda nhất quán). Lưới an toàn
`ar_verbatim_fallback` (cờ, chỉ crisp, mặc định ON): A3 phủ agenda < `essay_ar_min_cov` → dội đáp
truy hồi. Sửa **bất đối xứng truy hồi**: crisp trượt `best_match` → thử `facts_about` trước khi
buông W_seq. `generate_essay` CHƯA đụng (P1-β sau).

**ĐO (A/B unified `gen-hy` vs baseline §7.2; min_cov=0.6):**

| Checkpoint | QA (vb) | Para | Syn | Composition (cov/red/top) | Anaphor | So baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `grade_v3.pkl` | **100** (vb100→**29**) | 62 | 50 | 80/0/100 (n=5) | 100 | QA giữ, **sinh thật** |
| `grade_v3_big_a2.pkl` | **100** (vb100→**14**) | 88 | 67 | **58/0/92** (n=13) | 50 | QA/Para/Syn/anaphor **giữ** |

- **Thắng chính:** QA đã thuộc **BẰNG verbatim (100%)** trên CẢ hai checkpoint mà `verbatim_rate`
  rớt 100→14-29% ⇒ **đang SINH, không tra cứu** (đúng điều kiện mở "gỡ verbatim" của Trụ 4).
  Unified `gen-hy` (100) > raw `gen-ar` (86) nhờ hub-guard + facts_about-fallback + lưới.
- **SỔ NỢ — Composition big_a2 cov 69→58** (AR sinh phủ ít fact hơn nối-verbatim) NHƯNG
  **top 84→92, redund 2→0** (mạch lạc & đúng chủ đề hơn). Nghi: agenda đa-fact dài > `max_len=48`
  → cắt sớm (tune rẻ: nới max_len multi) + artifact ". ," (bridge sau period) — vá ở P1-β/P2.
- **SỔ NỢ — perf:** AR mặc định làm eval big checkpoint chậm hẳn (verbatim O(1) → AR
  O(max_len×vocab×spreading)); big_a2 full >14 CPU-phút (dùng `--fast`). Địa hạt TK_011.
- **297 test pass** (sửa 5 test khẳng định verbatim/markov/nối-tay → kiểm **hành vi sinh**; helper
  toy `_qa_state` nạp thêm W_seq qua `_g1_and_wseq` để A3 chảy như train thật).

**Graph UI (hợp nhất, 2026-07-03):** Parent Explorer nhánh A3 gọi `generate()` (không còn
`generate_autoregressive` raw) → đáp + trace token = engine hợp nhất; dropdown gọn còn **A3
(toggle 1 ý / nhiều ý) + Essay** (bỏ Hybrid/Compose — đã là tập con); default mode API =
`autoregressive`; bỏ dòng "Hybrid so sánh"; panel fact hiện khi A3+multi. `viz/server/main.py`,
`viz/web/src/{GraphExplorer.tsx,api.ts}`, `export_graph_v3.py`; `npm run build` (bundle mới).

**Còn lại:** P1-β (fold `generate_essay` vào `generate` — bỏ verbatim tails, đo essay `_t3_isg`
cov/ord/match/vb_seg); P2 (nới `ar_index_backbone` / cắm kênh Q·K doc 03 → verbatim_rate giảm
thêm, A/B); dọn dead code (`_markov_answer` + import `best_anaphor_for_entity`/`strip_leading_entity`).
Chưa commit (tác giả tự commit).

---

## 7. HANDOFF — trạng thái cuối phiên 2026-07-02 (đọc TRƯỚC khi làm tiếp)

> Phiên sau bắt đầu từ đây. Tóm tắt đủ ngữ cảnh để không phải đọc lại toàn bộ.

### 7.1. Đã hoàn thành & ĐO ĐƯỢC (phiên 07-01 → 07-02)

- **Trụ 1 (thước đo v4)** — `src/nn_aivie/eval/v4.py`, `scripts/eval_v4.py`, `tests/test_eval_v4.py`.
  Tách retrieval/generation + metric IDF nghiêm + `verbatim_rate` + composition (coverage/redundancy/
  **on_topic**) + **eval_anaphor**. Gold **theo checkpoint** (`_gold_for`): `composition.jsonl`
  (grade_v3, n=5), `composition.grade_v3_big_a2.jsonl` (**n=13, P0 07-02**), `anaphor.jsonl`.
- **Trụ 3/W3 (focus)** — `parent_store`: `_gdf`/`_gidf`, xếp hạng đơn-vị `IDF÷(1+vị_trí)`,
  `primary`=token phân biệt nhất, `facts_about` (cổng cụm liền kề + priority + dedup bỏ dấu).
- **A2 (anaphora heuristic)** — `best_anaphor_for_entity`: neo token phân biệt + cap-gate (≤0.12)
  + giữ chủ ngữ khi không chắc. **07-02:** vá `strip_leading_entity` cắt trọn tên riêng (hết "họ
  **Đạo** đánh thắng"); gate định từ `các` (successor overlap vị ngữ).
- **P1 Theme neuron (07-02)** — `ParentStore.theme_neuron` / `ThemeNeuron`: vật chất hóa
  `Concept:entity`, trọng số theme 1/(1+vị_trí); `facts_about(theme_weight=)`; cờ
  `HCAConfig.compose_theme_weight=1.0` (mặc định ON). Chỉ đụng composed/AR-multi; retrieval nguyên vẹn.
- **Trụ 7 T0** — E-episode intrinsic DƯƠNG, extrinsic Syn 33% bất động → ÂM TÍNH cho synonym-probe.
- **Trụ 7 T1 (07-02, cuối phiên)** — `AbstractNeuron`, `scripts/build_episodes.py`,
  `pick_facts_coherent`, `compose_episode_weight`. Checkpoint `grade_v3_big_a2_t1.pkl`: 9,241
  episode, 86% câu match cha. Graph UI: `AIVIE_CHECKPOINT=..._t1.pkl python scripts/run_viz.py`.
- **Bug ranking (07-02)** — (1) index prune cụm; (2) curated `best_match` gate primary entity +
  `_entity_parent_indices` fallback (Napoléon/Hybrid không còn Trung thu; Người dạy học→Thầy cô).
- **Bug triage (07-02)** — `case_equiv_ids` (Trái Đất≡Trái đất); `quality.py` chữ Nho/từ điển
  (cần re-ingest để có hiệu lực trên checkpoint).
- **Trụ 2 (order-2, R-slot)** — ĐO ÂM TÍNH → HOÃN. order-2 default TẮT.

### 7.2. Số liệu baseline hiện tại (min_cov=0.6, 2026-07-02, theme ON)

| Checkpoint | retr | gen-hy(vb) | gen-ar(vb) | Composition (cov/red/top) | Anaphor |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `grade_v3.pkl` | QA100 Para75 Syn50 | 100(vb100)/… | 71/62/33 | **93/0/100** (n=5; −7 artifact gold) | 100% (trivial — không có `,`) |
| `grade_v3_big_a2.pkl` | QA100 Para62 Syn33 | 100(vb100) | 86/62/33 | **69/2/84** (n=13; OFF: 67/0/82) | **50%** (3/6) |
| `grade_v3_big_a2_t1.pkl` | *(chưa eval đầy đủ)* | — | — | grade_v3 comp **80/0/90** (smoke) | 50% (smoke) |

Anaphor 50%: Nhà Lý/Trần→họ ✓, HCM→giữ chủ ngữ ✓; **Trái Đất/Nước/VN→họ ✗** (animacy).
Compose mode đúng entity cho Napoléon/Biển Đông sau Theme; Hybrid đúng entity sau bugfix curated gate.

### 7.3. ⚠️ P2 (anaphor học PMI) — THỬ & ÂM TÍNH, ĐÃ REVERT

Ý 1a (cạnh học thực-thể→đại-từ từ đồng-hiện, PMI): **đo 33% < 50% heuristic** → REVERT khỏi
đường mặc định. Lý do thất bại: **không tách được đại-từ khỏi từ-nội-dung đồng-hiện** (`Hồ Chí
Minh→"chủ"/"nãi"`) nếu thiếu POS; và base "họ" quá mạnh nên inanimate không lật. Cùng bức tường
"cần POS/pretrain" như synonym.
- **Method GIỮ LẠI (chưa xóa, KHÔNG wire mặc định):** `parent_store.build_pron_stats` +
  `entity_anaphor_pref`; `discourse.best_anaphor_for_entity(pref=...)` có param `pref` nhưng caller
  **không truyền** (generate_composed + eval_anaphor đã revert về không-pref).
- **Bằng chứng tín hiệu CÓ THẬT** (để lần sau thử lại đúng cách): đồng-hiện trong-cha
  Trái Đất→{nó23, đó9, họ2}, Nước→{đó3,đây3,họ0}, Việt Nam→{đó103,đây42,họ44}. Tín hiệu animacy
  đúng, chỉ THIẾU bộ lọc đại-từ sạch (POS-like) để dùng được.
- **Hướng sửa lần sau:** cần **inventory đại-từ SẠCH** (phân biệt đại-từ vs danh-từ đồng-hiện) —
  vd successor phải là VỊ NGỮ (là/có/được) chứ không phải danh từ; hoặc chấp nhận nới Luật 0.

### 7.4. TRẠNG THÁI GIT & TEST (2026-07-02, cuối phiên)

- **269 pass**, 1 skipped (sau T1 + bugfix ranking + bug triage).
- Checkpoint mới: `checkpoints/grade_v3_big_a2_t1.pkl` (episode layer — build bằng
  `scripts/build_episodes.py`, không train lại).
- Thay đổi chính chưa commit (tác giả tự commit): `parent_store.py` (T1, theme, ranking, case-equiv),
  `generate_hebbian.py`, `discourse.py`, `config.py`, `quality.py`, `eval/v4` tests, `build_episodes.py`.

### 7.5. VIỆC TIẾP THEO (theo ưu tiên)

1. ~~TK_002~~ **XONG 2026-07-02** (nhật ký "TK_002 IMPLEMENTED"): Syn 17→50, Para 88 giữ,
   comp top 83, episode_weight=0 (trung tính), 270 pass. Còn lại tách **`TK_003_ANAPHOR_ANCHOR_GOLD.md`**:
   (A) anchor họ-người phá chọn đại từ — 3 biến thể gate đã thử & đo tệ hơn, hướng mới: cap-share
   anchor → None sớm, hoặc anchor bằng CỤM; (B) refresh gold 2 ca Trái Đất/Mặt Trời (case-equiv
   tìm curated tốt hơn, gold khóa wiki cũ → cov 64 oan); (C) "sản/các" đo lại sau (A).
2. ~~Trụ 7 T2~~ **XONG (TK_004)** — essay cov 67/ord 75, 500 t1 SFT trên `_t2.pkl`. Việc nối:
   (a) `match_episode` theo ĐƠN VỊ CỤM (fail "đọc sách" — token instruction phổ biến lấn át);
   (b) metric on_topic riêng cho essay (metric composition không hợp — 41% oan);
   (c) đa nghĩa instruction ("căng thẳng" → bài chiến tranh — cùng trần T0);
   (d) scale essay 500→vài nghìn + T3 enrichment cụm↔cụm trên nền episode SFT.
3. **Re-ingest + train lại** checkpoint (quality gate chữ Nho/từ điển; token case normalize nếu quyết).
4. **Trụ 3 synonym — QUYẾT ĐỊNH TREO** (A/B/C/D) — chờ tác giả; T3 enrichment cụm↔cụm.
5. **P2 anaphor lần 2 / P3 SFT** — chờ inventory đại-từ sạch (POS) / gold đủ lớn.

### 7.6. Trần cứng đã xác nhận bằng đo (KHÔNG phá bằng kiến trúc thuần)

- **Synonym** (chữa≈trị): OOV/sai-nghĩa + corpus thưa ngữ cảnh → cần data hoặc pretrain (Trụ 3).
- **Animacy** (Huế→nó): tín hiệu đồng-hiện có, nhưng thiếu bộ lọc đại-từ sạch (POS) → cùng trần.
  Gate `các` đã vá (07-02); animacy vô tri (Trái Đất→họ) **chưa** sửa.
- **Ordering câu dài**: bigram đủ cho câu ngắn (98% pair); câu dài kém nhưng hệ chưa sinh câu dài
  token-by-token (ghép nguyên khối) → chưa cắn (R-slot hoãn tới Trụ 4).
- **Token case tokenization** ("Trái Đất"≠"Trái đất"): **vá đọc** `case_equiv_ids` (07-02); re-ingest
  normalize vẫn cần nếu muốn một token duy nhất trong vocab.
- **Đa nghĩa một-vector** (T0): "trị" bị "giá trị" nuốt, "bảo" bị "bảo tàng" — E tĩnh (window HAY
  episode) không tách nghĩa. Hướng thoát: đơn vị CỤM đa-token ("bảo vệ", "mắt sáng") ít đa nghĩa
  → cạnh enrichment cụm↔cụm (Trụ 7 T3), không phải token đơn.
