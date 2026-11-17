# RUNBOOK TK_044 P3.2 — kênh COPY trong sinh: argmax (baseline) vs Hopfield mềm (đòn bẩy)

⚠️ **Gated trên P3.1 qua cổng** (cạnh coref hợp lý — RUNBOOK_TK_044_P3.1.md §5). Design-ahead; chưa
chạy tới. Đọc TK_044 §2 (Hopfield=attention) + thảo luận "Hopfield: chất hay trang trí" (2026-07-22).

## 0. Mục tiêu

Cắm **kênh COPY** vào `generate_autoregressive`: ở slot-thực-thể, điền lại thực-thể MỚI đã bind (qua
coref) thay vì mosaic thực-thể TRAIN. A/B trên `bind_probe.jsonl` (thực-thể NGOÀI corpus) so nền P3.0
(copy 0/17).

## 1. RÀNG BUỘC Luật 0 (runtime) — kill-criteria, KHÔNG thương lượng

P3.1 dùng `cap_class`/`word-start`/string lúc **ingest** = HỢP LỆ (docs/study/09 §2: string OK ở
build). Nhưng **lúc SINH, kênh copy CHỈ được đọc TRỌNG SỐ đã học** — cạnh `coref`, `after_role`,
`E`, `entity_register_ids`. **CẤM** gọi `cap_class(...)`, `is_word_start(...)`, substring, `if
ner==...` để quyết token copy. Vi phạm điều này = hỏng Luật 0 (docs/study/09: cấm string-rule ở
`generate_hebbian`). → Nếu bản hiện thực cần string lúc sinh để chạy = **DỪNG, thiết kế lại**.

## 2. Ba việc mỗi bước sinh (xen như POC P1)

1. **Copy-slot?** left-context hiện tại có dự đoán slot-thực-thể-đa-filler không — dùng W_seq /
   `entity_attention_read` (Q·K sẵn có), KHÔNG cap_class.
2. **Route + fill:** slot → (coref) → antecedent-SIG → filler = thực-thể trong prompt
   (`entity_register_ids`) → **copy** token-id.
3. **role-backoff:** sau khi copy tên OOV (W_seq(tên) rỗng) → `after_role[SIG]` (token cấu-trúc kế).
   Xen: copy-slot → role-backoff → W_seq (đúng thứ tự P1).

**Lưu ý runtime:** lúc sinh chỉ có **left-context** (right chưa phát). SIG-phải chỉ dùng lúc INGEST
để đúc cạnh coref đúng khuôn (P2: "ăn kẹo,"≠"tặng Y kẹo,"); lúc sinh, left-context + cạnh coref (đã
mã hoá phân biệt đó) là đủ — như POC.

## 3. PHƯƠNG ÁN A (khuyến nghị) — truy hồi MỀM kiểu Hopfield

Ta đang có nửa "trọng số Hebbian" nhưng vận hành bằng **argmax cứng** (khớp SIG chính xác) → GIÒN với
BPE-fragment (đã thấy P3.1). Hopfield hiện đại = **truy hồi từ cue một phần**: đúng thuốc.

- **SIG-embedding:** `emb(sig) = normalize(mean(E[t] cho t ∈ left, t≠sentinel))` — vector từ E (đã có).
- **Index (build lúc load, từ `coref`):** K = ma trận `emb(source_sig_a)` mọi cạnh; lưu song song
  `coref[a]`, `after_role[a]`.
- **Truy hồi lúc sinh:** q = emb(left-context hiện tại); `w = softmax(β · K·q)`; lấy cạnh khối-lượng
  cao (hoặc chồng chập top-k) → antecedent → filler. **Khớp SIG GẦN ĐÚNG** (chịu BPE cắt lệch) + tie
  mềm tự nhiên (P2 phải vá tie-break cứng — đây là cách nguyên tắc).
- **= attention {Q=emb(left ctx), K=emb(source-SIG), V=filler-của-antecedent}, trọng số Hebbian, β
  điều nhiệt.** KHÔNG backprop → Luật-0-native. Đây là nửa "động lực" Hopfield còn thiếu.

## 4. PHƯƠNG ÁN B (baseline trước) — argmax cứng

`coref.get(sig_left_exact)` tra chính xác (dict đã có). Rẻ, chạy ngay sau P3.1. **GIÒN** với BPE
(SIG lệch 1 token → miss). Dùng làm **MỐC**: đo B trước → A phải VƯỢT B trên graph BPE-nhiễu mới
đáng độ phức tạp. (Measure-first: đừng nhảy thẳng A mà không có số B.)

## 5. A/B đo (harness P3.0 sẵn) — `--copy-channel`

`bind_probe.jsonl` (Krel/Mivor… NGOÀI corpus). Cấu hình `HCAConfig`: `ar_copy_channel ∈ {off, hard,
hopfield}` + `ar_copy_beta`. Chạy `probe_binding_graph.py` mỗi chế độ:

| Đo | Cổng P3.2 |
|---|---|
| single-copy (novel) | off ≈ 0 (nền) → hard > 0 → **hopfield ≥ hard** |
| drift (đọc tay, G15) | KHÔNG tăng so nền |
| guard essays bpb | Δ ≤ +0.05 (không hại fluency) |
| grounding cũ (`probe_retrieval_compose_v11`) | KHÔNG tụt (binding không phá grounding) |

**Cổng chốt:** hopfield-copy single novel > nền rõ + drift không tăng + 0 regression guard/grounding
+ `pytest -m "not slow"` xanh. Đạt → crossed/đại-từ (P3.3). Không → số + đọc tay, xem gãy đâu.

## 8. Kết quả Plan B (hard argmax, 2026-07-22) — smoke checkpoint + novel entity probes

**Checkpoint:** `network_v11_coref_smoke_v3.pkl` (2000 records/nguồn, 11,673 coref edges)

```
Plan B (--copy-channel hard): single 0/9 · crossed 0/4 · pronoun 0/4
P3.0 baseline (off):          single 0/9 · crossed 0/4 · pronoun 0/4
```

**Phân tích:** Plan B = exact SIG match. Novel entity probes ("Krel Zynq-47 rất chăm học .")
tạo left-context SIG KHÔNG khớp bất kỳ SIG nào trong training coref (viwiki/history/conv).
Lý do:
1. Smoke checkpoint chỉ có 2000 records/nguồn → SIG pool nhỏ, ít pattern cấu trúc chung.
2. Novel entity prompts dùng khuôn đơn giản ("X rất chăm học .") KHÔNG xuất hiện trong
   training text → _coref_left_set không có match → boost không bao giờ fire.

**Đây là kết quả đúng** — runbook §4 ghi "GIÒN với BPE (SIG lệch 1 token → miss)". Plan B
cần (a) full corpus coref (`network_v11_coref.pkl`) để có nhiều SIG hơn VÀ (b) test items với
khuôn cấu trúc khớp training (ví dụ: Wikipedia-style "X (tên khác) là một...").

**Bước kế:** chạy FULL corpus coref để có SIG pool đủ lớn, HOẶC chuyển Plan A (Hopfield
soft — robust BPE, không cần exact match). Nếu A > B=0 trên novel probes → tiêu chí đạt.

**Diagnostic (2026-07-22):** `entity_register_ids("Krel Zynq-47 rất chăm học .")` trả về:
```
['K', 'rel', 'ĠZ', 'yn', '-', '4', '7', 'q', 'Ġrất', 'Ġchăm', 'Ġ.', 'Ġhọc']
```
12 tokens — bao gồm cả functional words ("Ġrất", "Ġchăm", "Ġhọc"). Tại step đầu tiên
generation (pos=12), `_runtime_left_sig(12)` = 4 sentinels `(-1,-1,-1,-1)` vì 4 token cuối
cùng của prompt đều trong entity_ids_set. Không có coref key `(-1,-1,-1,-1)` → không fire.
Coref keys với sentinel: 2841/11673 (24%); sample: `(-1,-1,-1, 260)`, `(-1,-1,-1, 271)` =
3 sentinels + 1 real token. Plan A (Hopfield) — tiếp cận đúng: không cần exact match.

## 6. Trần & kill-criteria

- **Runtime-string = DỪNG** (§1) — ràng buộc cứng.
- A không vượt B trên graph thật dù POC hứa hẹn → ghi số, giữ B, đừng ép A.
- Binding-copy ≠ văn phong (P4). P3.2 chỉ là copy token đúng thực-thể.
- Probe phải NGOÀI corpus (đo bind, không phải trí-nhớ).

## 7. Plug point

`generate_hebbian.py:1268 generate_autoregressive` — thêm kênh copy cạnh `a_entity`/`a_micro`
(`a_final` blend). SIG-index build lúc `_ensure_*` / load. `HCAConfig` (config.py) thêm cờ.

## 9. M4 CÒN LÀM (2026-07-22, sau Plan B 0/9) — thứ tự có CỔNG CHẨN

§8 Plan B 0/9 lộ **2 vấn đề chồng**, không chỉ SIG-mismatch:
1. **SIG exact-match giòn** (đã lường trước §4).
2. **Sinh TRÔI khỏi thực-thể mới** (đọc `out/probe_bind_P32_planB.log`: "Krel…"→"IRC/Internet Relay
   Chat", "Noxel…"→"Acid cyanuric") — thực-thể mới không neo được sinh → không bao giờ có copy-slot để
   fire. **Kể cả Plan A cũng vô dụng nếu sinh đã trôi.**

Thêm nữa: khuôn probe cũ (`bind_probe.jsonl` Krel/Mivor) là **đồ chơi** ("X rất chăm học") — KHÔNG có
trong corpus → không SIG nào khớp. **Đã dựng `data/sft/bind_probe_real.jsonl`** (khuôn wiki/lịch-sử
THẬT: "X là một nhà thơ…", "X lên ngôi vua…", "X … Ông/Bà/Nơi này" + thực-thể Việt-MỚI).

**Việc 1 — full coref pass** (smoke 2000/nguồn → pool nhỏ):
```
PYTHONPATH=src python -u scripts/run_coref_pass.py --gidf-solo 6.0 \
    --out checkpoints/network_v11_coref.pkl --log-file out/coref_pass.log
```

**Việc 2 — re-run Plan B với khuôn THẬT trên checkpoint full** (CỔNG CHẨN):
```
PYTHONPATH=src python -u scripts/probe_binding_graph.py \
    --checkpoint checkpoints/network_v11_coref.pkl --copy-channel \
    --probe data/sft/bind_probe_real.jsonl --log-file out/probe_bind_P32_planB_full.log
```
→ **CỔNG:** đọc tay continuations:
  - **B > 0** (có copy) → cơ chế sống, chỉ giòn → **Việc 3 (Plan A mềm)**.
  - **B ~0 nhưng sinh BÁM khuôn** (nhắc thực-thể mới, chỉ chưa copy đúng chỗ) → Plan A đáng thử.
  - **B ~0 VÀ sinh vẫn TRÔI** (IRC/Acid…) → gãy ở **neo-sinh**, KHÔNG phải SIG → DỪNG copy-channel,
    xử "sinh bám thực-thể mới" trước (câu hỏi thiết kế lớn hơn — báo về, bàn hướng).

**Việc 3 (điều kiện) — Plan A Hopfield mềm** (§3): truy-hồi softmax trên emb-SIG. Dev sẽ soạn skeleton.

**Việc 4 — regression mỗi lần** (§5): `pytest -m "not slow"` + probe grounding cũ không tụt + guard
essays Δ≤+0.05.

**Cần dev (đã/đang làm):** `bind_probe_real.jsonl` ✅ · skeleton Plan A (khi tới Việc 3).

## 10. Review + vá 2 lỗi nghiêm trọng (2026-07-22, dev) — M4 pull rồi chạy Việc 1-2 §9

Review code Plan A/B của M4 (Luật-0 runtime: SẠCH ✓). Vá 2 lỗi khiến A chắc chắn 0/all:

1. **Plan A ngưỡng fire trên softmax-weight KHÔNG scale** → **cosine thô `dot.max()`** (`ar_copy_cos_thresh
   =0.5`). Self-test: match cosine 0.92 (fire) vs rand 0.22 (không) ở n=11k VÀ 100k; softmax-weight của
   match hoàn hảo = 0.008 (n=11k) / 0.001 (n=100k) — **cả hai < 0.02 cũ → không bao giờ fire**. Đây là
   lý do thật A sẽ 0, không phải Hopfield sai.
2. **`entity_ids` quá rộng** (M4 diagnostic: function word 'rất/học' lọt → SIG toàn sentinel) → **gate
   theo gidf** (`ar_copy_entity_gidf_min=3.0`, gidf=Luật-0 OK). Bỏ function word khỏi sentinel-SIG + boost.

+ `SIG_W`/`ENT` nay import từ `coref_pass` (1 nguồn, hết coupling ngầm). 245 pytest pass.

**CHƯA vá (ghi rõ, ngoài phạm vi baseline):** (a) chưa dùng `coref[sig]`(antecedent)/`after_role` → mới
là DETECT+boost-mọi-entity, chưa ROUTE (crossed X/Y chưa phân biệt, role-backoff chưa có); (b) K-matrix
dựng lại mỗi `generate()` (nên cache trên state ở scale). → làm khi single baseline dương.

## 11. Kết quả Plan B/A trên real khuôn (2026-07-22, ~19:40 UTC) — GATE THẤT BẠI

**Checkpoint:** `network_v11_coref.pkl` (full corpus: 500,276 coref edges, 294,178 records, 1.8GB)
**Probe data:** `data/sft/bind_probe_real.jsonl` (12 items: wiki-bio/place/org, history-king/battle, 
  kinship, pronoun-Ông/Bà/Nơi-này, control-TRAIN entity "Hà Nội")
**Items valid:** 11 novel (loại 1 control)

```
═══ BẢNG (chỉ item novel hợp lệ) ═══
Plan B (hard argmax):      single 0/6 · crossed 0/2 · pronoun 0/3 = 0/11
Plan A (hopfield cosine):  single 0/6 · crossed 0/2 · pronoun 0/3 = 0/11
P3.0 baseline (off):       single 0/6 · crossed 0/2 · pronoun 0/3 = 0/11

→ ĐỀU 0 — không cải thiện vs nền.
```

**Diagnosis từ logs:**
- **Entity được register** (`novelty=['reg=Y gidf=X']`): "Lữ Bằng Sơn" gidf=4.9 > 3.0 (pass gate ✓)
- **Nhưng sinh TRÔI** → "Lữ Bằng Sơn là một nhà thơ…" → sinh ra "Nguyễn Phi Phúc có tám người con…" 
  (hoàn toàn khác entity, kéo theo "Nguyễn Nhạc, Nguyễn Lữ, Nguyễn Huệ" — TRAIN entities)
- **Vậy không có copy-slot để fire**: kể cả SIG khớp (Plan B) hay cosine cao (Plan A) cũng vô dụng 
  vì sinh đã lựa chọn antecedent khác từ đầu.

**Root-cause**: Vấn đề không phải ở copy-channel (mechanism A/B không fire) mà ở **neo-sinh** 
(antecedent routing trong embedding-lookup ở bước sinh đầu tiên). Sinh-chain phát triển theo 
corpus patterns (TRAIN entities), không bị constrain bởi entity_ids bias.

**Gate decision**:
- ✗ **Điều kiện B > 0?** KHÔNG → Plan B mechanism cơ bản không work
- ✗ **A ≥ B?** Both 0 → Plan A không rescue
- ✓ **Drift?** YES — sinh kéo TRAIN entities (Nguyễn Phi Phúc, Acid cyanuric…)

**CỔNG THẤT BẠI** → không pass điều kiện "B > 0 hoặc (B~0+sinh-bám)". Ghi số: hệ thống sinh TRÔI 
khỏi thực-thể mới (neo-sinh issue). Copy-channel A/B chỉ có thể boost token khi antecedent đã lựa 
đúng — nếu sinh-chain chọn sai antecedent từ bước 1, copy-slot không bao giờ fire.

**Hành động kế:**
1. **Tạm dừng Plan A/B dev** (không thể cứu nếu neo-sinh gãy).
2. **Diagnose neo-sinh**: embedding-lookup ở bước đầu géneration có constrain entity_ids không? 
   Nếu không, thêm entity_ids bias vào scores TRƯỚC copy-slot (P3.3 design).
3. **Alternate**: nếu entity_ids bias đã có nhưng yếu, tăng độ khoảy (binding-factor) — thử 
   `ar_binding_boost` tăng từ 15.0 → 50.0 để xem sinh có bám entity_ids không.

→ Kịch bản này là "copy-channel chưa sử dụng được vì nền chưa sẵn sàng". Lưu y: binding 
(antecedent selection) ≠ copy (token-filling); binding phải OK trước.

**Data thêm:** logs đầy đủ ở `out/probe_bind_P32_{hard,hopfield}_real.log` (20-30 phút chạy mỗi); 
SIG-analysis ở `out/dump_coref_edges_v11.log` (khi chạy dump script — chưa chạy).

## 12. Diagnostic plan-vs-score (2026-07-22) — CHỐT trước khi quyết hướng

§11 gate fail (sinh trôi). Cần tách: drift ở tầng PLAN (agenda đã route train-entity từ đầu → copy-score
KHÔNG cứu) hay SCORE (agenda đúng, chỉ token lệch → boost đáng thử)? `scripts/probe_plan_agenda.py`:
```bash
PYTHONPATH=src python -u scripts/probe_plan_agenda.py \
    --checkpoint checkpoints/network_v11_coref.pkl --probe data/sft/bind_probe_real.jsonl \
    --log-file out/probe_plan_agenda.log
```
Đọc `plan.AGENDA` mỗi prompt:
- chứa **train-entity** (Nguyễn Lữ/Tây Sơn…) thay thực-thể-mới ⇒ **drift PLAN-level** ⇒ copy-channel
  (score) vô ích; M4 #3 (tăng boost) KHÔNG cứu — phải xử retrieval/plan hoặc đổi hướng (§11: 2 hoặc 3).
- **rỗng / đúng thực-thể-mới** nhưng sinh vẫn lạc ⇒ score-level ⇒ boost đáng thử.

(Dự đoán: PLAN-level — novel "Lữ Bằng Sơn" chia token `Lữ` với train "Nguyễn Lữ".) Dán log về.
## 13. Kết quả diagnostic plan-vs-score (2026-07-22, ~20:20 UTC) — PLAN-LEVEL DRIFT

**Chạy:** `probe_plan_agenda.py` ⇒ dò mỗi prompt: entity_register, plan.entity_ids, plan.AGENDA, facts_about#1

**Pattern đều:** 
```
[pronoun] 'Lữ Bằng Sơn là một họa sĩ tài năng . Ông'
  entity_register : 'Lữ Bằng họa sĩ...' ← entity-mới từ prompt ✓
  plan.entity_ids : 'Lữ Bằng họa sĩ...'
  plan.AGENDA     : 'Trong những năm đầu tiên, Nguyễn Nhạc đóng vai trò...' ← TRAIN entity (corpus)
  facts_about#1   : 'Lịch sử: Tiêu đề: Nguyễn Nhạc...'
```

**Kết luận:**
- ✓ entity_register cấp đúng: "Lữ Bằng Sơn" detected từ prompt
- ✗ plan.AGENDA **đã route TRAIN entities** từ đầu (agenda retrieval khéo): sinh-chain phát triển từ 
  {Nguyễn Nhạc, Tây Sơn, Nguyễn Ánh...} thay vì {Lữ Bằng Sơn}
- ✗ **Không thể fix bằng copy-channel Plan A/B**: copy chỉ có thể boost token khi agenda đã mention 
  entity-mới — nếu agenda chỉ mention TRAIN entities từ đầu, copy-slot không bao giờ có cơ hội.

→ **Root-cause: PLAN-LEVEL (agenda routing), KHÔNG phải SCORE-LEVEL (token-copy)**

**Gate Decision:**
- ✗ **Điều kiện copy-channel nền?** KHÔNG — Plan A/B phải dừng vì không thể fix PLAN-drift.
- ✗ **Có cứu được bằng ar_binding_boost (Option A §11)?** Không — binding-boost chỉ giúp step 1 
  token lựa chọn; binding-chain phát triển theo sinh logic, không bị constrain bởi single-step bias.
  Lỗi là agenda retrieval thứ, không phải binding yếu.
- ✓ **Drift xác nhận ở retrieval/plan (P3.3+ scope)**, không phải P3.2.

**Dừng P3.2**: Copy-channel A/B là dead-end với hiện tại. Yêu cầu fix agenda routing trước (P3.3 
design issue hoặc retrieval-overfitting).

**Data:** `out/probe_plan_agenda.log` (11 items mỗi agenda routing xác nhận TRAIN entities).
## 14. Test CÔ LẬP (continuation-mode) — cơ chế copy có sống khi TẮT plan-drift? (2026-07-22)

§13 chốt plan-level drift. Câu hỏi cuối: copy-channel (đã chứng minh 100% ở POC) có fire + giữ
thực-thể-mới KHI bỏ nguồn drift (`_build_segment_plan`) không? `probe_binding_graph.py --plan-mode
continuation` dựng plan tay (agenda = content prompt, entity_ids = entity_register) → generate KHÔNG
truy-hồi fact train, copy-channel vẫn có entity để boost. KHÔNG đụng lõi.
```bash
PYTHONPATH=src python -u scripts/probe_binding_graph.py \
    --checkpoint checkpoints/network_v11_coref.pkl --probe data/sft/bind_probe_real.jsonl \
    --plan-mode continuation --copy-channel hopfield \
    --log-file out/probe_bind_P32_continuation.log
# so nền cùng mode:
PYTHONPATH=src python -u scripts/probe_binding_graph.py \
    --checkpoint checkpoints/network_v11_coref.pkl --probe data/sft/bind_probe_real.jsonl \
    --plan-mode continuation --copy-channel off \
    --log-file out/probe_bind_P32_continuation_off.log
```
**Verdict cuối:**
- copy > nền (giữ thực-thể-mới) ⇒ **cơ chế SỐNG, drift là lỗi PLAN** → mở câu hỏi "khi nào plan vs
  continuation" (nhánh dùng được: continuation-coreference).
- vẫn ~0 / sinh rỗng ⇒ **cơ chế không chuyển được sang graph thật** → chốt âm tính §7, tái dùng 500k
  cạnh coref cho coreference entity ĐÃ BIẾT. Dán 2 log về.

## 15. Kết quả test continuation-mode (2026-07-22, ~20:40 UTC) — CƠ CHẾ CÓ THỰC LỖNG HƠN DỰ KIẾN

**Chạy:** continuation-mode baseline (copy OFF) + continuation-mode hopfield (copy ON)

```
continuation OFF:  single 0/6 · crossed 0/2 · pronoun 0/3 = 0/11
continuation HOPFIELD: single 0/6 · crossed 0/2 · pronoun 0/3 = 0/11
```

**Diagnostic:**
- continuation OFF (entity_ids bias, no plan-drift): sinh ra "Vĩnh Dương", "Cù tác Minh: Thác" 
  (contain some entity tokens, nhưng fragmented — entity_ids bias YẾU)
- continuation HOPFIELD (copy-channel ON): sinh ra "CCC", ". . .", "con con con" (output lộn xộn, 
  HOÀN HẢO hơn OFF)

→ **Cơ chế copy không bị chặn bởi PLAN-drift; root-cause ở embedding-lookup/scoring lúc sinh**:
  - entity_ids bias không đủ mạnh → sinh không bám thực-thể-mới
  - Copy-channel hopfield **làm tồi hơn** (boost sai -> sinh chọn punctuation/padding thay token thật)

**Ghi chép chốt:**
1. Plan-drift không phải nguyên nhân chính (test cô lập vẫn 0/11)
2. Nguyên nhân: **entity_ids bias toàn cục quá yếu** (đặc biệt ở bước 2+)
3. Copy-channel A/B **không thể cứu** nếu entity không ứng viên hàng đầu từ đầu (thứ bậc đã sai, 
   copy-boost không sửa thứ bậc, chỉ tăng score)

**P3.2 verdict:** ✗ **CỔNG THẤT BẠI** — Copy-channel A/B là kiến trúc sai lầm cho bài toán này. Yêu cầu:
- P3.3: Binding-layer (entity_ids constraint mạnh, early-exit khi khớp) TRƯỚC sinh
- Hoặc: Re-ranking/antecedent-selection thay vì copy-boost
- Hoặc: Retrieval-constraint (agenda + facts chỉ include entity_ids)

**Data:** 
- `out/probe_bind_P32_continuation_off.log` — baseline (0/11)
- `out/probe_bind_P32_continuation.log` — hopfield (0/11, output lộn xộn)
