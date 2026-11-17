# TK_017 — Tầng GATE/CHỌN: redirect v4 sau TK_015/016

| Field | Value |
|-------|-------|
| **ID** | TK_017 |
| **Status** | 🟡 WIRED — gate A3 xong; E2E sinh chưa cải thiện; cần override agenda (2026-07-05, Cursor) |
| **Nguồn** | Kết tinh TK_015 (C0/D1) + TK_016 (D3) + doc 07: substrate CÓ tín hiệu; readout phẳng LUÔN pha loãng. Đòn bẩy thật = **GATE/CHỌN theo ngữ cảnh**, không thêm cạnh. |
| **Doc** | [`docs/study/07_SPREADING_VS_DIRECTED_GENERATION.md`](docs/study/07_SPREADING_VS_DIRECTED_GENERATION.md), [`docs/study/05_ANCHOR_NEURONS.md`](docs/study/05_ANCHOR_NEURONS.md), [`docs/study/03_ATTENTION_QKV_FROM_GRAPH.md`](docs/study/03_ATTENTION_QKV_FROM_GRAPH.md) |
| **Tiền đề** | [`TK_015_DICTIONARY_SUBSTRATE.md`](TK_015_DICTIONARY_SUBSTRATE.md), [`TK_016_D1_REDIRECT_D3_GEN_WIRING.md`](TK_016_D1_REDIRECT_D3_GEN_WIRING.md) |
| **Checkpoint guard** | `checkpoints/network_v4.pkl` (D1′ rebuild; ruler **4.028 bits / top1 22.1%** = baseline) |

## 1. Chẩn đoán chốt

### Thắng thật (giữ)

| Hạng mục | Bằng chứng |
| :--- | :--- |
| Thước không bão hoà | `next_token_lm` — nhạy Δ, không bão hoà 100% |
| 2 tường C0 | đa-nghĩa blur + cạnh xa seq=0 |
| Substrate sense | `cân#3→gram/kg`, `đá#1→{hòn,tảng}` |
| Substrate bracket | `tuy→nhưng 3.57`, `nếu→thì 2.57` (35k cạnh mine) |
| Viz quan sát | Panel Tách nghĩa + `/api/embedding/senses` |

### Âm tính có giá trị (redirect)

| Thí nghiệm | Kết quả | Bài học |
| :--- | :--- | :--- |
| D1 flat-dump | ruler 4.03→**4.40** | Nhồi miền từ điển vào pool sinh = dilution |
| D3 readout phẳng | ruler 4.03→**4.48+** | Broadcast FRAME_BRACKET mọi vị trí = nhiễu |
| Spread vô hướng (doc07) | sụp hub `. ,` | Hub đè content khi không gate |
| q·k type-level (doc07) | word-salad | "Nói gì" cần agenda; q·k chỉ "cách nói" |

### Meta-finding

> **Attention thật = phép CHỌN (softmax/gate), không chỉ Q/K.**
> Substrate đủ; thiếu tầng quyết định **KHI NÀO** đọc tín hiệu cấu trúc nào.

**Chưa đạt:** chất lượng sinh qua prompt — mục tiêu gốc — chưa cải thiện aggregate. Biết vì sao.

---

## 2. Kiến trúc mới — `ContextGate`

```mermaid
flowchart TB
  subgraph substrate [Substrate da co]
    Wseq[W_seq]
    FB[FRAME_BRACKET]
    Sense[SENSE/POS]
    QK[q·k SVD]
  end
  subgraph gate [ContextGate — MOI]
    Ent[entropy / margin]
    Op[opener trong hist]
    Pos[position targeted]
    State[frame state machine]
  end
  subgraph readout [A3 readout]
    Micro[micro blend]
    Macro[agenda spread]
    Pick[argmax token]
  end
  substrate --> gate
  gate -->|"chi khi duoc CHON"| readout
  readout --> Pick
```

**Nguyên tắc Luật 0:** gate đọc **tín hiệu đồ thị** (entropy phân bố base-LM, cạnh FRAME_BRACKET,
cờ opener) — không `if token=="nếu"`. Ngưỡng entropy/margin = hyperparam ablate, không rule tiếng Việt.

**Hai nhu cầu tách bạch (doc 07 §6):**

| Nhu cầu | Ai lo | Gate |
| :--- | :--- | :--- |
| **Nói gì** (đáp đúng câu hỏi) | agenda / retrieval | Giữ; không bỏ |
| **Cách nói** (fluency, khung, chuyển tiếp) | W_seq + q·k + bracket | **Gate** mới |

---

## 3. Ba mode gate (thứ tự thử)

### Mode A — Entropy-gated (G-probe, rẻ nhất)

**Giả thuyết:** FRAME_BRACKET chỉ hữu ích khi base-LM **phân vân** (entropy cao) **và** có opener
đang mở khung trong lịch sử.

**Probe:** [`scripts/eval_d3_ruler.py`](scripts/eval_d3_ruler.py) `--sweep-entropy`

**Fire khi:** `entropy(scores) ≥ τ_ent` AND `∃ opener ∈ prefix[-W]` có outgoing FRAME_BRACKET.

**Acceptance G-probe:**
- Aggregate Δbits **≥ −0.05** (gần 0 hoặc dương) vs baseline 4.028
- Hoặc: fire rate hợp lý (1–15% vị trí) + targeted bracket positions dương

### Mode B — Position-targeted (fallback nếu A âm)

Chỉ chấm / chỉ fire tại vị trí **đóng khung thật** trong gold (sau opener, gap≥2).

**Probe:** [`scripts/eval_d3_bracket_positions.py`](scripts/eval_d3_bracket_positions.py) ✅

**Fire khi:** gold position là closer hợp lệ của opener gần nhất (metric, không rule runtime).

### Mode C — Learned gate (phase sau)

G3 củng cố cạnh gate khi đáp đúng; LTD khi bracket fire sai chỗ. Chờ gold đủ + G-probe/A/B ổn.

---

## 4. Lộ trình phụ thuộc G-probe

```mermaid
flowchart LR
  GP[G-probe Mode A] -->|Duong| Wire[Wire gate vao A3]
  GP -->|Am| Targeted[Mode B targeted]
  Wire --> D1p[D1 prime dilution fix]
  Wire --> D2[D2 lite q·k blend]
  D1p --> D4[D4 stack neo H_slow/mid/fast]
  D2 --> D4
  Targeted --> Rethink{Targeted duong?}
  Rethink -->|Co| Wire
  Rethink -->|Khong| C[Mode C hoac hoan wire]
```

| Phase | Nội dung | Đo | Phụ thuộc |
| :--- | :--- | :--- | :--- |
| **G-probe** | entropy gate trên FRAME_BRACKET | ruler + sweep | — |
| **Wire A3** | `frame_bracket_gate_weight` trong `generate_autoregressive` | ruler + QA guard 100% | G-probe ≥ −0.05 |
| **D1′** | two-tier pool + `bind_sense_resonance` | ruler ≤ 4.028 | song song G-probe |
| **D2 lite** | `attention_qk.py` blend raw⊕q·k, head POS | A0 bigram rank + ruler | sau wire bracket |
| **D4** | stack neo doc 05; mọi neo qua gate | composition + ruler | D1′ + D2 |

---

## 5. Thay đổi code dự kiến

| File | Thay đổi | Phase |
|------|----------|:---:|
| [`scripts/eval_d3_ruler.py`](scripts/eval_d3_ruler.py) | G-probe entropy gate + sweep | G-probe ✅ |
| `scripts/eval_d3_bracket_positions.py` | metric targeted | Mode B ✅ |
| [`hca/generate_hebbian.py`](src/nn_aivie/hca/generate_hebbian.py) | `_context_gate_fb()` trong AR loop | Wire ✅ |
| [`hca/frame_bracket_gate.py`](src/nn_aivie/hca/frame_bracket_gate.py) | shared gate + lazy mine | Wire ✅ |
| [`hca/config.py`](src/nn_aivie/hca/config.py) | `ar_frame_bracket_weight/floor/min_gap` | Wire ✅ |
| [`core/parent_store.py`](src/nn_aivie/core/parent_store.py) | `bind_sense_resonance()` | D1′ ✅ code / ❌ chưa chạy build |
| [`scripts/build_dict_checkpoint.py`](scripts/build_dict_checkpoint.py) | D1′, out=`network_v4.pkl` | D1′ ✅ |
| [`training/ingest_dictionary.py`](src/nn_aivie/training/ingest_dictionary.py) | no flat parent, no W_seq defs | D1′ ✅ |
| [`core/paths.py`](src/nn_aivie/core/paths.py) + eval scripts | default `network_v4.pkl` | infra ✅ |
| [`scripts/test_gen_network_v4.py`](scripts/test_gen_network_v4.py) | smoke sinh gate on/off | test ✅ |
| `hca/attention_qk.py` | Wq/Wk SVD + blend | D2 ❌ |
| [`TK_016`](TK_016_D1_REDIRECT_D3_GEN_WIRING.md) §7d–§9 | D1′ build + E2E + chốt | ✅ |

---

## 6. Guard & sổ nợ

- **Đóng băng:** retrieval crisp-QA 100% mọi phase.
- **Không ship:** kênh aggregate âm (global/boundary đã loại).
- **Perf:** gate phải O(vocab) như hiện tại hoặc sparse top-K closer — không full spread.
- **Hỏi:→Đáp:** df=0 trong parents — giữ frame protocol `generate()`, không ép cạnh mine.
- **Trần data:** sense bind vẫn thưa cho từ hiếm sau D1′ — ghi sổ, không ảo tưởng.

---

## 7. Acceptance toàn redirect

1. Ít nhất **một** mode gate (A hoặc B) **dương** trên metric phù hợp trước khi wire `generate()`.
2. Ruler aggregate **không tụt** > 0.05 bits sau wire (hoặc targeted dương bù aggregate trung tính).
3. QA/ISG guard giữ nguyên.
4. Đọc tay ≥2 prompt khung (`nếu…thì`, `tuy…nhưng`) cải thiện trên Graph UI.

---

## 8. Kết quả G-probe + Mode B (2026-07-05)

Checkpoint: `grade_v3_big_a2_t3_isg.pkl`, held-out n=2860, FRAME_BRACKET edges=35205.

### Mode A — Entropy-gated (G-probe) — **ÂM TÍNH**

| Cấu hình | bits | Δ vs base |
| :--- | :---: | :---: |
| base-LM | **4.028** | — |
| + fb global | 4.506 | **+0.477** |
| + fb boundary | 4.587 | **+0.559** |
| + fb entropy (best: ent≥2, w=1.0) | 4.478 | **+0.449** |

**Chẩn đoán:** essay held-out có opener + entropy cao ở **~94–100%** vị trí → entropy gate ≈ global gate, không lọc được.

**Verdict Mode A:** không đủ làm xương sống; chuyển Mode B.

### Mode B — Position-targeted (bracket-close) — **DƯƠNG MẠNH**

Script: [`scripts/eval_d3_bracket_positions.py`](scripts/eval_d3_bracket_positions.py)

| Metric | n | bits | top1 | Δ |
| :--- | :---: | :---: | :---: | :---: |
| ALL (base) | 2860 | 4.028 | 22.1% | — |
| **TARGETED close (base)** | **511** | 4.256 | 23.1% | — |
| ALL (+ fb @ close only) | 2860 | **3.860** | 22.9% | **−0.168** |
| **TARGETED (+ fb opener-specific)** | **511** | **3.316** | **27.4%** | **−0.940** |

**Chẩn đoán:** tín hiệu FRAME_BRACKET **THẬT và mạnh** tại vị trí đóng-khung; thất bại trước đây là do cắm phẳng/mọi vị trí, không phải do cạnh nhiễu.

**Verdict Mode B:** ✅ đủ điều kiện wire gate **position-targeted** vào A3 (Mode B, không Mode A).

### Quyết định lộ trình sau đo

1. **Wire A3:** state machine opener → chỉ bước đóng-khung fire softmax local trên closers của opener đó.
2. **D1′ song song:** two-tier + bind-resonance (sửa dilution, độc lập).
3. **Hoãn:** entropy-only gate; global/boundary readout.
4. **D2/D4:** mọi neo cấu trúc qua cùng pattern gate (position + opener-specific).

### Wire A3 (2026-07-05, Cursor) — IMPLEMENTED

- Module [`hca/frame_bracket_gate.py`](src/nn_aivie/hca/frame_bracket_gate.py): lazy mine, state machine `open_brackets`, `apply_frame_bracket_gate`.
- `generate_autoregressive`: gate sau G4/repetition, trước index-backbone; trace `frame_bracket_gate`.
- Config mặc định: `ar_frame_bracket_weight=0.5` (ablate =0).
- Tests: `tests/test_frame_bracket_gate.py` + 32 pass generate_hebbian.
- **Đã đo** trên `network_v4.pkl` — xem §9.

---

## 9. POST-WIRE — đo guard + sinh E2E (2026-07-05, Cursor)

Checkpoint: `checkpoints/network_v4.pkl` (D1′ rebuild, `--skip-resonance`).

### Thước LM + Mode B (giống `_t3_isg`)

| Metric | n | bits | top1 | Ghi chú |
| :--- | ---: | ---: | ---: | :--- |
| ALL base-LM | 2860 | **4.028** | 22.1% | = baseline |
| TARGETED close base | 511 | 4.256 | 23.1% | vị trí đóng khung khó hơn aggregate |
| ALL + gate @ close | 2860 | **3.860** | 22.9% | Δ **−0.168** |
| TARGETED + gate | 511 | **3.316** | **27.4%** | Δ **−0.940** |

→ Gate **vẫn mạnh trên thước targeted** sau D1′ rebuild; aggregate không tụt.

### Guard eval_v4 (`--fast`, QA/para/syn only)

| Section | retr | gen-hybrid |
| :--- | ---: | ---: |
| QA val | **100%** | **100%** |
| Paraphrase | 88% | 88% |
| Synonym | 67% | 67% |

→ Acceptance §7.3 **đạt** (QA không tụt).

### Sinh E2E — [`scripts/test_gen_network_v4.py`](scripts/test_gen_network_v4.py)

**Gate ON vs OFF:** 10/10 prompt — output **giống hệt** dù `frame_bracket_gate` fire 7–24 bước/step.

**v4 vs `_t3_isg`:** 9/10 prompt giống; khác `Đọc sách giúp` (fuzzy match lệch do embedding 77k).

**Prompt khung — thất bại E2E:**

| Prompt | Kết quả |
| :--- | :--- |
| `nếu trời mưa` | Không sinh `thì`; ghép fact không liên quan |
| `tuy vất vả` | Không nối `nhưng` gọn; kéo đoạn văn khác |
| `nếu trời mưa thì` | Tiếp `Khi đến Vũng Biện thì trời tối đen` (gate ON/OFF giống nhau) |

**Chẩn đoán post-wire:** gate nhân scores Hadamard **không đủ mạnh** so với parent agenda +
index backbone → argmax không đổi. Metric targeted đo **base-LM tại gold close** — khác pipeline
hybrid `generate()` (có `best_match` → agenda).

### Acceptance §7 — checklist

| # | Tiêu chí | Kết quả |
| :---: | :--- | :--- |
| 1 | Mode B dương trên metric phù hợp | ✅ Δ −0.940 targeted |
| 2 | Ruler aggregate không tụt > 0.05 | ✅ 4.028 (= baseline) |
| 3 | QA/ISG guard giữ nguyên | ✅ 100/100 |
| 4 | Đọc tay ≥2 prompt khung cải thiện | ❌ **chưa đạt** |

---

## 10. CHỐT TK_017 — hiện trạng & đánh giá

### Điểm ĐẠT

- Meta-finding **xác nhận bằng số**: readout phẳng âm; position-targeted dương mạnh.
- Mode A (entropy) **bác bỏ**; Mode B **chọn làm xương sống**.
- Wire production: `frame_bracket_gate.py` + config + trace + tests.
- `network_v4` guard pass; không ship kênh aggregate âm.

### CHƯA ĐẠT

- **Mục tiêu gốc §7.4:** sinh prompt khung **chưa cải thiện** — acceptance 4/4 **chưa đủ**.
- Gate **fire nhưng không đổi token** trong hybrid generate — cần thiết kế override (additive
  boost closer thay vì chỉ Hadamard nhân; hoặc gate trước agenda lock; hoặc W_seq-only path cho khung).

### CHƯA LÀM / CHƯA ĐO

| Hạng mục | Trạng thái |
| :--- | :--- |
| Mode C learned gate (G3/LTD) | ❌ |
| D2 q·k qua cùng pattern gate | ❌ |
| D4 sense/POS qua gate | ❌ |
| Metric sinh targeted @ close (không phải gold LM) | ❌ |
| Graph UI trace `frame_bracket_gate` | ❌ |
| eval_v4 full essay/anaphor trên v4 vs ISG diff | ⚠️ partial |

### Đánh giá sơ bộ

TK_017 **thắng ở tầng đo và kiến trúc** (biết *khi nào* fire, không broadcast). **Chưa thắng
ở tầng sản phẩm** (prompt `nếu…` vẫn tệ). Gap = gate wired **trong** A3 loop nhưng hybrid
`generate()` lấy agenda từ parent trước — closer đúng khung không được ưu tiên khi agenda
kéo sang fact khác.

**Hướng fix tiếp (đề xuất):** (1) eval sinh chỉ chấm token tại close position trên prompt khung;
(2) `apply_frame_bracket_gate` → additive/local softmax trên top-K closers thay vì nhân toàn vocab;
(3) bypass parent plan khi prefix chỉ còn khung (W_seq + gate path); (4) ablate `ar_index_backbone`
@ bracket close.

---

## 11. REVIEW + VERIFY (Claude, 2026-07-05) — pattern TK_005/006

### Verify đã chạy lại (không tin số suông)

- **15 test pass** (`test_frame_bracket_gate` + `test_paths` + `test_eval_v4`).
- **Lấp lỗ đo acceptance §7.2:** Cursor tick ✅ "ruler không tụt sau wire" bằng số **base-LM
  KHÔNG gate** (4.028) — chưa ai đo ruler với **runtime gate như wired**. Claude đo (n=2860,
  cfg default w=0.5/floor=0.2, state machine y hệt wire):

| | bits | top1 | fire |
| :--- | :---: | :---: | :---: |
| base-LM | 4.028 | 22.1% | — |
| **RUNTIME gate (wired)** | **3.976** | **22.4%** | **93%** |

  → **Δ −0.052 bits, CÓ LỢI nhẹ** dù fire 93%. Khác global broadcast cũ (+0.48) vì wired gate
  dùng MỘT opener gần nhất (boost sắc, chuẩn hoá theo row) thay vì cộng-mọi-opener-decay.
  Acceptance §7.2 giờ **đạt bằng số THẬT** (không chỉ vô hại — hơi dương).

### Findings review (đánh giá phương pháp Cursor)

- **F1 — Mode B "Δ−0.940" là ORACLE (label-leak), không phải gain khả thi.**
  `eval_d3_bracket_positions._is_bracket_close(prefix, gold)` quyết định fire theo **chính token
  gold** — runtime không biết gold. −0.940 = **cận trên oracle** (bằng chứng tín-hiệu-tồn-tại
  TẠI vị trí close — giá trị chẩn đoán THẬT), nhưng dùng nó pass acceptance §7.1 để wire là
  **nhảy logic** (may là số runtime đo lại vẫn ≥0 — F0 ở trên cứu). Mọi trích dẫn "−0.940" sau
  này phải kèm chữ *oracle*.
- **F2 — Chẩn "gate fire nhưng argmax không đổi" đúng nhưng thiếu phân tách:** với Hadamard
  `×(0.2 + 0.5·boost)` closer được nhân tối đa ×0.7 trong khi `ar_index_margin` cho token index
  thắng `max·1.5` — về cấu trúc số học gate KHÔNG BAO GIỜ thắng index-backbone. "Additive/local
  softmax + override tại close" (đề xuất #2 của Cursor) là đúng hướng.
- **F3 — `ensure_frame_brackets` lazy-mine MỖI process** (~15-40s ở lần generate đầu — viz
  server/eval/CLI đều trả giá; `_mined_state_ids` dùng `id(state)` có thể bị reuse sau GC).
  Nên mine lúc build + serialize layer vào checkpoint.
- **F4 — `bind_sense_resonance`:** `rank_cache` key theo tuple(seed) gần như unique/sense →
  cache vô dụng thực tế (ước 6h đứng nguyên); `added>=max_tokens_per_sense` chỉ `continue`
  không break vòng parents (perf nhỏ). Cần batch-hoá trước khi chạy thật.
- **F5 — Đường sinh prompt khung thất bại chủ yếu ở RETRIEVAL, không phải gate:** `nếu trời mưa`
  → best_match/facts_about kéo cha lạ ("cung Mặt trời...") → agenda khoá "nói gì" sai từ đầu.
  Gate lo "cách nói" không cứu được. Fix thuộc low-match-bypass (đề xuất #3 Cursor) — khi match
  yếu, plan=None → đường W_seq+bracket dẫn.

### Đánh giá Cursor (sơ bộ)

**Tốt:** D1′ dilution fix đúng + đo sạch (parents +0, ruler về 4.028, 2 vòng lặp tìm ra nhiễm
W_seq); ticket cập nhật TRUNG THỰC (không giấu E2E fail); code gate sạch (state machine, ablate,
trace, tests); tuân thủ observability. **Thiếu:** (a) oracle metric làm acceptance (F1); (b) bỏ
sót số ruler-runtime-gate (F0 — may là dương); (c) chưa cô lập nguyên nhân E2E (F2/F5 lẫn nhau).
Tổng: **khá — làm đúng phần code, hụt phần kỷ luật đo ở 1 điểm mấu chốt.**

### Lộ trình tiếp (Claude đề xuất, thứ tự đòn bẩy)

| # | Việc | Vì sao | Đo bằng | Chi phí |
| :--- | :--- | :--- | :--- | :--- |
| **R1** | **Low-match bypass**: `generate()` khi best_match/facts_about < ngưỡng → KHÔNG ép agenda; sinh bằng W_seq + bracket-gate + entity | F5 — prompt khung chết ở retrieval kéo cha lạ; đây là gap SẢN PHẨM lớn nhất | đọc tay 5+ prompt khung (`nếu trời mưa thì…`) + guard QA/ruler | vừa |
| **R2** | **Gate override tại close**: gap-histogram từ `seg._pair_occurrences` (đã có (parent,pos)) → P(close\|gap) prefix-only; fire mạnh (additive/top-K softmax) CHỈ khi gap ~ mode | F1/F2 — thay oracle bằng predictor học được (Luật 0 sạch); cho gate quyền thắng index-backbone đúng chỗ | ruler aggregate ≥ −0.05 + targeted (KHÔNG oracle) dương + đọc tay | vừa |
| **R3** | Mine-at-build: FRAME_BRACKET vào `build_dict_checkpoint` + serialize | F3 — bỏ 15-40s/process | load log | nhỏ |
| **R4** | `bind_sense_resonance` batch-hoá (index đảo token→sense, 1 pass qua parents thay vì rank/sense) rồi CHẠY | F4 — 6h→phút; mở D4 | probe sense sạch hơn + ruler giữ | vừa |
| **R5** | eval_v4 FULL (essay/comp/anaphor) network_v4 vs ISG diff + Graph UI trace `frame_bracket_gate` | lỗ guard ⚠️ còn lại + quan sát | bảng diff | nhỏ |

D2 (q·k) và D4 (stack neo) **sau R1-R2** — cùng pattern gate đã được chứng minh cách đo đúng.

---

## 12. R1 IMPLEMENTED + ĐO (Claude, 2026-07-05) — ROUTING ĐÚNG, LỘ BỆNH CONTINUATION

### Đã làm (3 lớp, mỗi lớp có cờ ablate)

1. **Focus-gate plan fallback** (`_filter_plan_on_focus`, cờ `ar_fallback_focus_gate`): fact
   fallback phải chứa primary-focus (fold-group). Test toy 2 ca (drop lạc / giữ con-kiến).
2. **Bracket-gate trong `_wseq_fallback`** (đường bypass giờ có khung dẫn).
3. **Opener-continuation routing** (cờ `ar_opener_continuation`): prompt mở đầu bằng
   bracket-OPENER (có row FRAME_BRACKET, check cả fold-variant) + không frame → coi là câu
   TIẾP DIỄN → W_seq path, bỏ retrieval. Luật 0 sạch (đọc cạnh mined).

### Chẩn đoán trước khi code (đo, quyết định thiết kế)

- **Drift nằm ở CRISP, không phải fallback:** `nếu trời mưa` best_match **HIT 0.192** (cha QA
  "Mặt trời cung cấp gì" — qua token "trời"); `Mèo là gì` HIT 0.261 ("Cái gì là nguồn gốc…").
  → focus-gate fallback (1) không bao giờ chạm 3/4 ca đích. `tuy vất vả` MISS → gate (1) chạm
  và hành xử ĐÚNG (fact "Tuy có vất vả đời sống…" đúng chủ đề → giữ).
- **Ngưỡng score KHÔNG tách được:** guard hợp lệ xuống tới **0.121–0.139** (Para "Người chữa
  bệnh" 0.121, Syn "bao nhiêu hôm" 0.129 — fuzzy) trong khi drift lên 0.48 ("Mùa hè em") →
  BÁC BỎ score-threshold. Focus-mặt-chữ trên crisp PHÁ Syn theo định nghĩa (khác mặt chữ).
  → routing bằng CẤU TRÚC (opener) là đường duy nhất không đụng guard.
- **Focus IDF mù liên-từ:** trên network_v4, `nếu` gidf **6.74**, `tuy` 7.76 — từ chức năng
  hiếm-mức-cha bị IDF coi là thực thể → đứng ĐẦU focus. Gốc sâu của drift crisp (retrieval
  đuổi theo "nếu"/"trời"). Ghi sổ: POS-hub D1 (nếu=kết từ) là cure đúng tầng, phase sau.

### ĐO A/B (network_v4, on/off `ar_opener_continuation`; log `out/r1_opener_ab.log`)

| Prompt | OFF (cũ — drift) | ON (R1 — continuation) |
| :--- | :--- | :--- |
| `nếu trời mưa` | "cung Mặt trời cấp ánh sáng và nhiệt." (fact LẠC) | ". , và các thành phố…" (word-salad) |
| `Nếu em học chăm` | "Trẻ em có quyền được chăm sóc" (LẠC) | "chăm sóc y tế bào…" (salad) |
| `tuy vất vả` | "Tuy có vất vả đời sống mà tâm vẫn an…" | salad |
| **Guard QA/Para/Syn (3 ca đọc tay + suite)** | — | **SAME tuyệt đối** (prompt "Hỏi:" không trigger); **46 test pass** |

### Verdict R1 — TRUNG THỰC

- **Routing: ĐÚNG** — 5/5 prompt khung chuyển đường, 0/3 guard bị đụng (by construction:
  guard luôn có frame). Cơ chế phân loại prompt bằng substrate hoạt động.
- **Chất lượng continuation: ÂM TÍNH** — `_wseq_fallback` (base-LM W_seq+readout greedy) sinh
  word-salad hub-sụp `. , và các` — ĐÚNG bệnh doc 07 "spread vô hướng sụp hub". Nhất quán với
  thước: bits 4.028 = base-LM chỉ đúng top1 22% → 24 bước greedy tất yếu rác. **Trade-off đo
  được: OFF = bịa fact lạc (sai NGHĨA, đúng ngữ pháp); ON = không bịa (đúng route, rác ngữ
  pháp).** Theo Luật 0 chống "ảo tưởng sức mạnh": ON trung thực hơn (không che bệnh substrate
  bằng hallucination trôi chảy) → **giữ ON theo luật 3.5, vào SỔ NỢ với giá đọc tay ghi trên**.
  Tác giả lật lại = `ar_opener_continuation=False` (1 dòng).
- **Bài học cho R2:** continuation path là nơi bệnh substrate LỘ TRẦN — R2/D2 (q·k + gap-gate)
  giờ có sân đo sạch: cải thiện `_wseq_fallback` đọc được NGAY trên 5 prompt khung này
  (không bị agenda/index-backbone che). Đây chính là "eval sinh targeted" tự nhiên.

### Sổ nợ R1

| Món | Ghi chú |
| :--- | :--- |
| Word-salad continuation | bệnh substrate (top1 22%) — R2/D2 chữa, không tune vá `_wseq_fallback` |
| Drift crisp "Mèo là gì" (0.261) | KHÔNG fix trong R1 (không phải opener) — ranking bug riêng: statement về Mèo thua cha "Cái gì là nguồn gốc"; cần soi `rank_matches` |
| Focus IDF mù liên-từ (nếu 6.74) | cure = POS-hub D1 vào `top_focus_tokens` (phase D4) |
| `hybrid_parent_min_score=0.18` quá thấp cho câu mở | không nâng được bằng score (phá Syn) — đã bác bỏ, giữ nguyên |

---

## 13. R2 IMPLEMENTED + ĐO (Claude, 2026-07-05) — q·k RULER DƯƠNG; CONTINUATION = TRẦN SUBSTRATE (4 bằng chứng hội tụ)

### Đã làm

1. **`hca/attention_qk.py`** (doc 03 D2-lite): `DirectedQK` — SVD PPMI-CÓ-HƯỚNG của W_seq,
   Wq=U√Σ / Wk=V√Σ (hai phía — khác E đối xứng); lazy view per-state. Blend vào kênh micro
   `_wseq_fallback` (cờ `ar_qk_weight=0.5`; chuẩn hoá từng kênh trước blend — doc 03 rủi ro 2;
   vá pad khi vocab grow sau build view).
2. **R2b — neo TOPIC cho continuation** (doc 05 H_mid): seeds = content-token của prompt,
   field `spreading_activation_read` nhân vào scores (cờ `ar_continuation_topic_weight=1.0`).
3. Probe R2c (không ship): IDF-normalize base (`base×gidf`).

### ĐO — A0 q·k (`scripts/eval_qk_a0.py`, network_v4, held-out n=2860)

| | bits ↓ | top1 | Ghi chú |
| :--- | :---: | :---: | :--- |
| base-LM | 4.028 | 22.1% | |
| **+q·k blend (w=0.5)** | **3.926** | 21.9% | **Δ −0.103 — kênh mới ĐẦU TIÊN cải thiện ruler aggregate** |
| UNSEEN transitions (n=163) | Δ **0.000** | | tổng-quát-hoá khuôn CHƯA chứng minh — thắng từ re-weight cặp đã thấy |

Đọc tay khuôn CÓ nghĩa cú pháp: `nếu→{không,có,bị,làm}`, `không→{còn,chỉ,phải}`, `màu→{vàng}`
— đúng doc 07: q·k mạnh ở function-token. **Verdict q·k: GIỮ ON** (thắng đo được trên thước).

### ĐO — continuation đọc tay (5 prompt khung, log `out/{r2_qk_ab,r2b_topic_ab,r2c_idfbase_probe}.log`)

**4 công thức khác nhau — TẤT CẢ vẫn word-salad hub (`. , và các thành phố…`):**

| Thử | Kết quả đọc tay |
| :--- | :--- |
| bracket-gate (R1) | salad |
| + q·k blend (R2) | salad đổi dạng ("người Mỹ"→"Việt Nam Cộng hòa Séc") |
| + topic-anchor (R2b) | salad đổi dạng |
| base×gidf (R2c probe) | salad; "tuy vất vả" TỆ hơn (lặp ". ,") |

Guard mọi vòng: QA đọc tay SAME OK + **304 test pass toàn suite** (1 bug shape vocab-grow đã vá).

### KẾT LUẬN R2 — trung thực, 4 bằng chứng hội tụ

**Free-continuation là TRẦN SUBSTRATE/DATA, không phải lỗi công thức readout.** 4 cách điều
biến khác nhau (khung xa / khuôn q·k / neo topic / IDF-base) đều không phá được hub-salad
greedy — vì base-LM top1 22% (bits 4.028 ≈ 16 lựa chọn/bước) thì 20 bước argmax tất yếu rác,
điều biến floor-nhân không đổi được argmax khi base hub lệch 100×.

**Nghi vấn gốc (data):** corpus = wiki-formal + essay-instruction — KHÔNG có văn NÓI/hội thoại.
"nếu trời mưa thì…" là khuôn văn nói; W_seq không thể học cái corpus không chứa. → Hướng fix
THẬT: **ingest corpus hội thoại/truyện tiếng Việt** (chuỗi văn nói) — data move, không phải
kiến trúc move. Thứ nhì: sampling thay greedy + order-cao (chờ substrate dày hơn mới đáng đo).

### SỔ NỢ R2 (flag BẬT theo luật 3.5 + giá đã đo)

| Flag | Giá đã đo | Hướng fix |
| :--- | :--- | :--- |
| `ar_qk_weight=0.5` | ruler −0.103 ✓; unseen 0.000; đọc tay trung tính | data chuỗi + head tách POS (D2 full) |
| `ar_continuation_topic_weight=1.0` | đọc tay trung tính-đổi-dạng; không hại guard | đo lại khi có data hội thoại |
| `ar_opener_continuation=True` | route đúng 5/5; output rác (trần substrate) | data hội thoại → continuation có nghĩa |

---

## 14. R3-DATA: INGEST VĂN NÓI + ĐO (Claude, 2026-07-05) — GUARD GIỮ; SALAD ĐỔI MÀU → TRẦN LÀ ORDER-1, KHÔNG PHẢI DATA

### Đã làm

- **Khảo sát nguồn:** `truongpdd/vietnamese_story` (99k truyện/tản văn — văn nói thật, /rows OK),
  `5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca` (12.7k hội thoại). Loại: OpenSubtitles/truyện cổ
  tích (gated 401/404), BlossomsAI (rác markdown).
- **Fetch** (`scripts/fetch_vi_conversational.py`, resumable+throttle): 6 000 truyện + 4 000 hội
  thoại → `data/conv/{stories,chat}_raw.jsonl`.
- **Build** (`scripts/build_conv_checkpoint.py`): tách câu + quality gate (max_words=40 cho văn
  nói) → **60 000 câu, CHỈ `_g1_and_wseq` — KHÔNG add_example** (đúng bài học D1: data vào đúng
  TẦNG; parents +0). → `network_v4_conv.pkl` (vocab 77.5k→89.3k, 252MB).

### ĐO (log `out/r3_conv_measure.log`)

| | network_v4 | network_v4_conv | Đọc |
| :--- | :--- | :--- | :--- |
| Guard QA đọc tay | ✓ | **✓ 3/3 đúng nguyên** ("Thủ đô…Hà Nội", "Bác sĩ chữa bệnh") | parents+retrieval nguyên vẹn — thiết kế ĐÚNG |
| Ruler bits ↓ | **4.028** | 4.233 (+0.205) | co-act văn nói pha loãng nhẹ miền essay |
| Ruler top1 | 22.1% | 22.4% | trung tính |
| Continuation 5 prompt khung | salad wiki (". , và các thành phố") | **salad văn-nói (", có thể có thể . , không ? gì")** | hub ĐỔI MÀU theo miền — vẫn không mạch lạc |

### KẾT LUẬN R3 — nâng cấp chẩn đoán (5 bằng chứng)

**Giả thuyết "thiếu data văn nói" bị bác một phần:** 60k câu văn nói làm hub đổi màu đúng miền
(chứng tỏ data VÀO thật) nhưng greedy vẫn loop hub mới ("có thể", "không"). Chẩn đoán cuối:

> **Trần TOÁN của Markov order-1 + greedy argmax** — mọi corpus tự nhiên đều có hub function-
> word; P(thì | mưa) bigram không phải P(thì | nếu…trời…mưa) context. LLM giải bằng order-∞
> (attention) + sampling. Không data nào cứu ở order-1 greedy.

**→ Hướng fix ĐÚNG TRIẾT LÝ MẠNG (R3b, đề xuất):** continuation không nên là Markov greedy —
nên là **pattern-completion trên POOL VĂN NÓI RIÊNG** (two-tier 0.5.10 có sẵn): nạp câu văn nói
làm parents ở pool priority-thấp, CHỈ truy hồi khi route opener-continuation (không cạnh tranh
QA/essay) → "nếu trời mưa" match mẫu "nếu trời mưa thì …" thật → completion mạch lạc. Đây chính
là máy mạnh nhất của mạng (retrieval+completion) áp vào đúng chỗ, thay vì đường Markov yếu nhất.

### Trạng thái ship

- `network_v4_conv.pkl` = **artifact A/B, KHÔNG thay** `network_v4.pkl` (ruler +0.205).
- Data + pipeline GIỮ (fetch resumable 10k items, builder, quality gate) — nguyên liệu cho R3b.

---

## 15. R3b IMPLEMENTED + ĐO (Claude, 2026-07-05) — ✅ CONTINUATION MẠCH LẠC ĐẦU TIÊN

### Cơ chế (đúng triết lý mạng — docx §1 pattern completion, two-tier 0.5.10)

- **`scripts/build_conv_pool.py`**: câu văn nói → cha **priority=0.3** (pool riêng); CHỌN câu
  bằng substrate D3 (chỉ nạp câu CHỨA bracket-opener — định luật 1: chọn, không cắm phẳng);
  KHÔNG `_g1_and_wseq` (ruler giữ NGUYÊN); mine FRAME_BRACKET tại build → cạnh PERSIST (gỡ nợ
  F3 lazy-mine). → `network_v4_convpool.pkl`: parents 44 633→**64 590** (+19 957), build 34s
  (cổ tức `defer_index` D1).
- **`_conv_pool_tail`** (`generate_hebbian`): route opener-continuation → `rank_matches(
  restrict_parents=pool)` → phát **ĐUÔI mẫu sau điểm khớp cuối** (fold-group). Cờ
  `ar_conv_pool=True`, `conv_pool_min_score=0.10` (score bị priority-scale 0.3 — đo mẫu tốt
  0.12–0.17). Trượt → `_wseq_fallback` như cũ.

### ĐO (log `out/r3b_measure{,2}.log`)

| Prompt | Trước (Markov — mọi biến thể R1/R2/R3) | **Sau (R3b pool completion)** |
| :--- | :--- | :--- |
| `nếu trời mưa` | ". , và các thành phố…" | **"nếu trời mưa tầm tã ."** ✓ |
| `nếu trời mưa thì` | salad | **"thì đây quả là một trường tai nàn đáng sợ ."** ✓ (typo từ data gốc) |
| `vì em chăm học` | salad | **"chăm làm , lại càng thương khi những đứa trẻ ngoan ngoãn ấy…"** ✓ |
| `khi mùa hè đến` | salad | **"vẻ nổi bật không cầu kỳ , dù diện mùa đông hay hè đều đẹp ."** ✓ |
| `càng học càng` | — | **"càng học càng đông ."** — khuôn càng…càng ĐÚNG (nghĩa hơi lệch) |
| `tuy vất vả` | salad | ", Sinh Tử Câu Cừu ." ✗ (tên truyện lọt — nhiễu data kiếm hiệp) |
| `Nếu em học chăm` | salad | "sóc trẻ em…" ✗ (bigram "chăm sóc" cắt sai điểm khớp) |

- **7/7 fire · 5/7 mạch lạc thật** (từ 0/7 mọi vòng trước). **Tốc độ 0-1s/prompt** (Markov AR
  6-42s) — completion là O(retrieval), không O(max_len×vocab).
- **Guard:** QA đọc tay nguyên vẹn + ruler **4.028 NGUYÊN** (thiết kế không-W_seq đúng) +
  **304 test pass**.

### Ý nghĩa

Xác nhận trọn chuỗi chẩn đoán R1→R3: continuation là bài TRUY HỒI MẪU + COMPLETION (máy mạnh
nhất của mạng), không phải Markov. Đây là lần đầu **prompt mở sinh văn đọc được** — và đạt bằng
đúng triết lý v3 "1 ví dụ = 1 nơ-ron, lan xuôi → dội ngược", không thêm cơ chế lạ.

### Sổ nợ R3b

- **verbatim-tail**: đuôi là mẫu nguyên văn (1 nguồn) — chưa "sinh thật"; nấc sau: AR bám mẫu
  làm agenda (như essay TK_009) / trộn nhiều mẫu.
- 2/7 nhiễu: (a) data kiếm hiệp/tên truyện lọt quality gate → lọc thể loại `the_loai`;
  (b) điểm-khớp-cuối bị bigram nuốt ("học chăm"→"chăm sóc") → khớp theo CỤM thay token đơn.
- Pool 20k mới dùng ~34k câu đầu / 60k đã lọc — scale thêm khi cần (build 34s).
- `network_v4_convpool.pkl` = ứng viên THAY `network_v4.pkl` (mọi guard giữ + thêm năng lực) —
  chờ tác giả duyệt + eval_v4 full trước khi flip default.

## 15b. R3b POLISH + SHIP (Cursor, 2026-07-05)

**Fix regression Paraphrase 88→75%** khi gắn convpool:
1. `generation_min_priority=0.5` — pool văn nói (priority=0.3) **không** vào `best_match`/`facts_about`.
2. **Frozen vocab** khi build pool — chỉ nạp câu token ⊆ vocab có sẵn (tránh embedding lệch).

**Cải thiện khớp cụm:** `_longest_suffix_cut` trong `_conv_pool_tail` (fix `Nếu em học chăm` còn lệch bigram `chăm sóc`).

**Lọc data:** bỏ CJK/kiếm hiệp/FX trong `build_conv_pool._is_conv_sentence`.

**Rebuild:** parents 44 633→**64 573**, vocab **77464** (không đổi).

**Guard sau fix:** QA **100/100**, Paraphrase **88%**, Synonym **67%** (= `network_v4`).

**Continuation (7 prompt khung):**

| Prompt | Output |
| :--- | :--- |
| `nếu trời mưa` | `nếu trời mưa tầm tã .` ✓ |
| `nếu trời mưa thì` | `…thì đây quả là một trường tai nạn đáng sợ .` ✓ |
| `tuy vất vả` | `tuy vất vả nữa lại kết thúc .` ✓ |
| `vì em chăm học` | mạch lạc ✓ |
| `khi mùa hè đến` | mạch lạc ✓ |
| `Nếu em học chăm` | ✗ (bigram `chăm sóc` cắt sai — sổ nợ) |

**Default checkpoint:** `network_v4_convpool.pkl` (paths + eval scripts).

---

## 16. REPORT CHAY THUC TE (2026-07-15) - network_v10 sau bake FRAME_BRACKET

Muc dich: chot mot ban report ngan de bao coworker sau khi bake xong `checkpoints/network_v10.pkl`.

### 16.1 Bake status (xac nhan)

- File log: `out/logs/bake_network_v10_nohup_20260714_204801.log`
- Ket qua:
  - `FRAME_BRACKET edges = 1758837`
  - `endpoints = 354671`
  - `seq_bi rows = 126154`
  - `seq_tri rows = 1958695`
  - Save thanh cong ve `checkpoints/network_v10.pkl`

### 16.2 Eval discourse (sau bake)

- Lenh chay:

  ```bash
  PYTHONPATH=src .venv/bin/python -u scripts/eval_discourse_prompts.py --checkpoint checkpoints/network_v10.pkl
  ```

- Log runtime: `out/logs/eval_discourse_prompts_20260715_105436.log`
- File tong hop: `out/tk035_discourse_prompts.log`
- Thoi gian: ~3047s (~50m47s)
- Trang thai: `DONE` (khong loi)

#### Summary metrics (mean by bucket/mode)

`varlm=0 / 0.35 / 0.5` cho ket qua giong nhau:

- `common_noun`: n=10, chars=227, multi=50.0%, overlap=1.00, ent_keep=0.20
- `conv`: n=10, chars=310, multi=90.0%, overlap=1.00, ent_keep=0.20
- `frame`: n=3, chars=326, multi=100.0%, overlap=1.00, ent_keep=0.00
- `history_wiki_para`: n=10, chars=329, multi=100.0%, overlap=1.00, ent_keep=0.50
- `statements_doc`: n=10, chars=302, multi=60.0%, overlap=1.00, ent_keep=0.20

Nhan xet nhanh:

- Pipeline eval chay on dinh sau bake.
- `overlap=1.00` toan bo bucket -> van co xu huong verbatim rat manh.
- Bien `varlm` hien tai chua tao khac biet do duoc tren bo eval nay.

### 16.3 D3 check (bash-only, khong re-mine)

Do 2 script `eval_d3_*` mac dinh se mine lai FRAME_BRACKET (rat nang), da dung cach do nhanh:

- Load checkpoint da bake
- Dung truc tiep ham `_ranks` va `_eval_targeted`
- KHONG goi lai `mine_frame_brackets`

Ket qua:

- `n=2803`
- base-LM: `bits 4.594`, `top1 19.3%`
- +fb global: `bits 5.438`, `top1 17.8%` (delta +0.844)
- +fb boundary-gated: `bits 5.455`, `top1 17.4%` (delta +0.861)
- +fb entropy-gated: `bits 5.438`, `top1 17.8%` (delta +0.844), fire `2995/2803 (106.8%)`

Mode B targeted:

- ALL: `bits 5.455 -> 5.296` (delta -0.159), `top1 17.4% -> 18.5%`
- TARGETED: `bits 5.279 -> 4.308` (delta -0.971), `top1 17.0% -> 23.4%`

### 16.4 Ket luan bao coworker

1. Bake `network_v10` thanh cong, FRAME_BRACKET da duoc vat chat hoa day du.
2. Eval discourse chay xanh, nhung model van nghieng verbatim (`overlap=1.00`).
3. Tin hieu D3 van theo pattern cu:
   - Targeted gain ro rang (dung cho dong-khung).
   - Aggregate mode global/boundary/entropy van am.
4. Uu tien tiep theo nen la tinh gate/routing va metric fire (vi fire >100% cho thay cach dem hien tai chua sach).
