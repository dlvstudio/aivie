# TK_016 — Redirect D1 (bind-resonance) + D3 khung xa + cắm neo vào ĐƯỜNG SINH

| Field | Value |
|-------|-------|
| **ID** | TK_016 |
| **Status** | 🟡 IN PROGRESS — D1′ + wire gate xong; sinh E2E chưa cải thiện (2026-07-05, Cursor) |
| **Nguồn** | Tiếp TK_015: D1 flat-dump đo ÂM TÍNH (ruler 4.03→4.40) → redirect. Tác giả muốn thấy kết quả TRỰC QUAN qua prompt+mode → cần cắm neo vào đường sinh + viz |
| **Doc** | [`docs/study/08_DICTIONARY_LABELED_SUBSTRATE.md`](docs/study/08_DICTIONARY_LABELED_SUBSTRATE.md) (D1–D4) |
| **Checkpoint** | `checkpoints/network_v4.pkl` — `_t3_isg` + substrate từ điển D1′ (**không** flat-dump parent; parents=44 633) |

## 1. Tổng quan

TK_015 dựng: thước không bão hoà (`next_token_lm`), xác nhận 2 tường (C0), và substrate từ điển
(sense/POS neuron). **Đo lộ 2 việc:** (a) flat-dump từ điển làm nơ-ron cha PHA LOÃNG sinh essay
(lặp bài học 0.5.8); (b) **chưa có gì "thấy được qua prompt"** vì đường sinh CHƯA đọc sense/POS/
khung. TK_016 = sửa gốc âm tính (D1′), xây tường xa (D3), và **cắm neo vào readout A3** để kết
quả hiện lên khi sinh + viz.

## 2. Hiện trạng chi tiết (đo TK_015)

- **Thước** `next_token_lm` (log2-rank base-LM, held-out essay 200 câu): baseline `_t3_isg`
  **4.028 bits / top1 22.1%**; `network_v4` (flat-dump) **4.400 / 20.0%** (TỆ hơn — dilution).
- **Sense-neuron TÁCH nghĩa THẬT** ở substrate (cân#3→gram/kg/lạng, đá#1→hòn/tảng) NHƯNG đa số
  NHIỄU/THƯA (từ điển 1-2 ví dụ/nghĩa → PPMI ít cạnh).
- **Đường sinh** `generate_autoregressive` ([generate_hebbian.py:477](src/nn_aivie/hca/generate_hebbian.py))
  chỉ đọc W_seq/spread/entity/discourse — **KHÔNG đọc** SENSE/POS_ANCHOR/FRAME_BRACKET.
- **Tường C0 chưa đụng:** cạnh xa `nếu→thì`/`Hỏi:→Đáp:` seq=0.00 (D3 chưa làm).
- **tsdocode THIẾU đồng/trái nghĩa** → cạnh SYN/ANT chưa có nguồn.
- **Viz:** `/api/embedding/*` có sẵn; sense/POS node bị `embedding_search` lọc (surface-only) →
  chưa khám phá được; đường sinh chưa hiện neo.

## 3. Kết quả hiện tại (chốt trung thực — cập nhật 2026-07-05)

| Hạng mục | Trạng thái |
| :--- | :--- |
| Thước không bão hoà (Trụ 1) | ✅ THẮNG (hạ tầng đo) |
| 2 tường C0 đo được | ✅ |
| Cơ chế sense tách (substrate) | ✅ chứng minh; ❌ chất lượng data-limited |
| D1′ dilution fix (ruler về baseline) | ✅ **4.028 bits / top1 22.1%** trên `network_v4` |
| D3 mine FRAME_BRACKET | ✅ 35 205 cạnh |
| Wire gate position-targeted (TK_017) | ✅ code + tests; ❌ **chưa đổi output sinh E2E** |
| Sinh tốt hơn (prompt khung) | ❌ CHƯA — `nếu…` vẫn truy hồi sai cha |
| Graph UI/API cho neo mới | ❌ chưa (viz sense/POS pending) |
| `bind_sense_resonance` trong checkpoint | ❌ chưa chạy (perf ~6h; build `--skip-resonance`) |
| D2 / D4 | ❌ chưa làm |

## 4. Đề xuất hướng đi

Thứ tự đòn bẩy: **D3 (thấy được nhanh) → D1′ (sửa gốc âm) → D2 (attention) → D4 (hợp nhất)**.
D3 độc lập với data-thưa của D1, nhắm đúng "ghép/chuyển không mượt", và tạo khác biệt sinh THẤY
được để wire UI test bằng prompt.

## 5. Giải pháp (từng phase, đo bằng thước TK_015)

### D3 — LINK NGANG XA `FRAME_BRACKET` (làm trước — visible)
- **Mine:** quét mỗi cha/episode cặp (token_mở, token_sau) KHÔNG kề, PMI có hướng; **gate POS**
  (đầu ∈ liên-từ/tiểu-từ/nhãn-frame từ `POS:` hub D1) → chống bùng nổ O(n²). Vật chất hoá cạnh
  `EdgeKind.FRAME_BRACKET` (đã khai báo). Ngưỡng PMI + df như phrase_assoc TK_008.
- **Cắm sinh:** kênh Hadamard thứ 5 trong `generate_autoregressive` — khi w0/lịch sử chứa
  token_mở "nếu"/"Hỏi:", nâng token_đóng "thì"/"Đáp:" (đọc `FRAME_BRACKET` như neo H_mid doc 05).
- **Đo:** ruler `next_token_lm` (kỳ vọng bits GIẢM ở held-out có cấu trúc khung) + đọc tay
  prompt `Hỏi: … nếu … Đáp:`; guard grade_v3/ISG không tụt.

### D1′ — SENSE bind-resonance (sửa gốc âm tính flat-dump)
- **KHÔNG** nạp định-nghĩa/ví-dụ làm cha vào pool sinh (bỏ dilution). Thay: sense-neuron **hút
  occurrence THẬT trong corpus** bằng resonance (`bind_fact_links`/`bind_episode_by_resonance` đã
  có) → nhiều cạnh co-activation THẬT → vector sense sạch, hết thưa.
- **Two-tier:** định nghĩa từ điển vào pool RIÊNG (chỉ tra khi hỏi "X là gì / nghĩa của X"),
  không cạnh tranh sinh essay (mẫu 0.5.10).
- **Đo:** ruler phải VỀ ≤ baseline 4.028 (hết dilution) + probe sense sạch hơn (ít nghĩa "chết").

### D2 — Attention Q/K có hướng (doc 03, A0)
- SVD PPMI-có-hướng của `W_seq` + `FRAME_BRACKET` → Wq/Wk; head tách theo POS (function-token
  riêng — vá nhiễu doc 07). Cắm kênh micro A3 thay/blend `ar_index_backbone`.
- **Đo:** held-out bigram rank (A0) + ruler.

### D4 — Sinh = đọc STACK NEO (doc 05) + SYN/ANT
- H_slow/mid/fast; sense/POS/theme/frame là các kênh readout. Nạp SYN/ANT từ **WordNet
  `zeloru/vietnamese-wordnet` + ViCon** (tsdocode thiếu) → cạnh `SYNONYM/ANTONYM` (đã khai báo).
- **Đo:** Synonym probe (kỳ vọng >50%) + ruler + composition.

## 6. Đề xuất thay đổi cụ thể

| File | Thay đổi | Phase |
|------|----------|:---:|
| `core/seg_graph.py` hoặc `training/frame_bracket.py` (mới) | mine cặp không-kề gate POS → promote `FRAME_BRACKET` | D3 |
| `hca/generate_hebbian.py` | kênh Hadamard thứ 5 đọc `FRAME_BRACKET`; (D4) stack neo sense/POS | D3/D4 |
| `hca/config.py` | cờ `frame_bracket_weight`, `sense_anchor_weight` (ablate=0) | D3/D4 |
| `core/parent_store.py` | `bind_sense_resonance()` (hút occurrence); two-tier pool định nghĩa | D1′ |
| `scripts/ingest_dictionary.py` caller | tách pool định nghĩa khỏi pool sinh | D1′ |
| `hca/attention_qk.py` (mới) | Wq/Wk SVD có hướng, head theo POS | D2 |
| `scripts/ingest_wordnet.py` (mới) | nạp SYN/ANT từ WordNet/ViCon | D4 |
| `viz/export_graph_v3.py` + `export_embedding.py` + `server/main.py` | hiện sense/POS/frame-bracket node + cạnh; endpoint `/api/embedding/senses`; trace sinh hiện neo | mỗi phase |
| `viz/web/src/*` | panel Sense/Frame trong Embedding + Parent Explorer | mỗi phase |

## 7. Lộ trình đo

Mỗi phase: A/B thước `next_token_lm` + guard (grade_v3/ISG QA không tụt) + đọc tay. Giữ nếp
"đo dương mới giữ; âm tính vào sổ nợ, không tắt" (v4 §3.5). Đóng băng retrieval crisp-QA.

## 7b. KẾT QUẢ ĐO — D3 (2026-07-05, Claude) — MINE DƯƠNG, KÊNH SINH ÂM TÍNH

**Làm:** `training/frame_bracket.py::mine_frame_brackets` (df-band func + đồng-hiện có hướng
gap≥2 + PMI + **lọc bất đối xứng hướng** fwd≥2·bwd) + `EdgeKind.FRAME_BRACKET`. Probe/eval:
`scripts/{probe_d3_brackets,eval_d3_ruler}.py`. Trên `_t3_isg`: 35 205 cạnh.

**① Mine — DƯƠNG (đúng "cấu trúc câu" tác giả muốn):** liên-từ tương liên nổi SẠCH:
`tuy→nhưng` **3.57** (top), `nếu→thì` 2.57 (+ "sẽ"), `không→gì/nào/nữa`, `càng→thêm`. Bắc cầu
đúng tường C0 (`nếu→thì` seq=0.00). Nhân-quả `vì/bởi/do` nhiễu (closer không cố định — đúng
ngôn ngữ). `Hỏi:→Đáp:` không mine được: **Hỏi: df=0** trong parents (frame xử lý riêng trong
`generate`, không cần cạnh).

**② Kênh sinh — ÂM TÍNH (đo, 4 cấu hình):** cắm frame-bracket làm kênh Hadamard readout trên
thước không-bão-hoà (held-out essay, n=2860):

| Cấu hình | bits ↓ | top1 | Δbits |
| :--- | :---: | :---: | :---: |
| base-LM | **4.028** | **22.1%** | — |
| + fb global (w=1.0) | 4.478 | 20.9% | +0.45 |
| + fb global (w=0.15) | 4.578 | 20.3% | +0.55 |
| + fb global (w=0.5) | 4.548 | 20.3% | +0.52 |
| + fb boundary-gated (w=0.5) | 4.587 | 20.3% | +0.56 |

**Chẩn:** closer chỉ đến ở MỘT vị trí sau opener, nhưng kênh bơm nó suốt cửa sổ → nhiễu phần
lớn vị trí không-phải-đóng-khung → hại aggregate. Boundary-gate (chỉ bơm sau dấu câu) KHÔNG cứu
(dấu phẩy ở khắp nơi). → **KHÔNG cắm frame-bracket vào sinh** (đã đo hại).

**③ META-FINDING (nhất quán D1 flat-dump + doc07 directed-attention + D3):** substrate CÓ tín
hiệu cấu trúc thật (sense tách, q·k tổng-quát, bracket tương-liên) — nhưng cắm làm kênh readout
PHẲNG/TOÀN-CỤC **luôn pha loãng** vì kích hoạt bừa. **Cái thiếu KHÔNG phải thêm cạnh cấu trúc,
mà là CƠ CHẾ GATE/CHỌN ngữ-cảnh** quyết định ĐỌC tín hiệu nào Ở ĐÂU — đúng tinh thần softmax-
attention (transformer) + agenda (A3). Đây là redirect lớn cho v4: đầu tư vào **tầng gate/chọn**,
không phải thêm substrate. `_t3_isg` giữ nguyên (không ship kênh hại).

**Sổ nợ D3:** thước aggregate không hợp feature thưa → cần metric TARGETED (rank closer TẠI vị trí
đóng-khung thật) để xác nhận "đúng chỗ có lợi"; nhưng dùng được vẫn cần gate chỉ-fire-đúng-chỗ.

## 7c. G-probe + Mode B (2026-07-05, Cursor) — entropy ÂM, targeted DƯƠNG

**G-probe (Mode A):** entropy-gated + opener — vẫn **≈ global** (Δ **+0.449** best) vì essay fire ~94–100%
vị trí. Giả thuyết entropy-only **bác bỏ**.

**Mode B:** [`scripts/eval_d3_bracket_positions.py`](scripts/eval_d3_bracket_positions.py) — n=511 vị trí
đóng-khung: targeted Δ **−0.940 bits** (4.256→3.316), ALL Δ **−0.168** (chỉ fire @ close).

→ **Wire A3 theo position-targeted gate** (TK_017), không entropy-only / global.

**Wire A3 (07-05):** `frame_bracket_gate.py` + default `ar_frame_bracket_weight=0.5` trong `generate_autoregressive`. Ablation: đặt weight=0.

Chi tiết: [`TK_017_GATE_LAYER_REDIRECT.md`](TK_017_GATE_LAYER_REDIRECT.md) §8–§9.

## 7d. D1′ BUILD + ĐO — `network_v4.pkl` (2026-07-05, Cursor)

### Việc đã làm

1. **`ingest_dictionary` (D1′):** bỏ `add_example()` (không flat-dump parent); bỏ `_g1_and_wseq` trên
   định nghĩa (tránh nhiễm W_seq essay). Chỉ giữ: sense/POS hub + cạnh `SENSE` + `CO_ACTIVATION`
   sense↔token định nghĩa.
2. **`bind_sense_resonance()`** trong `parent_store.py` — implement + tối ưu (cache rank, cap seed);
   **chưa chạy trong build** (`--skip-resonance`) vì ~0.6s/sense × 36k ≈ 6h.
3. **`build_dict_checkpoint.py`:** default out `network_v4.pkl`; phase `bind_sense_resonance` tùy chọn.
4. **Checkpoint defaults:** `paths.py`, `eval_*.py`, `probe_*.py` → ưu tiên `network_v4.pkl`.
5. **Rebuild:** `python scripts/build_dict_checkpoint.py --skip-resonance` (~74s).

### Thống kê checkpoint sau build

| | `_t3_isg` | `network_v4` (D1′) | flat-dump cũ |
| :--- | ---: | ---: | ---: |
| vocab | 34 528 | **77 464** (+42 936 sense/POS) | ~77k |
| parents | 44 633 | **44 633** (+0) | ~115k |
| kích thước | 121 MB | **209 MB** | 261 MB |
| LM ruler bits | **4.028** | **4.028** | 4.400 |
| top-1 | **22.1%** | **22.1%** | 20.0% |

**Chẩn:** dilution đã sửa; nhiễm W_seq từ định nghĩa đã loại (lần build đầu không bỏ `_g1_and_wseq`
vẫn 4.400 bits dù parents=0 — lần 2 bỏ W_seq → về 4.028).

### Guard eval_v4 (`network_v4`, `--fast`, skip essay/gen-ar/composition/anaphor)

| Section | retr | gen-hybrid |
| :--- | ---: | ---: |
| QA val | 100% | 100% |
| Paraphrase | 88% | 88% |
| Synonym | 67% | 67% |

→ **Không tụt** so với `_t3_isg` (QA 100/100 giữ nguyên).

### `bind_sense_resonance` — trạng thái

- **Code:** ✅ có trong `parent_store.py`
- **Checkpoint:** ❌ chưa áp dụng
- **Ảnh hưởng sinh hiện tại:** **≈ 0** — A3 chưa đọc sense-neuron; thiếu bind không giải thích lỗi sinh khung
- **Ảnh hưởng tương lai (D4):** quan trọng — sense thưa → gate chọn nghĩa sẽ yếu

## 7e. SINH END-TO-END — smoke test (2026-07-05)

Script: [`scripts/test_gen_network_v4.py`](scripts/test_gen_network_v4.py). Checkpoint: `network_v4.pkl`.

### Gate ON (w=0.5) vs OFF (w=0.0) — hybrid `generate()`

| Prompt | fb fire | Output khác khi tắt gate? |
| :--- | ---: | :--- |
| `Đọc sách giúp` | 24 | **Không** |
| `Mùa hè em` | 7 | Không |
| `Hỏi: Việt Nam là gì Đáp:` | 8 | Không → `Thủ đô Việt Nam là Hà Nội.` |
| `nếu trời mưa` | 9 | Không → **không sinh `thì`**, fact lạ |
| `tuy vất vả` | 23 | Không |
| `Napoléon là ai` | 23 | Không |

**Chẩn:** gate **fire** (trace `frame_bracket_gate=True`) nhưng **argmax không đổi** — parent agenda + index backbone vẫn thắng.

### v4 vs `_t3_isg` (gate=0.5)

9/10 prompt **giống hệt**. Khác duy nhất `Đọc sách giúp` (fuzzy `best_match` lệch do embedding
rebuild trên vocab 77k).

### Bracket metric trên `network_v4`

Giống baseline (`eval_d3_bracket_positions.py`):

| Metric | n | bits | Δ với gate @ close |
| :--- | ---: | ---: | ---: |
| ALL (base) | 2860 | 4.028 | −0.168 |
| TARGETED close | 511 | 4.256 → **3.316** | **−0.940** |

→ Substrate + gate **đúng trên thước targeted**; **chưa translate** sang sinh prompt khung.

## 9. CHỐT — code changes, đánh giá, tồn đọng

### Code changes (2026-07-05)

| File | Thay đổi | Phase |
|------|----------|:---:|
| `training/ingest_dictionary.py` | D1′: no parent, no W_seq on defs; sense/POS/co-act only | D1′ ✅ |
| `core/parent_store.py` | `bind_sense_resonance()` | D1′ ✅ code / ❌ chưa chạy |
| `scripts/build_dict_checkpoint.py` | out=`network_v4.pkl`, `--skip-resonance` | D1′ ✅ |
| `core/paths.py` | default + viz ưu tiên `network_v4.pkl` | infra ✅ |
| `hca/frame_bracket_gate.py` | position-targeted gate + state machine | D3/TK_017 ✅ |
| `hca/generate_hebbian.py` | wire gate sau G4/repetition; trace | TK_017 ✅ |
| `hca/config.py` | `ar_frame_bracket_weight/floor/min_gap` | TK_017 ✅ |
| `scripts/eval_d3_bracket_positions.py` | Mode B metric | D3 ✅ |
| `scripts/eval_d3_ruler.py` | G-probe entropy sweep | G-probe ✅ |
| `scripts/test_gen_network_v4.py` | smoke sinh gate on/off | test ✅ |
| `tests/test_frame_bracket_gate.py` | unit tests gate | TK_017 ✅ |
| `tests/test_paths.py` | network_v4 default | infra ✅ |
| eval/probe scripts | default checkpoint → `network_v4.pkl` | infra ✅ |

**Chưa làm:** `hca/attention_qk.py` (D2), `ingest_wordnet.py` (D4), viz sense/POS panel, wire sense readout (D4).

### Điểm ĐẠT

- Flat-dump dilution **sửa xong** — ruler về **4.028**, parents giữ **44 633**.
- Substrate từ điển **sạch**: 36 764 sense + 14 POS hub, không nhiễm W_seq essay.
- FRAME_BRACKET mine **35 205 cạnh**; Mode B targeted **Δ −0.940 bits**.
- Gate position-targeted **wired + tested**; QA guard **100/100** trên `network_v4`.
- Checkpoint chính **`network_v4.pkl`** — defaults cập nhật.

### CHƯA ĐẠT / CHƯA LÀM / CHƯA ĐO

| Hạng mục | Trạng thái |
| :--- | :--- |
| Sinh prompt khung (`nếu…thì`) cải thiện E2E | ❌ chưa đạt |
| Gate đổi token khi sinh hybrid | ❌ fire nhưng argmax không đổi |
| `bind_sense_resonance` trong checkpoint | ❌ chưa chạy |
| Probe sense sau D1′ (sạch hơn?) | ❌ chưa đo lại |
| D2 q·k blend | ❌ chưa làm |
| D4 stack neo sense/POS readout | ❌ chưa làm |
| Viz trace neo + `/api/embedding/senses` UI | ❌ chưa làm |
| eval_v4 full (essay + anaphor + composition) trên v4 | ⚠️ chạy 1 lần (anaphor 50%); chưa diff chi tiết vs ISG |
| Graph UI đọc tay ≥2 prompt khung | ❌ chưa |

### Đánh giá sơ bộ (Cursor, 2026-07-05)

**Kết quả cuối cùng hiện tại:** `network_v4.pkl` = **mạng essay sạch như `_t3_isg`** + **lớp
substrate sense/POS “ngủ”** (chưa vào readout) + **gate khung wired nhưng chưa thay đổi sinh
thực tế** khi parent plan chi phối.

Redirect TK_015/016 **đúng hướng về mặt đo lường**: không ship kênh aggregate âm; giữ substrate
cấu trúc; chứng minh gate chỉ hữu ích **targeted**. **Mục tiêu gốc** — sinh qua prompt thấy
khác — **chưa đạt**: cần (a) gate có quyền override khi đóng khung (không bị agenda nuốt),
(b) wire readout sense (D4), (c) có thể tối ưu + chạy `bind_sense_resonance`.

**Ưu tiên tiếp theo đề xuất:** (1) eval sinh chỉ tại vị trí đóng khung trên prompt khung;
(2) gate sau parent plan / có điều kiện override; (3) bind batch nhanh hoặc per-lemma; (4) D4 lite — 1 kênh sense readout ablate.

## 8. Sổ nợ / rủi ro

- D3 gate POS phụ thuộc POS-hub D1 (đã có 14 hub) — nếu POS thô, cạnh xa nhiễu.
- D1′ resonance có thể vẫn thưa cho từ hiếm — trần data (ghi sổ).
- D2 q·k nhiễu function-token (doc 07 §8) — probe token-đầu trước khi build.
- Perf: mỗi rebuild embedding trên vocab 77k ~vài phút (TK_011); build checkpoint ~7 phút.
