# TK_021 — DRIFT RETRIEVAL trên câu hỏi QA (4 ca tác giả 2026-07-06) — SPEC fix đúng tầng

| Field | Value |
|-------|-------|
| **ID** | TK_021 (+ T4 alias TK_022 ReadoutMode) |
| **Status** | 🟢 **DONE** — 4/4 ca + TK_022b bồi fact đoạn; guard QA 86 / Para 88 / Syn 67; 144 test pass |
| **Nguồn** | Tác giả test UI (A3): 4 ca sai ngữ cảnh + "chỉ sinh 1 câu" |
| **Checkpoint** | `network_v4_full.pkl` |
| **Log đo** | `out/tk022_measure.log` (final); `out/t2t3_measure.log` (T2+T3 trước T4) |

## 1. Chẩn đoán 4 ca (đo `out/diag_4cases.log`)

| Ca | Đường | Gốc |
| :--- | :--- | :--- |
| `Cách làm món phở` → "Cách mạng tháng Tám" | MISS→fallback | focus=[**Cách** 6.72, làm] — "phở" KHÔNG vào focus; "Cách" hoa-đầu-câu là token hiếm riêng → gidf artifact; fact chứa "Cách" pass gate |
| `Nếu trời mưa thì` → "…lẽ trời mệnh trời thì…" | HIT statement 0.266 | focus=[**Nếu** 5.83, trời] — "mưa" bị đẩy; cha không chứa "mưa" vẫn thắng (ngữ pháp nếu…thì đúng — cạnh ngang làm việc; NGỮ CẢNH sai — retrieval) |
| `Nhà Lý là gì` → "Nhà Trần thay nhà Lý 1226" | HIT 0.369 | token generic **Nhà** + power cao lấn át **Lý**; fact đúng không lọt top-20; `theme_neuron` có sẵn (v4 P1) nhưng chưa nối crisp `best_match` |
| `Nhà Trần là gì` → "Nhà Lý thành lập 1009" | HIT **QA** 0.251 | hallucination chéo filler khung "là gì" (hub-guard chỉ phủ khung "thành lập năm nào") |

## 2. VÁ NHANH ĐÃ THỬ — ÂM TÍNH, ĐÃ REVERT (bài học)

1. **fold-gidf cho focus** (diệt artifact hoa-đầu-câu): KHÔNG ăn — focus thường đi đường
   `entity_register_ids` (return sớm, ranking riêng mức UNIT), không qua nhánh sort.
2. **strong-set union 2 thang** trong `_match_has_focus`: đổi lựa chọn ứng viên → **Syn 67→50**.
3. **frame-opener route** (frame QA ruột continuation): không fire — `_prompt_is_open_continuation`
   dùng `query_anchor_clusters` **QUÁ RỘNG** (match MỌI cụm ISG, không riêng WH) → mọi câu chạm
   cụm nào đó đều bị coi "câu hỏi".
4. **WH-indexed retrieval** trong `best_match`: làm tụt guard — đã bỏ; indexed lookup chỉ fallback entity.
5. **slot-consistency trên mọi WH** (`là ai`, `bao nhiêu`): gãy Syn — chỉ bật cho cụm **`là gì`**.

Sau revert: QA 86 / Para 88 / **Syn 67** / 139 test — chuẩn TK_019. Hạ tầng `_ensure_fold_gdf`
/`_fold_gidf` GIỮ trong parent_store (wire T1 tại `entity_register_ids`).

## 3. FIX ĐÚNG TẦNG (4 việc T1–T4)

| Việc | Mục tiêu | Cơ chế |
| :--- | :--- | :--- |
| **T1** | Ca 1 focus đúng | `entity_register_ids`: fold-idf + `focus_pos_alpha` + demote FB-opener; gate crisp giữ legacy focus |
| **T2** | Ca 2 route pool | `wh_anchor_clusters` (filler đa dạng); frame QA ruột-continuation + `_opener_closer_pair_in_content` |
| **T3a** | Ca 3 tie-break | `topic_first_tiebreak` — primary xuất hiện sớm trong cha |
| **T3b** | Ca 4 slot | `slot_consistency_all_frames` — chỉ cụm **`là gì`**; primary trong `match_ids` |
| **T4 (TK_022)** | Ca 3 hoàn toàn | **ReadoutMode KNOWLEDGE** + `theme_neuron` pool + override khi `w_theme(top)<0.22` |

### T4 — Insight triết lý (TK_022)

Hai nhu cầu sinh ngôn ngữ cần **hàm đọc khác nhau** (doc 07 §6, TK_017 ContextGate):

| Mode | Khi nào | Token dẫn | Hàm đọc |
| :--- | :--- | :--- | :--- |
| **EXPRESSION** | continuation, conv pool, W_seq | generic, frame — **power cao** | `rank_matches` + `log(power)` + `match_common_clusters` |
| **KNOWLEDGE** | frame QA + WH **`là gì`** | đặc trưng — **power thấp** | `theme_neuron(primary)` → pool fact; score × `(0.5 + w_theme)`; **không** boost power |

**Không** tăng power cho token đặc trưng — đổi kênh đọc khi cần tri thức (`theme_neuron` = H_slow, v4 §P1).

## 4. "Chỉ sinh 1 câu" — trả lời + quyết định chờ tác giả

ĐÚNG THIẾT KẾ hiện tại: crisp = 1 fact agenda + EOS khi phủ đủ; chế độ ghép-nhiều-fact đã BỎ
theo chỉ đạo TK_019 (không nối fact/đại từ). Muốn đáp NHIỀU CÂU mà đúng triết lý mạng:
**pattern-completion mức ĐOẠN** — pool truyện đang cắt theo CÂU (build_conv_pool); nếu nạp
mức ĐOẠN (2-4 câu liền, đã có trong stories_raw) thì `_conv_pool_tail` phát đuôi dài nhiều câu
tự nhiên — KHÔNG cần cơ chế nối nào. Chờ tác giả duyệt trước khi làm (đổi shape pool).

---

## 5. T1 IMPLEMENTED + ĐO

**Ranking bare trong `entity_register_ids` (CHỈ đường fallback):**
`score = fold_gidf ÷ (1+pos)^0.3 × (0.1 nếu token là FB-opener)` — cờ `focus_pos_alpha` (<0 = cũ).

**Bài học scope:** ranking mới lọt vào gate crisp `best_match` → **Syn 67→33**. Fix: `best_match` ép
`focus_pos_alpha=-1` khi tính focus gate; ranking T1 chỉ ở fallback/facts_about.

| Ca 1 | Trước | Sau T1 |
| :--- | :--- | :--- |
| `Cách làm món phở` | "Cách mạng tháng Tám 1945" ✗ | **"Phở là món ăn…"** ✓ |

---

## 6. T2 IMPLEMENTED + ĐO

**WH-detector thật** — `ParentStore.wh_anchor_clusters()`: cụm anchor chỉ khi
`anchor_left_fillers` ≥ `wh_min_filler_diversity` (default 5). Thay `query_anchor_clusters` trong
`_prompt_is_open_continuation`.

**Frame QA ruột-continuation** — `_is_frame_opener_continuation()` + block trong `generate()`:
`Hỏi: Nếu trời mưa thì ? Đáp:` → conv pool trước crisp retrieval.

**Guard opener→closer** — `_opener_closer_pair_in_content()`: cần cặp mined trong query
(`nếu…thì`); tránh false route `Cách làm món phở` (Cách là FB opener nhưng không có closer).

| Ca 2 | Trước | Sau T2 |
| :--- | :--- | :--- |
| `Nếu trời mưa thì` | statement lạc đề ✗ | **conv pool** (nhà tồi tàn / mưa…) ✓ |

---

## 7. T3 IMPLEMENTED + ĐO

**T3a — topic-first tie-break** (`topic_first_tiebreak`, `topic_first_pos_alpha`):
`_distinct_focus_primary` = token focus **phải nhất** trong query (`Nhà Lý` → `Lý`);
`_pick_entity_candidate` chia score cho `(1+pos)^alpha`.

**T3b — slot-consistency** (`slot_consistency_all_frames`): chỉ WH **`là gì`**;
`_parent_has_distinct_primary` — primary trong `match_ids` (không completion answer).

| Ca | Sau T3 alone (`out/t2t3_measure.log`) |
| :--- | :--- |
| 3 `Nhà Lý là gì` | ✗ vẫn "Nhà Trần thay…" — fact đúng ngoài top-20 |
| 4 `Nhà Trần là gì` | ✓ |

---

## 8. T4 (TK_022) IMPLEMENTED + ĐO FINAL

### Giải pháp

1. **`ReadoutMode` enum** — `EXPRESSION` vs `KNOWLEDGE`; `_readout_mode()` trong `generate_hebbian.py`
   bật KNOWLEDGE chỉ khi `wh_slot_consistency_mode()` (cụm **`là gì`**).
2. **`_best_match_knowledge()`** — seed pool từ `theme_neuron(primary)`; rank trong pool;
   nhân score `(0.5 + retrieval_theme_weight × w_theme)`.
3. **Salient scoring** trong `rank_matches` khi đang rank theme pool: bỏ frame token WH khỏi
   coverage; tắt `log(power)` boost.
4. **Override gate** — chỉ thay top-1 thường khi `w_theme(top) < 0.22` (entity chỉ được **nhắc**
   cuối câu, không phải chủ đề). Tránh regression `Việt Nam có thủ đô là gì` (primary `Nam`
   artifact → theme pool sai nếu override quá rộng).

### Kết quả 4 ca + guard (`out/tk022_measure.log`)

| Ca | Kết quả |
| :--- | :--- |
| `Cách làm món phở` | ✓ Phở là món ăn… |
| `Nếu trời mưa thì` | ✓ conv pool |
| **`Nhà Lý là gì`** | ✓ **Nhà Lý thành lập năm 1009.** (was: Nhà Trần thay…) |
| `Nhà Trần là gì` | ✓ Nhà Trần đánh thắng quân Nguyên… |
| Guard QA / Para / Syn | **86% / 88% / 67%** (giữ) |
| Continuation 3 prompt | giữ |
| Tests | **146 passed**, 1 skipped |

---

## 9. Code changes

### `src/nn_aivie/core/parent_store.py`

| Thêm / sửa | Mô tả |
| :--- | :--- |
| `ReadoutMode` | Enum EXPRESSION / KNOWLEDGE |
| `wh_anchor_clusters()` | T2 — lọc cụm WH theo đa dạng filler |
| `wh_slot_consistency_mode()` | T3b — chỉ cụm `là gì` |
| `_frame_token_ids_in_query()` | T4 — token frame WH cho salient scoring |
| `rank_matches()` | Salient-only khi theme pool; tắt power boost; nhân theme weight |
| `_distinct_focus_primary()` | T3 — primary = rightmost focus trong query |
| `_primary_earliest_pos()`, `_pick_entity_candidate()` | T3a topic-first |
| `_parent_has_distinct_primary()`, `_entity_cands_filtered()` | T3b |
| `_best_match_knowledge()` | T4 — theme-first pool + rank |
| `best_match()` | Wire T3 + T4; override gate `w_top < 0.22` |
| `entity_register_ids()` | T1 fold-idf ranking (đã có từ trước) |

### `src/nn_aivie/hca/generate_hebbian.py`

| Thêm / sửa | Mô tả |
| :--- | :--- |
| `_readout_mode()` | T4 — phân loại KNOWLEDGE vs EXPRESSION |
| `_is_frame_opener_continuation()` | T2 — frame QA ruột continuation |
| `_opener_closer_pair_in_content()` | T2 — guard opener route |
| `generate()` | Sync cờ TK_021/T4 vào `ParentStore`; `readout_mode`, `retrieval_theme_weight` |
| `_prompt_is_open_continuation()` | T2 — dùng `wh_anchor_clusters` |

### `src/nn_aivie/hca/config.py`

| Cờ | Default | Ticket |
| :--- | :--- | :--- |
| `focus_pos_alpha` | 0.3 | T1 |
| `wh_min_filler_diversity` | 5 | T2 |
| `topic_first_tiebreak` | True | T3a |
| `topic_first_pos_alpha` | 0.4 | T3a |
| `slot_consistency_all_frames` | True | T3b |
| `retrieval_theme_weight` | 1.0 | T4 (0 = tắt theme crisp) |
| `knowledge_extra_facts` | 2 | T5 / TK_022b (0 = không bồi fact) |

### Tests

| File | Test mới |
| :--- | :--- |
| `tests/test_seg_graph.py` | `test_wh_anchor_clusters_…` |
| `tests/test_parent_store.py` | `test_topic_first_tiebreak_…`, `test_slot_consistency_…`, `test_theme_pool_beats_power_generic_parent`, `test_readout_mode_knowledge_salient_scoring` |

---

## 11. T5 (TK_022b) — Bồi fact đoạn dài IMPLEMENTED + ĐO (2026-07-06)

**Yêu cầu tác giả:** crisp chỉ sinh 1 câu — thiếu fact; muốn đáp **nhiều câu** về đúng thực thể
(không đại từ/bridge — TK_019).

### Giải pháp

1. **`_knowledge_extra_facts()`** — sau A3, bồi tối đa `knowledge_extra_facts` (default 2) completion
   nguyên khối từ `facts_about`, mỗi fact một câu, ngăn cách `.`.
2. **Fire khi:** frame QA + `wh_anchor_clusters` (mọi WH đa-filler: `là gì`, `là ai`, …) — không
   chỉ `là gì`.
3. **PRIMARY entity (copula split, Luật 0):** token có **gidf thấp nhất** trong query = copula
   `là` → tách `entity` (trước) / frame (sau). Primary = token **phải nhất** entity span
   (`Nhà Lý`→`Lý`, `Nguyễn Huệ`→`Huệ`, `Lý Thái Tổ`→`Tổ`).
   - Không dùng `_frame_token_ids_in_query` — misfire gắn `Nguyễn Huệ` thành cụm-WH.
4. **`facts_about(entity)`** — truyền cả span entity (không chỉ primary) để theme/cluster
   adjacency lọc đúng tên đa-token.
5. **Filter fact bồi:** ≥`min(2, len(entity))` token entity **literal** trong completion +
   có primary; khử trùng Jaccard > 0.6. Không `case_equiv` (artifact `Lý`≡`Society`).

### Kết quả (`out/tk022b_v3.log`)

| Prompt | Kết quả |
| :--- | :--- |
| Lý Thái Tổ là ai | ✓ dời đô + 2 fact **Vua Lý Thái Tổ…** (không còn Tổ tiên/Tổ chức) |
| Trần Nhân Tông là ai | ✓ Trúc Lâm + 2 fact bồi đúng tên |
| Nhà Lý là gì | ✓ thành lập 1009 + fact **Nhà Lý** (không bleed Trần/Lê) |
| Nguyễn Huệ là ai | ✓ 3 fact về Huệ |
| Mây là gì | ✓ định nghĩa + "Mây bao gồm những giọt nước…" |
| Guard QA/Para/Syn | **86% / 88% / 67%** acc giữ |

### Code (`generate_hebbian.py`)

- `_knowledge_extra_facts()` — copula split, `facts_about(entity)`, literal entity filter
- `generate()` — gọi bồi fact khi `knowledge_extra_facts > 0` và `wh_anchor_clusters` hit

### Config

| Cờ | Default |
| :--- | :--- |
| `knowledge_extra_facts` | 2 (0 = tắt, 1 câu như cũ) |

---

## 12. T6 (TK_023) — Chọn fact theo điểm kích thích lan xuôi IMPLEMENTED + ĐO (2026-07-06)

**Yêu cầu tác giả:** thay chọn fact thủ công (đếm token / cụm) bằng **tổng trọng số kích thích**
lan xuôi token → cụm ISG/phrase → cha.

### Giải pháp

1. **`parent_stimulus_score` / `_parent_stimulus_raw`** — runtime trong `parent_store.py`:
   - Token ∈ cụm ISG: `idf(token) × cluster.power`
   - Token rời `slot_chain`: `seg_link.weight`
   - Phrase khớp: `phrase_idf`
   - Token còn lại trên `match_ids`: `idf(token)`
   - `seed_tokens=entity` — chỉ tính kích thích từ span entity (bỏ frame WH)
2. **`rank_facts_by_stimulus`** — API đầy đủ (index + dedup + salient gate); dùng trong test.
3. **`_knowledge_extra_facts` (wire):** pool `facts_about(entity)` (cổng cụm liền kề + theme) →
   **rerank** theo `parent_stimulus_score(..., seed_tokens=entity)` — giữ chất lượng TK_022b,
   thay thứ tự chọn bằng điểm kích thích thay vì chỉ cov IDF trong `facts_about`.
4. **Viz DRY:** `stimulus_activation_edges` / delegate `_parent_match_activation_edges`.

### Kết quả (`out/tk023_measure.log`)

| Prompt | Kết quả |
| :--- | :--- |
| Lý Thái Tổ là ai | ✓ dời đô + 2 fact **Vua Lý Thái Tổ…** |
| Trần Nhân Tông là ai | ✓ Trúc Lâm + 2 fact bồi |
| Nhà Lý / Nhà Trần là gì | ✓ không bleed chéo triều đại |
| Nguyễn Huệ, Mây | ✓ giữ chất lượng v3 |
| Guard QA/Para/Syn | **86% / 88% / 67%** |

### Config

| Cờ | Default |
| :--- | :--- |
| `knowledge_fact_rank` | `"stimulus"` (`"facts_about"` = legacy không rerank) |
| `knowledge_min_salient_stimulus` | 0.15 (dùng trong `rank_facts_by_stimulus` standalone) |

---

## 13. Sổ nợ / hướng sau

- **Curriculum:** thêm parent định nghĩa `Nhà Lý là triều đại…` nếu muốn đáp định nghĩa thay vì fact "thành lập năm 1009" (data, không bắt buộc sau T4).
- **Phrase entity** (`Nhà+Lý` Re-Pair): kiểm tra phrase index checkpoint — boost phrase unit nếu chưa có.
- **Đoạn nhiều câu:** §4 — chờ duyệt tác giả.
- **Ngưỡng `w_top < 0.22`:** hyperparam ablate; đo tay trên thêm ca `là gì` có entity giữa cụm tên.
