# TK_035 — TRAINING ĐƠN VỊ ĐOẠN 2–4 CÂU (giữ , . ! ?)

| Field | Value |
|-------|-------|
| **ID** | TK_035 |
| **Status** | 🟢 DONE (2026-07-09) — v10 stack + hợp nhất A3 multi-only |
| **Nguồn** | Tác giả: cần học cụm nhiều câu có dấu phẩy, chấm, than — không chỉ câu-lẻ |

## 1. Tổng quan

Đơn vị học mới = **2–4 câu liên tiếp** trong cùng đoạn wiki/lịch sử, hoặc cửa sổ trượt trên văn nói.
Giữ nguyên dấu `,` `.` `!` `?` trong token stream — discourse học từ data, không rule.

Không tách mạng ngữ pháp riêng: **cùng ParentStore**, priority two-tier `0.36` (band `0.355–0.395`).

## 2. Kiến trúc

```
doc (doc_id, para) → cửa sổ 2–4 câu → parent priority=0.36
                                              ↓
generate() discourse-route → _discourse_pool_indices (conv ∪ paragraph)
                                              ↓
_pick_tail_from_ranked → ưu tiên đuôi DÀI từ pool đoạn
```

| Thành phần | File |
| :--- | :--- |
| Core | `src/nn_aivie/data/paragraph_units.py` |
| Build pool | `scripts/build_paragraph_pool.py` |
| Probe D0 | `scripts/probe_paragraph_units.py` |
| VarLM đoạn | `scripts/build_varlm.py --paragraph-units` |
| Wire | `hca/generate_hebbian.py`, `hca/config.py` |
| Tests | `tests/test_paragraph_units.py` |

## 3. Flags (`HCAConfig`)

| Flag | Default | Ý nghĩa |
| :--- | :--- | :--- |
| *(pool đoạn)* | **luôn bật** | `ar_paragraph_pool` đã gỡ — discourse-route luôn gộp conv ∪ paragraph |
| `paragraph_pool_min_score` | `0.08` | Ngưỡng khớp (priority 0.36 scale thấp hơn conv 0.3) |
| `paragraph_prefer_long_tail` | `True` | Ưu tiên đuôi dài từ parent đoạn |

## 4. Lộ trình đo

| Phase | Nội dung | Lệnh |
| :--- | :--- | :--- |
| **D0** | Probe thống kê dấu câu | `python scripts/probe_paragraph_units.py` |
| **D1** | Build pool | `python scripts/build_paragraph_pool.py --checkpoint … --out …` |
| **D2** | A/B continuation đọc tay | 7 prompt khung (giữ R3b) |
| **D3** | VarLM trên đoạn | `python scripts/build_varlm.py --paragraph-units` |
| **D4** | Guard QA/Para/Syn | spot-check / discourse eval (không còn `eval_v4` CLI) |

## 5. Acceptance

- D0: ≥30% đơn vị có `,` hoặc ≥2 câu; có mẫu `!`/`?` từ conv
- Unit tests: `pytest tests/test_paragraph_units.py` pass
- D2: ≥1 prompt cho đuôi ≥2 câu / có dấu phẩy (đọc tay)
- Guard: không đụng crisp QA (pool two-tier) — **2026-07-09:** crisp/hybrid mode đã gỡ; guard = discourse + spot QA

## 6. Sổ nợ

- Wiki đoạn chưa lọc hết rác từ điển → quality gate per-câu
- Pool chưa có FRAME_BRACKET gate (khác conv pool) — dựa rank + đuôi dài
- Essay AR chưa đo vb_seg sau paragraph pool (perf TK_011)
- Crisp multi-fact vẫn cần KNOWLEDGE mode (TK_024) — đoạn ≠ thay fact ranking

## 7. Kết quả implement (2026-07-09)

- Module `paragraph_units.py`: `iter_units_from_doc_jsonl`, `iter_units_from_conv_jsonl`, quality gate
- `build_paragraph_pool.py`: nạp từ statements_doc + history_wiki + conv
- Wire discourse pool + prefer long tail
- `build_varlm --paragraph-units` cho ctx xuyên dấu phẩy trong đoạn

### 7b. BUILD trên checkpoint mới nhất (`network_v9_histwiki.pkl`, 2026-07-09)

| | Trước | Sau |
| :--- | ---: | ---: |
| parents | 149 627 | **174 627** (+25 000) |
| vocab | 113 260 | 113 260 |
| đơn vị nạp | — | 25 000 (OOV 0) |
| output | — | `network_v9_histwiki_paragraph.pkl` (~397 MB) |
| thời gian build | — | ~531s (load 125s + index 129s + save 263s) |

**D0 probe:** 96 870 đơn vị unique trong corpus (100% multi_sent); stories có `!`/`?` dày.

**Default paths:** `default_checkpoint` → `network_v9_histwiki.pkl`; `viz_default_checkpoint` →
`network_v9_histwiki_paragraph.pkl` (nếu có).

### 7c. Guard `eval_v4 --fast` (`out/tk035_eval_fast.log`)

| Bộ | retr | gen-hy | vb-hy |
| :--- | ---: | ---: | ---: |
| QA val | 100% | **86%** | 14% |
| Paraphrase | 88% | **88%** | 50% |
| Synonym | 67% | **67%** | 17% |

→ **Guard GIỮ** — khớp baseline `network_v9_histwiki` (86/88/67). Fails là ca đã biết
(synonym role-query drift wiki), không regression TK_035.

### 7d. VarLM đoạn (`out/tk035_varlm_para.log`)

| | Giá trị |
| :--- | ---: |
| đoạn đếm | 231 546 |
| token | 11 942 090 |
| k=3 sau prune | 968 120 |
| k=4 sau prune | 1 258 702 |
| output | `network_v9_histwiki_paragraph.varlm.pkl` (59 MB) |

Sidecar tự load khi `load_checkpoint` → Graph UI / generate dùng VarLM đoạn.

### 7e. A/B paragraph pool (`out/tk035_paragraph_ab.log`)

7 prompt khung R3b, so `ar_paragraph_pool` ON vs OFF trên `network_v9_histwiki_paragraph.pkl`:

| Kết quả | |
| :--- | ---: |
| flips | **1/7** |
| on_longer | **1/7** |

Flip duy nhất — `'vì em chăm học'`:

- **ON:** đuôi dài đoạn văn (Kim Dung, Đông Nam nhật báo…)
- **OFF:** đuôi ngắn wiki (`Trẻ em có quyền được chăm sóc`)

5 prompt còn lại giữ nguyên; `'Nếu em học chăm'` đã multi-sent (`s=2`) cả ON/OFF.
→ Pool additive, không regression; đuôi dài chỉ kích hoạt khi rank khớp đoạn.

### 7f. Ruler VarLM in-harness (`out/tk035_varlm_ruler.log`)

Held-out 200 câu essay (`data/eval_v4/heldout_lm.jsonl`), `next_token_lm`:

| Checkpoint | w | bits | top-1 |
| :--- | ---: | ---: | ---: |
| `_paragraph` (base) | 0.00 | **4.401** | 19.9% |
| `_paragraph` + VarLM đoạn | 0.35 | 4.971 | 18.0% |
| `_paragraph` + VarLM đoạn | 0.50 | 4.969 | 17.9% |

→ **Substrate base-LM không đổi** (4.401 = v9). VarLM đoạn **âm trên essay held-out**
(+0.57 bits @ w=0.35) — kỳ vọng: corpus VarLM = wiki/lịch/sử đoạn, không phải vi-alpaca.
Lợi ích VarLM đoạn nằm ở **continuation khung** (ctx dài qua dấu phẩy), không ruler essay.

**Khuyến nghị runtime:** giữ `ar_varlm_weight` thấp (~0.2–0.35) khi QA crisp; tăng **~0.5**
khi prompt khung / đoạn dài (`nếu…`, `tuy…`, `Nếu em học chăm`).

### 7g. Ruler đúng domain — wiki/lịch sử đoạn (`out/tk035_paragraph_ruler.log`)

Held-out 400 đơn vị 2–4 câu (`data/eval_v4/heldout_paragraph_lm.jsonl`, split 20% doc_id):

| w | bits | top-1 | Δ vs base |
| :--- | ---: | ---: | ---: |
| 0.00 | 4.927 | 16.4% | — |
| 0.35 | 4.946 | 15.1% | +0.02 (âm nhẹ) |
| **0.50** | **4.787** | 15.4% | **−0.14** |

→ Trên domain đúng, VarLM đoạn **cải thiện ruler** @ w=0.5 (khác essay held-out §7f).
Script: `scripts/eval_paragraph_ruler.py`.

### 7h. v10 paragraph stack — DONE (2026-07-09)

Sau scale TK_024 C → `network_v10_histwiki.pkl` (151 430 parents):

| Bước | Kết quả |
| :--- | :--- |
| `build_paragraph_pool` | `network_v10_histwiki_paragraph.pkl` — **176 430** parents (+25 000) |
| `build_varlm --paragraph-units` | sidecar 59 MB; 237 577 đoạn / 12.26M token; k3=980 445 k4=1 276 037 |
| **Guard spot-check** | QA **86** / Para **88** / Syn **67** — GIỮ (`out/tk035_eval_v10_para_fast.log`; đo trước khi gỡ `eval_v4` CLI) |
| **A/B pool** | flips **1/7**, on_longer **1/7** — giống v9 (`out/tk035_paragraph_ab_v10.log`) |

**Default paths → v10:** `paths.py` `default_checkpoint` → `network_v10_histwiki.pkl`;
`viz_default_checkpoint` → `network_v10_histwiki_paragraph.pkl`. `test_paths.py` cập nhật (3 pass).

### 7i. Hợp nhất A3 multi-only (2026-07-09)

Chỉ đạo tác giả: **một** đường sinh — bỏ song song hybrid/crisp/essay/pool-OFF.

| Thay đổi | Chi tiết |
| :--- | :--- |
| **Code** | Xóa `generate_hybrid`, `generate_composed`, `generate_essay`; `ar_paragraph_pool` cờ gỡ (pool luôn on) |
| **Viz** | `completion_graph` luôn `multi=True`; GraphExplorer không dropdown mode |
| **Eval** | `eval_discourse_prompts.py` — prompt corpus-derived, sweep varlm; xóa `eval_paragraph_ab`, `eval_v4` CLI |
| **Tests** | 131 pass; xóa legacy episode/SFT/tk010, migration checkpoint, hf_corpus |

### 7j. VarLM cluster-aware + discourse eval (2026-07-09)

| Bước | Kết quả |
| :--- | :--- |
| `build_varlm.py --use-clusters` | Sidecar `network_v10_histwiki_paragraph.varlm.pkl` (~80 MB); ctx encode qua ISG cluster |
| Ruler paragraph held-out | w=0.50 → **4.730 bits** (tốt hơn base 4.849 trên v10) |
| `eval_discourse_prompts.py` | Bucket: history_wiki_para, statements_doc, conv, common_noun, frame; mode `gen multi varlm=W` |
| Log mẫu | `out/tk035_discourse_prompts_smoke.log`, `out/tk035_discourse_prompts_full.log` |
