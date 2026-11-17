# TK_005 — Essay: khớp t1 theo ĐƠN VỊ CỤM + metric match_accuracy

| Field | Value |
|-------|-------|
| **ID** | TK_005 |
| **Status** | In progress (Claude, 2026-07-02). **Superseded 2026-07-09:** `match_episode` / essay metric không còn trong runtime (TK_019 §7). |
| **Repo** | `nn-aivie`, nhánh `cursor2` |
| **Nguồn** | TK_004 mục 5 — 2/8 ca essay fail do khớp nhầm episode |

## 1. Tổng quan

`match_episode` (khớp câu hỏi → nơ-ron t1 SFT) đang chấm theo **bag token đơn** → token
instruction phổ biến ("lợi", "ích") lấn át cụm mang nghĩa ("đọc sách") → khớp nhầm bài.
Đồng thời metric `on_topic` mượn từ composition không hợp essay (41% oan — segment hợp lệ
không lặp từ chủ đề). Ticket này: khớp theo **đơn vị cụm liền kề** + thay on_topic bằng
**match_accuracy** (khớp đúng bài nguồn — đo trực tiếp cái ta quan tâm).

## 2. Hiện trạng chi tiết

- `parent_store.match_episode` (TK_004): coverage = Σ gidf(token ∩ match_ids) / Σ gidf(query),
  distinct-guard 75%. Token đơn, không cụm.
- Fail đo được (gold 8 ca, `_t2`): (a) "Nêu những lợi ích của thói quen đọc sách" → khớp bài
  khác chứa "lợi/ích" (đáp lẫn "Ích chi cực tắc…"); (b) "Căng thẳng thường có những biểu hiện
  nào" → bài chiến tranh (cũng chứa "căng thẳng" — đa nghĩa); bài đúng có instruction chứa CẢ
  "biểu hiện" + "căng thẳng" nhưng thua vì từng token bị chấm rời.
- `eval_essay.on_topic` = tỉ lệ segment chứa focus-token câu hỏi — nghĩa sai với essay.
- Gold `essay.grade_v3_big_a2_t2.jsonl` chưa ghi bài nguồn kỳ vọng → không đo được "khớp đúng bài".

## 3. Đề xuất giải pháp

1. **Khớp theo đơn vị liền kề:** ngoài token đơn, chấm **bigram liền kề** trùng giữa query và
   `match_ids` (không cần kho Re-Pair — instruction chưa được decompose; bigram surface là tín
   hiệu cấu trúc thuần, Luật 0, cùng tinh thần phrase-bonus 0.5.4). Bigram trùng ("đọc sách",
   "biểu hiện") được cộng gidf của CẢ HAI token — bài đúng (khớp nhiều đơn vị dài) vượt bài chỉ
   trùng token rời. Kỳ vọng sửa cả (a) lẫn (b): bài đúng của "căng thẳng" khớp thêm "biểu hiện".
2. **Metric:** thêm field `episode` (doc_id kỳ vọng) vào gold → `match_accuracy` = % ca khớp
   đúng bài nguồn. Bỏ on_topic khỏi bảng essay (giữ coverage/ordering/match).

## 4. Đề xuất thay đổi cụ thể

| File | Thay đổi |
|---|---|
| `core/parent_store.py::match_episode` | cộng điểm bigram liền kề (query×match_ids), giữ distinct-guard |
| `eval/v4.py::eval_essay` | + `matched_key` (episode khớp qua match_episode), `match_ok`; bỏ on_topic |
| `scripts/eval_v4.py` | in `match=…%` thay on_topic |
| `data/grade_03/qa/essay.grade_v3_big_a2_t2.jsonl` | + field `episode` (alpaca:N kỳ vọng) |
| `tests/test_parent_store.py` | toy: 2 episode chung token rời, khác cụm → chọn đúng cụm |

**Acceptance:** essay `_t2`: match_accuracy ≥ 7/8, coverage ≥ 75%, ordering ≥ 75%; 2 ca fail cũ
pass; guard (QA 100/Para 88/Syn 67/anaphor 50) giữ; pytest xanh.

## 5. Hoàn thành (2026-07-02, Cursor)

**Status:** Implemented.

| Hạng mục | File | Ghi chú |
|---|---|---|
| Bigram liền kề trong `match_episode` | `src/nn_aivie/core/parent_store.py` | Đã có sẵn (TK_005): token + bigram gidf-coverage, distinct-guard 75% |
| `eval_essay`: `matched_key`, `match_ok`; bỏ `on_topic` | `src/nn_aivie/eval/v4.py` | Sửa schema mismatch `EssayResult` |
| In `match=…%` thay `on_topic` | `scripts/eval_v4.py` | Mỗi ca in episode key + ok/BAD |
| Gold + field `episode` (8 ca) | `data/grade_03/qa/essay.grade_v3_big_a2_t2.jsonl` | alpaca:0, 14, 34, 35, 42, 43, 71, 18 |
| Test bigram thắng token rời | `tests/test_parent_store.py` | `test_match_episode_prefers_adjacent_bigram_units` |

**Test:** `pytest` 276 passed, 1 skipped.

**Chưa đo lại trên checkpoint `_t2`:** acceptance `match_accuracy ≥ 7/8` + coverage/ordering — cần chạy `python scripts/eval_v4.py --checkpoint checkpoints/grade_v3_big_a2_t2.pkl`.

## 6. REVIEW + ĐO (Claude, 2026-07-02)

Sửa thêm khi review: `eval_essay` dùng `min_score=0.0` (≠ generate 0.15) → eval báo khớp cả khi
generation fallback — đã đồng bộ ngưỡng qua `HCAConfig().essay_min_match`.

**Đo `_t2` (acceptance):** cov **79%** ✓(≥75) · ordering **88%** ✓(≥75) · match **6/8** ✗(đòi 7/8)
· guard giữ (QA 100/Para 88/Syn 67/anaphor 50) ✓ · 276 pass ✓.
- **2 ca fail cũ PASS đúng thiết kế**: "đọc sách"→alpaca:0 cov100, "căng thẳng"→alpaca:43 cov100.
- 2 ca BAD match: (a) "tập thể dục"→alpaca:322 — pool có nhiều bài thể dục, đáp ĐÚNG chủ đề,
  gold khóa-bài (artifact); (b) **"phở bò"→alpaca:477 (bí đỏ) — REGRESSION mới**: bigram generic
  "Hướng dẫn nấu/nấu món" (sum-unigram-gidf) thắng bài đúng. **Việc tiếp: trọng số bigram theo
  độ hiếm CỦA BIGRAM (đếm df bigram trên instruction), không phải tổng gidf 2 token.**

Kết luận: giữ (net +2 ca so trước, cov 67→79, ord 75→88); regression phở bò ghi việc tiếp.
