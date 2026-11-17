# TK_007 — Essay match: trọng số bigram theo DF-BIGRAM + gold đa-đích

| Field | Value |
|-------|-------|
| **ID** | TK_007 |
| **Status** | **Implemented 2026-07-02 (Claude)** — mục 5 |
| **Nguồn** | TK_005 §6 — regression "phở bò"→bài bí đỏ |

## 1. Tổng quan
`match_episode` chấm bigram bằng **tổng gidf 2 token** → bigram khung phổ biến giữa các
instruction ("hướng dẫn nấu", "viết một bài") mang trọng số lớn oan → bài sai thắng. Đổi sang
**IDF của chính bigram** (đếm trên tập instruction). Kèm: gold cho phép nhiều bài đích hợp lệ
(pool có nhiều bài cùng chủ đề — ca "tập thể dục").

## 2. Hiện trạng chi tiết
- `parent_store.match_episode` (TK_005): `bg_w[bg] = gidf(a)+gidf(b)`. "Hướng dẫn nấu món phở
  bò truyền thống" → alpaca:477 (bí đỏ) thắng alpaca:71 (phở) nhờ "Hướng dẫn"+"nấu món" (match 6/8).
- Gold `essay.*.jsonl` field `episode` chỉ 1 giá trị → "tập thể dục" bị BAD dù đáp đúng chủ đề
  (pool có ≥2 bài thể dục, alpaca:35 và :322).

## 3. Đề xuất giải pháp
1. Build lazy `_ep_bigram_df` (đếm #episode-SFT chứa mỗi bigram trong `match_ids`) — pattern
   `_ensure_seq_index`. `bg_w[bg] = log((N_ep+1)/(1+df_bg)) + 1` — bigram hiếm ("phở bò") nặng,
   khung ("hướng dẫn nấu") nhẹ. Cấu trúc thuần, Luật 0. Invalidate khi thêm episode.
2. Gold `episode` chấp nhận nhiều đích: `"alpaca:35|alpaca:322"`; eval tách `|`.

## 4. Đề xuất thay đổi cụ thể
| File | Thay đổi |
|---|---|
| `core/parent_store.py` | `_ensure_ep_bigram_df()`; `match_episode` dùng idf-bigram; `_register_episode` reset cờ |
| `eval/v4.py::eval_essay` | `match_ok = matched_key in c.episode.split("|")` |
| `data/.../essay.grade_v3_big_a2_t2.jsonl` | ca thể dục: `"alpaca:35\|alpaca:322"` |
| `tests/test_parent_store.py` | toy: bigram khung chung 2 episode (df cao) không lấn bigram riêng |

**Acceptance:** essay `_t2` match ≥ 7/8 (kỳ vọng 8/8, "phở bò" pass), cov ≥ 79, ord ≥ 88 giữ,
guard giữ, pytest xanh.

## 5. KẾT QUẢ (Claude, 2026-07-02)

3 vòng đo (whack-a-mole có kiểm soát, mỗi vòng một root cause thật):
1. df-bigram + gold đa-đích → match 75→88 ("thể dục" ✓) nhưng "phở bò" vẫn fail → lộ **token
   case** ("Phở"≠"phở") → so khớp trên bề mặt CASE-FOLD.
2. Case-fold → "phở bò" ✓ nhưng "bánh mì kẹp" lật sang bài cookies (gold question lặp khung
   "trình bày cách làm bánh" của nó; khung df=1 → idf cao oan) → **distinct-guard mức đơn vị
   bigram** (đơn vị phân biệt nhất phải được phủ).
3. Chốt: **match 7/8 ✓ · cov 79 ✓ · ord 88 ✓ · mix 50 · 277 pass · guard giữ.**

**Ca fail còn lại (đã hiểu, KHÔNG vá cosmetic):** "Trình bày cách làm bánh mì kẹp thịt gà" →
cookies — câu hỏi trùng nguyên văn khung HIẾM của bài khác; idf thuần không phân biệt
khung-hiếm vs nội-dung-hiếm. Hướng fix thật = TK_008 (2): liên tưởng cụm↔cụm đồng-episode
(biết "bánh mì kẹp"≈"sandwich" → bài đúng thắng bằng NGHĨA, không bằng mặt chữ).
