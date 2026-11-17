# TK_010B — HANDOFF cho Cursor: ĐO `_t3` + chốt TK_010 (Claude nghỉ vài giờ)

| Field | Value |
|-------|-------|
| **ID** | TK_010B |
| **Status** | **Done (Cursor, 2026-07-03)** — eval đầy đủ + ghi TK_010 §5; **TK_013** thêm `_t3_isg.pkl` |
| **Nguồn** | TK_010 đã code + build XONG; chỉ còn ĐO + ghi kết quả |
| **Nhắc** | Kỷ luật dự án: PHẢI chạy eval đầy đủ và dán số vào ticket (pattern cũ: code xong bỏ đo → Claude review phát hiện regression). KHÔNG commit — tác giả tự commit |

## 1. Tổng quan

TK_010 (episode = MỘT nơ-ron cha thật, tầng fact_links) đã implement + build xong
`checkpoints/grade_v3_big_a2_t3.pkl` (108M): 500 cha t1 nguyên vẹn, decompose 36 056 cụm,
**3 448 fact_links (TB 6.9/bài)**, 282 test pass. Việc còn lại: **đo acceptance + ghi kết
quả vào TK_010 + nhật ký v4**; nếu lệch thì xử theo hướng dẫn mục 4.

## 2. Hiện trạng chi tiết

**Đã xong (không đụng lại):**
- `ParentRecord.fact_links` + `tag` (+serialize); `bind_fact_links` (cộng hưởng, sắp theo
  vị trí); `t1_agenda` (bài tự chia theo dấu chấm) — `core/parent_store.py`.
- `generate_essay` nhánh THỐNG NHẤT: match_episode None → `best_match` chung → cha có
  fact_links/đáp dài → agenda của chính nó (ar-mode TK_009) + fact_links qua gate TK_008.
- Eval: `matched_key` qua `tag` khi đi đường unified; gold `essay.grade_v3_big_a2_t3.jsonl`
  (copy từ _t2, episode field = "alpaca:N" = tag).
- Viz: nhánh unified trong `export_graph_v3` (node episode → chuỗi segment → fact_links).
- **Bug nền đã vá**: `tokenize_punct` tách "Đáp:" → ingest_qa(split_punct=True) từng bỏ
  qua lặng lẽ MỌI cặp — giữ nhãn frame nguyên vẹn (`teaching/align.py`).

**CHƯA đo** (việc của ticket này). **Baseline để so:**
- Essay `_t2`: cov 79 / ord 88 / match 88 / vb_seg 34 (mode ar default).
- Guard big_a2: QA 100 / Para 88 / Syn 67 / comp 64-2-83 / anaphor 50.
- Guard grade_v3: QA 100 / Para 75(62 với fuzzy sổ nợ — số hiện hành 62) / Syn 50 / comp 93-0-100.

## 3. Các bước thực thi (đúng thứ tự, dán số vào ticket)

```bash
python -m pytest -q                                                      # ≥ 282 pass
python scripts/eval_v4.py --checkpoint checkpoints/grade_v3_big_a2_t3.pkl   # ĐO CHÍNH
python scripts/eval_v4.py --checkpoint checkpoints/grade_v3_big_a2.pkl      # regression
python scripts/eval_v4.py --checkpoint checkpoints/grade_v3.pkl             # regression
```

Trên `_t3` đọc kỹ: (a) bảng retr/gen-hy — **QA val PHẢI = 100** (rủi ro số 1: cha-dài
lấn QA ngắn trong retrieval); (b) section Essay (cov/ord/match/mix + từng ca, đối chiếu
baseline); (c) đọc tay 2-3 bài sinh ra (đầu segment không xáo, fact_links cuối bài không
lạc chủ đề).

A/B phụ (nên làm): essay `_t3` mode `verbatim` vs `ar` — dùng `scripts/essay_ab_t3.py`.

### A/B vb_seg `_t3` — KẾT QUẢ (2026-07-03, `scripts/essay_ab_t3.py`, n_gold=8)

| mode | cov | ord | match | vb_seg |
|---|---|---|---|---|
| verbatim | 75% | 88% | 88% | **100%** |
| ar | 75% | 88% | 88% | **96%** |

**Δvb_seg = +4%** (verbatim − ar). **Kết luận:** vb_seg cao ở `_t3` KHÔNG phải artifact của
verbatim-mode — chế độ `ar` (sinh token-level) cũng cho **96% segment nguyên văn**. Nghĩa là
`_t3` đang **tái tạo/tra cứu** nguyên văn bài nguồn chứ chưa **sinh** thực sự; grounding G4 của
A3 kéo mỗi bước về đúng token member t1. Đây là **món nợ** (không tắt cơ chế): việc "cách mạng
hóa v4" (giảm vb_seg mà giữ cov/ord) cần điều tra A3 grounding — ghi vào nhật ký v4. Kết quả
JSON: `out/essay_ab_t3.json`. (Ghi chú vận hành: A/B chạy ~51 phút — chính là điểm nóng TK_011
Phase B nhắm tới: essay AR-mode trên checkpoint 108MB.)

## 4. Xử lý khi lệch (đo TRƯỚC, vá SAU — ghi số cả trước lẫn sau)

| Triệu chứng | Hướng xử lý |
|---|---|
| **QA val < 100 trên `_t3`** (cha dài lấn) | GHI SỐ + liệt kê ca; thử tăng phạt specificity cho cha dài trong `rank_matches` (đã có `matched_units/len(pset)`) — chỉ chỉnh khi có số, đo lại cả 3 checkpoint |
| **Essay match < 88** | Mang bài học TK_007 vào đường best_match/rank_matches ở mức instruction: case-fold khi so match_ids + trọng số bigram theo df (xem TK_007 mục 5 — 3 vá đã mô tả sẵn) |
| **vb_seg > 34 nhiều** | Kiểm tra `essay_segment_mode="ar"` còn default không; ca fallback verbatim tăng → xem `essay_ar_min_cov` |
| **fact_links phát rác cuối bài** | Gate TK_008 (`essay_resonance_gate`) đang bảo vệ — nếu lọt, GHI ca vào sổ nợ TK_008 §6, đừng vá mặt-chữ thêm |
| Số TRUNG TÍNH/thấp hơn baseline | Luật sổ nợ: KHÔNG tắt cơ chế — ghi giá + nghi vấn gốc vào nhật ký v4 mục 3.5 |

## 5. Ghi kết quả (bắt buộc)

1. TK_010: thêm mục "## 5. KẾT QUẢ" — bảng số 3 checkpoint + verdict acceptance
   (essay ≥ 79/88/88, vb_seg ≤ ~34, guard giữ, QA `_t3` = 100).
2. `docs/NN-AIVIE v4.md` nhật ký mục 6: 1 dòng kết quả TK_010 (format như các dòng 07-02).
3. KHÔNG commit; KHÔNG xóa checkpoint cũ (`_t2` vẫn là baseline so sánh).

## 6. Việc TÙY CHỌN nếu còn thời gian (độc lập, rẻ)

- **C0 (doc `docs/study/06`)**: gieo 5-10 concept (định nghĩa+ví dụ) vào `_t3` qua
  `bind_fact_links`, đếm attach đúng miền — 1 buổi, ghi số vào doc 06.
- **N0a (doc `docs/study/05`)**: cluster episode×bigram-hiếm → đọc tay ≥5 cluster chủ đề.
- KHÔNG làm: scale essay, re-ingest, TK_003 (anaphor — tác giả hoãn), đổi default flags.

## 7. Cập nhật TK_013 (2026-07-03)

Checkpoint ISG: `grade_v3_big_a2_t3_isg.pkl` (rebuild từ `_t3`, log `out/rebuild_isg_t3.log`).
Eval fast: QA 100/100, Para 88%, Syn 67%. Chi tiết: `TK_013_INCREMENTAL_SEG_GRAPH.md`.
