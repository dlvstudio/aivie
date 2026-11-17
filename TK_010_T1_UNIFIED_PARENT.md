# TK_010 — ĐÍNH CHÍNH t1: episode là MỘT NƠ-RON CHA THẬT (train thống nhất top-down)

| Field | Value |
|-------|-------|
| **ID** | TK_010 |
| **Status** | **ĐO XONG (Cursor, 2026-07-03)** — build + vá match + eval đầy đủ |
| **Nguồn** | Chỉ đạo tác giả 2026-07-02 (đính chính TK_004): "không chia câu, không chia đoạn — cả episode là một nơ-ron, đối xử như nơ-ron fact" |

## 1. Tổng quan

TK_004 build t1 SAI nguyên lý: chia bài thành CÂU → ingest từng câu làm parent → t1 =
`AbstractNeuron` chỉ chứa con trỏ (không có chuỗi token riêng). Vi phạm nguyên lý train
thống nhất của mạng (v3 docx §1: "một câu / cụm / cả đoạn = MỘT nơ-ron cha, rồi mới phân rã").
Đính chính: **episode = MỘT ParentRecord thật** (chuỗi token đầy đủ), khác cha thường chỉ ở
**tầng link phụ**: fact-links CÓ THỨ TỰ + fact↔fact ngang + trung-gian→fact.

## 1b. Cách hiểu đầu bài (lời tác giả, diễn giải lại)

Một kiểu train DUY NHẤT, top-down, cho mọi quy mô: *cả cụm, cả câu, cả đoạn văn, cả episode
là MỘT nơ-ron* → từ nơ-ron phân rã token (index+power) → từ token tìm nơ-ron trung gian,
nối ngang theo index (token→token, token→trung gian, trung gian→token) → **từ list token tìm
các fact** (cha đã có) → **sắp fact theo thứ tự xuất hiện** → nối vào t1; fact↔fact cũng nối
ngang theo thứ tự đó; trung gian nối tới fact nếu là thành phần của fact. KHÔNG chia câu,
không chia đoạn. Hỏi-Đáp dài đối xử như QA parent thường (match=Hỏi, answer=Đáp), chỉ to hơn.

## 1c. Ý nghĩa (vì sao đáng làm)

Mạng hiện tại: fact là các hòn đảo (chỉ quen nhau qua token chung) — mọi khó khăn composition
(drift, thứ tự phải chế heuristic) quy về thiếu quan hệ HỌC ĐƯỢC giữa các fact. Thiết kế này
tạo **W_seq ở tầng fact**: đồ thị 2 tầng tự đồng dạng (token-graph + fact-graph), t1 là phương
tiện train ghi cạnh tầng trên. Kỳ vọng tác giả: (1) **gen fact→fact** = autoregression 2 cấp
(vĩ mô đi theo cạnh fact học từ diễn ngôn thật, vi mô sinh token TK_009); (2) **tìm fact hiệu
quả hơn** = t1 làm hub truy hồi (thay quét phẳng O(N)) + trung-gian→fact + lan theo cạnh ngang
("tìm fact bằng fact"). Bonus kiến trúc: khớp/sinh/phân rã dùng máy chung — bớt code riêng
(match_episode, lắp ráp essay thủ công thành thừa dần).

## 2. Hiện trạng chi tiết

| Khía cạnh | TK_004 (sai) | Thiết kế đúng (tác giả) |
|---|---|---|
| Episode | `AbstractNeuron{members, match_ids}` — không token seq | `ParentRecord` thật: emit = TOÀN bài; Hỏi-Đáp dài → is_qa (match=Hỏi, answer=Đáp) |
| Câu trong bài | TÁCH thành 3 680 statement parents riêng | KHÔNG chia — bài nguyên vẹn |
| Fact liên quan | member = mảnh của chính bài + resonance đuôi | **link phụ** tìm bằng cộng hưởng từ list token, sắp theo VỊ TRÍ xuất hiện, fact↔fact nối ngang theo thứ tự |
| Trung gian | không phân rã trên bài | **ISG (TK_013)** online + rewire; Re-Pair batch qua `--legacy-repair` |
| Máy sinh/khớp | match_episode + generate_essay chế riêng | rank_matches + completion + decompose DÙNG CHUNG |

Được giữ từ TK_004-009: data `essays.jsonl` (500 bài), cơ chế cộng hưởng (đúng bước 3),
ar-mode TK_009 (agenda giờ lấy từ chuỗi t1), phrase_assoc (xây trên màng episode — nguồn
wiki para giữ nguyên), gate chủ đề, eval essay (ordering/vb_seg).

## 3. Đề xuất giải pháp

1. **Build thống nhất** (`scripts/build_t1_unified.py`, thay build_sft_episodes):
   mỗi bài vi-alpaca → `add_example(encode(instruction + "Đáp:" + full_text))` → MỘT cha QA
   dài (match=instruction, answer=cả bài; R0 split_punct). KHÔNG ingest câu lẻ (xóa đường cũ
   khi rebuild). W_seq/G1 chạy trên chuỗi đầy đủ → liên kết ngang token theo index tự có.
2. **Phân rã**: **ISG** (`rebuild_seg_graph` / online tại ingest) mặc định từ TK_013;
   `decompose_repair` qua `--legacy-repair`. Trung-gian→fact = suy từ `phrase_ids` giao nhau.
3. **Tầng link phụ MỚI trên ParentRecord**: `fact_links: tuple[(parent_idx, pos), ...]`
   — quét cộng hưởng từ list token (tái dùng rank_matches như bind_episode_by_resonance),
   `pos` = vị trí khớp sớm nhất trong chuỗi t1 → sort theo pos = thứ tự fact trước/sau;
   quan hệ ngang fact↔fact suy từ thứ tự này (không thêm lớp cạnh vocab). Serialize.
4. **Sinh**: câu hỏi → rank_matches thường (t1 là cha, tự cạnh tranh bằng coverage/cụm/
   two-tier); đáp = TK_009 ar-mode với agenda = chuỗi t1 chia theo dấu chấm CỦA CHÍNH NÓ
   (không phải member list); fact_links + trung gian làm kênh macro/spreading bổ sung.
5. **Migration đo được**: rebuild `_t3.pkl` từ big_a2 (bỏ 3 680 cha-câu essay, thêm 500 cha
   t1) → chạy đủ: essay eval (kỳ vọng match/cov giữ hoặc tăng vì khớp bằng máy chung) +
   GUARD chặt (rủi ro chính: **cha-dài lấn QA ngắn** — specificity + distinct-ratio hiện có
   phạt cha dài, nhưng PHẢI ĐO); wiki-AbstractNeuron giữ nguyên (nguồn phrase_assoc) —
   migrate sau nếu hướng này thắng.

## 4. Đề xuất thay đổi cụ thể

| File | Thay đổi |
|---|---|
| `core/parent_store.py` | `ParentRecord.fact_links` (+serialize); `bind_fact_links(pi)` quét cộng hưởng + pos; helper `t1_agenda(pi)` (chia chuỗi theo dấu chấm) |
| `scripts/build_t1_unified.py` | build `_t3.pkl`: nạp bài nguyên vẹn + ISG/rebuild + bind_fact_links |
| `hca/generate_hebbian.py` | `generate_essay` v2: rank_matches → cha t1 → ar-mode trên agenda của chính nó; fallback composed giữ |
| `eval` | essay eval trỏ `_t3`; guard đủ 2 checkpoint cũ + `_t3` |
| docs | v4 Trụ 7 đính chính (đã ghi); README 0.6.8 note |

**Acceptance:** essay `_t3` ≥ mức hiện tại (cov 79/ord 88/match 88, vb_seg ≤ 34); guard
QA 100/Para 88/Syn 67 KHÔNG tụt (đặc biệt soi cha-dài lấn QA); pytest xanh; UI essay mode
chuyển sang đọc fact_links.

**Câu hỏi chốt với tác giả trước khi code:**
(a) 3 680 cha-câu tách từ TK_004 → XÓA trong build mới (đúng "không chia câu")? — đề xuất: xóa.
(b) Hỏi-Đáp dài xử lý như QA parent thường (match=Hỏi, answer=Đáp) — đúng ý?
(c) Wiki paragraphs: cũng migrate thành cha-đoạn thật, hay giữ AbstractNeuron làm nguồn assoc?

## 5. KẾT QUẢ (Cursor, 2026-07-03 — TK_010B)

**Build** (`grade_v3_big_a2_t3.pkl`, 108M): 44 633 cha, 500 t1 nguyên vẹn, 3 448 fact_links
(TB 6.9/bài), decompose 36 056 cụm. **282 pass.**

**Vá đo (Cursor):** (1) `eval_essay` crash `ep` trên đường unified; (2) `match_t1_parent` —
khớp instruction bằng bigram-df TK_007 (thay `best_match` BM25-style; regression đầu: match 25%).

### Bảng eval (min_cov=0.6)

| Checkpoint | QA | Para | Syn | Comp (cov/red/top) | Essay (cov/ord/match/vb_seg) | Anaphor |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| `_t3` | **100** | 75 | 67 | 80/0/90 (n=5) | **75/88/88/96%** | 50% |
| `big_a2` (guard) | **100** | **88** | **67** | **64/2/83** | — | 50% |
| `grade_v3` (guard) | **100** | 62 | 50 | **93/0/100** | — | 100% |

Baseline essay `_t2`: cov **79** / ord 88 / match 88 / vb_seg 34.

### Verdict acceptance

| Tiêu chí | Kết quả |
| :--- | :--- |
| QA val `_t3` = 100 | ✅ |
| Essay match ≥ 88 | ✅ 7/8 (ca cookies — sổ nợ TK_008) |
| Essay ord ≥ 88 | ✅ |
| Essay cov ≥ 79 | ⚠️ **75** (−4pp; ca thể dục :322 cov 33%) |
| vb_seg ≤ ~34 | ⚠️ **96%** — agenda t1_agenda verbatim + fact_links; metric chưa tương đương `_t2` |
| Guard big_a2 / grade_v3 | ✅ không regression |
| pytest | ✅ 282 pass |

**Đọc tay:** 7/8 bài đúng chủ đề; đầu segment không xáo ("Việc đọc sách" ✓); fact_links
cuối bài hợp lý (mix 88%). Ca fail: "bánh mì kẹp"→cookies (`alpaca:915`) — trùng khung
hiếm, cần phrase_assoc TK_008.

**Sổ nợ:** Para `_t3` 75 (1 ca paraphrase; QA val vẫn 100 — không lấn QA ngắn).

### Cập nhật TK_013 (2026-07-03)

Checkpoint ISG: `grade_v3_big_a2_t3_isg.pkl` — rebuild ~3 phút, 40k clusters, `ở→đâu` power=19.
Eval fast ISG: QA 100/100, Para 88%, Syn 67%. Xem `TK_013_INCREMENTAL_SEG_GRAPH.md`.
