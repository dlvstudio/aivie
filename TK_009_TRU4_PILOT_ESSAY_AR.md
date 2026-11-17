# TK_009 — Trụ 4 pilot: sinh TOKEN-LEVEL từng segment essay (verbatim → sinh thật)

| Field | Value |
|-------|-------|
| **ID** | TK_009 |
| **Status** | **Implemented 2026-07-02** — mục 5-6. **Superseded 2026-07-09:** essay AR pilot gộp vào A3 multi duy nhất; `generate_essay` xóa (TK_019 §7). |
| **Nguồn** | v4 Trụ 4; TK_004 định vị trung thực ("essay phase-1 = ghép verbatim") |

## 1. Tổng quan
Essay hiện chọn đúng bài/đúng ý/đúng thứ tự (79/88) nhưng từng câu vẫn **dội nguyên văn** —
phê phán gốc v4 ("máy tra cứu") còn nguyên ở mức câu. Pilot Trụ 4: mỗi segment essay sinh bằng
**A3 token-level** (member làm agenda) thay vì copy. Essay là sân thử lý tưởng: agenda rõ, thứ
tự có sẵn, thước đo sẵn (coverage/ordering/verbatim_rate).

## 2. Hiện trạng chi tiết
- `generate_essay`: `emit.extend(ps.completion(pi))` — verbatim từng member.
- `generate_autoregressive` (A3) đã có: 4 kênh Hadamard + index-backbone + G4 verify; đo tốt
  câu ngắn (agenda 7/9), yếu câu dài. Segment essay = câu đơn ~10-20 token → đúng vùng A3 mạnh.
- Sổ nợ chờ đất cắn tại đây: `ar_seq_order2_weight=0.5` (micro bậc-2), `hybrid_markov_answer`
  (đổi verbatim-first — thử lại trong pilot này, KHÔNG đổi suite nền).
- `eval_essay` chưa đo verbatim_rate mức segment.

## 3. Đề xuất giải pháp
1. `generate_essay(..., segment_mode="verbatim"|"ar")` — cờ `essay_segment_mode` (mặc định
   verbatim cho đến khi đo; pilot bật "ar" qua config/eval A/B).
2. Mỗi member: agenda = token nội dung của segment (thứ tự nguồn làm index-backbone), sinh bằng
   `generate_autoregressive` giới hạn trong segment; fallback verbatim khi A3 phủ agenda < ngưỡng
   (an toàn từng segment, không hỏng cả bài).
3. `eval_essay` + `segment_verbatim_rate` (% segment trùng nguyên văn member). A/B verbatim vs ar
   trên gold 8 ca + guard.
4. Trong pilot, ablate: order2 0↔0.5↔1.0, markov on/off — cập nhật giá sổ nợ bằng số thật.

## 4. Đề xuất thay đổi cụ thể
| File | Thay đổi |
|---|---|
| `hca/config.py` | `essay_segment_mode: str = "verbatim"`, `essay_ar_min_cov: float` (fallback) |
| `hca/generate_hebbian.py` | nhánh segment-AR trong `generate_essay` (tái dùng `generate_autoregressive` + agenda per-segment) |
| `eval/v4.py` | `segment_verbatim_rate` trong EssayResult |
| `scripts/eval_v4.py` | in vb-seg; chế độ so sánh A/B |
| tests | toy: segment AR phủ agenda; fallback khi A3 yếu |

**Acceptance (tiêu chí "máy sinh"):** segment_verbatim_rate **giảm rõ** (< 60% từ ~100%) TRONG KHI
coverage ≥ 0.9× và ordering ≥ 0.9× bản verbatim; guard giữ; mọi chênh ghi sổ nợ. Nếu A3 chưa đạt
→ ghi số thật + phân tích kênh (không tắt cờ — luật sổ nợ), pilot vẫn tính là đo xong.

## 5. KẾT QUẢ PILOT (Claude, 2026-07-02)

**Đã dựng:** `essay_segment_mode="verbatim"|"ar"` (+`essay_ar_min_cov=0.6` fallback từng
segment), `vb_seg` trong eval, toy test. 278+ pass.

**A/B trên `_t2` (gold 8):**
| mode | cov | ord | match | vb_seg |
|---|---|---|---|---|
| verbatim | 79 | 88 | 88 | 100% |
| **ar** | 79 | 88 | 88 | **4%** |

**ĐỌC KỸ OUTPUT — nói thẳng (chống ảo tưởng):** acceptance số học ĐẠT (vb_seg<60, cov/ord
giữ) nhưng phần khác-verbatim chủ yếu là **bug xáo 2-3 token ĐẦU segment, lặp hệ thống**:
"Việc đọc sách"→"đọc Việc sách", "Đầu tiên"→"tiên Đầu", "Tập thể dục"→"thể Tập dục",
"Điều này giúp"→"giúp Điều này". Phần còn lại của segment bám nguồn gần nguyên văn.
→ Đây là verbatim-bị-xáo-đầu, KHÔNG phải sinh có nghĩa. **Chẩn đoán:** bước đầu segment,
w0="Đáp:"/dấu chấm — index-backbone chọn token agenda ĐẠT ĐƯỢC từ w0 qua W_seq thay vì
agenda[0] (token đầu không có cạnh từ w0) → nhảy token giữa rồi mới quay về đầu.

**Kết luận pilot:** engine chạy end-to-end + thước đo vb_seg hoạt động; default GIỮ
"verbatim". **Việc kế tiếp (điều kiện flip):** vá trật tự đầu-segment trong A3 (boost
agenda[0] tại bước mở segment / prior thứ tự mạnh hơn ở step 1) → đo lại; chỉ nhận "máy
sinh" khi vb_seg thấp VÀ đọc tay không còn xáo hệ thống. Ablation order2/markov làm cùng
vòng đó.

## 6. VÒNG 2-4 — bóc 3 lớp bug đầu-segment → FLIP DEFAULT (Claude, 2026-07-02)

Chuỗi root-cause (mỗi lớp che lớp dưới, output y hệt qua 2 vòng fix đầu):
1. **Backbone điều kiện "đạt được từ w0"** — agenda[0] không có cạnh từ "Đáp:" → mất boost.
2. **G4 verify mask agenda[0] = −inf** (grounding <0.04) → boost isfinite bỏ qua. Fix: token
   agenda grounded BỞI XUẤT XỨ (từ cha đã học) → boundary được cứu khỏi −inf.
3. **`is_surface_token("Đáp:")=True`** → at_boundary không bao giờ kích hoạt → 2 fix trên
   đứng sau cổng đóng. Fix: ranh giới nhận thêm w0 kết thúc bằng ":" (nhãn frame, chính tả).

**Đo vòng 4 (`_t2`, gold 8):** vb_seg **100→34% THẬT** — đọc tay sạch xáo toàn bộ 8 ca
("Việc đọc sách" ✓, "Đầu tiên" ✓, "Trạng thái lo lắng" ✓); cov/ord/match giữ 79/88/88.
Non-verbatim còn lại chủ yếu = rút gọn dấu câu giữa vế (agenda không mang punct) + nối mềm —
đọc vẫn mạch lạc. **Điều kiện flip đạt cả hai vế → `essay_segment_mode="ar"` thành DEFAULT.**

Định vị trung thực: "sinh" phase này = tái dựng token-by-token bám agenda qua W_seq+backbone
(có thể lược/nối khác nguồn), CHƯA phải sáng tạo nội dung mới — nấc sau: agenda trừu tượng
hơn (ý, không phải token) + trộn resonance vào agenda. Giá đã ghi sổ: chậm hơn verbatim
(mỗi segment một vòng A3). Bài học kiến trúc: lỗi nảy ở GIAO ĐIỂM 3 cơ chế đúng-đơn-lẻ
(backbone×G4×is_surface) tại một hoàn cảnh (đầu segment sau nhãn frame).

---

## 7. `at_boundary`: từ danh sách dấu câu → TÍN HIỆU HỌC ĐƯỢC (2026-07-15/16)

> Chi tiết + số đo: **`TK_012_PURE_GRAPH_CONTENT_CACHE.md` §9.5**. Tổng quan: README §0.4.2.

Vòng 2 (§6) chốt: tại **ranh giới diễn ngôn** (`w0` = nhãn frame / **dấu câu**), W_seq từ `w0`
không mang tín hiệu trật tự → tin trật tự NGUỒN vô điều kiện. Bug đã đo: `"Việc đọc sách"` →
`"đọc Việc sách"`.

Cách nhận ranh giới lúc đó: `vocab.is_punct(w0)` — **danh sách dấu câu viết tay**, vi phạm Luật
0. Đợt 2026-07-15 gỡ toàn bộ `is_punct`/`PUNCT_TOKENS` khỏi repo (chỉ đạo tác giả: *"dấu câu
cũng là token như bình thường"*), nên `at_boundary` mất mốc.

**Thử 2 tín hiệu dẫn xuất để thay — CẢ HAI THẤT BẠI:**

| tín hiệu | `.` | `là` (nội dung) | tách được? |
| :--- | --: | --: | :--- |
| entropy hàng W_seq | 5.93 | **5.45** | ✗ — `?`=3.08 còn thấp hơn `học`=4.39 |
| tỉ lệ đứng cuối `slot_chain` | 0.152 | 0.000 | ✗ — `,`=`:`=`Đáp:`=**0.000**, ngang nội dung |
| **lift P(→EOS)** | **6.0–7.0** | **~0** | ✓ |

→ Đồ thị **không mã hoá** "kết thúc đơn vị" vì **dữ liệu train không đánh dấu nó**. Entropy đo
"đi được nhiều nơi" (hệ từ `là` cũng cao); tỉ lệ-cuối-chuỗi mù với ranh giới GIỮA câu (`,` `:`).

**Lối ra (ý tác giả): EOS cuối mỗi đơn vị train** → `at_boundary` đọc

```
lift(w0) = P(EOS|w0) / P(EOS) nền  ≥  config.ar_eos_boundary_min_lift  (2.0)
```

Dùng **lift** chứ không phải `P` tuyệt đối vì `P` phụ thuộc **độ dài đơn vị**: corpus 1 câu/đơn
vị → `P(.→EOS)=1.000`; corpus bài dài → `0.057`. Ngưỡng `P=0.25` hiệu chỉnh trên corpus toy đã
**trượt** trên mẻ thật. Lift gần như bất biến (6.0 vs 7.0 khi `P` đổi 6×).

Điều kiện hiện tại (`generate_hebbian`):

```python
at_boundary = (not is_surface_token(w0_tok))
              or w0_tok.endswith(":")           # nhãn frame — SỔ NỢ: so-string cuối cùng
              or _ends_unit(state, w0, config)  # HỌC ĐƯỢC từ EOS
```

**Lùi an toàn:** checkpoint train trước 2026-07-15 không có cạnh →EOS → `_ends_unit` = False →
`at_boundary` lùi về chỉ nhận mốc qua nhãn frame = **đúng hành vi §6**, không tệ hơn.

### 7.1 Verify 10/10 — lift hoạt động đúng ở tầng đồ thị

`network_v10` (train 2026-07-16), `scripts/verify_checkpoint.py`:

```
P(EOS) nền = 0.00358
'.' P(→EOS)=0.057  →  lift = 16.0  (ngưỡng 2.0)  ends_unit=True
từ nội dung ('là'/'của'/'và'/'Việt') nhận nhầm làm mốc: KHÔNG
```

**Lift chứng minh được thiết kế:** chính `P=0.057` này đã làm ngưỡng tuyệt đối `0.25` **trượt**
(verify lần đầu FAIL) — mà lift cho **16.0**, vượt xa ngưỡng 2.0. `P` phụ thuộc độ dài đơn vị,
lift thì không.

### 7.2 NHƯNG: `at_boundary` KHÔNG CHẠY trên đường mặc định

Eval `network_v10` sau khi rebuild sidecar (119 093 ctx EOS) + lift bật → summary **GIỐNG HỆT**
bản trước. Lý do (xem TK_012 §9.7.1):

```python
# generate() — mặc định ar_tail_as_agenda_opener=False
tail = _conv_pool_tail(...)
if tail:
    return _emit_tail(..., verbatim_only=True)
# _emit_tail:1023
if verbatim_only or not config.ar_tail_as_agenda:
    return generated + tail        # ← RETURN, KHÔNG vào generate_autoregressive
```

**`at_boundary` sống BÊN TRONG `generate_autoregressive`** → lưới verbatim return trước → nó là
**no-op** ở đường mặc định. Chỉ bật khi chạy `--no-verbatim-nets` hoặc khi prompt đi lối
retrieval→plan→A3.

Nghĩa là: **cơ chế đúng và đã verify, nhưng chưa được dùng.** Không phải hồi quy — chỉ là chưa
tới lượt. Khi nào bỏ được nhánh verbatim (mục tiêu Trụ 4) thì lift mới thật sự gánh việc.

### 7.3 SỔ NỢ — ca gốc §6 KHÔNG có test

`at_boundary` **đã đổi cơ chế nhận mốc** (`is_punct` → lift) mà ca đã sinh ra nó
(`"Việc đọc sách"` → `"đọc Việc sách"`) **không có test nào phủ**. 184 test + self-retrieval +
eval discourse đều mù — chúng không kiểm trật tự từ ở đầu segment sau dấu chấm.

Rủi ro **hiện đang thấp** vì §7.2: đường mặc định không chạy tới `at_boundary`. Nhưng nó sẽ
thành rủi ro thật ngay khi bỏ lưới verbatim. **Kiểm TRƯỚC khi làm việc đó.**

**Việc cần:** test dựng đúng ca §6 (agenda nhiều segment, `w0` = `.`, kiểm agenda[0] không bị
token giữa vượt lên) — chặn hồi quy im lặng cho mọi lần đụng `at_boundary` sau này.

### 7.4 Đo `--no-verbatim-nets` trên `network_v10` — lưới verbatim CỨU ở QA

Đọc tay 10 ca khác nhau (chi tiết + bảng: TK_012 §9.7.2): **OFF hơn 5 · hoà 2 · OFF KÉM 3.**

- **OFF thắng ở prompt mở**: `'8 ngày qua đã xuất hiện ba'` → *"ba **vùng thả dù** : "Natasha"…"*
  (nối đúng cú pháp + nghĩa). ON thì dội `Lịch sử : Tiêu đề :` — **scaffolding lạc đề**.
- **ON thắng ở QA**: `'Hỏi: Nhà Lý là gì Đáp:'` → ON đáp trọn vẹn (Lý Công Uẩn, 974, 1009, 1010);
  OFF ra mảnh giữa đoạn `: "Thân Vệ là người khoan thứ…"` **không trả lời câu hỏi**.
- **A3 phá text**: `'Chương 277'` → OFF sinh `"nhìn chằm **Mạn** chằm"` — chèn token phá từ láy.

→ Lưới verbatim **hại ở prompt mở, CỨU ở QA/frame** — đúng như tên gọi và README §0.4.1 *"giữ
đúng fact câu đã thuộc"*. **KHÔNG đổi mặc định toàn cục**; nếu làm thì tách theo bucket.

`Lịch sử : Tiêu đề :` lộ ra vấn đề **khác**: scaffolding tài liệu nằm **trong dữ liệu train** →
hại cả hai chế độ. **Dọn corpus rẻ hơn và không đánh đổi.**
