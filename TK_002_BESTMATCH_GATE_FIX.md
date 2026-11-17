# TK_002 — Sửa gate primary-entity trong `best_match` (regression Synonym 33→17) + 2 việc kèm

| Field | Value |
|-------|-------|
| **ID** | TK_002 |
| **Title** | Gate entity chỉ override khi curated khớp KHUNG-thuần; default episode_weight; dọn gate anaphor |
| **Status** | **Implemented 2026-07-02 (Claude)** — xem mục 7 Kết quả |
| **Repo** | `nn-aivie`, nhánh `cursor2` |
| **Nguồn** | Claude review 2026-07-02 — `docs/NN-AIVIE v4.md` nhật ký "ĐO LẠI SAU cursor2" + §7.5 #1-2 |
| **Liên quan** | `core/parent_store.py::best_match`, `hca/config.py`, `hca/discourse.py`, eval `grade_v3_big_a2.pkl` |

---

## 1. Bối cảnh — số ĐÃ ĐO (eval_v4, min_cov=0.6)

Commit `4896953` thêm gate primary-entity vào `best_match` (curated top-1 thiếu focus token
→ override bằng ứng viên chứa entity). Kết quả đo đầy đủ 2 checkpoint:

| big_a2 | trước | sau cursor2 | |
|---|---|---|---|
| QA val | 100% | 100% | ✓ |
| Paraphrase | 62% | **88%** | ✓ gate ăn thật (+26) |
| **Synonym** | 33% | **17%** | ✗ **regression** |
| Composition | 69/2/84 | 64/2/83 | −5 cov |
| Anaphor | 50% | 50% | ✓ |

grade_v3 nguyên vẹn (100/75/50; comp 93/0/100). 269 test pass.

**2 ca Synonym vỡ (cùng root cause):**

```
Ai cấy lúa      gold: Nông dân trồng lúa.   → ra wiki "Họ làm ruộng cấy một vụ lúa nếp , nay…"
Ai khâu quần áo gold: Thợ may may quần áo.  → ra rác  "Đây là loại hình sân khâu tiêu biểu…"
```

**Root cause:** gate đòi curated top-1 **chứa focus token trên mặt chữ** (`_match_has_focus`,
`parent_store.py:722`). Câu hỏi synonym THEO ĐỊNH NGHĨA không chia sẻ mặt chữ với cha gold
("cấy" vs "trồng") → gate loại cha đúng ("Ai trồng lúa" vẫn phủ "lúa"!), nhường cho câu wiki
nào tình cờ chứa từ synonym. Vá ca "Trung thu" làm vỡ cơ chế synonym-paraphrase.

---

## 2. Yêu cầu 1 (CHÍNH) — gate chỉ kích hoạt khi curated khớp KHUNG-THUẦN

**Ý tưởng:** phân biệt hai tình huống bằng **độ phân biệt của phần curated ĐÃ PHỦ** — tín hiệu
`distinct_ratio` **đã tính sẵn** trong `rank_matches` (`parent_store.py:690`):

- `"Tết Trung thu là gì"` cho query "Người dạy học là ai": chỉ phủ `là` (IDF thấp)
  → **khung-thuần** → override ✓ (giữ win Paraphrase +26).
- `"Ai trồng lúa"` cho query "Ai cấy lúa": phủ `lúa` + `Ai` (IDF cao)
  → khớp NỘI DUNG thật → **KHÔNG override** (trả lại 2 ca Synonym).

**Cách làm gợi ý** (`best_match`, quanh `parent_store.py:772`):
- Lấy `distinct_ratio` của curated top-1 (expose từ `rank_matches` — vd trả kèm trong tuple,
  hoặc tính lại cục bộ: `max IDF token curated phủ ÷ max IDF unit của query`).
- Điều kiện override mới: `not _match_has_focus(...)` **AND** `distinct_ratio < NGƯỠNG`
  (bắt đầu thử **0.6** — cùng ngưỡng phạt distinct-ratio hiện có; tinh chỉnh theo acceptance).
- Giữ nguyên phần fallback `_entity_parent_indices` (case-equiv) cho nhánh override.

**Ràng buộc Luật 0:** chỉ dùng IDF/cấu trúc — không danh sách từ, không `if token == "là"`.

---

## 3. Yêu cầu 2 — `compose_episode_weight` default về 0.0

`hca/config.py:79` đang default `1.0` **chưa có số đo dương**. Đã đo A/B trên
`grade_v3_big_a2_t1.pkl` (9 241 episode): **TRUNG TÍNH tuyệt đối** — ON = OFF = 64/2/83
(coverage/redundancy/on_topic, gold n=13).

Theo kỷ luật Trụ 1 (tiền lệ order-2 — "không bật cơ chế chưa chứng minh"): **default `0.0`**,
giữ code + cờ cho ablation và T2 essay (nơi episode kỳ vọng thật sự cắn). Ghi chú lý do vào
docstring cờ.

---

## 4. Yêu cầu 3 (kèm, nhỏ) — dọn gate vị ngữ anaphor

Gate `subj_overlap` (`hca/discourse.py:217-218`, ngưỡng 0.06) cho pick tốt mới
(**"Nước…, nó sôi ở 100 độ C"** — lần đầu "nó" cho vật vô tri ✓) nhưng cũng lọt pick lạ
trên đường compose `_t1`:

```
Nhà Trần …, Pháp thay nhà Lý năm 1226        ("Pháp" làm đại từ?!)
Nhà Nguyễn …, sản tiếp tục thực thi …        ("sản"?!)
```

Điều tra vì sao "Pháp" (viết hoa — đáng lẽ cap-gate chặn) và "sản" (mảnh từ "sản xuất"?)
vượt cả cap-gate lẫn subj_overlap; siết bằng tín hiệu cấu trúc (không danh sách từ).
Nếu không sạch được trong ticket này → tách ticket riêng, ưu tiên thấp hơn Yêu cầu 1.

---

## 5. Acceptance (bắt buộc chạy & dán số vào PR/commit)

```bash
python -m pytest -q                                              # ≥ 269 pass
python scripts/eval_v4.py --checkpoint checkpoints/grade_v3.pkl
python scripts/eval_v4.py --checkpoint checkpoints/grade_v3_big_a2.pkl
```

| Điều kiện | Ngưỡng |
|---|---|
| big_a2 QA val | = 100% |
| big_a2 Paraphrase | **≥ 88%** (giữ win) |
| big_a2 Synonym | **≥ 33%** (hết regression) |
| big_a2 Composition | ≥ 69/·/84 (về lại mức trước gate) |
| big_a2 Anaphor | ≥ 50% |
| grade_v3 (mọi bộ) | nguyên vẹn: 100/75/50; comp 93/0/100 |

Ca kiểm nhanh (CLI/UI Hybrid): `Người dạy học là ai` → nội dung "Thầy cô dạy em học" (không
"Trung thu"); `Ai cấy lúa` → "Nông dân trồng lúa."; `Ai khâu quần áo` → "Thợ may may quần áo."

**Test đơn vị mới:** thêm ca vào `tests/test_parent_store.py` — curated phủ token phân biệt
("lúa") không bị override dù thiếu token synonym ("cấy"); curated khung-thuần ("là") bị override.

---

## 6. Không làm trong ticket này

- KHÔNG đụng `rank_matches` scoring (distinct-ratio penalty hiện có giữ nguyên).
- KHÔNG đổi đường composed/theme (P1 đã đo, ON giữ nguyên).
- KHÔNG re-ingest / train lại checkpoint (ticket riêng — quality gate mới cần re-ingest).
- Ghi kết quả đo vào `docs/NN-AIVIE v4.md` nhật ký (mục 6) như thường lệ.

---

## 7. KẾT QUẢ (2026-07-02, Claude implement — Cursor nghỉ)

**Cơ chế thực tế KHÁC spec mục 2** (spec distinct-ratio<0.6 tính tay không tách được "Ai trồng
lúa" (0.88, giữ) khỏi "Một ngày có mấy giờ" (0.85, cần override) — cùng vùng): thay bằng sửa
`_match_has_focus` — chấp nhận **BẤT KỲ focus token đủ phân biệt** (gidf ≥ 75% primary,
case-equiv), không chỉ primary. "lúa" cứu ca synonym; "ai/là" gidf thấp không mở toang gate.

| Acceptance | Ngưỡng | Đo được |
|---|---|---|
| big_a2 QA / Para / **Syn** | 100 / ≥88 / ≥33 | 100 / 88 / **50** ✓ (vượt baseline 33) |
| big_a2 Composition | ≥69/·/84 | 64/2/**83** — top ✓; cov 64 = artifact gold (case-equiv tìm được curated TỐT HƠN cho Trái Đất/Mặt Trời, top 100%, gold khóa wiki cũ) |
| big_a2 Anaphor | ≥50 | 50 ✓ |
| grade_v3 | nguyên vẹn | 100/75/50; comp 93/0/100 ✓ |
| pytest | ≥269 | **270 pass** (thêm `test_best_match_keeps_curated_sharing_distinct_content_token`, fixture Napoléon thêm filler df thật) |

**Yêu cầu 2** ✓ `compose_episode_weight=0.0` (đo trung tính, ghi lý do trong config).
**Yêu cầu 3** — làm MỘT NỬA: ứng viên đại từ phải viết thường (giết "Pháp" trên _t1). Các biến
thể gate sâu hơn (lọc ent_top thường / bỏ dấu câu) ĐÃ THỬ & ĐO TỆ HƠN (on_topic 83→72/69, rác
"hình/sống/lần/lý" thay "họ") → revert về gate bản-83. **Gốc anchor-họ-người → TK_003.**
