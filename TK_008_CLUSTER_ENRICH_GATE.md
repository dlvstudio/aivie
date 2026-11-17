# TK_008 — T3: enrichment CỤM↔CỤM qua đồng-episode + gate chủ đề cho resonance

| Field | Value |
|-------|-------|
| **ID** | TK_008 |
| **Status** | **Implemented 2026-07-02 (Claude)** — (1) mục 5, (2) mục 6; phần chuyển tiếp ghi cuối mục 6 |
| **Nguồn** | TK_006 §6 (3/5 segment trộn lạc); T0 (tín hiệu episode intrinsic dương); ý gốc tác giả ("mắt sáng"↔"tinh anh") |

## 1. Tổng quan
Hai vấn đề cùng một gốc: (a) resonance bind bằng token bề mặt → segment trộn lạc kiểu đồng-âm
("tập trung QUÂN ĐỘI" vào bài tập-trung-công-việc); (b) synonym mức cụm chưa có đường học
("bảo vệ"↔"giữ gìn"). Fix chung: làm việc ở mức **CỤM đa-token trên màng episode** — T0 đã
chứng minh đồng-hiện mức đoạn mang tín hiệu paradigmatic thật, chỉ thua khi ở mức token đơn
(đa nghĩa "trị/giá trị"). Đây là mảnh cuối của ý tưởng t1 gốc (enrichment nghĩa).

## 2. Hiện trạng chi tiết
- `bind_episode_by_resonance` + build resonance: điểm rank_matches trên bag toàn bài, ngưỡng
  0.05-0.09 → khớp token đơn; "tập trung" đa nghĩa tự nó là bigram nên gate bigram-chung ĐƠN LẺ
  không đủ (ca quân đội vẫn chia "tập trung").
- Kho cụm: `phrases` (Re-Pair) + **ISG clusters (TK_013)** — `cl:*` online tại ingest; essay
  dùng chung `rank_matches` / `phrase_ids` materialize từ cụm ISG.
  (9241) + SFT (500) đã có member map — substrate cho đồng-thành-viên.
- E-episode T0 (token×đoạn) đã lưu bài học: cần đơn vị cụm, không token.

## 3. Đề xuất giải pháp
1. **Gate chủ đề khi PHÁT resonance** (`generate_essay` phase-2): member hợp lệ khi tổng
   idf-bigram (TK_007) của các bigram CHUNG với bài ≥ ngưỡng **và** ≥2 đơn vị chung phân biệt
   (một cụm đa nghĩa đơn lẻ như "tập trung" không đủ). Đo bằng đọc tay 5 ca + mix giữ >0.
2. **Cạnh liên tưởng cụm↔cụm** (`phrase_assoc`): hai cụm/bigram đồng-thành-viên nhiều episode
   → PMI trên màng episode (đơn vị = Re-Pair phrase + bigram instruction/câu). Lưu trong store,
   API `related_phrases(phrase)`. Ứng dụng đo được đầu tiên: mở rộng `match_episode` qua cụm
   liên quan (paraphrase instruction khác cụm vẫn khớp) + synonym probe mức cụm.
3. Thứ tự: (1) trước — rẻ, chặn rác ngay; (2) sau — nền T3 thật.

## 4. Đề xuất thay đổi cụ thể
| File | Thay đổi |
|---|---|
| `hca/generate_hebbian.py::generate_essay` | gate (1): shared-bigram idf ≥ `essay_resonance_gate` + ≥2 đơn vị |
| `hca/config.py` | `essay_resonance_gate: float` (đo để chọn; sổ nợ nếu chặt/lỏng) |
| `core/parent_store.py` | (2) `build_phrase_assoc()` PMI đồng-episode + `related_phrases()`; serialize |
| `eval` | synonym probe mức cụm (bộ nhỏ mới) + essay mix đọc tay |
| tests | toy gate chặn member chỉ-chia-1-cụm-đa-nghĩa; assoc PMI đúng |

**Acceptance:** (1) đọc tay: 0/5 segment trộn lạc trắng trợn, mix ≥ 25%, cov/ord/match giữ;
(2) `related_phrases` trả cặp có nghĩa trên `_t2` (ghi ví dụ vào ticket) + match_episode
paraphrase-khác-cụm cải thiện ≥1 ca đo được. Guard + pytest xanh.

## 5. KẾT QUẢ PHẦN (1) — gate chủ đề (Claude, 2026-07-02)

Gate = member phải chia ≥1 **bigram MỚI ngoài câu hỏi** với thân bài (`essay_resonance_gate=1`).
Đo `_t2`: **mix 62→38%**, cov/ord/match giữ 79/88/88, 277 pass.

Đọc tay: chặn đúng 2/4 rác (sandwich "99% thịt Đức", lập trình "Israel") ✓; giữ mix tốt
(thể dục↔dinh dưỡng/sức khỏe) ✓; **1 ca vẫn lọt** — bài tập-trung trộn "tập trung QUÂN ĐỘI"
+ "phong thủy" qua bigram chức năng ("tăng cường", "trung vào").

**SỔ NỢ:** gate mặt-chữ chạm trần — phân biệt cuối cùng cần NGHĨA mức cụm = phần (2)
`phrase_assoc` (chưa làm, việc kế tiếp). Acceptance "0/5 lạc trắng trợn" chưa đạt (còn 1);
theo luật: giữ ON, ghi giá, không tắt.

## 6. KẾT QUẢ PHẦN (2) — phrase_assoc (Claude, 2026-07-02)

3 vòng build trên `_t2` (9 741 episode, 160k bigram):
1. Top-12-hiếm-nhất → 8k cặp toàn mảnh tên riêng df=2, probe rỗng → **sampling sai**.
2. Band df [3,300], top-30 → 131k cặp, tín hiệu thật xuất hiện nhưng lẫn mảnh-chồng-lấp
   ("chú trọng"↔"trọng đến").
3. + filter NỐI-CHUỖI (giữ biến thể paradigm cùng-vị-trí như "mì kẹp"↔"mì sandwich") →
   **108k cặp, chất lượng thật**:
   - "thể dục" ↔ **"tập luyện" (5.7)**, "giấc ngủ", "giảm căng" — đúng loại "mắt sáng↔tinh anh"
   - "đọc sách" ↔ "nghe nhạc" (5.3), "để giảm" — cùng trường sở thích/thư giãn
   - "sức khỏe" ↔ "chăm sóc", "các bệnh", "bị suy"

**Acceptance:** (2a) cặp có nghĩa ✓ ĐẠT (ví dụ trên); (2b) flip ca thật ✗ CHƯA:
- "mì kẹp" assoc **rỗng** — df<3 trên corpus (SỔ NỢ: cần scale essay/wiki; đúng loại
  "thiếu data không phải bug");
- "tập trung" neighbors nhiễu — đa nghĩa cần ngữ cảnh sâu hơn.
Essay giữ 79/88/88/38; toy test khớp-qua-assoc pass; **278 pass**. Serialize vào checkpoint
`_t2` (build offline, match_episode đọc tự động khi có).

**Việc kế thừa:** scale corpus → rebuild assoc (kỳ vọng "mì kẹp" có df); dùng assoc cho gate
resonance (ca "quân đội") + synonym mức cụm; T3 tiếp tục trên nền này.
