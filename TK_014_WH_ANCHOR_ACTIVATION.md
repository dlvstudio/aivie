# TK_014 — Kích thích nơ-ron cụm WH/frame + anchor gom (X→"ở đâu")

| Field | Value |
|-------|-------|
| **ID** | TK_014 |
| **Status** | 🟡 Code + test + đo A/B XONG (2026-07-04, Claude) — chờ tác giả review/commit |
| **Nguồn** | Chỉ đạo tác giả 2026-07-04: nơ-ron "ở đâu" **không được loại bỏ**, phải được kích thích; sau này gom mọi query "nơi chốn". `X→"ở đâu"`, `Y→"ở đâu"` ⇒ gom X,Y. X = token / cụm / cha-câu / bài văn. Nối doc 03 (attention Q/K) + doc 05 (anchor neurons) |
| **Checkpoint** | `grade_v3.pkl` (guard) + `grade_v3_big_a2_t3_isg.pkl` (nơi "ở đâu" sống, power=19, cl:5780) |

## 1. Tổng quan

Debug prompt `Hỏi: Thủ đô Hà Nội ở đâu ? Đáp:`: nơ-ron cụm `ở đâu` (`cl:5780`, power=19)
**không kích hoạt**. Đo được 3 lớp chặn:

1. **Gate df trong matching** — [`query_match_units`](src/nn_aivie/core/parent_store.py) chỉ nạp
   cụm làm đơn vị match nếu `phrase_df ≤ phrase_max_df` (mặc định 8). `ở đâu` df=19 → loại; và
   thực tế MỌI cụm câu này bị loại (`Thủ đô` 33, `Hà Nội` 286, `ở đâu` 19, `là gì` 35) →
   `phrase_units=[]`, matching thoái hoá về phủ token rời. Ngưỡng 8 là hằng tuyệt đối, không
   scale theo 44k cha.
2. **UI chỉ vẽ cụm của cha THẮNG** — cha thắng của câu này là statement `Thủ đô Hà Nội ở Việt
   Nam` (không chứa "ở đâu"); QA `Thủ đô Việt Nam ở đâu` thua vì thiếu thực thể `Hà Nội`.
3. **Không có kênh "loại câu hỏi"** — matching thuần phủ IDF; "ở đâu" (ý niệm nơi chốn) bị coi
   là token phổ biến cần dìm → **vứt bỏ** thay vì định tuyến.

## 2. Hiện trạng chi tiết (đo 2026-07-04)

- Cụm `ở đâu` tồn tại: `cl:5780` power=19, cũng là Re-Pair phrase `j=246` df=19.
- `query_match_units("Thủ đô Hà Nội ở đâu")` df=8 → `phrase_units=[]`, bare=[Thủ,đô,Hà,Nội,ở,đâu].
- Cha thắng = `p57 "Thủ đô Hà Nội ở Việt Nam"` (statement), `clusters_in_span=[Thủ đô, Hà Nội,
  Việt Nam]` — không có "ở đâu".
- `seg._cluster_index[5780]` = **17 cha** chứa "ở đâu" — hạ tầng gom ĐÃ CÓ sẵn, chưa dùng.

## 3. Đề xuất giải pháp (đã triển khai)

**Hướng 1 — kích thích cụm phổ biến (WH/frame) làm đơn vị match.** Bỏ hard-gate df; nạp CẢ cụm
phổ biến làm unit, cân bằng bằng IDF thấp (đã có) + phạt distinct-ratio² (0.5.10) → không nhấn
chìm thực thể, nhưng nơ-ron WH được kích thích + hiện + làm anchor. Cờ `match_common_clusters`.

**Hướng 2 — anchor gom.** Dùng `seg._cluster_index` (đã có): cụm = anchor gom mọi cha `X→cụm`.
Thêm `anchor_members` / `anchor_left_fillers` (thực thể X đứng trước anchor) / `query_anchor_clusters`.
Viz: node cụm query được kích thích (`activated`) + field `anchor_groups` (nhóm + filler mẫu).

## 4. Đề xuất thay đổi cụ thể (đã làm)

| File | Thay đổi |
|------|----------|
| `hca/config.py` | `HCAConfig.match_common_clusters=True` |
| `core/parent_store.py` | field `match_common_clusters`; `query_match_units` bỏ cap df khi bật; `anchor_members` / `anchor_left_fillers` / `query_anchor_clusters` |
| `hca/generate_hebbian.py` | `generate()` đồng bộ cờ → store |
| `viz/export_graph_v3.py` | node cụm query kích thích + `anchor_groups` trong payload |
| `tests/test_seg_graph.py` | +2 test (anchor gom; gate ON/OFF) |

## 5. KẾT QUẢ ĐO (2026-07-04)

**A/B top-1 (df=8 vs df=∞):** grade_v3 QA+Para+Syn **21 ca → 0 flip**; ISG QA val **7 → 0 flip**.

**Full eval `grade_v3.pkl --fast` (cờ ON = default):** QA **100/100**, Para **62**, Syn **50**,
Comp 80/0/100 — **y hệt baseline** (trung tính hoàn toàn).

**Full eval ISG `grade_v3_big_a2_t3_isg.pkl --fast` (cờ ON):** QA **100/100**, Para **88**, Syn
**67**, Comp 60/0/93 — **= baseline TK_013** (QA 100/100/Para 88/Syn 67). Không regression.

**Kích hoạt (ISG):** df=∞ → `query_match_units("Thủ đô Hà Nội ở đâu")` = `['ở đâu','Thủ đô','Hà
Nội']` (bare rỗng). `anchor_left_fillers(5780)` = **{Việt Nam, Cà Mau, Amazon, Hồ Tây, nhà Nguyễn,
…}** — thực thể nơi chốn nổi lên đúng ý niệm, + ít nhiễu overlap ("xảy ra" từ "xảy ra ở đâu" —
sổ nợ overlap TK_013 §4).

**pytest:** 298 pass (296 + 2 mới), 1 skip.

## 6. Sổ nợ / phase sau

- **Overlap** `xảy ra ở đâu` vs `ở đâu` → filler "xảy ra" là động từ, không phải nơi chốn.
  Cần resolution ưu tiên span dài (TK_013 §4) + lọc filler theo vai (chưa làm).
- **Cụm nhiễu adjacency** (`đô Hà`, `Nội ở`) lọt query_anchor_clusters — đặc tính ISG bigram thô,
  không do TK_014. Có thể lọc theo df/power sau.
- **Answer-type biasing (doc 03 Wq/Wk)** — bước sau: dùng anchor "ở đâu" làm query-role key vào
  filler nơi chốn để ưu tiên đáp địa danh. Doc 07 §8 cảnh báo q·k trên function-token nhiễu →
  cần probe (thống kê token đầu đáp của member "X ở đâu") trước khi build.
- **Guard `big_a2` / `grade_v3` full (không --fast)** chưa chạy — perf chặn bởi TK_011.
