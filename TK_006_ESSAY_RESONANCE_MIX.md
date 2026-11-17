# TK_006 — Essay: PHÁT member cộng hưởng khi sinh (trộn tri thức cũ vào bài mới)

| Field | Value |
|-------|-------|
| **ID** | TK_006 |
| **Status** | Open (làm sau TK_005). **Superseded 2026-07-09:** `generate_essay` đã gỡ (TK_019 §7). |
| **Repo** | `nn-aivie`, nhánh `cursor2` |
| **Nguồn** | TK_004 — 2 080 member cộng hưởng đã nối nhưng chưa dùng; nửa còn lại ý tưởng t1 gốc |

## 1. Tổng quan

Khi build t1 SFT, mỗi bài đã **nối trung bình 4.2 cha cũ cộng hưởng** (np1/np2 — cơ chế tác giả:
kích hoạt toàn token bài → cha nào "rung" thì nối). Nhưng `generate_essay` phase-1 chỉ phát
member xuất xứ (w=1.0), bỏ qua member cộng hưởng (w<1) → t1 BIẾT tri thức cũ liên quan mà
không nói ra. Ticket này phát chúng — lần đầu mạng **trộn hai nguồn tri thức vào một bài**,
và là chỗ episode-coherence (đang trung tính trong sổ nợ) có đất cắn.

## 2. Hiện trạng chi tiết

- `AbstractNeuron.members` = [(pi, 1.0) xuất xứ theo thứ tự] + [(pi, score<1) cộng hưởng,
  score = điểm rank_matches ≥ 0.05, tối đa 8/bài, CHỈ cha có trước bài (wiki/curated/essay cũ)].
- `generate_essay` (`hca/generate_hebbian.py`): `if w < 1.0: continue`.
- Chất lượng member cộng hưởng CHƯA kiểm định bằng số — có thể lẫn cha wiki lạc chủ đề
  (điểm 0.05 là ngưỡng thấp). Rủi ro chính của ticket.
- Panel UI essay đã hiện `n_resonance` (TK_004) — chưa hiện segment nào là nguồn cũ.

## 3. Đề xuất giải pháp

1. Sau khi phát hết member xuất xứ (giữ nguyên agenda/closure), phát thêm **top-k member cộng
   hưởng** (k=`essay_resonance_k`, mặc định 2 — BẬT theo luật sổ nợ) xếp theo điểm, lọc:
   (a) điểm ≥ `essay_resonance_min` (đề xuất 0.10 — chặt hơn ngưỡng build 0.05);
   (b) khử trùng nội dung với segment đã phát (Jaccard 0.6 sẵn có);
   (c) không vượt `essay_max_segments` tổng.
2. **Đo trước khi tin**: metric `mix_rate` (% ca có ≥1 segment nguồn cộng hưởng được phát)
   + đọc tay 8 ca xem segment trộn có đúng chủ đề; coverage/ordering/match KHÔNG được tụt.
3. UI: đánh dấu segment cộng hưởng trong panel essay (kind node riêng hoặc prefix "⇠ tri thức cũ").

## 4. Đề xuất thay đổi cụ thể

| File | Thay đổi |
|---|---|
| `hca/config.py` | + `essay_resonance_k: int = 2`, `essay_resonance_min: float = 0.10` (sổ nợ nếu đo yếu) |
| `hca/generate_hebbian.py::generate_essay` | phát resonance sau provenance (lọc a-c) |
| `eval/v4.py::eval_essay` | + `mix_rate` |
| `viz/export_graph_v3.py` | node member cộng hưởng kind riêng + cạnh đứt về episode |
| `tests/test_generate_hebbian.py` | toy: t1 có resonance member đúng chủ đề → xuất hiện cuối bài; member trùng nội dung → bị khử |

**Acceptance:** mix_rate > 0 (cơ chế sống) + coverage/ordering/match_accuracy KHÔNG tụt so
TK_005 + guard giữ + đọc tay 8 ca: segment trộn không lạc chủ đề trắng trợn (ghi nhận xét vào
ticket). Nếu trộn ra rác → hạ k/tăng min, ghi sổ nợ, KHÔNG tắt.

## 5. Hoàn thành (2026-07-02, Cursor)

**Status:** Implemented.

| Hạng mục | File | Ghi chú |
|---|---|---|
| `essay_resonance_k=2`, `essay_resonance_min=0.10` | `src/nn_aivie/hca/config.py` | Mặc định BẬT (sổ nợ) |
| Phase-2 phát resonance sau provenance | `src/nn_aivie/hca/generate_hebbian.py` | Top-k theo w, dedup Jaccard 0.6, trần `essay_max_segments` |
| `EssayResult.mixed` + in `mix=…%` | `src/nn_aivie/eval/v4.py`, `scripts/eval_v4.py` | `mixed` = episode có member w<1.0 (proxy mix_rate) |
| Node `resonance_fact` + edge `resonance` | `src/nn_aivie/viz/export_graph_v3.py` | Nhánh phụ từ episode, sort theo w |
| Test resonance xuất hiện cuối bài | `tests/test_generate_hebbian.py` | `test_generate_essay_emits_resonance_members_after_provenance` |

**Test:** `pytest` 276 passed, 1 skipped.

**Chưa đo / đọc tay:** `mix_rate` trên gold 8 ca `_t2`, coverage/ordering/match không tụt, đọc tay segment trộn — cần `eval_v4` + Graph UI mode `essay` với checkpoint `_t2.pkl`.

## 6. REVIEW + ĐO (Claude, 2026-07-02)

**2 lỗi tìm thấy khi review, đã sửa:**
1. `mixed` của Cursor là proxy sai ("episode CÓ member w<1" — luôn True) → sửa thành "text member
   cộng hưởng THẬT SỰ nằm trong đáp". Proxy cũ che hoàn toàn lỗi #2.
2. **mix_rate thật = 0%**: cap 8 segment bị provenance nuốt (bài 13-17 member) + min 0.10 quá chặt
   (top thực tế 0.05-0.18). Fix: ĐẶT CHỖ k slot cho resonance trước khi phát provenance + min→0.07.

**Đo sau fix:** mix **62%** (5/8 ca có segment trộn) · cov/ord/match GIỮ 79/88/75 ✓ · 276 pass.

**Đọc tay 5 ca trộn (acceptance):** 2/5 tạm ổn (đọc sách→câu về sách; thể dục→dinh dưỡng/sức khỏe);
**3/5 lạc chủ đề kiểu đồng-âm bề mặt**: sandwich→"99% thịt sản xuất tại Đức", tập trung→"tập trung
QUÂN ĐỘI", kỹ năng→"Israel xếp hạng tư về kỹ năng". Root: resonance bind bằng rank_matches trên
bag toàn bài, ngưỡng thấp → khớp token bề mặt, không khớp CHỦ ĐỀ.

**SỔ NỢ (giữ ON theo luật):** giá = ~38% segment trộn lạc; gốc = bind thiếu gate chủ đề;
hướng fix = gate resonance bằng theme/episode-coherence lúc BUILD (T3 enrichment) hoặc đòi member
chia cụm đa-token với bài (không chỉ token đơn).
