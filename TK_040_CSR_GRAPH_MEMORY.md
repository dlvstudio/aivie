# TK_040 — CSR nén đồ thị: hạ RAM 20GB → vài GB (mở khóa numba + song song)

| Field | Value |
|-------|-------|
| **ID** | TK_040 |
| **Status** | 📋 SPEC (chưa làm) — phác sau khi train v10 chạm trần RAM 24GB (2026-07-13) |
| **Nguồn** | Full v10 (65k cha) chiếm **~20–22GB RAM** dạng dict-Python → swap trên MacBook 24GB → bake FRAME_BRACKET bò/treo. TK_039 giải thời-gian (skip_ledger); ticket này giải **bộ-nhớ**. |
| **Liên quan** | TK_011 Phase C (matrix/GPU); TK_039 (train perf); doc 03/07 (attention từ đồ thị cần matvec). |

## 0. Vấn đề (đo trước khi tin — BƯỚC 1 BẮT BUỘC)

Đồ thị ~65k cha ngốn ~20GB **không phải vì dữ liệu** (thật chỉ vài trăm MB) mà vì **overhead
object Python** trên hàng chục triệu object nhỏ:

| Cấu trúc | Dạng | Overhead |
|----------|------|----------|
| `adjacency._layers[*]` (CO_ACT ~4.6M+…) | `dict[int, dict[int, float]]` | ~100B/entry + float 24B → cho 1 số 4B |
| `seg._links_by_parent` (SegLink) | hàng triệu object 5-field (2 SlotRef) | ~120–200B/object |
| `parents[*].slot_chain` | tuple `SlotRef` | ~48–56B/slot |
| `seg.bigram_count` (~2M) | `dict[tuple[int,int], int]` | ~120B/cặp |

**⚠️ BƯỚC 1 = ĐO** (`tracemalloc` / `pympler.asizeof` / `sys.getsizeof` đệ quy) trên một checkpoint
đã load → bảng "cấu trúc → GB". KHÔNG rewrite trước khi biết cái nào chiếm phần lớn 20GB.

### M0 ĐO XONG (2026-07-13, `network_v10_histwiki.pkl`, 151k cha, RSS 3.26GB)

> ⚠️ Đây là v10_histwiki (KHÔNG phải full v10 22GB — chưa save được). Tỉ lệ cấu trúc tổng quát hóa.
> deep-sizeof cap 6M object → undercount (*), nhưng thứ hạng rõ.

| Cấu trúc | GB (≥) | |
|----------|-------:|--|
| **adj[CO_ACTIVATION]** `_data` (8.46M cạnh) + `_rows` | 0.50 + 0.27* | **#1, lưu 2 LẦN** |
| `parents[*].slot_chain` (SlotRef) | 0.36* | |
| `seg._links_by_parent` (SegLink) | 0.33* | |
| `parents` records | 0.30* | |
| adj[SEQUENTIAL] (578k) | 0.15 | 2 lần |
| bigram_count (1.8M) | 0.08 | |
| embedding (numpy) | 0.11 | đã gọn |

**Chốt:** (1) **CO_ACTIVATION adjacency = hog #1**, scale theo corpus (full v10 dày conv/paragraph →
~50–100M cạnh → ~10–15GB = phần lớn 22GB). (2) `SparseAdjacency` **lưu mỗi cạnh 2 lần** (`_data`
dict-theo-(i,j) + `_rows` dict-lồng); `_rows` là index dẫn-xuất → dư. (3) **`to_arrays()` đã có**
(dùng lúc pickle) → CSR runtime là mở rộng, không phát minh mới.

→ **M1 = CO_ACTIVATION (+ mọi layer) → CSR runtime, bỏ 2-lần.** Ước 8.46M: 0.77GB → ~68MB (~11×).
Script đo: `scratchpad/measure_ram.py`.

## 1. CSR là gì

Sparse layer `dict[src→{dst:w}]` → **3 mảng phẳng numpy**:
```
indptr : int32[N+1]     # cạnh của node i ở indices[indptr[i]:indptr[i+1]]
indices: int32[#cạnh]   # node đích nối đuôi
data   : float32[#cạnh] # trọng số nối đuôi
```
→ **~8 byte/cạnh** thay ~150 → **20–40× nhỏ hơn**. VD 4.6M cạnh: ~1GB → ~37MB.

## 2. Lợi ích kép

1. **RAM ~20GB → vài GB** → hết swap trên 24GB.
2. **Cache-locality** → hết page-miss stall (nhanh cả single-thread).
3. **numba JIT** chạy trên mảng (10–50×).
4. **multiprocessing** an toàn (numpy share COW sạch — process ~3.35GB non-graph còn lại → fork 4–6 worker khả thi).
5. **Generate nhanh** (đọc cạnh từ mảng).

## 3. Cái giá + kiến trúc kép

- CSR **tối ưu ĐỌC, kém GHI** (chèn cạnh phải dịch mảng).
- → **Ingest giữ dict/object** (grow được); **freeze → CSR** cho pha ĐỌC-NHIỀU (bake, generate, save/load).
- Chuyển 1 lần cuối train (như `finalize_skip_ledger_deferred` — cùng triết lý fix#5).

## 4. Lộ trình (measure-first, byte-identical)

| Phase | Nội dung | Verify |
|:---:|----------|--------|
| **M0** | **ĐO** breakdown RAM per-cấu-trúc trên checkpoint đã load | bảng GB — quyết cái nào đáng |
| **M1** ✅ core | CSR cho **layer nặng nhất** (nghi CO_ACTIVATION): `freeze_csr()` sau train + reader đọc CSR; ingest vẫn dict | RAM↓ đo được; `adjacency.weight/neighbors` byte-identical (chữ ký) |
| **M2** | CSR cho `_links_by_parent` (seg_h) + `slot_chain` (mảng token-id + offset thay SlotRef object) | links/generate byte-identical |
| **M3** | Bake + generate đọc CSR (không rã ra dict) | FRAME_BRACKET/skip scan trên CSR = kết quả cũ (FBSIG/SKIPSIG) |
| **M4** | (mở khóa) numba JIT `scan_parent` trên mảng; đo lại | tốc độ; kết quả byte-identical |
| **M5** | (mở khóa) multiprocessing pass record/bake khi RAM đã trống | tốc độ |

Mỗi phase: **đo RAM trước/sau + chữ ký byte-identical**, giữ nếu lợi thật (như TK_039).

## 4b. M1 CORE ĐÃ LÀM + VERIFY (2026-07-13)

`SparseAdjacency.freeze()` (`core/adjacency.py`): nén `_data`(dict) + `_rows`(dict) →
CSR **`_row_ids/_indptr/_indices` (int32) + `_vals` (float64)**, set `_frozen`, **bỏ 2 dict**.
Reader `get/row/items/__len__/to_arrays` có nhánh frozen (binary-search hàng+cột). `set/add` khi
frozen → `_thaw()` (dựng lại dict) — inference read-only nên không đụng. `TypedAdjacency.freeze()`
gọi mọi layer. **Opt-in** (mặc định KHÔNG frozen → path cũ nguyên vẹn, 0 regression).

**Verify (`network_v10_histwiki`, 8 layer, CO_ACT 8.46M cạnh):**
- **BYTE-IDENTICAL mọi layer** (hash `items()` trước/sau freeze khớp) + get/row/thaw unit-test ✓.
- **RAM adjacency: ~1.48GB (dict) → 0.113GB (CSR) = ~13×.** CO_ACT: 8.46M cạnh → **102MB**.
- freeze 1 lần ~92s (sort + build); vals GIỮ float64 nên không mất precision.

→ **Full v10** (CO_ACT ~50–100M cạnh) ước dict ~8–16GB → CSR ~0.6–1.2GB: **cắt ~10–15GB của 22GB**.

**CÒN LẠI (M3 integration, RỦI RO — chưa làm):** gọi `adjacency.freeze()` **sau load** để pha
đọc (bake/generate) hưởng. Điều kiện: **audit inference/bake KHÔNG mutate adjacency** (nếu mutate
→ auto-thaw, mất lợi). Đã biết: `mine_frame_brackets` set lại layer FRAME_BRACKET (nhỏ → thaw OK,
CO_ACT giữ frozen); cần grep `hca/` xem generate có `.set/.add` adjacency không. Script:
`scratchpad/verify_freeze.py`.

## 5. Rủi ro / trần

- Rewrite lớn, chạm `adjacency`, `seg_graph`, `parent_store`, save/load payload. Làm **từng layer**, verify từng bước.
- Ranh giới ingest(mutable)↔read(CSR) phải rõ; nếu code inference vô tình mutate CSR → lỗi.
- **KHÔNG** phải mọi thứ cần CSR — chỉ cấu trúc M0 chỉ ra là nặng. Tránh over-engineer.
- GPU vẫn **để sau/không hợp** cho scan bất quy tắc; CSR là điều kiện cần cho matvec (doc 03 attention) nếu sau này muốn.

## 6. Palliative trước khi làm TK_040 (ngay, rẻ)

- **Train KHÔNG `--bake`** → ghi pkl ngay sau train (an toàn) → bake riêng với RAM trống + `caffeinate -i`.
- **Đóng app** trước train (VSCode/Chrome/Slack/Mongo ~10GB) → giảm swap.
- Đây là băng-bó; TK_040 mới giải gốc trần 20GB.
