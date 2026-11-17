# TK_039 — TĂNG TỐC TRAIN: 6 fix (finalize + skip_ledger + bake)

| Field | Value |
|-------|-------|
| **ID** | TK_039 |
| **Status** | 🟢 fix #1–#4, #6 byte-identical; fix #5 đo-được (0% lõi/26% dict). **fix #1–#5 ĐÃ COMMIT**; **fix #6 + doc chưa commit**. |
| **Nguồn** | Train v10 rất chậm (~15.5h full; `history_articles` 158 bài ~22 phút; bake FRAME_BRACKET spike RAM → swap trên MacBook M4 Pro 24GB). |
| **Kỷ luật** | measure-first: profile trước, fix đúng hotspot, **verify byte-identical bằng chữ ký** (FULLSIG/SKIPSIG/FBSIG) trước khi tin. |

## 0. Chẩn đoán (measure-first)

**Profile `history_articles` 45 bài từ scratch = 1838s** (`cProfile`, sort tottime):

| Hàm | tottime | Ghi chú |
|-----|--------:|---------|
| `seg_graph._rebuild_parent_links` | 504s | dựng lại TOÀN BỘ link seg_h mỗi rewire (bài dài ~4400 link) |
| `skip_ledger._right_at_slot` (+genexpr+sum) | ~470s | dựng `flat` + `sum()` trong vòng lặp đôi si×sj |
| `seg_graph._refresh_pair_occurrences_parent` | 272s | quét lại chain + lọc global occurrence-list mỗi merge |

**Nút full-train** (bóc tách log thật `out/train_v10_full.log`, 8/8 dataset, 15.54h):

| Phase | Thời gian | % |
|-------|----------:|--:|
| ISG end_batch | 0.98h | 6% |
| skip_ledger `record_parent` | 6.89h | 44% |
| skip_ledger `materialize/gán` | 7.57h | 49% |
| **skip_ledger TỔNG** | **14.46h** | **93%** |

→ **skip_ledger = 93% thời gian** (lặp mỗi dataset trên số cha tăng dần). Riêng `qa_supervised` (dataset cuối) = 6.25h. **Không phải** index-rebuild (0.1s) như tưởng ban đầu.

**Bake:** `mine_frame_brackets` giữ **~6.5M cặp thô × {count, gaps-Counter, prov-list, edge_kinds}** đồng thời → spike **vài GB** ở đỉnh đồ thị → **swap 19GB / tổng 50GB** trên 24GB (nặng thêm sau sleep/wake).

## 1. Sáu fix

Tất cả **build-time thuần** — KHÔNG đụng `hca/` (hàm sinh), KHÔNG đổi format checkpoint.

| Fix | File | Hotspot | Cơ chế | Byte-id |
|:---:|------|---------|--------|:-------:|
| **#1** | `core/seg_graph.py` | `_rebuild_parent_links` (504s) | Cờ `_defer_link_rebuild`; hoãn rebuild link → **1 lần/cha** cuối `finalize_promotions` (link không đọc giữa promote; `link_weight` ổn định) | ✅ |
| **#2** | `core/skip_ledger.py` | `_right_at_slot` (~470s) | Hoist `flat` + prefix-sum ra khỏi vòng lặp đôi trong `scan_parent` → O(len²·gap)→O(len·gap) | ✅ |
| **#3** | `core/seg_graph.py` | `_refresh_pair_occurrences_parent` (93% sau #1/#2) | **Bỏ lưu vị trí**: `_pair_occurrences`(pos)→`_pair_members` (set cha); quét vị trí **on-demand** lúc promote; `composite_count` cập nhật CHỈ ở biên merge (`_apply_boundary_deltas_for`+`_deregister_composite`). Xóa hẳn `_refresh_pair_occurrences_parent` | ✅ |
| **#4** | `core/skip_ledger.py`, `core/parent_store.py` | `materialize` re-scan (gán skip_span_ids) | `materialize(reuse_hits=)`: cha vừa `record_parent` cùng flush (cùng `phrase_by_start`) → **tái dùng `_parent_hits`**, khỏi `scan_parent` lại | ✅ |
| **#5** | `core/parent_store.py`, `training/pipeline.py` | skip_ledger lặp mỗi dataset | `finalize_batch_ingest(defer_skip=True)` (mặc định qua `run_manifest`) → **hoãn record+materialize về 1 lần cuối** (`finalize_skip_ledger_deferred`). skip_span_ids KHÔNG đọc giữa dataset. O(dataset×cha)→O(cha) | ⚠️ **không** |
| **#6** | `training/frame_bracket.py` | `mine_frame_brackets` spike RAM (6.5M cặp) | **2-pass**: pass1 chỉ đếm; lọc survivor; pass2 gom gaps/prov CHỈ cho survivor. edge_kinds suy từ endpoints | ✅ |

## 2. Đo

### fix #1–#3 (`history_articles`, A/B back-to-back, cùng máy)

| Quy mô | Gốc | +#1+#2 | **+#3** | FULLSIG |
|--------|----:|-------:|--------:|:-------:|
| N=20 | 865.7s | 559.3s | **43.5s** | khớp |
| N=45 | 1838s | ~500s | **41.5s** | khớp |
| **Full 158 bài (grade+history end-to-end)** | ~2.5–3h | (kẹt 817s/710 promote) | **207.7s (~3.5 phút)** | — |

fix#3 verify FULLSIG (clusters, cluster_of_pair, composite_count, bigram_count, links, chains, skip) **khớp tuyệt đối N=6/20/45**; 21 test pass; generation ra đúng fact (Lý Thái Tổ→dời đô Thăng Long 1010).

### fix #4 (byte-identical) + #5 (đo mức khác)

fix#4 A/B N=20: fix#1+#2 = 559s → **43.5s**, FULLSIG+SKIPSIG khớp.

fix#5 — per-dataset vs deferred:

| Corpus | Kho span | Tổng gán | **Cha đổi skip_span_ids** |
|--------|:--------:|:--------:|:-------------------------:|
| grade+subject+history (lõi) | 626=626 | 3268=3268 | **0.0%** |
| + dictionary (phrase tăng mạnh) | 2248=2248 | 15959=15959 | **26.1%** |

→ kho span + tổng gán **BẤT BIẾN** (không mất tín hiệu, chỉ phân bố lại); non-skip state (chains/clusters) MATCH. **Ảnh hưởng sinh:** per-parent `skip_span_ids` = coverage phụ trong `rank_matches` (dòng ~1230); guard QA/Para/Syn đo domain lõi (0%) → **guard không đổi**; đáp cha-từ-điển có thể xê dịch nhẹ (nhất quán hơn). Nợ: chưa spot-check guard trực tiếp. **Tác giả chọn giữ fix#5 mặc định BẬT.**

### fix #6 (byte-identical, đánh đổi RAM/CPU)

A/B FRAME_BRACKET (grade+dict3k+history): edges=25631, pairs=1.56M — **FBSIG khớp** (`39dfa9b…`).

| | Peak RAM (mine) | CPU |
|--|:---------------:|:---:|
| 1-pass | vài GB | 94s |
| **fix#6 2-pass** | **~1GB** | 142s (**1.5×**) |

→ **fix RAM, không phải fix tốc độ.** Máy đủ RAM: chậm 1.5% thuần. Máy 24GB swap (ca thực): 1-pass thrash/treo → 2-pass **chạy xong**. Bake 1 lần/train nên chấp nhận.

## 3. Tổng lợi ích (ước tính full v10)

skip_ledger 14.46h → ~1–2.7h (fix#4+#5); ISG 0.98h → ~0.1h (fix#1+#3); bake hết spike RAM (fix#6).
→ **15.54h → ~2–3h** (hoặc ~1–1.5h nếu đủ RAM tránh swap — rate/cha đang chậm 23× vì swap). ≈ **5–8× nhanh hơn**, chưa đo end-to-end với đủ 5 fix (nợ).

## 4. Phương pháp verify (oracle byte-identical)

- **FULLSIG** = hash MỌI trường serialize: clusters, cluster_of_pair, composite_of_pair, bigram_count, composite_count, links, slot_chain, skip-spans.
- **SKIPSIG** = skip_spans inventory + per-parent skip_span_ids.
- **FBSIG** = FRAME_BRACKET edges + hlink_meta + endpoints.
- A/B: chạy sig với code mới → `git stash` file fix → chạy code cũ → so hash. Khớp = không đổi một bit.

## 5. Vận hành (bài học `--bake`)

`save_checkpoint(bake_derived=True)` chạy **bake TRƯỚC khi ghi `.pkl`** (`checkpoint_io.py:83→84`) → nếu bake OOM/treo thì **mất cả train**. Khuyến nghị:

```bash
# Train KHÔNG --bake → ghi pkl NGAY (an toàn), + chặn sleep
caffeinate -i python scripts/train.py --manifest config/train_v10.yaml --out checkpoints/network_v10.pkl
# Bake RIÊNG sau (retry được, còn pkl gốc nếu lỗi)
python scripts/bake_checkpoint_indices.py checkpoints/network_v10.pkl --in-place
```

Đóng app khác trước khi train (process ~3.35GB nhưng pressure hệ thống gây swap → stall).

## 6. Sổ nợ / hướng tiếp

- **Đo end-to-end full v10 với đủ 5 fix + RAM trống** (số ước tính §3 chưa verify thực).
- **Spot-check guard QA/Para/Syn với fix#5** (kỳ vọng không đổi vì domain lõi 0%).
- **fpos-cache cho fix#6** — cache `fpos` mỗi cha ở pass1 → bỏ recompute `_collect_frame_positions` ở pass2, cắt phần lớn 1.5× (đổi ít RAM). Đo rồi quyết.
- **CSR/mảng + numba** (TK_011): dạng dict-Python là gốc của memory-stall + bất-khả-song-song; CSR mở khóa RAM↓ + numba JIT + multiprocessing an toàn (process chỉ 3.35GB → fork 4–6 worker khả thi khi RAM trống). GPU KHÔNG hợp (scan bất quy tắc).
- **Parallelism:** thread=GIL chặn; multiprocessing cần mảng-share (CSR) để COW không bị refcount phá.

## 7. Files

| File | Fix |
|------|-----|
| `src/nn_aivie/core/seg_graph.py` | #1, #3 — **committed** |
| `src/nn_aivie/core/skip_ledger.py` | #2, #4 — **committed** |
| `src/nn_aivie/core/parent_store.py` | #4, #5 — **committed** |
| `src/nn_aivie/training/pipeline.py` | #5 (`run_manifest(defer_skip=True)`) — **committed** |
| `src/nn_aivie/training/frame_bracket.py` | #6 — **chưa commit** |
| `TRAINING_PIPELINE.md` §0.1 + `TK_039_TRAIN_PERF.md` | doc 6 fix — **chưa commit** |
