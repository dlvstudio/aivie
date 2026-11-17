# TK_013 — Mạng phân rã tăng dần qua link ngang (ISG)

| Field | Value |
|-------|-------|
| **ID** | TK_013 |
| **Status** | ✅ Done (2026-07-03) — code + rebuild `_t3` + eval + Graph UI |
| **Nguồn** | Ý tưởng tác giả: bigram ngang tích lũy → cụm + rewire; doc `02_HYBRID_HEBBIAN_MATRIX` §0–2 |
| **Checkpoint** | `checkpoints/grade_v3_big_a2_t3_isg.pkl` (từ `_t3.pkl` qua `rebuild_seg_graph.py`) |

## 0. Kết quả triển khai (2026-07-03)

### Code

- **`core/seg_graph.py`** — `SlotRef`, `SegLink`, `ClusterNeuron`, `SegmentGraph`: ingest online,
  batch ingest + `finalize_promotions` (heap), rewire toàn cục, index trên link (`parent_idx`,
  `slot_index`), cluster→token index 1/2 trên link dọc.
- **`parent_store.py`** — hook `add_example`; `rebuild_seg_graph()`; `source_ids` + `slot_chain` +
  `match_span`; materialize `match_ids`/`phrase_ids`; serialize `seg` trong checkpoint.
- **`query_match_units`** — span cụm ISG song song Re-Pair `phrases[]`.
- **`generate_hebbian.py`** — `_completion_from_parent()`.
- **`export_graph_v3.py`** — node `cl:*`, cạnh `seg_h` / `seg_cluster` trên cha thắng.
- **`util/progress.py`** — `FileProgressLog` (append file + stdout).
- **Scripts:** `rebuild_seg_graph.py`, `inference_samples.py`; `train_v3` / `build_t1_unified`
  mặc định ISG (`--legacy-repair` giữ Re-Pair batch).

### Rebuild `_t3` (ISG)

```bash
python scripts/rebuild_seg_graph.py \
  --checkpoint checkpoints/grade_v3_big_a2_t3.pkl \
  --out checkpoints/grade_v3_big_a2_t3_isg.pkl \
  --log-file out/rebuild_isg_t3.log \
  --log-every 2000 --max-promotions 40000
```

| Metric | Giá trị |
|--------|---------|
| parents | 44 633 |
| clusters | 40 000 |
| links | 535 627 |
| **cluster `ở→đâu`** | **power=19** (idx=5780) |
| wall time | ~3 phút |

Log: `out/rebuild_isg_t3.log`, tóm tắt `out/rebuild_isg_t3.summary.json`.

Migration checkpoint cũ: tái dựng `source_ids` từ `match_ids + Đáp: + answer_ids` (QA) hoặc
`emit_ids` (statement) khi thiếu `source_ids`.

### Eval (`grade_v3_big_a2_t3_isg.pkl`, `--fast`)

| Bộ | retrieval | gen-hybrid |
|----|-----------|------------|
| QA val | **100%** | **100%** |
| Paraphrase | 88% | 88% |
| Synonym | 67% | 67% |

Log: `out/eval_isg_t3.log`, timing `out/eval_isg_t3.json`. **296 pytest pass.**

### Graph UI

```bash
AIVIE_CHECKPOINT=checkpoints/grade_v3_big_a2_t3_isg.pkl python viz/server/main.py
```

Mode `hybrid` + `Hỏi: Hà Nội ở đâu` → hiển thị `cl:5780` (ở đâu) và cạnh `seg_h`.

## 1. Tổng quan

Thay Re-Pair batch (global `phrases[]` trên `match_ids`, greedy toàn corpus) bằng **ISG**:

- Mỗi cha: chuỗi slot + cạnh ngang `(parent_idx, slot_index, weight)`.
- Bigram token-token tích lũy khi ingest → đủ `min_frequency` → `ClusterNeuron` + rewire mọi cha
  chứa cặp.
- Khớp lan xuôi: đơn vị = slot (token hoặc cụm), không bag token rời.

## 2. Acceptance (đã đạt)

- [x] Toy 3 câu `ở đâu` → cluster power=3 (`tests/test_seg_graph.py`)
- [x] `rebuild_seg_graph.py` + log file + migration `source_ids`
- [x] `train_v3` / `build_t1_unified` ISG default; `--legacy-repair` A/B
- [x] Viz `cl:*` + `seg_h`
- [x] Rebuild `_t3`: `ở→đâu` power=19
- [x] Guard eval ≥ baseline trên QA val (100/100)

## 3. Files

| File | Thay đổi |
|------|----------|
| `core/seg_graph.py` | **Mới** |
| `core/parent_store.py` | ISG hook, rebuild, serialize |
| `hca/generate_hebbian.py` | `_completion_from_parent` |
| `viz/export_graph_v3.py` | ISG viz |
| `util/progress.py` | `FileProgressLog` |
| `scripts/rebuild_seg_graph.py` | migration + log |
| `scripts/inference_samples.py` | inference mẫu + log |
| `scripts/train_v3.py`, `build_t1_unified.py` | ISG default |
| `tests/test_seg_graph.py` | **Mới** (5 test) |

## 4. Sổ nợ / phase sau

- ~~**Cụm-của-cụm** (một đầu đã là cluster) — phase 2.~~ → **TK_038** (compositional promote)
- **Overlap** (`xảy ra ở đâu` vs `ở đâu`) — resolution ưu tiên span dài.
- **`fact_links`** — map position qua `expand_slots(match_span)`.
- Deprecate `decompose_repair` khi guard ổn định lâu dài trên ISG-only checkpoint.
