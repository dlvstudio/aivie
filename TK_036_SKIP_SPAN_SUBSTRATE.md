# TK_036 — SKIP_SPAN Substrate (kết nối ngang content xa + nơ-ron cụm)

| Field | Value |
|-------|-------|
| **ID** | TK_036 |
| **Status** | 🟡 S1+S2 (2026-07-10) — mine ✅ + AR gate ✅ + **retrieval ❌ chết, cố ý** |
| **Nguồn** | Ý tưởng tác giả: skip content (phở→Việt Nam) + promote SkipCluster; GIỮ link token→cha |

> **Retrieval path chết — đã đo, cố ý không sửa (2026-07-14).** `query_match_units.skip_units`
> luôn rỗng (`phrase_units` giành `used_pos` trước; đầu-phải luôn là cụm nên span luôn trùng →
> bất khả đạt). Sửa cho sống được (0/250→92/250 query) nhưng **retrieval TỤT**: rank-1
> 76.3%→70.0%, MRR 0.853→0.815 trên `grade_lessons_test.pkl`; **không trọng số nào có lợi**
> (w=0.1 trung tính, w=1.0 tụt 6.3pp). Skip span là bằng chứng **cấp họ** → nâng đều cả họ, loãng
> bằng chứng riêng của cha đúng. Guard README §0.5.12 **mù** với skip (`grade_v3.pkl` mine ra 0
> span — có trước ISG/TK_013). Giá trị skip span nằm ở **cổng AR** (§3 kênh 4), nơi nó là *cầu
> lúc sinh* chứ không phải *bằng chứng lúc xếp hạng*. Chi tiết: xfail strict
> `tests/test_skip_span.py::test_query_match_units_detect_skip`.

## 1. Khác ISG liền kề

| | ISG cluster (TK_013) | Skip span (TK_036) |
|--|---------------------|-------------------|
| Gap | 1 (kề) | 2–8 |
| Rewire slot_chain | ✅ | ❌ |
| Token `_index` | giữ | **giữ** (bắt buộc) |
| Đơn vị | `ClusterNeuron` | `SkipSpan` ảo `(left_token, right_phrase)` |
| Promote | power bigram | power×PMI ≥ ngưỡng → `promoted=True` |

## 2. Code

| File | Vai trò |
|------|---------|
| `training/skip_span.py` | `mine_skip_spans()` (bootstrap ledger + materialize) |
| `core/skip_ledger.py` | **S3** incremental ledger + slot-gap scan + rewire refresh |
| `hca/skip_span_gate.py` | `ensure_skip_spans`, `apply_skip_span_gate` |
| `core/parent_store.py` | `skip_spans`, `_skip_lookup`, `_skip_spans_in` (cổng AR đọc) — `query_match_units.skip_units` + nhánh skip trong `rank_matches` **chết, cố ý** (xem trên) |
| `hca/config.py` | `ar_skip_span_weight`, `ar_skip_gap_prior`, `ar_skip_ctx_weight` |
| `scripts/probe_skip_spans.py` | S0 probe top pairs |

## 3. Sinh next-token (4 kênh)

1. **W_seq** — bigram kề (giữ)
2. **macro** — spreading agenda (giữ)
3. **FRAME_BRACKET** — từ khung xa (TK_037: token/cluster/phrase endpoints)
4. **skip_bridge** — `w0=left` + agenda có `right` → boost × PMI × P(gap) × ctx

## 4. Chạy probe

```bash
PYTHONPATH=src python -u scripts/probe_skip_spans.py \
  --checkpoint checkpoints/network_v10_histwiki_paragraph.pkl
```

## 5. Sổ nợ

- VarLM `encode_tokens_with_skips` (phase 3)
- Viz `sk:*` nodes (phase 3)
- ~~Re-mine on ingest incremental~~ → **S3 `SkipLedger` online** (`core/skip_ledger.py`, hook `add_example` + `ingest`)
- **Bake lúc build:** `scripts/bake_checkpoint_indices.py` hoặc `save_checkpoint(..., bake_derived=True)`
