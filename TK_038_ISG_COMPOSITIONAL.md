# TK_038 — ISG compositional: cụm + token / cụm + cụm → nơ-ron cụm dài

| Field | Value |
|-------|-------|
| **ID** | TK_038 |
| **Status** | 🟢 Implemented (2026-07-11) |
| **Nguồn** | Tác giả: "Ví dụ" + "dụ mẫu" (2-token) thay vì một cụm "Ví dụ mẫu" (3-token) |
| **Phụ thuộc** | TK_013 ISG, TK_037 FRAME_BRACKET slot endpoints |

## 1. Vấn đề

ISG v1 chỉ đếm **token kề token** → `ClusterNeuron(left, right)` tối đa 2 surface token.

"Ví dụ mẫu" bị tách:
- promote `(Ví, dụ)` → cl₁
- đồng thời đếm `(dụ, mẫu)` → cl₂ (overlap, greedy chọn một)
- **không** có cl span 3 token

## 2. Giải pháp

Sau khi rewire bigram, đếm thêm **cặp slot kề** khi ít nhất một đầu là `cluster`:

| Cặp slot | Span mới |
|----------|----------|
| token + token | bigram cũ (TK_013) |
| cluster + token | `expand(cl) + (t,)` |
| token + cluster | `(t,) + expand(cl)` |
| cluster + cluster | `expand(a) + expand(b)` |

Đủ `min_frequency` → `promote_composite` → một `ClusterNeuron` với `span=(t0,…,tN-1)`, rewire 2 slot → 1 slot.

## 3. `ClusterNeuron`

```python
@dataclass
class ClusterNeuron:
    left: int          # token đầu (compat)
    right: int         # token cuối (compat)
    power: int
    span: tuple[int, ...] = ()  # rỗng = bigram (left, right)
```

`expand_slot()` / `cluster_children()` dùng `span` khi có.

## 4. Files

| File | Thay đổi |
|------|----------|
| `core/seg_graph.py` | composite count/promote/rewire |
| `core/parent_store.py` | `_sync_derived_from_slots` dùng full span → `phrases[]` |
| `tests/test_seg_graph.py` | `test_compositional_vi_du_mau` |

## 5. Giới hạn v1

- Chỉ merge **2 slot kề** mỗi lần promote (incremental: cl(Ví,dụ)+mẫu → 3 token; thêm token cần pass sau)
- **Overlap block**: không promote bigram `(a,b)` nếu đã có cụm `(x,a)` hoặc `(b,y)` — tránh `Ví,dụ` vs `dụ,mẫu`
- **Power-win (2026-08-07):** nới overlap khi `bigram_count(a,b) > max(power cụm chồng)` hoặc pair ∈ `seed_pairs`; rival **không** demote (cùng tồn tại). `finalize_promotions` đưa cả candidate đang block vào heap (quyết lúc pop).
- Checkpoint cũ (vd `network_v12_subst`): **cần rebuild ISG** / re-ingest parents mới có cụm kiểu `học sinh` — chỉ đổi code không sửa pkl.
- **Parent trùng** (`emit_ids` giống): `increment_parent_counts()` cộng thêm bigram/composite (trước đây chỉ `power+=1`)
- `cluster_of_pair` vẫn chỉ token-token; composite key riêng `composite_of_pair`
