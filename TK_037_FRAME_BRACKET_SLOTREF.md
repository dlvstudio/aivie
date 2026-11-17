# TK_037 — FRAME_BRACKET v2: SlotRef endpoints (token / cluster / phrase)

| Field | Value |
|-------|-------|
| **ID** | TK_037 |
| **Status** | 🟢 Implemented (2026-07-11) |
| **Nguồn** | Tác giả: FRAME_BRACKET cần triệt để hơn — không chỉ token→token mà cả cụm→cụm, token→cụm, cụm→token |
| **Thay thế** | Mở rộng TK_016/TK_020 mine path; gate readout tương thích checkpoint token→token cũ |

## 1. Vấn đề (TK_016)

`mine_frame_brackets` quét `source_ids` thô, chỉ ghép **token→token** trong band `func` (df cao).
Hệ quả:

- Cụm ISG đã promote (`Mặc dù`, `Không những`, `mà còn`) **không** thành endpoint
- Opener hiếm (`nếu`, df thấp) dễ bị loại khi `min_df` cao
- Gate chỉ boost **một** token closer

SKIP_SPAN (TK_036) đã có `token → phrase`; ISG seg_h (TK_013) đã có `SlotRef` kề. FRAME_BRACKET xa vẫn thiếu `SlotRef`.

## 2. Endpoint model

| Kind | Adj ID | Ví dụ |
|------|--------|-------|
| `token` | `token_id` | `nếu → thì` |
| `cluster` | `CLUSTER_OFFSET + ci` | `cl:Mặc_dù → vẫn` |
| `phrase` | `PHRASE_OFFSET + phrase_idx` | cụm 2–4 token đã học, chưa rewire ISG |

Registry: `ParentStore.hlink_endpoints: dict[int, FrameEndpoint]`

```python
@dataclass(frozen=True)
class FrameEndpoint:
    kind: Literal["token", "cluster", "phrase"]
    tokens: tuple[int, ...]   # expand cluster/phrase; (tid,) cho token
```

## 3. Mine (`training/frame_bracket.py`)

1. Duyệt `slot_chain` (fallback `source_ids` nếu chain rỗng)
2. Endpoint trái/phải = `SlotRef` token hoặc cluster **hoặc** phrase khung (từ `phrases` + ISG surface)
3. Gap = **chỉ số slot** giữa opener và closer (`min_gap`..`max_gap`)
4. Lọc: PMI có hướng + `dir_ratio` (giữ TK_016); token khung = func-band **hoặc** `df ≥ min_df_opener` (mặc định 2)
5. Cluster khung: power trong band, không phải pure content skip
6. Ghi `adjacency[(left_id, right_id)]` + `hlink_meta` (power, gaps, prov, left_kind, right_kind)
7. `prov`: `(parent_idx, opener_token_pos, gap_tokens)` — token pos để gate khớp runtime

## 4. Gate (`hca/frame_bracket_gate.py`)

- **Opener token**: mở khung khi token có outgoing `FRAME_BRACKET` (giữ TK_017)
- **Opener cluster**: mở sau token **cuối** cụm (`cluster_of_pair` khớp suffix prefix)
- **Opener phrase**: mở sau token cuối phrase khung
- **Closer cluster/phrase**: boost token **đầu** cụm (giống skip_span)
- **Closer token**: boost token đó
- Gap prior + ctx-resonance: giữ TK_020, đọc meta theo `(opener_id, closer_id)`

## 5. Serialize

`to_payload` / `from_payload`:

- `hlink_endpoints`: `{str(id): [kind, ...tokens]}`
- `hlink_meta` keys vẫn `"a,b"` (int adj ids); thêm `left_kind`, `right_kind` optional

Checkpoint cũ (chỉ token ids) vẫn load; thiếu `hlink_endpoints` → suy từ vocab token_of.

## 6. Files

| File | Vai trò |
|------|---------|
| `core/frame_bracket_ref.py` | `FrameEndpoint`, encode/decode, expand |
| `training/frame_bracket.py` | mine slot_chain + phrase |
| `hca/frame_bracket_gate.py` | opener/closer cluster+phrase |
| `core/parent_store.py` | serialize `hlink_endpoints` |
| `tests/test_frame_bracket_gate.py` | mine + gate cluster |

## 7. Chạy test

```bash
PYTHONPATH=src pytest tests/test_frame_bracket_gate.py -q
```

## 8. Sổ nợ

- Viz `export_hlinks.py` hiển thị cluster/phrase label
- Re-mine grade_lessons để thấy `Mặc dù→vẫn` thay `này→ra` dominator
- Per-item mine FRAME_BRACKET (hiện bake batch) — phase sau
