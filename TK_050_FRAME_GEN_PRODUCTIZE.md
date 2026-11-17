# TK_050 — ĐƯA KHUNG–ĐA-KHE VÀO ĐƯỜNG SINH (sau cổng 4)

> Mở 2026-08-06. **ĐÓNG 2026-08-06** (chốt ship A).
> Nối TK_049 P112: cổng (4) **ĐẠT** (`ep_da−chep = −3,3đ` · không-chép 41,7%).
>
> Nhãn sản phẩm: **truy hồi + khung có phép** — không quảng cáo “sinh sáng tạo”.
> Việc nghiên cứu tiếp (ghép ≥2 cha): `TK_051_MULTI_PARENT_COMPOSE.md`.

## 0. Trạng thái đóng (sau P113–P115)

| đã có (ship) | không primary / dừng |
|---|---|
| Bake `SYNONYM` + `Entity:` (`subst`) | Decode `thuc_the` làm mặc định (P114 Δ −6,7đ) |
| Decode **P112 token**: bam · min-gain=1 · max_thay=2 · no CO | Đóng khoảng −3,3đ → −2đ (P115 trượt) |
| CLI `gen_frame_slot.py` + `aivie generate --mode frame_da_khe` (opt-in) | Đổi mặc định `generate` sang `frame_da_khe` (giữ `w_skip`) |
| Cổng (1) wire ĐẠT; Hebbian-CO đo hại (P108) | Soft/hybrid đa-khe săn `n_thay≥2` |

## 1. Cổng G-prod (nêu TRƯỚC)

| # | điều kiện | nếu trượt |
|---|---|---|
| 1 | P113: mode frame tái lập P112 lệch ≤3đ dùng+tạm | wire sai — đừng đụng P114 |
| 2 | P114: `n_thay≥2` ≥35% *và* giữ cổng 4 | cụm chưa vào decode |
| 3 | P115: `ep_da−chep ≥ −2đ` (tuỳ chọn, sau 1–2) | dừng siết nếu (1)(2) đủ dùng |

## 2. Việc không làm

- Không bật Hebbian-CO vào `_co` (P108).
- Không `min-gain≥2` làm mặc định (P110/111 hết sáng tạo).
- Không đè `network_v12_sent.pkl`; `subst` vẫn artefact bake.

## 3. Runbook đầu

`RUNBOOK_TK_050_P113.md` — wire mode vào sinh + đo hồi quy vs P112.

## 4. KẾT QUẢ P113 (2026-08-06) — cổng (1) **ĐẠT**

| | |
|---|---|
| Module | `src/nn_aivie/hca/frame_slot_gen.py` |
| CLI | `scripts/gen_frame_slot.py` (defaults P112) |
| `aivie generate --mode frame_da_khe` | wire qua `HCAConfig.generation_mode` |
| `gen_three_arm` `ep_da_khe_*` | gọi `fill_da_khe` (một luật) |

Verify opener-slice replay vs `out/frame_p112_meta.json`:

| thước | giá trị |
|---|---:|
| khớp `n_thay` | **120/120 = 100%** ✅ (≥95%) |
| `ge1` | 41,7% → 41,7% |

Artefact: `out/frame_p113_verify_meta.json`. Việc kế: **P114** (decode thuc_the).

## 5. KẾT QUẢ P114 (2026-08-06) — cổng (2) **TRƯỢT hẹp** (meta đạt · chất lượng −6,7đ)

| | P112 token | P114 thuc_the+hybrid |
|---|---:|---:|
| `n_thay≥1` | 41,7% | **54,2%** ✅ |
| `n_thay≥2` | 23,3% | **43,3%** ✅ |
| dùng+tạm `ep_da` | 51,7% | **45,8%** |
| dùng+tạm `chep` | 55,0% | **52,5%** |
| `ep_da−chep` | **−3,3đ** | **−6,7đ** ❌ (≥−5) |
| độ trôi neo | — | **−5,0đ** |

Phân lớp: `n_thay=0` 74,5% ok · `n_thay=1` 69,2% · **`n_thay=2` 9,6%** — đa-khe soft/hybrid
phá câu. Module vẫn giữ; P115 phải siết.

Artefacts: `out/frame_p114_*` · `RUNBOOK_TK_050_P114.md`.

## 6. KẾT QUẢ P115 (2026-08-06) — cổng (3) **TRƯỢT**

Ba phép đo (cùng n=120 · neo 40 · thang tuyệt đối):

| biến thể | decode | flags | ge1 | ge2 | ep_da ok | chep ok | Δ |
|---|---|---|---:|---:|---:|---:|---:|
| **P115** | thuc_the | era+soft1+hy0+g2→1 | 52,5% | **36,7%** | 45,8% | 52,5% | **−6,7đ** |
| P115b | thuc_the | max_thay=1 +同上 | 52,5% | 0% | 44,2% | 52,5% | **−8,3đ** |
| P115c | token | era (P112+) | 41,7% | 23,3% | 51,7% | 58,3% | **−6,7đ** |

Chẩn: `n_thay=2` thuc_the ~**7–10%** dùng được — đa-khe soft/hybrid phá câu.
Lọc niên đại **không** đủ đóng ≥−2đ. P112 token (−3,3đ) vẫn là trần chất lượng tốt nhất đã đo.

**Quyết định:** dừng siết cổng 3 (TK_050 §1: tuỳ chọn). Giữ P112 token làm mặc định chất lượng;
`decode_unit=thuc_the` + filter flags vẫn có (meta creativity) nhưng **không** primary.

Artefacts: `out/frame_p115_*` · `out/frame_p115b_*` · `out/frame_p115c_*` · `RUNBOOK_TK_050_P115.md`.

## 7. CHỐT SHIP (2026-08-06) — đóng ticket

### Công thức ship (P112)

| tham số | giá trị |
|---|---|
| checkpoint | `checkpoints/network_v12_subst.pkl` (bake: `RUNBOOK_TK_049_P106.md`) |
| `decode_unit` | **token** |
| `slot_rank` / `min_gain` / `max_thay` | bam / 1 / 2 |
| CO-graph | tắt (`no_co_graph`) |
| licence | giao (SYNONYM bake ∩ bối cảnh) |
| cấm khe | nhãn `số` (`out/token_labels_v12.json`) |

Thước đóng: `ep_da−chep = −3,3đ` · `n_thay≥1 = 41,7%` · verify P113 120/120.

### Cách gọi

- Primary: `RUNBOOK_FRAME_P112_SHIP.md` → `scripts/gen_frame_slot.py`
- Opt-in trong generate: `aivie generate --mode frame_da_khe` (mặc định CLI vẫn `w_skip`)

### Cổng G-prod — tóm tắt

| # | kết quả |
|---|---|
| 1 P113 wire | **ĐẠT** |
| 2 P114 thuc_the | **TRƯỢT** (chất lượng) |
| 3 P115 siết −2đ | **TRƯỢT** / dừng |

Ticket đóng. Nghiên cứu composition đa-cha → **TK_051**.

