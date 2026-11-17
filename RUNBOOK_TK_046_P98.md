# RUNBOOK TK_046 P98 — đường thi hành cạnh suy ra (Hebbian + E)

> Sau §17: không cắm AR/truy hồi. Câu hỏi: *có đường nào biến liên kết suy ra
> thành câu đọc được không?*
>
> Thứ tự đã chốt với tác giả: (1) Hebbian + `E` tương đồng → viết cạnh có cổng
> → thi hành khung; (2) sau đó tầng khung/vai (TK_047) — **không** POS thô.

## 0. Cổng ghi TRƯỚC

### A — đồ chơi (bắt buộc trước mini)

Corpus đúng ví dụ tác giả (T1) + bẫy `bàn/nâu` (T2).

| # | điều kiện | nếu trượt |
|---|---|---|
| A1 | Viết cạnh: `khỉ → (là) động vật` có; `bàn → (là) động vật` **không** | dừng — cơ chế viết chưa tách lớp |
| A2 | Thi hành khung: prompt `con khỉ là` → surface chứa `động vật` | dừng — cạnh không thành câu |
| A3 | Thi hành âm: prompt `cái bàn là` → **không** ra `động vật` | dừng — nhiễm loại |

Cơ chế viết = **Hebbian lan truyền** (kiến trúc tác giả) + chọn bạn đồng lớp bằng
**đồng khung / hỗ trợ ≥2** trên đồ chơi (không có `E`). Thi hành = **điền khe**
khung `X là Y` từ cạnh đã viết — **không** chép cha, **không** AR token-kế.

### B — mini (chỉ khi A ĐẠT)

Cùng giao thức P93 (`probe_edge_precision`): nửa A suy / nửa B xác nhận.

| # | điều kiện |
|---|---|
| B1 | Bộ `hebbian_E` (tương đồng = cos `E`, chuyển giao trong khung) precision lỏng **≥ 1,0×** `embedding` P93 |
| B2 | **≥ 1,5×** `ngau_nhien` (khớp khung) |
| B3 | ≥200 chủ thể có ≥1 cạnh |

Nếu B ĐẠT → ghi cạnh vào artefact bake (chưa wire `generate` mặc định).
Nếu B TRƯỢT → giữ kết luận §9 (`E` một mình đã là mốc viết cạnh); không bake.

## 1. Lệnh

```bash
# A — đồ chơi
PYTHONPATH=src .venv/bin/python scripts/probe_analogy_exec_toy.py

# B — mini (sau A ĐẠT)
PYTHONPATH=src .venv/bin/python scripts/probe_edge_precision.py \
  --checkpoint checkpoints/mini_sent_ep_qa.pkl \
  --writer hebbian_E --n-subj 300 \
  --out out/probe_edge_prec_hebbian_E.json
```

## 2. Không làm

- Cắm cạnh suy vào `rank_matches` / chép cha (§11).
- Nhân nhãn POS để “cứu” analogy (P96 1,03×).
- Đổi mặc định `w_skip` / `compose_2cha`.

## 3. Kết quả (2026-08-06)

| cổng | kết quả |
|---|---|
| A đồ chơi | **ĐẠT** — `con khỉ là động vật`; bàn không nhiễm |
| B G-prec-HE | **ĐẠT** — `hebbian_E` lỏng 33,5% = **1,27×** `E` · **2,10×** ngẫu nhiên · n=280 |

Artefacts: `out/probe_edge_prec_hebbian_E.json` · `out/probe_analogy_exec_mini.json` · `TK_046` §18.

| cổng | kết quả |
|---|---|
| A đồ chơi thi hành | **ĐẠT** |
| B G-prec-HE (viết cạnh) | **ĐẠT** (1,27× `E`) |
| C G-exec (điền khung mini) | **TRƯỢT** (0,44× `E`, thưa) |

Chưa bake vào `generate`. Việc kế: rerank `E` bằng hebbian **hoặc** tín hiệu paradigmatic trong P112.
