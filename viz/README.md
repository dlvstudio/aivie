# AIVIE Graph UI (TK_001)

Minh họa đồ thị training, inference step-by-step, và subgraph theo rule.

## Yêu cầu

```bash
pip install -e ".[viz]"
cd viz/web && npm install
```

Checkpoint mặc định: `checkpoints/grade_v3_big_a2.pkl` (nếu có) → `grade_v3_big.pkl` → `grade_v3.pkl`

## Chạy nhanh (một lệnh)

```bash
pip install -e ".[viz]"
python scripts/run_viz.py
```

Mở **http://127.0.0.1:8765** — server tự build React nếu chưa có `dist/`, hoặc dùng fallback HTML.

## Chạy dev (hot reload frontend)

**Terminal 1 — API** (từ root repo):

```bash
python -m viz.server.main
```

**Terminal 2 — Web** (Vite, khuyên dùng khi sửa UI):

```bash
cd viz/web
npm run dev
```

Mở **http://127.0.0.1:5173** (proxy `/api` → :8765)

> ⚠️ Chỉ mở :8765 **không** có `viz/web/dist/` trước đây sẽ thấy trang trắng/JSON — đã sửa: luôn có fallback HTML tại `/`.

## Export JSON tĩnh (offline)

```bash
python -m nn_aivie.cli export-viz \
  --checkpoint checkpoints/grade_03.pkl \
  --rules vi_QA_hoi_dap vi_copula_S_la_NA \
  -o viz/web/public/graph.json \
  --trace-prompt "Hỏi: Em học gì ? Đáp:"
```

Hoặc: `python scripts/export_viz.py`

## Production build (API phục vụ static)

```bash
cd viz/web && npm run build
python -m viz.server.main
# Mở http://127.0.0.1:8765 nếu dist/ tồn tại
```

## Màn hình

| Tab | Mô tả |
|-----|--------|
| **Graph Explorer** | Subgraph BFS từ rule(s); màu theo kind node |
| **Inference Lab** | Trace prompt: active rules, Var, decay, top-k scores |

## API

| Endpoint | Mô tả |
|----------|--------|
| `GET /api/health` | Trạng thái + checkpoint path |
| `GET /api/rules` | Danh sách `Rule:…` |
| `GET /api/graph?rules=…` | Subgraph JSON |
| `POST /api/trace` | `{"prompt":"…","max_len":8,"top_k":12}` |

## Module Python

- `src/nn_aivie/viz/export_graph.py`
- `src/nn_aivie/viz/trace.py`
