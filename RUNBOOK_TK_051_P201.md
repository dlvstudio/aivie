# RUNBOOK TK_051 P201 — dual retrieve (smoke cổng 1–2)

> Cho M4. Ngày mở: 2026-08-06.
> Đọc `TK_051_MULTI_PARENT_COMPOSE.md` §0–§3 **trước** khi code.
> Ship P112 vẫn dùng `RUNBOOK_FRAME_P112_SHIP.md` — P201 không đè mặc định.

## 1. Mục tiêu P201

Chứng minh **retrieve được 2 cha khác nhau** cùng mạch đề đủ thường xuyên, chưa cần
nối tinh vi (P202) hay chấm mù đủ (P203).

Cổng:

| # | điều kiện |
|---|---|
| 1 | ≥30% mẫu `n_src≥2` trên n≥40 (smoke) hoặc n=120 |
| 2 | `ent_keep` (thực thể đề còn trong output) ≥ mức nhánh `ep_da` / `chep` cùng bộ đề |

Nếu (1) trượt → sửa lọc cha / top-N, **không** chạy chấm 400.

## 2. Việc làm (khi được bảo “làm P201”)

1. Arm `compose_2cha` (hoặc script `scripts/gen_compose_2cha.py`):
   - `rank_matches(đề, top_n≥8)` → lấy **2** cha khác `parent_idx` đề và khác nhau.
   - Lọc sơ: overlap token keep / entity với đề (tái dụng ý `ent_keep` từ
     `probe_retrieval_compose.py`).
   - Output tạm P201: nối đơn giản `decode(A) + " " + decode(B)` **hoặc** chỉ ghi meta
     hai cha + phát A (để đo tỷ lệ cặp hợp lệ trước khi nối).
2. Meta bắt buộc: `n_src`, `parents`, `ent_keep`, `fallback` nếu chỉ còn 1 cha.
3. Đối chứng cùng slice opener: `chep_cau`, `ep_da_khe_giao` (P112 flags).

Checkpoint: `checkpoints/network_v12_subst.pkl` (cùng ship).

## 3. Lệnh smoke

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_three_arm.py \
  --checkpoint checkpoints/network_v12_subst.pkl \
  --label-map out/token_labels_v12.json \
  --arms chep_cau,ep_da_khe_giao,compose_2cha \
  --max-thay 2 --slot-rank bam --min-gain 1 --no-co-graph \
  --n 40 \
  --out out/compose_p201_smoke.txt \
  --out-key out/compose_p201_smoke_key.json \
  --out-meta out/compose_p201_smoke_meta.json
```

Đếm nhanh từ meta:

```bash
PYTHONPATH=src .venv/bin/python -c "
import json
m=json.load(open('out/compose_p201_smoke_meta.json'))
for arm in ('chep_cau','ep_da_khe_giao','compose_2cha'):
    rows=[r for r in m['rows'] if r.get('cfg')==arm]
    ge2=sum(1 for r in rows if r.get('n_src',0)>=2)/max(len(rows),1)
    ek=sum(float(r.get('ent_keep') or 0) for r in rows)/max(len(rows),1)
    print(f'{arm}: n={len(rows)} n_src>=2={ge2:.1%} mean_ent_keep={ek:.3f}')
"
```

## 4. Việc không làm ở P201

- Soft đa-khe / `thuc_the` decode.
- Hebbian-CO.
- Chấm mù 400 mẫu trước khi cổng (1) đạt.
- Đổi default `generation_mode`.

## 5. Giao nộp P201

Meta smoke + cập nhật `TK_051` § kết quả P201 (đạt/trượt cổng 1–2).
Nếu đạt → mở `RUNBOOK_TK_051_P202.md` (nối có phép).
