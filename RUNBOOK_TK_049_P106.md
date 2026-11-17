# RUNBOOK TK_049 P106 — bake lớp thay thế + smoke sinh ĐA-KHE giao

> Cho M4. Ngày giao: 2026-08-06.
> Đọc `TK_049_FRAME_SLOT_ENTITY_GEN.md` §1 trước khi chạy.

## 1. Bake cạnh SYNONYM (lớp thay thế đông cứng)

```bash
PYTHONPATH=src .venv/bin/python scripts/bake_subst_class.py \
  --checkpoint checkpoints/network_v12_sent.pkl \
  --label-map out/token_labels_v12.json \
  --unit thuc_the \
  --min-share 3 --topk 8 \
  --out-checkpoint checkpoints/network_v12_subst.pkl \
  --out-meta out/bake_subst_v12_meta.json
```

Cổng bake (in ra cuối log):

- `ty_le_token_co_synonym` ≥ 0,20 trong tập token có ≥3 bối cảnh
- `n_entity` ≥ 100
- Hai lần chạy meta (n_edge, n_entity) **giống nhau** nếu chạy lại cùng lệnh

## 2. Smoke sinh đa-khe (không chấm mù đủ 400 ở P106)

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_three_arm.py \
  --checkpoint checkpoints/network_v12_subst.pkl \
  --label-map out/token_labels_v12.json \
  --arms chep_cau,ep_mot_khe_giao,ep_da_khe_giao \
  --n 40 \
  --out out/frame_da_khe_smoke.txt \
  --out-key out/frame_da_khe_smoke_key.json \
  --out-meta out/frame_da_khe_smoke_meta.json
```

Từ meta, đếm nhánh `ep_da_khe_giao`: `n_thay≥2` / 40 ≥ **25%**. In tỷ lệ `trung_nguyen_van`.

## 3. Giao nộp

Commit: `scripts/bake_subst_class.py`, vá `gen_three_arm.py`, `TK_049_*.md`,
`RUNBOOK_TK_049_P106.md`, `out/bake_subst_v12_meta.json`, smoke artefacts,
`checkpoints/network_v12_subst.pkl` **chỉ nếu** repo đã track checkpoint tương tự; nếu ckpt quá
lớn / bị ignore thì ghi rõ đường dẫn local trong commit message và meta.

Commit message ghi: n_edge SYNONYM · n_entity · ty_le_token_co_synonym · tỷ lệ n_thay≥2.
