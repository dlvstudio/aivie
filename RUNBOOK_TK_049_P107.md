# RUNBOOK TK_049 P107 — chấm mù khung–đa-khe trên ckpt subst

> Cho M4. Ngày giao: 2026-08-06.
> Đọc `TK_049_FRAME_SLOT_ENTITY_GEN.md` §1.3 cổng (4) trước khi chạy.
> P106 đã đạt cổng bake + smoke (`n_thay≥2` = 87,5%). P107 đo **chất lượng**.

## 1. Câu hỏi

`ep_da_khe_giao` điền nhiều khe từ lớp thay thế bake — có **mua chất lượng** gần `chep_cau`
không, hay chỉ tăng độ mới mà phá câu?

| kết quả | kết luận |
|---|---|
| `ep_da_khe − chep ≥ −5đ` **và** tỷ lệ không-chép (`n_thay≥1`) ≥30% | cổng (4) **đạt** — hướng khung–đa-khe sống |
| (3) đã đạt mà hiệu **< −20đ** | substrate chưa đủ cho đa-khe → Hebbian P108 trước scale |
| hiệu −5…−20đ | sống một phần — ghi số, đừng tuyên bố thắng |

Đối chứng bắt buộc: `ep_mot_khe_giao` (thủ tục cũ) trong cùng vòng.

## 2. Lệnh

Checkpoint: `checkpoints/network_v12_subst.pkl` (local, gitignore — đã bake P106).

### Bước 1 — sinh mẫu (n=120/nhánh, neo 40 như P103)

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_three_arm.py \
  --checkpoint checkpoints/network_v12_subst.pkl \
  --label-map out/token_labels_v12.json \
  --arms chep_cau,ep_mot_khe_giao,ep_da_khe_giao \
  --n 120 \
  --out out/frame_da_khe_samples.txt \
  --out-key out/frame_da_khe_key.json \
  --out-meta out/frame_da_khe_meta.json
```

Kiểm log: `cấm thay khe 'số'` với N>0 · `lớp thay thế từ SYNONYM bake` với N≫0.

### Bước 2 — trộn neo

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import re, random, pathlib
main = pathlib.Path("out/frame_da_khe_samples.txt").read_text(encoding="utf-8")
anch = pathlib.Path("out/anchor_samples.txt").read_text(encoding="utf-8")
pat = r"(\[[A-Z]?\w+\] ĐỀ: .*?\n   → .*?\n)"
blocks = re.findall(pat, main, re.S) + re.findall(pat, anch, re.S)
random.Random(107).shuffle(blocks)
with open("out/frame_da_khe_cham.txt", "w", encoding="utf-8") as f:
    f.write("# CHẤM MÙ TUYỆT ĐỐI — mỗi mẫu chấm ĐỘC LẬP, không so với mẫu nào khác.\n")
    f.write(f"# n={len(blocks)} mẫu · KHÔNG LỌC · thang: dung_duoc / tam / hong.\n\n")
    for b in blocks:
        f.write(b + "\n")
print("→ out/frame_da_khe_cham.txt", len(blocks), "mẫu")
PY
```

### Bước 3 — chấm mù

Chỉ đọc `out/frame_da_khe_cham.txt`. **Cấm** key/meta/verdict/anchor/TK/RUNBOOK.

Thang (NGUYÊN VĂN, không sửa chữ — lỗi P101):

> **dung_duoc** — phần sinh nối vào đề mà người đọc không khựng: đúng chủ đề/chủ thể đủ để
> tin là tiếp cùng mạch, câu tiếng Việt trôi, không salad từ.
> **tam** — còn nhận ra cùng chủ đề hoặc khung, nhưng cụt, lệch nhẹ, hoặc một chỗ lạc.
> **hong** — nhảy đề, salad, mảnh vụn, hoặc trôi chảy nhưng lệch nhân vật/lĩnh vực rõ.

Ghi `out/frame_da_khe_verdict.json` sau mỗi lô 40 (gộp đè).

### Bước 4 — độ trôi + hiệu

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import json, collections
v = json.load(open("out/frame_da_khe_verdict.json"))["grades"]
ak = json.load(open("out/anchor_key.json"))
mk = json.load(open("out/frame_da_khe_key.json"))
meta = json.load(open("out/frame_da_khe_meta.json"))["rows"]
ok = lambda ids: sum(1 for i in ids if v.get(i) in ("dung_duoc", "tam")) / max(len(ids), 1)
aid = [i for i in ak if i in v]
ref = sum(1 for i in aid if ak[i] in ("dung_duoc", "tam")) / len(aid)
print(f"NEO n={len(aid)} · phiên {ok(aid):.1%} · tham chiếu {ref:.1%} · TRÔI {ok(aid)-ref:+.1%}")
for arm in ("chep_cau", "ep_mot_khe_giao", "ep_da_khe_giao"):
    ids = [i for i, a in mk.items() if a == arm and i in v]
    c = collections.Counter(v[i] for i in ids)
    print(f"{arm:<18} n={len(ids):<4} dung {c['dung_duoc']/len(ids):>6.1%} · dùng+tạm {ok(ids):>6.1%}")
# không-chép từ meta
by = {}
for r in meta:
    by.setdefault(r["cfg"], []).append(r)
for arm in ("ep_mot_khe_giao", "ep_da_khe_giao"):
    rs = by[arm]
    ge1 = sum(1 for r in rs if r.get("n_thay", 0) >= 1) / len(rs)
    ge2 = sum(1 for r in rs if r.get("n_thay", 0) >= 2) / len(rs)
    print(f"meta {arm}: n_thay≥1={ge1:.1%} · n_thay≥2={ge2:.1%}")
PY
```

Cổng (4): `ep_da_khe_giao − chep_cau` (dùng+tạm) ≥ **−5đ** *và* `n_thay≥1` ≥ **30%**.

## 3. Giao nộp

`out/frame_da_khe_{samples,key,meta,cham,verdict}.*` · cập nhật `TK_049` § kết quả P107.
Commit message ghi: độ trôi · ba mức · hai hiệu · cổng (4) đạt/trượt.
