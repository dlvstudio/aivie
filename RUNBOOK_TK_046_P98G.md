# RUNBOOK TK_046 P98G — cầu cụm L=1..2

> Sau P98F/F+: đa-cầu token đơn vẫn hiếm. Thêm cầu **cụm** `(p1,p2)` từ trigram bề mặt
> `(x, p1, p2)` ⇒ cạnh `(p1+p2) → x`, bonus trọng số L=2.

## 0. Cổng (nêu TRƯỚC)

### Cầu

- L=1: bigram `(x, p)` ⇒ `p → x` (giữ P98E)
- L=2: trigram `(x, p1, p2)` ⇒ `(p1,p2) → x` · `compound_bonus` (mặc định 2.5)
- donor chỉ khi `lop_trai` giao non-empty (P98F)
- `score(c) = Σ_key strength` (P98F+)

### Ca / cổng G

| # | điều kiện |
|---|---|
| 1 | ≥10 ca dương điền được |
| 2 | Hit@1 ≥40% |
| 3 | FP âm ≤15% |
| 4 | creative (triple vắng) |

### Cổng phụ Q1

Không còn `Hồng`/`Sông` qua cầu `nuôi`.

## 1. Lệnh

```bash
# cổng đầy đủ
PYTHONPATH=src .venv/bin/python scripts/probe_analogy_exec_v5.py \
  --checkpoint checkpoints/mini_sent_ep_qa.pkl \
  --out out/probe_analogy_exec_v5.json

# thử prompt (resolve head: id props mạnh nhất; cụm → token cuối)
PYTHONPATH=src .venv/bin/python scripts/probe_analogy_exec_v5.py \
  --try-prompts "anh là,mẹ là,Thái là,ông là,bà là,Trung là,con khỉ là,Hồng là" \
  --out out/probe_analogy_try_v5.json
```

## 2. Kỳ vọng

- Giữ ĐẠT hình thức vs P98F
- `anh` có ≥2 cầu (`họ`+`thứ`); `Hồng` không điền
- L=2 có thể vẫn 0 trên mini (ít trigram sạch) — cơ chế sẵn, đo được

## 3. Kết quả (2026-08-07)

| | |
|---|---|
| bridge hit | 16/16 = **100%** · FP 0% |
| cổng G | **ĐẠT** |
| Q1 Hồng/Sông | **✅** |
| ca ≥2 cầu | **1/16** (`anh`) |
| ca có L2 | **0/16** (mini) |

### Prompt thử

| prompt | kết quả | cầu |
|---|---|---|
| `anh là` | `anh là Hồ` | `họ`,`thứ` (n=2) |
| `mẹ là` | `mẹ là Hồ` | `họ` |
| `Thái là` | `Thái là Hồ` | `thứ` |
| `ông`/`bà`/`Trung là` | `… là Nguyễn` | `mất` |
| `con khỉ là` | không điền | (khỉ không có cầu trên mini) |
| `Hồng là` | không điền | Q1 OK |

Artefacts: `out/probe_analogy_exec_v5.json` · `out/probe_analogy_try_v5.json` · `TK_046` §18.G

## 4. Bake opt-in generate (cùng ngày)

Module: `src/nn_aivie/hca/analogy_bridge_gen.py` · mode **`analogy_la`**.

```bash
PYTHONPATH=src .venv/bin/python -m nn_aivie.cli generate \
  --checkpoint checkpoints/mini_sent_ep_qa.pkl \
  --mode analogy_la \
  --prompt "anh là"
```

Ship runbook: `RUNBOOK_ANALOGY_P98G_SHIP.md`. **Không** đổi mặc định `w_skip`.

### Siết hub v12 (2026-08-07)

Triệu chứng: cầu `cẩn+thận` ~238 entity · giao `là cùng` chỉ 8 · điền `mẹ là cùng`.

Sửa trong `analogy_bridge_gen`:

1. `max_prop_df=40` — bỏ cầu quá phổ biến  
2. `require_la_complete` — cắt «là cùng nhau»; giữ «là Hồ Liêm» (pred hoa)  
3. `BAD_PRED`/`BAD_DONOR` — cùng, ngươi, đều, liền…  
4. phạt `log(prop_df)` trong strength · `pred_top_ban=40`

Smoke v12: hết hub `cùng`; `anh là Nguyễn` (donors mẹ/gái/trai/cha); `chó là Nguyễn` qua `ngoan+ngoãn`. Nghĩa vẫn mỏng — cải dần.

### Ưu tiên cụm Entity (2026-08-07) — không train lại ckpt

`prefer_compound=True`: bigram có `Entity:…` → neo **cụm**, không neo token cuối.
Triple `là` vẫn neo token (giữ `trai là Hồ` trên mini).

Đo v12: `sinh` hết `tự+tay`; `Entity:học sinh` có `tự+tay`.  
Emit cha vẫn tách token — **chưa** train lại `network_v12_subst.pkl`.
