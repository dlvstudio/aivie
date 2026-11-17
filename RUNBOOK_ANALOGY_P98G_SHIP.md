# RUNBOOK — analogy_la P98G (đường ship research)

> Chốt 2026-08-07. Đọc `TK_046_ANALOGY_FRAME_TENSOR.md` §18.G · `RUNBOOK_TK_046_P98G.md`.
> Nhãn: **điền `X là ?` qua cầu thuộc_tính** (L=1..2 + cầu ngữ cảnh `Lbl:` + Σ cầu).
> Không phải sinh tự do dài; không đổi mặc định `w_skip`.

## 0. Công thức

`min_support=2` · `prop_gidf_min=4.5` · `compound_bonus=2.5` · `min_strength=1.0` ·
`max_prop_df=40` · `require_la_complete=True` (cắt «là cùng nhau»; giữ «là Hồ Liêm») ·
`pred_top_ban=40` · ckpt đo: `mini_sent_ep_qa.pkl` /
smoke **`network_v12_subst_isg_pw.pkl`** (ISG power-win rebuild).

**Neo head (2026-08-07):** `prefer_compound=True` — thuộc_tính neo theo span dài nhất
ISG cluster → phrase → `Entity:` → token; dual-write token cuối (mini `anh/mẹ là`);
`prop_df` chỉ đếm head primary vocab/Entity (tránh phình do dual/cluster adj).
Ví dụ v12: `học sinh là` → ISG `(học,sinh)` power=1840; donor đuôi `sinh` bị loại lúc aggregate.

Module: `nn_aivie.hca.analogy_bridge_gen`.

## 1. Một lệnh sinh

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_analogy_la.py \
  --checkpoint checkpoints/mini_sent_ep_qa.pkl \
  --prompt "anh là"
```

## 2. Opt-in qua `aivie generate`

Mặc định `aivie generate` vẫn **`w_skip`**.

```bash
PYTHONPATH=src .venv/bin/python -m nn_aivie.cli generate \
  --checkpoint checkpoints/network_v12_subst_isg_pw.pkl \
  --mode analogy_la \
  --prompt "học sinh là"
```

## 3. Smoke / try (2026-08-07, ckpt `_isg_pw`)

```bash
./.venv/bin/python scripts/try_analogy_la_smoke.py \
  checkpoints/mini_sent_ep_qa.pkl checkpoints/network_v12_subst_isg_pw.pkl
# → out/analogy_la_smoke_isg_pw.log

./.venv/bin/python scripts/try_analogy_la_v12.py \
  checkpoints/network_v12_subst_isg_pw.pkl
# → out/analogy_la_try_network_v12_subst_isg_pw.json
# so sánh: scripts/compare_analogy_la_try.py → out/analogy_la_compare_isg_pw.txt
```

| thước | kết quả |
|---|---|
| mini `anh/mẹ/em là` | **Hồ** (giữ) |
| hub «là cùng» trên try 26 prompt | **0** (cũ `try_v12.json` gần như toàn `cùng`) |
| try fill dương | **17/26** pred thật (Tổng/Vương/…) |
| `học sinh là` | không còn `Vương` (sau siết donor đuôi + BAD_DONOR); head=ISG `học sinh` |
| index | L1≈160k L2≈439k triples=1001 · heads cl/ent ≈50k/40k |

**Chốt research:** giữ `analogy_la` **opt-in**; ckpt đo mặc định chuyển sang `_isg_pw`.
Không đổi default `w_skip`; không bake adjacency.

### Siết false analogy (2026-08-07) — case `Việt Nam là Tổng`

1. **Cấm cầu họ/tên:** `VN_SURNAME` / `Nguyễn+Văn` không neo prop; `ông+Họ` bỏ.
2. **Pred chức danh:** `Tổng`/`Phó`/… ∈ `BAD_PRED`; `TITLE_CONTINUATION` chặn `là Tổng thống` nhưng giữ `Hồ Liêm` / `Vương gia`.
3. **`lop_trai`:** `GEO_ENTITY` → `dia_ly`; `compatible` từ chối chỉ-chung-`khac` khi một bên có lớp mạnh (chặn Việt Nam←ông).

Smoke: `out/analogy_la_smoke_fix.log` — `Việt Nam là` → không điền; mini `anh/mẹ/em là` → `Hồ`.

### Siết donor trùng đuôi cụm (2026-08-07) — case `học sinh là Vương`

- `aggregate_scores`: bỏ donor là **token thành phần** của head cụm (`sinh` ⊂ `học sinh`).
- `BAD_DONOR` thêm discourse: `đặc biệt`, `Mặc dù`, …
- Giữ `min_support=2` → cầu chỉ còn `nàng` không đủ → hết đường `Vương` giả.
- Smoke: `out/analogy_la_smoke_donor_fix.log` — `học sinh là` ≠ `Vương`; mini + `Việt Nam` giữ.

### Siết ngữ cảnh lop (2026-08-07) — thay thế ≠ chỉ chung Z/`khac`

- `compatible`: cầu chỉ giao `khac` **không** đủ để suy `là` — cần `nguoi` hoặc `dia_ly`.
- `lop_trai`: head ∈ `KINSHIP` → luôn `nguoi` (mini `anh/mẹ+họ` không còn `khac`).
- Dedupe donor cụm: `nữ` ⊂ `Phụ nữ` không tính 2 support.
- Pred họ VN chỉ cộng khi cầu là `họ` (giữ mini `Hồ`; chặn `mẹ+cưng→Nguyễn`).
- Smoke: `out/analogy_la_smoke_lop_ctx.log` — `học sinh`/`XX` chết; mini `Hồ` giữ.

### Cầu ngữ cảnh `Lbl:` (2026-08-08) — ⚠️ **ĐÃ GỠ 2026-08-10**

> Xem tổng kết [`TK_046` §18.H](TK_046_ANALOGY_FRAME_TENSOR.md). Bake/scripts Lbl đã xóa;
> ckpt `isg_pw`/`mini` đã strip LEXATTR `Lbl:*`; `compatible` chỉ còn cầu thuộc_tính Z.
> Các mục densify/induced bên dưới chỉ là lịch sử.

~~Bake: `scripts/ingest_lbl_lop.py` → hub `Lbl:nguoi` / `Lbl:dia_ly` + `LEXATTR`
  (seed chỉ trong `lbl_lop_seed.py`, không gate lúc sinh).~~
~~`compatible` = giao neighbor `Lbl:*` (id); bỏ string `lop_trai` / KINSHIP/GEO runtime.~~
~~Ckpt đã bake: `mini_sent_ep_qa.pkl`, `network_v12_subst_isg_pw.pkl`.~~
~~Smoke: `out/analogy_la_smoke_lbl.log` — mini `Hồ`; `học sinh` không `XX`; `Việt Nam` khong_dien.~~

<details><summary>Lịch sử densify / induced (không chạy lại)</summary>

### Densify `Lbl:` (2026-08-08) — seed nghề/vật/tổ chức + NER

Hub: `nguoi` · `dia_ly` · `to_chuc` · `vat_the`.

```bash
# ĐÃ GỠ — không còn scripts/ingest_lbl_lop.py
```

- Seed: kinship + nghề (`học sinh` qua **Entity**, không gắn token `sinh`) + địa danh + org + vật.
- NER: PER→`nguoi`, LOC→`dia_ly`, ORG→`to_chuc` (cần `pip install underthesea`).
  Span đa token **chỉ** gắn `Entity:` (không phá token thành phần — tránh `sinh`←tên).
- Bake lại sạch: thêm `--reset` (xóa LEXATTR `Lbl:*` rồi seed+NER).
- Log: `out/bake_lbl_dense_mini.log`, `out/bake_lbl_dense_isg_pw.log`; smoke `out/analogy_la_smoke_lbl_dense.log`.

### Nhãn cảm sinh khung (2026-08-08) — `X là C` / prop chung

Không train lại: mine parents → `Lbl:{C}` (+ tuỳ chọn `Lbl:prop:{Z}`) + `LEXATTR`.

- `con mèo/chó/trâu là động vật` → `Lbl:động_vật` ← các head  
- Chiều lớp: `động vật 4 chân là mèo|chó|…` → `Lbl:cls:động_vật_4_chân` ← members  
- Prop chung: chỉ khi `--induced-prop` (siết df + bỏ Z viết hoa)  
- Log: `out/bake_lbl_induced_*.log`; smoke `out/analogy_la_smoke_lbl_induced.log`

</details>

## 4. Không làm trên đường ship

- Không đổi mặc định `generation_mode` sang `analogy_la`.
- Không kỳ vọng `con khỉ là động vật` trên mini (thiếu cầu gold).
- **Không bake lại** nơ-ron `Lbl:` ngữ cảnh (đã gỡ). Cầu thuộc_tính Z vẫn là xương sống `analogy_la`.
- Đường probe ý niệm không-Lbl: `hub_spread` / `scripts/probe_hub_spread.py` (chưa wire generate).
