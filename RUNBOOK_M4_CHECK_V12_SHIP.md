# RUNBOOK M4 — kiểm ckpt ship v12 + hồi quy bản vá niên đại

> Giao 2026-08-10. Máy dev (Intel 8GB) **không nạp nổi** ckpt v12 (~15–16GB RAM,
> đo tỉ lệ 13,7× trên `mini_sent.pkl`) → phải chạy ở M4.
> Đọc `TK_046_ANALOGY_FRAME_TENSOR.md` §18.G–H và `TK_051_MULTI_PARENT_COMPOSE.md` §11.

## 0. Bối cảnh — vì sao có runbook này

Phiên review 2026-08-10 sửa 5 lỗi trên đường sinh ship và thêm 28 test. Một
trong các bản vá **đụng vào khâu lọc cha của `compose_2cha`**, nên số P203 phải
được đo lại chứ không được giả định giữ nguyên.

| vá | file | ảnh hưởng |
|---|---|---|
| `eras_in_text` khớp **âm tiết** thay chuỗi con | `hca/frame_slot_gen.py` | 🔴 **đổi tập cha ứng viên** của `compose_2cha` |
| `la_emit_id` = biến thể `là` df cao nhất | `hca/analogy_bridge_gen.py` | có thể đổi mặt chữ đầu ra `analogy_la` |
| bỏ `acc_left` chết · bỏ `try/except` nuốt lỗi · xoá hàm chết | `hca/analogy_bridge_gen.py` | trung tính (nhưng thiếu `_gdf` nay **nổ** thay vì im lặng) |
| chỉ mục nghịch đảo chồng biên | `core/seg_graph.py` | chỉ tốc độ — 1 353× ở cỡ 260k cụm; kết quả khớp từng bit |

Lý do chuỗi con là lỗi: `Lê` trúng trong `Lên`, `Hồ` trong `Hồng`, `Ngô` trong
`Ngôn` → cha đúng đề bị gán niên đại giả rồi **bị loại oan**.

## 1. Chạy trước — test nhanh (không cần ckpt)

```bash
PYTHONPATH=src .venv/bin/python -m pytest -m "not slow" -q
```

Kỳ vọng **392 passed**. Trượt ở đây thì dừng, đừng chạy tiếp.

## 2. Kiểm hai ckpt ship

```bash
PYTHONPATH=src .venv/bin/python scripts/check_v12_ship_ckpt.py \
  --checkpoint checkpoints/network_v12_subst_isg_pw.pkl

PYTHONPATH=src .venv/bin/python scripts/check_v12_ship_ckpt.py \
  --checkpoint checkpoints/network_v12_subst.pkl
```

Ghi `out/check_ckpt_*.log` (có RSS + mốc thời gian) và `out/check_ckpt_*.json`.

Đọc ba con số, theo thứ tự:

| # | số | ý nghĩa nếu bất thường |
|---|---|---|
| 1 | `n_synonym_edges` | `subst` phải **>0**; =0 ⇒ chưa bake SYNONYM ⇒ `frame_da_khe`/`compose_2cha` không chạy được (xem `RUNBOOK_TK_049_P106.md` §1) |
| 2 | `n_triples` | **=0 ⇒ cơ chế `analogy_la` CÂM vì ngưỡng**, không phải vì âm tính. Mọi kết luận âm tính trên ckpt đó vô nghĩa cho tới khi hạ `pred_top_ban` / `prop_gidf_min` |
| 3 | `la_emit_doi_so_voi_min_id` | `true` ⇒ bản vá **đổi** token `là` phát ra trên v12 ⇒ output `analogy_la` sẽ khác bản đã ghi ở `RUNBOOK_ANALOGY_P98G_SHIP.md` §3 |

⚠️ Trên `mini_sent_ep_qa.pkl` (đối chứng đã chạy): `n_triples=33`,
`la_ids=[1495, 6137]` gdf `{1495: 1202, 6137: 3}` → emit **1495**, *trùng*
min-id ⇒ mini **không** đổi hành vi. `anh/mẹ là → Hồ` giữ nguyên. Nếu v12 ra
`true` ở (3) thì đó là chỗ khác biệt thật, cần đọc tay output.

## 3. 🔴 Hồi quy `compose_2cha` sau vá niên đại (BẮT BUỘC)

Đây là mục quan trọng nhất của runbook. Chạy lại smoke P201/P202 và so với số đã chốt:

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_three_arm.py \
  --checkpoint checkpoints/network_v12_subst.pkl \
  --label-map out/token_labels_v12.json \
  --arms chep_cau,ep_da_khe_giao,compose_2cha \
  --n 40 \
  --out out/compose_era_fix_smoke.txt \
  --out-key out/compose_era_fix_smoke_key.json \
  --out-meta out/compose_era_fix_smoke_meta.json
```

Từ meta, nhánh `compose_2cha`:

| thước | mốc đã chốt (P202, n=40) | phán |
|---|---:|---|
| `n_src≥2` | 100% | <95% ⇒ vá làm hỏng truy hồi → báo lại, **đừng** tự chỉnh ngưỡng |
| mean `ent_keep` | 0,872 | tụt >0,05 ⇒ như trên |
| `joined` | 100% | — |

**Nếu hai thước đầu giữ hoặc tốt lên:** vá là trung tính-hoặc-lợi trên đường ship
→ ghi số vào commit, không cần chấm mù lại.

**Nếu tụt:** đừng vá tiếp. Ghi lại số + 5 mẫu bị đổi (so `out/compose_p202_smoke.txt`)
và dừng — quyết định thuộc về tác giả, vì hướng đúng có thể là *bỏ hẳn* bộ lọc
niên đại hardcode chứ không phải chỉnh nó.

## 3b. A/B scaffold từ vựng trên v12 (nếu còn thời gian)

Trên mini đã đo: bảng từ viết tay **không** gánh kết quả trong probe
(`out/scaffold_ab_arm*.json`), nhưng **không chuyển được** sang module ship vì
gidf cấm luôn danh từ riêng tần suất cao (`Nguyễn` gidf 4,09 < 4,5). Chi tiết:
docstring `hu_tu_tu_do_thi` trong `src/nn_aivie/hca/analogy_bridge_gen.py`.

Câu hỏi cho v12: **corpus lớn hơn có tách được hai loại đó không?** (v12 nhiều
cha hơn ⇒ gidf của họ người có thể tách khỏi gidf hư từ.)

```bash
for A in full off gidf; do
  PYTHONPATH=src .venv/bin/python scripts/probe_analogy_exec_v5.py \
    --checkpoint checkpoints/network_v12_subst_isg_pw.pkl \
    --scaffold $A --out out/scaffold_ab_v12_$A.json
done
```

Đọc: `pos.bridge.rate` · `pos.bridge.n_fill` · tỉ số `bridge/embedding` ·
`neg.bridge.rate` (FP). Mốc mini: full 16 ca/100%/3,2× · off 31/87,1%/4,5× ·
gidf 40/90,0%/5,1× · FP 0% cả ba.

Ngoài ra in gidf của vài họ người trên v12 để biết ngưỡng 4,5 có còn cấm nhầm không:
`Nguyễn`, `Trần`, `Lê`, `Hồ` — nếu chúng đều ≥ 4,5 thì nhánh gidf **có thể**
bật được cho ship, và khi đó chạy lại smoke `analogy_la` để xác nhận.

## 4. Giao nộp

Commit: `out/check_ckpt_*.{log,json}`, `out/compose_era_fix_smoke*`, và một
dòng cập nhật vào `TK_046` §18.H / `TK_051` §11 nếu số dịch.

Commit message ghi đủ: `n_synonym_edges` · `n_triples` (cả hai ckpt) ·
`la_emit_doi_so_voi_min_id` · `n_src≥2` và `ent_keep` của `compose_2cha`.

## 5. Không làm trên runbook này

- Không đổi mặc định `generation_mode` (vẫn `w_skip`).
- Không tự hạ ngưỡng `pred_top_ban`/`prop_gidf_min` để "cứu" `n_triples` — chỉ **báo cáo** số.
- Không bake lại `Lbl:*` (đã gỡ, xem `TK_046` §18.H).
- Không chấm mù lại 120 mẫu trừ khi §3 cho thấy số dịch.
