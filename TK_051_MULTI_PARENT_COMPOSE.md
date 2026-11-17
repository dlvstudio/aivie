# TK_051 — GHÉP ≥2 CHA / ≥2 FACT (composition có phép)

> Mở 2026-08-06. **ĐÓNG 2026-08-06** (cổng 1–3 ĐẠT · chốt ship opt-in).
> Sau chốt ship P112 (`TK_050` đóng).
>
> P112 = **một** khung + vá khe hẹp. Ticket này hỏi: ghép **hai nguồn cha**
> cùng mạch đề mà không rơi về salad / lệch thực thể.

## 0. Câu hỏi (một dòng)

> Ghép **≥2 cha / ≥2 fact** cùng mạch đề mà dùng+tạm **≥ chep_cau − 5đ**,
> và tỷ lệ mẫu `n_src≥2` ≥ **30%**?

## 1. Bối cảnh đã đo (không làm lại)

| đã đo | kết luận |
|---|---|
| P112 1-khung + vá | chất lượng gần chép (−3,3đ); sáng tạo hẹp |
| P114/115 đa-khe / thuc_the | tăng `n_thay`, phá chất lượng |
| P108 Hebbian-CO | hại |
| `probe_retrieval_compose.py` (`generate(multi=True)`) | đã có thước `n_facts` / `ent_keep` — **tái dùng**, không săn hep đa-khe |

Nhãn thành công **không** phải `n_thay≥2`. Thành công = `n_src≥2` (hai cha thật vào output) + chất lượng.

## 2. Giả thuyết (ghi TRƯỚC)

Hai cha cùng **lĩnh vực / thực thể đề** (overlap theme/entity), nối bằng ranh giới câu
hoặc bản lề **đã thấy trong corpus** — không vá 2 khe soft trên một khung.

```
đề → rank top-2 cha (lọc entity)
    → nếu nối (cuối A → đầu B) có phép → emit A + emit B
    → không → fallback P112 (chỉ cha #1 + vá)
```

## 3. Cổng G-compose (nêu TRƯỚC)

| # | điều kiện | nếu trượt |
|---|---|---|
| 1 | Smoke: ≥30% mẫu `n_src≥2` | composition chưa bật — sửa retrieve/nối, **đừng chấm** |
| 2 | `ent_keep` ≥ mức P112 cùng bộ đề | ghép sai ngữ cảnh — siết lọc cha |
| 3 | Chấm mù: `compose − chep ≥ −5đ` **và** `n_src≥2` ≥30% | hướng chết — đóng B, giữ ship A |
| 4 (tuỳ chọn) | `compose − ep_da_P112 ≥ 0` dùng+tạm | không bắt buộc để tuyên bố sống |

## 4. Việc không làm

- Soft-giao / hybrid `thuc_the` để tăng `n_thay`.
- Hebbian-CO vào `_co`.
- Coi `n_thay≥2` = composition thành công.
- Đè `network_v12_sent.pkl`.
- Đổi mặc định `aivie generate` sang `frame_da_khe`.

## 5. Lộ trình

| bước | nội dung | runbook |
|---|---|---|
| **P201** | dual retrieve + meta `n_src` / parent ids; smoke cổng (1)(2) | `RUNBOOK_TK_051_P201.md` |
| **P202** | nối có phép (W_skip / tri count ≥ ngưỡng); fallback P112 | sau P201 đạt smoke |
| **P203** | chấm mù n=120 + neo 40; cổng (3) | sau P202 |

## 6. Giao nộp (khi code)

- Arm vd. `compose_2cha` cạnh `chep_cau` / `ep_da_khe_giao` trong `gen_three_arm` (hoặc CLI riêng).
- Meta: `n_src`, `parents=[i,j]`, `joined: bool`, `fallback_p112: bool`.
- Artefacts: `out/compose_p201_*` …

## 7. Trạng thái

**ĐÓNG.** P201–P203 **ĐẠT** cổng (1)(2)(3). Ship opt-in §11 — không đổi mặc định `w_skip`.

## 8. KẾT QUẢ P201 (2026-08-06) — cổng (1)(2) **ĐẠT**

Smoke n=40/nhánh · `chep_cau` · `ep_da_khe_giao` · `compose_2cha` · ckpt subst.

| nhánh | `n_src≥2` | mean `ent_keep` | joined |
|---|---:|---:|---:|
| `chep_cau` | 0% | 0,556 | — |
| `ep_da_khe_giao` | 0% | 0,535 | — |
| **`compose_2cha`** | **100%** | **0,793** | 100% |

| cổng | kết quả |
|---|---|
| (1) `n_src≥2` ≥30% | **100%** ✅ |
| (2) `ent_keep` ≥ min(chep, ep_da) | 0,793 ≥ 0,535 ✅ |

Arm: `compose_2cha` trong `gen_three_arm.py` — dual retrieve (theme overlap + era filter),
nối đơn giản `A + " " + B`. Artefacts: `out/compose_p201_smoke_*`.

**Chưa chấm mù** (cổng 3 = P203). Nối có phép = **P202**.

## 9. KẾT QUẢ P202 (2026-08-06) — smoke cổng nối **ĐẠT**

W_skip bản lề (cửa sổ 3×3, d≤8) · chọn B theo `ent_keep` · n=40.

| | `n_src≥2` | mean `ent_keep` | joined |
|---|---:|---:|---:|
| P201 (nối thô) | 100% | 0,793 | 100% |
| **P202** | **100%** ✅ | **0,872** ✅ (Δ=+0,08) | 100% |

Artefacts: `out/compose_p202_smoke_*` · `RUNBOOK_TK_051_P202.md`.

## 10. KẾT QUẢ P203 (2026-08-06) — cổng (3) **ĐẠT**

Chấm mù n=120/nhánh + neo 40 · seed=203 · ckpt subst · join d≤8 min_count≥1.

| nhánh | dung | dùng+tạm | hong | `n_src≥2` | ent_keep |
|---|---:|---:|---:|---:|---:|
| `chep_cau` | 36,7% | **54,2%** | 45,8% | 0% | 0,526 |
| `ep_da_khe_giao` | 26,7% | 44,2% | 55,8% | 0% | 0,427 |
| **`compose_2cha`** | 18,3% | **55,0%** | 45,0% | **99,2%** | **0,875** |

| cổng | kết quả |
|---|---|
| (3a) `compose − chep` (dùng+tạm) ≥ −5đ | **+0,8đ** ✅ |
| (3b) `n_src≥2` ≥30% | **99,2%** ✅ |
| (4 tuỳ chọn) `compose ≥ ep_da` | 55,0% ≥ 44,2% ✅ |

Neo: phiên 60,0% vs tham chiếu 67,5% · TRÔI **−7,5%** (ghi nhận; không cổng).

`compose` giữ dùng+tạm ngang/nhỉnh chép nhưng `dung_duoc` thuần thấp hơn (18% vs 37%) —
nhiều mẫu `tam` (nối được mạch, khe/seam). Fallback P112 chỉ **0,8%**.

Artefacts: `out/compose_p203_{samples,key,meta,cham,verdict}.*` · `RUNBOOK_TK_051_P203.md`.

## 11. CHỐT SHIP (2026-08-06) — đóng ticket

### Công thức ship (P203)

| tham số | giá trị |
|---|---|
| checkpoint | `checkpoints/network_v12_subst.pkl` |
| dual retrieve | theme keep-overlap + era filter · top_n=24 · ≤12 cand |
| bản lề | W_skip `d≤8` · `min_count≥1` · cửa sổ 3×3 |
| chọn B | max `(ent_keep, join_score)` |
| fallback | P112 trên cha #1 (bam · min_gain=1 · max_thay=2) |

Thước đóng: `compose−chep = +0,8đ` · `n_src≥2 = 99,2%` · `ent_keep` 0,875.

### Cách gọi

- Primary: `RUNBOOK_COMPOSE_P203_SHIP.md` → `scripts/gen_compose_2cha.py`
- Opt-in: `aivie generate --mode compose_2cha` (mặc định CLI vẫn `w_skip`)
- Module chung: `src/nn_aivie/hca/compose_2cha.py` (cùng luật với `gen_three_arm`)

### Cổng G-compose — tóm tắt

| # | kết quả |
|---|---|
| 1 `n_src≥2` smoke | **ĐẠT** |
| 2 `ent_keep` | **ĐẠT** |
| 3 chấm mù | **ĐẠT** (+0,8đ) |

Ticket đóng. Ship 1-khung vẫn `TK_050` / P112. Không mở siết `dung_duoc` trừ khi có câu hỏi mới + cổng trước.

**Hồi quy era syllable-match (2026-08-10, M4):** vá `eras_in_text` (âm tiết, không chuỗi con) → smoke n=40 `compose_2cha` giữ nguyên P202: `n_src≥2` 100% · mean `ent_keep` 0,872 · `joined` 100% (`out/compose_era_fix_smoke*`). Vá trung tính trên đường ship — không chấm mù lại.
