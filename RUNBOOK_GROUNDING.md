# RUNBOOK GROUNDING — vá retrieval trôi entity + confirm scale (TK_042, 2026-07-21)

> Thread riêng khỏi LM data-scale (`RUNBOOK_TK_042.md`). Máy dev Intel 8GB không load nổi graph
> 20GB; validate chính = **harness LOCAL nhỏ** (không cần M4) + **M4 confirm scale**.
> **Dán nguyên văn stdout, đừng tóm tắt.**

---

## 📍 TRẠNG THÁI

| | việc | trạng thái |
|---|---|---|
| ✅ | **Fix 1** (`efc55b0`): route agenda `_build_segment_plan` qua `facts_about(theme)` | XONG — nhưng KHÔNG phải đường OUTPUT (xem §2.4) |
| ✅ | **Fix 2** (`6ee491e`): primary phá thế-hòa gidf bằng theme-concentration | XONG · facts_about |
| ✅ | **Diag v11** (`6352059`): định vị `plan.agenda` ĐÚNG mà OUTPUT SAI → đường KHÁC | XONG (§2.4) |
| ✅ | **Fix 3** (`f9e3017`): cap-gated primary trong `knowledge_extra_fact_plan` (đường OUTPUT) | XONG · Thăng Long→Thăng · 245 test |
| ✅ | **Diag Fix 3** (`after_f9e3017`): facts_about-theme + agenda ĐÚNG mà OUTPUT vẫn Gia Long → đường THỨ TƯ | XONG (§2.6) |
| ✅ | **Fix 4** (`7f78c96`): đường OUTPUT crisp (`completion`→`verbatim`) ưu tiên facts_about-theme | XONG · 245 test |
| ✅ | **M4 confirm Fix 4 scale v11** (`1403a28`) | **ĂN — Thăng Long→Thăng Long, Nhà Trần→Nhà Trần; ans_recall 0.28→0.34; đọc tay ĐÚNG HẾT. GROUNDING VÁ XONG** |
| ⏳ | Residual query-intent (Nhà X = dynasty vs person) | để riêng |
| ⏳ | Aspect-split content-only (mở khoá sau grounding) | sau confirm |

---

## 1. Vấn đề đã vá

Đo (`probe_retrieval_compose_v11.txt`, `probe_aspect_split_v11.txt`): **composition ✓** (đoạn nhiều
câu mạch lạc) nhưng **grounding ✗** — trôi entity: Thăng Long→Gia Long (va chạm piece `ĠLong`),
Ngô Quyền→Đinh Tiên Hoàng, Bạch Đằng→Trần Hưng Đạo. `ans_recall 0.28 · ent_keep 0.00`.

**Gốc:** đường CHÍNH `rank_matches` xếp hạng coverage×power (theme-blind) → node prominent /
va-chạm-bề-mặt thắng. `facts_about` CÓ tín hiệu theme (nơ-ron chủ-đề, trọng số vị-trí entity) →
subject-aware ĐÚNG. **Bug: theme không dùng ở đường chính.**

**Fix:** `_build_segment_plan` ưu tiên `facts_about(theme)` cho truy vấn có focus; rank_matches
fallback (escape: `compose_theme_weight=0` → cũ). + primary phá thế-hòa gidf bằng theme nhỏ hơn.

Đo local (before→after, `scripts/probe_grounding_local.py`):
```
Thăng Long : Gia Long        → Thăng Long        ✓
Ngô Quyền  : Đinh Tiên Hoàng → Ngô Quyền         ✓
Bạch Đằng  : grade lessons   → Sông Bạch Đằng    ✓
Trần Hưng Đạo: Lý Chiêu Hoàng → Trần Hưng Đạo    ✓ (control)
Nhà Trần   : Nhà Ngô         → nội dung Trần     (~ đúng họ, chưa dynasty)
```

---

## 2. 🔄 VIỆC M4 — confirm scale v11

⚠️ **CHẠY `retrieval_compose`, KHÔNG chạy `probe_grounding`.** `probe_grounding` gọi
`top_focus/rank_matches/best_match` TRỰC TIẾP → **HANG** trên M4 (graph 20GB swap, đã thử 2 lần).
`retrieval_compose` gọi qua `generate()` (đường có fix) → **đã chạy trót lọt** + có `ans_recall`.

```bash
cd <repo> && git pull && export PYTHONPATH=src   # lấy efc55b0 + 6ee491e
python -u scripts/probe_retrieval_compose.py --checkpoint checkpoints/network_v11.pkl \
    2>&1 | tee out/probe_retrieval_compose_v11_fixed.txt
git add -f out/probe_retrieval_compose_v11_fixed.txt \
  && git commit -m "TK_042: retrieval_compose SAU fix grounding" && git push
```

**Đọc — so baseline `probe_retrieval_compose_v11.txt`:**

| thước | baseline (trước fix) | điều kiện phân định |
|---|---|---|
| ENTITY `ent_keep TB` | 0.00 | **tăng rõ** ⇒ fix giữ ở scale; ≈0 ⇒ power-bias v11 mạnh hơn harness, fix KHÔNG đủ |
| GRADE `ans_recall TB` | 0.28 | **tăng** ⇒ bớt trôi sang sử |
| đọc tay ENTITY | Thăng Long→Gia Long v.v. | entity ĐÚNG (Thăng Long→Thăng Long) |

⚠️ **Nếu KHÔNG tăng ở v11** ⇒ harness 659 parents chưa lột hết power-bias của 65k → cần xem lại
(có thể cần cluster-anchor cho va-chạm-bề-mặt, hoặc siết theme mạnh hơn). GHI THẬT, không tự huyễn.

**Bonus (tùy chọn):** `probe_aspect_split.py` cũng lộ fix (aspect Bạch Đằng giờ ra Bạch Đằng, không
Trần Hưng Đạo) — nhưng `retrieval_compose` có `ans_recall` nên là thước chính.

### 2.1. KẾT QUẢ M4 CONFIRM SCALE (2026-07-21)

Đã chạy đúng lệnh runbook, artifact:

`out/probe_retrieval_compose_v11_AFTER_FIX.txt` (commit `302fb65`, đã push)

So baseline `out/probe_retrieval_compose_v11.txt`:

| thước | baseline | AFTER_FIX | kết luận |
|---|---:|---:|---|
| ENTITY `n_facts TB` | 2.33 | 2.33 | composition giữ nguyên |
| ENTITY `copy TB` | 0.38 | 0.38 | giữ nguyên |
| ENTITY `ent_keep TB` | 0.00 | 0.00 | **không cải thiện** |
| GRADE `ans_recall TB` | 0.28 | 0.28 | **không cải thiện** |

Kết luận đúng tiêu chí runbook: fix local 4/5 **không scale** lên v11; giả thuyết power-bias 65k
parents vẫn thắng tín hiệu theme trên đường sinh thật.

### 2.2. TRẢ LỜI NGHI NGỜ "CHẠY CODE CŨ" (FORENSIC)

Đã kiểm chứng theo 3 lớp, kết luận: run AFTER_FIX **không chạy code cũ**.

1) **Chuỗi commit**

- `efc55b0` (17:07:33): route agenda qua `facts_about(theme)`
- `6ee491e` (17:16:41): tie-break primary bằng theme-concentration
- `302fb65` (19:03:13): commit artifact AFTER_FIX

⇒ artifact AFTER_FIX được commit SAU 2 commit fix.

2) **Dấu vân tay patch có trong source hiện hành**

- `src/nn_aivie/hca/generate_hebbian.py`: `_build_segment_plan` ưu tiên `facts_about(... theme_weight=...)`
   rồi mới fallback `rank_matches`.
- `src/nn_aivie/core/parent_store.py`: logic near-tie gidf (`<=0.3`) chọn token có theme nhỏ hơn.

3) **Timestamp + header runtime của file output**

- `out/probe_retrieval_compose_v11_AFTER_FIX.txt` mtime: `2026-07-21 18:58:24`
- header run ghi PID/loader mới (`pid=42057`, `load_network_v11_20260721_183751.log`), không phải
   artifact cũ.

=> Nghi ngờ "chạy nhầm code cũ" đã được bác bỏ bằng timeline + source fingerprint + runtime header.

### 2.3. DIAGNOSTIC RETRIEVAL + COMPOSE FLOW (COMMIT cb19742)

Chạy probe với `[diag]` traces để xác định **khớp locus của power-bias** (generate() scope hay
facts_about() scope?). Script thêm diagnostic output cho mỗi prompt:
```
[diag] focus='TOKEN1'(gidf1) 'TOKEN2'(gidf2) · facts_about-theme#1='FACT' · plan.agenda='PLAN'
```

**Kết quả 6 ENTITY prompts:**

| Prompt | Focus (gidf) | facts_about-theme pick | Final Output | Drift? |
|---|---|---|---|---|
| Nhà Lý | Nhà(5.13) Lý(4.90) | "Lý Thường Kiệt nổi tiếng..." | ✅ Lý Thường Kiệt | **Không** |
| Nhà Trần | Nhà(5.83) Trần(4.97) | "Nhà Trần ba lần đánh..." | ❌ Hồ Quý Ly | **CÓ** |
| Hà Nội | Hà(4.53) Nội(5.10) | "Hà Nội có Hồ Gươm..." | ❌ Ca trù Hà Nội | **CÓ** |
| Sông Hồng | Sông(6.89) Hồng(5.27) | "Sông Hồng chảy qua..." | ❌ Bài 1 luyện mẫu | **CÓ** |
| Trần Hưng Đạo | **Lý(1.89) Ai(3.69)** | "AI hỗ trợ đọc X-ray..." | ❌ Lý Chiêu Hoàng | **CÓ** |
| Thăng Long | Thăng(6.49) Long(4.72) | "Lý Thái Tổ dời kinh..." | ❌ Gia Long | **CÓ** |

**Kết luận:** 
1. `facts_about(theme)` **lựa chọn neo ĐÚNG** (fact relevant với theme) ✅
2. Nhưng `generate()` **thoát ra khỏi neo** → composition tự kích hoạt độc lập → vẫn drift ❌
3. **Thêm vấn đề focus:** Trần Hưng Đạo — focus tính sai (chọn "Lý"+"Ai" thay "Trần Hưng Đạo")
   → facts_about-theme pick nhầm sang AI.

**Nguyên nhân gốc:** Power-bias 65k nodes vẫn thắng tín hiệu theme **trong compose phase**,
không phải trong retrieval phase. Fix2 (tie-break gidf) làm yếu power-bias ở **retrieval**, nhưng
**generation LM vẫn ignores** theme neuron signal (scope limit).

### 2.4. ĐÍNH CHÍNH GỐC (soi code sau diag) — KHÔNG phải "power thắng theme", mà PRIMARY VA-CHẠM

Diag lộ mâu thuẫn: `plan.agenda` (từ `_build_segment_plan`) = **Thăng Long** ĐÚNG, nhưng OUTPUT =
verbatim **"Tiêu đề: Gia Long"**. ⇒ generate(multi) KHÔNG dùng agenda đó cho output — nó đi đường
**`knowledge_extra_fact_plan`** (đường bồi-fact multi), primary từ `_knowledge_entity_span` =
**`entity[-1]` (token PHẢI NHẤT)**. "Thăng Long" → primary=**'Long'** → `theme_neuron('Long')` kéo
Gia Long/Hạ Long vào pool → xếp `stim×power` → Gia Long (prominent viwiki) thắng → verbatim.

⇒ KHÔNG phải "power thắng theme ở compose". Mà **primary chọn nhầm token va-chạm ('Long')** ⇒ node
sai LỌT POOL từ đầu. `theme_neuron('Thăng')` sẽ KHÔNG chứa Gia Long → chọn đúng primary là loại
tận gốc. (Fix1/Fix2 vá `_build_segment_plan`/`facts_about` — đúng nhưng KHÔNG phải đường OUTPUT.)

### 2.5. 🔄 FIX 3 + VIỆC M4 (commit `f9e3017`)

`knowledge_extra_fact_plan`: primary `entity[-1]` đúng danh-từ-ghép ("Mùa hè"→'hè' head) nhưng SAI
tên riêng va-chạm. Tách bằng **cap_class**: entity TOÀN HOA (tên riêng) → primary = token ĐỊNH DANH
(theme_neuron NHỎ nhất, tie gidf); có token thường → GIỮ rightmost. Đo local: Thăng Long 'Long'→
'Thăng', Nhà Trần→'Trần', Ngô Quyền→'Ngô'; Mùa hè giữ 'hè'. **245 test pass**.

```bash
cd <repo> && git pull && export PYTHONPATH=src
python -u scripts/probe_retrieval_compose.py --checkpoint checkpoints/network_v11.pkl \
    2>&1 | tee out/probe_retrieval_compose_v11_fix3.txt
git add -f out/probe_retrieval_compose_v11_fix3.txt \
  && git commit -m "TK_042: retrieval_compose Fix 3 cap-gated (confirm scale)" && git push
```

**Đọc:** "Hỏi: Thăng Long là gì" → generate ra **Thăng Long** (KHÔNG còn "Tiêu đề: Gia Long"). `[diag]`
in `focus · facts_about-theme#1 · plan.agenda`. So baseline `ent_keep 0.00 / ans_recall 0.28`:
- **ent_keep/ans_recall tăng + Thăng Long đúng** ⇒ Fix 3 giữ ở scale ✓
- **vẫn Gia Long** ⇒ primary v11 vẫn ='Long' (min-theme chưa tách được ở 65k) → dán [diag], tôi xem
  focus+theme thật rồi siết tiếp. GHI THẬT.

⚠️ Vì sao KỲ VỌNG giữ ở scale: primary='Thăng' ⇒ `facts_about` đòi 'Thăng' có mặt ⇒ Gia Long (thiếu
'Thăng') **bị loại khỏi pool tận gốc**, bất kể power. Ở v11 'Long' theme CÀNG lớn (viwiki nhiều tên
'Long') nên min-theme chọn 'Thăng' càng chắc.

---

## 3. Harness LOCAL (iterate grounding KHÔNG cần M4)

`scripts/probe_grounding_local.py` — build graph NHỎ (history+subject+grade 659 parents) từ file
local, cache `checkpoints/mini_grounding.pkl` (gitignore, 84MB). **Load 12s → iterate fix 12s/vòng.**
Lần đầu build ~6 phút (g1 co-activation đoạn dài); sau đó load cache.

```bash
rm -f checkpoints/mini_grounding.pkl   # build lại nếu đổi FILES/corpus
PYTHONPATH=src python -u scripts/probe_grounding_local.py
```

⚠️ **Giới hạn:** 659 parents ≠ 65k — power-bias yếu hơn v11. Fix đúng trên harness PHẢI M4 confirm
(mục 2). Nhưng harness đủ để tái hiện cơ chế + chặn regression 4 ca chính, rẻ hơn M4 nhiều.

---

## 4. Còn lại (sau khi M4 confirm)

1. **Residual query-intent** (Nhà Trần → Trần Hưng Đạo thay bài dynasty): query "Nhà X là gì" muốn
   *triều đại*, focus "Trần" khớp cả *người* → theme ưu tiên người prominent. Vấn đề intent
   (dynasty vs person), sâu hơn primary — để riêng.
2. **Cluster-anchor** cho va-chạm-bề-mặt (Thăng Long fix nhờ theme, nhưng đòi cụm "Thăng"+"Long"
   kề là gốc rễ hơn — 1/4 ca gốc là surface collision).
3. **Aspect-split content-only** (ý đa-góc-nhìn tác giả): spreading đa-seed cho NỘI DUNG; style về
   LM+SFT (style-wall đã đo). Grounding vá xong ⇒ khả thi hơn.

---

## §2.6. FIX 4 — đường OUTPUT crisp (commit `7f78c96`) + việc M4

Diag sau Fix 3: `facts_about-theme#1` + `plan.agenda` ĐÚNG (Thăng Long) mà OUTPUT vẫn verbatim
"Gia Long" → OUTPUT đến từ **đường THỨ TƯ**. Dump `rank_matches("Thăng Long")` local:

```
8 bài sử ĐỒNG ĐIỂM 0.8922 (Gia Long, Hai Bà Trưng, Hồ Quý Ly, Lê Thái Tổ...)
→ bài "Thăng Long" KHÔNG có trong top!  ⇒ rank_matches THOÁI HOÁ (hoà hết, bài đúng biến mất)
```

⇒ KHÔNG phải cluster-anchor. rank_matches không phân biệt nổi cho tên riêng đa-token. Đường OUTPUT
crisp trong `generate()` (block `completion` → `verbatim`, dội khi A3 phủ agenda thấp) lấy completion
từ rank_matches/best_match thoái hoá → Gia Long. **Fix 4:** ưu tiên `facts_about(theme)` (subject-
aware, TÌM ĐÚNG — diag chứng minh) TRƯỚC rank_matches trong block này. 245 test pass. (Đây là đường
thứ 4; 3 fix trước feed AGENDA, không phải verbatim OUTPUT — đó là lý do metric không nhích.)

```bash
cd <repo> && git pull && export PYTHONPATH=src
python -u scripts/probe_retrieval_compose.py --checkpoint checkpoints/network_v11.pkl \
    2>&1 | tee out/probe_retrieval_compose_v11_fix4.txt
git add -f out/probe_retrieval_compose_v11_fix4.txt \
  && git commit -m "TK_042: retrieval_compose Fix 4 (output crisp)" && git push
```

**Đọc:** "Hỏi: Thăng Long là gì" → generate ra **Thăng Long** (KHÔNG "Tiêu đề: Gia Long") + ent_keep/
ans_recall tăng ⇒ Fix 4 ăn ở scale ✓. Vẫn Gia Long ⇒ còn đường THỨ NĂM (dán toàn văn + [diag], hoặc
grounding là hydra quá sâu → cân nhắc gác, pivot data-weighting 10d). GHI THẬT.

---

## FOCUS-BUG (2026-07-22) — từ-để-hỏi giành focus → sai fact

Sau khi CHỐT âm tính binding (TK_044), lever chất-lượng-sinh thật lộ ở `out/probe_retrieval_compose_
v11_fix4.txt`: 2 ca GÃY do FOCUS lệch:
- `"Trần Hưng Đạo là ai"` → focus `'là'(1.89) 'ai'(3.69)` (KHÔNG phải thực-thể) → `facts_about` kéo
  **bài AI/X-quang** (nghi: `'ai'` fold ≡ `'AI'` trí-tuệ).
- `"Hà Nội là gì"` → trôi "Ca trù".

**Hai tầng nghi:** (1) thực-thể NỔI TIẾNG gidf THẤP ("Trần Hưng Đạo" khắp corpus → idf thấp → thua
từ-hỏi); (2) case-fold chéo-lớp `'ai'(who)≡'AI'(trí-tuệ)` → focus sai kéo fact sai.

**M4 chạy diagnostic (phơi bày, chưa vá):**
```bash
PYTHONPATH=src python -u scripts/probe_focus_bug.py \
    --checkpoint checkpoints/network_v11.pkl --log-file out/probe_focus_bug.log
```
In per-token (gidf · fold_gidf · gdf · FB-opener? · case-equiv) + entity_register ranking + focus +
facts_about#1, cho ca GÃY (Trần Hưng Đạo, Hà Nội) vs ĐÚNG (Nhà Trần, Thăng Long). Dán log về →
xác định điểm vá: (a) demote từ-hỏi trong `entity_register_ids` (structural, không hardcode);
(b) chặn case-fold chéo-lớp (ai≠AI); (c) neo cụm thực-thể đa-token vượt token-hỏi đơn.

**KẾT QUẢ (2026-07-22) — focus-bug KHÔNG tái hiện:** `out/probe_focus_bug.log` cho thấy trên checkpoint
hiện tại focus ĐÚNG cả 4 ca: "Trần Hưng Đạo là ai"→focus 'Trần''Hưng'→facts đúng; "Hà Nội là gì"→
'Hà''Nội'→đúng. entity_register xếp thực-thể TRÊN từ-hỏi ('Trần'>'ai'). Case-fold 'ai'≡'AI' CÓ tồn tại
(caseq='AI Ai ai…') nhưng KHÔNG kích hoạt vì entity outrank. → Bug từ log Jul-21 (`compose_v11_fix4`)
đã hết (fix hoặc transient). **KHÔNG có focus-bug để vá.** Đừng đuổi lại. Residual thật còn lại =
SỐ nát (10809 — tường số học TK_041).
