# RUNBOOK TK_045 — P47 (tầng mệnh đề) + P48 (Hopfield phá thế hoà) cho M4

> **Hai phần độc lập, chạy song song được.**
> **PHẦN A (§0–9, dưới đây) = P47** — cổng sinh-tử tầng nơ-ron MỆNH ĐỀ (§13 Trụ A). Nặng RAM.
> **PHẦN B (cuối file) = P48** — Hopfield phá thế hoà cho `rank_matches` (§13 Trụ D). Nhẹ hơn.

---

# PHẦN A — P47: CỔNG SINH-TỬ tầng nơ-ron MỆNH ĐỀ

**Mục tiêu:** chạy `scripts/probe_clause_layer.py` trên **v12** và trả về **ba số**. Ba số đó
quyết định **§13 sống hay chết** — không phải "cải thiện bao nhiêu %", mà **có làm tiếp không**.

**Đọc trước (bắt buộc, theo thứ tự):** `TK_045_MATRIX_CORE.md` **§13.0 → §13.8**. Runbook này
không lặp lại lập luận ở đó; nó chỉ nói **chạy gì, đọc số thế nào, kết luận gì**.

**Vì sao có P47 (một đoạn):** xếp lại toàn bộ kết quả đo của TK_045 theo hình dạng bài toán thì
mọi ÂM TÍNH là *"sinh một phân bố trên 32k token"* và mọi DƯƠNG TÍNH là *"chọn trong một tập nhỏ
đã được chứng thực"*. ⇒ Giả thuyết §13: **nâng ĐƠN VỊ SINH từ token lên MỆNH ĐỀ**. P47 đo xem
tầng mệnh đề có tồn tại thật không, hay chỉ là phát lại cha (= chép, vô nghĩa).

---

## 0. Tiền đề & kỷ luật

- **Luật 0:** probe **chỉ ĐỌC** đồ thị (W_seq cho biên, W_skip cho mối nối). Không train, không
  hardcode từ vựng/dấu câu. Biên câu đến từ `P(EOS|w)` lift (§0.4.2); biên dưới-câu đến từ thung
  lũng PMI trên W_seq. **Không sửa hai tín hiệu này thành danh sách dấu câu** — nếu bạn thấy tiện
  tay làm thế, đọc lại §13.6 vấp 1.
- **Measure-first:** cổng đã **nêu trước** (mục 3). Dự đoán của Claude đã **ghi trước** (mục 5) để
  số có thể **bác** được. **Không chỉnh cổng sau khi thấy số** — nếu bạn tin cổng sai, viết một mục
  §13.x nói *tại sao sai* và *ngưỡng thay thế lấy từ tài liệu nào có TRƯỚC*, đúng như §13.6 đã làm
  với cổng `H_type` hỏng của Claude.
- **Observability:** probe tự ghi `out/logs/probe_clause_layer_<ts>.log` (gồm tiến độ nạp
  checkpoint). Giữ log, đừng chỉ giữ JSON.
- **Nạp checkpoint:** probe đã `gc.disable()` quanh `load_checkpoint` (1300×). Không gỡ.
- **Test:** `pytest -m "not slow"` (4 giây). `tests/test_clause_layer.py` phải **xanh 13/13** trước
  khi tin bất kỳ số nào.

---

## 1. Smoke trước (30 giây — xác nhận môi trường, KHÔNG bỏ qua)

```bash
python3 -m pytest tests/test_clause_layer.py -q -m "not slow"
```

Kỳ vọng **13 passed**. Nếu đỏ ở `test_eos_lift_is_BLIND_to_sub_sentence_boundary` thì tín hiệu
biên đã đổi bản chất — **dừng, báo Claude**, đừng chạy v12.

```bash
PYTHONPATH=src .venv/bin/python scripts/probe_clause_layer.py \
    --checkpoint checkpoints/mini_sent_ep_qa.pkl \
    --sample 300 --dump 0 --pmi-thetas none,0.0,0.5 \
    --out-json out/probe_clause_layer_mini.json
```

Kỳ vọng tái lập §13.6 (sai số nhỏ do mẫu ngẫu nhiên):

```
PMI none : 6,819 mệnh đề · dài TB 16.41 · KHÁC-CHA 99.5% · 0.230 bits/token
PMI 0.0  : 10,283 mệnh đề · dài TB 10.72 · KHÁC-CHA 99.8% · ~0.56 bits/token
VERDICT: MỞ
```

Lệch xa ⇒ môi trường/checkpoint khác — **báo trước khi chạy v12**.

---

## 2. Chạy THẬT trên v12

```bash
PYTHONPATH=src .venv/bin/python scripts/probe_clause_layer.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --sample 1500 --dump 8 \
    --out-json out/probe_clause_layer_v12.json \
    2>&1 | tee out/probe_clause_layer_v12.log
```

| | |
|---|---|
| **Nạp** | 430 447 cha · unpickle ~53s + adjacency ~560s + parents ~2–4 phút **nếu đủ RAM** |
| **W_skip** | `d_max=8, min_count=1, content_only=False` trên 430k cha — nặng nhất, cache lại trong `state` |
| **Sweep** | 6 ngưỡng PMI × (dựng tầng + 1500 mẫu mối nối) |
| **RAM** | ⚠️ **Đây là chỗ chết.** Claude chạy trên máy dev **Intel 8GB** → thrash, đứng yên 16 phút ở 39.5% cha, phải giết. **Cần máy ≥ 32GB.** Đúng blocker TK_040 mà README cảnh báo — **không phải lỗi probe** |

**Nếu vẫn thiếu RAM:** hạ quy mô bằng `--max-parents 100000` (giảm số cha đưa vào tầng mệnh đề).
⚠️ **Nó KHÔNG giảm chi phí NẠP** — nạp vẫn full 430k. Và số đo sẽ **không so được** với v12 đầy
đủ (ít cha ⇒ ít ứng viên ⇒ `H_type` thấp giả tạo). Nếu buộc phải dùng, **ghi rõ trong báo cáo**
và coi là *sơ bộ*, không phải phán quyết cổng.

Cờ khác — **để mặc định** trừ khi có lý do ghi lại được: `--theta 2.0` (= `ar_eos_boundary_min_lift`),
`--d-join 4`, `--w-skip-dmax 8`, `--w-skip-min-count 1`, `--seed 42`.

**Vá hiệu năng 2026-08-02 (đã kiểm bằng số):** `measure_joins` trước đây materialize một list
dài **bằng số ứng viên** cho mỗi mẫu (mini 1 821 · ước v12 **~138 000** × 1500 mẫu) chỉ để tính
`H_clause`. Thay bằng dạng giải tích tương đương

    H_clause = H_type + Σ_j q_j·log2(m_j)

(`m_j` = cỡ nhóm mệnh đề cùng token mở). Và `cross_parent` đếm **phần bù** qua `by_parent`
(mỗi cha vài mệnh đề) thay vì quét mọi thành viên. Đo lại trên mini: **số ra IDENTICAL**
(so từng trường tới 1e-9). Đẳng thức được khoá bằng `test_h_clause_analytic_identity`.
⇒ nút cổ chai còn lại ở v12 là **nạp checkpoint**, không phải `measure_joins`.

---

## 3. BA CỔNG (đã nêu trước — không đổi)

| # | đại lượng trong JSON | cổng | ý nghĩa nếu TRƯỢT |
|---|---|---|---|
| **M2** | `cross_parent_ratio` | **≥ 0.05** | tầng mệnh đề chỉ phát lại cha = chép ⇒ **ĐÓNG CẢ §13** |
| **M3** | `bits_per_token` | **< 2.0** | không thoát khe entropy ⇒ đóng hoặc đổi thiết kế (xem mục 6) |
| **M1** | `mean_len` | **∈ [4, 12]** | quá dài = thoái hoá về chép; quá ngắn = về salad |

`verdict` trong JSON = **có ít nhất một ngưỡng PMI qua cả ba**. Đó là con số duy nhất cần trả lời
"làm tiếp hay không".

**Cột `gate_h_type_OLD` là cổng HỎNG của Claude** (`H_type < 6.0`) — giữ để hồ sơ không mất dấu.
**Đừng dùng nó để kết luận.** Lý do hỏng: nó đo giá *mỗi quyết định*, trong khi §13 nói về *khấu
hao* — một quyết định 6.4-bit cam kết 9 token rẻ hơn hẳn chín quyết định 6.4-bit (§13.6 vấp 2).

---

## 4. Đọc đầu ra

Mỗi ngưỡng in 3 dòng + 1 dòng cổng:

```
M1: <n> mệnh đề · dài TB <x> (median <y>) · <k> token mở
M2: ứng viên TB <c> (<t> token mở) · KHÁC-CHA <p>% · chết <d>%
M3: H_type <a> bits · H_clause <b> · **<z> bits/token** (bigram = 6.42)
CỔNG: len=✓/✗ cross=✓/✗ bits/tok=✓/✗ ⇒ ĐẠT/TRƯỢT  [cổng-cũ H_type=…]
```

**Hai phép tự kiểm — chạy mắt qua trước khi tin số:**

1. **`lift cao nhất`** ở đầu log phải là `.` `!` `?` với lift ≫ 2, **mọi token nội dung = 0.0**.
   Nếu từ nội dung có lift cao ⇒ checkpoint hoặc W_seq có vấn đề.
2. **`H_type` phải tiến về ~6.42 khi `mean_len` co về ~6.** Đây là bằng chứng instrument đo đúng
   đại lượng nó tuyên bố (mini: len 5.90 → H_type 6.59). Nếu **không** thấy xu hướng này, số M3
   **không dùng được** — báo Claude.

**`--dump 8`** in 8 mệnh đề mỗi ngưỡng để **ĐỌC TAY**. Bắt buộc đọc — đây là lớp phòng thủ mà
test và metric đều mù (bài học TK_041 §5.2). Kỳ vọng: PMI ∈ [0.0, 0.5] ra mệnh đề thật
(`bị giặc đón chặn`, `ra đời thực sự`); PMI ≥ 1.5 ra vụn (`phó sứ`, `phần lớn`).

---

## 5. DỰ ĐOÁN GHI TRƯỚC (của Claude — để số BÁC được)

> Ở v12, `mean_candidates` **tăng mạnh** (nhiều cha ⇒ nhiều mệnh đề cùng token mở) ⇒ `H_type`
> **tăng** và `bits_per_token` **tăng** so với mini, có thể lên **1.0–1.5**.
> `cross_parent_ratio` **giữ ≥ 99%**.

Cơ chế nghi: cùng thứ v4 §3.7 đã đo — **entropy TĂNG theo scale** (mini 5.34 → 6.42). Nếu nó tái
diễn ở tầng mệnh đề thì phải biết **sớm**.

**Nếu số bác dự đoán này, ghi rõ** — dự đoán sai là thông tin, không phải điều đáng giấu.

---

## 6. BẢNG QUYẾT ĐỊNH (làm gì với từng kết cục)

| kết cục | kết luận | việc kế |
|---|---|---|
| **cả 3 cổng ĐẠT** ở ≥1 ngưỡng PMI | §13 Trụ A **MỞ** ở quy mô thật | Ghi ngưỡng PMI thắng làm mặc định cho `clause_layer.DEFAULT_PMI_THETA`. Sang **Trụ C** (cổng phủ) — đó mới là thứ quyết đúng-nội-dung |
| **M2 TRƯỢT** (`cross_parent` < 5%) | tầng mệnh đề = chép | **ĐÓNG §13 Trụ A–C.** Ghi §13.x âm tính. Trụ D (Hopfield) **vẫn sống** — độc lập |
| **M3 TRƯỢT** (`bits_per_token` ≥ 2.0) | khấu hao không đủ ở scale | **KHÔNG đóng vội.** Kiểm `mean_candidates`: nếu nó nổ (≫ mini) thì bệnh là **thiếu cổng lọc**, không phải thiếu khấu hao ⇒ **kéo Trụ C vào NGAY trong M3** (lọc ứng viên bằng PMI-cross với chủ đề trước khi tính H) rồi đo lại. Nếu đã lọc mà vẫn ≥ 2.0 ⇒ đóng |
| **M1 TRƯỢT mọi ngưỡng** | không có θ cho mệnh đề 4–12 token | Mở rộng `--pmi-thetas` (vd `2.0,3.0,4.0`) trước khi kết luận. Sweep hiện tại có thể chưa phủ vùng đúng ở v12 |
| **thrash / OOM** | giới hạn máy, **không phải kết quả** | Máy ≥32GB, hoặc chờ TK_040 (CSR). **Đừng ghi là âm tính** |

---

## 7. Báo cáo về đâu

Viết một mục mới **`### 12.x P47 — tầng mệnh đề trên v12 (<ngày>) — M4`** trong
`TK_045_MATRIX_CORE.md`, đúng văn phong ticket, gồm:

1. bảng 6 dòng (một dòng / ngưỡng PMI) với `n_clauses · mean_len · mean_candidates ·
   cross_parent_ratio · H_type · H_clause · bits_per_token · gate_pass`;
2. `verdict` cuối + **đọc tay** 2–3 mệnh đề đại diện ở ngưỡng thắng;
3. dự đoán mục 5 **đúng hay sai**;
4. đường dẫn `out/probe_clause_layer_v12.json` + `.log`;
5. cấu hình máy (RAM) — để lần sau biết chạy được ở đâu.

Rồi cập nhật **§13.8** (đổi từ "GIAO M4" sang "ĐÃ ĐO") và bảng **§13.4 hàng 1**.

---

## 8. Giới hạn PHẢI ghi kèm mọi trích dẫn số

**`bits_per_token` đo TRÔI CHẢY, KHÔNG đo ĐÚNG NỘI DUNG.** Nó giả định ruột mệnh đề tốn **0 bit**
— đúng về trôi chảy (ruột được chứng thực nguyên văn), **sai** về nội dung: chọn một mệnh đề là
cam kết luôn các fact trong đó. Giá nội dung là việc của **Trụ C** và cần phép đo riêng
(`lạc` / PMI-cross). **Không ai được trích `0.56 bits/token` như "mô hình ngôn ngữ 0.56 bits"** —
nó không phải thế.

---

## 9. Nợ đã biết (không thuộc P47, đừng gộp)

- **7 test đỏ có sẵn trên `a5f4862`** (⚠️ suite **không** xanh khi bạn bắt đầu — đừng tưởng do
  P47). Hai nhóm khác bản chất:
  - **6** = hồi quy §12.149, đáp mất chủ ngữ (`Hỏi: Cá voi là gì Đáp:` →
    `là động vật có vú lớn nhất.`). Ticket riêng.
  - **1** = `test_paths::test_viz_default_checkpoint_prefers_big_a2` — **môi trường**: chuỗi
    `elif` chưa liệt kê `grade_v3_big_a2_t3_isg.pkl` có trên máy. Đỏ trên mọi commit.
  - Xác minh: `git checkout a5f4862 && pytest -q -m "not slow"`.
  - `tests/test_clause_layer.py` (P47) phải **8/8 xanh** — đó là thứ duy nhất P47 chịu trách nhiệm.
- **`sent_intra._is_boundary` còn danh sách dấu câu hardcode** (`. ! ? … ; 。`) — trái §0.4.2.
  `clause_layer` dùng EOS-lift + PMI thay; nên trả nợ này khi nối Trụ A vào đường sinh.

---

# PHẦN B — P48 Trụ D (Hopfield phá thế hoà) trên v12

**Độc lập P47**, chạy song song được (không dựng W_skip, chỉ dùng `ParentBipartite.A`).

> ⚠️ **ĐÍNH CHÍNH 2026-08-02:** bản đầu tôi viết P48 *"nhẹ hơn nhiều"* — **sai**, đó là chưa
> nhân quy mô. Chi phí ≈ tuyến tính theo số cha × số query × số cấu hình. Đo mini (5 664 cha,
> 400 query): 2 cấu hình **18s** · 13 cấu hình **86s**. Ngoại suy v12 (×76 cha, ×2 query):
>
> | cấu hình | v12 ước tính |
> |---|---|
> | `--only A,E` (2) | **~45 phút** (gồm nạp) |
> | β sweep đầy đủ (13) | **~4.5 giờ** |
>
> ⇒ **Dùng `--only A,E` trên v12.** Nó trả lời trọn câu hỏi cần trả lời (E còn dương không);
> sweep β đầy đủ đã làm trên mini và **không cần lặp** ở v12.

## B.1 Đọc trước

`TK_045_MATRIX_CORE.md` **§13.9**. Tóm tắt: thay `rank_matches` bằng Hopfield **ÂM TÍNH**
(−7.5…−10pp rank-1); dạng sống sót là **phá thế hoà** (+2.25pp rank-1, MRR +0.053,
hoà 38.2%→0.0% trên mini).

## B.2 Smoke + chạy thật

```bash
python3 -m pytest tests/test_hopfield.py -q -m "not slow"      # kỳ vọng 6 passed

PYTHONPATH=src .venv/bin/python scripts/probe_hopfield_rank.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --n-query 800 --q-len 3 --only A,E \
    --out-json out/probe_hopfield_rank_v12.json
```

`--only A,E` giữ `A_rank_matches` (baseline luôn được chèn) + `E_tiebreak_s2_b32` (β=32 là
bản thắng mini). Đã kiểm: cho **đúng** số như chạy đủ 13 cấu hình, nhanh 4.8×.
Muốn xem lại `B/C/D` (đã âm tính trên mini) thì bỏ `--only` và chấp nhận ~4.5 giờ.

Bộ query **y hệt P31** (cùng `seed`/`seed+1`, cùng `_expected_rank`) ⇒ `A_rank_matches` phải
tái lập được baseline §12.126. **Nếu không khớp, dừng** — bộ query đã lệch, mọi Δ vô nghĩa.

RAM: nhẹ hơn P47 nhiều, nhưng vẫn phải nạp v12 (xem §2 phần A).

## B.3 Cổng G-hop (đã vá — dùng bản này)

```
ĐẠT  ⇔  Δrank-1 ≥ +5pp
     ∨  (Δtie ≥ 20% tương đối  ∧  Δrank-1 ≥ −0.5pp)
```

⚠️ Cổng **gốc** dùng OR trần và đã báo ĐẠT nhầm cho một cấu hình tụt 9.17pp — xem §13.9.
Vế hoà **chỉ hợp lệ trong cùng họ bộ chấm** (tức cấu hình `E_tiebreak`, vốn giữ điểm
`rank_matches` làm khoá chính). **Đừng** dùng vế hoà để bênh `B/C/D` — ở đó `tie=0` là hiện
vật của kiểu số.

## B.4 Cái cần xác nhận trên v12

| # | câu hỏi | mini | điều cần trả lời |
|---|---|---:|---|
| 1 | `E_tiebreak` có còn dương? | +2.25pp | dấu + độ lớn ở 430k cha |
| 2 | hoà đỉnh của `rank_matches` bao nhiêu? | 38.2% | nếu v12 **thấp hơn nhiều** thì trần lãi của E nhỏ đi tương ứng — báo cả hai số |
| 3 | β tốt nhất | 32 | v12 nhiều cha hơn ⇒ có thể cần β khác |
| 4 | `B/C/D` có còn âm? | −7.5…−10pp | nếu **dương** ở v12 thì kết luận §13.9(1) phải viết lại |

**Dự đoán ghi TRƯỚC:** hoà đỉnh ở v12 **thấp hơn** mini (nhiều cha ⇒ điểm coverage phân tán
hơn ⇒ ít trùng khít), nên lãi của E **nhỏ hơn +2.25pp**, có thể dưới ngưỡng. `B/C/D` vẫn âm.
Nếu số bác dự đoán này, ghi rõ.

## B.5 Nếu ĐẠT thì nối thế nào

Điểm nối ghi sẵn ở cuối `ParentStore.rank_matches`. `hca/hopfield.tiebreak_ranked(state, ...)`
nhận `state`. **`rank_matches` KHÔNG giữ tham chiếu `state`** — truyền từ call-site có state
(thêm tham số `tiebreak=` chẳng hạn). **Đừng bịa `self._state_ref`**: tôi đã thử, nó tạo lỗi
tiềm ẩn chỉ nổ khi bật cờ, và đã tự gỡ.

Sau khi nối: chạy lại `scripts/quality_baseline_v12.py` + `pytest -m "not slow"` để xác nhận
không đụng đường sinh.

## B.6 Báo cáo

Mục `### 12.x P48 — Hopfield phá thế hoà trên v12 (<ngày>) — M4`: bảng đầy đủ như §13.9 +
verdict cổng (bản đã vá) + dự đoán B.4 đúng/sai + đường dẫn JSON/log. Rồi cập nhật §13.9 và
bảng §13.4 hàng 2.

---

# PHẦN C — P57: cổng CẶP-ÂM-TIẾT trên v12

**Độc lập A và B.** Nặng RAM như PHẦN A (phải dựng tầng mệnh đề trên v12).

## C.1 Đọc trước

`TK_045_MATRIX_CORE.md` **§13.19 → §13.21**. Tóm tắt một đoạn:

Cổng phủ đang chạy khớp trên **fold ÂM TIẾT LẺ**. Tiếng Việt đơn âm tiết ⇒ `lục` khớp cả
*lục lạp* lẫn *lục bát*. Đo được: **86.4%** bằng chứng "đúng đề" là fold ≤4 ký tự, đứng đầu là
**chữ số**. Hệ quả: con số `lạc = 0.000` từng báo ba lần **là RỖNG** — cổng và thước dùng
**cùng một định nghĩa fold**. Dưới thước **nghiêm** (cặp âm tiết + bỏ chữ số), `lạc` thật của
cổng hiện tại là **0.966**.

Cổng **cặp âm tiết** đúng chiều (`pmi_cross` — thước ĐỘC LẬP, không dùng fold — lên
1.845 → 2.45–2.54) nhưng trên mini **chết đói**: 99–100% chuỗi tắc, 3/120 đầu ra chấm được.

**Giả thuyết cần bạn kiểm:** đây là bài toán **VẬT LIỆU**, không phải ngưỡng. v12 có
**1.76 triệu** mệnh đề vs mini **10 283** = **171×**.

## C.2 Smoke rồi chạy thật

```bash
python3 -m pytest tests/test_clause_layer.py -q -m "not slow"   # kỳ vọng 19 passed

PYTHONPATH=src .venv/bin/python scripts/probe_fold_gate.py \
    --checkpoint checkpoints/mini_sent_ep_qa.pkl --n-prompt 60 --dump 0 \
    --out-json out/probe_fold_gate_mini.json
```

Kỳ vọng mini (tái lập §13.21): `A` lạc ≈0.98 / chết ≈73% · `D` chết ≈95–99%, n≈0–3.
Probe sẽ **tự nói** rằng checkpoint nhỏ **không bác được** giả thuyết — đúng, đừng kết luận từ đó.

```bash
PYTHONPATH=src .venv/bin/python scripts/probe_fold_gate.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --n-prompt 120 --dump 4 \
    --out-json out/probe_fold_gate_v12.json
```

**Chi phí ước:** dựng tầng mệnh đề v12 ~**1 giờ** (P47 mất 3 765s cho 6 ngưỡng; đây chỉ 1
ngưỡng) + 4 cấu hình × 120 prompt soạn chuỗi. Tập ứng viên ở v12 **rất lớn** ⇒ nếu chậm quá
thì hạ `--n-prompt 60`, ghi rõ trong báo cáo. RAM: như PHẦN A, **cần ≥32GB**.

## C.3 CỔNG G-fold (đã nêu trước, KHÔNG đổi)

Cấu hình **`D_cap_bo_so`** phải đạt **CẢ BA**:

| # | điều kiện | mini để so |
|---|---|---|
| a | `dead_rate` ≤ **50%** | 99% |
| b | `lac_strict` ≤ **0.30** | (n quá nhỏ để tính) |
| c | `pmi_cross` ≥ `pmi_cross` của `A` | ✓ (2.45 vs 1.85) |

⚠️ `lac_strict` = **nan** nghĩa là **KHÔNG ĐỦ MẪU**, *không* phải "giữ đề hoàn hảo". Probe in
cảnh báo riêng cho ca này — đọc nó.

⚠️ Thước `lac_strict` **cố định** (cặp + bỏ chữ số) cho **mọi** cấu hình, tách khỏi định nghĩa
mà cổng dùng. Đừng "thống nhất" hai cái lại cho gọn — làm thế là tái lập đúng lỗi tự tại
§13.10 khiến `lạc = 0.000` trở nên vô nghĩa.

## C.4 DỰ ĐOÁN GHI TRƯỚC (Claude — để số BÁC được)

> Ở v12, `D_cap_bo_so` **sống được**: `dead_rate` xuống **20–40%** (mini 99%), `lac_strict`
> **≤ 0.30**, `pmi_cross` giữ ≥ 2.3. Lý do: §13.13 đo vật liệu co giãn **tuyến tính**, và
> 171× là rất nhiều.
>
> **Rủi ro ngược đã lường:** tập ứng viên lớn hơn cũng sinh nhiều **cặp trùng ngẫu nhiên** hơn
> ⇒ `lac_strict` có thể **không** xuống dưới 0.30 dù chuỗi sống. Nếu thấy `dead` ĐẠT mà `lạc`
> TRƯỢT thì kết luận là *"cặp âm tiết vẫn chưa đủ phân biệt ở quy mô lớn"* — cần đơn vị khớp
> cao hơn nữa (cụm ISG / nơ-ron cụm), không phải chỉnh ngưỡng.

## C.5 Bảng quyết định

| kết cục | kết luận | việc kế |
|---|---|---|
| **cả 3 ĐẠT** | giả thuyết VẬT LIỆU đúng; cổng cặp dùng được ở quy mô thật | đặt `pair_folds=True`+`drop_digits=True` làm mặc định `CoverageGate`; đo lại chấm mù vòng ba |
| **(a) ĐẠT, (b) TRƯỢT** | sống nhưng cặp âm tiết vẫn không đủ phân biệt | lên đơn vị khớp CAO HƠN (cụm ISG / `ParentStore.phrases`), **đừng** chỉnh ngưỡng |
| **(a) TRƯỢT** | ở **171×** vật liệu vẫn chết ⇒ **giả thuyết vật liệu BỊ BÁC** | đóng hướng cổng-fold; cơ chế khớp phải đổi hẳn (không phải fold) |
| thrash/OOM | giới hạn máy, **không phải kết quả** | máy ≥32GB; đừng ghi là âm tính |

## C.6 Báo cáo

Mục `### 12.x P57 — cổng cặp-âm-tiết trên v12 (<ngày>) — M4`: bảng 4 dòng đầy đủ
(`lac_strict` · `pmi_cross` · `dead_rate` · `n_tokens` · `n_scored`) + verdict G-fold + dự
đoán C.4 đúng/sai + **đọc tay 4 mẫu `--dump`** + đường dẫn JSON/log + RAM máy.
Rồi cập nhật §13.21 và bảng §13.4.

---

# PHẦN D — Xác nhận `E_tiebreak` (BẬT mặc định) không đụng ĐƯỜNG SINH ở v12

**Rẻ nhất trong bốn phần** — chạy chung lượt load v12 với PHẦN A hoặc C.

## D.1 Vì sao cần

§13.15 nối phá-thế-hoà Hopfield vào `rank_matches` và **bật mặc định** (quyết định tác giả sau
khi §12.151 xác nhận **+8.88pp** trên v12). Nhưng `rank_matches` là **đường truy hồi**, và
đường sinh đọc nó qua agenda MZ ⇒ **đầu ra sinh CÓ THỂ đổi**.

Baseline chất lượng §12.149 (`quality_pass=true`, 8 prompt) được đo **khi cơ chế này chưa tồn
tại**. Chưa ai kiểm nó còn đúng không. Trên mini tôi đã xác nhận suite xanh và `E` chồng lên
cho +0.00pp (idempotent), nhưng **mini không phải v12**.

## D.2 Chạy

```bash
# (1) BẬT — mã hiện tại
PYTHONPATH=src .venv/bin/python scripts/quality_baseline_v12.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --out-json out/quality_tiebreak_on.json --out-log out/quality_tiebreak_on.log

# (2) TẮT — tái lập điều kiện §12.149
PYTHONPATH=src .venv/bin/python scripts/quality_baseline_v12.py \
    --checkpoint checkpoints/network_v12_sent.pkl --no-tiebreak \
    --out-json out/quality_tiebreak_off.json --out-log out/quality_tiebreak_off.log
```

Cùng checkpoint, cùng seed, **chỉ khác một cờ**. Cờ `--no-tiebreak` đặt
`ps.hopfield_tiebreak_beta = 0.0` — thêm ở §13.15 đúng cho phép đo này.

## D.3 CỔNG G-tb (nêu TRƯỚC)

| # | điều kiện |
|---|---|
| a | **`quality_pass` = true** ở lượt BẬT — không được thủng baseline §12.149 |
| b | `pmi_mean` (BẬT) ≥ `pmi_mean` (TẮT) − 0.05 — không tụt bám đề |
| c | `salad_after_period_rate` (BẬT) ≤ (TẮT) — không tăng salad |

**Trượt (a)** ⇒ phá-thế-hoà **đụng đường sinh**: báo ngay, đặt
`HOPFIELD_TIEBREAK_BETA = 0.0` trong `core/parent_store.py` để lùi, rồi ghi §13.x.
Lãi **+8.88pp** là ở **truy hồi**; nếu nó phải trả bằng chất lượng **sinh** thì đó là đánh đổi
tác giả phải quyết, không phải mặc định im lặng.

## D.4 Dự đoán ghi TRƯỚC (Claude)

> **Đầu ra gần như không đổi.** Phá-thế-hoà chỉ xếp lại **trong nhóm điểm bằng nhau**; bậc
> điểm chính bất biến. Ở v12 hoà đỉnh là **50.1%** (§12.151) nên nó **có** chạm nhiều query —
> nhưng agenda MZ chỉ lấy vài cha đầu, và việc chọn đúng cha hơn thường **cải thiện** agenda.
> Dự đoán: `quality_pass` giữ **true**, `pmi_mean` **nhích lên** hoặc bằng.
>
> **Rủi ro ngược:** §12.149 đã chỉnh agenda/QA rất sát 8 prompt đó. Đổi thứ tự cha có thể làm
> vài prompt rơi khỏi đường đã tinh chỉnh ⇒ một-hai cổng con thủng dù truy hồi tốt hơn. Nếu
> thấy vậy thì **không** phải "tie-break hỏng" mà là **§12.149 quá khớp 8 prompt** — ghi rõ
> phân biệt đó.

## D.5 Báo cáo

Mục `### 12.x P58 — E_tiebreak trên đường sinh v12 (<ngày>) — M4`: hai `summary` JSON cạnh
nhau (BẬT/TẮT), verdict G-tb, dự đoán D.4 đúng/sai, và **đọc tay** 8 prompt xem đầu ra có đổi
gì đáng kể không. Rồi cập nhật §13.15.

---

# PHẦN E — P60: cổng PMI-cross thay cổng fold, trên v12

**Thay thế PHẦN C** (cổng fold đã đóng ở P57). Nặng RAM như A/C — chạy chung lượt load.

## E.1 Đọc trước

`TK_045_MATRIX_CORE.md` **§13.23**. Ba câu:

1. Hai cổng dựa trên **fold** đều hỏng — âm tiết lẻ quá lỏng, cặp quá thưa (P57 chết 94%).
2. **Đối chứng vàng** giải thích vì sao: fold-cặp chỉ khớp **5.2%** ngay cả giữa hai mệnh đề
   **liền kề trong cùng một cha** — tức đúng đề *theo định nghĩa*. Cổng đó đòi một sự kiện
   gần như không xảy ra. **Không phải thiếu vật liệu.**
3. ⛔ Kéo theo: **`lạc` cả hai bản NGHỈ HƯU** cho việc phán chủ đề — chúng đo *trùng lặp bề
   mặt*. Kể cả `lac_strict` mà PHẦN C dùng. **Đừng dùng `lạc` để kết luận gì ở phần này.**

## E.2 Mốc vàng — probe TỰ TÍNH trên v12

`gold` = `pmi_cross` trung bình giữa hai mệnh đề liền kề cùng cha, tính trên **chính
checkpoint đang chạy**. Nó là mức "đúng đề" mà corpus đó thể hiện ⇒ so được giữa checkpoint,
không phải hằng số ai đó bịa. Mini = **2.128**; v12 sẽ khác, probe in ra.

## E.3 Chạy

```bash
python3 -m pytest tests/test_clause_layer.py -q -m "not slow"   # kỳ vọng 19 passed

PYTHONPATH=src .venv/bin/python scripts/probe_pmi_gate.py \
    --checkpoint checkpoints/mini_sent_ep_qa.pkl --n-prompt 80 --dump 0 \
    --out-json out/probe_pmi_gate_mini.json          # smoke ~20s

PYTHONPATH=src .venv/bin/python scripts/probe_pmi_gate.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --n-prompt 120 --dump 4 \
    --out-json out/probe_pmi_gate_v12.json
```

Kỳ vọng smoke (mini): gold **2.128** · `pmi_1.75` **2.144 @ chết 52%** · fold 1.865 @ 71% ·
**G-pmi ĐẠT**. Chi phí v12 ≈ PHẦN C (~1h dựng tầng + 7 cấu hình × 120 prompt).

## E.4 CỔNG G-pmi (đã nêu trước)

Phải tồn tại **một θ** đạt **CẢ HAI** so với cổng fold đang chạy:

| # | điều kiện |
|---|---|
| a | `pmi_achieved` ≥ **gold − 0.05** — chạm mức đúng-đề của chính corpus |
| b | `dead_rate` ≤ `dead_rate(fold_dang_chay)` — không trả giá bằng tắc mạch |

Trượt ⇒ cổng PMI không chi phối được cổng fold ở quy mô thật ⇒ giữ hiện trạng, báo tác giả.

## E.5 DỰ ĐOÁN GHI TRƯỚC (Claude)

> **`gold` ở v12 THẤP HƠN mini** (~1.5–2.0): corpus v12 đa dạng hơn nhiều nên hai mệnh đề
> liền kề chia sẻ ít đồng-hiện hơn. Kéo theo **θ thắng cũng thấp hơn** — đoán **1.25–1.5**
> thay vì 1.75.
>
> **`dead_rate` THẤP HƠN mini ở cùng θ**: v12 có 1.76M mệnh đề (171×) ⇒ dễ tìm ứng viên vượt
> ngưỡng hơn. Đoán `pmi_1.5` chết **< 30%** (mini 39%).
>
> **G-pmi ĐẠT.** Cổng fold ở v12 chết **81%** (P57 §12.152) — mức rất dễ vượt.
>
> **Rủi ro ngược:** nếu `gold` v12 thấp thì điều kiện (a) dễ đạt một cách tầm thường, và
> "chạm mốc vàng" mất ý nghĩa. Nếu thấy `gold < 1.2` thì **báo lại**, đừng kết luận ĐẠT —
> mốc vàng lúc đó không phân biệt được gì và cần arbiter khác (chấm mù).

## E.6 Báo cáo

Mục `### 12.x P60 — cổng PMI-cross trên v12 (<ngày>) — M4`: **`gold` v12**, bảng 7 dòng
(`pmi` · `so mốc` · `chết%` · `tok`), verdict G-pmi, dự đoán E.5 đúng/sai, **đọc tay 4 mẫu**,
JSON/log, RAM. Rồi cập nhật §13.23 và §13.4.

**Nếu ĐẠT:** việc kế **không** phải bật mặc định ngay, mà là **chấm mù vòng ba**
(`clause_gated` cổng-fold vs cổng-PMI-θ-thắng) — arbiter duy nhất đã được kiểm. `pmi_cross`
là proxy, và ở đây nó vừa là **cổng** vừa là **thước** ⇒ tự tại, đúng cái bẫy §13.10.

---

# PHẦN F — P61: sinh cặp CHẤM MÙ vòng ba trên v12 (fold vs PMI)

**Rẻ** (~vài phút sau khi tầng mệnh đề đã dựng). Chạy chung lượt load với PHẦN E.

## F.1 Vì sao phải chấm mù, và vì sao phải là v12

P60 cho cổng PMI **chi phối** cổng fold trên v12. Nhưng ở đó `pmi_cross` vừa là **cổng** vừa
là **thước** ⇒ **tự tại** (bẫy §13.10). `lạc` cả hai bản đã **nghỉ hưu** (§13.23). ⇒ Arbiter
còn lại **duy nhất đã được kiểm** là **chấm mù**.

**Và mini KHÔNG thay được v12ở vòng này:** khác biệt then chốt là **độ dài**, mà nó rất khác
giữa hai quy mô:

| | fold | PMI θ=1.5 |
|---|---:|---:|
| mini | 34.7 tok | 30.1 tok (−13%) |
| **v12** | **26.1 tok** | **11.4 tok (−56%)** |

Ở v12 cổng PMI cho đầu ra **ngắn hơn một nửa**. "Ngắn" có thể đọc thành **cụt** hoặc thành
**gọn và đúng đề** — không metric nào phán được, người/AI đọc thì phán được. Chấm mù trên
mini sẽ **không** chạm tới hiện tượng này.

## F.2 Chạy

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_gate_blind_pairs.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --theta 1.25 --n-prompt 160 \
    --out-pairs out/gate_pairs_v12.txt --out-key out/gate_key_v12.json
```

θ=**1.25** (chết thấp nhất trong nhóm ĐẠT ở P60 — 42%). Muốn thêm điểm so sánh thì chạy lần
hai với `--theta 1.5` ra file khác; **đừng** gộp hai θ vào một bộ cặp.

Kỳ vọng ≥ **88** cặp (script tự cảnh báo nếu thiếu power). Script **chỉ sinh + trộn + giấu
nhãn**; **không tự chấm** (§12.111).

## F.3 Ai chấm

**Một phiên Claude riêng**, KHÔNG phải phiên đang chạy probe. Prompt giao việc: copy nguyên
mục F.4 dưới, đổi tên file.

⚠️ Người chấm **không được** đọc `out/gate_key_v12.json`, các khoá vòng trước, hay
`TK_045_MATRIX_CORE.md`.

## F.4 Rubric giao giám khảo (copy nguyên)

> Chấm mù 124+ cặp trong `out/gate_pairs_v12.txt`. Với từng cặp chọn **X / Y / hoà** theo:
> (a) đúng ngữ cảnh · (b) không salad · (c) văn phong trôi chảy. So sánh **trực tiếp** trong
> cặp, chọn "hoà" khi thật sự không phân biệt được.
>
> ⚠️ **Độ dài KHÔNG phải tiêu chí.** Hai bên khác nhau nhiều về độ dài. Đừng ưu ái bên dài;
> đừng phạt bên ngắn nếu nó dừng gọn và đúng đề. **Một câu ngắn đúng đề tốt hơn ba câu dài
> lạc đề.**
>
> ⚠️ **X/Y là VỊ TRÍ đã trộn**, không phải nhãn cấu hình — đừng suy luận thống kê ở mức
> X-vs-Y (vòng một đã có người nhầm chỗ này và kết luận ngược).
>
> Ghi `out/gate_verdict_v12.json`: `{"verdicts": {...}, "notes": "5-8 câu…"}`.

## F.5 Lật khoá & kết luận

```python
# đếm theo CẤU HÌNH, không phải theo vị trí — lỗi này đã xảy ra HAI lần (§13.19)
w = Counter()
for pid, v in verdicts.items():
    k = key[pid]
    w[k['left'] if v == 'X' else k['right'] if v == 'Y' else 'hoà'] += 1
```
Rồi `ruler.win_rate_ci(wins_pmi, wins_pmi + wins_fold)`; **tách khỏi 50%** thì mới kết luận.

**Nếu cổng PMI thắng** ⇒ đề xuất tác giả đổi mặc định `CoverageGate` sang PMI-θ.
**Nếu hoà** ⇒ hai cổng tương đương về chất lượng cảm nhận; chọn theo `dead_rate` (PMI 42% ≪
fold 81%) và ghi rõ đó là lý do **vận hành**, không phải chất lượng.
**Nếu cổng fold thắng** ⇒ P60 là dương tính GIẢ do tự tại; giữ fold, ghi §13.x.

---

# PHẦN G — P66: sinh cặp CHẤM MÙ vòng bảy trên v12 (mối nối `S1` vs **NEO-ĐỀ**)

> ⚠️ **G là P66, KHÔNG phải P61.** P61 = **PHẦN F**, chấm mù vòng ba, so hai **CỔNG**
> (`fold` vs PMI) — đã đóng, M4 chạy xong 02-08 (`011c89f`), kết luận ở §13.25/§13.27.
> PHẦN G so hai **BỘ XẾP MỐI NỐI** (`join_S1` vs `join_anchor`), **cùng** cổng fold.
> Cổng và bộ xếp là hai chỗ khác nhau trong `compose_chain` — đừng gộp hai vòng này.

**Giao M4.** Máy dev (Intel 8GB) đã thử và **thất bại**: sau 83 phút nó mới nạp xong
checkpoint rồi **swap**, 70 phút không chạm nổi cột mốc 50.000 cha đầu của
`build_clause_layer` (CPU 11%, pages free ~13 MB). Đây là việc của máy có RAM.

**Rẻ trên M4** — cùng dạng lượt chạy với PHẦN E/F: nạp một lần, dựng tầng mệnh đề, sinh cặp.

## G.1 Vì sao vòng này (đủ để chạy, không cần đọc chat)

`join_rescore` hiện **TẮT**. §13.30: bộ xếp `pmi_cross(A,B)` **thua** `S1` 59.7% dù AUC ngoại
tuyến .814 — nó tối ưu *mạch lạc cục bộ*, còn cái cần là *bám đề toàn cục*.

§13.31 đo tiếp và định vị được chỗ hỏng thật: **truy hồi không hỏng** (98.2% mệnh đề đúng tới
được qua `join_candidates`), **xếp hạng** mới hỏng — mệnh đề đúng nằm ở hạng trung vị
**51/125**, tức thứ tự ngẫu nhiên. Và cổng tiền-đăng ký của P65 **TRƯỢT** (không tín hiệu nào
đạt AUC .65 trên cả hai lớp âm bản) ⇒ hướng "điều kiện-PROMPT bằng pmi" đóng.

Ứng viên vòng này là `prompt_anchor_rescorer` — xếp ứng viên theo **Σ `gidf` của fold trùng
với ĐỀ**. Khác mọi lần trước ở đúng hai chỗ:

1. **XẾP, không CHẶN.** Mọi tiêu chí chặt trước đây đều làm **cổng** và chết đường (P57: 94%
   ngõ cụt). Bể ứng viên trung vị **~1.550** mệnh đề ⇒ làm **bộ xếp** thì chặt là *miễn phí*.
2. **Trọng số ĐỘ HIẾM, không ĐẾM.** §13.20: đếm thì rác (`nước`, `1`, `bằng`) chi phối.

`gidf` do `ParentStore` đếm sẵn ⇒ **Luật 0 sạch**, không có từ vựng cứng nào.

## G.2 Chạy

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_gate_blind_pairs.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --mode anchor --n-prompt 200 \
    --out-pairs out/anchor_pairs_v12.txt --out-key out/anchor_key_v12.json
```

Hai bên **cùng** cổng fold (đã thắng bốn vòng chấm mù); **chỉ khác** `join_rescore`:
`A = join_S1` (đang chạy, không rescore) · `B = join_anchor`. Script tự trộn trái/phải, giấu
nhãn, và **từ chối ghi đè** khoá khi đã có verdict — **đừng** dùng `--force`.

Smoke trước cho chắc (mini, ~1 phút):
```bash
PYTHONPATH=src .venv/bin/python scripts/gen_gate_blind_pairs.py \
    --checkpoint checkpoints/mini_sent_ep_qa.pkl --mode anchor --n-prompt 12 \
    --out-pairs out/anchor_pairs_smoke.txt --out-key out/anchor_key_smoke.json
```

## G.3 CỔNG G-anchor (nêu TRƯỚC khi chạy)

M4 **không** kết luận chất lượng ở phần này — chỉ **sinh cặp**. Nhưng lượt chạy chỉ **dùng
được** nếu cả ba đạt:

| # | điều kiện | vì sao |
|---|---|---|
| 1 | **≥ 88 cặp** trong `out/anchor_pairs_v12.txt` | ngưỡng power 80% cho +15pp (`min_pairs_for(15)`) |
| 2 | log **không** có `⚠️ … CHƯA đủ power` | script tự kiểm điều kiện 1 |
| 3 | tỷ lệ đề bị bỏ (do `a == b`) **< 70%** | nếu bộ xếp gần như không đổi được đầu ra ở v12 thì cơ chế **trơ** trên quy mô thật — §13.31 đo 38.3% đổi trên mini, số v12 phải báo lại |

**Nếu (1) trượt:** tăng `--n-prompt` lên 320 và ghi ra **tên file KHÁC** (`…_v12b`). Đừng ghi
đè.
**Nếu (3) trượt:** ĐỪNG chỉnh tham số để cứu. Báo lại số — "cơ chế trơ ở quy mô v12" tự nó là
kết quả, và nó bác luôn phép đo mini.

## G.4 Dự đoán ghi TRƯỚC (Claude — để số BÁC được)

- Tỷ lệ đổi đầu ra ở v12 **≥ 30%** (mini: 38.3%). Nếu v12 xuống **< 15%** ⇒ dự đoán của tôi
  **sai**, và nghĩa là bể v12 loãng tới mức trùng-fold-với-đề gần như không phân biệt được ai.
- Số cặp thu được từ 200 đề: **120–170**.
- Đầu ra bên `join_anchor` **không** ngắn đi đáng kể (đây là bộ xếp, không phải cổng — nó
  không thể gây ngõ cụt). Nếu độ dài tụt > 20% ⇒ có bug, **dừng và báo**, đừng giao chấm.
  (Smoke mini 12 đề đã chạy: script chạy thông, 3 cặp khác nhau, và bên `join_anchor` **dài
  hơn** ở cả 3 — nên mục độ dài này là **máy dò bug**, không phải tín hiệu chất lượng.)

## G.5 Báo cáo

Commit `out/anchor_pairs_v12.txt` + `out/anchor_key_v12.json`, kèm trong message:

- số cặp thu được / số đề đã thử;
- tỷ lệ đề bị bỏ vì `a == b`;
- độ dài token trung bình **mỗi bên** (để kiểm G.4 mục 3);
- thời gian chạy.

**KHÔNG chấm.** Chấm mù là **một phiên Claude riêng** theo `RUNBOOK_TK_045_P66.md` — phiên đó
bị cấm đọc `out/anchor_key_v12.json` và `TK_045_MATRIX_CORE.md`.

---

# PHẦN H — P69: đo **MỆNH ĐỀ NAM CHÂM RÁC** trên v12

**Giao M4.** Máy dev Intel 8GB không nạp nổi v12 (xem PHẦN G). Lượt này nặng gấp đôi PHẦN G:
sinh **hai** lần trên cùng bộ đề (bản thật + đối chứng xáo).

## H.1 Vì sao (đủ để chạy, không cần đọc chat)

§13.33 giám khảo mù lập bản kê 5 kiểu hỏng. §13.34 đo hai kiểu trong đó và tìm ra **một cơ chế
chung**:

- Corpus **KHÔNG** bẩn: 0,80% đơn vị dính dấu khuôn mẫu, và phần lớn số đó là bắt nhầm (ngoặc
  biên tập trong văn sử thật, danh sách nội dung thật, `{` `}` trong bài lập trình). Chỗ trống
  mẫu thư thật chỉ **~27 lần / 830 nghìn đơn vị**. ⇒ **Không lọc corpus.**
- Nhưng cụm `km 1661+100 QL1` — **hapax**, 1 lần duy nhất trong 829.513 đơn vị — xuất hiện ở
  **10/104 đề** khi sinh (**9,6%**), từ 10 đề không liên quan nhau. Dấu `*` khuếch đại **~9×**
  so với kỳ vọng.

**Chẩn:** mệnh đề đặc số/năm có fold là **rác §13.20** (`năm`, chữ số) nên **khớp với gần như
mọi đề**. Nó không thắng vì hiếm — nó thắng vì **khớp với mọi thứ**. Đây hợp nhất kiểu hỏng 2
(tan thành danh sách số) và kiểu 5 (rò rỉ khuôn mẫu) thành **một** cơ chế.

Nếu đúng, đây là mục tiêu can thiệp **khác hẳn** mọi thứ §13 đã thử: tấn công *cái bị chọn quá
nhiều*, không phải *cách chọn*. Và nó **Luật 0 sạch** — chỉ đếm.

## H.2 Chạy

```bash
PYTHONPATH=src .venv/bin/python scripts/probe_clause_magnet.py \
    --checkpoint checkpoints/network_v12_sent.pkl --n-prompt 400 \
    --out out/probe_clause_magnet_v12.json
```

Script tự chạy **hai** lượt trên **cùng** bộ đề, **cùng** cổng fold:
1. **thật** — cơ chế đang chạy;
2. **đối chứng** — `join_rescore` trả số ngẫu nhiên ⇒ xáo trật tự ứng viên trong cùng bể.

Đối chứng có mặt để **quy nhân**: tập trung ở **cả hai** ⇒ do bể/cổng (cái gì tới được); tập
trung **chỉ ở bản thật** ⇒ do **khâu xếp hạng**, đúng chẩn đoán nam châm.

Smoke mini đã chạy tại máy dev (`--n-prompt 20/40`, ~28s): script **thông**. ⚠️ Số của smoke
**vô nghĩa** — 103 lần phát trên 10.283 mệnh đề thì trùng lặp gần như không xảy ra dù cơ chế
có thật. Đừng trích số mini.

## H.3 CỔNG G-magnet (nêu TRƯỚC, đã ghi cứng trong script)

| # | điều kiện | vai trò |
|---|---|---|
| 1 | có **≥1 mệnh đề phát ở ≥5% số đề** | CHÍNH — nhân đôi quan sát §13.34 ở mức nửa (9,6% → 5%) |
| 2 | **top-10 mệnh đề chiếm ≥10%** tổng lần phát | tập trung có đáng kể không |
| 3 | tỷ số top-10 **thật / đối chứng ≥ 1,5×** | QUY NHÂN — do xếp hạng, không do bể |

**ĐẠT = cả ba.** Ngưỡng hiệu chỉnh từ quan sát **v12 thật**, không từ mini (bài học §13.32).

⚠️ **Phải chạy `--n-prompt` ≥ 400.** Ở n nhỏ, 5% = 1–2 đề nên cổng (1) **đạt tầm thường**, vô
nghĩa. Nếu vì lý do gì phải giảm n, báo lại và **đừng** đọc cổng (1).

## H.4 Dự đoán ghi TRƯỚC (Claude — để số BÁC được)

- Cổng **ĐẠT cả ba**. Cụ thể: đỉnh **≥8%** số đề (đã thấy 9,6% ở một cụm), top-10 **15–35%**
  tổng lần phát, tỷ số **2–4×**.
- Mệnh đề đứng đầu bảng **đặc số/năm** hoặc chứa rác kiểu `km 1661+100 QL1`, **không** phải câu
  văn xuôi bình thường.
- Đối chứng xáo cho tập trung **thấp hơn** bản thật. Nếu đối chứng **cao hơn** ⇒ tập trung là
  do **bể ứng viên**, chẩn đoán nam châm **sai**, và §13.34 phải viết lại.

## H.5 Báo cáo

Commit `out/probe_clause_magnet_v12.json` + log, kèm trong message:

- ba mục cổng ĐẠT/TRƯỢT và tỷ số thật/đối chứng;
- **8 mệnh đề đứng đầu** kèm văn bản đã giải mã và số đề của mỗi cái (script in sẵn) — phần
  này quan trọng ngang số: cần biết chúng **trông như thế nào**;
- thời gian chạy.

**Không** sửa cơ chế, **không** chỉnh tham số, **không** thêm bộ lọc — kể cả khi số xấu.
Can thiệp là quyết định sau, cần số trước.

---

# PHẦN I — P71: sinh cặp chấm mù vòng tám trên v12 — **ĐỀ BỎ TRỐNG TÍNH LÀ THUA**

**Giao M4.** Máy dev Intel 8GB không nạp nổi v12 (PHẦN G).

## I.1 Vì sao (đủ để chạy)

§13.36 đo được: chuỗi dừng sớm **không** vì đồ thị thưa mà vì **cổng phủ** — 63,3% số đề có
ứng viên nhưng bị loại sạch (87,7% do cổng), và **20,7% số đề không sinh được gì**. Bỏ cổng
thì 88% chạy hết `max_clauses`, độ dài 2,62 → 4,66.

Nhưng cổng fold là thứ **duy nhất** trong toàn §13 được chấm mù xác nhận (thắng 4 vòng). Mâu
thuẫn chỉ là bề ngoài: **mọi vòng chấm trước đều LOẠI cặp có bên rỗng** (`if not a or not b`)
⇒ giám khảo chưa bao giờ nhìn thấy cái giá của việc **không sinh được gì**.

Vòng này sửa đúng chỗ đó: **không loại bên trống**.

## I.2 Chạy

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_gate_blind_pairs.py \
    --checkpoint checkpoints/network_v12_sent.pkl --mode empty --n-prompt 600 \
    --out-pairs out/empty_pairs_v12.txt --out-key out/empty_key_v12.json
```

- **A = `fold`** — cổng fold hiện tại, **có thể trống**.
- **B = `fold_duphong`** — cổng fold, và **nếu rỗng thì soạn lại KHÔNG cổng**.

Hai bên **giống hệt nhau** ở mọi đề mà cổng sinh được ⇒ mọi cặp thu về đều là đề bị bỏ trống.
Bên trống được in rõ là `(TRỐNG — hệ không sinh được gì)`.

Smoke mini đã chạy tại máy dev (`--n-prompt 60` → 12 cặp, 20% — khớp 20,7% của §13.36):
script **thông**.

## I.3 CỔNG G-empty (nêu TRƯỚC)

| # | điều kiện | |
|---|---|---|
| 1 | **≥88 cặp** | power 80% cho +15pp |
| 2 | tỷ lệ đề bỏ trống ở v12 **trong khoảng 5–40%** | ngoài khoảng ⇒ v12 khác mini về BẢN CHẤT, báo lại trước khi chấm |

Nếu (1) thiếu: tăng `--n-prompt` lên 1000, ghi ra **tên file khác** (`_v12b`). Đừng ghi đè.

## I.4 Dự đoán ghi TRƯỚC (Claude)

- Tỷ lệ đề bỏ trống ở v12: **15–25%**.
- **TRỐNG thắng ≥55%.** Mẫu smoke cho thấy bản không-cổng là salad rõ (`chế độ mẫu quyền…
  Hán Vũ Đế… Lưu vào lịch điện thoại kèm nhắc trước một tuần`), và §13.30 cho thấy giám khảo
  **phạt rất nặng** việc bẻ gãy ngữ vực. Nếu **dự phòng** thắng ⇒ tôi sai, và im lặng là
  khuyết tật thật phải vá.
- Độ dài bên dự phòng ≈ 4,5 mệnh đề (bản không-cổng, §13.36).

## I.5 Báo cáo

Commit cặp + khoá, kèm: số cặp · **tỷ lệ đề bỏ trống** · độ dài trung bình bên dự phòng ·
thời gian. **Không chấm** — chấm là phiên Claude riêng theo `RUNBOOK_TK_045_P71.md`.

---

# PHẦN J — P74: sinh cặp chấm mù vòng chín trên v12 — **CHẶN BẢN LỀ ĐỔI VAI**

**Giao M4.** Máy dev Intel 8GB không nạp nổi v12 (PHẦN G).

## J.1 Vì sao (đủ để chạy)

Bệnh gốc §13.25 (bản lề đồng tự) đứng nguyên qua **tám** vòng và tới §13.39 mới có **số**.

Chữ ký thật đọc ra từ chính các ví dụ (`quốc gia` ⇒ `gia vị`, `tối đa` ⇒ `đa số`,
`tư duy` ⇒ `duy nhất`): bản lề là âm tiết **CUỐI** của một từ trong A, rồi mối nối đọc nó
thành âm tiết **ĐẦU** của một từ khác. Đếm được:

    word_pivot_strength = min( P(w_left ngay trước t) , P(w_right ngay sau t) )

| `min ≥ 0,05` | tỷ lệ mối nối |
|---|---:|
| **SINH** | **10,7%** |
| **THẬT** (người viết) | **0,0%** (không trừ self: 0,1%) |

Chênh ~**100 lần** ⇒ đây **không** phải hiện tượng của văn thật. Đây là con số cứng đầu tiên
cho bệnh gốc, và 10,7% mối nối là **đủ lớn để một vòng chấm mù phân giải** — khác hẳn 1,75%
của §13.38.

## J.2 Chạy

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_gate_blind_pairs.py \
    --checkpoint checkpoints/network_v12_sent.pkl --mode pivot --n-prompt 500 \
    --out-pairs out/pivot_pairs_v12.txt --out-key out/pivot_key_v12.json
```

- **A = `fold`** — cổng fold hiện tại.
- **B = `fold_chanbanle`** — cùng cổng fold, **cộng** `pivot_max=0.05`.

⚠️ Bên B **có thể rỗng** (chặn bớt ứng viên ⇒ có đề tắc hẳn), và mode này **KHÔNG giấu** chỗ
đó — giấu đi là lặp lại đúng sai lầm bảy vòng đầu (§13.37): thu lợi mà không tính giá. Bên
trống in rõ `(TRỐNG — hệ không sinh được gì)`, và rubric dạy giám khảo cách chấm nó.

Smoke mini (`--n-prompt 300`): **78 cặp = 26,0% đổi**, 11 cặp có bên trống — script thông.

## J.3 CỔNG G-pivot (nêu TRƯỚC)

| # | điều kiện | nếu trượt |
|---|---|---|
| 1 | **≥88 cặp** | tăng `--n-prompt` lên 900, ghi **tên file khác** |
| 2 | tỷ lệ đổi ∈ **10–45%** | ngoài khoảng ⇒ v12 khác mini về bản chất, **báo trước khi chấm** |
| 3 | tỷ lệ cặp có bên trống **< 35%** | vượt ⇒ vòng này thoái hoá thành vòng TRỐNG-vs-CHỮ mà §13.37 đã trả lời; báo lại, đừng chấm |

## J.4 Dự đoán ghi TRƯỚC (Claude)

- Tỷ lệ đổi ở v12: **15–35%** (mini 26,0%).
- **`fold_chanbanle` thắng ≥55%.** Đây là dự đoán **yếu**: tám vòng qua, 8/9 cải tiến "hiển
  nhiên đúng" đều thua hoặc hoà. Nhưng đây là ứng viên **duy nhất** nhắm vào một khuyết tật đã
  đo được ở mức **100× so với văn người viết** — chưa cơ chế nào trước đó có bằng chứng loại
  này. Nếu nó vẫn thua, kết luận rất mạnh: **đo được khuyết tật KHÔNG có nghĩa chữa nó là cải
  tiến**, và đó là bài học đáng hơn cơ chế.
- Tỷ lệ cặp có bên trống: tăng nhưng **≤ 20%**.
- ⚠️ Trong 10 ví dụ §13.39 có **≥2 cái ĐÚNG** (`nhà Lý` ⇒ `Lý Thái`, `người Việt` ⇒ `Việt Nam`)
  — bộ chặn giết cả chúng. Nếu giám khảo phàn nàn về **tên riêng bị cắt đôi**, đó là **cái giá
  đã lường trước**, không phải phát hiện mới.

## J.5 Báo cáo

Commit cặp + khoá, kèm: số cặp · tỷ lệ đổi · **tỷ lệ cặp có bên trống** · độ dài trung bình
mỗi bên · thời gian. **Không chấm** — chấm là phiên Claude riêng theo `RUNBOOK_TK_045_P74.md`.

---

# PHẦN K — P75: chấm mù vòng mười trên v12 — **DỪNG SỚM** (`max_clauses` 5 vs 2)

**Giao M4.** Máy dev Intel 8GB không nạp nổi v12 (PHẦN G).

## K.1 Vì sao (đủ để chạy)

§13.40 (chấm mù vòng chín) hoà, nhưng phân rã theo **hình dạng cặp** cho một thứ khác hẳn:

| nhóm | n | bên bị **cắt ngắn** thắng | CI95 |
|---|---:|---:|---|
| một bên là **tiền tố** của bên kia | 25 | **84,0%** | [65,3%, 93,6%] |
| **rẽ nhánh thật** | 76 | 51,3% | [40,3%, 62,2%] |

⇒ Giả thuyết: **mỗi mệnh đề nối thêm, trung bình, LÀM HẠI** — lợi ích của mọi cơ chế "cắt bớt"
đến từ chỗ nó **dừng**, không từ chỗ nó **chọn khéo**.

⚠️ Phân rã đó là **hậu nghiệm**. Vòng này là vòng **nêu trước** để xác nhận hoặc bác.

Vòng này **không có cơ chế mới nào**: cùng cổng fold, cùng rng, cùng mọi thứ — khác đúng **một
tham số** `max_clauses`. Hai bên vì thế chọn y hệt nhau ở từng bước, và bên ngắn **luôn** là
tiền tố nghiêm ngặt của bên dài ⇒ cô lập đúng câu hỏi *"nối thêm có lợi không"*.

## K.2 Chạy

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_gate_blind_pairs.py \
    --checkpoint checkpoints/network_v12_sent.pkl --mode len --short 2 --n-prompt 400 \
    --out-pairs out/len_pairs_v12.txt --out-key out/len_key_v12.json
```

- **A = `len5`** — `max_clauses=5`, mặc định hiện tại.
- **B = `len2`** — `max_clauses=2`.

Smoke mini (`--n-prompt 300`): **148 cặp = 49,3% đổi**, và **148/148 là tiền tố nghiêm ngặt** —
script thông, thiết kế đúng.

## K.3 CỔNG G-len (nêu TRƯỚC)

| # | điều kiện | nếu trượt |
|---|---|---|
| 1 | **≥88 cặp** | tăng `--n-prompt` lên 800, ghi **tên file khác** |
| 2 | tỷ lệ đổi ∈ **25–70%** | ngoài khoảng ⇒ báo trước khi chấm |
| 3 | **100% cặp là tiền tố nghiêm ngặt** | **DỪNG NGAY** — nếu có cặp rẽ nhánh thì hai bên đang khác nhau ở thứ **ngoài** `max_clauses`, tức có bug; báo, đừng chấm |

Mục (3) là **máy dò bug**, không phải tiêu chí chất lượng. Hai bên dùng cùng seed nên bất kỳ
sai lệch nào cũng là dấu hiệu tham số rò rỉ chỗ khác.

## K.4 Dự đoán ghi TRƯỚC (Claude)

- **`len2` thắng 60–75%.** Thấp hơn 84% hậu nghiệm, vì nhóm 25 cặp đó là chỗ bộ chặn cắt —
  nhiều khả năng đúng những chỗ đoạn nối thêm tệ nhất, tức mẫu đã thiên.
- Tỷ lệ đổi ở v12: **35–60%** (mini 49,3%).
- **Phản bác đã có sẵn câu trả lời:** *"giám khảo chỉ thích bên ngắn"* — **không đúng**, §13.25
  đã bác: ở vòng ba, cổng PMI cho đầu ra **ngắn hơn một nửa** (11,4 vs 26,1 token) và nó
  **THUA 85,6%**. Giám khảo không thưởng cho ngắn; họ phạt cho **cụt**.
- Nếu **hoà/thua** ⇒ con số 84% của §13.40 là **hiện vật**, giả thuyết "nối thêm làm hại"
  **chết**, và §13.40 phải ghi lại.

## K.5 Báo cáo

Commit cặp + khoá, kèm: số cặp · tỷ lệ đổi · **tỷ lệ cặp là tiền tố nghiêm ngặt** (phải 100%) ·
độ dài trung bình mỗi bên · thời gian. **Không chấm** — chấm là phiên Claude riêng theo
`RUNBOOK_TK_045_P75.md`.

---

# PHẦN L — P76: chấm mù vòng mười một trên v12 — **`len2` vs DỪNG KHI ĐÓNG ĐƯỢC CÂU**

**Giao M4.**

## L.1 Vì sao (đủ để chạy)

§13.41: `max_clauses=2` **thắng 78,7%** (CI95 [71,4%, 86,0%]) trước mặc định `=5`. Cơ chế
**không** phải "đuôi tệ hơn" — tỷ lệ mối nối hỏng **phẳng** theo vị trí (10,2% · 12,7% · 9,8%
· 10,4%). Chuỗi dài thua vì **TÍCH LUỸ**: 4 mối nối ⇒ ~34% hỏng, 1 mối nối ⇒ ~10%.

Nhưng `len2` là **hằng số mù**, và nó có giá đo được: **37% điểm dừng rơi giữa câu**, giám khảo
phạt 8 ca cụt rõ (`dừng ở 'hoặc các'`, `'và nếu'`, `'một cách'`, `'bằng'`).

Ứng viên: **dừng ở mệnh đề ĐÓNG ĐƯỢC CÂU đầu tiên** (EOS-lift — tín hiệu §0.4.2 đã có sẵn
trong `BoundarySignal`), sàn `min_clauses=2`, trần 5. Ý đồ: giữ cái thắng của `len2` (ít mối
nối), bỏ cái giá của nó (cụt).

⚠️ **Mốc so sánh là `len2`, KHÔNG phải `len5`.** So với `len5` nữa là so với thứ đã thua —
đúng cái bẫy làm một cải tiến trông đẹp mà thật ra chỉ lặp lại kết quả cũ.

## L.2 Chạy

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_gate_blind_pairs.py \
    --checkpoint checkpoints/network_v12_sent.pkl --mode close --short 2 --n-prompt 700 \
    --out-pairs out/close_pairs_v12.txt --out-key out/close_key_v12.json
```

- **A = `len2`** — mốc mới (thắng vòng mười).
- **B = `dong_cau`** — sàn 2 mệnh đề, đi tiếp **cho tới khi** một mệnh đề đóng được câu, trần 5.

Smoke mini (`--n-prompt 400`): **81 cặp = 20,2% đổi**, tiền tố **81/81** — script thông.

## L.3 CỔNG G-close (nêu TRƯỚC)

| # | điều kiện | nếu trượt |
|---|---|---|
| 1 | **≥88 cặp** | tăng `--n-prompt` lên 1400, ghi **tên file khác** |
| 2 | tỷ lệ đổi ∈ **10–45%** | ngoài khoảng ⇒ báo trước khi chấm |
| 3 | **100% cặp là tiền tố nghiêm ngặt** | **DỪNG NGAY** — máy dò bug, hai bên chỉ được khác nhau ở điểm dừng |

## L.4 Dự đoán ghi TRƯỚC (Claude)

- **`dong_cau` thắng 55–68%.** Đây là dự đoán **thật sự không chắc**, và tôi nói rõ vì sao:
  cơ chế **được** một thứ (bỏ cụt, ~37% ca) nhưng **trả** một thứ (mỗi mệnh đề đi thêm mang
  ~10% rủi ro mối nối hỏng — §13.41). Hai chiều ngược nhau, và tôi **không** biết chiều nào
  lớn hơn.
- Tỷ lệ đổi ở v12: **12–35%** (mini 20,2%).
- Độ dài bên `dong_cau` ≈ **1,3–1,8×** bên `len2`.
- **Nếu HOÀ** ⇒ kết luận sạch: *đóng được câu KHÔNG đáng giá bằng một mối nối tăng thêm*, và
  `len2` giữ nguyên là điểm vận hành. Đó là kết quả **hữu ích**, không phải thất bại.

## L.5 Báo cáo

Commit cặp + khoá, kèm: số cặp · tỷ lệ đổi · tỷ lệ tiền tố (phải 100%) · độ dài trung bình mỗi
bên · **số mệnh đề trung bình mỗi bên** · thời gian. **Không chấm** — phiên Claude riêng theo
`RUNBOOK_TK_045_P76.md`.
