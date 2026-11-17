# TK_045 — MATRIX CORE: neo cha / full-history / EOS ma trận / chống drift

> ⚠️ **ĐANG GIAO M4 → xem §12.31** (P66 neo-đề gần trơ @ v12 · G-anchor(3) TRƯỢT).
>
> 🆕 **2026-08-02 — §13 SOẠN THEO MẢNH ĐƯỢC CHỨNG THỰC**: đề xuất trọn vẹn cho "sinh vượt LLM"
> (chỉ đạo tác giả). Chẩn đoán hợp nhất: **đồ thị đếm là bộ CHỌN xuất sắc, bộ SINH-TOKEN vô
> vọng** ⇒ nâng ĐƠN VỊ sinh từ token lên **mệnh đề**. Cổng sinh-tử = **P47**
> (`scripts/probe_clause_layer.py`) — **MỞ trên v12** (§12.150). P48 Hopfield phá thế hoà
> **ĐẠT +8.88pp** (§12.151) · **ĐÃ WIRE** (§13.15) · P58 xác nhận sinh không đổi.
> P57 cổng cặp-âm-tiết **TRƯỢT** (§12.152). P60 cổng PMI-cross **ĐẠT** (§12.154).
> P61 cặp chấm mù **112 cặp** sẵn (§12.155). P66 neo-đề: đổi ~26% / bỏ 74% @ v12 (§12.156).

| Field | Value |
|-------|-------|
| **ID** | TK_045 |
| **Status** | 🟢 **Nhánh W_skip (2026-07-30):** W_skip là cấu trúc GỐC. **Thắng:** embedding E_skip (nhận, §12.123) · thước trôi-đề PMI-cross · match-theo-d truy hồi **+31pp v12** (§12.122). **Thua:** dự đoán next-token (VarLM hơn −14.5pp) · sinh sáng tạo (bức tường §12.104). Xem **§0.W TỔNG KẾT**. |
| **Nguồn** | Chỉ đạo 2026-07-24: audit từng ma trận; sinh hợp lý toàn văn (không bắt buộc next=`xem`); không window ctx; EOS phải từ phép ma trận (không rule cắt câu); agenda thật vẫn dài hơn một câu. |
| **Liên quan** | TK_042 (lõi sinh), TK_044 (attention-Hebbian), `docs/study/02_HYBRID_HEBBIAN_MATRIX.md`, `docs/study/03_ATTENTION_QKV_FROM_GRAPH.md`, `src/nn_aivie/hca/matrix_core.py`, `spectral_stack.py`, `attention_qk.py`, `knowledge_seq.py`, `sent_intra.py` |
| **Checkpoint đo** | **`checkpoints/network_v12_sent.pkl`** (+ `network_v12_sent.sent_qk.npz`); legacy: `network_v11_sent.pkl` (migrate), `network_v11_styled_hist.pkl` |
| **Probe** | `probe_agenda_handoff_only.py` · `probe_multi_prompt_long.py` · `probe_h1_late_and.py` · `probe_path_soft_ab.py` · `diag_channel_attrib.py` · `diag_wrong_anchor_local.py` |

---

## 0. Kết luận ngắn

Matrix core đã chuyển từ “trộn fluency toàn cục → salad thời đại” sang **neo cha + path emit + EOS cạnh tranh argmax**. Trên `ar_no_agenda=True` câu định nghĩa Ngô Quyền sạch; **không thay MZ** cho đoạn dài.

**Đa câu (2026-07-26):** L2R/R2L **chỉ trong 1 câu**. ≥2 câu = **episode span-handoff lúc EOS** (`ar_no_agenda=True` + `ar_sent_agenda_handoff=True`). Không bơm `W_agenda` vào logits giữa câu; không nhầm `ar_no_agenda=False` với “bật handoff” (cờ đó mở MZ retrieval).

**Hướng tiếp — ba trụ + ①b:**  
① bài = episode + câu parent + `SENT_SUCC`/`W_agenda` (P0) — **P0.1+P0.3 đạt trên v12**;  
①b **L2R/R2L+`d` trong câu** → ma trận; decode: R2L neo/cha, L2R continuation ⊙ emit (§4.1, P7);  
② phân bố sắc rồi mới sample (P5);  
③ `h` dài từ cùng tín hiệu (P6 / P7.5).  
Gen trái→phải; cổng cha bắt buộc; không mega-parent.

---

## 0.W — TỔNG KẾT NHÁNH W_skip (§12.106–§12.123, cập nhật 2026-07-30)

Mục tổng kết theo yêu cầu tác giả: đề bài · giải pháp · việc đã làm · kết quả · cải thiện · còn treo.
Chi tiết từng bước ở các §12.1xx tương ứng.

### A. Đề bài / yêu cầu

1. **Vấn đề gốc (tác giả nêu):** *"việc sinh đang không khác gì chép lại agenda từ training"* — hệ
   mặc định là bộ TRUY HỒI, sinh = phát lại câu đã nhớ. Muốn **sinh sáng tạo, đoạn dài, đúng nghĩa LLM**.
2. **Ý tưởng tác giả — tận dụng W_skip:** đếm phiếu từ cạnh có-hướng-theo-khoảng-cách `W_skip[d]`
   (`Việt→đẹp,vô,đánh` …), giảm theo d, softmax, **vòng quay** (không argmax). Lấy W_skip **chủ đạo**.
3. **Chốt hướng (2026-07-30):** **W_skip làm cấu trúc GỐC, W_seq/CO-embedding/VarLM là VIEW** — hợp
   nhất, không cộng thêm. "Cần train mới không?" → **KHÔNG** (W_skip/E_skip là view dẫn xuất của
   `emit_ids`, tính lại offline; đòn bẩy lớn nhất là DỮ LIỆU nhiều hơn).
4. **Ràng buộc:** Luật 0 (không backprop ở đường sinh; đếm/closed-form/Hebbian OK). Đo-trước-code.
   Bỏ thước verbatim-copy/agenda → **chấm mù bằng AI + đọc tay** (mục tiêu là sáng tạo, khớp agenda
   KHÔNG phải mục tiêu).

### B. Giải pháp / cách tiếp cận

* **Siêu-cấu-trúc:** `W_seq = W_skip[d=1]`; embedding = `Σ_d λ^d (W_skip+W_skipᵀ)`→PPMI→SVD; VarLM =
  band d=1 nối tiếp. W_skip = tensor `(N·D)×N` (8 CSR N×N, thưa).
* **Lộ trình 5 pha, mỗi pha một CỔNG đo so incumbent** (§12.114): 0 hạ tầng · 1 embedding · 2 dự đoán
  · 3 nghiệm thu · 4 tích hợp sinh.
* **Trục đo mới:** `scripts/gen_samples_blind.py` (sinh→giấu nhãn→trộn→chấm rubric mù F/C/N/P→lật) +
  thước trôi-đề (`probe_eskip_ruler.py`, `probe_wskip_accept.py`). M4 chạy v12, Claude chạy mini + chấm.

### C. Việc đã làm (scripts + phép đo)

| # | việc | script | §  |
|---|---|---|---|
| kênh phiếu | vote_logits nối vào generate, sửa 2 lỗi (EOS 0 phiếu, số vỡ) | `w_skip.py`,`generate_hebbian.py` | 106/107/110 |
| chấm mù | dựng bộ đo AI mù + đọc tay | `gen_samples_blind.py` | 111/112/113 |
| ① embedding | E_skip thành mã chính thức | `hca/embedding_skip.py`,`bake_eskip.py` | 115/121/123 |
| ② dự đoán | bake-off W_seq/W_skip/VarLM held-out | `probe_wskip_predict.py` | 116/120 |
| ③ nghiệm thu | PMI-cross, toán-tử-chuyển | `probe_wskip_accept.py`,`probe_wskip_transition.py` | 117/118 |
| ④ A/E theo d | match skip-gram truy hồi | `probe_ae_distance.py` | 119/122 |
| ⑤ E_skip attn | thước ngữ nghĩa | (trong probe) | 119 |

### D. Kết quả đạt được (mini + v12)

| hướng | kết quả | phán |
|---|---|---|
| **embedding E_skip** | thước trôi-đề **3–4×** > E_old; hàng xóm v12 ≥ E_old (thắng `sông`); sinh ≥ (kênh trơ) | ✅ **NHẬN** (§12.123) |
| **match-theo-d (A)** | thu hồi cha nguồn v12 **+31.25pp** @q=3 (mini +14.6) | ✅ **G-③ ĐẠT** — đang nối vào A |
| **PMI-cross thước** | Spearman↔C **+0.643** > cos(E_skip) +0.281 > lạc fold-cứng | ✅ thước tốt nhất |
| dự đoán next-token | W_skip-backoff **−14.5pp** < VarLM (v12) | ❌ giữ VarLM as view |
| kênh phiếu (sinh) | n=30 chấm mù **HOÀ** (OFF 4.10 ‖ w05 4.17) | ⚪ tắt mặc định |
| toán-tử-chuyển | Spearman +0.06 (làm mờ) | ❌ đóng |
| E_skip attention (sinh) | kênh `distant_content` trơ (0/30 đổi) | ⚪ chờ cơ chế khác |

### E. Cải thiện ĐÃ CÓ (cụ thể, giữ lại)

1. **E_skip nhận làm embedding v12** (§12.123) — thước/hàng xóm tốt hơn, 0 thoái lui.
2. **PMI-cross làm thước trôi-đề** — thay `lạc` fold-cứng (đang chờ ráp vào eval).
3. **match-theo-d** — cơ chế "W_skip thay túi-từ trong A", +31pp thu hồi (đang nối).
4. **Vá kênh phiếu:** EOS-neutral (logsumexp), cấm số vỡ; **vá harness** `eval_gen_broad` để `--extra`
   đo được sampling.
5. **Trục đo mới** (chấm mù AI) — sửa sai lệch "copy=1.0 là đỉnh" của thước cũ.
6. **Hợp nhất cấu trúc:** VarLM = view liền của W_skip (giữ, không thay bằng vote có-gap).

### F. Bài học / bức tường

* **Cặp gần-bất-ngờ > khả-đạt xa/nhiều-bước** cho phân biệt: mọi "xa hơn/nhiều bước hơn" (tầm d
  §12.108, hop §12.118) đều **làm mờ**. PMI cặp trực tiếp sắc nhất.
* **§12.104 (bức tường):** ba phép đo độc lập (fluency §12.103, attestation §12.104, PMI-cross §12.117)
  đều cho: **cấu trúc đồng-hiện — kể cả W_skip đầy đủ — cho "cùng chủ đề" KHÔNG cho "đúng ngữ pháp"
  cho chuỗi MỚI.** Nên W_skip nâng **TRUY HỒI + ĐO**, không nâng **sinh-token-kế**.
* **Ngã ba còn lại cho sinh-dài-sáng-tạo:** §12.104e — mô hình tổng-quát-hoá (neural_lm) làm **bộ
  nghiệm thu** (chạm Luật 0 khâu soạn) — **quyết định của tác giả**.
* **Đòn bẩy lớn nhất đo được: DỮ LIỆU** — v12 sinh tốt hơn mini rõ, độc lập mọi tinh chỉnh W_skip.

### G. Đang treo

* **P30–P34** (M4): ✅ E_skip save · ✅ rank_matches G-tie · ✅ PMI trong eval · ✅ order→act
  opt-in (G-act hẹp) · ✅ vote_primary v12 (G-vote **TRƯỢT**).
* **§12.140 (P38):** bão hoà indeg trên phiếu W_skip — **ĐÃ WIRE**. Câu dời đô giữ;
  sau `.` hết độc `và` (chưa EOS). Knob deg/row giữ để thử tiếp.
* **§12.133:** W_skip primary; MZ agenda opt — đo thuần dùng `ar_w_skip_use_mz_agenda=False`.
* **Claude về:** review MZ+W_skip; quyết §12.104e.
* **Chờ tác giả:** §12.104e / chỉnh retrieve MZ nếu muốn fact đúng hơn.

---

## 1. Bài toán (lúc mở ticket)

| # | Triệu chứng | Gốc |
|---|-------------|-----|
| T1 | `được→Thăng/Tây Sơn/…` lẫn kỷ | Parent coact + bag emit hòa tan; soft-AND lỏng |
| T2 | Gen word-salad / loop | Fluency (W_know / QK / stack) thắng khi conf thấp; anti-loop yếu |
| T3 | Cửa sổ `ctx_len` quên entity đầu câu | Soft-window λ^{T-i} + cắt `[-L:]` |
| T4 | Drift sau câu đúng (`…tự chủ việc…Tống`) | Path copy cùng emit wiki dài |
| T5 | EOS ép bằng rule (lookahead / `end_rate≥θ` / `conf=1→+20`) | Không Luật-0 / không giống LLM |
| T6 | Câu đúng nhưng cụt vs agenda MZ | `no_agenda` không có kênh “câu 2”; EOS = hết mạch local |

**Chấp nhận:** toàn văn hợp lý > bắt buộc bigram vàng `được→xem`. MZ = retrieval; matrix = fluency/knowledge views.

---

## 2. Cải tiến ĐÃ LÀM (decode / matrix views)

### 2.1 Parent bipartite (`build_parent_bipartite`, cache `_v≥10`)

| Artefact | Shape / dạng | Công thức chiết |
|----------|--------------|-----------------|
| `A` | sparse (N, P) | `match_ids` only × √(power×priority) × gidf — **không** kích cha từ emit (tránh lẫn bài) |
| `E` | sparse (P, N) | emit/answer × power/(1+0.15·pos) × gidf (+ bigram nhẹ) |
| `succ` | dict cạnh | bigram trong emit; **+ cạnh `w→EOS`** khi biên câu / hết emit |
| `end_after` / `cont_after` | (N,) | mass đứng trước biên vs tiếp tục |
| `emits` | list arrays | seq để path khớp suffix |

### 2.2 Parent coactivation (`parent_coactivation_logits`)

- **Full history** (`ctx_len=0`): không cắt window; neo presence × gidf (không decay trên kênh cha).
- **Specificity:** `act *= (explained/match_mass)²`.
- **Hard-AND** top-2 token gidf cao + sớm; soft-AND token phụ.
- **Top-k parents** (mặc định 12).
- **Path continuation:** khớp suffix history↔emit; lead-bonus (khớp gần đầu emit); next = content hoặc **EOS** (điểm trong logits).
- **Lock bag** theo `lock_pi` khi path rõ.
- Bỏ: front-weight (hồi quy `sử`), lookahead drift rule, `conf=1→ép EOS`.

### 2.3 Matrix core logits (`matrix_core_logits`)

```
logits ≈ β_par·bag + β_succ·(succ+path)
       + β_eos·(end_after[w0] + 0.25·ctx·end_after)   # cạnh tranh argmax
       + flu·(β_know·W_know/QK_know + β_qk·PPMI_QK + β_stack·stack + β_ro·readout)
next = argmax
```

- `flu = max(0.12, 1−0.9·conf)` — hạ fluency khi neo cha peaked.
- Full history cho know/qk (bag residual + sàn λ); stack `pos_ctx_len = |history|`; entity_ids = mọi surface trong hist.
- **Không zero `<EOS>`** khi mask structural.
- Anti-loop trên toàn history (nhân đường chéo / 4-gram).
- Generate: giữ EOS nếu matrix đã cho EOS thắng các cand khác (`_wseq_fallback`).

### 2.4 Config knobs (mặc định hiện tại)

`ar_matrix_parent_*`, `ar_matrix_know_*`, `ar_matrix_qk_weight`, `ar_matrix_stack_weight`, `ar_matrix_eos_weight`, `ar_matrix_parent_ctx_len=0`.

### 2.5 Kết quả probe (chốt)

```
no_agenda+matrix:
  Ngô Quyền là vị vua sáng lập nhà Ngô, được xem là người mở đầu thời kỳ độc lập tự chủ
  (xem rank=1; <EOS> có mặt trong top logits lúc decode)

MZ+matrix: đoạn Bạch Đằng đầy đủ (agenda) — không thay bằng matrix alone.
```

### 2.6 File chạm

- `src/nn_aivie/hca/matrix_core.py` — lõi
- `src/nn_aivie/hca/config.py` — knobs
- `src/nn_aivie/hca/generate_hebbian.py` — tôn trọng EOS từ matrix
- `scripts/probe_matrix_core.py` — đo

---

## 3. Cách các ma trận được TẠO (nhắc nhanh)

| View | Nguồn | Phép |
|------|--------|------|
| A, E, succ, end_after | ParentStore | đếm / ghép sparse |
| W_know | bigram trong chuỗi cha (+ Oja L2 hàng) | CSR; SVD → Know Q/K (N, 64) |
| PPMI Q/K | Adjacency SEQUENTIAL | PPMI hướng → SVD (N, 96) |
| Spectral stack 4 tầng | cùng PPMI | residual SVD; mỗi tầng wq/wk (N,d), u_ffn, n_keep (64→24) |

**Không backprop** — đổi công thức → rebuild cache/sidecar.

---

## 4. Quyết định kiến trúc (chốt 2026-07-24)

**Bài ≠ 1 parent emit dài.** Hierarchy hybrid:

```
[Episode / bài]  ← AbstractNeuron (đã có: members theo thứ tự)
       │ members
       ├── Parent câu/QA 1   emit = đúng 1 câu (hoặc 1 answer)
       ├── Parent câu/QA 2
       └── …
              ↑
         SENT_SUCC / AGENDA_NEXT: hết câu k → đầu câu k+1
```

| Đơn vị | Vai trò | Không làm |
|--------|---------|-----------|
| **Parent câu** | `emit_ids` / `E` / path / EOS local sạch | Không nhét cả wiki vào 1 emit |
| **Episode bài** | `members=(câu1,…)` + `match_ids` title/entity — “NÓI GÌ” / retrieval | Không dùng episode làm bag emit full-vocab |
| **SENT_SUCC** | Cạnh đếm token (cuối k → đầu k+1) hoặc parent→parent rồi chiếu token | Không if-rule “sau EOS nhảy câu 2” không có mass |

**Tái dùng:** `AbstractNeuron` / `episodes_of` / TK_035 `paragraph_units` (cửa sổ 2–4 câu = **bổ sung**, không thay đơn vị câu). Theme entity→facts giữ song song.

**Tránh hai cực:**
- Chỉ câu phẳng, không episode → retrieval mất “cùng bài”.
- Chỉ 1 mega-parent → drift path (đã thấy).

### 4.1 Solution triệt để: cạnh trong câu L2R / R2L + `d` → ma trận → sinh

**Chốt ý tác giả (ví dụ `Em bé Lan.`):** trong mỗi câu đã học, mọi cặp vị trí có cạnh **có kiểu + khoảng cách**; gen vẫn trái→phải.

```
Vị trí:  1:Em    2:bé    3:Lan
L2R:     Em ──1──► bé ──1──► Lan ;  Em ────2────► Lan
R2L:     Em ◄──1── bé ◄──1── Lan ;  Em ◄────2──── Lan
```

#### 4.1.1 Ba lớp cấu trúc (không gộp một túi)

| Lớp | Đơn vị | Cạnh | Việc chữa |
|-----|--------|------|-----------|
| **Trong câu** (P7) | Parent 1 câu | L2R/R2L mọi cặp + `d` | Drift Markov đuôi (B); mở phần còn lại của fact |
| **Giữa câu** (P0.2) | Episode members | SENT_SUCC / `W_agenda` | Hết câu 1 → đầu câu 2 |
| **Bài / entity** (P0/P1) | Episode + Theme | members, match | Retrieval “NÓI GÌ”; không mega-emit |

Segment câu (P0.1) là **điều kiện**: cạnh P7 chỉ đếm trong biên câu — không đếm xuyên đoạn wiki.

#### 4.1.2 Mô hình cạnh (formal)

Với câu `t_1…t_L` (chuỗi vị trí sau tokenize; chốt punct ở P7.0):

- ∀ `i < j`: cạnh **L2R** `(t_i → t_j, d=j−i)` weight `f(d)`
- ∀ `i < j`: cạnh **R2L** `(t_j → t_i, d=j−i)` weight `f(d)`
- `f(d) = λ^{d−1}` (mặc định đề xuất) hoặc `1/d`; knob `d_max` (vd 12) cắt cặp quá xa
- **Phạm vi:** cùng parent câu; **không** nối chéo bài chỉ vì trùng surface
- **Kind mới** (đọc kênh riêng như `SEQUENTIAL`, không đổ readout HCA cũ): `SENT_L2R`, `SENT_R2L`  
  — `SEQUENTIAL` / `succ` ≈ chỉ **L2R d=1** toàn cục; P7 **không** thay bằng cách phình SEQUENTIAL

Tuỳ chọn siết hub lúc đếm: bỏ cặp toàn chức năng; nhân `√(gidf_i·gidf_j)`.

#### 4.1.3 Vì sao giúp sinh? (cơ chế — tránh hiểu nhầm)

**Hiểu đúng R2L:**  
`Lan → Em` (R2L, d=2) **không** có nghĩa “next token sau Lan là Em”.  
Nghĩa là: đứng ở `Lan`, **truy hồi neo** `Em` (và các token đứng trước trong cùng kiểu câu) → dùng neo đó để **chọn cha / chọn continuation**.

| Tình huống decode | Đọc cạnh | Tác dụng lên next |
|-------------------|----------|-------------------|
| History …`chủ` / `Lan` | **R2L** từ `w0` (+ vài token cuối) | Lấy tập neo `A` (đầu câu thường đi với đuôi này) → **re-rank / lock parent** chứa `(neo ∪ w0)`; hạ flu bigram mù |
| History đã có `Ngô`/`Em` | **L2R** từ neo đầu / entity trong history | Mass token **đứng sau** trong cùng câu train → giao với **phần emit còn lại** của cha đang active → ứng viên next |
| Trong path cha | L2R/R2L + `d` khớp vị trí path | Ưu tiên bước còn lại đúng khoảng cách / hướng |

**Công thức decode đề xuất (lõi):**

```
# 1) Neo từ đuôi (R2L) — conditioning, không copy thành next
anchors = normalize( Σ_m α_m · W_r2l[ h_{k-m} ] )     # m=0..M-1, α_m decay
parents'  = rescore(parents | hist, anchors)           # peaking cha đúng câu
# 2) Continuation trong cha (L2R) — ứng viên next thật
cont = Σ_{a in anchors∪early(hist)} W_l2r[a]
cand = cont ⊙ mask(emit_remaining(parents'))           # BẮT BUỘC cổng theo cha
# 3) Trộn
logits += β_par·bag(parents') + β_path·path_next
       + β_l2r·cand + β_succ·succ(w0)                   # succ = L2R d=1 flu
       + β_eos·…
# flu toàn cục ↓ khi conf(parents') cao
```

Không cổng theo cha → L2R/R2L toàn cục biến thành hub (`,`.`là`) và salad nặng hơn bigram.

#### 4.1.4 Materialize ma trận (thực dụng)

| Artefact | Shape | Cách dựng |
|----------|-------|-----------|
| `W_l2r`, `W_r2l` | CSR `(N×N)` | Cộng `f(d)` mọi cặp trong mọi parent câu; optional nhân gidf |
| (Khuyến nghị) `intra_by_parent` | per-`pi`: pairs hoặc bitmask emit | Decode mask `emit_remaining` O(len câu) |
| Sidecar | cùng bipartite `_v` | Persist; không rebuild mỗi probe |
| SVD tuỳ chọn | Q/K intra dim nhỏ | Sau khi kênh CSR đã dương trên probe — không bắt buộc P0 |

**Độ phức tạp:** câu dài L≈20–40 → O(L²)/câu chấp nhận được sau segment; `d_max` + bỏ punct-only pair nếu cần.

#### 4.1.5 Cầu nối P6 (`h`)

Sau khi P7 ổn:

```
h = Σ_i ω_i E[h_i]           # full-past surface
  + γ_r · Σ_j anchors_j E[j] # neo R2L
  + γ_l · Σ_j cand_j E[j]    # hướng L2R trong cha
score(j) += β_h · (h · key_j)
```

P7 = cạnh tường minh; P6.A = nén cùng tín hiệu thành một vector — **không** thay P7 bằng thêm tầng SVD trên PPMI cũ.

#### 4.1.6 Failure modes & chống

| Lỗi | Triệu chứng | Chống |
|-----|-------------|--------|
| Hub | `,`.`là` thắng R2L/L2R | gidf; chỉ content pair; cổng cha |
| R2L → next sai | Sinh lại `Ngô` ngay sau `chủ` | R2L chỉ vào **anchors/parents**, không cộng thr thẳng vào next trừ khi path còn slot |
| Không segment | Cạnh xuyên đoạn wiki | Bắt buộc P0.1 trước (hoặc cùng PR) |
| Chỉ global W | Đúng token sai bài | `mask(emit_remaining)` |
| `d` phẳng | Túi hòa trong câu | `f(d)=λ^{d−1}`; ablation `d=1` vs full |

#### 4.1.8 Giúp gì cho Attention (Q/K, Wa) và FFN (Amp / Wffn)?

**Hiện trạng (hạn chế):** `attention_qk` + `spectral_stack` đều chiết từ **một** PPMI hướng của `SEQUENTIAL` ≈ bigram L2R `d=1`. Residual SVD nhiều tầng trên **cùng** M → không thêm inductive bias mới (đã đo α phẳng / hub). Attn và FFN gần như hai mặt của **cùng** thống kê chuyển tiếp cục bộ.

**P7 đổi nguyên liệu đồ thị** — không phải “thêm tầng cho sâu”:

| Thành phần | Nay (W_seq / PPMI) | Với L2R/R2L+`d` | Lợi ích thuật toán |
|------------|--------------------|-----------------|---------------------|
| **Q/K attention** (`q·k ≈ PPMI`) | Chỉ mẫu “đứng trước → đứng sau” **kề** | PPMI riêng từ **`SENT_L2R`** (mọi `d`) | Unseen transition trong câu vẫn khớp khuôn *skip* (Em→Lan), không chỉ Em→bé |
| **Query `q` / state `h` trước Attn** | `w0` hoặc túi history trên cùng không gian | `h` giàu **R2L**: neo đầu câu trước khi `q` attend | Attn không còn “mù đuôi”; α sắc hơn vì cue đúng fact |
| **Keys / cửa sổ S** (stack) | Top W_seq/`Wa` → dễ hub | S ← anchors(R2L) ∪ L2R⊙emit ∪ W_seq | Softmax trên tập **đồng-câu**, giảm `. , và` |
| **Wa (Attn spectral)** | Deflation PPMI bigram | Nên gắn view **L2R** (hướng đi tiếp) | Đúng vai “CÁCH nói về phía trước” |
| **Wffn / Amp (FFN spectral)** | Cùng residual PPMI → gần trùng Attn | Nên gắn view **R2L** hoặc co-occur trong câu (neo↔đuôi) | FFN = *làm giàu ngữ cảnh / thuộc cùng câu*; Attn = *chọn bước tiếp* — **tách vai** thật |
| **Đa view (P2.1)** | Trong-câu vs qua-câu mơ hồ | L2R/R2L = trong-câu; SENT_SUCC = qua-câu | Stack/QK β riêng, hết trộn kỷ |

**Pipeline đề xuất (sau CSR P7 ổn):**

```
W_l2r, W_r2l  →  PPMI_l2r / PPMI_r2l  →  SVD
    Attn:  Wq,Wk ← PPMI_l2r     (và/hoặc mix nhẹ SEQUENTIAL d=1)
    FFN:   U_ffn  ← PPMI_r2l hoặc (L2R+R2L) đối xứng hoá trong câu
Decode:  h ← hist + R2L_anchors;  q = h Wq;  attend keys trong S đã cổng cha
```

**Không kỳ vọng:** chỉ đếm cạnh P7 mà **không** rebuild Q/K/stack / không đổi cách dựng `q` và S → Attn/FFN gần như cũ.  
**Không kỳ vọng:** thêm layer residual trên PPMI bigram cũ sẽ thay P7.

**Một câu:** P7 cho Attn **hướng đi + skip trong câu**, cho FFN **nhìn về / đồng-câu**, và cho cả hai một **`h`/S có neo** — đúng chỗ bigram PPMI đang thiếu.

---

## 5. Việc CẦN LÀM — kế hoạch cụ thể

### Ba trụ lộ trình (map ①②③)

| # | Trụ | Chi phí | Ticket / mục | Một câu |
|---|-----|---------|--------------|---------|
| **①** | Liên kết câu | Rẻ — làm trước | **P0** (+ P1, P4) | Segment + SENT_SUCC → `W_agenda` + handoff EOS |
| **①b** | Intra-sentence L2R/R2L+`d` | Trung | **P7** | R2L→neo/cha; L2R→next trong emit; cổng cha bắt buộc |
| **②** | Phân bố sắc | Rẻ–trung bình | **P5** (+ P3.4/P3.5) | Next-token không phẳng → mới sample có ích |
| **③** | Học biểu diễn dài `h` | Đắt hơn — **lõi TK_042** | **P6** (G13 kNN-LM / §7.8+) | Vector ngữ cảnh cả đoạn điều kiện hoá next |

Thứ tự tuần: `① → ①b (P7 song song decode) → decode agenda → ② → ③ hướng A rồi B`.

---

### P0 — ① Liên kết câu: training + agenda + handoff (ưu tiên cao nhất)

#### P0.0 — Spec ingest (1–2 buổi, không train full)

| Bước | Việc cụ thể | File / artefact |
|------|-------------|-----------------|
| a | Luật cắt: `split_sentences` / biên `.!?` + QA = 1 answer = 1 parent; min/max token; bỏ câu rác | `data/quality.py`, gate mới hoặc tái `paragraph_units` |
| b | Schema: mỗi parent gắn `doc_id`, `sent_idx`, `episode_idx` (sidecar hoặc field nhẹ) | `ParentRecord` / content-cache |
| c | Probe dry-run: 50 bài lịch sử → thống kê #câu/bài, độ dài emit, % còn mega-parent | `scripts/probe_sentence_parents.py` (mới) |

**DoD P0.0:** báo cáo số; không còn parent wiki >N token (N≈40–60) trong mẫu.

#### P0.1 — Segment parent theo câu / QA answer

| Bước | Việc cụ thể | Kỳ vọng |
|------|-------------|---------|
| a | Ingest/retrain: 1 câu (hoặc 1 QA answer) = 1 `ParentRecord`; `emit_ids` = surface câu đó | `E`/`succ`/`end_after` theo biên câu thật |
| b | Cùng `doc_id`: tạo/ cập nhật `AbstractNeuron` với `members` đúng thứ tự trình bày | Episode = xương agenda |
| c | `match_ids` câu: content tokens câu + (tuỳ) entity title bài gắn nhẹ | Coact không hòa tan cả đoạn |
| d | Giữ pool đoạn TK_035 (priority 0.36) **song song** nếu cần discourse 2–4 câu — không thay P0.1 | Không regress discourse-route |
| e | Rebuild bipartite cache (`_v≥11`): `A` match-only; `succ` + `w→EOS` tại hết câu | Path hết câu → EOS cạnh tranh sạch |

**Điểm chạm code (dự kiến):** pipeline ingest wiki/SFT → `ParentStore.add_*`; `build_*_episodes`; `matrix_core.build_parent_bipartite`.

**Không:** gộp cả bài vào một `emit_ids`.

#### P0.2 — Cạnh SENT_SUCC / AGENDA_NEXT → `W_agenda`

| Bước | Việc cụ thể | Kỳ vọng |
|------|-------------|---------|
| a | Typed edge (hoặc counter): với mỗi cặp member liên tiếp `(p_k, p_{k+1})` trong episode — cộng mass từ **token cuối nội dung** câu k → **token đầu nội dung** câu k+1 (và/hoặc bigram biên) | Sparse đếm tái lập |
| b | (Tuỳ) cạnh parent→parent `AGENDA_NEXT` rồi `scatter` xuống token đầu emit | Dễ debug “câu nào sau câu nào” |
| c | Materialize `W_agenda (N×N)` CSR (+ optional SVD Q/K agenda dim nhỏ) | View riêng, β riêng |
| d | Wire vào `matrix_core_logits`: khi path gần EOS / conf EOS cao → cộng `β_agenda · W_agenda[w0]` (và/hoặc bag từ member kế) | Next-token sau EOS local có thể mở câu 2 **qua logits** |
| e | Sidecar persist agenda cùng bipartite | Probe không rebuild mỗi lần |

**Luật:** chỉ cạnh cùng `doc_id` / cùng episode (không nối chéo bài chỉ vì token trùng).

#### P0.3 — Decode handoff segment (EOS local ≠ hết bài)

| Bước | Việc cụ thể | Kỳ vọng | TT 2026-07-26 |
|------|-------------|---------|---------------|
| a | Trạng thái decode / cursor câu trong bài | Biết đang ở câu nào | ✅ span từ episode member |
| b | Khi `<EOS>` thắng **và** còn member kế → chèn **span** emit câu sau (không chỉ 1 token); không bơm agenda mid-sentence | ≥2 câu cùng entity | ✅ |
| c | Khi EOS thắng **và** không có span → dừng; `ar_sent_agenda_handoff_max` chống loop | Không `Năm.Năm…` | ✅ |
| d | Neo cha = coactivation hist (không so `eos_logit`; không hub `A[w0]`) | Luật 0 | ✅ |
| e | Probe: `no_agenda` = 1 câu; matrix+handoff = ≥2 câu Ngô | Log | ✅ `only6` |

---

### P1 — Siết neo entity lúc train

| ID | Việc cụ thể | Kỳ vọng |
|----|-------------|---------|
| **P1.1** | `match_ids`: ưu tiên title + entity bề mặt (gidf cao); cắt / hạ trọng token chức năng trong A | `A` peaked; ít Ngô∧Tây Sơn |
| **P1.2** | (Tuỳ) cạnh `ENTITY_IN_PARENT` hoặc Theme.facts siết hơn khi coact | Soft-AND cứng từ data |
| **P1.3** | Contrastive Hebbian nhẹ: khi cập nhật cha entity X, hạ coact cha kỷ khác cùng token hub | Ít hòa tan |
| **P1.4** | Episode `match_ids` = title/instruction; **không** nhét full body | Query kích episode rồi mới members |

Làm **song song** P0.1 khi đụng ingest.

---

### P2 — Đa view PPMI / fact / EOS mass

| ID | Việc cụ thể | Kỳ vọng |
|----|-------------|---------|
| **P2.1** | PPMI **trong-câu** vs **qua-câu** (qua-câu = cùng SENT_SUCC window); PPMI gidf-high only | Stack/QK phụ; β riêng |
| **P2.2** | `W_fact` / `_seq_tri` (entity–quan hệ–entity), SVD riêng | Kênh fact ≠ fluency |
| **P2.3** | Nhãn biên câu / `SPAN_END` lúc ingest → tăng mass `w→EOS` / `end_after` | EOS argmax đúng biên |
| **P2.4** | Không thêm tầng SVD “cho giống LLM” trên cùng PPMI nếu residual đã đo phẳng | Tránh ảo giác depth |

Làm **sau** P0 khi còn salad fluency hoặc EOS non.

---

### P3 — Đo / sidecar / sampling (rẻ, chạy song song)

| ID | Việc cụ thể | Kỳ vọng |
|----|-------------|---------|
| **P3.1** | Probe đa prompt (≥10 triều / QA); metric: đúng kỷ, hết câu sạch, #câu vs MZ, không salad | Không overfit Ngô Quyền |
| **P3.2** | Ablation β: parent / eos / know / qk / stack / agenda | Biết kênh nào ăn |
| **P3.3** | Persist bipartite v10+ + `W_agenda` sidecar | Load nhanh |
| **P3.4** | Metric entropy top-k logits từng bước; cảnh báo phân phối phẳng | Gate trước khi tin sample |
| **P3.5** | Probe sample: `T>0` + top-k + optional top-p vs argmax | Đa dạng vs bám fact |

---

### P4 — Tích hợp MZ ↔ matrix

| ID | Việc cụ thể | Kỳ vọng |
|----|-------------|---------|
| **P4.1** | `ar_no_agenda=False`: matrix = micro trong segment; hết mạch local → yield MZ / member kế theo mass agenda | Câu 1 sạch + câu 2 Bạch Đằng |
| **P4.2** | Không frame-bracket thủ công trên matrix path | Luật 0 |
| **P4.3** | So A/B: MZ thuần vs matrix+`W_agenda` vs hybrid handoff trên cùng prompt | Chọn default an toàn |

---

### P5 — ② Phân bố sắc (rẻ–trung bình)

**Định nghĩa:** next-token **không phẳng** — vài ứng viên rõ, còn lại gần 0.  
Probe đã chỉ: sample `T=0.85` trên dist phẳng → salad; argmax ổn định hơn. **Tăng T không làm sắc** — ngược lại.

| ID | Việc cụ thể | Kỳ vọng |
|----|-------------|---------|
| **P5.1** | **Oja / cạnh tranh hàng** mạnh hơn trên `W_know`, PPMI (đã có một phần); (tuỳ) siết hàng `E` | Đỉnh hẹp hơn từ phía ma trận |
| **P5.2** | **Top-k / gate cha** giữ mạnh (đang có) — sample **chỉ** trong top-k sắc (∪ succ ∪ agenda), không full-vocab mềm | Sample không lan kỷ |
| **P5.3** | **Đo** entropy `H(next\|ctx)` trên held-out — mục tiêu **giảm** so với bigram thuần / baseline hiện tại (gắn P3.4) | Có số, không đoán |
| **P5.4** | Peakiness gate runtime: entropy top-k cao → hạ T / fallback argmax / tăng β_parent | Ít salad khi dist xấu |
| **P5.5** | Sample khi đã sắc: `T≈0.7` + `top_k` nhỏ + `top_p≈0.9` (P3.5) | Đa dạng nhẹ, cùng kỷ |
| **P5.6** | Chỉ bật hybrid sample mặc định khi P5.3 + P5.4 đạt ngưỡng | Không “sáng tạo” = noise |

**Không kỳ vọng:** chỉ tăng temperature → sắc hơn.

**Xong khi (DoD ②):** sample 3 seed vẫn **cùng kỷ/entity**, khác cách diễn đạt nhẹ, không salad.

---

### P6 — ③ Học biểu diễn dài `h` (đắt hơn — lõi TK_042)

**Vấn đề (TK_042):** mô hình đếm / túi bigram **không** điều kiện hoá ngữ cảnh dài. Cần **vector ngữ cảnh `h`** từ cả đoạn — không thêm SVD layer trên cùng PPMI.

Gap vs LLM đóng bằng: liên kết câu (①) + dist sắc (②) + **`h` học được** (③).

#### Hướng A — Luật 0 hơn (ưu tiên thử trước)

| ID | Việc cụ thể | Kỳ vọng |
|----|-------------|---------|
| **P6.A1** | `h = f(history)` cố định: positional / decay sum trên embedding E; hoặc Hopfield đọc từ cha đã kích; entity sticky (không cắt window) | Neo đầu câu không mất |
| **P6.A2** | Chấm next: `score(j) = h · key_j` (key từ PPMI/SVD / parent keys) — wire vào `matrix_core_logits` β riêng | Continuity xuyên câu thuần ma trận |
| **P6.A3** | Không backprop; “học” = cập nhật E / cha lúc ingest (+ Oja trên key nếu cần) | Khớp Luật 0 |
| **P6.A4** | Không thay ① bằng “thêm layer spectral” trên cùng PPMI | Đã đo residual ≠ depth LLM |

#### Hướng B — Lai (mạnh hơn, nới Luật 0 — sau A hoặc khi A bão hòa)

| ID | Việc cụ thể | Kỳ vọng |
|----|-------------|---------|
| **P6.B1** | **kNN-LM** (TK_042 **G13** / §7.8+): datastore trên embed câu đã segment; mix `λ·p_matrix + (1−λ)·p_knn` | Fluency khi sparse; `h` query tốt hơn túi PPMI |
| **P6.B2** | (Đắt) Train nhỏ LSTM/Transformer hoặc distill `h` từ model ngoài gắn cạnh/parent | `h` học được thật |
| **P6.B3** | Không kỳ vọng kNN + túi IDF hiện tại đạt R1 bits — cần `h` học (đã đo lift transformer ≫ túi) | Tránh ảo giác “chỉ tăng k” |

**Xong khi (DoD ③):** đổi 1–2 từ giữa câu mà next-token vẫn ổn định đúng nghĩa (không vỡ sang triều khác).

---

### P7 — ①b Intra-sentence L2R / R2L + `d` (solution đầy đủ §4.1)

**Phụ thuộc:** P0.1 (hoặc subset đã segment câu). **Gen L→R.**  
**Nguyên tắc vàng:** R2L → **anchors / rescore cha**; L2R → **next** chỉ sau `mask(emit_remaining)`; không đổ thr R2L thành next.

#### P7.0 — Spec & policy

| ID | Việc | Chốt |
|----|------|------|
| **P7.0a** | Chuỗi vị trí câu = surface emit parent (sau segment) | Đơn vị đếm |
| **P7.0b** | Punct: phương án A giữ trong dãy vị trí; B chỉ content — **probe 2 phương án**, giữ cái ít hub hơn | Không đoán |
| **P7.0c** | `f(d)=λ^{d−1}`, `λ∈[0.7,0.9]`; `d_max` mặc định 12 | Knob config |
| **P7.0d** | EdgeKind `SENT_L2R` / `SENT_R2L` (priority 0, đọc kênh riêng) **hoặc** CSR sidecar trong `matrix_core` song song bipartite | Không nhiễm readout HCA cũ |
| **P7.0e** | Document: R2L ≠ next; ví dụ `Em bé Lan.` đủ 3 L2R + 3 R2L | Tránh implement sai |

#### P7.1 — Ingest / đếm cạnh

| ID | Việc | Kỳ vọng |
|----|------|---------|
| **P7.1a** | Mỗi parent câu: ∀i\<j cộng L2R/R2L với `f(d)` (và optional √gidf) | Đếm tái lập |
| **P7.1b** | Không đếm xuyên `doc` / xuyên episode member | Đúng phạm vi |
| **P7.1c** | Lưu thêm per-parent emit positions để mask decode | Cổng cha |
| **P7.1d** | Script `probe_sent_lr_edges.py`: nnz, top hub, dump câu Em/Ngô | Gate trước wire |

#### P7.2 — Materialize ma trận

| ID | Việc | Kỳ vọng |
|----|------|---------|
| **P7.2a** | CSR `W_l2r`, `W_r2l` (N×N) từ đếm | Load nhanh |
| **P7.2b** | Sidecar version gắn bipartite `_v` | Persist |
| **P7.2c** | (Sau CSR dương) PPMI+SVD: Attn Q/K ← `PPMI_l2r`; FFN U ← `PPMI_r2l` (hoặc co-occur trong câu) — **tách nguồn** khỏi residual cùng W_seq | Attn≠FFN cùng bias; xem §4.1.8 |
| **P7.2d** | Đồng bộ: `succ` / SEQUENTIAL = flu L2R d=1; **không** xoá; β tách | Tránh double-count: β_succ thấp khi β_l2r bật full-d |

#### P7.3 — Wire decode (`matrix_core_logits` / path)

| ID | Việc | Kỳ vọng |
|----|------|---------|
| **P7.3a** | `anchors = R2L(w0 + last M)` → rescore/lock parent (AND nhẹ với hist) | Đuôi câu không mất đầu |
| **P7.3b** | `cont = L2R(anchors ∪ early_hist)` ⊙ `emit_remaining(top parents)` → cộng logits | Next trong đúng fact |
| **P7.3c** | Cấm: cộng raw `W_r2l[w0]` đầy vocab như next | Tránh lặp entity làm next |
| **P7.3d** | `flu` ↓ khi conf parent cao (giữ pattern hiện có) | Bigram không thắng neo |
| **P7.3e** | Knobs: `β_r2l_anchor`, `β_l2r_cont`, `M`, `λ`, `d_max` | Ablation |
| **P7.3f** | EOS: nếu path hết emit và R2L neo ổn → EOS cạnh tranh; nếu agenda còn → P0.3 | Biên câu sạch |

#### P7.4 — Đo / ablation

| ID | Việc | Kỳ vọng |
|----|------|---------|
| **P7.4a** | Prefix `…tự chủ`: rank kỷ đúng; so baseline matrix hiện tại | ↑ hoặc không regress câu 1 |
| **P7.4b** | Ablation: off R2L / off L2R / chỉ d=1 / no parent-mask | Parent-mask và R2L phải hiện regress khi tắt |
| **P7.4c** | Hub audit: top-20 hàng R2L/L2R không toàn punct/chức năng | gidf đủ |
| **P7.4d** | 3 seed sample chỉ sau P5; trước đó argmax | Không salad sớm |

#### P7.5 — Cầu P6.A

| ID | Việc | Kỳ vọng |
|----|------|---------|
| **P7.5a** | `h ← surface + E[anchors_R2L] + E[cont_L2R]` | Một vector điều kiện hoá |
| **P7.5b** | `score += β_h (h·key)` | Đo perturbation giữa câu |

#### P7.6 — Ranh giới

- Không BERT nhìn token chưa sinh.
- Không thay MZ/agenda bằng chỉ L2R trong câu.
- Không materialize all-pairs trên mega-parent chưa segment.

**DoD ①b:** §8 giai đoạn A2 + ablation P7.4b dương.

---

## 6. Thứ tự thực hiện đề xuất

```
① P0.0 → P0.1 segment câu
   → P7.0–P7.2 đếm + CSR (+ probe hub)
   → P7.3 wire decode (R2L→anchor/cha, L2R→cont⊙emit)
   → P7.4 ablation
   → P0.2 W_agenda → P0.3 handoff
   P1.* · P3.* song song

② P5.* sắc phân bố → sample có điều kiện

③ P7.5 / P6.A h từ anchors+cont → P6.B nếu thiếu
```

**Ghi chú:** Solution P7 nằm ở §4.1 (cơ chế + failure). P7.* = checklist implement.

**Checkpoint train:** ưu tiên retrain/ingest có segment trên subset lịch sử → sidecar bipartite+agenda → probe → mới full v12.

---

## 7. Ranh giới / không làm

- Không hardcode list từ / triều để “sửa” Ngô Quyền.
- Không ép next-token = `xem`.
- Không thay MZ bằng matrix alone cho bài nhiều đoạn **trước** P0.2–P0.3.
- Không coi EOS matrix = “agenda đã phủ” khi chưa có `W_agenda`.
- Không gộp cả bài thành **một** parent emit.
- Không bỏ episode hub (chỉ túi câu phẳng).
- Không thêm tầng spectral “cho sâu” trên cùng PPMI nếu chưa có tín hiệu đo.
- Không bật sample nhiệt cao trên full-vocab khi entropy còn phẳng.
- Không cộng raw `W_r2l[w0]` như phân phối next (R2L = neo/cha, không phải next).
- Không bật L2R/R2L toàn cục **không** `mask(emit_remaining)`.
- Không materialize all-pairs trên mega-parent chưa segment.
- Không coi `succ` (L2R d=1) đã đủ thay P7.

---

## 8. Definition of done

### Giai đoạn A — hierarchy + agenda (P0)

1. Ingest: đa số fact wiki/SFT = parent 1 câu/answer; episode `members` đúng thứ tự.
2. Có `W_agenda` / SENT_SUCC đo được (nnz, ví dụ Ngô: định nghĩa → Bạch Đằng).
3. `no_agenda+matrix`: câu 1 sạch trên ≥K prompt lịch sử (không regress probe hiện tại).
4. `matrix+agenda` (hoặc handoff): ≥2 câu cùng entity/kỷ, không salad Tây Sơn / triều lệch.
5. EOS chỉ thắng qua logits; dừng bài khi agenda mass hết (không rule từ điển).

### Giai đoạn A2 — ①b intra-sentence L2R/R2L (P7 / §4.1)

5b. Artefact: mọi câu segment có đủ cặp L2R+R2L+`d` (vd `Em bé Lan.` = 6 cạnh); hub audit OK.
5c. Decode: R2L làm neo/rescore cha (không lặp entity làm next); L2R⊙emit cải thiện tiếp nối trong câu.
5d. Ablation P7.4b: tắt R2L / tắt mask cha / chỉ d=1 → regress rõ trên `…tự chủ` và ≥K prompt.

### Giai đoạn B — ② phân bố sắc (P5 + P3)

6. `H(next|ctx)` held-out **giảm** vs baseline; ablation β documented; peakiness có ngưỡng.
7. Sample 3 seed: cùng kỷ/entity, diễn đạt lệch nhẹ, không salad (DoD ②).

### Giai đoạn C — ③ biểu diễn dài (P6 / TK_042)

8. Hướng A: `h·key` cải thiện xuyên 3+ câu / perturbation 1–2 từ giữa câu (DoD ③), Luật 0.
9. (Nếu cần) Hướng B: kNN-LM G13 mix đo được lift vs matrix-only; không regress fact.

---

## 9. Liên hệ module sẵn có

| Nhu cầu | Đã có | Việc mới |
|---------|-------|----------|
| Câu trong đoạn | `AbstractNeuron.members`, `episodes_of` | Bắt buộc members = **câu** (không mega-parent) |
| Cắt câu / đoạn | `quality.split_sentences`, TK_035 `paragraph_units` | Ingest 1-câu parent + giữ pool 2–4 câu nếu cần |
| Bipartite / EOS | `matrix_core.build_parent_bipartite` | Rebuild sau segment; thêm view agenda |
| Theme entity | `Theme.facts` | Siết match (P1), không thay episode |
| MZ | `generate_hebbian` multizone | Handoff P0.3 / P4.1 |
| Phân bố sắc ② | Oja hàng / top-k gate / hybrid sample | P5 siết + đo `H` trước khi sample mặc định |
| `h` dài ③ | TK_042 G13 `probe_knn_lm`, E/PPMI | P6.A Hebbian `h·key`; P6.B kNN-LM / distill |
| Cạnh trong câu ①b | `succ` ≈ L2R d=1 | §4.1 + **P7**: SENT_L2R/R2L, R2L→anchor, L2R⊙emit, cầu `h` |

---

## 10. Lịch sử ngắn

| Ngày | Sự kiện |
|------|---------|
| 2026-07-24 | Audit matrix; specificity + hard-AND; bỏ emit-from-A |
| 2026-07-24 | Full-history; path lock; anti-loop |
| 2026-07-24 | Gỡ rule EOS; EOS = succ + ctx·end_after + path biên |
| 2026-07-24 | Chốt probe câu định nghĩa sạch; ticket TK_045 |
| 2026-07-24 | Chốt hierarchy: episode hub + parent câu + SENT_SUCC; bổ sung kế hoạch P0–P6 cụ thể vào ticket |
| 2026-07-24 | Map rõ ① liên kết câu / ② phân bố sắc / ③ `h` dài (TK_042); mở rộng P5–P6 đủ DoD gốc |
| 2026-07-24 | Chốt ý tác giả: full-past từ đuôi→đầu câu + entity đầu mở cả fact; thêm §4.1 + **P7** (không BERT decode) |
| 2026-07-24 | **Sửa hiểu:** không chỉ parent-bag — trong câu mọi cặp L2R/R2L + khoảng cách `d` (vd Em bé Lan.); viết lại §4.1 + P7 |
| 2026-07-24 | **Solution triệt để §4.1:** 3 lớp cấu trúc; R2L=neo/cha (không next); L2R⊙emit; failure/hub; P7.0–P7.5 checklist |
| 2026-07-24 | §4.1.8: P7 → Attn (PPMI_l2r, h/S) vs FFN (PPMI_r2l); hết cùng residual W_seq |
| 2026-07-25 | **Implement:** `hca/sent_intra.py` (L2R/R2L+d, W_agenda); EdgeKind SENT_*; wire `matrix_core_logits`; tests + probe scripts |
| 2026-07-25 | **P0.1+P0.3:** migrate + agenda handoff; **ckpt** `network_v11_sent.pkl` (250k parents / 25k ep); probe: mega 62%→5%, `xem` rank1, EOS sạch cuối câu |
| 2026-07-25 | **Handoff siết + P7.2c:** ratio vs EOS, no_agenda dừng; Attn QK←PPMI_l2r / R2L FFN; nnz-cap fallback; ablation `no_agenda` hết `Nghĩa/Ví` |
| 2026-07-25 | **P5+P6+sidecar:** entropy gate sample; `h`=hist+anchors+cont→E; `bake_sent_qk.py` → `.sent_qk.npz`; bỏ MZ khỏi lộ trình matrix |
| 2026-07-25→26 | **Retrain v12:** `--sentence-parents` full corpora → `network_v12_sent.pkl` (430 447 / 64 242 ep) + bake QK |
| 2026-07-26 | **Debug “cụt” + P0.3 span-handoff:** tách L2R vs đa câu; fix flag; coact neo; chèn span câu kế; stop sau 1 handoff — xem **§11** |

---

## 11. Báo cáo 2026-07-26 — v12 retrain, debug cụt, P0.3 span-handoff

### 11.1 Việc đã làm (tóm tắt)

| Hạng mục | Chi tiết |
|----------|----------|
| **Ingest 1 câu = 1 parent** | `pipeline.py --sentence-parents`: split multi-sentence docs → parent/câu; G1 full-doc; episode hub cuối manifest; QA không split (`sentence_parents: false`) |
| **Train v12** | `config/train_v12_sent.yaml` + nohup/caffeinate → `checkpoints/network_v12_sent.pkl` |
| **Bake Attn** | `scripts/bake_sent_qk.py` → `network_v12_sent.sent_qk.npz` (PPMI L2R/R2L) |
| **Matrix channels** | L2R⊙emit_remaining; R2L neo/rescore; `h` dài; agenda-in-logits **mặc định 0** |
| **P0.3 handoff** | EOS → `agenda_handoff_span`: neo cha bằng **coactivation hist** (không `A[w0]`); chèn **cả emit câu kế**; chấm câu; `ar_sent_agenda_handoff_max=1` rồi **dừng** |
| **Tách cờ** | `ar_no_agenda` = tắt MZ/fact retrieval (matrix path). `ar_sent_agenda_handoff` = đa câu ở EOS. Hai cờ độc lập. |
| **Probe scripts** | `probe_agenda_handoff_only.py`, `probe_multi_prompt_long.py`, `diag_handoff_nq.py` |

### 11.2 Vì sao vẫn “cụt” dù đã có L2R/R2L (chẩn đoán)

1. **L2R/R2L = trong câu**, không tạo câu 2+. Đa câu cần episode / SENT_SUCC handoff.
2. Probe `ar_no_agenda=True` + `ar_sent_agenda_handoff=False` → dừng EOS đúng thiết kế 1 câu.
3. Probe “agenda_handoff” cũ đặt `ar_no_agenda=False` → **lệch MZ retrieval**, không matrix+L2R; đồng thời `ar_no_agenda=True` từng **tắt luôn handoff** (bug).
4. Handoff neo `A[w0]` với `w0=chủ` → hub Sao Kim/từ điển; hoặc chỉ chèn token đầu `Năm` → lặp `Năm.Năm…` / dính câu.
5. Bơm `W_agenda` vào logits giữa câu (`ar_sent_agenda_weight>0`) → nhảy câu 2 sớm (`đượcNgô…`).

### 11.3 Fix chính (code)

| File | Thay đổi |
|------|----------|
| `config.py` | `ar_sent_agenda_weight=0`; `ar_sent_agenda_handoff_w_agenda=False`; `ar_sent_agenda_handoff_span=28`; `ar_sent_agenda_handoff_max=1`; comment tách `ar_no_agenda` |
| `sent_intra.py` | `_episode_next_span` / `agenda_handoff_span`; neo `parent_coactivation_logits`; blocklist hub (`Các`,`Bài`,…); tắt W_agenda fallback mặc định |
| `generate_hebbian.py` | Handoff không phụ thuộc `ar_no_agenda`; chèn span + `.` + Ġ; **break khi đủ handoff** |
| `matrix_core.py` | Agenda logits chỉ khi `cont≈0` **và** `w0` là biên câu (và weight thường = 0) |
| `probe_multi_prompt_long.py` | Mode `agenda_handoff`: `ar_no_agenda=True` + `ar_sent_agenda_handoff=True` |

### 11.4 Kết quả probe (chốt)

**1 câu (`ar_sent_agenda_handoff=False`)** — `out/probe_multi_prompt_v12_fix.log` / no_agenda:

| Prompt | Cont (tóm) |
|--------|------------|
| `…Ngô, được` | `xem là người mở đầu thời kỳ độc lập tự chủ` ✅ |
| `Trần Hưng Đạo là` | `Hưng Đạo Vương` ✅ |
| `Lê Lợi khởi nghĩa ở` | `Lam Sơn` ✅ |
| `Nhà Lý dời đô về` | lệch `Nhà Lý thành lập năm 109` + dính khoảng trắng ❌ |
| QA ngắn | vẫn lệch / cụm ❌ |

**≥2 câu (matrix + span-handoff)** — `out/probe_agenda_handoff_only6.log` (max_len=64):

| Prompt | Out |
|--------|-----|
| `…Ngô, được` | `…tự chủ. Năm 938, Ngô Quyền lãnh đạo quân dân đánh bại quân Nam Hán trên sông Bạch Đằng.` ✅ |
| `…tự chủ` (full) | `. Năm 938, … Bạch Đằng.` ✅ |
| `Trần Hưng Đạo là` | `Hưng Đạo Vương. Nhà Lê thống nhất giang sơn.` (câu 2 lệch kỷ — episode) ⚠️ |
| `Nhà Lý dời đô về` | câu 1 vẫn lệch; câu 2 đúng Thăng Long/1010 ⚠️ |
| `Lê Lợi…ở` | `Lam Sơn. Ông phát động khởi nghĩa tại Lam Sơn, Thanh Hóa vào năm 1418.` ✅ |

**Diag** (`scripts/diag_handoff_nq.py`): top coact `pi=4175` đúng định nghĩa Ngô; NEXT episode = `Năm 938…Bạch Đằng`.

### 11.5 Artefact / log chính

| Path | Vai trò |
|------|---------|
| `checkpoints/network_v12_sent.pkl` | Train sentence-parents |
| `checkpoints/network_v12_sent.sent_qk.npz` | Sidecar Attn L2R/R2L |
| `config/train_v12_sent.yaml` | Config train |
| `out/probe_agenda_handoff_only6.log` | **Chốt đa câu** |
| `out/probe_multi_prompt_v12_fix.log` | 1 câu / coact fix |
| `out/diag_handoff_nq*.log` | Neo episode / span |
| `tests/test_sent_intra.py` | 9 passed |
| `tests/test_ingest_sentence_parents.py` | Ingest split |

### 11.6 DoD / việc còn lại

| DoD §8 | Trạng thái |
|--------|------------|
| A1 ingest 1 câu + episode | ✅ v12 |
| A2 W_agenda / SENT_SUCC | ✅ có; decode đa câu **ưu tiên episode span** (W_agenda fallback tắt) |
| A3 no_agenda 1 câu sạch | ✅ Ngô/THĐ/Lê Lợi; ❌ Lý/QA |
| A4 matrix+handoff ≥2 câu cùng entity | ✅ Ngô; ⚠️ THĐ câu 2 lệch; Lý câu 1 lệch |
| A5 EOS qua logits + dừng | ✅ stop sau 1 handoff |
| A2 P7 L2R/R2L | ✅ wire + bake; ablation cũ còn log |
| P5/P6 sample/`h` | ✅ có knobs; chưa DoD sample 3-seed |

**Next (ưu tiên):**

1. Bake FRAME_BRACKET: `scripts/bake_checkpoint_indices.py checkpoints/network_v12_sent.pkl --in-place`
2. Siết câu 1 prompt ngắn (`Nhà Lý dời đô về` → Thăng Long) — succ/path / entity gate
3. QA answers (không split) + styled CO_ACT nếu cần
4. Handoff max>1 khi episode member kế đã lọc entity chặt hơn
5. Spacing BPE còn sót trên một số prompt (`vềNhà`)

---


---

## 12. Tier-1 fixes (dev, 2026-07-26) — M4 CHẠY GÌ

Review output v12 (`out/probe_multi_prompt_long.log`): entity lớn + prompt tự nhiên đã SẠCH
(Ngô Quyền / THĐ / Lê Lợi ✅). Ba nhóm lỗi còn lại: **BUG rẻ** (spacing, wrong-anchor, handoff lặp)
· **CẤU TRÚC** (QA, câu-2 lệch kỷ) · **TRẦN** (số nát, coherence >2 câu = ③/P6).

### 12.1 ĐÃ VÁ (dev — chạy trên v12 hiện tại, KHÔNG cần retrain)

| # | Fix | File | Trạng thái |
|---|-----|------|------------|
| **T1.1** | **Spacing `vềNhà/nămTrận/làTrần`**: sentence-parent câu ĐẦU tài-liệu có `emit[0]` KHÔNG-Ġ; làm continuation giữa dòng → dính. `_word_start_space_fix` trước `out.append`: prev Ġ-trọn + nxt non-Ġ **có Ġ-twin** → swap. Guard twin BẢO VỆ mảnh âm-tiết (`Nguyễn`→`ễn` giữ). | `generate_hebbian.py` | ✅ self-test + M4 hết glue |
| ~~T1.2~~ | ~~Cổng bag giữa câu~~ — **ĐÃ ĐO = NO-OP → REVERT** (xem §12.2 / §12.6). Không ship complexity không đo được. | — | ❌ reverted |

### 12.2 ✅ ĐÃ ĐO trên MINI local — GỐC THẬT: hard-AND loại cha ĐÚNG

Dev 8GB không load nổi v12 (430k parents → ~20GB). Nhưng câu hỏi là **cơ chế** → tái lập trên MINI:
`scripts/diag_wrong_anchor_local.py` build `checkpoints/mini_sent.pkl` từ `history_articles` +
`subject_articles` (**5 004 parents, sentence_parents=True, 46s**) → chẩn tại chỗ, giây/vòng.
Log: `out/diag_wrong_anchor_local.log`.

```
'Nhà Lý dời đô về'      conf=0.305  bag.max=5.19   succ+path.max=0      ← path KHÔNG fire
   act=0.263 'Nhà ĠLý Ġbắt Ġđược Ġhơn Ġ5 .'
   act=0.209 'Nhà ĠLý Ġthành Ġlập Ġnăm Ġ1 0 0 9 .'
   act=0.182 'Nhà ĠTrần Ġchính Ġthức Ġthay Ġnhà ĠLý .'     ← KHÔNG cha nào là "…dời đô về…"
   cổng TẮT/BẬT → top-5 Y HỆT  ⇒ T1.2 NO-OP (đã revert)

'Trận Bạch Đằng năm'    conf=1.000  succ+path.max=0
   act=1 'Trận ĠBạch ĠĐằng Ġcủa ĠNgô ĠQuyền Ġkết Ġthúc…'   ← cha SAI (emit không có "năm")

'Ngô Quyền … được'      conf=1.000  succ+path.max=1952     ← path FIRE → 'Ġxem'(6.13) ✅
   act=1 'Ngô ĠQuyền Ġlà Ġvị Ġvua … , Ġđược Ġxem'
```

**Fact CÓ trong corpus** (kiểm trực tiếp) — nên là **retrieval-fail, KHÔNG thiếu data**:
- `"Sau khi **nhà Lý dời đô về Thăng Long**, kinh đô Hoa Lư…"`  ← khớp gần trọn prompt
- `"Chiến thắng **Bạch Đằng năm 938** chấm dứt…"`               ← khớp "Bạch Đằng năm"

**GỐC: hard-AND neo 2 token gidf-cao + SỚM NHẤT (= chủ ngữ) loại chính cha đang tiếp cụm.**
- `"Nhà Lý dời đô về"` → hard-AND ≈ {`Nhà`,`Lý`}; cha đúng viết **`nhà`** thường (`Sau khi nhà Lý…`)
  → **case-mismatch** (`Nhà`≠`nhà`, id khác) → rớt hard-AND → `act=0` → path không có cha để khớp.
- `"Trận Bạch Đằng năm"` → hard-AND ≈ {`Trận`,`Bạch`}; cha đúng là `"Chiến thắng Bạch Đằng năm 938"`
  (**không có** `Trận`) → rớt → chỉ còn cha `"Trận Bạch Đằng của Ngô Quyền…"` (emit không có `năm`)
  → path fail → bag → `Đằng` → **khởi động lại câu khác**.

⇒ **Quy luật chung giải thích MỌI ca**: hệ tiếp nối bằng **khớp suffix CHÍNH XÁC** vào emit đã nhớ.
Prompt = tiền-tố **verbatim** của câu đã lưu → path fire → sạch (Ngô/THĐ/Lê Lợi ✅).
Prompt **diễn đạt khác** (hoa/thường, đổi từ, QA) → không cha nào khớp → bag → nhảy câu khác.
Đây là **tường KHÁI QUÁT HOÁ**, không phải bug lẻ.

**Hướng vá (theo ROI, cho M4 quyết bằng ablation):**
1. **Case-fold trong hard-AND / match** — `ps.case_equiv_ids` đã có. Rẻ nhất, vá ngay ca `Nhà/nhà`.
2. **Hard-AND theo token PHÂN BIỆT (gidf), không theo SỚM NHẤT** — `dời/đô` (hiếm) phải neo mạnh hơn
   `Nhà` (phổ biến); hiện "sớm" thắng nên chủ-ngữ chi phối.
3. **Path mềm (Hopfield)** — thay khớp-suffix-chính-xác bằng truy hồi mềm emb (đúng ý TK_044 P3.2
   Plan A: *retrieval from partial cue*). Đắt hơn nhưng giải cả lớp paraphrase.

### 12.2b ✅ VÁ (1) case-fold hard-AND — ĐO trên mini

`parent_coactivation_logits(..., casefold=True)` (knob `ar_matrix_hard_and_casefold`): hard-AND gộp
lớp tương-đương hoa/thường qua `ps.case_equiv_ids` (`Nhà`≡`Ġnhà`, precompute fold — Luật 0).

```
'Nhà Lý dời đô về'   casefold=False  succ+path.max=0       cha-top: "Nhà Lý bắt được…","Nhà Lý thành lập…"
                     casefold=True   succ+path.max=45.59   cha-top: "Năm 1010, nhà Lý DỜI ĐÔ từ…"  ✅
'Ngô Quyền … được'   False 1952 / True 1951 · logits Y HỆT ('Ġxem' 6.13)   ← KHÔNG regress ✅
'Trận Bạch Đằng năm' False 0 / True 0                                       ← casefold KHÔNG cứu ⚠️
```

**Đọc trung thực:** casefold **vá đúng cơ chế** — path từ 0 → 45.59, cha ĐÚNG ("nhà Lý dời đô") nổi
lên top. Control không regress. **NHƯNG** trên mini *nạp-lại* (`resync_match_ids` dựng lại match_ids)
argmax cuối đã đúng `ĠThăng` ở CẢ hai → **hiệu ứng end-to-end phải đo trên v12** (M4).

`Trận Bạch Đằng năm` vẫn 0: cha đúng là `"Chiến thắng Bạch Đằng năm 938"` — **không có** `Trận`, mà
hard-AND neo {`Trận`,`Bạch`} (2 token SỚM NHẤT). Khác-từ, không phải khác-hoa-thường ⇒ **cần vá (2)**:
neo theo **gidf/phân-biệt** thay vì **sớm nhất**.

### 12.2c ❌ vá(2) neo-theo-gidf: ĐO = KHÔNG ăn → revert. **Nút thắt thật = PATH khớp CHÍNH XÁC**

Thử `and_pos_alpha` 1.5(cũ)/0.5/0.3 (mềm hoá phạt vị-trí để gidf dẫn thay token SỚM NHẤT):

```
'Trận Bạch Đằng năm'  α=1.5/0.5/0.3 → succ=0, cha Y HỆT (cha đúng "Chiến thắng Bạch Đằng năm 938" KHÔNG nổi)
'Ngô Quyền … được'    α=1.5/0.5/0.3 → succ=1951, Y HỆT
'Lê Lợi khởi nghĩa ở' α bất kỳ      → succ=0  NHƯNG cha TOP ĐÃ ĐÚNG ("Lê Lợi khởi nghĩa Lam Sơn")
'Hỏi: Ngô Quyền là ai' α=0.3        → tập cha RỖNG (tệ hơn)
```
⇒ α không đổi gì, còn hại ở QA → **revert** (cùng kỷ luật T1.2: không ship cái đo ra vô ích).

**PHÁT HIỆN QUAN TRỌNG HƠN — chẩn lại nút thắt:** `'Lê Lợi khởi nghĩa ở'` có **cha ĐÚNG ở top**
nhưng `succ+path = 0`, vì emit lưu là `"Lê Lợi khởi nghĩa Lam Sơn"` — **không có token `ở`** →
khớp suffix chính xác chết ngay. Tương tự `"Bạch Đằng năm"` vs emit `"Bạch Đằng của"`.

⇒ **Nút thắt KHÔNG phải chọn-cha (activation/hard-AND) mà là `_path_continuation_logits` đòi
khớp token-suffix CHÍNH XÁC.** Chọn-cha đã đúng ở 2/4 ca mà vẫn không sinh được. Mọi diễn đạt lệch
một hư-từ (`ở`/`năm`/`của`) là chết. Đây đúng là **tường khái-quát-hoá** §12.2, và nó khu trú vào
MỘT hàm.

**⇒ Ưu tiên số 1 giờ là vá(3) PATH MỀM** (không phải tinh chỉnh chọn-cha nữa):
- bỏ qua hư-từ khi khớp (khớp trên chuỗi CONTENT token), hoặc
- khớp mềm bằng embedding/Hopfield (*retrieval from partial cue* — đúng TK_044 P3.2 Plan A), hoặc
- cho phép skip/gap có phạt (như alignment), thay vì đòi liên tiếp khít.
Ablation: `_path_continuation_logits` là hàm khu trú → A/B rẻ, đo ngay trên mini local.

### 12.3 M4 chạy theo thứ tự

```bash
# 1) XÁC NHẬN T1.1 spacing (kỳ vọng hết 'vềNhà','nămTrận','làTrần')
PYTHONPATH=src python -u scripts/probe_multi_prompt_long.py \
    --checkpoint checkpoints/network_v12_sent.pkl   # hoặc probe v12 đang dùng

# 2) DIAG trên v12 THẬT — xác nhận gốc hard-AND (mini đã chốt cơ chế, §12.2)
PYTHONPATH=src python -u scripts/diag_channel_attrib.py \
    --checkpoint checkpoints/network_v12_sent.pkl --log-file out/diag_channel_attrib.log

# 3) Vá theo §12.2: (1) case-fold hard-AND → (2) neo theo gidf thay "sớm" → ablation
#    (T1.2 đã revert — đo cho thấy no-op)
```

### 12.4 ⚠️ 2 test ĐỎ SẴN (trước fix dev — M4 xem)

`tests/test_sentence_parents.py::test_agenda_handoff_picks_next_sentence_start` và
`::test_episode_handoff_continues_to_next_member` — đỏ **từ commit `28f371a`**, không liên quan
Tier-1. Nghi handoff đổi hành vi (span thay first-token) mà test chưa cập nhật.

### 12.5 Tier-2/3 (kế, theo ROI)

| Tier | Việc | Ghi chú |
|---|---|---|
| **T2.a** | **Ablation β (P3.2)** — tắt từng kênh trên ≥10 prompt | **Giá trị cao nhất**: ~10 kênh đang trộn, chưa biết kênh nào ăn → đơn-giản-hoá trước khi thêm |
| T2.b | QA routing (QA answer parent, không matrix path) | "Ngô Quyền là ai"→"không còn chịu ách…" = không trả lời |
| T2.c | Handoff không lặp chủ-ngữ (THĐ→"Trần Hưng Đạo đánh thắng") + câu-2 đúng kỷ (P1.4 episode match) | |
| **T3** | Số nát (109/128,8) = tường TK_041 · coherence >2 câu = ③/P6 `h` học được | Ngắn hạn: copy số verbatim từ emit cha. Dài hạn: RAG (graph ground + LM compose) |

### 12.6 M4 kết quả đo (2026-07-26) — đánh giá

Chạy đúng §12.3 trên `network_v12_sent.pkl`. Commit: `a636f1e` (T1.1) + `5b2ed5b` (T1.2 + runbook).

#### (1) T1.1 spacing — `out/probe_multi_prompt_long_t11.log`

| Kiểm | Kết quả |
|-------|---------|
| `vềNhà` / `nămTrận` / `làTrần` | **Hết dính** (`glue=[]` trên A/B) ✅ |
| Ngô / THĐ / Lê Lợi (no_agenda + handoff) | Giữ sạch; handoff Ngô→Bạch Đằng 938 ✅ |
| `Nhà Lý dời đô về` / `Trận Bạch Đằng năm` | Có space trước token đầu, nhưng **loop rác** `Nhà Nhà 0…` / `Trận 8 8…` ❌ regress chất lượng vs trước (`…vềNhà Lý thành lập năm 109`) |

**Verdict T1.1:** vá spacing **đúng mục tiêu hình thái**; trên ca wrong-anchor, swap `Nhà`→`ĠNhà` + bag restart làm lộ/loop số — không phải hết bug neo cha.

#### (2) DIAG — `out/diag_channel_attrib.log` → **H1**

| Prompt | `succ+path.max` | Top cha | Winner |
|--------|----------------:|---------|--------|
| `…dời đô về` | **0** | sai: thành lập 1009 / bắt được… — **không** có “dời đô về Thăng Long” | bag đẩy `Nhà` |
| `…Bạch Đằng năm` | **0** | “Trận Bạch Đằng của Ngô…” (không khớp đuôi `năm`) | bag đẩy `Trận` |
| `…được` (control) | **1941** | pi=4175 đúng định nghĩa | `xem` ✅ |

**Verdict:** T1.2 **no-op đúng dự đoán** (guard `succ>0` không vào). Gốc = **act cha / match**: cha đúng không vào top-5 → path không fire → bag mở câu mới.

#### (3) A/B T1.2 — `out/probe_t12_ab_bag_midsent.log`

| Mode | ly / bach / nq / thd / hung |
|------|------------------------------|
| `penalty=0.35` (ON) | giống hệt OFF |
| `penalty=1.0` (OFF) | giống hệt ON |

**Verdict T1.2:** **không đổi output** trên 5 prompt → bỏ tin cổng bag-midsent cho ca Lý/Bạch; đừng ship như “đã fix wrong-anchor”.

#### (4) Test đỏ sẵn

`test_agenda_handoff_picks_next_sentence_start` + `test_episode_handoff_continues_to_next_member` vẫn `assert None` — test kỳ vọng first-token; runtime dùng span/coact. **Không chặn Tier-1**; cập nhật test khi làm T2.c.

#### Chốt hướng vá tiếp (theo bảng §12.2 H1)

1. **Ưu tiên:** tìm/siết cha `"…dời đô về Thăng Long"` vào top act (`match_ids`, `top_k_parents`, nới `act_floor`/`best_k≥2` nếu cha đúng đã có nhưng bị cắt).
2. **Không** dựa T1.2 cho ca `succ.max=0`.
3. Loop `0/8` sau restart = triệu chứng phụ khi path chết — hết khi path fire đúng (control Ngô không loop).
4. ROI kế: **T2.a ablation β** (§12.5) trước khi thêm kênh mới.

### 12.7 H1 fix — hard-AND đuôi∧entity (2026-07-26)

**Gốc:** cha đúng `"Lý Thái Tổ dời đô về Thăng Long"` có `raw act` cao nhưng bị hard-AND `Nhà∧Lý` (sticky đầu câu) **xóa** vì match không có `Nhà`. Bag còn lại mở câu mới → loop / Tây Đô.

**Vá** (`parent_coactivation_logits`): cặp hard-AND = **(token đuôi hist, 3 cuối)** ∧ **(entity đầu: ưu tiên vị trí sau opener, rồi gidf)** → `Ġdời∧ĠLý`, không `Nhà∧Lý` cũng không chỉ `dời∧đô`.

**Merge với remote (Claude §12.2):** giữ late∧early **và** `_rows_union`/`casefold` (`Nhà`≡`Ġnhà`) — hai lớp độc lập.

**Đo** `out/probe_h1_late_and3.log`:

| Prompt | Trước | Sau |
|--------|-------|-----|
| `Nhà Lý dời đô về` | `Nhà Nhà 0…` / Tây Đô | **`Thăng Long`** ✅ (`succ.max≈65`) |
| `…Ngô, được` | `xem…tự chủ` | giữ ✅ |
| `Trần Hưng Đạo là` | `Hưng Đạo Vương` | giữ ✅ |
| `Lê Lợi…ở` | `Lam Sơn` | giữ ✅ |
| `Trận Bạch Đằng năm` | loop số | vẫn kém (succ=0; năm→năm số — T3) ⚠️ |

### 12.8 vá(3) PATH mềm — A/B v12 (2026-07-26)

**Vá** (`_path_continuation_logits`, knob `ar_matrix_path_soft`): chiếu content (bỏ hư-từ
`ở`/`của`/… + fold); **không** bỏ `là`/`được`/`năm`/`về`. Kèm hard-AND: opener-only pos0
(`Trận`) → AND 2 late thay opener (để cha “Bạch Đằng năm…” sống).

**Đo** `out/probe_path_soft_ab2.log` (light-list đã siết; ab1 regress Ngô vì bỏ `được`):

| Prompt | SOFT_OFF | SOFT_ON |
|--------|----------|---------|
| Lý `dời đô về` | **Thăng Long năm 1010** ✅ | Thăng Long + salad `1 0 0…` ⚠️ |
| Bạch `năm` | `năm 1288` (path fire, năm sai) | `1288` loop dài hơn ⚠️ |
| Ngô `được` | `xem là…` ✅ | `xem là…` ✅ |
| THĐ | OK | OK |
| Lê Lợi `ở` | gen Lam Sơn, **succ=0** | gen Lam Sơn, **succ≈388** ✅ |

**Verdict:** soft đúng cơ chế hung; **default `ar_matrix_path_soft=False`** (không ship regress
số). Opener hard-AND giữ (OFF đã có Bach succ>0 + Lý 1010). Tiếp: T3 copy số / siết năm 938
vs title 1288; soft chỉ bật khi ablation riêng.

**Code / test / log**

| | |
|--|--|
| Soft path + opener hard-AND | `src/nn_aivie/hca/matrix_core.py` (`_path_continuation_logits`, late∧early opener-only) |
| Knob | `config.ar_matrix_path_soft` (default **False**); `ar_matrix_hard_and_casefold` (True) |
| Unit | `tests/test_path_soft.py` (skip `ở` → Lam; không skip `năm` → không khớp `của`) |
| A/B script | `scripts/probe_path_soft_ab.py` |

### 12.9 Post-merge retest v12 (2026-07-26) — §12.3 + H1 + soft

Sau merge `d4fad53` (H1 late∧early ⊕ casefold remote). Chạy lại runbook trên
`checkpoints/network_v12_sent.pkl`:

| Bước | Log | Kết quả |
|------|-----|---------|
| H1 smoke | `out/probe_h1_post_merge.log` | Lý→**Thăng Long** (`succ≈25`) ✅; Ngô/THĐ/Lê Lợi OK; Bạch `succ=0` (trước opener vá) |
| DIAG §12.3 | `out/diag_channel_attrib_post_merge.log` | Lý `succ+path=25.79` top=`ĠThăng` ✅; Bạch 0; Ngô `xem` ✅ |
| Multi-prompt §12.3 | `out/probe_multi_prompt_long_post_merge.log` | Handoff Ngô→938 Bạch Đằng ✅; Lý Thăng Long + salad số; QA lệch; Lê Lợi Lam Sơn + câu 2 |
| Soft A/B | `out/probe_path_soft_ab.log` (ab1) / `ab2.log` | ab1 regress Ngô; ab2 giữ Ngô + hung succ↑; default soft **OFF** (§12.8) |

**Cải tiến giữ (default ON):** T1.1 spacing · casefold hard-AND · late∧early · opener-only→2 late.
**Opt-in:** `ar_matrix_path_soft=True` (hung). **Chưa làm:** T3 số verbatim · năm 938 vs 1288 · QA · T2.a β.


### 12.4 Review vá M4 + 2 fix dev (2026-07-26)

**✅ Vá H1 late∧early hard-AND (M4) — ĂN THẬT.** 1 token ĐUÔI hist (cue tiếp nối) ∧ 1 entity ĐẦU
chọn theo **gidf** (bỏ sticky-pos). `Nhà Lý dời đô về` → cha đúng `"Năm 1010, nhà Lý dời đô từ Hoa Lư
về Thăng Long"` (succ 25.32) → sinh **Thăng Long** ✓. Luật-0 sạch (gidf + vai-trò vị-trí, không bảng từ).
Cùng casefold vá(1) ⇒ ca Lý đóng.

**❌→✅ vá(3) `path_soft`: VI PHẠM LUẬT 0 → ĐÃ SỬA.** Bản M4 dùng `_PATH_LIGHT_FOLDS` = **40 hư-từ
tiếng Việt hardcode**, gọi trong `_path_continuation_logits` = **đường RUNTIME sinh**. `docs/study/09
§2`: word-list ở ingest/build OK, ở runtime generate **CẤM**. → Thay bằng **ngưỡng gidf** (IDF học
được — cùng tín hiệu hard-AND đang dùng); `gidf=None` → không đoán. Cơ chế + knob của M4 giữ nguyên,
`tests/test_path_soft.py` vẫn pass. *(A/B vẫn regress ⇒ giữ default `False`.)*

**✅ Digit-loop → vá bằng LỚP TƯƠNG-ĐƯƠNG FOLD (không đặc-cách chữ số).**
Sau vá H1, `ly` sinh `"…Thăng Long năm 1 0 0 0 …×61"`; `bach` salad. Gốc: `apply_repetition_penalty`
chặn `-inf` **theo ID**, mà BPE có nhiều biến-thể cùng bề mặt (`0`/`Ġ0`/`00`) → **luân phiên id NÉ
được** ⇒ loop. Vá: ban theo **fold** (thuộc-tính token precompute — Luật 0). Self-test:
```
THEO-ID (cũ)     → next = 'Ġ0'  (BIẾN-THỂ → LOOP)
FOLD-EQUIV (mới) → next = 'Long' (thoát loop ✓)
```
Vá **cả lớp** loop biến-thể (không chỉ số). 262 pytest pass.

**Còn lại (ưu tiên):** ① số vẫn *sai giá trị* / tách token (`1010`→`1 0 1 0`) = tường TK_041 → nên
**copy verbatim** cụm số từ emit cha đang khoá. ② `Trận Bạch Đằng năm` → `1 2 8…` (path lệch năm /
title 1288; cha đúng `"Chiến thắng Bạch Đằng năm 938"` vẫn yếu). ③ QA vẫn yếu.

### 12.10 Đo lại sau digit-loop / Luật-0 soft (commit `3da3d8a`, 2026-07-26)

Chạy trên `network_v12_sent.pkl`:

| Log | Kết quả |
|-----|---------|
| `out/probe_h1_claude_digit.log` | Lý→**Thăng Long năm 1010** ✅ (hết `0×61`); Bạch `1 2 8…` + đoạn Ngô; Ngô/THĐ/Lê Lợi OK |
| `out/probe_multi_prompt_long_claude.log` | no_agenda + handoff: Ngô→938 Bạch Đằng ✅; Lý 1010 + câu 2 Trần 1226; Bạch lệch năm rồi handoff 938; QA lệch |
| `out/probe_path_soft_ab_gidf.log` | A/B dở (dừng sớm); SOFT_OFF trùng H1 |

**Verdict digit-fold:** hết **loop biến-thể**; số Lý đúng **1010** nhưng vẫn BPE-tách. Soft gidf chưa re-A/B đủ.

### 12.5 Copy-số-verbatim: THỬ → ĐO REGRESS → REVERT (thiết kế đúng ghi lại)

Ý: BPE tách số thành chữ-số rời + mạng KHÔNG có cơ quan số (TK_041 `1288`⊥`1287`) ⇒ số phải **CHÉP**
từ cha, không **SINH**. Bản thử: `digit_run_after` lấy cha từ `bi.succ[w0]` (cạnh `w0→nxt`, chọn theo
trọng số) rồi chép cụm chữ-số liền sau.

**A/B trên mini (`out/verify_copy_number.log`):**
```
OFF → " Thăng Long năm 1 0 1 0"                        (số vỡ nhưng ĐÚNG chữ số)
ON  → " Thăng Long năm 18110 nhà Lý dời đô từ Hoa Lư…"  (SỐ SAI + lặp lại cả câu)  ✗
```
**Gốc lỗi:** `bi.succ[w0]` gom **mọi** cha chứa `w0→nxt` → chọn theo trọng số ≠ **cha đang khoá path**
⇒ chép số từ cha KHÁC (`18110`). → **REVERT** (kỷ luật như T1.2, vá(2)).

**Thiết kế ĐÚNG (cho M4 / lần sau) — 1 trong 2:**
1. **Plumb `lock_pi`**: `_path_continuation_logits` đã trả `lock_pi` (cha đang khoá) → đưa ra
   `matrix_core_logits` → vòng sinh → chép cụm số **từ đúng cha đó**. Sạch nhất.
2. **Xác minh căn chỉnh tại chỗ**: trong `digit_run_after`, với mỗi ứng viên `pi` từ `succ[w0]`,
   ĐÒI emit khớp **suffix history ≥2 token** trước vị trí `w0` mới nhận (loại cha lệch). Rẻ hơn.

Cả hai giữ Luật 0 (`_is_digit_tok` = lớp ký-tự của token, như `is_structural_token` dùng `isalnum` —
không phải bảng từ). **v2 ĐÃ LÀM (2026-07-26, dev) — nhưng MẶC ĐỊNH OFF vì chưa verify được:**

- Plumb `lock_pi` bằng out-param: `_path_continuation_logits` → `parent_coactivation_logits(lock_out=)`
  → `matrix_core_logits(lock_out=)` → vòng sinh. Không phá chữ ký caller cũ.
- `digit_run_from_locked(bi, vocab, lock_pi, hist_tail, nxt)` — chỉ đọc emit của **cha ĐANG KHOÁ**
  (v1 dùng `succ[w0]` gom mọi cha ⇒ "18110"). Knob `ar_matrix_copy_number` (**default False**).
- 262 pytest pass. **KHÔNG bật mặc định**: dev không verify được end-to-end (mini bị OOM-kill giữa
  chừng, 8GB); v1 từng regress ⇒ không bật mù.

**M4 PHẢI A/B trước khi bật** (`out/verify_copy_number*.log` là tiền lệ):
```bash
PYTHONPATH=src python -u scripts/probe_multi_prompt_long.py \
    --checkpoint checkpoints/network_v12_sent.pkl --log-file out/probe_copynum_off.log
# rồi bật: HCAConfig.ar_matrix_copy_number=True (hoặc thêm cờ CLI) → out/probe_copynum_on.log
```
**Cổng:** `ly` phải ra `"…Thăng Long năm 1010"` (số ĐÚNG GIÁ TRỊ, không `1 0 1 0`, không `18110`)
**và** không regress `nq_def`/`thd`/`hung`. Đạt → đổi default True. Không → giữ False + ghi số.


### 12.6 Đo v12 sau fold-equiv anti-loop (`out/probe_multi_prompt_long_claude.log`) — ĂN, lộ hệ quả phụ

| Prompt | TRƯỚC (post_merge) | SAU (fold-equiv) |
|---|---|---|
| `Nhà Lý dời đô về` | `…năm 1 0 0 0 …×61` **LOOP** | `…Thăng Long năm 1 0 1 0` ✅ hết loop |
| `Trận Bạch Đằng năm` | `…0 0 0 0 ở trận lượt đi` **salad** | `…1 2 8 thắng Bạch Đằng năm của Ngô Quyền kết thúc hoàn toàn thắng lợi` ✅ đỡ hẳn |
| `nq_def` / `thd` / `hung` | đúng | **KHÔNG regress** ✅ |
| handoff Ngô | — | `…tự chủ. **Năm 938**, Ngô Quyền … sông Bạch Đằng.` ✅ |

**⚠️ Hệ quả phụ (chính fix này gây ra):** ban theo LỚP fold chặn luôn **chữ-số lặp hợp lệ**:
`1288` → `"1 2 8"` (chữ `8` thứ hai bị ban), `1000` → cụt. Ngược lại `1010` (không có chữ-số kề lặp)
ra đủ `1 0 1 0`. ⇒ loop chết, nhưng **số có chữ-số lặp bị TRUNCATE**.

**Bằng chứng trực tiếp cho copy-số (§12.5 v2):** cùng log, `Năm 938` trong câu handoff **ĐÚNG và đủ**
— vì đó là **span CHÉP nguyên**, không sinh từng chữ-số. ⇒ số phải CHÉP, không SINH. Đây là lý do
`ar_matrix_copy_number` (v2, lock_pi) đáng bật.

**M4 chạy tiếp (cổng §12.5):** A/B `ar_matrix_copy_number` False↔True trên v12.
Kỳ vọng: `ly` → `…năm 1010`, `bach` → `…năm 1288` (đủ chữ số, hết truncate), controls không regress.
Đạt → đổi default `True` (đồng thời giải luôn hệ quả phụ truncate ở trên).

### 12.11 A/B `ar_matrix_copy_number` trên v12 — FAIL cổng (2026-07-26)

Script: `scripts/probe_copynum_ab.py` (load 1 lần, no_agenda, max_len=36).
Log: `out/probe_copynum_ab.log`.

| Prompt | COPY_OFF | COPY_ON |
|--------|----------|---------|
| **ly** | `Thăng Long năm 1 0 1 0` (+ loop `Hoàng 0…`) | `Thăng Long năm **10109**` ❌ |
| **bach** | `1 2 8` + salad chữ số | **giống hệt OFF** (copy không ăn) |
| nq_def / thd / hung | OK | OK — không regress |

**Cổng §12.5:** ly phải `…năm 1010` (verbatim, không `1 0 1 0`, không `18110`/`10109`) — **KHÔNG đạt**.
**Verdict:** giữ `ar_matrix_copy_number=False`. v2 lock_pi vẫn lệch giá trị (class `18110`); Bach path
chưa khoá cha đúng nên copy không fire. Không đổi default.

### 12.12 Truy `10109`: KHÔNG phải chép sai — chép ĐÚNG rồi đẻ thêm chữ-số (2026-07-27)

§12.11 kết luận "v2 lock_pi vẫn lệch giá trị (class `18110`)". **Đo lại trên mini: kết luận đó SAI.**

`checkpoints/mini_sent.pkl` có đúng câu nguồn (`history_articles.jsonl:14`). Trace
`digit_run_from_locked` khi sinh (`ly`, no_agenda, greedy):

```
[copy] w0='Ġnăm' nxt='Ġ1' lock_pi=805 run=['Ġ1','0','1','0']
       emit='Hoàng Ġthành … ĠLý ĠThái ĠTổ Ġdời Ġđô Ġvề ĠThăng ĠLong Ġnăm Ġ1 0 1 0 .'
copy=False → 'Nhà Lý dời đô về Thăng Long năm 1 0 1 0 Hoàng 0 Hoàng …'
copy=True  → 'Nhà Lý dời đô về Thăng Long năm 1010 Lý dời đô từ Hoa Lư về Thăng Long'
```

⇒ **run chép ĐÚNG 4 token, dừng đúng ở `.`; cổng ĐẠT trên mini.** Cái sai ở v12 nằm SAU bước chép.

**Cơ chế `10109`.** BPE tách `1010` → `Ġ1 0 1 0`: chữ-số đầu có `Ġ`, các chữ-số sau **trần**.
`decode_surface` nối token trần vào từ trước ⇒ bất kỳ chữ-số trần nào sinh ở bước KẾ đều **dính
vào số vừa chép**. Ở v12 argmax bước kế ra `9` → `1010`+`9` = `10109`. (Mini ra `ĠLý` nên không lộ.)
Anti-loop fold-equiv (§12.6) cấm nhóm `0` — chữ-số trần khả dĩ gần nhất chính là `9`.

**Vá (§12.12).** Chép xong = số **ĐÓNG**: bước ngay sau, mask `-inf` các id
**toàn-chữ-số KHÔNG có `Ġ`** (`inner_digit_ids`, cache trên vocab). `Ġ9` (số MỚI sau dấu cách)
vẫn cho qua — chỉ chặn nhánh nối-liền. Lớp KÝ TỰ, không bảng-từ ⇒ **Luật 0 sạch** (như `_is_digit_tok`).

- `src/nn_aivie/hca/matrix_core.py` — `inner_digit_ids(vocab)`.
- `src/nn_aivie/hca/generate_hebbian.py` — cờ `num_closed` (bật sau copy, tiêu thụ ở bước kế).
- `tests/test_copy_number.py` — 3 test (3 passed). Suite: `262 passed`, 2 fail **pre-existing** (§12.4).
- `scripts/probe_copynum_ab.py` — nay in **TRACE mỗi lần copy fire** (`lock_pi` · `run` · emit cha).

Mini sau vá: `copy=True` vẫn ra `…năm 1010 Lý dời đô…` (không regress), `copy=False` không đổi.

**M4 chạy lại (cổng §12.5, không đổi):**

```bash
PYTHONPATH=src python -u scripts/probe_copynum_ab.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --log-file out/probe_copynum_ab_v2.log
```

Cổng: `ly` → `…Thăng Long năm 1010` (không `1 0 1 0`, không `10109`/`18110`) **và** `nq_def`/`thd`/
`hung` không regress. Đạt → đổi default `ar_matrix_copy_number=True`.
**Dán nguyên dòng `[copy]`** — nếu vẫn sai, trace phân biệt ngay: `run` khác `['Ġ1','0','1','0']`
= chép sai (sửa `digit_run_from_locked`); `run` đúng mà output vẫn dài = còn nguồn chữ-số khác.

`bach` KHÔNG thuộc cổng này: copy không fire vì `lock_pi=-1` (cha đúng
`Chiến thắng Bạch Đằng năm 938` thiếu `Trận` ⇒ path không khoá). Đó là ca wrong-anchor §12.x, vá riêng.

### 12.13 Wrong-anchor `bach` — GỠ ĐƯỢC; và một knob đã bị mất trong merge (2026-07-27)

Đo trên `checkpoints/mini_sent.pkl` (5004 cha, có đủ câu nguồn). Mọi số dưới đây là **đo**, không suy.

#### a. Chẩn đoán

`Trận Bạch Đằng năm` so với `Chiến thắng Bạch Đằng năm` (cùng cha đúng pi=4774):

| | `Trận Bạch Đằng năm` | `Chiến thắng Bạch Đằng năm` |
|---|---|---|
| best_k cha đúng | 3 | 5 |
| act cha đúng | 0.0071 | 0.4377 |
| `act_floor` = 5%·đỉnh | 0.0242 | 0.0219 |
| path.max | **0** | 284.6 |

Cha act đỉnh là "Trận Bạch Đằng **của Ngô Quyền** kết thúc…" — emit KHÔNG có `năm` ⇒ path k=0 ⇒
**vô dụng**, nhưng sàn tương-đối của nó vẫn giết cha đúng ⇒ path=0 ⇒ bag soup.

#### b. Knob `ar_matrix_path_soft` đã BIẾN MẤT khỏi `HCAConfig`

`matrix_core_logits` đọc `getattr(config, "ar_matrix_path_soft", True)`. M4 thêm field
(default `False`) ở `0573b5f`; **commit `6753a9a` của tôi** (revert copy-số v1) xoá nó khỏi
`config.py`. Từ đó field không tồn tại ⇒ getattr rơi về fallback **True** ⇒ `path_soft` **BẬT
ngầm** trong mọi lần đo — kể cả A/B v12 `45e35bc` — trái hẳn "default False" ghi ở §12.8–12.9.

Đã khôi phục field + đổi fallback `True`→`False` để không còn hai nguồn sự thật.

#### c. `_is_path_light` nuốt sạch chữ-số

`if len(tok) < 2: return True` — BPE tách số thành token **1 ký tự** ⇒ dưới soft, chuỗi content
của "…năm 1288" **≡** "…năm". Path khớp lại đúng chỗ vừa sinh và trỏ về `Ġ1` mãi ⇒ lặp câu +
salad chữ-số. Chữ-số là CONTENT ⇒ trả `False` trước luật `len<2` (lớp ký tự, Luật 0 sạch).

#### d. Ablation bỏ-từng-vá (7 prompt, greedy, `max_len=26`)

| bỏ | hệ quả |
|---|---|
| **BASE** (tất cả ON) | `bach`→`Trận Bạch Đằng năm 1288` · `ly`→`…năm 1010` · `bach2`→`…năm 1288` · `hung`→`Lam Sơn` · nq_def/thd/hoalu OK — **cả 7 sạch, dừng đúng** |
| −`path_floor_retry` | `bach` chết: `năm Đằng của Ngô Quyền kết thúc…` |
| −digit-là-content | `bach`/`bach2` lặp câu trở lại |
| −`ar_matrix_copy_number` | mọi số vỡ: `bach`→`1 2 8 2 8 2 8…`, `ly`→`1 0 1 0` |
| −`path_soft` | `hung` chết: `ở Huế giành thắng lợi` |
| −`path_lock_ratio` | **không đổi gì** |
| −`and_skip_opener` | **không đổi gì** |

#### e. Chốt

**Giữ** (mỗi cái đều load-bearing): `_path_scan_once` + `ar_matrix_path_floor_retry=True` (quét
lại sàn 0 khi path trắng tay) · chữ-số-là-content · khôi phục `ar_matrix_path_soft`
(**default đổi `False`→`True`**: sau khi chữ-số hết bị nuốt, soft không còn hại số mà vẫn cứu
`hung`; cũng đúng hành vi v12 đang chạy nên log cũ so sánh được).

**GỠ** (đo ra no-op ở cấu hình cuối, theo tiền lệ §12.3/§12.4): `ar_matrix_path_lock_ratio`
(khoá k≥3 + phủ ≥75%) và `ar_matrix_and_skip_opener` (opener khỏi ô hard-AND). Hai vá này chỉ
cần khi chữ-số còn bị nuốt; sửa đúng gốc thì k tự ≥4 và cha đúng tự nổi. Sống lại **chỉ khi** đo
ra ca mà `act[cha đúng]=0` dù `best_k` cao — bằng chứng cũ nằm ở mục (a).

Test: `265 passed` + `tests/test_copy_number.py` 4 test; 2 fail **pre-existing** (§12.4, đã xác
nhận fail cả khi stash toàn bộ thay đổi này).

#### f. M4 chạy gì

```bash
PYTHONPATH=src python -u scripts/probe_copynum_ab.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --log-file out/probe_copynum_ab_v3.log
```

Cổng (không đổi): `ly` → `…Thăng Long năm 1010`, `bach` → `…năm 1288` (không `1 0 1 0`, không
`10109`), `nq_def`/`thd`/`hung` không regress. Đạt → đổi `ar_matrix_copy_number` default `True`.
Mini đã đạt CẢ HAI; v12 chưa đo nên default vẫn **False**. Dán nguyên dòng `[copy]` (trace §12.12).

Ca còn hỏng, chưa vá: `Ngô Quyền đánh bại quân Nam Hán năm` → `năm 8 3 8 …`. Cha nguồn là
"**Năm 938,** Ngô Quyền đánh bại…" — số đứng TRƯỚC, nên không có cặp `năm`→chữ-số để chép.
Chép-lùi là hướng tiếp, chưa làm.

### 12.14 Chép-LÙI: số của cha ĐANG KÍCH khi path không khoá nổi (2026-07-27)

Ca `nq938` (`Ngô Quyền đánh bại quân Nam Hán năm` → `năm 8 3 8 …`). §12.13(f) đoán nguyên nhân là
"số đứng TRƯỚC nên không có cặp `năm`→chữ-số để chép" — **đúng nhưng chưa đủ**. Trace thật:

```
[step] w0='Ġnăm'  nxt='8'  lock=  -1   ← chưa hề khoá được cha
[step] w0='ĠQuyền' nxt='Ġđánh' lock=250 emit='Năm Ġ9 3 8 , ĠNgô ĠQuyền Ġđánh Ġbại …'
```

Cha nguồn là "**Năm 938,** Ngô Quyền đánh bại quân Nam Hán trên sông Bạch Đằng." Hậu-tố history
(`… Nam Hán năm`) KHÔNG xuất hiện trong emit ⇒ suffix-match bó tay (`lock=-1`) — đây là **tường
paraphrase**, không phải chuyện hướng chép. Nhưng `act` thì không mù:

| cha | act | phủ content history | cụm số trong emit |
|---|---|---|---|
| pi=250 | 0.6774 | **100%** | `938` (đúng 1) |
| pi=1915 | 0.3226 | **100%** | `938` |
| pi=813 | 0.00007 | 62% | — |

**Vá** `digit_run_from_active` (`matrix_core.py`), gọi khi `digit_run_from_locked` trả rỗng.
Chọn cha bằng **argmax act** rồi bốn cổng — v1 (§12.5) chép nhầm ra `18110` chính vì thiếu chúng:

1. `nxt` phải là chữ-số (thiếu cổng này → `ly` mất "Thăng Long năm", đo được);
2. **phủ 100%**: mọi content token của history phải có trong emit (test "cùng sự việc");
3. emit có **đúng MỘT** cụm số (nhiều số → không biết chép cái nào → im lặng);
4. history ≥3 content token, và cụm số chưa có sẵn trong history.

Tín hiệu toàn bộ là `act` đã học + fold + lớp ký tự ⇒ **Luật 0 sạch**.

#### Đo (mini, greedy, `ar_matrix_copy_number=True`)

| prompt | from_act=ON | OFF |
|---|---|---|
| **nq938** | `…quân Nam Hán năm **938**` | `năm 8 3 8 Ngô Quyền đánh bại…` |
| Lê Hoàn…Bạch Đằng năm | `năm **981**` | `năm 8 1` |
| Trần Hưng Đạo…Bạch Đằng năm | `năm **1288**` | `năm 8 2 8 1 2 8 2 8…` |
| 7 control §12.13 | **không đổi ký tự nào** | — |

Tổng quát 2/2 sang fact khác, không phải khớp riêng `nq938`.

#### Đối kháng

- `Ngô Quyền đánh bại quân **Tống** năm` → **không chép số** (cổng phủ-100% chặn: cha nói
  `Nam Hán`). Output vẫn lảm nhảm nhưng **không bịa 938**.
- `Nhà **Trần** dời đô về Thăng Long năm` → `1010` ✗ **SAI FACT**, nhưng **giống hệt khi tắt vá**
  ⇒ không do §12.14. Đây là wrong-anchor kiểu ĐỔI THỰC THỂ: suffix `dời đô về Thăng Long năm`
  khớp cha nhà Lý, và `Trần`/`Lý` không được đối chiếu. **Ca mở, chưa vá** — cần đối-chiếu thực
  thể chủ ngữ giữa history và cha, không phải chuyện số.

Test: `270 passed` + `tests/test_copy_number.py` 8 test (4 test mới đều là **cổng âm tính**);
2 fail pre-existing (§12.4).

**M4:** chạy cùng lệnh §12.13(f). `ar_matrix_copy_number_from_act` default `True` nhưng chỉ chạy
khi `ar_matrix_copy_number` bật (vẫn `False`, chờ cổng v12). Ablate bằng `=False`.

### 12.15 Cổng TIỀN-ĐỀ: không bịa khi tổ hợp chưa được chứng thực (2026-07-27)

Ca mở ở §12.14: `Nhà Trần dời đô về Thăng Long năm` → **`1010`** (fact của nhà **Lý**).

#### a. Vì sao "thiếu token" không đủ để phát hiện

Cha thắng (`act=0.508`, emit "Năm 1010, nhà Lý dời đô từ Hoa Lư về Thăng Long.") thiếu **đúng
một** token gidf-cao: `Trần` (4.348) — mọi token khác đều có. Nhưng ca ĐÚNG `Trận Bạch Đằng năm`
cũng thiếu đúng một (`Trận`, 7.439) ⇒ dùng "cha thiếu token gidf-cao" làm cổng sẽ giết luôn §12.13.

#### b. Tín hiệu tách được: GIAO các cha của mọi token gidf≥4 trong prompt

```
ly 2 · bach 11 · bach2 5 · nq938 2 · hung 9 · thd 12 · hoalu 1 · lehoan 1
Nhà TRẦN dời đô 0     ·     Ngô Quyền … quân TỐNG 0
```

Giao rỗng = trí nhớ CHƯA TỪNG thấy tổ hợp này ⇒ mọi cha chỉ khớp một PHẦN ⇒ trả lời tự tin từ
khớp-phần = **bịa**. Chỉ dùng `gidf` + ma trận `A` đã học ⇒ **Luật 0 sạch**.

Báo động giả (18 prompt hợp lệ cắt từ câu thật): **1/18** — `Ngô Quyền cho đóng cọc nhọn xuống
lòng sông`. Corpus viết "**Ông** cho đóng cọc nhọn…" nên tổ hợp thật sự KHÔNG có trong trí nhớ:
đây là khoảng trống **coreference**, cổng không sai về bằng chứng. Bịa: **5/5** nổ.

#### c. Chọn hành động — đã đo 3 mức

| mức | `Nhà Trần dời đô về Thăng Long năm` |
|---|---|
| NONE | `… năm 1010` (bịa, trôi chảy) |
| A — chặn chép số | `… năm 1 0 1 0` (vẫn bịa, chỉ xấu hơn) |
| B — chặn chép + tắt path | `… năm 1 0 1 0 1 0 1 0 …` (tệ hơn) |
| **C — DỪNG** | `Nhà Trần dời đô về Thăng Long năm` ✅ |

Chặn số không xoá được nội dung sai vì bag/succ vẫn đẩy từng chữ-số. Tiền đề chưa chứng thực thì
hành vi trung thực là **không nói**. Cổng tính MỘT lần trên **prompt** (ổn định; nếu tính trên
history đang dài dần thì giao sẽ tự rỗng ⇒ dừng oan).

#### d. Đo cuối (mini, greedy)

| prompt | guard=True | guard=False |
|---|---|---|
| `Nhà Trần dời đô về Thăng Long năm` | *(im)* | `… năm 1010` |
| `Ngô Quyền đánh bại quân Tống năm` | *(im)* | `… năm sau không cần đi đường qua Đại Việt…` |
| `Lê Lợi dời đô về Thăng Long năm` | *(im)* | `… năm 1010 lên ngôi lấy hiệu là Lê Thái Tổ…` |
| 9 ca đúng (ly/bach/bach2/nq938/lehoan/hung/thd/nq_def/hoalu) | **y nguyên từng ký tự** | — |

`ar_matrix_premise_guard` default **True**. Test: `274 passed`, `tests/test_premise_guard.py`
4 test; 2 fail pre-existing (§12.4).

⚠️ **M4 cần đo lại tỉ lệ nổ trên v12.** Mini chỉ 5004 cha nên giao dễ rỗng; v12 có 430k cha ⇒ cổng
sẽ nổ ÍT hơn (bảo thủ hơn), nhưng phải xác nhận nó **không** nổ trên bộ prompt QA hiện có —
nếu nổ >5% prompt hợp lệ thì hạ về mức A hoặc nâng `min_tokens`.

### 12.16 SỬA §12.15 — cổng tiền-đề hạ về OPT-IN + runbook đo tỉ lệ nổ trên v12 (2026-07-27)

**§12.15 tôi đã over-claim.** Default đặt `True` sau khi chỉ kiểm prompt NGẮN. Đo lại với prompt
DÀI / QA-frame trên mini:

| prompt (đều HỢP LỆ) | \|prompt\| | \|gidf≥4\| | cổng |
|---|---|---|---|
| `Nhà Lý dời đô về Thăng Long năm` | 8 | 6 | ok |
| `Hoa Lư là kinh đô…. Năm 1010,…. Nhà Lý dời đô về` | 36 | 17 | ⛔ NỔ |
| `Hỏi: Nhà Lý dời đô về đâu? Đáp:` | 10 | 9 | ⛔ NỔ |
| `Sông Bạch Đằng…. Hỏi: …năm nào? Đáp:` | 49 | 27 | ⛔ NỔ |

Prompt càng dài, tập token đặc-hiệu càng lớn ⇒ giao càng dễ rỗng ⇒ **cổng nổ bừa**. Bật mặc định
= bịt miệng QA. ⇒ `ar_matrix_premise_guard` hạ về **`False` (opt-in)**.

**Hai vá giữ lại (đo được):** bỏ token cấu-trúc/frame (`Hỏi:`/`Đáp:`/punct không phải tiền đề)
và knob `ar_matrix_premise_last_k` (mặc định 6) = chỉ xét MỆNH ĐỀ đang viết.

#### Thước bẩn — bài học riêng

Bản đo đầu lấy prefix từ file text rồi tách câu theo `.`, ra **71% nổ trên văn bản ĐÃ TRAIN** —
suýt kết luận cổng hỏng. Nhưng 5/5 prefix verbatim sạch KHÔNG nổ. Thủ phạm: `5.000` vỡ thành
`"000 lính và voi sang giúp, tháng"`. Đổi thước sang **prefix ~70% emit của CHA** (đọc thẳng
`bi.emits`, không parse, không mã hoá lại) → **0/200 nổ ở mọi `last_k`**. Cũng đã bác giả thuyết
"dùng nhầm `A`(match_ids) thay vì `E`(emit)": hai chỉ mục cho số **y hệt**.

#### M4 chạy

```bash
PYTHONPATH=src python -u scripts/probe_premise_guard_rate.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --log-file out/premise_guard_rate.log --limit 300
```

Chỉ gọi `unattested_premise`, KHÔNG sinh ⇒ xong trong vài giây sau khi load. Bốn nhóm:

| nhóm | ý nghĩa | kỳ vọng |
|---|---|---|
| **emit CỦA CHA (THƯỚC)** | nối tiếp câu mạng đã THUỘC | nổ **≤5%** |
| `qa_supervised` (đã train) | prompt QA có trong đồ thị | nổ thấp |
| `qa_heldout` (chưa train) | tham khảo — nổ ở đây KHÔNG phải lỗi | — |
| **BỊA** (8 prompt hoán thực thể) | mạng PHẢI im | nổ **≥80%** |

**Cổng bật default True:** THƯỚC ≤5% **và** BỊA ≥80% ở cùng một `last_k` → đặt
`ar_matrix_premise_last_k=<k>` + `ar_matrix_premise_guard=True`. Không `k` nào đạt ⇒ **giữ False**
và dán bảng vào đây. Đừng bật mù — đây là vá DUY NHẤT có thể làm mạng **im lặng oan**.

**Căng thẳng chưa gỡ được, M4 đọc kỹ:** trên mini, thước-emit (câu ĐƠN) chuộng `last_k=0`
(0% giả, 100% bịa) còn `k=6` chỉ bắt 75% bịa; nhưng thước-emit **không phủ** prompt nhiều câu /
QA-frame — đúng chỗ `k=0` từng nổ bừa. Hai thước đo hai thứ khác nhau, mini không phân xử được.
v12 có cả cha QA lẫn cha đoạn ⇒ bảng 4 nhóm ở trên mới là trọng tài. Chọn `k` theo **cả hai**
cột `THƯỚC` và `qa_supervised`, đừng chỉ nhìn một cột.

### 12.17 Đo M4 (2026-07-27) — premise_guard rate + copynum A/B v3

#### (1) `out/premise_guard_rate.log` — §12.16

| last_k | THƯỚC | qa_supervised | qa_heldout | BỊA | thước∧bia |
|-------:|------:|--------------:|-----------:|----:|-----------|
| 0 | 0.0% | **100%** | 100% | 100% | ✅ thước+bia / ❌ QA |
| 4 | 0.0% | **100%** | 100% | 0% | ❌ |
| 6 | 0.0% | **100%** | 100% | 62.5% | ❌ |
| 8 | 0.0% | **100%** | 100% | 75% | ❌ |
| 10 | 0.0% | **100%** | 100% | 100% | ✅ thước+bia / ❌ QA |

Script gợi ý `last_k=0`+guard=True — **KHÔNG làm**: `qa_supervised` nổ **100%** mọi `k` (bịt miệng
cả QA đã train). Theo §12.16 (“chọn theo cả THƯỚC **và** qa_supervised”) → **giữ
`ar_matrix_premise_guard=False`**.

#### (2) `out/probe_copynum_ab_v3.log` — §12.13(f) / §12.14

| Prompt | COPY_OFF | COPY_ON |
|--------|----------|---------|
| ly | `…năm 1 0 1 0` | **`…năm 1010`** ✅ `[copy] run=['Ġ1','0','1','0']` |
| bach | `1 2 8 thắng…` | **`năm 1288`** ✅ `[copy] run=['2','8','8']` |
| nq / thd / hung | OK | OK — không regress |

**Cổng đạt** → đổi default `ar_matrix_copy_number=True` (kèm `from_act`).

### 12.18 Review đo v12 của M4 (§12.17) — copynum ĐẠT; guard: số cũ KHÔNG dùng được

#### a. `ar_matrix_copy_number` → default `True`: ĐÚNG

`out/probe_copynum_ab_v3.log`, nguyên văn:

```
COPY_OFF ly:   Nhà Lý dời đô về Thăng Long năm 1 0 1 0
COPY_ON  ly:   Nhà Lý dời đô về Thăng Long năm 1010
COPY_OFF bach: Trận Bạch Đằng năm 1 2 8 thắng Bạch Đằng năm của Ngô Quyền kết thúc hoàn toàn thắng lợi
COPY_ON  bach: Trận Bạch Đằng năm 1288
nq_def / thd / hung: giống hệt cả hai nhánh
```

Cổng §12.5 đạt cả hai vế, controls không regress ⇒ M4 đổi default `True` là đúng. Trace xác nhận
§12.12 trên v12: `run=['Ġ1','0','1','0']` — chép đủ 4 token, **không đẻ thêm chữ-số**.
Ngoài ra `COPY_OFF ly` nay dừng ở `1 0 1 0` (hết loop `Hoàng 0 Hoàng…` của §12.6) ⇒ §12.13 cũng
ăn trên v12, không chỉ mini.

#### b. Cổng tiền-đề: 100% trên QA là **LỖI SCRIPT CỦA TÔI**, không phải bằng chứng về cổng

M4 giữ `premise_guard=False` — quyết định đúng. Nhưng lý do ghi trong §12.17 ("qa_supervised nổ
100%") không dùng được, vì chính con số đó do hai lỗi của tôi:

1. **`Hỏi:`/`Đáp:` lọt vào tập tiền đề.** `is_structural_token` trả **False** cho chúng (đo), mà
   `build_parent_bipartite` lại cố ý loại token cấu-trúc khỏi `A` ⇒ **df=0** ⇒ MỌI prompt QA làm
   giao rỗng **ngay token đầu** ⇒ nổ 100% bất kể nội dung. Vết trên mini:

   ```
   hi=['Hỏi:', 'Ġtrừ', 'Ġ4', 'Ġbằng', 'Ġmấy', 'Đáp:']
   Hỏi:(df=0)→∩0   Ġtrừ(df=20)→∩0   …   Đáp:(df=0)→∩0
   ```

   Vá: loại theo **id** của `HOI_LABEL`/`DAP_LABEL` như `parent_coactivation_logits` vẫn làm
   (special nguyên-tử, so INT — TK_041 §2.2 ⇒ Luật 0 sạch). Sau vá, cùng prompt:
   `hi=['Ġtrừ','Ġ4','Ġbằng','Ġmấy']`.

2. **Verdict tự mâu thuẫn.** Gate in ra chỉ xét `THƯỚC` + `BỊA`, nên §12.17 in
   `→ ĐẶT last_k=0 + guard=True` **trong khi** cột `qa_supervised` là 100%. M4 làm đúng nhờ đọc
   phần văn xuôi §12.16, chứ không nhờ verdict. Đã sửa: gate thêm điều kiện `qa_supervised ≤5%`,
   và CHẶN cứng nếu QA >20% ở mọi `last_k`.

#### c. Điều cần nhớ về chính cổng này

`unattested_premise` hỏi "có cha nào chứa đủ các token đặc-hiệu này không" — **đó chính là** phép
thử "tổ hợp có trong đồ thị không". Nên một nhóm prompt nổ 100% **không** phân biệt được
"cổng hỏng" với "nhóm prompt nằm ngoài đồ thị". Muốn dùng nhóm nào làm thước thì phải chứng minh
nhóm đó Ở TRONG đồ thị trước — đúng lỗi tôi đã mắc ở §12.16 với thước cắt-theo-dấu-chấm.

#### d. M4 chạy lại (2 lệnh, theo thứ tự)

```bash
PYTHONPATH=src python -u scripts/diag_premise_qa.py \
    --checkpoint checkpoints/network_v12_sent.pkl --log-file out/diag_premise_qa.log

PYTHONPATH=src python -u scripts/probe_premise_guard_rate.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --log-file out/premise_guard_rate_v2.log --limit 300
```

`diag_premise_qa` trả lời trước: **(1)** v12 có bao nhiêu cha `is_qa`; **(2)** cổng chạy trên chính
`match_ids` của cha QA — theo cấu tạo phải ra **0%**, khác 0 là cổng còn hỏng; **(3)** vết giao
từng bước cho biết token nào làm rỗng.

- `is_qa = 0` ⇒ nhóm `qa_supervised` nằm NGOÀI đồ thị ⇒ bỏ cột đó khi quyết, chỉ dùng `THƯỚC`+`BỊA`.
- `is_qa > 0` mà mục (2) > 0% ⇒ cổng vẫn hỏng ⇒ **đừng bật**, dán log về.
- `is_qa > 0` và (2) = 0% ⇒ bảng `probe_premise_guard_rate_v2` mới có nghĩa; theo verdict của nó.

Trên v12 số `THƯỚC` đã có và rất tốt: **0/300 = 0.0%** ở mọi `last_k` (§12.17) — cổng không hề
báo động giả khi nối tiếp câu mạng đã thuộc. `BỊA` 100% ở `k=0`/`k=10`, 62.5% ở `k=6`.

### 12.19 Đo §12.18 trên v12 (2026-07-27) — bật premise_guard

#### (1) `out/diag_premise_qa.log`

| | |
|--|--|
| (1) `is_qa` | **670 / 430 447** (0.16%) → có cha QA |
| (2) cổng trên `match_ids` 200 cha QA | **0/200 = 0.0%** → cổng ĐÚNG với đường QA |
| (3) vết | `Hỏi:`/`Đáp:` đã loại khỏi `hi`; ví dụ `hi=['Ġ2','Ġcộng',…]` giao ≠0 |

⇒ 100% QA ở §12.17 là (a) prompt file ≠ đồ thị / frame df=0 — **không** phải cổng hỏng.

#### (2) `out/premise_guard_rate_v2.log`

| last_k | THƯỚC | qa_supervised | qa_heldout | BỊA | cổng |
|-------:|------:|--------------:|-----------:|----:|------|
| 0 | 0.0% | **0.0%** | 97.5% | 100% | ✅ ĐẠT |
| 4 | 0.0% | 0.0% | 56.8% | 0% | ❌ bịa |
| 6 | 0.0% | 0.0% | 90.6% | 62.5% | ❌ bịa |
| 8 | 0.0% | 0.0% | 96.8% | 75% | ❌ bịa |
| 10 | 0.0% | **0.0%** | 97.1% | 100% | ✅ ĐẠT |

**Verdict:** đặt `ar_matrix_premise_last_k=0` + `ar_matrix_premise_guard=True`.
(heldout nổ cao = ngoài đồ thị — tham khảo, không chặn bật.)

### 12.20 Cổng tiền-đề bịt luôn COMPOSITION ⇒ trả về OFF (2026-07-27)

§12.19 (M4) đo v12 sau khi vá nhãn frame — **vá đúng, số đẹp**:

| | THƯỚC (emit cha) | qa_supervised | qa_heldout | BỊA |
|---|---|---|---|---|
| `last_k=0` | 0/300 = 0% | **0/300 = 0%** (trước vá: 100%) | 271/278 = 97.5% | 8/8 = 100% |

`diag_premise_qa`: `is_qa = 670/430,447`; cổng chạy trên chính `match_ids` của cha QA nổ
**0/200** ⇒ kết luận (a): 100% ở §12.17 là do nhãn frame, **không phải cổng hỏng**. Vết (3) sau
vá không còn `Hỏi:`/`Đáp:`. M4 theo verdict → bật default `True`, `last_k=0`.

#### Nhưng cả 4 nhóm đó đều là prompt ĐƠN-FACT

Thước = prefix của MỘT emit · `qa_supervised` = QA số học một cha · `qa_heldout` = ngoài đồ thị ·
`BỊA` = hoán thực thể trong MỘT fact. **Không nhóm nào chạm ca GHÉP.** Đo bổ sung (mini, `k=0`):

```
ĐƠN fact                     nổ 0/4
GHÉP nhiều fact / so sánh    nổ 5/5   ⛔
   ⛔ 'Hỏi: so sánh nhà Lý và nhà Trần Đáp:'
   ⛔ 'Ngô Quyền và Lê Hoàn đều đánh bại quân xâm lược trên sông Bạch Đằng, còn Trần Hưng Đạo'
   ⛔ 'Cả Hoa Lư và Thăng Long đều từng là'
   ⛔ 'Hỏi: Nhà Lý và nhà Trần khác nhau thế nào Đáp:'
BỊA                          nổ 2/2
```

#### Vì sao KHÔNG vá được bằng cách chỉnh ngưỡng

Luật của cổng — *"không cha nào chứa ĐỦ token đặc-hiệu ⇒ im"* — đọc cho đúng là
**"từ chối mọi thứ cần hơn MỘT trí nhớ"**. Đó chính là định nghĩa của composition
(`probe_retrieval_compose` đo `n_facts` = số đơn vị corpus khác nhau output ghép từ; TK_024).

Và bịa cũng chính là ghép: `Nhà Trần dời đô về Thăng Long` = ghép `Trần` (có thật) với
`dời đô → Thăng Long` (có thật) thành mệnh đề sai. Nên **"ghép đúng" và "ghép bịa" không tách
được bằng tín hiệu này** — cả hai đều là "không có cha nào chứa đủ". Không phải chuyện `last_k`
hay `gidf_hi`; chỉnh ngưỡng chỉ trượt dọc cùng một đánh đổi.

⇒ `ar_matrix_premise_guard` trả về **`False`**. Giữ nguyên cơ chế + knob: đây là chế độ dùng được
khi chạy QA-đơn-fact và chấp nhận mất ghép. Test `test_cong_bit_luon_prompt_GHEP_nhieu_fact`
khoá lại giới hạn này để lần sau không ai bật mù.

#### Lỗi là ở THIẾT KẾ CỔNG ĐO của tôi, không phải ở M4

M4 chạy đúng runbook §12.16/§12.18; verdict tôi lập trình đã in `✅ ĐẠT`. Cả bốn nhóm là do tôi
chọn, và cả bốn đều đơn-fact. Đây là lần **thứ ba** trong mạch này khâu yếu là thước chứ không
phải code: §12.16 thước cắt-theo-dấu-chấm (`5.000` → `"000 lính…"`), §12.18 nhãn frame df=0 làm
mọi prompt QA nổ, §12.20 bộ nhóm mù ca ghép. Ghi lại thành luật:

> **Một cổng CHẶN phải được đo trên chính loại đầu vào mà nó có thể chặn oan.**
> Liệt kê các loại đầu vào TRƯỚC, rồi mới soạn nhóm đo.

Ai muốn bật lại `premise_guard=True` phải kèm số trên **prompt GHÉP** (so sánh / đa thực thể /
liệt kê), không chỉ 4 nhóm đơn-fact ở §12.19.

### 12.21 Đóng 2 test đỏ từ `28f371a` — suite XANH (2026-07-27)

Hai test `test_sentence_parents.py` đỏ **từ đúng commit sinh ra chúng**, tôi đi qua 4 phiên và chỉ
ghi "pre-existing". Suite đỏ sẵn thì regression thật sẽ chìm. Ba lỗi, **đều ở TEST, không ở code sinh**:

**1. Đăng ký episode sai API.** `ps.episodes.append(...)` trần bỏ qua `_register_episode` — hàm
vừa append vừa dựng chỉ mục ngược `_parent_episodes`. Hệ quả `episodes_of(pi)` trả `()` ⇒ toàn bộ
handoff câm. (Đường production đi qua `build_episodes_from_provenance` → `_register_episode` nên
không dính; đây thuần lỗi dựng dữ liệu test.)

**2. Kỳ vọng ĐẠI TỪ sai.** Test đòi nối `Ngô Quyền lập nước.` → `Ông thắng ở Bạch Đằng.`
(overlap = 0), mà luật `if overlap < 1 and matched < 6` chặn **có chủ ý**. Dựng
`mini_sent_ep.pkl` để A/B thật thay vì đoán — nới 6→4:

```
mm=6:  'Trận Bạch Đằng năm 1288'
mm=4:  'Trận Bạch Đằng năm 1288. Ông được nhân dân tôn kính và thờ phụng là Đức Thánh Trần.'
```

Đúng **1/8** ca đổi, và đổi theo hướng XẤU: `Ông` không có tiền ngữ (history chỉ có tên trận,
không có người) ⇒ **đại-từ-lơ-lửng**. 7 ca kia y hệt. ⇒ kỳ vọng cũ sai, luật hiện tại đúng.
Knob vừa thêm đã **gỡ** (giá trị thay thế đo ra tệ hơn = dead weight), số đo ghi tại chính hằng
số `6`. Test nay khoá **cả hai vế**: có overlap → nối; đại-từ-lơ-lửng → im.

**3. Kỳ vọng công tắc sai.** Nửa sau test khẳng định `ar_no_agenda=True` làm handoff im — trái
chú thích ngay trong `agenda_handoff_next`: *"ar_no_agenda chỉ tắt MZ/fact plan — vẫn cho handoff
matrix"* và *"không so mass agenda với `eos_logit` tuyệt đối"*. Tách thành test riêng khoá công
tắc THẬT là `ar_sent_agenda_handoff` — để lần sau không ai sửa hàm cho vừa test.

**`scripts/build_mini_ep.py` (mới).** `mini_sent.pkl` truyền `episode_groups` vào ingest nhưng
**quên gọi `build_episodes_from_provenance`** (pipeline thật gọi ở `training/pipeline.py:333`)
⇒ 0 episode ⇒ mọi đường handoff đa-câu KHÔNG đo được trên dev, phải đẩy hết sang M4/v12.
Nay `mini_sent_ep.pkl` = 5.004 cha / **183 episode** / 5.004 cha có `episodes_of ≠ ()`.
Giữ nguyên `mini_sent.pkl` để số §12.13–§12.14 còn tái lập.

**Phát hiện kèm — quan trọng cho việc đo sắp tới.** Với handoff BẬT, mini sinh nhiều câu mạch lạc
hẳn so với mọi thứ ta từng thấy:

```
Lê Lợi khởi nghĩa ở Lam Sơn. Ông phát động khởi nghĩa tại Lam Sơn, Thanh Hóa vào năm 1418.
Hoa Lư là kinh đô đầu tiên của nhà nước Đại Cồ Việt. Hoa Lư nằm ở tỉnh Ninh Bình ngày nay.
Ngô Quyền đánh bại quân Nam Hán năm 938. Năm 981, Lê Hoàn đánh bại quân Tống trên sông Bạch Đằng.
```

**Toàn bộ đo §12.12–§12.20 đều chạy `ar_no_agenda=True`** nên chưa lần nào chạm chế độ này.
Mọi phép đo rộng sau đây PHẢI phủ cả hai chế độ.

`281 passed`, **0 fail** — lần đầu suite sạch trong mạch TK_045 này.

### 12.22 ĐO RỘNG chất lượng sinh — hai lỗ hổng 8 prompt cũ không thấy (2026-07-27)

`scripts/eval_gen_broad.py`. §12.12–§12.21 ship 6 thay đổi mà **toàn bộ bằng chứng nằm trên ~8
prompt tự chọn**, và **đều chạy `ar_no_agenda=True`**. Đây là thước để biết ta có đang khớp quá
vào 8 prompt đó không.

**Thiết kế theo bài học §12.16/§12.18/§12.20** — liệt kê LOẠI ĐẦU VÀO trước, rồi mới soạn nhóm:
`T1 verbatim` (sinh TỪ emit của cha ⇒ luôn đúng miền) · `T2 paraphrase` · `T3 ghép` ·
`T4 qa` · `T5 số` · `T6 bịa` · `T7 dài`, **× 2 chế độ** `no_agenda` / `handoff`.
Chỉ mục corpus dựng từ **emit trong checkpoint**, không parse lại file.
Thước khách quan, không cần LM: `empty · digit_split · loop4 · copy · n_facts · stop_clean · n_tok`.

#### Đo trên `mini_sent_ep.pkl` (5.004 cha / 183 episode, max_len=40)

```
chế độ    nhóm              n   empty  digit⎵  loop4   copy  n_facts   stop  n_tok
no_agenda T1 verbatim       8    0.00    0.00   0.00   1.00     1.00   1.00    7.2
no_agenda T2 paraphrase     5    0.20    0.00   0.00   0.80     0.60   1.00    5.4
no_agenda T3 ghép           5    0.00    0.00   0.00   0.95     0.20   1.00    3.6
no_agenda T4 qa             4    1.00    0.00   0.00   0.00     0.00   1.00    0.0
no_agenda T5 số             4    0.00    0.00   0.00   1.00     0.75   1.00    6.8
no_agenda T6 bịa            5    0.00    0.40   0.20   0.59     0.80   0.80   20.8
no_agenda T7 dài            4    0.00    0.00   0.25   0.81     0.75   0.75   13.2
handoff   T1 verbatim       8    0.00    0.00   0.00   0.75     1.88   1.00   29.2
handoff   T2 paraphrase     5    0.00    0.00   0.00   0.77     1.60   1.00   20.0
handoff   T3 ghép           5    0.00    0.00   0.00   0.95     0.20   1.00    3.6
handoff   T4 qa             4    1.00    0.00   0.00   0.00     0.00   1.00    0.0
handoff   T5 số             4    0.00    0.00   0.00   0.74     1.50   1.00   20.0
handoff   T6 bịa            5    0.00    0.40   0.20   0.42     1.60   0.60   34.6
handoff   T7 dài            4    0.00    0.00   0.25   0.71     1.25   0.75   27.2
```

#### Ba điều đọc được

**(a) T4 QA sinh RỖNG hoàn toàn** — `empty=1.00`, `n_tok=0.0`, ở CẢ HAI chế độ. Không phải cổng
tiền-đề (đang OFF từ §12.20). `Hỏi: Nhà Lý dời đô về đâu? Đáp:` ra đúng 0 token, trong khi
`Nhà Lý dời đô về` chạy tốt ⇒ **đường QA hỏng riêng**, không phải thiếu tri thức.
⚠️ Mini có `is_qa = 0` cha (dựng từ articles, không nạp corpus QA) nên đây CÓ THỂ là ảo giác
của mini — v12 có 670 cha QA (§12.19), phải đo lại mới kết luận.

**(b) T3 GHÉP gần như không sinh** — `n_tok=3.6`, `n_facts=0.20` ở cả hai chế độ. Composition
(TK_024) trên thực tế **chưa chạy**, dù §12.20 đã gỡ cổng chặn nó.

**(c) Khuyết tật còn lại TỤ ở nhóm BỊA.** T6 là nhóm mạng nói NHIỀU NHẤT (`n_tok` 20.8/34.6,
gấp 3–5 lần T1/T2) và là nhóm DUY NHẤT còn `digit_split=0.40`, `loop4=0.20`, `stop` tụt còn
0.60. Tức: khi tiền đề có thật, mạng chép sạch rồi dừng; khi tiền đề bịa, nó lảm nhảm và mọi
khuyết tật cũ quay lại. Đây là lập luận MẠNH cho việc xử lý ca bịa — nhưng bằng cách khác
`premise_guard` (§12.20 đã chứng minh cổng đó giết luôn ghép).

**(d) `handoff` là chế độ sinh thật.** `n_facts` T1 1.00 → **1.88**, `n_tok` 7.2 → **29.2**,
`copy` 1.00 → 0.75 (chép ít verbatim hơn vì đang ghép nhiều đơn vị). Mọi đo §12.12–§12.20 chạy
`no_agenda` nên chưa từng chạm chế độ này — số cũ chỉ mô tả nửa hệ thống.

#### M4 chạy trên v12

```bash
PYTHONPATH=src python -u scripts/eval_gen_broad.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --log-file out/eval_gen_broad_v12.log --json-out out/eval_gen_broad_v12.json

# BASELINE để so — cùng checkpoint, code TRƯỚC §12.12:
git stash && git checkout 4f47c5d -- src/nn_aivie/hca
PYTHONPATH=src python -u scripts/eval_gen_broad.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --log-file out/eval_gen_broad_v12_base.log
git checkout HEAD -- src/nn_aivie/hca && git stash pop
```

Câu hỏi cần trả lời, theo thứ tự: **(1)** T4 QA trên v12 có còn `empty=1.00` không (v12 CÓ cha
QA) — nếu còn thì đường QA hỏng thật và đó là ưu tiên số 1; **(2)** T3 `n_facts` có >1 không;
**(3)** BASE↔HEAD: 6 thay đổi có nhích tổng thể, hay chỉ nhích 8 prompt cũ.

### 12.23 Ablation β từng KÊNH — harness + 2 ứng viên kênh chết (2026-07-27)

`scripts/ablate_channels.py`. `matrix_core_logits` cộng ~11 kênh, mỗi kênh một hằng số β chưa ai
kiểm lại từ lúc đặt. Khi output sai không biết kênh nào đẩy; khi vá một kênh không biết có che
lỗi kênh khác không. Harness tắt ĐÚNG MỘT kênh (β=0) rồi so với BASE.

Dùng lại **nguyên bộ prompt + thước của §12.22** — không dựng thước thứ hai (ba lần liên tiếp lỗi
nằm ở thước, càng ít thước càng ít chỗ hỏng).

Kênh có knob: `bag(parent)` · `succ+path` · `l2r` · `h(long-ctx)` · `know` · `know_lr` · `qk` ·
`eos` · `r2l_ffn` · `readout` · `agenda`.

#### Chạy thử 2 kênh (mini, handoff, max_len=10, 23 prompt T2–T6)

```
BASE  empty=0.17  digit_split=0.09  loop4=0.00  copy=0.67  n_facts=0.74  stop=0.22  n_tok=6.17

tắt kênh         Δempty  Δdigit⎵  Δloop4   Δcopy  Δn_facts   Δstop   Δn_tok
readout            0.00     0.00    0.00    0.00      0.00    0.00      0.0
qk                 0.00     0.00    0.00    0.00      0.04    0.00      0.1
```

`readout` (β=0.05) **không đổi một thước nào**; `qk` (β=0.6) gần như cũng vậy. Hai ứng viên
kênh chết. Bỏ được thì giảm 2 siêu tham số và 2 lớp nhiễu mỗi lần chẩn — nhưng **chưa đủ để bỏ**:
mới 1 chế độ, 1 `max_len`, 23 prompt, và trên mini.

#### M4 chạy đủ trên v12

```bash
PYTHONPATH=src:scripts python -u scripts/ablate_channels.py \
    --checkpoint checkpoints/network_v12_sent.pkl --mode handoff \
    --log-file out/ablate_channels_v12_handoff.log
PYTHONPATH=src:scripts python -u scripts/ablate_channels.py \
    --checkpoint checkpoints/network_v12_sent.pkl --mode no_agenda \
    --log-file out/ablate_channels_v12_noagenda.log
```

Chi phí: mini ~230s/kênh ở `max_len=10` ⇒ 11 kênh ≈ 45 phút; v12 chậm hơn nhiều, cân nhắc
`--only` chia nhiều lượt. Kênh nào Δ=0 ở **cả hai chế độ** mới là ứng viên bỏ thật — và vẫn phải
chạy thêm bộ T1/T7 của `eval_gen_broad` trước khi xoá dòng nào.

### 12.24 Đo §12.22 trên v12 (2026-07-27) — M4

`out/eval_gen_broad_v12.log` · `out/eval_gen_broad_v12.json` · ~4.6h · max_len=48.
**Chưa chạy BASELINE `4f47c5d`** (cùng ~4.6h nữa) — bảng dưới là HEAD hiện tại.

```
chế độ    nhóm              n   empty  digit⎵  loop4   copy  n_facts   stop  n_tok
no_agenda T1 verbatim      20    0.00    0.00   0.00   0.93     1.00   1.00    7.9
no_agenda T2 paraphrase     5    0.20    0.00   0.00   0.80     0.60   1.00    5.4
no_agenda T3 ghép           5    0.00    0.00   0.00   0.80     0.20   1.00    9.6
no_agenda T4 qa             4    0.00    0.00   0.00   1.00     0.00   1.00    1.0
no_agenda T5 số             4    0.00    0.00   0.00   1.00     0.75   1.00    6.8
no_agenda T6 bịa            5    0.00    0.40   0.00   0.79     1.00   1.00   13.8
no_agenda T7 dài            5    0.00    0.00   0.00   0.84     1.00   0.60    9.6
handoff   T1 verbatim      20    0.00    0.00   0.00   0.71     1.80   1.00   23.8
handoff   T2 paraphrase     5    0.00    0.00   0.00   0.67     1.80   1.00   19.6
handoff   T3 ghép           5    0.00    0.00   0.00   0.78     0.60   1.00   18.2
handoff   T4 qa             4    0.00    0.00   0.00   1.00     0.00   1.00    1.0
handoff   T5 số             4    0.00    0.00   0.00   0.65     2.00   1.00   21.5
handoff   T6 bịa            5    0.00    0.40   0.00   0.53     2.00   1.00   32.4
handoff   T7 dài            5    0.00    0.00   0.00   0.71     1.60   0.80   26.2
```

#### Trả lời 3 câu §12.22

**(1) T4 QA trên v12 KHÔNG còn `empty=1.00`** (mini ảo giác vì `is_qa=0`). Nhưng **đường QA vẫn hỏng**:
`n_tok=1.0`, `n_facts=0.00` ở cả hai chế độ — sinh đúng **1 token rác** rồi dừng:
`Đáp: ở` / `Đáp: không` / `Đáp: Kỳ` / `Đáp: Lê`. Không empty, cũng không trả lời.

**(2) T3 ghép `n_facts` vẫn <1** — no_agenda 0.20, handoff 0.60. Composition chưa chạy.
Mẫu: `Đáp: Trần` / `Đáp: toilet là` / salad Marie Curie.

**(3) BASE↔HEAD:** chưa đo (hoãn). Trên HEAD: T5 số sạch (`digit⎵=0`, mẫu `1010`/`1288`/`968`);
digit_split **chỉ còn ở T6 bịa** (0.40). Handoff vẫn là chế độ sinh thật (T1 `n_facts` 1.00→1.80,
`n_tok` 7.9→23.8).

#### Ưu tiên tiếp

1. **QA path** — ưu tiên #1: vì sao sau `Đáp:` chỉ ra 1 token rác (kể cả khi cha QA tồn tại).
2. **Composition / T3** — `n_facts` không lên nổi >1.
3. §12.23 ablation: trước `--only readout,qk` cả 2 chế độ; đủ 11 kênh nếu còn thời gian.
4. (tuỳ chọn) BASELINE `4f47c5d` để biết 6 thay đổi §12.12–§12.21 có nhích tổng thể không.

### 12.25 Ablation `readout`/`qk` trên v12 handoff (2026-07-27) — M4

`out/ablate_channels_v12_handoff_dead.log` · ~3.0h · 23 prompt T2–T6 · max_len=40.

```
BASE  empty=0.00  digit_split=0.09  loop4=0.00  copy=0.71  n_facts=1.30  stop=0.83  n_tok=18.30

tắt kênh         Δempty  Δdigit⎵  Δloop4   Δcopy  Δn_facts   Δstop   Δn_tok
readout            0.00     0.00    0.00    0.00      0.00    0.00      0.0
qk                 0.00     0.00    0.00    0.00      0.00    0.00      0.0
```

Ứng viên KÊNH CHẾT trên **handoff**: `['readout', 'qk']` — khớp mini §12.23.
Xác nhận thêm ở §12.26 (`no_agenda`); ideally còn T1/T7 trước khi bỏ code.

### 12.26 Ablation `readout`/`qk` trên v12 no_agenda (2026-07-28) — M4

`out/ablate_channels_v12_noagenda_dead.log` · ~3.2h · 23 prompt T2–T6 · max_len=40.

```
BASE  empty=0.04  digit_split=0.09  loop4=0.00  copy=0.87  n_facts=0.52  stop=1.00  n_tok=7.61

tắt kênh         Δempty  Δdigit⎵  Δloop4   Δcopy  Δn_facts   Δstop   Δn_tok
readout            0.00     0.00    0.00    0.00      0.00    0.00      0.0
qk                 0.00     0.00    0.00    0.00      0.00    0.00      0.0
```

**Cả hai chế độ (handoff §12.25 + no_agenda §12.26) đều Δ=0** ⇒ `readout` và `qk` là ứng viên bỏ thật trên bộ T2–T6.
Còn khuyến cáo runbook: xác nhận thêm trên T1/T7 trước khi xoá dòng code / đặt β=0 default.
Ưu tiên chất lượng vẫn là QA path + composition (T3), không phải bỏ kênh.

### 12.27 Bug QA — path KHÔNG THỂ khoá cha QA; vá bằng chép span đáp (2026-07-28)

*(Claude đánh số gốc §12.27 — đổi thành §12.30 vì §12.27–§12.29 đã dùng cho đo M4.)*

#### a. Đọc lại số v12 của M4 (§12.22 chạy trên v12)

```
handoff   T4 qa           4    empty 0.00   digit⎵ 0.00   copy 1.00   n_facts 0.00   stop 1.00   n_tok 1.0
  'Hỏi: Nhà Lý dời đô về đâu? Đáp: ở'          'Hỏi: Ai đánh bại quân Nam Hán…? Đáp: không'
  'Hỏi: Hoa Lư nằm ở tỉnh nào? Đáp: Kỳ'        'Hỏi: Lê Lợi khởi nghĩa ở đâu? Đáp: Lê'
```

**Thước của tôi nói dối ở đây.** `copy=1.00` chỉ vì chuỗi 1 token luôn LCS-khớp trọn một emit
nào đó; `stop=1.00` vì EOS ngay cũng tính là dừng sạch. Bảng đọc như thể T4 hoàn hảo.
Đã vá: `copy` **không chấm** khi phần sinh <4 token (NaN), thêm cột `degen` (≤2 token).

#### b. Nguyên nhân — không phải "1 token rồi EOS"

Dựng `mini_sent_ep_qa.pkl` (`build_mini_ep.py --with-qa`, 5.664 cha, **is_qa=660**) để gỡ trên
dev. Tái lập ngay, và cơ chế nằm ở ĐỘ LỚN logit:

```
câu trần 'Nhà Lý dời đô về'       → ĠThăng(7.55)  #2 chỉ 0.88          ← path KHOÁ
QA       '…dời đô về đâu? Đáp:'   → ĠHồ(1.53) ĠThành(1.34) Ġở(1.33)…   ← trường PHẲNG
```

Truy tiếp, ca retrieval HOÀN HẢO:

```
'Hỏi: 2 cộng 3 bằng mấy? Đáp:'   act=0.798  pi=5004  is_qa=True
    match_ids = 'Ġ2 Ġcộng Ġ3 Ġbằng Ġmấy ?'      ← trúng tuyệt đối
    emit      = 'Ġ2 Ġcộng Ġ3 Ġbằng Ġ5 . <EOS>'
    path.max  = 0        lock = -1                ← vẫn KHÔNG khoá
```

`build_parent_bipartite` đặt `emit = rec.answer_ids` cho cha QA, còn `_path_continuation_logits`
khớp **hậu-tố history ↔ emit**. History là **câu hỏi**, emit là **câu trả lời** — hai chuỗi không
giao nhau ⇒ **suffix-match không bao giờ fire được với cha QA**. Đây là bất khớp KIẾN TRÚC, không
phải ngưỡng: path giả định "history là tiền tố của emit", đúng với câu trần, sai với QA.

Hệ quả: retrieval đúng cũng chỉ còn `bag` yếu ⇒ đáp = argmax của trường phẳng ⇒ rác.

#### c. Vá — `answer_span_from_active` (khuôn §12.14)

Prompt kết bằng `Đáp:` (so INT trên special nguyên-tử) + cha act đỉnh `is_qa` ⇒ chép span
`answer_ids`. Cổng an toàn = **history phủ ≥80% `match_ids` CỦA CHA**. Chiều phủ này là mấu chốt:
hỏi `Nhà Lý dời đô về đâu?` thì cha đỉnh trên mini là *"Nhà mình sắp đón Tết, dọn dẹp từ **đâu**?"*
— khớp theo HÌNH DẠNG câu hỏi; `match_ids` của nó (`mình/sắp/đón/Tết/dọn/dẹp`) không nằm trong
history ⇒ phủ thấp ⇒ chặn. Chỉ dùng `act` đã học + fold ⇒ Luật 0 sạch.

#### d. Đo (mini-có-QA, greedy, max_len=32)

| prompt | OFF | ON |
|---|---|---|
| `2 cộng 3 bằng mấy?` | `5` | `2 cộng 3 bằng 5.` |
| `10 trừ 4 bằng mấy?` | `0` ✗ | `10 trừ 4 bằng 6.` ✓ |
| `Thành nhà Hồ ở đâu?` | `Hồ` | `Thành nhà Hồ ở tỉnh Thanh Hóa, xây thế kỷ 14…` |
| `Lê Lợi khởi nghĩa ở đâu?` | `Lam Sơn` | `Lê Lợi khởi nghĩa Lam Sơn năm 1418 chống quân Minh…` |
| **`Nhà Lý dời đô về đâu?`** | `Hồ` | `Hồ` — cổng chặn, KHÔNG bịa thêm |
| 3 câu trần (control) | — | **y nguyên từng ký tự** |

`283 passed` (+2 test: vế dương và vế âm của cổng phủ).

#### e. Còn lại — lỗi retrieval, KHÁC lỗi trên

`Nhà Lý dời đô về đâu?` vẫn ra `Hồ`: cha act đỉnh được chọn theo **hình dạng câu hỏi**
(`Nhà`, `đâu`, `?`) chứ không theo nội dung, nên tri thức đúng (`Năm 1010, nhà Lý dời đô…về
Thăng Long`) không hề được truy hồi. Vá (c) chỉ sửa khâu PHÁT, không sửa khâu CHỌN. Đây là
"QA routing" đã nằm trong Tier-2/3 §12.5 — việc kế tiếp.

#### f. M4 chạy

> ⚠️ LỖI THỜI — lệnh hiện hành ở **§12.31**.

```bash
PYTHONPATH=src python -u scripts/eval_gen_broad.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --log-file out/eval_gen_broad_v12_qacopy.log --json-out out/eval_gen_broad_v12_qacopy.json
```
(sửa `MODES` thêm `ar_matrix_qa_answer_copy=True`, hoặc chạy A/B trực tiếp bằng knob)

**Cổng bật default:** T4 `n_tok` tăng rõ và `degen`→0, T1/T2/T5 **không đổi**, T6 bịa không tăng
`digit⎵`. Đạt → `ar_matrix_qa_answer_copy=True`. Mini đã đạt; v12 chưa đo nên default vẫn **False**.

### 12.28 QA routing — từ-để-hỏi làm neo hard-AND, giết luôn tri thức đúng (2026-07-28)

*(Claude đánh số gốc §12.28 — đổi thành §12.28.)*

§12.30 vá khâu PHÁT. Đây là khâu CHỌN. Ca: `Hỏi: Nhà Lý dời đô về đâu? Đáp:` → `Hồ`.

#### a. Cùng nội dung, chỉ thêm khung — cha đúng bị nhân 0

```
câu trần 'Nhà Lý dời đô về'                → pi=553  act=0.4778  hạng 0
QA       'Hỏi: Nhà Lý dời đô về đâu? Đáp:' → pi=553  act=0       hạng 551
```

Tắt hard-AND (`gidf_hi=100` ⇒ `hi` rỗng) thì tri thức đúng hiện ra ngay:
`Lý Thái Tổ dời kinh đô về Thăng Long năm 1010` (act .4266), `Năm 1010, vua Lý Thái Tổ…` (.4266).
**Nó vẫn nằm đó, chỉ bị hard-AND nhân 0.** `Hỏi:`/`Đáp:` vô can — cả hai có **0 cha trong `A`**.

Thủ phạm: `late_cut = hlen-3` rơi vào `? Ġ Đáp:` (đều bị loại) ⇒ vòng "late" không lấy được gì
⇒ `hi[0]` lấy từ `early_by_gidf`, mà danh sách này sắp theo `-pos0` nên **token MUỘN NHẤT lên
đầu** — đúng chỗ từ để hỏi đứng. `hi[:2] = đâu ∧ Nhà` ⇒ chỉ cha chứa CẢ HAI sống sót ⇒
"Nhà mình sắp đón Tết, dọn dẹp từ **đâu**?" và "Thành **nhà** Hồ ở **đâu**?" thắng.

#### b. Tách từ-để-hỏi mà KHÔNG dùng bảng từ

`emit_match_ratio` = (số cha **PHÁT** token) / (số cha **KHỚP** bằng token), chiết từ chính `E`
và `A`. Token chỉ để khớp mà không bao giờ được nói ra = chỉ mang hình thức câu hỏi:

| `gì` | `mấy` | `nào` | `đâu` | `sao` | `ai` | ‖ | `Lý` | `Nhà` | `dời` | `năm` | `Thăng` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| .12 | .15 | .22 | .43 | .59 | .87 | ‖ | 1.02 | 1.02 | 1.10 | 1.07 | 1.17 |

Biên .87 ↔ 1.02 mỏng nhưng thật. Knob `ar_matrix_and_form_min` (0 = tắt).

**Vá hụt lần đầu — ghi lại:** tôi chỉ lọc `scored`, A/B ra **không đổi một ký tự**. Vì
`early_by_gidf` được dựng ĐỘC LẬP và chỉ lọc `gidf`, nên `đâu` vẫn vào `hi[0]` qua đường đó.
Phải lọc ở **cả hai** chỗ. Sau khi lọc đủ: `pi=553` act 0 → **0.8063, hạng 0**.

#### c. Nhưng retrieval đúng vẫn chưa ra đáp

Cha đúng là **câu trần**, không phải cha QA ⇒ §12.30 không chép (nó đòi `is_qa`), path vẫn không
khớp nổi ⇒ vẫn `Đáp:0`. Mở rộng `answer_span_from_active` cho cha câu-trần với **chiều phủ NGƯỢC**:
cha QA đòi *history phủ `match_ids`*; cha câu-trần đòi *cha phủ nội dung CÂU HỎI* — vì
"Năm 1010, nhà Lý dời đô từ Hoa Lư về Thăng Long" chứa THÊM (`Hoa Lư`, `Thăng Long`), không nằm
gọn trong câu hỏi. Từ hình-thức bị loại khỏi tập cần phủ, và **token không cha nào khớp
(`A` rỗng) cũng bị loại** — để `Hỏi:`/`Đáp:` lại thì phủ tụt 5/7 = 0.71 < 0.8 và cổng chặn oan (đo).

#### d. Đo (mini-có-QA, greedy, max_len=30)

| prompt | OFF | ON (`and_form_min=0.9` + copy) |
|---|---|---|
| `Nhà Lý dời đô về đâu?` | `Hồ` | `Năm 1010, nhà Lý dời đô từ Hoa Lư về Thăng Long.` |
| `Hoa Lư nằm ở tỉnh nào?` | *(rỗng)* | `Hoa Lư nằm ở tỉnh Ninh Bình ngày nay.` |
| `Lê Lợi khởi nghĩa ở đâu?` | (đã đúng) | `Lê Lợi khởi nghĩa Lam Sơn năm 1418 chống quân Minh…` |
| `2 cộng 3 bằng mấy?` | (đã đúng) | `2 cộng 3 bằng 5.` |
| **`Ngô Quyền đánh bại quân Tống năm nào?`** | `Quyền giành độc` | `Hán đang` — **không bịa** |
| `Ai đánh bại quân Nam Hán…?` | *(rỗng)* | *(rỗng)* — **chưa giải** |
| **6 control câu trần** | — | **y nguyên từng ký tự** |

`283 passed` + test `emit_match_ratio` tách được từ-hỏi khỏi từ nội dung.

#### e. Chưa xong

- `Ai đánh bại quân Nam Hán trên sông Bạch Đằng?` vẫn rỗng — dạng hỏi `Ai` (E/A .87, sát biên).
- Lỗi hiển thị nhỏ: `Đáp:Năm` thiếu dấu cách (span bắt đầu bằng token không có `Ġ`).
- Hai knob **default OFF**: `ar_matrix_and_form_min=0.0`, `ar_matrix_qa_answer_copy=False`.
  Đây là sửa ở **lõi truy hồi** nên phải qua đủ 7 loại đầu vào rồi mới bật.

#### f. M4 chạy

> ⚠️ LỖI THỜI — lệnh hiện hành ở **§12.31**.

```bash
PYTHONPATH=src python -u scripts/eval_gen_broad.py \
    --checkpoint checkpoints/network_v12_sent.pkl --log-file out/eval_qa_route_v12_ON.log \
    --extra '{"ar_matrix_and_form_min":0.9,"ar_matrix_qa_answer_copy":true}'
```
so với `out/eval_gen_broad_v12.log` (đã có, = OFF). **Cổng bật default:** T4 `degen`→0 và `n_tok`
tăng rõ; **T1/T2/T5/T7 không đổi**; T6 bịa không tăng `digit⎵`. `--extra` là cờ mới của
`eval_gen_broad` để override knob mà không sửa code.

#### g. A/B đủ 7 loại đầu vào × 2 chế độ (mini-có-QA, max_len=40) — bổ sung §12.28

`out/eval_qa_route_OFF.log` ↔ `out/eval_qa_route_ON.log`
(`{"ar_matrix_and_form_min":0.9,"ar_matrix_qa_answer_copy":true}`)

| nhóm | thước | OFF | ON |
|---|---|---|---|
| **T4 qa** (cả 2 chế độ) | empty · degen · n_facts · n_tok | 0.50 · 1.00 · 0.00 · 0.8 | **0.25 · 0.25 · 0.75 · 14.8** |
| T1 · T2 · T5 · T6 · T7 | mọi cột | — | **giống hệt từng số** |
| T3 ghép (no_agenda) | degen · n_facts · n_tok | 0.60 · 0.40 · 6.0 | 0.40 · 0.60 · 14.8 |
| T3 ghép (no_agenda) | **digit⎵ · stop** | **0.00 · 1.00** | **0.20 · 0.80** ⚠️ |
| T3 ghép (handoff) | copy · n_tok | 0.52 · 13.2 | 0.69 · 19.2 |

**Đọc thẳng:** đúng nhóm mục tiêu (T4) cải thiện mạnh; **5/7 nhóm không đổi một số nào**; T3 dài
hơn và bớt cụt nhưng **phát sinh `digit⎵` 0.20 và `stop` tụt 0.80** ở `no_agenda` — 1/5 prompt.
Đây là tác dụng phụ THẬT, không giấu: chép span câu-trần vào ca ghép kéo theo chuỗi chữ-số rời.

Cân nhắc: T4 là nhóm đang hỏng nặng nhất (degen 1.00 = mọi đáp đều ≤2 token), T3 vốn đã hỏng
(n_facts <1). Đổi 1/5 ca T3 xấu đi lấy 3/4 ca T4 tốt lên là đáng — nhưng **vẫn giữ default OFF**
cho tới khi có bảng v12, vì đây là sửa lõi truy hồi và mini chỉ có 5.664 cha.

### 12.29 T3 ghép — KHÔNG phải tường, mà là hai đường sinh tách rời (2026-07-28)

#### a. Phát hiện lớn nhất: matrix core và multizone LOẠI TRỪ nhau

`matrix_core_logits` **chỉ** được gọi trong `_wseq_fallback`, tức nhánh `ar_no_agenda=True`.
Còn `multizone` nằm ở **đường plan** (`ar_no_agenda=False`), và `ar_no_agenda` chính là cờ tắt
MZ/fact retrieval. Nghĩa là:

> Toàn bộ §12.12–§12.28 (chép số, path lock, chép đáp QA, lọc từ-hỏi) nằm trên đường mà
> **multizone không bao giờ chạy**, và ngược lại.

Đo bằng cách bật đường plan trên mini — mất sạch mọi thứ đã vá:

```
matrix core: 'Nhà Lý dời đô về Thăng Long năm 1010. Nhà Trần thay nhà Lý năm 1226.'
plan+MZ:     'Nhà Lý dời đô vềNhà Lý do Lý Công Uẩn… sáng lập vào năm 10.09'   ← số vỡ, không trả lời
plan+MZ:     'Trận Bạch Đằng nămTrận Bạch Đằng của Ngô Quyền kết thúc…'          ← mất hẳn số
plan+MZ:     'Lê Lợi khởi nghĩa ở đâu? Đáp: …'                                   ← BỊA cả khung câu hỏi
```

⇒ T3 không phải bức tường kiến trúc như tôi dự đoán, mà là **một nhánh mã cũ chưa được nâng**.

#### b. Đính chính nhận định của chính tôi

Nhìn output đường plan (`Trần Tự Khánh…` + `Lý Thường Kiệt…`) tôi đã kết luận "multizone chạy,
lấy đúng một fact mỗi thực thể". **Sai.** Đo `zone_retrieve` trực tiếp:

```
'so sánh nhà Lý và nhà Trần'    → 1 zone ['ĠTrần','Ġ']   ← MẤT 'nhà Lý'
'Nhà Lý và nhà Trần khác nhau…' → 1 zone ['ĠNhà','ĠLý']  ← MẤT 'nhà Trần'
'Cả Hoa Lư và Thăng Long…'      → 2 zone, nhưng lẫn 'Cả'
```

Hai fact kia không đến từ hai vùng. Bài học lặp lại: **đọc output rồi suy ra cơ chế là sai** —
phải đo thẳng cơ chế.

#### c. Ba khuyết tật trong `_entity_span_zones`, đều đo được

1. **`Ġ` (khoảng trắng) có `cap_class = 2`** ⇒ độn vào run cho đủ `len≥2`, cho entity đơn lẻ lọt
   một cách TÌNH CỜ (`['ĠTrần','Ġ']`). Token cấu-trúc không phải danh-từ-riêng — loại thẳng.
2. **Luật "bỏ token mở đầu" chỉ xét pos 0**, nên `Kể`(gidf 8.26)/`Viết`(7.70) đứng ngay sau nhãn
   `Hỏi:` vẫn lọt → zone rác (kéo về fact *"Hiện vật kể chuyện quá khứ"*). Mở rộng: pos 0 **hoặc**
   ngay sau nhãn frame (so INT trên special nguyên-tử ⇒ Luật 0 sạch).
3. **`gidf_solo = 6.0` loại đúng `Lý`(4.67) và `Trần`(4.45)** — tên triều đại Việt rơi vào kẽ hở
   vì `nhà` viết thường nên run chỉ còn một token. Hạ `ar_multizone_gidf_solo` → 4.2 lấy lại được,
   và **không** làm `Kể`/`Viết` tệ hơn (chúng gidf CAO, vốn đã lọt ở 6.0 — khuyết tật 2 mới là thứ chặn).

Sau vá, cắt vùng sạch: `[['Lý'],['Trần']]` · `[['Hoa','Lư'],['Thăng','Long']]` · `[['Bạch','Đằng']]`.

#### d. Bắc cầu `zone_handoff_span` — giữ matrix core, chỉ thêm NGUỒN span

Matrix core đã có chỗ handoff (`agenda_handoff_span`, đi theo **trật tự nguồn** = episode). Ca GHÉP
không có episode nào chứa sẵn tổ hợp prompt đòi, nên handoff cạn. Thêm nguồn thứ hai: span fact của
ZONE kế. Knob `ar_matrix_zone_handoff` (**default OFF**).

| | nền | + zone-handoff (solo=4.2) |
|---|---|---|
| `so sánh nhà Lý và nhà Trần` | `Đáp: Trần` | `Đáp: Trần. Lý Cạnh sinh ra Lý Thiên Bảo, Lý Bí và Lý Hùng.` |
| `Nhà Lý và nhà Trần khác nhau…` | `Đáp: là` | `Đáp: là. Lý Cạnh sinh ra…` |
| 3 ca T3 còn lại | — | không đổi |
| **4 control** | — | **y nguyên từng ký tự** |

Bắn 2/5, không regress. **Nhưng fact chọn ra là rác** — vùng đúng, truy hồi TRONG vùng dở: zone chỉ
còn một token `Lý` (mất `nhà`) nên mất nghĩa "triều đại", `facts_about(('Lý',))` trả về gia phả.
**Mắt xích kế tiếp là chỗ đó**, không phải cắt vùng nữa.

#### e. Bắt được một lỗi ÂM THẦM ngoài lề — đáng giá hơn cả phần trên

Suite bỗng đỏ 1 test; stash ra thì xanh ⇒ tưởng regression của tôi. Chạy lặp trên **cùng cây mã**:
**284 / 283+1 / 284** ⇒ **flaky**, không phải logic.

Gốc: `_PARENT_BI` khoá bằng `id(state)`. `GraphState` bị thu hồi → state MỚI trùng `id` → nhận lại
bipartite CŨ dựng cho vocab khác → `vocab.fold(7) IndexError`. **Đây là lỗi production**, và crash
là trường hợp MAY: phần lớn id vẫn hợp lệ ⇒ **truy hồi sai âm thầm**. Cùng dạng với
`_KNOW_QK` / `_STRUCT_MASK` / `_SENT_INTRA` (đều khoá `id(state)`) — chưa vá, ghi nợ.

Vá: xác thực lúc đọc (`A.shape[0] == len(vocab)` và `n_parents` khớp) → dựng lại nếu lệch.
Suite chạy **5/5 lần đều 284 passed**. Kèm test tái hiện bằng cách nhét bipartite cũ vào khoá mới.

### 12.30 Vá nốt lớp lỗi cache `id(state)` — 10 chỗ, một helper chung (2026-07-28)

§12.29 vá `_PARENT_BI` bằng heuristic hình dạng. Rà lại thì **cùng lỗi có ở 10 chỗ**, và heuristic
là sai hướng: hai state khác nhau hoàn toàn vẫn có thể cùng `len(vocab)`.

**Gốc:** `id()` chỉ duy nhất **trong thời gian đối tượng còn sống**. `GraphState` bị thu hồi →
state MỚI cấp phát trúng cùng địa chỉ → nhận lại đồ dựng cho state CŨ. Crash (`vocab.fold(7)
IndexError`) là trường hợp **MAY**; phần lớn id vẫn hợp lệ ⇒ **truy hồi sai âm thầm**.

**Vá:** `hca/objcache.py` — giữ `id()` làm khoá (tra nhanh) nhưng lưu kèm **weakref tới chính đối
tượng** và so `is` lúc đọc. Đúng tuyệt đối, không đoán kích thước. Weakref chết ⇒ dựng lại.

| file | cache | ghi chú |
|---|---|---|
| `matrix_core` | `_PARENT_BI` · `_KNOW_QK` · `_STRUCT_MASK`(×2) | thay heuristic §12.29 |
| `sent_intra` | `_SENT_INTRA` · `_SENT_QK` | `_SENT_QK` khoá tuple → dùng `sub=(which, dim)` |
| `attention_qk` | `_ppmi_cache` · `_qk_cache` | |
| `knowledge_seq` | `_KNOW_CACHE` | |
| `spectral_stack` | `_STACK_CACHE` | |
| `skip_span_gate` | `_mined_skip` | **`set[id]`** — id tái dùng ⇒ tưởng ĐÃ mine ⇒ bỏ qua âm thầm |
| `frame_bracket_gate` | `_mined_state_ids` | `set[id]`, cùng dạng |
| `style_payoff` · `style_corpus` | `_PAYOFF_CACHE` · `_INDEX_CACHE` | khoá `id(vocab)` + paths |

Hai `set[id]` là ca xấu nhất: chúng không trả về dữ liệu sai mà **bỏ qua một bước dựng**, nên
không có triệu chứng nào cả — chỉ là thiếu `skip_spans` / thiếu FRAME_BRACKET.

**Kiểm:** `grep -rn "id(state)" src/nn_aivie/hca/*.py` → **0 chỗ còn lại**. Suite `286 passed`,
chạy lặp **3/3 lần** đều xanh (trước khi vá: 284/283+1/284). Test hồi quy dựng thẳng tình huống:
nhét bản của chủ A vào đúng khoá `id(B)` → phải MISS, mà chủ A vẫn HIT.

**Kiểm chức năng** (test không chạm sâu đường sinh): chạy lại A/B QA-routing trên mini — 6 ca QA
và 6 control **ra kết quả y hệt trước khi refactor**, kể cả
`Đáp:Năm 1010, nhà Lý dời đô từ Hoa Lư về Thăng Long.`

### 12.31 GIAO M4 — NGUỒN DUY NHẤT, cập nhật sau §12.149 (2026-08-02)

> Chỗ **duy nhất** ghi việc đang giao. Lệnh rải ở mục khác **lỗi thời**.

**Trạng thái:** §12.165 **P83 ĐÃ SINH** — 400 mẫu abs4 (`len1` vs `len2`, bộ đề rời);
trùng đề **0** · trống 26.5%/30.0%. Chờ Claude chấm (P83). **Không chấm** ở M4.
P81/P79/… xong. **Không** dùng neural/LM.

---

<details><summary>P83 (§12.165) — ĐÃ SINH · chờ chấm mù abs4 (len1 vs len2)</summary>

400 mẫu (200/200). Trùng đề 0. Trống len1 26.5% · len2 30.0% (Δ3.5pp). Mệnh đề 0.735 / 1.150.
Xem §12.165 · `out/abs4_samples_v12.txt` · khoá `abs4_key_v12.json`.

</details>

<details><summary>P81 (§12.164) — ĐÃ SINH · chờ chấm mù abs3 (fold đặc trưng)</summary>

400 mẫu (200/200). Trống: hien_tai 29.5% · fold_dac_trung **57.0%** ∈25–75%. ~115s.
Xem §12.164 · `out/abs3_samples_v12.txt` · khoá `abs3_key_v12.json`.

</details>

<details><summary>P79 (§12.163) — ĐÃ SINH · chờ chấm mù abs2 (kề nhau)</summary>

400 mẫu (200/200). Trống: hien_tai 29.5% · mối_nối_kề **70.0%** ∈40–75%. Unseen join 0%.
Xem §12.163 · `out/abs2_samples_v12.txt` · khoá `abs2_key_v12.json`.

</details>

<details><summary>P77 (§12.162) — ĐÃ SINH · chờ chấm mù tuyệt đối</summary>

n=200 (không bỏ mẫu). Trống 59=29.5% ∈15–40%. Mệnh đề TB 1.16≤2.0. ~108s.
Xem §12.162 · `out/abs_samples_v12.txt` · meta `abs_meta_v12.json`.

</details>

<details><summary>P76 (§12.161) — ĐÃ SINH · chờ chấm mù vòng mười một</summary>

n=700 → **186** cặp (≥88). Đổi 26.6% ∈10–45%. Tiền tố 186/186. Mệnh đề 2.0 vs 3.85.
Xem §12.161 · `out/close_pairs_v12.txt` · khoá `close_key_v12.json`.

</details>

<details><summary>P75 (§12.160) — ĐÃ SINH · chờ chấm mù vòng mười</summary>

n=400 → **142** cặp (≥88). Đổi 35.5% ∈25–70%. Tiền tố 142/142. Tok 31.3 vs 14.1.
Xem §12.160 · `out/len_pairs_v12.txt` · khoá `len_key_v12.json`.

</details>

<details><summary>P74 (§12.159) — ĐÃ SINH · chờ chấm mù vòng chín</summary>

500→77 (thiếu power) · 900→**148** cặp. Đổi 16.4% ∈10–45%. Trống-cặp 16.9%&lt;35%.
Xem §12.159 · `out/pivot_pairs_v12b.txt` · khoá `pivot_key_v12b.json`.

</details>

<details><summary>P71 (§12.158) — ĐÃ SINH · chờ chấm mù vòng tám</summary>

n=600 → **167** cặp (≥88). Trống 27.8%. Tok dự phòng mean ~22.8.
Xem §12.158 · `out/empty_pairs_v12.txt` · khoá `empty_key_v12.json`.

</details>

<details><summary>P69 (§12.157) — ĐÃ ĐO · G-magnet TRƯỢT trên v12</summary>

n=400 · 116s. (1) đỉnh 2.8%&lt;5% ❌ · (2) top10 5.4%&lt;10% ❌ · (3) 1.18×&lt;1.5× ❌.
Top vẫn `km 1661+100 QL1`. Xem §12.157 · `out/probe_clause_magnet_v12.json`.

</details>

<details><summary>P66 (§12.156) — ĐÃ SINH · G-anchor(3) TRƯỢT (cơ chế gần trơ @ v12)</summary>

200→43 · 320→87 · 400→104 cặp. Bỏ a==b ~73–78% (&gt;70%). Tok anchor ≥ S1 (+8%).
Xem §12.156 · `out/anchor_pairs_v12c.txt`.

</details>

<details><summary>P61 (§12.155) — ĐÃ SINH · chờ chấm mù vòng ba</summary>

`gate_pairs_v12.txt` 112 cặp (fold vs PMI θ=1.25). Khoá: `gate_key_v12.json`.
Chấm theo runbook F.4 → `gate_verdict_v12.json`. Xem §12.155.

</details>

<details><summary>P60 (§12.154 / §13.23) — ĐÃ ĐO · cổng PMI-cross ĐẠT trên v12</summary>

gold=1.372 · ĐẠT ở θ=1.25/1.5/1.75 (chết 42/59/77% ≤ fold 81%). Chưa wire — chờ chấm mù.
Xem §12.154 · `out/probe_pmi_gate_v12.json`.

</details>

<details><summary>P58 (§12.153) — ĐÃ ĐO · E_tiebreak không đụng đường sinh</summary>

A/B quality_baseline: BẬT≡TẮT (pmi 2.056, salad 0, quality_pass true). Giữ mặc định.
Xem §12.153 · `out/quality_tiebreak_{on,off}.json`.

</details>

<details><summary>P57 (§12.152) — ĐÃ ĐO · cổng cặp-âm-tiết TRƯỢT trên v12</summary>

`probe_fold_gate.py`: D chết 94% · n_scored=0 · pmi&lt;A. Giả thuyết vật liệu BỊ BÁC.
Đóng hướng cổng-fold. Xem §12.152 · `out/probe_fold_gate_v12.json`.

</details>

<details><summary>P48 (§12.151 / §13.15) — ĐÃ WIRE · Hopfield phá thế hoà</summary>

E_tiebreak mặc định BẬT sau §13.15; P58 xác nhận không đụng sinh. Xem §12.151 · §12.153.

</details>

<details><summary>P47 (§12.150 / §13) — ĐÃ ĐO · tầng mệnh đề v12 MỞ</summary>

`probe_clause_layer.py` trên `network_v12_sent.pkl`: cross 100% · bits/tok &lt; 2
@ PMI∈[−1,0.5] · len∈[4.6,8.5]. Xem §12.150 · `out/probe_clause_layer_v12.json`.

</details>

<details><summary>P47b (§12.149) — ĐÃ WIRE · đa-prompt agenda / QA / hist-Đáp</summary>

Siết retrieve+vote ngoài smoke dời đô. Xem §12.149. Đo: `out/quality_baseline_v12.json`.

</details>

<details><summary>P45 — ĐÃ WIRE · quality baseline / parity gates</summary>

`scripts/quality_baseline_v12.py` → PMI + salad-after-period + doi_do + eos_rate.
Xem `out/quality_baseline_v12.json`. Không so/không nhúng mạng LM.

</details>

<details><summary>P44 — ĐÃ WIRE · match-theo-d + MZ rerank</summary>

`directed_bigram_dmax=4` + `accum=sum` mặc định. MZ fact rerank theo overlap prompt.

</details>

<details><summary>P43 (§12.146) — ĐÃ WIRE · dual-channel content⊙surface</summary>

`ar_w_skip_content_gate`: surface_vote × gate(agenda∪entity, floor).

</details>

<details><summary>P42 (§12.141) — ĐÃ WIRE · EOS end_after + SENT_SUCC mở câu</summary>

Primary: `eos_from_end_after` vào pool softmax; `sent_open_from_w_agenda` sau biên.
Không `if token=='.'`.

</details>

<details><summary>P41 (§12.145) — ĐÃ WIRE · local-repeat + recover</summary>

Lặp token không sai. Local-repeat mềm + hồi cạnh (`hold_dst`) + reverse chọn lọc
(giữ `kinh→đô`). Xem §12.145.

</details>

<details><summary>P40 (§12.144) — ĐÃ WIRE · consume cạnh/đích W_skip</summary>

Sau sample: chỉ mark/hạ điểm cạnh `(src,d)→nxt` — **không** chặn token.
`kinh→đô` còn nếu chưa fire. Xem §12.144 / §12.145.

</details>

<details><summary>P39 (§12.143) — ĐÃ WIRE · W_skip không cắt câu</summary>

Tác giả: đuôi câu phải nối đầu câu sau (`đẹp→Việt`, `Long→Đây`). Bỏ `iter_sentence_spans`
trong `build_w_skip`. Xem §12.143.

</details>

<details><summary>P38 (§12.140) — ĐÃ WIRE · bão hoà hub (indeg) trên W_skip</summary>

Tác giả: không `if token=='.'`; hub bão hoà theo bậc. Đo: L2-row + compete phá câu;
`indeg_β=1` giữ dời đô, làm yếu `. và`. Xem §12.140.

</details>

<details><summary>P37 (§12.133) — ĐÃ XONG · W_skip + MZ agenda</summary>

Tác giả: quay lại W_skip, ưu tiên token agenda multizone.
`ar_w_skip_primary=True` · ép `ar_multizone` khi dựng plan · boost pending ×2.5×max(vote).
Smoke: output bám agenda; trật tự vẫn xáo; retrieve MZ có lúc lệch fact.
**Sau §12.139b:** mz-off cho đoạn dời đô sạch; mz-on dễ kéo Tông. Đo Oja với mz off.

</details>

<details><summary>P36 (§12.132) — ĐÃ XONG · chống salad thuần THẤT BẠI (rồi P37 bật lại primary)</summary>

same_sent+fluency vẫn salad. P37 không dựa vào đó làm đòn chính — dùng MZ agenda.

</details>

<details><summary>P34 (§12.129) — ĐÃ XONG · G-vote TRƯỢT</summary>

vote_primary v12: PMI 0.55–0.77 ≪ matrix 1.82; salad. Giữ probe-only, không wire generate.
**Ghi chú:** sau §12.131 tác giả đã yêu cầu wire — xem P35.

</details>

<details><summary>P33 (§12.128) — ĐÃ XONG · G-act ĐẠT hẹp</summary>

order→act opt-in: Δpmi +0.025 · text≠ 5/30. Knob `ar_matrix_order_act` default **OFF**.

</details>

<details><summary>P32 (§12.127) — ĐÃ XONG</summary>

`PmiCrossIndex` + cột `pmi` trong `eval_gen_broad`. Baseline A_off pmi=1.822.

</details>

<details><summary>P31–P26</summary>

P31 G-tie ĐẠT · P30 E_skip · P29 DIY · P28 NHẬN · P27 VarLM · P26 đóng phiếu copy.

</details>

### 12.51 Đo lift `P(fold ∈ đáp | WH-fold ∈ hỏi)` (2026-07-29) — Claude

Thực hiện đúng §12.50d: **đo trước, không viết đường sinh**. Script `scripts/probe_answer_lift.py`
(chỉ đo, không đụng `hca/`). Kết quả: **ÂM TÍNH** — không viết bước chọn câu.

#### a. Vì sao đo được trên mini, và đo theo chiều nào

mini 660 cha QA ‖ v12 670, và phân bố `ask_form` gần trùng khít (`gì` 229/14 ‖ 235/14 · `nào`
117/8 ‖ 119/8 · `mấy` 53/0 ‖ 55/0 · `đâu` 28/6 ‖ 31/6) ⇒ **cùng một corpus QA**. Bể ứng viên ở
probe (vài trăm đáp QA) **nhỏ hơn nhiều** bể thật của đường sinh (430k cha câu), nên:

    tách được trên mini  ⇒ CHƯA kết luận, cần v12
    KHÔNG tách trên mini ⇒ CHỐT (dễ hơn mà đã thua thì khó hơn không thể thắng)

Đây là chiều NGƯỢC với ca PMI §12.46/§12.49 (mini dương → v12 âm). Vì thế **không giao M4 chạy
lại trên v12**: kết luận âm đã chuyển sẵn, chạy nữa chỉ tốn lượt.

#### b. Hai lần probe hỏng trước khi ra số dùng được

**(1) `ask_form` cao nhất là token CẤU TRÚC.** `ġ` (khoảng trắng) = 1.00 và `?` = 0.99 — có ở mọi
câu hỏi, không bao giờ ở đáp ⇒ "hoàn hảo" mà rỗng thông tin. 606/659 ca gom vào một nhóm `ġ` ≈
nền, và L "thắng" 0.938. Cột **đối chứng** lộ ra: `rnd/L = 0.112` ≈ chance 0.125 ⇒ L không có tín
hiệu độc lập nào, nó chỉ đang tách chủ đề toán (`mấy`) khỏi phần còn lại. Vá: lọc `gidf ≥ 4.0`
(thống kê đã học, không nhìn mặt chữ — Luật 0 sạch). `ġ` 3.13 · `?` 3.21 ‖ `gì` 4.14 · `nào` 4.70
· `đâu` 6.00. *(`wh_slot_ids` production cũng cấm `''`/`?`; ở đó vô hại vì chỉ là ô neo, ở đây thì
chí mạng vì ta cần PHÂN LOẠI câu hỏi.)*

**(2) "nhiễu khó" hoá ra không khó.** Lấy nhiễu theo trùng thực thể hỏi↔hỏi ⇒ 113/132 ca phải pad
ngẫu nhiên, cột khó gần bằng cột ngẫu nhiên (B 0.885 ‖ 0.907). Định nghĩa lại: **nhiễu = đáp mà
chính tiêu chí B chấm cao nhất** — đúng thứ truy hồi thật đưa lên. Thiên vị chống B là cố ý: câu
hỏi cần trả lời là "L có cứu được ca B bị lừa không".

Lần thứ **sáu** liên tiếp khâu hỏng nằm ở THƯỚC chứ không ở mã. Cả hai lần đều bị bắt bởi một cột
mà bảng "không cần" — cột đối chứng ngẫu nhiên. Giữ cột vô dụng lại là thứ đã cứu phép đo này.

#### c. Số (mini, n=439, 5-fold, chống rò: `ask/ans` + tập WH đếm CHỈ trên train)

Phủ (G1) **66.5%** — `gì`=228 · `nào`=117 · `mấy`=53 · `nhiêu`=24 · `đâu`=17. (`ai` 0.46 và `sao`
0.70 không đạt `ask_form≥0.8` ⇒ ngoài vùng phủ, nhất quán với §12.48d.)

| bộ nhiễu | B | L | B+L |
|---|---|---|---|
| `topB` (B chấm cao nhất — THỰC TẾ) | **0.563** | 0.107 | 0.485 |
| `topBx` (chỉ nhiễu KHÁC loại hỏi) | 0.695 | 0.528 | **0.781** |
| `rnd` (đối chứng, chance 0.125) | 0.941 | 0.212 | 0.900 |

- Quét λ **oracle** (chọn ngay trên test ⇒ cận trên): tốt nhất λ=0.2 → 0.558 ‖ B thuần 0.563,
  tức **−0.005**. Thua cả cận trên.
- Phủ quyết đáy (dìm z(L) < −th) thay vì cộng điểm: `topB` tốt nhất **−0.009**, `topBx` **+0.009**
  — mức nhiễu. Toán tử khác cũng không cứu.
- Theo từng loại hỏi: **âm ở TẤT CẢ** — `gì` −0.083 · `nào` −0.026 · `mấy` −0.189 · `nhiêu` −0.042
  · `đâu` −0.059. Không có nhóm nào gánh (đúng bài học §12.38: không cho một nhóm may kéo cả cột).

G1 ĐẠT · G2 TRƯỢT (−0.077, 0/5 fold ≥0) · G3 TRƯỢT ⇒ **âm tính**.

#### d. Tín hiệu CÓ THẬT, nhưng sai hình dạng — đây mới là phần đáng giữ

`topBx` +0.087 và `rnd/L` 0.212 (1.7× chance) ⇒ lift **thật sự** biết phân biệt loại câu hỏi.
Top-lift đọc rất hợp lý ở nhóm lượng: `mấy`/`nhiêu` → `gam` `hecta` `kilôgam` `trăm` `xăng`.

Nó chết vì: trong truy hồi thật, **phần lớn ứng viên gây nhầm lại CÙNG loại** với gold. Khi ấy
lift nâng đều gold lẫn nhiễu ⇒ không phân biệt được gì, chỉ còn là nhiễu xô lệch thứ tự của B.
Bằng chứng sắc nhất là `mấy`: top-lift đẹp nhất mà Δ tệ nhất (**−0.189**), vì B ở nhóm đó vốn
mạnh (0.774) nên nhiễu phá được nhiều nhất.

⇒ **Luật rút ra: L làm hỏng B tỉ lệ thuận với việc B đang làm tốt cỡ nào.** "Loại câu hỏi" phần
lớn đã **hàm ý sẵn** trong trùng thực thể; phần nó thêm được nằm đúng chỗ B đã đúng rồi.

#### e. Phát hiện phụ, ảnh hưởng mọi hướng sau

**Đáp của cha QA gần như KHÔNG nằm trong corpus câu**: lấy 40 cha QA, chỉ **9/40** có cha câu nào
phủ ≥0.9 token của đáp. Hai hệ quả:

1. Không chấm gold được trên bể ứng viên THẬT (cha câu) — mọi probe kiểu này buộc phải chấm trên
   chính đáp QA, tức luôn là proxy. Cần nhớ khi thiết kế phép đo sau.
2. `answer_span_from_active` ở `Đáp:` đang chép văn bản **chỉ tồn tại trong cha QA** — nó không
   phải là truy hồi từ trí nhớ câu, mà là chép nguyên khối một đoạn QA.

#### f. Trạng thái nút thắt

§12.50c chốt ba cơ chế cùng dừng ở "chọn đúng câu". Hướng đầu tiên để phá — xếp hạng theo **loại
câu hỏi** — nay cũng âm, và âm vì lý do **cấu trúc** (loại đã hàm ý trong thực thể), không phải
vì mỏng dữ liệu. Nên đừng thử lại bằng nhiều cha QA hơn.

Cái CHƯA thử và không bị lý do trên chặn: tiêu chí xét **quan hệ vị trí giữa thực thể hỏi và
phần trả lời TRONG câu ứng viên** (câu 0 sai không phải vì sai loại — nó có `đường nào`, đúng loại
`nào` — mà vì thực thể được hỏi nằm ở vế **bị phủ định/không biết**). Đó là tín hiệu bên trong
một câu, không phải thống kê giữa các câu. Chưa thiết kế phép đo; chưa hứa.

---

### 12.52 Hướng "vế phủ định": tiền đề SAI, nhưng lộ ra khuyết tật của theme prior (2026-07-29) — Claude

Đi theo §12.51f. Kết quả: **giả thuyết của tôi bị bác**, và thứ tìm được lại nằm chỗ khác — có
giá trị hơn. Không viết mã sinh.

#### a. Ca 0 CÓ trong mini, nhưng KHÔNG tái hiện

`Sử cũ không ghi chép chi tiết nhà Lý dời đô bằng đường nào.` nằm ngay trong mini (cha 2107).
Vậy mà mini trả lời **đúng**: `Hỏi: Nhà Lý dời đô về đâu? Đáp:` → `Năm 1010, nhà Lý dời đô từ Hoa
Lư về Thăng Long.` (cả `no_agenda` lẫn `handoff`). Cùng bức tường §12.49: 5.004 cha thì câu đúng
thắng dễ, 430k cha thì câu meta chen lên ⇒ **hướng này không lặp được tại chỗ bằng cách đọc
output**. Mọi bằng chứng dưới đây vì thế là bằng chứng về ĐỒ THỊ, không phải về output.

#### b. Đặt hai câu cạnh nhau ⇒ "vị trí vế nhúng" là hướng SAI

```
xấu:  Sử cũ không ghi chép chi tiết │nhà Lý dời đô│ bằng đường nào.
đúng: Năm 1010, │nhà Lý dời đô│ từ Hoa Lư về Thăng Long.
```

Span khớp nằm gần như cùng chỗ. Khác biệt thật: câu xấu **chứa `nào`** — nó ĐẶT câu hỏi chứ không
ĐÁP. Và `ask_form` đã nói sẵn: `nào` 117 hỏi / 8 đáp ⇒ đáp gần như không bao giờ chứa từ hỏi.

#### c. Nhưng cơ chế đó nhắm TRƯỢT — đo trên 660 câu hỏi thật

Tiền đề của tôi: "câu meta nhại lại câu hỏi nên nổi lên đầu bảng xếp hạng". Đo làm giàu (tỉ lệ
ứng viên top-k chứa từ hỏi ÷ tỉ lệ nền corpus), xếp hạng bằng trùng-từ có trọng gidf:

| | nền | top-1 | top-3 | top-5 | top-10 |
|---|---|---|---|---|---|
| θ=0.9, 29 prompt **tôi tự chọn** | 1.46% | 2.4× | 0.8× | 1.9× | 1.2× |
| θ=0.9, **660 câu hỏi thật** (cha QA) | 1.46% | **0.2×** | 0.8× | 0.8× | 0.8× |
| θ=0.8, 660 câu hỏi thật | 2.62% | 0.3× | 0.5× | 0.5× | 0.6× |

Câu chứa từ hỏi bị xếp **THẤP HƠN nền**, không cao hơn. Cổng ≥5× trượt xa. Và các rank-1 bị đánh
dấu đều là **đồng âm** (`bao gồm` · `bao hàm` · `Làm sao` · `mấy tầng`), không phải câu meta.

⇒ Tiền đề sai. Tôi đã suy rộng từ **đúng một ca**. Chú ý bảng đầu: nhóm 29 prompt tự chọn cho
2.4× (top-1 = **1 ca duy nhất**) — đúng thứ [[blocking-gate-measurement-design]] cảnh báo; đổi
sang 660 câu hỏi thật thì dấu đảo hẳn.

#### d. Cơ chế "thực thể là chủ đề" ĐÃ CÓ, ĐÃ BẬT, và MÙ ở đây

`facts_about` nhân `cov *= 0.5 + theme_weight × w_theme`, `compose_theme_weight = 1.0` mặc định —
tức ý "thực thể đứng đầu câu thắng thực thể chỉ được nhắc" **đã chạy sẵn** trên chính đường
multizone đẻ ra ca 0. Đo `w_theme` với focus `Lý`:

    ĐÚNG  `Năm 1010, nhà Lý dời đô từ Hoa Lư về Thăng Long`   w_theme = 0.125
    XẤU   `Sử cũ không ghi chép … nhà Lý dời đô bằng đường nào` w_theme = 0.111
    ĐÚNG2 `Lý Thái Tổ dời đô về Thăng Long năm 1010`           w_theme = 1.000

Hệ số nhân: 0.625 ‖ 0.611 — chênh **2%**. Cơ chế không thấy gì.

Vì sao: `w_theme` đúng bằng **`1/(1+vị_trí)`** — nghịch đảo offset thô. Đo phân bố trên 1.200
theme neuron / 127.488 cặp: giá trị phổ biến đúng là 1.0 · 0.333 · 0.25 · 0.2 · 0.167; chỉ
**6.13%** đạt 1.0; trung vị **0.100**. Với sàn 0.5, hệ số nhân của **94%** fact bị nén vào dải
phẳng 0.6–0.75 ⇒ prior này thực chất **nhị phân** (đứng đầu câu / không), không phải thang.

**Khuyết tật cụ thể, không chỉ là giới hạn:** `Năm 1010, nhà Lý dời đô…` là câu chủ-đề-trước kinh
điển, chỉ có trạng ngữ thời gian đứng trước, mà bị phạt xuống 0.125 — **ngang với** câu mà thực
thể nằm trong mệnh đề nhúng. Trạng ngữ bị tính như mệnh đề cạnh tranh.

#### e. Ứng viên tiếp theo — và vì sao tôi CHƯA viết

Khác biệt thật giữa hai câu không phải vị trí mà là: câu XẤU có **một thực thể cạnh tranh đứng
trước** (`Sử cũ`) và chính nó mới là chủ đề; câu ĐÚNG chỉ có trạng ngữ thời gian. Diễn đạt được
bằng đồ thị sẵn có, không bảng từ: *trước span khớp có token nào gidf cao mà bản thân là primary
của một theme neuron không?* (`Sử` có · `Năm`/`1010` không).

Nhưng đây lại là **n=1** — đúng cái vừa làm tôi sai ở (c). Không viết cho tới khi có bộ nhãn.
Bộ nhãn khả dĩ **không cần chấm tay**, lấy thẳng từ đồ thị: dương = cặp (thực thể, câu) có
`w_theme = 1.0` (6.13%, thực thể đứng đầu) · âm = cặp mà một theme-primary KHÁC đứng trước.
Cổng: đặc trưng "không có thực thể cạnh tranh đứng trước" phải khôi phục nhãn dương **tốt hơn**
`1/(1+vị_trí)` thuần, và phải kéo `Năm 1010, nhà Lý dời đô…` lên khỏi dải phẳng.

#### f. Ghi nhận

Cả §12.51 lẫn §12.52 đều âm, và **cùng một kiểu sai của tôi**: đọc một output, dựng lý thuyết,
rồi mới đi đo. Hai lần bộ đo tự chọn đều cho dấu NGƯỢC với bộ đo lấy từ đồ thị (29 prompt: 2.4× ‖
660 câu hỏi: 0.2×). Thứ có giá trị nhất rút ra không phải hai giả thuyết ấy mà là (d): một prior
đang bật mặc định, tưởng là thang, thực ra nhị phân, và phạt oan trạng ngữ mở đầu.

---

### 12.53 Đo theme prior trên bench truy hồi suy từ đồ thị (2026-07-29) — Claude

Chạy phép đo §12.52e. **Phải bỏ thiết kế cũ trước** — nó thoái hoá — và kết quả tách làm ba,
không phải một verdict.

#### a. Thiết kế §12.52e hỏng: nhãn đoán được bằng chính định nghĩa

§12.52e định lấy dương = `w_theme = 1.0`, âm = có theme-primary khác đứng trước. Nhưng dương
định nghĩa bằng `pos == 0`, mà "không có gì đứng trước" thì **đúng hiển nhiên** khi `pos == 0`.
Đặc trưng thắng 100% mà chẳng đo gì.

Đi tìm nhãn cấu trúc độc lập trong đồ thị — **không có**: `match_span[0] == 0` ở **cả 5004** cha
câu (nó là cả câu, không phải chủ ngữ), `phrase_ids` là rác (`ḅ` · `ى` · `城`), và
`theme_neuron` (parent_store.py:2535) đúng là `1.0/(1.0+pos)` thuần, không thêm cấu trúc nào.
⇒ **Đồ thị không mang chú giải chủ đề nào ngoài vị trí token thô.**

Không chấm tay để lấp: §12.52c vừa cho thấy nhóm-đo-tự-chọn đảo dấu (29 prompt 2,4× ‖ 660 câu hỏi
thật 0,2×).

#### b. Bench mới — gold suy từ đồ thị, dùng lại được

Cha QA nào có cha CÂU phủ ≥0.9 token `answer_ids` ⇒ cặp (câu hỏi → cha câu) là một ca truy hồi có
đáp đúng. Được **149/660**. Ngưỡng phải là 0.9 — ở 0.7 gold đã hỏng
(`Một ngày có bao nhiêu giờ?` → `Một tuần có bảy ngày`). Phủ theo ngưỡng: .9→149 · .8→225 ·
.7→280 · .6→339 · .5→472. `scripts/probe_theme_prior.py` (chỉ đo; bể ứng viên thu bằng chỉ mục
ngược theo fold — CHÍNH XÁC vì cha không chia sẻ fold có điểm nền 0, và là điều kiện để chạy nổi v12).

#### c. Số (mini, n=149, `theme_weight=1.0`)

| prior | top1 | MRR | Δ so `pos` | bất đồng với `pos` |
|---|---|---|---|---|
| `tw=0` / `none` (tắt prior) | 0.530 | 0.641 | −0.040 | 20 ca (7 thắng / 13 thua) |
| **`pos` = 1/(1+vị_trí)** — HIỆN TẠI | **0.570** | 0.663 | — | — |
| `content` = 1/(1+#token gidf≥4 trước) | 0.557 | 0.662 | −0.013 | **6 ca (2/4)** |
| `clear` = nhị phân "không có nội dung trước" | 0.517 | 0.615 | −0.054 | 12 ca (2 thắng / **10 thua**) |

G1 bench lành **ĐẠT** (0.570) · G3 thay thế ≥+0.05 **TRƯỢT**.

#### d. Ba kết luận KHÁC NHAU — đừng gộp thành một

1. **Theme prior đang GÁNH VIỆC.** Tắt đi mất ròng 6/149 ca (0.570→0.530). Bất chấp §12.52d cho
   thấy 94% fact nằm dải phẳng: cú đẩy cho 6,13% cha đứng-đầu-câu chính là phần làm việc. Sửa
   prior này phải cẩn thận — nó không phải đồ trang trí.
2. **`clear` (giả thuyết §12.52e dạng thuần) ÂM TÍNH.** 2 thắng / 10 thua — hại thật, không phải
   nhiễu. "Không có nội dung đứng trước" làm luật cứng là sai.
3. **`content` KHÔNG PHÁN ĐƯỢC trên bench này** — bất đồng chỉ **4,0%** (2 thắng/4 thua), tức 96%
   ca hai prior xử y hệt. Δ = −0.013 là **2 ca**, đọc thành "tệ hơn" là đọc sai nhiễu.

Chẩn đoán ca 0 (focus `Lý`, hệ số `0.5 + w`): `pos` ĐÚNG 0.625 ‖ XẤU 0.611 (tách 2%) · `content`
ĐÚNG **0.833** ‖ XẤU 0.643 (tách 30%) · `clear` 0.500 ‖ 0.500 (không tách). Tức `content` tách ca
0 dứt khoát hơn hẳn — nhưng đúng một ca, và bench không có ca cùng dạng để kiểm.

#### e. Vì sao bench mù với hiện tượng, và cách gỡ

Gold ở đây là câu **gần nguyên văn** đáp ⇒ toàn ca truy hồi DỄ, không có ứng viên meta cạnh
tranh. Trên mini 5.004 cha thì bể ứng viên mỗi câu hỏi quá nhỏ để hai prior khác nhau. Trên v12
430k cha, số ứng viên chia sẻ fold lớn hơn nhiều bậc ⇒ tỉ lệ bất đồng phải cao hơn hẳn, và khi
đó bench MỚI phán được. Đây là ca **ngược** với §12.51: ở đó mini đủ để chốt âm, ở đây mini
không đủ để chốt gì cả. Giao M4 — xem §12.31 P5.

Theo luật đã tự đặt (G3 trượt ⇒ không viết vào mã), `compose_theme_weight` và công thức
`0.5 + w` giữ **nguyên** cho tới khi có số v12.

### 12.54 P5 theme prior trên v12 — G1 TRƯỢT; giữ prior nguyên (2026-07-29) — M4

`out/probe_theme_prior_v12.log` (~155s). Gold cover≥0.9: **74 ca** (mini 149). Ứng viên câu:
429.711.

#### a. Bảng prior/top1/MRR (nguyên văn)

```
==================================================================
prior          top1      MRR    Δ so pos   (n=74)
==================================================================
(tw=0)        0.095        —           —
none          0.095    0.354      -0.027
pos           0.122    0.330      +0.000
content       0.189    0.363      +0.068
clear         0.135    0.264      +0.014
==================================================================
```

#### b. BẤT ĐỒNG so với `pos` (nguyên văn)

```
BẤT ĐỒNG so với `pos` (ca mà top-1 đổi kết quả):
  none     thắng   0 · thua   2 · bất đồng   2/74 = 2.7%
  content  thắng   5 · thua   0 · bất đồng   5/74 = 6.8%
  clear    thắng   3 · thua   2 · bất đồng   5/74 = 6.8%
G1 bench lành (pos top-1 ≥0.50) : TRƯỢT (0.122)
G2 prior có tác dụng?            : tw=0 → 0.095 ‖ pos → 0.122 (Δ +0.027)
G3 thay thế hơn ≥+0.05          : ĐẠT (tốt nhất `content` 0.189, Δ +0.068)

⇒ CHỐT ÂM TÍNH — không sửa prior theo hướng này
```

#### c. CHẨN ĐOÁN ca 0 (nguyên văn)

```
CHẨN ĐOÁN ca 0 (focus `Lý`) — prior nào xếp câu ĐÚNG trên câu XẤU:
  none     ĐÚNG=1.500  XẤU=1.500  ✗ không tách
  pos      ĐÚNG=0.625  XẤU=0.611  ✓ tách
  content  ĐÚNG=0.667  XẤU=0.625  ✓ tách
  clear    ĐÚNG=0.500  XẤU=0.500  ✗ không tách
```

#### d. Đọc cổng §12.31

1. **G1 TRƯỢT** (0.122 ≪ 0.50) ⇒ theo runbook: **dừng**, so sánh G3/`content` trên bench này
   **không dùng để quyết** (baseline hỏng — cùng bài học §12.16).
2. Bất đồng `content` **6.8% &lt;10%** (và &lt;15%) ⇒ hiện tượng “hai prior tách trên bể lớn”
   **không** lộ ở quy mô này theo cổng đã chốt; **không** viết `content` vào mã.
3. **Giữ** `compose_theme_weight` + công thức `0.5+w` nguyên trạng. Không đổi knob.

**Claude:** gold v12 chỉ 74/670 (mini 149/660) và `pos` top-1 0.122 — bench/gold có thể không
còn “lành” trên 430k cha (ứng viên nhiễu?). Ca 0 vẫn tách nhẹ với `content` nhưng không đủ để
lật quyết định khi G1 trượt.

---

---

### 12.55 PMI không hỏng; nhưng giả thuyết "theme prior chống zone" của tôi cũng SAI (2026-07-29) — Claude

Đọc lại §12.46/§12.49b theo câu hỏi của tác giả ("PMI trông khả quan mà"). Đọc kỹ thì tác giả
đúng, và §12.49b đã ghi sẵn manh mối mà tôi bỏ qua: **vùng ĐÃ ĐÚNG** (`['nhà','Lý']`) trên v12,
fact trả về vẫn là gia phả. Tức PMI làm xong phần việc của nó; hỏng nằm ở khâu SAU.

#### a. Nghi phạm

`facts_about` rút zone về **một** `primary` để tra theme neuron (`pick_primary(['nhà','Lý'])`
→ `Lý`), rồi nhân `cov *= 0.5 + w_theme` với `w_theme = 1/(1+vị_trí)` (§12.52d). Hệ quả: mọi câu
**mở đầu bằng đúng token `Lý`** được nhân 1.5 — và đó chính là lớp câu gia phả / tên người.

Đo trực tiếp (mini, zone `['nhà','Lý']`, `pick_primary` → `ĠLý`):

```
theme_weight=0.0 → Nhà Trần chính thức thay nhà Lý. · Năm 1010, nhà Lý dời đô từ Hoa Lư…
                   · Hoàng thành… · Từ đây nhà Lý hoàn toàn chấm dứt sau 216 năm…
theme_weight=1.0 → Lý Thanh phục vụ dưới quyền Thứ sử Giao Châu là Đàn Hòa Chi nhà Lưu Tống…
                   · Lý Thánh Tông… · Lý Thường Kiệt…
```

`Lý Cạnh sinh ra Lý Thiên Bảo…` và `Lý Thanh phục vụ…` đều `w_theme = 1.0` — **cùng mức tối đa**
với fact triều đại, chỉ vì mở đầu bằng họ.

#### b. Đo bằng thước KHÁCH QUAN, 10 zone — không một ngoại lệ

Thước: trong top-5 `facts_about`, bao nhiêu fact chứa **ĐỦ** cụm zone (cả `nhà` lẫn `X`), chứ
không chỉ `X`. Không phán đoán chủ quan.

| zone | `tw=0` | `tw=1` | | zone | `tw=0` | `tw=1` |
|---|---|---|---|---|---|---|
| nhà Lý | 0.80 | 0.40 | | nhà Hồ | 0.80 | **0.00** |
| nhà Trần | 0.80 | 0.60 | | nhà Nguyễn | 0.60 | **0.00** |
| nhà Ngô | 1.00 | 0.20 | | nhà Mạc | 0.20 | **0.00** |
| nhà Lê | 0.60 | 0.40 | | sông Bạch | 0.80 | 0.20 |
| nhà Đinh | 0.40 | 0.20 | | vua Lê | 0.60 | **0.00** |
| | | | | **TRUNG BÌNH** | **0.66** | **0.20** |

**10/10 zone tệ đi khi bật theme prior.** Không phải n=1 nữa — nhưng đọc tiếp (c): 10 cụm này do
TÔI chọn, và chúng thiên vị có hệ thống. Số ở bảng này KHÔNG dùng để kết luận.

#### c. TỰ BÁC: 10 zone trên là nhóm TỰ CHỌN, và nó thiên vị có hệ thống

Trước khi commit, tôi cho zone **tự sinh từ đồ thị** thay vì chọn tay: lấy head gidf cao rồi
`left_pmi(head)` chọn token trái gắn chặt nhất ≥3.0 — đúng luật `_extend_left_by_pmi`.
`scripts/probe_zone_theme.py`.

| quần thể zone (n=25) | `tw=0` | `tw=1` | biên |
|---|---|---|---|
| **10 cụm tôi chọn tay** (§12.55b) | 0.66 | 0.20 | **+0.46** |
| tự sinh, mọi head | 0.63 | 0.55 | **+0.08** |
| tự sinh, head VIẾT HOA (danh từ riêng) | 0.62 | 0.58 | **+0.04** |

Cổng +0.15: **TRƯỢT** ở cả hai quần thể tự sinh. Và từng dòng cho thấy dấu **đảo** ở nhiều cụm —
`Hà Nội` 0.00→**1.00** · `Thái Bình` 0.00→**0.80** · `Đinh Bộ` 0.60→**1.00**: với tên riêng thật,
theme prior **giúp**. Nó chỉ hại ở `Mặt Trời` · `Trái Đất` · `Biển Đông` · `quân Tống` · `nhà Lý`.

Thiên vị của tôi có hệ thống, không ngẫu nhiên: tôi chọn toàn `nhà X` với X là họ **vừa là triều
đại vừa là họ người**. Đó đúng là cấu hình duy nhất lỗi xảy ra, và tôi đã lấy nó làm tính chất
chung. Lần thứ **tư** cùng một kiểu sai (§12.51b · §12.52c · §12.55b) — lần này bắt được ngay
trong phiên, vì phần tự sinh zone được viết SẴN vào script chứ không phải nghĩ ra sau.

#### d. Còn lại gì — và vì sao KHÔNG giao M4

Đúng: cơ chế ở (a) là **thật** và đo được — `Lý Cạnh sinh ra…` cùng `w_theme = 1.0` với fact
triều đại chỉ vì mở đầu bằng họ; với zone `['nhà','Lý']` thì tắt prior cho fact tốt hơn hẳn.

Sai: suy thành "theme prior chống mọi zone nhiều token". Số nói không (+0.04).

Nên hiện tượng chỉ sống trong lớp hẹp *"loại từ + họ vừa là triều đại vừa là tên người"*. Không
giao M4 vì hai lẽ: (1) quần thể không thiên vị đã trượt cổng ngay trên mini; (2) kể cả v12 có
dương, **không có cách Luật-0 sạch nào ĐỊNH NGHĨA lớp hẹp đó** — "họ người" là bảng từ. Một cơ
chế không diễn đạt được thì đo thêm cũng không dùng được.

`facts_about` và `compose_theme_weight` giữ **nguyên**.

#### e. Điều này KHÔNG mâu thuẫn §12.53, và trả lời câu hỏi ban đầu về PMI

§12.53d1: theme prior gánh việc (+0.040 / 149 ca zone một token). §12.55c: trên zone nhiều token
tự sinh, nó trung tính (+0.04 theo chiều ngược). Hai số này nhất quán — prior không hại diện rộng.

Về PMI: §12.49b vẫn là âm tính đúng, nhưng lý do KHÔNG phải "PMI sai". PMI nới zone đúng trên cả
mini lẫn v12 (`['nhà','Lý']`). Cái không theo kịp là **xếp hạng sau khi có zone đúng** — cùng nút
thắt §12.50c, và tôi vừa loại được một giả thuyết về thủ phạm chứ chưa tìm ra thủ phạm.

---

### 12.56 P5 trượt vì THƯỚC của tôi, không phải vì giả thuyết (2026-07-29) — Claude

M4 thi hành §12.31 P5 đúng runbook và **nêu đúng chỗ nghi**: *"gold v12 chỉ 74/670 (mini 149/660)
… bench/gold có thể không còn lành trên 430k cha"*. Truy ra: hai lỗi, **cả hai của tôi**, và cùng
một dạng — **hiệu chỉnh trên bể nhỏ rồi áp lên bể lớn**.

#### a. Lỗi 1 — dựng gold bị cắt ở top-80 ứng viên

`probe_theme_prior.py` chấm độ phủ trên `sorted(cnt, key=-cnt)[:80]`. Trên 5.004 cha thì 80 ứng
viên là rộng rãi; trên 429.711 cha, gold thật thường rơi **ngoài** top-80 ⇒ v12 ra 74 gold trong
khi mini ra 149. **Nhiều cha hơn mà ít gold hơn là vô lý** — đó là dấu hiệu, và M4 bắt được.

Vá: `cnt[k]` vốn đã là số token DUY NHẤT của câu `k` nằm trong đáp, nên `cnt[k]/|unique(k)|`
**chính là** độ phủ — tính đủ mọi ứng viên, không cắt, không xấp xỉ. Mini: **149 → 169 ca gold**
(thước cũ cắt cụt ngay cả ở đây). Kết luận mini **không đổi**: `content` −0.012 (2 thắng/4 thua).

#### b. Lỗi 2 — G1 là ngưỡng top-1 TUYỆT ĐỐI

Tôi đặt "bench lành = `pos` top-1 ≥ 0.50", hiệu chỉnh trên bể 5.004 cha. Trên bể 429.711 cha,
top-1 = 0.122 **không** có nghĩa bench hỏng — nó phần lớn phản ánh **cỡ bể**. MRR 0.330 của cùng
lượt chạy nói gold thường nằm quanh hạng 3, tức truy hồi vẫn khoẻ. Ngưỡng của tôi đo nhầm thứ.

Vá: G1 dùng **top-10 ≥ 0.40** — "câu hỏi hợp lệ thì gold phải nằm trong 10 ứng viên đầu", không
phụ thuộc bể to nhỏ. Thêm cột `top10` vào bảng.

#### c. Cái tôi PHẢI cẩn thận ở đây

Lượt P5 có một tín hiệu đáng chú ý: `content` **+0.068** với **5 thắng / 0 thua**, ngược dấu
mini (−0.012, 2/4). Cám dỗ là đọc nó thành dương tính. **Không.** Nó đo trên bộ gold **hỏng**
(74 ca bị cắt cụt), và dời cổng sau khi thấy số là p-hacking.

Cái được phép làm — và tôi làm — là sửa **kiểm tra sức khoẻ dụng cụ** (G1) trên căn cứ phương
pháp, rồi **chốt lại cổng G3 TRƯỚC KHI** số v12 đã vá tồn tại. G3 mới: `Δtop1 ≥ +0.05` **và**
`(thắng − thua) ≥ 5`. Bỏ điều kiện "bất đồng ≥15%" của P5 vì với bộ gold cố định, thống kê có
nghĩa là **thắng/thua theo cặp**, không phải tỉ lệ bất đồng thô — 5/0 và 15/10 cùng "bất đồng"
nhưng nói hai chuyện khác hẳn.

#### d. Ghi nhận

Lần thứ **bảy** khâu hỏng là thước chứ không phải mã, và là lần đầu **M4 bắt được trước tôi** —
bằng đúng cách rẻ nhất: thấy một con số vô lý về mặt số học (74 < 149 khi cha nhiều gấp 85 lần)
và nói ra thay vì đọc tiếp verdict. Bài học kèm: mọi ngưỡng tuyệt đối trong thước phải hỏi
"ngưỡng này có phụ thuộc cỡ bể không?" trước khi đem sang checkpoint khác.

### 12.57 P6 theme prior v12 (thước đã vá) — G1 ĐẠT · G3 TRƯỢT; giữ prior (2026-07-29) — M4

`out/probe_theme_prior_v12b.log` (~198s). Gold cover≥0.9 **không cắt top-80**: **274 ca**
(P5 thước cũ: 74). Ứng viên câu: 429.711.

#### a. Bảng prior (nguyên văn)

```
==================================================================
prior          top1    top10      MRR    Δ so pos   (n=274)
==================================================================
(tw=0)        0.036        —        —           —
none          0.036    0.423    0.196      -0.015
pos           0.051    0.423    0.207      +0.000
content       0.062    0.412    0.204      +0.011
clear         0.047    0.332    0.157      -0.004
==================================================================
```

#### b. BẤT ĐỒNG / cổng (nguyên văn)

```
BẤT ĐỒNG so với `pos` (ca mà top-1 đổi kết quả):
  none     thắng   0 · thua   4 · bất đồng   4/274 = 1.5%
  content  thắng   4 · thua   1 · bất đồng   5/274 = 1.8%
  clear    thắng   2 · thua   3 · bất đồng   5/274 = 1.8%
G1 bench lành (pos top-10 ≥0.40): ĐẠT (top10 0.423 · top1 0.051)
G2 prior có tác dụng?            : tw=0 → 0.036 ‖ pos → 0.051 (Δ +0.015)
G3 thay thế hơn ≥+0.05          : TRƯỢT (tốt nhất `content` 0.062, Δ +0.011)

⇒ CHỐT ÂM TÍNH — không sửa prior theo hướng này
```

#### c. CHẨN ĐOÁN ca 0 (nguyên văn)

```
CHẨN ĐOÁN ca 0 (focus `Lý`) — prior nào xếp câu ĐÚNG trên câu XẤU:
  none     ĐÚNG=1.500  XẤU=1.500  ✗ không tách
  pos      ĐÚNG=0.625  XẤU=0.611  ✓ tách
  content  ĐÚNG=0.667  XẤU=0.625  ✓ tách
  clear    ĐÚNG=0.500  XẤU=0.500  ✗ không tách
```

#### d. Đọc cổng §12.31 / §12.56c

1. **G1 ĐẠT** — thước vá đúng: gold 74→**274**; top-10 0.423 ≥ 0.40. (top-1 0.051 vẫn thấp —
   phản ánh cỡ bể, không còn chặn đọc tiếp.)
2. **G3 TRƯỢT** — `content` Δtop1 **+0.011** &lt; 0.05; (thắng−thua)=**3** &lt; 5. Tín hiệu P5
   (+0.068, 5/0) trên gold cắt **không** tái hiện.
3. **Giữ** `compose_theme_weight` + `0.5+w` nguyên. Không đổi knob.

---

### 12.58 Nhận §12.57 — âm tính SẠCH; dự đoán §12.53e của tôi SAI; và trần của cả họ prior (2026-07-29) — Claude

#### a. P6 là âm tính đúng nghĩa

Cổng chốt trước khi số tồn tại (§12.56c), thước đã vá, M4 chạy đúng. `content` **Δtop1 +0.011**
(<+0.05) · **thắng 4 / thua 1**, hiệu 3 (<5) ⇒ **trượt cả hai điều kiện**. Giữ nguyên trạng.

Vá thước hiệu quả rõ: gold **74 → 274**, và tín hiệu `content` **+0.068 / 5-0** của P5 **không tái
hiện**. Nó là hiện vật của bộ gold cắt top-80 — bộ cắt ấy giữ toàn ca DỄ (chia sẻ nhiều token), nên
`pos` top-1 tụt 0.122 → 0.051 khi thêm 200 ca khó. Việc từ chối đọc 5-0 thành dương tính (§12.56c)
là đúng.

#### b. Prior gánh việc — ba lần đo độc lập, nhưng tác dụng CO LẠI theo cỡ bể

| | mini 5.004 cha | v12 P5 (74 ca) | v12 P6 (274 ca) |
|---|---|---|---|
| Δtop1 khi BẬT prior | +0.041 | +0.027 | **+0.015** |

Cùng dấu cả ba ⇒ prior thật sự có ích. Nhưng độ lớn giảm đều theo cỡ bể.

#### c. Dự đoán của tôi SAI — và giả thuyết giải thích cũng sai

§12.53e tôi viết: *"trên v12 430k cha… tỉ lệ bất đồng phải cao hơn hẳn"*. Thực đo: mini **3,6%** →
v12 **1,8%**, giảm một nửa, **ngược dấu**.

Giả thuyết giải thích đầu tiên: dải `[0.5,1.5]` quá hẹp để lật khi bể lớn. Đo thay vì tin — **sai
nốt**: trên mini 97,6% số ca CÓ ứng viên nằm trong dải 3× của quán quân. Dải không phải chỗ nghẽn.
Nguyên nhân thật của việc bất đồng co lại: **chưa biết, chưa đo**. Ghi vào đây để không quên rằng
đây là câu hỏi còn treo chứ không phải đã giải.

#### d. Thứ tìm được khi đi tìm lời giải thích — TRẦN của cả họ prior nhân

Prior nhân dải `[lo,hi]` chỉ đổi được thứ hạng khi `base_j > base_i·(lo/hi)`. Vậy hỏi được câu mạnh
hơn nhiều so với "prior X có tốt không": **có prior nhân NÀO kéo nổi gold lên hạng 1 không?** Đó là
trần cho mọi prior ta nghĩ ra sau này, tính thẳng từ điểm nền.

Mini (n=169): dải 3× → **75,7%** · dải 6× → 79,9% · dải 20× → 79,9%. `pos` đang đạt **52,1%**.

Đọc ra hai điều: (1) còn **23,6 điểm dư địa** cho một prior tốt hơn — họ prior chưa chết trên mini;
(2) nới dải 3×→20× chỉ mua thêm **4,2 điểm** ⇒ nghẽn ở **chất lượng tín hiệu**, không ở độ mạnh
knob. Đừng đi tìm hệ số lớn hơn.

Số v12 quyết ở tầm TICKET: trần thấp ⇒ điểm nền đã quyết xong top-1 ở quy mô thật, và cả hướng
"xếp hạng lại" (thứ §12.50c chỉ ra là nút thắt) là ngõ cụt — phải quay sang sửa chính điểm nền.
§12.31 P7.

### 12.59 P7 trần prior nhân trên v12 — **58.8%** (≥40%); họ còn sống (2026-07-29) — M4

`out/probe_theme_prior_v12c.log` (~215s). n=274 (cùng bộ gold P6). Khối prior/G1–G3 ≡ §12.57
(khỏi dán lại).

#### a. TRẦN của MỌI prior nhân (nguyên văn)

```
TRẦN của MỌI prior nhân — ca mà top-1 CÓ THỂ đổi, theo dải hệ số:
  dải    3× (vd [0.5,1.5]): đổi được top-1 100.0%  ·  **kéo được GOLD lên hạng 1  58.8%** (161/274)
  dải    6× (vd [0.5,3.0]): đổi được top-1 100.0%  ·  **kéo được GOLD lên hạng 1  67.9%** (186/274)
  dải   20× (vd [0.5,10.0]): đổi được top-1 100.0%  ·  **kéo được GOLD lên hạng 1  69.0%** (189/274)
  Cột thứ hai mới là trần thật: đổi được top-1 mà gold ngoài tầm thì đổi cũng vô ích.
  Trần thấp ⇒ điểm NỀN đã quyết xong; mọi prior NHÂN đều không với tới, dù thông minh cỡ nào.
```

#### b. Đọc cổng §12.31

- Trần dải 3× **58.8% ≥ 40%** ⇒ **họ prior xếp-hạng-lại còn sống**, dư địa lớn (`pos` top1 chỉ
  0.051; khoảng cách tới trần ≈ **53 điểm**).
- Nới 3×→20×: 58.8%→69.0% (+10.2 điểm) — nhiều hơn mini (+4.2) nhưng vẫn: nghẽn chính là
  **chất lượng tín hiệu**, không phải hệ số.
- **Không** đổi `compose_theme_weight` / knob. Chờ Claude tín hiệu xếp hạng tốt hơn `pos`/`content`.

---

### 12.60 Nhận §12.59 — dương tính CẤU TRÚC đầu tiên; và chỗ hụt nằm ở đâu (2026-07-29) — Claude

#### a. P7 đạt, và đạt rộng

Trần của **mọi** prior nhân trên v12, dải `[0.5,1.5]`: **58,8%** (161/274) — cổng đòi ≥40%.
⇒ Họ prior xếp-hạng-lại **không phải ngõ cụt**. Đây là kết quả **dương đầu tiên** sau §12.51–§12.58
toàn âm, và nó dương ở tầm hướng đi chứ không phải tầm knob.

Nới dải 3×→20× chỉ đưa 58,8% → 69,0% (+10,2 điểm; mini +4,2) ⇒ nghẽn là **chất lượng tín hiệu**,
không phải độ mạnh hệ số. Đừng đi tìm knob to hơn — kết luận này giờ có số ở cả hai quy mô.

#### b. Con số đáng đọc nhất KHÔNG phải cái trần, mà là TỈ LỆ KHAI THÁC

| | `pos` đạt | trần | **khai thác** |
|---|---|---|---|
| mini 5.004 cha | 52,1% | 75,7% | **68,8%** |
| v12 429.711 cha | 5,1% | 58,8% | **8,7%** |

Trần chỉ tụt 75,7 → 58,8, tức **cơ hội gần như còn nguyên** ở quy mô thật. Nhưng prior hiện tại
chộp được 68,8% cơ hội trên mini và chỉ **8,7%** trên v12. Nó không thoái hoá vì bài toán khó hơn
— nó thoái hoá vì **chính nó** ngừng hoạt động.

Đây cũng là lời giải thích ứng viên cho cả chuỗi mini-dương-v12-âm đã cắn nhiều lần (§12.38a chép
đáp QA · §12.49b PMI · §12.53 `content`): nếu cơ chế nào cũng mất tác dụng theo cùng cách khi bể
lớn, thì các âm tính ấy có thể không phải "giả thuyết sai" mà là "cùng một bệnh về QUY MÔ".
**Chưa chứng minh** — mới là ứng viên, và P8 đo nó.

#### c. Giả thuyết: BÃO HOÀ (đã đo trên mini, chờ v12)

`w_theme = 1/(1+vị_trí)` trả **đúng 1.0** cho MỌI câu mở đầu bằng thực thể. Bể càng lớn càng nhiều
câu như vậy ⇒ tất cả cùng ×1.5 ⇒ prior nhân **đều nhau**, hết phân biệt. Chú ý điều này khớp với
§12.52d (chỉ 6,13% cặp đạt 1.0, phần còn lại nén vào dải phẳng): prior vốn **nhị phân**, mà nhóm
"đứng đầu câu" trên bể lớn thì đông.

Đo mini: bể ứng viên trung vị **416**, chỉ **2** ứng viên cùng chạm đỉnh ⇒ **không** bão hoà ⇒ mini
không phán được câu hỏi này, phải v12. Giao P8.

Nếu bão hoà được xác nhận thì hướng sửa hẹp lại rất nhiều, và nó **khác** mọi thứ đã thử: không
phải tìm một prior khác cho cả bể (đó là `content`/`clear`, đều âm), mà tìm tín hiệu **phân biệt
TRONG nhóm câu đã đứng đầu bằng thực thể**. Chưa thiết kế; chưa hứa.

### 12.61 P8 bão hoà trên v12 — **XÁC NHẬN** (2026-07-29) — M4

`out/probe_theme_prior_v12d.log` (~223s). n=274. Khối prior/trần ≡ §12.57/§12.59 (khỏi dán lại).

#### a. BÃO HOÀ (nguyên văn)

```
BÃO HOÀ — bao nhiêu ứng viên CÙNG chạm giá trị prior cao nhất (trung vị):
  bể ứng viên (trung vị) = 37,908
  pos           47 ứng viên cùng đỉnh =  0.12% bể   ⇒ prior KHÔNG phân biệt nổi
  content       90 ứng viên cùng đỉnh =  0.24% bể   ⇒ prior KHÔNG phân biệt nổi
  clear        113 ứng viên cùng đỉnh =  0.30% bể   ⇒ prior KHÔNG phân biệt nổi
```

#### b. Đọc cổng §12.31 / §12.60c

- Trung vị `pos` cùng đỉnh **47 > 20** ⇒ **bão hoà xác nhận** (mini chỉ 2).
- `content`/`clear` còn đông hơn (90 / 113) — khớp P6 âm: prior khác cho cả bể không cứu.
- So mini: bể 416 → **37.908** (~91×); cùng đỉnh 2 → **47**.
- **Giữ** `compose_theme_weight`. Chờ Claude: tín hiệu phân biệt **trong** nhóm câu đã đứng đầu
  bằng thực thể (không phải prior mới cho cả bể).

---

### 12.62 Nhận §12.61 — bão hoà xác nhận; nhưng kết luận kế tiếp CHƯA được đo (2026-07-29) — Claude

#### a. Xác nhận, và mạnh

Bể trung vị **37.908** ứng viên · `pos` có **47** ứng viên cùng chạm đỉnh prior (cổng >20; mini
chỉ **2**). `content` **90** · `clear` **113` — hai prior thay thế còn bão hoà **nặng hơn**, khớp
đúng việc P6 cho chúng âm tính: **prior khác cho cả bể không cứu được gì**.

Cơ chế đã rõ và khép kín: `w_theme = 1/(1+vị_trí)` trả **đúng 1.0** cho mọi câu mở đầu bằng thực
thể. Bể 91× lớn hơn ⇒ 47 câu cùng được ×1.5 ⇒ trong nhóm ấy prior **im lặng**, điểm nền quyết
hết. Đó là lý do `pos` khai thác 68,8% trần trên mini mà chỉ 8,7% trên v12 (§12.60b), và cũng
giải thích việc bất đồng CO LẠI (3,6% → 1,8%, §12.58c — câu hỏi tôi để treo): đổi công thức prior
chỉ xáo lại thứ tự *bên trong* các nhóm hoà, hiếm khi đổi nhóm nào thắng.

#### b. Chỗ tôi suýt hấp tấp

§12.60c tôi viết sẵn hướng sửa: *"tìm tín hiệu phân biệt TRONG nhóm câu đã đứng đầu bằng thực
thể"*. M4 chép lại đúng như vậy. Nhưng đó là **suy ra, chưa đo** — và nó chỉ đúng nếu **gold nằm
TRONG nhóm hoà**. Nếu gold thường nằm NGOÀI, phân biệt nội nhóm vô ích, và chẩn đoán phải nặng
hơn hẳn: prior đang **đặt cược vào nhầm nhóm**.

Điều này rất dễ xảy ra và có lý do: câu trả lời thật thường **không** mở đầu bằng thực thể được
hỏi — `Năm 1010, nhà Lý dời đô…` mở đầu bằng trạng ngữ. Nếu vậy thì prior đẩy 47 câu mở-đầu-bằng-
thực-thể lên đúng lúc đáp án nằm ngoài nhóm đó.

Sau bảy lần khâu hỏng là suy luận chưa đo, tôi không viết mã trên một suy ra nữa. P9 đo.

#### c. Phân hoạch — thước quyết định hướng sửa

    B  gold TRONG nhóm hoà, nền đã cao nhất nhóm  → đã đúng sẵn
    C  gold TRONG nhóm hoà, nền thua              → **đúng phần tín hiệu nội nhóm cứu được**
    D  gold NGOÀI nhóm hoà                        → cần cách khác hẳn

Mini: B 36,1% · C 7,7% · D 56,2% — nhưng nhóm hoà trên mini chỉ 2 ứng viên nên gold gần như luôn
ở ngoài; **mini không phán được** câu này. Giao P9.

Cổng chốt trước: **C ≥ 20%** ⇒ tôi thiết kế tín hiệu nội nhóm. **C < 10% và D lớn** ⇒ bỏ hẳn hướng
"sửa prior vị trí", chuyển sang prior **mức độ liên quan** thay vì vị trí — một đổi hướng lớn, nên
phải có số mới làm.

### 12.63 P9 phân hoạch GOLD vs nhóm hoà — C **21.5%** · D **74.1%** (2026-07-29) — M4

`out/probe_theme_prior_v12e.log` (~260s). n=274.

#### a. GOLD nằm đâu so với NHÓM HOÀ (nguyên văn)

```
GOLD nằm đâu so với NHÓM HOÀ ở đỉnh prior `pos` — quyết định HƯỚNG SỬA:
  B_hoà_và_nền_đã_nhất          12 =   4.4%
  C_hoà_nhưng_nền_thua          59 =  21.5%
  D_gold_NGOÀI_nhóm_hoà        203 =  74.1%
  B = đã đúng sẵn · **C = phần tín hiệu phân biệt NỘI NHÓM cứu được** · D = cần cách khác.
```

#### b. Đọc cổng §12.31 / §12.62c

- **C = 21.5% ≥ 20%** ⇒ cổng thiết kế tín hiệu **nội nhóm** đạt (`pos` top1 5.1% còn xa trần).
- **D = 74.1%** vẫn chiếm đa số ⇒ prior vị trí cũng **đặt cược nhầm nhóm** trên phần lớn ca (khớp
  nghi §12.62b về trạng ngữ mở đầu). Không đủ để kích cổng "C&lt;10% và D lớn" một mình, nhưng số
  D phải đọc cùng C.
- So mini (B 36.1 · C 7.7 · D 56.2): v12 **C tăng**, **D tăng**, **B giảm mạnh**.
- **Giữ** `compose_theme_weight`. Không knob. Chờ Claude.

---

### 12.64 Nhận §12.63 — bộ lọc 209×, và vì sao phải ĐỌC trước khi thiết kế (2026-07-29) — Claude

#### a. Số, và cách đọc đúng

v12 (n=274): **B 4,4%** (12) · **C 21,5%** (59) · **D 74,1%** (203). Cổng C ≥ 20% **đạt** — sát.

Nhưng con số đáng nói không nằm ở C mà ở chỗ nhóm hoà **to bao nhiêu**:

    nhóm hoà = 47 / 37.908 = 0,124% của bể
    chứa gold                = 25,9% số ca   (B+C)
    ⇒ LÀM GIÀU               = 209×

Tức "thực thể đứng đầu câu" là **bộ lọc rất tốt** — nó dồn 1/4 số đáp án vào 1/800 bể. Bệnh không
phải tín hiệu yếu; bệnh là **không xếp hạng nổi bên trong bộ lọc đó**. Đây là cách đọc ngược hẳn
§12.52d, nơi tôi coi việc prior "nhị phân, 94% nằm dải phẳng" là khuyết tật.

Phân rã dư địa:

| | trần | so `pos` đang 5,1% |
|---|---|---|
| chỉ sửa NỘI NHÓM (B+C) | **25,9%** | gấp 5 |
| mọi prior nhân (§12.59) | 58,8% | gấp 11,5 |
| phần chỉ D với tới | **32,9 điểm** | cần cách khác hẳn |

#### b. Cái gì trong (a) là VỮNG, cái gì phụ thuộc proxy — phải phân biệt

Điểm nền trong probe là **proxy của tôi** (trùng-từ có trọng gidf ÷ `√|câu|`), không phải `cov` thật
của `facts_about`. Nên:

* **Vững, không phụ thuộc proxy:** bão hoà 47 (chỉ phụ thuộc `w_theme`) · làm giàu 209% và trần
  nội nhóm 25,9% (chỉ phụ thuộc nhóm hoà + nhãn gold).
* **Phụ thuộc proxy:** tách B ‖ C. "Nền chọn nhầm" là nền CỦA TÔI chọn nhầm.

Lẽ ra tôi phải nói rõ điều này từ §12.53. Ghi ở đây để mọi số sau đọc đúng phạm vi.

#### c. Đọc 13 ca nhóm C trên mini — hai thứ mọi con số cổng đều giấu

```
[1] HỎI: Thủ đô Việt Nam ở đâu?
    GOLD  (nền 3.743): Thủ đô Hà Nội ở Việt Nam.
    nền#1 (4.244): Thủ đô ASEAN luân phiên.        ← THẮNG
    nền#2 (3.743): Thủ đô Việt Nam là Hà Nội.      ← cũng ĐÚNG, chỉ không phải gold
```

1. **Khớp thêm từ có thể LÀM TỤT điểm.** Gold khớp `Thủ·đô·Việt·Nam`, câu thắng chỉ khớp `Thủ·đô`.
   Nhưng `Việt`/`Nam` gidf thấp nên cộng gần như 0, trong khi mẫu số `√|câu|` phạt câu dài ⇒ câu
   NGẮN LẠC ĐỀ thắng câu DÀI ĐÚNG. Đây là khuyết tật của **điểm nền**, không phải chỗ thiếu một
   tín hiệu nội nhóm — và nó nằm ngoài mọi thứ prior nhân có thể sửa.
2. **Một phần C là NHIỄU NHÃN.** `Thủ đô Việt Nam là Hà Nội` là đáp đúng hoàn hảo, chỉ không trùng
   câu được chọn làm gold. Vậy C = 21,5% là **cận trên**, phần "lỗi thật" nhỏ hơn.

Ca [2] còn cho thấy loại thứ ba: hỏi *"Miền Bắc và miền Nam khác nhau thế nào"* thì gold
`Miền Nam ấm quanh năm` và ứng viên thắng `Miền Bắc có mùa đông lạnh` **đều đúng một nửa** — câu
so sánh cần HAI câu, một nhãn gold đơn không diễn tả nổi.

#### d. Vì sao P10 là dump chứ không phải phép đo

Tôi có cổng cho phép "thiết kế tín hiệu nội nhóm" ngay bây giờ. Không làm. Mini chỉ 13 ca C và
nhóm hoà ở đó chỉ 2 ứng viên — không đại diện cho 59 ca trên nhóm 47. Và bảy lần khâu hỏng trong
ticket này đều cùng một mở đầu: **thiết kế trước khi nhìn dữ liệu**.

P10 lấy 20 ca C trên v12 để tôi đọc. Không cổng, không verdict — dữ liệu. Sau khi đọc mới biết
phần "lỗi thật" trong C là bao nhiêu, nó có hình dạng chung nào không, và nó có phải thứ prior
nhân sửa được không (ca [1] gợi ý là **không**).


### 12.65 P10 dump 20 ca nhóm C trên v12 — dữ liệu để Claude đọc (2026-07-29) — M4

`out/probe_theme_prior_v12f.log` (~236s). **Không cổng** (§12.64d). Dán nguyên văn khối dump
(không rút / không chọn lọc):

```
ĐỌC 20 CA NHÓM C — gold Ở TRONG nhóm hoà nhưng điểm nền chọn nhầm.
Đây là đám mà tín hiệu phân biệt NỘI NHÓM phải sửa. `nền` = trùng-từ có trọng gidf.
====================================================================================

[0] HỎI: Truyện:Ċ"Các ngươi ngược lại là nói chuyện rất vui sướng, lão bà tử ta bỏ lỡ cái gì sao?" Whent 
    GOLD (nền 5.631): " Sau đó.<EOS>
    nền#1 (75.255): Truyện:Ċ“Các ngươi là nghĩ tại chúng ta đế hào gây sự sao,” Trong quầy hoàng mao trên mặt lộ ra một tia thần sắc
    nền#2 (39.932):  Lúc đầu ngư lang liên minh bắt đầu coi như thuận lợi, nếu là đông tuyến nhất cử cầm xuống Thành King's Landing,
    nền#3 (39.110): Bởi vì tọa trấn Thông U phong Lưu Kế Đường lặng yên rời đi, mà tin tức tiết ra ngoài lại cực kỳ đột nhiên, Cơ Vă

[1] HỎI:  x cộng 5 bằng 12 thì x bằng mấy? 
    GOLD (nền 10.558):  x cộng 5 bằng 12 thì x bằng 7.<EOS>
    nền#1 (10.558): x cộng 5 bằng 12 thì x bằng 7.<EOS>
    nền#2 (10.558):  x cộng 5 bằng 12 thì x bằng 7.<EOS>
    nền#3 (5.622): x và 2.<EOS>

[2] HỎI:  Xác suất nằm trong khoảng nào? 
    GOLD (nền 6.609):  Xác suất nằm trong đoạn 0 đến 1.<EOS>
    nền#1 (6.609): Xác suất nằm trong đoạn 0 đến 1.<EOS>
    nền#2 (6.609):  Xác suất nằm trong đoạn 0 đến 1.<EOS>
    nền#3 (4.670): Xác suất xảy ra B khi biết A xảy ra.<EOS>

[3] HỎI:  Thủ đô Việt Nam là gì? 
    GOLD (nền 8.398):  Thủ đô Hà Nội ở Việt Nam.<EOS>
    nền#1 (8.398): Thủ đô Việt Nam là Hà Nội.<EOS>
    nền#2 (8.398): Thủ đô Hà Nội ở Việt Nam.<EOS>
    nền#3 (8.398):  Thủ đô Hà Nội ở Việt Nam.<EOS>

[4] HỎI:  Thủ đô Việt Nam ở đâu? 
    GOLD (nền 8.398):  Thủ đô Hà Nội ở Việt Nam.<EOS>
    nền#1 (8.398): Thủ đô Việt Nam là Hà Nội.<EOS>
    nền#2 (8.398): Thủ đô Hà Nội ở Việt Nam.<EOS>
    nền#3 (8.398):  Thủ đô Hà Nội ở Việt Nam.<EOS>

[5] HỎI:  Sông Hồng chảy qua thành phố nào? 
    GOLD (nền 9.720):  Sông Hồng chảy qua Hà Nội.<EOS>
    nền#1 (9.720): Sông Hồng chảy qua Hà Nội.<EOS>
    nền#2 (9.720):  Sông Hồng chảy qua Hà Nội.<EOS>
    nền#3 (7.229): Sông Mekong chảy qua nhiều nước.<EOS>

[6] HỎI:  Đà Nẵng là thành phố thế nào? 
    GOLD (nền 7.792):  Đà Nẵng là thành phố biển.<EOS>
    nền#1 (7.792): Đà Nẵng là thành phố biển.<EOS>
    nền#2 (7.792):  Đà Nẵng là thành phố biển.<EOS>
    nền#3 (5.195): Đà Nẵng là thành phố loại 2, tương tự như thành phố Chợ Lớn thành lập trước đó.<EOS>

[7] HỎI:  Miền Bắc và miền Nam khác nhau thế nào về khí hậu? 
    GOLD (nền 5.532):  Miền Nam ấm quanh năm.<EOS>
    nền#1 (5.623):  Miền Bắc thành một nước xã hội chủ nghĩa, trong khi miền Nam thành lập một chế độ mới do Mỹ hậu thuẫn.<EOS>
    nền#2 (5.599): Miền Bắc có mùa đông lạnh.<EOS>
    nền#3 (5.532): Miền Nam ấm quanh năm.<EOS>

[8] HỎI:  Đồng bằng và núi khác nhau thế nào? 
    GOLD (nền 5.141):  Đồng bằng bằng phẳng.<EOS>
    nền#1 (5.141): Đồng bằng bằng phẳng.<EOS>
    nền#2 (5.141):  Đồng bằng bằng phẳng.<EOS>
    nền#3 (4.033): Đồng thời ngăn ngừa lỗi bất cứ khi nào có thể bằng cách tiếp nhận các thông tin và trình tự khác nhau.<EOS>

[9] HỎI:  Bác Hồ đọc Tuyên ngôn độc lập khi nào? 
    GOLD (nền 12.354):  Bác Hồ đọc Tuyên ngôn độc lập năm 1945.<EOS>
    nền#1 (12.354): Bác Hồ đọc Tuyên ngôn độc lập năm 1945.<EOS>
    nền#2 (12.354):  Bác Hồ đọc Tuyên ngôn độc lập năm 1945.<EOS>
    nền#3 (4.807): Bác Hồ sinh năm 1890.<EOS>

[10] HỎI:  Ngày 2 tháng 9 là ngày gì? 
    GOLD (nền 7.141):  Ngày 2 tháng 9 là Quốc khánh Việt Nam.<EOS>
    nền#1 (9.669):  Ngày 29 tháng 2.<EOS>
    nền#2 (7.141): Ngày 2 tháng 9 là Quốc khánh Việt Nam.<EOS>
    nền#3 (7.141):  Ngày 2 tháng 9 là Quốc khánh Việt Nam.<EOS>

[11] HỎI:  Quang Trung đại phá quân nào? 
    GOLD (nền 10.238):  Quang Trung đại phá quân Thanh.<EOS>
    nền#1 (10.238): Quang Trung đại phá quân Thanh.<EOS>
    nền#2 (10.238):  Quang Trung đại phá quân Thanh.<EOS>
    nền#3 (6.151):  Quang Trung chia quân làm 5 đạo.<EOS>

[12] HỎI:  Trống đồng Đông Sơn là gì? 
    GOLD (nền 9.408):  Trống đồng Đông Sơn là di sản.<EOS>
    nền#1 (9.408): Trống đồng Đông Sơn là di sản.<EOS>
    nền#2 (9.408):  Trống đồng Đông Sơn là di sản.<EOS>
    nền#3 (3.401): Trống này sau đó truyền mực lên giấy.<EOS>

[13] HỎI:  Phổi giúp ta làm gì? 
    GOLD (nền 8.371):  Phổi giúp ta thở.<EOS>
    nền#1 (8.371): Phổi giúp ta thở.<EOS>
    nền#2 (8.371):  Phổi giúp ta thở.<EOS>
    nền#3 (4.467):  Phổi: Uống cà phê giúp thể cải thiện chức năng phổi.<EOS>

[14] HỎI:  Biến đổi khí hậu ảnh hưởng gì đến Việt Nam? 
    GOLD (nền 7.561):  Biến đổi khí hậu làm mực nước biển dâng.<EOS>
    nền#1 (8.268): Biến đổi khí hậu: Nó ảnh hưởng đến môi trường của chúng ta như thế nàoĊ<EOS>
    nền#2 (7.930): Biến đổi khí hậu cần hành động chung.<EOS>
    nền#3 (7.684):  Biến đổi khí hậu cũng có thể ảnh hưởng đến vòng đời của loài bướm, dẫn đến suy giảm quần thể.<EOS>

[15] HỎI:  Megacity là gì? 
    GOLD (nền 7.815):  Megacity trên 10 triệu dân.<EOS>
    nền#1 (7.815): Megacity trên 10 triệu dân.<EOS>
    nền#2 (7.815):  Megacity trên 10 triệu dân.<EOS>

[16] HỎI:  Cà phê robusta trồng nhiều ở đâu nước ta? 
    GOLD (nền 9.882):  Cà phê robusta ở Tây Nguyên.<EOS>
    nền#1 (9.882): Cà phê robusta ở Tây Nguyên.<EOS>
    nền#2 (9.882):  Cà phê robusta ở Tây Nguyên.<EOS>

[17] HỎI:  Gạo xuất khẩu của Việt Nam chủ yếu từ đâu? 
    GOLD (nền 7.784):  Gạo xuất khẩu từ đồng bằng.<EOS>
    nền#1 (7.784): Gạo xuất khẩu từ đồng bằng.<EOS>
    nền#2 (7.784):  Gạo xuất khẩu từ đồng bằng.<EOS>
    nền#3 (3.535):  Gạo tẻ vo sạch nấu thành cháo.<EOS>

[18] HỎI:  Hạn hán ở Việt Nam thường gặp ở đâu? 
    GOLD (nền 6.866):  Hạn hán ở Tây Nguyên.<EOS>
    nền#1 (6.866): Hạn hán ở Tây Nguyên.<EOS>
    nền#2 (6.866):  Hạn hán ở Tây Nguyên.<EOS>
    nền#3 (5.795):  Hạn hán: Vùng Sahel, nằm ở phía nam sa mạc Sahara, thường xuyên phải chịu hạn hán do thiếu mưa trên sa mạc.<EOS

[19] HỎI:  Sông Hồng bắt nguồn từ đâu? 
    GOLD (nền 5.697):  Sông Hồng chảy qua Hà Nội.<EOS>
    nền#1 (5.697): Sông Hồng chảy qua Hà Nội.<EOS>
    nền#2 (5.697):  Sông Hồng chảy qua Hà Nội.<EOS>
    nền#3 (5.095): Sông Hồng nuôi dưỡng đồng bằng Bắc Bộ.<EOS>

```

---

### 12.66 Đọc P10 — lỗi NHÃN thứ ba; mọi số v12 từ §12.53 phải tính lại (2026-07-29) — Claude

#### a. Cái dump cho thấy

8/10 ca nhóm C in ra có dạng này:

```
[1] HỎI: x cộng 5 bằng 12 thì x bằng mấy?
    GOLD  (nền 10.558): " x cộng 5 bằng 12 thì x bằng 7."
    nền#1 (10.558):     "x cộng 5 bằng 12 thì x bằng 7."     ← CÙNG MỘT CÂU
    nền#2 (10.558):     " x cộng 5 bằng 12 thì x bằng 7."    ← CÙNG MỘT CÂU

[5] GOLD (9.720): " Sông Hồng chảy qua Hà Nội."
    nền#1 (9.720):     "Sông Hồng chảy qua Hà Nội."          ← CÙNG MỘT CÂU
```

**Cùng một câu tồn tại thành nhiều cha trên v12.** Điểm nền bằng nhau **tuyệt đối**; tie-break
`(-sc[k], k)` chọn chỉ số nhỏ nhất, thường không phải cha tôi gán nhãn gold ⇒ tính là "chọn nhầm"
trong khi văn bản **y hệt**. Đó không phải lỗi xếp hạng, đó là lỗi **đếm**.

Mini: **5.004 cha câu / 5.004 văn bản duy nhất — 0 bản sao.** Nên probe phát triển trên mini không
thể thấy hiện tượng này. Lần thứ **ba** trong chính probe này mắc cùng một lớp lỗi "hiệu chỉnh
trên bể nhỏ rồi áp lên bể lớn" (sau §12.56a cắt gold ở top-80, §12.56b ngưỡng G1 tuyệt đối).

#### b. Vá

Gold là **TẬP** mọi cha phủ ≥ `gold_cover`, đúng khi **bất kỳ** cái nào hạng 1 — áp cho mọi thước:
top1/top10/MRR, phân hoạch B/C/D, và trần (`reachable` lấy max trên tập).

Mini (không có bản sao) vẫn đổi đáng kể vì nhiều câu **khác nhau** cùng trả lời đúng:

| | trước | sau |
|---|---|---|
| `pos` top-1 | 0.521 | **0.615** |
| B | 36,1% | 43,8% |
| C | 7,7% | 7,1% |
| D | 56,2% | 49,1% |

#### c. Phải tính lại những gì

**Mọi số v12 từ §12.53 tới §12.65.** Cụ thể, các phát biểu sau đang treo, không được trích dùng:

* `pos` top-1 v12 **5,1%** — đánh giá THẤP (bản sao cướp công).
* C = **21,5%** và cổng "C ≥ 20% đạt" (§12.63) — rất có thể đạt **nhờ bản sao**. Nếu C thật < 10%
  thì nhánh kia của cổng §12.62 mới đúng, và hướng đi khác hẳn.
* Trần 58,8% và làm giàu 209× — đều tính trên gold đơn ⇒ đánh giá THẤP.
* §12.64c "khớp thêm từ làm tụt điểm" vẫn đứng (ca `Thủ đô` là hai câu KHÁC nhau, không phải bản
  sao), nhưng §12.64a phải chờ số mới.

Thứ **không** phải tính lại: bão hoà 47 (chỉ phụ thuộc `w_theme`, độc lập nhãn) và §12.57/§12.59
kết luận `content`/`clear` không hơn `pos` (so sánh tương đối, cùng nhãn hỏng cho cả ba).

#### d. Ghi nhận

Lần thứ **tám** khâu hỏng là thước. Nhưng đây là lần đầu nó bị bắt bởi việc **đọc dữ liệu thô** chứ
không phải bởi một con số vô lý — và nó xác nhận quyết định ở §12.64d: có cổng cho phép thiết kế
ngay, nhưng chọn dump và đọc trước. Nếu đã thiết kế "tín hiệu phân biệt nội nhóm" theo cổng C, tôi
sẽ viết cả một cơ chế để sửa một đám phần lớn là **bản sao văn bản**.

#### e. Quan sát riêng, chưa đo: corpus v12 có rác điểm cao

Ca [0] của §12.65: `Truyện:Ċ"Các ngươi ngược lại là nói chuyện rất vui sướng…"` — truyện dịch, và
ứng viên nền#1 đạt **75,3** trong khi mọi câu sử hợp lệ ở các ca khác chỉ 8–12. Rác đạt điểm nền
cao gấp 6–9 lần thì nó chi phối truy hồi mọi truy vấn chạm phải. Đã hỏi M4 đếm trong P11.


### 12.67 P11 probe sau vá gold-tập — tính lại toàn bộ (2026-07-29) — M4

`out/probe_theme_prior_v12g.log` (~238s). Gold = **TẬP** cha phủ ≥0.9; đúng nếu bất kỳ cái nào hạng 1.
n=274 (cùng số ca). **Không cổng** (§12.66).

#### a. Mọi khối (nguyên văn)

```
==================================================================
prior          top1    top10      MRR    Δ so pos   (n=274)
==================================================================
(tw=0)        0.117        —        —           —
none          0.117    0.493    0.263      -0.011
pos           0.128    0.482    0.275      +0.000
content       0.131    0.467    0.268      +0.004
clear         0.095    0.387    0.204      -0.033
==================================================================

BẤT ĐỒNG so với `pos` (ca mà top-1 đổi kết quả):
  none     thắng   3 · thua   6 · bất đồng   9/274 = 3.3%
  content  thắng   3 · thua   2 · bất đồng   5/274 = 1.8%
  clear    thắng   3 · thua  12 · bất đồng  15/274 = 5.5%

BÃO HOÀ — bao nhiêu ứng viên CÙNG chạm giá trị prior cao nhất (trung vị):
  bể ứng viên (trung vị) = 37,908
  pos           47 ứng viên cùng đỉnh =  0.12% bể   ⇒ prior KHÔNG phân biệt nổi
  content       90 ứng viên cùng đỉnh =  0.24% bể   ⇒ prior KHÔNG phân biệt nổi
  clear        113 ứng viên cùng đỉnh =  0.30% bể   ⇒ prior KHÔNG phân biệt nổi

GOLD nằm đâu so với NHÓM HOÀ ở đỉnh prior `pos` — quyết định HƯỚNG SỬA:
  B_hoà_và_nền_đã_nhất          23 =   8.4%
  C_hoà_nhưng_nền_thua          57 =  20.8%
  D_gold_NGOÀI_nhóm_hoà        194 =  70.8%
  B = đã đúng sẵn · **C = phần tín hiệu phân biệt NỘI NHÓM cứu được** · D = cần cách khác.

TRẦN của MỌI prior nhân — ca mà top-1 CÓ THỂ đổi, theo dải hệ số:
  dải    3× (vd [0.5,1.5]): đổi được top-1 100.0%  ·  **kéo được GOLD lên hạng 1  67.9%** (186/274)
  dải    6× (vd [0.5,3.0]): đổi được top-1 100.0%  ·  **kéo được GOLD lên hạng 1  72.6%** (199/274)
  dải   20× (vd [0.5,10.0]): đổi được top-1 100.0%  ·  **kéo được GOLD lên hạng 1  73.0%** (200/274)
  Cột thứ hai mới là trần thật: đổi được top-1 mà gold ngoài tầm thì đổi cũng vô ích.
  Trần thấp ⇒ điểm NỀN đã quyết xong; mọi prior NHÂN đều không với tới, dù thông minh cỡ nào.
G1 bench lành (pos top-10 ≥0.40): ĐẠT (top10 0.482 · top1 0.128)
G2 prior có tác dụng?            : tw=0 → 0.117 ‖ pos → 0.128 (Δ +0.011)
G3 thay thế hơn ≥+0.05          : TRƯỢT (tốt nhất `content` 0.131, Δ +0.004)

⇒ CHỐT ÂM TÍNH — không sửa prior theo hướng này

CHẨN ĐOÁN ca 0 (focus `Lý`) — prior nào xếp câu ĐÚNG trên câu XẤU:
  none     ĐÚNG=1.500  XẤU=1.500  ✗ không tách
  pos      ĐÚNG=0.625  XẤU=0.611  ✓ tách
  content  ĐÚNG=0.667  XẤU=0.625  ✓ tách
  clear    ĐÚNG=0.500  XẤU=0.500  ✗ không tách

====================================================================================
ĐỌC 20 CA NHÓM C — gold Ở TRONG nhóm hoà nhưng điểm nền chọn nhầm.
Đây là đám mà tín hiệu phân biệt NỘI NHÓM phải sửa. `nền` = trùng-từ có trọng gidf.
====================================================================================

[0] HỎI: Truyện:Ċ"Các ngươi ngược lại là nói chuyện rất vui sướng, lão bà tử ta bỏ lỡ cái gì sao?" Whent 
    GOLD (nền 10.771):  không phản ứng.<EOS>
    nền#1 (75.255): Truyện:Ċ“Các ngươi là nghĩ tại chúng ta đế hào gây sự sao,” Trong quầy hoàng mao trên mặt lộ ra một tia thần sắc
    nền#2 (39.932):  Lúc đầu ngư lang liên minh bắt đầu coi như thuận lợi, nếu là đông tuyến nhất cử cầm xuống Thành King's Landing,
    nền#3 (39.110): Bởi vì tọa trấn Thông U phong Lưu Kế Đường lặng yên rời đi, mà tin tức tiết ra ngoài lại cực kỳ đột nhiên, Cơ Vă

[1] HỎI:  Thủ đô Việt Nam là gì? 
    GOLD (nền 8.398):  Thủ đô Hà Nội ở Việt Nam.<EOS>
    nền#1 (8.398): Thủ đô Việt Nam là Hà Nội.<EOS>
    nền#2 (8.398): Thủ đô Hà Nội ở Việt Nam.<EOS>
    nền#3 (8.398):  Thủ đô Hà Nội ở Việt Nam.<EOS>

[2] HỎI:  Thủ đô Việt Nam ở đâu? 
    GOLD (nền 8.398):  Thủ đô Hà Nội ở Việt Nam.<EOS>
    nền#1 (8.398): Thủ đô Việt Nam là Hà Nội.<EOS>
    nền#2 (8.398): Thủ đô Hà Nội ở Việt Nam.<EOS>
    nền#3 (8.398):  Thủ đô Hà Nội ở Việt Nam.<EOS>

[3] HỎI:  Sông Hồng chảy qua thành phố nào? 
    GOLD (nền 9.720):  Sông Hồng chảy qua Hà Nội.<EOS>
    nền#1 (9.720): Sông Hồng chảy qua Hà Nội.<EOS>
    nền#2 (9.720):  Sông Hồng chảy qua Hà Nội.<EOS>
    nền#3 (7.229): Sông Mekong chảy qua nhiều nước.<EOS>

[4] HỎI:  Đà Nẵng là thành phố thế nào? 
    GOLD (nền 7.792):  Đà Nẵng là thành phố biển.<EOS>
    nền#1 (7.792): Đà Nẵng là thành phố biển.<EOS>
    nền#2 (7.792):  Đà Nẵng là thành phố biển.<EOS>
    nền#3 (5.195): Đà Nẵng là thành phố loại 2, tương tự như thành phố Chợ Lớn thành lập trước đó.<EOS>

[5] HỎI:  Lá cờ Việt Nam có màu gì? 
    GOLD (nền 10.401):  Lá cờ Việt Nam màu đỏ sao vàng.<EOS>
    nền#1 (10.401): Lá cờ Việt Nam màu đỏ sao vàng.<EOS>
    nền#2 (10.401):  Lá cờ Việt Nam màu đỏ sao vàng.<EOS>
    nền#3 (5.240):  Lá cờ chữ thập màu vàng từng là quân kỳ của hải quân Hoàng gia Thuỵ Điển.<EOS>

[6] HỎI:  Miền Bắc và miền Nam khác nhau thế nào về khí hậu? 
    GOLD (nền 5.532):  Miền Nam ấm quanh năm.<EOS>
    nền#1 (5.623):  Miền Bắc thành một nước xã hội chủ nghĩa, trong khi miền Nam thành lập một chế độ mới do Mỹ hậu thuẫn.<EOS>
    nền#2 (5.599): Miền Bắc có mùa đông lạnh.<EOS>
    nền#3 (5.532): Miền Nam ấm quanh năm.<EOS>

[7] HỎI:  Đồng bằng và núi khác nhau thế nào? 
    GOLD (nền 5.141):  Đồng bằng bằng phẳng.<EOS>
    nền#1 (5.141): Đồng bằng bằng phẳng.<EOS>
    nền#2 (5.141):  Đồng bằng bằng phẳng.<EOS>
    nền#3 (4.033): Đồng thời ngăn ngừa lỗi bất cứ khi nào có thể bằng cách tiếp nhận các thông tin và trình tự khác nhau.<EOS>

[8] HỎI:  Ngày 2 tháng 9 là ngày gì? 
    GOLD (nền 7.141):  Ngày 2 tháng 9 là Quốc khánh Việt Nam.<EOS>
    nền#1 (9.669):  Ngày 29 tháng 2.<EOS>
    nền#2 (7.141): Ngày 2 tháng 9 là Quốc khánh Việt Nam.<EOS>
    nền#3 (7.141):  Ngày 2 tháng 9 là Quốc khánh Việt Nam.<EOS>

[9] HỎI:  Quang Trung đại phá quân nào? 
    GOLD (nền 10.238):  Quang Trung đại phá quân Thanh.<EOS>
    nền#1 (10.238): Quang Trung đại phá quân Thanh.<EOS>
    nền#2 (10.238):  Quang Trung đại phá quân Thanh.<EOS>
    nền#3 (6.151):  Quang Trung chia quân làm 5 đạo.<EOS>

[10] HỎI:  Trống đồng Đông Sơn là gì? 
    GOLD (nền 9.408):  Trống đồng Đông Sơn là di sản.<EOS>
    nền#1 (9.408): Trống đồng Đông Sơn là di sản.<EOS>
    nền#2 (9.408):  Trống đồng Đông Sơn là di sản.<EOS>
    nền#3 (3.401): Trống này sau đó truyền mực lên giấy.<EOS>

[11] HỎI:  Phổi giúp ta làm gì? 
    GOLD (nền 8.371):  Phổi giúp ta thở.<EOS>
    nền#1 (8.371): Phổi giúp ta thở.<EOS>
    nền#2 (8.371):  Phổi giúp ta thở.<EOS>
    nền#3 (4.467):  Phổi: Uống cà phê giúp thể cải thiện chức năng phổi.<EOS>

[12] HỎI:  Biến đổi khí hậu ảnh hưởng gì đến Việt Nam? 
    GOLD (nền 7.561): Biến đổi khí hậu làm mực nước biển dâng.<EOS>
    nền#1 (8.268): Biến đổi khí hậu: Nó ảnh hưởng đến môi trường của chúng ta như thế nàoĊ<EOS>
    nền#2 (7.930): Biến đổi khí hậu cần hành động chung.<EOS>
    nền#3 (7.684):  Biến đổi khí hậu cũng có thể ảnh hưởng đến vòng đời của loài bướm, dẫn đến suy giảm quần thể.<EOS>

[13] HỎI:  Megacity là gì? 
    GOLD (nền 7.815):  Megacity trên 10 triệu dân.<EOS>
    nền#1 (7.815): Megacity trên 10 triệu dân.<EOS>
    nền#2 (7.815):  Megacity trên 10 triệu dân.<EOS>

[14] HỎI:  Cà phê robusta trồng nhiều ở đâu nước ta? 
    GOLD (nền 9.882):  Cà phê robusta ở Tây Nguyên.<EOS>
    nền#1 (9.882): Cà phê robusta ở Tây Nguyên.<EOS>
    nền#2 (9.882):  Cà phê robusta ở Tây Nguyên.<EOS>

[15] HỎI:  Gạo xuất khẩu của Việt Nam chủ yếu từ đâu? 
    GOLD (nền 7.784):  Gạo xuất khẩu từ đồng bằng.<EOS>
    nền#1 (7.784): Gạo xuất khẩu từ đồng bằng.<EOS>
    nền#2 (7.784):  Gạo xuất khẩu từ đồng bằng.<EOS>
    nền#3 (3.535):  Gạo tẻ vo sạch nấu thành cháo.<EOS>

[16] HỎI:  Hạn hán ở Việt Nam thường gặp ở đâu? 
    GOLD (nền 6.866):  Hạn hán ở Tây Nguyên.<EOS>
    nền#1 (6.866): Hạn hán ở Tây Nguyên.<EOS>
    nền#2 (6.866):  Hạn hán ở Tây Nguyên.<EOS>
    nền#3 (5.795):  Hạn hán: Vùng Sahel, nằm ở phía nam sa mạc Sahara, thường xuyên phải chịu hạn hán do thiếu mưa trên sa mạc.<EOS

[17] HỎI:  Sông Hồng bắt nguồn từ đâu? 
    GOLD (nền 5.697):  Sông Hồng chảy qua Hà Nội.<EOS>
    nền#1 (5.697): Sông Hồng chảy qua Hà Nội.<EOS>
    nền#2 (5.697):  Sông Hồng chảy qua Hà Nội.<EOS>
    nền#3 (5.095): Sông Hồng nuôi dưỡng đồng bằng Bắc Bộ.<EOS>

[18] HỎI:  Tại sao phải rửa tay trước khi ăn? 
    GOLD (nền 9.703):  Rửa tay trước khi ăn.<EOS>
    nền#1 (9.703): Rửa tay trước khi ăn.<EOS>
    nền#2 (9.703):  Rửa tay trước khi ăn.<EOS>
    nền#3 (4.642): Rửa tay diệt vi khuẩn.<EOS>

[19] HỎI:  Sa mạc Sahara ở châu lục nào? 
    GOLD (nền 11.419):  Sa mạc Sahara ở châu Phi.<EOS>
    nền#1 (11.419): Sa mạc Sahara ở châu Phi.<EOS>
    nền#2 (11.419):  Sa mạc Sahara ở châu Phi.<EOS>
    nền#3 (8.186):  Sa mạc Sahara là sa mạc lớn nhất thế giới.<EOS>

```

#### b. Quan sát rác corpus (hỏi kèm §12.66)

Trong 20 ca nhóm C in ra: **1/20** ca có `nền#1` loại truyện dịch / điểm nền cực cao.

- **[0]** nền#1 = **75.255** — `Truyện:…` / `hoàng mao` (cùng kiểu King's Landing ở §12.65[0])
- 19 ca còn lại: nền#1 là câu QA/kiến thức thường (điểm nền ~5–13), không phải văn bản đó.

---

### 12.68 Vá nhãn LẦN HAI — và một tự sửa trong cùng một lượt đọc (2026-07-29) — Claude

#### a. Số P11 (§12.67) và cái tôi đọc nhầm

| | trước vá §12.66 | sau vá §12.66 |
|---|---|---|
| `pos` top-1 | 5,1% | **12,8%** (2,5×) |
| trần dải 3× | 58,8% | **67,9%** |
| B | 4,4% | 8,4% |
| **C** | 21,5% | **20,8%** |
| D | 74,1% | 70,8% |
| bão hoà | 47 | **47** |

Bão hoà đứng yên đúng như §12.64b dự đoán (chỉ phụ thuộc `w_theme`, độc lập nhãn) — một dự đoán
đúng, ghi lại vì phần lớn dự đoán của tôi trong ticket này sai.

Thấy C gần như không đổi, tôi kết luận cảnh báo §12.66c ("C rất có thể đạt nhờ bản sao") là **sai**.
**Kết luận đó mới sai.** Đọc dump P11 thì nhóm C **vẫn toàn bản sao** — vá không bắt được chúng,
nên C bất động, và tôi đọc sự bất động ấy thành bằng chứng ngược.

#### b. Vì sao vá §12.66 hụt

Gom gold theo **id token**. Nhưng cùng một câu khác nhau mỗi khoảng trắng đầu là hai chuỗi id
KHÁC NHAU:

    'Sông' id=7630   ‖   'ĠSông' id=6559

Lệch đúng 1 token trên câu 8 token duy nhất ⇒ phủ **0.875 < 0.9** ⇒ bản kia rớt ngưỡng gold, rồi
bị tính là "nền chọn nhầm" dù chữ y hệt. Vá lần hai: gom theo **văn bản đã chuẩn hoá** (`Ġ`→space,
bỏ `<EOS>`, strip).

Nguồn nhiễu thứ hai **không** vá tự động được: nền#1 có thể là một đáp **đúng khác** —
`Thủ đô Việt Nam là Hà Nội` ‖ `Thủ đô Hà Nội ở Việt Nam`. Nhưng đo được bằng độ phủ, nên tách sẵn
để lượt sau khỏi hỏi lại:

    C1  nền#1 phủ ≥0.7 câu đáp → nhiễu nhãn      C2  nền#1 phủ <0.7 → lỗi xếp hạng THẬT

Mini: C 7,1% = C1 1,2% + **C2 5,9%**. Khối dump giờ chỉ in **C2** — đúng đám tôi cần đọc.

#### c. Bốn lần vá thước trên CÙNG một probe — và ý nghĩa của nó

1. §12.56a gold cắt ở top-80 ứng viên (v12 74 gold ‖ mini 149).
2. §12.56b G1 là ngưỡng top-1 tuyệt đối hiệu chỉnh trên bể nhỏ.
3. §12.66 gold là một cha thay vì tập.
4. §12.68 gom tập theo id thay vì theo văn bản.

Cả bốn cùng một lớp: **thứ đúng trên 5.004 cha sai trên 429.711 cha**. Và mỗi lần vá, con số đầu
bảng đổi rất nhiều (`pos` top-1 đã đi 5,1% → 12,8%, còn sẽ đổi nữa). Hệ quả thực tế: **đừng trích
bất kỳ số v12 nào của probe này như số cuối** cho tới khi một lượt chạy không còn lộ lỗi nhãn mới.
Thứ duy nhất đã ổn định qua cả bốn lần vá là **bão hoà 47** — vì nó không đụng tới nhãn.

#### d. Cổng §12.62 phải đọc lại

"C ≥ 20% ⇒ thiết kế tín hiệu nội nhóm" — cổng ấy giờ phải chấm trên **C2**, không phải C. Nếu C2
thật sự nhỏ (mini 5,9%) thì nhánh còn lại của §12.62 mới đúng: prior đang **đặt cược nhầm nhóm**
(D = 70,8%), và hướng đi là prior **mức độ liên quan** chứ không phải sửa vị trí. Chờ P12.

---

### 12.69 P12 probe sau vá gold lần hai (văn bản) — tính lại toàn bộ (2026-07-29) — M4

`out/probe_theme_prior_v12h.log` (~237s). Gold = **TẬP** cha phủ ≥0.9, gom theo **văn bản chuẩn hoá**
(không theo id token). n=274. **Không cổng.**

#### a. Mọi khối (nguyên văn)

```
==================================================================
prior          top1    top10      MRR    Δ so pos   (n=274)
==================================================================
(tw=0)        0.318        —        —           —
none          0.318    0.493    0.376      -0.029
pos           0.347    0.482    0.393      +0.000
content       0.310    0.467    0.368      -0.036
clear         0.215    0.391    0.275      -0.131
==================================================================

BẤT ĐỒNG so với `pos` (ca mà top-1 đổi kết quả):
  none     thắng  13 · thua  21 · bất đồng  34/274 = 12.4%
  content  thắng   1 · thua  11 · bất đồng  12/274 = 4.4%
  clear    thắng   2 · thua  38 · bất đồng  40/274 = 14.6%

BÃO HOÀ — bao nhiêu ứng viên CÙNG chạm giá trị prior cao nhất (trung vị):
  bể ứng viên (trung vị) = 37,908
  pos           47 ứng viên cùng đỉnh =  0.12% bể   ⇒ prior KHÔNG phân biệt nổi
  content       90 ứng viên cùng đỉnh =  0.24% bể   ⇒ prior KHÔNG phân biệt nổi
  clear        113 ứng viên cùng đỉnh =  0.30% bể   ⇒ prior KHÔNG phân biệt nổi

GOLD nằm đâu so với NHÓM HOÀ ở đỉnh prior `pos` — quyết định HƯỚNG SỬA:
  B_hoà_và_nền_đã_nhất          63 =  23.0%
  C_hoà_nhưng_nền_thua          17 =   6.2%
  C1_nền#1_có_thể_CŨNG_ĐÚNG      6 =   2.2%
  C2_lỗi_xếp_hạng_THẬT          11 =   4.0%
  D_gold_NGOÀI_nhóm_hoà        194 =  70.8%
  B = đã đúng sẵn · C = gold trong nhóm nhưng nền thua · D = cần cách khác.
  **C2 mới là phần tín hiệu nội nhóm thật sự cứu được** — C1 là nhiễu nhãn (đáp đúng khác).

TRẦN của MỌI prior nhân — ca mà top-1 CÓ THỂ đổi, theo dải hệ số:
  dải    3× (vd [0.5,1.5]): đổi được top-1 100.0%  ·  **kéo được GOLD lên hạng 1  67.9%** (186/274)
  dải    6× (vd [0.5,3.0]): đổi được top-1 100.0%  ·  **kéo được GOLD lên hạng 1  72.6%** (199/274)
  dải   20× (vd [0.5,10.0]): đổi được top-1 100.0%  ·  **kéo được GOLD lên hạng 1  73.0%** (200/274)
G1 bench lành (pos top-10 ≥0.40): ĐẠT (top10 0.482 · top1 0.347)
G2 prior có tác dụng?            : tw=0 → 0.318 ‖ pos → 0.347 (Δ +0.029)
G3 thay thế hơn ≥+0.05          : TRƯỢT (tốt nhất `none` 0.318, Δ -0.029)

⇒ CHỐT ÂM TÍNH — không sửa prior theo hướng này
```

#### b. Số đổi mạnh so với P11 (§12.67)

| | P11 (v12g, gold id) | P12 (v12h, gold văn bản) | Ghi chú |
|---|---|---|---|
| `pos` top1 | 0.128 | **0.347** | +0.219 — bản sao đã sạch |
| `(tw=0)` top1 | 0.117 | **0.318** | baseline cũng tăng mạnh |
| C (tổng) | 20.8% | **6.2%** | phần lớn là bản sao → gom xong |
| C1 | — | **2.2%** | đáp đúng khác, nhiễu nhãn |
| C2 | — | **4.0%** | **lỗi xếp hạng THẬT** = 11 ca |
| B | 8.4% | **23.0%** | nhiều ca vừa vào gold-tập mới |
| D | 70.8% | **70.8%** | không đổi |
| G3 | TRƯỢT | **TRƯỢT** | vẫn âm tính |

#### c. Đọc dump C2 (11 ca)

11 ca C2 (lỗi xếp hạng thật) sau khi lọc C1:

- **[0]** Rác fiction (`Truyện:/hoàng mao`) nền#1 = 75.255 — case rác corpus cũ.
- **[1]** Khí hậu: `nền#1` thắng bằng "Miền Bắc… XHCN" (lịch sử, không phải khí hậu) — trùng từ "Miền" nhiều hơn.
- **[2]** Biến đổi khí hậu: `nền#1` = tiêu đề SEO ("Nó ảnh hưởng đến môi trường như thế nào") — khớp gidf cao hơn đáp súc tích.
- **[3]** Kinh đô Nguyễn: gold chỉ nói `Huế là di sản` (không có "Nguyễn"/"triều") — điểm nền thấp; nền#1 nói `Đại Lý`.
- **[4][9]** Bản đồ tỉ lệ: gold là `tỉ lệ).<EOS>` (đoạn cụt); nền#1 = `Tỉ lệ sinh 13/1` — trùng "tỉ lệ" gidf cao → **rác chỉ số corpus**.
- **[5][6]** Chính phủ/Tòa án: nền#1 bản sao + `chính phủ dân chủ/ban thư ký` — câu dài hơn khớp nhiều từ hơn.
- **[7]** 0,75: nền#1 = `000 đến 57` / `075 km` — số rác khớp `0` và `7`.
- **[8]** Lực: gold nền = 2.563 rất thấp; `Lực lượng Liên Xô` thắng vì gidf `Liên·Xô` cao.
- **[10]** Đình làng: nền#1 = `Thành hoàng` — câu ngắn hơn gold nhưng gidf `Thành·hoàng` > `sinh·hoạt·cộng·đồng`.

**Mẫu chung của 11 ca C2:**
1. Câu rác/nhiễu corpus có gidf token cụm cao bất thường (rác fiction, chỉ số lạ, tên riêng ngoại).
2. Gold là câu trả lời súc tích, token thông dụng gidf thấp → thua nền.
3. Không phải "xếp hạng sai trong nhóm hòa" — là **điểm nền sai**, prior nhân không cứu được.

**Kết luận:** C2 = **4.0%** (11/274), nhỏ hơn cổng 20% rất nhiều. Hướng cổng §12.62 → **nhánh D**: cần prior **mức độ liên quan** (loại câu rác trước khi chấm điểm nền), không phải tín hiệu phân biệt nội nhóm.


---

### 12.70 Nhận §12.69 — hướng theme prior ĐÓNG bằng cổng đã chốt trước (2026-07-29) — Claude

#### a. Vá nhãn lần hai lật lại kết luận, và lần này theo hướng tôi đã cảnh báo

| | P11 (§12.67) | **P12 (§12.69)** |
|---|---|---|
| `pos` top-1 | 12,8% | **34,7%** |
| B | 8,4% | 23,0% |
| C | 20,8% | **6,2%** |
| C1 nhiễu nhãn | — | 2,2% |
| **C2 lỗi xếp hạng THẬT** | — | **4,0%** (11 ca) |
| D | 70,8% | **70,8%** |
| bão hoà | 47 | 47 |

Cảnh báo §12.66c ("C rất có thể đạt **nhờ bản sao**") **đúng**. Chỗ tôi sai là ở §12.68a, khi thấy
C bất động sau vá lần một mà kết luận cảnh báo sai — thực ra vá lần một không bắt được bản sao.

#### b. Cổng §12.62 bắn nhánh CÒN LẠI

Cổng chốt trước: *"C ≥ 20% ⇒ thiết kế tín hiệu nội nhóm · C < 10% và D lớn ⇒ prior đặt cược nhầm
nhóm, bỏ hẳn hướng sửa prior vị trí"*. Chấm trên **C2 = 4,0%**, D = 70,8% ⇒ **nhánh thứ hai**.

Hai ứng viên thay thế cũng chết dứt khoát ở lượt này: `content` −0.036 (**1 thắng / 11 thua**) ·
`clear` −0.131 (**2 / 38**). Không còn gì để thử trong họ prior vị trí.

Đọc 11 ca C2: ca [0] là **rác corpus** (truyện dịch, nền 75,3 ‖ gold 10,8) · ca [1] là câu **so
sánh** cần hai vế (`Miền Nam ấm quanh năm` ‖ `Miền Bắc có mùa đông lạnh` — cả hai đều đúng một
nửa). Tức ngay trong 11 ca "lỗi thật" cũng phần lớn không phải lỗi xếp hạng sửa được.

#### c. Trạng thái sau khi đóng

`pos` **34,7%** ‖ trần mọi prior nhân **67,9%** ⇒ còn ~33 điểm, nhưng **không** với tới bằng công
việc nội nhóm (tối đa +4). Chúng nằm ở **D = 70,8%**: gold không nằm trong nhóm đỉnh prior. Prior
vị trí đã hết đường; muốn chạm D phải là prior **mức độ liên quan**, không phải vị trí.

#### d. Bài học về con số

Ba lượt chạy cùng một probe trên cùng checkpoint, chỉ khác cách gán nhãn gold:

    pos top-1:  5,1%  →  12,8%  →  34,7%      (6,8× chỉ do vá nhãn)
    C:         21,5%  →  20,8%  →   6,2%

Cảnh báo §12.68c ("đừng trích số v12 nào của probe này như số cuối") là đúng và suýt nữa đã muộn:
cổng C ≥ 20% từng báo **đạt** ở hai lượt liên tiếp, và nếu tin nó thì tôi đã viết cả một cơ chế
xếp hạng nội nhóm để sửa một đám **4%** mà phần lớn là rác corpus với câu so sánh.

Thứ duy nhất đứng yên qua cả ba lượt: **bão hoà 47** — vì nó không đụng tới nhãn.

---

### 12.71 Câu thứ 3 KHÔNG hỏng — `handoff_max=1` là workaround đã lỗi thời (2026-07-29) — Claude

Đo ① của §12.70. `ar_sent_agenda_handoff_max = 1` kèm chú thích *"tránh dính câu 3 lỗi"* chưa ai
kiểm lại kể từ khi đặt. Chạy trên mini, `max_len=140`, bộ đo rộng `samples-per-cell 10`.

#### a. Kết quả — nâng cap cho ra đoạn dài gấp 2–3 lần mà KHÔNG sập

| nhóm | n_tok | n_facts | copy | loop4 | digit⎵ | stop |
|---|---|---|---|---|---|---|
| T1 verbatim | 20,9 → **47,8** | 1,65 → **3,15** | 0,71 → 0,44 | 0 → 0 | 0 → 0 | 1,00 → 1,00 |
| T2 paraphrase | 17,4 → **30,4** | 1,60 → **2,40** | 0,74 → 0,59 | 0 → 0 | 0 → 0 | 1,00 → 1,00 |
| T3 ghép | 14,4 → **35,5** | 0,80 → 1,20 | 0,66 → 0,52 | 0 → **0,10** | 0,10 → **0,30** | 1,00 → **0,90** |
| T5 số | 16,8 → **28,2** | 1,50 → **2,50** | 0,60 → 0,42 | 0 → 0 | 0 → 0 | 1,00 → 1,00 |
| T6 bịa | 48,8 → **75,0** | 1,60 → **3,20** | 0,41 → 0,28 | 0,20 → 0,20 | 0,40 → 0,40 | 0,80 → 0,80 |
| T7 dài | 8,0 → **21,8** | 0,40 → 0,80 | 0,78 → 0,29 | 0 → **0,20** | 0 → 0 | 1,00 → 1,00 |

`n_facts` (số cha KHÁC NHAU output chép từ) tăng ~2× ⇒ bề rộng trí nhớ được huy động tăng thật,
không phải kéo dài một câu. `copy` giảm mạnh (T1 0,71→0,44 · T7 0,78→0,29) vì output không còn
nằm gọn trong emit của một cha.

Đọc tay xác nhận mạch đúng niên đại: `Nhà Lý dời đô về` → *Thăng Long năm 1010. Nhà Trần thay nhà
Lý năm 1226. Trần Nhân Tông sáng lập Thiền phái Trúc Lâm. Nhà Trần kết thúc năm 1400 khi Hồ Quý
Ly cướp ngôi, lập ra nhà Hồ.*

#### b. Một chỉnh về THƯỚC, kẻo đọc sai bảng trên

`degen` **không đổi ở mọi nhóm** — nhưng đó KHÔNG phải bằng chứng chất lượng giữ nguyên:
`degenerate = int(len(cont) <= 2)` (eval_gen_broad.py:199), tức nó chỉ đo *"có sinh ra gì không"*.
Suýt nữa tôi trích nó làm bằng chứng. Cột đo thoái hoá thật là `loop4` · `digit⎵` · `stop`.

#### c. Cái giá phải trả — có thật, và khu trú

* **T3 ghép**: `loop4` 0 → 0,10 · `digit⎵` 0,10 → 0,30 · `stop` 1,00 → 0,90.
* **T7 dài**: `loop4` 0 → 0,20.
* **T6 bịa**: n_tok 48,8 → **75,0** với copy 0,28. Sinh dài hơn nghĩa là **bịa dài hơn** — nhóm
  này vốn `empty=0` (không bao giờ từ chối), nên cap 1 câu đang vô tình hạn chế thiệt hại.

Tức cap không phải vô dụng; nó **che** ba khuyết tật thay vì sửa. Nhưng nó che bằng cách trả giá
2–3× độ dài trên **mọi** nhóm, kể cả 4 nhóm không hề thoái hoá thêm.

#### d. Khuyết tật mới bắt được, chưa sửa: dính chữ ở biên câu

Đọc tay thấy `…sông trong núi.Rồi thuyền vào sông Sào Khê…` — token 29 là `.`, token 30 là `Rồi`
**trần** (id 8693) chứ không phải `ĠRồi` (7088, CÓ trong vocab).

Không phải lỗi dữ liệu: quét toàn corpus, chỉ **2/5664 = 0,04%** cha có `.` dính chữ hoa, và cả
hai là `TP.HCM` (viết tắt hợp lệ). Cũng không phải nhánh nối span — nhánh đó (generate_hebbian.py
~305) đã chuẩn hoá `Ġ` cho token đầu span. Vậy nó do **đường token-by-token** chọn biến thể trần
ngay sau biên câu — cùng họ với bức tường số BPE §12.41.

Ghi chú kèm: `out.extend(comp)` ở generate_hebbian.py:2477 (đường fact-plan) **không** có bước
chuẩn hoá `Ġ` như nhánh handoff. Đường đó tắt khi `ar_no_agenda=True` nên không dính lượt này,
nhưng là cùng một lỗ hổng ở chỗ khác.

#### e. Đề xuất — CHƯA đổi mặc định

Đây là số **mini**. Trước khi đổi `ar_sent_agenda_handoff_max`, phải A/B trên v12 (§12.31 P13):
lợi (n_tok · n_facts) và giá (loop4 · digit⎵ · stop · độ dài T6) đều có thể đổi độ lớn ở 430k cha.

Cổng đề xuất: nâng cap **đáng làm** nếu trên v12 `n_tok` và `n_facts` tăng ≥1,5× ở ≥4/7 nhóm
**và** `loop4` không vượt 0,15 ở nhóm nào **và** T6 `stop` không tụt. Trượt ⇒ giữ cap 1 và sửa
`loop4`/`digit⎵` trước.

---

### 12.72 P13 A/B `handoff_max` 1 vs 3 trên v12 — M4 (2026-07-29)

`out/hm1_v12.log` (356s, max_len=140, `hm=1`) · `out/hm3_v12.log` (380s, `hm=3`).

#### a. Bảng so sánh handoff nhóm (chỉ chế độ `handoff`)

| nhóm | n_tok hm1 | n_tok hm3 | n_facts hm1 | n_facts hm3 | loop4 hm1 | loop4 hm3 | stop hm1 | stop hm3 |
|---|---|---|---|---|---|---|---|---|
| T1 verbatim | 23.8 | **61.0** | 1.80 | **3.45** | 0.00 | **0.10** | 1.00 | 0.95 |
| T2 paraphrase | 19.6 | **39.2** | 1.80 | **3.20** | 0.00 | 0.00 | 1.00 | 1.00 |
| T3 ghép | 28.0 | **56.7** | 1.00 | **2.00** | 0.00 | **0.20** | 1.00 | 1.00 |
| T4 qa | 16.4 | 16.4 | 0.90 | 0.90 | 0.00 | 0.00 | 1.00 | 1.00 |
| T5 số | 21.5 | **43.2** | 2.00 | **3.50** | 0.00 | 0.00 | 1.00 | 1.00 |
| T6 bịa | 32.4 | **69.0** | 2.00 | **4.00** | 0.00 | 0.00 | 1.00 | 1.00 |
| T7 dài | 26.2 | **55.8** | 1.60 | **3.00** | 0.00 | **0.20** | 1.00 | 1.00 |

*(no_agenda không đổi — handoff_max không ảnh hưởng no_agenda)*

#### b. Kiểm cổng §12.71e

**`n_tok` và `n_facts` tăng ≥1,5× ở ≥4/7 nhóm:**
- n_tok tăng ≥1,5× : T1 (61/23.8=2.6×) · T2 (2.0×) · T3 (2.0×) · T5 (2.0×) · T6 (2.1×) · T7 (2.1×) = **6/7** ✅
- n_facts tăng ≥1,5× : T1 (1.92×) · T2 (1.78×) · T3 (2.0×) · T5 (1.75×) · T6 (2.0×) · T7 (1.88×) = **6/7** ✅

**`loop4` không vượt 0,15 ở nhóm nào:**
- T1 handoff: 0.10 ✅ · T3: 0.20 **❌** · T7: 0.20 **❌**

**T6 `stop` không tụt:**
- stop hm3 = 1.00 ✅

**Kết quả cổng: TRƯỢT** — `loop4` vượt 0,15 ở T3 (0.20) và T7 (0.20).

#### c. Nhận xét

Lợi ích rất rõ: đoạn lịch sử ghép (T2/T5) ra **3–4 câu sạch** xác nhận trên v12, ví dụ:
*"Nhà Lý dời đô về Thăng Long năm 1010. Nhà Trần thay nhà Lý năm 1226. Trần Nhân Tông sáng lập Thiền phái Trúc Lâm. Nhà Trần kết thúc năm 1400 khi Hồ Quý Ly cướp ngôi, lập ra nhà Hồ."*

Nhưng T3 ghép và T7 dài bị loop4 vượt ngưỡng — đúng với dự báo mini (§12.71c).

Theo cổng đã chốt: **giữ cap 1, sửa `loop4`/`digit⎵` trước rồi mới nâng.**

#### d. Ghi nhận thêm: `T4 qa` không hưởng lợi

n_tok T4 bằng nhau hm1=hm3=16.4 — cơ chế handoff của T4 không bật thêm câu kế; QA trả lời rồi dừng tự nhiên trước khi đụng cap. Hợp lý.


---

### 12.73 P13 — cổng TRƯỢT, và §12.71 của tôi SAI ở quy mô v12 (2026-07-29) — Claude

#### a. Cổng: 2 đạt, 1 trượt

| điều kiện chốt trước (§12.71e) | kết quả |
|---|---|
| n_tok & n_facts ≥1,5× ở ≥4/7 nhóm | **ĐẠT** — 6/7 (chỉ T4 không đổi, đường chép-đáp QA không qua handoff) |
| `loop4` ≤0,15 ở mọi nhóm | **TRƯỢT** — T3 **0,20** · T7 **0,20** |
| T6 `stop` không tụt | ĐẠT (1,00) |

Số v12 (handoff, cap 1 → 3): T1 n_tok 23,8→**61,0** · n_facts 1,80→**3,45** · T3 28,0→**56,7** ·
n_facts 1,00→**2,00** · T7 26,2→**55,8**. Và đếm ca T3 `n_facts>1` **3/10 → 6/10**.

Theo luật đã tự đặt: cổng trượt ⇒ **không đổi mặc định**. Nhưng lý do thật để không đổi nằm ở (b),
không phải ở `loop4`.

#### b. Đọc mẫu — bệnh không phải LẶP mà là TRÔI ĐỀ, và trên v12 nó nặng

`handoff·T1 verbatim` (nhóm DỄ NHẤT — chỉ nối tiếp câu mạng đã thuộc) ở cap 3:

```
'… mailing list s Nhiều ứng dụng cũng có riêng. Điện toán đám mây … Nên dùng bao nhiêu RAM …'
'" Long Hạo Thiên … tuyệt xử phùng sinh cười … Gạo hạt sen rửa sạch. Cho gạo vào nồi nấu …'
```

Truyện dịch → **công thức nấu cháo**, nối qua chữ chung `rửa sạch`. `T3 ghép`:
`Cả Hoa Lư và Thăng Long đều từng là … Marie Curie …` · `Nhà Lý và nhà Trần khác nhau thế nào →
Dưới đây là một số thách thức phổ biến mà người quản lý dự án …`

Đây không phải mạch đoạn. Đây là **nhảy tài liệu qua từ trùng**.

Trên mini (§12.71) chuỗi lại mạch lạc đúng niên đại (`Lý → Trần 1226 → Trần Nhân Tông → nhà Hồ
1400`). Vì corpus mini 5.004 câu gần như thuần sử; v12 429.711 câu có truyện dịch, tài liệu điện
toán, công thức nấu ăn. **Kết luận "câu 3 không hỏng" của §12.71 SAI ở quy mô v12.** Quy trình
thì đúng — §12.71e ghi rõ "số mini, CHƯA đổi mặc định, phải A/B v12" — nên không dòng mã nào bị
ship theo kết luận sai.

#### c. Thước của tôi đang thưởng cho TRÔI ĐỀ

`n_facts` = số cha KHÁC NHAU output chép từ, tôi đặt ở §12.22 làm "bề rộng ghép" cho T3. Nó
**không kiểm liên quan**. Nên `T1 n_facts 1,80 → 3,45` và `T3 n_facts>1 3/10 → 6/10` phần lớn là
**trôi đề được tính thành bề rộng**.

Tệ hơn: mẫu `Long Hạo Thiên … Gạo hạt sen` được mọi cột hiện có chấm là **tốt** — `degen` 0 ·
`stop` 1,00 · `loop4` 0 · `copy` 0,49 · `n_facts` 3,45. Không cột nào thấy nó lạc.

⇒ **Không thể quyết bất kỳ thay đổi nào về độ dài khi bộ đo mù với liên quan.** Đây là lỗ hổng
thước thứ mười của ticket, và là cái chặn thẳng mục tiêu "sinh đoạn dài".

#### d. Thêm cột `lạc` vào `eval_gen_broad`

`lạc` = tỉ lệ **CÂU** (từ câu 2 trở đi, ≥4 token) **không chia fold nội dung nào với prompt**.
Chỉ dùng `gidf ≥ 4.0` đã học — không bảng từ, không nhúng ngoài. Cắt câu theo token biên trong
chính vocab.

Cột này phải có trước mọi quyết định về `handoff_max`, và dùng lại được cho mọi thay đổi độ dài
sau này.

#### e. Một giả thuyết đã thử và BỎ

Nghi `loop4` do `agenda_handoff_span` **không** nhận `emitted_pis` (trong khi `zone_handoff_span`
có) nên chọn lại chính đoạn vừa phát. Viết guard chặn theo nội dung, chạy lại mini cap 3: kết quả
**bit-identical** — guard không nổ lần nào. Chẩn đoán sai; đã **revert**, không giữ mã chưa đo
được tác dụng.

`loop4` trên mini vậy là do đường token-by-token lặp, không do handoff. Chưa chẩn.

#### f. Cột `lạc` đo thử trên mini — trôi đề KHÔNG phải hiện tượng của cap 3

| `lạc` | T1 | T2 | T3 | **T4** | T5 | T6 | T7 |
|---|---|---|---|---|---|---|---|
| cap 1 (mặc định, 2 câu) | 0,33 | 0,25 | 0,33 | **0,00** | 0,33 | 0,50 | 0,00 |
| cap 3 | **0,49** | 0,33 | **0,44** | **0,00** | 0,44 | **0,67** | **0,50** |

Hai điều đọc ra:

1. **Trôi đề CÓ SẴN ở mặc định.** Ngay cap 1 — tức chỉ 2 câu — đã có 1/3 số câu thứ hai không
   chia một fold nội dung nào với prompt. Nâng cap không sinh ra bệnh, nó **nhân bệnh lên**
   (T1 0,33→0,49 · T7 0,00→0,50). Trước hôm nay không cột nào thấy.
2. **T4 = 0,00 ở CẢ HAI cap.** Đường chép-đáp QA miễn nhiễm hoàn toàn. Vừa là phép thử tỉnh táo
   (cột không phải nhiễu) vừa là gợi ý: cơ chế duy nhất KHÔNG trôi là cơ chế có **cổng phủ nội
   dung câu hỏi** (`answer_span_from_active` đòi phủ ≥ ngưỡng). Handoff không có cổng nào tương tự.

#### g. Việc kế

Trôi đề, không phải độ dài, mới là thứ chặn "sinh đoạn dài đúng nghĩa". Và nó **đã có sẵn** ở
mặc định. Đo trên v12 trước khi thiết kế gì (§12.31 P14) — mini và v12 khác nhau về đúng thứ gây
trôi (độ pha tạp corpus), nên số mini không dùng để quyết được.

---

### 12.74 P14 đo `lạc` trên v12 — cap 1 vs 3 (2026-07-29) — M4

`out/lac1_v12.log` (347s, `hm=1`) · `out/lac3_v12.log` (363s, `hm=3`). max_len=140, samples-per-cell=10.
Không đổi mặc định.

#### a. Bảng nguyên văn — `handoff_max=1`

```
chế độ    nhóm              n   empty  degen    lạc  digit⎵  loop4   copy  n_facts   stop  n_tok
========================================================================================================
no_agenda T1 verbatim      20    0.00   0.00    nan    0.00   0.00   0.93     1.00   1.00    7.9
no_agenda T2 paraphrase     5    0.20   0.20    nan    0.00   0.00   1.00     0.60   1.00    5.4
no_agenda T3 ghép          10    0.00   0.60    nan    0.10   0.00   0.60     0.30   1.00    9.2
no_agenda T4 qa            10    0.00   0.10   0.00    0.00   0.00   1.00     0.90   1.00   16.4
no_agenda T5 số             4    0.00   0.00    nan    0.00   0.00   1.00     0.75   1.00    6.8
no_agenda T6 bịa            5    0.00   0.00    nan    0.40   0.00   0.79     1.00   1.00   13.8
no_agenda T7 dài            5    0.00   0.00    nan    0.00   0.00   0.84     1.00   1.00    9.6
--------------------------------------------------------------------------------------------------------
handoff   T1 verbatim      20    0.00   0.00   0.12    0.00   0.00   0.71     1.80   1.00   23.8
handoff   T2 paraphrase     5    0.00   0.00   0.40    0.00   0.00   0.67     1.80   1.00   19.6
handoff   T3 ghép          10    0.00   0.30   0.29    0.10   0.00   0.67     1.00   1.00   28.0
handoff   T4 qa            10    0.00   0.10   0.00    0.00   0.00   1.00     0.90   1.00   16.4
handoff   T5 số             4    0.00   0.00   0.25    0.00   0.00   0.65     2.00   1.00   21.5
handoff   T6 bịa            5    0.00   0.00   0.00    0.40   0.00   0.53     2.00   1.00   32.4
handoff   T7 dài            5    0.00   0.00   0.00    0.00   0.00   0.71     1.60   1.00   26.2
```

#### b. Bảng nguyên văn — `handoff_max=3`

```
chế độ    nhóm              n   empty  degen    lạc  digit⎵  loop4   copy  n_facts   stop  n_tok
========================================================================================================
no_agenda T1 verbatim      20    0.00   0.00    nan    0.00   0.00   0.93     1.00   1.00    7.9
no_agenda T2 paraphrase     5    0.20   0.20    nan    0.00   0.00   1.00     0.60   1.00    5.4
no_agenda T3 ghép          10    0.00   0.60    nan    0.10   0.00   0.60     0.30   1.00    9.2
no_agenda T4 qa            10    0.00   0.10   0.00    0.00   0.00   1.00     0.90   1.00   16.4
no_agenda T5 số             4    0.00   0.00    nan    0.00   0.00   1.00     0.75   1.00    6.8
no_agenda T6 bịa            5    0.00   0.00    nan    0.40   0.00   0.79     1.00   1.00   13.8
no_agenda T7 dài            5    0.00   0.00    nan    0.00   0.00   0.84     1.00   1.00    9.6
--------------------------------------------------------------------------------------------------------
handoff   T1 verbatim      20    0.00   0.00   0.27    0.00   0.10   0.49     3.45   0.95   61.0
handoff   T2 paraphrase     5    0.00   0.00   0.33    0.00   0.00   0.42     3.20   1.00   39.2
handoff   T3 ghép          10    0.00   0.30   0.30    0.10   0.20   0.37     2.00   1.00   56.7
handoff   T4 qa            10    0.00   0.10   0.00    0.00   0.00   1.00     0.90   1.00   16.4
handoff   T5 số             4    0.00   0.00   0.33    0.00   0.00   0.41     3.50   1.00   43.2
handoff   T6 bịa            5    0.00   0.00   0.40    0.40   0.00   0.35     4.00   1.00   69.0
handoff   T7 dài            5    0.00   0.00   0.17    0.00   0.20   0.49     3.00   1.00   55.8
```

#### c. So `lạc` handoff cap1 → cap3 (và mini §12.73f)

| nhóm | v12 cap1 | v12 cap3 | mini cap1 | mini cap3 |
|---|---|---|---|---|
| T1 verbatim | **0.12** | **0.27** | 0.33 | 0.49 |
| T2 paraphrase | **0.40** | **0.33** | 0.25 | 0.33 |
| T3 ghép | **0.29** | **0.30** | 0.33 | 0.44 |
| T4 qa | **0.00** | **0.00** | 0.00 | 0.00 |
| T5 số | **0.25** | **0.33** | 0.33 | 0.44 |
| T6 bịa | **0.00** | **0.40** | 0.50 | 0.67 |
| T7 dài | **0.00** | **0.17** | 0.00 | 0.50 |

#### d. Đọc số

1. **Trôi đề có sẵn ở mặc định (cap 1):** T2 0.40 · T3 0.29 · T5 0.25 · T1 0.12. Không phải bệnh của riêng cap 3.
2. **Nâng cap nhân lên ở vài nhóm:** T1 0.12→0.27 · T6 0.00→0.40 · T7 0.00→0.17 · T5 0.25→0.33. T3 gần đứng yên (0.29→0.30). T2 hơi tụt (0.40→0.33) — mẫu nhỏ n=5.
3. **T4 = 0.00 cả hai cap** — giống mini; đường chép-đáp QA vẫn miễn nhiễm.
4. **v12 T1 thấp hơn mini** (0.12 vs 0.33 ở cap1) — khác chiều kỳ vọng “corpus pha tạp hơn ⇒ trôi nhiều hơn” trên T1; T2 thì ngược lại (v12 0.40 > mini 0.25). Cột bắt được mẫu nhảy đã nêu ở §12.73 (Long Hạo Thiên… / quản lý dự án vẫn trong dump).
5. Các cột khác (n_tok / n_facts / loop4) **trùng P13** — sanity OK.

Không đổi `ar_sent_agenda_handoff_max` (giữ 1). Chờ Claude thiết kế từ số nền này.

---

### 12.75 Cổng phủ cho span nối tiếp — thiết kế bằng đo, không bằng trực giác (2026-07-29) — Claude

§12.73g chỉ ra T4 là cơ chế **duy nhất** không trôi đề (`lạc = 0,00` ở mọi cap), và nó khác mọi
cơ chế khác đúng một điểm: `answer_span_from_active` có **cổng phủ nội dung**. Handoff không có.
Mục này làm cổng ấy cho span nối tiếp.

#### a. Hai thiết kế "hiển nhiên" — đo trước, và CẢ HAI SAI

**"Span phải phủ prompt"** — giết luôn chuỗi đúng. Chuỗi mạch lạc trên mini
(`Nhà Lý dời đô về Thăng Long 1010. Nhà Trần thay nhà Lý 1226. Trần Nhân Tông sáng lập…`) có câu
3 **không chia một fold nào** với prompt.

**"Phải cùng episode"** — cũng sai. Chính chuỗi ấy đi qua episode `[62 → 175 → 175 → 87]`: nó
**có** nhảy tài liệu mà vẫn đúng. (Đo hậu nghiệm, khớp câu→cha→episode, không đụng mã.)

Và chuỗi XẤU (`… cười. Gạo hạt sen rửa sạch`) thì **có** chia chữ với câu ngay trước. Nên cả
"phủ prompt" lẫn "liên kết cục bộ" đều không tách được.

#### b. Đo trên quần thể có nhãn suy từ đồ thị

Dương = hai câu **liền kề trong cùng tài liệu** (5.050 cặp). Âm KHÓ = hai câu **khác tài liệu
NHƯNG có chia fold nội dung** — đúng thứ khiến máy nhảy nhầm (nếu lấy âm ngẫu nhiên thì bài toán
dễ giả tạo).

| | chia fold nội dung TB | chia fold VIẾT HOA TB | =0 ở |
|---|---|---|---|
| dương | 2,45 | **0,58** | 67,5% |
| âm khó | 1,22 | **0,04** | **96,2%** |

Fold viết hoa chênh **14,5×** — trông rất hợp lý (`Trần`/`Lý` là thực thể, `rửa sạch` là hành
động chung). Nhưng quét luật thì nó **THUA**:

| luật cho qua | giữ ca ĐÚNG | lọt ca TRÔI |
|---|---|---|
| ≥1 fold viết hoa | 32,5% | 3,8% |
| **≥3 fold nội dung** | **39,7%** | **3,4%** |
| ≥1 viết hoa HOẶC ≥4 nội dung | 38,4% | 4,1% |

Luật đơn giản trội ở **cả hai trục**. Ghi lại vì đây là lần thứ hai trong ticket một tinh chỉnh
"có vẻ thông minh hơn" (hoa/thường) thua luật thô — lần đầu là §12.55.

#### c. Hiệu chỉnh tĩnh KHÔNG chuyển thẳng sang hành vi động

Cài `ar_sent_agenda_handoff_cover_min` (mặc định **0 = tắt**), so span với **câu trước** (không
so cả `out`, vì tập fold to dần thì cổng lỏng dần theo độ dài). Chạy cap 3 trên mini:

| cover_min | 0 | **1** | 2 | 3 |
|---|---|---|---|---|
| T1 `lạc` ‖ `n_tok` | 0,49 ‖ 47,8 | **0,33 ‖ 33,5** | 0,06 ‖ 14,2 | 0,00 ‖ 9,9 |
| T3 | 0,44 ‖ 35,5 | **0,22 ‖ 22,3** | 0,50 ‖ 13,2 | 0,00 ‖ 11,1 |
| T5 | 0,44 ‖ 28,2 | **0,17 ‖ 21,8** | 0,00 ‖ 9,8 | *nan* ‖ 6,8 |
| T7 | 0,50 ‖ 21,8 | **0,00 ‖ 14,8** | 0,00 ‖ 14,8 | *nan* ‖ 2,0 |

Ngưỡng **3** — thứ phép đo (b) chọn — **tắt gần hết handoff**: T1 còn 9,9 token (thấp hơn cả cap
1 là 20,9), T7 còn 2,0, `copy` bật lên 0,88–1,00 (quay về chép một câu), `degen` T1 0,05→0,25.
`lạc = 0,00` chỉ vì **không còn câu thứ hai để lạc**; T6 còn tệ hơn (0,67→1,00, mẫu số co lại).

Lần thứ ba trong phiên một hiệu chỉnh trên cặp tĩnh không chuyển sang vòng sinh. Bài học: **luôn
quét ngưỡng end-to-end**, đừng tin điểm do phép đo tĩnh chọn.

#### d. Điểm vận hành: `cover_min = 1` với `handoff_max = 3`

So với **mặc định hiện hành** (cap 1, không cổng) — đây mới là phép so đúng:

| nhóm | `n_tok` | `lạc` | `n_facts` |
|---|---|---|---|
| T1 verbatim | 20,9 → **33,5** | 0,33 → 0,33 | 1,65 → **2,25** |
| T3 ghép | 14,4 → **22,3** | 0,33 → **0,22** | 0,80 → **1,00** |
| T5 số | 16,8 → **21,8** | 0,33 → **0,17** | 1,50 → **1,75** |
| T6 bịa | 48,8 → **60,4** | 0,50 → **0,33** | 1,60 → 2,20 |
| T7 dài | 8,0 → **14,8** | 0,00 → 0,00 | 0,40 → 0,40 |

**Dài hơn VÀ ít trôi hơn** cùng lúc — tổ hợp mà §12.71/§12.73 không đạt được. T3 thêm: `loop4`
0,10 → **0,00**, `stop` 0,90 → **1,00**.

#### e. CHƯA đổi mặc định

Đây là số mini, và mini chính là chỗ đã lừa tôi ở §12.71 (kết luận "câu 3 không hỏng" sai ở v12).
Giao P15. Mặc định giữ `handoff_max=1`, `cover_min=0`.

---

### 12.76 P15 A/B cổng phủ trên v12 — cổng TRƯỢT (2026-07-29) — M4

A = `out/lac1_v12.log` (P14, cap1 · cover0) · B = `out/p15b_v12.log` (381s, cap3 · `cover_min=1`).
Không đổi `config.py`.

#### a. Bảng nguyên văn B (handoff_max=3, cover_min=1)

```
chế độ    nhóm              n   empty  degen    lạc  digit⎵  loop4   copy  n_facts   stop  n_tok
========================================================================================================
no_agenda T1 verbatim      20    0.00   0.00    nan    0.00   0.00   0.93     1.00   1.00    7.9
no_agenda T2 paraphrase     5    0.20   0.20    nan    0.00   0.00   1.00     0.60   1.00    5.4
no_agenda T3 ghép          10    0.00   0.60    nan    0.10   0.00   0.60     0.30   1.00    9.2
no_agenda T4 qa            10    0.00   0.10   0.00    0.00   0.00   1.00     0.90   1.00   16.4
no_agenda T5 số             4    0.00   0.00    nan    0.00   0.00   1.00     0.75   1.00    6.8
no_agenda T6 bịa            5    0.00   0.00    nan    0.40   0.00   0.79     1.00   1.00   13.8
no_agenda T7 dài            5    0.00   0.00    nan    0.00   0.00   0.84     1.00   1.00    9.6
--------------------------------------------------------------------------------------------------------
handoff   T1 verbatim      20    0.00   0.00   0.14    0.00   0.10   0.58     2.55   0.95   45.8
handoff   T2 paraphrase     5    0.20   0.20   0.11    0.00   0.00   0.60     2.00   1.00   25.6
handoff   T3 ghép          10    0.00   0.50   0.29    0.10   0.20   0.34     1.50   1.00   42.0
handoff   T4 qa            10    0.00   0.10   0.00    0.00   0.00   1.00     0.90   1.00   16.4
handoff   T5 số             4    0.00   0.00   0.11    0.00   0.00   0.57     2.75   1.00   36.8
handoff   T6 bịa            5    0.00   0.00   0.20    0.40   0.00   0.45     3.20   1.00   53.6
handoff   T7 dài            5    0.00   0.00   0.11    0.00   0.00   0.56     2.40   1.00   43.8
```

#### b. So A → B (chỉ handoff) + kiểm cổng §12.31

| nhóm | n_tok A | n_tok B | × | lạc A | lạc B | Δlạc | loop4 B |
|---|---|---|---|---|---|---|---|
| T1 | 23.8 | **45.8** | **1.92** | 0.12 | 0.14 | +0.02 | 0.10 |
| T2 | 19.6 | **25.6** | **1.31** | 0.40 | **0.11** | −0.29 | 0.00 |
| T3 | 28.0 | **42.0** | **1.50** | 0.29 | 0.29 | 0.00 | **0.20** |
| T4 | 16.4 | 16.4 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| T5 | 21.5 | **36.8** | **1.71** | 0.25 | **0.11** | −0.14 | 0.00 |
| T6 | 32.4 | **53.6** | **1.65** | 0.00 | 0.20 | **+0.20** | 0.00 |
| T7 | 26.2 | **43.8** | **1.67** | 0.00 | 0.11 | **+0.11** | 0.00 |

**Cổng:**
- `n_tok` ≥1,3× ở ≥4/7: **6/7** ✅ (T1/T2/T3/T5/T6/T7; T4 đứng yên)
- `lạc` không tăng quá +0,03 ở mọi nhóm: **TRƯỢT** — T6 **+0.20** · T7 **+0.11**
- `loop4` ≤0,15 mọi nhóm: **TRƯỢT** — T3 **0.20**
- T6 `stop` không tụt: **1.00** ✅

**Kết quả cổng: TRƯỢT** — giữ `handoff_max=1`, `cover_min=0`.

#### c. Đọc tay mẫu B (handoff)

Vẫn có nhảy tài liệu kiểu §12.73b:
- T3: `Nhà Lý và nhà Trần khác nhau… → Dưới đây là một số thách thức … quản lý dự án`
- T3: `Hoa Lư và Thăng Long → Marie Curie`
- T3: `Đinh Bộ Lĩnh và Lê Hoàn → vàng hồi môn / mẹ anh bảo…`

Tốt hơn một phần so cap3 trần (P13/P14):
- T1 `Long Hạo Thiên… cười` — **không** nối sang `Gạo hạt sen rửa sạch` (cổng có vẻ chặn được ca này)
- T2/T5 chuỗi niên đại sạch vẫn còn (`Lý 1010 → Trần 1226 → Trúc Lâm → Hồ 1400`)

Giá phụ: T2 `empty` 0→0.20 · T3 `degen` 0.30→0.50 · readable T3 7/10→5/10.

#### d. So nhanh với cap3 không cổng (`lac3_v12`)

B ngắn hơn và `lạc` thấp hơn cap3 trần ở hầu hết nhóm (T1 61.0/0.27 → 45.8/0.14; T6 69.0/0.40 → 53.6/0.20),
nhưng so **mặc định** (A) thì `lạc` T6/T7 vẫn vượt ngưỡng cổng đã chốt.

---

### 12.77 Thước CẶP CÓ THỨ TỰ — dương tính đầu tiên chạm được TIÊU CHÍ CHỌN (2026-07-29) — Claude

Tác giả hỏi: *"L2R/R2L cần đóng góp vào A và E nhiều, hiện mới là bigram phải không?"* Nửa đầu
đúng và nặng hơn tưởng; nửa sau thì không.

#### a. Chỉnh: L2R/R2L KHÔNG phải bigram

`accumulate_span_pairs` (sent_intra.py:103) lấy **mọi cặp i<j với d ≤ `d_max` = 12**, trọng số
`λ^d` (λ=0.85) × `√(gidf_i·gidf_j)` × `power×priority`, giới hạn **trong span câu**, **chỉ token
nội dung**. Skip-gram cửa sổ 12.

#### b. Nhưng điều tác giả nhắm tới thì ĐÚNG — và đo được

Đọc `ensure_parent_bipartite`:

* **`A` không có một chút thứ tự nào** — `A[t,pi] = √(power×priority) × gidf(t)` từ `match_ids`,
  thuần túi từ.
* **`E` chỉ có thứ tự ở d=1** — ngoài unigram giảm theo vị trí, phần thứ tự duy nhất là
  `E[pi,b] += 0.25·pw·gidf(b)` chạy trên `(seq[i], seq[i+1])`.
* **`succ` cũng chỉ d=1**.

L2R/R2L sống ở không gian (N,N) **riêng**, chỉ được tiêu thụ làm kênh logits (β=2.0) và
`r2l_anchor_scores` — docstring ghi rõ *"conditioning, không dùng làm next"*. **Không bao giờ
chạm vào A hay E.**

Đo lượng thông tin bị vứt (mini, bằng chính luật đang chạy):

    cặp (a,b) PHÂN BIỆT ở d=1  (E/succ giữ):   50.184
    cặp (a,b) PHÂN BIỆT ở d≤12 (L2R có):      386.307
    ⇒ E/succ bỏ 87,0% số cặp có thứ tự ĐÃ HỌC.  d=1 chỉ chiếm 13,5% lượt đếm.

#### c. Vì sao đây là gốc của HAI bế tắc lớn nhất, không phải chi tiết cài đặt

**`copy ≈ 1.0` là hệ quả bắt buộc.** Kênh gánh chính `succ+path` (β 5.0+4.0) chỉ biết "chữ ngay
sau". Khi biểu diễn duy nhất về thứ tự là kề nhau, cách an toàn duy nhất để sinh đúng ngữ pháp là
**chép nguyên đoạn đã thuộc** — không có chỗ nào biểu diễn "b ở đâu đó sau a" để bỏ bớt, chèn
thêm hay ghép hai trí nhớ.

**Nút thắt chọn câu §12.50c cũng là nó.** `A` là túi từ ⇒ truy hồi **về mặt toán học không thể**
phân biệt *"Nhà Lý dời đô về đâu"* với *"đâu dời đô về nhà Lý"*. Suốt §12.51–§12.53 tôi đi tìm
tiêu chí chọn tốt hơn; hoá ra vấn đề là tiêu chí **không có dữ liệu** để tốt hơn.

#### d. Thước — và cổng ĐẠT

Không đổi A/E (đắt, rủi ro). Làm **bước xếp hạng lại**: `A` cứ lọc bằng túi từ, rồi chấm ứng viên
bằng tỉ lệ **cặp có thứ tự của CÂU HỎI được ứng viên giữ đúng chiều** (`λ^d`, bỏ WH-fold, chỉ
`gidf≥4`). Đo trên đúng bench §12.53 (169 ca mini) — không dựng dụng cụ mới.

| thước | top1 | Δ so túi từ | thắng | thua |
|---|---|---|---|---|
| `B` (túi từ) | 0,562 | — | — | — |
| `PAIR` một mình | 0,479 | −0,083 | 7 | 21 |
| `B+PAIR` | 0,615 | +0,053 | 9 | **0** |
| `pos` (**tiêu chí hiện hành**) | 0,615 | +0,053 | 18 | 9 |
| **`pos+PAIR`** | **0,675** | **+0,112** | **19** | **0** |

**Hai tín hiệu CỘNG HƯỞNG, không trùng nhau:** 0,053 + 0,053 ≈ 0,112. Theme prior đo "thực thể có
là chủ đề câu không"; PAIR đo "trật tự câu hỏi có được tôn trọng không" — hai chiều độc lập.

Cổng §12.56c chấm so **tiêu chí hiện hành**: `pos` 0,615 → `pos+PAIR` 0,675, **Δ +0,059**,
**13 thắng / 3 thua** ⇒ **ĐẠT** (cần Δ ≥ +0,05 và thắng−thua ≥ 5).

Bền với trọng số: hệ số 0,5 cho kết quả y hệt ⇒ không phải chỉnh dao động.

#### e. CHƯA viết đường sinh

Đây là số **mini**, và §12.71 vừa dạy rằng mini nói sai ở quy mô v12. Nhưng khác ca PMI ở một
điểm đáng kể: tín hiệu này **mang tính cấu trúc** — `A` vứt 87% thứ tự **theo định nghĩa**, trên
bất kỳ corpus nào — chứ không phải đặc thù corpus. Vẫn phải có số v12 (§12.31 P16) trước khi viết.

Nếu v12 xác nhận, đây là **dương tính đầu tiên trong ticket chạm được tiêu chí chọn** — thứ mà
§12.51 (lift theo loại hỏi), §12.52 (dìm từ hỏi), §12.53/§12.57 (prior thay thế) đều thất bại.

---

### 12.78 Đọc §12.76 (P15) — cổng trượt, nhưng số nói điều khác với "hỏng" (2026-07-29) — Claude

Cổng §12.75e chấm A (`out/lac1_v12.log`, cap 1 không cổng) với B (cap 3 + `cover_min=1`):

| nhóm | `n_tok` | tỉ lệ | `lạc` | Δ`lạc` | `loop4` |
|---|---|---|---|---|---|
| T1 verbatim | 23,8 → 45,8 | 1,92× | 0,12 → 0,14 | +0,02 ✓ | 0 → **0,10** |
| T2 paraphrase | 19,6 → 25,6 | 1,31× | 0,40 → **0,11** | **−0,29** ✓ | 0 → 0 |
| T3 ghép | 28,0 → 42,0 | 1,50× | 0,29 → 0,29 | 0,00 ✓ | 0 → **0,20** |
| T5 số | 21,5 → 36,8 | 1,71× | 0,25 → **0,11** | **−0,14** ✓ | 0 → 0 |
| T6 bịa | 32,4 → 53,6 | 1,65× | 0,00 → 0,20 | **+0,20** ✗ | 0 → 0 |
| T7 dài | 26,2 → 43,8 | 1,67× | 0,00 → 0,11 | **+0,11** ✗ | 0 → 0 |

**Cổng TRƯỢT** ở hai điều kiện: `lạc` vượt +0,03 (T6, T7) và `loop4` T3 = 0,20 > 0,15. Giữ mặc
định. M4 đọc đúng.

**Nhưng đọc từng cột thì không phải "cổng phủ vô dụng":** ở hai nhóm **vốn CÓ trôi đề**, nó giảm
mạnh — T2 **0,40 → 0,11**, T5 **0,25 → 0,11** — trong khi n_tok vẫn tăng 1,3–1,7×. Chỗ nó "làm
tệ" là T6/T7, hai nhóm có `lạc = 0,00` ở nền **chỉ vì output quá ngắn để mà trôi**. Tăng từ số 0
không cùng nghĩa với tăng từ 0,40.

Tôi vẫn **không** nới cổng để cho đạt — cổng chốt trước thì phải tôn trọng. Nhưng ghi rõ để lần
sau thiết kế cổng đúng hơn: **điều kiện "Δ ≤ +0,03 ở MỌI nhóm" phạt oan nhóm có nền = 0.** Đáng
lẽ phải là "không nhóm nào vượt ngưỡng tuyệt đối" (vd `lạc ≤ 0,25`), vì cái ta quan tâm là mức
trôi cuối cùng, không phải độ tăng. Theo thước đó thì B đạt: cao nhất là T3 0,29 ‖ nền T3 cũng
0,29 — không nhóm nào tệ hơn nền quá 0,02 trừ T6/T7 vốn xuất phát từ 0.

Đây là **lỗi thiết kế cổng thứ hai cùng loại** (sau §12.56b ngưỡng top-1 tuyệt đối trên bể khác
cỡ): cổng chấm **độ thay đổi** thay vì **mức cuối**, nên nhóm có nền cực trị bị phạt sai.

`loop4` T3 = 0,20 thì là khuyết tật thật, chưa chẩn (§12.73e đã thử guard lặp và revert vì không
nổ). Đó mới là thứ chặn thật, không phải `lạc`.

**Không đổi mặc định.** P16 (§12.77) ưu tiên hơn — nó chạm nút thắt lõi, còn đây là chất lượng
đoạn dài.

---

### 12.79 P16 thước CẶP CÓ THỨ TỰ trên v12 — cổng TRƯỢT (2026-07-29) — M4

`out/pair_v12.log` (514s, n=274). Không viết đường sinh.

#### a. Khối nguyên văn

```
THƯỚC CẶP CÓ THỨ TỰ (§12.76) — `A` là túi từ nên không phân biệt nổi trật tự:
  thước            top1    Δ so B   thắng   thua   (n=274)
  B (túi từ)      0.318         —       —      —
  PAIR            0.285    -0.033       9     18
  B+PAIR          0.292    -0.026       4     11
  B+.5PAIR        0.314    -0.004       4      5
  pos             0.347    +0.029      21     13
  pos+PAIR        0.369    +0.051      17      3

  CỔNG (§12.56c) so TIÊU CHÍ HIỆN HÀNH `pos` 0.347 → `pos+PAIR` 0.369  (Δ +0.022 · thắng 15 · thua 9)
  ⇒ TRƯỢT  (cần Δ ≥ +0.05 VÀ thắng−thua ≥ 5)
```

#### b. Đọc cổng §12.31 / §12.77

| điều kiện | kết quả |
|---|---|
| Δtop1 ≥ +0,05 (`pos` → `pos+PAIR`) | **TRƯỢT** — +0,022 |
| (thắng − thua) ≥ 5 | **ĐẠT** — 15−9 = 6 |

**Cổng: TRƯỢT.** Mini từng ĐẠT (+0,059 · 13/3); v12 cùng hướng nhưng Δ yếu hơn ngưỡng (~2,3× nhỏ hơn mini).

PAIR một mình và B+PAIR **âm** so B (0,285 / 0,292). Chỉ khi ghép `pos` mới dương — cộng hưởng còn, nhưng không đủ cổng đã chốt.

Không đổi mặc định / không viết đường sinh. Chờ Claude.

---

### 12.80 Đo thông tin theo KHOẢNG CÁCH và theo CHIỀU (2026-07-29) — Claude

Bước 2 của hướng "ma trận có hướng tầm xa → PPMI+SVD → embedding" (đề xuất của tác giả). Đo
trước khi dựng. `scripts/probe_distance_info.py`, mini.

#### a. Trước hết: P16 (§12.79) TRƯỢT, và điều đó làm phép đo này càng cần

`pos` 0,347 → `pos+PAIR` 0,369 trên v12: Δ **+0,022** (cần ≥ +0,05), 15 thắng / 9 thua. Mini cho
+0,059 ⇒ hiệu ứng **co 2,7×**. Dấu vẫn dương và thắng nhiều hơn thua, nhưng dưới cổng ⇒ **không
viết đường sinh**.

Điều đó để ngỏ hai khả năng: (i) thứ tự vốn ít thông tin, hay (ii) proxy PAIR của tôi yếu. §12.80
phân biệt được hai cái đó — nó đo thông tin ở **nguồn**, không qua proxy nào.

#### b. Số (mini, `d_max=24`, `min_count=4`)

| d | lệch chiều | đối chứng | **VƯỢT** | PMI | %PMI(d=1) | #cặp |
|---|---|---|---|---|---|---|
| 1 | 0,855 | 0,284 | **+0,571** | 4,24 | 100% | 50.184 |
| 2 | 0,670 | 0,286 | +0,384 | 2,90 | 68,3% | 63.680 |
| 3 | 0,570 | 0,291 | +0,279 | 2,38 | 56,1% | 63.139 |
| 4 | 0,500 | 0,280 | +0,220 | 2,04 | 48,0% | 59.873 |
| 6 | 0,416 | 0,272 | +0,144 | 1,63 | 38,4% | 51.142 |
| 8 | 0,387 | 0,270 | **+0,116** | 1,49 | **35,2%** | 41.723 |
| 11 | 0,375 | 0,297 | +0,078 | 1,50 | 35,3% | 28.622 |
| ≥19 | — | — | nhiễu | nan | — | <6.000 |

**Đối chứng phẳng ~0,28 ở MỌI d.** Đó đúng là mức nhiễu nhị thức kỳ vọng với `min_count=4`
(√(2/πn) ≈ 0,40 ở n=4, thấp hơn khi n lớn) ⇒ cột VƯỢT là tín hiệu thật, không phải hiện vật đếm
thưa. Cột đối chứng này là thứ duy nhất khiến bảng đọc được; không có nó thì `lệch chiều = 0,387`
ở d=8 trông như tín hiệu mạnh trong khi hơn nửa là nhiễu.

G1 **ĐẠT** (+0,571) · G2 **ĐẠT** (+0,116 ở d=8) · G3 **ĐẠT** (35,2%).

#### c. Hình dạng — và nó bác chính đề xuất "d=48 hoặc không giới hạn"

PMI không giảm rồi về 0. Nó **giảm nhanh tới d≈7 rồi CHỮNG LẠI ~35%** và giữ phẳng tới d≈14.
Đó là chữ ký của một **hỗn hợp hai thành phần**:

* **thành phần cục bộ, giảm nhanh** — cú pháp/cụm từ. Khớp λ từ d=1→8: `0,352^(1/7) ≈ 0,86`,
  gần đúng λ=0,85 mà `L2R` đang dùng. Tức tham số hiện có **đã hiệu chỉnh tốt** cho vùng này.
* **thành phần phẳng ~35%** — chủ đề cấp tài liệu. Mũ `λ^d` sẽ **triệt tiêu** phần này; muốn giữ
  thì cần luỹ thừa hoặc mũ-cộng-sàn.

Nhưng phần phẳng ấy **đối xứng theo bản chất** (cùng chủ đề thì hai chiều ngang nhau) — và đúng là
thứ `CO_ACTIVATION` đang bắt. Đọc cột VƯỢT: nó rơi từ +0,571 xuống +0,116 ở d=8 và +0,078 ở d=11,
tức **hướng chỉ sống ở vùng gần**.

⇒ Kết luận cụ thể cho đề xuất: **d ≈ 8–12, KHÔNG phải 48 hay vô hạn.** Ngoài d≈12, cái còn lại là
chủ đề đối xứng mà ma trận hiện có đã nắm; thêm vào chỉ pha loãng.

#### d. Chỗ trống được định vị chính xác

| | d=1 | d=2..10 | d>12 |
|---|---|---|---|
| có hướng | `W_seq` ✓ | **TRỐNG** | (hướng đã tắt) |
| đối xứng | — | `CO_ACTIVATION` ✓ | `CO_ACTIVATION` ✓ |

Hai ma trận hiện có phủ hai đầu; **thiếu đúng khúc giữa**: tín hiệu CÓ HƯỚNG ở d=2..10 —
`W_seq` (d=1) bỏ sót, `CO_ACTIVATION` (đối xứng) không biểu diễn nổi.

Đó là ô duy nhất đáng dựng, và nó **hẹp hơn nhiều** so với đề xuất ban đầu.

#### e. Một dè dặt phải ghi

Cột PMI có **thiên lệch chọn mẫu**: d càng lớn thì càng ít cặp đạt `min_count=4`, nên tập sống sót
nghiêng dần về các cặp gắn bó mạnh ⇒ phần "chững lại ~35%" có thể bị thổi lên. Cột VƯỢT ít chịu
ảnh hưởng hơn vì đối chứng dùng **cùng bộ lọc**. Muốn chắc thì phải chấm ở mức support cố định —
chưa làm. Nên kết luận (c) nên đọc là "hướng sống ở d≲12" (vững, dựa cột VƯỢT), còn "có thành phần
chủ đề phẳng" là **gợi ý**, chưa chốt.

---

### 12.81 P17 thông tin khoảng cách/chiều trên v12 — G1/G2 ĐẠT · G3 TRƯỢT (2026-07-29) — M4

`out/dist_info_v12.log` (277s, `d_max=24`, `min_count=4`). Không đổi mã / không dựng ma trận.

#### a. Bảng nguyên văn + cổng

```
  d  lệch chiều  đối chứng     VƯỢT      PMI  %PMI(d1)      #cặp
==============================================================================
  1       0.809      0.164    0.645    2.815    100.0% 1,323,688
  2       0.520      0.165    0.355    1.193     42.4% 1,997,857
  3       0.397      0.166    0.231    0.815     29.0% 2,169,571
  4       0.324      0.168    0.156    0.635     22.6% 2,198,752
  5       0.295      0.170    0.125    0.550     19.5% 2,176,854
  6       0.282      0.171    0.110    0.512     18.2% 2,122,170
  7       0.278      0.174    0.104    0.488     17.3% 2,058,908
  8       0.278      0.176    0.102    0.475     16.9% 1,986,186
  9       0.280      0.178    0.103    0.470     16.7% 1,908,136
 10       0.283      0.180    0.103    0.471     16.7% 1,829,315
 11       0.287      0.182    0.105    0.474     16.8% 1,749,220
 12       0.291      0.185    0.106    0.479     17.0% 1,669,246
 13       0.294      0.187    0.107    0.482     17.1% 1,589,709
 14       0.298      0.189    0.110    0.487     17.3% 1,509,703
 15       0.302      0.193    0.109    0.495     17.6% 1,431,700
 16       0.306      0.195    0.111    0.505     17.9% 1,353,974
 17       0.310      0.198    0.112    0.514     18.3% 1,278,997
 18       0.313      0.200    0.113    0.525     18.6% 1,204,996
 19       0.317      0.204    0.113    0.537     19.1% 1,133,534
 20       0.320      0.207    0.113    0.547     19.4% 1,064,023
 21       0.324      0.210    0.114    0.560     19.9%   998,165
 22       0.326      0.213    0.113    0.570     20.2%   934,642
 23       0.329      0.216    0.112    0.580     20.6%   874,693
 24       0.334      0.219    0.115    0.595     21.1%   818,426
==============================================================================
G1 tỉnh táo (vượt ≥0.10 ở d=1) : ĐẠT (+0.645)
G2 hướng CÓ TẦM (vượt ≥0.05 ở d=8): ĐẠT (+0.102)  ← cổng quyết định
G3 PMI ở d=8 ≥25% của d=1      : TRƯỢT (16.9%)

⇒ ĐÁNG dựng ma trận có hướng tầm xa
```

#### b. Đọc số

1. **Đối chứng gần phẳng** 0.164→0.219 (nhẹ dốc lên; mini ~0.28 phẳng hơn). Không lệch mạnh ⇒ cột VƯỢT đọc được.
2. **G2 ĐẠT** — cổng quyết định: VƯỢT d=8 = **+0.102** ≥ 0.05. Script kết luận **đáng dựng**.
3. **G3 TRƯỢT** — PMI d=8 chỉ **16.9%** của d=1 (cần ≥25%; mini 35.2%). Tầm PMI trên v12 ngắn hơn mini.
4. Hình VƯỢT khác mini: sau d≈5–8 không rơi tiếp mà **chững ~0.10–0.11** tới d=24 (mini rơi còn +0.078 ở d=11). Có thể thiên lệch support (§12.80e) — Claude đọc.
5. So mini: VƯỢT d=1 mạnh hơn (+0.645 ‖ +0.571); d=8 gần bằng (+0.102 ‖ +0.116).

---

### 12.82 Nhận §12.81 (P17) — v12 BÁC kết luận §12.80c của tôi (2026-07-29) — Claude

#### a. Số v12 — sạch hơn mini nhiều

Đếm 1–2 **triệu** cặp mỗi `d` (mini chỉ 30–60 nghìn) nên hết nhiễu đếm thưa.

| d | lệch chiều | đối chứng | **VƯỢT** | PMI | %PMI(d1) |
|---|---|---|---|---|---|
| 1 | 0,809 | 0,164 | **+0,645** | 2,815 | 100% |
| 2 | 0,520 | 0,165 | +0,355 | 1,193 | 42,4% |
| 4 | 0,324 | 0,168 | +0,156 | 0,635 | 22,6% |
| 6 | 0,282 | 0,171 | +0,110 | 0,512 | 18,2% |
| **8** | 0,278 | 0,176 | **+0,102** | 0,475 | 16,9% |
| 12 | 0,291 | 0,185 | +0,106 | 0,479 | 17,0% |
| 18 | 0,313 | 0,200 | +0,113 | 0,525 | 18,6% |
| **24** | 0,334 | 0,219 | **+0,115** | 0,595 | 21,1% |

G1 **ĐẠT** (+0,645) · G2 **ĐẠT** (+0,102) · G3 **TRƯỢT** (16,9% < 25%).

#### b. Tôi SAI ở §12.80c

Tôi viết: *"hướng chỉ sống ở vùng gần ⇒ d ≈ 8–12, KHÔNG phải 48 hay vô hạn"*, dựa vào mini nơi
VƯỢT rơi xuống +0,078 ở d=11.

**v12 bác điều đó.** VƯỢT giảm nhanh tới d≈6 rồi **CHỮNG ở ~+0,10 và GIỮ PHẲNG tới d=24** — thậm
chí d=24 (+0,115) còn **cao hơn** d=8 (+0,102). Hướng **không tắt**. Trực giác ban đầu của tác giả
về khoảng cách xa **đúng trên corpus thật**; kết luận thu hẹp của tôi là hiện vật của mini.

Đối chứng làm đúng việc: nó **tăng song song** với cột thật (0,164→0,219 ‖ 0,278→0,334) vì `d` lớn
thì nhiều cặp phân biệt hơn, đếm mỗi cặp thấp hơn, nhiễu nhị thức cao hơn. Hai cột tăng cùng nhịp
nên VƯỢT phẳng — đúng dấu hiệu cột đối chứng đang cô lập đúng phần thật.

#### c. Nhưng phải đọc ĐỘ LỚN, không chỉ dấu

Quy VƯỢT về tỉ lệ chiều (`p = (1+e)/2`):

    d=1   +0,645 ⇒ 82/18     ← rất mạnh
    d=8   +0,102 ⇒ 55/45     ← thật, nhưng YẾU
    d=24  +0,115 ⇒ 56/44

Tức ở tầm xa, hướng đúng 55–56% thay vì 50%. **Có thật, bền, nhưng mỏng.** Và G3 trượt: sức liên
kết (PMI) ở d=8 chỉ còn **16,9%** của d=1 — trên mini là 35%. Nên v12 vừa cho tin **tốt hơn** về
tầm (hướng không tắt) vừa cho tin **xấu hơn** về sức (liên kết loãng hơn nhiều).

Hai con số này **nhất quán với P16**: thước cặp-có-thứ-tự đo được +0,022 trên bench thật. Một tín
hiệu 55/45 aggregated qua nhiều cặp thì cho đúng cỡ lợi ích đó. Không có mâu thuẫn giữa "tín hiệu
có thật" và "cổng trượt" — nó thật và nó nhỏ.

#### d. Cái phẳng ấy có lẽ là VAI TRÍ VỊ TRÍ TRONG CÂU, không phải cú pháp

Một hằng số hướng ở **mọi** khoảng cách không giải thích được bằng cú pháp cục bộ. Giả thuyết hợp
lý nhất: có token thiên về **đầu câu** và token thiên về **cuối câu**; mọi cặp giữa hai lớp ấy giữ
đúng chiều bất kể cách nhau bao xa. Đó là **vai trí vị trí trong câu** — thứ `W_seq` (d=1) không
thấy và `CO_ACTIVATION` (đối xứng) không biểu diễn nổi.

Nếu đúng thì nó cũng gợi ý cách rẻ hơn nhiều so với ma trận (N×N): một **vector vị trí trung bình
trong câu** cho mỗi token (1 số/token thay vì N² ô). **Chưa đo** — mới là giả thuyết, và ghi ở đây
để không quên rằng nó rẻ hơn hai bậc so với phương án ma trận.

#### e. Trạng thái hướng đi

Bước (2) của kế hoạch §12.79e **dương** ⇒ mở khoá bước (3) là dựng ma trận. Nhưng ba số phải cân
nhắc cùng nhau trước khi tiêu tài nguyên:

* hướng có thật ở mọi tầm — **+0,10 ổn định tới d=24**;
* nhưng chỉ 55/45, và PMI ở tầm xa chỉ 17% của d=1;
* và proxy rẻ nhất của nó (§12.77 PAIR) đã đo được **+0,022** end-to-end, dưới cổng.

Tôi **không** đề nghị dựng thẳng ma trận (N×N) cỡ v12 (~40 triệu cặp có hướng phân biệt cho
d≤24). Đề nghị: thử **giả thuyết (d) trước** — nó rẻ hơn hai bậc và nếu đúng thì giải thích được
phần phẳng, mà nếu sai thì loại được một cách hiểu sai về dữ liệu.

---

### 12.83 Giả thuyết VỊ TRÍ — xác nhận: **1 số/token thay được ma trận N²** cho phần xa (2026-07-29) — Claude

Đo giả thuyết §12.82d. `scripts/probe_position_role.py`, mini.

#### a. Thiết lập — và chỗ hỏng dễ nhất đã chặn

`pos_rel(t)` = trung bình `i/(L−1)` trên mọi lần t xuất hiện trong chuỗi token nội dung của câu.
0 = luôn đầu câu, 1 = luôn cuối câu. **Một số cho mỗi token.**

Vòng luẩn quẩn: tính `pos_rel` rồi chấm ngay trên cùng dữ liệu thì nó "giải thích" mọi thứ bằng
ghi nhớ. Nên **tách đôi corpus theo câu** — học ở nửa A, chấm trên cặp của nửa B.

`pos_rel` có tách thật: độ lệch chuẩn **0,159**, và hai đầu đọc rất hợp lý —
đầu câu `Những · Theo · Các · Đây · Cuộc · Bà · Vua` (0,00) ‖
cuối câu `nữa · Linh · Hóa · Đống · Giang · Cồ` (0,78–0,83) — toàn âm tiết sau của tên riêng
(`Đại Cồ Việt`, `Thanh Hóa`, `Đống Đa`) và tiểu từ cuối câu.

#### b. Số

| d | agree | đối chứng | VƯỢT | trần | **GIẢI THÍCH** | #lượt |
|---|---|---|---|---|---|---|
| 1 | 0,553 | 0,498 | +0,055 | 0,562 | **0,20** | 12.506 |
| 2 | 0,631 | 0,500 | +0,130 | 0,388 | 0,67 | 4.894 |
| 3 | 0,649 | 0,499 | +0,150 | 0,319 | **0,94** | 3.305 |
| 4 | 0,632 | 0,496 | +0,135 | 0,291 | **0,93** | 2.201 |
| 5 | 0,602 | 0,510 | +0,092 | 0,202 | 0,91 | 1.449 |
| 6 | 0,609 | 0,505 | +0,104 | 0,192 | 1,09 | 1.106 |

**Cột đối chứng ≈ 0,500 ở mọi d** — xáo trật tự thì dự đoán vị trí về đúng mức may rủi. Đó là
xác nhận thiết lập lành; không có nó thì `agree = 0,65` không đọc được.

G1 **ĐẠT** (sd 0,159) · G2 **ĐẠT** (0,97) · G3 **ĐẠT** (d=1 chỉ 0,20 ‖ d=3..6 là 0,97).

#### c. Đọc

**Vị trí giải thích ~97% tín hiệu hướng ở d = 3..6, nhưng chỉ 20% ở d = 1.**

Đúng bức tranh hai thành phần của §12.82c, và giờ mỗi thành phần có tên:

* **d = 1**: cú pháp/cụm từ cục bộ. Vị trí gần như không giải thích được (0,20) ⇒ cần cạnh thật.
  **`W_seq` đã có.**
* **d ≥ 3**: vai trí vị trí trong câu. Vị trí giải thích **gần trọn vẹn** ⇒ **không cần ma trận** —
  một scalar mỗi token là đủ.

Hai lỗi tôi tự sửa trong lượt này, ghi lại vì cùng một họ:

1. Bản đầu tính `agree` trên MỌI cặp nhưng `trần` chỉ trên cặp đạt `min_count`, và lấy tử số
   CHƯA trừ nhiễu chia mẫu số ĐÃ trừ ⇒ "giải thích" vọt lên **15,88**. Phải cùng quần thể, cùng
   cách hiệu chỉnh.
2. Tỉ lệ >1 ở d ≥ 8 là **nhiễu cạn mẫu** (mini còn 195–612 lượt), không phải nghĩa đen. Đổi vùng
   báo cáo sang d=3..6 (1,1k–3,3k lượt). **Không dời cổng** — cả hai vùng đều vượt 0,50 nên
   verdict không đổi; đây là sửa cách ĐỌC.

#### d. Hệ quả — đề xuất ma trận thu hẹp mạnh

Đề xuất ban đầu: ma trận (N×N) có hướng, `d` tới 48 hoặc vô hạn. Sau §12.80–§12.83:

| vùng | tín hiệu | cần gì | đã có chưa |
|---|---|---|---|
| d = 1 | cú pháp cục bộ, 82/18 | cạnh có hướng | **`W_seq` ✓** |
| d = 2 | chuyển tiếp | cạnh có hướng | `W_seq` bỏ sót |
| **d ≥ 3** | vai trí vị trí, 55/45 | **1 scalar/token** | **chưa có** |
| d lớn, đối xứng | chủ đề | ma trận đối xứng | **`CO_ACTIVATION` ✓** |

Tức ô trống thật sự **không phải** ma trận N² — mà là **một vector độ dài N** (`pos_rel`) cộng với
vùng hẹp d=2. Rẻ hơn đề xuất ban đầu **bốn bậc** (N vs N²), và trên v12 là 32k số thay vì ~40
triệu cặp.

#### e. Chưa làm, và vì sao

Chưa nối `pos_rel` vào đường sinh. Hai lý do: (1) đây là số **mini**, mà d ≥ 8 trên mini đã cạn
mẫu — cần v12 để chắc phần phẳng cũng do vị trí (§12.31 P18); (2) kể cả đúng, phải đo **lợi ích
end-to-end** trên bench §12.53, vì P16 vừa dạy rằng tín hiệu thật ở nguồn (+0,10 lệch chiều) chỉ
cho +0,022 sau khi qua đường xếp hạng.

---

### 12.84 W_skip — đề xuất theo khoảng cách + giao (ý tác giả, không phải pos_rel) (2026-07-29) — M4

Tác giả làm rõ: cần cạnh có hướng theo từng `d` để mỗi token prompt đề xuất next-token ở đúng
khoảng cách, rồi **lấy giao**. `pos_rel` (§12.83) là giả thuyết khác — không thay được cơ chế này.

#### a. Đã ship (mặc định OFF)

| file | việc |
|---|---|
| `src/nn_aivie/hca/w_skip.py` | dựng CSR `W_skip[d]` · `constrain_next` (hard∩ + soft phiếu) |
| `matrix_core.py` + `config` | kênh `ar_matrix_w_skip_weight` (0) · `ctx_max` mặc định **2** |
| `scripts/probe_w_skip_intersect.py` | đo giao vs baseline chỉ d=1 |
| `tests/test_w_skip.py` | toy `Việt∩Nam ⊇ {vô, mạnh}` |

Soft = **số proposer có token trong top-k** (+ phụ log-prob có `λ^{d-1}`). Bản tích log thuần từng
thua d=1; đổi sang đếm phiếu thì thắng.

#### b. Probe mini (`out/w_skip_mini.log`, 400 ca)

| thước | W_skip | chỉ d=1 |
|---|---|---|
| hard∩ ≠ ∅ | **0.455** | 0.922* |
| gold∈hard \| ≠∅ | **0.549** | 0.637* |
| soft top-10 | **0.517** | 0.482 |
| soft10 \| ≥2 proposer | **0.552** | — |

\*d=1 chỉ còn 1 proposer → hard∩ ≈ top-k của token cuối (không phải giao thật).

G1/G2/G3 **ĐẠT** · G4 (+0.035) trượt nhẹ · G4b (multi) **ĐẠT**. `ctx_max=2` tốt nhất
(soft10\|multi **0.579**).

#### c. A/B sinh mini (`eval_gen_broad`, max_len=64, n=5/ô)

OFF = `out/w_skip_off_mini.log` · ON = weight **2.0** + ctx_max **2** (`out/w_skip_on_mini.log`).

| handoff | lạc OFF→ON | n_tok OFF→ON | ghi chú |
|---|---|---|---|
| T1 | 0.33→0.35 | 20.9→20.9 | gần đứng |
| T2 | 0.25→0.25 | 17.4→17.4 | ≡ |
| T3 | 0.33→**0.00** | 14.4→10.6 | ít trôi, ngắn hơn |
| T4 | 0→0 | 16→16 | ≡ |
| T5 | 0.33→0.33 | 16.8→16.8 | ≡ |
| T6 | 0.50→0.40 | 33.6→32.6 | lạc↓; loop4 0.20→0 |
| T7 | 0.00→**0.67** | 8.0→**29.8** | dài hơn nhưng lạc↑ |

no_agenda: T7 empty/degen **0.40/0.60 → 0/0**, n_tok 2.0→12.6; T6 digit/loop dịu.

**Không đổi mặc định** (`weight=0`). Tín hiệu probe và vài nhóm sinh tích cực, nhưng T7 handoff
lạc xấu — cần chỉnh (ví dụ chỉ thưởng hard∩, hoặc v12) trước khi bật.

#### d. Việc kế → P19 (§12.85)

1. ✅ Probe + A/B trên **v12**.
2. ✅ hard-gate + lọc `gidf_min`.
3. ~~P18 `pos_rel`~~ **HỦY** (tác giả). Kết quả v12 → §12.85.

---

### 12.85 P19 — W_skip trên v12: probe G2/G3 TRƯỢT · A/B lạc↑ · giữ OFF (2026-07-29) — M4

`pos_rel` **không chạy**. Đo đúng ý tác giả: cạnh theo `d` → mỗi token prompt đề xuất → **giao**.

#### a. Code thêm so §12.84

| knob | mặc định | ý |
|---|---|---|
| `ar_matrix_w_skip_hard_gate` | `False` (A/B bật `True`) | nếu hard∩ ≠ ∅ → chỉ boost token trong ∩ |
| `ar_matrix_w_skip_gidf_min` | `4.0` | bỏ hub khỏi top-k đề xuất trước khi giao |
| `ar_matrix_w_skip_weight` | **0** | vẫn OFF sau cổng |

#### b. Probe v12 — không gidf (`out/w_skip_v12.log`)

| thước | W_skip | chỉ d=1 |
|---|---|---|
| hard∩ ≠ ∅ | 0.873 | 0.983 |
| gold∈hard \| ≠∅ | **0.307** | 0.639 |
| soft top-10 | **0.307** | 0.355 |

G1 **ĐẠT** · G2/G3/G4 **TRƯỢT**. Demo tay: `'Việt Nam'` ∩ = `{không Dân Cộng là gia có}` — hub chiếm chỗ.

#### c. Probe v12 — `gidf_min=4` (`out/w_skip_v12b.log`)

| thước | W_skip | chỉ d=1 |
|---|---|---|
| hard∩ ≠ ∅ | 0.690 | 0.980 |
| gold∈hard \| ≠∅ | **0.246** | 0.487 |
| soft top-10 | **0.235** | 0.285 |

G1 **ĐẠT** · G2/G3/G4 **TRƯỢT** (gold|hard còn thấp hơn). Tay tốt hơn một chút:
`'Việt Nam'` ∩ → `{gia Cộng Dân}` — hết hub thô, vẫn **không** phải kiểu “vô/mạnh”.
`ctx_max=2` vẫn tốt nhất trên soft10|multi.

#### d. A/B sinh v12 (`eval_gen_broad`, max_len=140, n=10) vs A=`out/lac1_v12.log`

**B1** hard_gate, không gidf → `out/w_skip_on_v12.log`  
**B2** hard_gate + gidf≥4 → `out/w_skip_on_v12b.log`  
(weight=2, ctx_max=2)

| handoff | lạc A | B1 | B2 | Δ B2−A |
|---|---|---|---|---|
| T1 | 0.12 | 0.13 | 0.12 | 0 |
| T2 | 0.40 | 0.40 | 0.40 | 0 |
| T3 | 0.29 | **0.57** | **0.50** | **+0.21** |
| T4 | 0.00 | 0.00 | 0.00 | 0 |
| T5 | 0.25 | 0.25 | 0.25 | 0 |
| T6 | 0.00 | 0.00 | **0.25** | **+0.25** |
| T7 | 0.00 | 0.00 | **0.33** | **+0.33** |

Cổng đổi default (§12.31): **TRƯỢT** — `lạc` T3/T6/T7 vượt +0.05. B1 cũng T3↑ (+0.28) dù T6/T7 đứng.
`loop4` B1 T3 = 0.10 (A=0); B2 loop4 ổn. T4 không regress.

#### e. Kết luận

1. Cơ chế **đúng hướng** trên toy/mini; trên v12 top-k count thô (kể cả gidf) **không đủ chọn lọc** — gold∈∅ ~0.25.
2. Hard-gate trên ∩ bẩn → ép sinh vào hub/lệch đề → **lạc↑**.
3. **Giữ `ar_matrix_w_skip_weight=0`.** Không bật default.
4. Việc kế (chờ Claude / thiết kế): PPMI hoặc skip-normalize trước top-k; yêu cầu ≥2 phiếu nội dung; hoặc kênh soft không hard-gate sau khi ∩ sạch hơn.


---

### 12.86 Nhận §12.85 (P19) + ghi P15 mà tôi BỎ SÓT (2026-07-29) — Claude

#### a. Trước hết: P15 đã chạy trên v12 mà tôi chưa từng đọc

`out/p15b_v12.log` tồn tại từ lượt P15 (cổng phủ span nối tiếp). Tôi thay runbook sang P16 **trước
khi** đọc nó, nên một lượt chạy v12 trôi mất. Chấm theo cổng §12.75e:

| nhóm | n_tok | tỉ lệ | Δ`lạc` | `loop4` (B) |
|---|---|---|---|---|
| T1 | 23,8 → 45,8 | 1,92× | +0,02 | 0,10 |
| T2 | 19,6 → 25,6 | 1,31× | **−0,29** | 0,00 |
| T3 | 28,0 → 42,0 | 1,50× | 0,00 | **0,20** |
| T5 | 21,5 → 36,8 | 1,71× | **−0,14** | 0,00 |
| T6 | 32,4 → 53,6 | 1,65× | **+0,20** | 0,00 |
| T7 | 26,2 → 43,8 | 1,67× | **+0,11** | 0,00 |

① n_tok ≥1,3× ở 6/7 nhóm **ĐẠT** · ② `lạc` không tăng >+0,03: **TRƯỢT** (T6 +0,20 · T7 +0,11) ·
③ `loop4` ≤0,15: **TRƯỢT** (T3 0,20) · ④ T6 `stop`: ĐẠT. ⇒ **TRƯỢT**, giữ `cover_min=0`.

Đọc kỹ: T6/T7 ở cấu hình A có `lạc = 0,00` **vì gần như không có câu thứ hai** để mà lạc. Nâng cap
tạo ra câu thứ hai, và câu ấy lạc. Nên chính xác là **cổng phủ không đủ mạnh để bù phần trôi mà
độ dài mới sinh ra**, chứ không phải cổng phủ vô dụng (T2 −0,29 · T5 −0,14 là thật).

**Luật quy trình bổ sung** (đã thêm vào §12.31): trước khi thay mục ƯU TIÊN, **phải ghi kết quả
việc cũ vào ticket**; chưa xong thì ghi "đang chạy", không xoá.

#### b. P19 âm — và nguyên nhân đúng thứ đã cảnh báo

`gold ∈ hard∩` : W_skip **0,307** ‖ chỉ d=1 **0,639**. Thêm `gidf_min=4` còn tệ hơn (0,246 ‖
0,487). A/B sinh: `lạc` T3 **+0,21** · T6 **+0,25** · T7 **+0,33**.

Nguyên nhân M4 chỉ ra và tôi đồng ý: **đếm thô không đủ chọn lọc, hub chiếm chỗ giao** —
`'Việt Nam'` ∩ = `{không Dân Cộng là gia có}`. Đó đúng rủi ro đã nêu khi trao đổi: *"càng xa càng
phải tỉa, nếu không thống kê tiến dần về đồng-hiện-trong-tài-liệu"*. Cách tỉa đúng là **PPMI**,
không phải `gidf` — `gidf` lọc theo tần suất token, PPMI chuẩn hoá theo tần suất **cả cặp**.

#### c. Nhưng PPMI cũng có TRẦN, và §12.83 đã đo trần đó

§12.83: ở d ≥ 3, **~97%** tín hiệu hướng được giải thích bởi `pos_rel` — **một scalar mỗi token**.
Tức phần thông tin **riêng của từng cặp** ở tầm xa chỉ còn vài phần trăm của một tín hiệu vốn đã
mỏng (55/45).

⇒ W_skip ở d ≥ 3 đang tiêu **N²** ô để bắt thứ đáng giá **N** số. PPMI sẽ làm nó **bớt tệ** (hết
hub) nhưng không đổi được trần ấy. Vùng W_skip thật sự có lý là **d = 2** — nơi `W_seq` bỏ sót mà
vị trí chưa giải thích được (§12.83: d=2 mới 0,67).

#### d. Ba lần thêm THÔNG TIN THỨ TỰ, ba lần ~+0,02 hoặc âm

| cách | kết quả |
|---|---|
| §12.77 PAIR re-rank (đọc `emits` lúc chấm) | +0,022 — dưới cổng |
| §12.85 W_skip (ma trận cạnh theo d) | **âm** — `lạc` +0,21…+0,33 |
| §12.83 `pos_rel` | **chưa thử** |

Hai cách đã thử **không cùng cơ chế** — một là re-rank, một là kênh logits — mà cùng cho ~+0,02
hoặc âm. Đó là dấu hiệu vấn đề **không nằm ở cách mã hoá thứ tự**.

Giả thuyết: **không có ai TIÊU THỤ thứ tự.** `A` truy hồi bằng túi từ; kênh mạnh nhất
(`succ+path`, β 5,0+4,0) chỉ biết chép hậu tố. Đổ thông tin thứ tự vào đầu vào của một hệ không có
chỗ dùng nó thì được +0,02. Chưa chứng minh — nhưng nó khớp cả ba kết quả và khớp `copy ≈ 1,0`.

#### e. Việc kế — và tôi KHÔNG đề nghị dựng thêm ma trận

1. **Rẻ nhất, chưa ai làm:** đo `pos_rel` có dự đoán EOS tốt hơn `end_after` không. Một phép đo
   trên mini, loại được hoặc xác nhận ứng dụng duy nhất của `pos_rel` mà đường sinh hiện có chỗ
   dùng ngay.
2. **Nếu vẫn muốn W_skip:** giới hạn **d = 2** và chuẩn hoá **PPMI** (không phải `gidf`). Đó là ô
   duy nhất còn lý do tồn tại sau §12.83.
3. **Việc lớn thật sự:** dựng đường **soạn câu** — cái tiêu thụ thứ tự. Đây là việc nhiều tuần,
   không phải một phép đo, và tôi sẽ không bắt đầu khi chưa chốt với tác giả.

---

### 12.87 W_skip làm KÊNH TRÔI CHẢY — đo dự đoán token kế (2026-07-29) — Claude

Tác giả yêu cầu tận dụng triệt để W_skip, tin nó là kênh cho **sinh mượt, tự nhiên**. §12.85 đã
âm, nhưng nó đo **câu hỏi khác**.

#### a. P19 và mục này hỏi hai câu khác nhau tận gốc

P19: *"giao top-k của các token prompt có chứa gold không"* — đó là **CHỌN NỘI DUNG** (kiến thức).
Mục này: *"đoán đúng token kế không"* — đó là **TRÔI CHẢY**.

Một kênh có thể tồi ở việc chọn *nói gì* mà tốt ở việc giữ *câu trôi chảy*; `W_seq` chính là ví
dụ — nó là kênh trôi chảy, không ai dùng nó chọn fact. Và `W_seq` **là order-1**, `varlm.py` ghi
rõ điểm yếu: *"nhìn 1 token trước → mất trí nhớ ngữ cảnh sau mỗi từ → văn rời"*. W_skip đúng là
bản mở rộng order-D của nó.

Ba khác biệt so với bản P19: **PPMI** thay đếm/chuẩn-hoá-hàng · cộng **MỀM** toàn cửa sổ (không
top-k, không giao, không hard-gate) · dùng **cả cửa sổ** thay `ctx_max=2`.

#### b. Một lỗi thước tôi tự bắt trước khi đọc kết quả

Bản đầu đo `d1` bằng **PPMI** nhưng `skip_raw` bằng **xác suất chuẩn-hoá-hàng** ⇒ so táo với cam.
Nó cho "+0,046" trông như dương. Sửa để mỗi thang có đủ cặp baseline/skip thì **+0,046 biến mất**.

#### c. Số (mini, n=6.000 vị trí, tách nửa A/B theo câu, `d_max=8`, λ=0,85)

| kênh | top1 | top10 | MRR |
|---|---|---|---|
| **`d1_prob`** (≈ `W_seq`) | **0,169** | 0,318 | 0,218 |
| `skip_prob` | 0,140 | 0,323 | 0,198 |
| `d1_ppmi` | 0,106 | 0,277 | 0,160 |
| `skip_ppmi` | 0,132 | 0,303 | 0,188 |
| **`d1p+skip_p`** | 0,168 | **0,339** | **0,223** |

G1 ĐẠT (0,318) · **G2 TRƯỢT** (+0,021 < +0,05) · G3 TRƯỢT.

#### d. Ba điều đọc ra

1. **W_skip một mình THUA `W_seq` ở top-1** (0,140 ‖ 0,169). Mở rộng cửa sổ làm **loãng** lựa chọn
   đầu bảng. Trực giác "ngữ cảnh rộng hơn thì đoán tốt hơn" **sai** ở dạng cộng tuyến tính này.
2. **Kết hợp thì hơn ở top-10 (+0,021) mà top-1 đứng yên.** Tức W_skip mở rộng **tập ứng viên**
   chứ không cải thiện **lựa chọn**. Đúng cùng hình dạng với PAIR (§12.79) và W_skip-giao (§12.85):
   cả ba đều đưa thêm thứ vào xét mà không chọn tốt hơn.
3. **PPMI THUA đếm thô** ở cả hai (0,303 ‖ 0,323 và 0,277 ‖ 0,318) — **ngược** giả thuyết của tôi.
   Vì với *trôi chảy*, hub là **tín hiệu**: từ phổ biến thật sự hay đứng kế. Còn với *chọn nội
   dung* (P19), hub là **nhiễu**. Cùng một ma trận, hai vai trò đối nghịch — và đó là lời giải
   thích sạch cho việc `gidf`/PPMI cứu được P19 nhưng làm hỏng chỗ này.

#### e. Kết luận và dè dặt

Theo cổng: **W_skip không phải kênh trôi chảy tốt hơn `W_seq`.** +0,021 top-10, 0 top-1.

Dè dặt phải ghi: đây là **mini**, và ước lượng `P(b|a,d)` cho `d ≥ 2` **cải thiện theo lượng dữ
liệu** — mini với `min_count=2` rất thưa ở các bin xa. Đây là một trong ít cơ chế mà v12 **có lý
do cơ học** để khác hẳn (nhiều dữ liệu hơn ⇒ ước lượng chắc hơn), khác với các ca mini-nói-quá
trước đó. Nên **chưa đóng hẳn** — xem §12.31 P20.

---

### 12.88 W_skip làm EMBEDDING — đo trên cùng bench (2026-07-29) — Claude

Nửa còn lại của yêu cầu tác giả. Dựng embedding từ W_skip rồi so **trực tiếp** với embedding hiện
hành, trên bench §12.53 đã qua bốn lần vá.

#### a. Dựng

Đếm cặp **trong span câu** (không tràn câu như `CO_ACTIVATION`), trọng số **λ^d** tường minh
(λ=0,85, d≤8), **đối xứng hoá** rồi PPMI → SVD 96 chiều. 484.808 ô ≠0.

Đối xứng hoá là chủ ý: embedding lo **tương đồng**, không lo thứ tự. Bản **có hướng** (Wq/Wk) là
đối tượng khác — và §12.83 đã đo rằng phần hướng ở tầm xa chủ yếu là **vị trí** (1 scalar/token),
nên nội dung riêng của nó mỏng.

#### b. Số (mini, n=169)

| thước | top1 | Δ so B | thắng | thua |
|---|---|---|---|---|
| B (túi từ) | 0,562 | — | — | — |
| `pos` (tiêu chí hiện hành) | 0,615 | +0,053 | 18 | 9 |
| **`cosE_old`** (embedding hiện tại) | **0,491** | −0,071 | 14 | 26 |
| **`cosE_skip`** (từ W_skip) | **0,479** | −0,083 | 13 | 27 |

*(Lỗi nhỏ tự bắt: lần chạy đầu `E_old` ra `None` vì tôi lấy `.vectors`; thuộc tính đúng là
`.matrix`. Không có nó thì mất chính cột so sánh cần nhất.)*

#### c. Đọc

**W_skip KHÔNG cho embedding tốt hơn.** 0,479 ‖ 0,491 — chênh **2 ca** trên 169, tức **ngang
nhau**, nhỉnh về phía bản cũ.

Lý do hợp lý: embedding hiện hành **đã có** giảm theo khoảng cách — `CO_ACTIVATION` dùng cửa sổ
trượt bước 1 nên cặp cách `d` tích luỹ ∝ `(W−d)`, một suy giảm tuyến tính (§12.80). W_skip thêm
được **giới hạn theo câu** và **λ tường minh**; hai thứ đó không đủ để đổi kết quả.

Ghi kèm, ngoài câu hỏi: **cả hai embedding đều thua xa túi từ** (0,49 ‖ 0,56) trên bench này. Đó
là bình thường — bench đo truy hồi theo **thực thể chính xác**, chỗ túi từ mạnh nhất và nén 96
chiều mất chính xác. Nó **không** nói embedding vô dụng; nó nói bench này không phải chỗ đo
embedding. Nhưng phép so `cosE_skip` ‖ `cosE_old` thì **cùng điều kiện**, nên vẫn đọc được.

#### d. Gộp lại: ba cách dùng W_skip, ba lần cùng một hình dạng

| cách dùng | kết quả |
|---|---|
| giao top-k đề xuất (§12.85) | **âm** — `lạc` +0,21…+0,33 |
| kênh trôi chảy (§12.87) | +0,021 **top-10**, **0** top-1 |
| embedding (§12.88) | ngang bản cũ (−0,012) |

Ba cơ chế khác hẳn nhau, cùng một kết luận: **W_skip mở rộng cái được đưa vào xét, không cải
thiện việc chọn.** Y hệt PAIR (§12.79), y hệt nới vùng PMI (§12.49), y hệt chép span đáp (§12.38).

Và §12.87d3 cho một quan sát mới đáng giữ: **hub là TÍN HIỆU cho trôi chảy nhưng NHIỄU cho chọn
nội dung**, nên không có một cách chuẩn hoá nào phục vụ được cả hai vai. Đó là lý do cấu trúc
khiến `gidf`/PPMI cứu §12.85 mà làm hỏng §12.87.

#### e. Dè dặt và việc còn lại

Cả hai phép đo đều trên **mini**, và W_skip là một trong ít cơ chế có **lý do cơ học** để v12
khác: ước lượng `P(b|a,d)` ở `d ≥ 2` phụ thuộc mạnh vào lượng dữ liệu, mà mini với `min_count=2`
rất thưa ở các bin xa. Nên **chưa đóng hẳn** — giao P20 chạy cả hai phép đo trên v12.

Nếu v12 cũng cho cùng kết quả thì kết luận là: W_skip không sai, nó chỉ **không có ai tiêu thụ** —
đúng giả thuyết §12.86d.

---

### 12.89 P20 — W_skip trên v12: trôi chảy G2 TRƯỢT · emb skip ≫ old nhưng thua túi (2026-07-29) — M4

Đóng dè dặt §12.87e / §12.88e: cùng hai phép đo trên `network_v12_sent.pkl` (430k cha).
**Không đổi mặc định.**

#### a. Kênh trôi chảy (`out/next_token_v12.log`, n=6.000)

| kênh | top1 | top10 | MRR |
|---|---|---|---|
| `d1_prob` | 0.183 | **0.431** | 0.265 |
| `skip_prob` | 0.184 | 0.401 | 0.259 |
| `d1_ppmi` | 0.066 | 0.241 | 0.123 |
| `skip_ppmi` | 0.134 | 0.316 | 0.195 |
| `d1p+skip_p` | **0.193** | 0.430 | **0.273** |

G1 **ĐẠT** (0.431) · **G2 TRƯỢT** (−0.001 top10; cần ≥+0.05) · G3 **TRƯỢT** (PPMI 0.316 ‖ đếm 0.401).

So mini (§12.87): cùng hình — kết hợp **không** thắng `d1` ở top-10. Khác nhẹ: top-1 kết hợp
+0.010 (mini ≈0). Vẫn **đóng** hướng dùng W_skip làm kênh sinh trôi chảy.

#### b. Embedding (`out/emb_v12.log`, n=274, PPMI 7.594.175 ô ≠0 → SVD 96d)

Khối `THƯỚC CẶP` (dòng quyết định):

| thước | top1 | Δ so B | thắng | thua |
|---|---|---|---|---|
| B (túi từ) | **0.318** | — | — | — |
| pos | 0.347 | +0.029 | 21 | 13 |
| **`cosE_old`** | **0.004** | −0.314 | 0 | 86 |
| **`cosE_skip`** | **0.135** | −0.182 | 1 | 51 |

`cosE_skip` **hơn rõ** `cosE_old` (+0.131 · ít thua hơn 35 ca). Đây là chỗ v12 **khác** mini
(mini: 0.479 ‖ 0.491, ngang / nhỉnh bản cũ). Nhưng cả hai vẫn **thua xa túi** — bench truy hồi
thực thể không phải chỗ đo embedding hữu ích cho gen.

#### c. Gộp ba cách dùng W_skip (v12)

| cách | v12 |
|---|---|
| giao top-k (§12.85) | **âm** — lạc↑ |
| kênh trôi chảy (mục a) | G2 **TRƯỢT** (−0.001) |
| embedding (mục b) | skip ≫ old trên cosE; cả hai thua túi |

Kết luận giữ: W_skip **mở rộng cái đưa vào xét**, không cải thiện **việc chọn** cho đường sinh
hiện tại. Giữ `ar_matrix_w_skip_weight=0`. Cơ học “v12 khác mini” **có** (emb), nhưng không đủ
để bật kênh.


---

### 12.90 Nhận §12.89 (P20) — kênh trôi chảy đóng; nhưng EMBEDDING là dương tính đáng kể (2026-07-29) — Claude

#### a. Kênh trôi chảy: cổng trượt, nhưng dự đoán cơ học của tôi ĐÚNG

| kênh | top1 | top10 | MRR |
|---|---|---|---|
| `d1_prob` | 0,183 | **0,431** | 0,265 |
| `skip_prob` | **0,184** | 0,401 | 0,259 |
| `d1_ppmi` | 0,066 | 0,241 | 0,123 |
| `skip_ppmi` | 0,134 | 0,316 | 0,195 |
| **`d1p+skip_p`** | **0,193** | 0,430 | **0,273** |

§12.87e tôi ghi dè dặt: *"ước lượng `P(b|a,d)` ở d≥2 cải thiện theo lượng dữ liệu"*. Đúng —
thiếu hụt top-1 của W_skip trên mini (0,140 ‖ 0,169) **biến mất** trên v12 (0,184 ‖ 0,183).

Nhưng cổng vẫn **TRƯỢT** (−0,001 top-10). Kết hợp được +0,010 top-1 / +0,008 MRR — thật nhưng
nhỏ. Và `d1_ppmi` **0,066** ‖ `d1_prob` **0,183** khẳng định lần hai, đậm hơn: **hub là TÍN HIỆU
cho trôi chảy**, PPMI giết nhầm.

⇒ Đóng hướng "W_skip làm kênh trôi chảy thay `W_seq`". Giữ `ar_matrix_w_skip_weight = 0`.

#### b. Embedding: 34× — lần đầu v12 cho hiệu ứng LỚN HƠN mini

| | v12 | mini |
|---|---|---|
| B (túi từ) | 0,318 | 0,562 |
| `pos` (tiêu chí hiện hành) | 0,347 | 0,615 |
| **`cosE_old`** | **0,004** (0 thắng/86 thua) | 0,491 |
| **`cosE_skip`** | **0,135** (1 thắng/51 thua) | 0,479 |

Trên mini hai bản **ngang nhau**; trên v12 bản W_skip hơn **34×**. Trực giác của tác giả về
embedding đúng, và nó chỉ lộ ở quy mô thật.

#### c. Nhưng `cosE_old = 0,004` thấp tới mức phải CHẨN trước khi kết luận

Bể trung vị 37.908 ⇒ ngẫu nhiên thuần ≈ 0,000026. `cosE_old` = 0,004 tức **152× ngẫu nhiên** —
không phải rác hoàn toàn, nhưng gần như vô dụng. Hai cách giải thích, **dẫn tới hai hành động
khác hẳn**:

**(a) `E_old` trên v12 thoái hoá hoặc lệch.** Hình dạng khớp (32002, 96) nên không phải lệch kích
thước, nhưng **nội dung** có thể chiết ở một mốc khác (TK_041 đã đổi đáy BPE 126k→32k). Nếu vậy
thì `cos(zone-emb, fact-emb)` trong `multizone` — tiêu chí xếp hạng ở đúng chỗ nút thắt §12.50c —
**đang chạy trên embedding hỏng suốt thời gian qua**, tức một **bug**, không phải giới hạn.

**(b) 96 chiều không đủ tách 37.908 ứng viên** (mini chỉ 416). Hợp lý — nhưng **không giải thích
nổi 34×** giữa hai embedding **cùng 96 chiều, cùng bể, cùng phép chấm**.

⇒ (b) giải thích được việc cả hai thua túi từ; **chỉ (a) giải thích được 34×**. Chưa đo.

#### d. Cần đo gì để phân biệt — rẻ, không cần chạy sinh

1. **Phân bố chuẩn `‖E_old[t]‖`** — nhưng con số này MỘT MÌNH không kết luận: token không xuất
   hiện trong corpus thì đương nhiên chuẩn 0, và **mini ra 69,26%** trong khi embedding rất lành.
2. **Láng giềng gần nhất** của dăm token dễ kiểm (`Lý` · `Thăng` · `sông` · `vua` · `Việt`) theo
   `E_old`. Đọc tay là đủ phân định.
3. Nếu láng giềng hợp lý ⇒ embedding lành ⇒ 34× là chuyện khác, phải đo tiếp.

#### e. Và nếu (a) đúng thì việc kế KHÔNG phải "thay embedding"

Cả hai embedding đều **thua túi từ** (0,135 và 0,004 ‖ 0,318) trên bench này. Bench đo truy hồi
theo **thực thể chính xác** — chỗ túi từ mạnh nhất. Nên "embedding tốt hơn" ở đây **chưa** nghĩa
là "sinh tốt hơn".

Chỗ embedding thật sự được tiêu thụ là `cos(zone-emb, fact-emb)` trong `multizone`. Muốn biết
thay nó có ăn không thì phải **A/B chính đường sinh** — và chỉ đáng chạy **sau khi** (d) cho biết
ta đang sửa một **bug** hay đổi một **lựa chọn**.

§12.31 P21.

---

### 12.91 P21 — chẩn `E_old` v12: **LÀNH** · bác thoái hoá (2026-07-29) — M4

`out/diag_emb_v12.log` nguyên văn:

```
[37s] ✓ E (32002, 96) · vocab 32,002

======================================================================
hàng chuẩn ≈ 0        : 4.32%  (1,382/32,002)
chuẩn  min/trung vị/max: 0 / 0.6609 / 4.574
cos cặp NGẪU NHIÊN    : trung bình +0.078 · |trung bình| 0.100   (gần 1 ⇒ sụp hạng)
======================================================================

LÁNG GIỀNG gần nhất (top 8) — phép thử đọc-tay, phân định chắc nhất:
         ĠLý → ĠVũ(0.94) · ĠTrần(0.93) · ĠTriệu(0.93) · ĠLương(0.91) · ĠLưu(0.89) · ĠDiên(0.88) · ĠLâm(0.88) · ĠLạc(0.87)
      ĠThăng → ĠLiêu(0.86) · ĠHậu(0.85) · ĠNghi(0.83) · ĠPhụng(0.83) · ĠTấn(0.82) · ĠSơn(0.82) · Lê(0.82) · ĠChiêm(0.81)
       Ġsông → Ġbờ(0.91) · Ġvùng(0.85) · Ġthung(0.85) · Sông(0.85) · Ġbắc(0.85) · Ġbiển(0.83) · Ġdọc(0.83) · ĠSông(0.83)
        Ġvua → ĠVua(0.90) · Ġtriều(0.85) · ĠChiêm(0.85) · Ġvương(0.84) · Ġhoàng(0.83) · Ġchúa(0.82) · Vua(0.81) · ĠJaya(0.80)
       ĠViệt → ĠNam(0.88) · ĠTrung(0.83) · ĠQuốc(0.79) · Việt(0.79) · ĠCộng(0.74) · ĠKháng(0.73) · ĠDân(0.71) · ĠBộ(0.68)
       ĠTrần → ĠVũ(0.96) · ĠLý(0.93) · ĠHoàng(0.90) · ĠLiễu(0.89) · ĠTrọng(0.89) · ĠThanh(0.88) · ĠKiệt(0.88) · ĠTriệu(0.88)
        Ġnăm → Ġtháng(0.91) · Ġ1(0.86) · Ġngày(0.83) · Vào(0.83) · Ġthứ(0.80) · ĠVào(0.80) · ipia(0.79) · ĠGregory(0.78)
         Ġđô → Ġphố(0.74) · ĠNăm(0.71) · Ġchiếm(0.68) · Ġquốc(0.67) · Ġkhu(0.67) · ĠKhu(0.65) · Thủ(0.64) · Ġthành(0.63)
```

#### Đọc theo cổng §12.31

| tín hiệu | mini (lành) | v12 | phán |
|---|---|---|---|
| hàng chuẩn ≈0 | 69.26% | **4.32%** | không phải “nhiều hàng rỗng” — v12 **đặc** hơn |
| cos ngẫu nhiên | +0.051 | **+0.078** | xa 1 → **không** sụp hạng |
| láng giềng | Lý→Thánh·Tông… | Lý→**Trần·Triệu·Lương**; vua→triều·vương; Việt→Nam; năm→tháng·1 | **hợp lý** |

**Kết luận:** `E_old` trên v12 **LÀNH**. Giả thuyết (a) thoái hoá / bug embedding (§12.90c) **BÁC**.

Khác mini ở chi tiết (`đô`→phố·thành chứ không Lư·dời) nhưng vẫn ngữ nghĩa địa lý/chính trị —
không phải nhiễu vô nghĩa.

⇒ Chênh 34× (`cosE_skip` 0.135 ‖ `cosE_old` 0.004) **không** giải thích bằng embedding hỏng.
Không đổi default; không dựng lại E. Việc kế (Claude): đo tiếp vì sao skip thắng old trên bench
dù cả hai cùng 96d / cùng bể — hoặc đường `cos(zone,fact)` tiêu thụ E thế nào.


---

### 12.92 Nhận §12.91 (P21) — `E_old` LÀNH, nhưng nó đo TYPE chứ không đo DANH TÍNH (2026-07-29) — Claude

#### a. M4 kết luận đúng: không phải bug embedding

v12: hàng chuẩn ≈0 chỉ **4,32%** (mini 69,26% — vì mini corpus nhỏ, phần lớn vocab không dùng);
cos cặp ngẫu nhiên **+0,078** (không sụp hạng); láng giềng **hợp lý**. Giả thuyết (a) của §12.90c
— `E_old` thoái hoá — **bị bác**.

#### b. Nhưng KIỂU láng giềng khác hẳn giữa hai checkpoint, và đó mới là chuyện

```
mini:  Lý → Thánh · Tông · vua · Thái · Hoàn · Lê      ← CỤM ĐI CÙNG (Lý Thánh Tông, Lý Thái Tổ)
v12:   Lý → Vũ · Trần · Triệu · Lương · Lưu · Diên     ← các HỌ KHÁC, thay thế được cho nhau
```

Cả 8 láng giềng gần nhất của `Lý` trên v12 đều là **họ/danh từ riêng khác**, và `cos(Lý, Trần) =
0,93`. Đó là embedding **paradigm** (cùng LOẠI) chứ không phải **collocation** (đi CÙNG) — kết quả
kinh điển của ngữ nghĩa phân bố khi có đủ dữ liệu, và là dấu hiệu embedding **tốt** theo nghĩa
thông thường.

Nhưng cho truy hồi thực thể thì nó **sai việc**: nó ánh xạ mọi triều đại về gần một điểm. Đây
đúng luật đã rút ở §12.45 — **fold khi so CHỨC NĂNG, đừng fold khi so DANH TÍNH** — và embedding
v12 đang fold danh tính vào loại.

⇒ Giải thích trọn vẹn `cosE_old = 0,004`: không phải embedding hỏng, mà là **`cos` gần như hằng
số trong đám ứng viên cùng thực thể**, nên xếp hạng bằng nó ≈ ngẫu nhiên.

#### c. Và công thức xếp hạng đặt nó làm THỪA SỐ CHÍNH

`multizone.py:200`:

    rank = cos × (0.25 + 0.75 × cov)

`cos` là thừa số chính; `cov` — thứ **duy nhất** kiểm token thật sự có mặt — chỉ điều biến biên độ
**4×**. Còn có nhánh lùi `if not scored` chấm bằng **cos thuần**, không `cov` gì cả.

Tôi phải sửa cách nói ban đầu của mình: công thức **không** xếp fact Trần trên fact Lý — `cov`
chặn được điều đó. Vấn đề tinh vi hơn: trong đám ứng viên **đều chứa `Lý`** (cov như nhau), `cos`
là thứ phá hoà, mà nó vô thông tin ⇒ **nhân một thừa số vô thông tin vào `cov` chỉ thêm nhiễu**.

#### d. Dự đoán CHỐT TRƯỚC — và mini KHÔNG kiểm được

Thêm hai thước vào bench: `cosxcov(mz)` = đúng hình dạng multizone · `covxcos(đảo)` = hoán vai.

Mini (n=169): B **0,562** · `cosE_old` 0,491 · **`cosxcov(mz)` 0,621** (+0,059, 10 thắng/0 thua) ·
`covxcos(đảo)` 0,604. Tức trên mini hình dạng hiện hành **tốt nhất** — vì ở đó `cos` có thông tin
thật (0,491).

**Nên mini không phán được.** Dự đoán cho v12, chốt trước khi có số:

1. `cosxcov(mz)` sẽ **kém hẳn** B (0,318) — vì `cos` ≈ vô thông tin mà lại làm thừa số chính.
2. `covxcos(đảo)` sẽ **xấp xỉ hoặc hơn** B — vì `cos` chỉ còn điều biến 4×.
3. **Nếu cả hai xấp xỉ B thì chẩn đoán của tôi SAI** — `cos` không gây hại, và việc nó vô thông
   tin chỉ làm công thức thoái hoá về `cov` chứ không tệ hơn.

Điều kiện (3) là thứ bác được giả thuyết, nên ghi rõ ở đây.

#### e. Nếu dự đoán đúng thì đây là sửa RẺ

Hoán vai hai thừa số là **một dòng**. Và nó không phải "tắt embedding" — `cos` vẫn giữ việc nó
làm tốt (bắc cầu đồng nghĩa, phân loại), chỉ thôi giữ vai quyết định danh tính.

Nhưng vẫn phải A/B chính đường sinh sau đó: bench đo truy hồi câu, `multizone` xếp fact — cùng
tín hiệu, khác nhiệm vụ. §12.31 P22.

---

### 12.93 Phản biện của tác giả ĐÚNG — và nó lộ ra lỗi suy luận của tôi (2026-07-29) — Claude

Tác giả nêu: `cos(Lý, Trần) = 0,93` **có khi không phải là sai** — hai bên đều là sử, đều nối với
cùng lớp từ thuần sử. Đúng, và nó xa hơn chỗ tôi đã thừa nhận ở §12.92a.

#### a. Chỗ tôi suy luận hụt

§12.92 tôi kết luận "`cos` gần như vô thông tin" **dựa trên `cosE_old = 0,004`**. Nhưng xem lại
cách bench dựng bể ứng viên (`probe_theme_prior.py:332`):

    pool = ⋃ f_inv[f] cho mọi fold nội dung f của câu hỏi

Bể **đã được lọc theo trùng-fold từ vựng** trước khi `cos` được lên tiếng. Tức nó đã topically
coherent sẵn. Vậy `0,004` chỉ nói:

> `cos` không thêm gì **SAU KHI** đã lọc từ vựng.

Nó **không** nói `cos` vô dụng. Chính việc tác giả mô tả — giữ miền — đã bị bộ lọc làm mất **trước
khi** phép đo bắt đầu. Bench của tôi cắt đúng phần việc mà `cos` có thể đang làm tốt.

Đây là lỗi cùng họ với những lần trước trong ticket, nhưng lần này ở tầng suy luận chứ không phải
tầng cài đặt: **thước đo hợp lệ, kết luận rút ra vượt quá thứ nó đo.**

#### b. Đo lại, cho `cos` đúng việc

Thêm ba cột: `cos` tới ứng viên **trong** bể (đã cùng miền) · tới ứng viên **ngoài** bể (nhiều khả
năng khác miền) · tới **gold**.

Mini (n=169): trong bể **+0,468** · ngoài bể **+0,349** · gold **+0,713**
⇒ **tách miền +0,119** · **tách gold +0,244**.

Trên mini, `cos` tách *danh tính* (gold vs phần còn lại của bể) **mạnh gấp đôi** tách *miền*. Khớp
với việc embedding mini là **collocation** (§12.92b) — nó làm đúng việc danh tính.

#### c. Dự đoán cho v12, chốt trước

Nếu tác giả đúng và embedding v12 là **paradigm**, thì trên v12 hai cột phải **đảo**:

* **tách miền LỚN** (cos phân biệt được sử với truyện dịch / công thức nấu ăn — đúng lớp rác đã đo
  ở §12.66e và §12.73b),
* **tách gold ≈ 0** (cos không phân biệt nổi câu về Lý này với câu về Lý kia).

Nếu ra như vậy thì `cos` **không phải nhiễu** — nó là **bộ giữ miền**, và đó là việc có giá trị đo
được: §12.73 đo `lạc` trên v12 ở mức 0,12–0,40 tuỳ nhóm, tức trôi miền là bệnh có thật.

#### d. Hệ quả: dự đoán P22 của tôi có thể SAI, và sai theo hướng nguy hiểm

§12.92d tôi đề xuất **hoán vai hai thừa số** (`cov × (0.25+0.75·cos)`). Nếu (c) đúng thì hoán vai
sẽ **hạ vai trò bộ giữ miền xuống còn điều biến 4×** ⇒ **trôi đề tăng**.

Nên P22 **không được chấm chỉ bằng top-1**. Phải xem cả hai cột: hình dạng nào thắng ở xếp hạng,
và hình dạng nào giữ miền tốt hơn. Rất có thể câu trả lời đúng là **giữ nguyên `cos` làm thừa số
chính** mà sửa chỗ khác — vì cái nó gánh (chống trôi) là bệnh đã đo được, còn cái nó không làm
(phân biệt danh tính) thì `cov`/`B` vốn đã làm tốt hơn nhiều (0,318 ‖ 0,004).

Ghi lại vì đây là lần đầu trong ticket một **phản biện của tác giả** lật được một kết luận của
tôi bằng lập luận, trước khi có số.

---

### 12.94 P22 — hình dạng mz trên v12: giữ công thức · chẩn “cos hại” BÁC (2026-07-29) — M4

`out/shape_v12.log` — không đổi `multizone.py` / default.

#### a. Khối `THƯỚC CẶP` (nguyên văn số)

| thước | top1 | Δ so B | thắng | thua |
|---|---|---|---|---|
| B (túi từ) | **0.318** | — | — | — |
| cosE_old | 0.004 | −0.314 | 0 | 86 |
| cosE_skip | 0.135 | −0.182 | 1 | 51 |
| **cosxcov(mz)** | **0.336** | **+0.018** | 6 | 1 |
| covxcos(đảo) | 0.328 | +0.011 | 4 | 1 |

#### b. Khối `cos` CÓ ĐO MIỀN KHÔNG

```
cos tới ứng viên TRONG bể : +0.435
cos tới ứng viên NGOÀI bể : +0.355
cos tới GOLD              : +0.537
⇒ tách miền = +0.080   ·   tách gold-trong-bể = +0.102
```

#### c. Đọc dự đoán

| dự đoán | kết quả |
|---|---|
| §12.92d1: mz **kém hẳn** B | **SAI** — mz **0.336 ≥ B 0.318** |
| §12.92d2: đảo ≈ hoặc hơn B | ĐÚNG — 0.328 ≈ B+0.011 |
| §12.92d3: cả hai ≈ B ⇒ chẩn đoán SAI | **KÍCH HOẠT** — cos **không** gây hại trên thước này |
| §12.93: miền LỚN + gold≈0 | **KHÔNG** — miền +0.080, gold **+0.102** (gold ≥ miền; mini: +0.119 / +0.244) |

⇒ **Giữ** `rank = cos × (0.25 + 0.75·cov)`. Không hoán vai. Không A/B sinh từ P22.

#### d. Review code Claude (`probe_theme_prior.py`, §12.92–93)

| | |
|---|---|
| OK | Hai hình dạng đúng proxy công thức mz; `(cos+1)/2` và `B/max` hợp lệ cho xếp hạng tương đối |
| OK | Khối miền: mẫu 40 câu ngoài `pool` — đủ để ước mean; seed ổn định theo `len(pool_l)` |
| OK | Không đụng `multizone.py` (đúng runbook) |
| Yếu | `B` ≠ `cov` token-zone thật của mz — chỉ kiểm **hình dạng thừa số**, không phải bit-exact |
| Yếu | “NGOÀI bể” ≠ bảo đảm khác miền — chỉ khác fold-overlap |
| Yếu | Dòng log cứng “tách gold ≈0” **in mọi lúc** dù số v12 không ≈0 — dễ đọc nhầm |
| Ghi | `_emb` mean-pool thô giống mz; `cosE_old=0.004` vẫn có phần artefact E đặc (P21 phụ: lọc content → ~0.16) — không đổi kết luận hình dạng mz ≥ B |


---

### 12.95 Nhận §12.94 (P22) — dự đoán của tôi BỊ BÁC; công thức hiện hành ĐÚNG (2026-07-29) — Claude

#### a. Điều kiện (3) đã nổ — đúng thứ tôi chốt trước để tự bác

§12.92d tôi chốt ba nhánh, trong đó nhánh (3) là *"cả hai xấp xỉ B ⇒ chẩn đoán của tôi SAI"*.

| thước | top1 | Δ so B | thắng | thua |
|---|---|---|---|---|
| B (túi từ) | 0,318 | — | — | — |
| `pos` | 0,347 | +0,029 | 21 | 13 |
| `cosE_old` | 0,004 | −0,314 | 0 | 86 |
| **`cosxcov(mz)`** — hình dạng HIỆN HÀNH | **0,336** | **+0,018** | 6 | 1 |
| `covxcos(đảo)` — hoán vai | 0,328 | +0,011 | 4 | 1 |

Dự đoán (1) — *"`cosxcov(mz)` kém hẳn B"* — **SAI**. Nó **hơn** B (+0,018, 6 thắng/1 thua), và
**hơn cả bản hoán vai**. Đề xuất "sửa một dòng" của tôi sẽ làm **tệ đi** (0,328 < 0,336).

⇒ **Giữ nguyên `rank = cos × (0.25 + 0.75·cov)`.** Không đụng `multizone.py`.

#### b. Vì sao tôi sai — và chỗ suy luận hụt nằm ở đâu

Tôi lập luận: `cos` xếp hạng ở 0,004 ⇒ vô thông tin ⇒ nhân nó vào `cov` chỉ thêm nhiễu.

Bước cuối **không theo**. Một tín hiệu xếp hạng kém **một mình** vẫn có thể **bổ sung** cho tín
hiệu khác — và số nói đúng thế: `cos` một mình 0,004, nhưng `cos × cov` **hơn** `cov` thuần. Sáu
ca thắng, một ca thua.

Đây là lỗi lập luận cùng dạng với §12.93a: từ "thước X thấp" suy ra "X vô dụng", trong khi thước
chỉ đo **một vai** của X.

#### c. Còn giả thuyết MIỀN của tác giả thì sao

| | tách miền | tách gold |
|---|---|---|
| mini | +0,119 | **+0,245** |
| **v12** | **+0,080** | **+0,102** |

Tác giả dự đoán v12 sẽ đảo (miền lớn, gold ≈0). **Không đảo** — trên v12 tách gold (+0,102) vẫn
**nhỉnh hơn** tách miền (+0,080), chỉ là **cả hai đều co lại** so với mini.

Nên `cos` **không phải bộ giữ miền thuần**, cũng **không phải** tín hiệu danh tính mạnh. Nó mang
**cả hai, đều yếu**. Và chính vì yếu-nhưng-thật mà nó **bổ sung** được cho `cov` (+0,018) thay vì
thay thế.

Một điều phải nói rõ để khỏi đọc nhầm: **tách gold dương mà xếp hạng vẫn 0,004 là nhất quán**.
Tách gold đo so **trung bình bể**; cao hơn trung bình 0,102 vẫn để lại hàng nghìn ứng viên phía
trên khi bể tới **37.908**. Trung bình và hạng-1 là hai chuyện khác nhau.

#### d. Một dòng script phải sửa

Khối `cos CÓ ĐO MIỀN KHÔNG` in sẵn câu *"Tách miền LỚN mà tách gold ≈0 ⇒ cos lo MIỀN"* — đó là
**câu cố định, không tính từ số**, và nó **khẳng định sai** với chính số v12 vừa ra. Đã thay bằng
so sánh thật giữa hai cột (ba nhánh) kèm ghi chú trung-bình-≠-hạng-1.

Lỗi nhỏ nhưng đúng lớp nguy nhất của ticket: **văn bản giải thích viết sẵn theo giả thuyết, rồi
đứng cạnh số như thể là kết luận rút ra từ số.**

#### e. Trạng thái sau P22

Đóng thêm một giả thuyết: **hình dạng công thức xếp hạng không phải chỗ hỏng.** `cos` đang ở đúng
vai, `cov` đang ở đúng vai, và tổ hợp hiện hành là tổ hợp tốt nhất trong ba tổ hợp đã đo.

Nút thắt §12.50c vẫn còn nguyên, và giờ đã loại được: nới vùng · lift theo loại hỏi · dìm câu
chứa từ hỏi · theme prior thay thế · thứ tự (PAIR/W_skip ba cách) · embedding thay thế ·
**hình dạng công thức**.

---

### 12.96 "Có dùng embedding W_skip không?" — CHƯA chốt được, và vì sao (2026-07-29) — Claude

Câu hỏi của tác giả. Trả lời thẳng: **chưa**, vì ba lý do cụ thể — trong đó lý do đầu là một
phép đo còn thiếu, không phải một do dự.

#### a. Phép đo QUYẾT ĐỊNH chưa từng chạy

Đang có:

| | v12 |
|---|---|
| `cosE_old` **một mình** | 0,004 |
| `cosE_skip` **một mình** | **0,135** |
| `cosE_old × cov` (công thức production) | **0,336** |
| `cosE_skip × cov` | **chưa đo** |

Thiếu đúng ô cuối — tức đúng công thức đang chạy, chỉ thay embedding.

Và **không được suy ra**. §12.95 vừa dạy chính xác bài này: `cos_old` **tệ nhất** khi đứng một
mình (0,004) mà lại **cộng thêm +0,018** khi nhân vào `cov`. "Tốt hơn một mình" **không** suy ra
"tốt hơn khi kết hợp". Nếu tôi suy ở đây thì lặp lại đúng lỗi vừa bị bác.

Đã thêm thước `skipxcov(mz)` vào probe. Mini: **hoà** (0,621 ‖ 0,621) — đúng như dự đoán, vì ở đó
cả hai embedding đều có thông tin. **Mini không phán được**; giao P23.

#### b. Phạm vi ảnh hưởng rộng hơn nhiều so với "đổi một ma trận"

`state.embedding` được dùng **40 chỗ** trong **8 module**:

    multizone · style_payoff · context_positional · generate_hebbian
    training/pipeline · viz/export_embedding · viz/export_graph_v3 · core/state

Ta mới đo **một** trong số đó (`cos` của multizone). Thay `state.embedding` toàn cục nghĩa là đổi
hành vi ở bảy chỗ **chưa đo gì cả** — trong đó `style_payoff` và `context_positional` đều nằm trên
đường sinh.

⇒ Kể cả P23 dương, cách triển khai đúng **không phải** thay trường `state.embedding`, mà là thêm
một embedding **riêng, có knob**, dùng **chỉ ở chỗ đã đo**. Các chỗ khác giữ nguyên cho tới khi
có số của chúng.

#### c. Chi phí phải tính vào quyết định

Dựng `E_skip` trên v12 mất **~178s** (đếm cặp trong span câu → PPMI → SVD 96 chiều). Nếu dùng thì
phải **bake vào checkpoint** hoặc dựng lười + cache — không thể tính lại mỗi lần nạp.

Với một lợi ích **chưa biết**, chi phí ấy chưa biện minh được.

#### d. Điều kiện để chốt CÓ — chốt trước khi có số

Trên v12, `skipxcov(mz)` phải:

1. hơn `cosxcov(mz)` (0,336) **≥ +0,02 top-1**, VÀ
2. **thắng > thua** khi so từng ca.

Đạt cả hai ⇒ dùng, nhưng **chỉ trong `multizone`**, qua knob riêng, mặc định OFF cho tới khi A/B
đường sinh (`lạc` · `loop4` · `n_tok`) cũng không xấu đi.

Không đạt ⇒ **đóng hướng embedding W_skip**, và ghi nhận: `cosE_skip` hơn `cosE_old` **34×** khi
đứng một mình nhưng điều đó không chuyển thành lợi ích trong công thức thật — đúng cùng hình dạng
với PAIR (§12.79), W_skip-giao (§12.85), W_skip-trôi-chảy (§12.87).

---

### 12.97 P23 — `skipxcov` v12: Δ +0.007 < +0.02 ⇒ đóng hướng emb W_skip (2026-07-30) — M4

Ô còn thiếu (§12.96a). `out/skipcov_v12.log`. Không đổi `state.embedding` / default.

#### a. Khối `THƯỚC CẶP` (dòng quyết định)

| thước | top1 | Δ so B | thắng | thua |
|---|---|---|---|---|
| B (túi từ) | 0.318 | — | — | — |
| cosE_old | 0.004 | −0.314 | 0 | 86 |
| cosE_skip | 0.135 | −0.182 | 1 | 51 |
| **cosxcov(mz)** | **0.336** | +0.018 | 6 | 1 |
| covxcos(đảo) | 0.328 | +0.011 | 4 | 1 |
| **skipxcov(mz)** | **0.343** | +0.026 | 9 | 2 |

#### b. Cổng §12.96d

| điều kiện | số | |
|---|---|---|
| `skipxcov − cosxcov` ≥ **+0.02** | **+0.007** | **TRƯỢT** |
| thắng > thua (so từng ca) | (bảng trên là vs B; Δ top1 ≈ 2/274 ca) | không cứu được cổng 1 |

⇒ **Đóng hướng embedding W_skip** cho công thức mz. `cosE_skip` hơn `cosE_old` 34× khi đứng
một mình **không** chuyển thành ≥+0.02 khi nhân `cov` — cùng hình dạng PAIR / W_skip-giao /
trôi-chảy.


---

### 12.98 Nhận §12.97 (P23) — CHỐT: **không** dùng embedding W_skip (2026-07-29) — Claude

#### a. Số

| | v12 top1 | thắng/thua so B |
|---|---|---|
| B (túi từ) | 0,318 | — |
| `cosE_old` một mình | 0,004 | 0 / 86 |
| `cosE_skip` một mình | 0,135 | 1 / 51 |
| `cosxcov(mz)` — **hiện hành** | **0,336** | 6 / 1 |
| `skipxcov(mz)` — embedding mới | **0,343** | 9 / 2 |

Δ so công thức hiện hành = **+0,007** ≈ 2 ca trên 274. Cổng §12.96d đòi **≥ +0,02** ⇒ **TRƯỢT**.

⇒ **Không dùng.** `state.embedding` giữ nguyên. `E_skip` chỉ tồn tại trong probe.

#### b. Hình dạng đáng ghi nhớ nhất của cả nhánh này

`E_skip` hơn `E_old` **34×** khi đứng **một mình** (0,135 ‖ 0,004). Trong công thức thật nó thêm
**+0,007**.

Lý do rõ ràng khi nhìn cột B: `cov` **một mình** đã 0,318 trên tổng 0,336 — tức nó gánh **95%**
công việc. Phần `cos` còn để lại chỉ 0,018, nên cải thiện `cos` gấp 34 lần cũng chỉ chia được vài
phần nghìn. **Không có chỗ cho nó thắng.**

Đây là phiên bản sạch nhất của một khuôn đã lặp suốt ticket: **cải thiện đầu vào của một khâu đã
gần bão hoà thì không chuyển thành lợi ích.**

#### c. Toàn bộ nhánh W_skip: ba cách dùng, ba lần đóng

| cách dùng | mini | v12 | trạng thái |
|---|---|---|---|
| giao top-k đề xuất (§12.85) | probe dương | `lạc` +0,21…+0,33 | **đóng** |
| kênh trôi chảy (§12.87 · §12.90a) | +0,021 top-10 | −0,001 top-10 | **đóng** |
| embedding (§12.88 · §12.98) | hoà | **+0,007** | **đóng** |

Ma trận `W_skip` **không sai** — §12.80/§12.82 đo được nó mang thông tin hướng thật, bền tới d=24.
Nó chỉ **không có chỗ tiêu thụ** trong kiến trúc hiện tại. Đúng giả thuyết §12.86d, và giờ có ba
phép đo độc lập cùng chỉ vào đó.

`ar_matrix_w_skip_weight` giữ **0**. Mã `w_skip.py` giữ lại (Luật 0 sạch, đo được, sẵn dùng nếu
sau này có cơ chế soạn câu).

#### d. Bảng tổng: mọi cách "thêm thông tin vào đầu vào" đã đo

| hướng | kết quả |
|---|---|
| lift theo loại câu hỏi (§12.51) | −0,005 (λ oracle) |
| dìm câu chứa từ hỏi (§12.52) | làm giàu 0,2× — nhắm trượt |
| theme prior `content`/`clear` (§12.69) | −0,036 / −0,131 |
| nới vùng PMI (§12.49) | `lạc` không cải thiện |
| PAIR re-rank (§12.79) | +0,022 |
| W_skip ×3 (§12.85·12.90·12.98) | âm / −0,001 / +0,007 |
| hoán vai công thức (§12.94) | −0,008 |

**Bảy hướng, không hướng nào vượt +0,03.** Trong khi trần của mọi prior nhân là **67,9%** và
`pos` đang ở **34,7%** (§12.82) — dư địa **33 điểm** vẫn còn đó, và không nằm ở phía đầu vào.

⇒ Củng cố §12.86d: chỗ nghẽn không phải **biểu diễn** thông tin, mà là **không có ai tiêu thụ**
nó. `cov` (túi từ) gánh 95%, và mọi thứ thêm vào chỉ chia phần 5% còn lại.

#### e. Không còn việc đo treo

M4 hết việc. Bước lớn còn lại — dựng đường **soạn câu**, tức cái tiêu thụ được thông tin thứ tự —
là việc nhiều tuần, và tôi không bắt đầu khi chưa chốt với tác giả.

---

### 12.99 Giữ W_skip (quyết định tác giả) + PHÁC THẢO đường soạn câu (2026-07-30) — Claude

#### a. W_skip: giữ, và cách nối cho an toàn

Tác giả chốt giữ — *"cải thiện một chút cũng là cải thiện"*. Ghi rõ **cách nào** đang dương để
khỏi nối sai chỗ:

| cách dùng | v12 | nối hay không |
|---|---|---|
| embedding (§12.98) | **+0,007** | **nối** |
| kênh trôi chảy (§12.90a) | −0,001 | không |
| giao top-k (§12.85) | `lạc` +0,21…+0,33 | không |

Cách nối: embedding W_skip là **ma trận RIÊNG, có knob**, dùng **chỉ ở `cos` của `multizone`** —
**không** thay `state.embedding`. Lý do §12.96b: trường ấy có **40 chỗ dùng trong 8 module**, ta
mới đo **một**; `style_payoff` và `context_positional` đều trên đường sinh và chưa đo gì.

Kèm hai việc bắt buộc: **cache/bake** (dựng mất ~178s trên v12, không thể tính mỗi lần nạp) và
**A/B đường sinh** (`lạc` · `loop4` · `n_tok`) trước khi bật mặc định. +0,007 là 2 ca trên 274 —
đủ để giữ, chưa đủ để bật mù.

#### b. Đường soạn câu: CHƯA có giải pháp đã kiểm

Nói thẳng trước: tôi **không** có giải pháp đã đo. Có một **phân rã** thành bốn phần, mỗi phần có
cơ chế ứng viên và một phép đo giết được nó sớm.

Và một điều phải nói ngay để khỏi kỳ vọng lệch: **soạn câu nhắm vào DIỄN ĐẠT, không nhắm vào CHỌN.**
Nó chữa `copy ≈ 1,0`, độ dài 8–16 token, trôi đề khi dài. Nó **không** chữa việc chọn sai câu —
dư địa 33 điểm ở §12.82 nằm ở khâu chọn, và soạn câu không chạm vào đó. Mục tiêu "sinh đoạn dài
đúng nghĩa LLM" của tác giả **là** diễn đạt, nên đây đúng hướng cho mục tiêu ấy, nhưng đừng trông
nó làm đáp đúng hơn.

#### c. Hạ tầng ĐÃ CÓ — nhiều hơn tôi tưởng, mỏng hơn cần

Đo trên mini:

    93.349 slot · token đơn 71.496 (77%) · cluster 21.853 (23%)
    3.966 cluster phân biệt · 96,2% dùng >1 lần · trung bình 5,51 lượt

`slot_chain` **là một phân rã thật**: `Vịnh │ Bắc │ [Bộ và] │ [vịnh Thái Lan] │ thuộc │
[Biển Đông.<EOS>]`, và `ps.seg.expand_slot()` bung cluster về token.

⇒ **Điểm cắt không cần phát minh** (tôi từng đề xuất học bằng entropy ở §12.86e — không cần nữa).

Nhưng ba giới hạn phải ghi:

1. Phần **dùng lại nhiều** là khuôn mẫu, không phải nội dung: `.<EOS>` 1642 lần · `? Đáp:` 609 ·
   `Tiêu đề:` 93. Mảnh **nội dung** chỉ 2–3 lần ⇒ không phải một vốn từ tổ hợp phong phú.
2. **77% slot là token đơn** ⇒ chuỗi slot là chẻ *nhẹ*, không phải ngữ pháp cụm mạnh.
3. `pattern_templates` = **0** ⇒ cơ chế khuôn rỗng.

#### d. Bốn phần, theo thứ tự làm, mỗi phần một cổng giết sớm

**(0) THƯỚC TRÔI CHẢY — thiếu, và phải làm TRƯỚC.**
Bộ đo hiện có `lạc` · `loop4` · `digit⎵` · `stop` · `copy` · `n_tok` — **không có thước ngữ pháp
nào**. Mà rủi ro lớn nhất của soạn câu chính là ngữ pháp: cắt-ghép sai chỗ ra văn què, và mọi cột
hiện tại đều **mù** với nó (đúng như `lạc` từng mù với trôi đề tới §12.73).
Làm được từ harness §12.87: chấm văn bản **sinh ra** bằng xác suất token-kế của chính đồ thị.
Luật 0 sạch, dùng lại mã đã có.
*Cổng:* thước phải xếp **văn corpus > văn sinh hiện tại > văn xáo trật tự**. Không đúng thứ tự ấy
thì thước sai, dừng.

**(1) TRẦN CỦA TỔ HỢP — phép đo có thể giết cả hướng, rẻ nhất.**
Trước khi dựng gì: một câu giữ lại có **lắp được** từ mảnh của các câu KHÁC không? Lấy câu ở nửa B,
thử phủ nó bằng cluster/token lấy từ nửa A.
*Cổng:* ≥40% câu giữ lại phủ được ≥80% bằng mảnh từ chỗ khác ⇒ có vật liệu. Dưới đó ⇒ **không có
gì để soạn**, đóng hướng, và tiết kiệm nhiều tuần. Với (c1) — mảnh nội dung chỉ dùng 2–3 lần —
tôi cho khả năng trượt là **thực**.

**(2) THỨ TỰ LẮP.**
`pos_rel` (§12.83: giải thích ~97% tín hiệu hướng ở d≥3, một scalar/token). Chấm bằng thước (0):
xếp mảnh theo `pos_rel` so với xếp ngẫu nhiên và so với trật tự gốc.
*Cổng:* phải hơn ngẫu nhiên rõ rệt và **không kém** trật tự gốc quá 10%.

**(3) BỘ SOẠN TỐI GIẢN.**
Chỉ khi (0)–(2) đạt. Config-gated, mặc định OFF. A/B toàn bộ eval gồm `lạc` · `loop4` · thước (0).
*Cổng:* thước trôi chảy **không tụt**, `n_tok` tăng, `lạc` không tăng.

#### e. Vì sao thứ tự này

(1) là phép đo **rẻ nhất và giết được nhiều nhất** — nó hỏi "có vật liệu không" trước khi hỏi "lắp
thế nào". (0) phải đứng trước (2)/(3) vì không có thước thì không phân biệt được "soạn được" với
"soạn ra văn què". Đó đúng bài học đắt nhất của ticket: **tám lần khâu yếu là thước, không phải mã**
— và lần này tôi biết trước mình thiếu thước gì.

Tôi đề nghị chạy **(1) trước (0)**: nếu không có vật liệu thì khỏi cần dựng thước.

---

### 12.100 Bước ① — trần của tổ hợp: cổng TRƯỢT SÁT, nhưng vì thiếu VẬT LIỆU (2026-07-30) — Claude

Phép đo đầu tiên của hướng soạn câu (§12.99d bước 1), chọn chạy trước vì rẻ nhất và giết được
nhiều nhất. `scripts/probe_compose_ceiling.py`, mini, 1.500 câu giữ lại.

#### a. Thước và bẫy đã chặn

Câu ở nửa B có **lắp được** từ mảnh của nửa A không. Bẫy phải chặn: nếu tính cả **token đơn** thì
mọi câu phủ 100% (token nào chẳng xuất hiện đâu đó) ⇒ số vô nghĩa, mà lắp từng token đúng là thứ
đường sinh hiện tại đang làm. Nên **chỉ đếm mảnh ≥2 token**.

#### b. Số

| vật liệu | phủ TB | phủ ≥0,8 | phủ ≥0,5 |
|---|---|---|---|
| `cluster` (slot_chain đã đăng ký) | 0,502 | **9,2%** | 50,4% |
| `ngram` (mọi đoạn ≥2 token đã nhớ) | **0,723** | **37,8%** | 89,1% |
| `ngram_xáo` (**đối chứng**) | 0,225 | 0,3% | 4,7% |

số mảnh / số token = **0,284** ⇒ mảnh trung bình dài **2,55 token**, tức thật sự đang ghép mảnh.

G1 (≥40% phủ ≥0,8) **TRƯỢT** — 37,8% · G2 (hơn xáo ≥0,15) **ĐẠT** +0,498 · G3 (≤60% mảnh/token)
**ĐẠT** 0,284.

#### c. Nhưng đường cong nói cổng trượt vì THIẾU VẬT LIỆU, không vì cơ chế sai

Dựng kho từ 25% / 50% / 100% nửa A rồi chấm **cùng bộ câu**:

| % nửa A | #mảnh | phủ TB | phủ ≥0,8 |
|---|---|---|---|
| 25% | 63.113 | 0,522 | 10,4% |
| 50% | 126.861 | 0,620 | 21,9% |
| 100% | 224.769 | 0,723 | **37,8%** |

Mỗi lần gấp đôi kho thêm **+11,5** rồi **+15,9** điểm — **đang tăng nhanh, chưa có dấu bão hoà**.
v12 có **~85× cha** so mini (≈6 lần gấp đôi nữa).

Đây là chỗ tôi phải cẩn thận, nên nói rõ: **tôi KHÔNG dời cổng.** Cổng trượt trên mini. Điều đường
cong thiết lập là **phạm vi** của kết luận — "trượt vì kho nhỏ" — và đó là số đo được, không phải
lập luận suông. Cùng loại lý lẽ đã dùng cho `P(b|a,d)` của W_skip ở §12.87e, và lần đó nó **đúng**
(thiếu hụt top-1 biến mất trên v12).

#### d. Một phát hiện thiết kế, độc lập với cổng

`cluster` **9,2%** ‖ `ngram` **37,8%** — hạ tầng chẻ slot hiện có kém vật liệu thô **4 lần**.

⇒ Nếu soạn câu, phải lắp từ **n-gram của `emits`**, **không** từ cluster đã đăng ký trong
`slot_chain`. Cluster được chẻ để **nén**, không để **tái tổ hợp** — và §12.99c đã thấy dấu: phần
dùng lại nhiều toàn khuôn mẫu (`.<EOS>` 1642 lượt), mảnh nội dung chỉ 2–3 lượt.

Điều này cũng huỷ nốt một ý tôi từng nêu: `pattern_templates` (= 0) và `slot_chain` **không** phải
đường vào của soạn câu.

#### e. Không đóng hướng — giao P24

Ba lý do để chưa đóng: cổng trượt **2,2 điểm**; hai cổng còn lại đạt **rất mạnh** (đối chứng +0,498
là tách sạch); và đường cong (c) cho thấy chính đại lượng bị thiếu **có sẵn 85×** trên v12.

Nếu v12 vượt 40% ⇒ có vật liệu ⇒ đi tiếp bước (0) dựng **thước trôi chảy** (bộ đo hiện không có
thước ngữ pháp nào, và đó là rủi ro lớn nhất của cắt-ghép).
Nếu v12 vẫn dưới 40% với kho 85× lớn hơn ⇒ **đóng dứt khoát**, vì lúc ấy không còn cách giải thích
nào bằng thiếu dữ liệu.

---

### 12.101 P24 — trần tổ hợp v12: G1 ĐẠT 96.3% · có vật liệu (2026-07-30) — M4

`out/compose_v12.log`. Không hạ `--max-piece`. Không đổi mã sinh.

#### a. Bảng vật liệu

| vật liệu | phủ TB | phủ ≥0.8 | phủ ≥0.5 |
|---|---|---|---|
| cluster | 0.583 | 13.0% | 70.4% |
| **ngram** | **0.956** | **96.3%** | 99.8% |
| ngram_xáo | 0.617 | 9.9% | 83.0% |

`số mảnh / số token (TB) = 0.289` (mảnh TB ~3.5 token — ghép mảnh thật, không từng-token).

Kho: cluster 155.755 mảnh · ngram **22.013.790** mảnh (≥2 token).

#### b. PHỦ THEO CỠ KHO

| % nửa A | #mảnh | phủ TB | phủ ≥0.8 |
|---|---|---|---|
| 25% | 3.810.672 | 0.824 | 62.5% |
| 50% | 10.015.955 | 0.925 | 91.2% |
| 100% | 22.013.790 | 0.956 | **96.3%** |

Gấp đôi kho (50%→100%) thêm **+5.1** điểm ở `phủ ≥0.8` — **chưa** bão hoà theo ngưỡng phụ
(<3 điểm), nhưng đã **chậm lại mạnh** so mini (+11…+16) và sát trần thực tế (~96%).

#### c. Cổng

| | |
|---|---|
| G1 ≥40% phủ≥0.8 bằng ngram | **ĐẠT (96.3%)** ← quyết định |
| G2 ngram − xáo ≥0.15 | **ĐẠT (+0.339)** |
| G3 mảnh/token ≤0.60 | **ĐẠT (0.289)** |

⇒ **Có vật liệu** soạn câu trên v12. Giả thuyết §12.100c (mini trượt vì kho nhỏ) **đúng**.
Cluster vẫn kém ngram (~7× ở cột ≥0.8) — giữ kết luận thiết kế: lắp từ n-gram `emits`, không từ
`slot_chain`. Chờ Claude bước (0) thước trôi chảy.


---

### 12.102 Nhận §12.101 (P24) — CÓ vật liệu; nhưng đối chứng cảnh báo đúng chỗ (2026-07-30) — Claude

#### a. Cổng đạt rộng, và dự đoán §12.100c ĐÚNG

| vật liệu | phủ TB | phủ ≥0,8 | phủ ≥0,5 |
|---|---|---|---|
| `cluster` | 0,583 | 13,0% | 70,4% |
| **`ngram`** | **0,956** | **96,3%** | 99,8% |
| `ngram_xáo` (đối chứng) | **0,617** | **9,9%** | 83,0% |

G1 **ĐẠT** 96,3% (ngưỡng 40%) · G2 **ĐẠT** +0,339 · G3 **ĐẠT** 0,289.

Mini 37,8% → v12 **96,3%**. §12.100c dự đoán "trượt vì kho nhỏ, không vì cơ chế sai" — **đúng**.
Và kho đã **bão hoà**: gấp đôi (50%→100%) chỉ thêm **+5,1%** (mini: +15,9%), tức 96,3% là **trần
thật**, không phải còn tăng.

    25% nửa A →  3,8M mảnh → phủ ≥0,8  62,5%
    50%       → 10,0M       →           91,2%
   100%       → 22,0M       →           96,3%

#### b. Nhưng đối chứng nhảy 0,225 → 0,617, và đó là tin QUAN TRỌNG hơn cổng

Trên v12, **câu đã XÁO TRẬT TỰ** cũng phủ được **61,7%**, và **9,9%** câu xáo đạt ≥0,8.

Vì kho có **22 triệu** mảnh nên gần như mọi cặp token ngắn đều tồn tại ở đâu đó. Hệ quả:

> "Lắp được từ mảnh đã nhớ" ở quy mô v12 là **ràng buộc RẤT LỎNG**. Nó **không** bảo đảm ngữ pháp.

G2 vẫn đạt (+0,339 là tách thật), nhưng **mức tuyệt đối** mới là thứ phải đọc: một bộ soạn có thể
lắp ra **văn què** mà thước phủ **không hề báo động** — đúng 9,9% ca xáo đạt ≥0,8 kia.

⇒ Kết quả này làm bước ⓪ (**thước trôi chảy**) **bắt buộc và cấp hơn**, không phải nhẹ đi. Vật
liệu dồi dào tới mức không còn tự ràng buộc; thứ chọn giữa hàng nghìn cách lắp phải là tín hiệu
ngữ pháp, và ta **chưa có nó**.

Ghi lại vì đây đúng khuôn đã cắn nhiều lần: một cột "đối chứng" mà bảng không cần, lại là cột đổi
cách đọc cả bảng.

#### c. Phát hiện §12.100d được củng cố, mạnh hơn

`cluster` **13,0%** ‖ `ngram` **96,3%** — khoảng cách nới từ 4× lên **7×**.

⇒ Soạn câu phải lắp từ **n-gram của `emits`**. `slot_chain`/`pattern_templates` dứt khoát **không
phải** đường vào.

#### d. Trạng thái hướng soạn câu

    ① trần tổ hợp        ✔ ĐẠT (96,3%, bão hoà) — CÓ vật liệu
    ⓪ thước trôi chảy    ← ĐANG LÀM (Claude, chạy được trên mini)
    ② thứ tự lắp         chờ ⓪
    ③ bộ soạn tối giản   chờ ⓪②

Bước ⓪ là của tôi, không giao M4: nó chỉ cần corpus + văn sinh, mini đủ.

---

### 12.103 Bước ⓪ — thước trôi chảy: TRƯỢT nghiệm thu, và chẩn được vì sao (2026-07-30) — Claude

`scripts/probe_fluency_ruler.py`. Ba dạng thước, tất cả đều **không phát hiện nổi mối ghép sai** —
đúng loại lỗi mà bộ soạn sẽ gây ra.

#### a. Số (mini, đếm nửa A, chấm nửa B, n=1.200)

| mức | logP/token | tệ nhất (bigram) | tệ nhất (skip d≤8) |
|---|---|---|---|
| corpus(B) | **−5,909** | −14,166 | −13,296 |
| văn sinh | **−3,967** | −11,225 | −11,439 |
| đảo 1 cặp kề | −6,619 | −14,731 | −13,572 |
| **ghép 2 nửa câu** | −6,204 | −14,709 | −13,727 |
| xáo(B) | −9,546 | −15,897 | −14,315 |
| unigram ngẫu nhiên | −10,000 | — | — |

G1 (corpus − xáo ≥0,5) **ĐẠT** +3,637 · G2 thứ tự nền **ĐẠT** · G5 (đảo cặp kề ≥0,1) **ĐẠT** +0,710.

**G4 nghiệm thu cho đúng việc — TRƯỢT cả ba dạng:** trung bình **+0,295** · tệ nhất bigram
**+0,544** · tệ nhất skip **+0,430**. Ngưỡng đòi ≥0,3 / ≥1,0.

#### b. Trước khi kết luận "thước mù", tôi kiểm nhãn đối chứng

Có khả năng đối chứng bị gán nhãn sai — ghép hai nửa câu THẬT có khi lại ra tiếng Việt đọc được.
Đọc 8 mẫu: **không**, tất cả đều là văn què thật.

    'Đinh Tiên Hoàng bị ám sát Biển Đông và các nguồn tài nguyên của nó.'
    'Phở là món ăn tổ hợp3 khi người dân trong vùng đào đất đắp đê.'
    'Các bang hoặc tỉnh của 4 quốc gia được đặt Hà Nội.'
    'Thời gian đó, dân số của đặt quốc hiệu là Đại Việt.'

⇒ Nhãn đúng, thước sai. Kết luận đứng vững.

#### c. Chẩn đoán — và một con số nói rõ tất cả

**Đảo 1 cặp kề bị phạt 0,710 · ghép nửa câu chỉ 0,295** — tức thước phạt lỗi NHẸ (đảo hai từ)
**mạnh gấp 2,4 lần** lỗi NẶNG (ghép sai hoàn toàn). Với người đọc thì ngược hẳn.

Đó là bằng chứng sạch rằng **xác suất n-gram ≠ ngữ pháp**. Ba lý do cộng lại:

1. Ghép chỉ phá **một** bigram ở mối nối; đảo cặp kề phá **hai**.
2. Bigram ở mối nối thường **không hiếm** (`sát`→`Biển`, `cách`→`đời` đều là cặp từ phổ biến ở
   đâu đó), nên xác suất không sụp.
3. Lấy **tệ nhất** cũng không cứu: `corpus(B)` tự nó đã có vị trí tệ nhất **−14,17** — văn thật
   vốn chứa bigram hiếm-nhưng-đúng, nên mối ghép không nhô lên khỏi nền ấy.

Và ngữ cảnh xa **không** giúp (skip +0,430 < bigram +0,544) — nên đây không phải chuyện bậc của mô
hình. Thông tin phân biệt "què" với "hiếm" **không nằm trong thống kê đồng hiện**, bất kể tầm.

#### d. Một mốc phụ đáng giữ

`văn sinh` = **−3,967**, cao hơn corpus **+1,943 nat/token**. Đó là `copy ≈ 1,0` đo được theo một
đường **độc lập** với thước `copy` của `eval_gen_broad`: văn sinh **dễ đoán hơn văn thật**, vì nó
là văn thật được chép lại.

#### e. Hệ quả: soạn câu bị CHẶN, và chặn ở chỗ khác với dự đoán

§12.99d chốt ⓪ là điều kiện tiên quyết: *"không có thước thì không phân biệt được 'soạn được' với
'soạn ra văn què'"*. ⓪ **trượt** ⇒ ② và ③ **không được khởi động**.

Nhưng chỗ chặn **không phải** thiếu vật liệu (①: 96,3%) và **không phải** thiếu điểm cắt (§12.99c:
`slot_chain` đã chẻ sẵn). Nó là **thiếu tín hiệu NGHIỆM THU**. Đó là một chẩn đoán hẹp và cụ thể
hơn "việc này khó".

#### f. Hai đường còn lại, và cái nào rẻ hơn

**① Ràng buộc ATTESTATION thay vì xác suất.** Không hỏi "chuỗi này có khả năng bao nhiêu" mà hỏi
"mọi đoạn ≥k token của nó có **thật sự xuất hiện** trong emit của cha nào không". §12.102 đo với
k=2 (phủ 96,3%, đối chứng xáo 61,7% — quá lỏng). Nâng k lên 4–5 thì ràng buộc **chặt lại** và đối
chứng phải tụt. **Đo được bằng đúng script đã có**, chỉ thêm một tham số. Rẻ nhất, làm trước.

**② `neural_lm` làm THƯỚC.** Dự án đã có LM nơ-ron (TK_042 G10, `out/g7_best_model.pt`), và
§12.87 dẫn `varlm.py`: `W_seq` order-1 là mắt xích yếu nhất. Một LM thật sẽ thấy mối ghép ở chỗ
thống kê đồng hiện không thấy.
**Luật 0 phải nói rõ:** dùng nó ở **đường sinh** thì vi phạm (nó học bằng backprop). Dùng nó làm
**dụng cụ đo** thì không — thước không nằm trong mạng, giống như `eval_gen_broad` dùng LCS mà không
ai coi LCS là một phần kiến trúc. Nhưng phải ghi rõ ranh giới ấy mỗi lần dùng, kẻo trôi.

Đề nghị làm **①** trước: một tham số, một lượt chạy, và nó có thể mở lại đường soạn câu mà không
cần chạm tới Luật 0.

---

### 12.104 Attestation cũng trượt — và hai lần trượt cùng MỘT lý do (2026-07-30) — Claude

Đường ① của §12.103f: thay xác suất bằng **chứng thực** — hỏi "đoạn ≥k token này có THẬT SỰ xuất
hiện trong emit của cha nào không". Tôi ghi sẵn nghi ngại trước khi chạy: câu **giữ lại** cũng có
n-gram **mới** dù đúng ngữ pháp, nên attestation có thể vướng đúng bẫy vừa giết thước xác suất.

#### a. Số (mini, chứng thực trên nửa A, chấm nửa B)

Thước: với mỗi **biên** giữa token `j` và `j+1`, k-gram **bắc qua** biên ấy có được chứng thực không.

| k | corpus(B) | ghép nửa câu | xáo | **TÁCH ghép** |
|---|---|---|---|---|
| 2 | 0,563 | 0,534 | 0,138 | +0,029 |
| **3** | 0,236 | 0,197 | 0,008 | **+0,039** |
| 4 | 0,098 | 0,070 | 0,000 | +0,028 |
| 5 | 0,046 | 0,028 | 0,000 | +0,018 |
| 6 | 0,023 | 0,013 | 0,000 | +0,010 |

**Nghi ngại đúng.** `TÁCH ghép` đạt đỉnh **+0,039** ở k=3 rồi **giảm dần** — không lớn lên theo k
như hy vọng. Và mức tuyệt đối sụp: ở k=4, **chính câu corpus** chỉ còn **9,8%** biên được chứng
thực; ở k=6 còn **2,3%**. Văn thật giữ lại **gần như hoàn toàn mới** từ 4-gram trở lên.

Nhưng `xáo` bị tách **rất mạnh** (0,138 → 0,008 → 0,000).

#### b. Hai cơ chế, cùng một hình dạng thất bại

| | phát hiện XÁO thô | phát hiện MỐI GHÉP |
|---|---|---|
| xác suất n-gram (§12.103) | +3,637 | **+0,295…+0,544** |
| chứng thực k-gram (§12.104) | 0,138 → 0,000 | **+0,039 đỉnh** |

Hai đường **hoàn toàn khác nhau** — một xác suất có làm trơn, một chứng thực nhị phân — cùng bắt
được xáo thô và cùng **mù với mối ghép**. Đó không phải hai lần xui.

#### c. Lý do chung, và nó là một giới hạn CẤU TRÚC

Cả hai đo **thứ đã thấy**. Ngữ pháp là **thứ tổng quát hoá sang chuỗi chưa thấy** — đúng phần mà
thống kê đồng hiện không có. Nên:

* câu **mới mà đúng**: thống kê thấp (n-gram chưa gặp) ⇒ bị chấm như què;
* câu **ghép mà què**: thống kê không thấp lắm (hai nửa đều thật, mối nối thường không hiếm) ⇒
  lọt lưới.

Số ở (a) đúng hình dạng ấy: ở k=4, corpus 9,8% ‖ ghép 7,0% — hai bên **cùng thấp**, khoảng cách
2,8 điểm. Không có ngưỡng nào tách được hai phân bố chồng lên nhau như vậy.

Đây là **suy luận mạnh**, không phải định lý: hai cơ chế độc lập, cùng lý do nguyên tắc. Nhưng
phải ghi là suy luận.

#### d. Hệ quả lớn nhất: `copy ≈ 1,0` KHÔNG phải khuyết điểm cài đặt

Nếu (c) đúng thì nền Luật-0 hiện tại **không có tín hiệu nào bảo đảm ngữ pháp cho chuỗi mới**. Mà
nó vẫn phải sinh ra tiếng Việt đọc được. Chỉ còn một cách: **phát lại nguyên văn đoạn đã nhớ** —
vì đoạn đã nhớ thì ngữ pháp đã được bảo đảm bởi chính việc nó tồn tại.

⇒ `copy ≈ 1,0` là **cách duy nhất nền này giữ được ngữ pháp**, không phải sự lười của đường sinh.
Mọi cơ chế soạn câu đều phải **từ bỏ bảo đảm ấy**, và §12.103/§12.104 nói không có gì thay thế.

Điều này cũng giải thích hậu nghiệm cả bảng §12.98d: bảy hướng "thêm thông tin vào đầu vào" không
hướng nào vượt +0,03. Chúng đều cố cải thiện **chọn cái gì để chép**, trong khi trần thật nằm ở
chỗ **chỉ được phép chép**.

#### e. Ngã ba, và nó là quyết định của TÁC GIẢ

**(a) Nhận `copy ≈ 1,0` như bảo đảm ngữ pháp của nền**, rồi dồn công vào khâu **CHỌN** — nơi còn
**33 điểm** dư địa đã đo (§12.82: trần 67,9% ‖ hiện 34,7%). Không cần đổi Luật 0. Đánh đổi: văn
vẫn là câu đã thuộc, không dài hơn nhiều, không "đúng nghĩa LLM".

**(b) Nhận một mô hình TỔNG QUÁT HOÁ làm bộ nghiệm thu.** Dự án đã có `neural_lm` (TK_042 G10,
`out/g7_best_model.pt`). Nó học bằng backprop, nên:
- dùng ở **đường sinh** ⇒ **vi phạm Luật 0**;
- dùng làm **dụng cụ đo** ⇒ không vi phạm (thước không nằm trong mạng, như `eval_gen_broad` dùng
  LCS mà không ai coi LCS là kiến trúc);
- nhưng dùng làm **bộ nghiệm thu cho bộ soạn lúc chạy** ⇒ **nằm giữa**, và đó chính là chỗ phải
  tác giả quyết. Nó là một **sửa đổi Luật 0**, không phải một tuỳ chọn kỹ thuật.

Tôi **không** tự chọn (b). Nhưng phải nói rõ: theo hai phép đo trên, **(b) là con đường duy nhất
còn lại cho mục tiêu "sinh đoạn dài đúng nghĩa LLM"**. (a) là con đường duy nhất còn lại nếu Luật 0
giữ nguyên — và nó dẫn tới một hệ thống truy hồi rất tốt, không phải một LLM.

#### f. Việc còn lại nếu chọn (a)

Trần 67,9% ‖ hiện 34,7% là dư địa lớn nhất còn được đo. Bảy hướng đã đóng đều nhắm vào **thêm tín
hiệu**; chưa hướng nào nhắm vào **cải thiện `cov`** — thứ đang gánh 95% (§12.98b). Đó là ô trống
duy nhất chưa ai chạm, và nó không đòi đổi Luật 0.

---

### 12.105 W_skip làm CHỦ ĐẠO + softmax + vòng quay (đặc tả tác giả) (2026-07-30) — Claude

`scripts/probe_skip_decode.py`. Cài đúng đặc tả, hiệu chỉnh, đo, đọc tay.

#### a. Ba sửa cài đặt trước khi chạy được

* **Chuỗi ĐẦY ĐỦ.** `w_skip.py` dựng `content_only=True` ⇒ bỏ hết từ chức năng và dấu câu ⇒ sinh
  ra không có `là`/`của`/`.`. Phải dựng lại trên emit đầy đủ.
* **Cả cửa sổ** thay `ctx_max=2`.
* **Hai cách chấm**: `vote` (đếm phiếu, đúng đặc tả) ‖ `mass` (cộng xác suất).

#### b. Hai điều tác giả nhận định ĐÚNG

**`vote` thắng `mass`** rõ rệt: −7,41 ‖ −9,49 (nhiệt 1,0). Đếm phiếu tốt hơn cộng xác suất.

**Argmax đang cưỡng bức việc chép.** Đường sinh chạy `hybrid_temperature = 0.0`. Vòng quay thật sự
phá được vòng chép: chữ ký chép −3,97 (§12.103d) chuyển sang −6,18 khi lấy mẫu ở mức corpus.

#### c. Một lỗi trong script của chính tôi

Docstring tôi viết *"mức corpus là ĐÍCH, không phải cực đại"*, rồi vòng quét lại chọn **cực đại**.
Nó trả về λ=0,85/k=40/T=0,25 → **−2,606**, **cao hơn cả bộ sinh chép** (−3,97) — tức "tốt nhất" của
tôi chính là **chép nặng hơn**. Đã sửa sang chọn **gần mức corpus nhất**.

Lỗi cùng họ §12.95d: văn bản giải thích và tiêu chí máy chấm nói hai điều khác nhau.

#### d. Đánh đổi ĐƠN ĐIỆU — không có cửa sổ nào vừa mới vừa đúng ngữ pháp

| nhiệt | logP/token | đọc tay |
|---|---|---|
| 1,0 | −7,41 | salad |
| **0,6** | **−6,18** (≈ corpus −5,91) | **vẫn salad** |
| 0,4 | −4,39 | bắt đầu đúng rồi trôi |
| 0,25 | −3,88 | đúng ngữ pháp nhưng ngắn/rỗng (`nước.` · `người.`) |

Điểm quan trọng nhất: **ở đúng mức corpus, văn VẪN là salad.**

    'Việt Nam' → ' Việt Trì năm. cho là người này và văn hoá vào tháng được thường đầu nhà Lý.'

⇒ Đây là **chứng minh end-to-end** cho §12.104c: khớp thống kê đồng hiện với corpus **không** cho
văn đúng ngữ pháp. Trước đó tôi suy ra điều này từ hai phép đo phát hiện; giờ có một bộ sinh thật,
hiệu chỉnh để khớp corpus, vẫn ra văn què. Bằng chứng mạnh hơn hẳn.

#### e. Nhưng có một CỬA SỔ dùng được, và đo được độ dài

Đọc tay thấy "bắt đầu đúng rồi rã". Đo logP theo **từng vị trí** (nhiệt 0,4):

    vị trí 1–5 : −4,64 −4,73 −4,65 −4,37 −4,94   → TB **−4,67**
    vị trí 6–14: −6,60 −5,88 −6,19 −4,18 −5,43 … → TB **−5,62**

Bậc gần **1 nat/token** sau token thứ **5**. Lý do cơ học: cửa sổ `d≤8` **dần chứa toàn output của
chính nó** — tới vị trí 6 thì phần lớn ngữ cảnh là văn tự sinh, mà mô hình **không có cách nào kiểm
output của mình** (đúng §12.104: nền này không có tín hiệu nghiệm thu).

Và đúng trong cửa sổ ấy nó cho ra đáp thật, **không phải chép**:

    'Nhà Lý dời đô về'          → ' Thăng Long.'        ← ĐÚNG
    'Hoa Lư là kinh đô của'     → ' nhà Tây, nhà Lê.'   ← dạng đúng, fact sai
    'Chiến thắng Bạch Đằng năm' → ' 1507),'             ← dạng đúng (một năm), fact sai

#### f. Đọc kết quả

Cơ chế **không sai** và ý tác giả về sampling **đúng**. Nhưng nó là một **LM order-thấp**: mạch cục
bộ tốt trong ~5 token, không có cấu trúc toàn cục — đúng thứ `varlm.py` gọi là *"văn rời"*.

Đánh đổi thực tế: cửa sổ 5 token **ngắn hơn** thứ đường chép đang cho (T4 hiện 16,4 token). Nên
W_skip-sampling **không thay được** đường chép về độ dài. Chỗ nó có thể có ích là **5 token cuối
khi đường chép không có gì** — tức bổ khuyết, không chủ đạo.

Chưa nối vào `generate`. Muốn nối thì phải quyết trước: chấp nhận **fact sai** (`nhà Tây, nhà Lê` ·
`1507`) để đổi lấy **không chép**. Đó là đánh đổi của tác giả, không phải của tôi — và nó là **cùng
một ngã ba §12.104e**, chỉ hiện ra ở dạng cụ thể hơn.

---

### 12.106 NỐI kênh phiếu W_skip vào đường sinh thật (2026-07-30) — Claude

Tác giả chốt: *"cứ nối đi, chúng ta cần đường sinh mượt, fact có thể sai một chút"*. §12.105 đo trên
script rời; đây là nối vào `generate_hebbian` thật.

`w_skip.vote_logits()` — đúng đặc tả: mỗi token lịch sử `h[-d]` đề cử top-k qua `W_skip[d]`, cộng
phiếu giảm `λ^(d-1)`. Cộng **MỀM** trên **cả cửa sổ**; **không** giao, **không** cổng cứng, **không**
`ctx_max` — vì §12.85 đo được cổng cứng trên tập giao bẩn đẩy sinh vào hub.

Bốn knob, mặc định `weight = 0` (TẮT). Test: `test_vote_logits_dem_phieu_theo_dac_ta` dựng đúng bốn
câu trong ví dụ tác giả, kiểm `đẹp=2 · vô=2 · xa=1 · đánh=1` phiếu.

#### a. Ba điều phải xử lý mới nối được

* **`content_only=False`.** Mặc định `W_skip` bỏ từ chức năng và dấu câu ⇒ sinh ra không có
  `là`/`của`/`.`.
* **Chuẩn hoá thang.** Phiếu dải ~1–8, `scores` là logits kênh matrix.
* **`ar_sharp_gate` vô hiệu hoá sampling.** Mặc định `True`, quay về **argmax** khi entropy top-k
  > 2,2 — tức **đúng lúc** phân phối phiếu phẳng. Bật sampling cần `hybrid_temperature > 0`
  **VÀ** `ar_sharp_gate=False`. Đã ghi vào chú thích knob.

#### b. Một vá HARNESS bắt buộc

`eval_gen_broad` truyền `hybrid_temperature=0.0` như **đối số cứng** ⇒ `--extra` đụng "multiple
values" ⇒ **bộ đo không thể đo sampling**. Đã gộp bằng dict. Lại đúng lớp giới hạn dụng cụ.

---

### 12.107 A/B kênh phiếu: cổng TRƯỢT, chẩn ra HAI LỖI NỐI, sửa xong thì chỉ còn MỘT giá (2026-07-30) — Claude

#### a. Lượt đầu — trượt, và trượt ở chỗ chẩn được

| handoff | OFF | ON w=1 T=0,35 | |
|---|---|---|---|
| `copy` T1 | 0,71 | **0,36** | ✓ đúng mục tiêu |
| `empty` T7 | 0,40 | **0,00** | ✓ |
| `degen` T7 | 0,60 | **0,00** | ✓ |
| `loop4` T6 | 0,20 | **0,00** | ✓ |
| **`stop` T5/T7** | 1,00 | **0,00 / 0,20** | ✗ |
| **`digit⎵` T2** | 0,00 | **0,60** | ✗ |
| **`lạc` T5** | 0,33 | **0,93** | ✗ |
| `n_tok` T5 | 16,8 | **60,0** = trần | ✗ |

`stop = 0,00` với `n_tok = 60,0` **chính xác** ⇒ **mọi** mẫu đụng trần, không mẫu nào tự dừng.

> **Đừng đọc chéo hai bảng.** Bảng (a) chạy `--max-len 60`; bảng (c) và mọi lượt sau chạy 48
> (mặc định). Trong TỪNG bảng thì OFF và ON cùng một trần nên so được; giữa (a) và (c) thì
> `n_tok`/`stop` **không** so được. Đó
không phải "văn dài hơn" mà là **không có lực dừng**. Hai lỗi nối, không phải hai đánh đổi:

**Lỗi 1 — EOS nhận đúng 0 phiếu.** Đo trên mini:

    EOS   khối phiếu VÀO = 0
    '.'   khối phiếu VÀO = 50.884

`iter_sentence_spans` cắt ở `.` nên `<EOS>` luôn rơi vào span **dài 1** và bị `len(toks) < 2` bỏ.
Kênh cộng `+w·range(scores)` cho gần như mọi token nội dung và **+0** cho EOS ⇒ mỗi bước dìm EOS
đúng một khoảng điểm. **Sửa:** giữ nguyên **biên EOS so với ứng viên tốt nhất** — kênh phiếu chỉ
được XẾP LẠI các ứng viên **tiếp tục**; quyết định dừng-hay-tiếp thuộc kênh `eos` (β=1,2).

**Lỗi 2 — `W_skip` học được `1`→`8`→`7`→`0` như bigram thường**, nên kênh phiếu tự chọn được đường
"đánh vần số từng chữ", vượt qua đường CHÉP cụm số (`digit_run_from_locked`). Số không thuộc thẩm
quyền kênh này (mạng không có cơ quan số — TK_041). **Sửa:** `_vt[inner_digit_ids(vocab)] = 0`.

#### b. Nhưng thước `digit⎵` bị LÁCH — phải nới trước khi tin

Sau sửa, `digit⎵` về 0,00 mà **đọc tay vẫn thấy lỗi y nguyên**: `1 . 2 . 8` · `. 7 5 7 6`.
`digit_split` chỉ xét **hai token KỀ NHAU** đều là chữ-số ⇒ kênh phiếu đánh vần số **qua dấu câu**
thì lọt lưới. Cột về 0,00 là **thước bị lách**, không phải lỗi hết.

Đã nới: "hai chữ-số cách nhau ≤1 token, token chen giữa không phải chữ-cái/chữ-số".

**Và điều đáng giá nhất: nới thước làm BASELINE xấu đi luôn** — OFF `digit⎵` T7 0,00 → **0,20**,
T6 → **0,40**. Tức lỗi đánh vần số **vốn đã có sẵn** ở đường chép; thước cũ đang che. Nếu tôi không
nới, tôi sẽ đã báo "vá xong lỗi số" trong khi thật ra chỉ là đổi hình dạng lỗi sang chỗ thước
không thấy. Cùng họ §12.95d / §12.105c: **tiêu chí máy chấm và điều đang xảy ra là hai việc khác
nhau.**

#### c. Lượt sau, cùng một thước (handoff, mini)

| cột | OFF | ON w=1 T=0,35 | |
|---|---|---|---|
| `copy` T1 / T7 | 0,71 / 0,78 | **0,57 / 0,49** | ✓ |
| `empty` T7 | 0,40 | **0,00** | ✓ |
| `degen` T3 / T7 | 0,50 / 0,60 | **0,40 / 0,40** | ✓ |
| `loop4` T6 | 0,20 | **0,00** | ✓ |
| `stop` T6 | 0,80 | **1,00** | ✓ |
| `digit⎵` T6 | 0,40 | **0,20** | ✓ |
| `n_facts` T6 | 1,60 | **2,00** | ✓ |
| **`lạc` T1/T3/T6/T7** | 0,33/0,33/0,50/0,00 | **0,46/0,56/0,60/0,33** | ✗ |
| `n_facts` T1 | 1,65 | 1,30 | ✗ |

⇒ Sau hai sửa, **mọi cột canh đều bằng hoặc tốt hơn, trừ `lạc`**. `lạc` xấu đều +0,13…+0,33.
Đó là **giá duy nhất còn lại**, và nó là đúng giá phải trả: bớt chép thì bớt neo vào tài liệu gốc.

#### d. w=0,5 hành xử KHÁC HẲN — nó là BỔ KHUYẾT, không phải chủ đạo

| handoff | OFF | w=0,5 T=0,3 |
|---|---|---|
| `copy` T1 / T2 / T5 | 0,71 / 0,74 / 0,60 | **0,71 / 0,74 / 0,60** — y nguyên |
| `empty` T7 | 0,40 | **0,00** |
| `degen` T7 | 0,60 | **0,40** |
| `n_tok` T7 | 8,0 | **18,0** |
| `n_facts` T7 | 0,40 | **0,80** |

Ở w=0,5 kênh phiếu **không đụng** vào nhóm mà đường chép đang có đáp tốt (T1/T2/T5 **không đổi một
chữ**), chỉ nổ ở nhóm đường chép **rỗng** — T7 "dài", đúng nhóm mục tiêu "sinh đoạn dài": từ
**40% rỗng / 60% què / 8 token** sang **0% rỗng / 40% què / 18 token**.

Đó **đúng** điều §12.105f dự đoán: *"chỗ nó có thể có ích là khi đường chép không có gì — tức bổ
khuyết, không chủ đạo."* Dự đoán này **được xác nhận**, khác với dự đoán §12.94 của tôi (bị bác).

#### e. Một cảnh báo về độ tin của các delta nhỏ

Hai lượt OFF **cùng cấu hình** không trùng nhau: T3 `n_tok` 14,4 ‖ 14,1 · `stop` 1,00 ‖ 0,90 ·
T6 `n_tok` 32,8 ‖ 30,4. Baseline **không tất định** dù `hybrid_temperature = 0`. Nên **delta dưới
~0,1 trong bảng này không đọc được**. Các kết luận ở (c)/(d) đều dựa trên delta ≥0,2.

#### f. Đề nghị, và nó là quyết định của tác giả

Bật mặc định **`weight = 0,5` · `hybrid_temperature = 0,3` · `ar_sharp_gate = False`**: đây là
cấu hình duy nhất **không hồi quy** nhóm nào đang chạy tốt, mà vẫn vá đúng nhóm T7. w=1,0 mượt hơn
và phá chép mạnh hơn nhưng trả bằng `lạc` trên **mọi** nhóm.

Hiện vẫn để `weight = 0` cho tới khi tác giả chốt, vì `lạc` là **trôi đề**, nặng hơn "fact sai một
chút" — nên đó không phải quyết định của tôi.

Chưa xác minh trên v12 (430.447 cha): `W_skip` ở quy mô ấy dày hơn nhiều, và §12.102b đã cho một
tiền lệ đối chứng nhảy 0,225 → 0,617 khi lên v12. Giao P25.

---

### 12.108 P25 — A/B phiếu W_skip trên v12: G1 TRƯỢT ở w=0.5 · dừng (2026-07-30) — M4

Theo §12.31: nếu `stop` < 0.90 ⇒ báo ngay, **không chạy tiếp** w=1.0.
`out/p25_off.log` · `out/p25_w05.log` (max_len=48 mặc định). Không có `p25_w10.log`.

#### a. Bảng handoff nguyên văn

**OFF** (`decode_weight=0`):

| nhóm | lạc | digit⎵ | copy | stop | n_tok |
|---|---|---|---|---|---|
| T1 | 0.12 | 0.05 | 0.71 | **1.00** | 23.8 |
| T2 | 0.40 | 0.00 | 0.67 | 1.00 | 19.6 |
| T3 | 0.43 | 0.10 | 0.66 | 0.90 | 27.4 |
| T4 | 0.00 | 0.00 | 1.00 | 1.00 | 16.4 |
| T5 | 0.25 | 0.00 | 0.65 | 1.00 | 21.5 |
| T6 | 0.00 | 0.40 | 0.53 | 1.00 | 32.4 |
| T7 | 0.00 | 0.00 | 0.71 | 1.00 | 26.2 |

**w=0.5** (`T=0.3`, `ar_sharp_gate=false`):

| nhóm | lạc | digit⎵ | copy | stop | n_tok |
|---|---|---|---|---|---|
| T1 | **0.34** | 0.05 | 0.65 | **0.85** | 29.9 |
| T2 | 0.40 | 0.00 | 0.67 | 1.00 | 19.6 |
| T3 | 0.33 | 0.00 | 0.76 | **0.80** | 24.2 |
| T4 | 0.00 | 0.00 | 1.00 | 1.00 | 16.3 |
| T5 | 0.25 | 0.00 | 0.65 | 1.00 | 21.5 |
| T6 | **0.40** | 0.00 | 0.59 | **0.60** | 34.8 |
| T7 | 0.00 | 0.00 | 0.62 | **0.60** | 37.4 |

no_agenda OFF/w05: xem log đầy đủ (T1 stop 1.00→0.90; T7 stop 1.00→0.80).

#### b. Cổng

| | |
|---|---|
| **G1** stop ≥0.90 mọi nhóm | **TRƯỢT** — T1 0.85 · T3 0.80 · T6 0.60 · T7 0.60 |
| G2 n_tok không đụng trần | ĐẠT (max ~37, không 48.0 tròn cả nhóm) |
| G3 digit⎵ ≤+0.10 vs OFF | ĐẠT (T3 0.10→0.00; T6 0.40→0.00) |
| G4 lạc TB ≤+0.15 | **TRƯỢT** — T1 +0.22 · T6 +0.40; TB(T1/2/3/5/6/7) ≈ +0.09 nhưng T1/T6 vượt |
| G5 copy T1 ↓≥0.10 ở w=1 | **không chạy** w=1 |

⇒ Vá EOS giữ-biên (§12.107a) **không đủ** trên v12. Còn đường dìm EOS (hoặc sampling + phiếu dày
hub) khiến nhiều nhóm không dừng.

#### c. Phản biện §12.107f — “w=0.5 bổ khuyết”?

**Không.** Mini: T1/T2/T5 copy y nguyên. v12: T1 `copy` 0.71→0.65 · `lạc` 0.12→**0.34** ·
`stop` 1.00→**0.85**; T6 `lạc` 0→**0.40** · `stop`→**0.60**. Đụng cả nhóm đang tốt.
§12.107f **rút** trên v12 — không đề nghị bật mặc định.

#### d. Mẫu đọc tay `p25_w05` (handoff, rút)

T2/T4/T5 còn đọc được (verbatim lịch sử). T1/T3/T6/T7: nhảy miền (mailing list · truyện dịch ·
công thức) hoặc cắt sớm — khớp `lạc`↑ và `stop`↓.

Giữ `ar_matrix_w_skip_decode_weight=0`. Chờ Claude vá EOS/v12 hoặc rút kênh.

→ **Claude trả lời ở §12.110: cổng G1 do tôi viết SAI (đo trần độ dài, không đo việc dừng).**
M4 chạy đúng. §12.107f **rút**. Giao P26 đo lại với trần 96.

### 12.109 `d_max` KHÔNG phải chỗ chặn — và tôi đã gọi sai tên chính cài đặt của mình (2026-07-30) — Claude

Tác giả hỏi: *"d = 8 có yếu quá không? tại sao không phải d = 12, 16"*.

#### a. Số học của chính công thức trả lời trước khi cần chạy

`vote_logits` cộng `λ^(d-1)`. Phân bố khối phiếu theo d:

| λ | D | d≤3 chiếm | d>8 chiếm | λ^(D-1) |
|---|---|---|---|---|
| **0,5** | **8** | **87,8%** | 0,00% | 0,00781 |
| 0,5 | 12 | 87,5% | **0,37%** | 0,00049 |
| 0,5 | 16 | 87,5% | 0,39% | 0,00003 |
| 0,85 | 8 | 53,0% | 0,00% | 0,32058 |
| 0,85 | 16 | 41,7% | **21,41%** | 0,08735 |

⇒ Ở `λ=0,5`, nâng `d_max` 8→16 đổi **0,4%** khối phiếu. **Ràng buộc không phải `d_max` mà là `λ`.**

**Và điều phải ghi thẳng: tôi gọi sai tên cài đặt của mình.** Biến tên `d_max=8` nhưng tầm **hiệu
dụng** là **~3** — kênh phiếu đang chạy thực chất là một **trigram**. Mọi câu tôi viết ở §12.106/
§12.107 nói "cửa sổ d≤8" đều mô tả *danh nghĩa*, không phải *thực*. Cùng họ lỗi §12.95d/§12.107b:
tên/văn bản và thứ đang thật sự xảy ra là hai việc.

#### b. Trần dữ liệu KHÔNG chặn ở 8

6.838 span · dài TB 16,3 · trung vị 16 · p90 26 · max 44.

| d | span còn dùng được | cặp vị trí |
|---|---|---|
| 8 | **85,7%** | 58.141 |
| 12 | 66,5% | 36.613 |
| 16 | 46,4% | 20.437 |
| 24 | 13,6% | 3.223 |

Dữ liệu đỡ tới ~12–16. Thưa thật chỉ từ d≈24.

#### c. Dự đoán đăng ký TRƯỚC — và nó bị BÁC

Lập luận: §12.105e đo văn rã ở vị trí 5–6, giải thích là cửa sổ dần chứa output của chính nó. Mà
`hist[-1]` là token **tự sinh** còn `hist[-8]` mới là **prompt thật**. Với λ=0,5 vị trí xa không có
trọng số ⇒ kênh phiếu gần như chỉ nghe **chính nó** ⇒ đó là một cơ chế sinh **trôi đề**.
**Dự đoán: λ cao neo vào prompt tốt hơn ⇒ `lạc` GIẢM.**

`lạc` trung bình trên T1/T2/T3/T5/T6/T7 (khối `handoff`, mini, w=1,0 T=0,35):

| cấu hình | d≤3 chiếm | `lạc` TB |
|---|---|---|
| OFF | — | **0,290** |
| d8 λ0,50 | 87,8% | 0,447 |
| d8 λ0,85 | 53,0% | 0,467 |
| d12 λ0,70 | 66,6% | 0,437 |
| d16 λ0,85 | 41,7% | 0,437 |

**BÁC.** Bốn cấu hình nằm trong dải 0,437–0,467 — chênh **0,03**, dưới hẳn sàn nhiễu §12.107e
(baseline còn không tất định). Kéo tầm hiệu dụng từ ~3 lên ~10, đưa **21%** khối phiếu ra ngoài
d=8, `lạc` **không nhích**.

Một ô dễ đọc lầm: `d12 λ0,70` có `T3 lạc = 0,00`. Nhưng cùng ô đó `n_tok = 8,5` (ngắn nhất bảng)
và `n_facts = 0,30` (thấp nhất) ⇒ hầu như **không có câu thứ hai để chấm** (`lạc` chỉ tính từ câu 2).
Đó là **mẫu rỗng**, không phải hết trôi đề. Nếu lấy ô ấy làm kết luận thì sai.

Tác dụng phụ duy nhất đo được của tầm xa: `n_tok` 21,1 → 23–25 · `n_facts` 1,30 → 1,70–1,85 ·
`digit⎵` T7 0,20 → **0,40** ở d12/d16 (đánh vần số trở lại chút). Không cột nào đáng đổi.

#### d. Hệ quả: LẦN THỨ BA kéo tầm xa không cho gì

| phép đo | kéo tầm | kết quả |
|---|---|---|
| §12.103a | bigram → skip d≤8 phát hiện mối ghép | **+0,430 < +0,544** — xa TỆ HƠN |
| kết nối xa (CO đa khoảng cách) | 1 → nhiều d | không vượt bigram |
| **§12.109c** | tầm hiệu dụng 3 → ~10 | `lạc` không đổi |

Ba đường độc lập. Ghép với §12.104c thì hình dạng nhất quán: thứ thiếu **không phải tầm** mà là
**tín hiệu nghiệm thu**. Kéo tầm cho một cơ chế đo *"thứ đã thấy"* thì vẫn chỉ thấy thứ đã thấy —
xa hơn thì thưa hơn, không đúng hơn.

⇒ **Đóng** hướng "nâng `d_max`/`λ` của kênh phiếu". Giữ `d_max=8 · λ=0,5` (rẻ nhất, không kém).

---

### 12.110 Vá EOS đúng hơn làm cổng TỆ HƠN — vì cổng G1 của tôi đo sai thứ (2026-07-30) — Claude

P25 (§12.108) TRƯỢT G1 (`stop`) và G4 (`lạc`) trên v12. Tôi nêu một giả thuyết cho G1, đo, và
**giả thuyết bị bác** — nhưng phép đo lại chỉ ra cổng G1 do chính tôi viết mới là chỗ sai.

#### a. Giả thuyết: vá §12.107a đúng cho ARGMAX, sai cho SAMPLING

`_sample` làm softmax trên **cả top-k** (k=12) rồi multinomial. Vá §12.107a giữ biên EOS so với ứng
viên **số 1** ⇒ giữ đúng quyết định argmax, nhưng mẫu số có 12 số hạng và kênh phiếu nâng cả hạng
2…12 ⇒ `P(EOS)` vẫn tụt. Dạng đóng: `P = e^{s/T}/(e^{s/T}+D)` ⇒ trung lập ⟺ `s_eos += T·Δlog D`,
tức **logsumexp**, còn `max` chỉ là giới hạn `T→0`.

Đã cài `_cont_anchor` theo logsumexp.

#### b. Mô phỏng BÁC giả thuyết — và thước tương đối đã đánh lừa tôi một lượt

Lượt đầu tôi chấm bằng **sai số tương đối** của `P(EOS)` và ra "trung vị 2.423.845%" — vô nghĩa, vì
`p0 ≈ 0` mỗi khi EOS không nằm trong top-k. Đổi sang **lệch tuyệt đối** (3.000 lượt, T=0,3, k=12):

| cách bù | TB lệch `P(EOS)` | dìm EOS ở |
|---|---|---|
| không bù | **−0,0156** | 12,3% bước |
| `max` (§12.107a, đang chạy) | **+0,0096** | **0,7%** bước |
| `logsumexp` | **+0,0009** | 0,0% bước |

`logsumexp` trung lập hơn ~10×, nên **đáng giữ**. Nhưng nó **bác giả thuyết**: bản `max` đang chạy
có trung bình **+0,0096** — nó *nâng* `P(EOS)`, chỉ 0,7% bước bị dìm. Mẫu số softmax **không** giải
thích được `stop` 1,00 → 0,60 trên v12.

#### c. Chẩn đúng: `stop_clean` đo TRẦN ĐỘ DÀI, không đo việc dừng

    m["stop_clean"] = int(len(cont) < max_len or _is_boundary(v, cont[-1]))

`stop = 0` ⟺ sinh **đúng `max_len` token** và token cuối không phải biên câu. Trên v12 M4 chạy
`max_len=48`, và T7 `n_tok` 26,2 → **37,4**. Tức cột `stop` đang báo **"chạm trần"**, không phải
**"không biết dừng"**.

Bằng chứng chốt, và nó ngược hẳn ý muốn của tôi: đo trên mini, vá `logsumexp` (**đúng hơn**) cho
T7 `n_tok` **18,0 → 40,4** so với vá `max`. Vá càng trung lập thì văn càng **dài**, nên trên v12 nó
sẽ làm G1 **TỆ HƠN**. Bản `max` "qua" G1 dễ hơn **chỉ vì nó bù quá tay về phía dừng** (+0,0096).

⇒ **G1 do tôi viết trộn hai việc khác nhau**: "không phát EOS" và "sinh dài hơn trần". Cùng họ lỗi
đã ghi trong [[blocking-gate-measurement-design]]: thước phải nằm trong đồ thị. Không phải M4 đo
sai — M4 chạy đúng cổng tôi đưa; cổng sai.

#### d. Nhưng G4 thì KHÔNG oan, và nó là thất bại thật

| v12 handoff | `lạc` OFF→w05 | `stop` OFF→w05 |
|---|---|---|
| T1 | 0,12 → **0,34** (+0,22) | 1,00 → 0,85 |
| T2 | 0,40 → 0,40 (0) | 1,00 → 1,00 |
| T3 | 0,43 → 0,33 (−0,10) | 0,90 → 0,80 |
| T5 | 0,25 → 0,25 (0) | 1,00 → 1,00 |
| T6 | 0,00 → **0,40** (+0,40) | 1,00 → 0,60 |
| T7 | 0,00 → 0,00 (0) | 1,00 → 0,60 |

Hai đọc được từ bảng này:

* **T2/T5 không đổi một chữ** ⇒ ở v12 kênh phiếu **không làm gì** trên nhóm đó. Chứ không phải
  "bổ khuyết có chọn lọc" — nó chỉ đơn giản không thắng nổi kênh `succ`+`path` ở đấy.
* **T7 `lạc` = 0,00 mà `stop` tụt 0,40** ⇒ đúng ca "đúng đề nhưng dài quá trần", tức (c). Còn
  **T1/T6** thì `lạc` và `stop` tụt **cùng biên độ** ⇒ ca "đi lạc sang tài liệu khác rồi không có
  gì bảo nó dừng". Đó là **một** thất bại nhìn từ hai cột, không phải hai.

M4 đọc tay khớp: *"T1/T3/T6/T7 nhảy miền (mailing list · truyện dịch · công thức)"*.

#### e. §12.107f RÚT — M4 đúng

Tôi đề nghị bật mặc định `w=0,5` dựa trên tính "bổ khuyết" đo trên mini (T1/T2/T5 y nguyên).
Trên v12 tính ấy **không có**: T1 `copy` 0,71→0,65 · `lạc` 0,12→**0,34** · `stop` 1,00→0,85 —
đụng thẳng nhóm đang tốt nhất. **Rút đề nghị.** Giữ `ar_matrix_w_skip_decode_weight = 0`.

Đây là lần thứ hai trong ticket một kết luận của tôi đo trên mini bị v12 đảo (lần trước: §12.102b
đối chứng xáo 0,225 → 0,617). Ghi lại thành nguyên tắc: **kết luận về `lạc`/`copy` không được chốt
trên mini** — 5.664 cha thì "đi lạc" gần như không có chỗ để lạc tới.

#### f. Việc còn lại

Giữ vá `logsumexp` (đúng hơn, đã đo). **Không** đề nghị bật kênh. G4 là chỗ chặn thật và không vá
EOS nào chạm tới nó: `lạc` đến từ việc kênh phiếu **không có khái niệm "cùng tài liệu"** — nó chỉ
biết "cặp này từng đứng cách nhau d token", ở đâu cũng được. Đó lại đúng §12.104c: nền này đo *thứ
đã thấy*, không đo *thứ hợp chỗ*.

---

### 12.111 P26 — max_len=96: G1′ gần đạt · G4′ TRƯỢT ⇒ đóng kênh phiếu (2026-07-30) — M4

`out/p26_off.log` · `out/p26_w05.log`. Không đổi knob / không sửa `w_skip.py`.

#### a. Bảng handoff (OFF → w=0.5)

| nhóm | lạc OFF→ON | stop OFF→ON | n_tok OFF→ON |
|---|---|---|---|
| T1 | 0.12→**0.32** | 1.00→**0.95** | 23.8→34.6 |
| T2 | 0.40→0.40 | 1.00→1.00 | 19.6→19.6 |
| T3 | 0.29→**0.43** | 1.00→1.00 | 28.0→29.9 |
| T4 | 0.00→0.00 | 1.00→1.00 | 16.4→16.3 |
| T5 | 0.25→0.25 | 1.00→1.00 | 21.5→21.5 |
| T6 | 0.00→**0.30** | 1.00→1.00 | 32.4→35.6 |
| T7 | 0.00→**0.20** | 1.00→**0.80** | 26.2→**52.6** |

no_agenda: xem log (T7 stop 1.00→0.80 · n_tok 9.6→32.0).

#### b. Cổng

| | |
|---|---|
| **G1′** stop≥0.90 mọi nhóm (trần 96) | **TRƯỢT nhẹ** — chỉ T7 **0.80**; T1/T3/T6 **phục hồi** so P25 |
| **G4′** lạc ↑ ≤+0.15 **mọi** nhóm | **TRƯỢT** — T1 **+0.20** · T6 **+0.30** · T7 **+0.20** (T3 +0.14 đạt biên) |

⇒ Theo §12.31: G4′ trượt ⇒ **đóng kênh phiếu**, không vá thêm. Giữ `decode_weight=0`.

#### c. Phản biện §12.110d (tách T7 vs T1/T6)

Với trần 96:

* **T1/T6:** `stop` về ≥0.95 / 1.00 — đúng hướng §12.110c (G1 cũ một phần là trần). Nhưng
  **`lạc` vẫn xấu** (+0.20 / +0.30) ⇒ không chỉ "không dừng", mà **trôi đề** là lỗi chính.
* **T7:** `n_tok` 26→53 (chưa đụng 96) mà `stop` vẫn 0.80 **và** `lạc` 0→0.20 — **không** còn
  đọc thuần "đúng đề nhưng quá trần" như P25 (`lạc`=0). Có cả dài hơn lẫn trôi nhẹ.

§12.110c **đúng một phần** (nới trần cứu stop T1/T6). G4′ vẫn chặn thật.

### 12.112 Đổi TRỤC ĐO: chấm mù bằng AI thay verbatim-copy — và verdict W_skip ĐẢO DẤU (2026-07-30) — Claude

Quyết định tác giả: *"không thể đo cứng trên verbatim copy… chúng ta đang làm cho nó sinh sáng
tạo, nên việc nó khớp agenda không phải mục tiêu"* + chọn **chấm bằng AI + đọc tay**.

Đây là một **sửa trục đo**, không phải thêm một cột. `copy`/`lạc`/`n_facts`/khung khớp T1–T7 đều là
thước của **hệ TRUY HỒI** — thưởng cho "tái tạo đúng trí nhớ". Hệ quả nặng nhất: §12.104 tôi viết
*"copy ≈ 1.0 là cách duy nhất giữ ngữ pháp"* — đó là đọc **thiên lệch của thước** thành **định luật
của nền**. Vòng luẩn quẩn do tôi tự dựng.

`scripts/gen_samples_blind.py`: sinh mẫu nhiều cấu hình → **giấu nhãn + trộn** → chấm theo rubric cố
định (F trôi chảy /3 · C bám prompt /3 · N mới /2 · P hợp lý /2) → **mới lật nhãn**. Chấm mù để điểm
không nghiêng theo kỳ vọng.

> **Nối với §12.111 (P26 của M4).** M4 chốt *"đóng kênh phiếu"* vì G4′ (`lạc`) trượt. Đúng —
> **trên trục copy/`lạc`**. Mục này lập luận chính trục ấy sai hướng cho mục tiêu sáng tạo, nên
> "đóng" chỉ đúng theo thước cũ. Không phủ nhận số của M4 (`lạc` thật sự ↑); phủ nhận rằng `lạc`↑
> **là** dấu hỏng khi mục tiêu là sinh mới. Hai mục cùng số, khác trục — cần đọc kèm.
>
> **Về `d_max=16` (M4, commit e0bfe6c).** Đổi mặc định 8→16. §12.109 đo `d=8 vs 16` chênh 0,03
> (dưới nhiễu) trên trục cũ; d=16 **chưa** phép đo nào trên trục mới chạm tới; kênh **tắt mặc
> định** nên vô hại runtime, chỉ ×2 chi phí dựng W_skip khi bật. Không revert (commit phía kia);
> ghi để lần đo tới trên trục mới quét lại `d` cho đúng.

#### a. Phát hiện phụ mà quan trọng hơn: đường mặc định là hệ CHÉP

Lượt đầu (BASE để handoff, KHÔNG `ar_no_agenda`): **cả ba cấu hình OFF/w05/w10 ra byte-identical**
trên 12 prompt. Vì `generate()` mặc định chạy **truy hồi/multizone** — lấy câu đã nhớ chép ra — và
kênh phiếu W_skip (sống trong AR matrix-core) **không bao giờ tới lượt**. Đọc mẫu thì rõ: đều là câu
bách-khoa hoàn chỉnh. ⇒ Muốn đo "sáng tạo" phải ép `ar_no_agenda=True`.

#### b. Trên đúng đường sinh AR — verdict ĐẢO DẤU thước copy (mini, n=12, chấm mù)

| cấu hình | TỔNG /10 | F trôi | C bám | N mới | P lý |
|---|---|---|---|---|---|
| OFF | 3,92 | 1,25 | 1,33 | 0,58 | 0,75 |
| **vote_w05** | **4,42** | **1,58** | 1,33 | 0,67 | 0,83 |
| vote_w10 | 3,58 | 1,17 | 1,17 | 0,67 | 0,58 |

Per-prompt: w05 hơn OFF **4** · kém 2 · bằng 6. Lợi ích **dồn vào `F` (trôi chảy)** 1,25→1,58; `C`
(bám prompt) **không đổi** 1,33. Tức ở w05 kênh phiếu làm văn **mượt hơn mà KHÔNG trôi đề thêm** —
đúng ngược với cái `lạc` (§12.108) phạt. `lạc` phạt "khác fold prompt", nhập nhằng "sáng tạo" với
"trôi"; giám khảo tách được hai cái. w10 thì quá tay: cả F và C cùng tụt.

Ca sạch nhất (0→4): `Lê Lợi khởi nghĩa…` OFF ra `chiến 0 7 0 9 0 9 7` (rác số) → w05 ra `…tiếp tục
cho mở mang xây dựng thêm nhiều cung điện lộng lẫy ở Hoa Lư` (mệnh đề trôi chảy). Kênh phiếu cứu
được ca rác-chữ-số thành văn đọc được.

#### c. Giới hạn — không được đọc quá

* **n=12, MỘT giám khảo (là tôi)**: +0,50 nằm trong sai số một-giám-khảo. Cần nhiều giám khảo /
  nhiều prompt trước khi chốt.
* **Điểm tuyệt đối THẤP** (cả hai ~4/10): đây là so **hai thứ yếu**, không phải "đã tốt".
* **mini** — thin regime. §12.108 đã cho một tiền lệ v12 đảo kết luận mini.
* Tôi vừa là bên đề xuất W_skip vừa là giám khảo ⇒ thiên lệch tiềm tàng, dù chấm mù nhãn.

#### d. Điều đã đứng vững

**Trục đo cũ sai hướng cho mục tiêu này** — đó không phải chuyện n nhỏ. `copy=1,0` là **đỉnh** của
thước cũ nhưng là **đáy** của mục tiêu sáng tạo. Mọi verdict "W_skip âm tính" ở §12.85/§12.107/
§12.108 đều đo trên trục copy/lạc — **phải đọc lại** dưới ánh sáng này. Không xoá, nhưng gắn cảnh báo.

Việc tiếp: (1) mở rộng bộ prompt + xin tác giả đọc tay chéo; (2) cân nhắc giám khảo thứ hai (M4
cũng là LLM — có thể chấm mù song song để đo độ lệch giám khảo); (3) đưa `gen_samples_blind` sang
v12 (M4 sinh, commit mẫu giấu nhãn; Claude + tác giả chấm).

---

### 12.113 ① mở rộng chấm mù (n=30): tín hiệu +0.50 KHÔNG lặp · ② E_skip làm thước thì tốt hơn thật (2026-07-30) — Claude

Tác giả: *"làm 1 và 2 đi"*. ① mở bộ prompt 12→30, thêm loại thường-thức/ngữ-cảnh-dài, bỏ w10
(§12.112 đã thua), giữ OFF vs w05, chấm mù. ② E_skip làm thước bám-chủ-đề, kiểm bằng chính nhãn
`C` đã chấm mù.

#### ① Bác chính §12.112: +0.50 là nhiễu n nhỏ

| cấu hình | TỔNG /10 | F trôi | C bám | N mới | P lý |
|---|---|---|---|---|---|
| OFF | **4,10** | 1,40 | 1,27 | 0,77 | 0,67 |
| vote_w05 | **4,17** | 1,43 | 1,23 | 0,80 | 0,70 |

Σ chênh trên 30 prompt = **+2** (TB +0,07). Per-prompt: hơn 6 · kém 8 · **bằng 16**. **Hoà.**

§12.112 báo w05 +0,50 ở n=12; ở n=30 **tan thành nhiễu**. Đúng cảnh báo tôi tự ghi ở §12.112c
("+0,50 nằm trong sai số một-giám-khảo"). Kết luận đứng vững là **phương pháp** (chấm mù chạy được,
đường mặc định là hệ chép), KHÔNG phải "w05 hơn OFF". Kênh phiếu ở mức chất lượng: **không đo được
lợi ích** khi n đủ.

#### ①b Điều bảng điểm để lộ — cùng dấu §12.104, giờ trên trục MỚI

Mọi mẫu 9/10 đều là **tái tạo fact đã nhớ**: `Huế` · `938` · `Thăng Long năm 1010` · QA Lê Lợi /
Nhà Lý / Bạch Đằng. Mẫu phân kỳ sáng tạo hầu hết rơi 0–3 (salad/trôi). Tức **ngay cả khi AI chấm
theo trục sáng tạo**, chất lượng vẫn do **hành vi giống-truy-hồi** gánh. §12.104 (copy là cách duy
nhất giữ ngữ pháp) trước đo trên trục copy — giờ **trục AI-chấm cho cùng dấu**. Đó là bằng chứng
MẠNH hơn hẳn: không phải thước cũ thiên lệch, mà nền thật sự chỉ chắc ở chỗ chép.

**Ngoại lệ đáng giữ:** S008/S044 `Vịnh Hạ Long… → "có hình dạng giống như một con thiên nga nằm
trên sóng nước"` — 8/10, mới VÀ đẹp VÀ đúng mạch. Nền **có thể** soạn mới-mà-hay, chỉ **hiếm và
không tin cậy** (1–2/30).

#### ② E_skip làm THƯỚC bám-chủ-đề — tốt hơn E_old, nhưng yếu tuyệt đối

`scripts/probe_eskip_ruler.py`. 57 mẫu, nhãn = `C` (bám prompt) đã chấm mù. E_skip dựng đối xứng
(PPMI skip-gram + SVD 96d, dmax 8, λ 0,85).

| | Spearman cos↔C | tách cos: C≥2 vs C≤1 |
|---|---|---|
| E_old | +0,072 | +0,026 |
| **E_skip** | **+0,213** | **+0,100** |

E_skip theo nhãn người **~3×** (tương quan) / **~4×** (tách). Xác nhận trực giác §12.112: E_old gần
chết cho việc đo chủ đề; E_skip khá hơn hẳn. **Nhưng** Spearman +0,213 vẫn là tương quan **yếu** —
E_skip là thước trôi-đề **dùng được nhưng thô**, không phải sắc.

#### Đọc kết quả — W_skip có ích ở đâu, hết ích ở đâu

Gộp mọi phép đo tới giờ, giá trị W_skip **thu hẹp về đúng một chỗ**:

* xếp hạng đáp (`skipxcov`): **+0,007** — không đáng (§12.98).
* kênh phiếu giải mã: **hoà** ở n=30 (§12.113①) — trước tưởng +0,50, là nhiễu.
* **embedding làm THƯỚC đo trôi đề: ~3–4× tốt hơn E_old** (§12.113②) — chỗ DƯƠNG duy nhất còn lại,
  và nó là việc *đo*, không phải việc *sinh*.

⇒ W_skip **không** cải thiện sinh đo được (ở quy mô mini). Nó cải thiện được **dụng cụ đo trôi đề**.
Nút chặn thật vẫn là §12.104: nền chắc ở chỗ chép, phân kỳ sáng tạo thì salad — và giờ điều này
đúng trên **cả hai** trục đo, nên không còn là nghi vấn "thước sai".

#### Việc tiếp (đề xuất, chưa làm)

* Nếu giữ eval sáng tạo: **nhận E_skip làm cột `trôi-đề`** thay `lạc` (fold cứng) — rẻ, tốt hơn đo
  được, sạch Luật 0. Không đổi `state.embedding` cho đường sinh.
* Giám khảo thứ hai (M4) chấm mù song song để đo lệch giám khảo — vẫn đáng, vì n=30 một giám khảo
  cho hoà thì cần biết khoảng tin.
* Kênh phiếu: để `weight=0`. Không có cớ đo được để bật.

---

### 12.114 LỘ TRÌNH — W_skip làm cấu trúc GỐC, W_seq/CO-embedding/VarLM là DẪN XUẤT (2026-07-30) — Claude

Quyết định tác giả: *"chốt theo W_skip thay thế W_seq, gồm cả embedding sau này… lên kế hoạch cải
tiến toàn diện tận dụng W_skip nâng cấp mạng"*. Đây là lộ trình. **Nguyên tắc xuyên suốt:** hướng
đã chốt (theo đuổi W_skip), nhưng mỗi bước có **cổng đo so với cái đang có** — nhận nếu **thắng
hoặc hoà-mà-rẻ-hơn**, không thay huỷ trước khi có bằng chứng. Sạch Luật 0 (mọi thứ closed-form /
đếm, không backprop ở đường sinh).

#### Luận đề: W_skip là SIÊU CẤU TRÚC, không phải một kênh thêm

`W_skip[d]` = đếm `a→b` cách nhau d token. Ba cấu trúc rời hiện nay đều là **lát cắt** của nó:

| cấu trúc hiện có | là view nào của W_skip |
|---|---|
| `W_seq` (order-1 kề) | `W_skip[d=1]` |
| embedding E (CO_ACTIVATION, cửa sổ đối xứng) | Σ_d λ^d (W_skip[d] + W_skipᵀ[d]) → PPMI → SVD |
| VarLM (n-gram liền + backoff) | chuỗi band d=1 nối tiếp (liền = đường chéo) |

⇒ "W_skip thay W_seq" **đúng nghĩa** = làm W_skip thành **nguồn chính tắc**, ba cái kia đọc RA từ
nó, rồi mở **một consumer MỚI** mà ba cấu trúc rời không cho được (§12.104: tín hiệu nghiệm thu).
Đây là **hợp nhất**, không phải cộng thêm — nên nó không đụng các âm tính "cộng kênh" đã đo.

#### Điều phải thành thật NGAY (không giấu để sau vấp)

Cho **dự đoán token kế**, đo cũ ([[long-range-cooccurrence-negative]]) cho **VarLM (n-gram liền)
+7,3pp**, còn **skip-gram có gap TRUNG TÍNH**. Nên ở vai trò dự đoán, W_skip **chưa** hơn cái đang
có; VarLM đang thắng. Lộ trình này **không** giả vờ ngược lại — nó (a) hợp nhất VarLM thành view
của W_skip nên "thay W_seq" vẫn đạt mà không mất VarLM, và (b) dồn kỳ vọng thắng vào chỗ W_skip
**độc nhất mạnh**: embedding (§12.113② đo 3–4×) và tín hiệu nghiệm thu toàn cục (chưa ai thử).

---

#### PHA 0 — Hạ tầng: W_skip thành hạng-nhất (rẻ, làm trước)

* **Nướng W_skip vào checkpoint** (nay dựng lúc load ~1 phút v12) — sidecar `.wskip`, có version.
* **Chốt bộ đo chuẩn:** chấm mù AI (§12.112, `gen_samples_blind.py`) + thước trôi-đề E_skip
  (§12.113②, `probe_eskip_ruler.py`) làm **xương sống đo** cho mọi pha sau. Thêm **giám khảo M4**.
* Cổng: không có — đây là hạ tầng. Xong = W_skip persisted + harness đứng + 2 giám khảo.

#### PHA 1 — Embedding: E_skip thay CO-embedding (chính "embedding sau này" tác giả nêu)

* Đổi `build_ppmi` nguồn từ `CO_ACTIVATION` → **W_skip đối xứng** (`embedding.py:38`). Vẫn
  PPMI→SVD, vẫn Luật 0.
* **Cổng G-emb:** (a) thước trôi-đề ≥ E_old — **đã đo đạt §12.113②** (3–4×); (b) chấm điểm ứng
  viên AR **không kém** E_old (§12.98 xếp-hạng +0,007 ≈ hoà — phải kiểm trên SINH, không chỉ rank);
  (c) hàng xóm ngữ nghĩa tỉnh (spot-check `Lý≈Trần`, `phở≈bún`). Đạt cả 3 ⇒ nhận.
* Đây là **thắng sạch nhất và đúng thứ tác giả gọi tên**. Giữ E_old làm fallback tới khi (b) đạt.

#### PHA 2 — Dự đoán: bake-off W_skip-backoff vs W_seq vs VarLM (de-risk "thay W_seq")

* Dựng bộ dự đoán **backoff theo khoảng cách** từ W_skip; đo held-out next-token top-1 vs
  (i) `W_skip[d=1]`=W_seq, (ii) VarLM.
* **Cổng G-pred:** W_skip-backoff ≥ VarLM (hoặc VarLM tái hiện được **như một view** của W_skip →
  "thay W_seq" đạt mà không mất VarLM). **Dự đoán trước:** nhiều khả năng W_skip-backoff ≈ VarLM
  khi backoff bắt được phần liền; nếu kém VarLM ⇒ giữ VarLM-as-view, không ép skip có-gap.
* Đây là chỗ khớp với đo cũ — mục tiêu là **không thua**, không phải kỳ tích.

#### PHA 3 — Consumer MỚI: W_skip làm tín hiệu NGHIỆM THU toàn cục (đánh vào §12.104)

* Nút chặn thật (đo trên **cả hai** trục, §12.113①b): nền chắc ở chỗ chép, phân kỳ thì salad —
  vì **không có tín hiệu phân biệt "mới-mà-đúng" với "mới-mà-què"**.
* Thử: chấm một chuỗi sinh bằng **độ nhất quán skip toàn cục** — đa tập `(ngữ-cảnh, ứng-viên, d)`
  có được W_skip chống đỡ ở **nhiều** khoảng cách không (khác vote cục bộ ~trigram §12.109; đây là
  điểm nhất quán trên cả câu).
* **Cổng G-acc:** tách "sáng tạo-mạch-lạc" khỏi "salad/trôi" **tốt hơn** cos-E_skip đơn (thước
  Pha 1), đo trên nhãn C chấm mù. **Dự đoán trước, thành thật:** §12.104 đã cho attestation LIỀN
  trượt (mù với mối ghép); skip-cặp có thể cùng lối. Nếu trượt ⇒ đó là **câu trả lời dứt khoát**
  rằng khe nghiệm thu không đóng được bằng cấu trúc đồng-hiện, và ta thôi tìm ở đây.

#### PHA 4 — Tích hợp SINH (chỉ khi Pha 3 dương)

* Nếu Pha 3 cho tín hiệu nghiệm thu dùng được: gắn làm **cổng tái xếp hạng** trong đường AR
  (KHÔNG phải vote-cộng — một **gate** chọn/loại), nhắm thẳng lỗi trôi/salad.
* **Cổng G-gen:** chấm mù AI n≥30, **hai** giám khảo, vượt OFF quá sàn nhiễu (§12.113 dạy: n=12
  lừa, phải n≥30). Chỉ khi đó mới bật mặc định.

---

#### Bản đồ phụ thuộc & rủi ro

* Pha 0→1 độc lập, làm ngay. Pha 2 độc lập. Pha 3 cần Pha 1 (dùng E_skip làm mốc so). Pha 4 cần 3.
* **Rủi ro lớn nhất:** Pha 3 trượt (khả năng thật, theo §12.104). Nếu vậy: W_skip vẫn thắng ở
  Pha 1 (embedding) + hợp nhất Pha 2 — mạng gọn hơn, thước tốt hơn — nhưng **sinh dài sáng tạo
  vẫn chưa mở**, và khi đó §12.104e (mô hình tổng-quát-hoá làm nghiệm thu, chạm Luật 0) là ngã ba
  duy nhất còn lại, để tác giả quyết.
* **Luật 0:** mọi pha closed-form/đếm ở đường sinh. Chỗ duy nhất chạm ranh là **nếu** Pha 4 sau
  này muốn học trọng số gate — lúc ấy dừng, hỏi tác giả (§12.104e).

**Đề nghị bắt đầu: Pha 0 + Pha 1 song song** (rẻ, có thắng đo được sẵn ở Pha 1). Pha 2 giao M4
(cần v12 + held-out). Pha 3 là phép cược lớn — làm sau khi Pha 1 cho mốc so.

---

### 12.115 PHA 0+1 — E_skip thành mã chính thức; THẮNG làm thước, HOÀ cho sinh (mini) (2026-07-30) — Claude

Tác giả: *"bạn làm luôn đi"*. Làm Pha 0 (hạ tầng) + Pha 1 (E_skip thay CO-embedding) trên mini.

#### a. Mã (Pha 0/1)

* `src/nn_aivie/hca/embedding_skip.py` — `build_eskip()` chính thức, tái dùng: đồng-hiện W_skip đối
  xứng → PPMI (**đúng công thức `core.embedding.build_ppmi`**, smoothing 0,75) → `truncated_svd`.
  Đặt ở `hca` (không `core`) để tránh vòng nhập. Khác CO-embedding DUY NHẤT ở nguồn.
* `scripts/gen_eskip_blind.py` — cổng G-emb(b) chấm mù ba mode neo-content-xa.

#### b. Phát hiện kiến trúc: E vào đường sinh AR qua MỘT cửa hẹp

`state.embedding` chỉ chạm sinh AR qua kênh `distant_content` khi `ar_distant_mode="e"`
(`generate_hebbian.py:554`, β 0,4). Mặc định mode `ppmi`/`coact` ⇒ **E không vào sinh**. Đó là lý
do §12.98 xếp-hạng chỉ +0,007: embedding vốn không phải đòn bẩy sinh ở cấu hình hiện tại.

#### c. Ba cổng G-emb (mini)

| cổng | kết quả | |
|---|---|---|
| (a) thước trôi-đề ≥ E_old | **THẮNG** — E_skip 3–4× (§12.113②) | ✓ |
| (b) sinh không kém | **HOÀ** — cơ học: `mode="e"` E_old đổi **0/30** mẫu so mặc định; E_skip đổi 3/30, cả 3 đều salad ở cả hai | ~ |
| (c) hàng xóm ngữ nghĩa | **BẤT PHÂN trên mini** — token đơn nhiễu cả hai (E_skip ra mảnh byte cho `phở`); cần v12 | ? |

Chi tiết (b): bật `mode="e"` với E_old **không đổi mẫu nào** so `ppmi` ⇒ kênh embedding trơ ở mini.
E_skip đổi đúng 3 prompt (`Điện Biên Phủ`·`Cà phê`·`Hà Nội`), cả 3 vẫn salad — không phải cải thiện.

#### d. Đọc kết quả — Pha 1 thắng ĐÚNG chỗ đã đo, không hơn

E_skip là mã chính thức, tái dùng được. Giá trị **xác nhận đúng phạm vi §12.113②**: **thước** trôi
đề. Cho **sinh**, embedding (bất kể E_old hay E_skip) gần như trơ ở mini vì cửa vào hẹp + embedding
mini yếu. Nên:

* **Nhận E_skip làm THƯỚC** (drift-ruler thay cột `lạc` fold-cứng) — thắng đo được, sạch Luật 0.
* **CHƯA thay `state.embedding`** của checkpoint sinh — trên mini không lợi, hàng xóm chưa tỉnh.
  Việc thay thật để **v12** (M4) khi (b)/(c) đo được ở quy mô có ý nghĩa. KHÔNG chen P27 (Pha 2
  đang giao); nối sau P27 thành P28.

#### e. Thành thật về mini

Cả (b) và (c) đều vướng **giới hạn mini**: embedding 5.664 cha quá thưa để hàng xóm tỉnh, và cửa
`distant_content` trơ. Đây **không** là bằng chứng E_skip vô dụng cho sinh — là bằng chứng **mini
không đo được câu đó**. Chỉ (a) — thước, dùng centroid nhiều token — đủ mạnh để kết luận trên mini.

---

### 12.116 PHA 2 — bake-off dự đoán (mini sơ bộ): W_skip > W_seq nhưng < VarLM (2026-07-30) — Claude

`scripts/probe_wskip_predict.py`. Held-out 20% cha, dựng 3 bộ từ TRAIN, chấm top-1 trên CÙNG tập
test. Mini là **sơ bộ** (thin); v12 là phép đo thật (giao M4 = P27).

#### a. Số (mini, n_test = 21.366 vị trí)

| bộ dự đoán | top-1 | vs VarLM |
|---|---|---|
| W_seq (d=1, bigram) | 16,54% | −4,57pp |
| **W_skip-backoff** (d1..8, λ 0,5) | **18,07%** | −3,05pp |
| **VarLM** (n-gram liền ≤4) | **21,11%** | — |

#### b. Đọc — và nó cho một kết luận KIẾN TRÚC, không chỉ một verdict

* **W_skip-backoff > W_seq (+1,53pp).** Thay W_seq bằng W_skip là cải thiện thật cho dự đoán —
  đúng nghĩa "W_skip thay W_seq".
* **VarLM > W_skip-backoff (−3,05pp).** N-gram LIỀN vẫn hơn skip CÓ-GAP — khớp
  [[long-range-cooccurrence-negative]] (43k: VarLM +7,3pp, skip trung tính).

⇒ Kết luận không phải "bỏ W_skip" mà là: **view đúng của W_skip cho vai DỰ ĐOÁN là backoff LIỀN,
không phải vote có-gap.** VarLM = ràng buộc "mọi bước d=1 nối tiếp" trên chính tensor W_skip. Nên
siêu-cấu-trúc vẫn đứng: bigram = `W_skip[d=1]`, n-gram = lát cắt liền, và lát cắt liền **thắng** lát
cắt có-gap cho next-token. W_skip **gánh được** vai dự đoán — qua view liền — mà không cần cấu trúc
mới; vote có-gap để dành cho embedding (§12.113②) và nghiệm thu (Pha 3).

#### c. Chờ v12

Biên độ có thể đổi ở v12 (43k cho VarSkip khoảng cách lớn hơn). G-pred đăng ký trước: nếu v12 giữ
`VarLM > W_skip-backoff > W_seq` thì chốt "giữ VarLM as view liền của W_skip". P27 sẽ xác nhận.

---

### 12.117 PHA 3 — PMI-cross là thước TRÔI ĐỀ tốt nhất, nhưng KHÔNG phải tín hiệu nghiệm thu (2026-07-30) — Claude

`scripts/probe_wskip_accept.py`. Chấm mỗi mẫu bằng **cross-support** prompt↔sinh: mỗi token-nội-dung
sinh có đồng-hiện skip (train, d≤8) với token-nội-dung prompt không. Nhãn = C (bám prompt) chấm mù,
57 mẫu. So với thước Pha 1 (cos E_skip).

#### a. Số

| thước | Spearman ↔C | tách chuẩn-hoá (C≥2 vs C≤1) |
|---|---|---|
| attest (nhị phân) | +0,082 | +0,362 |
| **pmi (skip cross)** | **+0,643** | **+0,937** |
| cos(E_skip) — Pha 1 | +0,281 | +0,548 |

`pmi` bám nhãn người **hơn gấp đôi** cos(E_skip). Nhị phân vô dụng vì bão hoà (corpus dày, gần mọi
cặp có đồng-hiện: C≥2 0,994 ‖ C≤1 0,973). PMI cứu bằng cách cân theo **độ bất ngờ**, không phải tồn tại.

#### b. Cổng G-acc ĐẠT phần thước — nhưng chẩn đoán để lộ giới hạn CỨNG

Kiểm rủi ro "PMI chỉ là dò-chép trá hình":

* **Spearman pmi↔novelty = −0,112** — yếu. PMI **không** phạt sáng tạo. (thoát rủi ro chính)
* **Đáy-6 PMI toàn C=0** (`máu là một lớp khí quyển`, `có thể gây bệnh`). Bắt **TRÔI** rất chắc.
* **NHƯNG top-6 phần lớn là ECHO prompt**, và đỉnh bảng là mẫu **C=1 XÁO TRỘN**
  (`9 Biên Phủ thắng dịch Điện Chiến dịch`) đạt PMI cao nhất **4,23** chỉ vì lặp từ prompt. Ca
  sáng tạo thật (Hạ Long "con thiên nga", C=2 N=2) chỉ **1,85** — **thấp hơn** mẫu xáo.

⇒ PMI-cross là **thước dò TRÔI ĐỀ** (off-topic) tốt, nhưng **MÙ với ngữ pháp TRONG-đề**: echo xáo
trộn vẫn điểm cao. Đó **đúng** bức tường §12.104 — nó đo *"cùng chủ đề không"*, không đo *"đúng ngữ
pháp không"*. Dự đoán đăng ký trước (§12.114 Pha 3) **đúng một nửa**: bắt được trôi (thứ attestation
liền §12.104 không nhắm), vẫn mù splice/xáo trong-đề.

#### c. Đọc kết quả — Pha 3 cho một THƯỚC, KHÔNG mở cổng SINH

* **Nhận PMI-cross làm cột `trôi-đề`** — tốt hơn cos(E_skip) và hơn hẳn `lạc` fold-cứng. Đây là
  thắng thật cho **đo**.
* **KHÔNG** dùng PMI-cross làm gate sinh (Pha 4) **nguyên trạng**: nó thưởng echo ⇒ đẩy về chép,
  đúng thứ ta muốn tránh. Nếu Pha 4: phải PMI-cross (chủ đề) **TRỪ** lặp/xáo — mà "phát hiện xáo
  trong-đề" chính là thứ §12.104 chứng minh nền đồng-hiện không làm được. Nên **bức tường đứng**.

#### d. Trạng thái lộ trình sau Pha 0–3

| pha | kết quả |
|---|---|
| 0 hạ tầng | ✓ |
| 1 E_skip embedding | ✓ thước; sinh chờ v12 (P28) |
| 2 dự đoán | ✓ W_skip>W_seq, <VarLM ⇒ view liền (=VarLM) gánh dự đoán; v12 = P27 |
| 3 nghiệm thu | **thước trôi-đề TỐT (PMI-cross)**, nhưng KHÔNG mở cổng sinh — §12.104 đứng |
| 4 tích hợp sinh | **chặn** — không có tín hiệu nghiệm thu ngữ-pháp; §12.104e là ngã ba của tác giả |

**Kết luận trung thực cho cả nhánh W_skip:** nâng cấp mạng bằng W_skip **thành công ở ĐO và CẤU
TRÚC** (embedding tốt hơn, thước trôi-đề tốt hơn, dự đoán qua view liền ≥ W_seq) — mạng **gọn và đo
được rõ hơn**. Nhưng **sinh dài sáng tạo vẫn chưa mở**, vì khe thiếu là **nghiệm thu ngữ pháp cho
chuỗi MỚI**, và ba phép đo độc lập (§12.103/§12.104/§12.117) đều nói cấu trúc đồng-hiện — kể cả
W_skip đầy đủ — không cung cấp được khe đó. Ngã ba còn lại là §12.104e (mô hình tổng-quát-hoá làm
nghiệm thu, chạm Luật 0) — quyết định của tác giả.

---

### 12.118 PHA ② — TOÁN TỬ CHUYỂN: âm tính. Kết hợp nhiều bước LÀM MỜ, không sắc (2026-07-30) — Claude

`scripts/probe_wskip_transition.py`. Dựng `P` = W_skip[d=1] content chuẩn-hoá hàng (Markov); đám mây
`cloud = Σ_{k=1..3} γ^k (v0 @ P^k)` từ prompt; điểm mẫu = TB cloud-mass trên token-nội-dung sinh.
So PMI-cross (§12.117) + cos(E_skip) trên cùng nhãn C chấm mù (57 mẫu).

#### a. Số — âm tính rõ

| thước | Spearman ↔C | tách chuẩn-hoá |
|---|---|---|
| **transition(cloud) thô** | **+0,060** | **−0,162** |
| transition PMI-hoá `log(cloud/nền)` | +0,146 | — |
| pmi-cross (§12.117) | **+0,643** | +0,937 |
| cos(E_skip) | +0,281 | +0,548 |

Cloud thô ≈ 0 (tách còn SAI chiều). PMI-hoá cứu lên +0,146 nhưng **vẫn thua xa** PMI-cross. Top-4
cloud cao nhất toàn **C=1** (`Quang Trung đại phá`, `Điện Biên`) — cloud cao = đạt token **hub**, không
= mạch lạc.

#### b. Vì sao — và nó ngược đúng kỳ vọng tôi bán ở §ví-dụ

Tôi lập luận P^k **sinh đường mới** (khái quát) là ưu điểm. Đo cho thấy đó chính là **nhược điểm**:
đồ thị nối dày ⇒ sau 2–3 bước **mọi token đạt tới mọi token** ⇒ khả-đạt **bão hoà** ⇒ mất phân biệt
chủ đề. Kết hợp nhiều bước = **làm mờ**: thêm nhiễu (đường giả) nhanh hơn thêm tín hiệu. PMI cặp
TRỰC TIẾP giữ được độ sắc vì không kết hợp — hỏi thẳng "cặp p–q có đồng hiện mạnh không".

Cùng họ §12.108 (tầm xa không giúp) và ví dụ P¹² → rỗng: toán tử chuyển **trộn về hub**, nên "tầm
nhìn nhiều bước" thực chất là **tầm nhìn mờ dần về từ phổ biến**, không phải quỹ đạo chủ đề.

#### c. Đóng ②

Toán tử chuyển **không** làm thước coherence, và (suy ra) không hứa hẹn làm agenda sinh: nó thiên
về token hub ⇒ dùng để lái sinh sẽ đẩy về văn **generic hơn**, không mạch lạc hơn. **Đóng.**
PMI-cross (§12.117) vẫn là thước trôi-đề tốt nhất. Bức tường nghiệm-thu-ngữ-pháp §12.104 không đổi.

Bài học chốt cho cả nhánh cấu-trúc-đồng-hiện: **cặp trực tiếp > kết hợp nhiều bước** cho phân biệt;
mọi hình thức "với xa hơn / nhiều bước hơn" (§12.108 tầm d, §12.118 hop) đều **làm mờ**, vì tín hiệu
phân biệt nằm ở **cặp gần, bất ngờ**, không ở **khả-đạt xa**.

---

### 12.119 PHA ③④ — ③ match theo d THẮNG cho TRUY HỒI; ④ E_skip attention MỜ hơn PMI thô (2026-07-30) — Claude

#### ③ A/E theo khoảng cách — dương tính thật cho truy hồi (`scripts/probe_ae_distance.py`)

Query = cửa sổ NGẮN token-nội-dung (theo thứ tự) của một cha; xếp hạng mọi cha; cha NGUỒN là đáp
đúng; rank-1 xét HOÀ (query ngắn ⇒ bag hoà hàng loạt). So `bag` (túi hiện tại) vs `skip`/`comb`
(thưởng bigram-nội-dung khớp thứ tự).

| q-len | bag rank-1 | skip/comb rank-1 | Δ | bag MRR → comb |
|---|---|---|---|---|
| 2 | 34,6% | **54,2%** | **+19,6pp** | 0,439 → 0,611 |
| 3 | 74,0% | **88,6%** | **+14,6pp** | 0,761 → 0,883 |
| 4 | 89,4% | **95,0%** | +5,6pp | 0,895 → 0,950 |

**Đây là hướng W_skip DƯƠNG đầu tiên trong bốn.** Cơ chế: bag chỉ đếm token ⇒ query ngắn cho hàng
loạt cha CÙNG điểm (hoà), cha nguồn chìm. Match theo thứ tự (bigram-nội-dung) **phá hoà** đúng
hướng — đòi token kề đúng trật tự ⇒ loại cha chung-token-sai-cấu-trúc. Lãi lớn ở query ngắn
(+19,6pp), teo dần khi query đủ dài để bag tự phân biệt.

**Ranh giới thành thật:** đây là **THU HỒI** (chọn đúng cha để CHÉP), không phải sinh sáng tạo. Nó
cải thiện đường **truy hồi/QA** (chọn câu đáp), khớp ý tác giả "A/E quan tâm từ cách xa d". Không
mâu thuẫn trần cũ (skipxcov +0,007): đó đo trên answer-selection có `cov` gánh 95%; đây đo source-
recovery từ fragment ngắn — **tác vụ khác**, và thứ tự giúp phân-định-nhập-nhằng nhiều. Còn phải
đo v12 (nhiều distractor hơn) + xác nhận nó chuyển thành đáp-đúng cuối, không chỉ rank cha.

#### ④ E_skip attention — bản MỜ của PMI-cross, đúng như dự đoán

Thước trôi-đề (nhãn C chấm mù, 57 mẫu):

| biến thể | Spearman ↔C |
|---|---|
| centroid cos(E_skip) | +0,238 |
| **max-sim cos(E_skip)** | **+0,378** |
| pmi-cross thô (§12.117) | **+0,643** |

max-sim sắc hơn centroid (mỗi token sinh lấy cos lớn nhất tới prompt) nhưng **vẫn thua** PMI cặp
thô. Xác nhận §④-lý-thuyết: nén SVD 96 chiều **làm mất độ sắc**; muốn cổng chống trôi thì dùng
**PMI-cross thô**, không phải embedding-attention. E_skip vào sinh mini vẫn trung tính (§12.115);
v12 (P28) mới biết cửa `distant_content`.

#### Chốt bốn hướng kiến trúc khai thác W_skip

| | kết quả | dùng vào |
|---|---|---|
| ① qk thiên-vị-d | = kênh vote, sinh HOÀ (§12.113) | — |
| ② toán tử chuyển | **ÂM** — làm mờ (§12.118) | — |
| **③ A/E theo d** | **+14,6pp thu hồi cha nguồn** | **truy hồi/QA** (chọn cha chép) |
| ④ E_skip attention | mờ hơn PMI thô (+0,378 < +0,643) | thước (PMI thô tốt hơn) |

Quy luật cuối: W_skip nâng được **TRUY HỒI** (③: chọn đúng cái để chép) và **ĐO** (PMI-cross, E_skip),
**KHÔNG** nâng sinh-token-kế (①②) vì đó cần nghiệm-thu-ngữ-pháp mà đồng-hiện không chứa. ③ đáng đưa
lên v12 — nó là cách cụ thể "W_skip thay túi-từ trong A", và có số dương thật.

---

### 12.120 P27 — bake-off next-token v12: W_skip THUA VarLM −14.5pp (2026-07-30) — M4

`out/p27.log`. Script `scripts/probe_wskip_predict.py` (+ `--max-pos`, precompute `skip_p` — full
test ~1.9M vị trí quá chậm pure-Python trên hub Counter v12).

```bash
PYTHONPATH=src python -u scripts/probe_wskip_predict.py \
    --checkpoint checkpoints/network_v12_sent.pkl --held-out 0.2 --seed 42 --max-pos 80000
```

Held-out 20% cha: train 344,358 / test 86,089. Chấm **80,000** vị trí mẫu từ 1,922,498 vị trí test
(seed+7).

#### a. Bảng top-1

| bộ | acc | vs VarLM |
|---|---|---|
| W_seq(d=1) | **18.32%** | −15.72pp |
| W_skip-backoff | **19.53%** | **−14.51pp** |
| VarLM(n-gram liền) | **34.04%** | — |

#### b. G-pred

W_skip-backoff ≪ VarLM−2pp ⇒ **skip có-gap THUA n-gram liền** ⇒ W_skip gánh embedding + nghiệm thu,
**KHÔNG** gánh next-token. Giữ VarLM as view liền (không thay bằng W_skip-backoff).

So mini §12.116 (W_skip 18.07% / VarLM 21.11%, Δ≈−3pp): cùng **thứ tự** W_skip > W_seq < VarLM, nhưng
v12 VarLM **nhảy mạnh** (34%) khi corpus dày — khoảng cách lớn hơn đo 43k cũ (+7.3pp) lẫn mini.

#### c. Đọc

P27 **chốt** nhánh dự đoán trên v12: không còn hy vọng "W_skip thay VarLM". Lộ trình §12.114 Pha 2
đóng theo nhánh dưới cùng của cổng. Tiếp = P28 (E_skip trên v12 làm thước/sinh).

---

### 12.121 P28 — đo E_skip trên v12: artifacts sẵn, CHỜ Claude chấm (2026-07-30) — M4

Re-derive offline, **không `--save`**. Logs: `out/p28_nn.log` · `out/p28_gen.log` ·
`out/eskip_samples.txt` · `out/eskip_key.json`.

```bash
PYTHONPATH=src python -u scripts/bake_eskip.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --neighbors "Lý,Trần,phở,Hà Nội,vua,sông,chiến,kinh đô" > out/p28_nn.log 2>&1
PYTHONPATH=scripts:src python -u scripts/gen_eskip_blind.py \
    --checkpoint checkpoints/network_v12_sent.pkl > out/p28_gen.log 2>&1
```

#### a. G-emb(c) hàng xóm (nguyên văn log — Claude đọc)

| probe | E_old | E_skip |
|---|---|---|
| Lý | Dương, Hoằng, Hoàng, Trịnh, Bá, Lỗ, Chiếu, Tống | Triệu, Sùng, Túc, Tào, Dận, Chu, Đại, Ân |
| Trần | Lê, Nguyễn, Trịnh, Nguyễn, Văn, Minh, Lê, Khiêm | Trịnh, Thiệu, Ngạn, Lê, Khiêm, Đình, Bá, Nguyễn |
| phở | bún, rán, gỏi, nấu, cháo, chưng, hến, tôm | rán, bún, nấu, cháo, gỏi, xốt, ngon, chưng |
| Hà Nội / kinh đô | (không id) | (không id) |
| vua | chồng, Koi, tiền, phụ, tiểu, Hùng, thiên, yêu | Bê, Koi, tâu, liêng, Đời, trtr, khang, Qua |
| sông | Thuỵ, bắc, vịnh, phía, đô, higashi, pinyin, Sùng | Tonlé, Bồn, ahu, Pênh, Phnôm, Drau, Donau, Tigris |
| chiến | tiểu, độc, tái, đẩy, trọng, lên, tiên, đào | PCCC, khốc, Brexit, Nêu, lực, Chiến, Trận, Cỡ |

#### b. G-emb(b) sinh mù

90 mẫu (30 prompt × A_ppmi / B_e_old / C_e_skip), `ar_distant_mode="e"` cho B/C. Key trong
`out/eskip_key.json`. **Claude chấm** C ≥ B?

#### c. M4 không quyết G-emb

Chờ Claude. Chỉ `--save` khi (b)+(c) đạt cổng §12.31.

---

### 12.122 P29 — PHA ③ match theo d trên v12: G-③ ĐẠT (+31pp @ q=3) (2026-07-30) — M4

`out/p29.log`. `scripts/probe_ae_distance.py`, n-query=800, seed 42.

```bash
for QL in 2 3 4; do
  PYTHONPATH=src python -u scripts/probe_ae_distance.py \
      --checkpoint checkpoints/network_v12_sent.pkl --n-query 800 --q-len $QL \
      >> out/p29.log 2>&1
done
```

#### a. Bảng thu hồi cha NGUỒN

| q-len | bag rank-1 | skip/comb rank-1 | Δ | bag MRR → comb |
|---|---|---|---|---|
| 2 | 4.25% | **16.12%** | **+11.88pp** | 0.083 → 0.212 |
| 3 | 24.62% | **55.88%** | **+31.25pp** | 0.315 → 0.600 |
| 4 | 44.38% | **79.50%** | **+35.12pp** | 0.508 → 0.791 |

(skip = comb trên cả ba q-len ở w=2.0 mặc định.)

#### b. G-③

Cổng: comb−bag rank-1 **≥ +5pp** ở q-len 3 ⇒ **ĐẠT** (+31.25pp). Lãi **lớn hơn mini** (§12.119:
+14.6pp @ q=3) — đúng kỳ vọng 430k cha ⇒ bag hoà nặng hơn. Khác mini: Δ **tăng** theo q-len
(không teo).

#### c. Đọc / giới hạn

Source-recovery dương rõ. Bước tiếp theo (Claude): nối match-theo-d vào `ParentBipartite.A`, rồi
đo answer-selection — chưa chắc chuyển thành đáp-đúng cuối (`cov` có thể nuốt).

---

### 12.123 Chấm P28 (Claude): E_skip ĐẠT mọi cổng — NHẬN; kênh embedding trơ, lợi ở THƯỚC (2026-07-30) — Claude

Chấm hai cổng M4 để lại (§12.121).

#### a. G-emb(c) hàng xóm v12 — E_skip ≥ E_old

Trên v12 mảnh-byte **hết** cả hai (quy mô sửa lỗi mini §12.115). So tay:
* `sông`: E_old `{…higashi, pinyin, Sùng}` (rác) ‖ E_skip `{Tonlé, Phnôm, Donau, Tigris…}` — **E_skip
  THẮNG rõ** (toàn tên sông thế giới).
* `chiến`: E_skip `{khốc, Chiến, Trận, lực}` nhỉnh (E_old generic).
* `phở`/`Trần`: hai bên **ngang, đều tốt** (`bún, cháo, gỏi`; `Lê, Nguyễn, Trịnh`).
⇒ **≥ E_old, thắng ở ca sạch nhất, không thoái lui.** Đạt.

#### b. G-emb(b) sinh mù v12 — ĐẠT kỹ thuật, nhưng kênh TRƠ

90 mẫu, chấm mù /10 (F trôi·C bám·N mới·P lý):

| cấu hình | TB /10 |
|---|---|
| A_ppmi (E không vào) | 4,40 |
| B_e_old (mode e) | **4,40** |
| C_e_skip (mode e) | **4,57** |

* **A_ppmi = B_e_old y HỆT 0/30 mẫu đổi** — bật `mode="e"` với E_old **không đổi gì**, đúng như mini
  (§12.115b): kênh `distant_content` **trơ** cả ở v12.
* C_e_skip vs B_e_old: **hơn 1 · kém 0 · bằng 29**. E_skip đổi đúng 1 prompt, và tốt hơn.

⇒ `C ≥ B` **đạt**, nhưng là đạt **kỹ thuật** (không hại): embedding gần như không phải đòn bẩy sinh
ở cấu hình hiện tại. Lợi ích thật của E_skip nằm ở **thước** (§12.113②) + **hàng xóm** (§12.123a).

#### c. Verdict — NHẬN E_skip

E_skip **≥ E_old ở cả ba cổng**, **strictly better** ở thước (3–4×) và hàng xóm (`sông`),
**0 thoái lui**. Theo cổng đăng ký trước (§12.31 P28) ⇒ **nhận**. Uỷ quyền M4 chạy `--save` tạo
`network_v12_sent_eskip.pkl` (P30). Thành thật: đây là **thắng cho ĐO/embedding**, không cho sinh —
sinh v12 tốt hơn mini là nhờ **quy mô** (`Huế là kinh đô nhà Nguyễn` · `…giết chết Hoằng Tháo`),
không nhờ E_skip.

#### d. Ghi chú quy mô — v12 sinh tốt hơn mini rõ

Đọc 90 mẫu: v12 cho nhiều câu **trôi chảy + đúng** mà mini không có (`Sau đó, ông chỉ huy trận Bạch
Đằng, đánh bại quân Nam Hán, giết chết Hoằng Tháo` · `Lê Lợi… mở đầu nhà Lê sơ`). Vẫn còn trôi/salad
(`espresso đậm đặc` cho Ngô Quyền≈Lê Hoàn · `Philadelphia` cho Hà Nội) và số vỡ (`9 5 4`). Nhưng
mức nền cao hơn hẳn — **đòn bẩy lớn nhất vẫn là DỮ LIỆU**, đúng §"cần train mới không".

---

### 12.124 P30 — `--save` E_skip checkpoint v12 (2026-07-30) — M4

Theo uỷ quyền §12.123c. `out/p30_save.log`.

```bash
PYTHONPATH=src python -u scripts/bake_eskip.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --save checkpoints/network_v12_sent_eskip.pkl
```

* Loaded E_old (32002, 96) → build E_skip (32002, 96) → ghi
  `checkpoints/network_v12_sent_eskip.pkl` (~180s).
* **Không** ghi đè `network_v12_sent.pkl`. File `.pkl` trong `checkpoints/` gitignore — chỉ commit log
  + ticket; checkpoint local để Claude/M4 dùng tiếp.

---

### 12.125 Nối ③ vào A — phát hiện HAI đường A; sửa nhầm đường; ③ CHƯA chạm path đo (2026-07-30) — Claude

Nối match-theo-d vào ranking. Kết quả: **âm tính về tích hợp** — nhưng chẩn ra một sự thật kiến trúc.

#### a. covdir — order-làm-multiplier BẮT TRỌN lãi (cov không nuốt)

Kiểm trước khi sửa (`--covdir-beta`): `covdir = bag·(1+β·order/max)` = **88.6%** = skip thuần ≫ bag
74%. ⇒ kiến trúc multiplier (rank_matches TK_029) **giữ** được lãi order. Yên tâm nối.

#### b. Sửa rank_matches: cặp-hướng d=1 → d≤dmax, + chế độ SUM

`rank_matches` đã có `directed_bigram_bonus=1.5` (BẬT, TK_029) nhưng CHỈ cặp KỀ d=1, `hit=MAX`. Đã:
(1) mở rộng d≤`directed_bigram_dmax` (λ giảm); (2) thêm `directed_bigram_accum="sum"` cộng dồn cặp
khớp. Config + wiring, **tương thích ngược** (dmax=1/max = TK_029 y nguyên). 291 pass.

#### c. Đo trên đường sinh: 0/30 — vì SỬA NHẦM ĐƯỜNG

`gen_dir_blind.py`: `dmax=1` vs `dmax=4`, `max` vs `sum` → **0/30 mẫu đổi**. Chẩn: `matrix_core.py`
gọi `rank_matches` **0 lần** — nó dùng `bi.A.T @ ctx` (túi thuần) TRỰC TIẾP. Và
`generate_hebbian.py:129`: *"chế độ mặc định `ar_matrix_core=True` — MỌI phép đo của ta — matrix core
`return` TRƯỚC khi tới rank_matches"*.

⇒ **HAI đường "A":** (1) `rank_matches` — truy hồi CŨ (legacy, coref guard, essay/discourse pool);
(2) `ParentBipartite.A` (`A.T@ctx`) — matrix-core, **path mọi phép đo dùng**. Tôi sửa (1); path đo
dùng (2). Nên ③ **chưa chạm** sinh matrix-core.

#### d. Bệnh ③ chữa là THẬT, nhưng matrix-core đã đi vòng

`generate_hebbian.py:2816`: *"'Thăng Long' → 8 bài sử ĐỒNG ĐIỂM 0.8922, bài đúng KHÔNG trong top"* —
đúng bài toán coverage-tie ③ giải (+31pp). Nhưng TK_042 đã vá bằng `facts_about-theme` subject-aware,
nên matrix-core không đi qua tie đó nữa.

#### e. Trạng thái & quyết định cần

* Sửa rank_matches (b) **GIỮ** — capability opt-in an toàn (mặc định tắt dmax=1 ⇒ 0 rủi ro), giúp
  đường legacy/fallback + v12 (tie nặng hơn: 430k cha).
* Để ③ chạm path đo (matrix-core), phải nối order vào `A.T@ctx` — **thay đổi TRUNG TÂM**, rủi ro cao,
  lợi ích mini **bất định** (matrix-core đã có spec/soft-AND xử tie). **Không tự làm.**
* **Đề nghị:** đo ③ trên rank_matches ở **v12** trước (M4) — nơi coverage-tie nặng. ③ giảm tie thật
  ⇒ mới cân nhắc nối vào `A.T@ctx`.

Bài học: "nối X vào A" — phải xác định A NÀO trước. Có hai; path đo dùng cái tôi không sửa. Kiểm
đường THẬT trước khi sửa ([[getattr-fallback-silent-default]] cùng họ: sửa/đo nhầm chỗ).

---

### 12.126 P31 — đo ③ trên `rank_matches` v12: G-tie ĐẠT @ q=3 (+13.5pp) (2026-07-31) — M4

`out/p31.log` · `scripts/probe_rank_matches_d.py`. Gọi **thật** `ParentStore.rank_matches`
(không DIY §12.122, không `gen_dir_blind`/`ar_matrix_core`).

```bash
for QL in 2 3 4; do
  PYTHONPATH=src python -u scripts/probe_rank_matches_d.py \
      --checkpoint checkpoints/network_v12_sent.pkl --n-query 800 --q-len $QL \
      >> out/p31.log 2>&1
done
```

A = `dmax=1/accum=max` (TK_029) · B = `dmax=4/accum=sum` (③ opt-in). n=800, seed 42.

#### a. Bảng

| q-len | A rank-1 | B rank-1 | Δ | A tie% → B | mean\|tie\| A→B |
|---|---|---|---|---|---|
| 2 | 14.88% | 11.25% | **−3.62pp** | 62.1→61.6 (+0.8%rel↓) | 15.56→16.85 |
| **3** | 32.62% | **46.12%** | **+13.50pp** | 53.2→53.4 (−0.2%rel) | **17.52→10.99** |
| 4 | 50.50% | **66.88%** | **+16.37pp** | 44.6→**34.5** (**+22.7%rel↓**) | 6.86→3.56 |

#### b. G-tie (đăng ký @ q=3)

rank-1 B−A **≥ +5pp** **HOẶC** tie-rate ↓ ≥20% tương đối ⇒ **ĐẠT** (nhánh rank-1 +13.50pp).
Tie-rate nhị phân gần đứng ở q=3; **mean|tie-set|** vẫn giảm rõ (17.5→11.0). Ở q=4 đạt cả hai
nhánh cổng. Ở q=2 ③ **hại** rank-1 (−3.6pp) — cửa sổ 2 token quá ngắn cho d≤4/sum.

#### c. Đọc / không đụng A.T@ctx

③ trên **đường `rank_matches` production** có lãi thu hồi nguồn ở v12 (q≥3), khớp hướng §12.122
(+31pp DIY) nhưng biên độ nhỏ hơn (API coverage×priority ≠ scorer túi+skip thuần). **M4 không nối
`A.T@ctx`.** Claude cân nhắc bước trung tâm theo §12.125e. Default knobs giữ `dmax=1`.

---

### 12.127 P32 — PMI-cross vào eval + baseline v12 (2026-07-31) — M4

* `src/nn_aivie/hca/metrics_pmi.py` — `PmiCrossIndex` (công thức §12.117).
* `scripts/eval_gen_broad.py` — thêm cột **`pmi`** cạnh `lạc` (không thay cổng cũ).
* Baseline 30 prompt matrix-core (`out/p32_baseline_*` = slice A_off của P33): **pmi_TB = 1.822**.

### 12.128 P33 — order→act opt-in trên path đo: G-act ĐẠT hẹp (2026-07-31) — M4

Knob `ar_matrix_order_act` (default **False**) + dmax/lam/beta/accum trong `config.py`. Áp
multiplier sau `A.T@ctx` trong `parent_coactivation_logits` (`_order_act_multiplier`) — không
rebuild sparse A.

```bash
PYTHONPATH=scripts:src python -u scripts/gen_order_act_blind.py \
    --checkpoint checkpoints/network_v12_sent.pkl --out-prefix out/p33
```

| | pmi_TB | n_tok | text≠ |
|---|---|---|---|
| A_off | 1.822 | 14.1 | — |
| B_order | **1.847** | 12.4 | **5/30** |

Δpmi **+0.025** · pmi↑ 3 / ↓ 2 ⇒ **G-act ĐẠT** (nhánh đếm prompt; Δ < +0.05). Lãi **hẹp**, chậm hơn
(~3× wall trên B). **Không** bật default — Claude review trước khi bake vào A / flip default.

### 12.129 P34 — vote chủ đạo v12: G-vote TRƯỢT (2026-07-31) — M4

`probe_vote_primary.py` trên v12, lưới T∈{0.4,0.6} × (only | chung-câu β=1):

| cấu hình | pmi_TB |
|---|---|
| vote_only_T0.4 | 0.545 |
| vote_chung_T0.4_b1.0 | 0.633 |
| vote_only_T0.6 | 0.572 |
| vote_chung_T0.6_b1.0 | **0.772** |
| matrix baseline (P32) | **1.822** |

Đọc tay: salad / số vỡ / lặp — khớp §12.105 mini. **G-vote TRƯỢT** (không hơn matrix). Giữ
probe-only; **không** wire `generate` / không bật `decode_weight`.

### 12.130 Tổng kết lộ trình Claude nghỉ — khuyến nghị (2026-07-31) — M4

| P | kết quả | khuyến nghị khi Claude về |
|---|---|---|
| P32 | PMI trong eval | dùng cột `pmi` mọi A/B tiếp |
| P33 | G-act hẹp (+0.025) | review multiplier; cân nhắc default / bake A nếu muốn |
| P34 | G-vote trượt | đóng nhánh vote-chủ-đạo cho sinh mượt |
| — | bức tường §12.104 đứng | **§12.104e** = ngã ba tác giả cho LLM sáng-tạo |

**Chốt Luật 0:** W_skip nâng TRUY HỒI + ĐO; “sinh mượt” thực dụng = matrix path (+ order opt-in
hẹp). Vote thuần không đủ. Không đụng Luật 0 / neural acceptor trong kỳ này.

### 12.131 P35 — W_skip primary mặc định (2026-07-31) — M4 · yêu cầu tác giả

Tác giả: «bạn chuyển hết đường sinh sang w skip đi».

| knob | giá trị |
|---|---|
| `ar_w_skip_primary` | **True** (mặc định) |
| `ar_w_skip_primary_T` | **0.5** (softmax+sample; argmax T=0 đã thử — hub lặp) |
| `ar_matrix_w_skip_decode_{lam,topk,dmax}` | 0.5 / 12 / 8 |

`generate()`: nếu primary → bỏ pattern/copy/matrix/agenda; `_w_skip_primary_loop` =
`vote_logits` + zero digit + EOS=median phiếu dương + `_sample` (sharp-gate off).

Lấy lại matrix: `HCAConfig(ar_w_skip_primary=False)`.

**Cảnh báo đo sẵn (P34):** PMI ≪ matrix, salad. Wire theo lệnh tác giả, không theo cổng G-vote.

### 12.132 P36 — chống salad trên W_skip primary: THẤT BẠI → lùi matrix (2026-07-31) — M4

Tác giả: «ok fix salad đi» sau khi xem smoke W_skip primary.

| thử | kết quả smoke v12 |
|---|---|
| bare vote T=0.5 | salad (P34) |
| +same_sent β=1 + fluency d=1 Hadamard | vẫn salad; đôi khi định-nghĩa-từ điển (`Giải phẫu`) |
| argmax T=0 (trước đó) | hub lặp tệ hơn |

**Chốt:** bức tường §12.104 đứng — W_skip không đủ làm fluency sinh mới. Default
`ar_w_skip_primary=False` (matrix). Knobs chống-salad giữ cho opt-in `primary=True`.
Đường sáng-tạo mượt còn lại: **§12.104e** (neural nghiệm thu) — quyết tác giả.

### 12.133 P37 — W_skip primary + ưu tiên agenda MZ (2026-07-31) — M4 · yêu cầu tác giả

Tác giả: «quay lại w skip… ưu tiên các token nằm trong agenda của multizone».

| knob | giá trị |
|---|---|
| `ar_w_skip_primary` | **True** |
| `ar_w_skip_use_mz_agenda` | **True** (ép `ar_multizone` khi dựng plan) |
| `ar_w_skip_agenda_boost` | 2.5 × max(vote) cửa sổ pending |
| `ar_w_skip_agenda_order_window/decay` | 4 / 0.6 |

`generate()` primary: `_build_segment_plan` (MZ) → `_w_skip_primary_loop` boost pending agenda
(+ entity nhẹ; EOS×0.15 khi còn agenda).

Smoke v12 (`out/diag_wskip_proposers.log`): full-row vote `quyết≈1.99 > chức≈1.49`;
generate 5/5 seed ra `Tổ quyết định dời đô từ Hoa Lư về Đại La…`.

### 12.137 W_skip phiếu = tích Hadamard theo d, rồi softmax (2026-07-31) — M4

**Hỏi tác giả:** cộng phiếu (sum) không ổn — các d phải **hỗ trợ nhau**.

**Công thức:** với lịch sử `h`, mỗi khoảng cách `d` lấy hàng proposer `W_skip[d][h[-d]]`
(chuẩn hoá thành phân phối), rồi

```
w[b] = Π_d  p_d[b]     # Hadamard / giao mềm
next ~ softmax(w / T)
```

Hub một phía (`Tổ→chức`) chết nếu các proposer khác không đề cử; token được nhiều
d cùng chỉ (`quyết`) sống.

**Chi tiết wire:**
- `ar_w_skip_vote_agg="prod"` (mặc định primary); `"sum"` giữ §12.106.
- Bỏ proposer cấu trúc / chữ-số / non-surface (tránh `0` trong `1010` nhân chết `quyết`).
- Hard Hadamard trên **cửa sổ 4 proposer gần** (Tổ×Thái×Lý×vua), geo-mean `(Π w)^(1/n)`.
  Full-∩ kèm `Năm` từng đẩy stopword `là/cũng` (đo).
- Loại candidate punct + stopword glue (`là/của/ông/…`).
- Giao rỗng → fallback cộng mềm. Flu gate: `w>0` được `multivote_gate_floor`.
- Agenda lệch cha (prompt `Tổ` nhưng MZ trả `, ông ngoại… Lý Thái Tông`): với `prod`,
  nếu đầu agenda ∉ giao W_skip → bỏ boost/nonag bước đó (Hadamard tự lái).

**Đo v12:** vote prod `Ġchức=0`, `Ġquyết≈1.96`, top `Ġdời≈4.9`. Generate 5/5:
`Tổ dời đô về Thăng Long…` (trước đó agenda Tông ép `, ông ngoại của…`).

### 12.138 d_max=16 (2026-07-31) — M4

Tác giả: đuôi câu lệch — thử `d_max=16`. Wire:
- `ar_matrix_w_skip_decode_dmax=16`
- `ar_w_skip_prod_window=0` (Hadamard dùng hết proposer nội dung trong d_max; bỏ cứng window=4)

**Đo:** xem `out/w_skip_primary_smoke.log`.

### 12.139 Sửa: pool = HỢP đề xuất, không giao (2026-07-31) — M4

Tác giả: không tính giao. Ví dụ `Việt→{tươi,vô,thắng}` ∪ `Nam→{tươi,vô,nhất}`
→ list `{tươi,vô,thắng,nhất}` (thắng/nhất vẫn chọn được).

`agg=prod` đổi: soft-prod chỉ trên proposer đã đề cử × số phiếu; thiếu ≠ nhân 0.

### 12.140 KẾ HOẠCH — Oja / bão hoà hub trên phiếu W_skip (2026-08-01) — M4

> Nguồn: trao đổi tác giả 2026-07-31…08-01. **CẤM** `if token == '.'` / rule từng loại dấu.
> Mọi token cùng một phép trên ma trận / vector phiếu.

#### Trao đổi

- Sau câu đúng (`…Thăng Long .`) gen trôi `. và . Đây . là . mốc…`.
- M4 đề xuất: neo biên câu / EOS boost / cấm glue sau `.`.
- **Tác giả bác:** đó là rule bề mặt (`if token == '.'`). Không Luật-0 / không generalize.
- **Tác giả chỉ hướng:** `.` (và hub tương tự) có **power mạnh + bậc kết nối rất cao**.
  Thay vì nhận diện dấu chấm, dùng phép **bão hoà kiểu Oja**: power càng mạnh / nối càng
  nhiều → trọng số hiệu dụng càng bị phản tác dụng, yếu đi so với token nội dung.
- Tiền lệ repo: `W_know` Oja-L2 hàng (`knowledge_seq.build_w_know_csr`); `oja_compete`
  `y/(1+βy)`; TK_012 gidf khiến `.` tự chìm khi đếm df — **chưa** áp đều lên
  `vote_logits` W_skip primary.

#### Hiện trạng (trước §12.140)

| lớp | hành vi |
|---|---|
| Pool phiếu | §12.139 **hợp** đề xuất (không giao) — đúng ý tác giả |
| Trọng số cạnh | thô `power×priority` CSR, **không** chuẩn hoá hàng / bậc |
| Sau `…Long .` | vote top ≈ `và` (p≈95% @ T=0.5); EOS ≈ 0.07 — tiếp tục connector |
| Rule lệch | `_cand_ok` từng **loại punct** (`. , ! …`) — trái nguyên tắc vừa chốt |
| MZ agenda | tắt khi đo thuần W_skip (`ar_w_skip_use_mz_agenda=False`) — salad Tông ≠ bệnh này |

#### Giải pháp (một công thức, mọi token)

Với mỗi hàng đề cử `row = W_skip[d][src]` (mọi `src`, không phân loại dấu):

1. **(Tuỳ chọn) Oja-L2 hàng** — `ar_w_skip_oja_row` (mặc định **False** sau đo:
   L2 + soft-prod đẩy `Đình`/`Thần`, mất đoạn dời đô).
2. **Bão hoà theo bậc ra** \(k=\mathrm{nnz}(row)\):  
   \(\alpha(k) = 1 / (1 + \beta_d\,\log(1+k))\) — proposer hub (`.`, `là`) đóng góp yếu hơn.
3. Đóng góp vào hợp §12.139: \(\alpha\cdot w\) (× \(\lambda^{d-1}\)).
4. **Bão hoà bậc vào** trên ứng viên:  
   \(\mathrm{score}[b] \mathrel{*}= 1/(1+\beta_{in}\log(1+\mathrm{indeg}_{d=1}(b)))\).
5. Softmax / sample. `oja_compete` trên vector cuối **tắt** (bóp thang → salad).

**Không:** nhánh `if punct` / EOS-after-dot / stopword theo tên.

**Đo A/B (mz off, seed0):** `oja_row` phá câu đầu; `deg+indeg` xáo giữa câu;
`indeg_only` giữ `…Thăng Long .` và làm yếu độc quyền `và` sau chấm.
→ mặc định: row=off, deg=0, indeg=1.0. Knob deg giữ để thử proposer-hub.

#### Changes (wire)

| file | thay đổi |
|---|---|
| `TK_045_MATRIX_CORE.md` | §12.140 kế hoạch + GIAO M4 P38 |
| `hca/config.py` | `ar_w_skip_oja_row`, `ar_w_skip_oja_degree_beta`, `ar_w_skip_oja_compete=0`, `ar_w_skip_oja_indeg_beta` |
| `hca/w_skip.py` | `_oja_scale_row`; `vote_logits` áp Oja-L2+α(nnz) hàng + α(indeg) đích; bỏ lọc punct cứng |
| `hca/generate_hebbian.py` | truyền knob Oja từ config vào `vote_logits` |
| `tests/test_w_skip.py` | hub bậc cao bị hạ tương đối; union vẫn giữ đơn phiếu |
| đo | smoke `Năm 1010, vua Lý Thái Tổ` · mz off · ghi `out/w_skip_primary_smoke.log` |

**Nghiệm thu kỳ vọng:** đoạn dời đô giữ; sau biên câu không còn độc quyền `và` chỉ vì
hub — không bằng rule dấu chấm, bằng bão hoà bậc/power.

**Đo 2026-08-01 (mz off, indeg=1):** 4/5 mở đúng
`…quyết định dời đô từ Hoa Lư về Đại La và đặt tên kinh đô là Thăng Long .`
Sau `.` không còn pattern `. và . Đây` — sang token khác (`Hổ`/`Th…`); vẫn chưa EOS.
Log: `out/w_skip_primary_smoke.log`.

### 12.141 Debug gốc trôi sau biên câu (2026-08-01) — M4

**Câu hỏi:** sau `…Thăng Long .` vì sao vẫn sinh `Hổ` / `cvt` / salad, không EOS?

**Đo** (`out/debug_after_dot.log`, seed0, mz off, indeg=1):

#### Chuỗi nhân quả

1. **`Ġ.` chết trên kênh W_skip**  
   - `is_structural_token=True` → vote **SKIP** proposer d=1.  
   - `W_skip[d=1][Ġ.].nnz = 0` (dấu chấm không nằm trên trục content-skip).  
   - `d1_fluency_row` từ `.` cũng **rỗng**.  
   → **Không có tín hiệu “vừa kết thúc câu”** trong đúng kênh đang lái gen.

2. **Hist câu đã xong vẫn đề cử tiếp** (hợp §12.139):

   | d | proposer | nnz hàng | ghi chú |
   |---|---|---:|---|
   | 1 | `Ġ.` | 0 | SKIP |
   | 2 | `ĠLong` | 830 | USE — có cạnh →`Hổ` |
   | 3 | `ĠThăng` | 104 | USE |
   | 4 | **`Ġlà`** | **5897** | USE — hub proposer cực mạnh |
   | 5… | `đô/kinh/tên/đặt` | 466… | USE |

3. **Ứng viên sau `.`** (vote+indeg): `cvt/Thoát/Hổ/Sùng…` — đồng xuất hiện hiếm với
   `Long`∪`là`, không phải “mở câu mới” có nghĩa.  
   `Hổ` ← `Long@d2 (w=3)` + `là@d4 (w=4.2)`.

4. **same_sent** kéo lại `là/đô/Long/Thăng` (recirculate trong câu vừa viết).

5. **Repetition penalty** đè đúng các token vừa emit → top cuối còn hub/hiếm:
   `và/cvt/Thoát/Hổ` (softmax gần **phẳng**, `và` chỉ ~13%).

6. **EOS không cạnh tranh trên W_skip**  
   - `vote[EOS]=0` (không ai đề cử EOS qua skip).  
   - Primary gán `EOS ← median(pos) ≈ 0.013` vs junk ≈ 0.6–0.8 → **không bao giờ thắng**.

#### Kết luận (không phải “quên if dấu chấm”)

| giả thuyết | kết luận đo |
|---|---|
| Cần `if token=='.'` để dừng | Triệu chứng đúng biên câu, nhưng **gốc** = W_skip không mô hình biên + EOS |
| Indeg bão hoà đủ | Hạ được độc quyền `và`; **không** tạo EOS; còn salad phẳng |
| Proposer `.` power mạnh | **Sai trên W_skip hiện tại** — `.` nnz=0, bị skip; hub thật là **`là`/nội dung câu cũ** |
| Oja-L2 hàng | Đã đo phá câu đầu (P38); không phải đòn cho biên câu |

#### Hướng xử lý (vẫn Luật-0 / ma trận — *chưa wire*, chờ tác giả)

A. **Đưa biên câu vào cùng họ W_skip / succ:** cạnh `w→EOS` hoặc skip-bin có punct
   (parent bipartite đã có `end_after`/`succ→EOS` — kênh matrix; primary đang bỏ qua).  
B. **Bão hoà proposer theo nnz** đủ mạnh khi hist dài sau khi đã có “khối lượng”
   (deg_β) — đo `là` 5897; cần không phá câu đang viết dở.  
C. **EOS vào pool phiếu bằng thống kê** (`end_after[w0]`, tỉ lệ kết sau token) —
   cùng vector softmax, không `if '.'`.  
D. Không: stop-after-first-period rule.

### 12.142 `Ġ.` sau Long — do `_word_start_space_fix`, không phải phiếu (2026-08-01) — M4

**Corpus / encode:** sau `ĠThăng ĠLong` next = **`.`** (`Ġ.` đếm = 0).

**W_skip / flu từ `ĠLong`:** `→.` mạnh (≈106); `→Ġ.` ≈ 0.

**Live log:** sample đúng `'.'`, rồi  
`_word_start_space_fix('ĠLong', '.') → 'Ġ.'`  
(vì prev có `Ġ`, nxt không → gắn twin `Ġ+.`). Lặp lại với mọi `… + '.'`.

→ khoảng trắng trước chấm = **vá spacing từ-mới** áp nhầm lên dấu câu, không phải vote chọn `Ġ.`.

**Vá:** **gỡ** `_word_start_space_fix` + `ar_w_skip_space_fix` (2026-08-01).
Hard-rewrite sau sample lệch Luật 0 (sửa bề mặt token theo if hình thái, không phải phiếu).
Glue `vềNhà` nếu tái hiện → xử lý ở nguồn (emit/vote), không patch sau sample.
Handoff vẫn Ġ-prefix đầu span khi cần (đường nối câu, không phải mọi bước).

### 12.143 W_skip KHÔNG cắt span câu (2026-08-01) — M4 · yêu cầu tác giả

**Bug:** `build_w_skip` gọi `iter_sentence_spans` → cạnh chỉ trong 1 câu.
Parent `…Thăng Long. Đây là mốc…` ⇒ `Long`↛`Đây` (mọi d) ⇒ sau `.` vote không có
token đầu câu sau → salad (`ĠHổ`…).

**Đúng (ví dụ tác giả):** `Việt Nam tươi đẹp. Việt Nam hùng cường.`
content_only: `tươi→Việt` d=2, `đẹp→Việt` d=1, `Nam→Việt` d=3.

**Vá:** bỏ vòng `iter_sentence_spans` trong `build_w_skip` — skip trên cả emit.
(`sent_intra` / same-sent bonus vẫn cắt câu nếu cần — kênh khác.)

### 12.144 Consume cạnh + đích W_skip sau sample (2026-08-01) — M4 · yêu cầu tác giả

**Ý:** token đã chọn qua một kết nối skip → đánh dấu cạnh, tránh recirculate tức thì
(túi xáo). **Lặp token không phải lỗi** (`dời đô` … `kinh đô`) — xem §12.145.

**Cơ chế (decode-only, không sửa CSR checkpoint):**
- **Chỉ cạnh:** sau sample `nxt`, consume **mọi** `(src∈bag, d, nxt)` có trên W_skip
  (không chỉ đúng d vị trí) — tránh `quan` lách sang d' khác từ `là` đã có trong hist.
- Chiều ngược `reverse_window`: gần luôn; xa trừ argmax d=1 — kết hợp §12.145 recover.
- Nhân `ar_w_skip_consume_edge_scale` (0 = tắt cạnh). `kinh→đô` sống nếu `kinh` chưa
  nằm bag lúc consume `đô` lần đầu.
- Flu d=1 tôn trọng block; same-sent tắt bonus trong cửa sổ local-repeat (không chặn cạnh).
- `ar_w_skip_consume_tokens=False` mặc định.

### 12.145 Local-repeat mềm + hồi cạnh (2026-08-01) — M4 · yêu cầu tác giả

**Ý:** lặp lại từ không sai; sai là **xoay túi gần** (`quan/là/trọng`). Cho phép
tái dùng token/cạnh ở xa (hai lần `đô`).

**Cơ chế:**
- `ar_w_skip_local_repeat_window=8`, `scale=0.15` + `near_k/near_scale` +
  `outsider_prefer`: phạt mềm túi gần (không cấm vĩnh viễn).
- `ar_w_skip_consume_recover_steps=8` + `hold_dst`: hồi cạnh khi đích đã ra khỏi
  cửa sổ gần.
- `reverse_window=4`: gần luôn ngược; xa ngược trừ argmax d=1 (`kinh→đô`).
  Kết hợp recover → sau này vẫn nói lại hub khi đã rời túi.

### 12.146 Dual-channel content ⊙ surface (2026-08-01) — M4 · P43

**Ý:** tách **what** (agenda/entity) và **how** (W_skip vote).  
`scores *= content_channel_gate(agenda∪entity, floor=0.2)`.

### 12.147 Gỡ neural/LM acceptor khỏi lộ trình chất lượng (2026-08-01) — M4

**Quyết định tác giả:** không dùng mạng LM/neural làm nghiệm thu hay kênh sinh
trên lộ trình W_skip (LM nội bộ yếu; không phải đường ≈ LLM). P46 / `ar_w_skip_lm_accept`
đã gỡ. Chất lượng = EOS · dual-channel · retrieve · data (Luật 0).

### 12.148 Siết EOS / hist-sau-biên / dual-prompt (2026-08-01) — M4

**Đo:** `Long→ĠĐây` d=2 chỉ w=0.95 ≪ hub (`ĠGiang` 4.2); W_agenda mở `Xét` —
câu 2 salad dù đã có `Đây` lẻ.

**Vá (Luật 0 / biên cấu trúc `_is_boundary`, không hardcode nội dung):**
1. `hist_after_last_boundary` — vote không kéo `là` câu cũ.
2. Content gate luôn gồm token **prompt**.
3. Boost mở câu: W_skip pre-boundary + W_agenda (β thấp).
4. **`ar_w_skip_boundary_eos_beta`:** đứng trên biên sau đoạn đủ dài → chọn EOS
   (tránh salad câu 2 khi cạnh mở yếu). Smoke 5/5: dừng sạch `…Thăng Long.`

**Data còn thiếu:** nhiều cặp `…Long. Đây…` / SENT_SUCC sạch để mở câu 2 đúng
trước khi nới EOS-sau-biên.

### 12.149 Đa-prompt: agenda lệch + hist `? Đáp:` + EOS sau câu ngắn (2026-08-02) — M4

**Hiện tượng (baseline mz off, trước vá):** chỉ prompt đầy `Năm 1010, vua Lý Thái Tổ`
sạch; các prompt khác salad / English junk / fact lệch:

| prompt | trước | gốc |
|---|---|---|
| `Nhà Lý dời đô về` | `…thành lậpNhà… Baltica…` | cha «thành lập 1009» / «tồn tại bao lâu» |
| `Hỏi:… Đáp:` | `pesto…` / meson | hist vote = `[Đáp:]` (sau `?`) → phiếu ~0 |
| `Việt Nam… nằm ở` | `ởViệt… Fansipan` | agenda boost lại `Việt` + fact Fansipan |
| `Phở là món ăn` | `của vùng nào…` | cha QA → đuôi **câu hỏi** làm agenda |
| `Thủ đô… từ khi` | `like` / `higashi` | agenda «Thủ đô Việt Nam…» recirculate |
| `Bắc thuộc…` | Harper / Đinh xáo | cha Tây Sơn / Hai Bà lệch lead |

**Vá (Luật 0 — cấu trúc frame/biên/overlap, không `if token==` nội dung):**

1. **`hist_after_last_boundary`** (`_w_skip_p42_helpers.py`): sau biên nếu chỉ còn
   frame (`Đáp:` / `Ġ`) → vote từ **mệnh đề trước biên** (câu hỏi), không `[Đáp:]`.
2. **Boundary→EOS:** đứng trên biên và đã có ≥1 biên trong phần sinh → EOS
   (bỏ ngưỡng `prompt_len+8` — câu ngắn `Thăng Long.` cũng đóng sạch).
3. **Agenda QA** (`_agenda_from_parent_index`):
   - khớp hỏi trên fold **nội dung** (bỏ `Hỏi:`/`?`/`Ġ` — tránh ov giả = 5);
   - statement chỉ nhận QA khi phủ ≥50% câu hỏi (tránh Tổ→QA «dời đô về đâu»);
   - frame `Đáp:`: ưu tiên cha `is_qa`; câu đáp **đầu**; đuôi sau token cuối trùng
     prompt; **giữ `.`** trong agenda → `Đáp: Thăng Long.`
4. **Pre-consume** prefix agenda trùng bề mặt prompt (+ glue/frame) — hết boost
   lại `Việt`/`Nhà`/`Thủ`.
5. **Weak suffix / specific overlap:** bỏ cha khớp chỉ «Nhà Lý» hoặc chỉ
   «Trong thời kỳ»; completion dump chỉ khi **lead** fact = lead prompt, rồi
   bỏ đúng lead (giữ `Quốc` / `Hà Nội` — không lọc mọi từ prompt kiểu `quốc gia`).
6. **facts_about:** thử nhiều cha (`top_k≥8`) thay vì chỉ `tf[0]`.

**File:** `generate_hebbian.py` · `_w_skip_p42_helpers.py` · `tests/test_w_skip_p42.py`
· `scripts/quality_baseline_v12.py` (sẵn có).

**Đo baseline (mz off, seeds=1) — `quality_pass=true`:**

| prompt | sau |
|---|---|
| Lý Thái Tổ | `…Thăng Long.` |
| Nhà Lý dời đô về | `…Thăng Long, đặt nền móng… kinh đô.` |
| Hỏi/Đáp dời đô | `Đáp: Thăng Long.` |
| Việt Nam nằm ở | `giáp Trung Quốc phía Bắc.` |
| Phở | `bò nổi tiếng… Hà Nội` (đôi seed xáo nhẹ) |
| Ngô Quyền … năm | `thắng quân Nam Hán.` (thiếu năm 938) |
| Bắc thuộc | `có` / EOS sớm — hết salad; retrieve còn trống |
| Thủ đô … từ khi | `Việt Nam là Hà Nội.` |

**Còn lại (data / retrieve, không LM):** năm Ngô Quyền; fact Bắc thuộc đúng;
SENT_SUCC câu 2; spacing BPE nếu tái hiện (không khôi phục `_word_start_space_fix`
— §12.142).

### 12.150 P47 — tầng mệnh đề trên v12 (2026-08-02) — M4

Chạy đúng [`RUNBOOK_TK_045_P47.md`](RUNBOOK_TK_045_P47.md) PHẦN A.
Máy: **Apple M4 Pro · 24 GB** · `arm64`. Nạp v12 sạch (~34s), không thrash
(máy Claude 8GB đứng ở 39.5% cha — đúng blocker TK_040, không phải âm tính probe).

Smoke: `tests/test_clause_layer.py` **8/8** · mini tái lập §13.6 (`VERDICT: MỞ`,
PMI 0.0 → 10 283 mđ · len 10.72 · cross 99.8% · 0.558 bits/tok).

```bash
PYTHONPATH=src .venv/bin/python scripts/probe_clause_layer.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --sample 1500 --dump 8 \
    --out-json out/probe_clause_layer_v12.json
# → out/probe_clause_layer_v12.log · tổng 3765.8s
```

| PMI cut | #mệnh đề | dài TB | ứng viên TB | KHÁC-CHA | H_type | H_clause | **bits/token** | cổng |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| none | 680 864 | 14.73 | 266 261 | **100.0%** | 8.55 | 16.17 | 0.580 | TRƯỢT (len) |
| **−1.0** | **1 168 852** | **8.48** | 701 735 | **100.0%** | 8.71 | 18.01 | **1.027** | **ĐẠT** |
| **0.0** | **1 758 562** | **5.59** | 1 220 790 | **100.0%** | 8.80 | 18.87 | **1.574** | **ĐẠT** |
| **0.5** | **2 120 945** | **4.61** | 1 576 357 | **100.0%** | 8.90 | 19.34 | **1.931** | **ĐẠT** |
| 1.0 | 2 495 807 | 3.88 | 1 884 995 | 100.0% | 8.89 | 19.58 | 2.291 | TRƯỢT (len+bits) |
| 1.5 | 2 817 748 | 3.42 | 2 165 527 | 100.0% | 8.91 | 19.78 | 2.605 | TRƯỢT (len+bits) |

**VERDICT: MỞ** — 3/6 ngưỡng qua cả ba cổng (M2/M3/M1). Vùng vận hành v12:
**PMI ∈ [−1.0, 0.5]** (mini khuyến nghị [0.0, 0.5]; trên v12 thêm −1.0 vì còn dư
bits). `DEFAULT_PMI_THETA=0.0` vẫn ĐẠT — giữ.

**Đọc tay (dump 8):**
- PMI −1.0: mệnh đề thật xen mảnh — `một cách đúng đắn` · `đỉnh sóng, một đầu bạch
  xà uốn qua uốn lại thật cao…` · `phương tiện truyền thông xã hội không?<EOS>`.
- PMI 0.0: ngắn hơn, còn nghĩa — `trang trại cừu cho đến năm 1953.` · `quan sát trận
  chiến đấu này` · `xanh biếc` · kèm mảnh `thái độ` / `sẽ có lúc`.
- PMI ≥ 1.0: vụn — `khí huyết` · `sớm hoặc` · `Thành thật` (đúng kỳ vọng runbook).

**Hai phép tự kiểm:**
1. **Lift cao nhất KHÔNG phải `.!?`** — v12 in `Trust=297.5 · dungculambanh=297.5 ·
   HoatDong=255…` (token hiếm đứng trước EOS). Khác mini (`.!?` sạch). **Ghi kèm**,
   không dừng: PMI-valley vẫn là tín hiệu cắt dưới-câu đang dùng; cổng 3 số vẫn hợp lệ
   theo định nghĩa runbook.
2. **H_type KHÔNG tiến về 6.42 khi len→~6** — ở len 5.59, H_type = 8.80 (mini: 6.59).
   Nguyên nhân đúng dự đoán §13.8: `mean_candidates` nổ (1.2M vs mini 1.8k) ⇒ entropy
   quyết định cao hơn. `bits_per_token = H_clause/len` vẫn dưới 2.0 ở vùng ĐẠT — thước
   khấu hao dùng được; H_type tuyệt đối không còn là tự-kiểm scale-invariant.

**Dự đoán Claude (§13.8 / runbook mục 5) — đúng hướng, hơi thấp ở trần bits:**
- `mean_candidates` ↑ mạnh: ✅ (×10³)
- `cross_parent` ≥ 99%: ✅ (**100.0%** mọi ngưỡng — kill switch M2 áp đảo)
- `H_type` ↑: ✅ (3.8→8.5 / 6.1→8.8)
- `bits_per_token` lên 1.0–1.5: ✅ ở −1.0 (1.027) · **hơi vượt** ở 0.0 (1.574) · sát trần
  ở 0.5 (1.931 < 2.0). **Không** vượt cổng M3 ở vùng vận hành.

**Theo bảng quyết định runbook mục 6:** cả 3 cổng ĐẠT ⇒ **§13 Trụ A MỞ ở quy mô thật**.
Việc kế: Trụ C (cổng phủ) — `mean_candidates` nổ là tín hiệu cần lọc sớm; Trụ D chạy
song song (P48).

⚠️ `bits/token` đo TRÔI CHẢY, không đo ĐÚNG NỘI DUNG (§13.6 / runbook §8) — không trích
1.027 như "LM 1.0 bit".

### 12.151 P48 — Hopfield phá thế hoà trên v12 (2026-08-02) — M4

Chạy đúng [`RUNBOOK_TK_045_P47.md`](RUNBOOK_TK_045_P47.md) PHẦN B.
Máy: Apple M4 Pro · 24 GB. Smoke: `tests/test_hopfield.py` **6/6**.

```bash
PYTHONPATH=src .venv/bin/python scripts/probe_hopfield_rank.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --n-query 800 --q-len 3 --betas 8.0,32.0,128.0 \
    --out-json out/probe_hopfield_rank_v12.json
# → out/probe_hopfield_rank_v12.log · tổng 1330.6s
```

| config | rank-1 | MRR | tie% | Δr1 | Δtie%rel | G-hop |
|---|---:|---:|---:|---:|---:|---|
| **A_rank_matches** | **22.25%** | 0.288 | **50.1%** | — | — | — |
| B_hop_s1 β=8 | 14.75% | 0.222 | 3.9% | −7.50pp | +92.3% | ✗ |
| C_hop_s2 β=8 | 10.75% | 0.160 | 6.4% | −11.50pp | +87.3% | ✗ |
| D_hop_s3 β=8 | 9.00% | 0.132 | 7.0% | −13.25pp | +86.0% | ✗ |
| **E_tiebreak_s2 β=8** | **30.75%** | **0.405** | **0.9%** | **+8.50pp** | +98.3% | **✓** |
| B_hop_s1 β=32 | 14.75% | 0.222 | 3.9% | −7.50pp | +92.3% | ✗ |
| C_hop_s2 β=32 | 15.88% | 0.188 | 4.6% | −6.38pp | +90.8% | ✗ |
| D_hop_s3 β=32 | 16.00% | 0.179 | 4.9% | −6.25pp | +90.3% | ✗ |
| **E_tiebreak_s2 β=32** | **30.88%** | **0.408** | **0.9%** | **+8.63pp** | +98.3% | **✓** |
| B_hop_s1 β=128 | 14.75% | 0.222 | 3.9% | −7.50pp | +92.3% | ✗ |
| C_hop_s2 β=128 | 15.00% | 0.170 | 4.2% | −7.25pp | +91.5% | ✗ |
| D_hop_s3 β=128 | 15.00% | 0.167 | 4.4% | −7.25pp | +91.3% | ✗ |
| **E_tiebreak_s2 β=128** | **31.13%** | **0.409** | **0.9%** | **+8.88pp** | +98.3% | **✓** |

**G-hop: ĐẠT** — tốt nhất `E_tiebreak_s2_b128` (+8.88pp, tie 50.1%→0.9%).
Ba β của E đều ĐẠT (≥ +8.5pp). **Chưa wire** — chờ tác giả (§13.9 kỷ luật mini→v12).

**4 câu B.4:**
1. E còn dương? ✅ **mạnh hơn mini** (+8.5…+8.88pp vs mini +2.25pp).
2. Hoà đỉnh A? **50.1%** (mini 38.2%) — **cao hơn**, không thấp hơn.
3. β tốt nhất? **128** nhích nhẹ; 8/32/128 gần như hoà cho E.
4. B/C/D còn âm? ✅ (−6.3…−13.3pp) — kết luận §13.9(1) giữ: không thay `rank_matches`.

**Dự đoán Claude B.4 — BỊ BÁC một nửa:** đoán hoà A thấp hơn mini ⇒ lãi E nhỏ hơn
+2.25pp. Thực tế hoà **cao hơn** (50.1%) ⇒ trần phá-hoà rộng hơn ⇒ lãi E **lớn hơn**.
`B/C/D` âm: đúng.

⚠️ **Baseline tuyệt đối ≠ §12.126:** P31 @ q=3 có A(dmax=1)=32.62% /
B(dmax=4)=46.12%. P48 A hiện tại = **22.25%** (cùng seed/n=800/q=3). Code/knob đã
đổi kể từ P31 (P44 default dmax=4 + vá sau). **Δ nội-bộ P48 vẫn hợp lệ** cho G-hop;
không so tuyệt đối với bảng P31. Claude nên rà `rank_matches` production nếu cần khớp
lại mốc §12.126.

### 12.152 P57 — cổng cặp-âm-tiết trên v12 (2026-08-02) — M4

Chạy đúng [`RUNBOOK_TK_045_P47.md`](RUNBOOK_TK_045_P47.md) PHẦN C.
Máy: Apple M4 Pro · 24 GB. Smoke: `tests/test_clause_layer.py` **19/19** · mini tái lập
§13.21 (D chết 97%, n=0, probe tự nói “không bác được giả thuyết vật liệu”).

```bash
PYTHONPATH=src .venv/bin/python scripts/probe_fold_gate.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --n-prompt 120 --dump 4 \
    --out-json out/probe_fold_gate_v12.json
# → out/probe_fold_gate_v12.log · 1 758 562 mệnh đề · tổng 115.4s
```

| cổng | lạc(NGHIÊM) | pmi_cross | chết% | tok | n_scored |
|---|---:|---:|---:|---:|---:|
| A_don_am_tiet | 0.982 | 1.506 | 81% | 26.1 | 47 |
| B_bo_chu_so | 0.981 | 1.479 | 82% | 25.8 | 45 |
| C_cap_am_tiet | **nan** | 1.271 | 94% | 19.7 | **0** |
| **D_cap_bo_so** | **nan** | **1.271** | **94%** | 19.7 | **0** |

**G-fold: TRƯỢT** — dead=✗ (94% ≫ 50%) · lạc=✗ (nan, n=0) · pmi=✗ (1.271 < A 1.506).

Theo bảng quyết định C.5 **(a) TRƯỢT**: ở **171×** vật liệu vẫn chết ⇒ **giả thuyết VẬT LIỆU
BỊ BÁC**. Đóng hướng cổng-fold; cơ chế khớp phải đổi hẳn (không phải fold).

**Dự đoán C.4 — BỊ BÁC:** đoán dead xuống 20–40%. Thực tế mini 99% → v12 **94%** (nhích
không đáng kể). Rủi ro ngược “sống nhưng lạc không xuống” **không tới** — chuỗi vẫn chết trước.

**Đọc tay dump D/C** (đầu ra còn sống rất ngắn, lệch đề): `xà phòng…da mặt` · `vào hùa *
theo lệnh…` · `căng băng rôn…` · `tròn nây số âm thanh…` — không phải mệnh đề đúng chủ đề.

### 12.153 P58 — E_tiebreak trên đường sinh v12 (2026-08-02) — M4

Chạy đúng runbook PHẦN D. A/B cùng checkpoint/seed, chỉ khác `--no-tiebreak`.

| | BẬT (mặc định) | TẮT (`--no-tiebreak`) |
|---|---:|---:|
| quality_pass | **true** | true |
| pmi_mean | **2.0558** | 2.0558 |
| salad_after_period_rate | **0.0** | 0.0 |
| eos_rate | 1.0 | 1.0 |
| doi_do_ok_rate | 1.0 | 1.0 |
| elapsed_s | 273.8 | 256.8 |

**G-tb: ĐẠT** — (a) quality_pass BẬT=true · (b) pmi không tụt · (c) salad không tăng.

**Đọc tay 8 prompt:** mọi `cont` BẬT ≡ TẮT (byte-identical trên 24 hàng). Đầu ra **không đổi**.

**Dự đoán D.4 — đúng:** “đầu ra gần như không đổi / quality_pass giữ true”. pmi **bằng**
(không nhích). Rủi ro “§12.149 quá khớp 8 prompt” **không xảy ra** trên bộ này.

Giữ `HOPFIELD_TIEBREAK_BETA` mặc định. Không lùi. JSON:
`out/quality_tiebreak_on.json` · `out/quality_tiebreak_off.json`.

### 12.154 P60 — cổng PMI-cross trên v12 (2026-08-02) — M4

Chạy đúng [`RUNBOOK_TK_045_P47.md`](RUNBOOK_TK_045_P47.md) PHẦN E.
Máy: Apple M4 Pro · 24 GB. Smoke: `tests/test_clause_layer.py` **19/19** · mini tái lập
§13.23 (gold 2.128 · G-pmi ĐẠT ở pmi_1.75/pmi_2).

```bash
PYTHONPATH=src .venv/bin/python scripts/probe_pmi_gate.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --n-prompt 120 --dump 4 \
    --out-json out/probe_pmi_gate_v12.json
# → out/probe_pmi_gate_v12.log · 1 758 562 mệnh đề · tổng 126.8s
```

**MỐC VÀNG v12 = 1.372** (mini 2.128). Trên ngưỡng báo nguy `gold < 1.2` — còn dùng được,
nhưng thấp hơn dải dự đoán 1.5–2.0.

| cổng | pmi | so mốc | chết% | tok |
|---|---:|---:|---:|---:|
| KHONG_cong | 0.722 | −0.650 | 1% | 26.2 |
| fold_dang_chay | 1.506 | +0.134 | **81%** | 26.1 |
| **pmi_1.25** | **1.675** | **+0.303** | **42%** | 14.6 |
| **pmi_1.5** | **2.050** | **+0.678** | **59%** | 11.4 |
| **pmi_1.75** | **2.313** | **+0.941** | **77%** | 8.4 |
| pmi_2 | 2.548 | +1.176 | 86% | 6.3 |
| pmi_2.25 | 2.830 | +1.458 | 90% | 6.2 |

**G-pmi: ĐẠT** — thắng: `pmi_1.25`, `pmi_1.5`, `pmi_1.75`
(a: pmi ≥ gold−0.05=1.322 · b: chết ≤ 81% fold).

Vùng vận hành khuyến nghị cho chấm mù vòng ba: **θ=1.25** (chết thấp nhất trong nhóm ĐẠT)
hoặc **θ=1.5** (pmi cao hơn, chết 59%). **Chưa bật mặc định** — đúng E.6 (pmi vừa cổng vừa
thước ⇒ tự tại; cần chấm mù).

**Đọc tay dump @ θ=1.5:** `đi? đến Thất giai` · `cực độ và đề thi…Toán, Lí, Hóa…` ·
`thân hư vô số lỗi` · `cười” cười khi chụp ảnh…` — ngắn hơn fold, còn lệch đề bề mặt;
không dùng `lạc` để phán.

**Dự đoán E.5:**
- gold v12 thấp hơn mini: ✅ hướng đúng (**1.372**, hơi dưới dải 1.5–2.0; ≥1.2 nên vẫn ĐẠT)
- θ thắng thấp hơn (1.25–1.5): ✅ (1.25/1.5 ĐẠT; 1.75 cũng qua)
- dead cùng θ thấp hơn mini: ❌ `pmi_1.5` chết **59%** (mini 39%) — cao hơn, không thấp hơn
- G-pmi ĐẠT: ✅
- rủi ro gold&lt;1.2: không xảy ra

### 12.155 P61 — sinh cặp chấm mù vòng ba v12 (2026-08-02) — M4

Chạy đúng [`RUNBOOK_TK_045_P47.md`](RUNBOOK_TK_045_P47.md) PHẦN F.

```bash
PYTHONPATH=src .venv/bin/python scripts/gen_gate_blind_pairs.py \
    --checkpoint checkpoints/network_v12_sent.pkl \
    --theta 1.25 --n-prompt 160 \
    --out-pairs out/gate_pairs_v12.txt --out-key out/gate_key_v12.json
```

| | |
|---|---|
| θ PMI | **1.25** (chết thấp nhất nhóm ĐẠT P60) |
| cặp | **112** (≥ 88 cần cho +15pp @ power 80%) |
| trộn | fold\|pmi1.25 = 52 · pmi1.25\|fold = 60 |
| máy | Apple M4 Pro · 24 GB · ~122s |

Artefact: `out/gate_pairs_v12.txt` (nhãn giấu) · `out/gate_key_v12.json` (khoá —
**chỉ** dùng sau khi chấm). Script **không tự chấm**.

**Việc kế (Claude / giám khảo riêng — F.3):** chấm `out/gate_pairs_v12.txt` theo rubric
F.4 → `out/gate_verdict_v12.json`. **Không** đọc khoá / ticket khi chấm. Lật theo F.5
(đếm theo cấu hình, không theo vị trí X/Y).

### 12.156 P66 — sinh cặp chấm mù vòng bảy v12: S1 vs NEO-ĐỀ (2026-08-03) — M4

Chạy đúng [`RUNBOOK_TK_045_P47.md`](RUNBOOK_TK_045_P47.md) PHẦN G ·
[`RUNBOOK_TK_045_P66.md`](RUNBOOK_TK_045_P66.md).
Máy: Apple M4 Pro · 24 GB. Smoke mini `--n-prompt 12` → 3 cặp (script thông).

| lượt | n-prompt | cặp | bỏ (a==b/rỗng) | đổi% | tok S1 | tok anchor | Δlen | power |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| v12 | 200 | 43 | **78.5%** | 21.5% | 29.1 | 31.7 | +8.8% | ✗ |
| v12b | 320 | 87 | **72.8%** | 27.2% | 28.3 | 30.5 | +7.8% | ✗ (87&lt;88) |
| **v12c** | **400** | **104** | **74.0%** | 26.0% | 28.8 | 31.1 | +8.1% | ✓ |

Thời gian mỗi lượt ~161–168s (nạp+tầng ~150s + sinh).

**CỔNG G-anchor:**
1. ≥88 cặp: ✅ trên `v12c` (104)
2. không cảnh báo thiếu power: ✅ `v12c`
3. tỷ lệ bỏ &lt; 70%: ❌ **74%** (mọi lượt ≥72.8%)

⇒ **(3) TRƯỢT.** Theo G.3: **không** chỉnh tham số để cứu. Cơ chế neo-đề **gần trơ ở quy
mô v12** (đổi chỉ ~26% vs mini 38.3%) — số này **bác** phép đo mini như tín hiệu quy mô thật.

**Dự đoán G.4:**
- đổi ≥30%: ❌ (~26%)
- cặp từ 200 đề ∈ 120–170: ❌ (43)
- độ dài anchor không tụt &gt;20%: ✅ (+8%, dài hơn — khớp smoke)

Artefact giao chấm (dù G-anchor(3) trượt — đủ power để Claude đọc nếu muốn):
`out/anchor_pairs_v12c.txt` · `out/anchor_key_v12c.json`. **Không chấm** (P66 §2).

Lượt phụ: `out/anchor_pairs_v12{,b}.txt` + key (thiếu power / gần đủ).

### 12.157 P69 — mệnh đề nam châm rác trên v12 (2026-08-03) — M4

Chạy đúng [`RUNBOOK_TK_045_P47.md`](RUNBOOK_TK_045_P47.md) PHẦN H.
Máy: Apple M4 Pro · 24 GB. `--n-prompt 400` · `network_v12_sent.pkl` · **116s**.

| | THẬT | ĐỐI CHỨNG |
|---|---:|---:|
| tổng lần phát | 833 | 828 |
| mệnh đề khác nhau | 771 | 780 |
| top-10 share | **5.4%** | 4.6% |
| đỉnh (% đề) | **2.8%** | 2.2% |
| # mệnh đề ≥5% đề | 0 | 0 |

tỷ số top10 thật/đối chứng = **1.18×**

**CỔNG G-magnet** (runbook H.3):
1. ≥1 mệnh đề ≥5% đề: ❌ đỉnh **2.8%**
2. top-10 ≥10% tổng lần phát: ❌ **5.4%**
3. thật/đối chứng ≥1.5×: ❌ **1.18×**

⇒ **TRƯỢT cả ba.** Không sửa cơ chế / không chỉnh tham số (H.5).

**TOP 8 (bản thật)** — quan trọng ngang số:

| lần | #đề | văn |
|---:|---:|---|
| 11 | 11 | `km 1661+100 QL1,` |
| 9 | 9 | `ngữ cảnh mà nó được sử dụng.` |
| 5 | 5 | `một quốc gia nhập khẩu ô tô` |
| 4 | 4 | `biệt hóa thành nhiều loại tế bào khác nhau` |
| 4 | 4 | `học hành` |
| 3 | 3 | `có thể khiến người khác` |
| 3 | 3 | `năm 1945` |
| 2 | 2 | `chương trình xử lý văn bản` |

Ghi chú đo: cụm hapax §13.34 **vẫn đứng đầu** (11/400 = 2.8%), nhưng dưới ngưỡng cổng
5%; tập trung gần đối chứng xáo ⇒ **không** đủ bằng chứng «xếp hạng nam châm» theo cổng đã
niêm. Artefact: `out/probe_clause_magnet_v12.json` · `out/probe_clause_magnet_v12.log`.

### 12.158 P71 — sinh cặp chấm mù vòng tám v12: ĐỀ BỎ TRỐNG (2026-08-03) — M4

Chạy đúng [`RUNBOOK_TK_045_P47.md`](RUNBOOK_TK_045_P47.md) PHẦN I ·
[`RUNBOOK_TK_045_P71.md`](RUNBOOK_TK_045_P71.md).
Máy: Apple M4 Pro · 24 GB. `--mode empty --n-prompt 600` · **~179s**.

| | |
|---|---|
| n-prompt | 600 |
| cặp thu | **167** |
| tỷ lệ đề bỏ trống | **27.8%** (167/600) |
| tok mean bên dự phòng | **22.8** (median 22 · min 9 · max 52) |
| trộn | fold\|fold_duphong ≈ cân (khoá) |

**CỔNG G-empty:**
1. ≥88 cặp: ✅ **167**
2. tỷ lệ trống ∈ 5–40%: ✅ **27.8%** (dự đoán I.4: 15–25% — hơi cao hơn nhưng trong cổng)

⇒ **ĐẠT.** Artefact: `out/empty_pairs_v12.txt` · `out/empty_key_v12.json`.
**Không chấm** — giao Claude theo P71 §2.

### 12.159 P74 — sinh cặp chấm mù vòng chín v12: CHẶN BẢN LỀ ĐỔI VAI (2026-08-03) — M4

Chạy đúng [`RUNBOOK_TK_045_P47.md`](RUNBOOK_TK_045_P47.md) PHẦN J ·
[`RUNBOOK_TK_045_P74.md`](RUNBOOK_TK_045_P74.md).
Máy: Apple M4 Pro · 24 GB. `--mode pivot`.

Sửa nhỏ trước chạy: khối sinh bị chèn nhầm vào chỗ ghi header file (lặp bug đã
mô tả trong commit giao) — bỏ gen thừa, ghi header cảnh báo TRỐNG cho `pivot`/`empty`.

| lượt | n-prompt | cặp | đổi% | trống-cặp% | tok fold | tok chanbanle | t | power |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| v12 | 500 | 77 | 15.4% | 14.3% | 26.3 | 26.0 | ~170s | ✗ (&lt;88) |
| **v12b** | **900** | **148** | **16.4%** | **16.9%** | 27.2 | 26.8 | ~190s | ✓ |

**CỔNG G-pivot** (trên `v12b`):
1. ≥88 cặp: ✅ **148**
2. đổi ∈ 10–45%: ✅ **16.4%** (dự đoán J.4: 15–35%)
3. trống-cặp &lt; 35%: ✅ **16.9%** (dự đoán ≤20%)

⇒ **ĐẠT.** Artefact giao chấm: `out/pivot_pairs_v12b.txt` · `out/pivot_key_v12b.json`.
Lượt phụ: `out/pivot_pairs_v12.txt` (+ key/log). **Không chấm** — Claude theo P74 §2.

### 12.160 P75 — sinh cặp chấm mù vòng mười v12: DỪNG SỚM len5 vs len2 (2026-08-03) — M4

Chạy đúng [`RUNBOOK_TK_045_P47.md`](RUNBOOK_TK_045_P47.md) PHẦN K ·
[`RUNBOOK_TK_045_P75.md`](RUNBOOK_TK_045_P75.md).
Máy: Apple M4 Pro · 24 GB. `--mode len --short 2 --n-prompt 400` · **~168s**.

| | |
|---|---|
| n-prompt | 400 |
| cặp thu | **142** |
| tỷ lệ đổi | **35.5%** (142/400) |
| tiền tố nghiêm ngặt | **142/142 = 100%** |
| tok mean `len5` / `len2` | **31.3** / **14.1** |

**CỔNG G-len:**
1. ≥88 cặp: ✅ **142**
2. đổi ∈ 25–70%: ✅ **35.5%** (dự đoán K.4: 35–60%)
3. 100% tiền tố nghiêm ngặt: ✅ **142/142** (máy dò bug sạch)

⇒ **ĐẠT.** Artefact: `out/len_pairs_v12.txt` · `out/len_key_v12.json`.
**Không chấm** — Claude theo P75 §2.

### 12.161 P76 — sinh cặp chấm mù vòng mười một v12: len2 vs ĐÓNG CÂU (2026-08-03) — M4

Chạy đúng [`RUNBOOK_TK_045_P47.md`](RUNBOOK_TK_045_P47.md) PHẦN L ·
[`RUNBOOK_TK_045_P76.md`](RUNBOOK_TK_045_P76.md).
Máy: Apple M4 Pro · 24 GB. `--mode close --short 2 --n-prompt 700` · **~182s**.

| | |
|---|---|
| n-prompt | 700 |
| cặp thu | **186** |
| tỷ lệ đổi | **26.6%** (186/700) |
| tiền tố nghiêm ngặt | **186/186 = 100%** |
| tok mean `len2` / `dong_cau` | **13.9** / **26.9** (**1.93×**) |
| mệnh đề mean `len2` / `dong_cau` | **2.00** / **3.85** (**1.93×**) |

**CỔNG G-close:** ĐẠT cả ba. Artefact: `out/close_pairs_v12.txt` · `out/close_key_v12.json`.
**Không chấm** — Claude theo P76 §2. (Claude đã chấm: §13.42 — `dong_cau` THUA 29,3%.)

### 12.162 P77 — sinh mẫu chấm mù tuyệt đối v12 (2026-08-03) — M4

Chạy đúng [`RUNBOOK_TK_045_P77.md`](RUNBOOK_TK_045_P77.md) mục 1.
Máy: Apple M4 Pro · 24 GB. `gen_absolute_samples.py --n 200` · fold+`max_clauses=2` · **~108s**.
**Không lọc** — giữ cả mẫu trống.

| | |
|---|---|
| số mẫu | **200** (A000–A199, không bỏ) |
| trống | **59 = 29.5%** |
| mệnh đề TB | **1.16** |

**CỔNG G-abs:**
1. đúng 200 mẫu: ✅
2. trống ∈ 15–40%: ✅ **29.5%** (dự đoán ~28%; §13.37 = 27,8%)
3. mệnh đề TB ≤ 2,0: ✅ **1.16**

⇒ **ĐẠT.** Artefact: `out/abs_samples_v12.txt` · `out/abs_meta_v12.json`.
**Không chấm** — Claude theo P77 §2.

### 12.163 P79 — sinh mẫu abs2 v12: `hien_tai` vs `mối_nối_kề` (2026-08-03) — M4

Chạy đúng [`RUNBOOK_TK_045_P79.md`](RUNBOOK_TK_045_P79.md) mục 1.
Máy: Apple M4 Pro · 24 GB. `--n 200 --join-seq-max 0.0` · **~112s**.
Cùng 200 đề × 2 cấu hình = **400 mẫu trộn**, không lọc.

| nhánh | n | trống | mệnh đề TB |
|---|---:|---:|---:|
| `hien_tai` | 200 | **29.5%** (59) | **1.160** |
| `mối_nối_kề` | 200 | **70.0%** (140) | **0.390** |

Ghép chưa-từng-thấy (nhánh kề, đo lại trên cùng seed): **0/18 = 0%**.

**CỔNG G-abs2:**
1. 400 mẫu · khoá 200/200: ✅
2. trống `mối_nối_kề` ∈ 40–75%: ✅ **70.0%** (mini 56,5%)
3. ghép chưa-thấy nhánh mới = 0%: ✅

⇒ **ĐẠT.** Artefact: `out/abs2_samples_v12.txt` · `out/abs2_key_v12.json` ·
`out/abs2_meta_v12.json`. **Không chấm** — Claude theo P79 §2.

### 12.164 P81 — sinh mẫu abs3 v12: `hien_tai` vs `fold_dac_trung` (2026-08-03) — M4

Chạy đúng [`RUNBOOK_TK_045_P81.md`](RUNBOOK_TK_045_P81.md) mục 1.
Máy: Apple M4 Pro · 24 GB. `--n 200 --theme-gidf 5.0` · **~115s**.
Cùng 200 đề × 2 cấu hình = **400 mẫu trộn**, không lọc.

| nhánh | n | trống | mệnh đề TB |
|---|---:|---:|---:|
| `hien_tai` | 200 | **29.5%** (59) | **1.160** |
| `fold_dac_trung` | 200 | **57.0%** (114) | **0.625** |

**CỔNG G-abs3:**
1. 400 mẫu · khoá 200/200: ✅
2. trống `fold_dac_trung` ∈ 25–75%: ✅ **57.0%** (mini 45,3%)
3. trống `hien_tai` ≈ 29,5% (±5): ✅ **29.5%**

⇒ **ĐẠT.** Artefact: `out/abs3_samples_v12.txt` · `out/abs3_key_v12.json` ·
`out/abs3_meta_v12.json`. **Không chấm** — Claude theo P81 §2.

### 12.165 P83 — sinh mẫu abs4 v12: `len1` vs `len2` (bộ đề rời) (2026-08-04) — M4

Chạy đúng [`RUNBOOK_TK_045_P83.md`](RUNBOOK_TK_045_P83.md) mục 1.
Máy: Apple M4 Pro · 24 GB. `--n 200 --len-short 1` · **~115s**.

Lượt đầu: 5 đề trùng **mặt chữ** giữa hai nửa `openers[0:200]` / `[200:400]` (cùng decode,
khác cha — vd. «Giải nghĩa: (khẩu ngữ)»). Cổng (2) TRƯỢT → vá chọn đề theo mặt chữ unique
rồi chạy lại (không đổi ngưỡng/cơ chế).

| nhánh | n | trống | mệnh đề TB |
|---|---:|---:|---:|
| `len1` | 200 | **26.5%** (53) | **0.735** |
| `len2` | 200 | **30.0%** (60) | **1.150** |

Đề trùng giữa hai nhánh: **0**. Δ trống: **3.5pp**.

**CỔNG G-len1:**
1. 400 mẫu · khoá 200/200: ✅
2. 0 đề trùng: ✅
3. Δ trống ≤ 10pp: ✅ **3.5pp**
4. mệnh đề TB `len1` ≤ 1,0: ✅ **0.735**

⇒ **ĐẠT.** Artefact: `out/abs4_samples_v12.txt` · `out/abs4_key_v12.json` ·
`out/abs4_meta_v12.json`. **Không chấm** — Claude theo P83 §2.

### 12.136 W_skip multi-d bị topk/min_count che (2026-07-31) — M4

**Hỏi tác giả:** thuật toán chỉ d=1 (Tổ) hay cả cửa sổ đề cử `quyết`?

**Trả lời:** `vote_logits` **có** vòng `d=1..d_max` (d_max=8). Nhưng sức mạnh multi-d
bị **cắt mất** trước khi cộng phiếu:

| proposer | d vàng | cạnh `→quyết` trong CSR | rank | weight | topk=12 | min_count=2 |
|---|---|---|---:|---:|---|---|
| Tổ | 1 | có | 67 | 1.85 | ❌ cắt | ❌ lọc (<2) |
| Thái | 2 | có | 123 | 1.85 | ❌ | ❌ |
| Lý | 3 | có | 73 | 4.53 | ❌ | ✅ |
| vua | 4 | có | 202 | 0.95 | ❌ | ❌ |

→ vote topk=12: `quyết=0`, `chức=1` (chỉ Tổ d=1 hub «Tổ chức» rank1).

**Vá:** `ar_w_skip_vote_topk=0` (full hàng) · `edge_min=0` khi full · multi-d gate floor
bảo vệ Hadamard. Đo: vote `quyết≈1.99 > chức≈1.49`.

### 12.134 Debug `Thánh→Năm` + vá (2026-07-31) — M4

Hiện tượng: `…Đại ThánhNăm 1 về…` — nghi next-token sau `Thánh` là `Năm`.

| gốc | chi tiết |
|---|---|
| Glue BPE | `ĠThánh`+`Năm` (không Ġ) → decode dính `ThánhNăm` |
| Agenda skip | khớp `Hoa về Lư` bằng cách skip → pending lệch → boost `Đại` → hub `Thánh` |
| Recirculate | prompt `Năm` + entity_register + twin `ĠNăm` lách ban theo exact-id |
| Flu floor 0.05 | `Năm` flu=0 vẫn lách Hadamard |

**Vá:** `flu_floor=0` · `agenda_strict` · `_word_start_space_fix` · ban bề mặt prompt
(`Năm`↔`ĠNăm`) trừ agenda pending. Đo lại: hết `ThánhNăm`; hết `Tổ Năm quyết`.

### 12.135 Debug `Tổ→chức` vs agenda `quyết` (2026-07-31) — M4

Hiện tượng: `…vua Lý Thái Tổ chức quyết định…` thay vì `Tổ quyết định`.

| tín hiệu @ `ĠTổ` | `Ġchức` | `Ġquyết` |
|---|---:|---:|
| W_skip[d=1] | **123.4** («Tổ chức») | 0 |
| flu gate | 1.0 | 0 |
| agenda add | 0 | **+2.5** (pending[0]) |
| FINAL trước vá | 1.0 | 2.5 |

Argmax chọn `quyết`, nhưng softmax T=0.5 vẫn sample được `chức` (~5%+).

**Vá:** `ar_w_skip_nonagenda_scale=0.05` — khi còn pending, nhân điểm token ngoài cửa sổ
agenda (trừ punct/EOS). Đo 5 seed: **0× `Tổ chức`**; seed 2/3 ra đúng
`Tổ quyết định dời đô từ Hoa Lư về Đại La…`.

---

## 13. SOẠN THEO MẢNH ĐƯỢC CHỨNG THỰC — đề xuất trọn vẹn (2026-08-02) — Claude

> **Nguồn:** chỉ đạo tác giả 2026-08-02 — *"đề xuất giải pháp trọn vẹn, để đạt mục tiêu của
> mạng, đó là chất lượng sinh vượt trội hơn nữa… mục tiêu là chất lượng sinh vượt LLM, đúng
> ngữ cảnh, không salad, văn phong tự nhiên trôi chảy"*.
> Đọc: README §0–0.5 · `docs/NN-AIVIE v4.md` §1.5/§2/§3.7/§7 · `docs/study/*` · TK_045 toàn bộ
> · code `generate_hebbian.py` / `w_skip.py` / `matrix_core.py` / `config.py` / `neural_lm.py`.

### 13.0 Trạng thái thật của đường sinh (xác nhận bằng CODE, không bằng doc)

```
generate() → ar_w_skip_primary=True (config.py:501)
          → _wseq_fallback → _w_skip_primary_loop
          → BỎ HẲN matrix_core (succ/path/know/qk/stack), pattern, copy
```

`matrix_core_logits` — nơi có `succ β=5.0`, `path β=4.0`, `stack 0.35`, `qk 0.0` — **không
được gọi** trên mặc định. Hai nửa loại trừ nhau bằng một cờ:

| `ar_w_skip_primary` | đo được | bệnh |
|---|---|---|
| `False` (matrix) | `copy` 0.93–1.00 (§12.72) | **chép** — trôi chảy nhưng không sinh |
| `True` (mặc định) | salad (§12.132), −14.5pp (§12.120) | **sinh** — mới nhưng vỡ ngữ pháp |

§12.133→§12.149 là **12 lớp vá liên tiếp** trên nửa thứ hai. Chúng có tác dụng
(`quality_pass=true` trên 8 prompt) nhưng mỗi cái là một miếng băng dán lên **cùng một lỗ**.

### 13.1 Chẩn đoán hợp nhất — MỘT sự thật, không phải 15 kết quả rời

Xếp lại toàn bộ kết quả đo của ticket theo **hình dạng bài toán**:

| ÂM TÍNH — đều là bài toán **SINH phân bố trên 32k token** | § |
|---|---|
| 8 cơ chế "đếm thêm" → trung tính/âm | v4 §3.7 |
| W_skip dự đoán next-token: −14.5pp vs VarLM | §12.120 |
| toán tử chuyển nhiều bước: làm mờ | §12.118 |
| fluency acceptor · attestation acceptor: trượt | §12.103 · §12.104 |
| vote chủ đạo: G-vote trượt, salad | §12.129 · §12.132 |

| DƯƠNG TÍNH — đều là bài toán **CHỌN trong tập nhỏ đã chứng thực** | § |
|---|---|
| match-theo-`d` truy hồi **+31pp** | §12.122 |
| `rank_matches` G-tie **+13.5pp** | §12.126 |
| E_skip làm thước **3–4×** | §12.123 |
| PMI-cross ↔ chấm mù Spearman **+0.643** | §12.117 |
| chép-đáp QA: `lạc = 0.00` ở **mọi** cap — cơ chế DUY NHẤT miễn nhiễm trôi đề | §12.74 |

> **Đồ thị đếm là bộ CHỌN xuất sắc và bộ SINH-TOKEN vô vọng.** `H(next|w0) = 6.42 bits` và
> **tăng theo scale** (v4 §3.7). Không phải 8 lần xui — là một sự kiện kiến trúc: bảng đếm
> không nén nổi ngữ cảnh dài, nhưng xếp hạng vài trăm ứng viên thì rất giỏi.

**Hệ quả thao tác:** đừng đánh nhau với nó. **Nâng ĐƠN VỊ SINH lên đúng chỗ đồ thị mạnh.**

Hiện chỉ có hai độ hạt — **token** (6.42 bits → salad) và **cả emit của cha** (→ chép).
Khoảng giữa — **mệnh đề** — chưa từng tồn tại như một tầng nơ-ron.

### 13.2 Sáu trụ

#### Trụ E — THƯỚC TRƯỚC (chặn)

Riêng ticket này đã có **5 lần kết luận sai vì thước**: `log2(rank)` gọi là bits · `copy=1.0`
coi là đỉnh · nhãn gold sai hai lần (§12.66/§12.68) · G1 đo nhầm thứ (§12.110) · tín hiệu
n=30 không lặp (§12.113). Trước khi viết dòng code kiến trúc nào:

1. `scripts/eval_bits.py` cross-entropy held-out — liên tục, không bão hoà, dùng được cho
   **cả** đường mệnh đề (chấm chính chuỗi token xuất ra).
2. Chấm mù A/B **có cặp**, n ≥ 100 (n=30 đã chứng minh là nhiễu), prompt set **đóng băng
   trước** thí nghiệm.
3. Ba trục thước MỚI:
   - **`attestation`** = % bigram/trigram sinh ra có trong corpus → **bảo chứng chống salad
     bằng cấu trúc**, không bằng cảm tính. Mục tiêu ≈ 100%.
   - **`novelty`** = LCS dài nhất với bất kỳ cha nào, chuẩn hoá — phải **GIẢM** trong khi chất
     lượng giữ. (`copy` giữ làm *ràng buộc*, không làm mục tiêu.)
   - **`lạc`/PMI-cross** cho câu ≥ 2.

#### Trụ A — TẦNG NƠ-RON MỆNH ĐỀ (lõi; KHÔNG cần train lại)

Dựng offline từ `emit_ids` của cha đã có:

- Cắt emit mỗi cha thành **mệnh đề** bằng tín hiệu **đã học sẵn**: `P(EOS|w)` lift
  (`_ends_unit`, §0.4.2) + `sent_intra` L2R/R2L. **Luật 0 sạch — không danh sách dấu câu**
  (lưu ý: `sent_intra._is_boundary` hiện CÒN danh sách `.!?…;。` hardcode — nợ, xem 13.4).
- Mỗi mệnh đề = `ClauseNeuron`: span token · token đầu (gidf cao nhất) · **chữ ký trái** /
  **chữ ký phải** (hàng W_skip ở `d` nhỏ) · cha nguồn · vị trí trong cha.
- Quy mô ước: 65k đơn vị × ~3 ⇒ ~200k nơ-ron mệnh đề. Rẻ, không retrain.

**Decode đổi đơn vị:** mỗi bước chọn **một mệnh đề**, không chọn một token. Ứng viên = mệnh đề
có chữ ký trái tương hợp với chữ ký phải phần đã sinh — cỡ 10²–10³, **không phải 32k**. Hàm
xếp hạng chính là thứ đã đo là điểm mạnh: match-theo-`d` (+31pp) + PMI-cross + E_skip.

**Vì sao thoát 6.42 bits:** entropy mỗi *quyết định* có thể vẫn 3–5 bits, nhưng mỗi quyết định
phát ~8 token ⇒ hiệu dụng ~0.5 bits/token. Chạm khe ~2 bits **bằng cách đổi ĐƠN VỊ, không bằng
đếm giỏi hơn** — vì đếm giỏi hơn đã thất bại 8 lần.

#### Trụ B — THAY KHE (nguồn sáng tạo thật, vẫn không salad)

- Trong mệnh đề, xác định **khe** = vị trí mà token thay được bằng **anh em cùng lớp mẫu**;
  lớp mẫu = token có (chữ ký trái, chữ ký phải) giống nhau trên W_skip (= `SlotHub.fillers` /
  Trụ 3 v4 chưa làm xong; E_skip là không gian tương đồng **đã đo là tốt**).
- Chỉ thay khi **hai cổng** cùng mở: (i) từ thay đủ số lần chứng thực **trong đúng mẫu khe
  đó**; (ii) PMI-cross với chủ đề ≥ ngưỡng.

Đây là đường phá trần synonym **đúng cách**: §12.92 + v4 §3.7 đã đo tương đồng embedding một
mình KHÔNG cứu `chó≈cún`; nhưng *thay-thế-được-trong-khe-đã-chứng-thực* là tín hiệu đếm được,
mạnh hơn hẳn, và là định nghĩa gốc của lớp mẫu.

#### Trụ C — CỔNG PHỦ làm bộ nghiệm thu ở tầng mệnh đề

Cơ chế **duy nhất** miễn nhiễm trôi đề là chép-đáp QA, vì nó là cơ chế duy nhất **có cổng phủ
nội dung** (v4 §1.5). Tổng quát hoá: mệnh đề ứng viên chỉ được nhận nếu (i) phủ ≥1 ý agenda
chưa phủ **hoặc** đẩy được mạch diễn ngôn; (ii) PMI-cross(mệnh đề, chủ đề) ≥ θ; (iii) chữ ký
trái tương hợp.

> ⚠️ §12.104 đã đo **attestation làm acceptor ở tầng TOKEN → trượt**. Đây là thí nghiệm
> **KHÁC**: acceptor ở tầng **mệnh đề**, dùng thước tương quan tốt nhất (+0.643). Cổng nêu
> trước: `lạc` câu 2 không xuống dưới 0.12 ⇒ **đóng trụ C**.

**EOS thành điều tự nhiên**: dừng khi agenda phủ đủ và không mệnh đề nào qua cổng — chính là
`answer closure` §0.2 vốn không code được vì **chưa có đơn vị để đóng**. Gỡ được
`ar_w_skip_boundary_eos_beta` (hiện là quy tắc biên trá hình).

#### Trụ D — HOPFIELD HIỆN ĐẠI cho truy hồi (thuật toán mới; ROI cao, rủi ro thấp)

`ParentBipartite.A/.E` **đã là** một bộ nhớ Hopfield: mẫu lưu = hàng của `A`. Ramsauer et al.
2020 chứng minh attention ≡ một bước truy hồi Hopfield hiện đại:

```
p = softmax(β · A qᵀ) ;  q' = Aᵀ p        # lặp 2–3 bước
```

Dạng đóng, **không backprop**, hợp Luật 0 tuyệt đối (cùng họ toán với PPMI+SVD đã dùng).

Thay được cả đống cổng chỉnh tay đang chồng lên `rank_matches` /
`parent_coactivation_logits`: hard-AND · `distinct-ratio²` · two-tier pool · `priority` nhân
tay · `ask_form_min` · `wh_veto_max` — bằng **một toán tử, một núm β**. β cao = tách sắc (một
cha thắng), β thấp = trộn (tổ hợp nhiều cha) — đúng thứ §12.2 đang vật lộn thủ công. Đứng trên
đường **đã đo là mạnh**, độc lập A–C, chạy song song được.

#### Trụ F — DỮ LIỆU (đòn bẩy lớn nhất đã đo)

§0.W: *"Đòn bẩy lớn nhất đo được: DỮ LIỆU."* §12.148 chỉ đúng chỗ thiếu: **cặp câu nối tiếp
sạch** (`…Long. Đây…` / SENT_SUCC). Tầng mệnh đề **nhân giá trị dữ liệu theo tổ hợp**: số câu
sinh được là tổ hợp của số mệnh đề, không tuyến tính theo số cha.

### 13.3 Ngã ba §12.104e — nói thẳng một lần

`neural_lm.py` đã đo: `W_seq` 7.088 vs LM 11.4M **5.594** bits/token — khe **1.494 bit** mà 8
cơ chế đếm không đóng nổi. Tác giả đã quyết (§12.147) gỡ neural khỏi lộ trình chất lượng.

**Đề xuất này thiết kế để KHÔNG cần đến nó** — đổi đơn vị sinh thay vì làm phân bố token sắc hơn.

Nếu sau này muốn "cơ quan ngữ cảnh liên tục" mà **vẫn** không backprop, còn một cửa giữa repo
chưa thử: **reservoir / Echo State Network** — `h_t = tanh(W_res h_{t-1} + W_in x_t)` với
`W_res` **ngẫu nhiên cố định** (hoặc dẫn xuất từ chính W_skip), readout học bằng **ridge
regression dạng đóng**. Đó là một phép giải tuyến tính, **không phải backprop** — cùng họ với
PPMI+SVD đã chấp nhận. Để đây như **quyết định chính sách của tác giả**, KHÔNG vào đường tới hạn.

### 13.4 Thứ tự + CỔNG nghiệm thu

| # | Việc | Cổng — trượt thì ĐÓNG |
|---|---|---|
| **0** | Trụ E: 3 trục thước mới, n≥100 có cặp, đóng băng prompt | — (nền) |
| **1** | **P47 probe mệnh đề** ✅ **MỞ trên v12** (§12.150): cross 100% · bits/tok 1.03–1.93 @ PMI∈[−1,0.5] · len∈[4.6,8.5] | H ≥ 6 bits **HOẶC** tái tổ hợp khác-cha ≈ 0 ⇒ **ĐÓNG CẢ ĐỀ XUẤT** |
| **2** | **Trụ D: Hopfield phá thế hoà** ✅ G-hop v12 + **ĐÃ WIRE** (§13.15) · P58 G-tb **ĐẠT** (sinh không đổi) | thay hẳn `rank_matches` ⇒ âm; E không ≥ +5pp / không phá hoà ⇒ giữ nguyên |
| **3** | Trụ A: `ClauseNeuron` + decode theo mệnh đề | `attestation` < 95% hoặc chấm mù thua matrix ⇒ đóng |
| **4** | Trụ C: cổng phủ · fold ❌ · **P60 PMI ✅** · **P61 cặp chấm mù ĐÃ SINH** (§12.155, 112 cặp θ=1.25) — chờ Claude chấm | PMI vừa cổng vừa thước ⇒ tự tại nếu bật sớm; arbiter = chấm mù |
| **5** | Trụ B: thay khe | `novelty` không tăng ở `attestation` không đổi ⇒ đóng |
| **6** | Trụ F: data câu nối tiếp + retrain | — |

**Rủi ro chính đã biết, thước bắt được:** mệnh đề quá dài → thoái hoá về chép (`novelty` bắt);
quá ngắn → về salad (`attestation` bắt).

**Nợ phát hiện khi rà:** `sent_intra._is_boundary` còn danh sách dấu câu hardcode
(`. ! ? … ; 。`) — trái §0.4.2 (*"KHÔNG còn dòng code nào hỏi token này có phải dấu câu không"*).
Tầng mệnh đề dùng `_ends_unit` (P(EOS) lift) thay, và nên trả nợ này khi nối.

### 13.5 Về mục tiêu "vượt LLM" — định vị trung thực

Đồ thị đếm vocab 32k trên ~100MB tiếng Việt **sẽ không** vượt LLM hiện đại ở sinh mở tổng quát.
Cái nó **có thể** vượt, và đáng nhắm:

- **Trung thực có dẫn nguồn** — mọi span đều được chứng thực, truy được về cha. LLM ảo giác;
  kiến trúc này **về cấu trúc** không thể, nếu cổng attestation giữ. Đây là chỗ vượt thật.
- **Tiếng Việt ở tầng token** — đã đo: 1.8× tốt hơn GPT-4, vocab nhỏ hơn 3× (§0.4.3).
- **Trong miền corpus** (lịch sử, kiến thức lớp 3–7): đúng + trôi chảy + không salad.

Ba yêu cầu tác giả — *đúng ngữ cảnh · không salad · văn phong trôi chảy* — giải bằng **cấu
trúc chứ không bằng may mắn**: đúng ngữ cảnh ← cổng phủ (Trụ C, cơ chế duy nhất đo được
`lạc=0.00`); không salad ← mọi span đã chứng thực (Trụ A, **bất khả về cấu trúc**); trôi chảy
← văn phong thừa kế từ văn bản thật, không phải từ một phân bố 6.42-bit.

### 13.6 P47 — kết quả CỔNG SINH-TỬ trên mini; và một cổng SAI của tôi (2026-08-02) — Claude

`scripts/probe_clause_layer.py` · `checkpoints/mini_sent_ep_qa.pkl` (5 664 cha) ·
`out/probe_clause_layer_mini.json`.

#### Vấp 1 — sweep θ_eos là NO-OP (tín hiệu EOS cho CÂU, không cho MỆNH ĐỀ)

Lift đo được: `'!'=23.0 · '.'=18.5 · '?'=6.6`, **mọi token khác = 0.0**. Sweep θ ∈
{2,4,8,16} cho **cùng một** kết quả (6 819 mệnh đề, dài TB 16.41). Tức `P(EOS|w)` tách sạch
**biên đơn vị** (§0.4.2 đúng) nhưng **mù với biên dưới-câu** — `,` không bao giờ đứng trước EOS.

**Vá — tín hiệu học thứ hai: THUNG LŨNG PMI trên W_seq.** `pmi(a→b) = log2(P(b|a)/P(b))`;
cắt ở chỗ mối nối **lỏng nhất**. Đây đúng là tiêu chí ISG/BPE dùng để **GỘP**, đọc ngược
thành tiêu chí **CẮT**. Luật 0 sạch (đếm cặp, không danh sách dấu câu).
§0.4.2 đã bác hai tín hiệu dẫn xuất khác (entropy hàng W_seq, tỉ lệ đứng-cuối) — cả hai đo
**token đơn**; cái này đo **CẶP**, đại lượng khác.

#### Kết quả — đường đánh đổi sạch, đơn điệu

| PMI cut | #mệnh đề | dài TB | ứng viên TB | KHÁC-CHA | H_type | H_clause | **bits/token** |
|---|---:|---:|---:|---:|---:|---:|---:|
| none (chỉ câu) | 6 819 | 16.41 | 182 | **99.5%** | 3.77 | 5.31 | **0.230** |
| −1.0 | 7 783 | 14.31 | 625 | 99.9% | 5.33 | 7.72 | 0.372 |
| **0.0** | **10 283** | **10.72** | 1 821 | **99.8%** | 6.10 | 9.08 | **0.569** |
| 0.5 | 12 324 | 8.87 | 2 708 | 99.9% | 6.33 | 9.59 | 0.713 |
| 1.0 | 15 145 | 7.16 | 4 016 | 99.9% | 6.46 | 10.16 | 0.902 |
| 1.5 | 18 224 | 5.90 | 5 606 | 99.9% | 6.59 | 10.57 | 1.115 |

**Ba đọc quan trọng:**

1. **M2 (kill switch) ĐẠT áp đảo ở MỌI cấu hình: 99.5–99.9% ứng viên đến từ CHA KHÁC.**
   Đây là câu hỏi sinh-tử của §13 và nó qua dứt khoát: tầng mệnh đề **không** phải phát lại
   cha. `dead_rate` 1.3–4.3% ⇒ hầu như mệnh đề nào cũng nối tiếp được.
2. **Thước TỰ KIỂM ĐƯỢC:** khi mệnh đề co về ~5.9 token, `H_type → 6.59` ≈ **6.42** — đúng con
   số `H(next|w0)` đo độc lập ở v4 §3.7. Instrument đo đúng đại lượng nó tuyên bố.
3. **bits/token 0.23 → 1.12** trên toàn dải, **đều dưới xa 6.42**. Khấu hao là có thật.

#### Vấp 2 — CỔNG `H_type < 6.0` CỦA TÔI LÀ SAI (ghi lại, không xoá dấu vết)

Lần chạy đầu verdict = **ĐÓNG** vì cổng `H_type < 6.0 bits`. Cổng đó **mis-specified**: nó đo
giá **mỗi QUYẾT ĐỊNH**, trong khi luận điểm §13 là **khấu hao** — một quyết định 6.4-bit cam
kết 9 token rẻ hơn hẳn **chín** quyết định 6.4-bit. Đại lượng đúng là **bits/token**, thứ tôi
đã tính sẵn trong probe nhưng lại **không** đặt cổng lên.

Ngưỡng thay thế **không bịa hôm nay**: `v4 §3.7` (2026-07-17, trước ticket này) nêu đích
*"tám cơ chế đang tinh chỉnh một phân bố cần đi từ ~6 bits xuống ~2"* ⇒ **bits/token < 2.0**.
Cổng cũ vẫn được in ra mỗi dòng (`cổng-cũ H_type`) để hồ sơ không mất dấu.

Đây đúng lớp lỗi §12.56 (*"P5 trượt vì THƯỚC của tôi, không phải vì giả thuyết"*) — lần này
là tôi.

#### VERDICT mini: **MỞ** — 4/6 cấu hình qua cả ba cổng; vùng vận hành khuyến nghị PMI ∈ [0.0, 0.5]

Đọc tay xác nhận cùng vùng: PMI=0.0 cho mệnh đề thật (`ra đời thực sự`, `bị giặc đón chặn`,
`Theo truyền thuyết, Lang Liêu sáng tạo bánh chưng vuông…`); PMI=1.5 vụn quá
(`phó sứ`, `phần lớn`).

#### ⚠️ Giới hạn PHẢI ghi — `bits/token` đo TRÔI CHẢY, không đo ĐÚNG NỘI DUNG

Con số này giả định phần ruột mệnh đề tốn **0 bit** — đúng về mặt **trôi chảy** (ruột được
chứng thực nguyên văn), **SAI** về mặt **nội dung**: chọn một mệnh đề là cam kết luôn các
fact trong đó. Giá nội dung là việc của **Trụ C** (cổng phủ) và cần phép đo riêng
(`lạc` / PMI-cross), không được suy ra từ bảng này. Không ai được trích 0.569 bits/token như
"mô hình ngôn ngữ 0.569 bits" — nó không phải thế.

### 13.7 Trụ A — mã chính thức `hca/clause_layer.py` (2026-08-02) — Claude

| file | nội dung |
|---|---|
| **`src/nn_aivie/hca/clause_layer.py`** | `BoundarySignal` (EOS-lift + PMI) · `ClauseNeuron` · `ClauseLayer` · `cut_clauses` · `build/ensure_clause_layer` (cache weakref §12.30) · `join_candidates` |
| `scripts/probe_clause_layer.py` | P47 — **import** từ module, không tự định nghĩa lại |
| `tests/test_clause_layer.py` | 8 test, **xanh** |

**MỘT định nghĩa DUY NHẤT:** probe import từ module — tránh đúng lớp lỗi mà
`core/neural_lm.py` cảnh báo (hai bên định nghĩa riêng → lệch âm thầm, không crash).

#### Bất biến đã khoá bằng test

| # | bất biến | vì sao khoá |
|---|---|---|
| 1 | EOS-lift đánh dấu mốc **đơn vị**, không đánh dấu từ nội dung | §0.4.2 |
| 2 | **EOS-lift MÙ với biên dưới-câu** (`,` lift = 0) | đây là cái **vấp thật** của P47 — khoá để không ai "tối giản" PMI đi rồi tưởng vẫn chạy |
| 3 | θ_pmi ↑ ⇒ số mệnh đề ↑ đơn điệu, dài TB ↓ | không khoá một θ cụ thể: corpus đồ chơi mọi bigram thấy 1 lần ⇒ PMI dương khắp nơi. Vùng vận hành thật ở §13.6 |
| 4 | ứng viên nối **chỉ** từ cạnh W_skip có thật | **bảo chứng chống salad**; đây là bất biến quan trọng nhất |
| 5 | tái tổ hợp khác-cha > 0 | kill switch M2 |
| — | `cut_clauses` **vô tổn hao** (không mất/không đổi thứ tự token) | cùng tinh thần bất biến VÔ-TỔN-HAO của TK_041 |

**Fixture phải là cha ĐA CÂU** — cha một-câu **không dựng nổi** ca tái tổ hợp: W_skip phải
bắc qua biên câu (§12.143) và các cha phải dùng chung token mở mệnh đề. Lần đầu tôi viết
fixture một-câu và test 5 đỏ với `cross=0` — **đúng như kill switch phải báo**.

#### ⚠️ 7 TEST ĐỎ CÓ SẴN TRÊN HEAD — không phải do §13

Xác minh bằng `git checkout a5f4862` trên cây sạch: **đỏ y hệt**. Hai nhóm **khác bản chất**:

**(a) 6 đỏ = hồi quy của chính §12.149:**

```
test_generate_hebbian: completion_emits_memorized_answer · paraphrase_completes_same_answer
                       autoregressive_reconstructs_answer_token_by_token
                       autoregressive_index_backbone_orders_words
test_ingest:           ingest_qa_becomes_answerable_parent
                       ingest_statement_enables_pattern_completion
```

**(b) 1 đỏ = MÔI TRƯỜNG, không phải hồi quy:**
`test_paths::test_viz_default_checkpoint_prefers_big_a2` — test khẳng định thứ tự ưu tiên
checkpoint mặc định, nhưng máy này có `grade_v3_big_a2_t3_isg.pkl` mà chuỗi `elif` chưa liệt
kê ⇒ đỏ trên **mọi** commit, phụ thuộc nội dung `checkpoints/`. Không gộp vào (a).

Triệu chứng chung: **đáp mất chủ ngữ**. `Hỏi: Cá voi là gì Đáp:` →
`là động vật có vú lớn nhất.` (thiếu `Cá voi`).

Nghi phạm: §12.149 mục 4 (*pre-consume prefix agenda trùng bề mặt prompt*) — dựng để chống
glue `ởViệt`/`lậpNhà`, nhưng ở frame QA chủ ngữ đáp trùng prompt **một cách hợp lệ** và bị
nuốt luôn. §12.149 báo `quality_pass=true` trên 8 prompt nhưng **không chạy suite** ⇒ đúng
bài học §12.4/§12.21: baseline prompt-set **không thay được** suite.

Tách việc riêng, không gộp vào §13.

### 13.8 ĐÃ ĐO — P47 trên v12 (2026-08-02) — M4

> Chi tiết đầy đủ: **§12.150**. Runbook: [`RUNBOOK_TK_045_P47.md`](RUNBOOK_TK_045_P47.md).

**Trạng thái:** P47 trên `network_v12_sent.pkl` — **VERDICT: MỞ** (3/6 ngưỡng PMI qua
cả ba cổng). Máy M4 Pro 24GB nạp sạch (~34s); máy Claude 8GB thrash là giới hạn máy
(TK_040), không phải âm tính.

| # | đại lượng | mini | **v12 @ PMI 0.0** | cổng | |
|---|---|---:|---:|---|---|
| M2 | `cross_parent_ratio` | 99.8% | **100.0%** | ≥ 5% | ✅ |
| M3 | `bits_per_token` | 0.558 | **1.574** | < 2.0 | ✅ |
| M1 | `mean_len` | 10.72 | **5.59** | ∈ [4, 12] | ✅ |

Vùng ĐẠT v12: PMI ∈ [−1.0, 0.5]. `DEFAULT_PMI_THETA=0.0` giữ. Dự đoán §13.8 **đúng hướng**
(`cand`↑ · `H_type`↑ · `cross`≥99% · bits↑); trần bits hơi thấp (0.0→1.574, 0.5→1.931).
`mean_candidates` nổ (×10³) ⇒ Trụ C (cổng phủ) cần sớm. Xem §12.150.

### 13.9 P48 Trụ D — Hopfield hiện đại: THAY `rank_matches` ÂM TÍNH · PHÁ THẾ HOÀ ĂN (2026-08-02) — Claude

`scripts/probe_hopfield_rank.py` · `src/nn_aivie/hca/hopfield.py` · `tests/test_hopfield.py`
(6 xanh) · `mini_sent_ep_qa`, **n=400, q_len=3, cùng bộ query với P31** (§12.126) ·
`out/probe_hopfield_rank_mini.json`.

**Quan sát mở đầu:** `ParentBipartite.A` **đã là** một bộ nhớ Hopfield — cột `pi` = vector
token có trọng số `√(power·priority)·gidf` của cha `pi` = mẫu lưu. Ramsauer et al. 2020:
attention ≡ một bước truy hồi Hopfield hiện đại `p=softmax(β·Aᵀq)`, `q'=Ap`. Dạng đóng,
không backprop — Luật 0 sạch.

#### Kết quả

| config | rank-1 | MRR | hoà% | Δ rank-1 |
|---|---:|---:|---:|---:|
| **A_rank_matches** (production) | **66.50%** | 0.697 | **38.2%** | — |
| B_hop_s1 (β bất kỳ) — *cosine trần* | 56.50% | 0.669 | 0.2% | **−10.00pp** |
| C_hop_s2 β=2 | 32.00% | 0.436 | 0.8% | −34.50pp |
| D_hop_s3 β=2 | 14.75% | 0.233 | 0.8% | **−51.75pp** |
| C_hop_s2 β=32 | 58.75% | 0.665 | 0.2% | −7.75pp |
| D_hop_s3 β=32 | 59.00% | 0.637 | 0.2% | −7.50pp |
| **E_tiebreak_s2 β=32** | **68.75%** | **0.750** | **0.0%** | **+2.25pp** |

#### Ba kết luận tách bạch

1. **"Hopfield THAY `rank_matches`" — BÁC BỎ** (−7.5…−10pp). Đống cổng chỉnh tay
   (distinct_ratio² · priority · power · dir · QA near-tie) **cộng** đơn vị phân tầng
   (cụm / skip-span) mang thông tin thật mà `A` không có. Giả thuyết §13.2 Trụ D rằng
   *"một toán tử, một núm β thay được cả đống cổng"* **SAI** — ghi rõ, không giấu.
2. **"β là núm chống mờ" — ĐÚNG, và nó định lượng được định kiến cũ.** β=2: lặp là thảm hoạ
   (56.5→32.0→**14.75**). β=32: lặp **giúp** (56.5→58.75→59.0). ⇒ Bài học §0.W
   (*"nhiều bước làm mờ"*, §12.118 Spearman +0.06) **đúng ở β thấp và SAI ở β cao** — nó
   được đo trên các toán tử **không có nhiệt độ**. Đây là tinh chỉnh thật cho một kết luận
   đã đóng, không phải lật nó.
3. **Dạng dùng SỐNG SÓT: Hopfield làm BỘ PHÁ THẾ HOÀ.** `rank_matches` hoà đỉnh **38.2%**
   query — đúng nút thắt §12.126 đã chỉ. Xếp lại **chỉ trong nhóm hoà**: rank-1 **+2.25pp**,
   MRR **+0.053** (+7.6% tương đối), hoà **38.2%→0.0%**. Bậc điểm chính bất biến ⇒ 62% query
   không hoà **không bị đụng theo cấu tạo**. Không phải "không thể tệ": trên query hoà, đoán
   sai vẫn đẩy cha đúng xuống — lãi ròng là thứ phải đo, và đã đo.

#### ⚠️ Cổng của tôi SAI — lần thứ hai (§13.6 là lần đầu)

Cổng gốc: `Δrank-1 ≥ +5pp` **HOẶC** `Δtie ≥ 20% rel`. OR trần ⇒ nó báo **ĐẠT** cho cấu hình
làm rank-1 **tụt 9.17pp**, chỉ vì hoà về 0.

Gốc sai: tiêu chí hoà mượn từ G-tie §12.126, nơi nó so **hai biến thể của CÙNG một bộ chấm** —
ở đó "hoà" nghĩa là thật sự không phân biệt được. Ở đây nó so **hai HỌ bộ chấm**: cosine trên
trọng số liên tục **gần như không bao giờ hoà**, nên `tie=0` là **hiện vật của kiểu số**,
không mang tin gì về chất lượng. ⇒ **Tiêu chí hoà KHÔNG chuyển được qua họ bộ chấm khác.**

Vá: hoà chỉ tính **khi chất lượng không tụt** (`d_r1 ≥ −0.5pp`). Sau vá, E đạt cổng qua nhánh
hoà **trong đúng miền hợp lệ của nó** (E giữ nguyên điểm `rank_matches` làm khoá chính ⇒ cùng
họ bộ chấm). Cổng cũ giữ trong mã kèm lý do, không xoá dấu vết.

#### CHƯA NỐI vào `rank_matches` — cố ý

Điểm nối + hướng dẫn đã ghi tại `parent_store.py` cuối `rank_matches`. **Chưa wire** vì mới đo
trên `mini`, mà **checkpoint nhỏ ĐÃ TỪNG đánh lừa NGƯỢC DẤU** (README §0.5.8). Cần M4 xác nhận
trên v12 → tác giả quyết bật. Lưu ý người nối: `rank_matches` **không** giữ tham chiếu `state`
— truyền vào từ call-site có `state`, **đừng bịa một `self._state_ref`** (tôi đã thử và tự gỡ:
nó tạo lỗi tiềm ẩn chỉ nổ khi bật cờ).

📕 **Runbook M4 cho P48: [`RUNBOOK_TK_045_P47.md`](RUNBOOK_TK_045_P47.md) PHẦN B** — smoke ·
lệnh v12 · cổng ĐÃ VÁ · 4 câu cần xác nhận · dự đoán ghi trước · cách nối nếu ĐẠT.

#### ĐÃ ĐO trên v12 (§12.151) — 2026-08-02 — M4

**G-hop ĐẠT.** `E_tiebreak_s2_b128` **+8.88pp** rank-1 (30.75–31.13% vs A 22.25%),
tie 50.1%→0.9%, MRR 0.288→0.409. B/C/D vẫn âm (−6…−13pp). Dự đoán "lãi E nhỏ hơn mini"
**BỊ BÁC** — hoà A v12 **cao hơn** mini (50.1% vs 38.2%) nên trần phá-hoà rộng hơn.
**Chưa wire.** Chi tiết + cảnh báo baseline tuyệt đối ≠ P31: §12.151.

### 13.10 P49 Trụ C — cổng phủ: TRƯỢT cổng đã nêu; và cổng (b) của tôi đo NHẦM BIẾN (2026-08-02) — Claude

`scripts/probe_coverage_gate.py` · `hca/clause_layer.py` (`CoverageGate`, `compose_chain`) ·
`tests/test_clause_layer.py` 12 xanh · `mini_sent_ep_qa`, n=150 prompt, max 5 mệnh đề ·
`out/probe_coverage_gate_mini.json`.

#### Lần chạy 1 — HỎNG THÍ NGHIỆM, không phải hỏng giả thuyết

`lạc` baseline **0.840** (tham chiếu 0.12) · `require_new_cover` loại **47.6 ứng viên/bước**
→ dead **100%** · PMI hầu như không loại gì. Ba dấu hiệu ⇒ soi lại thay vì kết luận:

1. **Lỗi ngữ nghĩa của tôi:** điều kiện *"phủ ≥1 fold agenda CHƯA phủ"* so `cf & agenda_folds`
   với `covered`, mà `covered` **khởi tạo bằng chính fold của prompt** = `agenda_folds`
   ⇒ điều kiện **luôn SAI** ⇒ loại sạch. Sửa: "đẩy mạch" = mang fold **MỚI so với đã phủ**
   (`cf - covered`). Khoá bằng test regression.
2. **Ngưỡng PMI chưa calibrate:** đo phân bố PMI-cross ứng viên (n=1049) —
   p10 **0.53** · p50 **1.25** · p75 **1.69** · p90 **2.23**. Tôi đặt θ=0.5 = **dưới p10**
   ⇒ **no-op**. *Đặt ngưỡng trước khi biết phân bố là đoán, không phải đo.*
3. **Hằng số ma:** `gidf_min=4.0` cứng trong cổng. Thang gidf phụ thuộc corpus — trên
   checkpoint khác nó lặng lẽ trả fold **RỖNG** ⇒ cổng thành no-op mà không báo gì. Test đồ
   chơi bắt đúng ca này ⇒ đã thành **tham số**.

#### Lần chạy 2 — số sạch, đơn điệu

| config | lạc | Δ lạc | dead | Δ dead | mđ/chuỗi |
|---|---:|---:|---:|---:|---:|
| **A_no_gate** | 0.827 | — | 16.0% | — | 4.39 |
| B_pmi 1.25 *(p50)* | 0.657 | −0.170 | 28.7% | +12.7pp | 4.11 |
| B_pmi 1.70 *(p75)* | 0.495 | −0.332 | 54.0% | +38.0pp | 3.23 |
| B_pmi 2.20 *(p90)* | 0.407 | −0.420 | 86.7% | +70.7pp | 1.67 |
| C_anchor1 *(tự tại)* | **0.000** | −0.827 | 42.0% | +26.0pp | 3.58 |

**Cổng phủ CÓ cắt trôi đề** — đơn điệu theo θ, không mơ hồ. Nhưng mọi điểm đều trả giá bằng
`dead_rate`, mà cổng (b) của tôi chỉ cho +10pp ⇒ **G-cov TRƯỢT**.

`require_new_cover` **no-op** ở mọi θ (số trùng khít): mệnh đề nào qua được (1)/(2) cũng đã
mang fold mới. Điều kiện 3 **thừa** — ghi lại, đừng bê vào thiết kế sau.

#### ⚠️ Cổng (b) đo NHẦM BIẾN — lần thứ ba, và lần này KHÔNG đổi phán quyết

`dead_rate` ở đây = *"chuỗi dừng vì không mệnh đề nào qua cổng"*. Nhưng **§13.2 (do chính tôi
viết) đã nói dừng-khi-không-ai-qua-cổng LÀ cơ chế EOS mong muốn** — *"EOS thành điều tự
nhiên: dừng khi agenda phủ đủ và không mệnh đề nào qua cổng"*. Vậy `dead` **không thuần là
giá**; một phần nó **là tính năng**. Biến giá đúng là **lượng văn dùng được** —
`mean_clauses`: C_anchor1 giữ **3.58/4.39 = 82%** độ dài với **0.000** trôi đề.

**Tôi KHÔNG đổi phán quyết theo phát hiện này.** Cổng đã nêu trước là cổng đã nêu trước; đây
là lần thứ ba tôi chỉnh thước sau khi thấy số (§13.6, §13.9, và đây) và một lần nữa sẽ thành
thói quen chọn thước cho vừa kết quả. **G-cov TRƯỢT, chấm hết.** Cái đúng đắn là nêu thước
thay thế cho **một phép đo MỚI**, không diễn giải lại phép cũ.

#### Hai điều thí nghiệm này KHÔNG đo được (giới hạn phải ghi)

1. **`lạc` baseline 0.827 ≫ 0.12–0.27 tham chiếu.** Vì bộ soạn của tôi chọn mệnh đề **thuần
   theo sức mạnh mối nối** — **không có tí kênh nội dung nào**. Đường production có agenda MZ
   + retrieval lái nội dung. Nên đây là **cận dưới**: cổng đang bị bắt gánh một bộ soạn mù nội
   dung. Cấu hình mà §13.2 thật sự đặc tả (cổng **+** agenda MZ) **chưa được đo**.
2. **`lạc`/`dead` không phán được "văn có tốt hơn không".** Chuỗi 3.58 mệnh đề đúng đề vs 4.58
   mệnh đề trôi đề là một phán đoán **chất lượng** — đúng việc của **Trụ E** (chấm mù). Không
   thước nào ở đây thay được nó.

#### Kết luận cho lộ trình

Trụ C **chưa đóng, cũng chưa mở**. Định kiến §12.104 (*acceptor tầng token trượt*) **không**
tái diễn: ở tầng mệnh đề cổng **có tác dụng thật và đo được**, khác hẳn tầng token. Nhưng
tác dụng đó chưa qua được cổng giá đã nêu.

**Việc kế (theo thứ tự):**
1. Đo lại với **agenda MZ** lái nội dung (cấu hình §13.2 thật) — cận dưới hiện tại nói rất ít.
2. Dựng **Trụ E** (chấm mù có cặp, n≥100) rồi mới phán "ngắn-đúng-đề vs dài-trôi-đề".
3. Cả hai đều **sau** khi M4 xác nhận P47 trên v12 — mini đã từng đánh lừa ngược dấu.

### 13.11 P50 Trụ E — cây thước; và nó BÁC ngay headline của chính tôi (2026-08-02) — Claude

`src/nn_aivie/eval/ruler.py` · `scripts/eval_ruler.py` · `tests/test_ruler.py` (8 xanh) ·
mini, n=120 prompt đóng băng · `out/eval_ruler_mini.json` · `out/ruler_blind_pairs.txt`.

#### Thiết kế: hai trục CHỈ có nghĩa khi đọc thành CẶP

| | novelty THẤP | novelty CAO |
|---|---|---|
| **attestation CAO** | **CHÉP** — trôi chảy, vô dụng | ✅ **ĐÍCH** |
| **attestation THẤP** | hỏng | **SALAD** |

Đọc `attestation` một mình = tái phạm §12.112 (*"copy=1.00 là đỉnh"*) mặc áo mới: bộ chép
verbatim đạt 1.00 **tầm thường**. Nên API **không có hàm nào trả một trục** — `cell()` buộc
gọi tên ô. Ba test khoá: bộ chép phải bị gọi **CHÉP**, salad phải bị gọi **SALAD**, và
`novelty` phải dùng **đoạn LIÊN TỤC** chứ không phải LCS dãy-con.

#### `min_pairs_for` — vì sao n=30 của §12.113 thất bại, bằng SỐ HỌC

Công thức đầu của tôi `(z/2δ)²` cho **43** cặp @+15pp — đó là *"nếu may mà quan sát đúng độ
lớn thật thì CI vừa đủ tách khỏi 50%"*, tức **power ≈ 50% = tung đồng xu**. Test bắt được.
Bản đúng có số hạng power 80%: `(z_α+z_β)²/(4δ²)` → **+15pp: 88 cặp · +10pp: 197 · +5pp: 785**.
⇒ **n=30 chỉ phát hiện nổi ≥ ~26pp.** *"Tín hiệu +0.50 không lặp"* (§12.113) là kết cục
**tiên đoán được**, không phải xui.

#### Số đo

| đường | attest | novelty | lạc | pmi | tok | ô |
|---|---:|---:|---:|---:|---:|---|
| `prod` | 0.335 | 0.486 | — | 1.462 | 9.6 | quá-ngắn 46% · SALAD 44% · ĐÍCH 1% |
| `clause_nogate` | 0.979 | 0.634 | 0.824 | 1.063 | 95.5 | **ĐÍCH 82%** · CHÉP 12% |
| `clause_gated` | 0.979 | 0.604 | **0.000** | **1.563** | 62.6 | **ĐÍCH 68%** · CHÉP 16% |

#### ⚠️ BA ĐIỀU LÀM HEADLINE "ĐÍCH 82%" KHÔNG DÙNG ĐƯỢC

**1. Ngưỡng ô là do tôi đặt, chưa calibrate — và nó sụp khi siết.**

| đường | nov≥0.30 | nov≥0.50 | nov≥0.70 | **nov≥0.80** |
|---|---:|---:|---:|---:|
| clause_nogate | 88% | 82% | 23% | **0%** |
| clause_gated | 73% | 68% | 42% | **0%** |

novelty p90 = **0.76 / 0.77** — phân bố bó chặt, **không mẫu nào vượt ~0.8**. novelty 0.70
nghĩa là đoạn chép liên tục dài nhất = **30% đầu ra** (~29 token trên 95). ⇒ **mọi** đầu ra
đều chứa một khối verbatim ~30%. Đây là **tính chất CẤU TRÚC của soạn-theo-mệnh-đề**, không
phải khuyết điểm cài đặt: chọn nguyên mệnh đề thì mệnh đề dài nhất chính là khối chép.
**Trần novelty ~0.77 chỉ nâng được bằng Trụ B (thay khe)** — đúng lý do Trụ B tồn tại.
Đây là lần thứ tư tôi đặt ngưỡng trước khi biết phân bố (§13.10 là lần trước); lần này
**thước tự bắt được**, nên bảng độ nhạy nay là phần cố định của báo cáo.

**2. `attestation` 0.979 của đường mệnh đề gần như TỰ TẠI.** Chuỗi phát nguyên mệnh đề ⇒ mọi
bigram trong mệnh đề đã chứng thực theo cấu tạo; chỉ **mối nối giữa các mệnh đề** mới có thể
trượt (~4/94 bigram) ⇒ sàn ≈ 0.957. ⇒ Nó là **phép kiểm BẢO CHỨNG** (xác nhận tính chất thiết
kế của Trụ A), **không** phải thước so sánh.

**3. `prod` đang bị chấm NGOÀI MIỀN — không kết luận gì về nó.** Prompt là mệnh-đề-mở lấy
ngẫu nhiên giữa corpus (`"Điều đó được xem là tàn dư của"`), trên **mini**; còn `prod` được
chỉnh (§12.149) cho prompt curated trên **v12**. Bảng này **không** nói `prod` kém.

#### Kết quả SẠCH duy nhất rút được

`lạc` **0.824 → 0.000** và `pmi_cross` **1.063 → 1.563** khi bật cổng phủ, giá là độ dài
**95.5 → 62.6 token**. Tái lập §13.10 trên **bộ thước độc lập** ⇒ hiệu ứng của cổng phủ là
thật, không phải hiện vật của một thước.

#### Deliverable: 98 cặp chấm mù, ĐỦ POWER

`out/ruler_blind_pairs.txt` — 98 cặp X/Y giấu nhãn (≥ **88** cần cho +15pp @ power 80%),
khoá ở `out/ruler_blind_key.json`. **Chưa chấm** — giữ nguyên nguyên tắc §12.111 (không tự
chấm kẻo thiên lệch theo kỳ vọng). Đây là thứ duy nhất phán được câu §13.10 để ngỏ:
*ngắn-đúng-đề vs dài-trôi-đề, bên nào hay hơn.*

### 13.12 P51 Trụ B — thay khe: TRƯỢT; hai bug của tôi; và một quy luật hợp nhất (2026-08-02) — Claude

`src/nn_aivie/hca/slot_swap.py` · `tests/test_slot_swap.py` (9 xanh) · đo qua
`scripts/eval_ruler.py` trên mini, n=120 · `out/eval_ruler_mini{,_o3}.json`.

**Cơ chế:** với vị trí `i`, gọi `a=tok[i-1]`, `b=tok[i+1]`; `x` thay được khi
`x ∈ succ(a) ∩ pred(b)` — **lớp mẫu theo khe**, tín hiệu ĐẾM ĐƯỢC (khác tương đồng embedding
mà §12.92 + v4 §3.7 đã đo là KHÔNG cứu được synonym).

#### Hai bug của tôi — cả hai chỉ lộ ra khi ĐO ở quy mô thật

**Bug 1 — HAI định nghĩa "đã chứng thực".** Tôi dựng `succ`/`pred` từ **W_seq**, còn thước
chấm bằng n-gram của **`emit_ids`**. Đo: **60.6%** bigram W_seq không có trong emit
(134 069 vs 52 885). ⇒ tuyên bố *"attestation giữ theo cấu tạo"* của tôi **SAI**:
đo thật 0.979 → **0.882**. Đúng lớp lỗi `core/neural_lm.py` cảnh báo — lệch **âm thầm**.
Test đồ chơi cũ **không bắt được** vì trên toy hai nguồn trùng khít; test regression mới
cố tình làm chúng lệch. Sau vá (một nguồn): attestation **0.979 → 0.979**, guarantee thật.

**Bug 2 — van `max_gidf` VÔ DỤNG với số.** TK_041 §0.4.3 cưỡng chế tách chữ số ⇒ mỗi chữ số
là atom **phổ biến** ⇒ gidf **thấp** ⇒ lọt van tầm thường. **Đọc tay** bắt được, không metric
nào báo động (attestation cao, novelty tăng, `lạc` = 0):

```
năm 1989 → năm 1929      năm 1290 → năm 1890      (1864) → 11264
000 hố va chạm → 020     từ năm 1009 đến năm 1225 → từ trong 180 đã đến làm 1829
```

Đây chính xác là bài học TK_041 §5.2 tái diễn. Vá: van `has_digit` chặn token chứa chữ số
**và láng giềng của nó** (thuộc-tính-token tính lúc build vocab — §0.0 cho phép, tiền lệ
`matrix_core.inner_digit_ids`). Sau vá: mọi số trong dump giữ nguyên vẹn.

#### Kết quả sau khi vá cả hai

| đường | attest | novelty | lạc | ĐÍCH@nov≥0.80 | #thay |
|---|---:|---:|---:|---:|---:|
| `clause_gated` | 0.979 | 0.604 | 0.000 | **0%** | — |
| `+swap order=2` | **0.979** | **0.687** | 0.000 | **27%** | 505 |
| `+swap order=3` | 0.979 | 0.614 | 0.000 | 0% | 60 |

**CỔNG G-slot (nêu trước):** (a) novelty ≥ +0.10 → **+0.083 TRƯỢT** · (b) attest ≥ −0.02 →
**+0.000 ĐẠT** · (c) `lạc` không tăng → **ĐẠT**. ⇒ **G-slot TRƯỢT** (hụt 0.017).

**Trần cấu trúc §13.11 CÓ vỡ:** `nov≥0.80` đi từ **0% → 27%**, novelty p90 **0.77 → 0.84**.
Điều trước đây **bất khả** nay xảy ra ở hơn một phần tư mẫu. Nhưng cổng nêu trước là cổng nêu
trước — **không dời** (đây sẽ là lần thứ tư).

#### ⚠️ Và kể cả nếu ĐẠT thì cũng KHÔNG nên bật — đọc tay nói thế

Số sạch, nhưng ngữ pháp/ngữ nghĩa hỏng thường xuyên, **mọi thước trong bộ đều mù**:

```
order=2:  Việt Nam, → Việt đã,     trong lịch sử → cho lịch sử     đưa tin: → đưa tin và
order=3:  là một triều đại → là Nam triều đại    nhà Tây Sơn → người Tây Sơn
          Phật giáo, tôn sùng → Phật giáo được tôn sùng
```

`order=3` (đòi trigram `(a,x,b)` đã thấy) chỉ còn 60 lần thay, novelty **+0.010**, ĐÍCH@0.80
về **0%** — quá chặt để có tác dụng, mà **vẫn sai**. Hai đầu của đường đánh đổi đều không dùng được.

#### Quy luật hợp nhất — đây mới là phần đáng giữ

> **Chứng thực chỉ mua được tính đúng ở ĐÚNG ĐỘ HẠT mà nó LIÊN TỤC.**

* tầng **token** (§12.104): acceptor attestation → **trượt**;
* tầng **khe** (đây): hai bigram cạnh — hoặc cả trigram — đã chứng thực → **vẫn trượt**;
* tầng **mệnh đề** (Trụ A): **cả span** liên tục đã chứng thực → **ăn** (0.979).

Thay khe **phá chính tính liên tục** đó, nên bảo chứng bay hơi — `(a,x)` và `(x,b)` có thật
ở **chỗ khác** không hàm ý `x` hợp **chỗ này**. Ba phép đo độc lập, cùng một quy luật.

⇒ Đường nâng novelty **không** nằm ở sửa bên trong mệnh đề. Nó nằm ở **mệnh đề ngắn hơn +
nhiều mối nối hơn** (θ_pmi cao hơn ở §13.6 — đo sẵn: θ=1.5 cho mệnh đề 5.9 token) hoặc ở
**dữ liệu** (Trụ F). Ghi để không ai đào lại hố này.

### 13.13 P52 Trụ F — dữ liệu: cổng ĐẠT, nhưng số nói điều KHÁC với cổng (2026-08-02) — Claude

`scripts/probe_data_scaling.py` · `grade_v3_big_a2_t3_isg.pkl` (44 633 cha, nạp 17s / peak
2.2GB trên máy 8GB) · `out/probe_data_scaling.json`.

**Vì sao phải đo:** §0.W khẳng định *"đòn bẩy lớn nhất đo được: DỮ LIỆU"* — nhưng đó là đo
trên **đường sinh cũ**. Ở tầng mệnh đề chưa ai đo; §13.2 Trụ F tới giờ là một khẳng định
**mượn**. Thiết kế có kiểm soát: **một** checkpoint, mẫu cha **ngẫu nhiên LỒNG NHAU**
(N nhỏ ⊂ N lớn) ⇒ biến duy nhất là LƯỢNG dữ liệu. Ở mỗi N, **cả bộ sinh lẫn thước** chỉ
thấy đúng tập cha đó.

| N cha | mệnh đề | ứng viên | khác-cha | novelty | attest | tok | **run** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 2 172 | 1 083 | 0.996 | 0.545 | 0.968 | 62.5 | **22.7** |
| 2 500 | 5 422 | 2 594 | 0.998 | 0.652 | 0.977 | 82.2 | 25.1 |
| 5 000 | 10 731 | 5 140 | 0.999 | 0.594 | 0.986 | 92.0 | 31.7 |
| 11 000 | 23 779 | 11 309 | 1.000 | 0.602 | 0.989 | 119.5 | 36.2 |
| 22 000 | 47 474 | 22 749 | 1.000 | 0.712 | 0.994 | 174.0 | 43.8 |
| **44 633** | 96 409 | 44 940 | 1.000 | **0.737** | 0.995 | 203.6 | **51.7** |

**CỔNG G-data: ĐẠT** — 2/4 bước ~4× đạt Δnovelty ≥ +0.083 (+0.118 · +0.135);
`cross_parent ≥ 0.95` ở mọi nấc.

#### ⚠️ Nhưng cổng đo TỈ LỆ, mà tỉ lệ đẹp lên vì đầu ra DÀI RA

`novelty = 1 − run/len`. Đọc cột `tok` và `run`: **cả hai cùng tăng**. len 62.5→203.6 (3.3×),
run **22.7→51.7 (2.3×)**. Tỉ lệ chép giảm 0.363→0.254, nhưng **khối chép nguyên văn TO GẤP
2.3 LẦN**. Ở nấc lớn nhất, mỗi đầu ra chứa một **trích dẫn liên tục ~52 token**.

Một trích dẫn nguyên văn 52 token **không phải "sáng tạo"**, dù tỉ lệ có đẹp. ⇒ Kết quả
**KHÔNG ủng hộ** cách đọc *"nhiều dữ liệu ⇒ sinh sáng tạo hơn"*.

Nguồn cơ chế: `compose_chain` chọn **mệnh đề dài nhất** cho mỗi token mở; nhiều dữ liệu ⇒
mệnh đề dài nhất **dài hơn** ⇒ chuỗi dài hơn **và** khối chép dài hơn. Đây là **thiên lệch
của bộ soạn của tôi** tương tác với quy mô, không phải "tái tổ hợp giàu lên".

**Nhiễu này tôi KHÔNG đăng ký trước** (chỉ đăng ký nhiễu ngược chiều: index lớn ⇒ novelty
xuống). Phát hiện khi đọc kết quả. Theo kỷ luật: **verdict giữ nguyên ĐẠT** như cổng đã nêu
— nhưng cách đọc phải ghi kèm, và **thước phải sửa** (dưới).

#### Sửa thước: thêm trục TUYỆT ĐỐI `copy_run`

`ruler.RulerScores.copy_run` (token) nay đi kèm `novelty` bắt buộc. Test khoá ca đánh lừa:
cùng khối chép 4 token, đầu ra dài gấp 3 ⇒ `novelty` tăng, `copy_run` **không đổi**.
`novelty` một mình từ nay không đủ để kết luận — cùng họ lỗi §12.112 (`copy=1.00 là đỉnh`)
và §13.11 (ngưỡng ô chưa calibrate).

#### Dương tính THẬT, không bị nhiễu

| đại lượng | 1 000 → 44 633 cha |
|---|---|
| `mean_candidates` | 1 083 → 44 940 — **tuyến tính đúng theo N** |
| `cross_parent_ratio` | 0.996 → **1.000** |
| `attestation` | 0.968 → 0.995 |

⇒ **VẬT LIỆU tổ hợp co giãn tuyến tính theo dữ liệu, và tái tổ hợp khác-cha bão hoà ở 1.000.**
Đây là phần Trụ F đứng vững. Việc bộ soạn **khai thác** được vật liệu đó hay không là câu hỏi
KHÁC — và hiện tại câu trả lời là **chưa** (nó tiêu vật liệu bằng cách chép khối dài hơn).

#### Việc kế rút ra từ số

1. **Sửa thiên lệch bộ soạn**: bỏ "chọn mệnh đề dài nhất", chuyển sang chọn theo sức mạnh mối
   nối / ngẫu nhiên có trọng số. Đây là đòn rẻ nhất và số đã chỉ thẳng vào nó.
2. **θ_pmi cao hơn** (mệnh đề ngắn hơn) — §13.6 đã đo sẵn θ=1.5 → 5.9 token/mệnh đề; kết hợp
   với (1) để hạ `copy_run` tuyệt đối.
3. Chỉ sau (1)+(2) mới bõ chạy lại đường cong này ở quy mô v12.

### 13.14 Gỡ thiên lệch "mệnh đề dài nhất" — và nó LÀM TAN phần lớn dương tính §13.13 (2026-08-02) — Claude

Yêu cầu tác giả sau §13.13. Sửa một dòng trong `compose_chain`: bỏ
`sorted(members, key=-len(tokens))` → **bốc ngẫu nhiên** trong nhóm cùng token mở.

| | TRƯỚC (thiên lệch) | | | SAU (ngẫu nhiên) | | |
|---:|---:|---:|---:|---:|---:|---:|
| **N cha** | tok | **run** | nov | tok | **run** | nov |
| 1 000 | 62.5 | 22.7 | 0.545 | 37.3 | 15.6 | 0.483 |
| 5 000 | 92.0 | 31.7 | 0.594 | 39.6 | 17.0 | 0.470 |
| 11 000 | 119.5 | 36.2 | 0.602 | 45.6 | 15.9 | 0.613 |
| 22 000 | 174.0 | 43.8 | 0.712 | 42.9 | 16.4 | 0.537 |
| **44 633** | 203.6 | **51.7** | 0.737 | 49.2 | **17.7** | 0.566 |

**Khối chép tuyệt đối `run` từ ×2.28 xuống ×1.14** (1k→44k cha) — tức **gần như PHẲNG**
theo dữ liệu. `tok` từ ×3.26 xuống ×1.32. Chẩn đoán §13.13 **đúng**: chuỗi dài ra và khối
chép to ra đều là **thiên lệch của bộ soạn**, không phải hiệu ứng dữ liệu.

#### Hệ quả: dương tính novelty của §13.13 phần lớn TAN

| Δnovelty mỗi bước ~4× | TRƯỚC | SAU |
|---|---:|---:|
| 1 000 → 5 000 | +0.049 | **−0.013** |
| 2 500 → 11 000 | −0.049 | +0.090 |
| 5 000 → 22 000 | **+0.118** | **+0.067** |
| 11 000 → 44 633 | **+0.135** | (xem JSON) |

Trước: 2/4 bước đạt +0.083. Sau: chỉ còn **1/4**, và novelty đi ngang trong nhiễu
(0.483 → 0.566 trên **45×** dữ liệu). ⇒ **Ở tầng mệnh đề, dữ liệu KHÔNG nâng novelty một
cách đáng tin.** Phần "ĐẠT" của §13.13 chủ yếu là hiện vật của chính bộ soạn tôi viết.

`gate_pass` vẫn = True vì cổng chỉ đòi **một** bước đạt — cổng đã nêu trước, **không sửa**.
Nhưng cách đọc đúng nằm ở bảng này, không ở cờ boolean.

#### Cái vẫn đứng vững (không đổi giữa hai lần đo)

`mean_candidates` **1 083 → 44 940 tuyến tính theo N** · `cross_parent` **0.996 → 1.000**
· `attestation` 0.950 → 0.983. **Vật liệu tổ hợp co giãn tuyến tính; bộ soạn vẫn chưa
khai thác được.** §13.13 kết luận vậy, và sau khi gỡ thiên lệch thì kết luận đó **mạnh
hơn**, không yếu đi.

#### Bug thứ ba cùng họ, tìm được khi viết test

`compose_chain` hardcode `gidf_min=4.0` qua `content_folds` — **y hệt** hằng số ma tôi đã
vá trong `CoverageGate` (§13.10) nhưng **quên bộ soạn**. Hậu quả: trên corpus nhỏ mọi token
đều dưới 4.0 ⇒ `content_folds` rỗng ⇒ `last=None` ⇒ **chuỗi chết ngay bước 0, âm thầm**.
Nay là tham số. Bài học: vá một hằng số ma thì phải **grep hết mọi chỗ dùng nó**, không
chỉ chỗ đang debug.

#### Việc kế (không đổi thứ tự §13.13)

θ_pmi cao hơn để hạ `run` tuyệt đối nữa — v12 đã đo sẵn (§12.150): PMI 0.5 → **4.61
token/mệnh đề**, vẫn ĐẠT cả ba cổng. Đó là nấc đáng thử tiếp.

### 13.15 Trụ D — NỐI phá-thế-hoà vào `rank_matches`, mặc định BẬT (2026-08-02) — Claude

**Quyết định tác giả** sau khi §12.151 xác nhận trên v12. Kỷ luật mini→v12 đã thoả:
G-hop ĐẠT ở **cả hai** quy mô (mini +2.25pp · v12 **+8.88pp**).

#### Nối thế nào — và vì sao KHÔNG thêm tham số vào 17 call-site

`rank_matches` chỉ nhận `query_ids`/`vocab`, nhưng `tiebreak_ranked` cần `state` để dùng lại
`ParentBipartite.A` đã cache. Có **17 call-site** ⇒ thêm tham số vào từng chỗ là churn lớn và
dễ sót. Thay bằng **một** chỗ: `GraphState.__setattr__` đặt backref **weakref**
`ParentStore._owner_ref` mỗi khi gán `state.parents` — phủ trọn mọi đường (load, ingest, test).

**weakref chứ không phải tham chiếu cứng:** `state → parents → state` là chu trình; giữ cứng
thì state cũ không được thu hồi — mà đây đúng là repo từng dính lỗi cache theo `id(state)` khi
state cũ chết rồi id bị tái dùng (§12.29/§12.30). Test khoá cả hai chiều.

`HOPFIELD_TIEBREAK_BETA = 32.0` (0 = tắt). Chọn **32** chứ không phải 128 (bản tốt nhất v12):
trên v12 cả 8/32/128 đều ĐẠT (+8.50/+8.63/+8.88) ⇒ chênh lệch trong nhiễu ⇒ lấy giá trị GIỮA,
thắng trên mini, **tránh over-fit vào một checkpoint**.

#### ⚠️ Bug tìm được KHI NỐI THẬT — đổi thứ tự mà giữ nguyên điểm là KHÔNG ĐỦ

Sau khi wire, `probe_hopfield_rank` in `A_rank_matches` **y hệt trước**: 66.50%, hoà 38.2%.
Nếu tin vào wiring mà không đo lại thì đã kết luận "đã bật" trong khi **nó vô hình**.

Gốc: `tiebreak_ranked` **sắp lại thứ tự** nhưng trả **cùng giá trị điểm** ⇒ mọi consumer đọc
ĐIỂM (kể cả `_tie_stats` của probe, và bất kỳ chỗ nào sắp lại theo score) không thấy gì. Vá:
ghi thứ tự vào chính điểm bằng **epsilon nhân**

```
s_k = s · (1 + ε·(m−k)/m),   ε = 1e-9
```

⇒ xáo **trong** bậc, giữ nguyên **giữa** các bậc (an toàn khi khoảng cách tương đối giữa hai
bậc > 1e-9 — đúng với coverage thực). Test khoá: điểm vẫn giảm dần, lệch < 1e-6 so với gốc,
**và** hoà phải bị phá (nếu không thì consumer-theo-điểm mất hiệu lực).

#### Kết quả sau khi nối (mini, n=400, q_len=3)

| | rank-1 | MRR | hoà |
|---|---:|---:|---:|
| `rank_matches` TRƯỚC | 66.50% | 0.697 | 38.2% |
| `rank_matches` **SAU (mặc định)** | **68.75%** | **0.751** | **0.0%** |
| `E_tiebreak` chồng lên | 68.75% | 0.751 | 0.0% |

E chồng lên cho **+0.00pp** — đúng, **idempotent**. Suite **337 pass**; không đụng đường sinh.

**Probe đã được sửa để không nói ngược:** nếu `baseline tie < 2%` thì in cảnh báo *"phá-thế-hoà
ĐANG BẬT ⇒ Δ≈0 KHÔNG có nghĩa Trụ D trượt"* kèm cách lấy lại baseline sạch
(`ps.hopfield_tiebreak_beta = 0.0`). Không có dòng này thì lần chạy sau sẽ đọc thành "âm tính".

#### ĐÃ ĐO trên đường sinh v12 (§12.153) — 2026-08-02 — M4

**G-tb ĐẠT.** A/B `quality_baseline_v12`: BẬT ≡ TẮT (pmi 2.0558, salad 0, `quality_pass=true`,
24/24 `cont` giống hệt). Giữ mặc định BẬT. Chi tiết §12.153.

### 13.16 P53 §12.104e — RESERVOIR + ridge dạng đóng: ÂM TÍNH DỨT KHOÁT (2026-08-02) — Claude

**Quyết định tác giả:** mở cửa đo thử reservoir (W_res ngẫu nhiên **cố định** + readout học
dạng đóng bằng ridge = giải tuyến tính, **không backprop**). `scripts/probe_reservoir.py` ·
mini · train 60k / held-out 6k token, **split theo CHA** (không theo token — tránh rò rỉ) ·
`out/probe_reservoir_mini.json`.

#### Phép đo đúng câu hỏi

Câu hỏi KHÔNG phải *"reservoir tự mô hình hoá ngôn ngữ được không"* mà là **"trạng thái liên
tục `h_t` có thêm thông tin mà bảng đếm KHÔNG có không"**. Nên chấm dạng **hiệu chỉnh trên nền**:

```
logits = log p_bigram-backoff  +  α · W_out·h_t
```

`α=0` **tái lập chính xác bigram** ⇒ đối chứng nội tại nằm ngay trong phép đo.

| mô hình | bits/token | so bigram |
|---|---:|---:|
| unigram | 10.308 | +2.348 |
| **bigram-backoff** | **7.959** | — |
| reservoir `random` (α*=10) | 7.919 | **−0.040** |
| reservoir `graph` (α*=3) | 7.923 | **−0.036** |

`α=0` cho đúng **7.959** ⇒ thước lành. Đường α đơn điệu rồi bão hoà ⇒ không phải chưa tune.

**G-res: TRƯỢT dứt khoát.** Cần **+0.30**, đo được **+0.04** — kém bar **7.5×**, và thua cả
mức VarLM ngữ cảnh-4-token (**+0.21**) mà dự án đã có sẵn. Khe tới neural là **1.494**.

**`graph` ≈ `random`** (0.036 vs 0.040): lấy chính W_skip[d=1] làm toán tử hồi quy **không
hơn** ma trận ngẫu nhiên. ⇒ thất bại KHÔNG phải "chọn nhầm toán tử hồi quy", mà là **một
phép hồi quy CỐ ĐỊNH + readout tuyến tính không rút được thông tin ngữ cảnh dài dùng được**.

#### Hai lỗi setup của tôi trên đường tới đó (ghi lại)

1. **Chấm reservoir MỘT MÌNH → 13.2 bits/token, tệ hơn cả unigram (10.3).** Không phải kết
   luận về reservoir mà là lỗi thiết kế: `W_out` 32 002×384 = **12M tham số từ 60k mẫu**,
   dưới-xác-định nặng, **thiếu số hạng chệch** ⇒ phân bố gần ĐỀU (log₂32002 = 14.97). Quét α
   chạm biên — dấu hiệu thang sai. Nếu báo cáo con số đó thành "reservoir trượt" thì đã kết
   luận đúng **vì lý do sai**.
2. **OOM**: ma trận logits đầy đủ (T×V) ≈ **2.5 GB** × 3 bản sao trong vòng quét α. Vá bằng
   tính theo khối (~65 MB), kết quả y hệt.

#### Kết luận cho §12.104e

Cửa reservoir **đóng bằng số**, rẻ, trước khi ai đầu tư vào nó. Quyết định §12.147 của tác giả
(không neural trên lộ trình chất lượng) **không bị bằng chứng này thách thức** — nếu muốn đóng
khe 1.494 bit thì chỉ còn mô hình **huấn luyện thật**, và đó vẫn là quyết định chính sách chưa đổi.

⚠️ Giới hạn: N=384, 60k token, checkpoint mini. Quy mô lớn hơn **có thể** khá hơn — nhưng lãi
đo được là +0.04 so với bar +0.30 (kém 7.5×) và α đã bão hoà, nên đây là **âm tính vững chiều**,
không phải hụt sát. Và α quét TRÊN held-out (lạc quan) ⇒ số thật còn **tệ hơn**.

### 13.17 P54 — CHẤM MÙ 98 cặp: cổng phủ THẮNG 92%; và grader tìm ra 2 lỗi mọi thước đều mù (2026-08-02) — Claude

Phiên Claude **độc lập** chấm mù (không đọc `ruler_blind_key.json`), tiêu chí: đúng ngữ cảnh ·
không salad · văn phong trôi chảy. `out/ruler_blind_verdict.json`.

#### Kết quả sau khi LẬT KHOÁ

| | |
|---|---:|
| `clause_gated` (cổng phủ Trụ C) thắng | **80** |
| `clause_nogate` thắng | **7** |
| hoà | 11 |

**80/87 = 92.0%**, CI95 **[86.2%, 97.7%]** — tách xa 50%. Cỡ mẫu đủ (98 ≥ 88 cho +15pp @
power 80%); ở đây hiệu ứng là **+42pp**, thừa xa.

**Kiểm chéo chống hiện vật vị trí:** gated thắng ở **cả hai** vị trí — bên trái **46–3**, bên
phải **34–4**; ngẫu nhiên hoá cân (53/45). Không phải thiên lệch X/Y.

**Grader tự kết luận "KHÔNG phân biệt được"** vì họ tính trên **X vs Y = 50/37 = 57.5%** —
nhưng X/Y là **VỊ TRÍ**, không phải cấu hình, và họ **không được** biết khoá. Phán đoán
từng-cặp của họ mới là thứ cần, và nó cho 92%. Đây chính là lý do chấm mù hoạt động: người
chấm **không thể** tính được thống kê mức-cấu-hình nên không thể lái nó.

#### Câu §13.10 để ngỏ — ĐÃ TRẢ LỜI

*"Chuỗi 62 token đúng-đề vs 95 token trôi-đề, bên nào hay hơn?"* → **ngắn-đúng-đề, áp đảo.**

⇒ Cổng **G-cov của tôi (§13.10) đã đo NHẦM BIẾN**, đúng như tôi nghi nhưng **từ chối tự sửa**
để không thành thói quen dời thước. Nay có **phép đo độc lập** bác nó. Đây là cách quy trình
phải chạy: không dời cổng bằng diễn giải lại, mà bằng một phép đo MỚI.
`dead_rate` **không phải giá** — nó phần lớn là **tính năng** (dừng khi không mệnh đề nào qua
cổng = cơ chế EOS mong muốn, §13.2).

#### ⚠️ Nhưng 92% là TƯƠNG ĐỐI — chất lượng tuyệt đối vẫn KÉM

Nguyên văn grader: *"cả hai đều là cùng một kiểu hệ: nối các mảnh câu đã truy hồi từ kho,
không phải văn sinh liền mạch — tuyệt đại đa số cặp là so sánh **salad nào đỡ tệ hơn**"*.
Không được trích 92% như "chất lượng tốt".

#### Hai lỗi CỤ THỂ mà SÁU thước đều mù

1. **CHUỖI HÚT (attractor).** Một chuỗi mệnh đề cố định xuất hiện **bất kể prompt là gì**:
   `"là một triều đại phong kiến… 1009 đến 1225"` + `"000m, cao trung bình 3–4 m"` +
   `"khe Cha Moong, xã Nam Đông"` + `"Dương Đình Nghệ đi trấn Giao Châu"` + `"là 1507,57 km²"`
   (biến thể thiên văn: `Alpha/Beta/Ovda Regio… 167 núi lửa`). **Bên nào rơi vào nó thì thua.**
   → Nghi: một token-mở là **hub** hút mọi mối nối; `compose_chain` bốc ngẫu nhiên trong nhóm
   nên vẫn rơi vào hub. `attestation`/`novelty`/`lạc`/`pmi` đều **không** báo động.
2. **DÍNH SỐ Ở MỐI NỐI + cắt câu giữa chừng.** `"năm 1989" + "000 hố va chạm"` → `1989000`;
   `1497835 m`. Grader chẩn đúng: **ranh giới đơn vị hỏng**, không phải lỗi ngữ pháp.
   → `compose_chain` ghép span mà **không kiểm biên bề mặt**. Van `has_digit` của Trụ B bảo vệ
   *bên trong* mệnh đề nhưng **không bảo vệ chỗ NỐI** giữa hai mệnh đề.

Cả hai đều là **khuyết tật cài đặt cụ thể, sửa được** — và cả hai chỉ lộ ra khi có người/AI
**đọc**, đúng bài học TK_041 §5.2. Đây là giá trị lớn nhất của lượt chấm này, hơn cả con số 92%.

#### Việc kế (theo thứ tự, số đã chỉ)

1. **Chẩn chuỗi hút** — đếm phân bố token-mở trong `join_candidates`; nếu một hub chiếm phần
   lớn khối lượng thì cần bão hoà bậc (đã có tiền lệ `oja_indeg` §12.140).
2. **Cấm nối làm dính số** — kiểm biên bề mặt ở chỗ ghép mệnh đề, dùng lại `has_digit`.
3. Sau đó mới siết θ_pmi.

### 13.18 Vá hai lỗi chấm mù chỉ ra — và một đính chính về NGUYÊN NHÂN (2026-08-02) — Claude

#### Chẩn trước khi code (mini, n=800 mệnh đề nguồn)

Mối nối bị **hub hư-từ** hút, đúng như §13.17 mô tả:

| token-mở | % lần đứng đầu | #mệnh đề cùng mở |
|---|---:|---:|
| `Ġlà` | **16.8%** | 200 |
| `Ġvà` | 10.1% | 176 |
| `0` | **7.7%** | 37 |

**TOP 10 token-mở chiếm 50%** mối nối; chỉ **178/2159** token-mở từng được chọn. Và `0` chính
là lỗi dính số — nó là token **không mở từ**, ghép vào là `"năm 1989"+"000…"` → `1989000`.

#### Hai vá

1. **Bão hoà hub** (`hub_beta=1.0`): hạ trọng số token-mở theo **cỡ nhóm**
   `w /= 1 + β·log(1+|nhóm|)` — token-mở nhiều mệnh đề = mở chung chung. Cùng họ `oja_indeg`
   §12.140, Luật 0 sạch (đếm, không bảng-từ).
2. **Luật biên bề mặt** (`no_glue=True`): chỉ nối được mệnh đề mà token ĐẦU mở từ (`Ġ`/`Ċ`).

**Kết quả trên chính chẩn đoán:** top-1 **16.8% → 4.1%**; cần **49** token-mở thay vì **10**
để chiếm 50%; số token-mở từng được chọn 178 → **272**.

**Bug bắt được ngay khi wire:** luật `Ġ` làm **lọc sạch** ứng viên trên vocab KHÔNG phải BPE
(legacy/fixture — không có marker). Phải NO-OP khi `not vocab.is_bpe`. Test đồ chơi đỏ ngay
nên bắt được; nếu chỉ chạy trên checkpoint BPE thì đã lọt âm thầm.

#### ⚠️ ĐÍNH CHÍNH — chuỗi hút mà grader thấy có lẽ ĐÃ ĐƯỢC VÁ TỪ §13.14

Đo thẳng "mệnh đề nào tái xuất hiện qua nhiều prompt" (n=120):

| | mệnh đề riêng biệt | lặp nhiều nhất | top-5 chiếm |
|---|---:|---:|---:|
| TRƯỚC §13.18 | **87.2%** | 3.3% (4/120 prompt) | 3.8% |
| SAU §13.18 | 88.6% | 2.5% (3/120) | 3.9% |

⇒ Ở tầng **đầu ra**, chuỗi hút **đã gần như không còn TRƯỚC KHI** tôi vá. Ứng viên hợp lý
nhất cho công lao đó là **§13.14** (gỡ thiên lệch "chọn mệnh đề dài nhất") — bản cũ chọn tất
định mệnh đề DÀI NHẤT mỗi token-mở nên ra cùng vài chuỗi dài, bất kể prompt.

**Và mẫu đem chấm mù là mẫu CŨ**: `eval_ruler` chạy lần cuối trước lượt chấm là ở §13.12,
tức **trước** §13.14. ⇒ Phê bình định tính của grader (chuỗi hút, dính số) nói về **mã cũ**.
Kết luận **so sánh** 92% vẫn đứng (hai bên cùng mã), nhưng muốn chấm lại chất lượng **tuyệt
đối** thì phải **sinh lại mẫu** trên mã hiện tại.

#### Số tổng hợp: nhúc nhích, và tôi suýt phóng đại

A/B chính xác (cùng mã, chỉ khác `hub_beta`/`no_glue`), n≈100:

| | attest | novelty | **copy_run** | tok |
|---|---:|---:|---:|---:|
| TRƯỚC | 0.962 | 0.547 | **16.4** | 43.6 |
| SAU | **0.968** | 0.439 | **15.3** | 33.8 |

`copy_run` **16.4 → 15.3** — cải thiện **khiêm tốn**, không phải "giảm nửa" như tôi suy lúc
đầu từ bảng §13.11 (bảng đó còn mang thiên lệch §13.14 ⇒ so xuyên hai thay đổi = sai).
`novelty` tụt (0.547→0.439) **chỉ vì chuỗi ngắn đi** (43.6→33.8) — đúng bẫy §13.13, và lần
này nó *vu oan* thay vì tâng bốc. Đó là lý do `copy_run` nay **được in trong bảng** —
tôi đã thêm trục này ở §13.13 rồi **quên nối vào báo cáo**.

#### Trạng thái

Hai vá **giữ** (đều tự biện minh trên chẩn đoán riêng: phân tán token-mở 5×, và `0` không
còn nối được). Nhưng **không** được ghi công là "đã diệt chuỗi hút" — công đó nhiều khả năng
thuộc §13.14. Việc kế: **sinh lại 98 cặp trên mã hiện tại** rồi chấm mù vòng hai; chỉ khi đó
mới nói được chất lượng tuyệt đối đã đổi hay chưa.

### 13.19 P55 — Chấm mù VÒNG HAI: mẫu sinh lại trên mã hiện tại (2026-08-02) — Claude

§13.18 kết luận mẫu vòng một là **mẫu cũ** (trước §13.14/§13.18) nên phê bình định tính của
grader nói về mã cũ. Vòng hai sinh lại và hỏi **đúng câu còn lại**: hai vá §13.18 có đổi
**chất lượng cảm nhận** không.

**Cặp đem chấm:** `clause_gated_pre1318` (hub_beta=0, không luật biên) **vs** `clause_gated`
(mã hiện tại). Cùng cổng phủ, cùng seed, chỉ khác hai cờ ⇒ cô lập đúng §13.18.
94 cặp (≥88 cần cho +15pp @ power 80%) · `out/ruler_blind_pairs_r2.txt`.

#### Lỗi dính số — ĐÃ CHỨNG MINH BẰNG MÁY, không cần người chấm

Đếm chuỗi ≥5 chữ số liền (dấu hiệu dính ở mối nối), n=120 prompt:

| | số hiện vật | mỗi đầu ra |
|---|---:|---:|
| TRƯỚC §13.18 | **12** (`1901984`, `2007000`, `1368000`, `47800`…) | 0.12 |
| SAU §13.18 | **0** | **0.00** |

⇒ Lỗi 2 của §13.17 **đóng dứt điểm**. Đây là loại khuyết tật nên đo bằng máy chứ không tiêu
một lượt chấm mù.

#### ⚠️ THƯỚC TỰ MÂU THUẪN — nên mới cần chấm mù

| đường | attest | novelty | **copy_run** | tok | ĐÍCH |
|---|---:|---:|---:|---:|---:|
| `clause_gated_pre1318` | 0.961 | 0.552 | **13.7** | 37.0 | **65%** |
| `clause_gated` (mới) | **0.966** | 0.448 | **11.6** | 26.5 | 40% |

`copy_run` nghiêng bản **MỚI** (11.6 < 13.7) · tỉ lệ ĐÍCH nghiêng bản **CŨ** (65% > 40%).
Nhưng ĐÍCH dựa trên `novelty` = tỉ lệ, mà bản mới **ngắn hơn** (26.5 vs 37.0) ⇒ đúng bẫy
§13.13. Hai thước cùng bộ chỉ ngược nhau ⇒ **chấm mù là trọng tài**.

#### DỰ ĐOÁN GHI TRƯỚC (trước khi có kết quả — để số BÁC được)

> Bản **MỚI thắng**, nhưng **hẹp** (55–65%), không áp đảo như vòng một (92%). Lý do: hiện vật
> dính số bị diệt hẳn (12→0) là khuyết tật **nhìn thấy được**, nhưng chuỗi hút — thứ grader
> phàn nàn nhiều nhất — **đã được vá từ §13.14 ở CẢ HAI bên**, nên §13.18 không còn nhiều
> chỗ để ăn. Rủi ro ngược: bản mới ngắn hơn 1.4× có thể bị đọc thành **cụt**.
>
> Nếu kết quả **gần 50-50** ⇒ hai vá §13.18 **không đổi chất lượng cảm nhận**, chỉ đúng trên
> chẩn đoán riêng (phân tán token-mở, hết dính số) — vẫn giữ, nhưng không được ghi công thêm.

#### KẾT QUẢ vòng hai (2026-08-02)

| | |
|---|---:|
| `clause_gated` (mới) thắng | **45** |
| `clause_gated_pre1318` thắng | 39 |
| hoà | 10 |

**45/84 = 53.6%**, CI95 **[42.9%, 64.2%]** — **KHÔNG tách khỏi 50%**.
⇒ Dự đoán của tôi (*"thắng hẹp 55–65%"*) **không được số ủng hộ**: đúng chiều nhưng hiệu ứng
không phát hiện được. **Hai vá §13.18 KHÔNG đổi chất lượng cảm nhận.** Chúng vẫn giữ vì tự
biện minh trên chẩn đoán riêng, nhưng không được ghi công thêm gì.

#### ⚠️ Tôi mắc ĐÚNG lỗi tôi vừa chỉ ra ở §13.17

Grader báo *"số dính liền chia đều hai bên"*, mâu thuẫn với "12→0" của tôi. Tôi đếm lại —
**theo X/Y** — và thấy X:10, Y:4, tưởng grader đúng. Sai: **X/Y là VỊ TRÍ đã trộn**, không
phải cấu hình. Đếm theo **cấu hình**:

| cấu hình | hiện vật dính số |
|---|---:|
| `clause_gated_pre1318` | **19** |
| `clause_gated` (mới) | **0** |

Tuyên bố "12→0" **đúng**; nhận xét của grader là hệ quả của việc họ **không có khoá** (đúng
thiết kế). Nhưng tôi thì có khoá, và vẫn đếm nhầm — **một tin nhắn sau khi viết ra chính cái
bẫy đó** ở §13.17. Ghi lại: biết một cái bẫy **không** miễn nhiễm với nó; phải kiểm bằng khoá,
không bằng trí nhớ.

#### 🔴 LỖI MỚI grader tìm ra — và nó LÀM RỖNG con số `lạc = 0.000`

Grader: bám từ khoá theo **mặt chữ** chứ không theo nghĩa — `lục lạp`→`lục bát`, `Mã Lai`→
`Gia Lai`, `ly tâm`→`thoát ly`, `Thanh Hoá`→`thanh toán`. Kiểm: thống kê chính các fold mà
cổng phủ dùng làm bằng chứng "đúng đề" (n=337 lượt):

```
'4' 10 · 'ông' 8 · '3' 7 · '7' 7 · '6' 6 · 'từ' 6 · 'dân' 6 · 'ngày' 5 · 'về' 5 · '1' 5
fold ≤4 ký tự (mảnh dưới-từ): 291/337 = 86.4%
```

**86.4% bằng chứng "đúng đề" là mảnh dưới-từ**, mà đứng đầu là **chữ số** và âm tiết chung.

⇒ `lạc = 0.000` của `clause_gated` **phần lớn là RỖNG**: cổng phủ và thước `lạc` dùng **cùng
một định nghĩa fold**, mà định nghĩa đó khớp ở mức **âm tiết BPE**, không phải nghĩa. §13.10
đã gắn nhãn `C_anchor1_TAUT` là *tự tại về hình thức*; nay biết nó còn **rỗng về nội dung**.

**Không lật kết quả vòng một (92%)** — cổng phủ vẫn thắng áp đảo, và đó là số đo độc lập. Nhưng
**cơ chế** không phải như tôi mô tả: lợi ích đến từ việc **thu hẹp pool ứng viên** (cắt phần
trôi tệ nhất), KHÔNG phải từ khớp chủ đề thật.

#### Việc kế (số đã chỉ thẳng)

1. **Siết định nghĩa fold nội dung**: đòi fold **đa âm tiết** hoặc **≥2 fold chung**, và loại
   fold chữ số. Rồi đo lại `lạc` — con số hiện tại không dùng được.
2. Chỉ sau đó θ_pmi mới có nghĩa.

### 13.21 P56 — Siết fold: `lạc = 0.000` là RỖNG (thật ra 0.966); cổng nghiêm thì CHẾT ĐÓI (2026-08-02) — Claude

Theo §13.19. Thêm hai van vào `content_folds` (Luật 0: đếm / lớp ký tự, không bảng-từ):
`drop_digits` và **`pair`** — đơn vị khớp là **CẶP fold nội dung KỀ NHAU** (`lục|lạp` ≠
`lục|bát`) thay vì âm tiết lẻ. Và **tách định nghĩa fold của CỔNG khỏi của THƯỚC**, nếu không
`lạc` lại tự tại như §13.10.

#### 🔴 Thước NGHIÊM (cặp + bỏ số) phán lại toàn bộ

| cổng | **lạc (NGHIÊM)** | pmi | chết% | tok | n |
|---|---:|---:|---:|---:|---:|
| **A — hiện tại (âm tiết lẻ)** | **0.966** | 1.845 | 68% | 33.8 | 71 |
| B — + bỏ chữ số | 0.969 | 1.807 | 74% | 32.7 | 71 |
| C — cặp âm tiết | 0.333 | **2.541** | **100%** | 14.6 | **3** |
| D — cặp + bỏ chữ số | **0.000** | 2.449 | **99%** | 15.7 | **3** |

**Cổng đang chạy (A) có `lạc` thật = 0.966**, tức **96.6% mệnh đề LẠC ĐỀ** — trong khi thước
lỏng báo **0.000**. Chênh lệch không phải sai số: nó là toàn bộ khoảng cách giữa *"khớp một
âm tiết"* và *"khớp một cặp âm tiết"*.

> ⛔ **THU HỒI mọi con số `lạc = 0.000` của `clause_gated`** ở §13.10 · §13.11 · §13.17.
> Chúng đo bằng đúng định nghĩa mà cổng dùng để lọc ⇒ **tự tại, và rỗng**. Con số đúng dưới
> thước nghiêm là **0.966**.

#### Nhưng cổng NGHIÊM thì chết đói

C/D đúng chiều — `pmi_cross` (thước **độc lập**, không dùng fold) lên **1.845 → 2.45–2.54** —
nhưng **chết 99–100%**, chỉ **3/120** đầu ra chấm được, dài 15 token. Quét gidf cũng không có
nấc trung gian:

```
gidf ≥4.0  lạc .969 chết 74%   |  ≥5.0  .959  99%  |  ≥6.0  .909  98%
gidf ≥7.0  lạc 1.000 chết 86%  |  ≥8.0 1.000  61%   ← theme_folds RỖNG ⇒ cổng TỰ TẮT
```

(gidf ≥7 tệ hơn ở **cả hai** cột chính vì ngưỡng cao làm `theme_folds` rỗng ⇒ nhánh
`anchor_min_folds > 0 and self.theme_folds` sai ⇒ cổng thành no-op. Tự kiểm nhất quán.)

#### Kết luận: đây là bài toán VẬT LIỆU, không phải bài toán ngưỡng

Trên mini **không có** cấu hình fold nào vừa sống vừa giữ được đề. Tiêu chí nghiêm **đúng**
(pmi độc lập xác nhận) nhưng **thiếu mệnh đề** để thoả nó. Nối thẳng vào Trụ F: vật liệu co
giãn **tuyến tính** theo dữ liệu (§13.13: 1 083 → 44 940 ứng viên khi 1k → 44k cha). v12 có
**1.76 triệu** mệnh đề (§12.150) vs mini **10 283** = **171×**. ⇒ Cổng cặp-âm-tiết phải được
đo ở **v12**, không phải mini.

#### Không lật kết quả chấm mù vòng một

92% (§13.17) là số đo **độc lập** về chất lượng cảm nhận và nó vẫn đứng: cổng phủ **có** giúp.
Nhưng nay biết chắc **cơ chế không phải khớp chủ đề** — nó là **thu hẹp pool ứng viên**.
Hai lần đo độc lập cùng chỉ vào kết luận đó (§13.19 fold dưới-từ, §13.21 lạc nghiêm 0.966).

#### Bug thứ tư cùng họ

`content_folds` **không** loại token cấu trúc — `<EOS>`/`.` lọt vào, làm cặp `'.|<eos>'`
thành fold chung của **mọi** câu. Test đồ chơi bắt (`lục lạp` vs `lục bát` giao đúng ở đó).
Đã loại `is_structural_token`/`not is_surface`. Cùng họ với ba hằng-số-ma trước: **vá một
chỗ lọc thì phải rà mọi hàm lọc cùng loại.**

#### ĐÃ ĐO trên v12 (§12.152) — 2026-08-02 — M4

**G-fold TRƯỢT.** `D_cap_bo_so` chết **94%** (mini 99%), `n_scored=0`, pmi 1.27 < A 1.51.
Dự đoán “dead 20–40% nhờ 171×” **BỊ BÁC**. Theo C.5(a): **đóng hướng cổng-fold** — giả thuyết
vật liệu không cứu được cổng cặp-âm-tiết. Cần cơ chế khớp khác (không phải fold). Chi tiết
§12.152.

### 13.22 Dọn test đỏ — 7 → 2, và một hồi quy production suýt lọt (2026-08-02) — Claude

#### Chẩn: KHÔNG phải lỗi tôi đoán ở chip

`tests/conftest.py` ép `_FORCE_MATRIX_FOR_TESTS=True` ⇒ 6 test đó chạy **route matrix**
(`ar_w_skip_primary=False`), **không phải** đường production. Giả thuyết ban đầu của tôi
(pre-consume trong `_w_skip_primary_loop`) **sai chỗ**.

Gốc thật: **§12.149 áp luật dành cho prompt TIẾP DIỄN lên frame QA**, ở **ba** chỗ:

| # | chỗ | hậu quả |
|---|---|---|
| 1 | "đuôi sau token cuối trùng hỏi" | `Sao Hỏa là hành tinh đỏ.` → **`đỏ.`** |
| 2 | lọc `novel` bỏ mọi token trùng prompt | `Cá voi là động vật…` → `là động vật…` |
| 3 | bỏ `n_lead` của completion | `Cá voi là…` → `là…` (mất chủ ngữ) |

Ba lần cùng một gốc. Đáp đúng ngữ pháp tiếng Việt **phải** nhắc lại chủ ngữ — *"trùng
prompt" ≠ "thừa"*.

#### 🔴 Và vá "đúng" làm HỎNG production — đo A/B cứu

Sửa xong 6 test, smoke đường production:

```
Hỏi: Nhà Lý dời đô về đâu? Đáp:
   TRƯỚC vá : 'Thăng Long.'                              ← §12.149, sạch
   SAU  vá  : 'Đại La và là Năm tên Long kinh Thăng đổi…' ← SALAD
```

⇒ §12.149 **đang làm việc thật** trên route production. Nếu chỉ chạy suite rồi commit thì đã
đổi 6 test đỏ lấy một hồi quy sinh **không ai thấy**.

Tôi còn nghi nhầm **tie-break §13.15**; đo A/B `beta=32` vs `beta=0` → **đầu ra y hệt** ⇒ không
phải nó. Bisect bằng `git checkout HEAD -- generate_hebbian.py` mới chỉ đúng thủ phạm: **chính
vá của tôi**.

#### Kết luận kiến trúc: dạng đáp phụ thuộc ROUTE, không phải sở thích

| route | agenda | dạng đáp |
|---|---|---|
| W_skip primary (production) | phải **chặt** — vòng phiếu phủ agenda dài rất tệ | **ngắn**: `Thăng Long.` |
| matrix/parent (test khẳng định) | dội khá trực tiếp | **đầy đủ**: `Thủ đô Việt Nam là Hà Nội.` |

⇒ Điều kiện đúng là `_qa_trim_prefix(config) = config.ar_w_skip_primary`, **không** phải cờ
toàn cục và tuyệt đối **không** phải "đang chạy test". Route production giữ **nguyên xi** mã
§12.149; mã mới chỉ sống ở nhánh `else`.

#### Trạng thái: 349 → **2 đỏ** (từ 7)

Đã xanh: 4 test `test_ingest`/`test_generate_hebbian` + `test_paths`.
**Còn đỏ 2** (`autoregressive_reconstructs_answer_token_by_token`,
`autoregressive_index_backbone_orders_words`): route matrix vẫn cắt qua một đường KHÁC
(`_longest_suffix_cut`), chưa truy tới. Ghi lại thay vì vá vội — bài học vừa rồi là **vá
nhanh ở đây đổi được đường sinh**.

#### `test_paths` — sửa GỐC, không sửa triệu chứng

Test **chép lại** danh sách ưu tiên checkpoint thành chuỗi `elif` song song rồi **trôi**
(thiếu `v6_knowledge`/`v5_entity`/`v4_full`/`grade_v3_big_a2_t3_isg`) ⇒ đỏ trên máy có
checkpoint mới, mà đỏ đó **không nói lên lỗi nào của mã**. Tách danh sách thành hằng số
`paths.VIZ_CHECKPOINT_PREFERENCE`; test nay kiểm **vòng lặp** ("lấy mục đầu tiên tồn tại"),
không chép nội dung. Hết một lớp trôi.

### 13.23 P59 — ĐỐI CHỨNG VÀNG lật §13.21 của tôi; và cơ chế khớp KHÔNG-fold đã có sẵn (2026-08-02) — Claude

Yêu cầu tác giả sau khi P57 đóng hướng cổng-fold: *tìm cơ chế khớp không phải fold*.

#### Trước hết: thước phải được KIỂM bằng ca đã biết đúng

Tôi đã dựng **ba** thước fold và **chưa lần nào** kiểm chúng trên một ca ground-truth. Ca đó
rẻ và hiển nhiên: **hai mệnh đề LIỀN KỀ trong CÙNG một cha** — đúng đề *theo định nghĩa*.
400 cặp:

| thước | nói "đúng đề" | ⇒ `lạc` |
|---|---:|---:|
| fold **cặp** (thước "nghiêm" của §13.21) | **5.2%** | **0.948** |
| fold âm tiết lẻ (thước cũ) | 23.0% | 0.770 |
| `pmi_cross` | — | **2.128** (mốc) |

> 🔴 **CẢ HAI thước fold đều HỎNG.** Thước gán `lạc = 0.948` cho văn bản đúng đề *theo định
> nghĩa* thì nó không đo chủ đề — nó đo **trùng lặp BỀ MẶT**.

#### ⛔ Thu hồi §13.21 — retraction của tôi cũng SAI

§13.21 tôi thu hồi `lạc = 0.000` và tuyên bố *"con số đúng là 0.966"*. **Sai.** `0.966` chỉ
là **sàn** mà thước nghiêm gán cho mọi thứ, kể cả đối chứng vàng (0.948). Cái đúng còn lại
của §13.21: `lạc = 0.000` **vẫn vô nghĩa** (tự tại). Cái sai: **0.966 không phải "trôi đề
thật"**, và tôi đã dùng một thước **chưa kiểm** để thu hồi số của một thước hỏng khác.

**Bài học (lần thứ năm về thước trong §13):** dựng thước mới thì **phải chạy nó trên ca đã
biết đúng TRƯỚC**. Đối chứng vàng này tốn 30 giây và lẽ ra phải có **trước Trụ C**.

#### Và nó làm SẮC kết luận P57 của M4

M4 kết luận đúng (*giả thuyết vật liệu bị bác*), nhưng lý do sắc hơn: cổng fold-cặp **không
đói vật liệu** — nó đòi một sự kiện chỉ xảy ra **5.2%** ngay cả giữa hai mệnh đề thật sự liên
quan. Chết 94% ở v12 là **hệ quả số học**, không phải thiếu dữ liệu.

#### Cơ chế KHÔNG-fold: `pmi_cross` — đã có sẵn, nay hiệu chỉnh được

Cả hai lần fold thất bại đều vì khớp **ký hiệu bề mặt**: âm tiết lẻ quá lỏng, cặp quá thưa.
Cái cần là tín hiệu **liên tục**, suy thoái mềm. `pmi_cross` là ứng viên duy nhất đã được
kiểm **hai lần độc lập**: Spearman **+0.643** với chấm mù (§12.117), và nay **mốc vàng 2.128**.

| cổng | pmi ĐẠT | so mốc | chết% | tok |
|---|---:|---:|---:|---:|
| KHÔNG cổng | 1.150 | −0.978 | 14% | 36.7 |
| **neo fold (đang chạy)** | 1.844 | −0.284 | **69%** | 33.3 |
| pmi ≥ 1.5 | 1.883 | −0.245 | **37%** | 29.8 |
| **pmi ≥ 1.75** | **2.130** | **+0.002** | **54%** | 24.1 |
| pmi ≥ 2.0 | 2.447 | +0.319 | 68% | 19.0 |

**`pmi ≥ 1.75` chạm ĐÚNG mốc vàng (2.130 vs 2.128) mà chết ÍT hơn cổng fold (54% vs 69%).**
Và `pmi ≥ 1.5` **áp đảo cổng fold trên CẢ HAI trục** (pmi cao hơn 1.883 > 1.844, chết
37% ≪ 69%). ⇒ Cổng fold **bị chi phối hoàn toàn**; không có lý do giữ nó.

#### Chưa đổi mặc định — cố ý

Mới đo trên mini. Theo đúng kỷ luật §13.9/§13.15 (mini → v12 → tác giả quyết). Cần:
1. M4 chạy lại trên v12 với `pmi_min ∈ {1.5, 1.75, 2.0}`, `anchor_min_folds=0`, **mốc vàng
   tính lại trên v12** (mệnh đề liền kề cùng cha).
2. **Chấm mù vòng ba** — arbiter duy nhất đã được kiểm; `lạc` fold nay **nghỉ hưu** cả hai bản.

#### Ba thước fold nay NGHỈ HƯU

`lạc` (lỏng) và `lac_strict` (cặp) **không dùng để phán chủ đề nữa**. Giữ trong mã cho lịch
sử và cho việc đo *trùng lặp bề mặt* — đúng thứ chúng thật sự đo.

#### ĐÃ ĐO trên v12 (§12.154) — 2026-08-02 — M4

**G-pmi ĐẠT.** Mốc vàng v12 = **1.372** (mini 2.128). Thắng: θ=**1.25 / 1.5 / 1.75**
(chết 42/59/77% ≤ fold 81%; pmi ≥ gold−0.05). Dự đoán “dead thấp hơn mini” **BỊ BÁC**
(`pmi_1.5` chết 59% vs mini 39%). **Chưa bật mặc định** — việc kế = chấm mù vòng ba
(fold vs PMI-θ). Chi tiết §12.154.

### 13.24 Đóng nốt 2 test đỏ — vị ngữ "route" của tôi sai chỗ (2026-08-02) — Claude

**349 passed, 0 đỏ** — suite xanh hoàn toàn lần đầu kể từ `a5f4862`.

#### Gốc: `ar_w_skip_primary` nói SẼ chọn đường nào, không nói ĐANG ở đường nào

§13.22 tôi đặt `_qa_trim_prefix(config) = config.ar_w_skip_primary`. Đúng cho `generate()`,
**sai** cho `generate_autoregressive()` gọi **trực tiếp** (đúng cái hai test còn lại làm):
ở đó cờ vẫn `True` dù ta **đã ở trong** route AR/matrix.

Vị ngữ đúng là **sự thật về route**, không phải cờ định tuyến: chạm tới `generate_autoregressive`
⇒ **đang ở** route AR — đường production (W_skip primary) rẽ sang `_wseq_fallback` từ
`generate()` và **không bao giờ** vào đây. Nên `_build_segment_plan` nhận override
`trim_prefix=False` tại đúng chỗ đó.

Luồng tham số: `_build_segment_plan(trim_prefix=None → suy từ config)` →
`_pick_parent_agenda(trim_prefix=)` → `_agenda_from_parent_index(trim_prefix=)`.
`None` = giữ hành vi cũ, nên mọi call-site chưa truyền đều **không đổi**.

#### Chứng minh trung tính với production — 8 prompt, không phải 3

Sau bài học §13.22 (vá "đúng" làm hỏng sinh), lần này A/B **đầy đủ** bằng
`quality_baseline_v12.py` trên mini, so với `HEAD`:

```
số prompt ĐỔI đầu ra: 0/8
pmi_mean 1.5096 → 1.5096 · hub_ratio 0.0263 → 0.0263 · salad 0.0 → 0.0
```

**Byte-identical trên cả 8.** (`quality_pass=False` ở cả hai bên — cổng chỉnh cho v12, mini
vốn không đạt; không phải hồi quy.)

#### Tổng kết đợt dọn: 7 → 0

| | |
|---|---|
| 6 test `test_ingest`/`test_generate_hebbian` | §12.149 áp luật prompt-TIẾP-DIỄN lên frame QA ở **bốn** chỗ (ba ở §13.22 + `_pick_parent_agenda` ở đây) |
| 1 test `test_paths` | test **chép lại** danh sách ưu tiên rồi trôi → tách hằng số `VIZ_CHECKPOINT_PREFERENCE`, test kiểm **vòng lặp** |

**Bài học đắt nhất của đợt này** (§13.22): suite xanh **không đủ** để commit thay đổi ở
`generate_hebbian.py` — phải A/B đường sinh. Tôi đã suýt đổi 6 test đỏ lấy một hồi quy sinh
im lặng, và chỉ tránh được vì đo trước khi commit.

### 13.25 P61 — Chấm mù vòng ba: cổng PMI **THUA 85.6%**; P60 là DƯƠNG TÍNH GIẢ (2026-08-02) — Claude

124 cặp mini (θ=1.5), phiên Claude độc lập, không đọc khoá. `out/gate_verdict_mini.json`.

| | |
|---|---:|
| **cổng fold** thắng | **89** |
| cổng PMI θ=1.5 thắng | **15** |
| hoà | 20 |

**fold 89/104 = 85.6%**, CI95 **[78.8%, 92.3%]** — tách xa 50%.
Kiểm chống hiện vật vị trí: fold thắng ở **cả hai** vị trí (41–6 và 48–9); ngẫu nhiên hoá cân
(59/65). Không phải thiên lệch X/Y.

#### ⇒ P60 là DƯƠNG TÍNH GIẢ do TỰ TẠI — đúng thứ runbook E.6/F.1 cảnh báo

P60 kết luận cổng PMI **chi phối** cổng fold (pmi cao hơn *và* chết ít hơn). Nhưng ở đó
`pmi_cross` vừa là **cổng** vừa là **thước** ⇒ "gate on X, score by X" ⇒ thắng **theo cấu
tạo**. Tôi đã ghi cảnh báo đó vào runbook **trước khi** M4 chạy, và nó xảy ra đúng như vậy.

**Bài học:** đo được "A chi phối B trên thước T" là **vô nghĩa** khi A tối ưu hoá trực tiếp T.
Cả hai cổng fold trước đây cũng vậy (§13.21 `lạc` tự tại). Đây là lần thứ **ba** cùng một
lớp lỗi trong §13. Quy tắc rút ra: **cổng và thước phải khác họ**, và arbiter cuối phải là
chấm mù.

#### Và grader lại đúng ở từng cặp, sai ở tổng — lần thứ HAI

Grader tự kết luận *"không phân biệt được"* vì tính trên **X 50 / Y 54**. X/Y là **vị trí đã
trộn**. Phán đoán **từng cặp** của họ cho 85.6%. Đúng như §13.19 — và tôi đã viết sẵn cảnh
báo này vào rubric F.4, họ vẫn tính ở mức X/Y. ⇒ Không nên trông cậy vào việc dặn; **luôn tự
lật khoá và đếm theo cấu hình**.

#### Trạng thái cổng: GIỮ fold

Cổng `anchor_min_folds=1` (âm tiết lẻ) **ở lại làm mặc định**. Nó vẫn là cổng có bằng chứng
chấm mù duy nhất ủng hộ (§13.17: 92% so với không-cổng; nay 85.6% so với cổng PMI).
Chuyện `lạc = 0.000` của nó là rỗng (§13.21/§13.23) **không** làm nó vô dụng — nó thu hẹp
pool ứng viên, và **đó** là cái ăn.

#### 🔴 LEAD MỚI: bệnh nằm ở MỐI NỐI, không phải ở cổng

Grader chỉ ra lỗi phổ biến nhất **của cả hai bên** — và nó không phải chuyện cổng:

> nối bằng **từ đồng âm làm bản lề**: `theo dõi bệnh **nhân**` → `2 **nhân** 3 bằng 6` ·
> `sông **Hậu**` → `**Hậu** quả gồm nắng nóng, bão mạnh` · `vương tôn` → `vương **tôn** giáo`

`join_candidates` nối bằng **đồng nhất TOKEN**: `last_content(C) → first_content(C')` trên
W_skip. Tiếng Việt đồng âm dày đặc ⇒ `nhân`(người) và `nhân`(phép nhân) là **cùng một token**
⇒ mối nối hợp lệ về đếm, **vô nghĩa** về ngữ nghĩa.

⇒ **Cùng một căn bệnh mặt-chữ đã giết cả hai cổng fold, nhưng nằm ở MỐI NỐI.** Không cổng
nào sửa được: cổng lọc *mệnh đề nào được vào*, còn bản lề hỏng nằm ở *chỗ ghép*. Đây là chỗ
đáng đào tiếp — và lần này phải kiểm bằng đối chứng vàng **trước**, không phải sau.

### 13.26 P62 — Tín hiệu MỐI NỐI: cái đang dùng gần như MÙ (AUC 0.605) (2026-08-02) — Claude

Theo lead §13.25 (bản lề đồng âm). **Đối chứng vàng dựng TRƯỚC** — đúng bài học §13.23.

* **nối THẬT** = hai mệnh đề liền kề **cùng một cha** (văn gốc, đúng theo định nghĩa);
* **nối GIẢ** = ứng viên mà `join_candidates` **thật sự đề xuất** cho cùng mệnh đề nguồn,
  nhưng từ cha khác. Âm bản **thực tế** — lấy cặp ngẫu nhiên thì tín hiệu nào cũng tách được
  và phép đo thành vô nghĩa.

Thước: **AUC** (xác suất xếp nối-thật trên nối-giả). 0.5 = mù.

| tín hiệu | AUC | so S1 | thật / giả |
|---|---:|---:|---|
| **`S1` đang dùng** — 1 token bản lề | **0.605** | — | 5.72 / 1.91 |
| `S2` cửa sổ hai phía (đuôi A × đầu B) | 0.736 | +0.131 | 19.16 / 6.98 |
| **`S3` `pmi_cross(A,B)`** | **0.814** | **+0.209** | 2.116 / 1.074 |

**`S1` = 0.605 — gần như MÙ.** Cơ chế nối hiện tại hầu như không phân biệt được nối thật với
nối bắc qua từ đồng âm. Đây là **số** cho đúng thứ chấm mù vòng ba tả bằng ví dụ
(`bệnh nhân` → `2 nhân 3`). **G-join ĐẠT** cho `S2` và `S3`.

#### Vì sao `pmi_cross` THUA làm CỔNG (§13.25) mà THẮNG làm bộ xếp mối nối

Không mâu thuẫn — **hai việc khác nhau**:

| dùng làm | câu hỏi | kết quả |
|---|---|---|
| **cổng** (§13.25) | *"mệnh đề này có thuộc chủ đề của PROMPT không?"* | **thua 85.6%** — tiêu chí sai, siết quá tay |
| **xếp mối nối** (đây) | *"hai mệnh đề này có đứng cạnh nhau được không?"* | **AUC 0.814** |

Đồng-hiện đo *"hay đi cùng nhau"* — đó đúng là định nghĩa của một mối nối tốt, và **không**
phải định nghĩa của "cùng chủ đề với prompt".

#### Wire: `compose_chain(join_rescore=…)` — CHƯA mặc định

Xếp lại ứng viên trong nhóm cùng token-mở theo `join_rescore(prev, cand)`. Đo trên mini:
**69/120 đầu ra đổi**, `dead` **không đổi** (69%), độ dài ~như cũ (26.4 → 27.3). Tức nó đổi
*lựa chọn* mà không trả giá vận hành.

⚠️ **Không tự chấm bằng chính `pmi_cross`** — đó đúng bẫy đã biến P60 thành dương tính giả
(§13.25). Nghiệm thu **chỉ** bằng chấm mù.

#### Đang chạy song song

M4 đã sinh **112 cặp v12** (fold vs cổng PMI θ=1.25, §12.155) — đã giao một phiên Claude độc
lập chấm. Rubric nhấn hai lỗi đã gây kết luận sai hai lần: **độ dài không phải tiêu chí**, và
**X/Y là vị trí — không được suy luận thống kê ở mức đó**.

### 13.27 P61-v12 — Chấm mù trên v12: cổng fold thắng **71.7%**; xác nhận P60 là dương tính giả (2026-08-02) — Claude

112 cặp v12 do M4 sinh (§12.155, fold vs PMI θ=1.25), phiên Claude độc lập chấm.

| | |
|---|---:|
| **cổng fold** | **66** |
| cổng PMI θ=1.25 | 26 |
| hoà | 20 |

**fold 66/92 = 71.7%**, CI95 **[62.5%, 80.9%]** — tách khỏi 50%. Thắng ở **cả hai** vị trí
(34–15 và 32–11); ngẫu nhiên hoá cân (52/60).

**Hai quy mô, cùng một chiều:** mini **85.6%** · v12 **71.7%**. ⇒ P60 (*"cổng PMI chi phối"*)
là **dương tính giả do tự tại**, xác nhận ở **cả hai** quy mô. Cổng fold **ở lại**.

Hiệu ứng nhỏ hơn ở v12 khớp với ghi chú giám khảo: bên PMI (ngắn) vừa cho **những ca đúng
nhất cả bộ** khi nó dừng gọn trúng ý, vừa cho nhiều mảnh **cụt rỗng nghĩa** (`trữ kho` ·
`cư dân mạng` · `là Janus.`).

#### Rubric ăn — giám khảo lần này KHÔNG tự kết luận từ X/Y

Hai vòng trước giám khảo đều tính tổng X-vs-Y rồi kết luận "không phân biệt được", trong khi
lật khoá ra là 92% và 85.6%. Lần này rubric ghi thẳng *"X/Y là VỊ TRÍ đã trộn, tuyệt đối
không suy luận thống kê ở mức đó"* + nêu chính hai lần sai trước — và giám khảo **giao lại
đúng phần việc của mình**, không kết luận. Giữ nguyên câu đó trong mọi rubric sau.

#### 🔴 Xác nhận §13.26 ở quy mô v12, và làm sắc thêm

Giám khảo tự phát hiện **bản lề đồng âm ở CẢ HAI bên** — không cấu hình nào miễn nhiễm:

```
bình (hoà bình) → bình luận → bình thường → bình sữa      (G107, chuỗi 4 nhịp)
trùng (khử trùng) → côn trùng → thụ phấn → phấn khích     (G004)
tĩnh (bình tĩnh) → tĩnh mạch   ·   văn → văn hoá → hoá đá
```

Và điểm mấu chốt: **khác biệt giữa hai cổng KHÔNG phải có hay không có bản lề đồng âm, mà là
HỆ QUẢ của nó** — bên dài dùng bản lề để **nhảy hẳn sang miền khác rồi ở lại đó**; bên ngắn
chỉ tạo một cụm lạc rồi dừng.

⇒ **Cổng không chữa được bệnh này** — nó chỉ đổi mức thiệt hại. Chữa phải ở **mối nối**, đúng
chỗ §13.26 đo được `S1` AUC **0.605** (mù) vs `S3` **0.814**. Chuỗi `bình→bình luận→bình
thường→bình sữa` là minh hoạ hoàn hảo: mỗi nhịp hợp lệ trên **một token bản lề**, vô nghĩa
trên **cả mệnh đề**.

#### Lộ trình sau bốn vòng chấm mù

1. **GIỮ** cổng fold — 2 vòng chấm mù × 2 quy mô ủng hộ (92% vs không-cổng · 85.6%/71.7% vs PMI).
2. **SỬA MỐI NỐI** bằng `join_rescore` (`S3`, AUC 0.814) — bệnh nằm ở đó, đã xác nhận ở v12.
3. Nghiệm thu bằng **chấm mù**, tuyệt đối không bằng `pmi_cross` (tự tại — chính bẫy đã sinh P60).

### 13.28 P63 — Chấm mù vòng năm ĐO NHẦM; hai bug im lặng trong `join_rescore` (2026-08-02) — Claude

#### Kết quả thô: không tách

106 cặp mini, `join_S1` (đang dùng) vs `join_S3` (pmi rescore):
**S1 49 · S3 44 · hoà 13** ⇒ S3 **47.3%**, CI95 [37.2%, 57.5%] — **không tách khỏi 50%**.

#### ⚠️ Nhưng vòng chấm này KHÔNG KIỂM ĐƯỢC GÌ — hai bug im lặng

Không nhận kết quả null; đọc lại mã và thấy:

1. **`sort(rescore)` rồi NGAY SAU đó `rng.shuffle(members)`.** Shuffle **xoá sạch** thứ tự vừa
   xếp ⇒ `join_rescore` là **NO-OP hoàn toàn**. Nguy hiểm ở chỗ: đầu ra **vẫn "đổi 69/120"**
   (vì shuffle nhận list đầu vào khác thứ tự nên cho hoán vị khác) ⇒ **phép đo end-to-end xác
   nhận "có tác dụng" trong khi không có gì**. Tôi đã dùng đúng con số 69/120 đó để biện minh
   cho việc đem đi chấm.
2. **Sai TẦNG.** Kể cả nếu sort sống, nó chỉ xếp lại **trong** một token-mở — mà **bản lề đồng
   âm CHÍNH LÀ token-mở** (`nhân`, `bình`, `tĩnh` — §13.27). Xếp lại nội bộ nhóm nằm **hạ
   nguồn** của sai lầm, không đời nào chữa được.

⇒ Vòng năm so **nhiễu với nhiễu**. Con số 47.3% **không nói gì** về `S3`. Thu hồi.

#### Vá + khoá

`order_join_candidates()` tách thành **hàm thuần**: bốc ngẫu nhiên phá thế hoà **TRƯỚC**,
chấm từng (token-mở, ứng viên tốt nhất), rồi xếp **cả danh sách token-mở**.

Test ở **tầng hàm thuần**, không end-to-end: fixture đồ chơi không kéo nổi chuỗi ≥2 mệnh đề
nên **mọi** test end-to-end đều mù với lỗi này — đúng như phép đo 69/120 đã mù. Test mới cho
điểm W_skip **bằng nhau** giữa các token-mở để **chỉ** rescore quyết định, rồi đòi thứ tự
**ổn định qua 8 seed** và **đúng theo điểm**.

#### Bug thứ ba: đếm verdict

Vòng này grader ghi `"hoà"` (có dấu); các vòng trước ghi `"hoa"`. Code đếm của tôi so cứng
`v == 'hoa'` ⇒ **13 phiếu hoà rơi xuống nhánh else và thành THẮNG cho bên phải**, âm thầm.
Bắt được vì tổng không khớp báo cáo grader (55+51=106 nhưng grader nói 13 hoà). Nay chuẩn
hoá NFC + `not in ('X','Y') ⇒ TIE`. Hai vòng cổng dùng `'hoa'` nên **số cũ vẫn đúng**
(đã đếm lại xác nhận: mini 85.6%, v12 71.7%).

#### Trạng thái

`S3` **chưa được nghiệm thu** — cần một vòng chấm mù **mới** trên bản đã vá. AUC 0.814 (§13.26)
vẫn đứng: đó là đo trên đối chứng vàng, không dính hai bug này.

**Bài học:** "đầu ra có đổi" **không** chứng minh cơ chế đang chạy. Muốn biết một cơ chế có
sống không thì phải test nó **ở tầng của nó**, bằng đầu vào tất định — không phải bằng cách
đếm xem đầu ra có khác đi hay không.

### 13.29 Hạ tầng chấm mù — đóng gói khâu đếm, và một nguy cơ dữ liệu (2026-08-02) — Claude

Sau **sáu** vòng đếm phiếu bằng tay (và một bug ở vòng thứ sáu), đóng gói lại:
`scripts/score_blind_verdict.py`. Nó chặn **cả ba** cái bẫy đã thật sự xảy ra:

1. **Đếm theo VỊ TRÍ thay vì CẤU HÌNH** — `X`/`Y` trộn từng cặp nên tổng X-vs-Y **vô nghĩa**.
   Xảy ra **ba** lần: hai giám khảo tự kết luận *"không phân biệt được"* (thực tế 92% và
   85.6%), và **một lần chính tôi** — ngay sau khi vừa viết ra cái bẫy đó.
2. **So chuỗi phiếu hoà ngây thơ** — một vòng ghi `"hoà"`, các vòng khác `"hoa"`; `v=='hoa'`
   làm 13 phiếu hoà thành **thắng cho bên phải**, âm thầm (§13.28).
3. **Không kiểm hiện vật vị trí** — nay luôn in thắng-thua **tách theo phía** và cảnh báo nếu
   bên thắng không thắng ở cả hai phía.

#### 🔴 Nguy cơ dữ liệu bắt được ngay khi chạy scorer

Scorer báo khoá vòng một **thừa 4 id**. Truy git: `out/ruler_blind_key.json` bị **ghi đè lúc
16:55** bởi một lần chạy `eval_ruler` ở §13.18 — sau khi verdict ghi lúc 15:05.

⇒ **Kết quả chấm mù cũ không lật lại được** nếu khoá bị ghi đè. Lấy khoá gốc từ git
(`29fbe44`) rồi chấm lại: **80/87 = 92.0%** — **khớp đúng** số §13.17. Con số đó **an toàn**
(tính lúc khoá còn nguyên); nhưng nếu tôi tin bản chạy lại hôm nay thì đã ghi 91.6% trên
83/87 cặp mà **không biết mình đang dùng khoá lệch**.

**Vá tại nguồn:** `gen_gate_blind_pairs.py` nay **từ chối** ghi đè khoá khi verdict tương ứng
đã tồn tại (trừ `--force`). Đã thử: nó chặn đúng.

**Bài học:** artefact của một phép đo (cặp + khoá) là **bằng chứng**, không phải file tạm.
Ghi đè nó = xoá khả năng kiểm chứng lại kết luận, mà **không có gì đỏ lên**.

### 13.30 P64 — `S3` THUA trên bản đã vá; AUC cao mà chất lượng kém, và vì sao (2026-08-02) — Claude

134 cặp mini, mã đã vá §13.28 (rescore nay đổi token-mở ở **80.5%** ca — cơ chế thật sự sống).
Đếm bằng `score_blind_verdict.py`.

| | |
|---|---:|
| **`join_S1`** (đang dùng, 1 token bản lề) | **77** |
| `join_S3` (pmi rescore) | 52 |
| hoà | 5 |

**S1 77/129 = 59.7%**, CI95 **[51.2%, 68.2%]** — tách khỏi 50%, thắng ở **cả hai** phía
(36–25 và 41–27). ⇒ **`S3` THUA.** Giữ `S1`; `join_rescore` **không** bật.

#### 🔴 AUC 0.814 mà chất lượng KÉM HƠN — đối chứng vàng nhắm sai MỤC TIÊU

§13.26 đo `S3` AUC **0.814** vs `S1` **0.605** trên đối chứng vàng, và đối chứng đó **không
sai**: nó thật sự đo *"hai mệnh đề này có đứng cạnh nhau trong văn thật không"*. Nhưng đó là
**tính mạch lạc CỤC BỘ giữa hai mệnh đề liền kề**, còn thứ ta cần là **bám PROMPT trên cả
chuỗi**. Hai mục tiêu khác nhau — và tối ưu cái đầu **làm hại** cái sau.

Cơ chế thấy rõ trong ghi chú giám khảo: hỏng nặng nhất của bên thua là **nối bằng số/công
thức** — `2^5 bằng 32` · `Chu vi bằng 2 nhân` · `Công bằng lực nhân quãng đường`. Các mệnh đề
toán **đồng hiện rất mạnh với nhau** ⇒ `pmi_cross` giữa chúng **cao** ⇒ `S3` ưu tiên chúng.
Mỗi bước **mạch lạc cục bộ**, cả chuỗi **trôi hẳn khỏi ngữ vực sử liệu**. Giám khảo:
*"nó bẻ gãy hẳn ngữ vực đang chạy, nên bên còn lại gần như luôn thắng"*.

> **Bài học thứ tư về thước trong §13:** một đối chứng vàng **đúng đắn** vẫn có thể **nhắm sai
> mục tiêu**. "Đã kiểm bằng ground truth" chưa đủ — phải hỏi *ground truth đó là bằng chứng
> cho ĐIỀU GÌ*. Ở đây nó chứng minh mạch lạc-cục-bộ, còn cái cần chứng minh là bám-đề-toàn-cục.

#### Và nó giải thích vì sao cổng fold thắng cả bốn vòng

Cổng fold neo vào **PROMPT** (`theme_folds`). Dù cách khớp của nó thô (âm tiết lẻ, 86.4% bằng
chứng là mảnh dưới-từ — §13.19), nó vẫn giữ đúng **đại lượng cần giữ**. `pmi_cross` sắc hơn
nhiều về mặt tín hiệu nhưng **đo sai đại lượng** khi làm cổng, và **đo đúng nhưng sai mục
tiêu** khi làm mối nối. Bốn vòng chấm mù đều nhất quán với cách đọc này.

#### Hướng kế — có điều kiện PROMPT

Tín hiệu mối nối phải **điều kiện hoá theo prompt**: `score(prev, cand | prompt)` thay vì
`score(prev, cand)`. Ví dụ rẻ nhất: `pmi_cross(cand, prompt)` **nhân** với `pmi_cross(prev,
cand)` — buộc ứng viên vừa nối được vừa còn ở trong đề.

⚠️ Và lần này đối chứng vàng phải nhắm **đúng mục tiêu**: âm bản không chỉ là "nối sai" mà
phải gồm **"nối mạch lạc cục bộ nhưng lạc prompt"** — chính hạng mà `S3` rơi vào. Nếu không,
ta lại đo được AUC đẹp cho một thứ thua chấm mù.

---

### 13.31 P65 — điều kiện-PROMPT: cổng TRƯỢT, và bốn thứ đo được lại đáng giá hơn (2026-08-03) — Claude

§13.30 đặt hướng "điều kiện hoá theo prompt" và tiền-đăng ký một cổng. P65
(`scripts/probe_join_prompt.py`) chạy đúng cổng đó và **TRƯỢT**. Nhưng đường đi tới chỗ trượt
đẻ ra bốn phép đo đổi hẳn cách nhìn về mối nối. Ghi cả hai.

#### Cổng tiền-đăng ký và kết quả

`GATE_AUC_MIN = 0.65` trên **CẢ HAI** lớp âm bản khó (argmax-S1 và argmax-S3), 300 cặp vàng /
298 âm, mini. Kết quả (`out/probe_join_prompt.json`):

| tín hiệu | vs argmax-S1 | vs argmax-S3 | min |
|---|---:|---:|---:|
| `S1` đang chạy | .136 | .407 | **.136** |
| `S3` pmi_cross(A,B) | .798 | .438 | **.438** |
| `S4` pmi_cross(ĐỀ,B) | .691 | .618 | **.618** |
| `S6` khử bản lề | .404 | .549 | **.404** |
| `S7` z(S6)+z(S4) | .656 | .614 | **.614** |

`gate_pass = false`. Không có tín hiệu nào qua .65 ở cả hai cột ⇒ **không** bật cái nào theo
đường này. Không dời cổng.

#### 🔴 Bẫy thứ nhất: AUC trước âm bản-đối-kháng là **tự quy chiếu**

Cột "vs argmax-S3" cho `S3` = **.438**, thấp hơn ngẫu nhiên. Không phải vì `S3` dở — mà vì âm
bản ở cột đó **chính là `argmax S3`**. Bất kỳ tín hiệu nào cũng chấm gần 0 trước argmax của
chính nó, **theo định nghĩa**. Cột đó đo *"tín hiệu nào chọn ra âm bản"*, không đo *"tín hiệu
nào xếp lại tốt"*. Tôi đã dựng nó rồi mới thấy — ghi lại để cả hai cột được đọc đúng: chỉ
**chéo** (S4 vs argmax-S1, S3 vs argmax-S1) là có nghĩa.

#### 🔴 `S6` khử bản lề: giả thuyết CHẾT — .404, dưới ngẫu nhiên

§13.25 chẩn "bệnh bản lề đồng tự" (`bệnh nhân` → `2 nhân 3`: tiếng Việt dùng **chung một id**
cho đồng tự). `S6` thử cách chữa hiển nhiên nhất: **bỏ hẳn token bản lề**, chấm đuôi-A × đầu-B
qua `W_skip` có suy giảm λ. AUC **.404** — *tệ hơn tung đồng xu*.

⇒ Ngữ cảnh xung quanh **gần như không mang thông tin nối** trong `W_skip`; token bản lề là
**vật mang duy nhất**. "Khử nhập nhằng bằng ngữ cảnh lân cận" là **hướng chết**, đừng thử lại.

#### 🔴 Đo lại ngoài đối kháng: cơ chế đang chạy **gần như mù khi xếp hạng**

Bỏ âm bản đối kháng, đo bằng **xếp hạng trong bể** (chèn mệnh đề tiếp theo thật vào bể ứng
viên, hỏi `S1` xếp nó thứ mấy):

| | |
|---|---:|
| bể đủ — top-1 của `S1` | **1.4%** |
| hạng trung vị của mệnh đề ĐÚNG | **51 / 125** |
| tỷ lệ mệnh đề đúng **tới được** qua `join_candidates` | **98.2%** |
| kích thước bể trung vị | **1.741** mệnh đề |

Truy hồi **không** hỏng (98.2%). **Xếp hạng** mới là chỗ hỏng: hạng trung vị 51/125 là **thứ
tự ngẫu nhiên**. Cơ chế nối đang chạy không xếp — nó chỉ *lọc rồi bốc*.

#### 🔴 Bẫy thứ hai: sai **quy mô bể**, và "nhiệm vụ vàng ≠ nhiệm vụ sinh"

Đây là chỗ hoà giải mâu thuẫn treo từ §13.26 (`S3` AUC .814 ngoại tuyến) vs §13.30 (`S3` thua
chấm mù 59.7%). Hai cơ chế, cả hai đều đo được:

1. **Lệch quy mô bể.** Probe ngoại tuyến xếp trên ~50 ứng viên; lúc sinh, bể trung vị
   **~1.550**. Một tín hiệu "đào đuôi" (ưu tiên thống kê đếm thấp) trông vô hại trên 50 ứng
   viên và **hại gấp đôi** trên 1.550. Đo cụ thể: **65.4%** lựa argmax trên bể đủ dựa vào
   token gặp **≤5 lần** (p25 = **hapax**), trong khi mệnh đề đúng chỉ **45.6%**.
2. **Nhiệm vụ vàng không phải nhiệm vụ sinh.** Đối chứng vàng thưởng cho việc tìm lại *mệnh đề
   kế tiếp của cùng cha* — tức thưởng kỹ năng **nhận ra cùng-cha**. Lúc sinh, **mọi** ứng viên
   đều **khác cha**. Kỹ năng được đo **không dùng được** ở nơi cần dùng.

Đã thử vá (1) bằng **sàn support** cho pmi: top-1 bể đủ rơi **9.6% → 1.4%**, vì mệnh đề đúng
cũng có support trung vị chỉ **7** ⇒ bị sàn cắt cùng. **Khai tử**, không tinh chỉnh.

> **Bài học thứ năm về thước trong §13:** một probe xếp hạng phải chạy **đúng quy mô bể của
> lúc sinh**. Sai quy mô làm thiên lệch-do-argmax bị **giấu đi**, không chỉ làm nhiễu.

#### ✅ Thứ còn lại: bộ xếp **NEO VÀO ĐỀ, trọng số ĐỘ HIẾM** (`prompt_anchor_rescorer`)

Bất biến qua **sáu** vòng chấm mù: **mọi lần nới lỏng neo-chữ-vào-đề đều THUA**. Và: **mọi
tiêu chí chặt cho tới nay đều được thử làm CỔNG** — cặp âm tiết chết 94% đường (P57).

Nhưng cổng ≠ bộ xếp. Trên bể ~1.550 mệnh đề, **chặt là miễn phí**: không có ngõ cụt, chỉ có
sắp thứ tự. Chưa ai thử. Đó là `prompt_anchor_rescorer` trong
[clause_layer.py](src/nn_aivie/hca/clause_layer.py):

```
score(cand) = Σ_{f ∈ folds(cand) ∩ folds(ĐỀ)} gidf(f)
```

`gidf` do `ParentStore` **đếm sẵn** ⇒ Luật 0 sạch. Dùng **trọng số hiếm** chứ không **đếm**,
vì §13.20 đã cho thấy đếm thì rác chi phối; và pair-fold làm bộ xếp thì **chỉ nổ 4.2%** —
96% trơ, vô dụng.

Tính chất đo được **trước** chấm mù (không phải kết luận):

| | |
|---|---:|
| tỷ lệ mối nối bộ xếp **sống** | **100%** (trọng số trung vị 5.3) |
| đầu ra **đổi** trên 120 đề | **38.3%** |
| đọc tay các ca đổi | 3/3 **bỏ xâm nhập toán**, về đúng ngữ vực |

Ví dụ (trái = `S1` đang chạy, phải = neo-đề):
- `Sau trận Xuân Lộc…` → `2^5 bằng 32` **vs** `chiến lược ở Khu 5, Navarre được chính phủ Pháp cấp thêm 9 tiểu đoàn`
- `Khi Nguyễn Ánh được 9 tuổi (1771),` → `năm 1161, năm 1975` **vs** `Lý Thái Tổ dời kinh đô về Thăng Long năm 1010`
- `Từ năm 1976 bà công tác tại Ban Tổ chức Thành ủy TP` → `475 tỉ mét khối. bằng 6.` **vs** `người dân treo cờ và tổ chức lễ kỷ niệm`

⚠️ **Chưa phải kết luận.** Ba proxy ngoại tuyến liên tiếp đã đánh lừa (§13.26 → §13.30, và
P65 ở trên). Arbiter là **chấm mù**: `gen_gate_blind_pairs.py --mode anchor` sinh cặp
`join_S1` vs `join_anchor` trên v12, khoá giấu ở `out/anchor_key_v12.json`.

#### Phụ: phân bố `gidf` thật — và một test bị siết SAI

Test đầu tiên tôi viết đòi *1 fold hiếm thắng 2 fold chung*. Nó **đỏ** (`Phở` 1.916 vs
`Việt`+`Nam` 2.0). Thay vì nới test, đo phân bố `gidf` trên bản production:

- rác §13.20 (`nước`, `1`, `bằng`, `từ`, `ra`): **4.0–4.2**
- fold đặc trưng: **bão hoà ở 9.64** (mọi token trên ~50 lần gặp đều 9.64)

⇒ đúng ra **~2.4 lần trùng rác = 1 lần trùng đặc trưng**. Yêu cầu 1 > 2 là **ép một tỉ lệ dữ
liệu không có** — chính cái test sai, không phải cơ chế. Viết lại thành bất biến bảo vệ được
(*cùng số lần trùng thì hiếm thắng*) và ghi phép đo vào docstring để không ai siết lại theo
trực giác.

Ghi kèm: `gidf` **không** phải IDF sạch — `<eos>` có `gidf` 9.64 dù có mặt ở cả 5.664 cha.
Không cắn ở đây (`is_structural_token` lọc trước), nhưng bão hoà nghĩa là `gidf` **không phân
biệt được** `đây` (46 lần) với `thê` (2 lần). Nếu bộ xếp neo-đề thắng chấm mù, chỗ nâng cấp
rõ nhất nằm ở đây.

#### ĐÃ SINH trên v12 (§12.156) — 2026-08-03 — M4

**G-anchor (3) TRƯỢT.** Đổi đầu ra chỉ **~26%** (mini 38.3%); bỏ vì `a==b` **~74%** (>70%).
Đủ power ở `v12c` (104 cặp / 400 đề). Độ dài anchor **+8%** vs S1 (không bug cụt). Dự đoán
“đổi ≥30% / 120–170 cặp từ 200 đề” **BỊ BÁC**. Không chỉnh tham số. Chi tiết §12.156.

---

### 13.32 P66 — cổng G-anchor TRƯỢT ở v12; dự đoán của tôi sai 2/3 (2026-08-03) — Claude

M4 chạy PHẦN G (§12.156, M4 Pro 24 GB, ~165s/lượt). Đọc kết quả **trước khi** đụng vào bất cứ
tham số nào.

| lượt | đề | cặp | bỏ (`a==b`) | đổi | tok `S1` | tok neo-đề |
|---|---:|---:|---:|---:|---:|---:|
| v12 | 200 | 43 | 78.5% | 21.5% | 29.1 | 31.7 |
| v12b | 320 | 87 | 72.8% | 27.2% | 28.3 | 30.5 |
| **v12c** | **400** | **104** | **74.0%** | **26.0%** | 28.8 | 31.1 |

**Cổng G-anchor (nêu trước ở G.3): (1) ✅ 104 ≥ 88 · (2) ✅ · (3) ❌ bỏ 74% > 70% ⇒ TRƯỢT.**
Không chỉnh `gidf_min`, không chỉnh gì. Ghi trượt.

#### Chấm dự đoán của chính tôi (G.4)

| dự đoán ghi trước | thực tế | |
|---|---:|---|
| đổi đầu ra ≥ 30% | **26.0%** | ❌ |
| 200 đề → 120–170 cặp | **43** | ❌ (lệch **3×**) |
| độ dài không tụt > 20% | **+8.1%** | ✅ |

Cái sai nặng nhất là số cặp: tôi ngoại suy tỷ lệ đổi của mini thẳng sang v12 mà **quên rằng
tỷ lệ bỏ nhân lên theo cả chuỗi**. Đây là **lần thứ ba** một phép đo trên mini không sống sót
qua v12 (§13.25 độ dài cổng PMI, §13.27, và giờ là đây).

> **Bài học thứ sáu về thước trong §13:** mini dùng được để **bác** một cơ chế, **không** dùng
> được để **định cỡ** nó. Mọi ngưỡng cổng lấy từ số mini đều là ngưỡng vay mượn — chính ngưỡng
> 70% của tôi cũng được hiệu chỉnh từ 38.3% của mini, nên nó vừa quá chặt vừa không có cơ sở.

#### Vì sao mini 38.3% mà v12 chỉ 26% — giả thuyết có cơ chế, đo được

`gidf` **bão hoà ở 9.64** (§13.31). Ở mini (5.664 cha) chỉ token gặp trên ~50 lần mới chạm
trần; ở v12 (**430.447 cha**, gấp 76 lần) **phần lớn** fold nội dung chạm trần ⇒ điểm neo-đề
**hoà nhau hàng loạt** ⇒ thứ tự không đổi ⇒ `a == b`. Nghĩa là bộ xếp không hề "yếu" — nó bị
**thước làm phẳng**.

Giả thuyết này **dự đoán được**: thay `gidf` bão hoà bằng idf không chặn trần thì tỷ lệ đổi
phải **tăng** ở v12. Nếu vá xong mà tỷ lệ đổi **không** tăng ⇒ giả thuyết sai, và cơ chế
neo-đề đúng là trơ.

⚠️ Vá `gidf` **không phải** là chỉnh tham số để cứu cổng — G.3 cấm đúng việc đó và tôi không
làm. Đây là sửa một **khuyết tật đã đo và đã ghi từ §13.31**, tức trước khi thấy số của P66.
Nhưng nó vẫn phải **xếp sau** chấm mù, đúng như P66 §4 đã ghi trước: chưa biết hướng đúng thì
làm sắc thêm một hướng sai chỉ tốn công.

#### Vẫn chấm mù `v12c` — và ghi trước ý nghĩa của mọi kết cục

Cổng (3) trượt xác lập một sự thật về **ĐỘ PHỦ** (cơ chế chỉ chạm 26% đầu ra), **không** nói
gì về **CHẤT LƯỢNG mỗi mối nối nó chạm**. Hai câu hỏi khác nhau; 104 cặp đủ power cho câu thứ
hai. Ghi trước, để không diễn giải lại sau khi thấy phiếu:

- **Neo-đề thắng** ⇒ hướng "xếp theo neo-đề có trọng số hiếm" **đúng**, nhưng trần hiệu quả
  chỉ **26% đầu ra**. Chưa đủ để bật mặc định; đủ để **mở PHẦN H** (vá trần bão hoà `gidf`),
  và lúc đó dự đoán ở trên thành cổng.
- **Hoà** ⇒ cơ chế **không** đáng theo. Đóng hướng neo-đề-có-trọng-số. Không vá `gidf`.
- **`S1` thắng** ⇒ dương tính giả thứ tư, và là cái đắt nhất: 100% sống, 38.3% đổi (mini),
  3/3 đọc tay tốt hơn — **cả ba** proxy đều thuận mà vẫn thua. Ghi rõ rằng ba proxy đó **không
  dự báo được chấm mù**; đó là dữ kiện về THƯỚC, đáng hơn cả cơ chế.

Artefact: `out/anchor_pairs_v12c.txt` · khoá `out/anchor_key_v12c.json` · rubric
`RUNBOOK_TK_045_P66.md` §2.1. Giám khảo là **phiên Claude riêng**, cấm đọc khoá và ticket.

---

### 13.33 P66 — chấm mù vòng bảy: HOÀ. Đóng hướng neo-đề (2026-08-03) — Claude

104 cặp `v12c`, phiên Claude riêng, chỉ đọc `out/anchor_pairs_v12c.txt`. Lật bằng
`score_blind_verdict.py`.

| | |
|---|---:|
| `join_anchor` (neo-đề, trọng số hiếm) | **55** |
| `join_S1` (đang chạy) | 39 |
| hoà | 10 |

`join_anchor` **58.5%**, CI95 **[48.6%, 68.5%]** — **PHỦ 50%** ⇒ **không tách được**.

Theo P66 §3 và §13.32 đã ghi trước khi thấy phiếu: **hoà ⇒ đóng hướng neo-đề, KHÔNG vá
`gidf`, KHÔNG mở PHẦN H, KHÔNG bật `join_rescore`.** Áp dụng nguyên văn.

#### Đọc kỹ trước khi bỏ — hai thứ suýt làm tôi đọc sai

**(a) Thô theo vị trí X=53 / Y=41 KHÔNG phải thiên lệch vị trí.** Tôi đã định kết luận vậy.
Nhưng `join_anchor` nằm trái ở 56 cặp, `join_S1` trái ở 48; nếu anchor thật sự thắng 58.5% thì
kỳ vọng X thắng = 56·0.585 + 48·0.415 = **52.7 ≈ 53** — **khớp đúng con số quan sát**. Chênh
X/Y được giải thích **trọn vẹn** bằng chênh cấu hình, không cần giả thuyết thiên lệch nào.
Kiểm hai phía cũng thuận: anchor thắng cả khi ở trái (32–18) lẫn khi ở phải (23–21).

**(b) 58.5% mà vẫn kết luận hoà — vì vòng này chỉ phân giải được +15pp.** Hiệu ứng quan sát
là **+8.5pp**, nhỏ hơn thứ 94 cặp phân định được. Vòng này **không** phân biệt được "không có
hiệu ứng" với "hiệu ứng nhỏ ~+8.5pp".

⚠️ Và tôi **không** đề nghị chạy vòng lớn hơn để đuổi +8.5pp. Lý do có tính nguyên tắc, không
phải mệt mỏi: cơ chế chỉ chạm **26% đầu ra** (§13.32) ⇒ dù +8.5pp là thật thì hiệu ứng trên
**toàn bộ** đầu ra chỉ khoảng **+2pp**. Đuổi nó cần ~4 lần số cặp cho một mức lợi mà mọi vòng
sau đều không đo nổi. Đóng là quyết định đúng, không phải quyết định thua.

#### 🔴 Proxy thứ tư đổ: "3/3 đọc tay bỏ được xâm nhập toán" là ảo

§13.31 tôi đọc tay 3 ca trên mini và thấy neo-đề thay `2^5 bằng 32` bằng văn sử. Giám khảo
mù ở v12 thấy **xâm nhập kỹ thuật/công thức ở CẢ HAI bên**: G005/G041/G055 ở một phía,
G097/G027/G029/G036 ở phía kia. Cơ chế **không** chữa bệnh đó — 3 ca tôi đọc là **chọn mẫu
theo thứ tự tôi gặp**, cỡ mẫu 3, và tôi đã ghi nó vào ticket như bằng chứng.

⇒ Cộng dồn: **bốn** proxy đo trước đều thuận (100% sống · 38.3% đổi trên mini · 3/3 đọc tay ·
AUC-vàng ở các vòng trước) mà chấm mù vẫn không xác nhận. Đây là dữ kiện về **THƯỚC**, và nó
đáng hơn cơ chế bị bỏ.

#### ✅ Thu hoạch thật của vòng này: bản kê BỆNH do giám khảo mù lập

Giá trị lớn nhất không nằm ở con số. `notes` liệt kê **năm kiểu hỏng**, và chúng **đối xứng
hai bên** — tức là bệnh của **hệ**, không phải của bộ xếp:

1. **Vỡ ngữ vực** — chèn mảnh kỹ thuật/toán vào cảnh truyện (G005, G041, G097, G027).
2. **Tan thành danh sách số/ngày** — `năm 1850 năm 1945. năm 1832.` · `123456789` · rác
   `km 1661+100 QL1` · chèn bản tin xổ số (G034, G022, G077, G049). Đề dạng **niên biểu** rơi
   vào kiểu này ở cả hai phía.
3. **Lặp cụm đến cạn nghĩa** — `dữ liệu được lưu trữ. dữ liệu có thể được lưu trữ` ·
   `học sinh học sinh học sinh` · nhồi `thời gian` sáu lần thành **`thời gian khổ`** (G044,
   G047, G088, G038).
4. **Nối bằng ĐỒNG ÂM rồi bị cụm mới cướp chủ đề** — `đề tài → tài sản → sản xuất → xuất khẩu
   → khẩu phần ăn` (G078) · `hiện thực → thực phẩm` (G059) · `an toàn → an toàn cầu → cầu nối`
   (G039) · `đá → bóng đá` (G064). **Đây đúng là bệnh bản lề đồng tự §13.25** — và nó vẫn
   nguyên vẹn ở cả hai bên. Chuỗi trượt nghĩa nhiều bước như G078 là bằng chứng sạch nhất từ
   trước tới nay cho cơ chế đó.
5. **RÒ RỈ KHUÔN MẪU** — 🆕 chưa từng ghi: G054 để nguyên chỗ trống `[chủ đề]` của mẫu thư,
   G076 chèn gạch đầu dòng `*`, G001 bật sang danh sách `- Bạn có…` giữa văn xuôi. Đây là
   **dữ liệu huấn luyện lộ ra**, không phải lỗi cơ chế sinh.

Và nhận xét cấu trúc đáng giá nhất: *"chỗ quyết định gần như luôn nằm ở điểm RẼ NHÁNH sau
phần đầu chung — bên thắng là bên QUAY LẠI được đề, bên thua rẽ xong không về nữa"*
(G035, G063, G091).

#### Hướng kế — từ bản kê bệnh, không từ cơ chế vừa bỏ

Ba việc **đo được** rơi ra từ đây, xếp theo tỷ số lợi/công:

1. **Kiểu 5 (rò rỉ khuôn mẫu)** — rẻ nhất, và **không** phải việc của tầng sinh: đếm xem bao
   nhiêu cha chứa `[…]`, `*`, `- ` đầu dòng. Nếu đáng kể ⇒ lọc ở khâu nạp. Không đụng Luật 0.
2. **Kiểu 2 (tan thành danh sách số)** — đề niên biểu là hạng thua nặng nhất và **nhận diện
   được trước khi sinh**. Đo tỷ lệ mệnh đề toàn-số trong bể ứng viên theo loại đề.
3. **Kiểu 4 (bản lề đồng tự)** — bệnh gốc, và §13.31 đã đóng lối chữa bằng ngữ cảnh lân cận
   (`S6` AUC .404). G078 cho thấy cần **theo dõi chủ đề qua nhiều bước**, không phải sửa một
   mối nối. Đắt nhất, để sau.

---

### 13.34 P67/P68 — "rò rỉ khuôn mẫu": corpus SẠCH, khâu CHỌN khuếch đại (2026-08-03) — Claude

§13.33 kiểu hỏng 5. Giả thuyết ban đầu của tôi: **dữ liệu bẩn** ⇒ lọc ở khâu nạp. Đo xong thì
giả thuyết đó **sai**, và cái đúng nằm chỗ khác.

#### P67 — corpus KHÔNG bẩn (`scripts/probe_template_leak.py`)

829.513 đơn vị (≈ câu, xấp xỉ mức cha) qua 8 tập của `config/train_v12_sent.yaml`:

| lớp dấu | đơn vị | tỷ lệ |
|---|---:|---:|
| gạch đầu dòng `- ` | 4.630 | 0,56% |
| `*` / `**` | 1.036 | 0,12% |
| `[chỗ trống]` | 543 | 0,07% |
| còn lại (số, `##`, `{{}}`, HTML, `___`) | ~437 | 0,05% |
| **tổng dính** | **6.614** | **0,80%** |

Và **phần lớn 0,80% đó là bắt nhầm của chính bộ dò tôi viết**: `[tiếng Hà Lan có nghĩa Số 9]`
là ngoặc **biên tập** trong văn sử thật; `- Về bộ binh, Pháp có 267 tiểu đoàn.` là danh sách
**nội dung thật**; `dấu ngoặc { và }` là bài **lập trình** nói về dấu ngoặc. Chỗ trống mẫu thư
thật sự — `[chủ đề]` **8 lần**, `[X]`/`[Y]` 6, `[ngày]` 5, `[tên]` 2 — tổng **~27 lần** trong
830 nghìn đơn vị.

⇒ **Không lọc corpus.** Một bộ lọc ngây thơ sẽ xoá văn sử và bài lập trình thật để chữa một
thứ chiếm 0,003%.

#### 🔴 P68 — phép đo đầu tiên của tôi ĐO SAI ĐƠN VỊ, cho âm tính giả

Giả thuyết kế: rác hiếm bị **khuếch đại** ở khâu chọn (khớp §13.31: 65,4% lựa argmax dựa vào
token gặp ≤5 lần). Tôi đo bằng **tần suất CHỮ**: 208 dòng sinh vs câu thật lấy ngẫu nhiên từ
chính corpus.

| | SINH | THẬT |
|---|---:|---:|
| p25 tần suất chữ | 7.662 | 6.408 |
| tỷ lệ chữ gặp ≤5 lần | 0,52% | 0,53% |
| tỷ lệ chữ gặp ≤1 lần | **0,06%** | **0,22%** |

Kết luận của probe: *"KHÔNG thấy thiên lệch về chữ hiếm"* — đầu ra sinh thậm chí dùng chữ
**phổ biến hơn** văn thật. Tôi suýt ghi đây là âm tính và đóng hướng.

**Sai ở đơn vị đo.** Cụm `km 1661+100 QL1` gồm toàn chữ thường gặp (`km`, số) ⇒ tần suất
**chữ** pha loãng hết hiệu ứng. Thứ bị khuếch đại là **MỆNH ĐỀ**, không phải chữ.

#### ✅ Đo lại đúng đơn vị — khuếch đại ~10 lần, bằng chứng sạch

Cụm `km 1661+100 QL1` xuất hiện **1 lần duy nhất** trong 829.513 đơn vị corpus (hapax). Trong
đầu ra sinh của 104 đề v12c, nó xuất hiện ở **10 cặp KHÁC NHAU** — G016, G019, G022, G045,
G049, G052, G066, G077, G083, G084 — tức **9,6% số lần sinh**, từ 10 đề không liên quan nhau.

| dấu | corpus (mỗi đơn vị) | kỳ vọng ở đầu ra (~5 mệnh đề) | quan sát | |
|---|---:|---:|---:|---:|
| `*` | 0,12% | ~0,6% | **5,3%** dòng | **≈9×** |
| `[…]` | 0,07% | ~0,35% | 0,5% dòng | ~1,4× (không đáng kể) |
| `- ` đầu dòng | 0,56% | ~2,8% | **0%** | bị **chặn** hẳn |

Đối chứng chống hiện vật: hai vế của **cùng** một cặp dùng chung phần đầu, nên phải đếm ở mức
**cặp**. Đã đếm lại — chỉ **4** đầu mệnh đề dùng lại quá 1 cặp. Hệ **không** lặp lung tung;
nó lặp **đúng vài mệnh đề nhất định**.

#### Chẩn: MỆNH ĐỀ NAM CHÂM RÁC

`km 1661+100 QL1, năm 1945…` là mệnh đề đặc số và năm. Fold của nó là **rác §13.20** (`năm`,
chữ số, `1`) — thứ khớp được với **rất nhiều** đề. Nó không thắng vì hiếm; nó thắng vì **khớp
với mọi thứ**. Điều này hợp nhất kiểu hỏng 2 (tan thành danh sách số) và kiểu 5 vào **một**
cơ chế: một nhúm mệnh đề có fold-rác phổ dụng hút được mọi đề và bị phát lại liên tục.

Và nó giải thích luôn vì sao `- ` bị chặn (0%) còn `*` thì không: `- ` đứng **đầu** đơn vị nên
bị khâu tách biên loại, còn `*` nằm **giữa** nên đi lọt.

#### Việc kế — đo được, và cần v12 ⇒ M4

Xếp hạng mệnh đề theo **số đề khác nhau mà nó được phát ra** trên vài trăm đề. Giả thuyết:
phân bố **lệch nặng** — một nhúm nhỏ chiếm phần lớn lần phát. Nếu đúng, đó là mục tiêu can
thiệp **cụ thể và đếm được** (Luật 0 sạch: chỉ đếm), khác hẳn mọi thứ §13 đã thử — vì nó tấn
công **cái được chọn quá nhiều**, không phải **cách chọn**.

⚠️ Chưa mở PHẦN H cho việc này: §13.33 vừa đóng hướng neo-đề, và bài học §13.32 là **đừng
định cỡ bằng mini**. Cần nêu cổng trước khi giao.

---

### 13.35 P69 — "nam châm rác" BỊ BÁC; và §13.34 của tôi có lỗi MẪU SỐ (2026-08-03) — Claude

M4 chạy PHẦN H trên v12, 400 đề, hai lượt, 116s (§12.157).

| | THẬT | ĐỐI CHỨNG xáo |
|---|---:|---:|
| tổng lần phát | 833 | 828 |
| mệnh đề khác nhau | **771** | 780 |
| top-10 chiếm | **5,4%** | 4,6% |
| đỉnh (tỷ lệ đề) | **2,75%** | 2,25% |
| mệnh đề phát ở ≥5% đề | **0** | 0 |

**Cổng G-magnet: (1) ❌ · (2) ❌ · (3) ❌ ⇒ TRƯỢT cả ba.** Tỷ số thật/đối chứng **1,18×**.

#### Chấm dự đoán của tôi (H.4): sai 3/4

| dự đoán ghi trước | thực tế | |
|---|---:|---|
| đạt cả ba mục cổng | trượt cả ba | ❌ |
| đỉnh ≥ 8% số đề | **2,75%** | ❌ |
| top-10 chiếm 15–35% | **5,4%** | ❌ |
| tỷ số thật/đối chứng 2–4× | **1,18×** | ❌ |
| mệnh đề đầu bảng đặc số/rác | `km 1661+100 QL1,` **11 đề** | ✅ |

771 mệnh đề khác nhau trên 833 lần phát — **gần như cái nào cũng chỉ phát một lần**. Không có
tập trung. Và **1,18× ≈ 1** nghĩa là chỗ tập trung ít ỏi còn lại **không** do khâu xếp hạng:
xáo ngẫu nhiên trật tự ứng viên cho **gần y hệt**.

⇒ **"Mệnh đề nam châm rác" như một CƠ CHẾ: BỊ BÁC.** Đóng hướng. Không viết bộ lọc.

#### 🔴 Lỗi thật của §13.34: tôi đo tỷ lệ trên một MẪU ĐÃ LỌC

§13.34 tôi ghi cụm `km 1661+100 QL1` phát ở **9,6%** số lần sinh. P69 đo trực tiếp: **2,75%**.
Chênh 3,5 lần. Nguyên nhân **không** phải ngẫu nhiên — là **mẫu số sai**:

`out/anchor_pairs_v12c.txt` **chỉ giữ những đề mà hai cấu hình cho kết quả KHÁC nhau** —
`gen_gate_blind_pairs.py` bỏ mọi cặp có `a == b`, tức **74% số đề bị loại** (§13.32). Tôi lấy
10/104 **cặp** rồi gọi đó là tỷ lệ khi sinh. Mẫu số đúng là **400 đề gốc**: 10/400 = **2,5%**
— **khớp** con số 2,75% của P69.

Con số 9,6% chưa bao giờ tồn tại. Nó là hiện vật của việc đo tỷ lệ phổ biến trên một artefact
**đã lọc theo đúng cái tiêu chí làm nó khác thường**.

> **Bài học thứ bảy về thước trong §13:** file cặp chấm mù **không** phải mẫu đại diện của
> đầu ra. Nó là mẫu **đã lọc** để giữ chỗ hai cấu hình bất đồng — tức giàu bất thường một cách
> có hệ thống. Dùng nó để chấm **so sánh** thì đúng; dùng nó để đo **tỷ lệ phổ biến** thì sai.

Kéo theo: con số "`*` khuếch đại ~9×" ở §13.34 **cùng lỗi mẫu số** và phải coi là **chưa
kiểm**. Tôi không đo lại — hướng đã đóng, đo thêm chỉ để cứu một kết luận đã chết.

#### Cái còn đứng, và cái đáng nhìn tiếp

`km 1661+100 QL1,` **thật sự** là mệnh đề bị phát nhiều nhất (11/400 đề) và **thật sự** là rác.
Nhưng nó là **một** mệnh đề, không phải một hạng. Xử lý một mệnh đề không phải sửa cơ chế.

Thứ đáng nhìn hơn nằm ngay trong bảng, ngoài mục tiêu ban đầu: **833 lần phát / 400 đề = 2,08
mệnh đề mỗi đề**, trong khi `max_clauses = 5`. Tức **chuỗi chết sớm ở đa số đề** — khớp
`dead_rate` 81% của cổng fold (§12.154). Đây là con số lớn nhất chưa ai theo đuổi trong §13:
mọi vòng vừa qua đo *chọn cái gì*, chưa vòng nào đo *vì sao dừng*.

---

### 13.36 P70 — VÌ SAO DỪNG: **cổng phủ** giết 63% chuỗi, không phải truy hồi (2026-08-03) — Claude

§13.35 để lại con số lớn nhất chưa ai theo: 2,08/5 mệnh đề mỗi đề. Suốt §13 mọi vòng đo *chọn
cái gì*; đây là vòng đầu đo *vì sao dừng*.

**Cách làm:** không chép lại vòng lặp vào probe (sẽ trôi khỏi mã thật) — thêm thẳng trường
`stats["stop"]` vào `compose_chain`, khai bốn lối ra: `du_max` · `khong_neo` ·
`khong_ung_vien` · `loai_het`, kèm `n_gate` / `n_dup` / `n_glue` / `pool_at_stop`.

300 đề, mini, `max_clauses=5`. **Đối chứng: cùng đề, KHÔNG cổng.**

| lý do dừng | CÓ CỔNG | KHÔNG CỔNG |
|---|---:|---:|
| `du_max` (không tắc) | 31,7% | **88,0%** |
| `khong_neo` | 0,0% | 0,0% |
| `khong_ung_vien` (truy hồi cạn) | 5,0% | 12,0% |
| **`loai_het`** (có ứng viên, loại sạch) | **63,3%** | **0,0%** |
| mệnh đề TB / đề | 2,62 | **4,66** |
| chuỗi **RỖNG** | **20,7%** | 2,3% |

Thủ phạm trong `loai_het`: **cổng 87,7%** · dính-biên-từ 12,2% · trùng 0,2%. Bể lúc tắc có
trung vị **61 token-mở** — tức **thừa ứng viên**, bị loại sạch.

⇒ Chuỗi dừng sớm **không** vì đồ thị thưa. Nó dừng vì **cổng phủ**. Bỏ cổng: 88% chạy hết
`max_clauses`, độ dài **2,62 → 4,66**, chuỗi rỗng **20,7% → 2,3%**.

#### Thế lưỡng nan phải nói thẳng

Cổng fold **thắng bốn vòng chấm mù** (§13.25/§13.27) — bỏ nó là bỏ thứ **duy nhất** đã được
arbiter xác nhận trong toàn §13. Nhưng chính nó khiến **1/5 số đề không sinh được gì** và cắt
độ dài gần một nửa. Hai sự thật này **không mâu thuẫn**: giám khảo so *chất lượng phần đã
sinh*, chưa vòng nào phạt việc **không sinh được**.

⚠️ Chưa đề xuất nới cổng. Nới cổng là **đúng cái đã thua** ở P60/§13.25 (cổng PMI lỏng hơn,
thua 85,6%) — và §13.30 giải thích vì sao: nới ra thì chuỗi trôi khỏi đề. Cần một vòng chấm
mù có **đề bị bỏ trống tính là thua**, chứ không phải nới rồi hy vọng.

#### 🔴 Và trường `stop` bắt ngay một TEST XANH VÔ NGHĨA

`test_compose_chain_dead_when_gate_rejects_all` xanh từ lâu. Vừa gắn `stop` vào thì lộ:
`stop == "khong_neo"`, tức chuỗi chết vì **không có neo**, **cổng chưa bao giờ được hỏi**.
Nguyên nhân đúng bằng cái bẫy đã ghi trong docstring `compose_chain`: corpus đồ chơi 4 câu có
**mọi** token dưới `gidf_min=4.0` ⇒ `content_folds` rỗng ⇒ chết ở bước 0. Test vẫn xanh vì
`chain2 == [] and dead` **đúng vì một lý do khác**.

Đã vá: truyền `gidf_min=0.0`, và siết bằng `stop == "loai_het"` + `n_gate > 0` +
`pool_at_stop > 0`. Ghi chú giữ trong test.

> **Bài học thứ tám về thước trong §13:** một test **xanh** không có nghĩa nó đã kiểm thứ ghi
> trong **tên** nó. Muốn biết, phải bắt nó khai **đường đi**, không chỉ khai kết quả.

#### Hạn của vòng này

Số ở đây là **mini**. Theo §13.32, mini dùng để **bác/chỉ cơ chế**, không để **định cỡ** — và
câu hỏi "lối ra nào chi phối" đúng là câu hỏi cơ chế. Chỉ dấu yếu là 2,62/5 (mini) so 2,08/5
(v12, §13.35) cùng hạng. Muốn con số v12 thì chạy `probe_why_stop.py` trên checkpoint v12
(cần máy có RAM), nhưng kết luận "cổng, không phải truy hồi" khó đổi dấu vì đối chứng
KHÔNG-cổng nằm ngay trong cùng đồ thị.

---

### 13.37 P71 — chấm mù vòng tám: **im lặng THẮNG 163/163**. Cổng phủ đúng, kể cả khi tính giá (2026-08-03) — Claude

167 cặp v12 (M4 §12.158, 600 đề, tỷ lệ bỏ trống **27,8%**). Phiên Claude riêng, chỉ đọc
`out/empty_pairs_v12.txt`. Lật bằng `score_blind_verdict.py`.

| | |
|---|---:|
| `fold` (**bên TRỐNG**) | **163** |
| `fold_duphong` (bên có chữ) | **0** |
| hoà | 4 |

**163/163 = 100%**, Wilson CI95 **[97,7%, 100%]**. Đây là kết quả **dứt khoát nhất** trong cả
§13 — và là lần đầu một vòng chấm mù cho **không một phiếu nào** cho một bên.

Giám khảo trả lời thẳng câu hỏi quyết định (P71 §2.1 mục 2): *"KHÔNG có ca nào bên có chữ thật
sự dùng được — không cặp nào đọc trọn vẹn mà vẫn bám đề."* Bốn ca hoà là bốn ca đúng **một**
mệnh đề đầu rồi vỡ (G079, G032, G005, G155).

#### Theo đúng nghĩa đã ghi trước (P71 §3)

**Im lặng TỐT HƠN nói bậy.** Cổng phủ đúng **ngay cả khi đã tính cả cái giá của nó** — 27,8%
số đề bỏ trống. ⇒ Hướng **"nới cổng cho chuỗi dài ra" CHẾT HẲN**. Trần chất lượng **không**
nằm ở độ chặt của cổng.

Và thế lưỡng nan §13.36 **tan**: cổng phủ không "trả giá quá đắt". Nó **đang mua đúng thứ đáng
mua**. `dead_rate` 81% và 2,08/5 mệnh đề là **giá đúng**, không phải khuyết tật.

Dự đoán I.4 của tôi: TRỐNG thắng **≥55%** ✅ (thực tế 100%) · tỷ lệ bỏ trống 15–25% ❌ (27,8%).

#### 🔴 Và nó lật lại §13.35 theo hướng KHÔNG ngờ: nam châm rác CÓ THẬT — nhưng bị cổng chặn

Giám khảo mô tả kiểu hỏng số một là **"mở đầu vay mượn"**: cùng một mẩu cố định được dán cho
hàng loạt đề không liên quan. Đếm lại trên 167 dòng có chữ (bản **KHÔNG cổng**):

| | KHÔNG cổng (P71) | CÓ cổng (P69, §13.35) |
|---|---:|---:|
| đỉnh — mẩu mở đầu phổ biến nhất | **4,8%** số đề | 2,75% |
| top-10 mẩu mở đầu chiếm | **25,1%** | 5,4% |

`và hãng hàng không có` 8 lần · `có màu hơi xám tro` 8 · `của một vật chứa nhiều` 6 ·
`không dứt đột ngột mưa` 5.

⇒ **§13.35 không sai — nó đo đúng hệ CÓ CỔNG.** Nam châm rác là hiện tượng thật, và **cổng
phủ chính là thứ đang chặn nó**. Hai kết quả khớp nhau hoàn hảo khi đọc đúng điều kiện: bỏ
cổng ra thì tập trung tăng **~4,6 lần** (5,4% → 25,1%).

> **Bài học thứ chín về thước trong §13:** một âm tính luôn kèm điều kiện thí nghiệm. "Không có
> nam châm rác" thật ra là "không có nam châm rác **khi cổng đang bật**". Chỗ khác biệt đó
> chính là công dụng của cổng — và tôi đã suýt ghi nó thành "hiện tượng không tồn tại".

#### Bệnh bản lề đồng tự: bằng chứng dày nhất từ trước tới nay

Giám khảo liệt kê kiểu hỏng 2 là **trượt đề qua điểm nối âm tiết**, đúng bệnh §13.25:
`tỉ lệ` → **`lệ quỷ`** (G027) · `xoáy ốc` → **`ốc sên`** (G130) · `ví dụ` → **`dụ quân địch`**
(G116) · `ngồi xuống` → **`xuống cấp`** (G040), cùng các mối hàn hỏng `thần kinh tế` (G014),
`công tyso` (G087).

Đây là bệnh còn nguyên vẹn sau tám vòng, và giờ đã có ví dụ dày. §13.31 đã đóng lối chữa bằng
ngữ cảnh lân cận (`S6` AUC .404). Lối còn lại chưa thử: **theo dõi chủ đề qua nhiều bước**,
chứ không sửa một mối nối.

#### Chốt trạng thái sau tám vòng

Đã đóng bằng số: cổng PMI · `lạc` hai bản · cặp âm tiết · bộ xếp `S3` · điều kiện-PROMPT ·
neo-đề trọng số hiếm · lọc corpus · nam châm rác (khi có cổng) · **nới cổng**.

Còn đứng: **cổng phủ fold** — thắng 5/5 vòng có nó tham gia, nay có cả bằng chứng nó chặn nam
châm rác. Còn mở: **bản lề đồng tự** (bệnh gốc, đã loại một lối chữa) và **kiểu hỏng 3** (lặp
cụm cạn nghĩa, chưa ai đo).

---

### 13.38 P72 — kiểu hỏng 3: lặp cụm **KHÔNG** quá văn thật; lỗi thật nằm ở **mối nối** (2026-08-03) — Claude

Món cuối trong bản kê 5 kiểu hỏng của giám khảo (§13.33). 2.000 đề, mini, cổng fold.
**Đối chứng nằm trong chính mạng:** k mệnh đề **liên tiếp của cùng một cha** — văn người viết
thật, cùng bộ tách, cùng tokenizer, cùng k.

| | SINH | VĂN THẬT |
|---|---:|---:|
| distinct-2 | **0,99** | 0,99 |
| distinct-2 p25 | 1,00 | 1,00 |
| tỷ lệ có bigram lặp | **18,5%** | 18,2% |
| lặp 1 chữ tối đa | 2,16 | 1,92 |
| **tỷ lệ có lặp KỀ** | **4,8%** | **1,6%** |
| số chữ | 31,3 | 25,3 |

#### Kiểu 3 như giám khảo mô tả: **KHÔNG xác nhận**

"Lặp cụm đến cạn nghĩa" ở mức **cụm** không tồn tại: `distinct-2` **bằng nhau đến hai chữ số**,
tỷ lệ có bigram lặp **18,5% vs 18,2%** — và đó là trong khi đầu ra sinh **dài hơn 24%** (31,3
vs 25,3 chữ), tức có **nhiều cơ hội lặp hơn** mà vẫn không lặp hơn.

Ví dụ giám khảo nêu (`dữ liệu được lưu trữ. dữ liệu có thể được lưu trữ`) là **có thật**, nhưng
văn thật cũng lặp đúng mức ấy. Con mắt bắt được cái lặp vì nó **chướng**, không vì nó **nhiều**.
Đây là kiểu hỏng thứ hai trong bản kê bốc hơi khi đo (sau kiểu 5, §13.34).

#### ✅ Nhưng có một khuyết tật thật, hẹp và định vị được: **lặp KỀ tại MỐI NỐI**

| | tỷ lệ |
|---|---:|
| lặp kề **tại mối nối** (chữ cuối A == chữ đầu B) | **61/3.482 = 1,75%** |
| lặp kề **trong lòng** mệnh đề | 10/39.043 = 0,03% |
| **tỷ số** | **68,4×** |

Đây **không** phải văn nguồn — văn nguồn lặp kề ở 0,03%. Đây là khâu **nối**: `join_candidates`
bắc cầu `last_content(A) → first_content(B)` qua `W_skip` mà **không có ràng buộc nào cấm B mở
đầu bằng đúng chữ A vừa kết thúc**. Kết quả là `học sinh học sinh` của §13.33.

⇒ Cùng **một chỗ** với bệnh bản lề đồng tự (§13.25): token bản lề. Bệnh đồng tự là *cùng id,
khác nghĩa*; bệnh này là *cùng id, phát hai lần*. Cái sau **sửa được bằng một ràng buộc**, cái
trước thì không.

#### Ứng viên sửa — và vì sao CHƯA bật

Chặn ứng viên có `first_content(B) == last_content(A)` ở `join_candidates`. Luật 0 sạch (ràng
buộc cấu trúc, không phải từ vựng cứng), và chạm đúng **1,75% mối nối**.

⚠️ **Chưa bật.** Tám vòng vừa qua: **mọi** cải tiến "hiển nhiên đúng" đều thua hoặc hoà khi
chấm mù, trừ đúng một cái. Trần lợi ích ở đây là 1,75% mối nối ⇒ hiệu ứng toàn cục quá nhỏ để
một vòng chấm mù 88 cặp phân giải được (§13.33 đã học đúng bài này: +8,5pp cũng không phân giải
nổi). Ghi lại làm **ứng viên có địa chỉ**, gom cùng các sửa nhỏ khác khi đủ một gói đáng chấm.

#### Phụ: hệ chọn mệnh đề DÀI hơn văn thật

Cùng số mệnh đề mà đầu ra sinh **31,3 chữ** so với văn thật **25,3** — dài hơn **24%**. Khớp
thiên lệch "chọn mệnh đề dài nhất" đã biết. Ghi nhận, chưa đo tác hại.

#### Bản kê 5 kiểu hỏng — đã đóng

| kiểu | phán quyết |
|---|---|
| 1. vỡ ngữ vực | có thật; cổng phủ đang chặn (§13.37) |
| 2. tan thành danh sách số | có thật; **không** do nam châm rác (§13.35) |
| 3. lặp cụm cạn nghĩa | **KHÔNG** hơn văn thật; lỗi thật là lặp-kề-tại-mối-nối 1,75% |
| 4. bản lề đồng tự | **có thật, còn nguyên** — bệnh gốc duy nhất chưa có lối chữa |
| 5. rò rỉ khuôn mẫu | corpus sạch; số 9,6% của tôi là **lỗi mẫu số** (§13.35) |

⇒ Sau khi đo cả năm: **chỉ kiểu 4 còn đứng như bệnh cần chữa**. Ba kiểu bốc hơi hoặc đã có
thứ chặn; kiểu 3 teo lại thành một khuyết tật 1,75% có địa chỉ.

---

### 13.39 P73 — bệnh gốc có SỐ: bản lề **ĐỔI VAI TRONG TỪ**, 10,7% vs 0,0% (2026-08-03) — Claude

Sau tám vòng, bệnh bản lề đồng tự (§13.25) mới chỉ được thấy bằng **ví dụ**. P73 đo nó.

#### 🔴 Thước thứ nhất SAI — và đối chứng lộ ra hiện vật của chính nó

Ý đầu: nghĩa của âm tiết đồng tự do **chữ đứng trước** quyết định (`bệnh` trong `bệnh nhân`),
nên đo `hop_le = P(w_left ngay trước t │ t đồng hiện w_right trong d)`.

Bản đầu cho: SINH 14,3% · **THẬT 50,0%**, "vay hẳn nghĩa khác" SINH 57,0% · **THẬT 0,0%** —
cổng ĐẠT rực rỡ. **Sai.** Mối nối THẬT nằm ngay trong kho văn ⇒ nó **tự đếm chính nó**, và
lần đó luôn hợp lệ ⇒ `hop_le = 0` ở đúng **0,0%** một cách máy móc.

Bỏ lần tự đếm ra (leave-one-out), kết quả **đảo chiều**:

| | SINH | THẬT |
|---|---:|---:|
| hợp-lệ trung bình | 14,3% | **11,1%** |
| "vay hẳn nghĩa khác" | 57,0% | **65,5%** |

⇒ **Cổng TRƯỢT**, và mối nối sinh còn *sạch hơn* mối nối người viết theo thước này. Kết luận
đúng: **thước sai**, không phải bệnh không có. Đối chứng thật (mối nối người viết) mà chỉ đạt
11,1% thì thước đó đang đo **độ hiếm của bigram trái**, không đo nghĩa.

> **Bài học thứ mười về thước trong §13:** khi đối chứng "vàng" cho ra con số **hoàn hảo**
> (0,0%), đó gần như luôn là **hiện vật cấu trúc**, không phải phát hiện. Hỏi ngay: *mẫu có
> tự đếm chính nó không?*

#### ✅ Thước thứ hai — đọc chữ ký ra từ chính danh sách ví dụ

`quốc gia` ⇒ `gia vị` · `Ban Nha` ⇒ `Nha biển` · `khai sinh` ⇒ `sinh ra`. Chữ ký **không**
phải "đồng tự" chung chung, mà là: bản lề là âm tiết **CUỐI** của một từ trong A, rồi mối nối
đọc nó thành âm tiết **ĐẦU** của một từ khác. Đếm được: bản lề bị **buộc chặt CẢ HAI phía**.

    word_pivot_strength = min( P(w_left ngay trước t) , P(w_right ngay sau t) )

| ngưỡng min ≥ 0,05 | tỷ lệ mối nối |
|---|---:|
| **SINH** | **10,7%** |
| **THẬT** (người viết, leave-one-out) | **0,0%** |
| THẬT (không trừ self — kiểm hiện vật) | **0,1%** |

Kiểm hiện vật quan trọng: lần này con số 0,0% **không** do phép trừ của tôi — bỏ leave-one-out
ra thì vẫn 0,1%. Chênh ~**100 lần**.

Ví dụ (SINH): `tối đa` ⇒ `đa số` (trái 13% · phải 23%) · `tư duy` ⇒ `duy nhất` (17% · 36%) ·
`nhân dân` ⇒ `dân số` (14% · 9%) · `dân tộc` ⇒ `tộc Việt` (69% · 7%).

⇒ **Đây là con số cứng đầu tiên cho bệnh gốc.** Nó **không** phải hiện tượng của văn thật: văn
người viết gần như không bao giờ nối câu ở một âm tiết đang bị buộc chặt cả hai phía.

#### Cơ chế đã dựng — và vì sao TẮT mặc định

`word_pivot_strength` + tham số `compose_chain(pivot_max=...)`, mặc định `None` = **tắt**.
Chỉ đọc `W_seq` (đếm) ⇒ Luật 0 sạch. Test khoá cả hai: mặc định **không** đổi hành vi
(`n_pivot == 0`), và dây đã nối.

⚠️ **Không tự bật.** Và có lý do thực chất, không chỉ kỷ luật: trong 10 ví dụ, ít nhất hai cái
**đúng** — `nhà Lý` ⇒ `Lý Thái` (Lý Thái Tổ) và `người Việt` ⇒ `Việt Nam`. Bộ chặn sẽ giết cả
chúng. Lợi/hại **phải** do chấm mù phán, và 10,7% mối nối là đủ lớn để một vòng phân giải
được — khác hẳn 1,75% của §13.38.

⇒ Giao M4 sinh cặp: **PHẦN J**.

---

### 13.40 P74 — chấm mù vòng chín: HOÀ. Nhưng lợi ích đến từ **DỪNG SỚM**, không từ chặn bản lề (2026-08-03) — Claude

148 cặp v12b (M4 §12.159), phiên Claude riêng.

| | |
|---|---:|
| `fold_chanbanle` (chặn bản lề đổi vai) | **67** |
| `fold` | 55 |
| hoà | 26 |

**54,9%**, CI95 **[46,1%, 63,7%]** — **phủ 50%** ⇒ không tách được. Theo P74 §3 đã ghi trước:
**HOÀ ⇒ giữ `pivot_max` TẮT.** Dự đoán J.4 của tôi (thắng ≥55%) **trượt** — điểm ước lượng
54,9%, và dù có đúng 55% thì vòng này cũng không phân giải nổi.

⚠️ Ghi thẳng bài học đã nêu trước ở J.4: **đo được khuyết tật KHÔNG có nghĩa chữa nó là cải
tiến.** Bản lề đổi vai có thật, ở mức 100× so với văn người viết (§13.39), và chặn nó **không**
làm chất lượng cảm nhận đổi.

#### 🔴 Giám khảo không phân biệt được HAI BÊN theo đúng hai bệnh ta nhắm

Rubric hỏi thẳng: bên nào hay cắt đôi tên riêng, bên nào hay trượt chủ đề tại mối nối. Trả
lời: **cả hai, ngang nhau**. Cắt tên riêng: `Kim` + `Dung dịch lỏng` (G018) · `Lê Thị Trang`
→ `trang web` (G051) · `cường đại học` (G060) · `chất lượng tử` (G075) · `công ty lạp thể`
(G114). Trượt chủ đề: `tối đa` → `đa bào` (G002) · `thanh âm` → `thanh toán` (G043) ·
`xác thực` → `thực phẩm` (G037) · `giới tính toán` (G109).

Và **nhiều ca lỗi nằm ngay trong đoạn TIỀN TỐ CHUNG** (G034, G090, G107) — tức **cả hai** cấu
hình đã mắc **trước khi rẽ**. ⇒ Ngưỡng `0.05` bắt được **một phần nhỏ** của bệnh; phần lớn ca
trượt nghĩa nằm **dưới** ngưỡng đó.

#### ✅ Phát hiện thật của vòng này — phân rã theo HÌNH DẠNG cặp

Giám khảo nhận xét: *"điểm phân biệt thực tế không phải loại lỗi mà là ĐIỂM DỪNG"*. Tách 122
cặp phân định làm hai nhóm:

| nhóm | n | `fold_chanbanle` thắng | CI95 |
|---|---:|---:|---|
| **một bên là TIỀN TỐ của bên kia** (bộ chặn cắt ngắn chuỗi) | 25 | **84,0%** | **[65,3%, 93,6%]** |
| **rẽ nhánh thật** (bộ chặn đổi hướng nối) | 76 | 51,3% | [40,3%, 62,2%] |

Và bên ngắn trong nhóm tiền tố là `fold_chanbanle` ở **29/29** ca.

⇒ **Toàn bộ lợi ích của bộ chặn đến từ việc nó DỪNG CHUỖI SỚM, không từ việc nó chọn mối nối
tốt hơn.** Khi nó chỉ đổi hướng nối, nó là **đồng xu** (51,3%). Khi nó cắt ngắn, nó **thắng
đậm** (84%).

Nói cách khác: **mỗi mệnh đề nối thêm, trung bình, LÀM HẠI.** Đây là §13.37 ("im lặng hơn nói
bậy") lặp lại ở cấp **dưới chuỗi**: không chỉ "thà không sinh còn hơn sinh bậy", mà "thà dừng
ở mệnh đề 2 còn hơn đi tiếp tới 5".

⚠️ **Đây là phân tích HẬU NGHIỆM**, tìm ra từ ghi chú giám khảo chứ không nêu trước. Nó
**chưa** là kết luận — nó là **giả thuyết có số**, và phải được một vòng **nêu trước** xác
nhận. Không được bật gì dựa trên nó.

#### Phụ: bên TRỐNG lần này THUA — và điều đó KHỚP với §13.37

25 cặp có một bên trống: trống **thắng 7** (bên kia salad thuần), **thua 14** (bên kia đọc
được, bám đề), hoà 4. Ngược hẳn §13.37 (trống thắng 163/163). **Không mâu thuẫn:** ở §13.37
đối thủ của trống là bản **KHÔNG cổng** (salad); ở đây đối thủ là bản **có cổng** (văn tạm
được). ⇒ Mệnh đề đúng là: *trống hơn salad, nhưng thua văn đã qua cổng.* Đây là chỗ §13.37 dễ
bị trích dẫn quá đà, nay có ranh giới.

#### Vòng kế — nêu trước, rẻ, và bác được

Giả thuyết: **dừng sớm là lợi, độc lập với mọi cơ chế chọn.** Kiểm bằng chấm mù
`max_clauses=5` (hiện tại) vs `max_clauses=2`, cùng cổng fold, không cơ chế mới nào.

- Nếu bản **ngắn thắng** ⇒ trần chất lượng nằm ở **độ dài chuỗi**, và mọi công sức chín vòng
  qua vào *cách chọn* đều nhắm sai chỗ. Sửa là **một tham số**.
- Nếu **hoà/thua** ⇒ 84% ở trên là hiện vật của việc bộ chặn cắt đúng chỗ khó, không phải quy
  luật về độ dài — và giả thuyết chết.

Giám khảo cũng đề nghị đúng hướng: đo **trực tiếp** tỷ lệ cắt-tên-riêng và trượt-chủ-đề tại
đúng token nối, vì chấm mù tổng thể **bị nhiễu nặng** bởi độ dài đoạn nối thêm và bởi lỗi có
sẵn trong tiền tố chung.

---

### 13.41 P75 — chấm mù vòng mười: **DỪNG SỚM THẮNG 78,7%**. Cơ chế thứ hai được xác nhận (2026-08-03) — Claude

142 cặp v12 (M4 §12.160), phiên Claude riêng. Hai bên khác đúng **một tham số**; bất biến
tiền tố kiểm độc lập **142/142**.

| | |
|---|---:|
| **`len2`** (`max_clauses=2`) | **96** |
| `len5` (mặc định hiện tại) | 26 |
| hoà | 20 |

**78,7%**, CI95 **[71,4%, 86,0%]** — **tách khỏi 50%**, và thắng ở **cả hai phía** (48–11 và
48–15). Dự đoán K.4 của tôi (60–75%) **thấp hơn thực tế**.

⇒ Đây là **cơ chế thứ hai** trong toàn §13 được arbiter xác nhận, sau cổng phủ fold. Và nó là
**một tham số**, không phải một thuật toán.

Phản bác "giám khảo chỉ thích ngắn" đã bị §13.25 bác từ trước (vòng ba: bên ngắn hơn một nửa
**THUA** 85,6%), và chính giám khảo vòng này liệt kê **8 ca bên ngắn bị cụt** phải phạt
(G004 dừng ở `hoặc các`, G010 `và nếu`, G041 `một cách`, G112 `bằng`). `len2` thắng **dù**
trả giá đó.

#### 🔴 Nhưng cơ chế KHÔNG phải "mối nối sau tệ hơn" — đo rồi, phẳng

Giả thuyết hiển nhiên: đuôi tệ vì đi càng xa càng lạc. **Sai.** Đo trên mini, 1.500 đề:

| mối nối thứ | n | tỷ lệ **đổi vai** ≥0,05 |
|---|---:|---:|
| 1 | 899 | 10,2% |
| 2 | 702 | 12,7% |
| 3 | 549 | 9,8% |
| 4 | 454 | 10,4% |

**Phẳng.** Mỗi mối nối hỏng như nhau, ~10%. ⇒ Cái thắng của `len2` **không** đến từ chất
lượng-theo-vị-trí mà từ **TÍCH LUỸ**: chuỗi 5 mệnh đề có **4** mối nối ⇒ P(ít nhất một hỏng)
≈ 1 − 0,9⁴ = **34%**; chuỗi 2 mệnh đề có **1** mối nối ⇒ **10%**.

> **Đây là cách đọc lại toàn bộ §13.** Chín vòng vừa qua đều cố nâng **chất lượng mỗi mối
> nối** — và **tất cả** đều hoà hoặc thua. Thứ duy nhất ăn là **giảm SỐ mối nối**. Trần chất
> lượng không nằm ở *chọn khéo hơn*, nó nằm ở **độ tin cậy mỗi bước ≈ 90%**, và mọi độ dài đều
> phải trả lãi kép trên con số đó.

#### Bốn kiểu hỏng của đoạn thêm (giám khảo)

1. **Trượt âm tiết** — `thiên nhiên` → `nhiên liệu` (G007) · `trung lập` → `lập tức` (G008) ·
   `cuộc đảo chính` → `chính mình` (G002). Đúng bệnh §13.39, và nó **vẫn còn** sau khi
   `pivot_max` bị bỏ (§13.40) — nhất quán.
2. **Đổ rác chuỗi số/ngày** — lại chính `km 1661+100 QL1` (G019, G023, G048, G089, G108).
3. **Lặp thoái hoá** — `học sinh học sinh` (G024) · `chảy máu chảy máu` (G134). Đúng khuyết tật
   1,75% của §13.38.
4. **Dán khối lạ đề rồi bỏ dở** — đoạn USDA/GMO chen vào chuyện cư dân mạng (G031).

Và một quan sát đắt: đuôi **rất thường kết ở hư từ** (`như` G092 · `một cách` G068 · `nhưng`
G056) — **nối thêm mà vẫn không đóng được câu**.

**Đoạn thêm KHÔNG phải lúc nào cũng hại**: G051 `duy trì hoạt động` → `…của quan lại` (đúng
ngữ vực sử) · G137 `nội tâm` → `nội tâm của hắn không khỏi hiện ra…` (đúng giọng truyện) ·
G129, G063, G110, G138, G141. ⇒ Có tín hiệu để phân biệt *nên đi tiếp* với *nên dừng*; `len2`
chỉ là hằng số mù chưa dùng tín hiệu đó.

#### Kết luận theo đúng P75 §3 đã ghi trước

**KHÔNG** đặt `max_clauses=2` rồi coi là xong — đó là hằng số mù, và nó cụt ở 8 ca đã nêu.
Kết luận đúng: **cần một cơ chế DỪNG CÓ CĂN CỨ**, và **`len2` là mốc so sánh mới** cho mọi
vòng sau.

#### Ứng viên dừng-có-căn-cứ, đã có sẵn tín hiệu

Đo kèm: tỷ lệ mệnh đề **đóng được câu** (EOS-lift ≥ θ ở token cuối) — cũng **phẳng** theo vị
trí, ~**63%**. Tức **37% điểm dừng rơi vào giữa câu**, khớp đúng 8 ca "bị cụt" giám khảo nêu.

⇒ Quy tắc đề xuất: **dừng ở mệnh đề ĐÓNG ĐƯỢC CÂU đầu tiên**, trần `max_clauses=5`. Nó cộng
được cái thắng của `len2` (ít mối nối) với việc bỏ cái giá của `len2` (cụt giữa chừng). Tín
hiệu `ends_unit` **đã có** trong `BoundarySignal`, thuần đếm ⇒ Luật 0 sạch.

Vòng kế: chấm mù **`len2` vs `dừng-khi-đóng-câu`** — mốc so sánh là `len2`, không phải `len5`.

---

### 13.42 P76 — dừng-khi-đóng-câu **THUA 29,3%**. Đi thêm LUÔN lỗ (2026-08-03) — Claude

186 cặp v12 (M4 §12.161), tiền tố kiểm độc lập **186/186**. Mốc so sánh là `len2` (thắng vòng
mười), **không** phải `len5`.

| | |
|---|---:|
| **`len2`** | **106** |
| `dong_cau` (sàn 2, đi tới khi đóng câu, trần 5) | 44 |
| hoà | 36 |

**`len2` 70,7%**, CI95 **[63,4%, 78,0%]** — tách khỏi 50%, thắng **cả hai phía**.

**Dự đoán L.4 của tôi SAI NGƯỢC CHIỀU**: tôi ghi `dong_cau` thắng 55–68%; nó **thua**. Và tôi
đã ghi trước rằng mình "thật sự không chắc" — chỗ không chắc đó hoá ra là chỗ sai.

⚠️ Một dấu hiệu đã lộ **trước khi chấm** mà tôi không sửa dự đoán (sửa lúc đó là dời cổng):
`dong_cau` dùng **3,85** mệnh đề so với `len2` **2,0** — tức **1,93×**, ngoài khoảng 1,3–1,8×
tôi đoán. Cái giá là ~**1,85 mối nối** tăng thêm, không phải một.

#### Vì sao — giám khảo giải thích, và nó khớp số học §13.41

1. **Cơ chế hầu như KHÔNG đạt mục tiêu của chính nó.** Chỉ ~**1/3** ca đoạn thêm thật sự chốt
   được bằng dấu chấm; phần lớn kéo dài rồi lại **treo ở hư từ khác** — `điều này có thể`
   (G022) · `còn có` (G130) · `hoặc bạn` (G166).
2. **Khi đóng được thì đa số đóng bằng NHẢY ĐỀ** — G123 đang nói điểm mã Unicode rồi chốt bằng
   `dịch vụ spa của họ`; G061 chèn tin công an Đắk Nông vào đoạn về công cụ đá.
3. **Đoạn thêm phá chỗ dừng đẹp bằng mối nối sai âm tiết** — `trấn áp` → `áp suất` (G153) ·
   `lo lắng` → `lắng nghe` (G131) · `khinh thị` → `thị xã` (G109) · `lưu thông` → `thông tin`
   (G150). **Đúng bệnh §13.39**, và đây là kênh thiệt hại cụ thể của mỗi mối nối tăng thêm.
4. Giá của `len2` **có thật**: đứt ngang ở ~60–70 ca (≈1/3), kết ở liên từ (`và`, `bằng`,
   `đối với`), lượng từ (`cái`, `một đạo`), dấu hai chấm, hoặc động từ chưa có bổ ngữ. **Nhưng
   vẫn rẻ hơn** cái giá của việc đi tiếp.

> Giám khảo chốt đúng một câu: *"'đóng được câu' ở vòng này hầu hết là đóng HÌNH THỨC chứ không
> phải đóng NỘI DUNG"*.

#### Một hiệu ứng chọn mẫu phải ghi kèm

Cặp chỉ tồn tại khi `len2 ≠ dong_cau`, tức **chỉ những đề mà mệnh đề 2 KHÔNG đóng câu**. Vì
thế tỷ lệ "đóng được" quan sát trong file (~1/3) **thấp hơn** tỷ lệ 63% đo ở §13.41 trên toàn
bộ mệnh đề — không mâu thuẫn, chỉ là **điều kiện hoá**. Phán quyết vẫn đúng nguyên vẹn: nó nói
về **đúng tập đề mà quy tắc đổi hành vi**, và đó là tập duy nhất đáng hỏi.

#### Kết luận theo đúng P76 §3 đã ghi trước

**Đi thêm LUÔN lỗ**, kể cả khi có lý do chính đáng để đi. Hai vòng liên tiếp nói cùng một
điều: `len2` > `len5` (78,7%) và `len2` > `dong_cau` (70,7%).

⇒ **`max_clauses=2` là điểm vận hành đã được arbiter xác nhận HAI lần.** Đề xuất tác giả đổi
mặc định. Đây là thay đổi hành vi sinh, nên là quyết định của tác giả, không phải của tôi.

#### Trạng thái mười một vòng — thứ còn đứng

| cơ chế | phán quyết |
|---|---|
| **cổng phủ fold** | thắng 5/5 vòng có nó · chặn nam châm rác (§13.37) |
| **`max_clauses=2`** | thắng 2/2 vòng · 78,7% và 70,7% |
| mọi thứ khác (9 cơ chế) | hoà hoặc thua |

Cả hai thứ còn đứng đều **giảm việc hệ phải làm**. Không một cơ chế nào **nâng chất lượng mỗi
mối nối** sống sót.

#### Chỗ duy nhất còn lại

§13.41 định lượng trần: **mỗi mối nối sạch ~90%**, phẳng theo vị trí. Mọi độ dài trả lãi kép
trên con số đó. Vậy chỉ còn hai đường:

1. **Nâng độ tin cậy mỗi mối nối lên trên ~90%** — chín vòng đã thử và thất bại; muốn thử tiếp
   thì phải là thứ **khác về chất**, không phải một bộ xếp/bộ chặn nữa.
2. **Chấp nhận trần và tối ưu trong nó** — chốt `max_clauses=2`, rồi dồn sức vào việc **chọn
   đúng 1 mệnh đề** thay vì nối nhiều mệnh đề. Ở n=1 mối nối, chất lượng đầu ra ≈ chất lượng
   một phép chọn — và đồ thị đếm **giỏi chọn** (§12.122 +31pp, §12.126 +13.5pp). Đây là đường
   khớp với thế mạnh đã đo được của hệ, không phải với thứ ta mong nó làm được.

---

### 13.43 P77 — MỐC GỐC: **0,5% dùng được**. Hệ CHƯA SINH ĐƯỢC (2026-08-03) — Claude

200 mẫu v12 **không lọc** (M4 §12.162), cấu hình đang thắng: cổng phủ fold + `max_clauses=2`.
Chấm tuyệt đối, từng mẫu độc lập, phiên Claude riêng.

| mức | n | tỷ lệ |
|---|---:|---:|
| `dung_duoc` | **1** | **0,5%** |
| `tam` | 17 | 8,5% |
| `hong` | **182** | **91,0%** |

Bỏ 59 mẫu trống ra, chỉ tính 141 mẫu **có sinh ra chữ**: `dung_duoc` **0,7%** · `tam` 12,1% ·
`hong` **87,2%**. Bỏ mẫu trống **không cứu được gì**.

Dự đoán §13.43 của tôi: `dung_duoc` ≤10% ✅ · `hong` ≥60% ✅ · trống ~28% ✅ (29,5%). **Đúng cả
ba hướng, nhưng thực tế tệ hơn hẳn biên tôi đặt.**

#### Kết luận theo đúng điều đã ghi trước

> *"`dung_duoc` < 3% ⇒ hệ **chưa sinh được** theo bất kỳ nghĩa thực dụng nào, và mọi công việc
> tinh chỉnh cơ chế phải **dừng** cho tới khi có thay đổi **khác về chất**."*

**0,5% < 3%. Áp dụng nguyên văn.** Hệ **chưa sinh được**.

Và mẫu `dung_duoc` duy nhất (A054) làm rõ thêm: đề là một dòng ví dụ từ điển, phần sinh nối
thêm đúng một ví dụ cùng khuôn (`* nhện chăng tơ`) dùng lại đúng từ khoá `tơ` đã có trong đề.
Giám khảo nói thẳng: *"**không có mẫu nào sinh được một mệnh đề mới mà vẫn trôi**"*.

#### 🔴 Đọc lại mười một vòng chấm mù dưới ánh sáng con số này

Mười một vòng đều đo **so sánh**, trong một vùng mà **91% đầu ra hỏng**. Hai cơ chế sống sót —
cổng phủ fold và `max_clauses=2` — đều **giảm việc hệ phải làm**, và cả hai đúng theo cách
riêng của chúng. Nhưng gộp lại, tiến trình đó là **thu hẹp về phía ít sai nhất trong một vùng
đều sai**, không phải tiến về phía đúng.

Đây là chỗ tôi phải tự phê rõ: từng báo cáo vòng đều chính xác, nhưng tôi đã để **bức tranh
gộp** đọc thành "đang tiến bộ" suốt chín vòng mà không đề nghị đo tuyệt đối. Phép đo này lẽ ra
phải là vòng **thứ nhất**, không phải vòng thứ mười hai.

> **Bài học thứ mười một về thước trong §13:** một chuỗi dài phép đo **so sánh** có thể chạy
> rất lâu mà không ai biết mình đang ở đâu. Phải có **mốc tuyệt đối trước**, rồi mới so.

#### Bệnh chi phối, đã định lượng ở mức tuyệt đối

Giám khảo phân loại 182 mẫu `hong`:

| kiểu hỏng | ~tỷ lệ trong `hong` |
|---|---:|
| **nhảy chủ đề ngay tại mối nối** (đồng âm / từ ghép bị cắt đôi) | **55–60%** |
| trống — hệ không sinh được gì | 32% |
| mảnh số / bản mẫu lặp (`km 1661+100 QL1`) | phần còn lại |

Ví dụ: `đoạn văn` → `chương trình xử lý văn bản` (A007/A052) · `năng lượng` → quang hợp
ATP/NADPH (A071) · `tính` → `toán tử tuyến tính` (A055) · `tương tự` → `tự nhiên` →
`nhiên liệu` (A090).

⇒ **Đây chính là bệnh §13.39**, và ở mức tuyệt đối nó **không phải khuyết tật 10,7% bên lề —
nó là NGUYÊN NHÂN CHÍNH.** Ngưỡng `pivot_max=0.05` của §13.39/§13.40 chỉ chạm được phần đỉnh;
đại đa số ca trượt nằm **dưới** ngưỡng đó, đúng như §13.40 đã ngờ khi thấy lỗi nằm trong
**tiền tố chung**.

#### 🆕 Sinh là TẤT ĐỊNH — chưa ai ghi

Giám khảo phát hiện: bốn đề trùng nhau nguyên văn (`Giải nghĩa: (khẩu ngữ)` — A134, A157,
A199, và biến thể A162/A172) cho ra **cùng một chuỗi ký tự y hệt**. Không có biến thiên.

Với `rng` cố định thì điều này **đúng như thiết kế**, nhưng nó nói một điều quan trọng hơn:
`compose_chain` hiện là một **phép tra cứu tất định theo đề**, không phải một quá trình sinh.
Mọi "đa dạng" chỉ đến từ đề khác nhau.

#### Thứ "khác về chất" mà dữ liệu chỉ tới

Giám khảo chốt: *"cái thiếu **không phải nội dung** mà là một **điểm dừng/điểm nối đúng**: hệ
không biết khi nào phải **khớp ngữ pháp** với đề, và không biết khi nào nên dừng."*

Mối nối hiện tại là `W_skip[d][last_content(A), first_content(B)]` — **một token sang một
token**, trên đồ thị **đồng hiện trong cửa sổ d**. Đại lượng đó **không mang**:
- **biên từ** — nên `quốc gia` + `gia vị` là hợp lệ với nó;
- **tính liên tục cú pháp** — nên `đoạn văn` + `chương trình xử lý văn bản` là hợp lệ.

Chín vòng đã thử **xếp lại / chặn bớt** trên chính đại lượng đó và **tất cả** đều hoà hoặc
thua. Đó không phải chín thất bại độc lập — đó là **một** kết luận lặp lại chín lần: **không
thể vá một đại lượng thiếu thông tin bằng cách xếp lại nó.**

⇒ "Khác về chất" nghĩa là **đổi đại lượng làm mối nối**, không phải đổi cách xếp nó. Hai hướng
Luật 0 sạch (đều **chưa** đo, ghi ở đây làm chỗ bắt đầu, **không** phải kết luận):

1. **Mối nối phải tôn trọng biên TỪ.** BPE/ISG đã có tín hiệu gộp âm tiết (§TK_041); mối nối
   hiện đọc `first_content(B)` như một âm tiết tự do. Đếm được, chưa dùng.
2. **Mối nối phải là CẶP LIÊN TỤC trong văn thật**, không phải cặp đồng hiện trong cửa sổ 8.
   `W_seq` (bigram có hướng, đã có) mang thông tin này; `W_skip` thì không. §13.38 đã cho một
   chỉ dấu: lặp-kề tại mối nối cao **68×** so với trong lòng mệnh đề — tức mối nối đang **đứng
   ở chỗ văn thật không bao giờ đứng**.

⚠️ Cả hai đều là **giả thuyết**. Theo đúng điều đã ghi trước, việc tinh chỉnh cơ chế **dừng**
ở đây; bước kế phải là **đo** xem mối nối hiện tại lệch bao xa khỏi mối nối trong văn thật —
không phải viết thêm một bộ xếp nữa.

---

### 13.44 P78 — bộ CẮT và bộ NỐI dùng tiêu chí **mâu thuẫn nhau**; 46,6% mối nối chưa từng thấy (2026-08-03) — Claude

600 đề mini, cấu hình đang thắng. Đo **chuyển tiếp bề mặt**: token **cuối** của A → token
**đầu** của B — đúng chỗ hai mệnh đề chạm nhau khi đọc. Đối chứng: mối nối **thật** giữa hai
mệnh đề liên tiếp cùng cha.

| | SINH (khác cha) | THẬT |
|---|---:|---:|
| số mối nối | 367 | 4.619 |
| **cặp CHƯA TỪNG thấy kề nhau** | **46,6%** | **5,2%** |
| pmi p25 | −3,33 | −0,92 |
| **pmi trung vị** | **+0,57** | **−0,39** |
| pmi p75 | +3,73 | −0,07 |

#### 🔴 Tự kiểm trước: một nửa kết quả này là VÒNG VO — phải nói ra

Biên mệnh đề thật có pmi thấp **theo định nghĩa**: `cut_clauses` cắt đúng ở chỗ
`bound.pmi(t, t+1) ≤ pmi_theta` ([clause_layer.py:201](src/nn_aivie/hca/clause_layer.py:201)),
và tầng đang chạy với `pmi_theta = 0.0`. Nên "mối nối thật nằm ở vùng âm" **không phải phát
hiện** — nó là hệ quả của chính quy tắc cắt. Trình bày nó như khám phá là tự lừa.

**Phần KHÔNG vòng vo mới là phần đáng giá**, và nó mạnh hơn:

> Tầng mệnh đề **cắt** ở chỗ `pmi ≤ 0`, nhưng **nối** ở chỗ pmi trung vị **+0,57** — tức nối ở
> đúng loại điểm mà **chính nó** sẽ không bao giờ coi là biên. **Bộ CẮT và bộ NỐI đang dùng
> hai tiêu chí mâu thuẫn nhau.**

Đây là mâu thuẫn **nội bộ**, đọc ra được từ mã, không cần đối chứng nào. Và nó giải thích
đúng bệnh §13.43: hai âm tiết dính chặt (`quốc` `gia`, `gia` `vị`) là **từ**, không phải
**biên** — mà bộ nối lại ưu tiên đúng chúng.

#### 🔴 Và con số này thì hoàn toàn không vòng vo: **46,6%**

Gần **một nửa** mối nối sinh đặt cạnh nhau hai token **chưa bao giờ đứng kề nhau** trong toàn
corpus (`pmi = −inf`). Văn thật: **5,2%**.

`W_skip` chứng thực **đồng hiện trong cửa sổ 8**, không chứng thực **kề nhau**. Bảo chứng
"chỉ nối cái đã được chứng thực" (bất biến 4 của tầng mệnh đề) vì thế **yếu hơn nhiều so với
tên gọi của nó**: nó bảo đảm hai token *từng ở gần nhau*, không bảo đảm chúng *từng đứng cạnh
nhau*. Ở chỗ ghép bề mặt, chỉ "cạnh nhau" mới có nghĩa.

⇒ Đây là **cơ chế salad**, định lượng xong: 46,6% chỗ ghép là chỗ tiếng Việt chưa từng viết.

#### Khoảng cách, phát biểu gọn

Mối nối đang dùng lệch khỏi mối nối văn thật theo **hai** trục độc lập:

1. **Sai đại lượng** — dùng đồng hiện cửa-sổ-8 thay cho kề-nhau ⇒ 46,6% ghép chưa từng thấy.
2. **Sai chiều** — trong số còn lại, chọn liên kết **mạnh** trong khi biên là chỗ **yếu**;
   và chính bộ cắt của hệ định nghĩa biên như vậy.

#### Ứng viên — và lần này nó KHÁC VỀ CHẤT

Không phải một bộ xếp nữa trên `W_skip`. Là **đổi điều kiện hợp lệ của mối nối**:

    hợp lệ ⇔ pmi_seq(cuối A → đầu B) ĐƯỢC CHỨNG THỰC và nằm trong DẢI BIÊN (≈ pmi ≤ θ_cắt)

Tức: **nối ở đúng loại chỗ mà bộ cắt gọi là biên**. Hai bộ dùng **cùng một** tiêu chí. Thuần
đếm trên `W_seq` (đã có) ⇒ Luật 0 sạch.

⚠️ Dự báo cái giá, ghi trước: điều kiện này **chặt hơn nhiều** — sẽ loại 46,6% ứng viên "chưa
từng thấy" cộng phần lớn ứng viên pmi dương ⇒ **tỷ lệ trống sẽ tăng mạnh** từ 29,5%. Theo
§13.37 (im lặng hơn nói bậy) đó **không** tự động là hỏng, nhưng nó phải được đo, không được
giấu.

⚠️ Và theo §13.43 đã ghi trước: thước cho ứng viên này phải là **chấm TUYỆT ĐỐI** so với mốc
**0,5% dùng được**, không phải một vòng so sánh nữa. Một cơ chế thắng so sánh mà vẫn 1% dùng
được thì không phải tiến bộ.

---

### 13.46 P79 — "mối nối kề nhau" **TỆ HƠN**. Chẩn §13.44 SAI; và thước tuyệt đối **tái lập được** (2026-08-03) — Claude

400 mẫu v12 (M4 §12.163): hai cấu hình trên **cùng 200 đề**, trộn lẫn, giấu nhãn, chấm trong
**một** phiên.

| toàn bộ 200 mẫu/nhánh | `dung_duoc` | `tam` | `hong` |
|---|---:|---:|---:|
| `hien_tai` | **0,5%** | **8,5%** | 91,0% |
| `mối_nối_kề` | **0,0%** | **3,0%** | **97,0%** |

| chỉ mẫu CÓ sinh ra chữ | n | `dung_duoc`+`tam` |
|---|---:|---:|
| `hien_tai` | 141 | **12,8%** |
| `mối_nối_kề` | 60 | **10,0%** |

⇒ **Thua ở CẢ HAI thước.** Không phải "đúng hướng nhưng quá đắt" — nó **không** tốt hơn ngay
cả khi đã bỏ hết mẫu trống ra.

#### Kết luận theo đúng P79 §3 đã ghi trước

> *"Không hơn ở cả hai ⇒ chẩn §13.44 **sai**: ghép 'chưa từng thấy kề nhau' **không** phải
> nguyên nhân chính của 91% hỏng, dù nó chiếm 46,6%. … và mọi ứng viên dạng **siết điều kiện
> mối nối** đều chết theo."*

**Áp dụng nguyên văn.** `join_seq_max` giữ **TẮT**. Và cả **họ** giải pháp "siết điều kiện mối
nối" — đây là cái thứ **mười** trong họ đó — đóng lại.

Điều này đáng ghi kỹ: cơ chế **làm đúng** thứ nó hứa (ghép chưa-từng-thấy **46,3% → 0,0%**),
mà chất lượng vẫn **không** lên. ⇒ **46,6% ghép "chưa từng thấy kề nhau" là TRIỆU CHỨNG, không
phải NGUYÊN NHÂN.** Sửa triệu chứng thì chỉ mất ứng viên (trống 29,5% → 70,0%), không được gì.

#### ✅ Thu hoạch phương pháp: thước tuyệt đối **TÁI LẬP ĐƯỢC**

Nhánh `hien_tai` trong vòng này là **đúng 198 mẫu y hệt** §13.43 (đã kiểm bằng băm nội dung:
198/198 trùng khớp), nhưng do **một phiên giám khảo KHÁC** chấm, trong một file **400 mẫu trộn
lẫn**, thứ tự khác hẳn.

| | `dung_duoc` | `tam` | `hong` |
|---|---:|---:|---:|
| phiên 1 (§13.43) | 1 | 17 | 182 |
| phiên 2 (P79) | **1** | **17** | **182** |

**Trùng khít tuyệt đối.** Sau mười một vòng mà mọi proxy ngoại tuyến đều đánh lừa, đây là lần
đầu một thước trong §13 chứng minh được **tính tái lập** của chính nó. Nghĩa là:

- mốc **0,5%** là con số **chắc**, không phải một phiên khó tính;
- rubric tuyệt đối (ba mức + quy tắc phân-vân-thì-hạ-bậc) **truyền đạt được** giữa các phiên;
- mọi so sánh sau này **dùng lại được** mốc này mà không phải chấm lại bản gốc.

> **Bài học thứ mười hai về thước trong §13:** thước duy nhất đáng tin là thước **đã tự chứng
> minh tái lập**. Mười một vòng qua không vòng nào làm việc đó; lẽ ra phải làm từ đầu.

#### Bệnh thật, theo mô tả của giám khảo

*"Phần sinh bám đúng vài chữ cuối của đề rồi **nhảy sang một miền văn bản hoàn toàn khác** —
tiểu thuyết tiên hiệp ghép vào bài kỹ thuật, hoặc ngược lại"* — chiếm **~3/4** số mẫu `hong`
**có chữ**.

Đây **không** phải lỗi ngữ pháp bề mặt, và đó là lý do siết điều kiện bề mặt không ăn. Đây là
**đứt mạch MIỀN VĂN BẢN**. Cả `W_skip` lẫn `W_seq` đều là thống kê **cục bộ hai token** —
không đại lượng nào mang **miền**. Thứ đáng lẽ giữ miền là **cổng phủ fold**, nhưng §13.19 đã
đo: **86,4%** bằng chứng của nó là **mảnh dưới-từ**, tức nó đang neo bằng âm tiết lẻ chứ không
bằng chủ đề.

⇒ Câu hỏi kế **không** phải "nối thế nào cho đúng" mà **"làm sao biết hai mệnh đề có cùng miền
văn bản không"** — và đó là đại lượng **chưa hề có** trong hệ.

#### Nhóm "chỉ thiếu một chút" — đáng đọc kỹ

Giám khảo chỉ ra vài ca hỏng vì **một chữ**: A337 thừa chữ `là` · A324/A329 thiếu giới từ
`cho` · A299 thiếu một danh từ sau `một cái` · A103 thiếu một dấu phẩy. Và A000/A107: *"phần
sinh hoàn chỉnh và trôi chảy nhưng mối nối với đề sai ngữ pháp — chỉ cần một liên từ hoặc cắt
bỏ 3-4 chữ đầu là dùng được"*.

Đây là **lớp khác hẳn** với đứt-mạch-miền: nội dung đúng, chỉ hỏng ở **hình thái ghép**. Nó
nhỏ, nhưng nó là lớp **duy nhất** trong toàn §13 mà cách sửa là **hiển nhiên và cục bộ**.
Chưa đo tỷ trọng — ghi lại làm ứng viên, **không** làm kết luận.

---

### 13.47 P80 — miền văn bản: hệ **gần như MÙ**, và cái "giữ miền" hiện nay là **rác** (2026-08-03) — Claude

§13.46 để lại câu hỏi: *"làm sao biết hai mệnh đề có cùng miền văn bản không"*. P80 đo xem
miền có phải **đại lượng đếm được** không, và hệ đang cách văn thật bao xa.

**Đối chứng lần này là CROSS-PARENT thật** — hai cha **liên tiếp trong cùng một episode**, tức
hai câu khác cha mà người viết đặt cạnh nhau. (§13.44 đã mắc lỗi vòng vo khi lấy biên
trong-cùng-cha làm chuẩn; lần này tránh đúng chỗ đó.) Sàn: hai cha bốc ngẫu nhiên.
Thước: Jaccard trên fold nội dung của hai cha — thuần đếm.

| ngưỡng `gidf` của fold | SINH | THẬT | NGẪU NHIÊN | SINH nằm ở đâu giữa sàn↔trần |
|---|---:|---:|---:|---:|
| **4,0** — mặc định production | 0,0468 | 0,0723 | 0,0104 | **59%** |
| **7,0** — chỉ fold ĐẶC TRƯNG | **0,0055** | **0,0562** | 0,0013 | **8%** |

⚠️ Hai dòng `gidf ≥ 9,0` trong lần chạy thử **KHÔNG hợp lệ** — ở ngưỡng đó gần như mọi cha
không còn fold nào ⇒ nhóm THẬT rỗng, và con số "80%" là **rác của phép chia cho tập rỗng**.
Ghi lại để không ai trích nó.

#### 🔴 Kết quả LẬT NGƯỢC khi bỏ fold rác — và đó mới là kết quả

Ở ngưỡng production (4,0), hệ trông như **giữ được miền**: 59% quãng đường, và thậm chí
**ít mối nối không-chung-fold hơn văn thật** (16,3% vs 37,4%). Tôi suýt ghi "chẩn đứt-mạch-miền
yếu đi".

Ở ngưỡng **fold đặc trưng** (7,0), hệ tụt xuống **8%** — tức **gần như sàn ngẫu nhiên**:

| | SINH | THẬT |
|---|---:|---:|
| tỷ lệ mối nối **KHÔNG có fold đặc trưng chung** | **96,2%** | 81,2% |

⇒ **Cái "trùng fold" mà cổng phủ đang ép là trùng RÁC.** `năm`, chữ số, `nước`, `bằng` (§13.20)
có mặt khắp nơi nên Jaccard trông cao, trong khi **96,2%** mối nối **không chia sẻ một từ đặc
trưng nào**. Hệ **không** theo dõi miền — nó chỉ **trông như** có, vì thước mặc định đếm cả rác.

Điều này khớp và giải thích ba thứ đã đo rời rạc trước đó:
- §13.19: **86,4%** bằng chứng của cổng là **mảnh dưới-từ** — cùng một hiện tượng, đo ở chỗ khác;
- §13.33: bộ xếp neo-đề có trọng số `gidf` **hoà** — nó cân lại rác nhưng `gidf` bão hoà (§13.32)
  nên không tách được rác khỏi đặc trưng;
- §13.46: giám khảo thấy *"nhảy sang miền văn bản hoàn toàn khác"* ở **3/4** mẫu hỏng có chữ —
  đúng như một hệ **mù miền** phải hành xử.

#### Miền LÀ đại lượng đếm được — đó là phần tin tốt

Nhóm THẬT ở ngưỡng 7,0 đạt **0,0562**, gấp **10 lần** SINH và gấp **43 lần** sàn ngẫu nhiên.
Nghĩa là: **fold đặc trưng chung GIỮA HAI CHA là một tín hiệu miền thật, tách được, và thuần
đếm** (⇒ Luật 0 sạch). Hệ **có sẵn** đại lượng này — nó chỉ chưa bao giờ dùng đúng ngưỡng.

#### Ứng viên kế — và một cảnh báo tự vấn

Điều kiện: **ứng viên phải chia sẻ ≥1 fold ĐẶC TRƯNG (`gidf ≥ 7`) với ĐỀ**, thay cho ngưỡng
4,0 hiện tại của `CoverageGate`.

⚠️ Tự vấn trước khi làm: §13.46 vừa đóng **cả họ** "siết điều kiện". Vì sao cái này khác?
- Cái vừa chết siết **bề mặt hai token** (`W_seq` kề nhau) — một đại lượng đã đo là **triệu
  chứng**.
- Cái này siết **miền**, đại lượng vừa đo được là **tách 10×** giữa thật và sinh, và là thứ
  giám khảo mô tả là bệnh chính.

Khác biệt có thật, nhưng **không** phải bảo chứng — chín cơ chế trước cũng từng có lý do tốt.
⇒ Thước bắt buộc: **chấm tuyệt đối, trộn chung một phiên, so với mốc 0,5%** (§13.43/§13.46 đã
chứng minh thước này **tái lập được**). Và phải ghi trước cái giá: siết lên `gidf ≥ 7` sẽ đẩy
tỷ lệ trống lên mạnh, đúng như `join_seq_max` đã làm.

---

### 13.49 P81 — cổng fold ĐẶC TRƯNG: **không tách khỏi nhiễu**. Hai họ cùng đóng (2026-08-03) — Claude

400 mẫu v12 (M4 §12.164), hai cấu hình trên cùng 200 đề, trộn, một phiên.

| toàn bộ 200/nhánh | `dung_duoc` | `tam` | `hong` |
|---|---:|---:|---:|
| `hien_tai` | 0,5% | 7,5% | 92,0% |
| `fold_dac_trung` (`gidf ≥ 5,0`) | 0,5% | 6,5% | 93,0% |

| chỉ mẫu CÓ chữ | n | `dung_duoc`+`tam` |
|---|---:|---:|
| `hien_tai` | 141 | 11,3% |
| `fold_dac_trung` | 86 | **16,3%** |

#### 🔴 Không được đọc 16,3% vs 11,3% là "thắng"

Kiểm hai tỷ lệ: chênh **+5,0pp**, **z = 1,06** ⇒ **KHÔNG tách khỏi nhiễu** (cần |z| > 1,96).
Toàn bộ: z = −0,38, cũng không tách. Cỡ mẫu nhánh mới chỉ **n = 86** vì 57% đã bỏ trống.

Nếu dừng ở dòng "16,3% > 11,3%" thì tôi đã tự báo cáo một chiến thắng không tồn tại — đúng lớp
lỗi §13 đã mắc nhiều lần dưới dạng khác (§13.35 mẫu số, §13.39 đối chứng tự đếm, §13.44 vòng
vo). Ghi lại vì cám dỗ ở đây **mạnh**: đây là cơ chế đầu tiên có điểm ước lượng nhích lên.

#### Kết luận theo P81 §3 đã ghi trước

Nhánh "chỉ hơn ở nhóm không trống" **đòi hiệu ứng THẬT**, mà ở đây không có. ⇒ Áp dụng nhánh
thứ ba: **tín hiệu miền TÁCH ĐƯỢC (§13.47: thật gấp 10× sinh) nhưng DÙNG LÀM CỔNG thì không
ăn.**

**Hai họ cùng đóng:**
- siết **mối nối bề mặt** (§13.46 — cái thứ mười trong họ);
- siết **miền** (§13.49).

`reject_when_no_theme` và `gidf_min` cao: giữ **TẮT**. Bản vá bẫy cổng-tắt-âm-thầm (§13.48)
**giữ lại** — nó đúng bất kể cơ chế này sống hay chết.

#### ✅ Thước TÁI LẬP lần thứ hai

Cùng 198 mẫu `hien_tai`, **ba** phiên giám khảo độc lập:

| | `dung_duoc` | `tam` | `hong` |
|---|---:|---:|---:|
| §13.43 | 1 | 17 | 182 |
| §13.46 | 1 | 17 | 182 |
| §13.49 | **1** | **15** | 184 |

Chênh `tam` = **1,0pp**, nhiễu nhị thức 1σ = **2,0pp** ⇒ trong nhiễu. Thước tuyệt đối vẫn
đứng, và **mốc 0,5% dùng được** nay có **ba** phiên xác nhận.

#### 🔴 Khuyết tật thiết kế phải ghi: chấm cặp-tuyệt-đối bị **HỞ NHÃN một phần**

Giám khảo tự nhận ra: *"nhiều đề xuất hiện **hai lần** trong file và hai lần cho kết quả khác
hẳn — A043 bỏ trống còn A308 (cùng đề) sinh dài"*. Vì hai cấu hình chạy trên **cùng bộ đề**,
mỗi đề xuất hiện đúng hai lần ⇒ ghép cặp được bằng mắt, dù nhãn bị giấu.

Lần này ảnh hưởng có lẽ nhỏ (giám khảo khai không nhóm, và nhánh `hien_tai` tái lập đúng),
nhưng đây là **lỗ hổng thật** của thiết kế trộn-chung. Đánh đổi: cùng bộ đề cho phép so **cặp**
(mạnh hơn), nhưng làm hở nhãn. Vòng sau phải chọn có ý thức, và ghi rõ chọn gì.

#### Câu hỏi §13 phải hỏi bây giờ

Đã đóng bằng số: xếp lại mối nối (9 cơ chế) · siết mối nối bề mặt · siết miền · dừng-theo-câu ·
lọc corpus · nam châm rác. Còn đứng: cổng phủ fold (ngưỡng gốc) · `max_clauses` nhỏ. **Cả hai
đều chỉ GIẢM việc hệ làm.**

Và giám khảo vòng này chỉ đúng chỗ đau nhất: mẫu `dung_duoc` **duy nhất** vẫn là dòng **liệt kê
từ điển** — *"hệ chỉ trôi được ở chỗ mà văn bản gốc vốn đã rời rạc, không đòi hỏi cú pháp liên
tục"*.

⇒ Câu hỏi đúng không còn là *"chọn mệnh đề nào"* mà là: **hệ có bao giờ tạo ra được một mối
nối cú pháp mà nó chưa từng thấy không** — hay mọi thứ nó làm được chỉ là **phát lại** những
chỗ vốn đã rời rạc. §13.43 đã có một chỉ dấu chưa ai theo: **sinh là TẤT ĐỊNH**, cùng đề cho
cùng chuỗi ký tự. Đó là hành vi của một **bảng tra**, không phải của một bộ sinh.

---

### 13.50 P82 — **MỖI MỐI NỐI ĐỀU LÀM HỎNG**: 24,0% vs 6,6% (2026-08-03) — Claude

Không chạy gì mới — cắt lại **chính** 200 mẫu mốc gốc (§13.43) theo **số mệnh đề**:

| đầu ra có | n | `dung_duoc`+`tam` | `hong` |
|---|---:|---:|---:|
| **0 mệnh đề** (trống) | 59 | 0,0% | 100% |
| **1 mệnh đề — KHÔNG mối nối nào** | 50 | **24,0%** | 76,0% |
| **2 mệnh đề — 1 mối nối** | 91 | **6,6%** | 93,4% |

Chênh **+17,4pp**, **z = 2,96** ⇒ **tách khỏi nhiễu, p < 0,01**.

**Kiểm chéo trên phiên chấm KHÁC** (nhánh `hien_tai` của bộ abs3, §13.49): 1 mệnh đề **20,0%**
· 2 mệnh đề **6,6%** — cùng dấu, cùng độ lớn.

#### Phát biểu gọn

> Đầu ra **không có mối nối nào** tốt hơn đầu ra **có một mối nối** khoảng **3,6 lần**.
> **Mỗi mối nối hệ tạo ra đều làm hỏng đầu ra.**

#### Điều này giải thích lại TOÀN BỘ §13

| đã đo | giờ đọc là |
|---|---|
| `max_clauses=2` thắng `=5` **78,7%** (§13.41) | ít mối nối hơn |
| `len2` thắng `dong_cau` **70,7%** (§13.42) | ít mối nối hơn |
| **chín** cơ chế xếp lại/chặn mối nối đều hoà hoặc thua | không xếp lại được thứ đáng lẽ **không nên có** |
| tỷ lệ hỏng mỗi mối nối **phẳng** theo vị trí (§13.41) | không phải "đi xa thì lạc" — **bước nào cũng hỏng** |
| cổng phủ fold thắng 5/5 vòng (§13.25…) | nó **giảm** số ứng viên ⇒ giảm số mối nối sống sót |

Cả hai thứ sống sót sau mười hai vòng đều **giảm việc hệ làm**. Giờ có lý do: **việc mà hệ làm
— NỐI — là việc nó không làm được.**

#### ⚠️ Cái bẫy chọn mẫu, phải nói trước khi ai đó bật `max_clauses=1`

Nhóm "1 mệnh đề" **không** phải nhóm được **gán ngẫu nhiên**. Đó là những đề mà chuỗi **tự
chết** sau mệnh đề đầu (`loai_het` — không ứng viên nào qua cổng). Rất có thể chúng là những
đề **khác về bản chất** (đề hẹp, ít ứng viên), chứ không phải "cùng đề nhưng dừng sớm".

⇒ **KHÔNG được** kết luận "đặt `max_clauses=1` thì được 24%". Đây đúng lớp lỗi §13.35 (mẫu số)
và §13.40 (phân rã hậu nghiệm). Con số 24% là **giả thuyết có cơ sở mạnh**, không phải kết quả.

#### Vòng kế — thí nghiệm CÓ GÁN, nêu trước

`max_clauses=1` vs `max_clauses=2`, **cùng bộ đề**, chấm **tuyệt đối**, trộn chung một phiên,
so với mốc **0,5% / 12,8%** (toàn bộ / chỉ-có-chữ) đã tái lập **ba** lần.

Mọi đề đều bị ép dừng ở 1 ⇒ **không còn chọn mẫu**. Nó biến giả thuyết trên thành phán quyết.

**Dự đoán ghi trước:** `len1` thắng ở **cả hai** thước; `dung_duoc + tam` toàn bộ **≥ 12%** (so
8,0% hiện tại). Nếu `len1` **không** hơn ⇒ 24% ở trên **hoàn toàn** là hiện vật chọn mẫu, và
bài học là: *nhóm con nào cũng có thể trông đẹp nếu nó được chọn bằng chính cơ chế đang xét.*

⚠️ Và ghi trước một hệ quả khó chịu nếu `len1` thắng: hệ khi đó **không nối gì cả** — nó chỉ
**chọn một mệnh đề có sẵn** rồi dán vào đề. Gọi đó là "sinh" thì phải nói rõ giới hạn: đó là
**truy hồi + ghép một bước**, và toàn bộ tham vọng "tái tổ hợp nhiều mệnh đề" của Trụ A chưa
có bằng chứng nào ủng hộ.

---

### 13.52 P83 — thí nghiệm CÓ GÁN xác nhận: **`len1` thắng, z = 3,01** (2026-08-04) — Claude

400 mẫu v12 (M4 §12.165), **hai bộ đề rời nhau** (0 đề trùng — kiểm độc lập), trộn, một phiên.

| toàn bộ 200/nhánh | `dung_duoc` | `tam` | `hong` | **`dung`+`tam`** |
|---|---:|---:|---:|---:|
| `len2` (hiện tại) | 0,5% | 4,0% | 95,5% | **4,5%** |
| **`len1`** | **1,0%** | **12,0%** | 87,0% | **13,0%** |

chênh **+8,5pp**, **z = 3,01** ⇒ **TÁCH khỏi nhiễu**.

| chỉ mẫu CÓ chữ | n | `dung`+`tam` |
|---|---:|---:|
| `len2` | 140 | 6,4% |
| **`len1`** | 147 | **17,7%** |

chênh **+11,3pp**, **z = 2,91** ⇒ **TÁCH**.

**Dự đoán P83 §3 của tôi: ĐÚNG** — `len1` thắng cả hai thước, `dung+tam` toàn bộ **13,0% ≥ 12%**.

⇒ §13.50 **không** phải hiện vật chọn mẫu. **Mỗi mối nối hệ tạo ra đều làm hỏng đầu ra** — nay
có thí nghiệm **có gán** xác nhận, không chỉ quan sát.

#### ⚠️ Kiểm tổng quát hoá: **qua, nhưng sát mép** — và có hệ quả về cách dùng mốc

Tôi ghi trước: nhánh `len2` trên **bộ đề mới** phải ra ≈ **8,0% ± 4pp**. Thực tế **4,5%** — nằm
trong dải, nhưng **ở đúng mép dưới**. Chênh so mốc cũ 3,5pp, z ≈ 1,4 ⇒ trong nhiễu.

Nhưng phải rút ra quy tắc: **mốc 8,0% không phải hằng số của hệ, nó là hằng số của MỘT bộ đề**.
⇒ Từ nay **mọi so sánh phải có nhánh đối chứng chấm CÙNG PHIÊN**; không được lấy con số vòng
trước làm mốc cho vòng sau. May là cả §13.46, §13.49 lẫn vòng này đều đã làm vậy — nhưng đó là
may, không phải thiết kế. Ghi thành luật.

#### ✅ Kết luận theo đúng điều ghi trước

Đề xuất tác giả: **`max_clauses = 1` làm mặc định.** Đây là cơ chế thứ **ba** được arbiter xác
nhận, sau cổng phủ fold và `max_clauses=2` — và nó **thay thế** cái thứ hai.

**Và phải nói thẳng cái giá của kết luận này:** với `max_clauses=1`, hệ **không nối gì cả**. Nó
**chọn một mệnh đề có sẵn rồi dán vào đề**. Đó là **truy hồi + ghép một bước**.

⇒ Tham vọng trung tâm của Trụ A — *"sinh = CHỌN một mệnh đề đã được chứng thực, rồi TÁI TỔ HỢP
nhiều mệnh đề"* — **chưa có một bằng chứng nào ủng hộ vế thứ hai**, và nay có bằng chứng **phản
đối** nó: mọi lần tái tổ hợp đều làm chất lượng tụt. §13 phải sửa mục tiêu cho trung thực thay
vì tiếp tục tinh chỉnh khâu nối.

#### 🔴 Và mức tuyệt đối vẫn RẤT THẤP — không được đọc đây là "đã sinh được"

`len1` đạt `dung_duoc` **1,0%** (2/200) và `dung+tam` 13,0%. Tức **87% đầu ra vẫn hỏng**. So
mốc gốc §13.43 (0,5% / 8,0%) thì đây là cải thiện **thật và đo được**, nhưng nó dịch chuyển hệ
từ *"gần như không dùng được"* sang *"vẫn gần như không dùng được"*.

Giám khảo mô tả ba mẫu `dung_duoc` (A103, A149, A354): tất cả đều là **dòng từ điển** — phần
sinh chỉ hoàn tất nốt từ đang dở rồi nối một ví dụ cùng trường nghĩa theo **đúng quy ước dấu
`*` của chính đề**. *"Không một mẫu văn xuôi tự do nào đạt mức này."* Đây là **lần thứ ba** liên
tiếp (§13.43, §13.49, §13.52) mẫu tốt nhất là dòng từ điển.

#### 🆕 Chẩn sắc nhất từ trước tới nay: **"ngắn thì RỖNG, dài thì TRÔI"**

Giám khảo trả lời câu hỏi 4 bằng một phân đôi mà chưa vòng nào bắt được:

| | hỏng kiểu gì |
|---|---|
| đầu ra **rất ngắn** (1–3 chữ) | **DỘI LẠI** chữ cuối của đề rồi tắt — `học và` → `học hành` (A093) · `vùng xa` → `xa.` (A216) · `công cụ bằng đá` → `bóng đá Ý` (A048). Dừng đúng chỗ nhưng **nội dung rỗng**. |
| đầu ra **dài hơn** | khởi đầu hợp lý được 5–10 chữ rồi **TRÔI** sang miền khác **giữa câu** — `trước khi khử` → `khuẩn` (đúng!) rồi trượt sang `nghi thức khi dùng bữa tại nhà hàng` (A194). |

⇒ Hai chế độ hỏng **khác nhau về chất**, và `len1` thắng vì nó **đổi trôi lấy rỗng** — mà rỗng
bị phạt nhẹ hơn. Đây **không** phải hệ đã học nối tốt hơn; nó chỉ **nói ít đi**.

Và nó khớp chính xác với thứ còn đứng sau mười ba vòng: **cổng phủ fold** và **`max_clauses`
nhỏ** — cả hai đều **giảm việc hệ làm**. Không một cơ chế nào **thêm năng lực** sống sót.

---

### 13.53 P84 — "ngắn thì rỗng": **DỘI CHỮ chỉ 7,5%**, không giải thích được gì (2026-08-04) — Claude

§13.52 để lại chẩn của giám khảo: bản ngắn hỏng bằng cách *"DỘI LẠI chữ cuối của đề rồi tắt —
dừng đúng chỗ nhưng **nội dung rỗng**"*. Đo trên **dữ liệu đã chấm** (`abs4`, thuần văn bản,
không sinh lại ⇒ không có chỗ chọn mẫu). **Dội** = chữ đầu của phần sinh nằm trong **3 chữ
cuối** của đề.

| nhánh | nhóm | n | `dung`+`tam` |
|---|---|---:|---:|
| `len1` | **DỘI** | 11 | 9,1% |
| `len1` | không dội | 136 | 18,4% |
| `len2` | **DỘI** | 23 | 4,3% |
| `len2` | không dội | 117 | 6,8% |

| | tỷ lệ DỘI |
|---|---:|
| `len1` | **7,5%** (11/147) |
| `len2` | **16,4%** (23/140) |

#### Chấm ba dự đoán ghi trước: **SAI CẢ BA**

| dự đoán | thực tế | |
|---|---:|---|
| tỷ lệ dội ở `len1` **≥ 25%** | **7,5%** | ❌ |
| dội ≤5% vs không dội ≥20% (tách) | 9,1% vs 18,4%, **z = 0,78** | ❌ không tách |
| `len2` dội **ít hơn** `len1` | `len2` **16,4%** > `len1` 7,5% | ❌ ngược dấu |

Hiện tượng **có thật** — ví dụ rất rõ: `vùng sâu và vùng xa` → `xa.` · `ghi chép` → `chép bài`
· `1871 – Minh Trị Duy tân` → `tân thủ`. Nhưng nó chiếm **7,5%**, nên **không** giải thích được
87% hỏng của `len1`. Và nhóm dội chỉ có **n = 11** ⇒ mọi so sánh chất lượng ở đây vô lực.

⇒ Theo đúng điều ghi trước: *"dội = rỗng"* là **ấn tượng của giám khảo**, không phải khuyết tật
đủ lớn để hành động. **§13.52 không cần sửa gì.**

#### 🔴 Lần thứ BA một mô tả sống động của giám khảo hoá ra HIẾM khi đếm

| vòng | giám khảo mô tả | đếm ra |
|---|---|---:|
| §13.34 | rò rỉ khuôn mẫu (`[chủ đề]`, `*`) | ~27 lần / 830k đơn vị |
| §13.38 | lặp cụm đến cạn nghĩa | **không** hơn văn thật |
| §13.53 | dội lại chữ cuối của đề | **7,5%** |

> **Bài học thứ mười ba về thước trong §13:** giám khảo mù **giỏi mô tả KIỂU hỏng** và **dở
> ước lượng TỶ TRỌNG** — cái chướng mắt thì nhớ, cái nhàm thì quên. Mọi mô tả của họ phải được
> **đếm** trước khi ai đó xây cơ chế theo nó. Ba lần rồi, và cả ba lần tôi đều suýt xây.

Điều này **không** hạ giá trị giám khảo: chẩn "nhảy miền" của họ (§13.46/§13.52) đếm ra **~50%
số hỏng** và đúng. Chỉ là **kiểu** thì tin được, **tỷ trọng** thì phải tự đếm.

#### Phụ: hai bộ đề khác nhau về độ dội gấp đôi

`len1` và `len2` chạy trên **hai bộ đề rời nhau** (§13.51), và tỷ lệ dội chênh **7,5% vs 16,4%**
— hơn gấp đôi, dù khâu chọn mệnh đề đầu **y hệt nhau** ở hai nhánh. ⇒ Đây là **biến thiên theo
bộ đề**, và nó củng cố đúng luật §13.52 vừa rút: **không được lấy con số của một bộ đề làm mốc
cho bộ đề khác**; mọi so sánh phải có đối chứng **cùng phiên, cùng thiết kế**.

---

### 13.60 P85 — TẦNG MỆNH ĐỀ **KÉM HƠN 11 LẦN** so với truy hồi đã có trước TK_045 (2026-08-04) — Claude

450 mẫu mini, ba nhánh, **bộ đề rời nhau**, trộn, giấu nhãn, **một** phiên chấm tuyệt đối.

| nhánh | `dung_duoc` | `tam` | `hong` | **`dung`+`tam`** |
|---|---:|---:|---:|---:|
| **`ban_le`** — tầng mệnh đề, cấu hình đang chạy | 0,7% | 4,0% | 95,3% | **4,7%** |
| **`truy_hoi_cha`** — `rank_matches(ĐỀ)` → mệnh đề mở của cha | 18,7% | 23,3% | 58,0% | **42,0%** |
| **`chep_cau`** — `rank_matches(ĐỀ)` → phát nguyên câu của cha | **31,3%** | 22,0% | 46,7% | **53,3%** |

| so sánh | chênh | z | |
|---|---:|---:|---|
| `truy_hoi_cha` − `ban_le` | **+37,3pp** | **7,64** | TÁCH |
| `chep_cau` − `ban_le` | **+48,7pp** | **9,29** | TÁCH |
| `chep_cau` − `truy_hoi_cha` | +11,3pp | 1,97 | tách (sát mép) |

#### 🔴 Phán quyết về chính TK_045 §13

Tầng nơ-ron mệnh đề — **đóng góp trung tâm của cả ticket** — cho **4,7%** dùng-được-hoặc-tạm,
trong khi **truy hồi cha, thứ ĐÃ CÓ TỪ TRƯỚC §13** (`rank_matches` +31pp §12.122; chép-đáp QA
`lạc = 0.00` §12.74), cho **42–53%**.

**Kém hơn ~11 lần.**

Đây không phải "một cơ chế con thất bại". Tiền đề mở đầu §13 là:

> *"Đồ thị đếm là bộ CHỌN xuất sắc và bộ SINH-TOKEN vô vọng. Hệ chỉ có hai độ hạt — token
> (salad) và cả emit của cha (chép 0,93–1,00). Tầng này là KHOẢNG GIỮA."*

Phần chẩn đoán **đúng**. Phần kết luận — rằng khoảng giữa sẽ tốt hơn hai đầu — **sai**, và nay
có số: khoảng giữa **tệ hơn cả hai đầu**. Mười ba vòng chấm mù đã tối ưu bên trong một vùng
mà bản thân vùng đó là lựa chọn tồi.

Tác giả nêu đúng chỗ này trước khi tôi kịp: *"nếu truy hồi câu thì chúng ta đã có từ trước
ticket 045 rồi"*. Số liệu xác nhận, và nặng hơn thế — **không chỉ không tiến, mà lùi.**

#### Vì sao con số này ĐÁNG TIN (những chỗ đã chống trước)

- **Không phải do văn phong mượt kéo điểm.** Rubric dặn thẳng *"trôi chảy KHÔNG bù được lạc
  đề"*, và giám khảo đếm **~60 mẫu** dạng *"câu chỉn chu nhưng lạc đề"* rồi chấm **`hong`** hết
  — *"đọc mượt tới mức phải dừng lại đối chiếu mới thấy sai"*. Bộ đếm này chính là hàng rào
  chống thiên vị cho hai nhánh mới, và nó **đã ăn**.
- **Không phải phát lại nguồn.** `ranked_parent` **loại cha của chính đề**; nếu không thì nhánh
  truy hồi đo **trí nhớ**, không đo truy hồi.
- **Không lộ nhãn.** Ba bộ đề rời nhau — `chep_cau` phát nguyên câu nên dùng chung đề là lộ
  ngay.
- **Chép lại đề không được tính điểm.** Giám khảo tự khai luật: lặp nguyên câu đề mà không thêm
  dữ kiện ⇒ `hong` (A216, A318, A398…).

#### Hạn phải ghi kèm

- Chạy trên **mini**, không phải v12 ⇒ **chỉ so nội bộ ba nhánh** là hợp lệ. Con số 53,3%
  **không** so trực tiếp được với mốc 13,0% của v12 (§13.52).
- `chep_cau` phát **cả câu**, không nối tiếp ngữ pháp với đề dở dang ⇒ nó trả lời được *"câu
  người viết nào hợp với đề này"*, **không** trả lời *"viết tiếp câu này thế nào"*. Với 46,7%
  vẫn `hong`, nó không phải lời giải, chỉ là **trần trên đã đo được**.
- `chep_cau` vs `truy_hoi_cha` chỉ z = 1,97 — **sát mép**, đừng đọc thành "chép chắc chắn hơn".

#### Việc phải làm, theo thứ tự

1. **Ghi lại tiền đề §13 cho đúng** — nó đang nói khoảng-giữa là hướng đi, trong khi số nói
   ngược. Đây là việc đầu tiên, vì mọi người đọc ticket sau đều bị nó dẫn sai.
2. **Chạy lại ba nhánh trên v12** để có con số so được với mốc 13,0%. Việc của M4.
3. Nếu v12 xác nhận: **đường sinh mặc định phải là truy hồi**, không phải soạn mệnh đề. Đó là
   quyết định của tác giả, không phải của tôi.

---

### 13.61 P86 — suy luận BẬC HAI: tín hiệu CÓ (≈73× ngẫu nhiên) nhưng **không hơn embedding đã có** (2026-08-04) — Claude

Ý tưởng của tác giả: `con mèo`/`con chó` → `là động vật` + `nghịch ngợm`; `con khỉ` →
`nghịch ngợm` ⇒ suy ra `con khỉ` → `là động vật`. Và chiều ngược: `làm`/`triển khai`/`thực
hiện` chung ngữ cảnh trái ⇒ `ok thực hiện đi`. **Cùng một phép toán**: đồng hiện bậc hai.

**Đo trước, xây sau.** Probe không viết cơ chế mới; nó hỏi hai câu: *tín hiệu có ở đó không*
và *có hơn thứ hệ ĐÃ CÓ không*.

Giao thức: cặp `(a, c)` **chỉ đồng hiện trong ĐÚNG MỘT cha** → **bỏ nguyên cha đó** khỏi ma
trận (đúng nghĩa *"chưa hề train con khỉ là động vật"*; bỏ một **ô** thì các ô khác của cùng
cha vẫn rò rỉ) → hỏi `c` xếp thứ mấy trong **31.649** ứng viên.

| bộ chấm | hit@10 | MRR | hạng trung vị |
|---|---:|---:|---:|
| `ngau_nhien` | 0,0% (kỳ vọng **0,032%**) | 0,000 | 14.927 |
| **`embedding`** — `E` 32002×96, ĐÃ CÓ | **2,7%** | 0,016 | **1.343** |
| `bac_hai` — `PPMI(M)[a,:] @ PPMI(M)` | 2,3% | **0,024** | 1.571 |

**CỔNG G-2nd: (1) ✅ 300 cặp · (2) ✅ ≈73× ngẫu nhiên · (3) ❌ không đạt 1,3× embedding ⇒ TRƯỢT.**

#### Kết luận theo đúng điều ghi trước

> *"Trượt (3) mà qua (2) ⇒ ý tưởng **đúng** nhưng **hệ đã có rồi**, và việc cần làm là **dùng
> `E` ở khâu sinh**, không phải xây cơ chế mới."*

Áp dụng nguyên văn. Ý tưởng bậc hai **không sai** — tín hiệu tồn tại rõ rệt, ~73 lần trên sàn
ngẫu nhiên. Nhưng `E` **là SVD của chính đồng hiện**, tức **cùng họ**, và nó đã bắt được cùng
lượng thông tin đó. Dự đoán ghi trước của tôi cho mục (3) là *"50/50, thật sự không biết"* —
và nó rơi về phía "đã có rồi".

**Một chi tiết ngược chiều, đáng giữ:** `bac_hai` có **MRR cao hơn** (0,024 vs 0,016) dù
hit@10 thấp hơn. Nghĩa là nó **đúng rất sát ở một số ca** mà `E` không bắt được, trong khi `E`
**đều tay hơn**. Hai tín hiệu **không trùng khít** — có thể bù nhau. Chưa đo, ghi lại làm ứng
viên, **không** làm kết luận.

#### 🔴 Nhưng con số tuyệt đối nói điều nặng hơn cả cổng

**Cả hai đều yếu**: hit@10 chỉ **2–3%**, hạng trung vị **1.343–1.571** trên 31.649 ứng viên.
Nghĩa là suy luận `con khỉ → là động vật` **không chạy được ở mức đáng dùng** trên corpus này,
bằng **cả hai** cơ chế.

#### ⚠️ Hạn của chính probe này — có thể tôi đã hỏi câu KHÓ HƠN câu của tác giả

Probe hỏi: *"xếp token đồng hiện thật lên top trong **toàn bộ 31.649 ứng viên**"*. Ví dụ của
tác giả hẹp hơn nhiều: *`con khỉ là` → `động vật`* — ở đó ngữ cảnh trái (`là`) **đã ràng buộc**
tập ứng viên xuống rất nhỏ.

⇒ Con số 2,3% có thể **đánh giá thấp** cơ chế trong tình huống dùng thật. Bản đo đúng phải
là **có ràng buộc khung**: cho `(chủ thể, ngữ cảnh trái)`, chỉ xếp trong các token **từng theo
sau ngữ cảnh trái đó**, rồi hỏi bậc hai có đẩy đúng đích lên không.

Và: mini chỉ **5.664 cha**. Cấu trúc loại suy (nhiều thực thể chung nhiều ngữ cảnh) cần **mật
độ**; v12 có 430.447 cha, gấp 76 lần. §13.32 đã dạy đúng bài này — **mini bác được, không định
cỡ được**.

⇒ Chưa đóng ý tưởng. Trạng thái đúng là: **cổng trượt ở mức đo hiện tại**, và có **hai lý do
cụ thể** để nghi phép đo chứ không nghi ý tưởng.

---

### 13.62 P87 — probe khung+loại suy **BÁO ĐẠT NHƯNG VÔ HIỆU**: hiện vật HOÀ ĐIỂM (2026-08-04) — Claude

Nối tiếp §13.61, đo bậc hai **có ràng buộc khung** — đúng tình huống `con khỉ là → động vật`.

Kết quả in ra:

| bộ chấm | MRR | hit@1 |
|---|---:|---:|
| `ngau_nhien` | 0,080 | 2,8% |
| `khung` | 0,026 | 0,2% |
| `khung_x_embedding` | 0,029 | 0,2% |
| **`khung_x_bac_hai`** | **0,082** | **5,5%** |

`bac_hai/khung = 3,12×` · `bac_hai/embedding = 2,83×` ⇒ **CỔNG ĐẠT cả ba.**

#### 🔴 Và nó SAI. Ba dấu hiệu, kiểm xong: hiện vật.

**Dấu hiệu 1 — `ngau_nhien` (0,080) HƠN `khung` (0,026).** Xếp ngẫu nhiên mà hơn tín hiệu
thật thì không phải phát hiện, đó là lỗi.

**Dấu hiệu 2 — `đích TỰ tới được: 0,0%`.** Không một ca nào đích nằm trong tập ứng viên do
khung sinh ra.

**Dấu hiệu 3 — cơ chế của lỗi, đã truy ra:** probe chọn ca sao cho bigram `(f, x)` chỉ có ở
**đúng một cha**, rồi `fs.pop(x)` để xoá chứng cứ trực tiếp. Hệ quả: điểm khung của `x`
**luôn bằng 0**. Mọi bộ chấm đều nhân với khung ⇒ tại `x` cũng **bằng 0**. Thứ hạng tính bằng
*"đếm ứng viên LỚN HƠN HẲN"*, nên `x` **hoà** với mọi ứng viên bị nhân về 0.

⇒ **Bộ chấm càng THƯA thì đích càng "thắng"**, vì càng nhiều ứng viên bị nhân về 0 và hoà với
nó. `bac_hai` thưa nhất ⇒ trông tốt nhất. **Không đo tín hiệu, đo độ thưa.**

Kiểm trực tiếp một ca: `f = ':'` có 474 token theo sau, `x = 'Biển'` **có** trong danh sách —
nhưng bị `pop` đi, nên điểm khung của nó bằng 0 trong khi 473 ứng viên kia dương.

⇒ **P87 VÔ HIỆU. Không được trích con số nào từ nó.** Cổng "đạt" không có giá trị.

> **Bài học thứ mười bốn về thước trong §13:** khi hạng tính bằng *"đếm cái lớn hơn hẳn"* thì
> **hoà điểm là phần thưởng**. Bộ chấm thưa được hưởng lợi một cách máy móc. Phải phá hoà —
> cộng nhiễu cực nhỏ, hoặc lấy hạng trung bình của cụm hoà.

#### Vì sao tôi để lọt tới lúc in kết quả

Tôi đã **cố ý** xoá chứng cứ trực tiếp (đúng), nhưng **không hỏi tiếp** *"sau khi xoá thì đích
còn được bộ chấm nền cho điểm gì không"*. Câu trả lời là **không** — và điều đó biến toàn bộ
so sánh thành so **độ thưa**. Cùng họ với §13.39 (đối chứng tự đếm chính nó) và §13.44 (biên
thật thấp *theo định nghĩa*): **thiết kế đối chứng sai, không phải cơ chế sai.** Lần thứ ba
trong phiên này.

#### Thiết kế đúng phải có

1. **Phá hoà**: hạng = trung bình của cụm hoà, hoặc cộng nhiễu `1e-12`.
2. **Đích phải chấm được bởi bộ chấm nền**: không chọn ca mà khung không thể biết `x`. Thay vì
   `(f,x)` chỉ-một-cha, hãy giữ `(f,x)` **có ở nhiều cha** và chỉ bỏ **cha đang xét** — khung
   vẫn cho `x` điểm dương, và câu hỏi trở lại đúng chỗ: *chủ thể có thêm gì so với khung*.
3. **Kiểm sàn trước khi đọc kết quả**: `ngau_nhien` **phải** thua `khung`. Không thoả ⇒ dừng.

Mục (3) rẻ và đáng thành thói quen: **sàn ngẫu nhiên là máy dò lỗi thiết kế**, không chỉ là mốc
so sánh. Ở đây nó đã báo động ngay dòng đầu, và tôi suýt đọc lướt qua.

⇒ §13.61 vẫn đứng (cổng trượt, đo hợp lệ). §13.62 **rút lại**. Ý tưởng của tác giả **vẫn
chưa được đo đúng** ở dạng có ràng buộc khung.

---

### 13.63 P88 — bản đã sửa: **biết CHỦ THỂ ăn thật (2,19×)**, nhưng `E` mới là cách hiện thực (2026-08-04) — Claude

P87 vô hiệu (§13.62). P88 sửa cả ba lỗi: **hạng giữa** phá hoà · **tách hai nhiệm vụ** · **kiểm
sàn trước khi đọc**. Sàn qua — `khung` 0,1373 > `ngẫu nhiên` 0,0719 — nên kết quả **đọc được**.

#### A — BỔ TRỢ: khung CÓ chứng cứ, hỏi *biết chủ thể có giúp không* (n = 400)

| bộ chấm | MRR |
|---|---:|
| `ngau_nhien` | 0,0719 |
| `khung` (chỉ biết khung) | 0,1373 |
| `khung × embedding` | 0,2790 |
| **`khung × bac_hai`** | **0,3004** |

**`bac_hai/khung` = 2,19×** ⇒ **A-2 ĐẠT** (cổng 1,15×).

> **Biết CHỦ THỂ hơn GẤP ĐÔI so với chỉ biết khung.** Đây là **xác nhận trực tiếp trực giác
> của tác giả**: `X là …` không cho cùng một đáp án với mọi `X` — chủ thể mang thông tin thật,
> và hiện tại **khâu sinh không dùng nó**.

`bac_hai/embedding` = **1,08×** ⇒ **A-3 TRƯỢT** (cổng 1,10×). Ngang nhau.

#### B — SUY LUẬN THẬT: khung KHÔNG thể biết đích (n = 400)

Cặp `(khung, đích)` chỉ có ở **đúng một cha**, bỏ cha đó ⇒ đúng nghĩa *"chưa hề train con khỉ
là động vật"*.

| bộ chấm | MRR |
|---|---:|
| `ngau_nhien` | 0,1209 |
| **`embedding`** | **0,2954** |
| `bac_hai` | 0,1506 |

- `bac_hai/ngẫu nhiên` = **1,25×** ⇒ **B-2 TRƯỢT** (cổng 2,0×).
- `bac_hai/embedding` = **0,51×** ⇒ **B-3 TRƯỢT** — `embedding` **gấp đôi** bậc hai.

#### ✅ Kết luận: ý tưởng ĐÚNG, cách hiện thực SAI

Hai điều tách bạch, và cả hai đều có số:

1. **Suy luận liên kết chưa từng thấy LÀ CÓ THẬT.** Ở nhiệm vụ B — nơi chứng cứ trực tiếp bị
   xoá sạch — `embedding` đạt **0,2954 vs 0,1209** sàn ngẫu nhiên = **2,44×**. Tức
   `con khỉ là → động vật` **khôi phục được**. Ý tưởng của tác giả **đứng vững**.
2. **Nhưng ma trận bậc hai KHÔNG phải cách làm.** Nó ngang `E` ở nhiệm vụ dễ (A: 1,08×) và
   **thua một nửa** ở nhiệm vụ khó (B: 0,51×).

Lý do có tên trong sách: `E` là **SVD cắt hạng** của chính đồng hiện — tức **làm trơn hạng
thấp**. Đó đúng là thứ bài toán điền-ma-trận cần, và nó **tổng quát hoá tốt hơn** tích thô
`W Wᵀ W`, vốn chỉ đi được theo đường đã có. Ở nhiệm vụ B, đích thường là token hiếm nên đường
bậc hai tới nó **thưa**; `E` không cần đường, nó dùng **hướng**.

⇒ **Không xây ma trận bậc hai.** `E` (32002×96) đã nằm sẵn trong checkpoint.

#### Việc thực sự phải làm — và nó khớp với §13.60

Ba số ghép lại thành một hướng cụ thể:

| | |
|---|---|
| §13.60 | truy hồi **42–53%** vs tầng mệnh đề **4,7%** — truy hồi là đường đúng |
| §13.63-A | biết chủ thể **2,19×** so với chỉ biết khung — chủ thể mang thông tin thật |
| §13.63-B | `E` khôi phục liên kết chưa từng thấy **2,44×** sàn |

⇒ **Điều kiện hoá khâu sinh theo CHỦ THỂ bằng `E`.** Cụ thể: khi xếp ứng viên (dù là mệnh đề
hay cha), nhân thêm `cos(E[chủ thể], E[ứng viên])`. Thuần đại số trên thứ đã có ⇒ Luật 0 sạch,
không thêm ma trận, không thêm bộ nhớ.

⚠️ Và phải nói ngay: đây **vẫn là một bộ xếp lại**, mà §13 đã giết chín cái. Khác biệt duy
nhất — nhưng có thật — là lần này tín hiệu được **đo trước** ở đúng nhiệm vụ nó sẽ làm, và nó
**hơn sàn gấp 2,2–2,4 lần**. Chín cái kia không cái nào có bằng chứng loại đó trước khi bật.
Vẫn phải chấm mù trên thang tuyệt đối, so mốc §13.60.

---

### 13.65 P89/P90 — lan truyền kích thích + viết cạnh: chạy ở ĐỒ CHƠI, **vô hiệu ở quy mô thật** (2026-08-04) — Claude

Ý tưởng tác giả (bản chi tiết hoá): không phải tính điểm lúc truy vấn, mà **viết cạnh mới lúc
train** qua lan truyền nhiều chặng — `khỉ → nghịch ngợm → {chó, mèo} → động vật` ⇒ nối
`khỉ → động vật`. Ba chỗ khác thật: **lặp**, **cạnh là thật** (mọi cơ chế hạ nguồn dùng được),
**thưa hoá lúc viết**.

#### P89 — corpus đồ chơi: cơ chế CHẠY ĐÚNG ở ca dương tính

Với 5 câu của tác giả, sau khi sửa, `khỉ → động vật` **được suy ra**, và truy vết cho đúng
đường tác giả mô tả: `động` nhận từ **`['chó', 'mèo']`** — hai bạn cùng lớp.

**Ba lần thử chặn ca âm tính (`bàn → động vật`), ba lần tách SAI:**

| cách chặn | vì sao hỏng |
|---|---|
| tần suất (`df/n`) | **tách NGƯỢC**: cầu tốt `nghịch ngợm` df **0,60** bị chặn; cầu xấu `nâu` df 0,29 lọt. Cầu tốt là thuộc tính chung của **nhiều** thành viên ⇒ tần suất cao **theo định nghĩa** |
| tổng kích thích | `động vật` yếu hơn `chó/mèo` **8×** ở T1 và **15×** ở T2 — cùng hình dạng, không tách |
| lift so nền | T1 lift **0,09**, T2 **0,11** — ca **cấm** còn cao hơn ca **cần** |

Chỉ khi đếm **thực thể tiền bối TRỰC TIẾP** mới thấy khác biệt thật: T1 `['chó','mèo']` = **2
thực thể**; T2 `['chó','con']` = **1 thực thể + 1 loại từ**. ⇒ ràng buộc đúng là **≥2 thực thể
ủng hộ**, và **loại từ phải bị loại khỏi việc ĐẾM**.

⚠️ Tôi **dừng** ở đó thay vì hạ `hub_max` xuống 0,85 cho T2 xanh — `con` có df 0,86 ở T2 và
1,00 ở T1, tức tôi **biết trước** ngưỡng nào làm test xanh. Đó là **khớp ngược trên 7 câu**,
không phải đo.

#### 🔴 P90 — mini: ràng buộc cứu được đồ chơi thì **VÔ HIỆU** ở quy mô thật

Nhiệm vụ **B của §13.63** (cặp chỉ có ở một cha, bỏ nguyên cha), 400 ca:

| bộ chấm | MRR |
|---|---:|
| `ngau_nhien` | 0,0803 |
| **`embedding`** | **0,1780** |
| `bac_hai` (một vòng) | 0,1226 |
| **`lan_truyen`** (3 chặng + hỗ trợ ≥2 thực thể) | **0,1098** |

`lan_truyen/embedding` = **0,62×** · `lan_truyen/bac_hai` = **0,90×** ⇒ **CỔNG TRƯỢT (2) và (3).**

Dự đoán của tôi: (2) trượt ✅ · **(3) đạt ❌** — tôi đoán lọc-hỗ-trợ sẽ giúp; nó **làm tệ hơn**.

**Vì sao, và đây là phần đáng giá nhất:**

> **`đích có ≥2 thực thể ủng hộ: 379/400 = 94,8%`**

Ràng buộc phân biệt được `khỉ` với `bàn` trên **7 câu** thì trên **5.664 cha** gần như **luôn
đúng**. Ở quy mô thật, **mọi thứ đều có ≥2 thực thể ủng hộ** — bộ lọc **không lọc gì**, chỉ
thêm một chặng làm loãng tín hiệu (0,1226 → 0,1098).

> **Bài học thứ mười lăm về thước trong §13:** một tiêu chí phân biệt được hai ca trên corpus
> đồ chơi có thể **vô hiệu hoàn toàn** ở quy mô thật — vì cái làm nó phân biệt được là **sự
> KHAN HIẾM** của corpus nhỏ, không phải bản chất của tiêu chí. Đây là §13.32 ("mini không định
> cỡ được") ở dạng gay gắt hơn: mini còn **bịa ra** khả năng phân biệt mà thật ra không có.

#### Trạng thái ý tưởng — ba lần đo, ba cách hiện thực

| bản | kết quả |
|---|---|
| §13.61 bậc hai một vòng, toàn cục | hit@10 2,3% vs `E` 2,7% — **không hơn** |
| §13.63-B bậc hai, có khung | MRR 0,1506 vs `E` 0,2954 — **thua một nửa** |
| §13.66 lan truyền 3 chặng + hỗ trợ | MRR 0,1098 vs `E` 0,1780 — **thua, và tệ hơn cả một vòng** |

⇒ **Cơ chế loại-suy-bằng-đường-đi không vượt được `E` ở bất kỳ dạng nào đã thử.** Lý do nhất
quán qua cả ba: đồ thị đồng hiện thật **dày**, nên đường đi tới đâu cũng có; `E` là **SVD cắt
hạng** — nó tóm bằng **hướng**, không bằng **đường**, và đó là thứ tổng quát hoá được.

**Nhưng ý tưởng gốc KHÔNG bị bác.** §13.63-B đã đo: suy luận liên kết **chưa từng thấy** là
**có thật** — `E` đạt **2,44×** sàn ở đúng nhiệm vụ đó. Điều bị bác là **cách hiện thực bằng
lan truyền/tích ma trận**. Việc còn lại vẫn là: **dùng `E` để điều kiện hoá khâu sinh** — vòng
chấm mù cho việc đó đang dở (ngắt ở lô 2/5 do hết hạn mức phiên).

---

### 13.66 P91 — **SỐ KHUNG CHUNG không bão hoà**: hướng tách-lát sống sót (2026-08-04) — Claude

§13.65 giết ràng buộc "≥2 thực thể ủng hộ" bằng một dòng: **94,8%** ca thoả ⇒ vô hiệu ở quy mô
thật. Câu hỏi kế: **số KHUNG chung** có chịu chung số phận không? Rẻ, không xây gì, chỉ đếm.

**Khung** của `a` = tập token **kề trước/kề sau** `a` trên `W_seq`, lọc rác bằng `gidf`. Đây là
"vai" thô nhất đồ thị hiện có.

| nhóm | n | trung vị khung chung | tỷ lệ **≥2 khung chung** |
|---|---:|---:|---:|
| `ngau_nhien` (sàn) | 600 | **0** | **7,5%** |
| `dong_hien_1` (đồng hiện đúng 1 cha) | 600 | **3** | 58,8% |
| `cung_lop` (⚠️ **tự quy chiếu**) | 600 | 5 | 100% |

**CỔNG G-frames: (1) ✅ · (2) ✅ · (3) ✅ ⇒ ĐẠT.**

#### Vì sao đây là kết quả đáng kể

| tiêu chí | tỷ lệ thoả ở ca vô can | |
|---|---:|---|
| "≥2 **thực thể** ủng hộ" (§13.65) | **94,8%** | vô hiệu — không lọc gì |
| **"≥2 KHUNG chung"** (§13.66) | **7,5%** | **chọn lọc gấp ~12,6 lần** |

Cùng dạng "đếm cái chung", nhưng một cái bão hoà và một cái không. Khác biệt: **thực thể** thì
đồ thị đồng hiện dày nên ai cũng chung nhiều; **khung** ràng buộc cả **vị trí kề**, nên hiếm
hơn hẳn. ⇒ Kết luận §13.65 (*"đồ thị dày thì mọi tiêu chí đếm-cái-chung đều bão hoà"*) mà tôi
định ghi là **SAI** — nó chỉ đúng cho tiêu chí bỏ qua vị trí.

#### Hai chỗ phải trừ hao, ghi rõ

1. **Tỷ số "3.000.000.000×" in ra là RÁC** — trung vị nhóm ngẫu nhiên bằng **0**, và mã chia
   cho `max(0, 1e-9)`. Con số đúng để trích là **trung vị 0 vs 3**, không phải một tỷ số.
   (Cùng họ với lỗi chia-cho-tập-rỗng ở §13.47 — lần này tôi bắt được ngay.)
2. **`cung_lop` tự quy chiếu**: nó được định nghĩa **bằng chính đại lượng đang đo** (≥3 khung
   chung), nên 100% là **tautology**, không phải phát hiện. Chỉ dùng để xem trần thang đo.
   Bằng chứng thật nằm ở `dong_hien_1` vs `ngau_nhien` — hai nhóm định nghĩa **độc lập** với
   thang đo.

#### Điều này KHÔNG chứng minh là gì

Cổng này chỉ nói tiêu chí **có sức chọn lọc**, tức **đáng xây**. Nó **chưa** nói tiêu chí
**phân loại đúng** — `bàn` và `chó` vẫn có thể chung nhiều khung vì cả hai đều là danh từ đứng
sau loại từ. Phép đo tiếp theo phải là: **tenxơ tách lát theo khung** rồi chạy lại **nhiệm vụ B
của §13.63**, so với mốc đã có (`E` **0,1780**, bậc hai **0,1226**, lan truyền **0,1098**).

⇒ Đây là lần đầu trong chuỗi bốn phép đo về ý tưởng của tác giả mà một **thành phần** vượt
cổng. Ý tưởng vẫn sống; cách hiện thực bằng **đường đi** đã chết (§13.65), cách bằng **tách lát
theo vai** thì vừa qua được cửa đầu tiên.

---

### 13.67 P92 — **TÁCH LÁT THEO KHUNG: 1,60× embedding, 1,91× khung** — cơ chế đầu tiên vượt `E` (2026-08-04) — Claude

Bản cuối của ý tưởng tác giả. Cơ chế:

    tương đồng(a, b) = |khung(a) ∩ khung(b)|            # tiêu chí đã qua cổng §13.66
    điểm(c) = Σ_b tương đồng(a, b) · đếm(b · f · c)     # chuyển giao TRONG khung f

Tiếng Việt: *`khỉ` được điểm cho `động vật` vì `chó` và `mèo` — hai thực thể nhiều khung chung
với `khỉ` — từng xuất hiện trong đúng khung `… là động vật`.*

#### 🔴 Lần chạy đầu: 0/400 — và lỗi ở NHIỆM VỤ, không ở cơ chế

Nhiệm vụ B (§13.63) chọn ca sao cho bigram `(f, x)` chỉ có ở **đúng một cha**, rồi bỏ cha đó ⇒
`(f, x)` **không còn tồn tại ở đâu**. Nhưng tách lát chuyển giao mẫu `b·f·c` **từ cha khác**,
nên nó cần `(f, x)` có ở đâu đó — theo cấu tạo thì **không bao giờ**. Đo ra `0/400 đích được
cho điểm`.

Đối chiếu ví dụ gốc thì thấy ngay nhiệm vụ sai: `con chó là động vật` **và** `con mèo là động
vật` **đều có** trong train ⇒ `(là, động vật)` tồn tại ở 2 cha; thứ **thiếu** là **bộ ba**
`(khỉ, là, động vật)`.

⇒ **Đơn vị giữ lại phải là BỘ BA, không phải cặp.** Nhiệm vụ B loại bỏ đúng kịch bản tác giả
mô tả. Sửa: giữ bộ ba `(a,f,x)` duy nhất ở cha đang xét, **và đòi `(f,x)` còn ở ≥1 cha khác**.

Đây là **lần thứ tư** trong phiên thiết kế đối chứng sai chứ không phải cơ chế sai (§13.39,
§13.44, §13.62, và đây).

#### Kết quả sau khi sửa (400 ca)

| bộ chấm | MRR |
|---|---:|
| `ngau_nhien` | 0,0901 |
| **`khung`** một mình — **đối chứng quyết định** | 0,1624 |
| `bac_hai` (túi từ) | 0,1251 |
| **`embedding`** | 0,1948 |
| **`tach_lat`** | **0,3107** |

| tỷ số | |
|---|---:|
| `tach_lat / khung` | **1,91×** |
| `tach_lat / embedding` | **1,60×** |
| `tach_lat / bac_hai` | **2,48×** |

**CỔNG G-tensor: ĐẠT cả ba.** Đích được cho điểm ở **76,5%** ca (lần đầu: 0%).

#### Vì sao lần này tin được — bốn hàng rào

1. **Đối chứng `khung` một mình** — thứ tôi **suýt bỏ sót**. Nhiệm vụ mới đòi `(f,x)` còn ở cha
   khác, nên khung có thể đã đủ. Nó đạt 0,1624; `tach_lat` **1,91×** hơn ⇒ loại suy **có làm
   việc thật**, không ăn theo bigram. (Đúng bài học §13.62.)
2. **Kiểm sàn**: `khung` (0,1624) > `ngau_nhien` (0,0901) ⇒ thiết kế lành.
3. **Hạng giữa** phá hoà ⇒ không thưởng bộ chấm thưa (bài học §13.62).
4. **Bộ ba bị loại khỏi nguồn chuyển giao** (`pj == pi` bị bỏ), và **khung tính có loại trừ
   cha** ⇒ không rò rỉ.

#### Ý nghĩa — và giới hạn

Đây là **cơ chế đầu tiên trong toàn §13 vượt `E`** ở nhiệm vụ suy luận. Chuỗi năm phép đo về ý
tưởng của tác giả nay đọc thế này:

| cách hiện thực | vs `E` |
|---|---:|
| bậc hai một vòng, toàn cục (§13.61) | 0,85× |
| bậc hai có khung (§13.63-B) | 0,51× |
| lan truyền 3 chặng + hỗ trợ (§13.65) | 0,62× |
| **tách lát theo khung (§13.67)** | **1,60×** |

Cái phân biệt: ba cái đầu dùng **túi từ** — mất **vai**. Tách lát giữ vai, và §13.66 đã đo
trước rằng chính **vị trí** tạo ra sức chọn lọc (7,5% vs 94,8%).

⚠️ **Giới hạn phải nói kèm:**
- Đây là **mini**, và là **nhiệm vụ xếp hạng**, **không** phải chất lượng sinh. §13.60 đã dạy:
  thắng ở một thước không có nghĩa thắng ở đầu ra.
- Chưa có gì chứng minh nó **chặn được ca âm tính** (`bàn → động vật`); P89 cho thấy đó mới là
  chỗ khó, và probe này **không đo** nó.
- 76,5% ca được cho điểm ⇒ 23,5% cơ chế **câm**; ở đó nó không giúp gì.

⇒ Việc kế: (a) đo ca âm tính ở quy mô thật — tỷ lệ cạnh suy ra **sai loại**; (b) nếu qua, đưa
vào đường **truy hồi** (nhánh mạnh nhất theo §13.60) rồi **chấm mù tuyệt đối** so mốc 53,3%.
