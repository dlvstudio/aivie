# TK_061 — SINH LAI: W_skip (grounding) × causal-LM (trôi chảy), cổng theo độ tin

## Bằng chứng thúc đẩy (thăm dò TK_056-v2)

`scripts/tham_do_causal_lm.py` / `..._lm2.py` — sinh THẲNG từ logits causal-LM
(weight-tied, TK_059), không W_skip:

- Greedy TRẦN → lặp vòng ("cây cây, cây…", "loài loài…") — bệnh LM yếu kinh điển.
- **+ repetition penalty + no-repeat-bigram → MẠCH LẠC, đúng ngữ pháp:**
  - «nước là chất» → "nước là CHẤT LỎNG, có thể được dùng để tạo ra…" (mở đúng)
  - «sắt là» → "sắt là một loại CÂY TRỒNG…" (ngữ pháp mượt, SAI sự thật)
  - «chó là» → "chó là loài chim mọc CÓ VÚ…" (đúng lẫn sai)

## Chẩn đoán — hai nửa bù nhau

| | Sự thật | Trôi chảy/khái quát |
|---|---|---|
| W_skip (đếm) | ✓ grounded verbatim | ✗ cứng; hết chuỗi → cụt/rác (Rocky/Foucault) |
| causal-LM | ✗ ảo giác (sắt→cây) | ✓ mượt, ngữ pháp, khái quát |

Lặp vòng chỉ là DECODING (sửa được). LM nền CÓ năng lực trôi chảy mà W_skip thiếu.
Nới (TK_056) thất bại vì blend lệ thuộc tập W_skip; kNN reorder không lật được vote.

## Thiết kế đề xuất — cổng theo ĐỘ TIN của W_skip

KHÔNG dùng `λ·P_kNN+(1−λ)·P_causalLM` trần (thừa hưởng ảo giác). Thay:

- **W_skip TỰ TIN** (vote đỉnh cao, có cha verbatim) → theo W_skip (giữ SỰ THẬT).
- **W_skip PHÂN VÂN / CẠN** (vote phẳng, hết chuỗi) → để causal-LM (+ kNN) DẪN
  tiếp cho MƯỢT, thay vì cụt/rác. Áp repetition-penalty của đường sinh hiện có.
- Đo: độ tin = biên vote đỉnh / nền (đã có `rv`, median trong generate_hebbian).

Kỳ vọng: „Rocky"/„Foucault"/cụt → câu mượt; «hổ là loài thú…» (W_skip mạnh) GIỮ
nguyên. Rủi ro: causal-LM kéo lệch sự thật ở vùng phân vân — cần ngưỡng cổng chặt
+ chỉ cho causal-LM điền phần „nói thêm", không đè mệnh đề factual đầu.

## Trạng thái
- [x] Thăm dò: causal-LM sinh mượt (repetition-penalty) — ĐÁNG dựng.
- [ ] Dựng đường blend cổng-độ-tin trong generate_hebbian (hoặc path song song).
- [ ] Đo TRUNG THÀNH (sự thật) vs TRÔI CHẢY trên bộ prompt, so W_skip thuần.

## Kết quả v1 (đã dựng) — an toàn TUYỆT ĐỐI, lợi trôi-chảy CHƯA tới

`hybrid_blend` (cổng khe-hở-đỉnh) + chống-cụt (dìm EOS sớm khi output ngắn để
causal-LM nối). Test 5 prompt MẠNH + 5 YẾU (`scripts/thu_hybrid.py`):

**MẠNH — sự thật giữ 5/5 (y hệt off):** «chó là», «nước là chất», «vàng là kim
loại», «hổ là loài thú», «người nông dân». Cổng độ-tin hoạt động HOÀN HẢO —
W_skip tự tin (verbatim) → w_lm=0 → không đụng. **Facts an toàn tuyệt đối.**

**YẾU — lẫn lộn, không đạt trôi chảy:**
- «dòng sông chảy» → chống-cụt KÍCH: "dòng sông chảy **xiết** Foucault là" — "chảy
  xiết" MƯỢT+ĐÚNG, nhưng nối tiếp trôi vào RÁC ("Foucault").
- «sắt và»→thiếc (không đổi), «cây cối cần ánh»→"triển", «trong rừng»→"trọt" — lẫn.

**Chẩn đoán:** kiến trúc cổng-độ-tin ĐÚNG + AN TOÀN, nhưng **causal-LM d=128
(perplexity ~110) quá YẾU**: thi thoảng ra token hay ("xiết") rồi ẢO GIÁC ngay
("Foucault"). Nút thắt giờ là **DUNG LƯỢNG LM**, không phải tích hợp.

**Trạng thái:** hybrid + chống-cụt để **opt-in** (`/api/gen?hybrid=1`), CHƯA default
production (vẫn kNN reorder causal — phân bố 2× tốt đã chứng minh). Cổng-độ-tin là
nền đúng; mở khoá thật khi có LM mạnh hơn (d/tầng lớn + train dài — "train LM nhỏ
thật sự").

- [x] Dựng cổng-độ-tin + chống-cụt; chứng minh facts an toàn 5/5.
- [x] Đo: causal-LM d=128 quá yếu để nối mượt tin cậy → nút thắt là dung lượng LM.
- [ ] (mở) train causal-LM lớn hơn / dài hơn rồi bật hybrid default.

## v2 — causal-LM LỚN (d=256) + BÀN GIAO SPAN (đúng kiến trúc)

**Train lớn (TK_059 scale-up):** d=256/6 tầng/8 head, 12.9M tham số, 25k bước.
Held-out LM bits **40→5.68** (d=128 là 6.78; perplexity ~110→51). kNN-LM bits
λ=0.5: MASKED 8.44 · d128 7.39 · **BIG 7.02**.

**Sinh trực tiếp big (repetition-penalty): MƯỢT hơn HẲN d128** — "nước là chất
lỏng… dùng để đun sôi, nung nóng"; "dòng sông chảy nước chảy xiết"; học cả văn
phong từ điển ("(nói khái quát)"). NHƯNG vẫn ẢO GIÁC sự thật (sắt→chất lỏng, hổ
là loài→"cây họ vịt"). ⇒ scale sửa TRÔI CHẢY, KHÔNG sửa SỰ THẬT.

**Phát hiện kiến trúc:** hybrid per-step (nhấp 1 bước trong vòng W_skip) KHÔNG khai
thác được — prompt yếu dừng sau 1 token, big-LM chưa kịp phát huy. Fluency cần
KIỂM SOÁT LIÊN TỤC. ⇒ đổi sang **BÀN GIAO SPAN** (`KnnLM.lm_extend`):

- W_skip sinh phần đầu (GROUNDED). Nếu DÀI (định nghĩa) → **giữ nguyên, không nối**.
- Nếu CỤT/NGẮN (< extend_min token) → causal-LM BIG **nối tiếp liên tục** (greedy +
  repetition-penalty + no-repeat-bigram).

**Kết quả (span handoff):**
- «chó là» (15 tok), «nước là chất» (22 tok) → KHÔNG nối, facts nguyên vẹn ✓
- «sắt và» → "sắt và thiếc **là chất độc, có thể được sử dụng để làm nguyên liệu
  hoặc chất lỏng, thường dùng…**" (mượt trọn câu)
- «dòng sông chảy» → "…**dòng suối trôi qua bờ biển**" (mượt+đúng)
- «trong rừng có nhiều» → "…**núi lửa và thung lũng sông… khu rừng nhiệt đới**"

Đuôi mượt+đọc-được (nửa đúng chủ đề, nửa trôi nghĩa) thay cho cụt/rác. Giới hạn
còn lại: LM nhỏ ⇒ đuôi có thể lệch sự thật; nhưng đây là bước NHẢY về trôi chảy.

**Wire:** `get_knn` ưu tiên BIG d256; `/api/gen?extend=1` bật bàn giao span (mặc
định TẮT — đuôi có drift, để opt-in + đánh giá thêm). Facts an toàn vì chỉ nối khi
W_skip cụt.

- [x] Scale causal-LM d=256 (bits 5.68); [x] bàn giao span lm_extend; [x] facts an toàn.
- [ ] (mở) chặn drift đuôi: neo chủ đề/thực thể prompt vào causal-LM khi nối.
