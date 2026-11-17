# 03 — ATTENTION Q/K/V CHO NN-AIVIE: CHIẾT TỪ ĐỒ THỊ, KHÔNG BACKPROP

> Nguồn: trao đổi tác giả ↔ Claude, 2026-07-02/03. Tác giả đặt bài: *"Q = E·Wq, K = E·Wk,
> V = E·Wv; Q·Kᵀ ra hệ số chú ý. E chính là mạng nơ-ron của chúng ta — vậy xây Wq/Wk/Wv
> thế nào nếu chỉ dùng nơ-ron thuần túy?"* — think outside the box.

## 1. Wq/Wk/Wv thật sự làm gì (nhìn lại từ gốc)

- **Wk**: chiếu token thành "tôi trình diện thế nào khi kẻ khác tìm tôi" (key).
- **Wq**: chiếu token thành "tôi đang tìm cái gì" (query) — BẤT ĐỐI XỨNG với Wk
  ("đọc" ở vai query tìm tân ngữ; ở vai key nó chào mình như một hành động).
- **Wv**: "tôi đóng góp gì khi được chú ý" — nội dung truyền đi, ≠ danh tính.

Điểm cốt tử: `Q·Kᵀ` tạo quan hệ **có hướng, phụ thuộc vai**. Bỏ W thì `E·Eᵀ` chỉ là
similarity đối xứng — chính là fuzzy cosine đang có. **Toàn bộ việc của Wq/Wk là phá vỡ
tính đối xứng.**

## 2. Insight trung tâm: W = một lớp synapse — và đồ thị CÓ HƯỚNG đã nhân tử hóa sẵn

Một phép chiếu tuyến tính W không là gì khác ngoài **đồ thị hai phía có trọng số giữa hai
quần thể nơ-ron**. Và mạng ta có sẵn thứ bất đối xứng: các lớp cạnh có hướng
(`W_seq[i→j] ≠ W_seq[j→i]`). Theo đúng toán Levy-Goldberg (nền của E hiện tại):

```
M_seq (PPMI có hướng) ≈ U · Σ · Vᵀ      (SVD, giữ d=96 trị lớn nhất)
Wq := U·√Σ        Wk := V·√Σ        ⇒   q_i · k_j ≈ PPMI(i→j)
```

**Giải nghĩa bằng nơ-ron** (đã trao đổi với tác giả): SVD "phát minh" d nơ-ron ẩn — mỗi
nơ-ron là một *khuôn chuyển tiếp* (vd khuôn "động-từ-ăn-uống → danh-từ-món" tự nổi).
- `U[i]` = token i tham gia khuôn ở vai **đứng trước** → vector query.
- `V[j]` = token j tham gia khuôn ở vai **đứng sau** → vector key.
- `Σ` = độ quan trọng từng khuôn; **chỉ giữ d khuôn lớn nhất chính là chỗ tạo khái quát
  hóa** ("nhắm" chưa từng đứng trước "nước" nhưng q·k vẫn dương nếu trúng khuôn).
- `√Σ` = chia đều độ lớn cho hai phía để `(U√Σ)(V√Σ)ᵀ = UΣVᵀ`.

E hiện tại cũng là PPMI+SVD nhưng trên ma trận ĐỐI XỨNG (co-activation) nên chỉ lấy một
phía U → chỉ biết "giống nhau", không biết "ai trước ai sau". **Lấy cả hai phía U và V
của ma trận có hướng — đó là toàn bộ bí mật của Wq/Wk.**

## 3. Multi-head MIỄN PHÍ: mỗi loại cạnh = một head

Transformer phải HỌC để head chuyên hóa; ta có chuyên hóa **theo cấu trúc**:

| Head | Lớp nhân tử hóa | Vai chú ý |
|---|---|---|
| Cú pháp | `W_seq` (bigram có hướng) | token nào theo sau token nào |
| Chủ đề | CO_ACTIVATION (cửa sổ) | cùng trường ngữ cảnh |
| Diễn ngôn | membership token↔episode t1 | cùng bài/đoạn |
| Tham chiếu | theme (thực thể→fact) | anaphora/coreference |
| **Fact-level** (TK_010 mở ra) | `fact_links` có thứ tự | fact nào theo sau fact nào |

## 4. Hai mảnh còn lại

- **Wv không cần tồn tại riêng**: "giá trị khi được chú ý" = **trường axon** của nơ-ron —
  hàng readout A[j,·]. `entity_attention_read` hiện tại đã đúng nguyên văn công thức này
  (docs vốn gọi là "attention không softmax").
- **Q phụ thuộc ngữ cảnh**: transformer có nhờ cả chuỗi nhân qua Wq; bản Hebbian = query
  là **chồng chập kích hoạt H** (spreading + decay λ theo k_max) — đã có trong code.
- **Tính dẻo**: transformer tinh chỉnh W bằng gradient; ta tinh chỉnh **cạnh chú ý bằng G3
  dopamine** (LTP khi token được chú ý góp vào đáp đúng) — khớp P3 SFT roadmap.

**Trung thực về giới hạn:** đây là attention *type-level* (mỗi token một q/k cố định);
tính ngữ-cảnh-hóa per-occurrence đến từ chồng chập H, KHÔNG phải contextual QKV per-token
như transformer thật. Đủ cho khái quát hóa chuyển tiếp; không thay thế được contextual
disambiguation sâu (đa nghĩa "trị/giá trị" vẫn cần đơn vị cụm).

## 5. Kho ma trận hiện tại & chi phí

Checkpoint chỉ persist: **A** (kề thưa nhiều lớp — nguồn sự thật duy nhất), **E** (đặc
duy nhất, view dẫn xuất), ParentStore (chuỗi + index suy ra). Wq/Wk mỗi head = ma trận
đặc N×96 **cùng loài với E** (view offline, build lại mỗi epoch): 4-5 head × 2 phía ≈
100-125MB f32 (~nửa nếu f16). Không thêm tham số backprop nào. Bất biến đẹp giữ nguyên:
**A là bộ nhớ, mọi thứ khác là góc nhìn chiết từ A.**

## 6. GIẢI PHÁP TRỌN VẸN (lộ trình đo được — TK_011 tương lai)

| Phase | Nội dung | Đo bằng | Chi phí |
|---|---|---|---|
| **A0 — thí nghiệm rẻ** | SVD PPMI-có-hướng của `W_seq` (d=96) trên big_a2; giữ 10% bigram held-out; so RANK successor thật: (a) bigram thô vs (b) `q·k` tái dựng | held-out bigram rank; kỳ vọng q·k dự đoán được cặp CHƯA đồng hiện | 1 buổi |
| **A1 — cắm kênh micro** | A3: `micro' = blend(W_seq thô, q·k)` (cờ `ar_qk_weight`, sổ nợ) — sửa đúng lỗ hổng đầu-segment ("Đáp:"→"Việc" không có cạnh thô nhưng q·k>0 nếu cùng khuôn mở câu) | essay vb_seg/cov/ord + guard; ca đầu-segment không cần rescue −inf nữa | vừa |
| **A2 — head diễn ngôn/fact** | SVD lớp fact_links (khi ≥ vài nghìn t1 sau scale) → q·k mức FACT cho bước nhảy fact→fact (gen 2 cấp TK_010) | fact-chain held-out (đoán fact kế tiếp) | vừa, sau scale |
| **A3 — G3 plasticity** | Lớp cạnh `A_attn[q_x,k_y]` cập nhật LTP/LTD theo eval đáp (P3 SFT) — phần "học tiếp" sau khi chiết | style/quality metric theo phiên | lớn, cuối |

**Acceptance A0 (điều kiện mở A1):** trên held-out, mean-rank của successor thật qua
`q·k` TỐT HƠN bigram-thô (thô cho rank ∞ với cặp chưa thấy); đọc tay 10 khuôn U/V lớn
nhất có nghĩa (khuôn mở câu, khuôn động-tân...).

**Rủi ro ghi trước:** (1) svds trên ma trận không đối xứng — dùng đúng u/vt hai phía,
không lấy nhầm một phía; (2) thang q·k khác thang W_seq thô → blend cần chuẩn hóa;
(3) type-level như đã nói ở §4 — không hứa contextual thật.

**Luật 0:** sạch tuyệt đối — mọi thứ chiết từ A đã học, không pretrain, không backprop,
không hardcode từ vựng.

### Phase stack (orthogonal residual — không gán nghề tầng)

Bổ sung 2026-07: thay vì multi-head theo **nhãn** loại cạnh, xây **4 tầng đồng cấu**
`Attn→Amp→top-n` với `Wa_ℓ`/`Wffn_ℓ` từ **residual SVD** trên một nguồn PPMI(`W_seq`).
Tầng sau sống trên phổ còn lại → trực giao gần đúng theo xây dựng. Vai trò (cú pháp /
hub / …) chỉ kết luận sau `scripts/probe_spectral_stack.py` — không hardcode vào
`spectral_stack.py`. Cờ runtime: `ar_spectral_stack` (blend AR; tắt copy khi bật).
