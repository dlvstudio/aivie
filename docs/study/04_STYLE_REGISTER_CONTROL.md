# 04 — VĂN PHONG TÙY BIẾN: NƠ-RON PHONG CÁCH TỪ HUB-ĐẢO + ĐỒNG-EPISODE

> Nguồn: trao đổi tác giả ↔ Claude, 2026-07-03. Tác giả đặt bài: cùng một tri thức,
> mạng chọn được văn phong theo tính cách agent / người dùng / ngữ cảnh:
>
> ```
> X là thứ duy nhất mà Y      (quả quyết — độc tôn)
> X là cần thiết để Y         (khẳng định — trung tính)
> X có thể cần để Y           (dè dặt — hedge)
> X là tuyệt đối để Y         (nhấn mạnh — cực đoan)
> ```
> Nghĩa (X→Y) giữ nguyên; chỉ SẮC THÁI đổi.

## 1. Insight 1 — Văn phong = KHUNG = slot hub ĐẢO NGƯỢC

Slot hub hiện tại học chiều xuôi: *khung cố định, filler thay đổi* ("Thủ đô của ___ là
gì" → {Mỹ, Pháp, Anh}). Văn phong là hiện tượng **đối ngẫu**: *nội dung (X,Y) cố định,
KHUNG thay đổi* — họ các khung cùng nối một cặp nội dung = **paradigm của văn phong**.

→ Cùng thuật toán `induce_slot_hubs`, chỉ **đảo vai trò khung/filler** khi quét: gom các
cha có cùng cặp filler nội-dung nhưng khác phần giữa. Khung được MINE từ data (Re-Pair +
hub-đảo), không một danh sách từ nào hardcode — Luật 0 sạch.

## 2. Insight 2 — Episode là NHÃN VĂN PHONG miễn phí

"duy nhất/tuyệt đối" vs "có thể cần" phân biệt bằng gì nếu không gán tay? Bằng **phân bố
nguồn bài**: bài quảng cáo/blog nhiệt huyết dùng khung cực đoan; hướng dẫn trung tính
dùng "cần/nên"; văn khoa học dùng "có thể/thường". Corpus vi-alpaca vốn trộn đủ ba loại
— **mỗi episode t1 (TK_010: bài = MỘT nơ-ron nguyên vẹn) là một mẫu văn phong nhất quán
tự nhiên.**

→ Chạy `phrase_assoc` (đồng-episode — máy TK_008 có sẵn) trên riêng các **cụm-khung**:
khung hay đứng cùng bài tự co cụm thành **style cluster** (cực quả-quyết / dè-dặt /
trang-trọng...). Mỗi cluster vật chất hóa thành **NƠ-RON VĂN PHONG** — cùng loài với
theme/hub/t1 (giữ tính phân dạng của mạng).

## 3. Insight 3 — Chọn văn phong lúc sinh: kênh thứ 5 của Hadamard + PRIMING

A3 sinh bằng `a_micro ⊙ a_macro ⊙ a_entity ⊙ a_discourse`. Thêm **`a_style`** = readout
từ nơ-ron văn phong đang được **mồi kích hoạt**: token thuộc khung cùng cực được nâng,
khung cực đối bị nén. **Nghĩa không đổi** vì agenda (X, Y, fact) vẫn do t1/fact_links
quyết — chỉ con đường NỐI giữa chúng đổi màu.

Persona — 3 nguồn kích hoạt, đều thuần đồ thị:
1. **Tường minh**: frame token `Persona:` trỏ vào style neuron (frame protocol được phép
   như `Hỏi:`/`Đáp:`).
2. **Mirroring người dùng**: user hay dùng khung nào → G3 củng cố style neuron đó trong
   phiên — persona nổi lên từ hội thoại, không cấu hình.
3. **Tính cách agent cố định**: vector mồi lưu sẵn (agent "nhà khoa học" → cực hedge;
   "sales" → cực tuyệt đối).

## 4. Vì sao ĐẾN GIỜ mới làm được — hai điều kiện vừa xong

- **TK_009 (essay sinh token-level là default)**: verbatim thì copy cả khung — không bao
  giờ đổi được văn phong. Chỉ khi từng token đi qua các kênh thì `a_style` mới có chỗ
  can thiệp. Đây là giá trị định tính đầu tiên mà "máy sinh" mở ra ngoài điểm số.
- **TK_010 (t1 = cha thật, bài nguyên vẹn)**: văn phong của bài nhất quán trong MỘT
  nơ-ron — nhãn sạch cho style cluster (bài băm câu thì nhãn vỡ vụn).

## 5. GIẢI PHÁP TRỌN VẸN (lộ trình đo được — TK_012 tương lai)

| Phase | Nội dung | Đo bằng | Chi phí |
|---|---|---|---|
| **S0 — thí nghiệm rẻ** | Trên 500 essay + wiki `_t3`: (a) mine khung bằng hub-đảo trên cặp nội dung lặp; (b) cluster khung theo đồng-episode (`phrase_assoc` giới hạn vào cụm-khung); (c) ĐỌC TAY: cực quả-quyết vs dè-dặt có TỰ TÁCH không | đọc tay + PMI nội-cluster vs liên-cluster; ghi ví dụ cluster vào ticket | 1 buổi |
| **S1 — nơ-ron văn phong** | `StyleNeuron{key, frames[], polarity_examples}` vật chất hóa cluster; serialize; API `style_of_frame(cụm)` | roundtrip + đọc tay | nhỏ |
| **S2 — kênh a_style** | Term Hadamard thứ 5 trong A3 (`ar_style_weight` — sổ nợ nếu yếu); mồi qua tham số `style=` của generate_essay | metric mới **style_consistency** (% khung trong bài sinh thuộc đúng cực persona) + meaning coverage KHÔNG đổi + guard | vừa |
| **S3 — persona** | Frame `Persona:` + mirroring G3 trong phiên | style_consistency theo persona đặt/học | vừa, sau S2 |

**Acceptance S0 (điều kiện mở S1):** ≥2 cluster khung tách rõ khi đọc tay, mỗi cluster
≥5 khung, ví dụ đối cực đúng trực giác ("duy nhất/tuyệt đối" ≠ "có thể/thường").

**Rủi ro ghi trước:** (1) 500 bài có thể chưa đủ phân cực → S0 âm tính thì ghi sổ chờ
scale essay (mọi đường hiện đều chờ cùng một thứ: thêm bài); (2) khung lẫn nội dung khi
mine (hub-đảo cần gate IDF: khung = phần IDF THẤP giữa hai filler IDF cao — đảo đúng
gate hiện có của induce_slot_hubs); (3) style ⊗ nghĩa không tách tuyệt đối ("có thể"
đổi cả truth-value — hedge là thay đổi cam kết nhận thức, không chỉ trang trí; metric
meaning-giữ phải soi kỹ chỗ này, đừng tự lừa).

**Luật 0:** khung mine từ data; cluster tự nổi từ đồng-episode; persona chỉ TRỎ vào
cluster đã nổi; frame `Persona:` thuộc frame protocol được phép. Không hardcode danh
sách từ văn phong.

## 6. Liên kết chéo

- `03_ATTENTION_QKV_FROM_GRAPH.md`: a_style về sau có thể là một HEAD (SVD lớp
  khung↔episode) trong cùng khung nhân tử hóa.
- TK_008 (`phrase_assoc`), TK_009 (ar-mode), TK_010 (t1 nguyên vẹn) — ba điều kiện nền.
- P3 SFT/G3: mirroring persona dùng chung máy củng cố.
