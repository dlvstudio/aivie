# 06 — HỌC MỒI: GIEO NƠ-RON KHÁI NIỆM + TỰ ĐẤU DÂY BẰNG RESONANCE + G3

> Nguồn: trao đổi tác giả ↔ Claude, 2026-07-03. Tác giả đặt bài: LLM có các tầng attention
> khác nhau nhờ hàng tỷ bước train + thuật toán ép các ma trận chú ý khác nhau. Ta không có
> gradient tỷ-bước, nhưng có 3 đòn bẩy học:
> - **Học giám sát (SFT)**: đã có — trọng số gia giảm theo ưu tiên.
> - **Học mồi / học trọng điểm**: người dạy CHỦ ĐỘNG tiêm khái niệm/ý niệm/trừu tượng
>   (curriculum lớp 3→6→9→12→ĐH→TS).
> - *Vấn đề cốt tử tác giả nêu*: mồi chỉ TẠO điểm neo; làm sao **các nơ-ron khác TỰ nối tới
>   nơ-ron trừu tượng mồi với TRỌNG SỐ THÍCH HỢP**, kết hợp với các quá trình train khác? Khi
>   có neo mồi, lúc sinh nó nổi lên làm điểm neo → sinh từ chính xác hơn.

## 1. Vấn đề thật: auto-wiring, không phải seeding

Gieo một nơ-ron mà không gì nối tới → nơ-ron CHẾT. LLM đấu dây bằng gradient (mọi trọng số
tự dịch về khái niệm hữu ích). Không có gradient thì câu hỏi là: **cơ chế Hebbian nào khiến
nơ-ron khác tự nối vào neo mồi, và cái gì cho trọng số ĐÚNG?**

## 2. Insight: máy auto-wiring ĐÃ CÓ = resonance / bind (TK_010)

Nơ-ron mồi = một **t1 đặc biệt** mà "episode" của nó là ĐỊNH NGHĨA/VÍ DỤ do thầy đưa (thay
vì rút từ corpus). Cơ chế "các nơ-ron khác tự nối vào" = `bind_fact_links` / resonance
sweep: kích hoạt token định nghĩa → quét mọi cha cộng hưởng → nối, trọng số khởi tạo = độ
khớp (PMI/IDF-coverage). Động cơ tự-đấu-dây đã xây rồi; mồi chỉ là loại t1 do người dạy tạo.

## 3. VÒNG HỌC 4 BƯỚC (hợp nhất mọi cơ chế đang có)

```
① MỒI   thầy tiêm nơ-ron khái niệm + định nghĩa/ví dụ  → NEO      (lesson/imprint — Mục 0)
② ĐẤU   resonance sweep: khái niệm ↔ nội dung khớp      → trọng số KHỞI TẠO  (bind_fact_links)
③ TINH  G3 khi sinh: neo active → đáp đúng? LTP / lạc? LTD → trọng số THÍCH HỢP  (SFT/P3)
④ NEO   khái niệm ở H_slow lúc sinh                     → sinh chính xác hơn  (doc 05)
```

**Bước ③ = câu trả lời cho "trọng số thích hợp":** resonance cho trọng số *khởi tạo* (khớp
bề mặt), G3 cho trọng số *đúng* — cạnh giúp sinh đáp tốt → mạnh; cạnh kéo lạc → yếu.
"Gradient của người nghèo": KHÔNG lan ngược toàn mạng, chỉ củng cố CỤC BỘ cạnh-của-neo theo
phần thưởng eval. Đây là chỗ SFT (đòn bẩy 1) và học mồi (đòn bẩy 2) GẶP NHAU.

## 4. Curriculum = THANG TRỪU TƯỢNG (concept→concept), và vì sao vượt LLM về hiệu suất mẫu

Lớp 3→6→9→12→ĐH→TS không chỉ "thêm data" mà **gieo khái niệm bậc cao NỐI VÀO khái niệm bậc
thấp đã gieo**: "quang hợp" (L6) → "chu trình carbon" (L9) → "cân bằng sinh thái" (ĐH).
Nơ-ron mồi nối tới nơ-ron mồi khác, không chỉ tới fact → tầng trừu tượng nhiều bậc.

**Đảo ngược LLM (out of the box):**
> LLM: học biểu diễn từ tỷ mẫu RỒI dò khái niệm trong đó (data-first).
> nn-aivie: thầy TIÊM khái niệm RỒI mạng mọc biểu diễn quanh nó (concept-first).

Cái đắt nhất của trí tuệ = khám phá "thang trừu tượng nào đáng có"; LLM tốn tỷ bước, ở đây
**thầy tặng cái thang miễn phí**, mạng chỉ còn học "nội dung nào lấp bậc nào" (few-shot,
dễ hơn nhiều) → khả thi với data nhỏ.

## 5. Luật 0 — vị trí sạch

Mồi khái niệm KHÔNG phải hardcode: Mục 0 cho phép tường minh "teach, curriculum, imprint,
lesson". Ranh giới: thầy ĐẶT TÊN trừu tượng + cho ví dụ, nhưng KHÔNG hardcode từ nào kích
hoạt nó — "từ/câu nào thuộc khái niệm" do resonance + G3 học từ đồ thị. Nơ-ron khái niệm =
node CẤU TRÚC; NGHĨA đến từ cạnh học được. Đúng phía được phép.

## 6. GIẢI PHÁP TRỌN VẸN (TK_014 tương lai)

Định dạng lesson mồi (mở rộng lesson.yaml đã có):
```yaml
concept: "quang hợp"
parents: ["chu trình carbon"]      # nối lên khái niệm bậc cao đã gieo (thang)
define: "Quang hợp là quá trình cây xanh dùng ánh sáng tạo chất hữu cơ ."
examples: ["Cây xanh hấp thụ khí CO2 .", "Lá cây chứa diệp lục ."]
```

| Phase | Nội dung | Đo bằng | Chi phí |
|---|---|---|---|
| **C0 — thí nghiệm rẻ** | Gieo 5-10 concept thủ công (định nghĩa+ví dụ) vào `_t3`; chạy resonance sweep; ĐẾM mỗi concept hút được bao nhiêu cha đúng chủ đề | ≥N cha/concept, đọc tay đúng miền; concept "chết" (0 attach) = data thiếu | 1 buổi |
| **C1 — ConceptNeuron + lesson** | Dataclass concept (kind="primed"; parents cho thang); parser lesson mồi; ② đấu = bind_fact_links tái dụng; serialize | roundtrip; thang concept→concept nối đúng | vừa |
| **C2 — G3 tinh chỉnh** (③) | Sinh trên gold có concept liên quan → concept active làm neo → eval đúng thì LTP cạnh concept↔fact, lạc thì LTD. Tái mục đích `g3_dopamine` (đang tắt) | trọng số conc" hội tụ; đáp bám concept đúng hơn baseline | lớn |
| **C3 — neo lúc sinh** (④) | Concept vào H_slow (doc 05); query chạm concept → nổi lên thiên vị readout | sinh-từ chính xác hơn trên câu thuộc miền concept; guard | vừa |

**Acceptance C0 (điều kiện mở C1):** ≥70% concept gieo hút được ≥3 cha đúng miền khi đọc
tay; concept "chết" ghi rõ (chỉ ra data cần bổ sung — không tự lừa bằng attach mờ nhạt).

**Rủi ro ghi trước:**
1. **Concept chết** — gieo mà corpus không có nội dung → dangling anchor. Metric coverage
   concept bắt buộc; concept chết = tín hiệu cần data, KHÔNG phải giữ cho đẹp.
2. **G3 mất ổn định** — LTP/LTD cục bộ có thể trôi; đóng băng đường crisp-QA (như P3 chốt
   an toàn README 0.6 Trụ 6 ý 2) + học suất nhỏ + đo mỗi vòng.
3. **Không phải trí tuệ LLM** — concept-first cho hiệu suất mẫu + diễn giải được, KHÔNG
   thay được năng lực suy luận nổi lên của LLM tỷ-tham-số. Định vị đúng: dạy có cấu trúc,
   không phải phát minh khái niệm mới.
4. **Đặt tên ≠ hiểu** — neo mồi chỉ hữu ích nếu ② đấu + ③ tinh chạy; gieo suông = nhãn
   trang trí. Metric phải chứng minh neo THAY ĐỔI được đầu ra (ablate concept → sinh tệ đi).

## 7. Liên kết chéo

- doc 05 (tầng neo): concept mồi = một `AnchorNeuron` kind="primed"; H_slow là chỗ nó neo.
- doc 03 (Q/K heads): concept có thể thành một head khi đủ nhiều (SVD lớp concept↔fact).
- TK_010 (`bind_fact_links`/t1) = máy ② đấu dây; P3 SFT/G3 = máy ③ tinh chỉnh.
- Mục 0 (teach/curriculum/imprint) + lesson.yaml = hạ tầng ① mồi đã có một phần.
