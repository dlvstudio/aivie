# Báo cáo kết thúc P3 (P3.3' vòng thầy chung chữ + ảnh)

Thời điểm đọc: 15/10/2026 09:32 giờ VN (02:32 UTC) · box Vast 1×H100 `<IP>:43856`

## Tóm tắt

- **P3.3' XONG lúc 09:10 giờ VN 15/10** (02:10 UTC). Checkpoint cuối: `ckpt/aivie_p3_r1_bf16.pt` (7,2 GB), đã đẩy lên HF `nn-aivie-2p6b-p3-snap/aivie_2p6b_p3_r1_bf16.pt`.
- Vòng thầy mất khoảng 18 giờ 20 phút (07:47 UTC 14/10 → 02:10 UTC 15/10): 1500 hội thoại, 6336 lượt; sửa 5036 lượt chữ và 1293 lượt có ảnh; DPO 4198 cặp chữ và 1216 cặp ảnh.
- **Kết quả thật: lên xuống lẫn lộn, nhìn chung đứng yên.** Dịch, kiến thức và đời sống tăng rõ. Toán tăng nhẹ. **Excel giảm mạnh (3,44 → 2,31)**. Đọc ảnh gần như không đổi: các thay đổi chỉ ±1,5 điểm, nằm trong mức nhiễu của 200 câu.

## Chữ — trước (P3/P3.5) và sau (P3r1)

| Chỉ số | Trước | Sau | Δ |
|---|---|---|---|
| Lập trình (eval_v1, đúng /5) | 1,0 | 1,1 | ≈ |
| Văn phòng (eval_v1) | 3,7 | 3,7 | = |
| Tóm tắt | 2,8 | 2,5 | −0,3 |
| Dịch | 2,0 | **3,2** | **+1,2** |
| Kiến thức | 3,2 | 3,6 | +0,4 |
| Đời sống | 2,5 | 3,0 | +0,5 |
| Pháp luật | 2,0 | 1,6 | −0,4 |
| Suy luận | 1,8 | 1,8 | = |
| Mở: tự nhiên / đúng (gemma4-26b chấm) | 4,53 / 2,38 | 4,58 / 2,56 | +0,18 đúng |
| Sự kiện: đúng / không biết / sai | 5% / 30% / 65% | 2% / 32% / 65% | ≈ |
| Không có thật: nhận / bịa | 45% / 55% | 45% / 55% | = |
| Toán eval_v1 | 11% | 17% (3/18) | +1 câu |
| easy20 | 17/20 | 16/20 | −1 |
| **Excel (giám khảo văn phòng 150 câu)** | **3,44** | **2,31** | **−1,13** |
| Văn phòng 150 câu, trung bình | 3,29 | 3,00 | −0,29 |
| · thao tác / email / văn bản | 3,54 / 4,29 / 3,34 | 3,32 / 3,79 / 3,41 | email −0,5 |
| · tự động hoá / biên tập | 2,00 / 2,80 | 1,91 / 3,12 | biên tập +0,3 |
| Excel đúng tuyệt đối | — | 0/27 | |

Chưa có mốc trước để so (chỉ có số sau): SGK 400 câu: đúng 52,8% · một phần 14,8% · sai 32,5%. Toán sinh: 75/200 (78 khi dùng máy tính). Vi-GSM8K: 26/300 (29 khi dùng máy tính). TK_105 ghi 1/100 cho model 1.3B, nhưng đó là thân khác nên không so trực tiếp được.

Bảng chấm thứ hai bằng SeaLLM-7B trong `p3loop_report.txt` **trống**: giám khảo này không ra số. Mọi số ở trên đều do gemma4:26b chấm.

## Đọc ảnh — trước (P3.4) và sau (P3r1), % đúng, mỗi bộ 200 câu

| Bộ | Trước | Sau | Δ | khớp giá trị % (trước → sau) |
|---|---|---|---|---|
| Màn hình | 59,0 | 60,5 | +1,5 | → 73,4 |
| Code | 49,5 | 48,0 | −1,5 | 49,6 → 50,4 |
| Tin tức | 27,0 | 26,0 | −1,0 | 57,0 → 55,7 |
| Hỏi đáp mở | 37,5 | 35,5 | −2,0 | 25,4 → 25,4 |
| Biểu đồ | 35,5 | 35,5 | 0 | 17,4 → 15,2 |
| Ảnh tự nhiên | 1,5 | 2,5 | +1,0 | 21,5 → 17,7 |

Ở mức n=200, chênh ±2 điểm chỉ là nhiễu: **vòng thầy không làm khả năng đọc ảnh tốt lên**. Điểm thầy chấm trong vòng cũng thấp nhất ở tin tức (0,97), màn hình (1,27) và code (1,51). Phần an toàn chỉ được 0,52, nghĩa là dữ liệu sửa ở các phần này nhiều nhưng chưa chuyển thành điểm đo.

## Ví dụ thật (eval_v1, cùng câu hỏi, gemma chấm trước → sau)

1. **Dịch, 1 → 4.** Câu hỏi: dịch sang tiếng Việt câu *"The weather is getting colder, so you should wear more warm clothes."*
   - Trước: chép lại nguyên câu tiếng Anh rồi giảng từ vựng.
   - Sau: *"Thời tiết đang trở nên lạnh hơn, vì vậy bạn nên mặc thêm quần áo ấm."* → dịch đúng.
2. **Tóm tắt, 2 → 5** (đoạn về lừa đảo qua mạng).
   - Trước: lạc đề thành "khái niệm lừa đảo".
   - Sau: tóm đúng ý (giả danh công an/ngân hàng, đòi OTP).
3. **Tóm tắt, 5 → 1** (quy định phân loại rác).
   - Sau: chép gần như nguyên văn đoạn gốc thay vì tóm tắt. Lỗi chép lại vẫn còn.
4. **Kiến thức, 5 → 1** (Thế chiến thứ nhất).
   - Sau: bịa ra rằng chiến tranh "bắt đầu bằng việc Đức ký Hiệp ước Versailles" và "Nga tấn công Ba Lan". Model vẫn bịa sự kiện.
5. **Excel, 5 → 2** (chuẩn hoá tên chủ hộ ở cột B).
   - Sau: `=PROPER(TRIM(B2)) & " " & TRIM(A2) & ...`, tức là gộp nhầm các cột khác vào tên. Dạng lỗi này giải thích vì sao Excel tụt điểm.
6. **Đọc ảnh màn hình** (log sinh): *"54 + 950 + 70 + 220 = 1386. Tổng các giá trị là 1386 triệu đồng."* Model có đọc số và tự cộng.

## GPU / ổ đĩa

- H100: **42 °C**, tải 100%, công suất 226–238 W trên giới hạn 700 W. Đang chạy C1, nhiệt độ mát.
- Ổ `/`: 215G / 250G đã dùng (86%), **còn 36G**.

## Trạng thái C1 (TK_108)

- **Đã bắt đầu** lúc 02:10 UTC trên `ckpt/aivie_p3_r1_bf16.pt`, với 3 UI chạy song song (cổng 8811/8812/8813). Trong log, tag ghi là `p3`, nhưng checkpoint được đo là r1.
- 02:11 UTC: **HumanEval và MBPP LỖI** do thư viện `datasets`. Script đã tự sửa và **chạy lại lúc 02:29 UTC**.
- Tiến độ lúc 02:32 UTC:
  - HumanEval: 1/164.
  - MBPP: 1/257.
  - Tool-use: 11/40, thành công 0%, **87% lượt không gọi công cụ**. Kết quả này khớp với mốc v5 tool-use 0%: model gần như chưa biết gọi công cụ.
- Ước tính thô (theo tốc độ tool-use khoảng 85 giây/bài): tool-use xong khoảng 10:10 giờ VN. HumanEval và MBPP mới có 1 bài nên chưa đủ dữ liệu để ước tính chính xác, dự kiến vài giờ, xong vào khoảng trưa–chiều 15/10 giờ VN.

## Nhận xét và đề xuất

- Vòng thầy chung **có ích cho chữ tổng quát** (dịch +1,2, đời sống +0,5, kiến thức +0,4) nhưng **làm hỏng Excel (−1,13) và email (−0,5)**. Nhiều khả năng dữ liệu sửa thiên về hội thoại chung, đã ghi đè phần expert office. Nên kiểm tra tỷ lệ mẫu office/Excel trong `r1_txt` trước khi chạy vòng r2.
- Đọc ảnh không cải thiện. Gợi ý cho lần sau: tăng tỷ trọng ảnh tin tức, màn hình và code trong SFT/DPO, hoặc tách thành vòng riêng.
- Model vẫn bịa sự kiện (65% sai, 55% bịa với điều không có thật) và vẫn chép lại thay vì tóm tắt. Vòng thầy chưa chạm tới hai lỗi này.
- Bên mình chỉ đọc dữ liệu: không commit, không push, không xoá gì trên box.
