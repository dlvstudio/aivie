# TK_108 — Gemma 4 trên M4: tiếng Việt · văn phòng · tool (tự chấm)

*Ngày 06/11/2026 · Ollama · `gemma4:12b` (7,7 GB) và `gemma4:26b-a4b` (16 GB), bật chế độ suy nghĩ (mặc định), `MAXNEW=6144`, `CTX=16384`.*

*Bộ đề là 28 ca thực tế do mình tự soạn (`data/eval_clean/real_vto_28.json`). Mình đọc từng câu trả lời và chấm 0–10, cùng cách chấm với [22 ca code](TK_108_DANH_GIA_GEMMA4_THUC_TE.md). Với tool, kết quả công cụ được viết sẵn và đưa vào ở lượt thứ hai.*

## Tóm tắt

| | Gemma 4 12B | Gemma 4 26B-A4B |
|---|---|---|
| Tiếng Việt (12 ca) | 7,9 | **8,8** |
| Văn phòng (9 ca) | 9,3 | **9,4** |
| Tool (7 ca) | 9,7 | **10** |
| **Chung (28 ca)** | **8,8** | **9,3** |
| Code (22 ca, lần trước) | 9,3 | 9,8 |
| Thời gian mỗi lượt / tốc độ | ~43 s · ~27 token/s | **~22 s · ~55 token/s** |

- **Tool:** cả hai bản gần như hoàn hảo. Gọi đúng tên, đúng tham số, tự quy ra ngày ("ngày mai", "chủ nhật này"). Không gọi công cụ khi không cần, hỏi lại khi thiếu email, gọi song song 2 lần, viết SQL đúng, và trả lời đúng theo kết quả công cụ.
- **Văn phòng:** công thức Excel đúng (COUNTIFS, TEXTAFTER, cách tách tên cho bản Excel cũ, NETWORKDAYS.INTL mã 11). VBA chuẩn, có tính toán trên bảng doanh số. Biên bản và công văn có đủ quốc hiệu, tiêu ngữ, số hiệu; bản 26B còn dẫn đúng Nghị định 30/2020.
- **Tiếng Việt:** viết rất tự nhiên. Xưng hô "bác–cháu" nhất quán suốt 2 lượt, giọng miền Nam có "nè, nhen, á". Sửa đúng lỗi dành/giành, điều/đều, xung/sung, kỷ/kỹ; thêm dấu hoàn hảo.
- **Hai điểm yếu thật:**
  1. **Kiến thức cũ.** Cả hai bản đều nói Việt Nam có 63 tỉnh thành; đúng ra là 34 kể từ 1/7/2025. Kiến thức của model dừng trước khi sáp nhập.
  2. **Bản 12B bịa.** Nó tóm tắt rất trơn tru cuốn "Dòng sông không tên" của Nguyễn Huy Thiệp, một tác phẩm không có thật. Bản 26B thì nói rõ là không có tác phẩm này.
- **Bất ngờ:** bản 26B-A4B **nhanh gấp đôi** bản 12B trên M4. Lý do là 26B-A4B chỉ chạy khoảng 4 tỷ tham số cho mỗi token, còn 12B chạy cả 12 tỷ. Nếu RAM đủ (khoảng 16–18 GB), **26B-A4B tốt hơn ở mọi mặt**.

---

## Bảng điểm

### Tiếng Việt

| Ca | 12B | 26B | Nhận xét |
|---|---|---|---|
| v01 email xin nghỉ gửi sếp | 9 | 9 | Lễ phép, xưng "em – anh Tuấn" đúng, có nói sẽ bàn giao việc. Đưa 3 phương án, hơi dài so với yêu cầu "đừng dài quá". |
| v02 tóm tắt 3 ý | 8 | 8 | Đúng 3 ý, đúng ý chính, nhưng bỏ mất các con số (34 tỉnh, 3.321 xã). Bản 12B viết "sẽ vận hành" là sai thì. |
| v03 dịch Anh → Việt (báo kinh tế) | 9 | 9 | Dịch hay ("biên lợi nhuận bị thu hẹp", "lạc quan thận trọng"). Riêng "đơn hàng tồn đọng" mang nghĩa tiêu cực, chưa sát với "backlog". |
| v04 dịch thành ngữ Việt → Anh | 8 | 10 | Cả hai dùng "last minute / eleventh hour / headless chickens". Bản 12B có cụm "left it until the wire" không phải tiếng Anh chuẩn (đúng là "down to the wire"). |
| v05 giải thích 3 thành ngữ | 10 | 10 | Nghĩa đúng, ví dụ đời thường hợp lý. |
| v06 sửa chính tả | 10 | 10 | Sửa đủ 4 lỗi và giải thích dành/giành đúng. |
| v07 thêm dấu tin nhắn | 10 | 10 | Hoàn hảo. |
| v08 số tỉnh thành hiện nay | **2** | **2** | Trả lời "63 (58 tỉnh + 5 thành phố)". Sai, vì từ 1/7/2025 chỉ còn 34. |
| v09 lãi kép và lãi đơn | 10 | 10 | 119.101.600 đ và lãi 18.000.000 đ, đều đúng, có bảng so sánh. |
| v10 bài Facebook bán bánh, giọng miền Nam | 9 | 9 | Giọng tự nhiên, đúng giá và combo, dưới 120 chữ. Có vài từ hơi gượng ("tiết kiệm tốn luôn"). |
| v11 tác phẩm bịa | **1** | **10** | Bản 12B bịa cả một bài tóm tắt dài. Bản 26B nói rõ không có tác phẩm đó và gợi ý các khả năng nhầm. |
| v12 hội thoại "cháu – bác", 2 lượt | 9 | 9 | Nhập vai "bác" nhất quán, nội dung đúng (tỉ lệ xanh/nâu, ủ hiếu khí, xử lý mùi). Hơi dài. |

### Văn phòng

| Ca | 12B | 26B | Nhận xét |
|---|---|---|---|
| o01 COUNTIFS tháng 3/2026 | 10 | 8 | 12B dùng `">="&DATE(2026,3,1)`, đúng ở mọi máy. 26B viết ngày dạng chữ `">=01/03/2026"`, có thể bị hiểu sai trên máy cài định dạng ngày kiểu Mỹ (vẫn có cách SUMPRODUCT đúng). |
| o02 tách họ / tên | 10 | 10 | `TEXTAFTER(…," ",-1)` và cách cũ `TRIM(RIGHT(SUBSTITUTE(…REPT(" ",100)),100))` đều đúng. |
| o03 ngày làm việc, chỉ nghỉ Chủ nhật | 10 | 10 | `NETWORKDAYS.INTL(A2,B2,11,E2:E10)` đúng, mã 11 = chỉ nghỉ Chủ nhật. |
| o04 biên bản họp, thể thức hành chính | 7 | 9 | Đủ các phần. Bản 12B làm vỡ bảng tiêu đề (ngày tháng bị lặp ở cả hai cột). Bản 26B trình bày đúng hơn và dẫn Nghị định 30/2020. |
| o05 mục lục tự động trong Word | 10 | 10 | Heading, rồi References → Table of Contents, rồi Update Field. Đúng và đủ. |
| o06 dàn ý 8 slide | 9 | 9 | Đủ 8 slide, mỗi slide có 3 ý và gợi ý biểu đồ, bám đúng số liệu (+15%, −3%). |
| o07 tính trên bảng doanh số | 10 | 10 | Tổng 380/345/295/233, xếp hạng đúng, Dũng +63,3% mạnh nhất, Chi −40%. Nhận xét hợp lý. |
| o08 VBA tô dòng tồn kho < 10 | 10 | 10 | Tìm dòng cuối, kiểm tra sheet tồn tại, tô hoặc xoá màu cả dòng, tắt cập nhật màn hình khi chạy. |
| o09 công văn đòi nợ | 8 | 9 | Lịch sự và kiên quyết, có số tiền bằng chữ, dẫn đúng số hợp đồng. Bản 12B ghi số hiệu "…/2024" lệch với năm 2026. |

### Tool

| Ca | 12B | 26B | Nhận xét |
|---|---|---|---|
| t01 "ngày mai" ở Đà Nẵng | 10 | 10 | `get_weather(Đà Nẵng, 2026-11-07)`. |
| t02 tìm vé rẻ nhất rồi gửi mail (2 bước) | 10 | 10 | Tìm vé, rồi gửi mail chọn đúng chuyến Vietjet 13:30, giá 1.290.000. |
| t03 câu không cần tool | 10 | 10 | Trả lời thẳng "Tokyo". |
| t04 thiếu email | 10 | 10 | Hỏi lại địa chỉ email, không tự bịa ra. |
| t05 hai thành phố | 10 | 10 | Gọi song song 2 lần. |
| t06 công cụ báo lỗi | 8 | 10 | Cả hai tính đúng "chủ nhật" là 08/11. 26B gọi "Sài Gòn", gặp lỗi thì thử lại bằng "Ho Chi Minh City", rất hợp lý. 12B ngay lượt đầu đã gọi "Ho Chi Minh City", nên thông báo lỗi giả của mình không khớp. Nó thử lại bằng "Saigon", vẫn ổn nhưng không giải thích gì. |
| t07 SQL rồi trả lời | 10 | 10 | Viết 2 câu SQL đúng (SUM; GROUP BY + ORDER BY + LIMIT 3), sau đó trả lời đúng con số. |

---

## Kết luận và việc nên làm

1. **Dùng ngay được** cho tool và văn phòng ở cả hai bản. Nếu RAM cho phép, nên dùng **26B-A4B**: điểm cao hơn, nhanh gấp đôi, ít bịa hơn.
2. **Hai lỗ hổng cần vá nếu làm trợ lý tiếng Việt thật:**
   - **Kiến thức mới về Việt Nam:** địa giới sau 1/7/2025, luật mới, giá cả. Fine-tune không sửa tận gốc được việc này. Nên **ghép RAG** với kho tài liệu của mình (BM25/viwiki đã có), cộng thêm vài nguồn văn bản pháp luật mới.
   - **Bịa ở bản 12B:** nếu chọn 12B làm nền, LoRA cần có dữ liệu "nhận không biết" (bộ unanswerable cũ có sẵn). Bản 26B vốn đã tự từ chối đúng.
3. **Cả hai đều dài dòng.** Model hay đưa 2–3 phương án dù người dùng chỉ hỏi một. Đây là thứ LoRA văn phong sửa được dễ.
4. **Hướng đi:** tiếp tục phương án 1 (LoRA tiếng Việt trên Gemma 4), ưu tiên nền 26B-A4B, kèm RAG cho kiến thức mới. Không cần chuyển sang Mamba (xem [kế hoạch P0](TK_108_KE_HOACH_CHUNG_CAT_GEMMA_MAMBA.md)).

## Chạy lại

```bash
MODEL=gemma4:26b-a4b OUT=out/real_vto_gemma4_26b_a4b MAXNEW=6144 CTX=16384 python3 scripts/real_use_eval_ollama.py data/eval_clean/real_vto_28.json
MODEL=gemma4:12b OUT=out/real_vto_gemma4_12b MAXNEW=6144 CTX=16384 python3 scripts/real_use_eval_ollama.py data/eval_clean/real_vto_28.json
```
