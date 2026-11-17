# Kết quả sinh — mạng AIVIE (causal-LM)

- **Model**: d512MIX (d=512, 12 tầng, ~54M tham số) + W_skip grounding + kNN
- **Data train**: 174M token (FineWeb2-HQ + Wikipedia vi + prose factual)
- **Đường sinh**: /api/gen (W_skip vote → causal-LM nối, conf_stop chống-drift)
- **Ngày**: 2026-08-31
- **Phần cứng**: MacBook Pro M4 Pro 24GB (MPS)


## Định nghĩa

| Prompt | Kết quả sinh | nối? |
|---|---|---|
| «nước là chất» | nước là chất lỏng không màu, không mùi và tồn tại trong tự nhiên ở ao hồ, sông biển | — |
| «chó là con vật» | chó là con vật của các loài động vật có vú, nhưng nó được tìm thấy trong các khu vực khác nhau. | ✓ |
| «mặt trời là» | mặt trời là một ngôi sao thuộc dãy chính màu vàng chiếm khoảng | — |
| «cây xanh là» | cây xanh là cỏ được trồng để bảo vệ môi trường, nói chung . | — |
| «trái tim là» | trái tim là con người, coi là biểu tượng của tình cảm . | — |
| «âm nhạc là» | âm nhạc là nghệ thuật dùng những hình thức tổ hợp âm thanh nhất định trong diễn | — |
| «tình yêu là» | tình yêu là cảm yêu mến làm cho gắn bó mật thiết và có trách nhiệm với người, họ với | — |
| «giáo dục là» | giáo dục là một trong những trường hợp đặc biệt của các trường đại học, thường được gọi là trường trung học.  Các trường cao | ✓ |
| «tế bào» | tế bào đơn vị cơ bản của sự sống. | — |
| «máy tính» | máy tính thực hiện tự động các phép tính . | — |
| «hạnh phúc là» | hạnh phúc là trạng thái sung sướng vì cảm thấy hoàn toàn đạt được ý nguyện . | — |
| «sức khỏe là» | sức khỏe là một phần vô cùng quan trọng đóng góp cho việc phát triển cuộc sống của bạn ở hiện tại và kể cả trong tương | — |

## Thực thể / Factual

| Prompt | Kết quả sinh | nối? |
|---|---|---|
| «Hà Nội là» | Hà Nội là thủ đô của nước Cộng hòa xã hội chủ nghĩa Việt Nam. | — |
| «Việt Nam là» | Việt Nam là thành viên WTO. USD | — |
| «Trái Đất là» | Trái Đất là hành tinh lớn nhất trong bốn hành tinh đất đá của hệ Mặt Trời, về cả kích thước và khối lượng. | — |
| «internet là» | internet là Martin, một trong những người đã từng làm việc với các nhà sản xuất và nhà thiết kế, người ta đã làm được | ✓ |
| «mặt trăng» | mặt trăng tròn tự tin tự tin, nhưng không thể nào bị bắt. | ✓ |
| «sông Hồng» | sông Hồng chảy qua Hà Nội. | — |
| «thủ đô của Việt Nam là» | thủ đô của Việt Nam là Hà Nội.Tiêu đề: Nguyễn Văn Sơn  Nguyễn Văn Thiệu (1945-10 tháng 12 năm 1 | ✓ |
| «chiến tranh thế giới» | chiến tranh thế giới thứ hai kết thúc năm nào? Đáp: Chiến tranh Việt Nam 1 trong | — |

## Khoa học / Trừu tượng

| Prompt | Kết quả sinh | nối? |
|---|---|---|
| «trí tuệ nhân tạo» | trí tuệ nhân tạo thay đổi lao động. | — |
| «biến đổi khí hậu» | biến đổi khí hậu cần hành động chung. | — |
| «vũ trụ» | vũ trụ là khoảng cách giữa hai vật thể, một vật liệu có thể được tạo ra bởi các vật chất khác nhau.  Các vật | ✓ |
| «con người cần» | con người cần ăn uống đầy đủ. - Ăn nhiều rau, rau xanh, cà chua, dưa chuột, quả táo, các loại rau củ, khoai | ✓ |

## Nối / Thủ tục / Câu hỏi

| Prompt | Kết quả sinh | nối? |
|---|---|---|
| «ngày xưa có một» | ngày xưa có một liêng Một giờ Một. “Tôi không biết rằng tôi đã có một người bạn của mình. Tôi không thể nói rằng họ có thể | ✓ |
| «buổi sáng mùa thu» | buổi sáng mùa thu vào mùa đông, Đà Lạt có mưa phùn và trời ít khi có nắng. | — |
| «học sinh cần» | học sinh cần phải đăng ký tạo tài khoản, lựa chọn gói học phù hợp và thanh toán phải | — |
| «ăn nhiều rau để» | ăn nhiều rau để quả tươi có nhiều sinh tố, ăn điều độ, đủ chất (đạm, béo tươi, đủ chất béo bột | — |
| «tập thể dục giúp» | tập thể dục giúp buổi sáng. Từ đó, bạn có thể sử dụng các loại thuốc này để điều trị bệnh tim. Ví dụ, một số | ✓ |
| «mặt trời mọc ở» | mặt trời mọc ở Cối, một số loài động vật có vú khác như cá heo. | ✓ |
| «một cộng một bằng» | một cộng một bằng dụ về các thành viên của nhóm này.  Các thành phố khác  Mury, 1987. | ✓ |
| «nấu cơm cần» | nấu cơm cần salad, cho vào nồi, để cho thêm nước sôi. - Dùng một miếng nhỏ, cắt nhỏ để làm sạch. | ✓ |

---

## Đánh giá

- **Điểm mạnh**: định nghĩa & fact có trong kho grounding → chính xác, đọc được như từ điển/bách khoa (Hà Nội, Trái Đất, tế bào, âm nhạc, hạnh phúc...).
- **Điểm yếu**: prompt trừu tượng / thủ tục / toán → trôi chủ đề hoặc rác (không suy luận, chỉ khớp mẫu).
- **Trần**: model 54M tham số + ctx 128 + 174M token — đã vắt kiệt máy M4 Pro. Nâng 'yếu→ổn' cần model lớn hơn (GPU CUDA).

*Ghi chú: cột 'nối?' = ✓ nghĩa là W_skip ngắn nên causal-LM nối thêm; '—' = định nghĩa grounded đủ dài, không nối.*
---

## Sáng tạo hay thuần chép? (kiểm bằng grep data train)

**Kết luận: CHỦ YẾU CHÉP + TÁI TỔ HỢP MẪU — chưa phải sáng tạo hiểu biết.**

**1. Output ĐÚNG = chép nguyên văn.** Mọi định nghĩa hay khớp verbatim trong data:
- «nước là chất lỏng không màu, không mùi...» → có nguyên văn trong `data/text_corpus/dinh_nghia_ho.jsonl`, `dictionary_ho.jsonl`.
- «hạnh phúc: trạng thái sung sướng...», «âm nhạc: tổ hợp âm thanh...», «tế bào: đơn vị cơ bản của sự sống» → đều là mục từ điển/QA có sẵn.

**2. Output "mới" = ghép template đã học (novel nhưng thường SAI):**
- «chó là một xã ở tỉnh Somme, vùng Hauts-de-France, Pháp. Thông tin nhân khẩu. Xem thêm. Tham khảo...» — câu-với-"chó" KHÔNG có trong data, nhưng **template** "là một xã ở tỉnh Somme..." có nguyên văn trong wiki, và **6358 bài** dùng mẫu "là một xã ở tỉnh". → thuộc lòng template + gắn sai chủ đề.

**3. Đôi khi ghép tạm ổn:** «con người cần → ăn nhiều rau, cà chua, dưa chuột, quả táo» (không verbatim, ghép mới, hợp lý — nhưng hên-xui).

**Model làm được**: chép (đúng) + tái tổ hợp mẫu (novel nhưng hên-xui đúng).
**Model CHƯA làm được**: "sáng tạo mà đúng" — chuỗi mới, mạch lạc VÀ đúng sự thật. Cần model đủ lớn (emergence ở hàng trăm triệu–tỷ tham số) = GPU CUDA.
