# RAG + SeaLLM — kiến trúc lai (grounding của ta + LLM hiểu-thật)

Đường sinh: prompt → BM25 truy đoạn Wikipedia THẬT (viwiki 285k bài) → SeaLLM-7B sinh có căn cứ → verify (đối chiếu nguồn).

## Tiến hóa 3 thế hệ (prompt «chó»)
| Hệ | Kết quả |
|---|---|
| Mạng 54M (from-scratch) | "chó là một xã ở tỉnh Somme, Pháp" (ghép mẫu Wikipedia sai) |
| SeaLLM-7B đơn | "chó là con vật trung thành với con người" (hiểu, chung chung) |
| RAG + SeaLLM | "chó là loài động vật có vú, thuộc họ Canidae, thuần hóa hàng nghìn năm... chăn cừu, cảnh sát, cứu hộ, làm bạn" (đúng + có nguồn) |

## Kết quả RAG demo (8 câu, grounded_ratio)
- «chó là gì»  (8.6s, grounded=0.77)
  TRẢ LỜI: Chó là loài động vật có vú, thuộc họ Canidae, là loài vật nuôi phổ biến và được con người thuần hóa từ hàng nghìn năm trước. Chó có nhiều giống khác nhau, từ chó chăn cừu Đức đến chó rừng vằn hông, và chúng có nhiều vai trò trong cuộc sống hàng ngày của con người, từ việc chăn cừu, bảo vệ, cảnh sát, cứu hộ đến làm bạn.
  NGUỒN: Chó cắn hay bị chó cắn là thuật ngữ chỉ về những cuộc tấn công của loài chó gây ra cho con…
--
- «Hà Nội là gì»  (2.7s, grounded=0.4)
  TRẢ LỜI: Hà Nội là thủ đô của Việt Nam, là trung tâm chính trị, văn hóa và kinh tế của đất nước.
  NGUỒN: Đại học Hà Nội có thể là:  Trường Đại học Hà Nội, trường đại học công lập ở Hà Nội, tiền t…
--
- «internet là gì»  (3.6s, grounded=0.32)
  TRẢ LỜI: Internet là một mạng lưới toàn cầu kết nối hàng tỷ thiết bị thông qua các giao thức mạng như TCP/IP, cho phép trao đổi thông tin và dữ liệu giữa các máy tính và thiết bị khác nhau. Nó cung cấp một nền tảng cho việc truy cập và chia sẻ thông tin, giao tiếp và trao đổi dữ liệu thông qua các ứng dụng và dịch vụ trực tuyến.
  NGUỒN: Về bài luận Wikipedia về troll, xem Wikipedia:Troll là gì?  Một con Internet troll, hay đơ…
--
- «tế bào là gì»  (4.9s, grounded=0.68)
  TRẢ LỜI: Tế bào là đơn vị cơ bản của sự sống, là đơn vị cấu thành nên tất cả các sinh vật sống. Tế bào là một cấu trúc có khả năng tự tái tạo và thực hiện các chức năng sống như trao đổi chất, hấp thụ và giải phóng năng lượng, sinh sản và phân chia. Tế bào có thể được chia thành hai loại chính là tế bào nhân chuẩn và tế bào nhân sơ. Tế bào nhân chuẩn có nhân chứa DNA và có cấu trúc phức tạp hơn, trong khi tế bào nhân sơ không có nhân và cấu trúc đơn giản hơn.
  NGUỒN: Vách tế bào thứ cấp là một cấu trúc được tìm thấy trong nhiều tế bào thực vật, nằm giữa vá…
--
- «biến đổi khí hậu là gì»  (3.6s, grounded=1.0)
  TRẢ LỜI: Biến đổi khí hậu là sự thay đổi của hệ thống khí hậu gồm khí quyển, thủy quyển, sinh quyển, thạch quyển, băng quyển hiện tại và trong tương lai bởi các nguyên nhân tự nhiên và nhân tạo trong một giai đoạn nhất định tính bằng thập kỷ hay hàng triệu năm.
  NGUỒN: Ủy ban Liên chính phủ về Biến đổi khí hậu (tên tiếng Anh: Intergovernmental Panel on Clima…
--
- «trí tuệ nhân tạo là gì»  (3.7s, grounded=0.93)
  TRẢ LỜI: Trí tuệ nhân tạo là một lĩnh vực khoa học đang định hình lại xã hội của chúng ta, với sự phát triển nhanh chóng về kiến thức và tiến bộ trong lĩnh vực này đang thúc đẩy nhu cầu về các dịch vụ kỹ thuật số mới để khai thác công nghệ này với tiềm năng cao nhất.
  NGUỒN: Đây là bài con của Trí tuệ nhân tạo, nội dung chú trọng vào sự phát triển và lịch sử ngành…
--
- «mặt trời là gì»  (2.7s, grounded=0.7)
  TRẢ LỜI: Mặt Trời là một ngôi sao ở trung tâm hệ Mặt Trời của chúng ta, cung cấp năng lượng và ánh sáng cho Trái Đất.
  NGUỒN: Tháp năng lượng mặt trời (tiếng Anh: Solar power tower) là một loại lò năng lượng mặt trời…
--
- «sông Hồng ở đâu»  (3.2s, grounded=0.93)
  TRẢ LỜI: Sông Hồng là một con sông lớn ở miền Bắc Việt Nam, chảy qua các tỉnh như Hà Nội, Hưng Yên, Bắc Ninh, Hải Dương, Hải Phòng và là một phần của hệ thống sông Hồng.
  NGUỒN: Hệ thống sông Hồng là một mạng lưới các con sông, tập hợp quanh con sông chính là sông Hồn…


## Nhận xét
- **Chất lượng nhảy bậc**: câu trả lời đúng sự thật + mạch lạc + CÓ NGUỒN kiểm chứng được — thứ mạng 54M không bao giờ đạt.
- **grounded_ratio** = tỷ lệ từ nội dung trong câu trả lời xuất hiện trong nguồn; thấp = SeaLLM dùng kiến thức riêng (do BM25 truy lệch) → tín hiệu cần kiểm chứng.
- **Điểm cần cải thiện**: retrieval (BM25 trên lead) đôi khi truy bài phụ (chó→"chó cắn"). Nâng cấp: index toàn bài + boost tiêu đề, hoặc embedding retrieval (bge-m3).
- **Vai trò hệ của ta**: kho factual tiếng Việt + retrieval + verify = lớp NEO SỰ THẬT cho SeaLLM (chống ảo giác ở fact hiếm/địa phương). Công from-scratch không phí.
