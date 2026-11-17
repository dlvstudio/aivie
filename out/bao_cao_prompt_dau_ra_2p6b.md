# Thử model nền: prompt và đầu ra cụ thể — 2,5B (mốc 20 tỷ) vs 1.3B

*12/10/2026 · cả hai là model NỀN (chưa SFT) · viết tiếp, nhiệt độ 0,7, tối đa 120 token (đầu ra bị cắt ở 120 token) · chạy trên box P2 bằng `scripts/probe_base_prompts.py` · checkpoint: `cpt_b.pt` bước 9536 (2,49B) và `hybrid_1p3b.pt` (1.3B gốc)*

Đầu ra giữ NGUYÊN VĂN (kể cả chỗ sai, chỗ lặp). Dòng *Nhận xét* là ghi chú khi đọc, không phải điểm máy chấm.

## 1. Trắc nghiệm kiến thức (44 câu, chấm bằng log-xác-suất: model chọn đáp án đúng hay đáp án nhiễu)

**Tổng: 1.3B 36/44 · 2,5B 38/44**

| # | Câu | Đáp án đúng | 1.3B | 2,5B |
|---|---|---|---|---|
| 1 | Thủ đô của Việt Nam là | Hà Nội | ✓ | ✓ |
| 2 | Thủ đô của Pháp là | Paris | ✓ | ✓ |
| 3 | Thủ đô của Nhật Bản là | Tokyo | ✓ | ✓ |
| 4 | Sông dài nhất Việt Nam chảy qua nhiều nước là sông | Mê Kông | ✓ | ✓ |
| 5 | Đỉnh núi cao nhất Việt Nam là | Fansipan | ✗ Bạch Mã | ✗ Bạch Mã |
| 6 | Chiến thắng Điện Biên Phủ diễn ra vào năm | 1954 | ✓ | ✓ |
| 7 | Ngày Quốc khánh của Việt Nam là ngày | 2 tháng 9 | ✓ | ✓ |
| 8 | Tác giả của Truyện Kiều là | Nguyễn Du | ✓ | ✓ |
| 9 | Nước sôi ở nhiệt độ | 100 độ C | ✓ | ✓ |
| 10 | Công thức hoá học của nước là | H2O | ✓ | ✓ |
| 11 | Hành tinh gần Mặt Trời nhất là | Sao Thủy | ✗ Trái Đất | ✓ |
| 12 | Một năm có | 12 tháng | ✓ | ✓ |
| 13 | 7 nhân 8 bằng | 56 | ✗ 64 | ✗ 64 |
| 14 | Thành phố lớn nhất Việt Nam theo dân số là | Thành phố Hồ Chí Minh | ✓ | ✓ |
| 15 | Quốc gia đông dân nhất Đông Nam Á là | Indonesia | ✓ | ✓ |
| 16 | Con vật được gọi là chúa sơn lâm là | con hổ | ✓ | ✓ |
| 17 | Đơn vị đo cường độ dòng điện là | ampe | ✓ | ✓ |
| 18 | Người đầu tiên đặt chân lên Mặt Trăng là | Neil Armstrong | ✓ | ✓ |
| 19 | The capital of Germany is | Berlin | ✓ | ✓ |
| 20 | Water is made of hydrogen and | oxygen | ✓ | ✓ |
| 21 | Vịnh Hạ Long thuộc tỉnh | Quảng Ninh | ✓ | ✓ |
| 22 | Phở là món ăn nổi tiếng của | Việt Nam | ✓ | ✓ |
| 23 | Trong Python, để in ra màn hình ta dùng hàm | print | ✗ printf | ✓ |
| 24 | Tết Nguyên Đán là ngày đầu năm theo | âm lịch | ✓ | ✓ |
| 25 | Vua đầu tiên của nhà Nguyễn là | Gia Long | ✓ | ✓ |
| 26 | Kinh đô của nhà Lý được dời từ Hoa Lư về Thăng Long vào năm | 1010 | ✓ | ✓ |
| 27 | Trận Bạch Đằng năm 938 do | Ngô Quyền | ✓ | ✓ |
| 28 | Tác giả của Bình Ngô đại cáo là | Nguyễn Trãi | ✓ | ✓ |
| 29 | Tác phẩm Tắt đèn là của nhà văn | Ngô Tất Tố | ✓ | ✓ |
| 30 | Tỉnh có diện tích lớn nhất Việt Nam là | Nghệ An | ✗ Sơn La | ✗ Sơn La |
| 31 | Cố đô Huế nằm bên bờ sông | Hương | ✓ | ✓ |
| 32 | Cao nguyên đá Đồng Văn thuộc tỉnh | Hà Giang | ✓ | ✓ |
| 33 | Định luật vạn vật hấp dẫn do nhà khoa học | Newton | ✓ | ✓ |
| 34 | Thuyết tiến hóa bằng chọn lọc tự nhiên do | Darwin | ✓ | ✓ |
| 35 | Kim loại ở thể lỏng ở nhiệt độ phòng là | thủy ngân | ✓ | ✓ |
| 36 | Cơ quan sản xuất insulin trong cơ thể người là | tuyến tụy | ✓ | ✓ |
| 37 | Căn bậc hai của 144 là | 12 | ✓ | ✓ |
| 38 | Tổng các góc trong của một tam giác bằng | 180 độ | ✓ | ✓ |
| 39 | 15% của 200 bằng | 30 | ✗ 20 | ✗ 20 |
| 40 | Số nguyên tố nhỏ nhất là | 2 | ✗ 1 | ✗ 1 |
| 41 | Trong SQL, câu lệnh để lấy dữ liệu từ bảng là | SELECT | ✗ UPDATE | ✗ INSERT |
| 42 | Đồng tiền chính thức của Nhật Bản là | yên | ✓ | ✓ |
| 43 | The largest ocean on Earth is the | Pacific Ocean | ✓ | ✓ |
| 44 | The author of Romeo and Juliet is | William Shakespeare | ✓ | ✓ |

Cả hai cùng sai: đỉnh núi cao nhất (chọn Bạch Mã), 7×8 (chọn 64), 15% của 200 (chọn 20), số nguyên tố nhỏ nhất (chọn 1), tỉnh lớn nhất (chọn Sơn La — đáp án gốc Nghệ An theo địa giới trước 2025), SQL SELECT. 2,5B sửa được 2 câu 1.3B sai: hành tinh gần Mặt Trời nhất (Sao Thủy), hàm in của Python (print).

## 2. Viết tiếp: 30 prompt

### 1. [tin]

**Prompt:**

~~~~text
Hà Nội, ngày 15/3 – Sở Giao thông vận tải thành phố cho biết
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
, đơn vị này đang tích cực phối hợp với các cơ quan chức năng để tiến hành xử lý vi phạm về việc kinh doanh thực phẩm không đảm bảo vệ sinh an toàn thực phẩm (ATVSTP).
Theo ông Nguyễn Xuân Tân – Phó Giám đốc Sở Giao thông vận tải Hà Nội, hiện nay, trên địa bàn thành phố có hơn 50% số lượng xe ô tô khách lưu thông không đăng ký và cấp biển kiểm soát, thậm chí còn là “xe dù, bến cóc” nên khó quản lý. Nhiều xe hoạt động không có giấy phép kinh doanh vận tải,
~~~~

**1.3B (gốc):**

~~~~text
, từ ngày 16 đến 31/3 sẽ cấm tất cả các phương tiện xe thô sơ, xe gắn máy, xe ba gác tự chế và xe thô sơ, xe cơ giới hoạt động trên địa bàn thành phố trong khung giờ từ 6h00 đến 22h00. Đây là một trong những biện pháp mạnh tay của Hà Nội nhằm kiềm chế tình trạng xe mô tô, xe gắn máy, xe ba bánh tự chế không rõ nguồn gốc, xuất xứ được sản xuất, lắp ráp tràn lan trên địa bàn thành phố. Theo ông Nguyễn Hoàng Linh – Phó giám đốc
~~~~

*Nhận xét:* 2,5B: đúng văn phong tin tức nhưng nội dung lạc (Sở GTVT → an toàn thực phẩm), có tên người bịa.

### 2. [tin]

**Prompt:**

~~~~text
Theo Tổng cục Thống kê, GDP của Việt Nam trong quý I năm nay
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
tăng 6,89% so với cùng kỳ năm trước. Đây là mức tăng trưởng cao nhất của quý I kể từ năm 2012 trở lại đây.
Những con số trên cho thấy kinh tế Việt Nam đang phục hồi mạnh mẽ sau những tác động tiêu cực do ảnh hưởng của dịch Covid-19.
Mặc dù có sự phục hồi ấn tượng, song TS. Võ Trí Thành - nguyên Phó Viện trưởng Viện Nghiên cứu và Quản lý kinh tế Trung ương (CIEM), cho rằng vẫn còn nhiều điều đáng lo ngại.
Thứ nhất, tăng trưởng GDP
~~~~

**1.3B (gốc):**

~~~~text
đạt 4,03% so với cùng kỳ năm trước. Tuy nhiên, con số này vẫn thấp hơn nhiều so với mục tiêu 6,7% mà Quốc hội đã thông qua hồi đầu năm. Nếu xét theo mức tăng trưởng kinh tế dự kiến 6,7% thì đây vẫn là một con số khá thấp và chưa đạt được mục tiêu đề ra của Chính phủ. |Sản xuất công nghiệp tháng 4/2019 giảm 1,3% so với cùng kỳ năm trước| Trước tình hình này, nhiều chuyên gia nhận định, để đạt được mục tiêu tăng trưởng GDP 6
~~~~

*Nhận xét:* 2,5B: viết như tin thật; số liệu (6,89%) không kiểm chứng được — có thể bịa.

### 3. [tin]

**Prompt:**

~~~~text
Đội tuyển bóng đá Việt Nam đã có chiến thắng
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
lịch sử khi đánh bại đội tuyển Thái Lan với tỉ số 3–0. Tại trận đấu này, đội tuyển Thái Lan chỉ có thể ghi được 2 bàn thắng vào lưới của đội tuyển Việt Nam, trong khi đó các cầu thủ của đội tuyển Việt Nam cũng đã lập cú hat-trick của Nguyễn Quang Hải, Phạm Thành Lương và Nguyễn Văn Toàn, qua đó trở thành những cầu thủ ghi bàn thắng với tổng số pha lập công nhiều nhất ở vòng bảng.

Sau trận đấu 
Sau trận đấu, huấn luyện viên Nishino Akira đã dành thời gian nằm nghỉ trên sân vận động
~~~~

**1.3B (gốc):**

~~~~text
đầu tiên ở vòng loại World Cup 2022 khu vực châu Á. Tiền vệ Nguyễn Quang Hải là một trong những người hùng của đội tuyển Việt Nam với cú đúp bàn thắng, góp phần mang về trận thắng cho đội nhà trước đối thủ Saudi Arabia tại lượt trận thứ 3 vòng loại thứ 3 World Cup 2022 khu vực châu Á tối 14/10. Với kết quả này, đoàn quân HLV Park Hang-seo tiếp tục dẫn đầu bảng G với 11 điểm sau 4 trận toàn thắng. Đội tuyển Việt Nam đã có chiến thắng đầu tiên ở vòng loại
~~~~

*Nhận xét:* 2,5B: bịa trận đấu và hat-trick; mâu thuẫn tỉ số (3–0 nhưng Thái Lan ghi 2 bàn).

### 4. [wiki]

**Prompt:**

~~~~text
Vịnh Hạ Long là một vịnh nhỏ thuộc phần bờ tây vịnh Bắc Bộ,
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
tại khu vực biển Đông Bắc Việt Nam. Vịnh Hạ Long được UNESCO nhiều lần công nhận là di sản thế giới vì vẻ đẹp kỳ vĩ của tạo hóa.
Năm 1994, vịnh Hạ Long đã được UNESCO công nhận là di sản thiên nhiên thế giới bởi giá trị thẩm mỹ và địa chất, địa mạo.
Vịnh Hạ Long nằm ở khu vực Đông Bắc Việt Nam, phía tây bắc Vịnh giáp đảo Cát Bà, phía đông nam giáp biển Đông, phía tây nam giáp đất liền với đường bờ biển dài 120 km.
Vịnh Hạ Long có tổng diện tích khoảng
~~~~

**1.3B (gốc):**

~~~~text
bao gồm vùng biển đảo thuộc thành phố Hạ Long, thành phố Cẩm Phả và một phần của huyện đảo Vân Đồn của tỉnh Quảng Ninh. Vịnh Hạ Long được UNESCO nhiều lần công nhận là di sản thiên nhiên của thế giới với hàng nghìn hòn đảo được làm nên bởi tạo hoá kỳ vĩ và sống động. Vịnh Hạ Long có phong cảnh tuyệt đẹp nên đã từng được chọn là một trong những địa điểm tổ chức đám cưới lãng mạn nhất Việt Nam. 2. Phong cách trang trí tiệc cưới ngoài trời Với sự phát triển không ngừng của xã hội hiện đại, các cặp đôi cũng mong muốn đám
~~~~

*Nhận xét:* 2,5B: đúng (UNESCO 1994, giáp Cát Bà, bờ biển 120 km).

### 5. [wiki]

**Prompt:**

~~~~text
Chủ tịch Hồ Chí Minh sinh ngày 19 tháng 5 năm 1890 tại
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
Kim Liên, xã Nam Đàn, tỉnh Nghệ An.
Năm 1911: Người ra đi tìm đường cứu nước.
Tháng 7 năm 1919: Người gia nhập Đảng Xã hội Pháp.
Tháng 6 năm 1920: Người tham dự Đại hội Tua.
Tháng 12 năm 1920: Người tham gia thành lập Hội Liên hiệp các dân tộc thuộc địa ở Pari và xuất bản báo “Người cùng khổ”.
Năm 1921: Người tham dự Đại hội V của Quốc tế Cộng sản.
Năm 1924
~~~~

**1.3B (gốc):**

~~~~text
xã Kim Liên, huyện Nam Đàn, tỉnh Nghệ An. Cha là cụ Nguyễn Sinh Sắc, một nhà Nho nghèo, mẹ là bà Hoàng Thị Loan, người đã cưu mang Bác trong những ngày đầu đi tìm đường cứu nước. Năm 1906, Cụ Sắc làm quan tri phủ tỉnh Quảng Bình và cụ mất tại quê nhà vào năm 1915. Mẹ của Bác Hồ là Hoàng Thị Loan, con gái của một gia đình giàu có ở làng Hoàng Trù, xã Kim Liên, huyện Nam Đàn, Nghệ An. Ngay từ nhỏ, Bác Hồ đã phải sống trong cảnh nghèo khổ
~~~~

*Nhận xét:* 2,5B: đúng (Kim Liên, Nam Đàn, Nghệ An) + niên biểu hợp lý.

### 6. [wiki]

**Prompt:**

~~~~text
Sông Mê Kông bắt nguồn từ
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
Trung Quốc, chảy qua Lào, Myanma, Thái Lan, Campuchia và đổ ra Biển Đông ở Việt Nam. Tính theo độ dài đứng thứ 12 (thứ 7 tại châu Á), còn tính theo lưu lượng nước đứng thứ 10 trên thế giới (lưu lượng hàng năm đạt khoảng 475 triệu m³). Lưu lượng trung bình 13.200m³/s, vào mùa nước lũ có thể lên tới 30.000m³/s. Lưu vực của nó rộng khoảng 795.000km² (theo số liệu của Ủy ban sông Mê
~~~~

**1.3B (gốc):**

~~~~text
cao nguyên Tây Tạng, chảy qua các nước Lào, Myanmar, Thái Lan và Việt Nam. Sông Mê Kông có chiều dài hơn 6.300 km, trong đó đoạn sông chảy qua Trung Quốc là dài nhất (1.850 km). Tuy nhiên, phần lớn dòng chảy của sông Mê Kông lại tập trung ở lãnh thổ Trung Quốc với chiều dài 5.280 km, chiếm gần 99% tổng chiều dài toàn bộ con sông này. Với chiều dài như vậy nên sông Mê Kông được mệnh danh là “con sông dài nhất thế giới”. Sông Mê
~~~~

*Nhận xét:* 2,5B: đúng phần lớn (chảy qua TQ–Lào–Myanma–Thái–Campuchia–VN).

### 7. [wiki]

**Prompt:**

~~~~text
Quang hợp là quá trình
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
chuyển hóa năng lượng của ánh sáng mặt trời thành các chất dinh dưỡng cần thiết cho cây xanh. Quá trình này diễn ra trong lục lạp, một bào quan có chức năng quang hợp nằm ở bên trong màng tế bào. Lục lạp chứa nhiều hệ thống enzyme đóng vai trò quan trọng trong việc chuyển hóa năng lượng ánh sáng thành năng lượng hóa học.
 Quang tự dưỡng: Đây là quá trình sử dụng năng lượng ánh sáng để tổng hợp các hợp chất hữu cơ từ CO2 và H2O. Các hợp chất này được tạo ra thông qua quá trình quang phân ly nước và quá
~~~~

**1.3B (gốc):**

~~~~text
sinh tổng hợp chất hữu cơ từ các chất vô cơ và sản sinh năng lượng từ phản ứng phân hủy các chất hữu cơ. Quá trình này xảy ra trong tế bào thực vật khi có ánh sáng chiếu vào. + Sinh trưởng: Là sự tăng lên về kích thước của tế bào, mô, cơ quan theo sự tác động của các điều kiện môi trường sống. + Phân hóa mầm hoa: Là hiện tượng hạt phấn được giải phóng từ một điểm sinh trưởng trên một cây, từ đó hình thành nên quả. + Quả: Là đơn vị chức năng của cây, có nhiệm vụ dự trữ
~~~~

*Nhận xét:* 2,5B: đúng ý chính (lục lạp, năng lượng ánh sáng → hoá học).

### 8. [wiki]

**Prompt:**

~~~~text
Chiến thắng Điện Biên Phủ năm 1954
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
, Hà Nội đã được tặng thưởng nhiều huân chương cao quý. Trong đó có một số huân chương chiến công như: Huân chương Quân công hạng nhất, Huân chương Chiến thắng hạng nhất, Huân chương Kháng chiến hạng nhất, Huân chương Lao động hạng nhất, Huân chương Độc lập hạng nhất, Huân chương Chiến sĩ vẻ vang hạng nhất, hai lần được tặng danh hiệu Anh hùng Lực lượng vũ trang nhân dân.

Lịch sử 
Ngày 20 tháng 12 năm 1953, Bộ Chính trị Ban Chấp hành Trung ương Đảng quyết định mở chiến dịch Điện Biên Phủ. Trong giai
~~~~

**1.3B (gốc):**

~~~~text
đã ghi một dấu mốc chói lọi trong lịch sử dân tộc Việt Nam, đánh bại chủ nghĩa thực dân cũ và mới, giải phóng miền Bắc, tiến tới thống nhất đất nước. Chiến thắng này là thắng lợi của đường lối cách mạng đúng đắn, sáng tạo của Đảng ta và Chủ tịch Hồ Chí Minh; là thắng lợi của sự đoàn kết chiến đấu, hy sinh cao cả, ý chí quyết chiến, quyết thắng của toàn dân tộc Việt Nam trong cuộc kháng chiến trường kỳ chống thực dân Pháp xâm lược. Đó cũng là thắng lợi của đường lối cách mạng đúng đắn, sáng tạo của
~~~~

*Nhận xét:* 2,5B: lạc đề sang huân chương; 1.3B đúng hơn ở câu này.

### 9. [wiki]

**Prompt:**

~~~~text
Thành phố Đà Lạt nằm trên cao nguyên
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
Langbiang, với độ cao 1.500m so với mực nước biển, khí hậu mát mẻ quanh năm, rất thích hợp cho du lịch nghỉ dưỡng.
- Với độ cao lý tưởng cộng với khí hậu ôn hòa, mát mẻ quanh năm, Đà Lạt là một trong những địa điểm du lịch nghỉ dưỡng hàng đầu Việt Nam.
- Ngoài ra, Đà Lạt còn là thành phố của nhiều hồ nước đẹp như hồ Xuân Hương, hồ Than Thở, hồ Đa Thiện… Các hồ này đều có những nét riêng, tạo nên vẻ đẹp thơ mộng cho thành phố.
~~~~

**1.3B (gốc):**

~~~~text
Lâm Viên, tỉnh Lâm Đồng. Thành phố này được thành lập vào ngày 23 tháng 10 năm 1976 theo Quyết định số 37-CP của Hội đồng Chính phủ. Đà Lạt có một vị trí địa lý thuận lợi, nằm ở độ cao khoảng 1.500 mét so với mực nước biển, nên khí hậu nơi đây mát mẻ quanh năm. Nhiệt độ trung bình hàng năm khoảng 20°C, là một trong những thành phố ôn đới đẹp nhất Việt Nam. Địa hình của Đà Lạt gồm nhiều đồi núi và thung lũng, có các dãy núi cao
~~~~

### 10. [truyện]

**Prompt:**

~~~~text
Ngày xửa ngày xưa, ở một làng nhỏ ven sông, có hai anh em mồ côi
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
cha mẹ từ sớm, nhà nghèo sống với nhau rất hòa thuận. Tuy vậy, người anh lại rất tham lam còn người em thì lương thiện, thật thà.
Một hôm, cả hai anh em rủ nhau đi câu cá. Người anh câu được con cá rất to, đem về định nấu cho em ăn nhưng khi đưa lên miệng thì nó đã bay mất. Sau đó, người em cũng câu được một con cá, con cá này lớn hơn con cá hôm qua nhiều nên cậu nghĩ chắc hẳn là người anh tham lam kia ăn mất rồi. Cậu liền mang về nhà và khoe với người
~~~~

**1.3B (gốc):**

~~~~text
cha mẹ từ nhỏ sống với nhau. Người anh tên là Tí và người em tên là Bét. Tí chăm chỉ, hiếu thảo, chịu khó làm ăn, tính tình hiền lành. Còn Bét thì lười biếng, ham chơi, thường hay trộm cắp vặt. Hai anh em rất thương yêu nhau. Một hôm, Bét đi chăn trâu ở đồng xa, để Tí ở nhà một mình trông nom ngôi nhà nhỏ. Bét về, không thấy Tí đâu liền đi tìm khắp nơi. Bỗng thấy có con chim lạ từ trên cây bay xuống đậu trên nóc nhà, kêu mấy tiếng rồi
~~~~

### 11. [truyện]

**Prompt:**

~~~~text
Mưa rơi lất phất trên phố cổ. Lan đứng nép dưới mái hiên, tay cầm lá thư
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
xanh đỏ và cây bút chì nhỏ nhắn. Bỗng cô nghe tiếng gọi mình, Lan giật mình: - Sao em biết anh ở đây? - Anh có phải là người đã gửi bức thư này cho em không? - Anh đang đi tìm em đây! Em đứng đợi anh nhé! Nói rồi, Lan bỏ chạy, gió thổi lạnh buốt cả vai. Cô ngồi xuống bậc thềm, tay cầm lá thư xanh đỏ và cây bút chì nhỏ nhắn. Gió lại thổi, lá thư bay bay trong gió. Nhìn theo bóng dáng bé nhỏ của Lan khuất dần sau cánh cổng, lòng tôi tự hỏi:
~~~~

**1.3B (gốc):**

~~~~text
tay của con trai, mắt rưng rưng nhìn người cha già nua ngồi trong ngôi nhà nhỏ. Ba Lan chết vì bị ung thư phổi. Ông để lại cho vợ và hai đứa con gái một số vốn ít ỏi. Ông đã phải bán hết tài sản có giá trị để chạy chữa cho con. “Giá như ngày ấy tôi đừng quyết định đi xuất khẩu lao động, giá như tôi không lấy vợ sớm…” – Lan tự trách mình. Mẹ của Lan cũng là người đàn bà chịu nhiều đau khổ. Bà sinh được ba người con nhưng cả ba đều chết trẻ. Chồng chết, bà
~~~~

### 12. [thơ]

**Prompt:**

~~~~text
Trăm năm trong cõi người ta,
Chữ tài chữ mệnh khéo là ghét nhau.
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
Trải qua một cuộc bể dâu,
Những điều trông thấy mà đau đớn lòng.
Lạ gì bỉ sắc tư phong,
Trời xanh quen thói má hồng đánh ghen.
Ngẫm hay muôn sự tại trời,
Trời kia đã bắt làm người có thân.
Bắt phong trần phải phong trần,
Cho thanh cao mới được phần thanh cao.
Có đâu thiên vị người nào,
Chữ tài chữ mệnh dồi dào cả hai,
Có tài mà cậy chi tài,
Chữ tài liền với chữ tai một vần.
Đã mang lấy nghiệp vào thân,
Cũng đừng trách lẫn
~~~~

**1.3B (gốc):**

~~~~text
Chữ tài chữ mệnh khéo là ghét nhau Có thể nói, bài thơ “Hai câu luận” của Hồ Chí Minh là một sáng tạo đặc biệt về nghệ thuật thơ văn chính luận, vừa sâu sắc, vừa có tính thuyết phục cao. Bài thơ không chỉ thể hiện tâm tư, tình cảm của Bác Hồ mà còn bộc lộ quan điểm riêng của Người về con người và cuộc đời.
~~~~

*Nhận xét:* 2,5B: chép đúng nguyên văn Truyện Kiều; 1.3B lạc đề.

### 13. [văn]

**Prompt:**

~~~~text
Mùa thu Hà Nội đẹp nhất là vào những buổi sáng
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
mùa thu, khi mà tiết trời se se lạnh, được ngồi bên gia đình thưởng thức một bát phở nghi ngút khói, kèm theo một đĩa quẩy giòn tan thì còn gì tuyệt vời bằng.
Để làm món ăn này cũng rất đơn giản, các bạn chỉ cần chọn mua được những chiếc bánh phở tươi ngon nhất là được. Với công thức nấu phở gà thơm ngon chuẩn vị dưới đây, đảm bảo sẽ mang đến cho gia đình bạn một bữa sáng thật ngon miệng và đầy đủ dinh dưỡng đấy nhé.
Nguyên liệu làm món phở gà
- Gà ta: 1 con khoảng 1,
~~~~

**1.3B (gốc):**

~~~~text
sớm khi tiết trời còn se lạnh, bạn có thể cùng người thân dạo quanh hồ Tây để cảm nhận hết nét đặc trưng của mùa thu Hà Nội. Bạn có thể đi bộ hoặc thuê xe đạp đôi để ngắm cảnh Hồ Tây trong lành, yên bình. Ngoài ra bạn cũng có thể lên cầu Long Biên để ngắm nhìn toàn cảnh phố cổ Hà Nội và ngắm những chiếc lá vàng rơi trên mặt hồ vào buổi chiều tà. Một điều đặc biệt nữa mà bạn không nên bỏ qua đó là hãy thử đi xe máy quanh hồ Tây để cảm nhận hết cái vẻ đẹp của nơi đây. Khi đến
~~~~

### 14. [hướng dẫn]

**Prompt:**

~~~~text
Cách nấu phở bò Hà Nội:
Nguyên liệu:
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
– 500g thịt bò thăn hoặc sườn non, thái miếng to bản.
– 2 củ hành khô, 3 cây sả, 2 củ tỏi, 1 củ gừng, 1 quả ớt (nếu bạn ăn cay)
– Nước mắm, gia vị, hạt nêm, đường, giấm, chanh, rượu trắng và nước
– Rau thơm: Hành lá, rau mùi, ngò gai, húng quế, rau diếp cá… tùy theo sở thích của bạn.
Cách làm:
Bước 1: Ướp thịt bò với một chút muối, tiêu, dầu ăn và gừng băm nhỏ trong khoảng
~~~~

**1.3B (gốc):**

~~~~text
– Một món ăn ngon cho ngày cuối tuần1. Cách nấu phở bò Hà Nội: Cháo lòng Nguyên liệu làm cháo lòng Cách nấu phở bò Hà Nội: Cháo lòng2. Cách nấu phở bò Hà Nội: Bún thang Nguyên liệu: bún thang, tôm khô, trứng gà, nấm hương, củ cải trắng, nước mắm, hạt nêm, tiêu, hành lá, rau mùi… Cách nấu phở bò Hà Nội: Bún thangNguyên liệu làm bún thang3. Cách nấu phở bò Hà Nội: Bánh cuốn Nguyên liệu: bột gạo, thịt lợn băm nhỏ, mộc nhĩ, nấm hương,
~~~~

### 15. [hướng dẫn]

**Prompt:**

~~~~text
5 mẹo giúp bạn ngủ ngon hơn mỗi đêm:
1.
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
Giảm ánh sáng xanh vào buổi tối
Ánh sáng xanh là một loại ánh sáng có bước sóng ngắn, nó thường được phát ra từ màn hình máy tính, tivi và các thiết bị điện tử khác. Ánh sáng này làm gián đoạn quá trình sản xuất melatonin – một hormone giúp cơ thể dễ đi vào giấc ngủ hơn.
Vì vậy, nếu bạn muốn dễ ngủ hơn thì hãy giảm ánh sáng xanh trên các thiết bị điện tử của mình nhé!
2. Tạo không gian ngủ thoải mái
Một phòng ngủ tối và yên tĩnh sẽ giúp bạn có được giấc ngủ sâu hơn.
~~~~

**1.3B (gốc):**

~~~~text
Thư giãn trước khi đi ngủ Giữ một tâm trí thư giãn là một trong những cách tốt nhất để có được giấc ngủ ngon. Bạn hãy thử tập các bài thiền hoặc thực hành yoga, đây là những phương pháp đơn giản nhưng rất hiệu quả cho giấc ngủ của bạn. Nếu bạn không có thời gian để làm việc này thì hãy nghe nhạc nhẹ nhàng. 2. Chuẩn bị giường ngủ phù hợp Một chiếc giường thoải mái và chăn gối sạch sẽ sẽ giúp bạn dễ dàng đi vào giấc ngủ hơn. Bạn nên chọn giường ngủ bằng gỗ, tránh dùng đệm quá dày vì nó có thể gây đau
~~~~

### 16. [pháp luật]

**Prompt:**

~~~~text
Theo Bộ luật Lao động năm 2019, người lao động có quyền
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
được trả lương trực tiếp, đầy đủ và đúng thời hạn. Tuy nhiên, trong trường hợp người sử dụng lao động chậm trả lương thì phải trả thêm một khoản tiền ít nhất bằng số tiền lãi của số tiền trả chậm tính theo lãi suất huy động tiền gửi có kỳ hạn 01 tháng do ngân hàng nơi người sử dụng lao động mở tài khoản trả lương cho người lao động công bố tại thời điểm trả lương.
2. Nghĩa vụ của người sử dụng lao động khi chậm trả lương
Người sử dụng lao động phải trả lương trực tiếp, đầy đủ và đúng thời hạn cho người
~~~~

**1.3B (gốc):**

~~~~text
đơn phương chấm dứt hợp đồng lao động mà không cần phải báo trước. Vậy, khi nào người lao động được đơn phương chấm dứt hợp đồng lao động? Hãy cùng Luật sư X tìm hiểu qua bài viết dưới đây. Căn cứ pháp lý: - Bộ luật Lao động năm 2019; - Nghị định 05/2015/NĐ-CP hướng dẫn Bộ luật Lao động năm 2019. Nội dung tư vấn 1. Quyền đơn phương chấm dứt hợp đồng lao động của người lao động Căn cứ theo Điều 38 Bộ luật Lao động năm 201
~~~~

### 17. [sức khoẻ]

**Prompt:**

~~~~text
Triệu chứng thường gặp của bệnh sốt xuất huyết gồm
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
:
- Sốt cao đột ngột từ 2 – 7 ngày, đau đầu, đau sau hốc mắt.
- Đau cơ, khớp, nhức hai hố mắt.
- Thường kèm theo cảm giác buồn nôn, nôn ói, hoặc phát ban trên da.
Sốt xuất huyết có nguy hiểm không?
Theo các bác sĩ chuyên khoa, sốt xuất huyết là căn bệnh truyền nhiễm cấp tính do virus Dengue gây ra và lây lan qua vết đốt của muỗi vằn. Sốt xuất huyết nếu không được điều trị kịp thời sẽ dẫn đến nhiều biến chứng nguy hiểm như:
- Sốc
~~~~

**1.3B (gốc):**

~~~~text
: Sốt cao đột ngột, ớn lạnh, đau đầu dữ dội, chân tay run rẩy, buồn nôn và nôn mửa. Một số trường hợp có thể bị chảy máu mũi, đi cầu phân đen hoặc ra máu. Ở trẻ nhỏ, các biểu hiện thường diễn biến nhanh hơn và nghiêm trọng hơn. Bệnh sốt xuất huyết được chia thành 2 loại là sốt xuất huyết Dengue và sốt xuất huyết dengue: Sốt xuất huyết Dengue là bệnh do virus Dengue gây ra. Bệnh lây truyền từ người sang người qua đường máu. Muỗi cái sau khi hút máu người bệnh
~~~~

### 18. [hỏi đáp]

**Prompt:**

~~~~text
Câu hỏi: Thủ đô của Nhật Bản là gì?
Trả lời:
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
Thủ đô của Nhật Bản là Tokyo.
Thủ đô của Nhật Bản là gì?
Thủ đô của Nhật Bản là Tokyo.
Thủ đô của Nhật Bản là gì?
Thủ đô của Nhật Bản là Tokyo.
Thủ đô của Nhật Bản là gì?
Thủ đô của Nhật Bản là Tokyo.
Thủ đô của Nhật Bản là gì?
Thủ đô của Nhật Bản là Tokyo.
Thủ đô của Nhật Bản là gì?
Thủ đô của Nhật Bản là Tokyo.
Thủ đô của Nhật Bản là gì?
Thủ đô của Nhật Bản là Tokyo.
Thủ đô
~~~~

**1.3B (gốc):**

~~~~text
Tokyo, Yokohama, Osaka. Câu hỏi: Tại sao các thành phố lớn ở Nhật Bản lại được gọi là Tokyo? Trả lời: Tokyo nằm ở phía Đông của đảo Honshu và có thủ đô là một trong những thành phố lớn nhất trên thế giới. Câu hỏi: Thủ đô của Nhật Bản là gì? Trả lời: Tokyo, Yokohama, Osaka. Câu hỏi: Tại sao các thành phố lớn ở Nhật Bản lại được gọi là Tokyo? Trả lời: Tokyo nằm ở phía Đông của đảo Honshu và có thủ đô là một trong những thành phố lớn nhất trên thế
~~~~

*Nhận xét:* 2,5B: đúng Tokyo nhưng lặp không dừng (việc của SFT).

### 19. [hỏi đáp]

**Prompt:**

~~~~text
Câu hỏi: Tại sao bầu trời có màu xanh?
Trả lời:
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
Ánh sáng mặt trời là ánh sáng trắng, gồm 7 màu cơ bản: đỏ, cam, vàng, lục, lam, chàm, tím. Ánh sáng mặt trời đi qua khí quyển (chứa nhiều bụi) sẽ bị phân tán thành nhiều màu sắc khác nhau. Khi bầu trời quang đãng, không có mây, mắt chúng ta nhìn thấy những đám mây trắng ở gần và các tia sáng xanh ở xa hơn.
Ánh sáng mặt trời gồm 7 màu cơ bản.
Khi ánh sáng mặt trời đi qua khí quyển (chứa nhiều bụi) sẽ bị phân tán thành nhiều màu sắc
~~~~

**1.3B (gốc):**

~~~~text
Bầu trời có màu xanh vì trong không khí có chứa một chất hóa học gọi là oxy. Oxy là một chất khí mà chúng ta hít thở hàng ngày và nó cung cấp cho các tế bào trong cơ thể chúng ta năng lượng để hoạt động. Khi không khí có chứa oxy, thì sẽ có màu xanh. Các nguyên tố khác như sắt, kẽm, đồng,… cũng có thể làm cho bầu trời có màu xanh nhưng những nguyên tố này chỉ chiếm một phần rất nhỏ trong không khí mà thôi. Và khi các nguyên tố này kết hợp với nhau sẽ tạo ra màu xanh. Bầu trời
~~~~

### 20. [hội thoại]

**Prompt:**

~~~~text
A: Chào bạn, hôm nay bạn có khỏe không?
B:
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
Tôi cũng vậy.
C: Hôm nay có ai đến thăm bạn không?
B: Có một người đàn ông, anh ấy bảo tôi gọi cho anh ấy vì anh ấy muốn mời tôi đi ăn tối.
C: Thật tuyệt! Chúng ta sẽ gặp nhau ở đó nhé!
B: Tôi chắc chắn rồi.
Hãy cùng xem thêm các mẫu câu tiếng Anh giao tiếp theo chủ đề khác trong video sau đây nhé!
~~~~

**1.3B (gốc):**

~~~~text
Tôi thấy khỏe. Xin cảm ơn anh đã cho biết về tình trạng của tôi. LM: Anh có khỏe không khi tôi hỏi là anh đang ở đâu?ZB: Tôi ở trên giường bệnh, và đang được điều trị tại một trung tâm phục hồi chức năng. LM: Anh có thể kể cho tôi nghe về việc anh được chẩn đoán ra sao?AfghanB: Lúc đó tôi cảm thấy rất đau đớn. Tôi nằm đó và không cử động được. Khi người ta gọi tên tôi thì tôi mới tỉnh lại. Tôi đã trải qua bốn lần phẫu thuật và một cuộc giải
~~~~

### 21. [toán]

**Prompt:**

~~~~text
Bài 1: 12 + 7 = 19
Bài 2: 25 + 18 = 43
Bài 3: 36 + 47 =
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
69
Bài 4: 55 + 10 = 65
Bài 5: 84 + 25 = 111
Bài 6: 96 + 48 = 142
Bài 7: 128 + 36 = 172
Bài 8: 144 + 76 = 216
Bài 9: 158 + 23 = 180
Bài 10: 160 + 12 = 182
Bài 11: 192 + 48 = 246
Bài 12: 176
~~~~

**1.3B (gốc):**

~~~~text
81achter) Vậy, để vẽ đường tròn chúng ta phải vẽ hai giao điểm của đường thẳng đi qua hai điểm đó và đường thẳng cắt hai đường thẳng đó.Bạn đang đọc: Đường tròn (Oy): Định nghĩa và các tính chất cơ bản Một số người chỉ thích ngồi vẽ vòng tròn xã giao, cho đó là những vòng tròn vô ích, to lớn để rồi ngăn cản sự phát triển của trẻ nhỏ những câu hỏi như :Tại sao lại không có vòng tròn trên sân chơi ?Tại sao trẻ con phải đếm số ?Tại sao phải dạy trẻ em đếm đúng mực ?
~~~~

*Nhận xét:* 2,5B: đúng khuôn nhưng 36+47=69 SAI (đúng 83); các bài sau phần lớn sai.

### 22. [toán]

**Prompt:**

~~~~text
Một cửa hàng có 120 kg gạo, đã bán 45 kg. Hỏi cửa hàng còn lại bao nhiêu ki-lô-gam gạo?
Bài giải:
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
Câu hỏi 2:
Đắp một ụ đất cao 5 cm trên đường, rồi dùng phấn màu vẽ hình chữ nhật có chiều dài 45 cm và chiều rộng 30 cm. Tính chu vi hình chữ nhật đó.
Bài giải:
Chu vi hình chữ nhật là:
45 x 30 = 1350 (cm)
Đáp số: 1350 cm
Bài tập 5:
Người ta lấy 2/3 của một tảng đá rồi đem nung cho nó nóng lên. Hỏi thể tích của tảng đá sau khi nung tăng thêm bao nhiêu phần trăm so
~~~~

**1.3B (gốc):**

~~~~text
= 25,6034 (kg) x 0,0012 x 0,0012 = 2,7819. Cửa hàng còn lại 120 kg gạo, đã bán 45 kg. Hỏi cửa hàng còn lại bao nhiêu ki-lô-gam gạo?
~~~~

*Nhận xét:* Cả hai không giải bài; 2,5B nhảy sang bài khác, tính chu vi sai (45×30).

### 23. [toán]

**Prompt:**

~~~~text
Giải phương trình 2x + 6 = 0.
Lời giải:
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
Phương trình có nghiệm x = – 3/2.
Vậy phương trình có hai nghiệm – 3/2 và x = – 3/2.
Bài 5. Cho phương trình x2 – (m+1)x + m = 0 có nghiệm là x = – 1, m.
a) Tìm điều kiện của m để phương trình có hai nghiệm trái dấu.
b) Tìm m để phương trình có hai nghiệm cùng dấu.
Lời giải:
Phương trình có nghiệm x = – 1.
Nghiệm x = – 1 thỏa mãn điều kiện m ≥ 2
~~~~

**1.3B (gốc):**

~~~~text
(1) Tìm giá trị của a, b, c và d. Ngoài ra, còn có các trò chơi như:Xem thêm: Chia sẻ cách bắt cầu tài xỉu chuẩn nhất hiện nay Mini game: Tài Xỉu, Bầu Cua, Cao Thấp, Pokémon, Vòng quay may mắn….Game bài đổi thưởng: Tiến lên miền Nam, Poker, Mậu binh, Sâm lốc, Tá lả ….Game Slots: Sơn Tinh Thủy Tinh, Ăn Khế Trả Vàng, Mỹ Nữ, Kim Cương, Phúc Lộc Thọ…Hướng dẫn liên hệ với cổng game Zindo
~~~~

*Nhận xét:* 2,5B: x = −3/2 SAI (đúng −3).

### 24. [dịch]

**Prompt:**

~~~~text
Tiếng Anh: Good morning.
Tiếng Việt: Chào buổi sáng.
Tiếng Anh: I love my family.
Tiếng Việt:
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
Tôi yêu gia đình tôi.
Tiếng Anh: My sister is a very good student.
Tiếng Việt: Em gái tôi là một học sinh giỏi.
Tiếng Anh: Please use this phone number as soon as possible.
Tiếng Việt: Xin vui lòng sử dụng số điện thoại này càng sớm càng tốt.
Tiếng Anh: Thank you for your inquiry.
Tiếng Việt: Cảm ơn bạn đã liên hệ với chúng tôi.
Tiếng Anh: I am sorry to hear that.
Tiếng Việt: Tôi xin lỗi vì
~~~~

**1.3B (gốc):**

~~~~text
Tôi yêu gia đình tôi. I love my family Tiếng Anh: Tôi yêu gia đình của mình. I love my family Tiếng Anh: Tôi yêu gia đình của mình. I love my family Tiếng Anh: Tôi yêu gia đình của mình. I love my family Tiếng Anh: Tôi yêu gia đình của mình. I love my family Tiếng Anh: Tôi yêu gia đình của mình. I love my family Tiếng Anh: Tôi yêu gia đình của mình. I love my family Tiếng Anh: Tôi yêu gia đình của mình. I love my family Tiếng Anh: Tôi
~~~~

*Nhận xét:* 2,5B: dịch đúng + tự sinh thêm cặp dịch đúng; 1.3B lặp.

### 25. [english]

**Prompt:**

~~~~text
The Industrial Revolution began in Britain in the late 18th century and
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
spread throughout Europe. During the same period, America was experiencing a revolution of its own that also began in Britain. This revolution was known as the Industrial Revolution because it was a time when people were creating things they had never seen before. This was during the 19th century, which is often referred to as the age of steam and the age of machines.
The Industrial Revolution
~~~~

**1.3B (gốc):**

~~~~text
it is now one of the most important economic factors in Europe. It is the key to managing the growth of a country, which is also an important part of the world economy. The problem is that the economy is prone to losses and lack of infrastructure, but that the economic factor is not the only reason for this. The growth rate is relatively low, but it is also very high, with
~~~~

### 26. [english]

**Prompt:**

~~~~text
Photosynthesis is the process by which plants
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
convert carbon dioxide and water into sugars and oxygen, respectively. The process begins when sunlight enters the leaves through a process called photosynthesis. This reaction takes place between chlorophyll and light.
The first step in the process is the reaction between carbon dioxide and water. Carbon dioxide is the main component of plant tissue, while water is a key component of the plant’s structure. As carbon dioxide is taken
~~~~

**1.3B (gốc):**

~~~~text
react to the sunlight to receive nutrients. The ideal plant’s growth factor is between 30 to 40%. This means that the plants need a high temperature and low moisture conditions to grow. At this point, increased heat and lighting at the same time can reduce the amount of sunlight needed for these plants. This means that the plant will receive enough nutri
~~~~

*Nhận xét:* 2,5B: đúng ý (CO2 + nước → đường + oxy); 1.3B sai.

### 27. [code]

**Prompt:**

~~~~text
def fibonacci(n):
    """Trả về số Fibonacci thứ n."""
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
Fibonacci = [1];
for i in range (0, n + 1):
Fibonacci.append(Fibonacci[-1] + Fibonacci[-2])
return Fibonacci[n]
``````

Output:
``````3
``````

This code example uses the `fibonacci` function to calculate the nth Fibonacci number.

### Usage:

``````def fib(n):
if n
~~~~

**1.3B (gốc):**

~~~~text
""""Lưu ý: Nếu giá trị của hàm (x) là số dương, nó sẽ trở thành 0 hoặc 1 và ngược lại. Khi đó, số dương (y) là số âm.""""Một số ví dụ: y= 2.76185 x 10 = 2.7618500."""""Ví dụ: y= 2.7618500.0 = 2.7618500.0.""""Lưu ý: Nếu giá trị của hàm
~~~~

*Nhận xét:* 2,5B: đúng dạng code nhưng logic sai (khởi tạo [1], lấy [-2] khi mới 1 phần tử).

### 28. [code]

**Prompt:**

~~~~text
# Đọc file CSV và tính trung bình cột 'gia'
import pandas as pd
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
from sklearn.linear_model import LinearRegression
import csv
df = pd.read_csv("/home/pi/Downloads/20140817-a0394369_1_0_000000_gpic_dna.csv", header=None)
#print "Reading CSV"
#df = pd.read_csv("/home/pi/Download
~~~~

**1.3B (gốc):**

~~~~text
, write in device. #use the selected textbox to rewrite it. #enable the Variable Function(); #use the selected textbox to reference to the fields (with the properties of the fields); #use the selected textbox to compare the values with the value which is given. #use the selected textbox to fix the value of the values. #use the selected text
~~~~

### 29. [code]

**Prompt:**

~~~~text
function isPrime(n) {
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
for(var i = 2; i <= n; i++) {
if(n % i == 0) {
return true;
}
}
return false;
}
return null;
}
/*
Function to return the count of odd numbers from 1 to n
Using Recursion
*/
public static int countOdd(int n) {
if (n == 0 || n == 1) {
return 0;
}

if
~~~~

**1.3B (gốc):**

~~~~text
(n); } return NP_NUM. - As a result, the final expression of this formula was first analyzed into a kind of perfectly completely and then sampled to some of the same kind of principle. - Những con cá nhỏ có thể cung cấp cho bạn từ 20 đến 50 điểm, trong khi những con cá lớn hơn có thể mang lại cho bạn từ 100 đến 200 điểm. - Tuy nhiên, cách chơi bài bip này sẽ
~~~~

*Nhận xét:* 2,5B: đúng cú pháp nhưng logic ngược (tìm được ước thì trả true).

### 30. [nhớ]

**Prompt:**

~~~~text
Anh Minh có ba con: Hoa 12 tuổi, Nam 9 tuổi và Bình 5 tuổi. Cả nhà sống ở Huế. Hằng ngày anh đi làm bằng xe máy, chị Thu ở nhà bán hàng tạp hoá. Cuối tuần cả nhà thường về quê thăm ông bà. Người con lớn nhất của anh Minh tên là
~~~~

**2,5B (mốc 20 tỷ):**

~~~~text
Minh, năm nay 20 tuổi, đang học tại Hà Nội. Người con thứ hai tên Nam 16 tuổi, đã đi làm. Người con út tên Bình 5 tuổi, đang học mẫu giáo. Theo lời chị Thu, con gái lớn của anh Minh có tính tình trầm mặc, ít nói và không thích giao tiếp với ai. Con gái nhỏ của anh Minh cũng khá nhút nhát và ngại ngùng khi gặp người lạ. “Bình thường trong nhà tôi chỉ có hai cháu là hay chơi cùng nhau nhưng gần đây chúng không còn chơi chung nữa”, chị Thu chia sẻ. Theo chị Thu
~~~~

**1.3B (gốc):**

~~~~text
Hoa (12 tuổi) đang học lớp 4 Trường Tiểu học Trần Quốc Toản. Nam và Bình cùng học một trường mẫu giáo với nhau. Cả hai đều rất ngoan và thương mẹ. Hằng ngày, ngoài giờ lên lớp, các con còn giúp mẹ bán hàng tạp hoá để phụ giúp gia đình. Anh Minh cho biết, Hoa là người ít nói, không thích nói chuyện với mọi người xung quanh. Tuy nhiên, em lại có khả năng nhớ bài rất tốt. Theo anh Minh, để giúp con học giỏi và hiểu bài nhanh, hằng ngày anh thường cho các con làm bài tập về
~~~~

*Nhận xét:* 2,5B: SAI (con lớn nhất là Hoa 12 tuổi, model viết 'Minh, 20 tuổi'); 1.3B đúng.

## 3. Thử trong UI chat (M4, cổng 8797)

**Hỏi:** Giới thiệu ngắn về Vịnh Hạ Long.

**2,5B (nền):** viết như bài bách khoa — vị trí (Đông Bắc, giáp Cát Bà, bờ biển 120 km), diện tích 1.553 km², 1.969 hòn đảo, rồi sang mục "1.2. Điều kiện tự nhiên" (khí hậu, nhiệt độ 24–25 °C, lượng mưa 2.000–2.200 mm) và không dừng. Số liệu chính khớp số phổ biến; dạng trả lời là của model nền (chưa SFT).
