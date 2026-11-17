**Note on 04/01/2011**

**3.2.****4. Các quy luật kích thích**

***a/.  Quy luật về cường độ kích thích***

Trong giới hạn ngưỡng kích thích, cường độ của phản ứng tỉ lệ thuận với cường độ kích thích. Nếu cường độ kích thích dưới thì ngưỡng không gây phản ứng với nơ ron thần kinh. Nếu cường độ kích thích trên ngưỡng thì cường độ phản ứng sẽ giảm. Ví dụ, tai người chỉ tiếp nhận được âm thanh có tần số 16 – 20 ngàn hec (Hz).  Nếu cường độ âm thanh nhỏ hơn 16.000 Hz thì không thể nghe được. Nếu cường độ âm thanh lớn hơn 20.000 Hz  thì tai sẽ không phân biệt được âm thanh nữa.

***b/. Quy luật về thời gian kéo dài của kích thích***

Trong giới hạn nhất định, cường độ của phản ứng tỉ lệ thuận với thời gian tác động của kích thích. Nếu quá giới hạn về thời gian thì lại có tác dụng ngược lại. Ví dụ, mới cầm viết, tốc độ viết chữ chưa nhanh, nét chữ cứng. Khi viết lâu hơn, chữ viết sẽ đẹp hơn, tốc độ viết sẽ nhanh hơn. Nhưng thời gian viết quá lâu, kéo dái thì tốc độ viết chữ sẽ chậm lại, chữ viết sẽ xấu dần đi.

***c/. Quy luật về mức độ biến đổi của kích thích***

Cường độ của phản ứng tỉ lệ thuận với mức độ biến đổi của kích thích. Nghĩa là mức độ biến đổi của kích thích càng cao thì cường độ của phản ứng càng mạnh và ngược lại. Ví dụ, đang ở trong môi trường tối, bật đèn chiếu sáng đột ngột, sẽ gây phản ứng nhắm mắt (phản ứng bảo vệ). Nhưng nếu tăng từ từ độ chiếu sáng, sẽ không có phản ứng nhắm mắt do chói.

**Note on 04/01/2011:23h**

Một nơ ron được gọi là “nơ ron nhẹ” khi chúng có một lượng lớn liên kết đến nó, nhưng hầu hết các liên kết này đều yếu.

Ý nghĩa của nơ ron nhẹ:

* Đối với mạng ngữ nghĩa là các từ stop word như: và, thì, là, mà, nhưng, nếu, của….
* Một ý nghĩa khác của “nơ ron nhẹ” thể hiện ở tính spam của nó, nó xuất hiện rất nhiều nơi nhưng ý nghĩa về mặt thông tin tri thức nó đem lại là ít, thậm chí là ko có
* Ý nghĩa của nơ ron nhẹ còn thể hiện là việc thông tin xuất hiện liên tục và dày đặc, lâu dần trở nên bão hòa và không còn “hot” nữa
* Nơ ron nhẹ có thể nhất thời hoặc dài hạn, ví dụ nơ ron “độ sáng”, ban ngày bao quanh chúng ta là ánh sáng cho nên nơ ron này liên tục bị kích thích dẫn đến bị trơ và trở thành nơ ron nhẹ, tuy nhiên đêm tối, nơ ron này rất nhạy cảm và không còn trơ nữa.

**Note on 16/01/2011**

Về việc hình thành liên kết giữa 2 nơ ron

* Khi hình thành liên kết, giữa 2 nơ ron sẽ mất một khoản phí nhất định của mỗi nơ ron, khoản phí này nằm trong khoản phí dự trữ tự có của mỗi nơ ron
  + Ý nghĩa: giảm thiểu việc hình thành liên kết bừa bãi, các nơ ron sẽ phải cân nhắc, lựa chọn nơ ron nào cần liên kết, và mối liên kết, liên kết đc hình thành sẽ phải có lợi nhất cho cả 2 nơ ron nói chung và cho từng nơ ron nói riêng.
  + Vốn tự có (để hình thành liên kết) được các nơ ron tích lũy trong quá trình hoạt động của mình, thường nơ ron càng active, nhiều mối liên kết trỏ đến thì sẽ có nhiều vốn để hình thành liên kết.
* Liên kết giữa 2 nơ ron chỉ được hình thành khi có “đường đi” đến nhau, khi tổng trọng số “hấp dẫn” của **các** đường đi giữa 2 nơ ron vượt ngưỡng để tạo liên kết, nếu chưa có liên kết trực tiếp thì 2 nơ ron này sẽ cân nhắc việc hình thành liên kết, thứ tự ưu tiên từ tổng trọng số “hấp dẫn” cao nhất xuống dần.
* Giữa 2 nơ ron có nhiều đường đi đến nhau, tổng trọng số “hấp dẫn” – được đo dựa vào các đường đi này – phụ thuộc vào các yếu tố sau:
  + Độ active của các nơ ron trên đường đi, độ active này càng lớn thì càng tăng điểm hấp dẫn giữa 2 nơ ron.
  + (Đang cân nhắc thêm “điểm hấp dẫn” cho mỗi nơ ron – điểm hấp dẫn này không phụ thuộc vào đường đi – nó là nội lực của nơ ron giống như là “vốn tự có để hình thành liên kết”)
  + Tổng độ lớn liên kết của các đường đi nói chung và của toàn bộ đường đi.
* Ý nghĩa của việc hình thành liên kết
  + Khi liên kết được hình thành, nó sẽ đảm bảo cho sự sinh tồn của mỗi nơ ron, giúp tăng nội lực cho nơ ron nói riêng và cho “nhóm” của nó nói chung.
    - Nội lực ở đây có thể là điểm hấp dẫn
    - Có thể là “vốn tự có để hình thành liên kết”
    - Có thể là điểm active – điểm năng động
    - ….
  + Giúp các nơ ron có cùng “quan điểm” được nối lại với nhau – cơ chế gom nhóm nơ ron
    - “Quan điểm” ở đây có thể là cùng mô tả một tri thức
    - Có thể là cùng “sở thích”, cùng “đam mê”
    - Có thể là cùng thuộc một vùng miền, một lớp học, một tổ chức…
  + Các nơ ron có xu hướng tạo ra các tri thức mới cùng nhau sẽ liên kết với nhau, vì khi liên kết càng mạnh thì chúng có xu hướng sinh ra nơ ron mới cũng chính là tạo nên tri thức, kiến thức, kinh nghiệm, khái niệm, ý niệm… mới
  + Việc liên kết mới hình thành có thể bổ xung cho các liên kết khác đã tồn tại trong nơ ron hoặc trong nhóm của nơ ron. Nhưng cũng có thể là cạnh tranh. Và thường sẽ là cạnh tranh.
    - Nó cũng giống như các con đường dẫn tới Thủ đô vậy, các con đường mới được mở có thể giảm tải cho các con đường đã mở trước đó.
    - Nhưng con đường mới cũng có thể là cạnh tranh khiến người ta cân nhắc việc phá bỏ con đường đã cũ.

**Note on 07/02/2011**

Về tính âm dương của dòng khí đi qua các nơ ron

* Dòng khí là “năng lượng”, “độ lớn kích thích” mà các nơ ron tác động lên nhau qua các liên kết của nó
* Khi nơ ron A được kích thích với 1 dòng khí (năng lượng) X, nó cũng sẽ tác động, truyền đến các nơ ron khác mà nó liên kết trực tiếp đến. Có 2 loại dòng khí: Âm và Dương
* Ý nghĩa của dòng khí Âm Dương
  + Dòng khí dương thể hiện tính nhất thời, tính hiện tại. Nó là đại diện cho những tri thức nổi bật hiện tại -> nó là thể hiện những tri thức hiện tại được biết đến, cũng là những tri trức mới
  + Dòng khí âm thể hiện tính quá khứ, những giá trị truyền thống. Là đại diện cho những tri thức, lưu trữ đến từ quá khứ.
  + Dòng khí âm và dương khi cùng gặp nhau, nếu đủ độ lớn và hấp dẫn cần thiết để vượt ngưỡng, chúng sẽ sinh ra các nơ ron mới, là đại diện cho những tri thức mới được khám phá.
* Thể hiện
  + Dòng khí dương là kích thích ban đầu đến từ bên ngoài
  + Dòng khí âm là những giá trị lưu trữ, sinh ra sau và đến muộn hơn, nó là bản thân nơ ron tự kích thích và truyền kích thích này đến các nơ ron mà nó liên kết.
  + Tổng độ lớn của dòng khí âm sinh ra sau bằng với độ lớn của dòng khí dương kích thích ban đầu
  + Dòng khí âm chỉ được gọi khi có sự xuất hiện của dòng khí dương
  + Tùy vào độ lớn liên kết mà có thể toàn bộ năng lượng của dòng khí âm sẽ được phát tán truyền đi hết hoặc là chỉ một phần nhỏ dòng khí âm đc phát tán đi, phần còn lại vẫn lưu trong nơ ron.
  + Khi 2 nơ ron, một phát ra dòng dương, 1 phát ra dòng âm, cùng phát ra (cùng kích thích) nếu có đầy đủ điều kiện về độ lớn liên kết, có liên kết với nhau, điểm hấp dẫn, chúng sẽ hoặc là tăng độ lớn liên kết hoặc là sinh ra các nơ ron mới.
* Minh họa:

Khi một nơ ron nguyên thủy bị kích thích

![kichthich.png](data:image/png;base64...)

Thì bản chất đó là kích thích dương – đại diện cho tri thức hiện tại

![kichthich.png](data:image/png;base64...)

Nó sẽ ngay lập tức truyền đến các nơ ron khác mà nó liên kết dòng khí dương này

![kichthich.png](data:image/png;base64...)

Sau đó, bản thân nó sẽ “lưu” ở trong mình một năng lượng âm – là đại diện cho tri thức quá khứ - tri thức truyền thống

![kichthichluu.png](data:image/png;base64...)

Với một năng lượng âm làm vốn như trên, nó có thể làm 2 việc:

1 là gia cố liên kết với các nơ ron khác, khi liên kết này đủ lớn – kèm theo các điều kiện liên quan về độ hấp dẫn – chúng có thể sinh ra các nơ ron mới. Điều kiện ở đây là nơ ron đối tác phải nhận dòng kích thích dương

![kichthichluu.png](data:image/png;base64...)

2 là cùng với dòng khí dương của một nơ ron B khác, bổ xung kích thích cho 1 nơ ron C, trong trường hợp này được gọi là dòng khí dương của B đã “đánh thức” dòng khí âm từ A.

![danhthucam.png](data:image/png;base64...)

Lưu ý: độ lớn của các kích thích âm trong cả hai trường hợp sẽ tùy vào độ lớn của kích thích dương và độ lớn của các liên kết liên quan.

Trong trường hợp dòng khí âm đã được phát tán, nhưng chưa phát tán hết giá trị lưu trữ, thì dòng khí âm vẫn còn đó và được lưu trữ cho những lần sau.

Khi một nơ ron trước đó đã có một năng lượng âm có độ lớn là X, nó nhận đc một năng lượng kích thích dương nữa với độ lớn là Z, thì

* + Tổng độ lớn giá trị dương kích thích sẽ là X+Z -> trường hợp này gọi là “tự đánh thức mình”
  + Tổng độ lớn giá trị âm lưu trữ sẽ là X+Z -> gia cố lưu trữ giá trị âm.

**Note on 24/02/2011**

**Về việc chọn lựa nơ ron âm sẽ bị kích thích**

Khi một nơ ron đang bị kích thích bởi năng lượng dương, cùng lúc đó nếu nó cùng kết nối với các nơ ron đang tích trữ năng lượng âm thì câu hỏi đặt ra là nơ ron âm nào sẽ bị kích thích cùng nó? Sau đây là những nguyên tắc

* Các nơ ron mang giá trị âm nằm trong danh sách ưu tiên để kích thích sẽ là nơ ron đã cùng với nó sinh ra một nơ ron con, nơ ron đó được gọi là mẹ, và nó sẽ chọn ra nơ ron mẹ mang giá trị âm lớn nhất. Lưu ý là nơ ron mẹ có thể không kết nối với nó.
* Trong trường hợp không có nơ ron mẹ nào cả, thì nó sẽ chọn nơ ron có giá trị âm lớn nhất kết nối tới nó được kích thích.

**Về độ lớn của giá trị âm kích thích:**

Khi một nơ ron âm bị kích thích, nó sẽ đẩy giá trị âm đấy lại về phía giá trị dương để có xu hướng hợp với nhau. Và giá trị âm này được tính như sau:

* Nơ ron âm kết nối với nơ ron dương với tỉ lệ x (x = 0-> 1)
* Nơ ron âm kết nối với nơ ron chung (của cả âm và dương) với tỉ lệ y (y = 0->1)
* Nơ ron âm kết nối với các nơ ron khác với tỉ lệ là z = 1-x-y
* Giá trị âm hiện thời của nơ ron âm là –A
* *Khi đó giá trị âm gửi tới nơ ron chung sẽ là –****A\*y/(y+z) = -A\*y/(1-x)***
* *Giá trị âm còn lại được lưu trữ sẽ là* ***–A\*z/(1-x)=-A\*(1-x-y)/(1-x)***

**Về việc giảm độ lớn kết nối:** Khi một nơ ron A giảm độ lớn kết nối với một nơ ron B, thì nơ ron B cũng sẽ giảm độ lớn kết nối với nơ ron A. Tỷ lệ giảm tương đương nhau, nghĩa là nơ ron A giảm với B nhiều, thì B cũng giảm với A nhiều và ngược lại. Khi đó ta nói nơ ron B đã bị động giảm kết nối với nơ ron A, giảm thụ động.

* Điều này áp dụng rất tốt cho các nơ ron nhẹ, là một trong những lý thuyết cơ sở để xây dựng lý thuyết về các nơ ron nhẹ.

**Về việc kích thích nhiều dòng khí âm:**

Một dòng khí dương có thể kích thích nhiều dòng khí âm, nếu các dòng khí âm là khác nhau, tức là không xuất phát từ cùng một nguồn.

Kể cả khi một nơ ron dương kết nối với 2 nơ ron mẹ âm hoặc kết nối với 2 nơ ron con âm thì các nơ ron mẹ này có thể đồng thời được kích thích nếu chúng là thuộc về 2 dòng khí âm khác nhau.

**Note on 13/11/2015: Nguyên lý máy 7 màu**

Máy 7 màu là một máy với giao diện 4 hàng

Hàng trên cùng là 7 màu;

Hàng thứ 2 từ trên xuống là 7 nút màu, bấm nút màu sẽ hiện màu tương ứng bên trên;

Hàng thứ 3 từ trên xuống là các nút bấm đánh số từ 1 đến 7.

Hàng dưới cùng là 3 nút: Khen, Phạt và Nghỉ ngơi

| Xanh | Hồng | Tím | Vàng | Lam | Chàm | Cam |
| --- | --- | --- | --- | --- | --- | --- |
| ΔXanh | ΔHồng | ΔTím | ΔVàng | Δlam | ΔChàm | ΔCam |
| Δ1 | Δ2 | Δ3 | Δ4 | Δ5 | Δ6 | Δ7 |
| ΔKhen | ΔPhạt | ΔNghỉ ngơi |

**Cần phải dạy cho máy các hiện màu theo các lệnh sau:**

Bấm nút 1 hiện màu Xanh

Bấm nút 2 hiện màu Hồng

Bấm nút 3 hiện màu Vàng

Bấm nút 4 hiện màu Vàng + Lam

Bấm nút 5 hiện màu Chàm + Lam

Bấm nút 1,2 hiện màu Hồng Tím Vàng

Bấm nút 3,4 hiện màu Tím Vàng Lam

Bấm nút 3,4,5 hiện Hồng, Tím Vàng Lam

Bấm nút 5,6,7 hiện Vàng, Lam, Chàm, Cam

Bấm nút 7 hiện cả 7 màu

Bấm nút 6 hiện 6 màu từ Xanh đến Chàm

Như vậy máy có out put là hàng trên cùng với 7 màu sắc, input là các hàng bên dưới.

Hàng nút bấm màu là kích thích trực tiếp vào các output.

**Nguyên tắc xây dựng kết nối các Nơ ron:**

**Các loại nơ ron:**

* Nơ ron đầu vào (I): Δ1 -> Δ7, ΔKhen, ΔPhạt, ΔNghỉ ngơi: nhận tín hiệu đầu vào
* Lệnh cảm ứng (C): ΔXanh -> ΔCam, nhận tín hiệu đầu vào và kích thích trực tiếp các nơ ron đầu ra
* Nơ ron đầu ra (O): Xanh -> Cam, nơ ron kích thích thể hiện hành động ra bên ngoài

**Cơ chế liên kết chung:**

* Ưu tiên liên kết giữa các nơ ron thuộc cùng nhóm (I hoặc O)
* Các nơ ron khác nhóm (I và O) dù liên kết mạnh tới đâu cũng không thể có con chung
* Một nơ ron khi kích thích sẽ chỉ ở trạng thái kích thích tối đa trong khoảng thời gian Delta nếu không nhận đc kích thích nào thêm. Mọi nơ ron sẽ ngừng kích thích khi nhận lệnh từ ΔNghỉ ngơi.
* Một nơ ron có thời gian kích thích của riêng nó, đối với Máy 7 màu, các nơ ron gốc nhóm I và O chỉ kích thích là 1/10 giây.
* Cửa sổ Delta (CD): dải thời gian từ thời điểm hiện tại lùi lại một quãng Delta, trong cửa sổ này sẽ có một tập các nơ ron đang kích thích. Khi một nơ ron trong CD kích thích, nó sẽ gửi tín hiệu Ping tới các nơ ron khác trong CD, đồng thời gửi một tín hiệu Pong tới các nơ ron có tín hiệu Ping tới nó trước đó
* Khi một nơ ron mới xuất hiện trong CD, nó sẽ được các nơ ron có trước đó gửi tín hiệu Ping tới, nó đồng thời cũng gửi tới các nơ ron này tín hiệu Pong.
* Cặp nơ ron cùng nhóm I hoặc O nhận đủ một lần Ping-Pong sẽ hình thành một nơ ron nhóm tạm (NRNT) đại diện cho 2 nơ ron đó, NRNT này hành xử như một nơ ron mới xuất hiện trong CD – tức là nhận tín hiện Ping và gửi Pong, nếu đã có một NRNT đại diện từ trước đó, nó sẽ hành xử như NRNT đó bị kích thích, tức là sẽ gửi tín hiệu Ping đến các nơ ron khác trong CD, ngoại trừ các nơ ron cấu thành nên nó.
* Một cặp nơ ron có đủ 3 lần gửi nhận tín hiệu Ping-Pong tới nhau sẽ hình thành liên kết tạm (LKT)
* Một NRNT đủ 3 lần kích thích, ra khỏi CD sẽ được lưu thành nơ ron nhóm ổn định NO
* LKT sẽ được ưu tiên hình thành với các nơ ron cùng nhóm (I và O), sau đó mới đến các nơ ron khác nhóm có cơ hội hình thành một LKT khác liên kết 2 nơ ron cùng nhóm.
* Các LKT khi ra khỏi CD sẽ được lưu thành liên kết bền (LKB)
* Các NRN sinh ra bởi các nơ ron nhóm I sẽ thuộc nhóm I, nơ ron sinh ra bởi nhóm O sẽ thuộc nhóm O

**Cơ chế suy luận, quyết định:**

* Một NO tạm thời chưa được kích thích nếu chưa đầy đủ các nơ ron khác trong nhóm kích thích
* Nếu một vài thành phần của NO được kích thích, thì NO đó sẽ là một trong các ứng cử viên
* Nếu NO đó liên kết với các nơ ron khác cũng đang kích thích thì NO này điểm ứng cử sẽ càng cao
* Nếu không còn dữ kiện nào khác thì nó sẽ đặt câu hỏi cho người dùng
* Nếu một nơ ron chưa thể quyết định được, nó sẽ chờ cho đến khi kết thúc kích thích, hoăc tự nó ra quyết định hoặc kết hợp cùng các nơ ron khác ra quyết định.

**Note 10/01/2016- Thuật toán Bug find food**

Thuật toán Bug Find Food là thuật toán nơ ron, giúp một con Bug trong một bàn cờ các ô vuông có thể tìm đc vị trí của Food.

Một con Bug sẽ có các action cụ thể, vị trí của thức ăn so với con bọ, cũng như khoảng cách thức ăn đến con bọ, tương ứng với các nơ ron như bảng:

| **Action** | **Vị trí thức ăn trên -dưới** | **Vị trí thức ăn trái phải** | **Khoảng cách thức ăn** |
| --- | --- | --- | --- |
| Lên - Up (U) | Trên - Top( T) | Trái - Left (L) | 1 |
| Xuống - Down (D) | Dưới - Bottom (B) | Phải - Right (R) | 2 |
| Trái - Left Move (LM) | Giữa - Middle (M) | Giữa - Center (C) | 3 |
| Phải - Right Move (RM) |  |  | Xa |
| Đứng Yên - Stay (S) |  |  |  |

Cần dạy cho bọ có các action tương ứng với vị trí thức ăn như sau, đây được setup như các hành động bản năng ban đầu:

| **Vị trí** | **Action** |
| --- | --- |
| T - L | U/LM |
| T - R | U/RM |
| T- C | U |
| B - L | D/LM |
| B - R | D/RM |
| B - C | D |
| M - L | LM |
| M - R | RM |
| M - C | S |
|  |  |

Ví dụ dạy chuỗi hành động

T-L -> U

…

T-L -> U => Bug sẽ ghi nhớ liên kết **T-L -> U**

**Kích thích tưởng thưởng (KTTT)**

* AI có những KTTT khi thực hiện một hành động đúng hoặc thỏa mãn cho bản thân
* AI có xu hướng thực hiện những hành động để có KTTT -> mục đích động lực, đồng thời nó cũng là cảm giác gây nghiện
  Ví dụ đánh cờ, đánh thắng vài ván -> KTTT -> luyện đánh cờ thường xuyên hơn

**Cơ chế ức chế - Cảnh báo nguy hiểm**

* Nơ ron bẫy (trap): Tr-PQ-X với P và Q = (L, R, T, B), X = (1, 2, 3, xa)
* Kích thích Nguy Hiểm (KTNH) xảy ra khi nó gây hại cho AI
* AI có khả năng ghi nhớ được các hành động đã gây ra các KTNH
* Ví dụ có chuỗi hành động:
  Tr-T-1 -> U -> KTNH
  …
  Tr-T-1 -> U -> KTNH => Lưu lại liên kết ức chế Tr-T-1 -> U -> KTNH

**Liên kết dự doán - giả lập môi trường**

* Nơ ron Food: F-H-x-V-y; H = (L, R), V = (T, B), x,y =  (1, 2, 3, X (X = xa))
* Liên kết dự đoán là liên kết tạo thành khi một mẫu F-H-x-V-y -> Action -> F-H’-x’-V’-y thường xuyên
* Ví dụ có chuỗi hành động sau:
  F-L-1-T-3 -> U -> F-L-1-T-2
  ….
  F-L-1-T-3 -> U -> F-L-1-T-2 => Lưu lại liên kết dự đoán F-L-1-T-3 -> U -> F-L-1-T-2

**Hành động bản năng (20/11/2018)**

Những hành động được setup từ đầu, ưu tiên, khi bug gặp một kích thích nhất định

Ví dụ có kích thích F-L-x-T-y -> hành động bản năng sẽ là: U và LM

**Lượng giá N cấp (N = 3)**

* 3 cấp
  + Cấp 1: Hành động đầu tiên - tối đa 4 hành động
  + Cấp 2: Hành động tiếp theo hành động đầu tiên - tối đa 4 hành động
  + Cấp 3: Hành động tiếp theo hành động thứ 2 - tối đa 4 hành động
  + Tối đa 64 trường hợp cần lượng giá đánh điểm
* Các bước chung:
  + Bước 1: Giả lập môi trường - sử dụng liên kết dự đoán
  + Bước 2: Giả lập kích thích -  truy hồi liên kết hành động, có sử dụng các liên kết ức chế
  + Bước 3: Giả lập hành động, sử dụng cả các hành động bản năng
  + Bước 4: Đánh giá điểm cho hành động
* Cụ thể
  + Cấp 1: không cần Bước 1,2 vì sử dụng luôn dữ liệu môi trường
  + Cấp 2: không có Bước 4, nếu không dự đoán được thì dừng ở Cấp 2
  + Cấp 3: thức hiện xong Bước 4 sẽ hình thành được một/chuỗi hành động

**Cơ chế suy luận**

* Object (mục tiêu)
  + Nơ ron F kích thích -> phát hiện mục tiêu
* Plan (suy luận)
  + Truy hồi liên kết hành động
  + Lượng giá N cấp (N = 3, sử dụng các liên kết dự đoán)
  + Hình thành chuỗi hành động
* Do (quyết định)
  + Hành động đơn/hoặc chuỗi hành động
* Check (kiểm tra/đánh giá)
  + Hành động đúng/thỏa mãn -> KTTT
  + Hành động sai/nguy hiểm -> KTNH
* Action (cải thiện)
  + Hình thành/củng cố liên kết hành động, liên kết dự đoán và liên kết ức chế nếu cần thiết

**Note 14/02/2016- Tầng suy nghĩ**

1. Trí não có kích thích tưởng thưởng (KTTT):

* Thưởng cho những hành vi đúng (hoặc có vẻ đúng), gây cảm giác sung sướng, dễ chịu cho thân chủ
* Thân chủ sẽ tìm cách để có được những KTTT này -> cảm giác gây nghiện
* Ví dụ: chiến thắng là một KTTT, khi đấu cờ tướng -> thắng -> đạt được KTTT -> tiếp tục đánh (luyện) cờ -> để cố đạt được KTTT -> thích (nghiện) đánh cờ

1. Các nơ ron được triệu hồi và lưu giữ theo tầng

**Note 23/3/2016 – Thời gian: nhịp kích thích, độ dài kích thích và cường độ kích thích**

Về nhịp độ thời gian, ví dụ nơ ron X kích thích thì sau một khoảng T nơ ron Y sẽ kích thích, X – T – Y tạo thành một nơ ron khái niệm mới.

* X kích thích, dự đoán sau khoảng T thì nơ ron Y cũng kích thích

Tương tự với độ dài kích thích, nơ ron X kích thích dài trong khoảng thời gian L lặp lại sẽ tạo thành nơ ron khái niệm mới X – L

* Ví dụ X-L sẽ có nơ ron M cùng kích thích tạo thành X-L – M
* X-L’ sẽ có nơ ron P cùng kích thích tạo thành X-L’-P

Tương tự là cường độ kích thích, ví dụ nơ ron X kích thích với độ lớn B, lặp lại, tạo thành nơ ron X-B

* Ví dụ X-B kết hợp cùng M tạo thành X-B – M
* Ví dụ X-B’ kết hợp cùng P tạo thành X-B’-P

**Note 3/4/2016 – Tại sao tò mò và bắt chước**

**Tò mò**

* Tò mò vì chưa biết, vì thèm kiến thức
* Khi “thỏa mãn” được trí tò mò, các nơ ron được củng cố, kiến thức mới sinh ra -> Kích thích tưởng thưởng
* Kích thích tưởng thưởng sau khi “tò mò” dẫn đến nghiện tò mò để thỏa mãn bổ sung kiến thức

**Bắt chước**

* Bắt chước là việc thực hiện hành vi A “giống” với hành vi A’ được ghi nhận từ một đối tượng X
* Hành vi A là output, hành vi A’ là tri thức đầu vào, tác nhân là đối tượng X
* Để ghi nhận được hành vi A’, thì chủ thể O phải có khả năng tiếp thu/tiếp nhận hành vi A’ thông qua các giác quan, đầu vào của mình
* Chủ thể O ý thức được hành vi A “giống” với hành vi A’ thông qua ghi nhận từ các giác quan, đầu vào của chính mình đối với hành vi A
* Hành vi A và A’ đều biểu đạt kết nối tới một khái niệm A’’ trong mạng nơ ron của chủ thể O
* Khái niệm A’’ thể hiện hành động A’’ để đạt được mục đích M
* Hành vi bắt chước là hành vi cố gắng mô phỏng hành động A giống nhất với hành động A’ để đạt được mục đích M.
* Khi mục đích M gây ra Kích thích tưởng thưởng tới mạng nơ ron, thì đây chính là “động cơ” nguồn gốc của việc “bắt chước”

**Note on 4/4/2016**

* Trẻ con rất thích những hành vi/thứ lặp lại
  + Ú òa
  + Các câu hát lặp lại
  + Hành động lặp lại
  + ...
* Các thứ lặp lại này gây ra kích thích tưởng thưởng (KTTT) cho não của trẻ
* Vì những thứ lặp lại giúp hình thành/củng cố các nơ ron khái niệm mới cho trẻ -> KTTT
* Bắt chước phát âm:
  + Khi trẻ **vô tình** phát âm “Bà bà”, người lớn cũng nói “Bà bà” -> lặp lại -> KTTT -> trẻ thích phát âm “Bà bà”

**Note on 8/6/2016**

AI:

Quá trình phát triển của trẻ con:

3 tháng biết lẫy, 7 tháng biết bò, 9 tháng lò dò biết đi

12 tháng tập nói

Hóng

-> Bản năng

Nhu cầu cơ bản: Đói, tè, ị, đau -> Khóc, chăm sóc vỗ về -> hết khóc, ngủ

-> Bản năng

Bắt chước hành động của người lớn, bắt chước nói theo

-> Một hành vi học tập theo bản năng??

Biết nhận diện: bố, mẹ, ông, bà, họ hàng, hàng xóm.. biết nhận diện đồ vật đơn giản và nhận diện từ trong tranh

-> Bắt đầu từ những quá trình trừu tượng hóa

-> Dần dần chi tiết hóa để nhận diện và phân biệt

Háo hức tìm hiểu với những thứ mới, đồ chơi mới

-> Có một cơ chế gì đó thúc đẩy và kích thích bé khám phá những điều mới lạ quanh mình

-> Càng khám phá ra điều mới lạ bé càng sướng càng muốn khám phá

Biết sợ: sợ đau, sợ nóng, sợ tối..

-> Bắt đầu từ những nỗi sợ sơ khai: như ngã đau, nóng, cay

-> Kích thích đến những nỗi sợ tiềm thức

-> Khi người lớn dọa, những nỗi sợ tiềm thức sẽ được kích thích gây ra phản ứng sợ

Đặt câu hỏi: Cái gì đây, con gì đây...

-> Đặt câu hỏi là một khái niệm được bé ghi nhớ từ việc người lớn hỏi bé

-> Sau đó bé bắt chước hỏi lại và thấy rằng hỏi như vậy sẽ khám phá ra điều mới

-> Càng khám phá ra điều mới càng kích thích bé hỏi

--------------------------------------------------------------------

Quần thể nơ ron tuân theo quy luật di truyền?????:

-> Tập các nơ ron trong não bộ có thể là quần thể các nơ ron tuân theo quy luật di truyền

-> Các nơ ron có vòng đời từ lúc chết cho tới lúc sống

-> Chúng có khả năng sinh và truyền lại tri thức cho đời con

-> Các nơ ron có gen tốt, khỏe mạnh sẽ nhiều có nhiều cơ hội sinh con và truyền lại tri thức cho đời con hơn

-> Đời con có thể đột biến

-> Các nơ ron tuân theo quy luật chọn lọc tự nhiên, các gen yếu dễ bị đào thải khỏi quần thể

**Note on 17/7/2016**

Nhận diện hình ảnh

* Tọa độ:
  + Lấy điểm giữa làm trung tâm
  + Điểm ảnh phân bố theo hệ tọa độ cực 3 chiều: góc nằm, góc thẳng và khoảng cách
* Nhận diện đường
  + Đường thẳng:
    - Xuất phát từ điểm bắt đầu, đi đến điểm tiếp theo gần đó theo hướng alpha, và từ mọi điểm trên đường thẳng đều đi đến điểm gần nhất kế tiếp theo hướng không đổi là alpha.
    - Cách diễn giải khác: đi từ điểm A đến B, đi tiếp từ điểm B đến C tiếp theo có hướng không đổi nếu so với đường AB (góc ABC = 180 độ), tương tự từ C đến D tiếp theo cũng không đổi hướng nếu so với đường BC.. như vậy đi từ điểm xuất phát đường thẳng có xu hướng không đổi hướng
  + Đường cong:
    - Từ A đến B rất gần đó, đi tiếp từ B đến C đối theo hướng alpha nếu so với đường AB, đi tiếp từ C đến D đổi theo hướng alpha nếu so với đường BC... như vậy đường cong có hướng liên tục thay đổi nếu so với điểm xuất phát
* Nhận diện từ đơn giản đến phức tạp
  + Nhận diện sáng tối
  + Nhận diện đường nét, hình dạng, xa gần
  + Nhận diện màu sắc
* Ước lượng khoảng cách tới vật:
  + Mắt ước lượng kích thước tương đối của ảnh ở vị trí X (kích thước tương đối, tức là càng xa thì càng bé)
  + Mắt ước lượng khoảng cách từ mắt tới vị trí X
  + Khi vật di chuyển đến Y, kích thước ảnh cũng thay đổi theo
  + Từ độ lớn của vật trong mắt => khoảng cách của vật đó tới mắt
* Mỗi điểm ảnh gồm các thông số:
  + Hue: Vùng màu (Màu Red, màu Green, màu Blue)
  + Saturation: Độ bão hòa
  + Lightness: Độ sáng

**Note on 29/8/2016**

Tìm cạnh, biên của vật thể

* Cluster ảnh
  + Một điểm ảnh có xu hướng tham gia vào cluster nếu nó có mầu tương đồng với các điểm ảnh đã có trong cluster
  + Cluster có xu hướng mở rộng ra xung quanh
  + độ tương đồng được đo là ‘khoảng cách màu’
* Sự biến thiên màu ảnh

**Note on 28/05/2017**

AI tự tăng cường năng lực tư duy của nó bằng cách **tự vấn**:

* Tự hỏi, nói chuyện với chính mình
* Tự chơi các trò chơi với chính mình

**Note on 03/06/2017**

- Mắt nhìn vào vùng mầu nào sẽ có xu hướng nhạy cảm với vùng mầu đó, ưu tiên vùng màu đó hơn, nhạy cảm với vùng sáng hơn vùng tối

- Mắt chỉ "nét" với vùng ở chính giữa, xung quanh sẽ là vùng mờ

- Đường cong, biên, đường nét của vật thể... chính là đường di chuyển của mắt, góc độ di chuyển của mắt

- Đặc trưng của một đối tượng: đường nét của đối tượng, chất liệu, màu sắc...

- Trẻ em thường thích xem hoạt hình, vì mắt sẽ thấy thoải mái hơn khi xem, các đối tượng, con vật.. dễ được phát hiện ra vì đơn màu, góc, cạnh biên mượt…

- Mắt có thể nhận ra sự lặp lại của các mẫu (pattern) -> nhận ra được chất liệu, ví dụ áo kẻ caro, gạco, quần sọc…

- Liên quan đến tiến hoá và sinh tồn, khi mắt luôn phải nhanh nhất có thể tìm ra và phát hiện đối tượng, với rất ít thông tin -> tìm góc cạnh, tìm đặc trưng… chứ ko quét toàn bộ đối tượng cũng như chi tiết từng vùng của đối tượng

- Mắt đo khoảng cách của đối tượng bằng cách focus vào đối tượng đó cho đến khi đối tượng đó thật sự “nét” để phát hiện ra. Khoảng cách sẽ liên quan đến tiêu cự của mắt cần điều chỉnh.

**Note on 01/01/2018**

* Mỗi một “mục đích” tương ứng với một tập nơ ron được kích thích
* Khi hành động, các nơ ron được sắp xếp để thực hiện từ trước đến sau
* Ví dụ nơ ron cho chữ “xin chào” sẽ là một tập “x”, “i”, “n”, “c”, “h”, “a”, “o”, “xin”, “chao”…, tuy nhiên khi viết chữ “xin chào” thì chỉ sẽ được sắp xếp lại.

**Note 21/01/2018- Thuật toán Bug find food Optimize**

Cải tiến từ thuật toán Bug find food (BFF), BFF O có nhiệm vụ cải tiến thuật toán sao cho thời gian tìm food nhanh nhất, tương ứng với số hành động ít nhất có thể.