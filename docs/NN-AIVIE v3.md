# KẾ HOẠCH PHÁT TRIỂN KIẾN TRÚC MẠNG LAI NEURO-SYMBOLIC CHO NN-AIVIE

*(Lộ trình tiến hóa từ Ý tưởng phân cấp sinh học đến Hiện thực hóa ma trận trên GPU)*

## 1. Hiện trạng & Triết lý khởi nguồn (Ý tưởng ban đầu của Tác giả)

### 1.1. Bản chất ý tưởng gốc

Ý tưởng bắt đầu từ việc xây dựng một mạng nơ-ron phân cấp dựa trên nguyên lý ngữ nghĩa trực quan và cơ chế tự động sinh nơ-ron:

* **Nơ-ron Khái niệm/Ngữ cảnh lớn:** Khi mạng học một cụm từ hay cả một câu văn (Ví dụ: *"đặc điểm của con bò là gì? đáp: chúng có 4 chân..."*), toàn bộ cụm/câu này được coi là một nơ-ron cha vĩ mô ($n\_1$).
* **Nơ-ron Token cơ sở:** Mỗi từ/token đơn lẻ đóng vai trò nơ-ron input, ban đầu nối thẳng vào nơ-ron cha.
* **Cơ chế sinh nơ-ron trung gian:** Khi xuất hiện các cụm từ con lặp đi lặp lại giữa các câu khác nhau (như *"đặc điểm của"*, *"hình dạng"*, *"chúng có 4 chân"*), mạng sẽ tự động bóc tách và sinh ra các nơ-ron trung gian mới. Các nơ-ron token cơ sở sẽ chuyển sang nối vào nơ-ron trung gian này, và nơ-ron trung gian lại kết nối lên nơ-ron cha.
* **Cơ chế Thứ tự (Index) & Sức mạnh (Power):** \* Các nơ-ron con khi nối vào nơ-ron cha sẽ được đánh chỉ mục cứng (index = 1, 2, 3...) để duy trì thứ tự xuất hiện của từ.
  + Mỗi nơ-ron lưu trữ giá trị power đại diện cho tần suất xuất hiện (đếm số lần lặp trong quá trình train).

### 1.2. Cơ chế vận hành nguyên bản (Inference)

* **Quá trình Lan xuôi (Bottom-up):** Khi nhận một Prompt (Ví dụ: *"Hỏi: đặc điểm của loài chó là gì? Đáp:"*), năng lượng từ các token tầng dưới sẽ kích hoạt lan dần lên các nơ-ron tầng trên. Nơ-ron cha chứa nhiều thành phần khớp nhất sẽ ở đỉnh và tích tụ nhiều năng lượng kích hoạt nhất.
* **Quá trình Lan ngược (Top-down / Pattern Completion):** Từ nơ-ron cha ở đỉnh, hệ thống kiểm tra các thành phần con theo chỉ mục index. Những thành phần nào đã có trong prompt thì bỏ qua, thành phần nào còn thiếu (Ví dụ: đoạn trả lời phía sau từ *"Đáp:"*) sẽ được dội ngược năng lượng xuống các tầng nơ-ron con thấp hơn cho đến tầng token cơ sở để sinh từ, lấp đầy thành phần thiếu.
* **Xử lý tri thức thiếu hụt (Case con kiến):** Nếu prompt hỏi về một chủ đề chưa từng được học cấu trúc hỏi-đáp trực tiếp nhưng đã có dữ liệu mô tả (Ví dụ: *"đặc điểm của kiến là sống theo bầy đàn..."*), các nơ-ron đặc điểm vĩ mô của kiến vẫn sẽ kích hoạt mạnh. Quá trình lan ngược sẽ tự động chọn nhánh nơ-ron con chưa được kích hoạt để đổ năng lượng xuống, sinh ra từ tiếp theo (Ví dụ từ *"là"*).

## 2. Quá trình trao đổi & Các bước cải tiến ý tưởng

Để đưa ý tưởng sinh học trực quan này vào môi trường tính toán thực tế của dự án nn-aivie, quá trình thảo luận phản biện đã bóc tách và tinh chỉnh các góc khuất kỹ thuật qua 3 luận điểm lớn:

### 2.1. Giải quyết bài toán bùng nổ tổ hợp (Combinatorial Explosion)

* *Vấn đề:* Nếu câu nào khác biệt một chút cũng sinh nơ-ron câu riêng, số nơ-ron sẽ bằng số câu độc bản trong tập dữ liệu. Hệ thống sẽ bị rác và không thể tổng quát hóa.
* *Cải tiến:* Áp dụng tư duy của thuật toán BPE/Re-Pair (Grammar Induction). **Chỉ gộp và tạo nơ-ron mới cho các cụm từ ngắn đi liền nhau có tần suất xuất hiện vượt qua một ngưỡng quy định (min\_frequency)**. Không bao giờ lưu nguyên một câu dài làm một nơ-ron độc lập trừ khi câu đó lặp lại nguyên bản nhiều lần.

### 2.2. Xóa bỏ tính "giòn" (Brittle) bằng Không gian Vector (Embeddings)

* *Vấn đề:* Nếu chỉ khớp chuỗi rời rạc (exact match), từ *"chó"* và *"cún"* hay *"kiến"* và *"loài kiến"* sẽ thuộc về các nơ-ron hoàn toàn biệt lập, làm trượt xích toàn bộ dòng năng lượng lan truyền nếu prompt thay đổi từ ngữ.
* *Cải tiến:* Tích hợp thêm **Lớp Embedding mềm (Fuzzy Matching)**. Mỗi nơ-ron (kể cả nơ-ron cụm mới sinh) sẽ sở hữu một vector embedding. Khi prompt đi vào, ta tính Cosine Similarity để kích hoạt mềm tất cả các nơ-ron có nghĩa gần nhau, cho phép năng lượng truyền lên tầng cha một cách uyển chuyển.

### 2.3. Mềm hóa văn phong bằng Ma trận dịch chuyển Markov thay cho Index cứng

* *Vấn đề:* Việc gán chỉ mục cứng (index 1, 2, 3) khiến mạng bị phụ thuộc tuyệt đối vào trật tự câu chữ tĩnh, câu từ sinh ra sẽ bị máy móc, què cụt và không xử lý được hiện tượng đảo ngữ trong ngôn ngữ mở.
* *Cải tiến:* Thay vì lưu index cố định trong nơ-ron cha, ta chuyển sang dùng **Ma trận Xác suất Chuyển dịch Vị trí Ngữ cảnh ($\mathbf{W}\_{seq}$)** kết hợp **Bộ nhớ ngắn hạn (Hebbian Trace)**. Từ tiếp theo được chọn dựa trên sự giao thoa giữa: "Năng lượng vĩ mô dội từ cha xuống" và "Xác suất từ đi sau chuỗi từ vừa sinh trước đó", sau đó lấy mẫu qua hàm Softmax có Temperature để tạo văn phong tự nhiên.

## 3. Chi tiết đề xuất kiến trúc kỹ thuật cuối cùng

Mô hình được chuẩn hóa thành một mạng lai **Neuro-Symbolic Hybrid Network** chạy hoàn toàn trên GPU bằng PyTorch, chia làm 3 tầng dữ liệu:

[Tầng Đỉnh: Các Nơ-ron Ý niệm vĩ mô / Nơ-ron Cha]
 ▲ │
 Lan xuôi│ │Lan ngược
 (Aᵀ · a)│ │(A · a) kết hợp W\_seq
 │ ▼
[Tầng Trung gian: Các Nơ-ron Cụm từ được sinh tự động]
 ▲ │
 │ ▼
[Tầng Đáy: Token Embeddings cố định - Xử lý khớp mờ Cosine]

1. **Tầng Embedding Mềm ($\mathbf{E}$):** Ma trận kích thước $[N \times d]$ (với $d=384$). Phụ trách ánh xạ các nơ-ron vào không gian vector liên tục.
2. **Ma trận Phân cấp Thưa ($\mathbf{A}$):** Ma trận thưa kích thước $[N \times N]$ lưu mối quan hệ giữa Nơ-ron cha và các Nơ-ron con. Trọng số tỷ lệ thuận với sức mạnh (power).
3. **Ma trận Dịch chuyển Ngữ cảnh ($\mathbf{W}\_{seq}$):** Ma trận kích thước $[N \times N]$ lưu tỷ lệ phân phối vị trí kề cận của các từ.

## 4. Giải pháp triển khai cụ thể trên GPU (Implementation Blueprint)

### Bước 1: Huấn luyện tăng trưởng Offline kết hợp Online

1. **Trích xuất cụm (Offline - CPU):** Quét văn bản bằng thuật toán gom cụm tham lam, lọc các cụm từ lặp cao hơn min\_frequency.
2. **Cấp phát nơ-ron và Khởi tạo không gian (GPU):** Khi sinh ra một nơ-ron cụm mới, vector embedding của nó được tính tức thời bằng phương pháp *Mean Pooling* (Trung bình cộng vector của các nơ-ron con cấu thành).
3. **Cập nhật trọng số Hebbian (GPU):** Thay đổi các giá trị liên kết trong ma trận thưa dựa trên sự đồng kích hoạt của các token bằng quy tắc Hebbian dạng ma trận hóa (torch.sparse):
   $$\mathbf{W}\_{seq} = \mathbf{W}\_{seq} + \eta \cdot (\mathbf{a}\_t^T \times \mathbf{a}\_{t+1})$$

### Bước 2: Thuật toán suy luận tự nhiên (Inference Module)

Khi nhận Prompt, tiến trình xử lý trên GPU diễn ra như sau:

1. **Khớp mờ:** Tính tương đồng Cosine giữa prompt và ma trận embedding $\mathbf{E}$ để tạo vector năng lượng ban đầu $\mathbf{a}\_0$.
2. **Lan xuôi:** Chạy phép nhân ma trận thưa lặp để tìm nơ-ron cha vĩ mô: $\mathbf{a}\_{t+1} = \sigma(\mathbf{A}^T \cdot \mathbf{a}\_t)$.
3. **Lan ngược tạo màng lọc vĩ mô:** Dội năng lượng từ cha xuống để định hình bộ khung kiến thức: $\mathbf{a}\_{macro} = \mathbf{a}\_{parent} \cdot \mathbf{A}$.
4. **Tính toán mạch văn vi mô:** Lấy vector dấu vết của các từ vừa sinh (bộ nhớ ngắn hạn có suy giảm $\gamma$) nhân với ma trận vị trí để biết từ nào hợp logic tiếp theo: $\mathbf{a}\_{micro} = \mathbf{a}\_{context\\_trace} \cdot \mathbf{W}\_{seq}$.
5. **Lấy mẫu sinh từ:** Kết hợp hai luồng năng lượng $\mathbf{a}\_{final} = \mathbf{a}\_{macro} \odot \mathbf{a}\_{micro}$. Đưa qua torch.softmax(a\_final / Temperature) và dùng hàm torch.multinomial để bốc ngẫu nhiên từ tiếp theo, đảm bảo câu văn mượt mà và biến hóa tự nhiên.

### Bước 3: Lộ trình tích hợp vào hệ thống nn-aivie

* **Lưu trữ tài liệu:** Lưu bản thiết kế này thành study/02\_HYBRID\_HEBBIAN\_MATRIX.md.
* **Mã nguồn:** Xây dựng module prototype độc lập mang tên generate\_hebbian.py sử dụng thư viện PyTorch để kiểm tra hiệu năng tính toán ma trận thưa.
* **Ứng dụng:** Ánh xạ module này trực tiếp vào hàm assess\_answer\_closure() để làm màng lọc kiểm tra tính bao phủ của câu trả lời, đảm bảo không bị thiếu hụt ý niệm ngữ nghĩa của Agenda đặt ra.

NGHIÊN CỨU BỔ SUNG: TRỪU TƯỢNG HÓA CẤU TRÚC TRONG SINH CHUỖI DÀI VÀ MÃ NGUỒN (STORY & CODE GENERATION)

* Trần kiến trúc của Cơ chế Phân rã Gốc
  Cơ chế gộp cụm nguyên bản (BPE/Re-Pair) vận hành hoàn toàn dựa trên Sự lặp lại bề mặt (Lexical Surface Repetition). Thuật toán bắt ngữ nghĩa rất tốt cho các cụm từ cố định xuất hiện y hệt nhiều lần (ví dụ: "chúng có 4 chân", "đặc điểm của").

Tuy nhiên, khi nâng cấp lên quy mô Sinh một truyện ngắn cực dài hoặc Code một chương trình phần mềm, mạng sẽ đối mặt với Trừu tượng hóa cấu trúc/chức năng (Structural/Functional Abstraction) – một vùng không gian mà cơ chế đếm lặp từ vựng bị mù hoàn toàn:

* Trong Văn bản dài: Đoạn mở đầu (Setup) của Truyện A ("Ngày xửa ngày xưa...") và Truyện B ("Mưa rơi suốt đêm trên thành phố vắng...") không chia sẻ bất kỳ token chung nào. Khái niệm "Mở bài" là một vị trí chức năng, không phải là một chuỗi ký tự cố định.
* Trong Mã nguồn: Hàm def foo() và def calculate\_tax() hoàn toàn khác biệt về mặt từ vựng nhưng chia sẻ chung vai trò cấu trúc cú pháp FunctionDef.

Bản chất lý thuyết: Trừu tượng hóa cấu trúc là một hiện tượng Liên-mẫu (Cross-example), không tồn tại trong phạm vi Nội-mẫu (Within-example). Hệ thống không thể rút ra "khung truyện" hay "khung thuật toán" nếu chỉ xử lý tuần tự từng mẫu văn bản cô lập.

* Chiến lược 2 Chế độ Trí nhớ (Dual-mode Memory) cho Truyện dài và Code

Để xử lý mạch lạc tầm xa và tính chính xác tuyệt đối của cấu trúc, mạng lai cần tích hợp song song hai chế độ: Không gian nhúng mềm (Soft Embedding Space) và Bảng ký hiệu cứng (Hard Symbol Table).

Chế độ 1: CHẾ ĐỘ MỀM (Soft Embedding)

* Vai trò: Quản lý mạch văn, ngữ cảnh chung, gợi ý "viết kiểu gì, cấu trúc nào hợp". Cơ chế này cho phép dung sai mềm (Fuzzy Interpolation).
* Ứng dụng đối với Truyện dài: Sử dụng Hebbian trên Vị trí và Ngữ cảnh chung. Các nơ-ron-đoạn có cùng vị trí cấu trúc (cùng index trong nơ-ron cha) và có phân phối hàng xóm giống nhau sẽ bị luật Hebbian kéo lại gần nhau trong không gian vector. Hiện tượng Hội tụ phân phối (Clustering) sẽ tự động diễn ra, chưng cất ra tâm cụm (Centroid) chính là nơ-ron trừu tượng biểu diễn cho vai trò cấu trúc (Ví dụ: nơ-ron [SETUP], [CONFLICT]) mà không cần gán nhãn thủ công.
* Ví dụ: [STORY] [SETUP] Ngày xửa ngày xưa... [/SETUP] [CONFLICT] Nhưng một ngày kia... [/CONFLICT] [CLIMAX] ... [/CLIMAX] [RESOLUTION] ... [/RESOLUTION] [/STORY]

Chế độ 2: CHẾ ĐỘ CỨNG (Symbol Table)

* Vai trò: Ghi nhớ chính xác tên biến, khóa kiểu dữ liệu, chữ ký hàm, áp đặt ràng buộc chính xác tuyệt đối.
* Ứng dụng đối với Mã nguồn: Lập trình không có dung sai cho sự gần đúng; sai một ký tự (user\_count thành userCount) hệ thống sẽ sụp đổ. Do đó, Chế độ Cứng (Blackboard / Symbol Table) đóng vai trò bắt buộc. Mỗi khi mạng sinh ra một định nghĩa biến hoặc hàm, định nghĩa đó lập tiếp được ghi nhận dưới dạng chuỗi ký tự chính xác tuyệt đối để tra cứu cứng khi sinh các khối logic tiếp theo.
* Tối ưu hóa Huấn luyện bằng Cây cú pháp trừu tượng AST (Sân thử nghiệm tối ưu cho nn-aivie)

So với văn bản tự nhiên, Mã nguồn (Code Gen) chính là vùng đất lý tưởng nhất để triển khai kiến trúc này nhờ 3 lợi thế tuyệt đối từ cấu trúc hình thức:

1. Cấu trúc tường minh miễn phí: Ta không cần dùng Hebbian tự mò mẫm cấu trúc của mã nguồn. Khi nạp dữ liệu train, chạy trực tiếp bộ Parser của ngôn ngữ (ví dụ: Python AST Parser) để trích xuất ra Cây cú pháp trừu tượng (Abstract Syntax Tree). Các nhãn cấu trúc như FunctionDef, ForStmt, Assign, Return chính là các token cấu trúc tối cao được gán sẵn hoàn hảo và không có nhiễu.
2. Tập trừu tượng hữu hạn: Khác với truyện có vô số kiểu biểu đạt, cấu trúc mã nguồn chỉ đóng khung trong một tập hữu hạn các loại nút cú pháp. Mạng lai sẽ hội tụ nhanh hơn gấp hàng ngàn lần.
3. Vòng phản hồi Vàng (Self-Verifying): Tính đúng sai của code là khách quan. Trình biên dịch, Linter và Unit Test cung cấp tín hiệu phản hồi ngay lập tức để mạng tự động điều chỉnh năng lượng kích hoạt, tối ưu hóa quá trình sinh (Generate-and-Verify Pipeline).
4. Giải pháp thuật toán: Cập nhật Ma trận Chuyển dịch Cấu trúc liên tục (W\_struct)

Bổ sung một pha So sánh chéo định kỳ (Cross-sample Phase) vào lõi Training. Khi phát hiện hai nơ-ron đoạn/khối i và j xuất hiện ở cùng vị trí chức năng trên đồ thị phân cấp của nhiều mẫu khác nhau, ma trận liên kết cấu trúc liên tục được cập nhật theo công thức: tích ngoài Delta W\_struct bằng hệ số eta nhân với tích chập của e\_pos\_i và e\_context\_j chuyển vị.

Lúc suy luận Top-Down, mạng áp đặt một màng lọc cú pháp thông qua ngăn xếp cấu trúc (Context Stack). Năng lượng cuối cùng a\_final bằng tích chập của ba luồng: a\_macro nhân a\_micro nhân a\_symbol\_mask. Trong đó:

1. a\_macro: Định hướng vai trò khối logic dội ngược từ AST cha.
2. a\_micro: Mạch chuyển dịch Markov thời gian ngắn của token kề cận.
3. a\_symbol\_mask: Mặt nạ lọc từ Chế độ Cứng (Symbol Table), triệt tiêu toàn bộ các token vi phạm quy tắc cú pháp hoặc sai lệch tên thực thể.

Kết luận về Trần sáng tạo

Mô hình phân rã cấu trúc kết hợp bộ nhớ kép này hoạt động như một hệ thống Template-Aware và Self-Verifying Generator. Hệ thống sẽ tái tổ hợp cực kỳ xuất sắc các khuôn mẫu cấu trúc đã học (Ví dụ: viết các script CRUD chuẩn, sinh cấu trúc truyện 3 hồi, cấu trúc hàm xử lý lỗi).

Trần kiến trúc không thể vượt qua: Hệ thống chỉ nội suy trong không gian khuôn mẫu cũ, không thể tự phát minh ra một thuật toán hoàn toàn mới (ví dụ: một kiến trúc sắp xếp lạ) hoặc một cốt truyện đột phá vượt ngoài biên độ thống kê liên-mẫu. Định vị thực dụng nhất của module này trong nn-aivie là tạo ra lõi sinh code/văn bản chuẩn cấu trúc văn phạm 100%, làm màng lọc ngữ cảnh đóng chặt chẽ cho mô hình.