# ĐỀ ÁN THIẾT KẾ NN-AIVIE V2: ĐỒ THỊ KHÁI NIỆM TRỪU TƯỢNG TỰ PHÁT (DYNAMIC LATENT ABSTRACTION & NEUROGENESIS)

---

## 1. Tầm nhìn chiến lược & Tính cách mạng

Trong phiên bản **NN-AIVIE v1 (POC)**, khả năng khái quát hóa (generalization) dựa trên cơ chế ghép cặp song sinh (sibling pairs) và thừa hưởng cạnh tĩnh Jaccard (`consolidate_emergent`). Tuy nhiên, cách tiếp cận này gặp phải giới hạn:
1. **Tính chất phẳng (Flat Graph):** Sự kế thừa chỉ diễn ra ở 1 tầng duy nhất giữa các token bề mặt (surface tokens), không tạo ra các tầng tri thức trừu tượng trung gian.
2. **Khó khăn trong suy luận logic đa cấp (Multi-hop Reasoning):** Không thể suy luận bắc cầu như: `bố` $\rightarrow$ `phụ huynh` $\rightarrow$ `con người` $\rightarrow$ `cần thở` $\rightarrow$ `sinh học`.
3. **Phụ thuộc vào thống kê tĩnh:** Việc kế thừa tính chất diễn ra ngoại tuyến (offline consolidation) thay vì tự phát và mượt mà trong quá trình huấn luyện trực tuyến (online training).

**NN-AIVIE v2** thay đổi hoàn toàn cuộc chơi bằng cách đưa vào cơ chế **Unsupervised Neurogenesis (Tự sinh nơ-ron không giám sát)** và **Multi-tier Hierarchical Spreading (Lan truyền kích hoạt phân cấp đa tầng)**. 

Mạng sẽ tự tạo ra các nơ-ron đại diện cho **khái niệm ẩn (Latent Concept Nodes)** có ký hiệu `Concept:XXXX`. Quá trình này hoàn toàn tự phát dựa trên các kích hoạt Hebbian tự nhiên của dòng dữ liệu từ Lớp 3 đến Lớp 7, mang tính chất sinh học thuần khiết và **tuyệt đối không hardcode**.

---

## 2. Các nguyên lý kiến trúc cốt lõi (Tuân thủ nghiêm ngặt Mục 0)

Để đảm bảo không vi phạm **Mục 0 (Luật kiến trúc)** của dự án, thiết kế v2 tuân thủ các nguyên tắc sau:
* **Không ontology định sẵn:** Không khai báo bất kỳ nhãn khái niệm nào như `"con người"`, `"động vật"`, `"toán học"`. Mạng chỉ sinh ra các ID nơ-ron ẩn vô danh như `Concept:0001`, `Concept:0002`. Ý nghĩa của chúng hoàn toàn do các kết nối đồ thị định nghĩa.
* **Suy luận thuần đồ thị (Graph-only Inference):** Mọi sự kích hoạt và suy luận ở runtime đều dựa trên lan truyền xung lực (spreading activation) qua ma trận kề `TypedAdjacency` và các Priority Weight, không dùng bất kỳ câu lệnh `if/else` ngôn ngữ hay cấu trúc logic lập trình nào.
* **Học động từ dữ liệu:** Sự hình thành nơ-ron khái niệm và các liên kết đa tầng được điều hướng hoàn toàn bởi các pha học: G1 Passive (Học thụ động), G2 Imitation (Học bắt chước), và G3 Dopamine (Củng cố/Phạt lặp dựa trên LTD).

---

## 3. Kiến trúc kỹ thuật chi tiết

### A. Cơ chế tự sinh nơ-ron ẩn (Unsupervised Neurogenesis)

Trong quá trình huấn luyện G1 Passive và G2 Imitation, ma trận đồng kích hoạt (co-activation matrix) ghi nhận tần suất các nơ-ron đồng thời kích hoạt cùng nhau.

#### 1. Thuật toán phát hiện Clique (Co-firing Cliques)
Khi một nhóm nơ-ron bề mặt $\{n_1, n_2, \dots, n_k\}$ có trọng số đồng kích hoạt chéo vượt qua ngưỡng dẻo $\theta_{neuro}$ và có chung một lượng ngữ cảnh đầu ra (out-degree contexts), mạng sẽ kích hoạt cơ chế sinh nơ-ron:
$$\text{Neurogenesis-Trigger}(N_{\text{clique}}) \rightarrow \text{True} \quad \text{if} \quad \forall i, j \in N_{\text{clique}}, \ \text{Co-activation}(i, j) \ge \theta_{neuro}$$

#### 2. Tuyển mộ nơ-ron Concept (Concept Recruitment)
Khi điều kiện trên thỏa mãn, một nơ-ron ẩn mới $C_{\text{new}}$ (được đặt tên `Concept:XXXX` trong `Vocabulary`) sẽ được tạo ra.
* **Liên kết Bottom-Up:** Tạo các cạnh hướng lên từ các thành viên clique đến nơ-ron khái niệm:
  $$\mathbf{W}(n_i \rightarrow C_{\text{new}}, \text{EdgeKind.CO\_ACTIVATION}) = w_{\text{init}}$$
* **Liên kết Top-Down:** Tạo các cạnh hướng xuống để nơ-ron khái niệm củng cố ngược lại các thực thể của nó:
  $$\mathbf{W}(C_{\text{new}} \rightarrow n_i, \text{EdgeKind.REINFORCED}) = w_{\text{init}} \times \gamma_{\text{top-down}}$$

```
               [ Concept:010 ]  <-- Khái niệm bậc cao hơn (vd: Con người)
                ^         ^
               /           \ (Co-activation)
              /             \
       [ Concept:001 ]   [ Concept:002 ] <-- Khái niệm bậc trung (vd: Phụ huynh vs Đồng nghiệp)
        ^   ^   ^   ^
       /    |   |    \ (Co-activation / Reinforced)
      /     |   |     \
    [bố]  [mẹ] [anh]  [chị]             <-- Nơ-ron bề mặt (Surface tokens)
```

### B. Học liên kết đa tầng (Hierarchical Association & Lateral Inhibition)

Để tránh hiện tượng toàn bộ nơ-ron ẩn co cụm lại thành một siêu kết nối (mega-hub), mạng áp dụng các luật dẻo hóa Hebbian cải tiến:

1. **Ức chế cạnh bên (Lateral Inhibition):**
   Khi nơ-ron $C_1$ đã được tuyển chọn và liên kết mạnh với một clique, nó sẽ phát ra một xung lực ức chế cạnh bên đến các nơ-ron ẩn lân cận khác để ngăn chúng liên kết với cùng tập hợp thực thể đó. Điều này đảm bảo tính phân tách đặc trưng (feature disentanglement).
2. **Kích hoạt liên tầng (Inter-tier Association):**
   Bản thân các nơ-ron `Concept:A` và `Concept:B` cũng có thể co-fire. Khi chúng đồng kích hoạt mạnh, một nơ-ron ẩn cấp cao hơn `Concept:Super` sẽ được tuyển mộ để liên kết chúng lại, hình thành nên cấu trúc cây khái niệm (concept tree) sâu một cách tự phát.

### C. Cơ chế lan truyền kích hoạt hai chiều (Bidirectional Spreading Activation)

Khi thực hiện Inference (Suy luận từng bước tại `/api/trace` hoặc CLI `generate`):

1. **Bottom-Up Abstraction (Tập trung ngữ cảnh):**
   Khi người dùng nhập prompt chứa các từ khóa cụ thể (vd: `bố`, `mẹ`), xung lực kích hoạt lan truyền lên các nơ-ron khái niệm `Concept:001` (Gia đình), rồi tiếp tục lan truyền lên `Concept:010` (Con người).
2. **Top-Down Generalization (Lan tỏa thuộc tính):**
   Nếu tại nơ-ron `Concept:010` (Con người) có các liên kết ngữ nghĩa mạnh dẫn đến các thuộc tính cụ thể (vd: `cần thở`, `biết đi`, `học tập`), các thuộc tính này sẽ nhận được một lượng kích hoạt lan truyền ngược lại thông qua các nơ-ron khái niệm bậc trung, boost điểm số của chúng lên trong `scores_base`.
3. **Công thức tính lan truyền tổng hợp:**
   $$\mathbf{S}_{\text{concept}}(t) = \sum_{c \in \text{Concepts}} \mathbf{Activation}(c) \times \mathbf{W}(c \rightarrow t, \text{EdgeKind.REINFORCED}) \times \pi_{\text{concept}}$$

---

## 4. Tận dụng Dataset & Quy trình huấn luyện (Curriculum)

NN-AIVIE v2 sẽ tận dụng toàn bộ bộ dữ liệu curriculum hiện có từ Lớp 3 đến Lớp 7 để huấn luyện tích lũy thành một checkpoint duy nhất (`grade_cumulative.pkl`), nhưng thay đổi cách thức xử lý ở mỗi pha huấn luyện:

### Pha 1: Direct Teaching (Nhập tâm cấu trúc cơ bản)
* Nạp các mẫu ngữ pháp và câu hỏi mẫu để xây dựng khung cấu trúc nền tảng (`Rule:`, `Var:`).
* Các từ ngữ bề mặt được gán vào các vị trí slot.

### Pha 2: G1 Passive - Tự sinh khái niệm (Concept Emergence)
* Cho mô hình đọc lượng lớn văn bản tri thức (General Corpus + Subject Knowledge từ Lớp 3 - Lớp 7).
* Đây là pha vàng để kích hoạt **Unsupervised Neurogenesis**. Khi các thực thể xuất hiện trong các cấu trúc câu giống nhau, các nơ-ron `Concept:` sẽ liên tục được sinh ra và phân tầng.

### Pha 3: G2 Imitation - Neo giữ ngữ nghĩa vào khung câu hỏi (Semantic Grounding)
* Huấn luyện bắt chước trên các cặp QA mẫu.
* Các nơ-ron khái niệm tự động liên kết với các cấu trúc câu hỏi (`QFocus:`). Ví dụ: `QFocus:Con_mèo_màu_gì` sẽ liên kết với `Concept:Color` thông qua hoạt động co-fire tự nhiên.

### Pha 4: G3 Dopamine & LTD - Sắc bén hóa đồ thị (Synaptic Pruning)
* Chạy mô hình phản hồi dopamine.
* Sử dụng cơ chế suy giảm dài hạn (Long-term Depression - LTD) để **cắt tỉa** các kết nối khái niệm bị mờ nhạt hoặc gây nhiễu (cross-example bleed). Nếu một nơ-ron khái niệm bị kích hoạt sai ngữ cảnh dẫn đến câu trả lời lỗi, các kết nối dẫn đến nó sẽ bị phạt giảm mạnh trọng số.

---

## 5. Kế hoạch hành động và Lộ trình triển khai (Action Plan)

### Bước 1: Thiết lập cấu trúc từ vựng mới (Vocabulary & State Expansion)
* Cập nhật `src/nn_aivie/core/vocab.py` để hỗ trợ thêm prefix ẩn mới: `Concept:`.
* Viết thêm module quản lý neurogenesis tự động trong quá trình huấn luyện: `src/nn_aivie/training/neurogenesis.py`.

### Bước 2: Tích hợp thuật toán tự sinh Concept vào G1 Passive
* Sửa đổi `src/nn_aivie/training/g1_passive.py`.
* Sau mỗi epoch của G1, tiến hành quét ma trận co-activation, phát hiện các clique và tiến hành tuyển chọn nơ-ron khái niệm ẩn mới, tự động nối các cạnh `EdgeKind.CO_ACTIVATION` và `EdgeKind.REINFORCED`.

### Bước 3: Cải tiến Matcher lan truyền hai chiều (Bidirectional Spreading Matcher)
* Sửa đổi `src/nn_aivie/hca/forward.py` để thêm hàm `concept_hierarchical_read()`.
* Hàm này sẽ tính toán lượng kích hoạt lan truyền đa tầng (multi-hop activation propagation) từ các token lịch sử lên các nơ-ron Concept và dội ngược lại các token thuộc tính.
* Tích hợp hàm này vào pipeline tính điểm chính của `generate()` và `trace_generate_step()`.

### Bước 4: Kiểm thử, Tối ưu hóa & Trực quan hóa trên Graph UI
* Viết bộ test kiểm thử suy luận bắc cầu tại `tests/test_concept_abstraction.py` (Ví dụ: Dạy `bố` là gia đình, `mẹ` là gia đình, gia đình là con người, con người biết đi $\rightarrow$ Kiểm tra xem `bố` có tự động biết đi mà không cần dạy trực tiếp hay không).
* Cập nhật Graph UI để hiển thị các nơ-ron `Concept:` vô danh với một màu sắc đặc trưng (vd: màu tím neon) để người dùng có thể click vào xem cấu trúc phân cấp đồ thị cực kỳ đẹp mắt và sống động.

---

## 6. Các quy luật sinh học cốt lõi từ Tài liệu Nghiên cứu Hệ thống (study_nn.docx)

Để tăng cường chiều sâu sinh học và triệt để cách mạng hóa mô hình, chúng tôi tích hợp trực tiếp các nguyên lý lịch sử sâu sắc từ ghi chú nghiên cứu của bạn vào kiến trúc kỹ thuật của **NN-AIVIE v2**:

### A. Cơ chế quản lý "Vốn tự có" (Neuron Capital & Reserve Fee for Plasticity)
*   **Nguyên lý cốt lõi (Note 16/01/2011):** Mỗi nơ-ron sở hữu một giá trị nội lực gọi là "vốn tự có" (Neuron Capital). Khi hai nơ-ron hình thành liên kết trực tiếp, cả hai đều phải trả một "khoản phí hình thành liên kết" (connection fee) khấu trừ từ vốn tích lũy này. Vốn tự có chỉ được tích lũy khi nơ-ron hoạt động năng nổ (high activation) hoặc có nhiều kết nối trỏ đến.
*   **Giải pháp mô hình hóa v2:** Thay vì khởi tạo liên kết bừa bãi hoặc cập nhật Hebbian liên tục gây bão hòa ma trận kề, chúng tôi đưa vào một vector nội lực $\mathbf{C} \in \mathbb{R}^{V}$ biểu diễn Vốn tự có của từng nơ-ron:
    $$\mathbf{C}_i \leftarrow \mathbf{C}_i + \alpha_{\text{accum}} \times \mathbf{Activation}(i)$$
    Mối liên kết giữa nơ-ron $i$ và $j$ chỉ được nâng cấp hoặc hình thành mới khi và chỉ khi:
    $$\mathbf{C}_i \ge \theta_{\text{fee}} \quad \text{và} \quad \mathbf{C}_j \ge \theta_{\text{fee}}$$
    Khi liên kết được thiết lập, cả hai nơ-ron đều bị trừ phí $\theta_{\text{fee}}$, tạo ra cơ chế tối ưu hóa **L1/Sparse Regularization sinh học tự nhiên cực kỳ mạnh mẽ**, triệt tiêu kết nối nhiễu bừa bãi và tăng hiệu quả tính toán của đồ thị mỏng (sparse graph).

### B. Dòng khí Âm Dương & Cơ chế đánh thức ký ức (Yin-Yang Stimulus Flow & Memory Awakening)
*   **Nguyên lý cốt lõi (Note 07/02/2011 & 24/02/2011):** 
    *   **Dòng khí Dương (Yang Flow):** Kích thích tức thời, hiện tại, hướng ngoại. Khi nơ-ron bị kích hoạt bởi bên ngoài, nó truyền dòng khí Dương đi khắp mạng.
    *   **Dòng khí Âm (Yin Flow):** Lưu trữ quá khứ, nội tại, truyền thống. Khi dòng khí Dương đi qua, bản thân nơ-ron giữ lại một năng lượng Âm đại diện cho ký ức ngắn/trung hạn.
    *   **Sự giao thoa (Yin-Yang Collision):** Khi một kích thích Dương mới từ nơ-ron B chạm tới nơ-ron A, kích thích Dương này "đánh thức" (awakens) dòng khí Âm tích lũy trong A, giải phóng nó để cùng củng cố nơ-ron chung C (nơ-ron khái niệm hoặc nơ-ron con).
*   **Giải pháp mô hình hóa v2:** Chúng tôi xây dựng mô hình lan truyền kích hoạt hai dòng độc lập:
    *   **Dương Vector ($\mathbf{a}^{+} \in \mathbb{R}^V$):** Trạng thái kích hoạt hiện tại (Forward Activation) dùng trực tiếp cho scoring next-token.
    *   **Âm Vector ($\mathbf{a}^{-} \in \mathbb{R}^V$):** Vết lưu giữ ký ức suy giảm theo thời gian (Trace/Decay memory):
        $$\mathbf{a}^{-}_i(t) \leftarrow (1 - \lambda_{\text{decay}}) \mathbf{a}^{-}_i(t-1) + \gamma_{\text{store}} \mathbf{a}^{+}_i(t)$$
    *   **Đánh thức thông qua "Nơ-ron Mẹ" (Mother Node Routing):** Khi chọn lựa nơ-ron để liên kết hoặc kích hoạt, nơ-ron Dương ưu tiên đánh thức nơ-ron Mẹ (Mother node - nơ-ron đã cùng với nó đồng phát sinh ra nơ-ron con/khái niệm trước đó) có giá trị Âm lớn nhất, phục dựng lại toàn bộ mạch ký ức liên quan một cách chính xác mà không bị bleed sang các chủ thể khác có liên kết phẳng yếu hơn.

### C. Cơ chế kiểm soát "Nơ-ron nhẹ" & Tính trơ thích ứng (Light Neurons & Adaptive Habituation)
*   **Nguyên lý cốt lõi (Note 04/01/2011):** "Nơ-ron nhẹ" là các nơ-ron có cực kỳ nhiều liên kết nhưng đa phần rất yếu (các stop words như *và, thì, là, mà, nhưng*...). Chúng có xu hướng bão hòa nhanh, dễ bị "trơ" (adaptation) khi xuất hiện dày đặc để tránh nhiễu thông tin (spam), nhưng có thể lấy lại độ nhạy trong các ngữ cảnh đặc biệt (như nơ-ron "độ sáng" trơ ban ngày, nhạy bén ban đêm).
*   **Giải pháp mô hình hóa v2:** Chúng tôi áp dụng cơ chế **Dynamic Habituation (Tính trơ động)** cho các nơ-ron có độ phủ (in-degree/out-degree) cực cao:
    *   Mỗi nơ-ron có một ngưỡng trơ thích ứng $\mathbf{H}_i(t) \in [1.0, H_{\max}]$:
        $$\mathbf{H}_i(t) \leftarrow \mathbf{H}_i(t-1) + \beta_{\text{fatigue}} \times \mathbf{Activation}_i(t) - \beta_{\text{recovery}}$$
    *   Điểm số lan truyền hoặc kích thích thực tế sẽ bị suy giảm bởi tỉ lệ trơ này:
        $$\mathbf{Activation}_{\text{effective}}(i) = \frac{\mathbf{Activation}_{\text{raw}}(i)}{\mathbf{H}_i(t)}$$
    *   Nhờ đó, các từ stop words xuất hiện liên tiếp sẽ tự động bị dập tắt (habituate) nhanh chóng, ngăn chặn việc mô hình bị kẹt vào các vòng lặp sinh từ vô nghĩa (*là là là*, *và và*), đồng thời tự hồi phục độ nhạy bén khi chuyển sang ngữ cảnh trống mới.

### D. Cửa sổ Delta Ping-Pong động (Delta Sliding-Window Binding)
*   **Nguyên lý cốt lõi (Note 13/11/2015 - Máy 7 màu):** Sự hình thành liên kết tạm thời (LKT) và nơ-ron tạm thời (NRNT) diễn ra trong một cửa sổ thời gian trượt (Delta Window). Các nơ-ron trong cửa sổ liên tục gửi tín hiệu Ping và Pong đến nhau. Khi đạt đủ 3 lần Ping-Pong, chúng liên kết bền vững thành liên kết bền (LKB) và nơ-ron ổn định (NO).
*   **Giải pháp mô hình hóa v2:** Chúng tôi thay thế cơ chế parse frame tĩnh bằng một hàng đợi trượt thời gian thực (Delta Queue) lưu trữ các token mới kích hoạt. Khi hai token $A$ và $B$ cùng nằm trong hàng đợi, chúng thực hiện cơ chế Ping-Pong đếm số lần co-firing. Sự tích lũy đủ 3 nhịp Ping-Pong đóng vai trò là điều kiện dẻo hóa khớp thần kinh thực tế (STDP) để lưu ma trận kề `REINFORCED`, đưa tri thức tạm thời vào trí nhớ dài hạn.

---
*Tài liệu lưu trữ định hướng phát triển NN-AIVIE v2.*
