# 05 — TẦNG NEO: NƠ-RON = BẤT BIẾN, ĐỒNG HỒ ĐA TẦNG KHI SINH

> Nguồn: trao đổi tác giả ↔ Claude, 2026-07-03. Tác giả đặt bài: *"để sinh từ/văn phong
> chuẩn cần những nơ-ron nổi lên đóng vai trò ĐIỂM NEO (nơ-ron câu/đoạn/episode); muốn
> train tự học ra các THỂ LOẠI nơ-ron khác — cảm xúc, chủ đề (như head của LLM) — để làm
> neo."* — suy nghĩ thấu đáo, out of the box, phương hướng khả thi trọn vẹn.

## 1. Điểm neo, định nghĩa chính xác

Neo = **biến CHẬM trong trạng thái kích hoạt H**: cháy bền qua nhiều bước sinh, thiên vị
mọi readout bên dưới. Đối chiếu: vỏ não phân tầng theo cửa sổ thời gian (cảm giác nhanh →
trán chậm); cảm xúc = neuromodulation (tín hiệu chậm nhất, tắm toàn mạng); feature
chủ đề/sắc thái trong LLM sáng bền suốt đoạn. `agenda` của TK_009 chính là **neo-câu
nguyên thủy** ta đã dùng mà chưa gọi tên.

## 2. Hiện trạng: 4 loại neo ĐÃ CÓ — đều thiết kế tay

câu = ParentRecord (v3) · đoạn/bài = t1 (TK_010) · thực thể = Theme (P1) · khung = Hub
(0.5.14, + Style doc 04). Câu hỏi thật: **nguyên lý để mạng tự đẻ THỂ LOẠI nơ-ron mới**,
không phải ta liệt kê từng loại.

## 3. NGUYÊN LÝ THỐNG NHẤT: NƠ-RON = MỘT BẤT BIẾN

Mọi loại nơ-ron hiện có đều theo một công thức: *tập tín hiệu đồng-kích-hoạt lặp lại,
BẤT BIẾN qua một phép biến đổi*. **Loại bất biến quyết định loại nơ-ron:**

| Bất biến qua… | Cái còn lại | Loại nơ-ron | Trạng thái |
|---|---|---|---|
| vị trí trong câu | cụm lặp | trung gian Re-Pair | ✅ |
| filler | khung câu | hub | ✅ |
| khung | nội dung + sắc thái | style | 🟡 doc 04 |
| **câu trong cùng bài** | **chủ đề** | **topic neuron** | ⬜ |
| **chủ đề** (xuất hiện mọi topic, phân cực theo bài) | **đánh giá/cảm xúc** | **emotion neuron** | ⬜ |
| bài cùng người viết | giọng cá nhân | voice neuron | ⬜ xa |

→ Xây MỘT **máy chiếu-bất-biến** (invariance projector) trên A thay vì hand-design từng
loại. Đối ngẫu với doc 03: bên đó chiết **W** (cách nhìn — cấu trúc, offline); bên này
chiết **neo** (cái-còn-sáng — trạng thái, runtime). Head và anchor = hai mặt của cùng
phép nhân tử hóa.

## 4. Hai loại mới khả thi ngay

**Topic neuron** — cluster episode theo tập bigram hiếm chung (tái dụng `phrase_assoc` /
SVD episode×bigram = hạ tầng T0+TK_008). Mỗi cluster = một nơ-ron chủ đề; membership
episode→topic có trọng số. Ứng dụng neo: (a) coherence khi sinh — fact rác "tập trung
quân đội" khác topic với bài đang sinh → nén ở TẦNG ĐÚNG (thay các gate mặt-chữ);
(b) truy hồi: query → topic → bó episode/fact hẹp.

**Emotion neuron** — từ đánh giá TỰ LỘ DIỆN bằng chữ ký thống kê: *bất biến qua chủ đề*
("tuyệt vời" ở bài nấu ăn lẫn thể thao lẫn công nghệ) *nhưng phân cực trong bài* (đứng
cùng phe các từ dương khác trong cùng episode). Cực tính nổi từ đồng-episode — KHÔNG
hardcode danh sách từ cảm xúc. **Điều kiện data nói thẳng:** cần corpus review/bình luận
phân cực (HF tiếng Việt sẵn — vd bộ sentiment/review); vi-alpaca hiện trung tính, tín
hiệu sẽ mỏng nếu chưa nạp.

## 5. Cơ chế NEO lúc sinh: ĐỒNG HỒ ĐA TẦNG

```
H_slow  (persona / emotion / topic)  — mồi đầu phiên; decay ≈ 0; G3 cập nhật (mirroring)
H_mid   (t1 + nơ-ron câu hiện hành)  — cập nhật tại ranh giới "." (re-prime từ segment vừa phát)
H_fast  (token, decay λ)             — như hiện tại
a_final = Π (sàn + kênh đọc từ từng tầng)     ← Hadamard, đúng phong cách A3
```

- Mỗi neo active đóng góp một kênh đọc = readout từ nơ-ron neo (trường axon — nhất quán
  với V trong doc 03).
- Ranh giới "." là sự kiện cập nhật H_mid (đã có boundary detection từ TK_009 vòng 4).
- agenda hiện tại = trường hợp riêng của H_mid; a_style (doc 04) = trường hợp riêng của
  H_slow — đề xuất này là TỔNG QUÁT HÓA chúng thành một máy, không thêm cơ chế lạ.
- Neo bắt lỗi theo tầng: neo-câu giữ segment không trôi (có rồi), neo-topic giữ mạch
  giữa segment, neo-style/emotion giữ giọng — thay thế dần các gate mặt-chữ vá rời rạc.

## 6. LỘ TRÌNH TRỌN VẸN (ticket riêng sau TK_013 ISG — anchor neurons; xem `TK_013` = phân rã link ngang)

| Phase | Nội dung | Đo bằng | Chi phí |
|---|---|---|---|
| **N0a — topic rẻ** | SVD/cluster episode×bigram-hiếm trên `_t3` (~10k episode) → đọc tay cluster chủ đề | ≥5 cluster có nghĩa khi đọc tay (nấu ăn / sức khỏe / công nghệ…) | 1 buổi |
| **N0b — emotion rẻ** | Probe chữ ký "bất biến-chủ-đề + phân-cực-bài" trên corpus hiện có; nếu mỏng → ingest 1 bộ review VI (HF) rồi probe lại | tách được ≥2 cực khi đọc tay | 1 buổi (+ingest) |
| **N1 — AnchorNeuron** | Dataclass thống nhất `{kind, key, members, signature}`; kind = phép bất biến sinh ra nó; serialize; topic+emotion materialize | roundtrip + thống kê membership | nhỏ |
| **N2 — đồng hồ đa tầng** | H_fast/mid/slow trong generate_essay/A3; kênh `a_anchor` (tổng quát hóa a_style); cập nhật H_mid tại "." | essay: coherence giữa segment (đo ca "quân đội" hết lọt KHÔNG cần gate mặt-chữ) + guard + style/meaning giữ | vừa |
| **N3 — G3 anchoring** | Neo học/mirror qua phiên (P3); persona ghim | style_consistency + phiên dài | lớn, cuối |

**Acceptance nguyên tắc:** mỗi loại neo mới phải (1) TỰ NỔI từ data (đọc tay cluster có
nghĩa), (2) khi làm kênh neo thì thay được ÍT NHẤT một gate mặt-chữ hiện có mà metric
không tụt — neo là để GỠ vá, không phải thêm vá.

**Rủi ro ghi trước:** (1) emotion cần data phân cực — không có thì N0b âm tính trung
thực, ghi sổ chờ ingest; (2) đa tầng đồng hồ thêm siêu tham số (decay/floor mỗi tầng) —
mỗi kênh phải ablate được về 0 (truyền thống config hiện tại); (3) nguy cơ "neo đè nghĩa"
— topic/emotion nén quá tay làm mất fact; metric meaning-coverage phải giữ nguyên vẹn
như style (doc 04 §5 rủi ro 3).

**Luật 0:** mọi neo nổi từ thống kê bất biến trên data; không danh sách từ cảm xúc/chủ
đề; frame `Persona:`/mồi neo thuộc frame protocol được phép.

## 7. Liên kết chéo

- doc 03 (Q/K heads) — đối ngẫu cấu trúc/trạng thái; head topic và neo topic dùng chung
  phép nhân tử hóa episode×bigram.
- doc 04 (style) — a_style = trường hợp riêng đầu tiên của kênh neo H_slow.
- TK_008 `phrase_assoc` (máy đồng-episode), TK_009 (boundary + kênh Hadamard), TK_010
  (t1/fact_links = neo-bài và xương H_mid) — nền đã sẵn.
- P3 SFT/G3 — máy củng cố dùng chung cho N3.
