# 07 — SINH TOKEN-BY-TOKEN: LAN TOẢ VÔ HƯỚNG vs ATTENTION CÓ HƯỚNG (đo thực nghiệm)

> Nguồn: phiên làm việc tác giả ↔ Claude 2026-07-03/04. Nối tiếp **doc 03** (Q/K từ đồ thị) —
> lần này **ĐO THỰC NGHIỆM** thay vì chỉ thiết kế. Toàn bộ code probe đã xoá sau khi ghi lại
> (không cần giữ); tài liệu này là kết quả để nghiên cứu tiếp.

## 0. Bối cảnh & câu hỏi

Sau khi hợp nhất sinh về **một chế độ A3 token-by-token** (README 0.6.9 / v4 §Trụ4 P1-α), tác giả
đặt vấn đề cốt lõi:

> *"Việc gom fact hay không KHÔNG thể quyết ở tầng plan (`multi`). Mạng phải **tự lan toả → đi qua
> các bước chú ý → bùng nổ → gom nhóm nơ-ron → chọn lọc** rồi mới chọn token kế. Làm sao sinh đủ
> tri thức nếu chỉ dựa vào 1 hay nhiều fact?"*

Tức: nghi ngờ chính cơ chế **agenda** (danh sách token nội dung chọn-sẵn-lúc-truy-hồi) đang giới
hạn tri thức và làm việc sinh "un-LLM-like". Câu hỏi thí nghiệm: **có thể để mạng tự lan toả trên
toàn đồ thị mỗi bước → chọn token, KHÔNG cần agenda cứng không?**

Kỷ luật: **đo trước, không đại phẫu mù**. Chuỗi 3 probe rẻ, mỗi cái bác/khẳng định một giả thuyết.

## 1. Cơ chế A3 hiện tại (để biết đang gỡ cái gì)

`generate_autoregressive` mỗi bước: `a_final = (sàn+micro) ⊙ (sàn+macro) ⊙ (sàn+entity) ⊙ (sàn+discourse)`
- **micro** = `W_seq[w0]` (bigram CÓ HƯỚNG — token nào theo sau token nào).
- **macro** = `spreading_activation_read(seed = agenda CÒN THIẾU)` — lan toả VÔ HƯỚNG qua cạnh readout.
- **index-backbone**: token agenda kế theo **thứ tự nguồn**, nếu đạt được từ w0 → **ép điểm = max×(1+margin)** → luôn thắng. ⇒ đây là "xương sống" khiến A3 **replay đáp truy hồi gần verbatim** (`vb_seg≈96%`).
- **EOS** ép bật khi agenda phủ đủ ⇒ **độ dài = độ dài agenda**.

Nghĩa là: agenda quyết "nói gì" + độ dài; index-backbone quyết thứ tự (cứng). Spreading chỉ được
dùng để *kéo về agenda*, không phải để mạng tự bung.

## 2. Probe 1 — LAN TOẢ VÔ HƯỚNG THUẦN (bỏ agenda) → SỤP ĐỔ HUB

Thử: mỗi bước seed spreading từ **toàn ngữ cảnh** (w0 + lịch sử decay + entity register) ra cả đồ
thị, `micro⊙macro⊙entity` + G4 + repetition, EOS tự nổi, **KHÔNG agenda / KHÔNG index-backbone**.
Checkpoint dày nhất `grade_v3_big_a2_t3_isg.pkl` (vocab 34.5k, 44.6k cha).

**Kết quả:** mọi prompt → `. , . , . , …` (uniq 0.03). Sụp đổ hoàn toàn về dấu câu.

Trace từng bước (w0=`Đáp:`) chỉ rõ nguyên nhân:

| Kênh | Top | Bản chất |
|---|---|---|
| micro (W_seq) | `Tôi=2.05  Xác=1.3  Mở=1.15  Một=1` | **CÓ content** (token mở đáp thật) |
| macro (spread) | `.=572  ,=62.7  và=49.5  của=26.6` | **hub `.` đè gấp 9×** |
| combined | `.=0.178  Tôi=0.008` | `.` thắng **17×** |

**Nguyên nhân (dứt điểm):** lan toả VÔ HƯỚNG chấm theo độ mạnh cạnh readout; hub (`.` `,` `và` `của`
— co-occur với **mọi thứ**) có out-degree cao nhất → tích activation lớn nhất **bất kể ngữ cảnh**.
Hadamard để hub *veto* content. → **bác bỏ giả thuyết "bỏ agenda, spread tự do"**.

**Vá thử — IDF-gate (nén token df cao): PHẢN TÁC DỤNG.** Nhân macro với `_idf(t)`: `.` từ 572 → 6690
(còn to hơn!). Vì dấu câu **không nằm trong bảng df** của parent store (bị loại khỏi content) →
`_idf(".")` trả điểm CAO → gate *khuếch đại* hub. Bài học: IDF của retrieval không áp được cho hub
bề mặt.

## 3. Probe 2 (A0, doc 03) — Q·K TỪ SVD CỦA W_seq CÓ HƯỚNG

Insight từ Probe 1: **content nằm ở cạnh CÓ HƯỚNG (micro/W_seq)**; lan toả vô hướng chỉ là nhiễu hub.
Doc 03: SVD ma trận **PPMI CÓ HƯỚNG** của W_seq (`M≈UΣVᵀ`, d=96) → `Wq=U√Σ`, `Wk=V√Σ` →
`q_i·k_j ≈ PPMI(i→j)` — bất đối xứng, phá thế hub, tổng quát hoá cặp chưa đồng hiện.

Đo trên `_t3_isg` (270k cạnh W_seq, PPMI nnz 224k, held-out 10%):

**ĐO 1 — held-out bigram rank:** median **1033**/34.5k (top 3%), recall@10 **10.6%**. Hơn thô (∞ với
cặp chưa thấy) nhưng **yếu** — successor-đúng thường chôn ~1000 sâu.

**ĐO 2 — hub-domination (q·k vs thô):** *HỖN HỢP*

| w0 | q·k top | |
|---|---|---|
| `.` | Bạn, Tôi, Nếu, Đây, Chúng (KHÔNG có `.`) | ✅ **nén hub** → mở câu |
| `Nhà` | **Nguyễn, Trần, Lê, Ngô, Lý, Đinh** | ✅✅ **paradigm** — cả họ triều đại (thô: xuất/thờ/nước = nghĩa khác) |
| `Trần` | Văn, Xuân, Ngọc, Quang (tên đệm) | ✅ |
| `Đáp:` | Gan, Não, Vitamin, Pi (đồng hạng) | ❌ **nhiễu token hiếm** |
| `,` | TCDD, router, Brahmaputra (đồng) | ❌ **nhiễu** |
| `thắng` | tại, thuộc, trở, của (giới từ) | ~ generic (thô: lợi/quân/trận = content hơn) |

**ĐO 3 — khuôn (thành phần SVD):** top `σ=231` = **hub/hư-từ** (`và , là :` → `, . và`); có vài khuôn
nghĩa thật: `năm→2007/2018/1905` (khuôn năm), `Đáp:/:→Lựa/Ngày/Bắt` (khuôn mở câu). Nhưng khuôn nghĩa
KHÔNG phải hướng trội.

**Verdict A0:** ý tưởng attention-có-hướng **CÓ tín hiệu thật** (Nhà→triều đại là paradigm-generalization
nổi từ SVD thuần — thứ spreading vô hướng lẫn bigram thô đều trượt; `.`→mở-câu = nén hub). **NHƯNG**
sụp thành nhiễu trên token tần-số-cao (`Đáp:`, `,` — chính là token *lái* chuyển tiếp lúc sinh), và
held-out yếu. **Không phải drop-in.**

## 4. Probe 3 (A1) — BLEND raw W_seq ⊕ q·k, + KÉO-TOPIC CÓ HƯỚNG

Doc 03 A1: raw chắc ở token thường + q·k gánh tổng-quát-hoá → **blend bù lỗ hổng của nhau**.
`blend = norm(raw) + w·norm(relu(q·k))`, `w=0.6`.

**ĐO A — blend có bù không? CÓ ✅:**

| w0 | BLEND top | |
|---|---|---|
| `Đáp:` | Tôi, Xác, Một, Mở, Hãy | ✅ content từ raw, **hết nhiễu** Gan/Pi |
| `,` | chúng, họ, và, ông, nhà, nó | ✅ **hết nhiễu** Thucydides/router |
| `Nhà` | Nguyễn, Lê, Trần, Lý (boost lên top) | ✅ generalization từ q·k |
| `.` | `.` vẫn top | ⚠ raw `.`=65 quá lớn, blend không nén nổi |

→ blend **xoá nhiễu function-token của q·k** (raw thắng ở đó) **và** boost generalization (Nhà→triều
đại). Đúng doc 03.

**ĐO B — SINH có-hướng, KHÔNG agenda** (field = blend ⊙ directed-topic-pull từ entity; no agenda, no
spread vô hướng): **THẤT BẠI (nhưng tiến bộ)**

```
"Thủ đô Việt Nam là gì" → Chim có một ngôn từ Niềm nói về các nhà Nguyễn Văn nằm ở Hầm Hải thuộc…
"Nhà Trần thế nào"      → Văn Minh cũng có nằm trong những người Đức . - Nguyễn Văn Minh cũng có nằm…
   (generate() có agenda: "Thủ đô Việt Nam là Hà Nội." / "Nhà Trần đánh thắng quân Nguyên ba lần." ✓)
```

**Tiến bộ:** blend **chữa sụp-đổ-hub** — sinh ra **từ thật, ngữ pháp cục bộ trôi chảy** (hết `. , . ,`).
**Nhưng** là **word-salad trôi dạt** — lạc sang "Chim"/"người Đức", lặp, **không bao giờ trúng đáp**. Vì
`topic-pull = Σ q·k(entity)` bị nhiễu (entity register lẫn hư-từ `gì`/`là`; tổng q·k khuếch đại nhiễu),
và q·k **type-level** không chọn-lọc-theo-ngữ-cảnh được.

## 5. Thử "thay index-backbone bằng blend" (giữ agenda)

Hiện thực: cờ `ar_directed_blend` — bật thì `micro=blend(raw⊕q·k)` + **bỏ ép-cứng index-backbone**
(để agenda-boost mềm + blend quyết thứ tự); **giữ agenda** (lo "nói gì") + EOS-phủ-agenda.

Đo nhanh `grade_v3` (2k vocab):
- **Crisp QA: `base == blend`** — grounding giữ nguyên ("Bác sĩ chữa bệnh.", "Thủ đô Việt Nam là Hà Nội.").
- **Multi (câu mở, sinh dài):** blend **đổi output** (bớt cứng): `Nhà Lý mở… là Lý Thái Tổ` → `Nhà mở…
  Thái Tổ` (bớt lặp chủ ngữ NHƯNG rụng "Lý"); thêm **trôi đuôi nhẹ** ("Trần đánh thắng"). Trên đồ thị
  **thưa** grade_v3 → **hơi lộn xộn**.
- **Essay `_t3_isg` (đồ thị dày — nơi kỳ vọng blend phát huy): CHƯA đo được** — ~3 phút/bài (SỔ NỢ perf:
  AR chấm 34.5k vocab + spreading mỗi token × ~150 bước). Vòng lặp quá chậm cho phiên tương tác.

## 6. KẾT LUẬN DỨT ĐIỂM

Vòng sinh token-by-token có **HAI nhu cầu tách bạch**:

| Nhu cầu | Trạng thái | Bằng chứng |
|---|---|---|
| **CÁCH nói** (fluency cục bộ + tổng-quát-hoá) | ✅ **giải được** bằng blend(raw⊕q·k) | ĐO A + hết sụp hub, ra từ ngữ-pháp-đúng |
| **NÓI GÌ** (trúng đáp cho *câu hỏi này*) | ❌ **KHÔNG** đến từ đồ thị thuần | spread vô hướng→hub; q·k type-level→trôi dạt |

**Điều này khớp cảnh báo doc 03 §4:** q·k là **type-level** (mỗi token 1 vector cố định) — làm được
*chuyển tiếp/tổng-quát*, KHÔNG làm được *chọn-lọc-theo-ngữ-cảnh* ("câu hỏi *này* → fact *kia*"). Chọn-
đáp-theo-ngữ-cảnh là thứ **chỉ retrieval (agenda) cung cấp** — **agenda KHÔNG phải nạng vứt được**, nó
gánh phần "nói gì" mà động lực nội tại của đồ thị (không backprop) không tạo ra.

⇒ **Tầm nhìn "bỏ agenda, mạng tự lan toả quyết định nói gì" là bất khả trên substrate hiện tại** —
chứng minh bằng **2 negative** (sụp-hub + trôi-dạt). Kiến trúc đúng: **agenda lo "nói gì" (cần thiết) +
blend có-hướng lo "cách nói" (học được)**.

## 7. ĐIỂM MẠNH / YẾU / NGUYÊN NHÂN (bảng tổng)

| Cơ chế | Mạnh | Yếu | Nguyên nhân gốc |
|---|---|---|---|
| Spread vô hướng | lan rộng, đa-hop | **sụp hub** `. ,` | chấm theo out-degree → hub thắng; đối xứng (doc 03) |
| IDF-gate spread | rẻ | **phản tác dụng** | dấu câu không có df → IDF cao → khuếch đại hub |
| q·k (SVD directed) | **paradigm** (Nhà→triều đại), nén hub `.` | **nhiễu** ở function-token, held-out yếu (rank~1033) | low-rank không dựng nổi successor của token df-cao; d=96 nhỏ; data thưa |
| blend raw⊕q·k | **bù lỗ hổng** (ĐO A), fluency | không target đáp (ĐO B) | q·k type-level, không contextual |
| blend thay index-backbone | grounding giữ, bớt replay | drift trên đồ thị thưa; essay chưa đo | đồ thị thưa + agenda-boost mềm chưa đủ giữ thứ tự |

## 8. HƯỚNG NGHIÊN CỨU TIẾP (khi quay lại)

1. **Blend thay index-backbone — đo cho xong trên `_t3_isg`** (đồ thị dày): vb_seg giảm mà cov/ord giữ?
   Cần **hạ tầng perf trước** (TK_011: CSR/Numba/GPU cho spreading + readout) vì hiện ~3 phút/bài.
2. **Nén hub bề mặt đúng cách**: df CÓ tính dấu câu/hư-từ (bảng df riêng cho surface token), hoặc mask
   hub trong kênh macro — thứ IDF-retrieval không làm được.
3. **Tăng d SVD + nhiều data** để q·k bớt nhiễu ở function-token (held-out rank kỳ vọng giảm).
4. **Contextual attention thật** (vượt type-level): cần chồng-chập-H theo ngữ cảnh (docx §4) hoặc học
   `A_attn` bằng G3 (doc 03 A3) — đây mới là thứ *có thể* thay agenda, nhưng đắt.
5. **Multi-scale anchor (doc 05)**: topic/theme neuron làm H_slow để "kéo topic" thay `Σ q·k(entity)`
   (đang nhiễu) — directed topic-pull đúng phải từ neo chủ đề, không phải tổng q·k token hỏi.

## 9. Chốt cho sản phẩm (đã ship, KHÔNG phụ thuộc thread này)

Việc **hợp nhất một chế độ A3** (P1-α: `generate()`, bỏ hybrid/composed/essay-verbatim, QA giữ 100%,
Graph UI gộp) **đã xong và độc lập** với nghiên cứu directed-attention ở trên. Thread này (blend/q·k)
là **hướng cải tiến "cách nói"** cho tương lai — đã ghi đủ để hiện thực lại khi có hạ tầng perf. Toàn
bộ code probe (probe_spread / trace_spread / probe_qk_a0 / probe_qk_a1 / essay_blend_ab / directed_attn
/ cờ ar_directed_blend) **đã gỡ** sau khi ghi tài liệu này — không giữ code thử nghiệm chưa dùng.

**Liên kết:** doc 03 (thiết kế Q/K — tài liệu này là phần ĐO), doc 05 (neo đa tầng — hướng #5),
TK_011 (perf — chặn #1).
