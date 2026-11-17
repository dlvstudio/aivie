# TK_029 — Thưởng cặp token CÓ HƯỚNG (dùng hướng liên-kết đã học)

| Field | Value |
|-------|-------|
| **ID** | TK_029 |
| **Status** | 🟢 IMPLEMENTED, guard-safe (86/88/67) — order-awareness đúng nguyên tắc; "Triều Lý" giúp một phần nhưng CHƯA tip (tổ hợp coverage/specificity). |
| **Nguồn** | Tác giả nhận xét "Triều Lý đã làm gì" lọt fact có "Lý"/"Triều" NGƯỢC ("Lý…Triều Tiên"); gợi ý dùng HƯỚNG liên-kết (đã tạo lúc train là có-hướng). |

## Xác nhận: liên-kết có-hướng đã có
- W_seq (token→kế), FRAME_BRACKET (opener→closer), ISG cluster (left→right): **có hướng**.
- CO_ACTIVATION (đồng-hiện): VÔ HƯỚNG — và tầng `stimulus` khớp token-trần là **bag-of-words vô-hướng**
  → bỏ mất thông tin hướng → "Lý…Triều Tiên"(ngược) vẫn khớp query "Triều Lý".

## Probe (measure-first) — hướng TÁCH sạch
Cặp có-hướng fold "triều→Lý" CHỈ có ở 3 fact curated dynasty ("Triều Lý trải qua chín đời vua",
"triều Lý dời đô", "triều Lý mở khoa thi"). Fact TQ "Triều Kim/Triều Nguyên", Korea "Lý…Triều Tiên"
→ chứa cả 2 token nhưng KHÔNG có cặp-kề-đúng-hướng.

## Code
`rank_matches` (parent_store): tính `q_dir` = cặp fold kề của query (theo `q_seq` giữ thứ tự), weight
= min-idf hai token. Mỗi cha: nếu `match_ids` có cặp-kề khớp `q_dir` → `score *= 1 + bonus×(idf-cặp/
max_q_idf)`. Cờ `directed_bigram_bonus=0.6` (gán `ps.` ở generate). Dùng thứ-tự đã học, không rule tay.

## ĐO (`out/tk029_dir.log` + diag)
- **Guard QA 86 / Para 88 / Syn 67 — GIỮ** (guard-safe); "Nhà Lý"/sử vẫn sạch.
- Bonus có tác dụng: "Triều Lý trải qua…" từ rank-4 (0.323) → boost (0.742 ở bonus 3.0). **NHƯNG vẫn
  không #1**.

## Vì sao "Triều Lý" chưa tip (chẩn diag_dir)
Tổ hợp 3 yếu tố, không phải chỉ hướng:
1. Fact curated "Triều Lý trải qua chín đời vua" DÀI (11 token nội dung, khớp 2) → **specificity thấp
   → điểm nền thấp**.
2. Query có cặp phụ **(Lý,đã)** GENERIC → nhiều fact "…Lý đã…" (Korea/Trần) cũng được thưởng.
3. Fact "Tổ tiên nhà Trần…Lý đã đến" ngắn → coverage/specificity cao → #1 bền.
→ Bonus kéo curated lên nhưng không vượt được nền cao của fact-generic-ngắn.

## Kết luận
Ý HƯỚNG của tác giả ĐÚNG và đã cài (guard-safe, order-awareness — sửa lỗ vô-hướng của stimulus).
Giữ `directed_bigram_bonus=0.6`. Nhưng "Triều Lý" (lần thử **thứ 5**: margin/resonance/fold-cluster/
data-priority/directed-bigram) = **nợ đa-tầng bệnh-lý**: fact-đúng dài→specificity thấp + bigram-phụ
generic + fact-sai ngắn→nền cao. Không cưỡng bằng 1 lever. Ghi sổ, dừng — honest.
