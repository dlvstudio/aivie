# TK_028 — Data SỬ VIỆT NAM curated (nhà-sử-học soạn)

| Field | Value |
|-------|-------|
| **ID** | TK_028 |
| **Status** | 🟢 DONE — 30 bài, 173 câu, train vào v7; guard 86/88/67 giữ; sử answerable sạch. "Triều Lý" vẫn nợ (retrieval, không phải data). |
| **Nguồn** | Tác giả: "đóng vai nhà sử học, tạo data ~30 bài, train vào" — bổ khuyết wiki-dump thưa/nhiễu. |

## Minh bạch
Nội dung **AI-soạn** từ kiến thức sử phổ thông — bám sự kiện/ngày LỚN, ổn định (Nhà Lý 1009, dời đô
1010, Văn Miếu 1070, sông Như Nguyệt 1077, Bạch Đằng 938/1288, Lam Sơn 1418-1428, Ngọc Hồi Đống Đa
1789…). Tránh chi tiết mơ hồ. Dùng nghiêm túc nên đối chiếu nguồn; sử VN cơ bản thì độ tin cao.

## Code
- `scripts/gen_history_data.py`: 30 bài (triều đại/vua/sự kiện/di tích), title-anchored →
  `data/real_vi/history_vi.jsonl` (173 câu, format statements_doc).
- `scripts/build_history_checkpoint.py`: v6 → `ingest_statements`(priority **0.85**, curated — lấn
  wiki 0.6) + `ingest_entities`(gộp mọi nguồn title) → **network_v7_history.pkl** (parents +173,
  3083 entity). `viz_default_checkpoint`→v7.

## ĐO (`out/tk028_history.log`)
- ✅ **"Nhà Lý có gì" → "Nhà Lý là một triều đại phong kiến… 1009-1225 . Lý Thường Kiệt đánh
  Chiêm/Tống"** — data curated priority-cao THẮNG, mở đầu bằng ĐỊNH NGHĨA sạch.
- ✅ "Văn Miếu"(trường ĐH đầu tiên,1076), "Lê Lợi"(Lam Sơn,1427,Hậu Lê), "Quang Trung"(thống nhất),
  "Ngô Quyền"(thắng Nam Hán): đáp curated sạch.
- ✅ **Guard QA 86 / Para 88 / Syn 67 — GIỮ TRỌN** (priority cao không phá curated grade).
- ❌ **"Triều Lý đã làm gì" VẪN Korea** — kể cả có fact "triều Lý dời đô" sạch 0.85. Vì query khớp
  Korea fact qua **"Lý"+"đã"+"làm"** (generic, stimulus cao) > cụm "triều Lý". **Nợ RETRIEVAL, không
  phải nợ DATA** — thêm data sạch cũng không cứu vì tín-hiệu-phân-biệt "triều Lý" yếu ở tầng khớp token.

## Kết luận
Data sử curated = thắng rõ cho Q&A lịch sử (Nhà Lý/Văn Miếu/Lê Lợi/Quang Trung/Ngô Quyền sạch), guard
giữ. Khẳng định "Triều Lý" là **nợ retrieval đa-tầng** (sau 4 lần thử: margin/resonance/fold-cluster/
data-priority) — token "Triều"+"Lý"+"đã làm" generic để Korea fact thắng; cần hướng token-disambiguation
sâu hơn, không phải data. Ghi sổ, không ép.
