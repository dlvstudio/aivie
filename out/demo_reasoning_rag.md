# Demo: Reasoning-model + RAG trên M4 (không cần GPU khủng)

**Kiến trúc:** gemma4:26b (reasoning, Ollama, ~66 tok/s trên M4) + RAG BM25 viwiki (grounding của ta) + verify.
Ba đột phá chạy-local: test-time compute (model NGHĨ lâu) + RAG (nguồn thật) + neo/kiểm chứng.

## Kết quả (single-shot vs reasoning+RAG)

**1. «Tế bào vi khuẩn phân chia mỗi 20 phút, sau 2 giờ có bao nhiêu?» (BÀI TOÁN — mạng 54M của ta ra RÁC)**
→ reasoning+RAG: **64 tế bào**, có LỜI GIẢI từng bước (120/20 = 6 lần → 2^6 = 64), CÓ TRÍCH "Nguồn 1, 3" cho dữ kiện "mỗi lần chia → 2 tế bào". (nghĩ 444 từ)

**2. «Sông Hồng hay Mê Kông dài hơn?»**
→ reasoning+RAG: **Mê Kông dài hơn**, và THÀNH THẬT: "các nguồn không có chiều dài sông Hồng → dùng kiến thức chung". (grounding-awareness — biết khi nào nguồn thiếu)

**3. «Tại sao lá cây màu xanh lục?»** → diệp lục (chlorophyll), có nguồn "màu xanh lá cây".
**4. «Hà Nội hay HCM đông dân hơn?»** → HCM, honest về nguồn.

## Ý nghĩa
- **Suy-luận THẬT trên M4**: gemma nghĩ 400-600 từ → giải toán/logic đúng (2^6=64 có lời giải). Mạng 54M tự-train KHÔNG bao giờ làm được ("một cộng một bằng" → rác).
- **RAG + verify**: truy nguồn tiếng Việt thật, model tự nhận khi nguồn thiếu + trích nguồn.
- **100% local, không cloud GPU**: gemma 26B (15GB) chạy ngay trên M4 Pro 24GB.

## Đây là con đường "LLM-quality trên M4"
```
Model-suy-luận local (gemma/Qwen-reasoning, MLX/Ollama)
  + RAG/grounding CỦA ANH (viwiki, kNN, verify)
  + QLoRA (tùy chọn: chuyên tiếng Việt/lĩnh vực, ~90 phút trên M4)
= trợ lý tiếng Việt chất-lượng-cao, của riêng anh, không cần GPU khủng.
```
Tái dùng TOÀN BỘ công đã xây (RAG, data pipeline, grounding). `scripts/rag_reasoning.py`.
