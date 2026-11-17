"""§TK_077 — SO 3 cấu hình trên CÙNG bộ prompt đa dạng:
  BASE  = Qwen2.5-3B-Instruct-4bit (gốc)
  OLD   = + adapter train TOÀN 1 kiểu "viết đoạn văn" (task-narrowing)
  NEW   = + adapter train DATA ĐA DẠNG 12 kiểu (iter-400, val tốt nhất)

Nhấn vào các prompt TỪNG HỎNG ở OLD: dịch, email. + on-task + toán + QA.

    .venv/bin/python scripts/test_qlora_compare.py 2>&1 | tee out/qlora_compare_div.log
"""
import time, sys
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

BASE = "mlx-community/Qwen2.5-3B-Instruct-4bit"
# decoding chống lặp: temp thấp + top_p + repetition_penalty
SAMPLER = make_sampler(temp=0.3, top_p=0.9)
LOGITS = make_logits_processors(repetition_penalty=1.3, repetition_context_size=40)

CONFIGS = [("BASE", None),
           ("NEW (đa dạng)", "adapters/vi_qlora_div400")]

PROMPTS = [
    ("on-task: đoạn văn", "Viết một đoạn văn ngắn bằng tiếng Việt về sông Mê Kông."),
    ("DỊCH (từng hỏng)", "Dịch câu sau sang tiếng Anh: Hôm nay trời đẹp nên tôi đi dạo trong công viên cùng gia đình."),
    ("EMAIL (từng lan man)", "Viết một email ngắn xin sếp cho nghỉ phép 1 ngày vì lý do gia đình."),
    ("TOÁN", "Một cửa hàng bán 15 chiếc áo, mỗi chiếc giá 120000 đồng. Hỏi tổng số tiền thu được là bao nhiêu?"),
    ("QA", "Thủ đô của Nhật Bản là thành phố nào?"),
    ("GIẢI THÍCH", "Giải thích ngắn gọn quá trình quang hợp ở cây xanh."),
    ("TÓM TẮT", "Tóm tắt bằng 1 câu: Trí tuệ nhân tạo là lĩnh vực của khoa học máy tính nhằm tạo ra các hệ thống có khả năng thực hiện những nhiệm vụ thường đòi hỏi trí thông minh của con người, như nhận diện hình ảnh, hiểu ngôn ngữ và ra quyết định."),
]


def run(name, adapter):
    print(f"\n{'#'*72}\n# {name}\n{'#'*72}", flush=True)
    t = time.time()
    model, tok = load(BASE, adapter_path=adapter) if adapter else load(BASE)
    print(f"(nạp {time.time()-t:.0f}s)", flush=True)
    for tag, p in PROMPTS:
        msgs = [{"role": "user", "content": p}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
        out = generate(model, tok, prompt=prompt, max_tokens=170, verbose=False,
                       sampler=SAMPLER, logits_processors=LOGITS)
        print(f"\n▶ [{tag}] {p[:55]}", flush=True)
        print(f"  {out.strip()[:400]}", flush=True)


def main():
    for name, ad in CONFIGS:
        run(name, ad)
    print("\n### hết", flush=True)


if __name__ == "__main__":
    main()
