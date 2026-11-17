"""§TK_077 — TEST NĂNG LỰC hệ C (QLoRA Qwen-3B `vi_qlora_div400`) do TA train.
Bao quát: lập trình, dịch, email, soạn thảo, tóm tắt, toán, giải thích, liệt kê, sáng tác.
Decoding chống-lặp: temp0.3 top_p0.9 repetition_penalty1.2.

    .venv/bin/python scripts/test_C_caps.py 2>&1 | tee out/test_C_caps.log
"""
import time
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

BASE = "mlx-community/Qwen2.5-3B-Instruct-4bit"
ADAPTER = "adapters/vi_qlora_div400"
SAMPLER = make_sampler(temp=0.3, top_p=0.9)
LOGITS = make_logits_processors(repetition_penalty=1.2, repetition_context_size=40)

TESTS = [
    ("LẬP TRÌNH 1", "Viết một hàm Python kiểm tra một số có phải số nguyên tố không. Kèm ví dụ gọi hàm.", 320),
    ("LẬP TRÌNH 2", "Viết hàm Python nhận vào một danh sách số và trả về danh sách đã sắp xếp tăng dần, không dùng hàm sorted có sẵn.", 320),
    ("LẬP TRÌNH 3", "Giải thích đoạn code sau làm gì:\n\ndef f(n):\n    return n if n < 2 else f(n-1) + f(n-2)", 220),
    ("DỊCH V→A", "Dịch sang tiếng Anh: Trí tuệ nhân tạo đang thay đổi cách con người làm việc và học tập mỗi ngày.", 120),
    ("DỊCH A→V", "Dịch sang tiếng Việt: The committee will review all applications and announce the results by the end of next week.", 120),
    ("EMAIL", "Viết email gửi khách hàng xin lỗi vì giao hàng trễ 3 ngày và đề nghị giảm giá 10% cho đơn tiếp theo.", 320),
    ("SOẠN THẢO", "Viết một đoạn mô tả sản phẩm hấp dẫn cho một chiếc bình giữ nhiệt inox 500ml, giữ nóng 12 giờ.", 260),
    ("TÓM TẮT", "Tóm tắt đoạn sau thành 2 câu: Biến đổi khí hậu là sự thay đổi lâu dài về nhiệt độ và các hình thái thời tiết. Nguyên nhân chính từ giữa thế kỷ 20 là hoạt động của con người, đặc biệt việc đốt nhiên liệu hóa thạch làm tăng khí nhà kính trong khí quyển, khiến Trái Đất nóng lên và gây ra các hiện tượng thời tiết cực đoan.", 160),
    ("TOÁN", "Một bể chứa 2400 lít nước. Mỗi giờ rút ra 150 lít. Hỏi sau bao nhiêu giờ thì bể cạn? Trình bày các bước.", 240),
    ("GIẢI THÍCH", "Giải thích ngắn gọn cho học sinh lớp 5: tại sao ban ngày trời sáng còn ban đêm trời tối?", 220),
    ("LIỆT KÊ", "Liệt kê 5 bước để nấu một nồi cơm ngon bằng nồi cơm điện.", 240),
    ("SÁNG TÁC", "Viết một bài thơ 4 câu về Hà Nội mùa thu.", 160),
]


def main():
    t = time.time()
    model, tok = load(BASE, adapter_path=ADAPTER)
    print(f"# Hệ C = Qwen-3B + adapter '{ADAPTER}'  (nạp {time.time()-t:.0f}s)", flush=True)
    print(f"# decoding: temp0.3 top_p0.9 rep_penalty1.2\n", flush=True)
    for tag, p, mx in TESTS:
        msgs = [{"role": "user", "content": p}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
        t = time.time()
        out = generate(model, tok, prompt=prompt, max_tokens=mx, verbose=False,
                       sampler=SAMPLER, logits_processors=LOGITS)
        print(f"\n{'='*72}\n▶ [{tag}] ({time.time()-t:.0f}s)  {p[:60]}\n{'-'*72}", flush=True)
        print(out.strip(), flush=True)
    print("\n### hết", flush=True)


if __name__ == "__main__":
    main()
