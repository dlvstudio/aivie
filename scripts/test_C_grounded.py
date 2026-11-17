"""§TK_077 — CÓ ngữ cảnh chữa được BỊA không? So cùng câu: KHÔNG-ctx vs CÓ-ctx.
Đưa đoạn thông tin THẬT vào prompt (ý tưởng RAG, dùng kho của TA).

    .venv/bin/python scripts/test_C_grounded.py 2>&1 | tee out/test_C_grounded.log
"""
import time
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

BASE = "mlx-community/Qwen2.5-3B-Instruct-4bit"
ADAPTER = "adapters/vi_qlora_div400"
SAMPLER = make_sampler(temp=0.3, top_p=0.9)
LOGITS = make_logits_processors(repetition_penalty=1.2, repetition_context_size=40)

MARS = ("Sao Hỏa hay Hỏa Tinh là hành tinh thứ tư ở Hệ Mặt Trời và là hành tinh đất đá ở xa "
        "Mặt Trời nhất, với bán kính bé thứ hai. Sao Hỏa có màu cam đỏ do bề mặt được bao phủ "
        "bởi lớp vụn sắt(III) ôxít, nên còn gọi là 'hành tinh đỏ'. Khí quyển Sao Hỏa khá mỏng "
        "với thành phần chính là cácbon điôxít. Ở hai cực Sao Hỏa là lớp băng làm bằng nước. "
        "Sao Hỏa có hai vệ tinh là Phobos và Deimos.")

DAYNIGHT = ("Trái Đất tự quay quanh trục của nó một vòng mất khoảng 24 giờ. Nửa Trái Đất đang "
            "quay về phía Mặt Trời được chiếu sáng nên là ban ngày; nửa còn lại quay đi khỏi "
            "Mặt Trời, không nhận được ánh sáng nên là ban đêm.")

RICE = ("Cách nấu cơm bằng nồi cơm điện: vo gạo 2-3 lần cho sạch, đổ nước theo tỉ lệ khoảng "
        "1 phần gạo 1,2 phần nước, cho nồi vào và bấm nút Cook, khi nồi tự chuyển sang Warm thì "
        "ủ thêm 10 phút cho cơm chín đều rồi xới tơi.")

# (nhãn, câu hỏi, ngữ_cảnh|None)
CASES = [
    ("NGÀY/ĐÊM — KHÔNG ctx", "Giải thích cho học sinh lớp 5: tại sao ban ngày trời sáng còn ban đêm trời tối?", None),
    ("NGÀY/ĐÊM — CÓ ctx", "Giải thích cho học sinh lớp 5: tại sao ban ngày trời sáng còn ban đêm trời tối?", DAYNIGHT),
    ("NẤU CƠM — KHÔNG ctx", "Liệt kê các bước nấu một nồi cơm ngon bằng nồi cơm điện.", None),
    ("NẤU CƠM — CÓ ctx", "Liệt kê các bước nấu một nồi cơm ngon bằng nồi cơm điện.", RICE),
    ("SAO HỎA — CÓ ctx (hỏi khó)", "Sao Hỏa có mấy vệ tinh, tên là gì? Vì sao nó có màu đỏ?", MARS),
    ("SAO HỎA — CÓ ctx (bẫy)", "Sao Hỏa có sự sống của con người không? Chỉ trả lời dựa trên đoạn đã cho.", MARS),
]


def main():
    t = time.time()
    model, tok = load(BASE, adapter_path=ADAPTER)
    print(f"# Hệ C grounded test (nạp {time.time()-t:.0f}s)\n", flush=True)
    for tag, q, ctx in CASES:
        if ctx:
            user = (f"Dựa DUY NHẤT vào thông tin sau, hãy trả lời. Nếu thông tin không đủ, "
                    f"nói 'không có trong tài liệu'.\n\n[Tài liệu]\n{ctx}\n\n[Câu hỏi] {q}")
        else:
            user = q
        prompt = tok.apply_chat_template([{"role": "user", "content": user}],
                                         add_generation_prompt=True)
        t = time.time()
        out = generate(model, tok, prompt=prompt, max_tokens=220, verbose=False,
                       sampler=SAMPLER, logits_processors=LOGITS)
        print(f"\n{'='*72}\n▶ [{tag}]\n{'-'*72}", flush=True)
        print(out.strip(), flush=True)
    print("\n### hết", flush=True)


if __name__ == "__main__":
    main()
