"""§TK_089 — TEST loạt prompt THỰC TẾ qua trợ lý của ta (BM25F + QLoRA-3B, có nguồn).

    .venv/bin/python scripts/test_assistant_real.py 2>&1 | tee out/test_assistant_real.log
"""
import sys, time
sys.path.insert(0, "scripts")
from assistant_ours import Assistant

QS = [
    "Sông Mê Kông chảy qua những quốc gia nào?",
    "Chó nhà có nguồn gốc và đặc điểm gì?",
    "Biến đổi khí hậu là gì và do đâu?",
    "Sao Hỏa có mấy vệ tinh và vì sao có màu đỏ?",
    "Thủ đô của Nhật Bản là thành phố nào?",
    "Tế bào là đơn vị gì của sự sống?",
    "Trí tuệ nhân tạo là lĩnh vực nghiên cứu về cái gì?",
    "Việt Nam nằm ở khu vực nào và giáp những nước nào?",
    "Mặt Trời là gì?",
    "Hà Nội là thành phố như thế nào?",
]


def main():
    ast = Assistant()
    print(f"\n{'#'*74}\n# TEST {len(QS)} PROMPT THỰC TẾ — trợ lý của ta (BM25F + QLoRA)\n{'#'*74}", flush=True)
    tot = 0.0
    for q in QS:
        t = time.time(); r = ast.answer(q, k=3); dt = time.time() - t; tot += dt
        print(f"\n{'='*74}\n❓ {q}   ({dt:.1f}s · grounded={r['grounded']})\n{'-'*74}", flush=True)
        print(f"💬 {r['answer']}", flush=True)
        print(f"📚 {r['src'][0][:90]}…", flush=True)
    print(f"\n>>> {len(QS)} câu · trung bình {tot/len(QS):.1f}s/câu", flush=True)
    print("### hết", flush=True)


if __name__ == "__main__":
    main()
