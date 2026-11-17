"""§TK_092 — Chạy loạt prompt thực tế qua trợ lý của ta, XUẤT kết quả ra file .md.

    .venv/bin/python scripts/report_assistant.py --out out/ket_qua_tro_ly.md
"""
import sys, time, argparse, datetime
sys.path.insert(0, "scripts")
from assistant_ours import Assistant, BASE, ADAPTER

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/ket_qua_tro_ly.md")
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    ast = Assistant()
    rows = []
    for q in QS:
        t = time.time(); r = ast.answer(q, k=a.k); dt = time.time() - t
        rows.append((q, r["answer"], r["grounded"], dt, r["src"][0]))
        print(f"· {q[:40]} ({dt:.1f}s, g={r['grounded']})", flush=True)

    avg = sum(x[3] for x in rows) / len(rows)
    ng = sum(1 for x in rows if x[2] >= 0.9)
    L = []
    L.append("# Kết quả trợ lý AIVIE — loạt prompt thực tế\n")
    L.append(f"*Sinh lúc {datetime.datetime.now():%Y-%m-%d %H:%M} · "
             f"BM25F (285k viwiki, của ta) + QLoRA-3B (ta fine-tune) · KHÔNG LLM ngoài*\n")
    L.append(f"**Tổng quan:** {len(rows)} câu · grounded≥0.9: {ng}/{len(rows)} · "
             f"trung bình {avg:.1f}s/câu\n")
    # bảng tổng quan
    L.append("| # | Câu hỏi | grounded | Thời gian |")
    L.append("|---|---------|:--------:|:---------:|")
    for i, (q, _, g, dt, _) in enumerate(rows, 1):
        L.append(f"| {i} | {q} | {g:.2f} | {dt:.1f}s |")
    L.append("\n---\n\n## Chi tiết\n")
    for i, (q, ans, g, dt, src) in enumerate(rows, 1):
        L.append(f"### {i}. {q}\n")
        L.append(f"> **grounded {g:.2f}** · {dt:.1f}s\n")
        L.append(f"**Trả lời:**\n\n{ans}\n")
        L.append(f"**Nguồn:** {src[:160]}…\n")
    L.append("---\n")
    L.append("*grounded = tỉ lệ từ trong câu trả lời có xuất hiện ở nguồn "
             "(đo độ trung thành / chống ảo giác). 1.0 = mọi từ đều có căn cứ.*\n")
    open(a.out, "w").write("\n".join(L))
    print(f"\n[LƯU] {a.out} ({len(rows)} câu, grounded≥0.9: {ng}/{len(rows)}, {avg:.1f}s/câu)", flush=True)
    print("### hết", flush=True)


if __name__ == "__main__":
    main()
