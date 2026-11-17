"""TK_104 §16 · HỘI THOẠI NHIỀU LƯỢT — chữa "câu đầu ổn, vài câu sau lan man".

Dữ liệu cũ gần như toàn 1 lượt (văn phòng/code/hiệu chỉnh 100%, SFT v2 ~91%) → model ít khi phải đọc lịch sử để trả lời.
Thầy (Gemma-4-31B) viết hội thoại 3–6 lượt giữa người dùng Việt và trợ lý, mỗi hội thoại theo một KIỂU nối tiếp:
sửa theo yêu cầu · hỏi sâu · tham chiếu ý trước ("cái thứ hai bạn nêu") · đổi chủ đề rồi quay lại · bổ sung thông tin dần
· đính chính khi người dùng chỉ ra lỗi. Train trên MỌI lượt trợ lý (build_example đã gắn nhãn mọi lượt "Đáp:").

    python scripts/gen_multiturn.py gen --n 9000 --workers 48 --ollama http://127.0.0.1:8000/v1
    python scripts/gen_multiturn.py build     # → data/multiturn/train.jsonl + probe.jsonl (lượt NGƯỜI DÙNG giữ riêng để đo)
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, time
import concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_office import ollama, CJK

D = "data/multiturn"
T0 = time.time()
DOMAINS = {
    "van_phong": ["soạn email rồi chỉnh sửa nhiều lần", "lập kế hoạch công việc", "viết báo cáo, thông báo", "công thức Excel và chỉnh tiếp",
                  "thao tác Word/PowerPoint"],
    "code": ["viết hàm Python rồi sửa lỗi", "giải thích code rồi tối ưu", "thêm tính năng vào đoạn code trước", "tìm lỗi chương trình"],
    "kien_thuc": ["hỏi bài học SGK rồi hỏi sâu thêm", "hỏi lịch sử / địa lý Việt Nam", "khoa học tự nhiên phổ thông", "văn học Việt Nam"],
    "toan": ["giải bài toán rồi hỏi cách khác", "bài toán nhiều bước, hỏi lại từng bước", "ôn công thức rồi làm ví dụ"],
    "doi_song": ["tư vấn nấu ăn", "lên lịch du lịch", "lời khuyên sức khoẻ thường thức", "chọn quà tặng", "học tiếng Anh"],
    "tro_chuyen": ["tán gẫu về sở thích", "chia sẻ tâm trạng sau ngày làm việc", "bàn về phim, sách"],
}
PATTERNS = ["người dùng yêu cầu SỬA kết quả trước (ngắn lại / trang trọng hơn / thêm ý)",
            "người dùng HỎI SÂU vào một chi tiết trong câu trả lời trước",
            "người dùng THAM CHIẾU ý trước bằng từ như 'cái thứ hai', 'như bạn nói ở trên', 'đoạn đó'",
            "người dùng ĐỔI CHỦ ĐỀ ở giữa rồi QUAY LẠI chủ đề đầu",
            "người dùng BỔ SUNG thông tin dần qua các lượt (lượt sau dùng thông tin lượt trước)",
            "người dùng chỉ ra một chỗ CHƯA ĐÚNG/chưa rõ và trợ lý đính chính lịch sự"]

P_GEN = """Viết một đoạn hội thoại tiếng Việt TỰ NHIÊN gồm {n} lượt giữa NGƯỜI DÙNG và TRỢ LÝ AI giỏi (lĩnh vực: {dom}; tình huống: {task}).
Kiểu nối tiếp chính: {pat}.
Yêu cầu:
- Mỗi câu trả lời của TRỢ LÝ phải bám đúng câu hỏi MỚI NHẤT và nhất quán với các lượt trước (nhớ tên, số liệu, yêu cầu đã nêu).
- Trả lời đúng, có cấu trúc khi cần (gạch đầu dòng, khối code ```), không lặp lại nguyên văn câu trả lời cũ, không lan man.
- Với câu hỏi kiến thức / trò chuyện / tư vấn: KHÔNG trả lời cộc lốc một câu — nêu thẳng ý chính trước, rồi giải thích thêm,
  ví dụ hoặc gợi ý thực tế (thường 3–6 câu); nhiều lượt (không phải tất cả) kết bằng một câu hỏi mở tự nhiên để tiếp tục
  câu chuyện (vd: "Bạn muốn mình gợi ý thêm … không?").
- Viết công thức bằng ký hiệu thường (vd: V = π × r² × h, x² − 1 = 0), KHÔNG dùng LaTeX ($...$, \\frac, \\times).
- Người dùng nói như người thật (có khi ngắn, thiếu chủ ngữ, xưng hô anh/chị/em/mình).
Định dạng (đúng {n} cặp, không thêm gì khác):
### NGƯỜI DÙNG
...
### TRỢ LÝ
..."""


def parse(t):
    parts = re.split(r"^###\s*(NGƯỜI DÙNG|TRỢ LÝ)\s*$", t.strip(), flags=re.M)
    msgs = []
    for k in range(1, len(parts) - 1, 2):
        role = "user" if parts[k] == "NGƯỜI DÙNG" else "assistant"
        c = parts[k + 1].strip()
        if c:
            msgs.append({"role": role, "content": c})
    return msgs


def spec(rng):
    dom = rng.choice(list(DOMAINS))
    return {"dom": dom, "task": rng.choice(DOMAINS[dom]), "pat": rng.choice(PATTERNS), "n": rng.choice([3, 4, 4, 5, 5, 6])}


def work(args):
    i, sp, url = args
    try:
        msgs = parse(ollama(P_GEN.format(**sp), url, temp=0.85, num_predict=3000))
        ok = len(msgs) >= 4 and all(m["role"] == ("user" if k % 2 == 0 else "assistant") for k, m in enumerate(msgs))
        if not ok or msgs[-1]["role"] != "assistant":
            return {"i": i, "spec": sp, "drop": f"sai cấu trúc ({len(msgs)} tin)"}
        if CJK.search(" ".join(m["content"] for m in msgs)):
            return {"i": i, "spec": sp, "drop": "lẫn chữ CJK"}
        return {"i": i, "spec": sp, "messages": msgs}
    except Exception as e:
        return {"i": i, "spec": sp, "drop": f"{type(e).__name__}: {str(e)[:80]}"}


def cmd_gen(a):
    os.makedirs(D, exist_ok=True)
    rng = random.Random(a.seed)
    specs = [spec(rng) for _ in range(a.n)]
    out = f"{D}/raw.jsonl"
    done = {json.loads(l)["i"] for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
    todo = [(i, s, a.ollama) for i, s in enumerate(specs) if i not in done]
    kept = 0
    with open(out, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(work, todo), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            kept += "drop" not in r
            if k % 100 == 0 or k == len(todo):
                el = time.time() - T0
                print(f"[{el:.0f}s] {k}/{len(todo)} · đạt {kept} · {60*k/el:.1f}/phút", flush=True)
    print("### multiturn xong", flush=True)


def cmd_build(a):
    rng = random.Random(7)
    rows = []
    for l in open(f"{D}/raw.jsonl", encoding="utf-8"):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if "drop" not in r:
            rows.append(r)
    rng.shuffle(rows)
    probe, train = rows[:60], rows[60:]
    with open(f"{D}/train.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps({"messages": r["messages"], "src": "nhieu_luot_" + r["spec"]["dom"]}, ensure_ascii=False) + "\n")
    with open(f"{D}/probe.jsonl", "w", encoding="utf-8") as f:          # chỉ lấy lượt NGƯỜI DÙNG — model tự trả lời từng lượt
        for k, r in enumerate(probe):
            f.write(json.dumps({"id": f"mt_{k}", "dom": r["spec"]["dom"], "pat": r["spec"]["pat"],
                                "users": [m["content"] for m in r["messages"] if m["role"] == "user"],
                                "ref": [m["content"] for m in r["messages"] if m["role"] == "assistant"]}, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"đạt {len(rows)} → train {len(train)} · probe {len(probe)} · số lượt {dict(Counter(len(r['messages'])//2 for r in rows))}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen"); g.add_argument("--n", type=int, default=9000); g.add_argument("--seed", type=int, default=0)
    g.add_argument("--workers", type=int, default=48); g.add_argument("--ollama", default="http://127.0.0.1:8000/v1")
    sp.add_parser("build")
    a = ap.parse_args()
    {"gen": cmd_gen, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
