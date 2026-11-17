"""TK_104 §15 · "SÁCH GIÁO KHOA" LẬP TRÌNH TIẾNG VIỆT kiểu Phi-1 — ít mà chất, MỌI mẫu đều được MÁY KIỂM.

Phi-1 (1.3B) đạt ~50% HumanEval nhờ dữ liệu "chất lượng sách giáo khoa" + bài tập, không nhờ khối lượng.
Ở đây Gemma làm thầy viết bằng tiếng Việt, ta chạy thật để lọc:
  bai_tap  : đề tiếng Việt (có tên hàm) + lời giải (dẫn ngắn → code chú thích tiếng Việt → giải thích) + 3–6 assert
             → GIỮ chỉ khi code + assert chạy QUA (tiến trình con, timeout 10s)
  bai_giang: giảng một khái niệm (định nghĩa → ví dụ chạy được → lỗi hay gặp → mẹo) — GIỮ khi MỌI khối code chạy không lỗi
Lưới đề: ~70 chủ đề Python/thuật toán × độ khó × bối cảnh đời thường Việt Nam → đa dạng.
Cũng là dữ liệu CẦU NỐI tiếng Việt chất lượng cao cho expert code (chú thích code bằng tiếng Việt).

    python scripts/gen_code_textbook.py gen --n 20000 --workers 16 --ollama http://127.0.0.1:11434   # box
    python scripts/gen_code_textbook.py build      # → data/code_tb/train.jsonl (định dạng messages) + thống kê
"""
from __future__ import annotations
import argparse, json, os, random, re, subprocess, sys, tempfile, time
import concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_office import ollama

RAW = "data/code_tb/raw.jsonl"
T0 = time.time()

TOPICS = [
    "biến, kiểu số và phép toán", "chuỗi: cắt, nối, định dạng f-string", "phương thức chuỗi (split, join, strip, replace)",
    "list: thêm, xoá, cắt lát", "tuple và unpacking", "dict: duyệt, đếm, gom nhóm", "set và phép toán tập hợp",
    "câu lệnh if/elif/else", "vòng lặp for và range", "vòng lặp while và break/continue", "hàm, tham số mặc định",
    "*args và **kwargs", "đệ quy", "list comprehension", "dict comprehension", "lambda, map, filter",
    "sorted với key", "enumerate và zip", "xử lý ngoại lệ try/except", "đọc ghi file văn bản", "module math",
    "datetime: ngày tháng", "re: biểu thức chính quy cơ bản", "collections.Counter", "collections.defaultdict",
    "collections.deque", "heapq", "itertools (combinations, permutations, product)", "class và đối tượng",
    "kế thừa và phương thức đặc biệt __str__, __eq__", "generator và yield", "decorator đơn giản",
    "tìm kiếm tuyến tính và nhị phân", "sắp xếp nổi bọt, chọn, chèn", "sắp xếp trộn (merge sort)",
    "quick sort", "hai con trỏ", "cửa sổ trượt", "tổng tiền tố", "ngăn xếp (stack)", "hàng đợi (queue)",
    "duyệt ma trận 2 chiều", "xoay/chuyển vị ma trận", "số nguyên tố, sàng Eratosthenes", "ước chung lớn nhất, bội chung nhỏ nhất",
    "chữ số của một số", "chuyển đổi hệ cơ số", "thao tác bit", "chuỗi đối xứng", "anagram", "đếm tần suất ký tự/từ",
    "gộp khoảng (merge intervals)", "quy hoạch động cơ bản (Fibonacci, leo cầu thang)", "quy hoạch động: dãy con tăng dài nhất",
    "quy hoạch động: balo 0/1", "tham lam (greedy)", "BFS trên đồ thị", "DFS trên đồ thị", "đường đi ngắn nhất trên lưới",
    "danh sách liên kết (cài bằng class)", "cây nhị phân tìm kiếm cơ bản", "xử lý chuỗi số tiền, định dạng nghìn",
    "xử lý dữ liệu dạng bảng bằng list của dict", "kiểm tra dữ liệu đầu vào hợp lệ", "tính toán thống kê: trung bình, trung vị, mode",
    "làm tròn và số thực", "mô phỏng đơn giản (máy bán hàng, hàng đợi khách)", "phân tích chuỗi ngày dd/mm/yyyy",
]
LEVELS = ["dễ", "vừa", "khó vừa phải"]
CONTEXTS = ["điểm học sinh", "đơn hàng shop online", "lương nhân viên", "nhiệt độ các tỉnh", "vé xe khách",
            "kho hàng tạp hoá", "giao dịch ngân hàng", "lịch học", "giải bóng đá", "chi tiêu gia đình", "thư viện sách",
            "bãi gửi xe", "quán cà phê", "biển số xe", "số điện thoại", "toạ độ trên bản đồ", "không cần bối cảnh"]

P_EX = """Bạn là giáo viên lập trình Python giỏi, viết BÀI TẬP cho sách giáo khoa tiếng Việt.
Chủ đề: {topic}. Độ khó: {level}. Bối cảnh: {ctx}.
Viết đúng 3 phần theo định dạng dưới (không thêm gì khác):
<<<DE_BAI>>>
Đề bài tiếng Việt rõ ràng: yêu cầu viết MỘT hàm Python tên cụ thể (ghi rõ chữ ký hàm, ví dụ `def tinh_tong(ds):`),
nói rõ đầu vào, đầu ra, 1–2 ví dụ.
<<<LOI_GIAI>>>
Câu dẫn ngắn tiếng Việt, rồi MỘT khối ```python chứa hàm hoàn chỉnh (chú thích bằng tiếng Việt, chỉ dùng thư viện chuẩn,
KHÔNG đọc input(), KHÔNG in ví dụ), rồi giải thích ngắn gọn cách làm và độ phức tạp bằng tiếng Việt.
<<<KIEM_THU>>>
3–6 dòng `assert ...` gọi hàm ở trên với kết quả đúng (không đặt trong khối ```)."""

P_LS = """Bạn là giáo viên lập trình Python giỏi, viết MỘT MỤC BÀI GIẢNG cho sách giáo khoa tiếng Việt.
Chủ đề: {topic}. Trình độ: {level}. Ví dụ lấy bối cảnh: {ctx}.
Viết theo định dạng (không thêm gì khác):
<<<CAU_HOI>>>
Một câu học viên hỏi tự nhiên về chủ đề này (tiếng Việt).
<<<BAI_GIANG>>>
Bài giảng tiếng Việt: giải thích khái niệm dễ hiểu → 2–3 ví dụ trong khối ```python TỰ CHẠY ĐƯỢC (có print kết quả,
chú thích tiếng Việt, chỉ thư viện chuẩn, không input(), không đọc file ngoài) → lỗi hay gặp → mẹo ghi nhớ. Không lan man."""

CODE = re.compile(r"```python\n(.*?)```", re.S)


def run_py(src, timeout=10):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
    try:
        p = subprocess.run([sys.executable, f.name], capture_output=True, timeout=timeout, text=True)
        return p.returncode == 0, p.stderr[-200:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(f.name)


def spec(rng):
    return {"kind": "bai_tap" if rng.random() < 0.7 else "bai_giang", "topic": rng.choice(TOPICS),
            "level": rng.choice(LEVELS), "ctx": rng.choice(CONTEXTS)}


def split3(t, a, b, c=None):
    if a not in t or b not in t or (c and c not in t):
        return None
    x = t.split(a, 1)[1]
    y, z = x.split(b, 1)
    if c:
        z, w = z.split(c, 1)
        return y.strip(), z.strip(), w.strip()
    return y.strip(), z.strip()


CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")   # Gemma bản nén đôi khi lẫn chữ Trung/Nhật/Hàn → loại


def work(args):
    i, sp, url = args
    try:
        if sp["kind"] == "bai_tap":
            t = ollama(P_EX.format(**sp), url, num_predict=1600)
            parts = split3(t, "<<<DE_BAI>>>", "<<<LOI_GIAI>>>", "<<<KIEM_THU>>>")
            if not parts:
                return {"i": i, "spec": sp, "drop": "sai định dạng"}
            de, lg, kt = parts
            blocks = CODE.findall(lg)
            asserts = [l for l in kt.splitlines() if l.strip().startswith("assert")]
            if len(blocks) != 1 or len(asserts) < 3:
                return {"i": i, "spec": sp, "drop": f"khối code {len(blocks)} · assert {len(asserts)}"}
            ok, err = run_py(blocks[0] + "\n\n" + "\n".join(a.strip() for a in asserts) + "\n")
            if not ok:
                return {"i": i, "spec": sp, "drop": f"test trượt: {err[-80:]}"}
            if CJK.search(de + lg):
                return {"i": i, "spec": sp, "drop": "lẫn chữ CJK"}
            return {"i": i, "spec": sp, "q": de, "a": lg, "tests": asserts}
        t = ollama(P_LS.format(**sp), url, num_predict=1800)
        parts = split3(t, "<<<CAU_HOI>>>", "<<<BAI_GIANG>>>")
        if not parts:
            return {"i": i, "spec": sp, "drop": "sai định dạng"}
        q, a = parts
        blocks = CODE.findall(a)
        if not blocks:
            return {"i": i, "spec": sp, "drop": "không có ví dụ code"}
        for b in blocks:
            ok, err = run_py(b)
            if not ok:
                return {"i": i, "spec": sp, "drop": f"ví dụ lỗi: {err[-80:]}"}
        if CJK.search(q + a):
            return {"i": i, "spec": sp, "drop": "lẫn chữ CJK"}
        return {"i": i, "spec": sp, "q": q, "a": a}
    except Exception as e:
        return {"i": i, "spec": sp, "drop": f"{type(e).__name__}: {str(e)[:80]}"}


def cmd_gen(a):
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rng = random.Random(a.seed)
    specs = [spec(rng) for _ in range(a.n)]
    done = {json.loads(l)["i"] for l in open(a.out)} if os.path.exists(a.out) else set()
    todo = [(i, s, a.ollama) for i, s in enumerate(specs) if i not in done]
    print(f"[tb] {len(done)} đã có · còn {len(todo)}", flush=True)
    kept = drop = 0
    with open(a.out, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(work, todo), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            kept += "drop" not in r; drop += "drop" in r
            if k % 50 == 0 or k == len(todo):
                el = time.time() - T0
                print(f"[{el:.0f}s] {k}/{len(todo)} · giữ {kept} · loại {drop} · {60*k/el:.1f}/phút", flush=True)
    print("### tb xong", flush=True)


def cmd_build(a):
    from collections import Counter
    rows = []
    for l in open(a.inp, encoding="utf-8"):
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:                       # dòng cuối đang ghi dở
            pass
    good = [r for r in rows if "drop" not in r]
    seen, uniq = set(), []
    for r in good:
        key = re.sub(r"\W+", " ", r["q"].lower())[:100]
        if key not in seen:
            seen.add(key); uniq.append(r)
    os.makedirs("data/code_tb", exist_ok=True)
    with open("data/code_tb/train.jsonl", "w", encoding="utf-8") as f:
        for r in uniq:
            f.write(json.dumps({"messages": [{"role": "user", "content": r["q"]}, {"role": "assistant", "content": r["a"]}],
                                "src": r["spec"]["kind"]}, ensure_ascii=False) + "\n")
    print(f"tổng {len(rows)} · đạt {len(good)} · khử trùng {len(uniq)}")
    print("loại:", Counter(r["spec"]["kind"] for r in uniq))
    print("lý do loại:", Counter(r["drop"].split(":")[0][:30] for r in rows if "drop" in r).most_common(6))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen"); g.add_argument("--n", type=int, default=100); g.add_argument("--seed", type=int, default=0)
    g.add_argument("--workers", type=int, default=2); g.add_argument("--out", default=RAW)
    g.add_argument("--ollama", default="http://127.0.0.1:11434")
    b = sp.add_parser("build"); b.add_argument("--inp", default=RAW)
    a = ap.parse_args()
    {"gen": cmd_gen, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
