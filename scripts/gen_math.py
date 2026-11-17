"""TK_104 §16 · TOÁN tiếng Việt CÓ KIỂM bằng máy: thầy viết đề + lời giải từng bước + Python tính đáp số → chỉ giữ khi
Python ra ĐÚNG đáp số khai báo VÀ trùng đáp số cuối trong lời giải. Lời giải viết kiểu "biểu thức = kết quả" (khớp máy tính
lúc trả lời, TK_104 §14.4). Kèm Vi-GSM8K train (7,4 nghìn bài có lời giải tiếng Việt, khớp đáp số với GSM8K gốc).

    python scripts/gen_math.py gen --n 12000 --workers 48 --ollama http://127.0.0.1:8000/v1
    python scripts/gen_math.py build        # → data/math/train.jsonl + test.jsonl (Vi-GSM8K test + bài sinh giữ riêng)
"""
from __future__ import annotations
import argparse, json, os, random, re, subprocess, sys, tempfile, time
import concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_office import ollama, loads_loose, CJK

D = "data/math"
T0 = time.time()
TOPICS = {
    1: ["cộng trừ trong phạm vi 100", "so sánh số", "bài toán thêm bớt"],
    2: ["cộng trừ có nhớ", "bảng nhân chia 2, 5", "xem đồng hồ, ngày tháng"],
    3: ["nhân chia số có ba chữ số", "gấp lên, giảm đi một số lần", "chu vi hình chữ nhật, hình vuông", "bài toán hai bước"],
    4: ["phân số cơ bản", "trung bình cộng", "tổng và hiệu của hai số", "diện tích hình chữ nhật", "đổi đơn vị đo"],
    5: ["số thập phân", "tỉ số phần trăm", "toán chuyển động đều", "diện tích hình thang, hình tròn", "thể tích hình hộp chữ nhật"],
    6: ["phép tính với số nguyên", "phân số và số thập phân", "ước chung lớn nhất, bội chung nhỏ nhất", "tỉ số và tỉ lệ"],
    7: ["tỉ lệ thuận, tỉ lệ nghịch", "biểu thức đại số", "tam giác và góc", "thống kê: số trung bình, mốt"],
    8: ["phương trình bậc nhất một ẩn", "giải bài toán bằng cách lập phương trình", "định lý Pythagore", "hằng đẳng thức đáng nhớ"],
    9: ["hệ hai phương trình bậc nhất", "phương trình bậc hai, công thức nghiệm", "hệ thức Vi-ét", "hàm số bậc nhất", "xác suất đơn giản"],
    10: ["lãi suất ngân hàng, phần trăm tăng giảm", "bài toán năng suất, công việc chung", "bài toán tuổi", "toán hỗn hợp nồng độ"],
}
CONTEXTS = ["đi chợ", "trường học", "nông trại", "cửa hàng", "giao thông", "xây dựng", "thể thao", "nấu ăn", "tiết kiệm tiền",
            "du lịch", "nhà máy", "thư viện", "vườn cây", "không cần bối cảnh"]

P_GEN = """Viết MỘT bài toán tiếng Việt (trình độ lớp {lop}, dạng: {dang}, bối cảnh: {ctx}) có đáp số là MỘT SỐ.
Viết công thức bằng ký hiệu thường (vd: V = π × r² × h, x² − 1 = 0), KHÔNG dùng LaTeX ($...$, \\frac, \\times).
Trả về DUY NHẤT JSON:
{{"de": "đề bài", "loi_giai": "lời giải từng bước, mỗi phép tính viết dạng biểu thức = kết quả, kết thúc bằng câu: Đáp án: <số>",
 "python": "đoạn Python ngắn tính đáp số, gán vào biến ans", "dap_so": <số>}}"""


def run_ans(code):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("import math\nfrom fractions import Fraction\n" + code + "\nprint(repr(float(ans)))\n")
    try:
        p = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=5)
        return float(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 and p.stdout.strip() else None
    except (subprocess.TimeoutExpired, ValueError):
        return None
    finally:
        os.unlink(f.name)


def num_last(s):
    m = re.findall(r"-?\d[\d.,]*", s or "")
    if not m:
        return None
    x = re.sub(r"[.,](?=\d{3}\b)", "", m[-1].rstrip(".,")).replace(",", ".")
    try:
        return float(x)
    except ValueError:
        return None


def close(a, b):
    return a is not None and b is not None and abs(a - b) <= 1e-6 * max(1.0, abs(a)) + 1e-6


def spec(rng):
    lop = rng.choice(list(TOPICS))
    return {"lop": min(lop, 9) if lop < 10 else 9, "dang": rng.choice(TOPICS[lop]), "ctx": rng.choice(CONTEXTS)}


def work(args):
    i, sp, url = args
    try:
        o = loads_loose(ollama(P_GEN.format(**sp), url, temp=0.9, fmt="json", num_predict=1000))
        de, lg, code = o["de"].strip(), o["loi_giai"].strip(), o["python"]
        ds = float(str(o["dap_so"]).replace(",", "."))
        got = run_ans(code)
        if not close(got, ds) or not close(num_last(lg), ds):
            return {"i": i, "spec": sp, "drop": f"lệch: python {got} · khai báo {ds} · lời giải {num_last(lg)}"}
        if CJK.search(de + lg):
            return {"i": i, "spec": sp, "drop": "lẫn chữ CJK"}
        return {"i": i, "spec": sp, "de": de, "loi_giai": lg, "dap_so": ds}
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
    print("### math xong", flush=True)


def cmd_build(a):
    from tk105_stage3 import gsm_pairs
    rng = random.Random(5)
    gen = []
    for l in open(f"{D}/raw.jsonl", encoding="utf-8"):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if "drop" not in r:
            gen.append(r)
    seen, uniq = set(), []
    for r in gen:
        k = re.sub(r"\W+", " ", r["de"].lower())[:90]
        if k not in seen:
            seen.add(k); uniq.append(r)
    rng.shuffle(uniq)
    g_test, g_train = uniq[:200], uniq[200:]
    gsm = gsm_pairs()
    train = [{"messages": [{"role": "user", "content": r["de"]}, {"role": "assistant", "content": r["loi_giai"]}], "src": "toan_sinh"}
             for r in g_train]
    train += [{"messages": [{"role": "user", "content": x["vi_q"]}, {"role": "assistant", "content": x["vi_a"]}], "src": "vi_gsm8k"}
              for x in gsm["train"]]
    rng.shuffle(train)
    test = [{"q": r["de"], "gold": r["dap_so"], "src": "toan_sinh"} for r in g_test]
    test += [{"q": x["vi_q"], "gold": x["gold"], "src": "vi_gsm8k"} for x in gsm["test"][:300]]
    with open(f"{D}/train.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{D}/test.jsonl", "w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"bài sinh đạt {len(gen)} (khử trùng {len(uniq)}) · train {len(train)} (sinh {len(g_train)} + Vi-GSM8K {len(gsm['train'])}) · test {len(test)}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen"); g.add_argument("--n", type=int, default=12000); g.add_argument("--seed", type=int, default=0)
    g.add_argument("--workers", type=int, default=48); g.add_argument("--ollama", default="http://127.0.0.1:8000/v1")
    sp.add_parser("build")
    a = ap.parse_args()
    {"gen": cmd_gen, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
