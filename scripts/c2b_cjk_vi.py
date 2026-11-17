"""C2b — cứu mẫu đã kiểm có ĐỀ tiếng Trung/Nhật/Hàn (lẫn từ OpenCoder, ~35% verified_*): thầy dịch ĐỀ sang tiếng Việt,
giữ NGUYÊN mọi khối code và chuỗi trích dẫn (test vẫn khớp). Lời giải + test đã chạy thật giữ nguyên.
Ghi đè data/c2b/verified_<lang>.jsonl (bản gốc → .bak). Mẫu dịch hỏng giữ nguyên (bộ trộn sẽ bỏ).
    python scripts/c2b_cjk_vi.py [--workers 24]
"""
import argparse, glob, json, os, re, shutil, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
VI = re.compile(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", re.I)
BLOCK = re.compile(r"```.*?```", re.S)
P = """Dịch đề bài lập trình dưới đây sang TIẾNG VIỆT tự nhiên (như lập trình viên Việt Nam viết).
QUY TẮC: giữ NGUYÊN Y HỆT mọi khối code ``` (không sửa một ký tự); giữ nguyên tên hàm, tên biến, tên tham số; giữ nguyên
mọi chuỗi nằm trong dấu nháy và mọi ví dụ đầu vào/đầu ra (kể cả khi chúng là chữ Trung) vì bộ test kiểm đúng chuỗi đó.
Chỉ dịch phần chữ giải thích. Không thêm bớt nội dung. Chỉ trả về đề đã dịch, không viết gì khác.
=== ĐỀ ===
{q}
=== HẾT ==="""


def chat(llm, model, prompt):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": 2500}
    for k in range(3):
        try:
            req = urllib.request.Request(llm + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"] or ""
        except Exception:
            time.sleep(5 * (k + 1))
    return ""


def fix(a, r):
    q = r["messages"][0]["content"]
    if len(CJK.findall(q)) < 3:
        return r, None
    out = chat(a.llm, a.model, P.format(q=q)).strip()
    out = re.sub(r"^=== ĐỀ ===\s*|\s*=== HẾT ===$", "", out).strip()
    ok = out and BLOCK.findall(out) == BLOCK.findall(q) and len(VI.findall(out)) >= 5 and \
        len(CJK.findall(BLOCK.sub("", re.sub(r"(['\"“「]).*?(['\"”」])", "", out)))) < 3
    if not ok:
        return r, False
    r = dict(r, messages=[dict(r["messages"][0], content=out)] + r["messages"][1:], dich_de=True)
    return r, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--llm", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma4-31b")
    a = ap.parse_args()
    tot = {"dịch": 0, "hỏng": 0}
    for p in sorted(glob.glob("data/c2b/verified_*.jsonl")):
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        if not any(len(CJK.findall(r["messages"][0]["content"])) >= 3 for r in rows):
            continue
        shutil.copy(p, p + ".bak")
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(lambda r: fix(a, r), rows))
        with open(p, "w", encoding="utf-8") as f:
            for r, _ in res:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        d, h = sum(x is True for _, x in res), sum(x is False for _, x in res)
        tot["dịch"] += d; tot["hỏng"] += h
        print(f"[cjk] {os.path.basename(p)}: dịch {d} · hỏng {h}", flush=True)
    print(f"### c2b dịch đề Hán→Việt: {tot['dịch']} mẫu · hỏng (bị bỏ khi trộn) {tot['hỏng']}", flush=True)


if __name__ == "__main__":
    main()
