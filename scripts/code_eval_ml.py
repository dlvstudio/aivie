"""BỘ ĐO CODE ĐA NGÔN NGỮ cho aivie (TK_108 C2b) — MultiPL-E (nuprl/MultiPL-E, MIT): HumanEval đã dịch sang JS, TS, Java,
C#, PHP, Go, Bash, chấm bằng CHẠY THẬT test ẩn trong sandbox (khung chạy của c2b_teacher).

Model trò chuyện thường trả cả hàm chứ không viết tiếp từ giữa → ta tìm dòng chữ ký trong câu trả lời, lấy thân hàm tới
ngoặc đóng tương ứng rồi ghép: prompt gốc + thân + test (giống cách chấm gốc của MultiPL-E). Hàm phụ model viết ngoài hàm
chính được giữ lại (JS/TS/PHP/Go/Bash). Chỉ đo pass@1 (một lần, nhiệt độ 0).

    python scripts/code_eval_ml.py build                                   → data/code_eval/multipl_e.jsonl (cố định)
    python scripts/code_eval_ml.py selftest                                → kiểm ghép + chạy từng ngôn ngữ bằng lời giải tay
    python scripts/code_eval_ml.py run --url http://localhost:8798 --tag c4b [--langs js,java] [--n 40]
    python scripts/code_eval_ml.py report --tags c4b,c4c
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import code_agent as A
import c2b_teacher as T

DATA = "data/code_eval/multipl_e.jsonl"
OUT = "out/code_eval"
JT = "/opt/c2b/javatuples-1.2.jar" if sys.platform == "linux" else os.path.expanduser("~/.cache/c2b/javatuples-1.2.jar")
JT_URL = "https://repo1.maven.org/maven2/org/javatuples/javatuples/1.2/javatuples-1.2.jar"   # đề Java của MultiPL-E import nó
LANGS = {   # mã MultiPL-E → (tên, nhãn khối code, file, lệnh chạy, file phụ)
    "js": ("JavaScript", "javascript", "prog.js", "node prog.js", {}),
    "ts": ("TypeScript", "typescript", "prog.ts", "tsc --target es2020 --module commonjs --skipLibCheck --lib es2020,dom "
           "--typeRoots /usr/local/lib/node_modules/@types --types node prog.ts && node prog.js", {}),
    "java": ("Java", "java", "Problem.java", f"javac -cp {JT} Problem.java && java -ea -cp {JT}:. Problem", {}),
    "cs": ("C#", "csharp", "Program.cs", "dotnet run -c Debug --nologo -v q", {"app.csproj": T.CSPROJ.replace(
        "<ImplicitUsings>enable</ImplicitUsings>", "<ImplicitUsings>disable</ImplicitUsings>")}),
    "php": ("PHP", "php", "prog.php", "php prog.php", {}),
    "go": ("Go", "go", "prog_test.go", "go test ./...", {"go.mod": "module prog\n\ngo 1.22\n"}),
    "sh": ("Bash", "bash", "prog.sh", "bash prog.sh", {}),
}
LANG_TAG = {"javascript": "js", "typescript": "ts", "csharp": "cs", "bash": "sh"}


def build():
    import pyarrow.parquet as pq, io
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    n = 0
    with open(DATA, "w", encoding="utf-8") as f:
        for lg in LANGS:
            url = f"https://huggingface.co/api/datasets/nuprl/MultiPL-E/parquet/humaneval-{lg}/test/0.parquet"
            with urllib.request.urlopen(url, timeout=120) as r:
                rows = pq.read_table(io.BytesIO(r.read())).to_pylist()
            for x in rows:
                f.write(json.dumps({"id": f"{lg}/{x['name']}", "lang": lg, "name": x["name"], "prompt": x["prompt"],
                                    "tests": x["tests"]}, ensure_ascii=False) + "\n")
                n += 1
            print(f"[ml] {lg}: {len(rows)} bài", flush=True)
    print(f"### multipl-e: {n} bài → {DATA}")


def load(langs=None):
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    return [r for r in rows if not langs or r["lang"] in langs]


# ── ghép câu trả lời vào khung chấm ─────────────────────────────────────────────────────────
def _match_close(s, i, lg):
    """s[i:] là phần ngay sau "{" mở thân hàm → vị trí ngay sau "}" đóng tương ứng (bỏ qua chuỗi + chú thích). None nếu thiếu."""
    d, k, n = 1, i, len(s)
    while k < n:
        c = s[k]
        if c in "\"'`":
            q = c; k += 1
            while k < n and s[k] != q:
                k += 2 if s[k] == "\\" else 1
        elif s.startswith("//", k) and lg != "sh" or (c == "#" and lg in ("sh", "php") and (k == 0 or s[k - 1] in " \t\n")):
            while k < n and s[k] != "\n":
                k += 1
        elif s.startswith("/*", k) and lg != "sh":
            e = s.find("*/", k + 2); k = n if e < 0 else e + 1
        elif c == "{":
            d += 1
        elif c == "}":
            d -= 1
            if d == 0:
                return k + 1
        k += 1
    return None


def code_of(ans, fence):
    bl = re.findall(r"```[\w#+.-]*[^\n]*\n(.*?)```", ans, re.S)
    if not bl:                                         # khối code chưa đóng (model bị cắt) → bỏ dòng ``` mở đầu
        m = re.search(r"```[\w#+.-]*[^\n]*\n(.*)", ans, re.S)
        return m.group(1) if m else ans
    same = [b for t, b in re.findall(r"```([\w#+.-]*)[^\n]*\n(.*?)```", ans, re.S) if t.lower() in (fence, LANG_TAG.get(fence, fence))]
    return max(same or bl, key=len)


def assemble(p, ans):
    """→ chương trình đầy đủ để chạy (prompt + thân hàm của model + [hàm phụ] + test)."""
    lg = p["lang"]
    name, fence, _, _, _ = LANGS[lg]
    code = code_of(ans, fence)
    sig = [l for l in p["prompt"].rstrip().split("\n") if l.strip()][-1].strip()     # dòng chữ ký, kết thúc bằng "{"
    head = re.escape(sig[:-1].strip()) if sig.endswith("{") else re.escape(sig)
    m = re.search(re.sub(r"\\\s+", r"\\s*", head) + r"\s*\{", code) or \
        re.search(r"(function\s+|func\s+|\b)" + re.escape(_fname(sig)) + r"\s*\([^)]*\)[^{\n]*\{", code)
    tests_close = p["tests"].lstrip().startswith("}")
    if m:
        end = _match_close(code, m.end(), lg)
        if end is None:
            body = code[m.end():] + "\n}"
            end = len(code)
        else:
            body = code[m.end():end]
        extra = ""
        if lg in ("js", "ts", "php", "go", "sh"):     # hàm phụ model viết ngoài hàm chính
            rest = code[:m.start()] + "\n" + code[end:]
            rest = "\n".join(l for l in rest.split("\n") if not re.match(
                r"\s*(<\?php|\?>|package\s|import\s|import\s*\(|\"[\w/]+\"\s*$|\)\s*$|module\.exports|export\s+default|console\.log|"
                r"echo\s|print|main\s*\(|[a-z_]+\s+\"\$@\"\s*$)", l))
            if re.search(r"\b(function|func)\b|\w+\s*\(\)\s*\{", rest):
                extra = "\n" + _keep_defs(rest, lg) + "\n"
    else:                                              # không thấy chữ ký → coi như model chỉ viết thân hàm
        body = code.rstrip() + "\n}"
        extra = ""
    if tests_close:                                    # test của Java / Bash tự đóng hàm → bỏ "}" cuối của thân
        body = body.rstrip()
        body = body[:-1] if body.endswith("}") else body
    prog = p["prompt"] + body + "\n" + (extra if not tests_close else "") + p["tests"]
    if tests_close and extra:                          # Bash: hàm phụ đặt sau khi đóng hàm chính
        prog = p["prompt"] + body + "\n}\n" + extra + p["tests"].lstrip()[1:]
    return prog


def _fname(sig):
    m = re.search(r"(\w+)\s*\(", sig)
    return m.group(1) if m else "___"


def _keep_defs(rest, lg):
    """chỉ giữ các khối định nghĩa hàm hoàn chỉnh trong phần còn lại."""
    out, k = [], 0
    pat = re.compile(r"(?m)^[ \t]*(?:export\s+)?(?:async\s+)?(?:function\s+\w+|func\s+(?:\([^)]*\)\s*)?\w+|\w+\s*\(\))[^{\n]*\{")
    for m in pat.finditer(rest):
        if m.start() < k:
            continue
        e = _match_close(rest, m.end(), lg)
        if e:
            out.append(rest[m.start():e].replace("export ", "", 1)); k = e
    return "\n\n".join(out)


def grade(p, prog):
    _, _, fname, cmd, extra = LANGS[p["lang"]]
    r = T.run_project({fname: prog, **extra}, cmd, 90 if p["lang"] in ("cs", "java", "ts", "go") else 30)
    return r["rc"] == 0 and not r["timeout"], r


TASK = "Hoàn thành hàm {L} sau (giữ nguyên chữ ký, chỉ dùng thư viện chuẩn). Trả về toàn bộ hàm trong MỘT khối ```{f}.\n```{f}\n{p}\n```"


def run(a):
    langs = a.langs.split(",") if a.langs else list(LANGS)
    probs = load(langs)
    if a.n:
        by = {}
        for p in probs:
            by.setdefault(p["lang"], []).append(p)
        probs = [p for v in by.values() for p in v[:a.n]]
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/ml_{a.tag}.jsonl"
    done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()
    probs = [p for p in probs if p["id"] not in done]
    model = A.Model(a.url, a.tag, max_new=a.max_new, model_id=a.model_id)
    ensure_jt()
    T0 = time.time()

    def one(p):
        L, f = LANGS[p["lang"]][0], LANGS[p["lang"]][1]
        ans = model.chat([{"role": "user", "content": TASK.format(L=L, f=f, p=p["prompt"].strip())}])
        prog = assemble(p, ans)
        ok, r = grade(p, prog)
        return {"id": p["id"], "lang": p["lang"], "ok": ok, "rc": r["rc"], "out": r["out"][-600:], "answer": ans, "prog": prog}

    with open(path, "a", encoding="utf-8") as fo, ThreadPoolExecutor(a.workers) as ex:
        for k, rec in enumerate(ex.map(one, probs)):
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
            if k % 20 == 0 or k == len(probs) - 1:
                print(f"[ml {a.tag}] {k+1}/{len(probs)} · {time.time()-T0:.0f}s · {summary(path)}", flush=True)
    print("### ml", a.tag, json.dumps(summary(path), ensure_ascii=False))


def summary(path):
    paths = path if isinstance(path, list) else [path]
    rows = [json.loads(l) for p in paths if os.path.exists(p) for l in open(p, encoding="utf-8")]
    by = {}
    for r in rows:
        by.setdefault(r["lang"], []).append(r["ok"])
    s = {lg: round(100 * sum(v) / len(v), 1) for lg, v in sorted(by.items())}
    if rows:
        s["TB"] = round(sum(s.values()) / len(s), 1)
        s["n"] = len(rows)
    return s


def report(a):
    for t in a.tags.split(","):                     # "c4c-1+c4c-2" → gộp nhiều file (mỗi tiến trình đo ghi file riêng)
        print(f"{t:12s} {json.dumps(summary([f'{OUT}/ml_{x}.jsonl' for x in t.split('+')]), ensure_ascii=False)}")


SELF_ADD = {   # HumanEval/53 add(x, y): câu trả lời kiểu model (cả hàm, có khi kèm hàm phụ) — đúng phải QUA, sai phải HỎNG
    "js": "```javascript\nfunction helper(a, b) { return a + b; }\n\nfunction add(x, y) {\n  // cộng\n  return helper(x, y);\n}\n```",
    "ts": "```typescript\nfunction add(x: number, y: number): number {\n  return x + y;\n}\n```",
    "java": "```java\nimport java.util.*;\n\nclass Problem {\n    public static long add(long x, long y) {\n        return x + y;\n    }\n}\n```",
    "cs": "```csharp\nclass Problem {\n    public static long Add(long x, long y) {\n        return x + y;\n    }\n}\n```",
    "php": "```php\n<?php\nfunction add($x, $y) {\n    return $x + $y; // {\n}\n```",
    "go": "```go\npackage add_test\n\nfunc add(x int, y int) int {\n\treturn x + y\n}\n```",
    "sh": "```bash\nadd() {\n    echo $(( $1 + $2 ))\n}\n```",
}


def ensure_jt():
    if not os.path.exists(JT):
        os.makedirs(os.path.dirname(JT), exist_ok=True)
        urllib.request.urlretrieve(JT_URL, JT)
        os.chmod(JT, 0o644)


def selftest():
    ensure_jt()
    good = True
    probs = {p["lang"]: p for p in load() if p["name"].startswith("HumanEval_53_")}
    for lg, ans in SELF_ADD.items():
        if lg not in probs:
            print(f"[selftest] {lg}: không có bài add"); good = False; continue
        p = probs[lg]
        ok1, r1 = grade(p, assemble(p, ans))
        bad = ans.replace("x + y", "x - y").replace("$x + $y", "$x - $y").replace("$1 + $2", "$1 - $2").replace("a + b", "a - b")
        ok2, r2 = grade(p, assemble(p, bad))
        ok = ok1 and not ok2
        good &= ok
        print(f"[selftest] {lg}: {'OK' if ok else 'HỎNG'} (đúng rc={r1['rc']}, sai rc={r2['rc']})"
              + ("" if ok1 else f"\n--- chương trình ---\n{assemble(p, ans)[-900:]}\n--- lỗi ---\n{r1['out'][-600:]}"), flush=True)
    print(f"### ml selftest: {'OK' if good else 'HỎNG'}")
    return good


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "selftest", "run", "report"])
    ap.add_argument("--url", default="http://localhost:8798")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--tag", default="c4b"); ap.add_argument("--tags", default="c4b")
    ap.add_argument("--langs", default="")
    ap.add_argument("--n", type=int, default=0, help="chỉ n bài đầu mỗi ngôn ngữ")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=900)
    a = ap.parse_args()
    if a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    {"build": build, "run": lambda: run(a), "report": lambda: report(a)}[a.cmd]()
