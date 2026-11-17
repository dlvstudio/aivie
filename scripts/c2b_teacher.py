"""TK_108 C2b phần B — code ĐA NGÔN NGỮ có thầy, KIỂM BẰNG CHẠY THẬT, LƯU NHẬT KÝ ĐẦY ĐỦ.

Lấy mẫu từ data/c2b/<lang>.jsonl (phần A, đã qua kiểm cú pháp; ưu tiên dạng hỏi–đáp, không lấy commit):
  javascript typescript java csharp php go bash  thầy viết lại thành dự án nhỏ chạy độc lập (lời giải + test chỉ thư viện chuẩn)
                                                → CHẠY trong sandbox (user sbx) → hỏng thì đưa lỗi thật cho thầy sửa (tối đa
                                                --fix lượt) → giữ khi test QUA và test đủ mạnh (≥ 3 phép kiểm)
  sql                                           thầy giải ĐỘC LẬP câu hỏi gretel trên SQLite → kết quả phải TRÙNG lời giải gốc
  tsql                                          thầy chuyển bài gretel (SQLite) sang SQL Server: setup.sql + query.sql →
                                                CHẠY trên SQL Server 2022 Express (DB riêng mỗi bài, login quyền thấp)
                                                → kết quả phải TRÙNG lời giải gốc chạy trên SQLite
  html_css                                      mở trang bằng Chromium không giao diện → lỗi JS → thầy sửa → mở lại
  --vi-frac                                     một phần mẫu QUA được thầy dịch đề + giải thích sang tiếng Việt (code y hệt)

Ra:  data/c2b/verified_<lang>.jsonl          mẫu SFT (đề → giải thích + lời giải đã qua test; kèm test để dùng cho RL)
     data/c2b/teacher_log/<lang>.jsonl       MỌI lượt: lời gọi thầy (prompt + trả lời thô), file, lệnh, mã thoát, output, qua/hỏng
                                             → làm DPO (bản hỏng ↔ bản qua), quỹ đạo lỗi → sửa, bài khó cho C5 (RL)

  python scripts/c2b_teacher.py --langs javascript,typescript,java,csharp,php,go,bash,sql,html_css --per-lang 4000 --hours 7
"""
from __future__ import annotations
import argparse, itertools, json, os, random, re, resource, shutil, sqlite3, subprocess, sys, tempfile, threading, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
import code_agent as CA

OUT = "data/c2b"
A = None
_LK = threading.Lock()
_LOGS = {}
WORK = "/opt/c2b/work" if sys.platform == "linux" else os.path.join(tempfile.gettempdir(), "c2b_work")  # box: chmod 1777
RUN_SEM = threading.Semaphore(24)                  # số tiến trình biên dịch/chạy cùng lúc (javac / dotnet / go nặng CPU)
ENV = {"PATH": ("/opt/homebrew/opt/openjdk@21/bin:/opt/homebrew/bin:/usr/local/share/dotnet:" if sys.platform == "darwin" else "")
               + "/usr/local/go/bin:/usr/lib/go/bin:/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
       "GOCACHE": "/opt/c2b/gocache" if sys.platform == "linux" else os.path.join(tempfile.gettempdir(), "c2b_gocache"), "GOFLAGS": "-mod=mod", "GOTOOLCHAIN": "local", "GO111MODULE": "on",
       "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_EnableWriteXorExecute": "0",  # W^X dùng memfd rất lớn → vướng RLIMIT_FSIZE
       "DOTNET_NOLOGO": "1", "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1",
       "NODE_PATH": "/usr/local/lib/node_modules",
       "MSBUILDDISABLENODEREUSE": "1", "DOTNET_CLI_USE_MSBUILD_SERVER": "0", "UseSharedCompilation": "false"}  # không để tiến trình nền
CSPROJ = ('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework>'
          '<ImplicitUsings>enable</ImplicitUsings><Nullable>disable</Nullable><TreatWarningsAsErrors>false</TreatWarningsAsErrors>'
          '</PropertyGroup></Project>')

LANG = {
    "javascript": dict(name="JavaScript (Node.js 18)", files={"solution.js": "lời giải, xuất bằng module.exports",
                        "test.js": "test: const assert = require('assert'); const {...} = require('./solution'); lỗi thì assert ném ra"},
                       cmd="node test.js", fence="javascript"),
    "typescript": dict(name="TypeScript 5 (Node.js 18)", files={"solution.ts": "lời giải, dùng export",
                        "test.ts": "test: import assert from 'assert'; import {...} from './solution';"},
                       cmd="tsc --target es2020 --module commonjs --esModuleInterop --skipLibCheck --strict --outDir out "
                           "--typeRoots /usr/local/lib/node_modules/@types --types node solution.ts test.ts && node out/test.js",
                       fence="typescript"),
    "java": dict(name="Java 17", files={"Solution.java": "lời giải (class Solution, có thể thêm class phụ không public)",
                  "Main.java": "test: public class Main với main(); kiểm sai thì throw new AssertionError(...)"},
                 cmd="javac -d out *.java && java -cp out Main", fence="java"),
    "csharp": dict(name="C# (.NET 8)", files={"Solution.cs": "lời giải (class Solution, namespace tuỳ ý hoặc không)",
                    "Program.cs": "test: class Program với static void Main(); kiểm sai thì throw new Exception(...)"},
                   cmd="dotnet run -c Release --nologo -v q", fence="csharp", extra={"app.csproj": CSPROJ}),
    "php": dict(name="PHP 8.3", files={"solution.php": "lời giải", "test.php": "test: require 'solution.php'; kiểm sai thì "
                "throw new Exception(...) (không dùng PHPUnit)"},
                cmd="php -d zend.assertions=1 -d assert.exception=1 test.php", fence="php"),
    "go": dict(name="Go 1.22", files={"solution.go": "package main — lời giải (có thể có func main rỗng)",
                "solution_test.go": "package main — test dùng gói testing (func TestXxx(t *testing.T), t.Errorf/t.Fatalf)"},
               cmd="go test ./...", fence="go", extra={"go.mod": "module sol\n\ngo 1.22\n"}),
    "bash": dict(name="Bash 5", files={"solution.sh": "lời giải: định nghĩa hàm bash (không chạy gì khi được source)",
                  "test.sh": "test: source ./solution.sh; kiểm từng trường hợp, sai thì echo lý do >&2 và exit 1"},
                 cmd="bash test.sh", fence="bash"),
}
CHECKS = {"javascript": r"assert", "typescript": r"assert", "java": r"AssertionError|assert", "csharp": r"throw|Assert",
          "php": r"throw|assert", "go": r"t\.(Errorf|Fatalf|Error|Fatal)", "bash": r"exit 1|return 1"}

P_PROJ = """Bạn là kỹ sư phần mềm cẩn thận. Dưới đây là một bài lập trình {L} và lời giải tham khảo (có thể SAI, thiếu, hoặc là
đoạn trích không chạy độc lập được).
Nhiệm vụ: biến nó thành một dự án nhỏ CHẠY ĐƯỢC NGAY bằng lệnh `{cmd}`, CHỈ dùng thư viện chuẩn, gồm đúng các file:
{files}
Yêu cầu:
- Giữ đúng ý đồ của đề. Lời giải đầy đủ, chạy thật được (không để trống, không TODO, không giả lập kết quả).
- Test kiểm tra ít nhất 4 trường hợp có ĐÁP ÁN CỤ THỂ, gồm trường hợp biên; sai thì phải thoát với mã khác 0.
- Không dùng mạng, không đọc file ngoài, không chờ nhập từ bàn phím, chạy dưới 10 giây.
=== ĐỀ ===
{q}
=== LỜI GIẢI THAM KHẢO ===
{a}
=== HẾT ===
Trả lời theo khuôn: với MỖI file, một dòng `FILE: <tên file>` rồi ngay sau là một khối code. Cuối cùng một dòng bắt đầu bằng
`GIẢI THÍCH:` gồm 1–3 câu tiếng Việt nói lời giải làm gì. Không viết gì khác."""
P_FIX = """Chạy `{cmd}` bị lỗi (mã thoát {rc}):
```
{out}
```
Hãy sửa. Nếu test sai so với ĐỀ thì sửa test cho ĐÚNG đề, nhưng KHÔNG được nới lỏng hay xoá phép kiểm chỉ để cho qua.
Trả lời lại TOÀN BỘ các file và dòng GIẢI THÍCH theo đúng khuôn cũ."""
P_SQL = """Cho cơ sở dữ liệu SQLite:
```sql
{ctx}
```
Câu hỏi: {q}
Trả lời bằng TIẾNG VIỆT: viết truy vấn SQLite trong MỘT khối ```sql, rồi 1–3 câu giải thích truy vấn làm gì."""
P_TSQL = """Bạn là chuyên gia SQL Server (T-SQL). Dưới đây là một bài SQL viết cho SQLite cùng lời giải đúng.
Hãy chuyển sang SQL Server 2022:
- `setup.sql`: tạo bảng + dữ liệu y hệt (đổi kiểu cho hợp T-SQL: NVARCHAR, INT, DECIMAL(p,s), DATE/DATETIME2, BIT;
  AUTOINCREMENT → IDENTITY; không dùng GO, không USE, không tạo database).
- `query.sql`: MỘT câu truy vấn T-SQL trả lời câu hỏi, cho ra CÙNG kết quả với lời giải SQLite. Dùng cách viết tự nhiên của
  T-SQL khi hợp lý (TOP thay LIMIT, ISNULL/COALESCE, DATEADD/DATEDIFF/FORMAT, CTE, window function…).
=== LƯỢC ĐỒ + DỮ LIỆU (SQLite) ===
```sql
{ctx}
```
=== CÂU HỎI ===
{q}
=== LỜI GIẢI SQLite ===
```sql
{ref}
```
Trả lời theo khuôn: `FILE: setup.sql` rồi một khối ```sql, `FILE: query.sql` rồi một khối ```sql, cuối cùng một dòng
`GIẢI THÍCH:` 1–3 câu tiếng Việt nói truy vấn T-SQL làm gì. Không viết gì khác."""
P_VI = """Dịch đề bài lập trình và phần giải thích dưới đây sang TIẾNG VIỆT tự nhiên (như lập trình viên Việt Nam viết).
Giữ NGUYÊN Y HỆT mọi khối code ``` (không sửa một ký tự), giữ nguyên tên hàm/biến/ví dụ. Không thêm bớt nội dung.
=== ĐỀ ===
{q}
=== GIẢI THÍCH ===
{e}
=== HẾT ===
Trả lời đúng khuôn:
<<<DE>>>
(đề tiếng Việt)
<<<GT>>>
(giải thích tiếng Việt)"""
VI_CH = re.compile(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", re.I)


# ── nhật ký ─────────────────────────────────────────────────────────────────────────────────
def logw(lang, **d):
    with _LK:
        if lang not in _LOGS:
            os.makedirs(f"{OUT}/teacher_log", exist_ok=True)
            _LOGS[lang] = open(f"{OUT}/teacher_log/{lang}.jsonl", "a", encoding="utf-8")
        _LOGS[lang].write(json.dumps({"t": round(time.time(), 1), "lang": lang, **d}, ensure_ascii=False) + "\n")


def chat(msgs, lang, rid, step, max_tokens=3500, temp=0.2):
    body = {"model": A.model, "messages": msgs, "temperature": temp, "max_tokens": max_tokens}
    for k in range(3):
        try:
            req = urllib.request.Request(A.llm.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                out = json.loads(r.read())["choices"][0]["message"]["content"] or ""
            pr = msgs[-1]["content"]
            if isinstance(pr, list):                       # có ảnh → chỉ ghi phần chữ
                pr = " ".join(x.get("text", "[ảnh]") for x in pr)
            logw(lang, id=rid, ev="teacher", step=step, prompt=pr, raw=out)
            return out
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            time.sleep(5 * (k + 1))
    logw(lang, id=rid, ev="teacher_error", step=step, err=err)
    return ""


# ── chạy thật trong sandbox ─────────────────────────────────────────────────────────────────
def _lim(cpu):
    def f():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 30, 1 << 30))   # dotnet/go ghi file trung gian lớn
        CA._nproc()                                                         # trần tiến trình sbx (chống fork bomb)
        os.setsid()
    return f


def run_project(files, cmd, timeout):
    os.makedirs(WORK, exist_ok=True)
    wd = tempfile.mkdtemp(prefix="c2b_", dir=WORK)       # không đặt ngay dưới /tmp: go bỏ qua go.mod ở gốc thư mục tạm
    try:
        for p, c in files.items():
            if "/" in p or p.startswith("."):
                return {"rc": -1, "out": f"tên file không hợp lệ: {p}", "timeout": False}
            open(os.path.join(wd, p), "w", encoding="utf-8").write(c)
        os.makedirs(os.path.join(wd, ".tmp"), exist_ok=True)
        CA._own(wd)
        env = dict(ENV, HOME=wd, DOTNET_CLI_HOME=wd, NUGET_PACKAGES=os.path.join(wd, ".nuget"),
                   TMPDIR=os.path.join(wd, ".tmp"))                     # go bỏ qua go.mod nằm ngay ở thư mục TMPDIR
        os.makedirs(env["TMPDIR"], exist_ok=True)
        with RUN_SEM:
            rc, out, to = CA.run_group(CA._wrap(["bash", "-c", cmd], wd), wd, env, timeout, _lim(timeout * 4))
            return {"rc": rc, "out": out[-4000:] + ("\n[HẾT GIỜ]" if to else ""), "timeout": to}
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def parse_files(raw, want):
    files = {}
    for m in re.finditer(r"FILE:\s*`?([\w.\-]+)`?\s*\n+```[\w#+.-]*[^\n]*\n(.*?)```", raw, re.S):
        files[m.group(1)] = m.group(2)
    ex = re.search(r"GIẢI THÍCH:\s*(.+)", raw, re.S)
    ok = set(want) <= set(files)
    return ({k: files[k] for k in want} if ok else None), (ex.group(1).strip()[:800] if ex else "")


def strong(lang, test):
    return len(re.findall(CHECKS[lang], test)) >= 3


# ── các kiểu bài ────────────────────────────────────────────────────────────────────────────
def do_code(lang, r):
    S, rid = LANG[lang], r["id"]
    q, a = r["messages"][0]["content"], r["messages"][1]["content"]
    want = list(S["files"])
    fdesc = "\n".join(f"- `{k}`: {v}" for k, v in S["files"].items())
    msgs = [{"role": "user", "content": P_PROJ.format(L=S["name"], cmd=S["cmd"], files=fdesc, q=q[:5000], a=a[:7000])}]
    hist = []
    for step in range(A.fix + 1):
        raw = chat(msgs, lang, rid, step)
        msgs.append({"role": "assistant", "content": raw})
        files, expl = parse_files(raw, want)
        if files is None:
            res = {"rc": -1, "out": f"thiếu file — cần đúng các file {want}, mỗi file một dòng FILE: rồi một khối code", "timeout": False}
        else:
            res = run_project({**files, **S.get("extra", {})}, S["cmd"], A.timeout)
        weak = files is not None and res["rc"] == 0 and not strong(lang, files[want[-1]])
        ok = res["rc"] == 0 and not res["timeout"] and files is not None and not weak
        logw(lang, id=rid, ev="run", step=step, files=files, cmd=S["cmd"], rc=res["rc"], timeout=res["timeout"],
             out=res["out"], ok=ok, weak_test=weak)
        hist.append({"step": step, "ok": ok, "rc": res["rc"]})
        if ok:
            sol = files[want[0]].rstrip()
            ans = (expl + "\n\n" if expl else "") + f"```{S['fence']}\n{sol}\n```"
            return {"id": rid + "_v", "lang": lang, "src": r.get("src"), "messages": [{"role": "user", "content": q},
                    {"role": "assistant", "content": ans}], "files": files, "cmd": S["cmd"], "rounds": step + 1}
        if weak:
            res = dict(res, out=res["out"] + "\n[TEST QUÁ YẾU: cần ít nhất 4 phép kiểm có đáp án cụ thể]")
        msgs.append({"role": "user", "content": P_FIX.format(cmd=S["cmd"], rc=res["rc"], out=res["out"][-2500:])})
    return None


def ordered(q):
    """lời giải có ORDER BY ở câu ngoài cùng → thứ tự là một phần của đáp án (top N, xếp hạng…)."""
    return bool(re.search(r"ORDER\s+BY[^()]*$", re.sub(r"\([^()]*\)", "", q.strip().rstrip(";")), re.I | re.S))


def trivial(rows):
    return all(v is None for r in rows for v in r)


def sql_rows(ctx, q, keep_order=False):
    con = sqlite3.connect(":memory:")
    try:
        con.executescript(ctx)
        rows = con.execute(q).fetchall()
        if trivial(rows):
            return "LỖI: kết quả toàn NULL"
        rows = list(map(repr, rows))
        return rows if keep_order else sorted(rows)
    except Exception as e:
        return f"LỖI: {e}"
    finally:
        con.close()


def do_sql(r):
    rid = r["id"]
    u = r["messages"][0]["content"]
    m = re.search(r"```sql\n(.*?)```\s*(.*)", u, re.S)
    ref = re.search(r"```sql\n(.*?)```", r["messages"][1]["content"], re.S)
    if not m or not ref:
        return None
    ctx, question = m.group(1), m.group(2).strip()
    keep = ordered(ref.group(1))
    want = sql_rows(ctx, ref.group(1), keep)
    if not isinstance(want, list) or not want:
        logw("sql", id=rid, ev="ref_bad", ref=ref.group(1), got=want)
        return None
    for step in range(A.fix + 1):
        raw = chat([{"role": "user", "content": P_SQL.format(ctx=ctx, q=question)}], "sql", rid, step, max_tokens=1200,
                   temp=0.2 if step == 0 else 0.7)
        qs = re.findall(r"```sql\n(.*?)```", raw, re.S)
        got = sql_rows(ctx, qs[0], keep) if len(qs) == 1 else "không có đúng 1 khối ```sql"
        ok = got == want
        logw("sql", id=rid, ev="run", step=step, query=qs[0] if qs else None, ref=ref.group(1), ordered=keep, want=want[:50],
             got=got[:50] if isinstance(got, list) else got, ok=ok)
        if ok:
            return {"id": rid + "_v", "lang": "sql", "src": r.get("src"), "messages": [
                {"role": "user", "content": f"Mình có cơ sở dữ liệu SQLite như sau:\n```sql\n{ctx.strip()}\n```\n{question}"},
                {"role": "assistant", "content": raw.strip()}], "expected_rows": want, "rounds": step + 1}
    return None


MSSQL_SEM = threading.Semaphore(8)                # SQL Server Express: tối đa 4 lõi → ít kết nối cùng lúc


def norm(rows, keep_order=False):
    """chuẩn hoá để so kết quả SQLite ↔ SQL Server (Decimal/float, ngày giờ, bool, khoảng trắng)."""
    import datetime, decimal
    out = []
    for r in rows:
        t = []
        for v in r:
            if isinstance(v, bool):
                v = int(v)
            if isinstance(v, (int, float, decimal.Decimal)):
                v = round(float(v), 2)
            elif isinstance(v, (datetime.datetime, datetime.date)):
                v = v.isoformat()[:10] if isinstance(v, datetime.date) and not isinstance(v, datetime.datetime) or \
                    (isinstance(v, datetime.datetime) and v.time() == datetime.time()) else v.isoformat(sep=" ")[:19]
            elif isinstance(v, str):
                v = v.strip()
                try:
                    v = round(float(v), 2)
                except ValueError:
                    pass
            t.append(v)
        out.append(repr(tuple(t)))
    return out if keep_order else sorted(out)


def mssql_run(setup, query, keep_order=False):
    import pymssql, uuid
    db = "c2b_" + uuid.uuid4().hex[:12]
    pw = open("/root/.c2b_mssql_login").read().strip()
    with MSSQL_SEM:
        con = pymssql.connect("127.0.0.1", "c2b", pw, "master", autocommit=True, timeout=30, login_timeout=15)
        try:
            cur = con.cursor()
            cur.execute(f"CREATE DATABASE [{db}]")
            cur.execute(f"USE [{db}]")
            cur.execute(setup)
            while cur.nextset():
                pass
            cur.execute(query)
            rows = cur.fetchall()
            if trivial(rows):
                return "LỖI: kết quả toàn NULL"
            return norm(rows, keep_order)
        except Exception as e:
            return f"LỖI: {str(e)[:600]}"
        finally:
            try:
                cur.execute("USE master")
                cur.execute(f"ALTER DATABASE [{db}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE; DROP DATABASE [{db}]")
            except Exception:
                pass
            con.close()


def do_tsql(r):
    rid = r["id"]
    u = r["messages"][0]["content"]
    m = re.search(r"```sql\n(.*?)```\s*(.*)", u, re.S)
    ref = re.search(r"```sql\n(.*?)```", r["messages"][1]["content"], re.S)
    if not m or not ref:
        return None
    ctx, question = m.group(1), m.group(2).strip()
    con = sqlite3.connect(":memory:")
    try:
        con.executescript(ctx)
        keep = ordered(ref.group(1))
        rows = con.execute(ref.group(1)).fetchall()
        want = [] if trivial(rows) else norm(rows, keep)
    except Exception as e:
        logw("tsql", id=rid, ev="ref_bad", err=str(e)[:200])
        return None
    finally:
        con.close()
    if not want:
        return None
    msgs = [{"role": "user", "content": P_TSQL.format(ctx=ctx, q=question, ref=ref.group(1))}]
    for step in range(A.fix + 1):
        raw = chat(msgs, "tsql", rid, step, max_tokens=3000)
        msgs.append({"role": "assistant", "content": raw})
        files, expl = parse_files(raw, ["setup.sql", "query.sql"])
        got = mssql_run(files["setup.sql"], files["query.sql"], keep) if files else "thiếu file setup.sql / query.sql"
        ok = got == want
        logw("tsql", id=rid, ev="run", step=step, files=files, ref=ref.group(1), want=want[:50],
             got=got[:50] if isinstance(got, list) else got, ok=ok)
        if ok:
            user = (f"Mình dùng SQL Server, có các bảng sau:\n```sql\n{files['setup.sql'].strip()}\n```\n{question}")
            ans = (expl + "\n\n" if expl else "") + f"```sql\n{files['query.sql'].strip()}\n```"
            return {"id": rid + "_tsql", "lang": "tsql", "src": r.get("src"), "messages": [
                {"role": "user", "content": user}, {"role": "assistant", "content": ans}], "files": files,
                "expected_rows": want, "rounds": step + 1}
        why = got if isinstance(got, str) else (f"kết quả khác lời giải SQLite{' (thứ tự dòng cũng phải đúng)' if keep else ''}."
                                                f"\nCần: {want[:10]}\nĐược: {got[:10]}")
        msgs.append({"role": "user", "content": f"Chạy trên SQL Server: {why}\nHãy sửa và trả lời lại TOÀN BỘ theo đúng khuôn cũ."})
    return None


_PW = threading.local()


SHOTS = f"{OUT}/html_shots"
P_LOOK = """Bạn là người duyệt giao diện web khó tính. Ảnh 1: trang ở khổ máy tính (1280px). Ảnh 2: cùng trang trên điện thoại (390px).
=== YÊU CẦU GỐC CỦA NGƯỜI DÙNG ===
{q}
=== HẾT ===
Chấm từng mục 0–10: dung_yeu_cau (trang làm đúng điều được yêu cầu), bo_cuc (thẳng hàng, khoảng cách hợp lý, không chồng
chéo, không vỡ), de_doc (chữ rõ, đủ tương phản, cỡ chữ hợp lý), dien_thoai (trên điện thoại vẫn dùng được, không tràn ngang).
Trả về DUY NHẤT JSON: {{"dung_yeu_cau": n, "bo_cuc": n, "de_doc": n, "dien_thoai": n, "loi": ["lỗi cụ thể nhìn thấy", ...]}}"""
P_HTML_FIX = """Trang web dưới đây chưa đạt khi kiểm tra:
{errs}
=== YÊU CẦU GỐC ===
{q}
=== TRANG ===
```html
{html}
```
Sửa trang cho hết các lỗi trên, giữ đúng yêu cầu gốc. Trả lời: MỘT khối ```html chứa toàn bộ trang đã sửa, rồi một dòng
`GIẢI THÍCH:` 1–2 câu tiếng Việt nói đã sửa gì."""
JS_METRICS = """() => {
  const vw = window.innerWidth, sw = document.documentElement.scrollWidth;
  const imgs = [...document.images].filter(i => i.complete && i.naturalWidth === 0 && !i.src.startsWith('data:')).length;
  const leaves = [...document.querySelectorAll('body *')].filter(e => e.children.length === 0 && e.innerText &&
      e.innerText.trim().length > 1).slice(0, 300).map(e => e.getBoundingClientRect()).filter(r => r.width > 2 && r.height > 2);
  let ov = 0;
  for (let i = 0; i < leaves.length; i++) for (let j = i + 1; j < leaves.length; j++) {
    const a = leaves[i], b = leaves[j];
    const w = Math.min(a.right, b.right) - Math.max(a.left, b.left), h = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
    if (w > 0 && h > 0 && w * h > 0.5 * Math.min(a.width * a.height, b.width * b.height)) ov++;
  }
  return {overflow: sw - vw, broken_img: imgs, overlap: ov, text: (document.body ? document.body.innerText.trim().length : 0)};
}"""


def render(html, rid=None, save=False):
    """mở trang bằng Chromium không giao diện ở 2 khổ → (lỗi JS, chỉ số bố cục, ảnh desktop, ảnh điện thoại [JPEG bytes]).
    Chỉ cho tải từ CDN phổ biến; bỏ lỗi tải mạng."""
    if not hasattr(_PW, "b"):
        from playwright.sync_api import sync_playwright
        _PW.p = sync_playwright().start()
        _PW.b = _PW.p.chromium.launch(args=["--no-sandbox"])
    allow = ("cdn.jsdelivr.net", "cdnjs.cloudflare.com", "unpkg.com", "cdn.tailwindcss.com", "code.jquery.com",
             "fonts.googleapis.com", "fonts.gstatic.com", "stackpath.bootstrapcdn.com", "maxcdn.bootstrapcdn.com")
    errs, met, shots = [], {}, {}
    for kind, vp in (("d", {"width": 1280, "height": 800}), ("m", {"width": 390, "height": 844})):
        pg = _PW.b.new_page(viewport=vp, device_scale_factor=1)
        try:
            pg.route("**/*", lambda rt: rt.continue_() if rt.request.url.startswith(("data:", "about:")) or
                     any(h in rt.request.url for h in allow) else rt.abort())
            if kind == "d":
                pg.on("pageerror", lambda e: errs.append(f"lỗi JS: {e}"))
                pg.on("console", lambda m: errs.append(f"console.error: {m.text}") if m.type == "error" and
                      "Failed to load resource" not in m.text and "net::" not in m.text else None)
            pg.set_content(html, timeout=15000, wait_until="load")
            pg.wait_for_timeout(600)
            met[kind] = pg.evaluate(JS_METRICS)
            h = min(pg.evaluate("document.documentElement.scrollHeight"), 2400)
            shots[kind] = pg.screenshot(type="jpeg", quality=70, clip={"x": 0, "y": 0, "width": vp["width"], "height": max(h, 200)})
        except Exception as e:
            errs.append(f"không mở được ({kind}): {str(e)[:200]}")
        finally:
            pg.close()
    # ── tầng 1: kiểm tự động trên ảnh + bố cục ──
    if "d" in shots:
        import io
        from PIL import Image, ImageStat
        im = Image.open(io.BytesIO(shots["d"])).convert("L")
        if max(ImageStat.Stat(im).stddev) < 4:
            errs.append("trang gần như một màu (trắng/trống)")
    if met.get("d", {}).get("text", 1) == 0 and "d" in shots:
        errs.append("trang không có chữ nào")
    if met.get("m", {}).get("overflow", 0) > 8:
        errs.append(f"trên điện thoại (390px) trang tràn ngang {met['m']['overflow']}px")
    if met.get("d", {}).get("broken_img", 0):
        errs.append(f"{met['d']['broken_img']} ảnh không hiển thị được")
    if met.get("d", {}).get("overlap", 0) > 2:
        errs.append(f"{met['d']['overlap']} cặp khối chữ chồng lên nhau")
    if save and rid and "d" in shots:                     # ── tầng 3: lưu ảnh ↔ HTML (dữ liệu cho expert đọc ảnh) ──
        os.makedirs(SHOTS, exist_ok=True)
        for k, b in shots.items():
            open(f"{SHOTS}/{rid}_{k}.jpg", "wb").write(b)
    return errs[:10], met, shots


def page_errors(html):
    return render(html)[0]


def look(q, shots, rid, step):
    """── tầng 2: thầy NHÌN ảnh 2 khổ, chấm theo yêu cầu gốc ──"""
    import base64
    from gen_office import loads_loose
    if not A.vision or "d" not in shots or "m" not in shots:
        return True, {}
    content = [{"type": "text", "text": P_LOOK.format(q=q[:2500])}] + [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(shots[k]).decode()}}
        for k in ("d", "m")]
    raw = chat([{"role": "user", "content": content}], "html_css", rid, f"look{step}", max_tokens=600, temp=0.0)
    try:
        sc = loads_loose(raw)
        keys = ("dung_yeu_cau", "bo_cuc", "de_doc", "dien_thoai")
        v = [float(sc[k]) for k in keys]
        return (min(v) >= 6 and sum(v) / 4 >= 7), sc
    except Exception:
        return False, {"loi": ["thầy chấm không ra JSON"], "raw": raw[:300]}


def do_html(r):
    rid = r["id"]
    q, a = r["messages"][0]["content"], r["messages"][1]["content"]
    hs = re.findall(r"```html\n(.*?)```", a, re.S)
    if len(hs) != 1:
        return None
    html, ans = hs[0], a
    for step in range(A.fix + 1):
        errs, met, shots = render(html)
        good, sc = (look(q, shots, rid, step) if not errs else (False, {}))
        ok = not errs and good
        logw("html_css", id=rid, ev="run", step=step, html=html, errors=errs, metrics=met, look=sc, ok=ok)
        if ok:
            render(html, rid + "_v", save=True)
            return {"id": rid + "_v", "lang": "html_css", "src": r.get("src"), "messages": [
                {"role": "user", "content": q}, {"role": "assistant", "content": ans}], "rounds": step + 1,
                "look": sc, "shots": [f"{SHOTS}/{rid}_v_d.jpg", f"{SHOTS}/{rid}_v_m.jpg"]}
        if step == A.fix:
            break
        why = errs + [f"người duyệt (nhìn ảnh): {x}" for x in (sc.get("loi") or [])][:8] + \
            ([f"điểm duyệt: {', '.join(f'{k}={sc[k]}' for k in ('dung_yeu_cau', 'bo_cuc', 'de_doc', 'dien_thoai') if k in sc)}"]
             if sc and not errs else [])
        raw = chat([{"role": "user", "content": P_HTML_FIX.format(errs="\n".join(f"- {x}" for x in why), q=q[:3000],
                                                                  html=html[:9000])}], "html_css", rid, step + 1, max_tokens=6000)
        hs = re.findall(r"```html\n(.*?)```", raw, re.S)
        if len(hs) != 1:
            logw("html_css", id=rid, ev="parse_fail", step=step + 1)
            return None
        html = hs[0]
        ex = re.search(r"GIẢI THÍCH:\s*(.+)", raw, re.S)
        ans = (ex.group(1).strip() + "\n\n" if ex else "") + f"```html\n{html.rstrip()}\n```"
    return None


def vision_ok():
    """thầy có nhìn được ảnh qua vLLM không? (trang nền đỏ → hỏi màu)"""
    _, _, shots = render("<html><body style='background:#d00;margin:0'><h1 style='color:#fff'>XIN CHAO</h1></body></html>")
    import base64
    content = [{"type": "text", "text": "Màu nền của trang trong ảnh là màu gì và dòng chữ lớn là gì? Trả lời ngắn."},
               {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(shots["d"]).decode()}}]
    raw = chat([{"role": "user", "content": content}], "html_css", "_vision_check", "check", max_tokens=60, temp=0.0)
    return bool(re.search(r"đỏ|red", raw, re.I)) and "XIN" in raw.upper(), raw


def to_vi(res):
    lang, rid = res["lang"], res["id"]
    q, a = res["messages"][0]["content"], res["messages"][1]["content"]
    code = re.findall(r"```.*?```", a, re.S)
    expl = re.sub(r"```.*?```", "", a, flags=re.S).strip()
    raw = chat([{"role": "user", "content": P_VI.format(q=q, e=expl or "(không có)")}], lang, rid, "vi", max_tokens=3000)
    m = re.search(r"<<<DE>>>\s*(.*?)\s*<<<GT>>>\s*(.*)", raw, re.S)
    if not m or re.findall(r"```.*?```", m.group(1), re.S) != re.findall(r"```.*?```", q, re.S) or len(VI_CH.findall(m.group(1))) < 5:
        logw(lang, id=rid, ev="vi", ok=False)
        return None
    gt = m.group(2).strip()
    gt = "" if gt.startswith("(không có") else gt
    logw(lang, id=rid, ev="vi", ok=True)
    return dict(res, id=rid + "_vi", messages=[{"role": "user", "content": m.group(1).strip()},
                {"role": "assistant", "content": (gt + "\n\n" if gt else "") + "\n\n".join(code)}])


def work(item):
    lang, r = item
    try:
        res = do_sql(r) if lang == "sql" else do_tsql(r) if lang == "tsql" else do_html(r) if lang == "html_css" \
            else do_code(lang, r)
        logw(lang, id=r["id"], ev="final", ok=bool(res), rounds=res["rounds"] if res else None)
        vi = to_vi(res) if res and random.Random(r["id"]).random() < A.vi_frac else None
        return lang, res, vi
    except Exception as e:
        logw(lang, id=r["id"], ev="crash", err=f"{type(e).__name__}: {str(e)[:300]}")
        return lang, None, None


def pick(lang, n):
    rows = [json.loads(l) for l in open(f"{OUT}/{'sql' if lang == 'tsql' else lang}.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["src"] != "commit" and len(r["messages"]) == 2]       # cần dạng hỏi–đáp
    if lang == "sql":
        rows = [r for r in rows if r["src"] == "gretel"]                                  # có lược đồ + dữ liệu để chạy
    random.Random(208).shuffle(rows)
    if lang == "tsql":
        rows = rows[len(rows) // 2:]                                                     # khác tập bài với "sql"
    done = set()
    p = f"{OUT}/teacher_log/{lang}.jsonl"
    if os.path.exists(p):
        done = {json.loads(l)["id"] for l in open(p, encoding="utf-8") if '"ev": "final"' in l}
    return [r for r in rows[:n] if r["id"] not in done]


SELF = {   # (lời giải, test đúng, test sai) — kiểm khung chạy của từng ngôn ngữ trước khi tốn GPU
    "javascript": ("module.exports = { add: (a, b) => a + b };", "const assert=require('assert');const {add}=require('./solution');"
                   "assert.strictEqual(add(1,2),3);assert.strictEqual(add(0,0),0);assert.strictEqual(add(-1,1),0);", "X"),
    "typescript": ("export function add(a: number, b: number): number { return a + b; }",
                   "import assert from 'assert'; import { add } from './solution'; assert.strictEqual(add(1, 2), 3);"
                   "assert.strictEqual(add(0, 0), 0); assert.strictEqual(add(2, 2), 4);", "X"),
    "java": ("class Solution { static int add(int a, int b) { return a + b; } }",
             "public class Main { public static void main(String[] a) { if (Solution.add(1,2)!=3) throw new AssertionError(); "
             "if (Solution.add(0,0)!=0) throw new AssertionError(); if (Solution.add(2,2)!=4) throw new AssertionError(); } }", "X"),
    "csharp": ("public class Solution { public static int Add(int a, int b) => a + b; }",
               "class Program { static void Main() { if (Solution.Add(1,2)!=3) throw new Exception(); "
               "if (Solution.Add(0,0)!=0) throw new Exception(); if (Solution.Add(2,2)!=4) throw new Exception(); } }", "X"),
    "php": ("<?php function add($a, $b) { return $a + $b; }", "<?php require 'solution.php'; if (add(1,2)!==3) throw new Exception('a');"
            "if (add(0,0)!==0) throw new Exception('b'); if (add(2,2)!==4) throw new Exception('c');", "X"),
    "go": ("package main\n\nfunc Add(a, b int) int { return a + b }\n\nfunc main() {}\n",
           "package main\n\nimport \"testing\"\n\nfunc TestAdd(t *testing.T) {\n\tif Add(1, 2) != 3 { t.Errorf(\"a\") }\n"
           "\tif Add(0, 0) != 0 { t.Errorf(\"b\") }\n\tif Add(2, 2) != 4 { t.Errorf(\"c\") }\n}\n", "X"),
    "bash": ("add() { echo $(( $1 + $2 )); }", "source ./solution.sh\n[ \"$(add 1 2)\" = 3 ] || exit 1\n"
             "[ \"$(add 0 0)\" = 0 ] || exit 1\n[ \"$(add 2 2)\" = 4 ] || exit 1\n", "X"),
}


def selftest():
    good = True
    for lang, (sol, test, _) in SELF.items():
        S = LANG[lang]; f = list(S["files"])
        r1 = run_project({f[0]: sol, f[1]: test, **S.get("extra", {})}, S["cmd"], 120)
        bad = sol.replace("a + b", "a - b").replace("$1 + $2", "$1 - $2").replace("$a + $b", "$a - $b")
        r2 = run_project({f[0]: bad, f[1]: test, **S.get("extra", {})}, S["cmd"], 120)
        ok = r1["rc"] == 0 and r2["rc"] != 0
        good &= ok
        print(f"[selftest] {lang}: {'OK' if ok else 'HỎNG'} (đúng rc={r1['rc']}, sai rc={r2['rc']})"
              + ("" if ok else f"\n{r1['out'][-800:]}"), flush=True)
    e1 = page_errors("<html><body style='font-family:sans-serif;padding:24px'><h1>Cửa hàng hoa</h1><p>Hoa tươi giao trong ngày "
                     "tại Hà Nội.</p><button style='padding:8px 16px;background:#2a7;color:#fff;border:0'>Đặt hoa</button>"
                     "<script>document.title='x'</script></body></html>")
    e2 = page_errors("<html><body><p>x</p><script>khongco()</script></body></html>")
    e3 = page_errors("<html><body><h1>Rộng</h1><div style='width:900px;height:40px;background:#39c'>bảng</div></body></html>")
    e4 = page_errors("<html><body></body></html>")
    ok = not e1 and bool(e2) and any("tràn ngang" in x for x in e3) and bool(e4); good &= ok
    print(f"[selftest] html_css: {'OK' if ok else 'HỎNG'} tốt={e1} lỗiJS={e2} tràn={e3} trống={e4}", flush=True)
    if "tsql" in (A.langs or ""):
        g = mssql_run("CREATE TABLE t (a INT, b NVARCHAR(10)); INSERT INTO t VALUES (1, N'x'), (2, N'y');",
                      "SELECT TOP 1 b, SUM(a) OVER () FROM t ORDER BY a DESC")
        b = mssql_run("CREATE TABLE t (a INT);", "SELECT khongco FROM t")
        ok = g == ["('y', 3.0)"] and isinstance(b, str); good &= ok
        print(f"[selftest] tsql: {'OK' if ok else 'HỎNG'} {g} {str(b)[:80]}", flush=True)
    print(f"### c2b selftest: {'OK' if good else 'HỎNG'}", flush=True)
    return good


def mssql_login():
    """tạo login c2b (chỉ quyền dbcreator) — mỗi bài T-SQL chạy trong database riêng của nó."""
    import pymssql
    sa, pw = open("/root/.c2b_mssql_pw").read().strip(), open("/root/.c2b_mssql_login").read().strip()
    c = pymssql.connect("127.0.0.1", "sa", sa, "master", autocommit=True)
    k = c.cursor()
    k.execute("SELECT SUSER_ID('c2b')")
    if k.fetchone()[0] is None:
        k.execute("CREATE LOGIN c2b WITH PASSWORD = %s, CHECK_POLICY = OFF", (pw,))
    k.execute("ALTER SERVER ROLE dbcreator ADD MEMBER c2b")
    print("login c2b ok", flush=True)


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="javascript,typescript,java,csharp,php,go,bash,sql,tsql,html_css")
    ap.add_argument("--per-lang", type=int, default=4000)
    ap.add_argument("--hours", type=float, default=0)
    ap.add_argument("--fix", type=int, default=2, help="số lượt thầy sửa sau khi chạy hỏng")
    ap.add_argument("--timeout", type=int, default=60, help="giây cho biên dịch + chạy test")
    ap.add_argument("--vi-frac", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--llm", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma4-31b")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mssql-login", action="store_true")
    ap.add_argument("--no-vision", action="store_true", help="tắt tầng 2 (thầy nhìn ảnh)")
    A = ap.parse_args()
    if A.mssql_login:
        return mssql_login()
    if A.selftest:
        A.vision = False
        sys.exit(0 if selftest() else 1)
    langs = A.langs.split(",")
    A.vision = False
    if "html_css" in langs and not A.no_vision:
        A.vision, said = vision_ok()
        print(f"[c2bt] thầy nhìn ảnh: {'OK' if A.vision else 'KHÔNG — chỉ dùng tầng 1 + 3'} ({said.strip()[:80]})", flush=True)
    per = {l: pick(l, A.per_lang) for l in langs}
    print(f"[c2bt] cần làm: " + " · ".join(f"{l} {len(v)}" for l, v in per.items()), flush=True)
    items = [x for grp in itertools.zip_longest(*[[(l, r) for r in per[l]] for l in langs]) for x in grp if x]  # xen kẽ ngôn ngữ
    deadline = time.time() + A.hours * 3600 if A.hours else float("inf")
    fo = {l: open(f"{OUT}/verified_{l}.jsonl", "a", encoding="utf-8") for l in langs}
    n, ok, vi = {l: 0 for l in langs}, {l: 0 for l in langs}, {l: 0 for l in langs}
    T0, it = time.time(), iter(items)
    with ThreadPoolExecutor(A.workers) as ex:
        futs = set()

        def feed():
            while len(futs) < A.workers * 2 and time.time() < deadline:
                x = next(it, None)
                if x is None:
                    return
                futs.add(ex.submit(work, x))
        feed()
        k = 0
        while futs:
            f = next(as_completed(futs)); futs.discard(f)
            lang, res, v = f.result()
            n[lang] += 1; k += 1
            for x in (res, v):
                if x:
                    fo[lang].write(json.dumps(x, ensure_ascii=False) + "\n")
            ok[lang] += bool(res); vi[lang] += bool(v)
            if k % 200 == 0:
                for h in list(fo.values()) + list(_LOGS.values()):
                    h.flush()
                print(f"[c2bt] {k} · " + " · ".join(f"{l} {ok[l]}/{n[l]}" for l in langs) + f" · {time.time()-T0:.0f}s", flush=True)
            feed()
    for h in list(fo.values()) + list(_LOGS.values()):
        h.close()
    print("### c2b B: " + " · ".join(f"{l} {ok[l]}/{n[l]} (+vi {vi[l]})" for l in langs) +
          f" · tổng qua {sum(ok.values())} · {time.time()-T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
