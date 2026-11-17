"""KHUNG CHẠY CÓ CÔNG CỤ cho aivie viết code (TK_108): model viết code → CHẠY THẬT trong sandbox → đọc lỗi → sửa.

Hai chế độ:
  debug  vòng "viết → chạy test CÔNG KHAI → đưa lỗi lại → sửa" tối đa N vòng (dùng được với MỌI model chat ngay)
  tool   giao thức gọi công cụ trong thư mục dự án (bộ công cụ kiểu Claude Code / Cursor): mỗi lượt đúng một công cụ
             <tool>{"name": "edit_file", "args": {"path": "src.py", "old": "...", "new": "..."}}</tool>  → <obs>…</obs>
         list_dir · glob · grep · search_code · read_file (theo đoạn, có số dòng) · edit_file (thay đoạn DUY NHẤT, phải đọc
         trước) · write_file · delete_file · bash (sandbox, không mạng) · run_tests · lint ;  xong: <done>tóm tắt</done>
         → quỹ đạo (từng lượt + kết quả thật) ghi jsonl — sau này dùng làm dữ liệu SFT/RL tool-use.

Sandbox: macOS `sandbox-exec` (cấm mạng, cấm ghi ngoài thư mục làm việc) · Linux `bwrap` nếu có · luôn giới hạn CPU/bộ nhớ/
thời gian/độ dài đầu ra. Model: UI của ta (`/api/chat`, serve_hybrid_ui.py) hoặc máy chủ kiểu OpenAI (`/v1`, vLLM/Ollama — thầy).
"""
from __future__ import annotations
import json, os, re, resource, shutil, subprocess, sys, tempfile, time, urllib.error, urllib.request

PY = sys.executable
GUARD = os.environ.get("AIVIE_TOOL_GUARD", "1") != "0"   # khung v2: chống lặp + báo rõ lệch khoảng trắng (0 = khung cũ, để so)
OUT_CAP = 3000                                        # ký tự đầu ra tối đa đưa lại cho model


# ── sandbox ──────────────────────────────────────────────────────────────────────────────
def _limits(cpu=20, mem_mb=1024):
    def f():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
        except (ValueError, OSError):                 # macOS không cho đặt RLIMIT_AS → bỏ qua
            pass
        resource.setrlimit(resource.RLIMIT_FSIZE, (50 << 20, 50 << 20))
        _nproc()
        os.setsid()
    return f


NPROC_SBX = int(os.environ.get("AIVIE_NPROC", "2500"))  # trần tiến trình/luồng user sandbox (16/10: fork bomb bash 5.653 tiến trình làm sập box)


def _nproc():
    if os.environ.get("AIVIE_SANDBOX_USER") and hasattr(resource, "RLIMIT_NPROC"):
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (NPROC_SBX, NPROC_SBX))
        except (ValueError, OSError):
            pass


def kill_group(pgid):
    """diệt CẢ nhóm tiến trình kể cả khi nó đang tự nhân bản: đóng băng (SIGSTOP) lặp tới khi không còn ai sinh thêm,
    rồi SIGKILL lặp tới khi nhóm trống. Diệt một lần không đủ — bản sao mới sinh sau lúc gửi tín hiệu sẽ sống sót."""
    import signal
    for sig, n in ((signal.SIGSTOP, 50), (signal.SIGKILL, 200)):
        for _ in range(n):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return
            except PermissionError:
                break
            if sig == signal.SIGKILL:
                time.sleep(0.02)


def run_group(cmd, cwd, env, timeout, preexec, stdin=None):
    """chạy trong NHÓM tiến trình riêng (preexec phải setsid); output ghi ra file tạm (tiến trình nền không giữ pipe);
    xong hoặc hết giờ đều diệt cả nhóm → không sót tiến trình con. → (rc, out, hết_giờ)"""
    with tempfile.TemporaryFile() as fo:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                             stdout=fo, stderr=subprocess.STDOUT, preexec_fn=preexec)
        if stdin is not None:
            try:
                p.stdin.write(stdin.encode() if isinstance(stdin, str) else stdin); p.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        to = False
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            to = True
        kill_group(p.pid)
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        fo.seek(0)
        out = fo.read().decode("utf-8", errors="replace")
    return (-9 if to else p.returncode), out, to


def _wrap(cmd, workdir):
    """bọc lệnh trong sandbox của hệ điều hành."""
    wd = os.path.realpath(workdir)
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        tmp = os.path.realpath(tempfile.gettempdir())
        home = os.path.realpath(os.path.expanduser("~"))
        secret = "".join(f' (subpath "{home}/{d}")' for d in (".ssh", ".cache/huggingface", ".aws", ".config", "Library/Keychains"))
        prof = (f'(version 1)(allow default)(deny network*)(deny file-write*)'
                f'(allow file-write* (subpath "{wd}") (subpath "{tmp}") (literal "/dev/null"))'
                f'(deny file-read*{secret})')                                       # khoá SSH, token HF… không đọc được
        return ["sandbox-exec", "-p", prof] + cmd
    if shutil.which("bwrap"):
        return ["bwrap", "--ro-bind", "/", "/", "--bind", wd, wd, "--tmpfs", "/tmp", "--dev", "/dev", "--proc", "/proc",
                "--unshare-net", "--die-with-parent", "--chdir", wd] + cmd
    su = os.environ.get("AIVIE_SANDBOX_USER")       # box Linux không bwrap: chạy dưới user riêng KHÔNG ghi được /workspace (C3)
    if su and hasattr(os, "geteuid") and os.geteuid() == 0:
        return [shutil.which("runuser") or "/usr/sbin/runuser", "-u", su, "--"] + cmd
    return cmd                                        # chỉ còn giới hạn tài nguyên (box dùng xong bỏ)


def _own(wd):
    """thư mục làm việc thuộc về user sandbox (để công cụ ghi/sửa được trong đó, và CHỈ trong đó)."""
    su = os.environ.get("AIVIE_SANDBOX_USER")
    if su and hasattr(os, "geteuid") and os.geteuid() == 0:
        for root, dirs, files in os.walk(wd):
            for x in [root] + [os.path.join(root, f) for f in dirs + files]:
                shutil.chown(x, su, su)


def run(cmd, workdir, timeout=20, stdin=None):
    """→ {"rc", "out", "timeout"} — out = stdout+stderr đã cắt gọn."""
    rc, out, to = run_group(_wrap(cmd, workdir), workdir, {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": workdir,
                                                          "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"},
                            timeout, _limits(cpu=timeout + 2), stdin)
    if len(out) > OUT_CAP:
        out = out[:OUT_CAP // 2] + "\n…(cắt bớt)…\n" + out[-OUT_CAP // 2:]
    return {"rc": rc, "out": out, "timeout": to}


def run_python(code, workdir=None, timeout=20):
    own = workdir is None
    wd = workdir or tempfile.mkdtemp(prefix="aivie_run_")
    try:
        with open(os.path.join(wd, "_main.py"), "w", encoding="utf-8") as f:
            f.write(code)
        _own(wd)
        return run([PY, "-B", "-I", "_main.py"], wd, timeout)
    finally:
        if own:
            shutil.rmtree(wd, ignore_errors=True)


# ── model ────────────────────────────────────────────────────────────────────────────────
class Model:
    """--url http://localhost:8798 → UI của ta (/api/chat, luồng SSE) · …/v1 → máy chủ kiểu OpenAI (vLLM / Ollama)."""
    def __init__(self, url, name="", max_new=600, temp=0.0, model_id=""):
        self.url, self.name, self.max_new, self.temp, self.model_id = url.rstrip("/"), name, max_new, temp, model_id
        self.n_call, self.n_tok_s = 0, 0.0

    def chat(self, msgs):
        self.n_call += 1
        t = time.time()
        if self.url.endswith("/v1"):
            body = {"model": self.model_id, "messages": msgs, "temperature": self.temp, "max_tokens": self.max_new,
                    "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(self.url + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            out = json.loads(urllib.request.urlopen(req, timeout=900).read())["choices"][0]["message"]["content"]
        else:
            body = {"conv_id": f"agent-{os.getpid()}-{self.n_call}", "messages": msgs, "temperature": self.temp, "top_p": 1.0,
                    "rep_pen": 1.1, "max_new": self.max_new, "rag": False}
            out = None
            for _ in range(120):
                try:
                    req = urllib.request.Request(self.url + "/api/chat", data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=900) as res:
                        for line in res:
                            line = line.decode().strip()
                            if line.startswith("data: "):
                                ev = json.loads(line[6:])
                                if ev["type"] == "done":
                                    out = ev.get("text", "")
                                elif ev["type"] == "error":
                                    out = ""
                    break
                except urllib.error.HTTPError as e:
                    if e.code != 429:
                        raise
                    time.sleep(3)                           # UI đang bận câu khác
            out = out or ""
        self.n_tok_s += time.time() - t
        return out.strip()


def extract_code(text):
    """khối ```python … ``` DÀI nhất (không có khối → cả câu trả lời nếu trông như code)."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len).strip("\n")
    return text if re.search(r"^\s*(def |import |from |class )", text, re.M) else ""


# ── chế độ debug: viết → chạy test công khai → sửa ────────────────────────────────────────
P_SOLVE = """{task}

Viết lời giải Python hoàn chỉnh (kèm các dòng import cần thiết), đặt trong MỘT khối ```python. Không viết test."""
P_FIX = """Mình đã chạy code của bạn với các test sau:
```python
{tests}
```
Kết quả:
```
{out}
```
Code chưa đúng. Hãy đọc lỗi, tìm nguyên nhân và viết lại TOÀN BỘ lời giải đã sửa trong MỘT khối ```python."""


def solve_debug(model, task, public_tests, prefix="", rounds=3, timeout=10):
    """→ (code cuối, lịch sử các vòng). public_tests: chuỗi assert/lệnh gọi mà model ĐƯỢC THẤY khi sửa."""
    msgs = [{"role": "user", "content": P_SOLVE.format(task=task)}]
    hist, code = [], ""
    for r in range(rounds + 1):
        ans = model.chat(msgs)
        code = extract_code(ans) or code
        res = run_python(prefix + code + "\n\n" + public_tests + "\n", timeout=timeout)
        ok = res["rc"] == 0 and not res["timeout"]
        hist.append({"round": r, "answer": ans, "code": code, "public_ok": ok, "out": res["out"][-800:]})
        if ok or r == rounds:
            break
        msgs += [{"role": "assistant", "content": ans},
                 {"role": "user", "content": P_FIX.format(tests=public_tests.strip(), out=(res["out"] or "(hết thời gian)").strip()[-1500:])}]
    return code, hist


# ── chế độ tool: bộ công cụ kiểu Claude Code / Cursor trong thư mục dự án ─────────────────────
TOOLS_DOC = """- list_dir {"path": "."} — liệt kê thư mục (file + thư mục con)
- glob {"pattern": "**/*.py"} — tìm file theo mẫu tên
- grep {"pattern": "regex", "path": ".", "glob": "*.py", "context": 0, "ignore_case": false} — tìm bằng regex trong file
- search_code {"query": "mô tả bằng lời"} — tìm hàm/lớp liên quan theo từ khoá (khi chưa biết tên chính xác)
- read_file {"path": "...", "offset": 1, "limit": 200} — đọc file có SỐ DÒNG (offset/limit để đọc từng đoạn)
- edit_file {"path": "...", "old": "...", "new": "...", "all": false} — thay ĐÚNG một đoạn: "old" phải xuất hiện DUY NHẤT
  (copy y nguyên từ read_file, bỏ phần "số dòng| " ở đầu); "all": true để thay mọi chỗ. Phải read_file trước khi sửa
- write_file {"path": "...", "content": "..."} — tạo file MỚI (file đã có: dùng edit_file, hoặc read_file trước rồi mới ghi đè)
- delete_file {"path": "..."} — xoá file
- bash {"command": "...", "timeout": 30} — chạy lệnh shell trong sandbox (KHÔNG có mạng), vd: python3 x.py, pytest -q, git diff
- run_tests {} — chạy bộ test của dự án
- lint {"path": "..."} — kiểm lỗi cú pháp / tên chưa định nghĩa trước khi chạy"""

P_TOOL = """Bạn là aivie, lập trình viên làm việc trong một thư mục dự án. Nhiệm vụ:
{task}

Mỗi lượt bạn gọi ĐÚNG MỘT công cụ, viết đúng dạng:
<tool>{{"name": "<tên>", "args": {{...}}}}</tool>
Các công cụ:
""" + TOOLS_DOC.replace("{", "{{").replace("}", "}}") + """
Sau mỗi lần gọi bạn nhận kết quả trong <obs>…</obs>. Cách làm tốt: tìm file liên quan → đọc → sửa nhỏ bằng edit_file → chạy test →
đọc lỗi → sửa tiếp cho tới khi test qua. Khi xong, viết: <done>tóm tắt ngắn đã làm gì</done>"""


def parse_tool(text):
    if re.search(r"<done>", text) and not re.search(r"<tool>", text):
        return {"name": "done", "args": {"summary": re.sub(r".*<done>|</done>.*", "", text, flags=re.S).strip()}}
    m = re.search(r"<tool>\s*(\{.*\})\s*</tool>", text, re.S)
    if not m:
        return {"name": "done", "args": {"summary": ""}} if re.search(r"<done>", text) else None
    try:
        j = json.loads(m.group(1))
        return {"name": str(j.get("name", "")), "args": j.get("args") or {}}
    except json.JSONDecodeError:
        return {"name": "_bad_json", "args": {"raw": m.group(1)[:300]}}


def _num(lines, start):
    return "\n".join(f"{i:>4}| {l}" for i, l in enumerate(lines, start))


SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache", ".pytest_cache"}


class Workspace:
    """thư mục dự án + trạng thái phiên (file đã đọc — bắt buộc đọc trước khi sửa, như Claude Code)."""
    def __init__(self, wd, test_cmd=None):
        self.wd, self.test_cmd, self.read = os.path.realpath(wd), test_cmd, set()

    def path(self, p):
        full = os.path.realpath(os.path.join(self.wd, str(p or ".")))
        return full if full == self.wd or full.startswith(self.wd + os.sep) else None

    def rel(self, full):
        return os.path.relpath(full, self.wd)

    def files(self, base=None):
        for r, ds, fs in os.walk(base or self.wd):
            ds[:] = sorted(d for d in ds if d not in SKIP_DIRS and not d.startswith("."))
            for f in sorted(fs):
                if not f.startswith((".", "_main")):
                    yield os.path.join(r, f)

    # ---- công cụ ----
    def t_list_dir(self, path="."):
        p = self.path(path)
        if not p or not os.path.isdir(p):
            return f"Lỗi: không có thư mục {path!r}"
        out = [f"{n}/" if os.path.isdir(os.path.join(p, n)) else n for n in sorted(os.listdir(p))
               if n not in SKIP_DIRS and not n.startswith((".", "_main"))]
        return "\n".join(out) or "(trống)"

    def t_glob(self, pattern="**/*"):
        import fnmatch
        pat = str(pattern)
        hits = [self.rel(f) for f in self.files() if fnmatch.fnmatch(self.rel(f), pat)
                or fnmatch.fnmatch(os.path.basename(f), pat) or fnmatch.fnmatch(self.rel(f), pat.replace("**/", ""))]
        return "\n".join(hits[:200]) + (f"\n…(+{len(hits)-200} file)" if len(hits) > 200 else "") if hits else "(không có file khớp)"

    def t_grep(self, pattern, path=".", glob=None, context=0, ignore_case=False):
        import fnmatch
        try:
            rx = re.compile(str(pattern), re.I if ignore_case else 0)
        except re.error as e:
            return f"Lỗi regex: {e}"
        base = self.path(path)
        if not base:
            return "Lỗi: đường dẫn ngoài dự án"
        files = [base] if os.path.isfile(base) else list(self.files(base))
        out, n = [], 0
        ctx = max(0, min(int(context or 0), 5))
        for f in files:
            if glob and not fnmatch.fnmatch(os.path.basename(f), glob):
                continue
            try:
                lines = open(f, encoding="utf-8", errors="replace").read().split("\n")
            except OSError:
                continue
            for i, l in enumerate(lines):
                if rx.search(l):
                    n += 1
                    if n > 80:
                        continue
                    lo, hi = max(0, i - ctx), min(len(lines), i + ctx + 1)
                    out += [f"{self.rel(f)}:{k+1}{':' if k == i else '-'} {lines[k]}" for k in range(lo, hi)]
                    if ctx:
                        out.append("--")
        if not n:
            return "(không thấy)"
        return "\n".join(out)[:OUT_CAP] + (f"\n…(tổng {n} dòng khớp, mới hiện 80)" if n > 80 else "")

    def t_search_code(self, query):
        """tìm khối def/class theo từ khoá chung với câu hỏi (BM25 đơn giản, không cần embedding)."""
        import math
        words = lambda t: [w for w in re.findall(r"[a-zA-ZÀ-ỹ_]{2,}", t.lower().replace("_", " ")) if len(w) > 1]
        q = set(words(str(query)))
        chunks = []
        for f in self.files():
            if not f.endswith((".py", ".js", ".ts", ".md", ".txt", ".sql", ".sh")):
                continue
            lines = open(f, encoding="utf-8", errors="replace").read().split("\n")
            starts = [i for i, l in enumerate(lines) if re.match(r"\s*(def |class |function |async def )", l)] or [0]
            for a, b in zip(starts, starts[1:] + [len(lines)]):
                chunks.append((self.rel(f), a + 1, "\n".join(lines[a:min(b, a + 40)])))
        if not chunks or not q:
            return "(không thấy)"
        df = {w: sum(1 for c in chunks if w in set(words(c[2]))) for w in q}
        score = lambda c: sum(math.log(1 + len(chunks) / (1 + df[w])) for w in q if w in set(words(c[2])))
        best = sorted(chunks, key=score, reverse=True)[:5]
        return "\n\n".join(f"{p}:{ln}\n{body[:600]}" for p, ln, body in best if score((p, ln, body)) > 0) or "(không thấy)"

    def t_read_file(self, path, offset=1, limit=200):
        p = self.path(path)
        if not p or not os.path.isfile(p):
            return f"Lỗi: không có file {path!r}" + (f" — có thể ý bạn là: {self.t_glob('**/' + os.path.basename(str(path)))}" if path else "")
        lines = open(p, encoding="utf-8", errors="replace").read().split("\n")
        a = max(1, int(offset or 1)); b = min(len(lines), a - 1 + max(1, min(int(limit or 200), 400)))
        self.read.add(p)
        more = f"\n…(file có {len(lines)} dòng; đọc tiếp: offset={b+1})" if b < len(lines) else ""
        return (_num(lines[a - 1:b], a) + more)[:OUT_CAP * 2]

    def t_edit_file(self, path, old, new, all=False):
        import difflib
        p = self.path(path)
        if not p or not os.path.isfile(p):
            return f"Lỗi: không có file {path!r} (tạo file mới thì dùng write_file)"
        if p not in self.read:
            return f"Lỗi: phải read_file {path!r} trước khi sửa"
        txt = open(p, encoding="utf-8").read()
        old, new = str(old), str(new)
        if old == new:
            return "Lỗi: old và new giống nhau"
        k = txt.count(old) if old else 0
        if k == 0 and GUARD:
            fixed = _reindent(txt, old, new)               # khung v3: chỉ lệch thụt lề + khớp duy nhất → tự khớp rồi sửa
            if fixed:
                txt2, ln = fixed
                open(p, "w", encoding="utf-8").write(txt2)
                lines = txt2.split("\n")
                a, b = max(1, ln - 3), min(len(lines), ln + new.count("\n") + 3)
                return (f"Đã sửa {path} (1 chỗ; đoạn 'old' lệch thụt lề so với file — đã tự khớp theo thụt lề của file). "
                        "Đoạn sau khi sửa:\n" + _num(lines[a - 1:b], a))
            ws_hint = _ws_mismatch(txt, old)
            if ws_hint:
                return f"Lỗi: không tìm thấy đoạn 'old' NGUYÊN VĂN trong {path}. {ws_hint}"
        if k == 0:
            first = (old.strip().split("\n") or [""])[0].strip()
            cand = difflib.get_close_matches(first, [l.strip() for l in txt.split("\n")], n=1, cutoff=0.5)
            hint = f" Dòng gần giống nhất: {cand[0]!r} — copy y nguyên (đúng khoảng trắng thụt lề)." if cand else ""
            return f"Lỗi: không tìm thấy đoạn 'old' trong {path}.{hint}"
        if k > 1 and not all:
            return f"Lỗi: đoạn 'old' xuất hiện {k} lần — thêm vài dòng xung quanh cho duy nhất, hoặc đặt \"all\": true"
        txt2 = txt.replace(old, new) if all else txt.replace(old, new, 1)
        open(p, "w", encoding="utf-8").write(txt2)
        i = txt2.find(new) if new else txt.find(old)
        ln = txt2[:max(0, i)].count("\n") + 1
        lines = txt2.split("\n")
        a, b = max(1, ln - 3), min(len(lines), ln + new.count("\n") + 3)
        return f"Đã sửa {path} ({k if all else 1} chỗ). Đoạn sau khi sửa:\n" + _num(lines[a - 1:b], a)

    def t_write_file(self, path, content):
        p = self.path(path)
        if not p or p == self.wd:
            return "Lỗi: đường dẫn ngoài dự án"
        if os.path.exists(p) and p not in self.read:
            return f"Lỗi: {path!r} đã có — read_file trước (hoặc dùng edit_file để sửa một đoạn)"
        if GUARD and os.path.exists(p) and open(p, encoding="utf-8").read() == str(content):
            return (f"Lỗi: nội dung y hệt {path} hiện tại — ghi lại không thay đổi gì. Nếu test còn hỏng thì lời giải này "
                    "SAI: đọc lại thông báo lỗi của test và viết lời giải KHÁC.")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w", encoding="utf-8").write(str(content))
        self.read.add(p)
        return f"Đã ghi {path} ({str(content).count(chr(10)) + 1} dòng)"

    def t_delete_file(self, path):
        p = self.path(path)
        if not p or not os.path.isfile(p):
            return f"Lỗi: không có file {path!r}"
        os.remove(p); self.read.discard(p)
        return f"Đã xoá {path}"

    def t_bash(self, command, timeout=30):
        r = run(["/bin/bash", "-c", str(command)], self.wd, timeout=max(1, min(int(timeout or 30), 120)))
        return f"(mã thoát {r['rc']}{', HẾT THỜI GIAN' if r['timeout'] else ''})\n{r['out']}"

    def t_run_tests(self):
        if not self.test_cmd:
            return "Lỗi: dự án không khai báo lệnh test — dùng bash"
        r = run(self.test_cmd, self.wd, timeout=60)
        return f"(mã thoát {r['rc']}{', HẾT THỜI GIAN' if r['timeout'] else ''} — {'TẤT CẢ QUA' if r['rc'] == 0 else 'CÓ TEST HỎNG'})\n{r['out']}"

    def t_lint(self, path="."):
        targets = [self.path(path)] if path and os.path.isfile(self.path(path) or "") else \
                  [f for f in self.files(self.path(path) or self.wd) if f.endswith(".py")]
        code = ("import sys, ast\nok = True\nfor f in sys.argv[1:]:\n    src = open(f, encoding='utf-8').read()\n"
                "    try:\n        ast.parse(src, f)\n    except SyntaxError as e:\n        ok = False; print(f'{f}:{e.lineno}: lỗi cú pháp: {e.msg}')\n"
                "        continue\n    try:\n        from pyflakes.api import check\n        from pyflakes.reporter import Reporter\n"
                "        if check(src, f, Reporter(sys.stdout, sys.stdout)): ok = False\n    except ImportError:\n        pass\n"
                "print('Không thấy lỗi.' if ok else '')\n")
        open(os.path.join(self.wd, "_main_lint.py"), "w").write(code)
        r = run([PY, "-B", "-I", "_main_lint.py"] + [self.rel(t) for t in targets if t], self.wd, timeout=30)
        os.remove(os.path.join(self.wd, "_main_lint.py"))
        return r["out"].strip() or "Không thấy lỗi."

    # cũ (tương thích quỹ đạo trước): list_files / search / run_python
    def t_list_files(self):
        return "\n".join(self.rel(f) for f in self.files())[:OUT_CAP] or "(trống)"

    def t_search(self, pattern):
        return self.t_grep(re.escape(str(pattern)))

    def t_run_python(self, code):
        r = run_python(str(code), self.wd)
        return f"(mã thoát {r['rc']}{', HẾT THỜI GIAN' if r['timeout'] else ''})\n{r['out']}"

    def call(self, c):
        n, a = c["name"], c["args"] if isinstance(c["args"], dict) else {}
        if n == "_bad_json":
            return 'Lỗi: JSON trong <tool> không hợp lệ — nhớ thoát dấu " (\\") và xuống dòng (\\n) trong chuỗi.'
        f = getattr(self, "t_" + n, None)
        if not f:
            return f"Lỗi: không có công cụ {n!r}. Có: list_dir, glob, grep, search_code, read_file, edit_file, write_file, delete_file, bash, run_tests, lint"
        try:
            return f(**a)
        except TypeError as e:
            return f"Lỗi tham số cho {n}: {e}"
        except Exception as e:
            return f"Lỗi khi chạy {n}: {type(e).__name__}: {str(e)[:200]}"


def _ws_mismatch(txt, old):
    """'old' khớp file nếu bỏ qua khác biệt khoảng trắng đầu/cuối dòng (TAB ↔ dấu cách) → gợi ý đoạn ĐÚNG, TAB hiện là \\t."""
    norm = lambda l: l.expandtabs(4).rstrip()
    ol = [norm(l) for l in old.strip("\n").split("\n")]
    tl = txt.split("\n")
    tn = [norm(l) for l in tl]
    hits = [i for i in range(len(tn) - len(ol) + 1) if tn[i:i + len(ol)] == ol] or \
        [i for i in range(len(tn) - len(ol) + 1) if [x.strip() for x in tn[i:i + len(ol)]] == [x.strip() for x in ol]]
    if len(hits) != 1:
        return ""
    real = tl[hits[0]:hits[0] + len(ol)]
    kind = "TAB" if any("\t" in l[:len(l) - len(l.lstrip())] for l in real) else "số dấu cách khác"
    show = "\n".join(l.replace("\t", "\\t") for l in real)
    return (f"Có đoạn giống hệt nếu bỏ qua khoảng trắng (dòng {hits[0] + 1}–{hits[0] + len(ol)}), nhưng file thụt lề bằng {kind}. "
            f"Đoạn đúng trong file (\\t = một ký tự TAB, trong JSON viết \\t):\n{show}\nHãy copy y nguyên đoạn này làm 'old' "
            "và thụt lề 'new' cùng kiểu.")


def _indent_unit(lines):
    ks = sorted({len(l) - len(l.lstrip()) for l in lines if l.strip()})
    d = [b - a for a, b in zip(ks, ks[1:]) if b > a]
    return min(d) if d else (ks[0] if ks and ks[0] else 4)


def _reindent(txt, old, new):
    """'old' khớp DUY NHẤT một đoạn của file nếu bỏ qua thụt lề → thay đoạn đó bằng 'new' đã đổi thụt lề theo file
    (giữ thụt lề tương đối; đổi đơn vị 4↔2 dấu cách, dấu cách↔TAB). → (nội dung mới, dòng bắt đầu) hoặc None."""
    ol = [l for l in old.strip("\n").split("\n")]
    tl = txt.split("\n")
    key = lambda l: l.strip()
    hits = [i for i in range(len(tl) - len(ol) + 1) if [key(x) for x in tl[i:i + len(ol)]] == [key(x) for x in ol]]
    if len(hits) != 1 or not any(x.strip() for x in ol):
        return None
    i = hits[0]
    blk = tl[i:i + len(ol)]
    use_tab = any(l[:len(l) - len(l.lstrip())].count("\t") for l in blk)
    ex = lambda l: len(l[:len(l) - len(l.lstrip())].expandtabs(4))
    o_lines = [l.expandtabs(4) for l in ol if l.strip()]
    f_lines = [l.expandtabs(4) for l in blk if l.strip()]
    o_base, f_base = min(ex(l) for l in o_lines), min(ex(l) for l in f_lines)
    o_unit, f_unit = _indent_unit(o_lines + [l.expandtabs(4) for l in new.split("\n")]), _indent_unit(f_lines + [l.expandtabs(4) for l in tl])
    out = []
    for l in new.strip("\n").split("\n"):
        if not l.strip():
            out.append(""); continue
        rel = max(0, ex(l) - o_base)
        col = f_base + round(rel / max(o_unit, 1)) * f_unit
        out.append(("\t" * (col // 4) + " " * (col % 4) if use_tab else " " * col) + l.lstrip())
    txt2 = "\n".join(tl[:i] + out + tl[i + len(ol):])
    return txt2, i + 1


def _sig(call):
    return json.dumps([call.get("name"), call.get("args")], sort_keys=True, ensure_ascii=False) if call else None


def do_tool(call, wd, test_cmd):                     # tương thích: gọi lẻ không giữ trạng thái đọc-trước-khi-sửa
    ws = Workspace(wd, test_cmd); ws.read = {os.path.realpath(os.path.join(wd, f)) for f in os.listdir(wd)}
    return ws.call(call)


REPEAT = ("Lỗi: bạn vừa gọi Y HỆT lời gọi ngay trước — kết quả sẽ không đổi. Hãy ĐỔI CÁCH: đọc lại file hoặc thông báo lỗi "
          "của test, tìm nguyên nhân, rồi sửa khác đi.")


def solve_tool(model, task, files, test_cmd, max_steps=12, prefix=()):
    """files: {đường dẫn: nội dung}. → (thành công?, quỹ đạo, thư mục làm việc — người gọi xoá).
    prefix: các lượt trợ lý CHÈN SẴN (vd một bản sửa sai) — chạy thật để lấy kết quả, đánh dấu injected (C3-R dạy phục hồi)."""
    wd = tempfile.mkdtemp(prefix="aivie_proj_")
    for p, c in files.items():
        os.makedirs(os.path.dirname(os.path.join(wd, p)) or wd, exist_ok=True)
        open(os.path.join(wd, p), "w", encoding="utf-8").write(c)
    _own(wd)
    ws = Workspace(wd, test_cmd)
    msgs = [{"role": "user", "content": P_TOOL.format(task=task)}]
    traj, last = [], None
    for ans in prefix:
        call = parse_tool(ans)
        obs = ws.call(call) if call and call["name"] != "done" else "Lỗi: lượt chèn không hợp lệ"
        if GUARD and _sig(call) == last:
            obs = REPEAT
        last = _sig(call)
        traj.append({"step": len(traj), "assistant": ans, "call": call, "obs": obs, "injected": True})
        msgs += [{"role": "assistant", "content": ans}, {"role": "user", "content": f"<obs>\n{obs}\n</obs>"}]
    for s in range(len(traj), len(traj) + max_steps):
        ans = model.chat(msgs)
        call = parse_tool(ans)
        if call is None:
            obs = "Lỗi: không thấy lời gọi công cụ. Hãy gọi đúng một công cụ dạng <tool>{...}</tool> hoặc kết thúc bằng <done>…</done>."
        elif call["name"] == "done":
            traj.append({"step": s, "assistant": ans, "call": call, "obs": None})
            break
        elif GUARD and _sig(call) == last:          # gọi y hệt lượt ngay trước → không chạy lại, nhắc đổi cách
            obs = REPEAT
        else:
            obs = ws.call(call)
        last = _sig(call)
        traj.append({"step": s, "assistant": ans, "call": call, "obs": obs})
        msgs += [{"role": "assistant", "content": ans}, {"role": "user", "content": f"<obs>\n{obs}\n</obs>"}]
    final = run(test_cmd, wd, timeout=60)                  # chấm: bộ test của dự án chạy lại sau cùng
    return final["rc"] == 0 and not final["timeout"], traj, wd


if __name__ == "__main__":                                 # tự kiểm sandbox
    print(run_python("print(sum(range(10)))"))
    print(run_python("import urllib.request as u; u.urlopen('http://example.com', timeout=5)")["out"][-160:])
    print(run_python("open('/tmp/../Users/x_test_write','w')")["out"][-160:] if sys.platform == "darwin" else "")
    print(run_python("while True: pass", timeout=2))
