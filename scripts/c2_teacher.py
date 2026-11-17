"""TK_108 C2 — dữ liệu code CÓ THẦY, mọi mẫu đều được KIỂM BẰNG CHẠY THẬT. Thầy = vLLM (OpenAI API) trên box.

Nguồn: data/c2/oci_verified.jsonl (99 658 bài OpenCodeInstruct CC-BY-4.0, lời giải đã qua test — c2_build_verified.py).
Các kiểu bài (mỗi lệnh ghi data/c2/<kiểu>.jsonl, chạy lại được — bỏ qua id đã làm):
  vi       dịch đề + lời giải sang tiếng Việt, khối code GIỮ Y HỆT (kiểm: mọi khối ```python khớp bản gốc)
  debug    cài 1 lỗi vào lời giải đúng → chạy test lấy thông báo lỗi → người dùng hỏi "sửa giúp"; thầy giải thích + code sửa
           (kiểm: code sửa của thầy QUA toàn bộ test)
  predict  "hàm này trả về gì với đầu vào X?" — kết quả THẬT lấy bằng chạy code; thầy viết lập luận ngắn
           (kiểm: câu trả lời chứa đúng kết quả thật)
  sql      thầy dựng bảng SQLite + dữ liệu + câu hỏi tiếng Việt + truy vấn; một lượt thầy KHÁC giải độc lập
           (kiểm: cả hai truy vấn chạy được, kết quả khác rỗng và TRÙNG nhau)
Dùng: python scripts/c2_teacher.py vi --n 30000 [--workers 96] [--llm http://127.0.0.1:8000/v1] [--tag _v2]
Nhật ký thầy ĐẦY ĐỦ (mọi lượt gọi thầy + mọi lần chạy test, cả hỏng) → data/c2/teacher_log/<kiểu><tag>.jsonl
"""
from __future__ import annotations
import argparse, json, os, random, re, sqlite3, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
import code_agent as CA
from gen_screens import mutate

SRC = "data/c2/oci_verified.jsonl"
OUT = "data/c2"
A = None
_LOG, _LK, _CUR = None, threading.Lock(), threading.local()


def logw(**d):
    """NHẬT KÝ THẦY đầy đủ (cả lượt hỏng) → data/c2/teacher_log/<kiểu><tag>.jsonl — để làm DPO / dữ liệu sửa lỗi / RL."""
    if _LOG is None:
        return
    d = {"id": getattr(_CUR, "id", None), "kind": A.kind, "t": round(time.time(), 1), **d}
    with _LK:
        _LOG.write(json.dumps(d, ensure_ascii=False) + "\n")


def chat(prompt, max_tokens=1500, temp=0.3):
    body = {"model": A.model, "messages": [{"role": "user", "content": prompt}], "temperature": temp, "max_tokens": max_tokens}
    for k in range(3):
        try:
            req = urllib.request.Request(A.llm.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                out = json.loads(r.read())["choices"][0]["message"]["content"]
            logw(ev="teacher", prompt=prompt, raw=out, temp=temp)
            return out
        except Exception:
            time.sleep(5 * (k + 1))
    return ""


def pycode(t):
    return re.findall(r"```python\n(.*?)```", t, re.S)


def run_tests(code, tests, what="run"):
    r = CA.run_python(code + "\n\n" + "\n".join(t.strip() for t in tests) + "\n", timeout=10)
    ok = r["rc"] == 0 and not r["timeout"]
    logw(ev=what, code=code, rc=r["rc"], timeout=r["timeout"], out=r["out"][-3000:], ok=ok)
    return ok, r["out"]


def msgs(q, a):
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


# ── vi ──────────────────────────────────────────────────────────────────────────────────────
P_VI = """Dịch bài lập trình dưới đây sang TIẾNG VIỆT tự nhiên (như lập trình viên Việt Nam viết).
QUY TẮC: giữ NGUYÊN Y HỆT mọi khối code ``` (không dịch, không sửa một ký tự nào, kể cả chú thích trong code);
giữ nguyên tên hàm, tên biến, ví dụ đầu vào/đầu ra; chỉ dịch phần chữ bên ngoài khối code; không thêm bớt nội dung.

=== ĐỀ ===
{q}
=== LỜI GIẢI ===
{a}
=== HẾT ===
Trả lời ĐÚNG khuôn sau, không viết gì khác:
<<<DE>>>
(đề bằng tiếng Việt)
<<<LOIGIAI>>>
(lời giải bằng tiếng Việt, khối code giữ nguyên)"""

VI_CH = re.compile(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", re.I)


def do_vi(r):
    q, a = r["messages"][0]["content"], r["messages"][1]["content"]
    out = chat(P_VI.format(q=q, a=a), max_tokens=3000, temp=0.2)
    m = re.search(r"<<<DE>>>\s*(.*?)\s*<<<LOIGIAI>>>\s*(.*)", out, re.S)
    if not m:
        return None
    q2, a2 = m.group(1).strip(), m.group(2).strip()
    if pycode(a2) != pycode(a) or len(VI_CH.findall(q2)) < 5:
        return None
    return {"id": r["id"] + "_vi", "kind": "vi", "messages": msgs(q2, a2), "tests": r["tests"]}


# ── debug ───────────────────────────────────────────────────────────────────────────────────
ASK_DEBUG = ["Code của mình chạy test bị lỗi, bạn xem giúp lỗi ở đâu và sửa giúp nhé:",
             "Hàm này mình viết mà chạy sai, đây là code và thông báo lỗi. Sửa giúp mình với:",
             "Mình đang làm bài này: {task}\n\nCode của mình bị lỗi khi chạy test:",
             "Debug giúp mình đoạn code Python này:", "Why does this fail? Please fix it:"]
P_DEBUG = """Bạn là lập trình viên giỏi, trả lời người dùng Việt Nam bằng TIẾNG VIỆT.
Người dùng gửi code lỗi và thông báo lỗi khi chạy test. Thông tin cho riêng bạn (KHÔNG nhắc tới trong câu trả lời):
lỗi nằm ở dòng {line}: `{bad}` — đúng ra phải là `{good}`.

=== NGƯỜI DÙNG ===
{user}
=== HẾT ===
Viết câu trả lời: 1–3 câu chỉ ra nguyên nhân (dòng nào, vì sao sai), rồi đưa TOÀN BỘ hàm đã sửa trong MỘT khối ```python,
cuối cùng 1 câu ngắn giải thích bản sửa. Không dùng LaTeX."""


def do_debug(r):
    rng = random.Random(r["id"])
    good_code = pycode(r["messages"][1]["content"])[0]
    mu = mutate(good_code, rng)
    if not mu:
        return None
    bad_code, line, good_l, bad_l, _ = mu
    ok, err = run_tests(bad_code, r["tests"], "run_bad")
    if ok:                                     # lỗi cài vào không làm test hỏng → bỏ
        return None
    err = "\n".join(err.strip().splitlines()[-12:])
    ask = rng.choice(ASK_DEBUG).format(task=r["messages"][0]["content"][:600])
    test_show = "\n".join(t.strip() for t in r["tests"][:3])
    user = f"{ask}\n\n```python\n{bad_code.strip()}\n```\n\nTest:\n```python\n{test_show}\n```\n\nLỗi:\n```\n{err}\n```"
    ans = chat(P_DEBUG.format(line=line, bad=bad_l, good=good_l, user=user), max_tokens=1800)
    cs = pycode(ans)
    ok, out = (run_tests(cs[0], r["tests"], "run_fix") if len(cs) == 1 else (False, "không có đúng 1 khối ```python"))
    rnd = 0
    if not ok and A.retry:                     # lượt sửa thêm: đưa lỗi thật của bản sửa cho thầy
        rnd = 1
        ans = chat(P_DEBUG.format(line=line, bad=bad_l, good=good_l, user=user) +
                   f"\n\nBản trả lời trước của bạn:\n{ans}\n\nChạy test vẫn lỗi:\n```\n{out.strip()[-1500:]}\n```\n"
                   "Viết lại TOÀN BỘ câu trả lời theo đúng yêu cầu ở trên (không nhắc tới lần trả lời trước).", max_tokens=1800)
        cs = pycode(ans)
        ok = len(cs) == 1 and run_tests(cs[0], r["tests"], "run_fix2")[0]
    logw(ev="final", ok=ok, rounds=rnd + 1)
    if not ok:
        return None
    return {"id": r["id"] + "_dbg", "kind": "debug", "messages": msgs(user, ans), "tests": r["tests"], "rounds": rnd + 1}


# ── predict ─────────────────────────────────────────────────────────────────────────────────
P_PRED = """Bạn là lập trình viên giỏi, trả lời bằng TIẾNG VIỆT. Người dùng hỏi:
=== ===
{user}
=== ===
Kết quả THẬT khi chạy (cho riêng bạn): {out}
Viết câu trả lời: lần theo code từng bước ngắn gọn (3–8 dòng) cho đúng đầu vào này, rồi kết luận bằng dòng cuối dạng
"Kết quả: `{out}`". Không dùng LaTeX."""


def do_predict(r):
    code = pycode(r["messages"][1]["content"])[0]
    calls = [m.group(1) for t in r["tests"] for m in [re.match(r"\s*assert\s+(.+?)\s*==\s*.+", t.strip())] if m]
    calls = [c for c in calls if len(c) < 160 and "(" in c]
    if not calls:
        return None
    call = random.Random(r["id"]).choice(calls)
    res = CA.run_python(code + f"\n\nprint(repr({call}))\n", timeout=10)
    out = res["out"].strip().splitlines()[-1] if res["rc"] == 0 and res["out"].strip() else None
    if not out or len(out) > 200:
        return None
    user = f"Đoạn code sau trả về gì khi gọi `{call}`? Giải thích ngắn.\n\n```python\n{code.strip()}\n```"
    ans = chat(P_PRED.format(user=user, out=out), max_tokens=900)
    if f"`{out}`" not in ans:
        return None
    return {"id": r["id"] + "_pred", "kind": "predict", "messages": msgs(user, ans), "expected": out}


# ── sql ─────────────────────────────────────────────────────────────────────────────────────
DOMAINS = ["cửa hàng bán lẻ", "trường học", "bệnh viện", "nhân sự công ty", "thư viện", "khách sạn", "ngân hàng (dữ liệu giả)",
           "giao hàng", "quán cà phê", "phòng khám nha khoa", "nhà hàng", "kho hàng", "rạp chiếu phim", "trung tâm tiếng Anh",
           "cửa hàng điện thoại", "shop thời trang online", "công ty xây dựng", "phòng gym", "tiệm sửa xe", "nông trại"]
LEVELS = ["một bảng, lọc WHERE đơn giản", "GROUP BY + hàm tổng hợp", "JOIN hai bảng", "JOIN + GROUP BY + HAVING",
          "truy vấn con (subquery)", "ORDER BY + LIMIT để tìm top N", "xử lý ngày tháng dạng 'YYYY-MM-DD'", "CASE WHEN"]
P_SQL_GEN = """Tạo MỘT bài tập SQL (SQLite) bối cảnh "{dom}", độ khó: {lvl}.
Trả về DUY NHẤT JSON:
{{"schema": "các lệnh CREATE TABLE + INSERT (ít nhất 6 dòng dữ liệu mỗi bảng, tên bảng/cột tiếng Anh không dấu)",
  "question": "câu hỏi tiếng Việt tự nhiên như người dùng hỏi, rõ ràng, chỉ có MỘT đáp án",
  "query": "truy vấn SQLite trả lời câu hỏi"}}"""
P_SQL_SOLVE = """Cho cơ sở dữ liệu SQLite:
```sql
{schema}
```
Câu hỏi: {question}
Trả lời bằng TIẾNG VIỆT: viết truy vấn SQLite trong MỘT khối ```sql, rồi 1–3 câu giải thích truy vấn làm gì."""


def sql_run(schema, q):
    con = sqlite3.connect(":memory:")
    try:
        con.executescript(schema)
        return sorted(map(repr, con.execute(q).fetchall()))
    except Exception:
        return None
    finally:
        con.close()


def do_sql(item):
    k, dom, lvl = item
    from gen_office import loads_loose
    try:
        g = loads_loose(chat(P_SQL_GEN.format(dom=dom, lvl=lvl), max_tokens=1800, temp=0.9))
        schema, question, q1 = g["schema"], g["question"], g["query"]
    except Exception:
        return None
    r1 = sql_run(schema, q1)
    if not r1:
        return None
    ans = chat(P_SQL_SOLVE.format(schema=schema, question=question), max_tokens=900, temp=0.2)
    m = re.findall(r"```sql\n(.*?)```", ans, re.S)
    if len(m) != 1 or sql_run(schema, m[0]) != r1:
        return None
    user = f"Mình có cơ sở dữ liệu SQLite như sau:\n```sql\n{schema.strip()}\n```\n{question}"
    return {"id": f"sql_{k}", "kind": "sql", "messages": msgs(user, ans), "schema": schema, "expected_rows": r1}


def safe(fn, x):
    _CUR.id = x["id"] if isinstance(x, dict) else f"sql_{x[0]}"
    try:
        return fn(x)
    except Exception as e:                         # một mẫu hỏng không được làm sập cả lượt
        print(f"[c2t] lỗi mẫu: {type(e).__name__}: {str(e)[:120]}", flush=True)
        return None


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["vi", "debug", "predict", "sql"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--llm", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma4-31b")
    ap.add_argument("--seed", type=int, default=108)
    ap.add_argument("--tag", default="", help="hậu tố file ra, vd _v2 → debug_v2.jsonl (chạy lại không đè bản cũ)")
    ap.add_argument("--retry", type=int, default=1, help="debug: cho thầy sửa thêm 1 lượt khi bản sửa còn hỏng")
    ap.add_argument("--no-log", action="store_true")
    A = ap.parse_args()
    global _LOG
    if not A.no_log:
        os.makedirs(f"{OUT}/teacher_log", exist_ok=True)
        _LOG = open(f"{OUT}/teacher_log/{A.kind}{A.tag}.jsonl", "a", encoding="utf-8")
    path = f"{OUT}/{A.kind}{A.tag}.jsonl"
    seen_path = f"{OUT}/{A.kind}{A.tag}.seen"
    seen = set(open(seen_path).read().split()) if os.path.exists(seen_path) else set()
    if A.kind == "sql":
        items = [(k, DOMAINS[k % len(DOMAINS)], LEVELS[(k // len(DOMAINS)) % len(LEVELS)]) for k in range(A.n)]
        items = [x for x in items if str(x[0]) not in seen]
        fn, key = do_sql, (lambda x: str(x[0]))
    else:
        rows = [json.loads(l) for l in open(SRC)]
        random.Random(A.seed + ["vi", "debug", "predict"].index(A.kind)).shuffle(rows)   # mỗi kiểu một tập con khác
        items = [r for r in rows[:A.n] if r["id"] not in seen]
        fn, key = {"vi": do_vi, "debug": do_debug, "predict": do_predict}[A.kind], (lambda r: r["id"])
    print(f"[c2t] {A.kind}: cần làm {len(items)} (đã làm {len(seen)})", flush=True)
    ok = n = 0; T0 = time.time()
    with open(path, "a") as fo, open(seen_path, "a") as fs, ThreadPoolExecutor(A.workers) as ex:
        for it, res in zip(items, ex.map(lambda x: safe(fn, x), items)):
            n += 1
            fs.write(key(it) + "\n")
            if res:
                ok += 1; fo.write(json.dumps(res, ensure_ascii=False) + "\n")
            if n % 500 == 0:
                fo.flush(); fs.flush(); _LOG and _LOG.flush()
                print(f"[c2t] {A.kind} {n}/{len(items)} · giữ {ok} ({ok/n:.0%}) · {time.time()-T0:.0f}s", flush=True)
    print(f"### c2 {A.kind}: làm {n} · giữ {ok} ({ok/max(n,1):.0%}) · {time.time()-T0:.0f}s → {path}", flush=True)


if __name__ == "__main__":
    main()
