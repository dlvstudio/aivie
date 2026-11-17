"""TK_108 C2b — dữ liệu code ĐA NGÔN NGỮ (top 10: JS, TS, HTML/CSS, Python, SQL, Java, C#, PHP, Go, Bash).

Phần A (CPU, chạy được khi GPU bận): lấy mẫu theo ngôn ngữ từ các nguồn cho dùng thương mại → KIỂM CÚ PHÁP mọi khối code
bằng tree-sitter (không còn nút ERROR/MISSING; đoạn trích thiếu khung thì thử bọc class/hàm) → SQL của gretel CHẠY THẬT
trên SQLite (lược đồ + dữ liệu mẫu + câu truy vấn) → khử trùng + bỏ tên hàm HumanEval → data/c2b/<lang>.jsonl.
Phần B (GPU, sau): thầy viết test + chạy thật từng ngôn ngữ, viết lại sang T-SQL chạy trên SQL Server Express, dịch tiếng Việt.

Nguồn (giấy phép đã kiểm trên HF):
  ling      inclusionAI/Ling-Coder-SFT          Apache-2.0  hỏi–đáp code tổng hợp, >20 ngôn ngữ
  opc       OpenCoder-LLM/opc-sft-stage1        MIT         largescale_diverse_instruct + realuser_instruct
  magic     ise-uiuc/Magicoder-OSS-Instruct-75K MIT         (sinh bằng GPT-3.5 — user đã xin phép dùng)
  commit    bigcode/commitpackft                MIT         commit thật: file cũ + lời nhắn → file mới
  gretel    gretelai/synthetic_text_to_sql      Apache-2.0  text-to-SQL có lược đồ + dữ liệu

  python scripts/c2b_multilang.py extract --raw /workspace/c2b_raw --per-lang 40000 --workers 32
"""
from __future__ import annotations
import argparse, collections, glob, hashlib, json, logging, os, re, sqlite3
from concurrent.futures import ProcessPoolExecutor

OUT = "data/c2b"
LANGS = ["javascript", "typescript", "html_css", "python", "sql", "java", "csharp", "php", "go", "bash"]
ALIAS = {"javascript": "javascript", "js": "javascript", "jsx": "javascript", "node": "javascript", "mjs": "javascript",
         "typescript": "typescript", "ts": "typescript", "tsx": "typescript",
         "html": "html_css", "htm": "html_css", "css": "html_css",
         "python": "python", "py": "python", "python3": "python",
         "sql": "sql", "tsql": "sql", "mysql": "sql", "postgresql": "sql", "postgres": "sql", "sqlite": "sql", "plsql": "sql",
         "java": "java", "csharp": "csharp", "cs": "csharp", "c#": "csharp", "php": "php", "go": "go", "golang": "go",
         "bash": "bash", "sh": "bash", "shell": "bash", "zsh": "bash",
         # C2b+ (ngôn ngữ hiện đại)
         "kotlin": "kotlin", "kt": "kotlin", "dart": "dart", "svelte": "svelte", "vue": "vue", "rust": "rust", "rs": "rust",
         "swift": "swift", "cpp": "cpp", "c++": "cpp", "ruby": "ruby", "rb": "ruby"}
# tên grammar tree-sitter theo nhãn khối code
TS_GRAMMAR = {"javascript": "javascript", "js": "javascript", "node": "javascript", "mjs": "javascript", "jsx": "javascript",
              "typescript": "typescript", "ts": "typescript", "tsx": "tsx", "html": "html", "htm": "html", "css": "css",
              "python": "python", "py": "python", "python3": "python", "java": "java", "csharp": "csharp", "cs": "csharp",
              "c#": "csharp", "php": "php", "go": "go", "golang": "go", "bash": "bash", "sh": "bash", "shell": "bash",
              "zsh": "bash", "sql": "sql", "tsql": "sql", "mysql": "sql", "postgresql": "sql", "postgres": "sql",
              "sqlite": "sql", "plsql": "sql", "kotlin": "kotlin", "kt": "kotlin", "dart": "dart", "svelte": "svelte",
              "vue": "vue", "rust": "rust", "rs": "rust", "swift": "swift", "cpp": "cpp", "c++": "cpp", "ruby": "ruby", "rb": "ruby"}
PY_HEAVY = 0.25                 # python đã có ~160K mẫu kiểm bằng chạy thật (C2) → chỉ lấy 1/4 hạn mức
FENCE = re.compile(r"```([\w#+.-]*)[^\n]*\n(.*?)```", re.S)
_P = {}
logging.getLogger("sqlglot").setLevel(logging.CRITICAL)


def parser(g):
    if g not in _P:
        from tree_sitter_language_pack import get_parser
        _P[g] = get_parser(g)
    return _P[g]


def ok_tree(g, code):
    return not parser(g).parse(code.encode("utf-8", "ignore")).root_node.has_error


def syntax_ok(tag, code):
    """khối code đúng cú pháp? đoạn trích hay thiếu khung → thử bọc."""
    g = TS_GRAMMAR[tag]
    if g == "sql":
        from sqlglot import exp
        STMT = tuple(getattr(exp, n) for n in ("Query", "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter",
                                               "Set", "Use", "Transaction", "Commit", "Rollback", "Declare", "TruncateTable")
                     if hasattr(exp, n))
    if g == "sql":              # grammar SQL của tree-sitter yếu với phương ngữ → dùng sqlglot (đọc được mọi phương ngữ)
        import sqlglot
        for d in ("tsql", "postgres", "mysql", "sqlite", None):
            try:
                es = [e for e in sqlglot.parse(code, read=d) if e is not None]
                if es and all(isinstance(e, STMT) for e in es):        # phải là câu lệnh thật, không phải biểu thức lẻ
                    return True
            except Exception:
                pass
        return False
    if g == "php" and "<?php" not in code:
        code = "<?php\n" + code
    tries = [code]
    if g == "java":
        tries += [f"class _W {{\n{code}\n}}", f"class _W {{ void _m() {{\n{code}\n}} }}"]
    elif g == "go":
        tries += [f"package main\n{code}", f"package main\nfunc _() {{\n{code}\n}}"]
    elif g == "csharp":
        tries += [f"class _W {{\n{code}\n}}"]
    elif g in ("kotlin", "dart", "swift"):
        tries += [f"class _W {{\n{code}\n}}"]
    elif g == "rust":
        tries += [f"fn _w() {{\n{code}\n}}"]
    elif g in ("javascript", "typescript", "tsx"):
        tries += [f"class _W {{\n{code}\n}}"]
    return any(ok_tree(g, c) for c in tries)


def judge(ans):
    """→ (ngôn ngữ chính, mọi khối có nhãn đều đúng cú pháp?) ; None nếu không có khối code nào nhận ra."""
    blocks = [(t.lower(), c) for t, c in FENCE.findall(ans)]
    known = [(t, c) for t, c in blocks if t in ALIAS]
    if not known or len(blocks) > 8:
        return None
    w = collections.Counter()
    for t, c in known:
        w[ALIAS[t]] += len(c)
    lang = w.most_common(1)[0][0]
    if sum(len(c) for _, c in known) < 60:
        return None
    try:
        return lang, all(syntax_ok(t, c) for t, c in known)
    except Exception:
        return lang, False


def sql_runs(ctx, q):
    db = sqlite3.connect(":memory:")
    try:
        db.executescript(ctx)
        db.execute("PRAGMA query_only = 1") if q.lstrip().lower().startswith(("select", "with")) else None
        db.executescript(q) if ";" in q.strip().rstrip(";") else db.execute(q).fetchall()
        return True
    except Exception:
        return False
    finally:
        db.close()


# ── đọc từng nguồn → (id, src, messages, lang gợi ý) ──
def rows_ling(raw):
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(f"{raw}/ling/**/*.parquet", recursive=True)):
        for b in pq.ParquetFile(f).iter_batches(batch_size=4096, columns=["mid", "messages"]):
            for r in b.to_pylist():
                ms = [{"role": "user" if m["role"].upper() in ("HUMAN", "USER") else "assistant", "content": m["content"]}
                      for m in r["messages"] if m["role"].upper() != "SYSTEM"]
                if len(ms) >= 2 and ms[0]["role"] == "user":
                    yield f"ling_{r['mid']}", "ling", ms


def rows_opc(raw):
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(f"{raw}/opc/**/*.parquet", recursive=True)):
        if "filtered_infinity" in f:
            continue
        sub = "real" if "realuser" in f else "div"
        for b in pq.ParquetFile(f).iter_batches(batch_size=4096, columns=["instruction", "output"]):
            for i, r in enumerate(b.to_pylist()):
                yield (f"opc{sub}_{hashlib.md5(r['instruction'].encode()).hexdigest()[:16]}", f"opc_{sub}",
                       [{"role": "user", "content": r["instruction"].strip()}, {"role": "assistant", "content": r["output"].strip()}])


def rows_magic(raw):
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(f"{raw}/magic/**/*.parquet", recursive=True)) + sorted(glob.glob(f"{raw}/magic/**/*.jsonl", recursive=True)):
        it = (r for b in pq.ParquetFile(f).iter_batches(batch_size=4096) for r in b.to_pylist()) if f.endswith(".parquet") \
            else (json.loads(l) for l in open(f))
        for r in it:
            yield f"magic_{r['index']}", "magic", [{"role": "user", "content": r["problem"].strip()},
                                                     {"role": "assistant", "content": r["solution"].strip()}]


CP_LANG = {"javascript": "javascript", "typescript": "typescript", "html": "html", "css": "css", "python": "python",
           "sql": "sql", "java": "java", "c#": "csharp", "php": "php", "go": "go", "shell": "bash"}


def rows_commit(raw):
    for lg, tag in CP_LANG.items():
        for f in sorted(glob.glob(f"{raw}/commit/data/{lg}/*.jsonl")):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                old, new = r.get("old_contents", ""), r.get("new_contents", "")
                if not old.strip() or not new.strip() or old == new or len(old) + len(new) > 7000:
                    continue
                q = (f"File `{r['new_file']}`:\n```{tag}\n{old.rstrip()}\n```\n\nHãy sửa file theo yêu cầu: "
                     f"{r.get('message') or r.get('subject')}\nTrả về toàn bộ file sau khi sửa.")
                yield (f"commit_{r['commit'][:12]}_{hashlib.md5(r['new_file'].encode()).hexdigest()[:6]}", "commit",
                       [{"role": "user", "content": q}, {"role": "assistant", "content": f"```{tag}\n{new.rstrip()}\n```"}])


def rows_gretel(raw):
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(f"{raw}/gretel/**/*.parquet", recursive=True)):
        if "test" in os.path.basename(f):
            continue
        for r in pq.read_table(f).to_pylist():
            q = (f"Lược đồ và dữ liệu:\n```sql\n{r['sql_context'].strip()}\n```\n\n{r['sql_prompt'].strip()}")
            a = f"```sql\n{r['sql'].strip()}\n```\n\n{r['sql_explanation'].strip()}"
            yield f"gretel_{r['id']}", "gretel", [{"role": "user", "content": q}, {"role": "assistant", "content": a}], \
                (r["sql_context"], r["sql"])


def check(item):
    rid, src, ms, extra = item
    ans = "\n".join(m["content"] for m in ms if m["role"] == "assistant")
    j = judge(ans)
    if j is None:
        return None
    lang, ok = j
    if ok and extra is not None:                    # gretel: phải CHẠY được thật trên SQLite
        ok = sql_runs(*extra)
    return rid, src, ms, lang, ok


def extract(a):
    os.makedirs(OUT, exist_ok=True)
    he = set()
    if os.path.exists("data/c2b/humaneval_names.txt"):
        he = {l.strip() for l in open("data/c2b/humaneval_names.txt") if len(l.strip()) >= 10}
    cap = {l: a.per_lang if l != "python" else int(a.per_lang * PY_HEAVY) for l in LANGS}
    cap_src = {}                                    # một nguồn không chiếm quá 60% một ngôn ngữ
    n = collections.Counter(); seen, bad, stat = set(), collections.Counter(), collections.Counter()
    fout = {l: open(f"{OUT}/{l}.jsonl", "w", encoding="utf-8") for l in LANGS}

    def gen():
        for g in (rows_gretel, rows_commit, rows_magic, rows_opc, rows_ling):
            for t in g(a.raw):
                yield (t[0], t[1], t[2], t[3] if len(t) > 3 else None)

    def batches(it, k=4096):
        b = []
        for x in it:
            q = x[2][0]["content"]
            h = hashlib.md5(re.sub(r"\s+", " ", q.lower()).encode()).hexdigest()
            if h in seen or len(q) > 6000 or sum(len(m["content"]) for m in x[2]) > 12000:
                continue
            seen.add(h); b.append(x)
            if len(b) >= k:
                yield b; b = []
        if b:
            yield b

    with ProcessPoolExecutor(a.workers) as ex:
        for b in batches(gen()):
            for r in ex.map(check, b, chunksize=64):
                if r is None:
                    continue
                rid, src, ms, lang, ok = r
                stat[(lang, src, ok)] += 1
                if not ok:
                    bad[lang] += 1; continue
                if n[lang] >= cap[lang] or cap_src.get((lang, src), 0) >= 0.6 * cap[lang]:
                    continue
                if he and any(x in m["content"] for m in ms for x in he):
                    bad["_humaneval"] += 1; continue
                fout[lang].write(json.dumps({"id": rid, "lang": lang, "src": src, "messages": ms}, ensure_ascii=False) + "\n")
                n[lang] += 1; cap_src[(lang, src)] = cap_src.get((lang, src), 0) + 1
            print(f"[c2b] {dict(n)}", flush=True)
            if all(n[l] >= cap[l] for l in LANGS):
                break
    for f in fout.values():
        f.close()
    json.dump({f"{k[0]}|{k[1]}|{'ok' if k[2] else 'loi'}": v for k, v in sorted(stat.items())},
              open(f"{OUT}/stats.json", "w"), indent=1)
    tot = sum(n.values())
    print(f"### c2b A: {tot} mẫu · " + " · ".join(f"{l} {n[l]}" for l in LANGS) +
          f" · loại cú pháp/chạy hỏng {sum(v for k, v in bad.items() if k != '_humaneval')} · trùng HumanEval {bad['_humaneval']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract"])
    ap.add_argument("--raw", default="/workspace/c2b_raw")
    ap.add_argument("--per-lang", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    extract(a)


if __name__ == "__main__":
    main()
