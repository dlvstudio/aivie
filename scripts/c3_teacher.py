"""TK_108 C3 — quỹ đạo DÙNG CÔNG CỤ: thầy (vLLM) giải bài dạng dự án nhỏ bằng khung code_agent (đúng giao thức
<tool>{json}</tool> mà aivie dùng), CHỈ giữ quỹ đạo thành công (bộ test chạy lại sau cùng QUA + mọi lượt gọi công cụ hợp lệ).

Bài dựng từ data/c2/oci_verified.jsonl (đã bỏ bài giống HumanEval/MBPP; KHÔNG dùng 40 bài đo tool-use của C1):
  fix        src.py có 1 lỗi cài sẵn (test đang hỏng) → tìm và sửa
  implement  src.py chỉ có khung hàm + docstring → viết thân hàm cho test qua
  (+ đôi khi có file phụ utils.py / README.md để phải tìm đúng file)
Đề tiếng Việt nếu bài đó có bản dịch trong data/c2/vi.jsonl.

  python scripts/c3_teacher.py build --n 40000                        → data/c3/tasks.jsonl (đã kiểm: gốc qua, bài giao hỏng)
  python scripts/c3_teacher.py run --model gemma4-31b --tag pilot_gemma --n 50 --offset 0
  python scripts/c3_teacher.py run --model gemma4-31b --tag main --offset 100 --hours 5
Ra data/c3/traj_<tag>.jsonl: {"id", "kind", "ok", "steps", "messages" (đúng khuôn lúc dùng thật)}; chạy lại được.
"""
from __future__ import annotations
import argparse, collections, json, os, random, re, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
import code_agent as A
from gen_screens import mutate

SRC, VI, OUT = "data/c2/oci_verified.jsonl", "data/c2/vi.jsonl", "data/c3"
TEST_CMD = [A.PY, "-B", "-E", "-s", "test_src.py"]
UTILS = ['"""Tiện ích dùng chung (không liên quan bài này)."""\n\n\ndef clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n',
         '"""Đọc cấu hình."""\nimport json\n\n\ndef load_config(path):\n    with open(path, encoding="utf-8") as f:\n        return json.load(f)\n']


def run_tests(files):
    import tempfile
    wd = tempfile.mkdtemp(prefix="aivie_chk_")
    try:
        for p, c in files.items():
            open(os.path.join(wd, p), "w", encoding="utf-8").write(c)
        A._own(wd)
        return A.run(TEST_CMD, wd, timeout=20)["rc"]
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def build(a):
    vi = {}
    if os.path.exists(VI):
        for l in open(VI):
            r = json.loads(l); vi[r["id"].removesuffix("_vi")] = r["messages"][0]["content"]
    rows = [json.loads(l) for l in open(SRC)]
    rng = random.Random(308); rng.shuffle(rows)
    os.makedirs(OUT, exist_ok=True)
    cands, k = [], 0
    for r in rows:
        if len(cands) >= int(a.n * 1.4):
            break
        code = re.search(r"```python\n(.*?)```", r["messages"][1]["content"], re.S).group(1)
        code = re.split(r"\n# (?:Example|Test|Sample)", code)[0].rstrip() + "\n"      # bỏ phần ví dụ chạy ở cuối
        fn = re.search(r"^def (\w+)\(", code, re.M)
        if not fn:
            continue
        spec = vi.get(r["id"]) or r["messages"][0]["content"]
        if len(spec) > 2500:
            continue
        test_py = "from src import *\n\n" + "\n".join(t.strip() for t in r["tests"]) + "\nprint('TẤT CẢ TEST QUA')\n"
        kind = "fix" if k % 2 == 0 else "implement"
        if kind == "fix":
            m = mutate(code, rng)
            if not m:
                continue
            src = m[0]
            task = ("Trong dự án có `src.py` và bộ test `test_src.py`. Chạy test đang HỎNG vì `src.py` có lỗi. "
                    f"Hãy tìm và sửa lỗi trong `src.py` (không sửa test). Yêu cầu gốc:\n{spec}")
        else:
            sm = re.search(r"^(def " + fn.group(1) + r"\(.*?\)[^:\n]*:)", code, re.M | re.S)
            if not sm:
                continue
            head = "\n".join(l for l in code.split("\n") if l.startswith(("import ", "from ")))
            others = code[:sm.start()].strip()                                           # hàm phụ / lớp đứng trước
            others = "\n".join(l for l in others.split("\n") if not l.startswith(("import ", "from ")))
            src = ((head + "\n\n") if head else "") + ((others + "\n\n\n") if others else "") + \
                  f'{sm.group(1)}\n    """Xem yêu cầu trong nhiệm vụ."""\n    raise NotImplementedError\n'
            task = (f"Trong dự án có `src.py` (hàm `{fn.group(1)}` chưa viết) và bộ test `test_src.py`. "
                    f"Hãy viết thân hàm trong `src.py` cho tất cả test qua. Yêu cầu:\n{spec}")
        files_ok = {"src.py": code, "test_src.py": test_py}
        files_bad = {"src.py": src, "test_src.py": test_py}
        if rng.random() < 0.4:                                                          # file phụ → phải tìm đúng file
            extra = {"utils.py": rng.choice(UTILS), "README.md": "# Dự án nhỏ\n\nChạy test: `python test_src.py`\n"}
            files_ok.update(extra); files_bad.update(extra)
        cands.append((files_ok, {"id": f"c3_{kind}_{r['id']}", "kind": kind, "task": task, "files": files_bad,
                                 "vi": r["id"] in vi}))
        k += 1

    def check(c):                                       # gốc phải QUA · bài giao phải đang HỎNG
        return run_tests(c[0]) == 0 and run_tests(c[1]["files"]) != 0
    n = 0
    with open(f"{OUT}/tasks.jsonl", "w", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        for c, good in zip(cands, ex.map(check, cands)):
            if good and n < a.n:
                f.write(json.dumps(c[1], ensure_ascii=False) + "\n"); n += 1
    print(f"### c3 build: {n} bài (từ {len(cands)} ứng viên) → {OUT}/tasks.jsonl", flush=True)


# ── C3-R: quỹ đạo PHỤC HỒI (dạy: làm sai → nhận lỗi → đổi cách) ──────────────────────────────
def tabify(code):
    """thụt lề bằng TAB (mỗi 4 dấu cách đầu dòng → 1 TAB) — Python vẫn chạy như cũ."""
    out = []
    for l in code.split("\n"):
        k = len(l) - len(l.lstrip(" "))
        out.append("\t" * (k // 4) + " " * (k % 4) + l[k:])
    return "\n".join(out)


def T(name, **a):
    return "<tool>" + json.dumps({"name": name, "args": a}, ensure_ascii=False) + "</tool>"


def build_r(a):
    """từ tasks.jsonl (bỏ phần C3 chính đã dùng) → data/c3/tasks_r.jsonl, mỗi bài có 'inject' = các lượt SAI chèn sẵn:
       tab        file thụt lề TAB + một edit_file viết bằng dấu cách (công cụ báo lệch khoảng trắng)
       bad        implement: viết 'return None' · fix: cài THÊM một lỗi (mutate) → test vẫn hỏng
       repeat     như bad nhưng gửi lại y hệt lần 2 (khung chống lặp nhắc đổi cách)
       none       không chèn, cho 15 bước (bài khó tự nhiên)"""
    rng = random.Random(408)
    rows = [json.loads(l) for l in open(f"{OUT}/tasks.jsonl")][a.offset:]
    rng.shuffle(rows)
    if a.out != "tasks_r.jsonl" and os.path.exists(f"{OUT}/tasks_r.jsonl"):   # đợt sau: bỏ bài đã dùng ở C3-R đầu
        used = {json.loads(l)["id"][:-2] for l in open(f"{OUT}/tasks_r.jsonl")}
        rows = [t for t in rows if t["id"] not in used]
    suf = "_" + a.out.split("_")[-1].split(".")[0] if a.out != "tasks_r.jsonl" else "_r"
    out = []
    for t in rows[: int(a.n * 1.3)]:
        src = t["files"]["src.py"]
        kind = rng.choices(["tab", "bad", "repeat", "none"], [float(x) for x in a.weights.split(",")])[0]
        inj = [T("run_tests")] if t["kind"] == "fix" else []
        inj.append(T("read_file", path="src.py"))
        if kind == "tab":
            src = tabify(src)
            lines = [l for l in src.split("\n") if l.startswith("\t") and l.strip() and "NotImplementedError" not in l]
            if not lines:
                continue
            l = rng.choice(lines)
            spaced = l.replace("\t", "    ")
            inj.append(T("edit_file", path="src.py", old=spaced, new=spaced.rstrip() + "  "))
        elif kind in ("bad", "repeat"):
            if t["kind"] == "implement":
                ed = T("edit_file", path="src.py", old="    raise NotImplementedError", new="    return None")
            else:
                mu = mutate(src, rng)
                if not mu:
                    continue
                bad_l, good_l = mu[3], mu[2]
                if src.count(good_l) != 1:
                    continue
                ed = T("edit_file", path="src.py", old=good_l, new=bad_l)
            inj += [ed] + ([ed] if kind == "repeat" else []) + [T("run_tests")]
        else:
            inj = []
        t = dict(t, files=dict(t["files"], **{"src.py": src}), inject=inj, rkind=kind, id=t["id"] + suf)
        out.append(t)
        if len(out) >= a.n:
            break
    with open(f"{OUT}/{a.out}", "w", encoding="utf-8") as f:
        for t in out:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"### c3 build_r: {len(out)} bài · " + str(dict(collections.Counter(t["rkind"] for t in out))), flush=True)


def clean(t):
    """chỉ giữ phần trước + khối <tool>…</tool> đầu tiên (hoặc <done>…</done>) — bỏ token rác của định dạng công cụ riêng
    của thầy (vd Gemma: <tool_call|><|tool_response>) để aivie học đúng giao thức của mình."""
    m = re.search(r"<tool>.*?</tool>", t, re.S) or re.search(r"<done>.*?</done>", t, re.S)
    if m:
        t = t[:m.end()]
    return re.sub(r"<\|[^<>]*>|<[a-z_]+\|>", "", t).strip()                 # chỉ token có dấu |


def one(model, t, max_steps):
    ok, traj, wd = A.solve_tool(model, t["task"], t["files"], TEST_CMD, max_steps=max_steps, prefix=t.get("inject", ()))
    shutil.rmtree(wd, ignore_errors=True)
    own = [s for s in traj if not s.get("injected")]
    valid = bool(own) and own[-1]["call"] and own[-1]["call"]["name"] == "done" and \
        all(s["call"] is not None and s["call"]["name"] != "_bad_json" for s in own)
    msgs = [{"role": "user", "content": A.P_TOOL.format(task=t["task"])}]
    for s in traj:
        m = {"role": "assistant", "content": s["assistant"] if s.get("injected") else clean(s["assistant"])}
        if s.get("injected"):
            m["nolearn"] = True                 # lượt SAI chèn sẵn: cho model THẤY, không cho HỌC
        msgs.append(m)
        if s["obs"] is not None:
            msgs.append({"role": "user", "content": f"<obs>\n{s['obs']}\n</obs>"})
    rec = sum(1 for s in own if s["obs"] and ("Lỗi" in s["obs"][:40] or "CÓ TEST HỎNG" in s["obs"]))
    return {"id": t["id"], "kind": t["kind"], "rkind": t.get("rkind"), "ok": ok, "valid": valid, "steps": len(traj),
            "n_injected": len(traj) - len(own), "own_errors": rec, "messages": msgs}


def run(a):
    tasks = [json.loads(l) for l in open(f"{OUT}/{a.tasks}")][a.offset:]
    if a.n:
        tasks = tasks[:a.n]
    path = f"{OUT}/traj_{a.tag}.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
    tasks = [t for t in tasks if t["id"] not in done]
    model = A.Model(a.llm, a.model, max_new=1500, temp=0.2, model_id=a.model)
    print(f"[c3] {a.tag} · thầy {a.model} · {len(tasks)} bài", flush=True)
    T0, deadline = time.time(), time.time() + a.hours * 3600 if a.hours else float("inf")
    n = keep = 0
    with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        futs, it = set(), iter(tasks)
        def feed():
            while len(futs) < a.workers * 2 and time.time() < deadline:
                t = next(it, None)
                if t is None:
                    return
                futs.add(ex.submit(one, model, t, a.steps))
        feed()
        while futs:
            fut = next(as_completed(futs)); futs.discard(fut)
            try:
                r = fut.result()
            except Exception as e:
                print(f"[c3] lỗi bài: {type(e).__name__}: {str(e)[:120]}", flush=True); feed(); continue
            n += 1; keep += r["ok"] and r["valid"]
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if n % 200 == 0:
                print(f"[c3] {a.tag} {n} · thành công {keep} ({keep/n:.0%}) · {time.time()-T0:.0f}s", flush=True)
            feed()
    rows = [json.loads(l) for l in open(path)]
    good = sum(r["ok"] and r["valid"] for r in rows)
    print(f"### c3 {a.tag} · {a.model}: {len(rows)} bài · thành công {good} ({good/max(len(rows),1):.0%}) · "
          f"fix {sum(r['ok'] and r['valid'] for r in rows if r['kind']=='fix')} · "
          f"implement {sum(r['ok'] and r['valid'] for r in rows if r['kind']=='implement')} · {time.time()-T0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "build_r"])
    ap.add_argument("--tasks", default="tasks.jsonl", help="run: file bài (tasks_r.jsonl cho C3-R)")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--weights", default="0.35,0.3,0.2,0.15", help="build_r: tỉ lệ tab,bad,repeat,none")
    ap.add_argument("--out", default="tasks_r.jsonl", help="build_r: file bài ra (trong data/c3)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--model", default="gemma4-31b")
    ap.add_argument("--llm", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--hours", type=float, default=0, help="dừng nhận bài mới sau N giờ (0 = chạy hết)")
    a = ap.parse_args()
    {"build": build, "build_r": build_r, "run": run}[a.cmd](a)


if __name__ == "__main__":
    main()
