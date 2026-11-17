"""BỘ ĐO CODE + TOOL-USE cho aivie (TK_108). Chạy model qua khung chạy có công cụ (code_agent.py), code chạy THẬT trong sandbox.

  humaneval  164 bài (openai/openai_humaneval). Test công khai = ví dụ >>> trong docstring; chấm = test ẩn đầy đủ.
  mbpp       MBPP sanitized, phần TEST (257 bài). Test công khai = assert đầu tiên; chấm = cả 3 assert.
  agent      bài tool-use trong thư mục dự án (dựng từ MBPP phần train/validation — KHÔNG trùng phần test):
               fix        src.py bị cài 1 lỗi logic, test_src.py đang hỏng → sửa cho test qua
               implement  src.py chỉ có chữ ký + docstring → viết thân hàm cho test qua
  Mỗi bài HumanEval/MBPP đo 2 số: pass@1 (một lần, không chạy thử) và pass sau ≤N vòng "chạy test công khai → sửa".

    python scripts/code_eval.py build-agent                    → data/code_eval/agent_tasks.jsonl (cố định, seed 108)
    python scripts/code_eval.py run --suite humaneval --url http://localhost:8798 --tag sft2500 [--n 40] [--rounds 3]
    python scripts/code_eval.py run --suite agent --url http://localhost:8798 --tag sft2500
    python scripts/code_eval.py report --tags v5,sft2500        → bảng so sánh
"""
from __future__ import annotations
import argparse, doctest, json, os, random, re, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import code_agent as A

OUT = "out/code_eval"
TASKS = "data/code_eval/agent_tasks.jsonl"


# ── dữ liệu ──────────────────────────────────────────────────────────────────────────────
def humaneval():
    from datasets import load_dataset
    cache = "data/code_eval/humaneval_public_ok.json"
    ok_cache = json.load(open(cache)) if os.path.exists(cache) else {}
    out = []
    for r in load_dataset("openai/openai_humaneval", split="test"):
        ex = doctest.DocTestParser().get_examples(r["prompt"])          # ví dụ >>> trong docstring → test công khai
        want = lambda e: e.want.strip().split("\n")[0].strip()                 # bỏ dòng thừa (vd dấu """ đóng docstring)
        pub = "\n".join(f"assert ({e.source.strip()}) == {want(e)}" for e in ex
                        if want(e) and '"""' not in want(e) and "\n" not in e.source.strip())
        if pub:                                       # chỉ giữ ví dụ mà lời giải CHUẨN qua (vài docstring gốc ghi sai)
            key = r["task_id"]
            if key not in ok_cache:
                ok_cache[key] = [l for l in pub.split("\n")
                                 if A.run_python(r["prompt"] + r["canonical_solution"] + "\n" + l, timeout=10)["rc"] == 0]
            pub = "\n".join(ok_cache[key])
        imports = "\n".join(l for l in r["prompt"].split("\n") if l.startswith(("import ", "from ")))
        task = ("Hoàn thành hàm Python sau (giữ nguyên tên hàm và tham số):\n```python\n" + r["prompt"].strip() + "\n```")
        out.append({"id": r["task_id"], "task": task, "prefix": imports + "\n", "public": pub,
                    "hidden": r["test"] + f"\ncheck({r['entry_point']})\n"})
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(ok_cache, open(cache, "w"), indent=0)
    return out


def mbpp():
    from datasets import load_dataset
    out = []
    for r in load_dataset("google-research-datasets/mbpp", "sanitized", split="test"):
        t = r["test_list"]
        task = f"{r['prompt']}\nHàm cần qua test mẫu: {t[0]}"
        out.append({"id": f"mbpp_{r['task_id']}", "task": task, "prefix": "\n".join(r.get("test_imports") or []) + "\n",
                    "public": t[0], "hidden": "\n".join(t)})
    return out


def build_agent(n=40, seed=108):
    """bài tool-use từ MBPP train+validation: kiểm thật — lời giải gốc QUA test, bản cài lỗi phải HỎNG."""
    from datasets import load_dataset
    from gen_screens import mutate
    rng = random.Random(seed)
    rows = [r for sp in ("train", "validation") for r in load_dataset("google-research-datasets/mbpp", "sanitized", split=sp)]
    rng.shuffle(rows)
    out = []
    for r in rows:
        if len(out) >= n:
            break
        code, tests = r["code"].strip(), r["test_list"]
        fn = re.search(r"^def (\w+)\(", code, re.M)
        if not fn:
            continue
        test_py = "\n".join(r.get("test_imports") or []) + f"\nfrom src import *\n\n" + "\n".join(tests) + "\nprint('TẤT CẢ TEST QUA')\n"
        kind = "fix" if len(out) % 2 == 0 else "implement"
        if kind == "fix":
            m = mutate(code, rng)
            if not m:
                continue
            src = m[0]
            task = ("Trong dự án có `src.py` và bộ test `test_src.py`. Chạy test đang HỎNG vì `src.py` có lỗi. "
                    f"Hãy tìm và sửa lỗi trong `src.py` (không sửa test). Yêu cầu gốc của hàm: {r['prompt']}")
        else:
            sm = re.search(r"^(def \w+\(.*?\)[^:\n]*:)", code, re.M | re.S)          # có thể có "-> kiểu"
            if not sm:
                continue
            sig = sm.group(1)
            head = "\n".join(l for l in code.split("\n") if l.startswith(("import ", "from ")))
            src = (head + "\n\n" if head else "") + f'{sig}\n    """{r["prompt"]}"""\n    raise NotImplementedError\n'
            task = (f"Trong dự án có `src.py` (hàm `{fn.group(1)}` chưa viết) và bộ test `test_src.py`. "
                    f"Hãy viết thân hàm trong `src.py` cho tất cả test qua. Yêu cầu: {r['prompt']}")
        files_ok = {"src.py": code + "\n", "test_src.py": test_py}
        files_bad = {"src.py": src, "test_src.py": test_py}
        if _run_tests(files_ok) != 0 or _run_tests(files_bad) == 0:           # gốc phải qua · bài giao phải đang hỏng
            continue
        out.append({"id": f"agent_{kind}_{r['task_id']}", "kind": kind, "task": task, "files": files_bad})
    os.makedirs(os.path.dirname(TASKS), exist_ok=True)
    with open(TASKS, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"### {len(out)} bài tool-use → {TASKS} (fix {sum(x['kind'] == 'fix' for x in out)} · "
          f"implement {sum(x['kind'] == 'implement' for x in out)})")


def _run_tests(files):
    import tempfile
    wd = tempfile.mkdtemp(prefix="aivie_chk_")
    try:
        for p, c in files.items():
            open(os.path.join(wd, p), "w", encoding="utf-8").write(c)
        return A.run([A.PY, "-B", "-E", "-s", "test_src.py"], wd, timeout=20)["rc"]
    finally:
        shutil.rmtree(wd, ignore_errors=True)


# ── chạy ─────────────────────────────────────────────────────────────────────────────────
def run_suite(a):
    model = A.Model(a.url, a.tag, max_new=a.max_new, model_id=a.model_id)
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{a.suite}_{a.tag}.jsonl"
    done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()
    if a.suite == "agent":
        probs = [json.loads(l) for l in open(TASKS, encoding="utf-8")]
    else:
        probs = humaneval() if a.suite == "humaneval" else mbpp()
    probs = [p for p in probs[:a.n or None] if p["id"] not in done]
    T0 = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for k, p in enumerate(probs):
            t = time.time()
            if a.suite == "agent":
                ok, traj, wd = A.solve_tool(model, p["task"], p["files"], [A.PY, "-B", "-E", "-s", "test_src.py"], max_steps=a.steps)
                shutil.rmtree(wd, ignore_errors=True)
                rec = {"id": p["id"], "kind": p["kind"], "ok": ok, "steps": len(traj),
                       "n_tool_ok": sum(1 for s in traj if s["call"] and s["call"]["name"] not in ("_bad_json",)),
                       "n_no_call": sum(1 for s in traj if s["call"] is None), "traj": traj}
            else:
                code, hist = A.solve_debug(model, p["task"], p["public"] or "pass", p["prefix"], rounds=a.rounds)
                first = hist[0]["code"]
                grade = lambda c: (A.run_python(p["prefix"] + c + "\n\n" + p["hidden"], timeout=15)["rc"] == 0) if c else False
                rec = {"id": p["id"], "pass1": grade(first), "pass_debug": grade(code), "rounds": len(hist), "hist": hist}
            rec["sec"] = round(time.time() - t, 1)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if k % 5 == 0 or k == len(probs) - 1:
                print(f"[{a.suite} {a.tag}] {k+1}/{len(probs)} · {time.time()-T0:.0f}s · {summary(path)}", flush=True)
    print("###", a.suite, a.tag, summary(path))


def summary(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []
    if not rows:
        return {}
    n = len(rows)
    if "ok" in rows[0]:
        by = {}
        for r in rows:
            by.setdefault(r["kind"], []).append(r["ok"])
        return {"n": n, "thành_công_%": round(100 * sum(r["ok"] for r in rows) / n, 1),
                **{f"{k}_%": round(100 * sum(v) / len(v), 1) for k, v in by.items()},
                "lượt_không_gọi_công_cụ_%": round(100 * sum(r["n_no_call"] for r in rows) / max(1, sum(r["steps"] for r in rows)), 1)}
    return {"n": n, "pass@1_%": round(100 * sum(r["pass1"] for r in rows) / n, 1),
            "sau_sửa_%": round(100 * sum(r["pass_debug"] for r in rows) / n, 1)}


def report(a):
    print(f"{'bộ':10s} | " + " | ".join(f"{t:>28s}" for t in a.tags.split(",")))
    for s in ("humaneval", "mbpp", "agent"):
        cells = [json.dumps(summary(f"{OUT}/{s}_{t}.jsonl"), ensure_ascii=False) for t in a.tags.split(",")]
        print(f"{s:10s} | " + " | ".join(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-agent", "run", "report"])
    ap.add_argument("--suite", default="humaneval", choices=["humaneval", "mbpp", "agent"])
    ap.add_argument("--url", default="http://localhost:8798")
    ap.add_argument("--model-id", default="", help="tên model khi --url là máy chủ /v1 (vLLM/Ollama)")
    ap.add_argument("--tag", default="sft2500"); ap.add_argument("--tags", default="v5,sft2500")
    ap.add_argument("--n", type=int, default=0, help="chỉ chạy n bài đầu (0 = tất cả)")
    ap.add_argument("--rounds", type=int, default=3, help="số vòng sửa tối đa (debug)")
    ap.add_argument("--steps", type=int, default=10, help="số lượt gọi công cụ tối đa (agent)")
    ap.add_argument("--max-new", type=int, default=600)
    a = ap.parse_args()
    {"build-agent": lambda: build_agent(), "run": lambda: run_suite(a), "report": lambda: report(a)}[a.cmd]()
