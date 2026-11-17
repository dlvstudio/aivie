"""TK_104 §18 · SO SÁNH CÔNG BẰNG với model nhỏ ngoài thị trường — CÙNG bộ đề, CÙNG giám khảo (Gemma-4-26B qua Ollama)
như đã chấm 1.3B của ta (v3/v4/v5).

Model ngoài phục vụ bằng vLLM (OpenAI API, cổng --url); hỏi đúng câu tiếng Việt như hỏi model của ta (không câu dặn hệ thống),
nhiệt độ 0. Bộ đề:
  sgk (400 câu giữ riêng) · toán (Vi-GSM8K 300 + bài sinh 200 + eval_v1 18, đáp số máy kiểm) · văn phòng (150 đề, Excel tính thật)
  · easy20 · chat 5 lượt (60 hội thoại) · eval_v1 (158 câu → eval_clean judge/report) · MBPP-500 · HumanEval-164

    /workspace/vllm_env/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct --served-model-name ext --port 8001 &
    /workspace/zenv/bin/python scripts/eval_external.py --tag qwen25_1p5b --url http://127.0.0.1:8001/v1
"""
from __future__ import annotations
import argparse, collections, json, os, re, subprocess, sys, tempfile, time, urllib.request
import concurrent.futures as cf
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]

OUT = "ckpt/expert_code"


def chat(url, msgs, n=512):
    body = {"model": "ext", "temperature": 0, "max_tokens": n, "messages": msgs,
            "chat_template_kwargs": {"enable_thinking": False}}          # Qwen3/Gemma-4: tắt chế độ "suy nghĩ" cho công bằng
    req = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=600).read())["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(lỗi: {e})"


def pmap(fn, xs, w=32):
    with cf.ThreadPoolExecutor(w) as ex:
        return list(ex.map(fn, xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--judge", default="http://127.0.0.1:11434", help="giám khảo (Ollama Gemma-26B như các bản của ta)")
    ap.add_argument("--only", default="sgk,math,off,easy20,mt,v1,mbpp,he")
    a = ap.parse_args()
    import expert_office_1p3b as X
    from gen_math import num_last, close
    res = {"tag": a.tag}
    U = lambda q: [{"role": "user", "content": q}]
    todo = a.only.split(",")

    if "sgk" in todo:
        rows = [json.loads(l) for l in open("data/sgk/test.jsonl", encoding="utf-8")][:400]
        ans = pmap(lambda q: chat(a.url, U(q["q"]), 400), rows)
        def j(p):
            q, an = p
            try:
                return X.llm_json(X.P_SGK.format(q=q["q"], ref=q["ref"], ans=an), a.judge).get("v", "?")
            except Exception:
                return "?"
        v = collections.Counter(pmap(j, list(zip(rows, ans)), 8))
        res["sgk"] = {k: round(100 * c / len(rows), 1) for k, c in v.items()}
        print("[sgk]", res["sgk"], flush=True)

    if "math" in todo:
        rows = [json.loads(l) for l in open("data/math/test.jsonl", encoding="utf-8")]
        ev = [json.loads(l) for l in open("data/eval_clean/eval_v1.jsonl", encoding="utf-8")]
        rows += [{"q": x["q"], "gold": num_last(x["gold"]), "src": "eval_v1"} for x in ev if x["type"] == "math"]
        ans = pmap(lambda q: chat(a.url, U(q["q"]), 700), rows)
        m = collections.defaultdict(lambda: [0, 0])
        for q, an in zip(rows, ans):
            m[q["src"]][0] += 1; m[q["src"]][1] += close(num_last(an), q["gold"])
        res["math"] = {k: f"{v[1]}/{v[0]}" for k, v in m.items()}
        print("[math]", res["math"], flush=True)

    if "off" in todo:
        from gen_office import check_excel, judge_work
        rows = [json.loads(l) for l in open("data/office/eval_office.jsonl", encoding="utf-8")]
        ans = pmap(lambda q: chat(a.url, U(q["q"]), 900), rows)
        judged = pmap(lambda p: judge_work(({**p[0], "request": p[0]["q"], "answer": p[1]}, a.judge)), list(zip(rows, ans)), 8)
        by, xl = collections.defaultdict(list), []
        for r, an in zip(judged, ans):
            if "j" in r:
                by[r["cat"]].append(r["j"])
            if "formula" in r:
                f_ = X.extract_formula(an)
                try:
                    xl.append(bool(f_) and check_excel(r["table"], f_, r["expected"])[0])
                except Exception:
                    xl.append(False)
        res["off"] = {c: {"ok": round(sum(x["ok"] for x in v) / len(v), 2), "vi": round(sum(x["vi"] for x in v) / len(v), 2)} for c, v in by.items()}
        res["off"]["excel_dung"] = f"{sum(xl)}/{len(xl)}"
        print("[off]", res["off"], flush=True)

    if "easy20" in todo:
        rows = [json.loads(l) for l in open("data/eval_clean/easy20.jsonl", encoding="utf-8")]
        ans = pmap(lambda q: chat(a.url, U(q["q"]), 150), rows)
        res["easy20"] = sum(q["gold"].lower() in an.lower() for q, an in zip(rows, ans))
        print("[easy20]", res["easy20"], flush=True)

    if "mt" in todo:
        rows = [json.loads(l) for l in open("data/multiturn/probe.jsonl", encoding="utf-8")]
        def conv(c):
            msgs, sc = [], []
            for t, u in enumerate(c["users"]):
                msgs.append({"role": "user", "content": u})
                an = chat(a.url, msgs, 450)
                hist = "\n".join(f"{'NGƯỜI DÙNG' if m_['role'] == 'user' else 'TRỢ LÝ'}: {m_['content'][:600]}" for m_ in msgs[:-1])
                try:
                    jj = X.llm_json(X.P_MT.format(hist=hist or "(chưa có)", q=u, ans=an[:2000]), a.judge)
                    sc.append((t + 1, float(jj.get("bam", 0)), float(jj.get("ok", 0))))
                except Exception:
                    pass
                msgs.append({"role": "assistant", "content": an})
            return sc
        by = collections.defaultdict(list)
        for sc in pmap(conv, rows, 8):
            for t, b, o in sc:
                by[t].append((b, o))
        res["mt"] = {f"luot_{t}": round(sum(b for b, _ in v) / len(v), 2) for t, v in sorted(by.items())}
        print("[mt]", res["mt"], flush=True)

    if "v1" in todo:
        rows = [json.loads(l) for l in open("data/eval_clean/eval_v1.jsonl", encoding="utf-8")]
        ans = pmap(lambda q: chat(a.url, U(q["q"]), 700), rows)
        with open(f"out/evalclean_{a.tag}.jsonl", "w", encoding="utf-8") as f:
            for q, an in zip(rows, ans):
                f.write(json.dumps({**q, "tag": a.tag, "answer": an}, ensure_ascii=False) + "\n")
        print("[v1] đã sinh → chấm bằng eval_clean.py judge", flush=True)

    if "mbpp" in todo or "he" in todo:
        from datasets import load_dataset
        def run(prog):
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(prog)
            try:
                return subprocess.run([sys.executable, f.name], capture_output=True, timeout=10).returncode == 0
            except subprocess.TimeoutExpired:
                return False
            finally:
                os.unlink(f.name)
        code_of = lambda r: (re.search(r"```(?:python)?\n(.*?)```", r, re.S) or [None, r])[1]
        if "mbpp" in todo:
            ds = list(load_dataset("google-research-datasets/mbpp", "full", split="test"))[:500]
            qs = [f"{ex['text']}\nYour code should pass this test:\n{ex['test_list'][0]}\nWrite the Python function." for ex in ds]
            ans = pmap(lambda q: chat(a.url, U(q), 600), qs)
            ok = sum(run(code_of(an) + "\n\n" + "\n".join(ex["test_list"]) + "\n") for ex, an in zip(ds, ans))
            res["mbpp500"] = round(ok / len(ds), 3)
            print("[mbpp]", res["mbpp500"], flush=True)
        if "he" in todo:
            ds = list(load_dataset("openai/openai_humaneval", split="test"))
            ans = pmap(lambda ex: chat(a.url, U(f"Complete the following Python function:\n```python\n{ex['prompt']}```"), 600), ds)
            ok = 0
            for ex, an in zip(ds, ans):
                c = code_of(an)
                if f"def {ex['entry_point']}" not in c:
                    c = ex["prompt"] + c
                ok += run(c + "\n\n" + ex["test"] + f"\ncheck({ex['entry_point']})\n")
            res["humaneval"] = round(ok / len(ds), 3)
            print("[he]", res["humaneval"], flush=True)
    json.dump(res, open(f"{OUT}/external_{a.tag}.json", "w"), ensure_ascii=False, indent=1)
    print("### XONG", json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
