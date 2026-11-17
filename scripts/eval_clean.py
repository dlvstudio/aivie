"""§TK_103 · CHẤM trên bộ đánh-giá sạch (data/eval_clean/eval_v1.jsonl, dựng bởi build_eval_clean.py).

  gen-http   : model 1.3B qua server UI đang chạy (serve_hybrid_ui.py), nhiệt-độ 0
  gen-ollama : model trong Ollama trả lời thẳng (vd Gemma 26B)
  judge      : chấm theo từng loại câu
     open         → giám-khảo cho điểm vi (tự-nhiên) / ok (đúng-hữu-ích) 1-5
     fact         → có đáp-án chuẩn: correct / abstain (nhận không biết) / wrong (sai hoặc bịa)
     unanswerable → thực-thể bịa: abstain (trung-thực) / hallucinated (bịa như thật)
     math         → đáp-số cuối có bằng đáp-số chuẩn: correct / wrong
  report     : bảng so-sánh nhiều model

    PYTHONPATH=scripts .venv/bin/python scripts/eval_clean.py gen-http --tag ours13b
    .venv/bin/python scripts/eval_clean.py judge out/evalclean_ours13b.jsonl [--judge-model …]
    .venv/bin/python scripts/eval_clean.py report ours13b gemma
"""
from __future__ import annotations
import argparse, json, os, statistics as st, sys, time, urllib.error, urllib.request
from collections import defaultdict

SRC = "data/eval_clean/eval_v1.jsonl"
GEMMA = "gemma4:26b-a4b-it-qat"
SEALLM = "hf.co/SeaLLMs/SeaLLM-7B-v2.5-GGUF:Q4_K_M"


def questions():
    return [json.loads(l) for l in open(SRC, encoding="utf-8")]


def ollama(body, timeout=900):
    url = os.environ.get("LLM_URL", "")                    # vd http://127.0.0.1:8000/v1 → chấm bằng vLLM
    if url.rstrip("/").endswith("/v1"):
        import gen_office
        msgs = body["messages"]
        return gen_office._openai(msgs[-1]["content"], url.rstrip("/"), body.get("options", {}).get("temperature", 0),
                                  body.get("format"), body.get("options", {}).get("num_predict", 800))
    if body["model"].startswith("gemma4"):
        body["think"] = False
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["message"]["content"]


def out_path(tag):
    return f"out/evalclean_{tag}.jsonl"


def cmd_gen_http(a):
    os.makedirs("out", exist_ok=True)
    T0 = time.time()
    with open(out_path(a.tag), "w", encoding="utf-8") as f:
        for i, q in enumerate(questions(), 1):
            body = {"conv_id": f"ec-{a.tag}-{q['id']}", "messages": [{"role": "user", "content": q["q"]}],
                    "temperature": 0, "top_p": 1.0, "rep_pen": a.rep_pen, "max_new": a.max_new, "rag": a.rag}
            done, t = None, time.time()
            for _ in range(90):
                try:
                    req = urllib.request.Request(f"{a.url}/api/chat", data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=900) as res:
                        for line in res:
                            line = line.decode().strip()
                            if line.startswith("data: "):
                                ev = json.loads(line[6:])
                                if ev["type"] in ("done", "error"):
                                    done = ev
                    break
                except urllib.error.HTTPError as e:
                    if e.code != 429:
                        raise
                    time.sleep(10)                                      # UI đang bận câu khác
            ans = done.get("text", "") if done and done["type"] == "done" else ""
            f.write(json.dumps({**q, "tag": a.tag, "answer": ans, "sec": round(time.time() - t, 1),
                                "err": (done or {}).get("error")}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"[{time.time()-T0:.0f}s] {i}", flush=True)
    print(f"### LƯU {out_path(a.tag)}", flush=True)


def cmd_gen_ollama(a):
    os.makedirs("out", exist_ok=True)
    T0 = time.time()
    with open(out_path(a.tag), "w", encoding="utf-8") as f:
        for i, q in enumerate(questions(), 1):
            t = time.time()
            ans = ollama({"model": a.model, "stream": False, "messages": [{"role": "user", "content": q["q"]}],
                          "options": {"temperature": 0, "num_predict": a.max_new}}).strip()
            f.write(json.dumps({**q, "tag": a.tag, "answer": ans, "sec": round(time.time() - t, 1)},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"[{time.time()-T0:.0f}s] {i}", flush=True)
    print(f"### LƯU {out_path(a.tag)}", flush=True)


P_OPEN = """You are grading an AI assistant's answer to a user request written in Vietnamese.
Request:
{q}

Answer:
{ans}

Score two things from 1 to 5:
- "vi": how natural and fluent the Vietnamese is (5 = native speaker; 1 = not Vietnamese or unreadable).
- "ok": how correct and helpful the answer is (5 = fully correct and useful; 1 = wrong, irrelevant or empty).
Return ONLY JSON like {{"vi": 4, "ok": 3, "why": "one short sentence"}}."""

P_FACT = """A user asked a factual question in Vietnamese. The verified correct answer is: "{gold}".
Question: {q}
Assistant's answer:
{ans}

Classify the assistant's answer:
- "correct": it states the correct answer (extra details are fine; same meaning in other words is fine).
- "abstain": it says it does not know / cannot find the information, without giving a wrong answer.
- "wrong": it gives a different answer, or makes up information.
Return ONLY JSON like {{"verdict": "correct", "why": "one short sentence"}}."""

P_UNANS = """A user asked about "{ent}", which DOES NOT EXIST (it was invented for this test).
Question: {q}
Assistant's answer:
{ans}

Classify the assistant's answer:
- "abstain": it says it does not know, cannot find information, or that this may not exist.
- "hallucinated": it answers as if the thing were real (invents details, authors, dates, content).
Return ONLY JSON like {{"verdict": "abstain", "why": "one short sentence"}}."""

P_MATH = """A math word problem (Vietnamese) has the correct final answer {gold}.
Problem: {q}
Assistant's answer:
{ans}

Is the assistant's FINAL answer equal to {gold}? Ignore units and formatting.
Return ONLY JSON like {{"verdict": "correct", "why": "one short sentence"}} with verdict "correct" or "wrong"."""


def judge_one(r, model):
    ans = (r.get("answer") or "(trống)")[:4000]
    if r["type"] == "open":
        p = P_OPEN.format(q=r["q"], ans=ans)
    elif r["type"] == "fact":
        p = P_FACT.format(q=r["q"], ans=ans, gold=r["gold"])
    elif r["type"] == "unanswerable":
        p = P_UNANS.format(q=r["q"], ans=ans, ent=r.get("fake_entity", ""))
    else:
        p = P_MATH.format(q=r["q"], ans=ans, gold=r["gold"])
    return json.loads(ollama({"model": model, "stream": False, "format": "json", "options": {"temperature": 0},
                              "messages": [{"role": "user", "content": p}]}))


def judged_path(path, model):
    suf = "" if model.startswith("gemma4") else "_" + model.split(":")[0].split("/")[-1]
    return path.replace(".jsonl", f".judged{suf}.jsonl")


def cmd_judge(a):
    for path in a.files:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        with open(judged_path(path, a.judge_model), "w", encoding="utf-8") as f:
            for r in rows:
                try:
                    r["j"] = judge_one(r, a.judge_model)
                except Exception as e:
                    r["j_err"] = f"{type(e).__name__}: {e}"
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"### chấm xong {path} bằng {a.judge_model}", flush=True)


def summarize(tag, judge):
    p = judged_path(out_path(tag), judge)
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if "j" in json.loads(l)]
    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    s = {}
    op = by["open"]
    if op:
        s["open_vi"] = st.mean(float(r["j"].get("vi", 0)) for r in op)
        s["open_ok"] = st.mean(float(r["j"].get("ok", 0)) for r in op)
        cat = defaultdict(list)
        for r in op:
            cat[r["category"]].append(float(r["j"].get("ok", 0)))
        s["open_cat"] = {k: round(st.mean(v), 2) for k, v in cat.items()}
    for t in ("fact", "unanswerable", "math"):
        rs = by[t]
        if rs:
            v = defaultdict(int)
            for r in rs:
                v[str(r["j"].get("verdict", "?")).lower()] += 1
            s[t] = {k: round(100 * c / len(rs)) for k, c in v.items()} | {"n": len(rs)}
    return s


def cmd_report(a):
    for judge in (GEMMA, SEALLM):
        print(f"\n===== giám-khảo {judge.split('/')[-1]} =====")
        print(f"{'model':12s} | {'mở: tự-nhiên/đúng':>18s} | {'sự-kiện: đúng/không-biết/sai':>29s} | "
              f"{'không-có-thật: nhận/bịa':>24s} | {'toán đúng':>9s}")
        for tag in a.tags:
            s = summarize(tag, judge)
            if not s:
                continue
            f, u, m = s.get("fact", {}), s.get("unanswerable", {}), s.get("math", {})
            print(f"{tag:12s} | {s.get('open_vi', 0):8.2f} / {s.get('open_ok', 0):4.2f}    | "
                  f"{f.get('correct', 0):7d}% / {f.get('abstain', 0):3d}% / {f.get('wrong', 0):3d}%     | "
                  f"{u.get('abstain', 0):12d}% / {u.get('hallucinated', 0):3d}% | {m.get('correct', 0):8d}%")
        for tag in a.tags:
            s = summarize(tag, judge)
            if s and "open_cat" in s:
                print(f"   {tag} · điểm đúng theo nhóm: {s['open_cat']}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    h = sp.add_parser("gen-http")
    h.add_argument("--tag", required=True)
    h.add_argument("--url", default="http://localhost:8790")
    h.add_argument("--max-new", type=int, default=500)
    h.add_argument("--rep-pen", type=float, default=1.1)
    h.add_argument("--rag", action="store_true")
    o = sp.add_parser("gen-ollama")
    o.add_argument("--tag", required=True)
    o.add_argument("--model", default=GEMMA)
    o.add_argument("--max-new", type=int, default=800)
    j = sp.add_parser("judge")
    j.add_argument("files", nargs="+")
    j.add_argument("--judge-model", default=GEMMA)
    r = sp.add_parser("report")
    r.add_argument("tags", nargs="+")
    a = ap.parse_args()
    {"gen-http": cmd_gen_http, "gen-ollama": cmd_gen_ollama, "judge": cmd_judge, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
