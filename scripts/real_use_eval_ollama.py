# cùng bộ ca dùng thực tế (data/eval_clean/real_code_22.json) nhưng hỏi model Ollama — so với c6l1 cùng cài đặt sinh
# Dùng: MODEL=qwen2.5-coder:7b OUT=out/real_eval_qwen7b python3 scripts/real_use_eval_ollama.py data/eval_clean/real_code_22.json [id…]
import json, os, sys, time, urllib.request
MODEL = os.environ.get("MODEL", "qwen2.5-coder:7b")
OUT = os.environ.get("OUT", "out/real_eval_" + MODEL.replace(":", "_").replace("/", "_")); os.makedirs(OUT, exist_ok=True)
URL = os.environ.get("URL", "http://localhost:11434") + "/api/chat"
MAXNEW = int(os.environ.get("MAXNEW", 0))   # >0: ghi đè max_new (model có "suy nghĩ" như Gemma 4 tốn thêm token)
CTX = int(os.environ.get("CTX", 8192))

def chat(msgs, max_new=2048):
    body = json.dumps({"model": MODEL, "messages": msgs, "stream": False,
                       "options": {"temperature": 0.7, "top_p": 0.9, "repeat_penalty": 1.1, "num_predict": MAXNEW or max_new, "num_ctx": CTX}}).encode()
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=1800) as r:
        d = json.loads(r.read())
    return {"text": d["message"]["content"], "thinking": d["message"].get("thinking", ""), "tokens": d.get("eval_count", 0), "sec": round(time.time() - t0, 1)}

CASES = json.load(open(sys.argv[1]))
only = set(sys.argv[2:])
for c in CASES:
    if only and c["id"] not in only: continue
    msgs, turns = [], []
    for u in c["turns"]:
        msgs.append({"role": "user", "content": u})
        r = chat(msgs, c.get("max_new", 2048))
        msgs.append({"role": "assistant", "content": r["text"]})
        turns.append({"user": u, **r})
        print(f"[{c['id']}] {r['tokens']} tok · {r['sec']}s · {r['tokens']/max(r['sec'],1e-3):.1f} tok/s", flush=True)
    json.dump({"id": c["id"], "cat": c["cat"], "model": MODEL, "turns": turns}, open(f"{OUT}/{c['id']}.json", "w"), ensure_ascii=False, indent=1)
