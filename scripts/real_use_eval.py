# chạy bộ ca dùng thực tế (thiên về code) qua /api/chat của UI c6l1 (cổng 8805), lưu từng câu trả lời
import json, os, sys, time, urllib.request
URL = os.environ.get("URL", "http://localhost:8805") + "/api/chat"
OUT = os.environ.get("OUT", "out/real_eval_c6l1"); os.makedirs(OUT, exist_ok=True)

def chat(msgs, max_new=2048):
    body = json.dumps({"messages": msgs, "max_new": max_new, "temperature": 0.7, "top_p": 0.9, "rep_pen": 1.1}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); done = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data:"):
                ev = json.loads(line[5:])
                if ev.get("type") == "done": done = ev
                if ev.get("type") == "error": return {"text": "[LỖI] " + ev["error"], "tokens": 0, "sec": time.time() - t0}
    return {"text": done["text"], "tokens": done.get("tokens", 0), "sec": round(time.time() - t0, 1)}

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
    json.dump({"id": c["id"], "cat": c["cat"], "turns": turns}, open(f"{OUT}/{c['id']}.json", "w"), ensure_ascii=False, indent=1)
