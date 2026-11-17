"""20 câu phổ thông dễ qua UI /api/chat — kiểm nhanh ghép expert không làm hỏng hội thoại chung.
    python scripts/easy20_probe.py http://localhost:8801 <tag>"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from code_agent import Model
url, tag = sys.argv[1], sys.argv[2]
m = Model(url, max_new=120)
rows = [json.loads(l) for l in open("data/eval_clean/easy20.jsonl", encoding="utf-8")]
out, ok = [], 0
for r in rows:
    ans = m.chat([{"role": "user", "content": r["q"]}])
    hit = r["gold"].lower() in ans.lower(); ok += hit
    out.append({**r, "answer": ans, "hit": hit})
json.dump(out, open(f"out/easy20_probe_{tag}.json", "w"), ensure_ascii=False, indent=1)
print("EASY20", tag, f"{ok}/{len(rows)}")
