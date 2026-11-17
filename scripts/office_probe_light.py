"""Đo Excel 27 bài (chạy công thức thật) + easy20 qua UI /api/chat — KHÔNG cần torch (chạy bằng evalenv trên box).
    python scripts/office_probe_light.py http://localhost:8821 <tag>     → out/office_probe_<tag>.jsonl"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from code_agent import Model
from gen_office import check_excel


def extract_formula(ans):                       # giống expert_office_1p3b.extract_formula (chấm chặt: ưu tiên khối ```)
    f_ = None
    for blk in re.findall(r"```[a-zA-Z]*\n(.*?)```", ans, re.S):
        f_ = next((ln.strip() for ln in blk.splitlines() if ln.strip().startswith("=")), None)
        if f_:
            break
    if not f_:
        m_ = re.search(r"(=[A-Z]+\(.*\))", ans)
        f_ = m_.group(1).strip() if m_ else None
    if f_ and ";" in f_ and "," not in f_:
        f_ = f_.replace(";", ",")
    return f_


url, tag = sys.argv[1], sys.argv[2]
m = Model(url, max_new=600)
xl = [r for r in (json.loads(l) for l in open("data/office/eval_office.jsonl", encoding="utf-8")) if "formula" in r]
ok = blk = 0
with open(f"out/office_probe_{tag}.jsonl", "w", encoding="utf-8") as f:
    for r in xl:
        ans = m.chat([{"role": "user", "content": r["q"]}])
        fm = extract_formula(ans)
        try:
            good = bool(check_excel(r["table"], fm, r["expected"])[0]) if fm else False
        except Exception:
            good = False
        ok += good; blk += bool(re.search(r"```[a-zA-Z]*\n\s*=", ans))
        f.write(json.dumps({"id": r["id"], "q": r["q"], "answer": ans, "formula": fm, "ok": good}, ensure_ascii=False) + "\n")
e20 = [json.loads(l) for l in open("data/eval_clean/easy20.jsonl", encoding="utf-8")]
m.max_new = 120
hit = sum(r["gold"].lower() in m.chat([{"role": "user", "content": r["q"]}]).lower() for r in e20)
print(f"### {tag} · Excel {ok}/{len(xl)} · công thức trong khối {blk}/{len(xl)} · easy20 {hit}/{len(e20)}", flush=True)
