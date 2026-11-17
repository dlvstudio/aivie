"""Kiểm chứng cổng expert — phần văn phòng: 27 bài Excel (kiểm thật bằng formulas) + 6 bài văn phòng khác (lưu để so) qua UI /api/chat.
    python scripts/gate_probe_office.py http://localhost:8801 open"""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from code_agent import Model
from expert_office_1p3b import extract_formula
from gen_office import check_excel
url, tag = sys.argv[1], sys.argv[2]
m = Model(url, max_new=600)
R = [json.loads(l) for l in open("data/office/eval_office.jsonl", encoding="utf-8")]
xl = [r for r in R if "formula" in r]
other = random.Random(0).sample([r for r in R if "formula" not in r], 6)
path = f"out/gate_probe_office_{tag}.jsonl"
done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()
with open(path, "a", encoding="utf-8") as f:
    for r in xl + other:
        if r["id"] in done:
            continue
        ans = m.chat([{"role": "user", "content": r["q"]}])
        rec = {"id": r["id"], "cat": r["cat"], "q": r["q"], "answer": ans}
        if "formula" in r:
            f_ = extract_formula(ans)
            try:
                good, _ = check_excel(r["table"], f_, r["expected"]) if f_ else (False, None)
            except Exception:
                good = False
            rec.update(excel_ok=bool(good), formula_model=f_, formula_ref=r["formula"])
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
xs = [r["excel_ok"] for r in rows if "excel_ok" in r]
print("EXCEL", tag, f"{sum(xs)}/{len(xs)}")
