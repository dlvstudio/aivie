"""TK_108 C2 (phần không cần thầy): dữ liệu code ĐÃ CHẠY THẬT từ nguồn có test sẵn.

Nguồn: nvidia/OpenCodeInstruct (CC-BY-4.0, dùng thương mại được) — mỗi bài có lời giải + danh sách assert.
Lọc:  điểm test gốc = 1.0 · giám khảo gốc logical_correctness = 5 · có khối ```python · không quá dài
      · BỎ bài gần giống HumanEval / MBPP (n-gram 13 từ của đề + tên hàm) — tránh rò bộ đo
      · CHẠY LẠI toàn bộ assert trong sandbox của code_agent (cấm mạng, cấm ghi ngoài) → chỉ giữ bài qua hết.
Ra:   data/c2/oci_verified.jsonl  {"messages": [user, assistant], "tests": [...], "src", "id", "domain"}
      (giữ tests để dùng lại cho C5 — RL thưởng bằng test)

Dùng: python scripts/c2_build_verified.py --shards 0,1,2,3 [--limit N] [--workers 10]   (chạy lại được — bỏ qua id đã chấm)
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
import code_agent as CA

RAW = "data/c2/raw/data/train-{:05d}-of-00050.parquet"
OUT = "data/c2/oci_verified.jsonl"
SEEN = "data/c2/oci_checked.jsonl"                    # mọi id đã chạy (qua/trượt) — để chạy lại không làm lại
N = 13


def words(t):
    return re.findall(r"[a-z0-9_]+", t.lower())


def ngrams(t, n=N):
    w = words(t)
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def eval_index():
    """n-gram + tên hàm của HumanEval (164) và MBPP (đủ 4 phần) — mọi bài dùng để đo."""
    from datasets import load_dataset
    grams, names = set(), set()
    for r in load_dataset("openai/openai_humaneval", split="test"):
        grams |= ngrams(r["prompt"]); names.add(r["entry_point"])
    for cfg in ("full", "sanitized"):
        for sp in ("train", "validation", "test", "prompt"):
            try:
                ds = load_dataset("google-research-datasets/mbpp", cfg, split=sp)
            except Exception:
                continue
            for r in ds:
                grams |= ngrams(r.get("text") or r.get("prompt") or "")
                m = re.search(r"def (\w+)", r["code"])
                if m:
                    names.add(m.group(1))
    return grams, names


def code_of(out):
    m = re.search(r"```python\n(.*?)```", out, re.S)
    return m.group(1) if m else None


def judge_ok(j):
    try:
        return json.loads(j)["logical_correctness"]["score"] >= 5
    except Exception:
        return False


def check(row):
    code = code_of(row["output"])
    tests = json.loads(row["unit_tests"])
    prog = code + "\n\n" + "\n".join(t.strip() for t in tests) + "\n"
    r = CA.run_python(prog, timeout=10)
    return row, tests, r["rc"] == 0 and not r["timeout"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="0,1,2,3")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N bài (thử)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-chars", type=int, default=5000, help="bỏ lời giải dài hơn (ký tự)")
    a = ap.parse_args()
    import pandas as pd
    seen = set()
    if os.path.exists(SEEN):
        seen = {json.loads(l)["id"] for l in open(SEEN)}
    grams, names = eval_index()
    print(f"[c2] chỉ mục bộ đo: {len(grams)} n-gram · {len(names)} tên hàm · đã chấm trước {len(seen)}", flush=True)
    cand, stat = [], {"tong": 0, "diem<1": 0, "giam_khao": 0, "khong_code": 0, "qua_dai": 0, "giong_bo_do": 0, "da_cham": 0}
    for s in map(int, a.shards.split(",")):
        df = pd.read_parquet(RAW.format(s))
        for row in df.to_dict("records"):
            stat["tong"] += 1
            if row["average_test_score"] < 1.0:
                stat["diem<1"] += 1; continue
            if not judge_ok(row["llm_judgement"]):
                stat["giam_khao"] += 1; continue
            code = code_of(row["output"])
            if not code:
                stat["khong_code"] += 1; continue
            if len(row["output"]) > a.max_chars:
                stat["qua_dai"] += 1; continue
            fn = re.findall(r"def (\w+)", code)
            if (ngrams(row["input"]) & grams) or any(f in names and len(f) > 6 for f in fn[:1]):
                stat["giong_bo_do"] += 1; continue
            if row["id"] in seen:
                stat["da_cham"] += 1; continue
            cand.append(row)
    if a.limit:
        cand = cand[:a.limit]
    print(f"[c2] lọc: {stat} → chạy {len(cand)} bài", flush=True)
    ok = n = 0; T0 = time.time()
    with open(OUT, "a") as fo, open(SEEN, "a") as fs, ThreadPoolExecutor(a.workers) as ex:
        for row, tests, passed in ex.map(check, cand):
            n += 1; ok += passed
            fs.write(json.dumps({"id": row["id"], "pass": passed}) + "\n")
            if passed:
                fo.write(json.dumps({"id": row["id"], "src": "OpenCodeInstruct (CC-BY-4.0)", "domain": row["domain"],
                                     "messages": [{"role": "user", "content": row["input"]},
                                                  {"role": "assistant", "content": row["output"]}],
                                     "tests": tests}, ensure_ascii=False) + "\n")
            if n % 2000 == 0:
                fo.flush(); fs.flush()
                print(f"[c2] {n}/{len(cand)} · qua {ok} ({ok/n:.1%}) · {time.time()-T0:.0f}s", flush=True)
    print(f"### c2 xong: chạy {n} · qua {ok} ({ok/max(n,1):.1%}) · {time.time()-T0:.0f}s → {OUT}", flush=True)


if __name__ == "__main__":
    main()
