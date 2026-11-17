"""§TK_081 — PHA 0: EVAL HARNESS cho mạng Hebbian + causal LM.

Mục tiêu: THAY "chấm bằng mắt" bằng số đo lặp lại được, để mọi cải tiến Pha 1-4
đều so được với baseline. Gọi /api/gen (server đã warm, d512mix) trên bộ prompt
cố định × các biến thể decoding, rồi tính metric TỰ ĐỘNG trên phần SINH (tail).

Metric (chỉ cần text, không cần tham chiếu):
  distinct1/2  : đa dạng uni/bi-gram (thấp = lặp)
  rep4         : tỉ lệ 4-gram lặp lại (cao = lặp thoái hóa)
  max_run      : chuỗi token GIỐNG nhau dài nhất liên tiếp
  drift        : Jaccard từ-nội-dung giữa 1/3 đầu và 1/3 cuối (thấp = trôi chủ đề)
  anchor       : tỉ lệ cửa sổ còn dính ≥1 từ-nội-dung của prompt (thấp = mất neo)
  frag         : tỉ lệ token ngắn ≤2 ký tự (proxy rác/mảnh từ-điển)

    .venv/bin/python scripts/eval_hebbian.py 2>&1 | tee out/eval_hebbian_baseline.log
"""
import sys, json, time, urllib.request, urllib.parse
from collections import Counter

BASE = "http://127.0.0.1:8765/api/gen"

FUNC = {"là","và","của","một","các","những","đã","đang","sẽ","có","ở","cho","với",
        "khi","thì","mà","ra","vào","nên","do","bị","được","này","đó","hay","hoặc",
        "từ","đến","trong","trên","dưới","cũng","rất","để","vì","nếu","tại","theo","như"}

PROMPTS = [
    "Chó là loài động vật",
    "Hà Nội là thành phố",
    "Mặt Trời là",
    "Trí tuệ nhân tạo là",
    "Sông Mê Kông là con sông",
    "Việt Nam là một quốc gia",
    "Vào một buổi sáng mùa thu",
    "Cuộc cách mạng công nghiệp",
    "Trong cơ thể người, tế bào",
    "Biến đổi khí hậu là",
]

# (nhãn, tham số /api/gen)
VARIANTS = {
    "counting": {"knn": 0, "lm_led": 0, "hybrid": 0, "extend": 0},        # đếm W_skip DẪN (ref)
    "lm_t04":   {"knn": 1, "lm_led": 1, "ground_a": 0.5, "lm_temp": 0.4},  # §TK_085 quét temp
    "lm_t05":   {"knn": 1, "lm_led": 1, "ground_a": 0.5, "lm_temp": 0.5},
    "lm_t06":   {"knn": 1, "lm_led": 1, "ground_a": 0.5, "lm_temp": 0.6},
}
MAX_LEN = 60


def call(prompt, params):
    q = {"prompt": prompt, "max_len": MAX_LEN, **params}
    url = BASE + "?" + urllib.parse.urlencode(q)
    for _ in range(2):
        try:
            r = urllib.request.urlopen(url, timeout=120)
            return json.load(r)
        except Exception as e:
            err = str(e); time.sleep(1)
    return {"error": err}


def content(tokens):
    return [t for t in tokens if len(t) >= 3 and t not in FUNC]


def metrics(tail, prompt):
    toks = tail.split()
    n = len(toks)
    if n == 0:
        return {"n_tok": 0}
    d1 = len(set(toks)) / n
    bigrams = list(zip(toks, toks[1:]))
    d2 = len(set(bigrams)) / max(len(bigrams), 1)
    g4 = list(zip(toks, toks[1:], toks[2:], toks[3:]))
    rep4 = 0.0 if not g4 else 1 - len(set(g4)) / len(g4)
    # max run token giống nhau liên tiếp
    run = mx = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        mx = max(mx, run)
    # drift: Jaccard content 1/3 đầu vs 1/3 cuối
    third = max(n // 3, 1)
    c1, c3 = set(content(toks[:third])), set(content(toks[-third:]))
    drift = len(c1 & c3) / len(c1 | c3) if (c1 | c3) else 0.0
    # anchor: cửa sổ (3 phần) còn dính từ-nội-dung prompt
    pset = set(content(prompt.split()))
    wins = [toks[:third], toks[third:2*third], toks[-third:]]
    anchor = sum(1 for w in wins if pset & set(content(w))) / 3 if pset else 0.0
    frag = sum(1 for t in toks if len(t) <= 2 and t not in FUNC) / n
    return {"n_tok": n, "distinct1": round(d1,3), "distinct2": round(d2,3),
            "rep4": round(rep4,3), "max_run": mx, "drift": round(drift,3),
            "anchor": round(anchor,3), "frag": round(frag,3)}


def main():
    rows = []
    for p in PROMPTS:
        for var, params in VARIANTS.items():
            r = call(p, params)
            if "error" in r:
                print(f"[LỖI] {var} «{p}»: {r['error']}", flush=True); continue
            tail = r.get("tail", "")
            grounded = r.get("grounded", False)
            m = metrics(tail, p)
            m.update({"prompt": p, "variant": var, "grounded": grounded,
                      "ms": r.get("meta",{}).get("elapsed_ms"), "tail": tail})
            rows.append(m)
            flag = " [GROUNDED-lookup]" if grounded else ""
            print(f"\n▶ [{var}]{flag} «{p}»  ({m.get('n_tok')}tok, {m.get('ms')}ms)", flush=True)
            print(f"   d1={m.get('distinct1')} d2={m.get('distinct2')} rep4={m.get('rep4')} "
                  f"run={m.get('max_run')} drift={m.get('drift')} anchor={m.get('anchor')} frag={m.get('frag')}", flush=True)
            print(f"   ⟶ {tail[:150]}", flush=True)

    # tổng hợp theo biến thể (chỉ tính mẫu SINH, bỏ grounded-lookup)
    print(f"\n{'='*72}\nTRUNG BÌNH theo biến thể (chỉ mẫu sinh, bỏ grounded-lookup)\n{'='*72}", flush=True)
    keys = ["distinct1","distinct2","rep4","max_run","drift","anchor","frag","n_tok"]
    print(f"{'variant':8} " + " ".join(f"{k:>9}" for k in keys), flush=True)
    agg = {}
    for var in VARIANTS:
        sub = [r for r in rows if r["variant"]==var and not r["grounded"] and r.get("n_tok")]
        if not sub: continue
        avg = {k: round(sum(r[k] for r in sub)/len(sub), 3) for k in keys}
        agg[var] = {"n_samples": len(sub), **avg}
        print(f"{var:8} " + " ".join(f"{avg[k]:>9}" for k in keys), flush=True)

    json.dump({"rows": rows, "agg": agg, "max_len": MAX_LEN, "checkpoint": "d512mix"},
              open("out/eval_hebbian_baseline.json","w"), ensure_ascii=False, indent=1)
    print(f"\n[LƯU] out/eval_hebbian_baseline.json ({len(rows)} mẫu)", flush=True)
    print("### hết", flush=True)


if __name__ == "__main__":
    main()
