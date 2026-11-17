"""TK_108 C4 lần 3 — dữ liệu "SAI → LỖI → SỬA" từ nhật ký thầy (C2b B + C2 debug v2) → data/c2b/mix_fix.jsonl (khoá c2bF).

Chỉ lấy bài thầy SAI ở một lượt rồi SỬA ĐƯỢC ở lượt sau (đã chạy thật). Hội thoại:
  người dùng: đề gốc
  trợ lý   : lời giải SAI của thầy                 ← nolearn (model thấy, KHÔNG học)
  người dùng: lỗi thật khi chạy (traceback / lỗi SQL Server / kết quả lệch / lỗi giao diện)
  trợ lý   : lời giải đã sửa — QUA                 ← học
Bỏ: lượt hỏng vì thầy trả sai khuôn (thiếu file) hoặc vì test quá yếu (lỗi của bài kiểm, không phải của lời giải).
SQL (SQLite) không có: thầy giải độc lập, không được báo lỗi.
    python scripts/c2b_fixmix.py
"""
import collections, glob, json, os, random, re

D = "data/c2b"
FENCE = {"javascript": "javascript", "typescript": "typescript", "java": "java", "csharp": "csharp", "php": "php",
         "go": "go", "bash": "bash"}
SOL = {"javascript": "solution.js", "typescript": "solution.ts", "java": "Solution.java", "csharp": "Solution.cs",
       "php": "solution.php", "go": "solution.go", "bash": "solution.sh"}
ASK = ["Mình chạy thử thì bị lỗi:", "Chạy test vẫn lỗi, bạn xem lại giúp:", "Code này chạy ra lỗi như sau:",
       "Mình chạy thì báo thế này, sửa giúp mình nhé:"]


def logs(path):
    by = collections.defaultdict(list)
    for l in open(path, encoding="utf-8"):
        try:
            r = json.loads(l)
        except Exception:                                  # dòng cuối có thể đang ghi dở
            continue
        by[r.get("id")].append(r)
    return by


def verified(lang):
    p = f"{D}/verified_{lang}.jsonl"
    out = {}
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if not r["id"].endswith("_vi"):
                out[r["id"].rsplit("_", 1)[0]] = r          # id gốc → mẫu đã qua
    return out


def tail(s, n=1500):
    s = (s or "").strip()
    return s if len(s) <= n else "…\n" + s[-n:]


def chain(q, wrong, err, fix, rng):
    return [{"role": "user", "content": q}, {"role": "assistant", "content": wrong, "nolearn": True},
            {"role": "user", "content": f"{rng.choice(ASK)}\n```\n{tail(err)}\n```"}, {"role": "assistant", "content": fix}]


def code_lang(lang, rng):
    V, out = verified(lang), []
    for rid, evs in logs(f"{D}/teacher_log/{lang}.jsonl").items():
        if rid not in V:
            continue
        runs = sorted([e for e in evs if e.get("ev") == "run"], key=lambda e: e["step"])
        k = next((i for i, e in enumerate(runs) if e.get("ok")), None)
        if not k:                                          # qua ngay lượt đầu (k = 0) hoặc không qua
            continue
        bad = runs[k - 1]
        if not bad.get("files") or bad.get("weak_test") or SOL[lang] not in bad["files"]:
            continue
        v = V[rid]
        wrong = f"```{FENCE[lang]}\n{bad['files'][SOL[lang]].rstrip()}\n```"
        err = f"$ {bad.get('cmd', '')}\n{bad.get('out', '')}" + ("\n[HẾT GIỜ]" if bad.get("timeout") else "")
        out.append({"id": rid + "_fix", "lang": lang, "messages": chain(v["messages"][0]["content"], wrong, err,
                                                                        v["messages"][1]["content"], rng)})
    return out


def tsql(rng):
    V, out = verified("tsql"), []
    for rid, evs in logs(f"{D}/teacher_log/tsql.jsonl").items():
        if rid not in V:
            continue
        runs = sorted([e for e in evs if e.get("ev") == "run"], key=lambda e: e["step"])
        k = next((i for i, e in enumerate(runs) if e.get("ok")), None)
        if not k:
            continue
        bad = runs[k - 1]
        if not bad.get("files"):
            continue
        got = bad.get("got")
        err = got if isinstance(got, str) else f"Kết quả không đúng. Cần:\n{bad.get('want', [])[:8]}\nNhưng ra:\n{(got or [])[:8]}"
        v = V[rid]
        out.append({"id": rid + "_fix", "lang": "tsql", "messages": chain(
            v["messages"][0]["content"], f"```sql\n{bad['files']['query.sql'].strip()}\n```", err, v["messages"][1]["content"], rng)})
    return out


def html(rng):
    V, out = verified("html_css"), []
    for rid, evs in logs(f"{D}/teacher_log/html_css.jsonl").items():
        if rid not in V:
            continue
        runs = sorted([e for e in evs if e.get("ev") == "run"], key=lambda e: e["step"])
        k = next((i for i, e in enumerate(runs) if e.get("ok")), None)
        if not k:
            continue
        bad = runs[k - 1]
        why = list(bad.get("errors") or []) + [f"người xem góp ý: {x}" for x in ((bad.get("look") or {}).get("loi") or [])]
        if not why or not bad.get("html"):
            continue
        v = V[rid]
        out.append({"id": rid + "_fix", "lang": "html_css", "messages": [
            {"role": "user", "content": v["messages"][0]["content"]},
            {"role": "assistant", "content": f"```html\n{bad['html'].rstrip()}\n```", "nolearn": True},
            {"role": "user", "content": "Mình mở trang thì thấy:\n" + "\n".join(f"- {x}" for x in why[:8]) + "\nSửa giúp mình."},
            {"role": "assistant", "content": v["messages"][1]["content"]}]})
    return out


def debug_v2(rng):
    """C2 debug: lượt 1 của thầy vẫn hỏng → thầy sửa lại (rounds = 2)."""
    p, s = "data/c2/teacher_log/debug_v2.jsonl", "data/c2/debug_v2.jsonl"
    if not (os.path.exists(p) and os.path.exists(s)):
        return []
    L, out = logs(p), []
    for l in open(s, encoding="utf-8"):
        r = json.loads(l)
        if r.get("rounds") != 2:
            continue
        evs = L.get(r["id"].removesuffix("_dbg"), [])
        t = [e for e in evs if e.get("ev") == "teacher"]
        f1 = next((e for e in evs if e.get("ev") == "run_fix"), None)
        if len(t) < 2 or not f1 or f1.get("ok"):
            continue
        out.append({"id": r["id"] + "_fix", "lang": "python", "messages": chain(
            r["messages"][0]["content"], t[0]["raw"].strip(), f1.get("out", ""), r["messages"][1]["content"], rng)})
    return out


CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")   # chữ Hán / Nhật / Hàn (lẫn từ OpenCoder)


def sql_pairs():
    """SQL (SQLite): thầy giải độc lập — lượt đầu ra kết quả SAI, lượt sau ĐÚNG → một cặp DPO (không có lỗi để nói lại)."""
    V, out = verified("sql"), []
    p = f"{D}/teacher_log/sql.jsonl"
    if not os.path.exists(p):
        return out
    for rid, evs in logs(p).items():
        if rid not in V:
            continue
        runs = sorted([e for e in evs if e.get("ev") == "run"], key=lambda e: e["step"])
        k = next((i for i, e in enumerate(runs) if e.get("ok")), None)
        if not k or not runs[k - 1].get("query"):
            continue
        v = V[rid]
        out.append({"id": rid + "_dpo", "lang": "sql", "ctx": [{"role": "user", "content": v["messages"][0]["content"]}],
                    "chosen": v["messages"][1]["content"], "rejected": f"```sql\n{runs[k - 1]['query'].strip()}\n```"})
    return out


def main():
    rng = random.Random(508)
    allr, stat = [], {}
    for lg in FENCE:
        if os.path.exists(f"{D}/teacher_log/{lg}.jsonl"):
            x = code_lang(lg, rng); allr += x; stat[lg] = len(x)
    for name, fn in (("tsql", tsql), ("html_css", html), ("c2_debug", debug_v2)):
        try:
            x = fn(rng)
        except FileNotFoundError:
            x = []
        allr += x; stat[name] = len(x)
    n0 = len(allr)
    allr = [r for r in allr if len(CJK.findall(r["messages"][0]["content"])) < 3]
    stat["bỏ đề Hán/Nhật/Hàn"] = n0 - len(allr)
    rng.shuffle(allr)
    with open(f"{D}/mix_fix.jsonl", "w", encoding="utf-8") as f:
        for r in allr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"### mix_fix: {len(allr)} chuỗi sai → lỗi → sửa · " + " · ".join(f"{k} {v}" for k, v in stat.items()))
    # C5a — cặp DPO: cùng đề, lời giải ĐÚNG (đã chạy qua) ↔ lời giải SAI của thầy
    dpo = [{"id": r["id"].replace("_fix", "_dpo"), "lang": r["lang"], "ctx": [r["messages"][0]],
            "chosen": r["messages"][3]["content"], "rejected": r["messages"][1]["content"]} for r in allr]
    dpo += [x for x in sql_pairs() if len(CJK.findall(x["ctx"][0]["content"])) < 3]
    rng.shuffle(dpo)
    with open(f"{D}/dpo_code.jsonl", "w", encoding="utf-8") as f:
        for r in dpo:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"### dpo_code: {len(dpo)} cặp (đúng ↔ sai) · " + str(dict(collections.Counter(r["lang"] for r in dpo))))


if __name__ == "__main__":
    main()
