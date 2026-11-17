"""TK_108 C2b+ — code HIỆN ĐẠI: Next.js/React, Tailwind, Svelte 5, Kotlin + Jetpack Compose, Flutter/Dart, shadcn/ui
+ glaive bù TypeScript / Go / Rust / Swift / Kotlin / Dart / C++ / Ruby. Cùng chuẩn với C2b phần A: mọi khối code phải
qua kiểm cú pháp tree-sitter (Kotlin Compose đã được tác giả BIÊN DỊCH thật; Flutter diff-steps kiểm bằng áp lại bản vá).

  nextjs    Tesslate/Next.js-Dataset                          Apache-2.0
  websight  HuggingFaceM4/WebSight v0.2 (chỉ HTML, qua API)   CC-BY-4.0
  tailwind  kiiwee/tailwind_component_llama3_dataset          Apache-2.0
  svelte    Mungus451/svelte-5-expert-sft                     MIT
  kotlin    giggiovpg/android-kotlin-compose-compiler-verified MIT (compile_status = verified)
  flutter   bbidpa/flutter-full-examples-v1 + flutter-diff-steps-v1  Apache-2.0
  fecook    Tensoic/FrontendCookbook                          Apache-2.0 (sinh bằng Bard → chỉ giữ mẫu qua kiểm)
  glaive    glaiveai/glaive-code-assistant-v3                 Apache-2.0
  shadcn    valentin-marquez/react-shadcn-codex               MIT (code chép từ GitHub — user đồng ý dùng)

  python scripts/c2b_plus.py --raw /workspace/c2b_raw/plus --workers 24 → data/c2b/plus_<nhóm>.jsonl
"""
from __future__ import annotations
import argparse, ast, collections, glob, hashlib, json, os, re, time, urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import c2b_multilang as M

OUT = M.OUT
CAP = {"nextjs": 40000, "websight": 40000, "tailwind": 5000, "svelte": 5000, "kotlin": 10000, "flutter": 30000,
       "fecook": 20000, "shadcn": 5000}
GLAIVE = {"typescript": 15000, "go": 10000, "rust": 15000, "swift": 10000, "kotlin": 10000, "dart": 5000, "cpp": 15000,
          "ruby": 8000}


def read_all(d, skip=()):
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(f"{d}/**/*", recursive=True)):
        if any(s in f for s in skip) or os.path.isdir(f):
            continue
        if f.endswith(".parquet"):
            for b in pq.ParquetFile(f).iter_batches(batch_size=2048):
                yield from b.to_pylist()
        elif f.endswith(".jsonl"):
            yield from (json.loads(l) for l in open(f, encoding="utf-8"))
        elif f.endswith(".json") and "dataset_info" not in f:
            try:
                x = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            yield from (x if isinstance(x, list) else [])


def U(q, a):
    return [{"role": "user", "content": q.strip()}, {"role": "assistant", "content": a.strip()}]


def strip_think(t):
    return re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()


# ── nguồn → (nhóm, id, messages, kiểu kiểm) ; kiểu kiểm: "judge" | "trust" (đã biên dịch / đã kiểm bản vá) ──
def src_nextjs(R):
    for r in read_all(f"{R}/nextjs"):
        yield "nextjs", "nx_" + hashlib.md5(r["question"].encode()).hexdigest()[:16], U(r["question"], r["response"]), "judge"


def src_tailwind(R):
    for r in read_all(f"{R}/tailwind"):
        m = re.search(r"user<\|end_header_id\|>(.*?)<\|eot_id\|>.*?assistant<\|end_header_id\|>(.*?)(?:<\|eot_id\|>|$)", r["text"], re.S)
        if m:
            yield "tailwind", "tw_" + hashlib.md5(m.group(1).encode()).hexdigest()[:16], \
                U(f"Tạo component bằng HTML + Tailwind CSS theo mô tả:\n{m.group(1).strip()}", f"```html\n{m.group(2).strip()}\n```"), "judge"


def src_svelte(R):
    for r in read_all(f"{R}/svelte"):
        ms = [m for m in r["messages"] if m["role"] != "system"]
        if len(ms) >= 2:
            yield "svelte", "sv_" + hashlib.md5(ms[0]["content"].encode()).hexdigest()[:16], ms, "judge"


def src_kotlin(R):
    for r in read_all(f"{R}/kotlin", skip=("dpo", "validation", "eval", "test")):
        if r.get("compile_status") != "verified" or not r.get("code"):
            continue
        why = strip_think(r.get("output") or "")
        why = re.sub(r"```.*?```", "", why, flags=re.S).strip()[:1500]
        yield "kotlin", "kt_" + hashlib.md5(r["instruction"].encode()).hexdigest()[:16], \
            U(r["instruction"], (why + "\n\n" if why else "") + f"```kotlin\n{r['code'].strip()}\n```"), "trust"


def src_flutter(R):
    for r in read_all(f"{R}/flutter_full"):
        q = r["goal"] + (f"\n\nCode hiện tại:\n```dart\n{r['initial_code'].strip()}\n```" if (r.get("initial_code") or "").strip() else "")
        yield "flutter", r["id"], U(q, f"```dart\n{r['final_code'].strip()}\n```"), "judge"
    for r in read_all(f"{R}/flutter_diff"):
        try:                                            # kiểm: áp lại bản vá search/replace lên code cũ phải ra đúng code mới
            ch = ast.literal_eval(r["changes"]) if isinstance(r["changes"], str) else r["changes"]
            code = r["backward_code"]
            for c in ch:
                if c["search"] not in code:
                    raise ValueError
                code = code.replace(c["search"], c["replace"], 1)
            if code.strip() != r["forward_code"].strip():
                continue
        except Exception:
            continue
        q = (f"Mục tiêu: {r['goal']}\n\nFile Dart hiện tại:\n```dart\n{r['backward_code'].strip()}\n```\n\n"
             f"Bước tiếp theo: {r['forward_action']}\nTrả về toàn bộ file sau khi sửa.")
        yield "flutter", r["id"], U(q, f"```dart\n{r['forward_code'].strip()}\n```"), "judge"


def src_fecook(R):
    for r in read_all(f"{R}/fecook"):
        yield "fecook", "fe_" + hashlib.md5(r["question"].encode()).hexdigest()[:16], U(r["question"], r["answer"]), "judge"


def src_shadcn(R):
    for r in read_all(f"{R}/shadcn"):
        if not r.get("prompt") or not r.get("content"):
            continue
        yield "shadcn", "sh_" + hashlib.md5(r["content"].encode()).hexdigest()[:16], \
            U(r["prompt"], f"```tsx\n{r['content'].strip()}\n```"), "judge"


def src_glaive(R):
    for r in read_all(f"{R}/glaive"):
        yield "glaive", "gl_" + hashlib.md5(r["question"].encode()).hexdigest()[:16], U(r["question"], r["answer"]), "judge"


def src_websight(R):
    f = f"{R}/websight.jsonl"
    if not os.path.exists(f):
        return
    for r in (json.loads(l) for l in open(f, encoding="utf-8")):
        yield "websight", r["id"], U(f"Tạo trang web bằng HTML + Tailwind CSS theo ý tưởng:\n{r['idea']}",
                                    f"```html\n{r['html'].strip()}\n```"), "judge"


def fetch_websight(R, n):
    """chỉ lấy cột HTML + ý tưởng qua datasets-server (không tải ảnh) → websight.jsonl"""
    f = f"{R}/websight.jsonl"
    if os.path.exists(f) and sum(1 for _ in open(f)) >= n:
        return

    def page(off):
        u = f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceM4/WebSight&config=v0.2&split=train&offset={off}&length=100"
        for t in range(5):
            try:
                with urllib.request.urlopen(u, timeout=60) as h:
                    return [(off + i, x["row"]) for i, x in enumerate(json.load(h)["rows"])]
            except Exception:
                time.sleep(5 * (t + 1))
        return []
    offs = [i * 100 for i in range(0, 2_700_000 // 100, (2_700_000 // 100) // (n // 100))][: n // 100]  # rải đều cả bộ
    with open(f, "w", encoding="utf-8") as fo, ThreadPoolExecutor(8) as ex:
        for rows in ex.map(page, offs):
            for i, r in rows:
                if r.get("text") and r.get("llm_generated_idea"):
                    fo.write(json.dumps({"id": f"ws_{i}", "idea": r["llm_generated_idea"], "html": r["text"]}, ensure_ascii=False) + "\n")


def fetch_websight_shards(R, k):
    """bổ sung: tải thẳng k phần parquet (rải đều 738 phần), CHỈ đọc cột HTML + ý tưởng rồi xoá ngay (mỗi phần ~0,42 GB có ảnh)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    files = sorted(f for f in HfApi().list_repo_files("HuggingFaceM4/WebSight", repo_type="dataset")
                   if f.startswith("v0.2/") and f.endswith(".parquet"))
    pick = files[::max(1, len(files) // k)][:k]
    tmp = f"{R}/_ws_tmp"
    with open(f"{R}/websight.jsonl", "a", encoding="utf-8") as fo:
        for f in pick:
            p = hf_hub_download("HuggingFaceM4/WebSight", f, repo_type="dataset", local_dir=tmp)
            t = pq.read_table(p, columns=["text", "llm_generated_idea"]).to_pylist()
            base = files.index(f) * 10_000_000
            for i, r in enumerate(t):
                if r["text"] and r["llm_generated_idea"]:
                    fo.write(json.dumps({"id": f"ws_{base + i}", "idea": r["llm_generated_idea"], "html": r["text"]},
                                        ensure_ascii=False) + "\n")
            os.remove(p)
            print(f"[c2b+] websight {f}: +{len(t)}", flush=True)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def check(item):
    grp, rid, ms, how = item
    ans = "\n".join(m["content"] for m in ms if m["role"] == "assistant")
    if how == "trust":
        return grp, rid, ms, "kotlin", True
    j = M.judge(ans)
    if j is None:
        return None
    return grp, rid, ms, j[0], j[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/workspace/c2b_raw/plus")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--websight", type=int, default=40000)
    ap.add_argument("--websight-shards", type=int, default=0, help="bổ sung WebSight bằng k phần parquet")
    ap.add_argument("--only", default="", help="chỉ làm lại các nhóm này (vd websight,svelte), giữ nguyên file nhóm khác")
    ap.add_argument("--per-q", type=int, default=3, help="svelte: số câu trả lời khác nhau tối đa cho mỗi câu hỏi")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    if a.websight_shards:
        fetch_websight_shards(a.raw, a.websight_shards)
    else:
        fetch_websight(a.raw, a.websight)
    he = {l.strip() for l in open(f"{OUT}/humaneval_names.txt") if len(l.strip()) >= 10} \
        if os.path.exists(f"{OUT}/humaneval_names.txt") else set()
    n, bad, seen = collections.Counter(), collections.Counter(), set()
    fout = {}

    def out(name):
        if name not in fout:
            fout[name] = open(f"{OUT}/plus_{name}.jsonl", "w", encoding="utf-8")
        return fout[name]

    def gen():
        perq = collections.Counter()
        for s in (src_kotlin, src_svelte, src_tailwind, src_shadcn, src_nextjs, src_websight, src_flutter, src_fecook, src_glaive):
            if only and s.__name__[4:] not in only:
                continue
            for x in s(a.raw):
                q = x[2][0]["content"]
                if x[0] == "shadcn" and q.startswith("Request the implementation"):     # đề máy sinh, không nói cần làm gì
                    continue
                h = hashlib.md5(re.sub(r"\s+", " ", q.lower()).encode()).hexdigest()
                if x[0] == "svelte":            # bộ Svelte: ít câu hỏi, nhiều câu trả lời khác nhau → giữ tối đa per_q
                    ha = hashlib.md5(x[2][1]["content"].encode()).hexdigest()
                    if perq[h] >= a.per_q or (h + ha) in seen:
                        continue
                    perq[h] += 1; seen.add(h + ha)
                    yield x
                    continue
                if h in seen or len(q) > 8000 or sum(len(m["content"]) for m in x[2]) > 16000:
                    continue
                seen.add(h)
                yield x

    with ProcessPoolExecutor(a.workers) as ex:
        it = gen()
        while True:
            b = [x for _, x in zip(range(4096), it)]
            if not b:
                break
            for r in ex.map(check, b, chunksize=64):
                if r is None:
                    continue
                grp, rid, ms, lang, ok = r
                key = f"glaive_{lang}" if grp == "glaive" else grp
                if not ok:
                    bad[key] += 1; continue
                if grp == "glaive":
                    if lang not in GLAIVE or n[key] >= GLAIVE[lang]:
                        continue
                elif n[key] >= CAP[grp]:
                    continue
                if he and any(x in m["content"] for m in ms for x in he):
                    bad["_humaneval"] += 1; continue
                out(key).write(json.dumps({"id": rid, "lang": lang, "src": grp, "messages": ms}, ensure_ascii=False) + "\n")
                n[key] += 1
            print(f"[c2b+] {dict(n)}", flush=True)
    for f in fout.values():
        f.close()
    json.dump({"giu": dict(n), "loai": dict(bad)}, open(f"{OUT}/plus_stats{'_' + '_'.join(sorted(only)) if only else ''}.json", "w"), indent=1, ensure_ascii=False)
    print(f"### c2b+: {sum(n.values())} mẫu · " + " · ".join(f"{k} {v}" for k, v in sorted(n.items())) +
          f" · loại {sum(v for k, v in bad.items() if k != '_humaneval')}", flush=True)


if __name__ == "__main__":
    main()
