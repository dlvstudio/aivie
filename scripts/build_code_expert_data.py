"""TK_108 pilot EXPERT CODE LỚN — dữ liệu PRETRAIN cho riêng expert (thân đóng băng), nguồn THÂN CHƯA THẤY:
  opc_algo   OpenCoder opc-annealing-corpus / algorithmic_corpus     (ODC-BY) code thuật toán (từ The Stack v2) + đề
  opc_snip   …/ synthetic_code_snippet                               đoạn code chất lượng cao viết lại từ algorithmic
  opc_files  …/ synthetic_qa (thực chất là file code thật kèm tên file)
  web        OpenCoder-LLM/opc-fineweb-code-corpus (MIT) trang web về lập trình (hướng dẫn, tài liệu) — vài file đầu
  vi         1 shard tiếng Việt đã mã hoá sẵn (donaldluong/nn-aivie-cpt-data/vi_web) — ôn ngôn ngữ chống trôi
Lọc: code xấu (dòng > 400 ký tự, quá ngắn/dài) · bỏ văn bản chứa tên hàm HumanEval (chống rò) · khử trùng chính xác.
Ra: data/cx/<nhóm>/shard_*.npy (uint16, bpe_v1, mỗi văn bản kết thúc EOS) + data/cx/val/<nhóm>.npy (2 triệu token giữ riêng).
    python scripts/build_code_expert_data.py --budget opc_algo:100e6,opc_snip:100e6,opc_files:100e6,web:60e6 --workers 30
CHẠY THẬT (--profile cx3 → data/cx3, ~3,4 tỷ token): toàn bộ OpenCoder annealing theo hạn mức + thêm nhóm
  gh   codeparrot/github-code-clean — CHỈ file giấy phép mở (MIT/Apache-2.0/BSD/ISC/Unlicense/CC0), chia hạn mức theo
       ngôn ngữ (đẩy TS/C#/Go/SQL lên vì MultiPL-E pilot yếu nhất ở đó) — code thật nhiều ngôn ngữ, OpenCoder thiên Python
  vi   5 shard vi_fineweb2 (~0,5 tỷ, 15%)
Tải file NGAY TRƯỚC khi đọc và xoá bản thô sau khi đọc (ổ box nhỏ); file pilot xếp đầu → tập giữ riêng trùng pilot (so được).
    python scripts/build_code_expert_data.py --profile cx3 --workers 40
"""
from __future__ import annotations
import argparse, collections, glob, hashlib, json, os, re, sys, time
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
import build_cpt_data as B

RAW = "/workspace/cx_raw"
OUT = "data/cx"
SRC = {   # nhóm → (repo, các file — vừa đủ hạn mức pilot, kiểu lọc)
    "opc_algo": ("OpenCoder-LLM/opc-annealing-corpus", [f"algorithmic_corpus/{i}.jsonl" for i in (1, 12, 23, 34, 45, 50)], "code"),
    "opc_snip": ("OpenCoder-LLM/opc-annealing-corpus", [f"synthetic_code_snippet/{i}.jsonl" for i in (1, 11, 21)], "code"),
    "opc_files": ("OpenCoder-LLM/opc-annealing-corpus", [f"synthetic_qa/{i}.jsonl" for i in (1, 41, 81)], "code"),
    "web": ("OpenCoder-LLM/opc-fineweb-code-corpus", [f"data/train-{i:05d}-of-00510.parquet" for i in (0, 170, 340)], "plain"),
}
VAL = 2_000_000
PERMISSIVE = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "unlicense", "cc0-1.0"}
GH_LANG = {"Java": .11, "JavaScript": .11, "TypeScript": .11, "Python": .09, "C#": .10, "GO": .09, "PHP": .09, "HTML": .06,
           "CSS": .02, "SQL": .04, "Shell": .04, "C++": .05, "C": .04, "Rust": .03, "Ruby": .02}
PROFILES = {"pilot": ("data/cx", "opc_algo:100e6,opc_snip:100e6,opc_files:100e6,web:60e6"),
            "cx3": ("data/cx3", "opc_algo:0.9e9,opc_snip:0.6e9,opc_files:0.5e9,gh:1.0e9,web:0.3e9")}


def cx3_src():
    """file pilot đứng đầu (giữ đúng tập val), rồi đến phần còn lại."""
    def order(first, rest):
        return first + [x for x in rest if x not in first]
    return {
        "opc_algo": ("OpenCoder-LLM/opc-annealing-corpus", order(SRC["opc_algo"][1], [f"algorithmic_corpus/{i}.jsonl" for i in range(54)]), "code"),
        "opc_snip": ("OpenCoder-LLM/opc-annealing-corpus", order(SRC["opc_snip"][1], [f"synthetic_code_snippet/{i}.jsonl" for i in range(31)]), "code"),
        "opc_files": ("OpenCoder-LLM/opc-annealing-corpus", order(SRC["opc_files"][1], [f"synthetic_qa/{i}.jsonl" for i in range(106)]), "code"),
        "web": ("OpenCoder-LLM/opc-fineweb-code-corpus", order(SRC["web"][1], [f"data/train-{i:05d}-of-00510.parquet" for i in range(0, 510, 7)]), "plain"),
        "gh": ("codeparrot/github-code-clean", [f"data/train-{i:05d}-of-00880.parquet" for i in range(0, 880, 5)][:150], "code"),
    }


def download():
    from huggingface_hub import hf_hub_download
    for g, (repo, files, _) in SRC.items():
        for f in files:
            hf_hub_download(repo, f, repo_type="dataset", local_dir=f"{RAW}/{g}")
        print(f"[cx] tải {g}: {len(files)} file", flush=True)
    os.makedirs(f"{OUT}/vi", exist_ok=True)
    p = hf_hub_download("donaldluong/nn-aivie-cpt-data", "vi_web/shard_vi_hplt_00014.npy", repo_type="dataset",
                        local_dir=f"{RAW}/vi")
    a = np.load(p, mmap_mode="r")
    np.save(f"{OUT}/vi/shard_vi_00000.npy", np.asarray(a[VAL:VAL + 60_000_000]))
    os.makedirs(f"{OUT}/val", exist_ok=True)
    np.save(f"{OUT}/val/vi.npy", np.asarray(a[:VAL]))
    print(f"[cx] vi: 60 triệu token ôn + {VAL} token giữ riêng", flush=True)


def download_vi(n):
    """cx3: n shard vi_fineweb2 (ôn tiếng Việt); 2 triệu token đầu shard đầu = tập giữ riêng."""
    from huggingface_hub import hf_hub_download
    os.makedirs(f"{OUT}/vi", exist_ok=True); os.makedirs(f"{OUT}/val", exist_ok=True)
    for i in range(n):
        dst = f"{OUT}/vi/shard_vi_{i:05d}.npy"
        if os.path.exists(dst):
            continue
        p = hf_hub_download("donaldluong/nn-aivie-cpt-data", f"vi_web/shard_vi_fineweb2_{i:05d}.npy", repo_type="dataset",
                            local_dir=f"{RAW}/vi")
        x = np.load(p, mmap_mode="r")
        if i == 0:
            np.save(f"{OUT}/val/vi.npy", np.asarray(x[:VAL])); x = x[VAL:]
        np.save(dst, np.asarray(x)); os.remove(p)
    print(f"[cx] vi: {n} shard", flush=True)


def texts(g, lazy=False):
    """(chữ, ngôn ngữ) — lazy: tải từng file ngay trước khi đọc, đọc xong xoá (dòng nào đã đọc thì không cần nữa)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    repo = SRC[g][0]
    for f in SRC[g][1]:
        p = f"{RAW}/{g}/{f}"
        if lazy and not os.path.exists(p):
            try:
                hf_hub_download(repo, f, repo_type="dataset", local_dir=f"{RAW}/{g}")
            except Exception as e:
                print(f"[cx] bỏ {g}/{f}: {e}", flush=True); continue
        if g == "gh":
            for b in pq.ParquetFile(p).iter_batches(batch_size=2048, columns=["code", "language", "license"]):
                for t, lg, li in zip(b.column("code").to_pylist(), b.column("language").to_pylist(), b.column("license").to_pylist()):
                    if li in PERMISSIVE and lg in GH_LANG:
                        yield t, lg
        elif p.endswith(".parquet"):
            for b in pq.ParquetFile(p).iter_batches(batch_size=2048, columns=["text"]):
                for t in b.column("text").to_pylist():
                    yield t, ""
        else:
            for l in open(p, encoding="utf-8"):
                try:
                    r = json.loads(l)
                    yield r.get("text") or "", r.get("lang", "")
                except Exception:
                    continue
        if lazy:
            os.remove(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="pilot", choices=list(PROFILES))
    ap.add_argument("--budget", default=None)
    ap.add_argument("--vi-shards", type=int, default=5, help="cx3: số shard vi_fineweb2 (~100 triệu token mỗi shard)")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    global OUT
    OUT, dflt = PROFILES[a.profile]
    a.budget = a.budget or dflt
    lazy = a.profile != "pilot"
    if lazy:
        SRC.update(cx3_src())
        download_vi(a.vi_shards)
    elif not a.skip_download:
        download()
    budget = {k: float(v) for k, v in (x.split(":") for x in a.budget.split(","))}
    he = [l.strip() for l in open("data/c2b/humaneval_names.txt") if len(l.strip()) >= 8] \
        if os.path.exists("data/c2b/humaneval_names.txt") else []
    he_re = re.compile(r"\b(" + "|".join(map(re.escape, he)) + r")\b") if he else None
    pool = mp.get_context("spawn").Pool(a.workers, initializer=B.init, initargs=("checkpoints/bpe_v1.json", "", 0, 0, 0))
    seen, man = set(), {}
    for g, cap in budget.items():
        kind = SRC[g][2]
        w = B.Writer(f"{OUT}/{g}", 50_000_000, g)
        val, n_val, stat = [], 0, {"giữ": 0, "loại": 0, "trùng": 0, "rò HumanEval": 0}
        T0 = time.time()
        caps = {lg: f * cap for lg, f in GH_LANG.items()} if g == "gh" else None
        used, langs = collections.Counter(), collections.deque()

        def items():
            for t, lg in texts(g, lazy):
                if not t:
                    continue
                if caps is not None and used[lg] >= caps[lg]:
                    continue
                h = hashlib.blake2b(re.sub(r"\s+", " ", t[:2000]).encode(), digest_size=8).digest()
                if h in seen:
                    stat["trùng"] += 1; continue
                seen.add(h)
                if he_re and he_re.search(t):
                    stat["rò HumanEval"] += 1; continue
                langs.append(lg)
                yield (t, kind)
        for r in pool.imap(B.work, items(), chunksize=64):
            lg = langs.popleft()
            if r[0] != "ok":
                stat["loại"] += 1; continue
            ids = r[1]
            if n_val < VAL:
                val.append(ids); n_val += len(ids); continue
            if caps is not None and used[lg] >= caps[lg]:
                continue
            w.add(ids); stat["giữ"] += 1; used[lg] += len(ids)
            if w.total >= cap or (caps is not None and all(used[k] >= v for k, v in caps.items())):
                break
        w.flush()
        os.makedirs(f"{OUT}/val", exist_ok=True)
        np.save(f"{OUT}/val/{g}.npy", np.concatenate(val))
        man[g] = {"token": int(w.total), **stat, "giây": round(time.time() - T0)}
        if used and any(used.values()):
            man[g]["theo ngôn ngữ (triệu)"] = {k: round(v / 1e6, 1) for k, v in used.most_common()}
        print(f"[cx] {g}: {w.total/1e6:.0f} triệu token · {stat} · {time.time()-T0:.0f}s", flush=True)
    pool.close()
    man["vi"] = {"token": sum(len(np.load(f, mmap_mode="r")) for f in glob.glob(f"{OUT}/vi/*.npy"))}
    json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1, ensure_ascii=False)
    print("### cx: " + " · ".join(f"{g} {v['token']/1e6:.0f}M" for g, v in man.items()), flush=True)


if __name__ == "__main__":
    main()
