"""P0/P1 · DỰNG DỮ LIỆU TRAIN TIẾP (CPT) cho thân nuôi lớn 2,6B · ngữ cảnh 8K → shard uint16 theo NHÓM.

Mỗi nguồn: đọc luồng HF → chuẩn hoá (GIỮ xuống dòng — bản cũ gộp hết thành 1 dòng nên lọc "lặp dòng" vô tác dụng)
→ lọc heuristic (tiếng Việt) → chấm GIÁ TRỊ GIÁO DỤC (edu_classifier, chỉ web tiếng Việt; giữ ≥ --edu-min)
→ khử trùng lặp CHÍNH XÁC giữa các nguồn (băm 2 000 ký tự đầu) → tokenize (bpe_v1, song song)
→ văn bản ≥ --long-min token sang nhóm "long" (cho giai đoạn 8K) · còn lại vào nhóm của nguồn.
Dừng mỗi nhóm khi đủ hạn mức token (--budget).

    nhóm: vi_web (FineWeb-2, HPLT, CulturaX) · vi_wiki (Wikipedia, Wikisource) · long · en_edu (FineWeb-Edu)
          math (FineMath-4+) · code (github-code-clean, giấy phép thoáng)
    python scripts/build_cpt_data.py --sources vi_fineweb2,vi_hplt,vi_wiki --edu-clf data/cpt/edu_clf.pt \\
        --budget vi_web:10e9,vi_wiki:1e9,long:2e9 --out data/cpt --workers 30
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time, unicodedata
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np

SOURCES = {   # tên → (repo, cấu hình, cột chữ, nhóm, kiểu lọc)
    "vi_fineweb2": ("HuggingFaceFW/fineweb-2", "vie_Latn", "text", "vi_web", "vi"),
    "vi_hplt": ("HPLT/HPLT2.0_cleaned", "vie_Latn", "text", "vi_web", "vi"),
    "vi_culturax": ("uonlp/CulturaX", "vi", "text", "vi_web", "vi"),                 # gated: cần đăng nhập HF
    "vi_wiki": ("wikimedia/wikipedia", "20231101.vi", "text", "vi_wiki", "wiki"),
    "vi_wikisource": ("wikimedia/wikisource", "20231201.vi", "text", "vi_wiki", "wiki"),
    "en_edu": ("HuggingFaceFW/fineweb-edu", "sample-100BT", "text", "en_edu", "plain"),
    "math": ("HuggingFaceTB/finemath", "finemath-4plus", "text", "math", "plain"),
    "code": ("codeparrot/codeparrot-clean", None, "content", "code", "code"),      # Python; github-code-clean dùng script cũ (datasets 5 bỏ)
}
PERMISSIVE = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "cc0-1.0", "unlicense", "mpl-2.0"}
CODE_LANGS = {"Python", "JavaScript", "TypeScript", "Java", "C#", "SQL", "HTML", "Shell", "Go", "C++", "PHP"}

G = {}


def norm(t):
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[ \t ]+", " ", t)
    t = re.sub(r"\n\s*\n\s*\n+", "\n\n", t)
    return t.strip()


def init(tok_path, clf_path, edu_min, minc, minw):
    from nn_aivie.core.tokenizer import BPETokenizer
    G["tok"] = BPETokenizer.load(tok_path); G["eos"] = G["tok"].token_to_id["<EOS>"]
    G["edu"] = None
    if clf_path:
        import edu_classifier as E
        G["edu"] = E.EduScorer.load(clf_path)
    G.update(edu_min=edu_min, minc=minc, minw=minw)


def work(item):
    """(chữ, kiểu lọc) → ('ok', ids, điểm) | ('loai', lý do)."""
    import build_pretrain_pipeline as B
    t, kind = item
    t = norm(t)
    if kind == "code":
        if not (100 <= len(t) <= 60000) or max(len(l) for l in t.split("\n")) > 400:
            return ("loai", "code-xấu")
    elif kind != "plain":
        r = B.quality_reason(t, G["minc"], G["minw"])
        if r:
            return ("loai", r)
    s = -1.0
    if kind == "vi" and G["edu"] is not None:
        s = G["edu"].score([t[:3000]])[0]
        if s < G["edu_min"]:
            return ("loai", "edu-thấp")
    ids = G["tok"].encode(t); ids.append(G["eos"])
    return ("ok", np.asarray(ids, dtype=np.uint16), s)


UP = {"repo": None, "threads": [], "sem": None, "ok": 0, "fail": 0}


def upload_then_delete(path, rel):
    """ổ box nhỏ (120–150GB): mỗi shard ghi xong → đẩy lên HF dataset riêng tư → thành công mới xoá bản trên ổ."""
    from huggingface_hub import HfApi
    with UP["sem"]:
        for k in range(5):
            try:
                HfApi().upload_file(path_or_fileobj=path, path_in_repo=rel, repo_id=UP["repo"], repo_type="dataset")
                os.remove(path); UP["ok"] += 1
                return
            except Exception as e:
                print(f"[up] lỗi {rel} lần {k + 1}: {type(e).__name__} {str(e)[:120]}", flush=True)
                time.sleep(30 * (k + 1))
        UP["fail"] += 1                                         # GIỮ file trên ổ — đẩy tay sau


class Writer:
    """shard_<tag>_<số>.npy — mỗi tiến trình (1 nguồn) một tag → nhiều nguồn chạy SONG SONG ghi chung thư mục nhóm."""
    def __init__(self, out, cap, tag):
        self.out, self.cap, self.tag = out, cap, tag
        self.group = os.path.basename(out.rstrip("/"))
        os.makedirs(out, exist_ok=True)
        self.idx = len([f for f in os.listdir(out) if f.startswith(f"shard_{tag}_")])
        self.buf, self.n, self.total = [], 0, 0

    def add(self, ids):
        self.buf.append(ids); self.n += len(ids); self.total += len(ids)
        if self.n >= self.cap:
            self.flush()

    def flush(self):
        if self.buf:
            fn = f"shard_{self.tag}_{self.idx:05d}.npy"
            np.save(os.path.join(self.out, fn), np.concatenate(self.buf))
            if UP["repo"]:
                import threading
                th = threading.Thread(target=upload_then_delete, args=(os.path.join(self.out, fn), f"{self.group}/{fn}"))
                th.start(); UP["threads"].append(th)
            self.buf, self.n, self.idx = [], 0, self.idx + 1


def stream(name, local=None):
    if local:
        import pyarrow.parquet as pq
        for t in pq.read_table(local, columns=["text"]).column("text").to_pylist():
            yield t
        return
    from datasets import load_dataset
    repo, cfg, col, _, kind = SOURCES[name]
    for r in load_dataset(repo, cfg, split="train", streaming=True):
        if kind == "code" and (r.get("license") not in PERMISSIVE or r.get("language", "Python") not in CODE_LANGS
                               or str(r.get("autogenerated", "False")) == "True"):
            continue
        yield r[col]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="vi_fineweb2,vi_hplt,vi_wiki")
    ap.add_argument("--local", default="", help="parquet cục bộ thay cho nguồn HF (thử trên M4; dùng với 1 nguồn)")
    ap.add_argument("--out", default="data/cpt")
    ap.add_argument("--budget", default="vi_web:10e9,vi_wiki:1e9,long:2e9,en_edu:3e9,math:1e9,code:2e9")
    ap.add_argument("--edu-clf", default="")
    ap.add_argument("--edu-min", type=float, default=2.5)
    ap.add_argument("--long-min", type=int, default=4096, help="văn bản ≥ bấy nhiêu token → nhóm long")
    ap.add_argument("--tokenizer", default="checkpoints/bpe_v1.json")
    ap.add_argument("--shard-tokens", type=int, default=200_000_000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--hf-upload", default="", help="dataset HF riêng tư: đẩy từng shard lên rồi XOÁ trên ổ (box ổ nhỏ)")
    a = ap.parse_args()
    if a.hf_upload:
        import threading
        from huggingface_hub import HfApi
        HfApi().create_repo(a.hf_upload, repo_type="dataset", private=True, exist_ok=True)
        UP["repo"], UP["sem"] = a.hf_upload, threading.Semaphore(3)
    budget = {k: float(v) for k, v in (x.split(":") for x in a.budget.split(","))}
    tag = a.sources.replace(",", "+")
    writers = {g: Writer(os.path.join(a.out, g), a.shard_tokens, tag) for g in budget}
    seen = set()
    stats = {}
    T0 = time.time()
    pool = mp.get_context("spawn").Pool(a.workers, initializer=init,
                                        initargs=(a.tokenizer, a.edu_clf, a.edu_min, a.min_chars, a.min_words))
    for name in a.sources.split(","):
        grp, kind = SOURCES[name][3], SOURCES[name][4]
        if grp not in budget:
            continue
        st = stats.setdefault(name, {"doc": 0, "giu": 0, "trung": 0, "token": 0, "loai": {}})
        it = stream(name, a.local or None)
        done = False
        while not done:
            batch = []
            for t in it:
                st["doc"] += 1
                h = hashlib.blake2b(re.sub(r"\s+", " ", t[:2000]).encode(), digest_size=8).digest()
                if h in seen:
                    st["trung"] += 1; continue
                seen.add(h); batch.append((t, kind))
                if len(batch) >= a.chunk:
                    break
            if not batch:
                break
            for r in pool.imap(work, batch, chunksize=32):
                if r[0] != "ok":
                    st["loai"][r[1]] = st["loai"].get(r[1], 0) + 1; continue
                ids = r[1]
                g = "long" if (len(ids) >= a.long_min and "long" in budget and kind in ("vi", "wiki")
                               and writers["long"].total < budget["long"]) else grp
                if writers[g].total >= budget[g]:
                    continue
                writers[g].add(ids); st["giu"] += 1; st["token"] += len(ids)
            if writers[grp].total >= budget[grp]:
                done = True
            if st["doc"] % (a.chunk * 25) < a.chunk:
                print(f"[{time.time()-T0:.0f}s] {name}: {st['doc']} văn bản · giữ {st['giu']} · "
                      f"{st['token']/1e6:.0f}M token · nhóm {grp} {writers[grp].total/1e9:.2f}/{budget[grp]/1e9:.1f}B", flush=True)
        print(f"[{time.time()-T0:.0f}s] XONG {name}: {json.dumps(st, ensure_ascii=False)}", flush=True)
    pool.close()
    for w in writers.values():
        w.flush()
    man = {"created": time.strftime("%Y-%m-%d %H:%M"), "dtype": "uint16", "V": 32002, "EOS": 1, "tokenizer": "bpe_v1",
           "groups": {g: w.total for g, w in writers.items()}, "sources": stats, "edu_min": a.edu_min, "long_min": a.long_min}
    mp_ = os.path.join(a.out, f"manifest_{tag}.json")
    json.dump(man, open(mp_, "w"), ensure_ascii=False, indent=1)
    if UP["repo"]:
        for th in UP["threads"]:
            th.join()
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=mp_, path_in_repo=os.path.basename(mp_), repo_id=UP["repo"], repo_type="dataset")
        print(f"[up] đã đẩy {UP['ok']} shard · lỗi {UP['fail']} (lỗi thì file còn trên ổ)", flush=True)
    print("### " + json.dumps(man["groups"]), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)          # luồng đọc nền của datasets (streaming) còn sống lúc thoát → Python báo "Fatal … finalizing"; dữ liệu đã ghi + đẩy xong
