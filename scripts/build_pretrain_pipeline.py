"""§TK_100 · M0 — PIPELINE DỰNG DATA PRETRAIN TIẾNG VIỆT (tại-chỗ, không tốn GPU).

Luồng: stream HuggingFace → LỌC chất-lượng (heuristic vi) → DEDUP near-dup (MinHash-LSH
thuần numpy, không cần datatrove/datasketch) → KHỬ-NHIỄM với eval (tuỳ chọn) → TOKENIZE
BPE `bpe_v1` → ghi SHARD uint16 (V=32002 < 65536) + manifest.json.

Shard là mảng token phẳng, mỗi tài-liệu kết bằng <EOS> — mmap được, khung-nào cũng nạp
được (Megatron/nanotron/loader của ta). Đây là bước NẶNG NHẤT của M0 và chạy được ngay
trên máy dev; scale thật chỉ việc tăng --limit / thêm --sources.

SMOKE (rẻ, ~vài nghìn doc):
    PYTHONPATH=src .venv/bin/python scripts/build_pretrain_pipeline.py \
        --sources fineweb2 wiki --limit 5000 --out data/pretrain_vi_smoke

THẬT (mở --limit, thêm nguồn; CulturaX cần `huggingface-cli login`):
    PYTHONPATH=src .venv/bin/python scripts/build_pretrain_pipeline.py \
        --sources fineweb2 culturax wiki --limit 2000000 --shard-tokens 100000000 \
        --out data/pretrain_vi
"""
import sys, os, re, time, json, argparse, unicodedata
sys.path.insert(0, "scripts")
import numpy as np
from nn_aivie.core.tokenizer import BPETokenizer

T0 = time.time()

# ── đa-tiến-trình: LỌC + tokenize song-song (nút thắt là vòng-lọc serial ở main) ───
_TOK = None; _EOS = None; _MINC = 200; _MINW = 40


def _init_worker(tok_path, minc=200, minw=40):
    global _TOK, _EOS, _MINC, _MINW
    _TOK = BPETokenizer.load(tok_path); _EOS = _TOK.token_to_id["<EOS>"]
    _MINC, _MINW = minc, minw


def _encode(text):                                        # chỉ tokenize (khi lọc đã làm ở main)
    ids = _TOK.encode(text); ids.append(_EOS)
    return np.asarray(ids, dtype=np.uint16)


def _filter_encode(text):                                 # LỌC + tokenize trong worker → trả array(giữ) hoặc str(lý-do loại)
    t = norm_ws(text)
    if not t:
        return "rỗng"
    r = quality_reason(t, _MINC, _MINW)
    if r:
        return r
    ids = _TOK.encode(t); ids.append(_EOS)
    return np.asarray(ids, dtype=np.uint16)
MERSENNE = (1 << 61) - 1                                   # số nguyên-tố cho hash MinHash

# ── Nguồn HF: name → (repo, config, split, cột-text). Sửa/thêm tự-do. ──────────────
SOURCES = {
    "fineweb2": ("HuggingFaceFW/fineweb-2", "vie_Latn", "train", "text"),   # ⭐ ODC-BY, không gated
    "culturax": ("uonlp/CulturaX",          "vi",       "train", "text"),   # gated → cần HF login
    "wiki":     ("wikimedia/wikipedia",     "20231101.vi", "train", "text"),
    "madlad":   ("allenai/madlad-400",      "vi",       "clean", "text"),
}

# ── Ký-tự tiếng Việt (Latin + dấu) để đo tín-hiệu ngôn-ngữ ─────────────────────────
_VI = "aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵđ"
VI_CHARS = frozenset(_VI + _VI.upper())
LATIN = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") | VI_CHARS
WORD_RE = re.compile(r"\w+", re.UNICODE)


def norm_ws(t: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t)).strip()


# ── LỌC chất-lượng (Gopher/FineWeb rút gọn, chỉnh cho tiếng Việt) ──────────────────
def quality_reason(t: str, min_chars: int, min_words: int):
    """Trả về None nếu ĐẠT, hoặc chuỗi lý-do BỊ LOẠI."""
    if len(t) < min_chars:
        return "ngắn"
    words = WORD_RE.findall(t)
    n = len(words)
    if n < min_words:
        return "ít-từ"
    mean_wl = sum(len(w) for w in words) / n
    if not (2.5 <= mean_wl <= 12):
        return "độ-dài-từ-lạ"
    alpha = sum(c.isalpha() for c in t)
    if alpha / len(t) < 0.55:
        return "ít-chữ-cái"
    digit = sum(c.isdigit() for c in t)
    if digit / len(t) > 0.20:
        return "nhiều-số"
    latin = sum(c in LATIN for c in t if c.isalpha())
    if alpha and latin / alpha < 0.80:                    # loại doc chủ-yếu CJK/khác
        return "không-phải-vi"
    lines = [ln for ln in t.split("\n") if ln.strip()]
    if lines:
        uniq = len(set(lines)) / len(lines)
        if uniq < 0.60:                                   # lặp dòng nhiều = boilerplate
            return "lặp-dòng"
    # tỉ-lệ từ nằm trong top lặp (spam): nếu 1 từ chiếm >30% → loại
    if n >= 20:
        from collections import Counter
        top = Counter(words).most_common(1)[0][1]
        if top / n > 0.30:
            return "lặp-từ"
    return None


# ── DEDUP near-dup: MinHash-LSH thuần numpy (chuẩn SlimPajama: trùng 1 band = dup) ──
class MinHashLSH:
    def __init__(self, num_perm=64, bands=16, k=5, seed=0):
        assert num_perm % bands == 0, "num_perm phải chia hết bands"
        self.num_perm, self.bands, self.rows, self.k = num_perm, bands, num_perm // bands, k
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, MERSENNE, num_perm, dtype=np.uint64)
        self.b = rng.integers(0, MERSENNE, num_perm, dtype=np.uint64)
        self.seen = set()                                 # {(band_idx, hash băng)}

    def _shingles(self, t: str):
        t = re.sub(r"\s+", " ", t.lower())
        if len(t) <= self.k:
            return np.array([hash(t) & 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
        s = {t[i:i + self.k] for i in range(len(t) - self.k + 1)}
        return np.fromiter((hash(x) & 0xFFFFFFFFFFFFFFFF for x in s), dtype=np.uint64, count=len(s))

    def is_dup(self, t: str) -> bool:
        x = self._shingles(t)                             # (m,) base-hash các shingle
        # signature: min_j (a_i*x_j + b_i) mod P  → (num_perm,)
        h = (np.multiply.outer(self.a, x) + self.b[:, None]) % MERSENNE
        sig = h.min(axis=1)                               # (num_perm,)
        keys, dup = [], False
        for bi in range(self.bands):
            band = sig[bi * self.rows:(bi + 1) * self.rows].tobytes()
            key = (bi, hash(band))
            keys.append(key)
            if key in self.seen:
                dup = True
        if not dup:
            self.seen.update(keys)
        return dup


# ── KHỬ-NHIỄM: bỏ doc chứa n-gram (từ) trùng tập eval ─────────────────────────────
def load_eval_ngrams(path, ngram=13):
    if not path or not os.path.exists(path):
        return None
    grams = set()
    for line in open(path, encoding="utf-8"):
        w = WORD_RE.findall(line.lower())
        for i in range(len(w) - ngram + 1):
            grams.add(hash(" ".join(w[i:i + ngram])))
    print(f"[{time.time()-T0:.0f}s] khử-nhiễm: {len(grams)} n-gram({ngram}) từ {path}", flush=True)
    return grams


def contaminated(t, grams, ngram=13):
    if not grams:
        return False
    w = WORD_RE.findall(t.lower())
    for i in range(len(w) - ngram + 1):
        if hash(" ".join(w[i:i + ngram])) in grams:
            return True
    return False


# ── GHI SHARD uint16 ──────────────────────────────────────────────────────────────
class ShardWriter:
    def __init__(self, out, shard_tokens):
        self.out, self.cap = out, shard_tokens
        os.makedirs(out, exist_ok=True)
        self.buf, self.idx, self.total = [], 0, 0

    def add(self, ids):
        self.buf.append(np.asarray(ids, dtype=np.uint16))
        self.total += len(ids)
        if sum(len(b) for b in self.buf) >= self.cap:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        arr = np.concatenate(self.buf)
        p = os.path.join(self.out, f"shard_{self.idx:05d}.npy")
        np.save(p, arr)
        print(f"[{time.time()-T0:.0f}s] LƯU {p} ({len(arr)/1e6:.2f}M token)", flush=True)
        self.buf, self.idx = [], self.idx + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["fineweb2", "wiki"],
                    help=f"chọn trong {list(SOURCES)}")
    ap.add_argument("--out", default="data/pretrain_vi")
    ap.add_argument("--limit", type=int, default=5000, help="số doc TỐI-ĐA mỗi nguồn (mở lớn khi train thật)")
    ap.add_argument("--tokenizer", default="checkpoints/bpe_v1.json")
    ap.add_argument("--shard-tokens", type=int, default=20_000_000)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--num-perm", type=int, default=64)
    ap.add_argument("--bands", type=int, default=16)
    ap.add_argument("--eval-ngrams", default=None, help="file text eval (mỗi dòng 1 mẫu) để khử-nhiễm")
    ap.add_argument("--no-dedup", action="store_true", help="tắt dedup (dùng cho FineWeb-2/CulturaX ĐÃ dedup tại nguồn → tránh nổ RAM)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="số tiến-trình tokenize song-song (nút thắt CPU M4)")
    ap.add_argument("--chunk", type=int, default=3000, help="số doc gom mỗi lần tokenize song-song")
    ap.add_argument("--max-tokens", type=float, default=0, help="DỪNG khi đạt tổng token này (0=vô-hạn) — chặn tràn đĩa")
    a = ap.parse_args()

    from datasets import load_dataset                     # nạp trễ để --help nhanh
    tok = BPETokenizer.load(a.tokenizer)
    EOS = tok.token_to_id["<EOS>"]; V = len(tok.tokens)
    assert V <= 65535, "V vượt uint16"
    global _TOK, _EOS, _MINC, _MINW                        # cho _encode/_filter_encode nhánh đơn-luồng
    _TOK, _EOS, _MINC, _MINW = tok, EOS, a.min_chars, a.min_words
    print(f"[{time.time()-T0:.0f}s] tokenizer V={V}, EOS={EOS}", flush=True)

    lsh = None if a.no_dedup else MinHashLSH(a.num_perm, a.bands)
    eval_grams = load_eval_ngrams(a.eval_ngrams)
    sw = ShardWriter(a.out, a.shard_tokens)
    stats = {"per_source": {}, "dropped": {}, "total_words": 0}

    import multiprocessing as mp
    pool = (mp.Pool(a.workers, initializer=_init_worker, initargs=(a.tokenizer, a.min_chars, a.min_words))
            if a.workers > 1 else None)
    # LỌC-SONG-SONG: khi dedup tắt & không khử-nhiễm → đẩy CẢ lọc vào worker (phá nghẽn 1-core main)
    par_filter = pool is not None and a.no_dedup and not eval_grams
    mode = f"{a.workers} tiến-trình (lọc+tokenize)" if par_filter else \
           (f"{a.workers} tiến-trình (chỉ tokenize)" if pool else "đơn-luồng")
    print(f"[{time.time()-T0:.0f}s] xử-lý: {mode}", flush=True)
    pending = []                                          # par_filter: RAW text · else: text đã-lọc

    def drop(reason):
        stats["dropped"][reason] = stats["dropped"].get(reason, 0) + 1

    def flush():                                          # trả #doc GIỮ
        if not pending:
            return 0
        if par_filter:
            res = pool.map(_filter_encode, pending, chunksize=64)
        else:
            res = pool.map(_encode, pending, chunksize=64) if pool else [_encode(t) for t in pending]
        k = 0
        for r in res:
            if isinstance(r, np.ndarray):
                sw.add(r); k += 1
            else:
                drop(r)
        pending.clear()
        return k

    reached = False
    for name in a.sources:
        if reached:
            break
        if name not in SOURCES:
            print(f"⚠️ bỏ qua nguồn lạ: {name}", flush=True); continue
        repo, cfg, split, col = SOURCES[name]
        print(f"[{time.time()-T0:.0f}s] === {name}: {repo} ({cfg}/{split}) ===", flush=True)
        try:
            ds = load_dataset(repo, cfg, split=split, streaming=True)
        except Exception as e:
            print(f"⚠️ KHÔNG mở được {name}: {e}\n   (CulturaX gated → `huggingface-cli login`)", flush=True)
            continue
        seen_in = kept = 0
        for row in ds:
            if seen_in >= a.limit:
                break
            seen_in += 1
            if par_filter:                                # LỌC ở worker: đẩy RAW text
                pending.append(row.get(col) or "")
            else:                                         # LỌC ở main (cần cho dedup/khử-nhiễm)
                t = norm_ws(row.get(col) or "")
                if not t:
                    drop("rỗng"); continue
                r = quality_reason(t, a.min_chars, a.min_words)
                if r:
                    drop(r); continue
                if eval_grams and contaminated(t, eval_grams):
                    drop("nhiễm-eval"); continue
                if lsh and lsh.is_dup(t):
                    drop("trùng"); continue
                pending.append(t)
                stats["total_words"] += len(WORD_RE.findall(t))
            if len(pending) >= a.chunk:
                kept += flush()
                if a.max_tokens and sw.total >= a.max_tokens:
                    print(f"[{time.time()-T0:.0f}s]   ĐẠT max-tokens {sw.total/1e9:.2f}B → dừng", flush=True)
                    reached = True; break
            if seen_in % 20000 == 0:
                print(f"[{time.time()-T0:.0f}s]   {name}: đọc {seen_in} · giữ {kept} · "
                      f"{sw.total/1e6:.2f}M token", flush=True)
        kept += flush()
        stats["per_source"][name] = {"repo": repo, "read": seen_in, "kept": kept}
        print(f"[{time.time()-T0:.0f}s] {name}: đọc {seen_in} → giữ {kept}", flush=True)

    flush()
    if pool:
        pool.close(); pool.join()
    sw.flush()
    total_kept = sum(s["kept"] for s in stats["per_source"].values())
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "out": a.out, "dtype": "uint16", "V": V, "EOS": EOS,
        "num_shards": sw.idx, "total_tokens": sw.total,
        "total_docs_kept": total_kept,
        "fertility_tok_per_word": (round(sw.total / stats["total_words"], 3) if stats["total_words"] else None),
        "params": {"limit": a.limit, "min_chars": a.min_chars, "min_words": a.min_words,
                   "dedup": (not a.no_dedup), "num_perm": a.num_perm, "bands": a.bands},
        "per_source": stats["per_source"], "dropped": stats["dropped"],
    }
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[{time.time()-T0:.0f}s] ===== XONG =====", flush=True)
    print(f"  giữ {total_kept} doc · {sw.total/1e6:.2f}M token · {sw.idx} shard", flush=True)
    print(f"  fertility ≈ {manifest['fertility_tok_per_word']} token/từ", flush=True)
    print(f"  loại: {stats['dropped']}", flush=True)
    print(f"  manifest: {os.path.join(a.out, 'manifest.json')}", flush=True)


if __name__ == "__main__":
    main()
