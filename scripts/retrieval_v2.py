"""§TK_079 — RETRIEVAL v2: BM25F (tiêu đề + toàn thân) thay BM25-chỉ-lead.

Vá lỗi "Hà Nội → Hà Đông": query khớp TIÊU ĐỀ bài đúng phải được ưu tiên mạnh.
- 2 trường: title (trọng số w_title cao) + body (index sâu hơn, không chỉ 200 char đầu).
- tiêu đề = cụm đầu trước ' là '/'('/','; trang định-hướng ("có thể là") bị phạt.

Dùng như BM25 cũ: bm.top(query, k) -> list doc_id. Kèm snippets để nạp cho C.

    .venv/bin/python scripts/retrieval_v2.py            # tự test so cũ/mới
"""
import sys, time, json, re, math, argparse
from collections import defaultdict, Counter

VIWIKI = "data/text_corpus/viwiki.jsonl"
_TOK = re.compile(r"\w+", re.UNICODE)


def tok(s):
    return _TOK.findall(s.lower())


# từ dừng tiếng Việt (từ hỏi/hư từ) — bỏ khi tính "query phủ tiêu đề"
STOP = {"là", "như", "thế", "nào", "gì", "của", "có", "và", "các", "một", "những",
        "cái", "người", "ở", "đâu", "bao", "nhiêu", "khi", "được", "cho", "để",
        "này", "đó", "thì", "mà", "ra", "vào", "về", "với", "trong", "hay", "hoặc",
        "bằng", "bởi", "tại", "trên", "dưới", "sao", "vì", "nên"}


def extract_title(text):
    """Tiêu đề = cụm đầu trước ' là '/'('/','/xuống dòng."""
    head = text[:120]
    m = re.split(r"\s+là\s+|\s+hay\s+|\s+\(|[,\n:]", head, maxsplit=1)[0].strip()
    return m if 1 <= len(m) <= 60 else text[:40]


class BM25F:
    """BM25F 2 trường (title, body) CHUẨN: chuẩn-hoá độ dài TỪNG trường riêng,
    rồi mới gộp theo trọng số → title match không bị body dài làm loãng."""

    def __init__(self, titles, bodies, is_disamb, k1=1.5,
                 b_title=0.6, b_body=0.75, w_title=6.0, w_body=1.0):
        self.k1 = k1
        self.N = len(bodies)
        self.is_disamb = is_disamb
        self.titles = titles             # per-doc token list (giữ THỨ TỰ cho phrase-match)
        self.tf_t, self.tf_b = [], []    # per-doc Counter mỗi trường
        dlt, dlb = [], []
        df = defaultdict(int)
        self.inv = defaultdict(list)
        for i in range(self.N):
            ct, cb = Counter(titles[i]), Counter(bodies[i])
            self.tf_t.append(ct); self.tf_b.append(cb)
            dlt.append(sum(ct.values())); dlb.append(sum(cb.values()))
            for t in set(ct) | set(cb):
                df[t] += 1; self.inv[t].append(i)
        self.dlt, self.dlb = dlt, dlb
        self.avgt = sum(dlt) / max(self.N, 1)
        self.avgb = sum(dlb) / max(self.N, 1)
        self.b_title, self.b_body = b_title, b_body
        self.w_title, self.w_body = w_title, w_body
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    @staticmethod
    def _contig(sub, seq):
        """sub (token list tiêu đề) có là dãy LIÊN-TIẾP trong seq (query) không."""
        n = len(sub)
        if n == 0 or n > len(seq):
            return False
        for i in range(len(seq) - n + 1):
            if seq[i:i + n] == sub:
                return True
        return False

    def top(self, query, k=3, cov_boost=2.5, q_boost=1.5, phrase_boost=4.0):
        q = tok(query)
        qset = {t for t in q if t not in STOP}      # từ nội-dung của query
        scores = defaultdict(float)
        for t in set(q):
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i in self.inv[t]:
                # chuẩn hoá độ dài RIÊNG mỗi trường trước khi gộp
                nt = self.tf_t[i].get(t, 0) / (1 - self.b_title + self.b_title * self.dlt[i] / max(self.avgt, 1e-9))
                nb = self.tf_b[i].get(t, 0) / (1 - self.b_body + self.b_body * self.dlb[i] / max(self.avgb, 1e-9))
                wtf = self.w_title * nt + self.w_body * nb
                scores[i] += idf * wtf * (self.k1 + 1) / (self.k1 + wtf)
        for i in list(scores):
            # title-coverage: query phủ được bao nhiêu phần của TIÊU ĐỀ
            ttok = set(self.tf_t[i])
            if ttok and qset:
                matched = len(ttok & qset)
                cov = matched / len(ttok)            # query phủ tiêu đề (1.0=khớp trọn chủ đề)
                qcov = matched / len(qset)           # tiêu đề giải thích bao nhiêu phần query
                scores[i] *= (1 + cov_boost * cov + q_boost * qcov)
                # PHRASE: query GỌI ĐÍCH DANH thực thể (tiêu đề là cụm liên-tiếp trong query)
                if len(self.titles[i]) >= 1 and self._contig(self.titles[i], q):
                    scores[i] *= phrase_boost
            if self.is_disamb[i]:
                scores[i] *= 0.5          # phạt trang định-hướng
        return sorted(scores, key=scores.get, reverse=True)[:k]


def load_corpus_v2(max_docs, body_chars=3500, snippet_chars=3500):
    # §TK_090 — index+snippet SÂU hơn (1500/700→3500): fact quan trọng hay nằm sâu
    # trong bài, không ở lead (vd "thủ đô Tokyo" ở ký tự 3302 của bài Nhật Bản).
    titles, bodies, is_disamb, snippets = [], [], [], []
    T0 = time.time()
    for i, line in enumerate(open(VIWIKI)):
        if i >= max_docs:
            break
        try:
            t = json.loads(line)
        except Exception:
            continue
        if not isinstance(t, str) or len(t) < 40:
            continue
        title = extract_title(t)
        titles.append(tok(title))
        bodies.append(tok(t[:body_chars]))
        is_disamb.append(("có thể là" in t[:60].lower()) or ("có thể chỉ" in t[:60].lower()))
        snippets.append(t[:snippet_chars].replace("\n", " "))
        if i % 50000 == 0:
            print(f"[{time.time()-T0:.0f}s] nạp {i}…", flush=True)
    print(f"[{time.time()-T0:.0f}s] {len(bodies)} bài; dựng BM25F (title+body)…", flush=True)
    bm = BM25F(titles, bodies, is_disamb)
    print(f"[{time.time()-T0:.0f}s] BM25F sẵn sàng (avg title={bm.avgt:.1f} body={bm.avgb:.0f})", flush=True)
    return bm, snippets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=285000)
    a = ap.parse_args()
    sys.path.insert(0, "scripts")
    from rag_seallm import load_corpus as load_old   # BM25 cũ để so
    print(">>> nạp BM25 CŨ (chỉ-lead)…", flush=True)
    old, osn = load_old(a.max_docs)
    print(">>> nạp BM25F MỚI (title+body)…", flush=True)
    new, nsn = load_corpus_v2(a.max_docs)
    QS = ["Hà Nội là thành phố như thế nào?", "sông Mê Kông chảy qua nước nào?",
          "chó là loài gì?", "Sao Hỏa có mấy vệ tinh?", "tế bào là gì?",
          "Việt Nam có bao nhiêu tỉnh?"]
    for q in QS:
        oi = old.top(q, 1)[0]; ni = new.top(q, 1)[0]
        same = "＝" if oi == ni else "≠"
        print(f"\n❓ {q}", flush=True)
        print(f"  CŨ : {osn[oi][:78]}", flush=True)
        print(f"  MỚI: {nsn[ni][:78]}  {same}", flush=True)
    print("\n### hết", flush=True)


if __name__ == "__main__":
    main()
