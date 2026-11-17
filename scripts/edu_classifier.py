"""P0/P1 · BỘ CHẤM "GIÁ TRỊ GIÁO DỤC" cho văn bản web tiếng Việt (kiểu FineWeb-Edu) → lọc giữ phần tốt nhất để train tiếp.

  sample  đọc luồng web tiếng Việt (FineWeb-2 / HPLT / parquet cục bộ) → N đoạn (≤3000 ký tự) đa dạng
  label   thầy Gemma-31B (vLLM) chấm 0–5 theo thang cộng điểm (như FineWeb-Edu, Việt hoá)
  train   model nhỏ kiểu fastText (túi n-gram từ băm → nhúng 32 chiều → hồi quy điểm), CPU vài phút;
          báo cáo MSE + F1 ở ngưỡng ≥3 trên tập giữ riêng
  EduScorer.load(path).score(texts) → điểm (dùng trong build_cpt_data.py để lọc hàng chục triệu văn bản)

    python scripts/edu_classifier.py sample --n 60000 --out data/cpt/edu_sample.jsonl
    python scripts/edu_classifier.py label  --inp data/cpt/edu_sample.jsonl --llm http://127.0.0.1:8000/v1 --workers 128
    python scripts/edu_classifier.py train  --inp data/cpt/edu_labels.jsonl --out data/cpt/edu_clf.pt
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, time, zlib
import concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn

P_EDU = """Dưới đây là một đoạn trích văn bản từ web tiếng Việt. Hãy đánh giá GIÁ TRỊ GIÁO DỤC của nó (hữu ích cho việc dạy–học
từ tiểu học đến đại học) theo thang CỘNG ĐIỂM 0–5:
- +1 nếu có một chút thông tin cơ bản liên quan đến chủ đề học tập, dù lẫn quảng cáo hay nội dung không liên quan.
- +1 nếu đề cập một số kiến thức phù hợp cho học tập nhưng chưa mạch lạc, chưa đầy đủ.
- +1 nếu phù hợp cho học tập và giới thiệu được khái niệm chính, mạch lạc dù chưa thật đầy đủ.
- +1 nếu rất hữu ích cho học tập, rõ ràng, nhất quán, giống một đoạn sách giáo khoa hay bài giảng, ít nội dung thừa.
- +1 nếu xuất sắc: giải thích sâu, có lập luận/ví dụ, không quảng cáo, như sách giáo khoa chất lượng cao.
Trừ hết điểm nếu là quảng cáo, rao vặt, nội dung người lớn, tin giật gân, danh sách liên kết hoặc văn bản vô nghĩa.

Văn bản:
<<<
{t}
>>>
Trả về DUY NHẤT JSON: {{"ly_do": "một câu ngắn", "diem": <0-5>}}"""


# ── sample ────────────────────────────────────────────────────────────────────────────
def stream_vi(source, local=None):
    if local and os.path.exists(local):
        import pyarrow.parquet as pq
        for t in pq.read_table(local, columns=["text"]).column("text").to_pylist():
            yield t
        return
    from datasets import load_dataset
    repo, cfg = {"fineweb2": ("HuggingFaceFW/fineweb-2", "vie_Latn"), "hplt": ("HPLT/HPLT2.0_cleaned", "vie_Latn"),
                 "culturax": ("uonlp/CulturaX", "vi")}[source]
    for r in load_dataset(repo, cfg, split="train", streaming=True):
        yield r["text"]


def cmd_sample(a):
    r = random.Random(a.seed)
    out = []
    per = a.n // len(a.sources.split(","))
    for src in a.sources.split(","):
        k = 0
        for t in stream_vi(src, a.local):
            if len(t) < 200 or r.random() > a.keep:
                continue
            s = r.randrange(0, max(1, len(t) - 3000)) if len(t) > 3000 else 0
            out.append({"src": src, "text": t[s:s + 3000]})
            k += 1
            if k >= per:
                break
        print(f"[sample] {src}: {k}", flush=True)
    r.shuffle(out)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"### {len(out)} đoạn → {a.out}", flush=True)


# ── label ─────────────────────────────────────────────────────────────────────────────
def cmd_label(a):
    from gen_office import loads_loose
    import gen_vlm_data as G
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    out = a.out or a.inp.replace("sample", "labels")
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["text"][:200] for l in open(out, encoding="utf-8")}
    todo = [x for x in rows if x["text"][:200] not in done]
    print(f"[label] {len(todo)} đoạn cần chấm", flush=True)

    def work(x):
        try:
            j = loads_loose(G.chat(a.llm, P_EDU.format(t=x["text"][:2500]), temp=0, max_tokens=200))
            d = float(j.get("diem", -1))
            return {**x, "diem": d} if 0 <= d <= 5 else None
        except Exception:
            return None
    n, T0 = 0, time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex, open(out, "a", encoding="utf-8") as f:
        for k, r_ in enumerate(ex.map(work, todo)):
            if r_:
                f.write(json.dumps(r_, ensure_ascii=False) + "\n"); n += 1
            if k % 2000 == 0:
                print(f"  {k}/{len(todo)} · {time.time()-T0:.0f}s", flush=True)
    print(f"### chấm {n}/{len(todo)} → {out}", flush=True)


# ── model kiểu fastText ───────────────────────────────────────────────────────────────
NB = 1 << 21


def feats(t, max_words=600):
    """băm từ (chữ thường) + cặp từ liền nhau → chỉ số trong NB ô."""
    w = re.findall(r"\w+", t.lower())[:max_words]
    ids = [zlib.crc32(x.encode()) % NB for x in w]
    ids += [zlib.crc32((x + " " + y).encode()) % NB for x, y in zip(w, w[1:])]
    return ids or [0]


class EduNet(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.emb = nn.EmbeddingBag(NB, dim, mode="mean")
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, flat, offs):
        return self.out(self.emb(flat, offs)).squeeze(-1)


def pack(texts):
    fs = [feats(t) for t in texts]
    offs = torch.tensor([0] + [len(f) for f in fs[:-1]]).cumsum(0)
    return torch.tensor([i for f in fs for i in f]), offs


class EduScorer:
    def __init__(self, net):
        self.net = net.eval()

    @classmethod
    def load(cls, path):
        net = EduNet(); net.load_state_dict(torch.load(path, map_location="cpu")["state"]); return cls(net)

    @torch.no_grad()
    def score(self, texts, bs=512):
        out = []
        for i in range(0, len(texts), bs):
            out += self.net(*pack(texts[i:i + bs])).clamp(0, 5).tolist()
        return out


def cmd_train(a):
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    random.Random(0).shuffle(rows)
    nv = max(200, len(rows) // 10)
    va, tr = rows[:nv], rows[nv:]
    net = EduNet()
    opt = torch.optim.Adam([{"params": net.emb.parameters(), "lr": 0.05}, {"params": net.out.parameters(), "lr": 3e-3}])
    yv = torch.tensor([float(x["diem"]) for x in va])

    def evalv():
        p = torch.tensor(EduScorer(net).score([x["text"] for x in va]))
        net.train()
        tp = ((p >= 2.5) & (yv >= 3)).sum().item(); fp = ((p >= 2.5) & (yv < 3)).sum().item(); fn = ((p < 2.5) & (yv >= 3)).sum().item()
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        return round(float(((p - yv) ** 2).mean()), 3), round(f1, 3)
    print(f"[train] {len(tr)} train · {nv} val · điểm TB {yv.mean():.2f} · tỉ lệ ≥3: {(yv >= 3).float().mean():.2f}", flush=True)
    for ep in range(a.epochs):
        random.Random(ep).shuffle(tr)
        for i in range(0, len(tr), 64):
            b = tr[i:i + 64]
            loss = ((net(*pack([x["text"] for x in b])) - torch.tensor([float(x["diem"]) for x in b])) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        mse, f1 = evalv()
        print(f"[train] lượt {ep + 1}: MSE val {mse} · F1(≥3) {f1}", flush=True)
    torch.save({"state": net.state_dict(), "mse": mse, "f1": f1}, a.out)
    print(f"### {a.out} · MSE {mse} · F1 {f1}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "label", "train"])
    ap.add_argument("--sources", default="fineweb2,hplt")
    ap.add_argument("--local", default="", help="parquet cục bộ thay cho đọc luồng (thử trên M4)")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--keep", type=float, default=0.2, help="xác suất giữ mỗi văn bản khi lấy mẫu (đa dạng hơn)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inp", default="data/cpt/edu_sample.jsonl")
    ap.add_argument("--out", default="")
    ap.add_argument("--llm", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "sample" and not a.out:
        a.out = "data/cpt/edu_sample.jsonl"
    if a.cmd == "train" and not a.out:
        a.out = "data/cpt/edu_clf.pt"
    {"sample": cmd_sample, "label": cmd_label, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
