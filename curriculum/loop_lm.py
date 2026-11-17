"""TK_104 · Model LÕI-LẶP + thí-nghiệm "học theo lớp" trên chương-trình Toán tiểu-học (curriculum/gen_toan.py).

Kiến-trúc: mở-đầu (P tầng) → LÕI (C tầng) LẶP r vòng → kết (K tầng). Mỗi vòng lõi được "nhắc lại đề":
    e = mở-đầu(x);  s = 0;  lặp r lần: s = lõi(s + e);  y = kết(s)
Train với r ngẫu-nhiên 1..R → lúc dùng chọn r tuỳ ý (nghĩ ít / nghĩ nhiều).  Mã-hoá theo KÝ-TỰ (mỗi chữ số 1 token).
Khối = Transformer pre-norm (RMSNorm, RoPE, SDPA nhân-quả, SwiGLU). Lõi Mamba để sau (trên Mac quét tuần-tự rất chậm).

Lịch học (cùng tổng số bước, cùng lô → so được công-bằng):
    random       trộn đều lớp 1-5 suốt quá-trình
    curr         học lần-lượt lớp 1 → 5, mỗi lớp 1/5 số bước, KHÔNG ôn
    curr_review  như curr nhưng mỗi lô 30% là bài các lớp TRƯỚC (ôn bài)

    PYTHONPATH=curriculum .venv/bin/python curriculum/loop_lm.py --schedule curr_review --steps 6000 --tag cr_s1
    → out/curriculum/<tag>.json : độ-chính-xác từng lớp 1-6 × số vòng r, đo sau MỖI giai-đoạn (thấy được sự quên)
"""
from __future__ import annotations
import argparse, json, math, os, random, re, time, zlib
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = "data/curriculum/toan"
PAD, BOS, EOS = 0, 1, 2
EXTRA = ["(", ")", "C", "x", "è", "í", "ắ", "ọ", "ớ", "ò"]   # chỉ THÊM CUỐI (không đổi thứ-tự) → checkpoint cũ vẫn nạp được


# ── Dữ-liệu ──────────────────────────────────────────────────────────────────────────
class Vocab:
    def __init__(self):
        path = f"{DATA}/vocab.json"
        if not os.path.exists(path):
            chars = set()
            for fn in os.listdir(DATA):
                if fn.endswith(".jsonl"):
                    for l in open(f"{DATA}/{fn}", encoding="utf-8"):
                        r = json.loads(l)
                        chars |= set("Đề: " + r["q"] + "\nGiải: " + r["sol"])
            json.dump(sorted(chars), open(path, "w", encoding="utf-8"), ensure_ascii=False)
        self.itos = ["<pad>", "<bos>", "<eos>"] + json.load(open(path, encoding="utf-8")) + EXTRA
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def enc(self, s):
        return [self.stoi[c] for c in s]

    def dec(self, ids):
        return "".join(self.itos[i] for i in ids if i > 2)


def load_level(level, split):
    p = f"{DATA}/lop{level}_{split}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def quote_sol(r):
    """LỜI GIẢI TRÍCH CÂU: mỗi bước nhắc lại câu đề đang xử-lý → model chỉ cần tìm câu KẾ TIẾP chưa làm
    (theo dõi tiến-trình bằng nội-dung, không phải đếm). 'Lan làm rơi mất 57 quả: 92 - 57 = 35. …'"""
    sents = re.split(r"(?<=\.) ", r["q"])
    clauses = [re.sub(r"^(Rồi|Sau đó) ", "", c).rstrip(".") for c in sents[1:-1]]
    steps = r["sol"].split(". Đáp số:")[0].split(". ")
    assert len(clauses) == len(steps), (r["q"], r["sol"])
    return ". ".join(f"{c}: {st_}" for c, st_ in zip(clauses, steps)) + f". Đáp số: {r['ans']}."


def rewrite_sol(r):
    """VIẾT LẠI ĐỀ SAU MỖI BƯỚC (như người gạch câu đã làm): tính phép của câu ĐẦU, rồi chép phần đề CÒN LẠI với số
    mới → mỗi bước luôn là 'xử-lý câu đầu của một đề ngắn dần' (giống hệt bài 1 bước), không cần đếm hay tra vị-trí.
    '92 - 57 = 35. Còn: Lan có 35 quả táo. Sau đó … Hỏi …? 35 + 67 = 102. Còn: … Đáp số: 102.'"""
    sents = re.split(r"(?<=\.) ", r["q"])
    head, ops, ask = sents[0], sents[1:-1], sents[-1]
    m_ = re.match(r"(.+ có )(\d+)( .+\.)", head)
    steps = r["sol"].split(". Đáp số:")[0].split(". ")
    out = []
    for i, st_ in enumerate(steps):
        val = st_.split(" = ")[-1]
        rest = ops[i + 1:]
        out.append(f"{st_}. Còn: " + " ".join([f"{m_.group(1)}{val}{m_.group(3)}"] + rest + [ask]))
    return " ".join(out) + f" Đáp số: {r['ans']}."


def index_row(r, quote=True, top=40):
    """NHÃN CÂU, ĐIỂM BẮT ĐẦU NGẪU-NHIÊN (index hints): đề '(17) Rồi Lan … (18) Sau đó …', lời-giải '(17) …: a - b = c'.
    Nhãn chạy từ off ∈ [1, top-n] → model thấy quan-hệ 'câu kế = nhãn kế' trên cả dải 1..top, kể cả khi bài
    chỉ ≤4 bước → bài 12 bước (nhãn 1..12 hay 20..31) không còn lạ. Điểm bắt đầu cố-định theo băm của đề."""
    sents = re.split(r"(?<=\.) ", r["q"])
    ops, n = sents[1:-1], len(sents) - 2
    off = 1 + zlib.crc32(r["q"].encode()) % (top - n)
    q = " ".join([sents[0]] + [f"({off + i}) {c}" for i, c in enumerate(ops)] + [sents[-1]])
    steps = r["sol"].split(". Đáp số:")[0].split(". ")
    clauses = [re.sub(r"^(Rồi|Sau đó) ", "", c).rstrip(".") for c in ops]
    sol = ". ".join(f"({off + i}) {c}: {st_}" if quote else f"({off + i}) {st_}"
                    for i, (c, st_) in enumerate(zip(clauses, steps)))
    return {**r, "q": q, "sol": sol + f". Đáp số: {r['ans']}."}


THUA = ["{w} rất vui.", "Hôm đó trời nắng đẹp.", "{w} năm nay {n} tuổi.", "Sau đó {w} đi học về.", "Nhà {w} có {n} người.",
        "{w} thích màu xanh.", "Sau đó trời bắt đầu mưa.", "Bạn của {w} tên là {w2}.", "{w} đi bộ {n} phút.",
        "Sau đó {w} ngồi nghỉ một lúc.", "Lớp của {w} có {n} bạn.", "{w} học bài rất chăm.", "Mẹ của {w} đi chợ.",
        "Sau đó {w} gặp {w2}.", "Con mèo nhà {w} có {n} cái chân."]
TEN2 = ["Lan", "Minh", "An", "Hoa", "Nam", "Mai", "Tùng", "Linh", "Bình", "Hà"]


def distract_row(r, rnd, kmax=4):
    """ĐỀ CÓ THÔNG-TIN THỪA: chèn 0..kmax câu không liên-quan (có câu chứa số, có câu bắt đầu 'Sau đó') xen giữa
    các câu phép-tính → số câu trong đề KHÔNG còn bằng số bước; model phải chọn câu theo NỘI-DUNG.
    Lời-giải = trích câu (chỉ các câu phép-tính)."""
    sents = re.split(r"(?<=\.) ", r["q"])
    who = sents[0].split(" có ")[0]
    mid = list(sents[1:-1])
    for _ in range(rnd.randint(0, kmax)):
        d = rnd.choice(THUA).format(w=who, w2=rnd.choice([t for t in TEN2 if t != who]), n=rnd.randint(2, 40))
        mid.insert(rnd.randint(0, len(mid)), d)
    return {**r, "q": " ".join([sents[0]] + mid + [sents[-1]]), "sol": quote_sol(r)}


def encode_example(v, r, answer_only=False, quote=False, rewrite=False):
    p = [BOS] + v.enc("Đề: " + r["q"] + "\nGiải: ")
    t = v.enc(f"Đáp số: {r['ans']}." if answer_only else
              (rewrite_sol(r) if rewrite else (quote_sol(r) if quote else r["sol"]))) + [EOS]
    return p + t, [-100] * len(p) + t                       # loss CHỈ trên lời giải


# ── Model ────────────────────────────────────────────────────────────────────────────
POS = {"t": None}                                            # vị-trí NGẪU-NHIÊN (Ruoss 2023) — None = 0..l-1


def rope(x):                                                 # x (b,h,l,hd)
    b, h, l, hd = x.shape
    pos = POS["t"][:l].to(x.device) if POS["t"] is not None else torch.arange(l, device=x.device, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd))
    ang = torch.outer(pos, inv)
    c, s = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2).type_as(x)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        hid = int(8 * d / 3)
        self.w12 = nn.Linear(d, 2 * hid, bias=False)
        self.w3 = nn.Linear(hid, d, bias=False)

    def forward(self, x):
        b, l, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, l, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(b, l, d))
        g, u = self.w12(self.n2(x)).chunk(2, -1)
        return x + self.w3(F.silu(g) * u)


def ssd_scan(x, dt, A, B, C):
    """Quét SSM Mamba2: h_t = exp(dt·A)·h_{t-1} + dt·x_t⊗B_t ; y_t = h_t·C_t.
    CUDA → kernel Triton của mamba-ssm; Mac/CPU → vòng tuần-tự (chỉ để thử nhỏ)."""
    if x.is_cuda:
        from mamba_ssm import mamba_chunk_scan_combined
        return mamba_chunk_scan_combined(x, dt, A, B[:, :, None], C[:, :, None], chunk_size=64)
    b, l, nh, hd = x.shape
    h = x.new_zeros(b, nh, hd, B.shape[-1], dtype=torch.float32)
    ys = []
    for t in range(l):
        dA = torch.exp(dt[:, t] * A)                                  # (b,nh)
        h = h * dA[..., None, None] + (dt[:, t, :, None, None] * x[:, t, :, :, None].float()) * B[:, t, None, None, :].float()
        ys.append((h * C[:, t, None, None, :].float()).sum(-1))
    return torch.stack(ys, 1).type_as(x)


class MambaBlock(nn.Module):
    """Khối Mamba2 (SSD) + SwiGLU — CÙNG khung với Block (thay attention bằng quét trạng-thái).
    Không vị-trí tuyệt-đối (không RoPE) → ứng-viên tổng-quát-hoá ĐỘ DÀI tốt hơn."""
    def __init__(self, d, heads, d_state=64, headdim=32, d_conv=4):
        super().__init__()
        self.h, di = heads, 2 * d
        self.nh, self.hdim, self.n, self.di = di // headdim, headdim, d_state, di
        self.n1, self.n2, self.ny = nn.RMSNorm(d), nn.RMSNorm(d), nn.RMSNorm(di)
        self.inp = nn.Linear(d, 2 * di + 2 * d_state + self.nh, bias=False)   # z, x, B, C, dt
        self.conv = nn.Conv1d(di + 2 * d_state, di + 2 * d_state, d_conv, groups=di + 2 * d_state, padding=d_conv - 1)
        self.dt_bias = nn.Parameter(torch.log(torch.expm1(torch.exp(
            torch.rand(self.nh) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)))))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.nh + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.nh))
        self.o = nn.Linear(di, d, bias=False)
        hid = int(8 * d / 3)
        self.w12 = nn.Linear(d, 2 * hid, bias=False)
        self.w3 = nn.Linear(hid, d, bias=False)

    def forward(self, x):
        b, l, d = x.shape
        z, xbc, dt = self.inp(self.n1(x)).split([self.di, self.di + 2 * self.n, self.nh], -1)
        xbc = F.silu(self.conv(xbc.transpose(1, 2))[..., :l].transpose(1, 2))   # tích-chập NHÂN-QUẢ
        xs, B, C = xbc.split([self.di, self.n, self.n], -1)
        dt = F.softplus(dt.float() + self.dt_bias)
        A = -torch.exp(self.A_log.float())
        xs = xs.reshape(b, l, self.nh, self.hdim)
        y = ssd_scan(xs.contiguous(), dt.contiguous(), A, B.contiguous(), C.contiguous())
        y = y + xs * self.D[None, None, :, None].to(xs.dtype)
        y = self.ny(y.reshape(b, l, self.di) * F.silu(z))
        x = x + self.o(y)
        g, u = self.w12(self.n2(x)).chunk(2, -1)
        return x + self.w3(F.silu(g) * u)


def make_block(kind, d, heads):
    return MambaBlock(d, heads) if kind == "m" else Block(d, heads)


class LoopLM(nn.Module):
    def __init__(self, V, d=256, heads=8, P=2, C=2, K=2, arch=""):
        """arch: chuỗi 'a'/'m' cho từng khối mở-đầu+lõi+kết (a=attention, m=Mamba2); rỗng = toàn attention."""
        super().__init__()
        arch = arch or "a" * (P + C + K)
        assert len(arch) == P + C + K, "arch phải dài đúng P+C+K"
        self.emb = nn.Embedding(V, d)
        self.pre = nn.ModuleList([make_block(arch[i], d, heads) for i in range(P)])
        self.core = nn.ModuleList([make_block(arch[P + i], d, heads) for i in range(C)])
        self.post = nn.ModuleList([make_block(arch[P + C + i], d, heads) for i in range(K)])
        self.norm = nn.RMSNorm(d)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, ids, r=1):
        e = self.emb(ids)
        for b in self.pre:
            e = b(e)
        s = torch.zeros_like(e)
        for _ in range(r):                                   # LÕI LẶP — nhắc lại đề mỗi vòng
            s = s + e
            for b in self.core:
                s = b(s)
        for b in self.post:
            s = b(s)
        return self.norm(s) @ self.emb.weight.T


# ── Lịch học ─────────────────────────────────────────────────────────────────────────
def grow_core(m, n_new, dev):
    """'LỚN DẦN': thêm tầng lõi; chiếu-ra (o, w3) = 0 → tầng mới là ĐỒNG-NHẤT lúc chèn, không phá cái đã học."""
    d, h = m.emb.weight.shape[1], m.core[0].h
    new = []
    for _ in range(n_new):
        b = Block(d, h).to(dev)
        nn.init.zeros_(b.o.weight); nn.init.zeros_(b.w3.weight)
        m.core.append(b); new += list(b.parameters())
    return new


def level_weights(schedule, step, steps, review=0.3, levels=5):
    """Phân-phối chọn lớp cho lô ở bước `step` → (giai-đoạn, trọng-số từng lớp 1..levels)."""
    if schedule == "random":
        return 0, [1 / levels] * levels
    stage = min(levels - 1, step * levels // steps)          # 0..4 = lớp 1..5
    w = [0.0] * levels
    if schedule == "curr" or stage == 0:
        w[stage] = 1.0
    else:                                                    # curr_review: 70% lớp hiện tại + 30% ôn lớp trước
        w[stage] = 1 - review
        for j in range(stage):
            w[j] = review / stage
    return stage, w


# ── Đánh-giá ─────────────────────────────────────────────────────────────────────────
RANDPOS = {"max": 0}


def rand_positions(n, rng=random):
    """n vị-trí tăng dần lấy ngẫu-nhiên trong [0, RANDPOS.max) — cùng một phân-bố khi train và khi chấm."""
    return torch.tensor(sorted(rng.sample(range(RANDPOS["max"]), n)), dtype=torch.float32)


@torch.no_grad()
def evaluate(m, v, rows, r, dev, max_new=170, bs=100):
    """Giải tham-lam (greedy) toàn-bộ đề; đúng = 'Đáp số: N' khớp tuyệt-đối."""
    m.eval()
    ok = 0
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        seqs = [[BOS] + v.enc("Đề: " + x["q"] + "\nGiải: ") for x in chunk]
        lens = torch.tensor([len(s) for s in seqs], device=dev)
        L = max(len(s) for s in seqs) + max_new
        ids = torch.full((len(seqs), L), PAD, dtype=torch.long, device=dev)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s)
        done = torch.zeros(len(seqs), dtype=torch.bool, device=dev)
        start = lens.clone()
        if RANDPOS["max"]:                                   # một bộ vị-trí cố-định cho cả lần giải (nhất-quán giữa các bước)
            POS["t"] = rand_positions(min(L, RANDPOS["max"]), random.Random(i))
        for _ in range(max_new):                             # đệm PHẢI + nhân-quả → đệm không ảnh-hưởng token thật
            W = int(lens.max())
            lo = m(ids[:, :W], r)
            nxt = lo[torch.arange(len(seqs), device=dev), lens - 1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ids[torch.arange(len(seqs), device=dev), lens] = nxt
            lens = lens + (~done).long()
            done = done | (nxt == EOS)
            if bool(done.all()):
                break
        for j, x in enumerate(chunk):
            out = v.dec(ids[j, start[j]:lens[j]].tolist())
            tail = out.split("Đáp số:")[-1].strip().rstrip(".") if "Đáp số:" in out else ""
            ok += tail == str(x["ans"])
    POS["t"] = None
    m.train()
    if dev == "cuda":
        torch.cuda.empty_cache()                             # trả bộ-nhớ chấm bài → nhiều thí-nghiệm chung 1 GPU không tràn
    return ok / len(rows)


MAX_NEW = 170


def eval_all(m, v, tests, rs, dev):
    return {f"r{r}": {f"lop{lv}": round(evaluate(m, v, tests[lv], r, dev, max_new=MAX_NEW), 4) for lv in tests} for r in rs}


# ── Train ────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", choices=["random", "curr", "curr_review", "mastery"], required=True)
    ap.add_argument("--answer-only", action="store_true", help="NGHĨ THẦM: bỏ lời giải, chỉ ghi 'Đáp số: N'")
    ap.add_argument("--pass-acc", type=float, default=0.95, help="mastery: lên lớp khi bài kiểm-tra nhỏ đạt mức này")
    ap.add_argument("--check-every", type=int, default=200)
    ap.add_argument("--stage-cap", type=int, default=2400, help="mastery: kẹt quá số bước này thì vẫn cho lên lớp")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--P", type=int, default=2)
    ap.add_argument("--C", type=int, default=2)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--r-train", type=int, default=4, help="số vòng lõi khi train: ngẫu-nhiên 1..r-train")
    ap.add_argument("--r-eval", default="1,2,4,8")
    ap.add_argument("--n-test", type=int, default=300, help="số đề kiểm-tra mỗi lớp (≤500)")
    ap.add_argument("--review", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plasticity", choices=["none", "lrdecay", "freeze"], default="none",
                    help="phân-tầng độ mềm-dẻo TỪ LỚP 2 (lớp 1 = giai-đoạn then-chốt, mọi tầng học tự-do): "
                         "lrdecay = tốc-độ học tầng dưới ×0,01 → tầng trên ×1; freeze = mỗi lần lên lớp khoá thêm 1 khối từ dưới")
    ap.add_argument("--preschool", type=int, default=0,
                    help="MẪU-GIÁO: N bước đầu (tính trong ngân-sách) chỉ 'đọc' đề của MỌI lớp (LM trên đề, không lời-giải)")
    ap.add_argument("--freeze-bottom", action="store_true",
                    help="hết mẫu-giáo → khoá VĨNH-VIỄN bảng-từ + khối mở-đầu (đặc-trưng chung)")
    ap.add_argument("--expand", type=int, default=0,
                    help="MỖI LỚP MỌC KHỐI MỚI: lên lớp → khoá mọi khối cũ, thêm N khối (đồng-nhất lúc chèn) trên đỉnh")
    ap.add_argument("--lr-sched", choices=["cos", "stage", "wsd"], default="cos",
                    help="cos = cosine cả quá-trình (vòng 1-4) · stage = MỖI LỚP MỘT CHU-KỲ (khởi-động lại) · "
                         "wsd = phẳng, chỉ giảm 10%% cuối — để lớp khó không rơi vào lúc tốc-độ học ≈ 0")
    ap.add_argument("--quote", action="store_true", help="lời-giải TRÍCH CÂU đề mỗi bước (theo dõi tiến-trình)")
    ap.add_argument("--index", choices=["", "quote", "plain"], default="",
                    help="nhãn câu điểm-bắt-đầu-ngẫu-nhiên: quote = nhãn + trích câu, plain = chỉ nhãn")
    ap.add_argument("--rand-pos", type=int, default=0, help="vị-trí ngẫu-nhiên trong [0, N) khi train VÀ chấm (0 = tắt)")
    ap.add_argument("--rewrite", action="store_true", help="lời-giải VIẾT LẠI ĐỀ CÒN LẠI sau mỗi bước")
    ap.add_argument("--distract", type=int, default=0, help="chèn 0..N câu THỪA vào đề train (lời-giải trích câu)")
    ap.add_argument("--only-level", type=int, default=0, help="chỉ train MỘT lớp (vd 5) — lớp dưới có phải nền-tảng bắt-buộc?")
    ap.add_argument("--n-train", type=int, default=0, help="ÍT DỮ-LIỆU: chỉ dùng N bài train đầu mỗi lớp (0 = tất-cả)")
    ap.add_argument("--arch", default="", help="loại từng khối mở-đầu+lõi+kết: a=attention, m=Mamba2 (vd mmmmmm, mammmm)")
    ap.add_argument("--grow", default="", help="số tầng lõi theo 5 giai-đoạn, vd 1,1,2,2,2 (thêm tầng khi lên lớp)")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    global MAX_NEW
    RANDPOS["max"] = a.rand_pos
    if a.quote or a.index or a.distract:
        MAX_NEW = 520
    if a.rewrite:
        MAX_NEW = 1500
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    v = Vocab()
    raw = {lv: load_level(lv, "train") for lv in range(1, 6)}
    if a.distract:
        drng = random.Random(1000 + a.seed)
        raw = {lv: [distract_row(r, drng, a.distract) for r in rows] for lv, rows in raw.items()}
    if a.index:
        raw = {lv: [index_row(r, a.index == "quote") for r in rows] for lv, rows in raw.items()}
    dev_rows = {lv: raw[lv][-200:] for lv in raw}            # tập XÉT LÊN LỚP — tách khỏi train, KHÔNG đụng test
    train = {lv: [encode_example(v, r, a.answer_only, a.quote, a.rewrite) for r in raw[lv][:-200][: a.n_train or None]] for lv in raw}
    tests = {lv: load_level(lv, "test")[:a.n_test] for lv in range(1, 7)}
    if a.index:
        tests = {lv: [index_row(r, a.index == "quote") for r in rows] for lv, rows in tests.items()}
    reading = []                                             # mẫu-giáo: chỉ ĐỀ của mọi lớp, học đọc (loss trên mọi ký-tự)
    if a.preschool:
        for lv in raw:
            for r in raw[lv][:-200]:
                x = [BOS] + v.enc("Đề: " + r["q"]) + [EOS]
                reading.append((x, [-100] + x[1:]))
    grow = [int(x) for x in a.grow.split(",")] if a.grow else None
    m = LoopLM(len(v.itos), a.d, 8, a.P, grow[0] if grow else a.C, a.K, a.arch).to(dev)
    npar = sum(p.numel() for p in m.parameters())
    # nhóm tham-số theo KHỐI (dưới → trên) để phân-tầng độ mềm-dẻo; emb dùng chung với tầng xuất → luôn học tự-do
    units = list(m.pre) + list(m.core) + list(m.post)
    groups = [{"params": list(m.emb.parameters()) + list(m.norm.parameters()), "mult": 1.0, "unit": -1}]
    groups += [{"params": list(u.parameters()), "mult": 1.0, "unit": i} for i, u in enumerate(units)]
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def set_plasticity(stage):
        """stage ≥ 1 (đã qua lớp 1): áp phân-tầng. Hệ-số nhân tốc-độ học theo vị-trí khối."""
        if a.plasticity == "none" or stage < 1:
            return
        n = len(units)
        for g in opt.param_groups:
            u = g.get("unit", -1)
            if u < 0:
                continue
            if a.plasticity == "lrdecay":
                g["mult"] = 0.01 ** (1 - u / max(1, n - 1))   # khối dưới cùng ×0,01 … khối trên cùng ×1
            else:
                g["mult"] = 0.0 if u < stage else 1.0          # sau lớp k: khoá k khối dưới cùng
    def freeze(mod):
        for p_ in mod.parameters():
            p_.requires_grad_(False)

    def expand_top():
        """lên lớp: khoá mọi khối cũ (lõi + kết + chuẩn-hoá cuối) → thêm a.expand khối mới trên đỉnh."""
        for mod in list(m.core) + list(m.post) + [m.norm]:
            freeze(mod)
        d_, h_ = m.emb.weight.shape[1], m.core[0].h
        for _ in range(a.expand):
            b = Block(d_, h_).to(dev)
            nn.init.zeros_(b.o.weight); nn.init.zeros_(b.w3.weight)
            m.post.append(b)
            opt.add_param_group({"params": list(b.parameters()), "mult": 1.0, "unit": -2})
        print(f"  ⇑ mọc khối: kết {len(m.post)} khối · đang học "
              f"{sum(p_.numel() for p_ in m.parameters() if p_.requires_grad)/1e6:.2f}M tham-số", flush=True)
    def lr_at(s):
        warm = min(1, (s + 1) / 200)
        if a.lr_sched == "wsd":
            return a.lr * warm * min(1.0, (a.steps - s) / (0.1 * a.steps))
        if a.lr_sched == "stage" and a.schedule in ("curr", "curr_review") and s >= a.preschool:
            L = (a.steps - a.preschool) / 5                 # mỗi lớp: khởi-động 100 bước + cosine về 10%
            t = (s - a.preschool) % L
            return a.lr * min(1, (t + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / L)))
        return a.lr * warm * 0.5 * (1 + math.cos(math.pi * s / a.steps))
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else torch.autocast("cpu", enabled=False)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    rs = [int(x) for x in a.r_eval.split(",")]
    print(f"[{a.tag}] {a.schedule} · {npar/1e6:.2f}M tham-số · {dev} · {a.steps} bước × lô {a.bs}", flush=True)
    log = {"args": vars(a), "params": npar, "stages": [], "curve": []}
    T0, last_stage, layer_steps = time.time(), 0, 0
    cur, stage_start = 1, 0                                  # mastery: lớp đang học
    log["promotions"] = []
    for step in range(a.steps):
        if a.schedule == "mastery":
            if step and step % a.check_every == 0 and cur < 5:
                acc = evaluate(m, v, dev_rows[cur][:100], min(2, a.r_train), dev)
                if acc >= a.pass_acc or step - stage_start >= a.stage_cap:
                    log["promotions"].append({"from": cur, "step": step, "dev_acc": acc,
                                              "reason": "đỗ" if acc >= a.pass_acc else "hết hạn"})
                    print(f"  ⇧ lên lớp {cur+1} @{step} (bài kiểm-tra lớp {cur}: {100*acc:.0f}%)", flush=True)
                    cur, stage_start = cur + 1, step
            w = [0.0] * 5
            w[cur - 1] = 1.0 if cur == 1 else 1 - a.review
            for j in range(cur - 1):
                w[j] = a.review / (cur - 1)
            stage = cur - 1
        elif step < a.preschool:
            stage, w = 0, None
        else:
            stage, w = level_weights(a.schedule, step - a.preschool, a.steps - a.preschool, a.review)
        if a.preschool and step == a.preschool:
            print(f"  ✎ hết mẫu-giáo @{step}" + (" → khoá bảng-từ + khối mở-đầu" if a.freeze_bottom else ""), flush=True)
            if a.freeze_bottom:
                freeze(m.emb); [freeze(b) for b in m.pre]
        gstage = min(4, step * 5 // a.steps)                 # giai-đoạn theo thời-gian (dùng cho lớn-dần)
        if a.schedule in ("curr", "curr_review") and stage != last_stage:   # hết một giai-đoạn → đo (học & quên)
            res = eval_all(m, v, tests, [1, a.r_train], dev)
            log["stages"].append({"after_level": last_stage + 1, "step": step, "acc": res})
            print(f"  ✦ xong lớp {last_stage+1} @{step}: {res}", flush=True)
            last_stage = stage
            set_plasticity(stage)
            if a.expand:
                expand_top()
            if a.plasticity != "none":
                print(f"  ⚙ mềm-dẻo theo khối: {[round(g['mult'], 3) for g in opt.param_groups if g.get('unit', -1) >= 0]}", flush=True)
        if grow and len(m.core) < grow[gstage]:
            new = grow_core(m, grow[gstage] - len(m.core), dev)
            opt.add_param_group({"params": new})
            print(f"  ⇑ lớn dần @{step}: lõi {len(m.core)} tầng", flush=True)
        for g in opt.param_groups:
            g["lr"] = lr_at(step) * g.get("mult", 1.0)
        if w is None:                                        # mẫu-giáo
            lv, batch = [0], [rng.choice(reading) for _ in range(a.bs)]
        else:
            if a.only_level:
                w = [float(l == a.only_level) for l in range(1, 6)]
            lv = rng.choices(range(1, 6), weights=w, k=a.bs)
            batch = [rng.choice(train[l]) for l in lv]
        Lm = max(len(x) for x, _ in batch)
        X = torch.full((a.bs, Lm), PAD, dtype=torch.long)
        Y = torch.full((a.bs, Lm), -100, dtype=torch.long)
        for j, (x, y) in enumerate(batch):
            X[j, :len(x)] = torch.tensor(x); Y[j, :len(y)] = torch.tensor(y)
        X, Y = X.to(dev), Y.to(dev)
        r = rng.randint(1, a.r_train)
        layer_steps += len(m.pre) + len(m.core) * r + len(m.post)   # đại-diện compute (tầng × bước)
        if a.rand_pos:
            POS["t"] = rand_positions(X.shape[1] - 1, rng)
        with amp:
            lo = m(X[:, :-1], r)
        POS["t"] = None
        loss = F.cross_entropy(lo.float().reshape(-1, lo.size(-1)), Y[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p_ for p_ in m.parameters() if p_.requires_grad], 1.0)
        opt.step()
        if step % 250 == 0:
            log["curve"].append({"step": step, "loss": round(float(loss), 4)})
            print(f"  bước {step} · loss {float(loss):.3f} · r={r} · lớp-lô {lv[0]} · {time.time()-T0:.0f}s", flush=True)
    final = eval_all(m, v, tests, rs, dev)
    log["final"] = final
    log["sec"] = round(time.time() - T0)
    log["layer_steps"] = layer_steps
    log["params_final"] = sum(p.numel() for p in m.parameters())
    print(f"[{a.tag}] CUỐI ({log['sec']}s): {json.dumps(final)}", flush=True)
    os.makedirs("out/curriculum", exist_ok=True)
    json.dump(log, open(f"out/curriculum/{a.tag}.json", "w"), ensure_ascii=False, indent=1)
    torch.save({"args": vars(a), "state": m.state_dict()}, f"out/curriculum/{a.tag}.pt")


if __name__ == "__main__":
    main()
