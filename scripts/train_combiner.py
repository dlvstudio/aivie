"""§TK_096 — COMBINER: nhiều expert nơ-ron ĐÓNG BĂNG → đầu vào cho một mạng chung
tổng-hợp (chỉ combiner backprop). Kiểm chứng ý tưởng "train riêng → freeze → combiner học".

Kiến trúc: K expert (Mamba/hybrid) ĐÓNG BĂNG cho (logits_i, hidden_i). Combiner NHẬN
hidden các expert làm ĐẦU VÀO → Linear+GELU + 1 tầng self-attention nhân-quả → hidden
tổng-hợp z. Ba CHẾ-ĐỘ trộn (--mode):
  • prob   : gate per-token trên z → trộn XÁC-SUẤT p=Σ g_i·softmax(logits_i). Chỉ CHỌN/PHA
             đầu-ra expert (không tạo dự-đoán mới), nhưng xuất-phát ở chất-lượng expert.
  • hidden : z → ĐẦU-RA RIÊNG head(z)→logits. TỰ TỔNG-HỢP dự-đoán mới từ tầng-ẩn expert
             (biểu-đạt mạnh hơn) nhưng head khởi-tạo ngẫu-nhiên → phải học lại ánh-xạ vocab.
  • hybrid : nền = log(mixture prob) + hiệu-chỉnh head(z). head zero-init → BẮT ĐẦU = prob,
             rồi cộng thêm phần tổng-hợp. Lấy cả hai: khởi-tạo tốt + biểu-đạt mạnh.

So sánh: B0 (expert đơn) · B1 (trộn đều, KHÔNG học) · B2 (combiner học). B2<B1 → routing có ích.

    PYTHONPATH=src .venv/bin/python scripts/train_combiner.py \
        --experts checkpoints/mamba_d256_2k.pt checkpoints/expert_qa_d256.pt \
        --data checkpoints/mix_wiki_qa.npz --mode hybrid --steps 600
"""
import sys, argparse, time, math, re
sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from train_mamba import Mamba, AttnBlock, RMSNorm
T0 = time.time()


def load_expert(ckpt, V, dev):
    obj = torch.load(ckpt, map_location="cpu"); c = obj["cfg"]; state = obj["state"]
    al = sorted({int(m.group(1)) for k in state
                 for m in [re.match(r"blocks\.(\d+)\.attn\.", k)] if m})
    m = Mamba(V, c["d"], c["layers"], None, c["d_state"], al, c.get("heads", 4)).to(dev)
    m.load_state_dict(state); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)                            # ĐÓNG BĂNG
    tag = f"hybrid(attn@{al})" if al else "mamba-thuần"
    return m, c["d"], tag


class KnnExpert:
    """§TK_097 RAG expert tri-thức: truy-xuất kho viwiki → phân-phối token GROUNDED.
    Không tham-số học (retrieval thuần); combiner học TRỌNG-SỐ tin nó qua gate."""
    def __init__(self, path, dev, V, k=16, temp=0.1):
        ds = np.load(path)
        self.keys = F.normalize(torch.tensor(ds["keys"].astype(np.float32), device=dev), dim=-1)  # (N,d)
        self.vals = torch.tensor(ds["vals"].astype(np.int64), device=dev)                          # (N,)
        self.d = int(ds["d"]); self.V = V; self.k = k; self.temp = temp; self.dev = dev

    @torch.no_grad()
    def logits(self, query_h):                             # (b,l,d) → (b,l,V) log-prob grounded
        b, l, d = query_h.shape
        q = F.normalize(query_h.reshape(-1, d), dim=-1); M = q.shape[0]
        out = torch.empty(M, self.V, device=self.dev)
        for i in range(0, M, 128):                         # chunk truy-vấn (giới-hạn bộ-nhớ sim)
            sim = q[i:i + 128] @ self.keys.T               # (ch,N) cosine
            tv, ti = sim.topk(self.k, dim=-1)              # top-k láng-giềng
            w = F.softmax(tv / self.temp, dim=-1)          # trọng-số theo độ gần
            dist = torch.zeros(tv.shape[0], self.V, device=self.dev)
            dist.scatter_add_(1, self.vals[ti], w)         # gom vào token
            out[i:i + tv.shape[0]] = torch.log(dist + 1e-6)
        return out.reshape(b, l, self.V)


class KnnExpertFAISS:
    """§TK_098 RAG expert QUY-MÔ LỚN qua FAISS IVFPQ (ANN) — vượt trần brute-force 2M.
    Cùng giao-diện .logits(query_h) như KnnExpert; tìm gần-đúng trên CPU (faiss-cpu)."""
    def __init__(self, prefix, dev, V, k=16, temp=0.1):
        import faiss
        faiss.omp_set_num_threads(1)                       # §TK_098 tránh segfault xung-đột OpenMP với PyTorch
        self.index = faiss.read_index(f"{prefix}.index")
        self.vals = np.load(f"{prefix}_vals.npy")
        self.d = int(np.load(f"{prefix}_meta.npy")[0])
        self.V = V; self.k = k; self.temp = temp; self.dev = dev

    @torch.no_grad()
    def logits(self, query_h):                             # (b,l,d) → (b,l,V) log-prob grounded
        import torch.nn.functional as _F
        b, l, d = query_h.shape
        q = _F.normalize(query_h.reshape(-1, d), dim=-1).cpu().numpy().astype(np.float32)
        sim, idx = self.index.search(q, self.k)            # (M,k) IP + chỉ-số (FAISS, CPU)
        toks = torch.tensor(self.vals[idx], dtype=torch.long, device=self.dev)          # (M,k)
        w = torch.softmax(torch.tensor(sim, device=self.dev) / self.temp, dim=-1)        # (M,k)
        dist = torch.zeros(q.shape[0], self.V, device=self.dev)
        dist.scatter_add_(1, toks, w)
        return torch.log(dist + 1e-6).reshape(b, l, self.V)


def make_knn(path, dev, V, k, temp):
    """§TK_098 chọn retriever: .index → FAISS (ANN lớn); .npz → brute-force."""
    return KnnExpertFAISS(path[:-6], dev, V, k, temp) if path.endswith(".index") \
        else KnnExpert(path, dev, V, k, temp)


class Combiner(nn.Module):
    """§TK_096 mạng chung: nhận hidden expert → z tổng-hợp → trộn theo --mode. Trả (logp, gate)."""
    def __init__(self, dims, dh=256, heads=4, mode="prob", V=32002):
        super().__init__()
        self.K = len(dims); self.mode = mode; self.V = V
        self.inp = nn.Linear(sum(dims), dh)
        self.attn = AttnBlock(dh, heads)                   # "mạng chú ý từ input"
        self.n1 = RMSNorm(dh); self.n2 = RMSNorm(dh)
        self.ff = nn.Sequential(nn.Linear(dh, dh * 2), nn.GELU(), nn.Linear(dh * 2, dh))
        if mode in ("prob", "hybrid"):
            self.gate = nn.Linear(dh, self.K)
            nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)  # bắt đầu = gate đều (=B1)
        if mode in ("hidden", "hybrid"):
            self.head = nn.Linear(dh, V)                   # đầu-ra riêng của combiner
            if mode == "hybrid":                           # zero-init → bắt đầu = prob-mixture
                nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def _z(self, hiddens):
        z = F.gelu(self.inp(torch.cat(hiddens, dim=-1)))   # (b,l,dh)
        z = z + self.attn(self.n1(z))
        z = z + self.ff(self.n2(z))
        return z

    def forward(self, hiddens, logits):
        z = self._z(hiddens)
        eps = 1e-9
        if self.mode == "hidden":
            return F.log_softmax(self.head(z), dim=-1), None
        g = F.softmax(self.gate(z), dim=-1)                # (b,l,K)
        probs = torch.stack([F.softmax(lo, dim=-1) for lo in logits], dim=-1)  # (b,l,V,K)
        p_mix = (probs * g.unsqueeze(2)).sum(-1)           # (b,l,V) mixture Σ=1
        if self.mode == "prob":
            return torch.log(p_mix + eps), g
        # hybrid: nền mixture + hiệu-chỉnh head(z)
        return F.log_softmax(torch.log(p_mix + eps) + self.head(z), dim=-1), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", nargs="+", required=True)
    ap.add_argument("--knn-datastore", default=None, help="§TK_097 thêm RAG expert kNN (query = hidden expert[0])")
    ap.add_argument("--knn-k", type=int, default=16)
    ap.add_argument("--knn-temp", type=float, default=0.1)
    ap.add_argument("--data", default="checkpoints/enc_data_viwiki.npz")
    ap.add_argument("--mode", choices=["prob", "hidden", "hybrid"], default="prob")
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--dh", type=int, default=256)
    ap.add_argument("--out", default="checkpoints/combiner.pt")
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    d = np.load(a.data); X = d["X"][:, :a.ctx]; L = d["L"]; V = d["E"].shape[0]
    keep = np.where(L >= a.ctx // 2)[0]
    rng = np.random.default_rng(0); rng.shuffle(keep)
    ho, tr = keep[:400], keep[400:]
    Xt = torch.tensor(X, dtype=torch.long)

    experts, dims, tags = [], [], []
    for ck in a.experts:
        m, dd, tg = load_expert(ck, V, dev); experts.append(m); dims.append(dd); tags.append(tg)
    knn = None
    if a.knn_datastore:                                    # §TK_097 RAG expert tri-thức (query=hidden expert[0])
        knn = make_knn(a.knn_datastore, dev, V, a.knn_k, a.knn_temp)
        dims.append(dims[0]); tags.append(f"RAG-kNN(k={a.knn_k})")   # hidden slot = tái dùng hidden expert[0]
    comb = Combiner(dims, a.dh, mode=a.mode, V=V).to(dev)
    ncp = sum(p.numel() for p in comb.parameters() if p.requires_grad)
    print(f"[{time.time()-T0:.0f}s] dev={dev} · {len(experts)} expert ĐÓNG BĂNG: {tags}", flush=True)
    print(f"[{time.time()-T0:.0f}s] combiner MODE={a.mode} train-được {ncp/1e6:.3f}M (dh={a.dh}) · V={V}", flush=True)
    opt = torch.optim.AdamW(comb.parameters(), lr=a.lr)

    @torch.no_grad()
    def run_experts(ids):
        los, hs = [], []
        for m in experts:
            lo, h = m(ids, return_hidden=True); los.append(lo); hs.append(h)
        if knn is not None:                                # §TK_097 RAG expert: logits truy-xuất + tái dùng hidden expert[0]
            los.append(knn.logits(hs[0])); hs.append(hs[0])
        return los, hs

    nK = len(dims)                                         # số expert (gồm slot RAG-kNN nếu có)
    @torch.no_grad()
    def evaluate(cap=64):
        comb.eval()
        b0 = [0.0] * nK; b1 = 0.0; b2 = 0.0
        gsum = torch.zeros(nK); has_g = False; c = 0
        for i in range(0, min(len(ho), cap), a.bs):
            ids = Xt[ho[i:i + a.bs]].to(dev); tgt = ids[:, 1:]
            los, hs = run_experts(ids)
            for k, lo in enumerate(los):
                b0[k] += float(F.cross_entropy(lo[:, :-1].reshape(-1, V), tgt.reshape(-1)))
            pmean = torch.stack([F.softmax(lo, -1) for lo in los], 0).mean(0)
            b1 += float(F.nll_loss(torch.log(pmean[:, :-1] + 1e-9).reshape(-1, V), tgt.reshape(-1)))
            logp, g = comb(hs, los)
            b2 += float(F.nll_loss(logp[:, :-1].reshape(-1, V), tgt.reshape(-1)))
            if g is not None:
                gsum += g.mean(dim=(0, 1)).cpu(); has_g = True
            c += 1
        comb.train(); ln2 = math.log(2)
        gout = (gsum / c).tolist() if has_g else None
        return ([x / c / ln2 for x in b0], b1 / c / ln2, b2 / c / ln2, gout)

    def gstr(g):
        return "n/a" if g is None else str([f"{x:.2f}" for x in g])
    b0, b1, b2, g = evaluate()
    print(f"[{time.time()-T0:.0f}s] TRƯỚC — B0={[f'{x:.3f}' for x in b0]} · B1(trộn đều)={b1:.3f} · "
          f"B2(combiner)={b2:.3f} · gate={gstr(g)}", flush=True)
    for step in range(1, a.steps + 1):
        ids = Xt[tr[rng.integers(0, len(tr), a.bs)]].to(dev); tgt = ids[:, 1:]
        los, hs = run_experts(ids)
        logp, _ = comb(hs, los)
        loss = F.nll_loss(logp[:, :-1].reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(comb.parameters(), 1.0)
        opt.step()
        if step % a.eval_every == 0:
            b0, b1, b2, g = evaluate(); best = min(b0)
            mark = "✓ THẮNG" if b2 < min(b1, best) - 1e-3 else ("~ ngang" if b2 < best else "✗ thua")
            print(f"[{time.time()-T0:.0f}s] step {step} loss {loss.item():.3f} | B1={b1:.3f} B2={b2:.3f} "
                  f"(expert tốt nhất {best:.3f}) gate={gstr(g)} {mark}", flush=True)
            torch.save({"state": comb.state_dict(), "dims": dims, "dh": a.dh,
                        "mode": a.mode, "experts": a.experts,
                        "knn": ({"datastore": a.knn_datastore, "k": a.knn_k, "temp": a.knn_temp} if knn else None)}, a.out)
    torch.save({"state": comb.state_dict(), "dims": dims, "dh": a.dh,
                "mode": a.mode, "experts": a.experts,
                        "knn": ({"datastore": a.knn_datastore, "k": a.knn_k, "temp": a.knn_temp} if knn else None)}, a.out)
    print(f"[{time.time()-T0:.0f}s] LƯU {a.out}", flush=True)
    print("### hết", flush=True)


if __name__ == "__main__":
    main()
