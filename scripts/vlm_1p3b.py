"""VLM — model của ta ĐỌC ẢNH (như Gemma 26B/31B): ảnh và chữ đan xen trong CÙNG một chuỗi, MỘT checkpoint.

    Ảnh ──► tách ô 384px (+ 1 ảnh thu nhỏ toàn cảnh) ──► SigLIP2 so400m (mắt, 27×27 mảnh/ô)
        ──► gộp 3×3 mảnh kề nhau (pixel-shuffle: 729 → 81 token/ô) ──► bộ chiếu MLP ──► token ảnh (d=2048)
    Chuỗi:  "Hỏi: " <ảnh> [toàn cảnh 81] <hàng> [ô 1,1][ô 1,2]… <hàng> […] </ảnh> "\\n" câu hỏi "\\n" "Đáp: " …
    Thân 1.3B (đóng băng) + 3 expert cũ (đóng băng) + EXPERT THỊ GIÁC (GatedFFN mới, cổng tự mở ở chỗ cần nhìn ảnh).

Bài học TK_105 (token mềm phình ×3,4 → BLEU 0): đầu ra bộ chiếu RMSNorm rồi nhân ĐÚNG cỡ của embedding chữ (emb_rms) →
token ảnh luôn cùng độ lớn với token chữ ngay từ bước đầu.

Ô chữ trong ảnh chụp màn hình: ô 384px gốc + ghép ô theo tỉ lệ khung (kiểu InternVL) → 1080p ≈ 4×2–5×3 ô ≈ 0,7–1,3K token.
Trong chuỗi, chỗ token ảnh đánh dấu bằng id ÂM (-1); embed_ids() thay bằng véc-tơ ảnh.

    # M4 — kiểm đường ống (V0)
    .venv/bin/python scripts/vlm_1p3b.py smoke   --base checkpoints/hf_hybrid/hybrid_1p3b_assistant_v5.pt --image t.png
    .venv/bin/python scripts/vlm_1p3b.py overfit --base checkpoints/hf_hybrid/hybrid_1p3b_assistant_v5.pt   # 8 ảnh tự vẽ
    # box — train (V2): stage 1 = chỉ bộ chiếu · stage 2 = bộ chiếu + expert thị giác (+ vài tầng cuối của mắt)
    python scripts/vlm_1p3b.py train --stage 1 --data data/vlm/align.jsonl --base ckpt/hybrid_1p3b_assistant_v5.pt …
    python scripts/vlm_1p3b.py pack  --base …assistant_v5.pt --vision ckpt/vlm/s2_vision.pt --out …vlm_v1.pt
"""
from __future__ import annotations
import argparse, io, json, math, os, random, re, sys, time
import concurrent.futures as cf
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")]
import torch
import torch.utils.checkpoint                  # NGC torch không tự nạp submodule này (Mac thì có)
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

TOWER = "checkpoints/hf_hybrid/vision/siglip2_so400m_384_bf16.safetensors"
TILE, SHUF = 384, 3                      # 384/14 = 27 mảnh/cạnh → gộp 3×3 → 9×9 = 81 token/ô (V1)
CFG = {"shuf": 3, "thumb": True}         # V4: gộp 2×2 (đệm 27→28 → 14×14 = 196 token/ô), bỏ ảnh toàn cảnh khi nhiều ô


def set_cfg(shuf=3, thumb=True):
    CFG["shuf"], CFG["thumb"] = int(shuf), bool(thumb)
IMG = -1                                 # id giữ chỗ cho token ảnh trong chuỗi
from expert_ffn_1p3b import POS         # điểm cắm expert = các khối attention (CÙNG một list, load_base cập nhật theo thân)


# ── ảnh → các ô ───────────────────────────────────────────────────────────────────────
def grid_for(w, h, max_tiles):
    """chọn lưới (cột, hàng) có tỉ lệ gần tỉ lệ ảnh nhất; hoà → lưới lớn hơn nếu ảnh đủ to (kiểu InternVL)."""
    ar, best, bd = w / h, (1, 1), float("inf")
    cands = sorted({(c, r) for c in range(1, max_tiles + 1) for r in range(1, max_tiles + 1) if c * r <= max_tiles},
                   key=lambda g: g[0] * g[1])
    for c, r in cands:
        d = abs(ar - c / r)
        if d < bd - 1e-9:
            best, bd = (c, r), d
        elif abs(d - bd) < 1e-9 and w * h > 0.5 * TILE * TILE * c * r:
            best = (c, r)
    return best


def tiles_of(img: Image.Image, max_tiles=12):
    """→ (danh sách ô 384×384, (cột, hàng)). Ảnh nhỏ / max_tiles=1 → 1 ô. Nhiều ô → thêm ảnh thu nhỏ toàn cảnh ở đầu."""
    img = img.convert("RGB")
    w, h = img.size
    if max_tiles <= 1 or (w <= TILE * 1.2 and h <= TILE * 1.2):
        return [img.resize((TILE, TILE), Image.BILINEAR)], (1, 1)
    c, r = grid_for(w, h, max_tiles)
    if c * r == 1:
        return [img.resize((TILE, TILE), Image.BILINEAR)], (1, 1)
    big = img.resize((c * TILE, r * TILE), Image.BICUBIC)
    out = [img.resize((TILE, TILE), Image.BICUBIC)] if CFG["thumb"] else []
    for j in range(r):
        for i in range(c):
            out.append(big.crop((i * TILE, j * TILE, (i + 1) * TILE, (j + 1) * TILE)))
    return out, (c, r)


def pixels(tiles):
    import numpy as np
    a = torch.from_numpy(np.stack([np.asarray(t, dtype=np.uint8) for t in tiles])).permute(0, 3, 1, 2).float()
    return (a / 255.0 - 0.5) / 0.5                                  # chuẩn SigLIP: mean = std = 0,5


def n_tokens(grid):
    """số token ảnh: <ảnh> + [toàn cảnh k] (+ với nhiều ô: <hàng> trước mỗi hàng + k/ô) + </ảnh>; k = ⌈27/gộp⌉²."""
    c, r = grid
    k = (-(-(TILE // 14) // CFG["shuf"])) ** 2
    if c * r == 1:
        return 2 + k
    return 2 + (k if CFG["thumb"] else 0) + r + c * r * k


def load_image(src):
    if isinstance(src, Image.Image):
        return src
    if isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(src))
    if isinstance(src, str) and src.startswith("data:"):
        import base64
        return Image.open(io.BytesIO(base64.b64decode(src.split(",", 1)[1])))
    return Image.open(src)


# ── mắt + bộ chiếu ────────────────────────────────────────────────────────────────────
def vm(tower):
    """phần thân của mắt (transformers 4: tower.vision_model · transformers 5: chính tower)."""
    return getattr(tower, "vision_model", tower)


def load_tower(path=TOWER, dev="cpu", dtype=torch.bfloat16):
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import SiglipVisionConfig, SiglipVisionModel
    meta = safe_open(path, "pt").metadata()
    cfg = SiglipVisionConfig(**{k: v for k, v in json.loads(meta["vision_config"]).items() if k != "model_type"},
                             vision_use_head=False)
    m = SiglipVisionModel(cfg)
    sd = {k: v for k, v in load_file(path).items() if not k.startswith("head.")}
    missing, extra = vm(m).load_state_dict(sd, strict=False)
    assert not missing and not extra, (missing, extra)
    return m.to(dev, dtype).eval(), json.loads(meta["vision_config"])


class Projector(nn.Module):
    """81×(1152·9) → d. Ra = RMSNorm(·) × emb_rms × gain (gain học được, khởi = 1) → token ảnh cùng cỡ token chữ."""
    def __init__(self, dv=1152, d=2048, emb_rms=1.0, shuf=None):
        super().__init__()
        self.shuf = shuf or CFG["shuf"]
        din = dv * self.shuf * self.shuf
        self.ln = nn.LayerNorm(din)
        self.fc1 = nn.Linear(din, 2 * d)
        self.fc2 = nn.Linear(2 * d, d)
        self.gain = nn.Parameter(torch.ones(d))
        self.special = nn.Parameter(torch.randn(3, d))              # 0 <ảnh> · 1 </ảnh> · 2 <hàng>
        self.register_buffer("emb_rms", torch.tensor(float(emb_rms)))

    def scale(self, y):
        return F.rms_norm(y, (y.shape[-1],)) * self.emb_rms * self.gain

    def forward(self, feats):                                       # (n_ô, 729, dv)
        n, L, dv = feats.shape
        q = self.shuf
        s = int(math.isqrt(L)); g = -(-s // q)
        if g * q != s:                                              # 27 không chia hết cho 2 → đệm lặp hàng/cột cuối
            f = feats.view(n, s, s, dv).permute(0, 3, 1, 2)
            feats = F.pad(f, (0, g * q - s, 0, g * q - s), mode="replicate").permute(0, 2, 3, 1)
        x = feats.reshape(n, g, q, g, q, dv).permute(0, 1, 3, 2, 4, 5).reshape(n, g * g, q * q * dv)
        return self.scale(self.fc2(F.gelu(self.fc1(self.ln(x)))))  # (n_ô, g², d)

    def sp(self, i):
        return self.scale(self.special[i])


class Vision(nn.Module):
    """mắt (SigLIP2, mặc định đóng băng) + bộ chiếu → token ảnh cho từng ảnh."""
    def __init__(self, tower, proj):
        super().__init__()
        self.tower, self.proj = tower, proj

    def feats(self, px):
        dt = next(self.tower.parameters()).dtype
        return self.tower(pixel_values=px.to(dt)).last_hidden_state.float()

    def forward(self, px_list, grids):
        """px_list: [(n_ô_i,3,384,384)] · grids: [(c,r)] → [(n_token_i, d)] theo đúng thứ tự n_tokens()."""
        allp = torch.cat(px_list)
        f = self.proj(self.feats(allp.to(self.proj.fc1.weight.device)))
        out, k = [], 0
        for px, (c, r) in zip(px_list, grids):
            t = f[k:k + len(px)]; k += len(px)
            th = 1 if (c * r == 1 or CFG["thumb"]) else 0      # có ảnh toàn cảnh ở đầu?
            seq = [self.proj.sp(0)[None]] + ([t[0]] if th else [])
            if c * r > 1:
                for j in range(r):
                    seq.append(self.proj.sp(2)[None])
                    seq += [t[th + j * c + i] for i in range(c)]
            seq.append(self.proj.sp(1)[None])
            out.append(torch.cat(seq))
        return out


class LoRA(nn.Module):
    """W x + (B A x)·(α/r) — thân đóng băng, chỉ A, B học (B khởi 0 → lúc đầu y hệt).
    LoRA.on (chung mọi lớp): UI chỉ bật khi hội thoại CÓ ẢNH → chat chữ giữ nguyên thân v5 (V4 gộp hẳn làm SGK 40,7→36%)."""
    on = True
    def __init__(self, base: nn.Linear, r=64, alpha=64):
        super().__init__()
        self.base, self.r, self.s = base, r, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        if not LoRA.on:
            return self.base(x)
        return self.base(x) + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.s

    # lõi CUDA của mamba_ssm đọc THẲNG in_proj.weight / out_proj.weight (không gọi forward) → trả trọng số đã cộng LoRA
    @property
    def weight(self):
        w = self.base.weight
        return w if not LoRA.on else w + (self.B @ self.A).to(w.dtype) * self.s

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features


def lora_targets(trunk, mamba=False):
    """(khối, tên lớp): wq/wk/wv/wo của các khối attention (V4) [+ in_proj/out_proj của các khối Mamba (V5)]."""
    out = [(i, nm) for i in POS for nm in ("wq", "wk", "wv", "wo")]
    if mamba:
        out += [(i, nm) for i in range(len(trunk.blocks)) if i not in POS for nm in ("in_proj", "out_proj")]
    return out


def add_lora(trunk, r, mamba=False, targets=None):
    """LoRA lên các lớp chiếu của thân — nơi model 'nhìn lại' token ảnh để CHÉP chữ."""
    ps = []
    for i, nm in targets or lora_targets(trunk, mamba):
        blk = trunk.blocks[i]
        if not isinstance(getattr(blk, nm), LoRA):
            base = getattr(blk, nm)
            setattr(blk, nm, LoRA(base, r, r).to(base.weight.device))
        ps += [getattr(blk, nm).A, getattr(blk, nm).B]
    return ps


def lora_state(trunk):
    return {f"{i}.{nm}.{k}": getattr(getattr(b, nm), k).detach().to(torch.bfloat16).cpu()
            for i, b in enumerate(trunk.blocks) for nm in ("wq", "wk", "wv", "wo", "in_proj", "out_proj")
            if isinstance(getattr(b, nm, None), LoRA) for k in ("A", "B")}


def load_lora(trunk, sd):
    r = next(v for k, v in sd.items() if k.endswith(".A")).shape[0]
    add_lora(trunk, r, targets=sorted({(int(k.split(".")[0]), k.split(".")[1]) for k in sd}))
    for k, v in sd.items():
        i, nm, ab = k.split(".")
        getattr(getattr(trunk.blocks[int(i)], nm), ab).data.copy_(v.float())


def merge_lora(state, sd):
    """gộp LoRA vào state_dict thân (khoá blocks.{i}.{tên}.weight) → checkpoint không cần lớp LoRA."""
    for k in [k for k in sd if k.endswith(".A")]:
        i, nm, _ = k.split(".")
        A, B = sd[k].float(), sd[f"{i}.{nm}.B"].float()
        w = f"blocks.{i}.{nm}.weight"
        state[w] = (state[w].float() + B @ A).to(state[w].dtype)                  # α = r → hệ số 1


def apply_vision_file(vsd, trunk, vision, sets, c):
    """nạp file bộ chiếu/expert/mắt/LoRA đã train (cấu hình gộp đọc từ file)."""
    vision.proj.load_state_dict(vsd["proj"])
    if "tower" in vsd:
        vm(vision.tower).load_state_dict(vsd["tower"], strict=False)
    if "expert" in vsd:
        ve = vision_expert(c["d"], vsd.get("hid", 4096)); ve.load_state_dict(vsd["expert"]); sets["vision"] = ve
    if vsd.get("lora"):
        load_lora(trunk, vsd["lora"])


def prep_images(srcs, max_tiles=12):
    px, grids = [], []
    for s in srcs:
        t, g = tiles_of(load_image(s), max_tiles)
        px.append(pixels(t)); grids.append(g)
    return px, grids


# ── khuôn chuỗi ───────────────────────────────────────────────────────────────────────
def build_example(msgs, tok, grids, max_len=2048, last_only=False):
    """như train_hybrid_sft.build_example + ảnh: mỗi tin nhắn người dùng có 'n_img' ảnh (lấy lần lượt từ grids) đặt
    ngay sau "Hỏi: ". Trả (ids có IMG, labels)."""
    EOS = tok.token_to_id["<EOS>"]
    ids, lab, gi = [], [], 0
    last_a = max((i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=-1)

    def add(e, train):
        ids.extend(e); lab.extend(e if train else [-100] * len(e))
    for k, m in enumerate(msgs):
        r, c = m["role"], m["content"].strip()
        if r == "system" and c:
            add(tok.encode(c + "\n"), False)
        elif r == "user":
            ni = m.get("n_img", 0)
            if ni:
                add(tok.encode("Hỏi: "), False)
                for _ in range(ni):
                    add([IMG] * n_tokens(grids[gi]), False); gi += 1
                add(tok.encode("\n" + c + "\n"), False)
            else:
                add(tok.encode("Hỏi: " + c + "\n"), False)
        elif r == "assistant" and c:
            tr = (not last_only) or k == last_a
            add(tok.encode("Đáp: "), False)
            add(tok.encode(c), tr)
            ids.append(EOS); lab.append(EOS if tr else -100)
    assert gi == len(grids), (gi, len(grids))
    if len(ids) > max_len:                       # cắt phải; KHÔNG cắt giữa ảnh (mẫu như vậy bị loại ở bộ nạp)
        ids, lab = ids[:max_len], lab[:max_len]
    return ids, lab


def embed_ids(emb, ids, vis):
    """ids (b,L) có IMG → (b,L,d); vis = véc-tơ ảnh nối theo thứ tự xuất hiện (tổng = số IMG)."""
    x = emb(ids.clamp(min=0))
    if vis is not None:
        m = ids < 0
        assert int(m.sum()) == vis.shape[0], (int(m.sum()), vis.shape)
        x = x.masked_scatter(m.unsqueeze(-1), vis.to(x.dtype))
    return x


# ── model đầy đủ (train / overfit) ────────────────────────────────────────────────────
def load_base(path, dev, kernel=True):
    """thân + các bộ expert có sẵn trong checkpoint (mục "experts"), tất cả đóng băng."""
    import train_hybrid_scale as ths
    import expert_ffn_1p3b as E
    import expert_office_1p3b as X
    if not kernel:
        ths.HAVE_KERNEL = False
    obj = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    c = obj["cfg"]
    trunk = ths.build_from_cfg(c, obj["state"]["emb.weight"].shape[0])   # đúng attn_layers + rope_base của thân
    trunk.load_state_dict(obj["state"])
    E.set_pos(c)
    sets = {}
    for nm, e in obj.get("experts", {}).items():
        s = X.make_set(c["d"], e.get("hid", 8192), 0.0)
        s.load_state_dict({k: v.float() for k, v in e["state"].items()})
        for b in s.values():
            b.use_gate = e.get("gated", True)
        sets[nm] = s
    for p in list(trunk.parameters()) + [p for s in sets.values() for p in s.parameters()]:
        p.requires_grad_(False)
    return trunk, sets, c, obj


class VLM(nn.Module):
    def __init__(self, trunk, sets, vision, ckpt_from=None):
        super().__init__()
        self.t, self.sets, self.v = trunk, nn.ModuleDict(sets), vision
        self.ckpt_from = POS[0] if ckpt_from is None else ckpt_from

    def forward(self, ids, px_list, grids, grad_ckpt=True):
        vis = torch.cat(self.v(px_list, grids)) if px_list else None
        x = embed_ids(self.t.emb, ids, vis)
        for i, (blk, nm) in enumerate(zip(self.t.blocks, self.t.norms)):
            if grad_ckpt and self.training:
                x = x + torch.utils.checkpoint.checkpoint(lambda z, b=blk, n=nm: b(n(z)), x, use_reentrant=False)
            else:
                x = x + blk(nm(x))
            for s in self.sets.values():
                if str(i) in s:
                    x = s[str(i)](x)
        return self.t.norm_f(x) @ self.t.emb.weight.T


def make_vision(trunk, dev, tower_path=TOWER, tower_dtype=torch.bfloat16):
    tower, vcfg = load_tower(tower_path, dev, tower_dtype)
    for p in tower.parameters():
        p.requires_grad_(False)
    rms = float(trunk.emb.weight.float().pow(2).mean().sqrt())
    proj = Projector(vcfg["hidden_size"], trunk.emb.weight.shape[1], rms).to(dev)
    return Vision(tower, proj), vcfg


def vision_expert(d, hid=4096):
    import expert_office_1p3b as X
    return X.make_set(d, hid, -2.0)                  # lớp ra = 0 → lúc đầu y hệt model cũ; cổng khởi ~0,12


# ── lệnh: smoke (M4) ──────────────────────────────────────────────────────────────────
def demo_image(text="Xin chào", bg=(30, 90, 200), size=(384, 384)):
    from PIL import ImageDraw, ImageFont
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 54)
    except Exception:
        f = ImageFont.load_default()
    d.text((24, size[1] // 2 - 30), text, fill=(255, 255, 255), font=f)
    return img


def cmd_smoke(a):
    import hybrid_infer as hi
    from nn_aivie.core.vocab import Vocabulary
    import expert_ffn_1p3b as E
    dev = a.device
    t0 = time.time()
    trunk, sets, c, _ = load_base(a.base, "cpu", kernel=False)
    vsd = torch.load(a.vision, map_location="cpu") if a.vision else {}
    set_cfg(**vsd.get("cfg", {"shuf": 3, "thumb": True}))
    vision, _ = make_vision(trunk, "cpu")
    if vsd:
        apply_vision_file(vsd, trunk, vision, sets, c)
    trunk.to(dev).eval(); vision.to(dev)
    per = {str(i): nn.Sequential(*[s[str(i)] for s in sets.values()]) for i in POS}
    r = hi.Runner(trunk, dev, nn.ModuleDict(per).to(dev).eval())
    tok = E.tok_(); voc = Vocabulary(tok); eos = tok.token_to_id["<EOS>"]
    print(f"[smoke] nạp {time.time()-t0:.0f}s · expert {list(sets)}", flush=True)
    img = load_image(a.image) if a.image else demo_image()
    for mt in (1, a.max_tiles):
        px, grids = prep_images([img], mt)
        t1 = time.time()
        with torch.no_grad():
            vis = torch.cat(vision(px, grids))
        er = float(trunk.emb.weight.float().pow(2).mean().sqrt())
        print(f"[smoke] ảnh {img.size} · lưới {grids[0]} · {len(px[0])} ô → {vis.shape[0]} token ảnh "
              f"(= {n_tokens(grids[0])}) · rms ảnh {float(vis.pow(2).mean().sqrt()):.4f} vs chữ {er:.4f} · mắt {time.time()-t1:.1f}s")
        ids, _ = build_example([{"role": "user", "content": a.q, "n_img": 1}], tok, grids)
        ids += tok.encode("Đáp: ")
        t1 = time.time()
        out = list(r.generate(ids, n=40, temp=0, eos=eos, vis=vis))
        print(f"[smoke] prefill+sinh {time.time()-t1:.1f}s · {len(ids)} token vào · ra: {voc.detokenize(out)!r}", flush=True)


# ── lệnh: overfit (M4) — kiểm gradient chảy qua bộ chiếu: 8 ảnh tự vẽ, học thuộc lời mô tả ──
COLORS = {"xanh dương": (30, 90, 200), "đỏ": (200, 40, 40), "xanh lá": (40, 160, 70), "vàng": (230, 200, 40)}
WORDS = ["Hà Nội", "Mèo", "Toán", "Excel"]


def cmd_overfit(a):
    import expert_ffn_1p3b as E
    import hybrid_infer as hi
    from nn_aivie.core.vocab import Vocabulary
    dev = a.device
    trunk, sets, c, _ = load_base(a.base, "cpu", kernel=False)
    trunk.to(dev)
    for s in sets.values():
        s.to(dev)
    vision, _ = make_vision(trunk, dev)
    m = VLM(trunk, sets, vision).to(dev).train()
    tok = E.tok_(); voc = Vocabulary(tok); eos = tok.token_to_id["<EOS>"]
    data = []
    for k, (cn, col) in enumerate(COLORS.items()):
        for w in WORDS[k % 2::2]:
            cap = f"Ảnh có chữ \"{w}\" màu trắng trên nền {cn}."
            data.append((demo_image(w, col), cap))
    q = "Mô tả ảnh này."
    params = [p for p in vision.proj.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    print(f"[overfit] {len(data)} ảnh · học {sum(p.numel() for p in params)/1e6:.0f}M tham số (bộ chiếu) · {dev}", flush=True)
    T0 = time.time()
    for s_ in range(a.steps):
        img, cap = data[s_ % len(data)]
        px, grids = prep_images([img], 1)
        ids, lab = build_example([{"role": "user", "content": q, "n_img": 1}, {"role": "assistant", "content": cap}], tok, grids)
        X_ = torch.tensor([ids], device=dev); Y_ = torch.tensor([lab], device=dev)
        lo = m(X_[:, :-1], px, grids, grad_ckpt=False)
        loss = F.cross_entropy(lo.float().reshape(-1, lo.size(-1)), Y_[:, 1:].reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if s_ % 8 == 0 or s_ == a.steps - 1:
            print(f"[overfit] bước {s_} · loss {float(loss):.3f} · gain {float(vision.proj.gain.mean()):.3f} · {time.time()-T0:.0f}s", flush=True)
    m.eval()
    per = {str(i): nn.Sequential(*[s[str(i)] for s in sets.values()]) for i in POS}
    r = hi.Runner(trunk, dev, nn.ModuleDict(per).eval())
    ok = 0
    for img, cap in data:
        px, grids = prep_images([img], 1)
        with torch.no_grad():
            vis = torch.cat(vision(px, grids))
        ids, _ = build_example([{"role": "user", "content": q, "n_img": 1}], tok, grids)
        out = voc.detokenize(list(r.generate(ids + tok.encode("Đáp: "), n=30, temp=0, eos=eos, vis=vis))).strip()
        ok += out == cap
        print(f"  {'✓' if out == cap else '✗'} {out!r}  (đúng: {cap!r})")
    print(f"### overfit: {ok}/{len(data)} mô tả đúng nguyên văn", flush=True)
    if a.save:
        torch.save({"proj": vision.proj.state_dict()}, a.save)


# ── lệnh: train (box, CUDA) ───────────────────────────────────────────────────────────
class Data:
    """jsonl: {"images": [đường dẫn…], "messages": [{"role","content", "n_img"?}…], "last_only"?}
    Không ghi n_img → toàn bộ ảnh gắn vào tin nhắn người dùng ĐẦU TIÊN."""
    def __init__(self, path, tok, max_tiles, max_len, root=None):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        self.tok, self.mt, self.ml = tok, max_tiles, max_len
        self.root = root or os.path.dirname(path)

    def get(self, r):
        imgs = [p if os.path.isabs(p) else os.path.join(self.root, p) for p in r.get("images", [])]
        msgs = [dict(m) for m in r["messages"]]
        if imgs and not any(m.get("n_img") for m in msgs):
            next(m for m in msgs if m["role"] == "user")["n_img"] = len(imgs)
        px, grids = prep_images(imgs, min(r.get("max_tiles", self.mt), self.mt))
        ids, lab = build_example(msgs, self.tok, grids, 10 ** 6, r.get("last_only", False))
        if len(ids) > self.ml:
            return None                                          # quá dài (ảnh nhiều ô + hội thoại dài) → bỏ
        return ids, lab, px, grids


def cmd_train(a):
    import contextlib
    import expert_ffn_1p3b as E
    dev = a.device
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if dev == "cuda" else contextlib.nullcontext
    torch.manual_seed(0)
    tok = E.tok_()
    trunk, sets, c, _ = load_base(a.base, "cpu", kernel=dev == "cuda")
    set_cfg(a.shuf, not a.no_thumb)
    vision, _ = make_vision(trunk, dev, a.tower)
    ve = vision_expert(c["d"], a.hid)
    lora_p = add_lora(trunk, a.lora, mamba=a.lora_mamba) if a.lora else []
    if a.init:                                                   # tiếp từ giai đoạn trước
        vsd = torch.load(a.init, map_location="cpu")
        assert vsd.get("cfg", {"shuf": 3, "thumb": True}) == dict(CFG), ("cấu hình gộp khác file init", vsd.get("cfg"), CFG)
        vision.proj.load_state_dict(vsd["proj"])
        if "expert" in vsd:
            ve.load_state_dict(vsd["expert"])
        if "tower" in vsd:
            vm(vision.tower).load_state_dict(vsd["tower"], strict=False)
        if vsd.get("lora") and a.lora:                            # LoRA cũ (vd V4 chỉ attention) → nạp phần trùng
            for k, v in vsd["lora"].items():
                i, nm, ab = k.split(".")
                mod = getattr(trunk.blocks[int(i)], nm)
                if isinstance(mod, LoRA) and getattr(mod, ab).shape == v.shape:
                    getattr(mod, ab).data.copy_(v.float())
    for p in lora_p:
        p.requires_grad_(a.stage >= 2)
    sets["vision"] = ve
    m = VLM(trunk.to(dev), sets, vision).to(dev).train()
    groups = [{"params": list(vision.proj.parameters()), "lr": a.lr}]
    for p in ve.parameters():
        p.requires_grad_(a.stage >= 2)
    if a.stage >= 2:
        groups += [{"params": [p for k, p in ve.named_parameters() if ".gate." not in k], "lr": a.lr_expert},
                   {"params": [p for k, p in ve.named_parameters() if ".gate." in k], "lr": 1e-3}]
        if lora_p:
            groups.append({"params": lora_p, "lr": a.lr_lora})
        if a.tower_last:
            lay = vm(vision.tower).encoder.layers[-a.tower_last:]
            tp = [p for l in lay for p in l.parameters()] + list(vm(vision.tower).post_layernorm.parameters())
            vision.tower.float()
            for p in tp:
                p.requires_grad_(True)
            groups.append({"params": tp, "lr": a.lr_tower})
    for g in groups:
        g["base"] = g["lr"]
    params = [p for g in groups for p in g["params"]]
    print(f"[vlm] giai đoạn {a.stage} · học {sum(p.numel() for p in params)/1e6:.0f}M tham số", flush=True)
    data = Data(a.data, tok, a.max_tiles, a.max_len)
    val = [data.rows[i] for i in range(min(a.n_val, len(data.rows) // 20))]
    train = data.rows[len(val):]
    opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
    rng = random.Random(a.seed)
    pool = cf.ThreadPoolExecutor(a.loaders)                       # đọc + cắt ô ảnh song song với GPU

    def micro(exs):
        """gộp vài mẫu thành 1 lô có đệm phải (Mamba/attention nhân quả → đệm cuối không ảnh hưởng token trước)."""
        L = max(len(e[0]) for e in exs)
        X_ = torch.zeros(len(exs), L, dtype=torch.long); Y_ = torch.full((len(exs), L), -100, dtype=torch.long)
        for j, (ids, lab, _, _) in enumerate(exs):
            X_[j, :len(ids)] = torch.tensor(ids); Y_[j, :len(lab)] = torch.tensor(lab)
        px = [p.to(dev, non_blocking=True) for e in exs for p in e[2]]
        grids = [g for e in exs for g in e[3]]
        X_, Y_ = X_.to(dev), Y_.to(dev)
        with amp():
            lo = m(X_[:, :-1], px, grids)
        n = int((Y_[:, 1:] != -100).sum())
        return F.cross_entropy(lo.float().reshape(-1, lo.size(-1)), Y_[:, 1:].reshape(-1), ignore_index=-100,
                               reduction="sum"), n

    def chunks(exs):
        """chia lô theo NGÂN SÁCH TOKEN (mẫu dài nhiều ô ảnh tự đứng riêng) → tránh hết bộ nhớ."""
        exs = sorted([e for e in exs if e is not None], key=lambda e: len(e[0]))
        out, cur = [], []
        for e in exs:
            if cur and (len(cur) + 1) * len(e[0]) > a.tok_budget:
                out.append(cur); cur = []
            cur.append(e)
        return out + ([cur] if cur else [])

    @torch.no_grad()
    def vloss():
        m.eval()
        tot = n = 0
        for ch in chunks(list(pool.map(data.get, val))):
            l, k = micro(ch); tot += float(l); n += k
        m.train()
        return round(tot / max(1, n), 4)
    print(f"[vlm] {len(train)} mẫu train · {len(val)} val · val trước {vloss()}", flush=True)
    T0 = time.time()
    nxt = pool.map(data.get, rng.sample(train, a.bs))
    for s_ in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = g["base"] * min(1, (s_ + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s_ / a.steps)))
        exs = list(nxt)
        nxt = pool.map(data.get, rng.sample(train, a.bs))       # nạp lô sau trong lúc GPU tính lô này
        cs = chunks(exs)
        ntok = sum(int(sum(x != -100 for x in e[1])) for ch in cs for e in ch)
        tot = 0.0
        for ch in cs:
            l, _ = micro(ch)
            (l / max(1, ntok)).backward(); tot += float(l)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if s_ % 50 == 0 or s_ == a.steps - 1:
            print(f"[vlm] bước {s_}/{a.steps} · loss {tot/max(1,ntok):.3f} · gain {float(vision.proj.gain.mean()):.3f} · "
                  f"{len(cs)} lô con · {time.time()-T0:.0f}s", flush=True)
        if (s_ + 1) % a.val_every == 0:
            print(f"[vlm] val bước {s_+1}: {vloss()}", flush=True)
        if (s_ + 1) % a.save_every == 0 or s_ == a.steps - 1:
            sd = {"proj": vision.proj.state_dict(), "hid": a.hid, "stage": a.stage, "cfg": dict(CFG)}
            if lora_p:
                sd["lora"] = lora_state(trunk)
            if a.stage >= 2:
                sd["expert"] = ve.state_dict()
                if a.tower_last:                                  # chỉ các tầng mắt được học (đỡ ổ: ~0,2GB thay vì 0,86GB)
                    nl = len(vm(vision.tower).encoder.layers)
                    keep = tuple(f"encoder.layers.{i}." for i in range(nl - a.tower_last, nl)) + ("post_layernorm.",)
                    sd["tower"] = {k: v.to(torch.bfloat16) for k, v in vm(vision.tower).state_dict().items() if k.startswith(keep)}
            torch.save(sd, a.out + ".tmp")                          # ghi tạm rồi đổi tên → hết ổ giữa chừng không hỏng bản cũ
            os.replace(a.out + ".tmp", a.out)
    print(f"[vlm] val sau {vloss()}", flush=True)


# ── lệnh: pack — MỘT checkpoint (thân + expert + mắt + bộ chiếu + expert thị giác) ─────
def cmd_pack(a):
    from safetensors import safe_open
    from safetensors.torch import load_file
    obj = torch.load(a.base, map_location="cpu", weights_only=False)
    vsd = torch.load(a.vision, map_location="cpu")
    tower = {k: v for k, v in load_file(a.tower).items() if not k.startswith("head.")}
    if "tower" in vsd:
        tower.update(vsd["tower"])
    meta = safe_open(a.tower, "pt").metadata()
    cfg = vsd.get("cfg", {"shuf": 3, "thumb": True})
    obj["vision"] = {"tower": {k: v.to(torch.bfloat16) for k, v in tower.items()},
                     "tower_cfg": json.loads(meta["vision_config"]), "proj": vsd["proj"],
                     "tile": TILE, "shuf": cfg["shuf"], "thumb": cfg["thumb"], "max_tiles": a.max_tiles}
    if vsd.get("lora"):
        if a.merge_lora:                                        # gộp hẳn vào thân (chat chữ cũng đổi)
            merge_lora(obj["state"], vsd["lora"])
        else:                                                   # giữ riêng → UI chỉ bật khi hội thoại có ảnh
            obj["vision"]["lora"] = vsd["lora"]
    if "expert" in vsd:
        obj.setdefault("experts", {})["vision"] = {"state": {k: v.to(torch.bfloat16) for k, v in vsd["expert"].items()},
                                                  "hid": vsd.get("hid", 4096), "gated": True}
    obj["note"] = (obj.get("note", "") + " · ĐỌC ẢNH: SigLIP2 so400m + bộ chiếu + expert thị giác").strip(" ·")
    torch.save(obj, a.out)
    print(f"### {a.out} · {os.path.getsize(a.out)/1e9:.2f}GB · expert {list(obj['experts'])}", flush=True)


def cmd_assemble(a):
    """box: thân chat_v2 + 3 expert (file rời fp32) → 1 checkpoint trợ lý (như hybrid_1p3b_assistant_v5.pt trên Mac)."""
    obj = torch.load(a.base, map_location="cpu", weights_only=False)
    obj["experts"] = {}
    for nm, path, hid in (("code", a.code, 8192), ("office", a.office, 4096), ("hoc", a.hoc, 8192)):
        sd = torch.load(path, map_location="cpu")
        obj["experts"][nm] = {"state": {k: v.to(torch.bfloat16) for k, v in sd.items()}, "hid": hid, "gated": True}
    obj.pop("opt", None)
    torch.save(obj, a.out)
    print(f"### {a.out} · expert {list(obj['experts'])}", flush=True)


# ── lệnh: evalvlm — đo trên bộ test giữ riêng ─────────────────────────────────────────
P_VJ = """Chấm câu trả lời của một trợ lý AI về MỘT BỨC ẢNH (bạn không thấy ảnh, chỉ có đáp án chuẩn do người/thầy viết).
Câu hỏi: {q}
Đáp án chuẩn: {ref}
Trả lời của trợ lý: {ans}
"dung" = đúng ý chính của đáp án chuẩn, không thêm chi tiết sai; "mot_phan" = đúng một phần hoặc có chi tiết sai; "sai" = sai/lạc đề/bịa.
Trả về DUY NHẤT JSON: {{"v": "dung|mot_phan|sai"}}"""


def keys_of(ref):
    """các giá trị phải có trong câu trả lời: số, chuỗi trong ngoặc kép, công thức, tên trong `…`."""
    f = re.findall(r"=[A-Z][^\s]*[^\s.,]", ref)
    rest = re.sub(r"=[A-Z][^\s]*[^\s.,]", " ", ref)
    ks = f + re.findall(r"`[^`]+`|\"[^\"]{1,60}\"|\d[\d.,]*\d|\d", rest)
    return [k.strip('`"') for k in ks]


def cmd_evalvlm(a):
    import difflib
    import hybrid_infer as hi
    import expert_ffn_1p3b as E
    from nn_aivie.core.vocab import Vocabulary
    dev = a.device
    tok = E.tok_(); voc = Vocabulary(tok); eos = tok.token_to_id["<EOS>"]
    trunk, sets, c, _ = load_base(a.base, "cpu", kernel=False)
    vsd = torch.load(a.vision, map_location="cpu") if a.vision else {}
    set_cfg(**vsd.get("cfg", {"shuf": 3, "thumb": True}))
    if vsd.get("lora") and not a.no_vexpert and not a.no_lora:
        load_lora(trunk, vsd["lora"])
    trunk.to(dev).eval()
    vision, _ = make_vision(trunk, dev, a.tower)
    if vsd:
        vision.proj.load_state_dict(vsd["proj"])
        if "tower" in vsd:
            vm(vision.tower).load_state_dict(vsd["tower"], strict=False)
        if "expert" in vsd and not a.no_vexpert:
            ve = vision_expert(c["d"], vsd.get("hid", 4096)); ve.load_state_dict(vsd["expert"]); sets["vision"] = ve
    per = {str(i): nn.Sequential(*[s[str(i)] for s in sets.values()]) for i in POS}
    r = hi.Runner(trunk, dev, nn.ModuleDict(per).to(dev).eval())
    root = os.path.dirname(a.test)
    rows = [json.loads(l) for l in open(a.test, encoding="utf-8")][:a.n]
    out = []
    T0 = time.time()
    for k, row in enumerate(rows):
        q, ref = row["messages"][0]["content"], row["messages"][1]["content"]
        imgs = [os.path.join(root, p) for p in row.get("images", [])]
        msgs = [{"role": "user", "content": q, "n_img": len(imgs)}]
        if imgs:
            px, grids = prep_images(imgs, min(row.get("max_tiles", a.max_tiles), a.max_tiles))
            with torch.no_grad():
                vis = torch.cat(vision(px, grids))
        else:
            grids, vis = [], None
        ids, _ = build_example(msgs, tok, grids, 10 ** 6)
        ans = voc.detokenize(list(r.generate(ids + tok.encode("Đáp: "), n=a.max_new, temp=0, eos=eos, vis=vis))).strip()
        out.append({"src": row.get("src", ""), "q": q, "ref": ref, "ans": ans})
        if k % 25 == 0:
            print(f"[eval] {k}/{len(rows)} · {time.time()-T0:.0f}s · {ans[:80]!r}", flush=True)
    os.makedirs("ckpt/vlm", exist_ok=True)
    name = f"ckpt/vlm/eval_{a.tag}_{os.path.basename(a.test).split('.')[0]}.json"
    json.dump({"tag": a.tag, "test": a.test, "rows": out}, open(name, "w"), ensure_ascii=False, indent=1)
    print(f"### sinh xong {len(out)} câu → {name} (chấm: lệnh judgevlm)", flush=True)


def cmd_judgevlm(a):
    """chấm các file ckpt/vlm/eval_*.json: thầy chấm ý (vLLM) + khớp giá trị + độ giống chuỗi (chép lại)."""
    import difflib, glob
    from gen_office import loads_loose
    import gen_vlm_data as G

    def judge(o):
        try:
            return loads_loose(G.chat(a.llm, P_VJ.format(q=o["q"], ref=o["ref"][:2500], ans=o["ans"][:2500]), temp=0)).get("v", "?")
        except Exception:
            return "?"
    for name in sorted(glob.glob(a.files)):
        d = json.load(open(name))
        out = d["rows"]
        with cf.ThreadPoolExecutor(48) as ex:
            vs = list(ex.map(judge, out))
        cnt = {"dung": 0, "mot_phan": 0, "sai": 0}
        km, tr = [], []
        for o, v in zip(out, vs):
            o["judge"] = v
            if v in cnt:
                cnt[v] += 1
            if o["q"].startswith("Chép lại"):
                tr.append(difflib.SequenceMatcher(None, o["ans"], o["ref"]).ratio())
            else:
                ks = keys_of(o["ref"])
                if ks:
                    km.append(all(x in o["ans"] for x in ks))
        n = max(1, len(out))
        d["res"] = {"n": len(out), "dung_%": round(100 * cnt["dung"] / n, 1), "mot_phan_%": round(100 * cnt["mot_phan"] / n, 1),
                    "khop_gia_tri_%": round(100 * sum(km) / len(km), 1) if km else None,
                    "chep_lai_giong": round(sum(tr) / len(tr), 3) if tr else None}
        json.dump(d, open(name, "w"), ensure_ascii=False, indent=1)
        print(f"### {d['tag']} · {os.path.basename(d['test'])} · {d['res']}", flush=True)


def vision_from_ckpt(obj, trunk, dev):
    """cho UI: dựng Vision từ mục "vision" của checkpoint (None nếu không có)."""
    v = obj.get("vision")
    if not v:
        return None, 12
    from transformers import SiglipVisionConfig, SiglipVisionModel
    cfg = SiglipVisionConfig(**{k: x for k, x in v["tower_cfg"].items() if k != "model_type"}, vision_use_head=False)
    tower = SiglipVisionModel(cfg)
    vm(tower).load_state_dict(v["tower"])
    set_cfg(v.get("shuf", 3), v.get("thumb", True))
    if v.get("lora"):
        load_lora(trunk, v["lora"])
        trunk.to(dev)
    proj = Projector(cfg.hidden_size, trunk.emb.weight.shape[1], 1.0)
    proj.load_state_dict(v["proj"])
    return Vision(tower.to(dev, torch.bfloat16).eval(), proj.to(dev).eval()), v.get("max_tiles", 12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "overfit", "train", "pack", "assemble", "evalvlm", "judgevlm"])
    ap.add_argument("--base", default="checkpoints/hf_hybrid/hybrid_1p3b_assistant_v5.pt")
    ap.add_argument("--tower", default=TOWER)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--image", default="")
    ap.add_argument("--q", default="Mô tả ảnh này.")
    ap.add_argument("--max-tiles", type=int, default=12)
    ap.add_argument("--vision", default="", help="file bộ chiếu/expert thị giác đã train")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save", default="")
    # train
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--data", default="data/vlm/align.jsonl")
    ap.add_argument("--init", default="")
    ap.add_argument("--hid", type=int, default=4096)
    ap.add_argument("--lr-expert", type=float, default=2e-4)
    ap.add_argument("--tower-last", type=int, default=0, help="giai đoạn 2: mở N tầng cuối của mắt")
    ap.add_argument("--shuf", type=int, default=3, help="gộp k×k mảnh/token (V1: 3 → 81 token/ô · V4: 2 → 196)")
    ap.add_argument("--no-thumb", action="store_true", help="bỏ ảnh toàn cảnh thu nhỏ khi ảnh nhiều ô")
    ap.add_argument("--lora", type=int, default=0, help="hạng LoRA trên 6 khối attention của thân (giai đoạn 2)")
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lora-mamba", action="store_true", help="LoRA cả in_proj/out_proj của 42 khối Mamba (V5)")
    ap.add_argument("--merge-lora", action="store_true", help="pack: gộp LoRA vào thân thay vì giữ riêng")
    ap.add_argument("--no-lora", action="store_true", help="evalvlm: bỏ LoRA (giữ expert thị giác) — như UI khi không có ảnh")
    ap.add_argument("--lr-tower", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--tok-budget", type=int, default=12000, help="số token tối đa mỗi lô con (đệm theo mẫu dài nhất)")
    ap.add_argument("--loaders", type=int, default=12)
    ap.add_argument("--out", default="ckpt/vlm/vision.pt")
    # assemble / evalvlm
    ap.add_argument("--code", default=""); ap.add_argument("--office", default=""); ap.add_argument("--hoc", default="")
    ap.add_argument("--test", default="data/vlm/test_screen.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--llm", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--files", default="ckpt/vlm/eval_*.json")
    ap.add_argument("--no-vexpert", action="store_true", help="bỏ expert thị giác (so sánh: kỹ năng chữ có bị ảnh hưởng?)")
    a = ap.parse_args()
    {"smoke": cmd_smoke, "overfit": cmd_overfit, "train": cmd_train, "pack": cmd_pack, "assemble": cmd_assemble,
     "evalvlm": cmd_evalvlm, "judgevlm": cmd_judgevlm}[a.cmd](a)


if __name__ == "__main__":
    main()
