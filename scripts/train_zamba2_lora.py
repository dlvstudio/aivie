"""§TK_103 · PILOT PHƯƠNG ÁN B — LoRA Zamba2-2.7B-Instruct cho tiếng Việt (thầy = Gemma).

Dữ-liệu (data/pilot_b/, sinh bởi gen_pilot_b.py + lát PhoMT):
  • B2   : hỏi Việt → "<en>\\n{nháp EN}\\n</en>\\n<vi>\\n{trả lời VI}\\n</vi>"   (distill.jsonl)
  • ôn EN: hỏi EN (q_en) → trả lời EN — chống quên tiếng Anh/code           (--replay-frac của distill)
  • dịch : "Dịch sang tiếng Việt: …" → VI · "Translate to English: …" → EN   (phomt.jsonl)
Loss CHỈ trên phần assistant (che câu hỏi); cắt sau <|im_end|>.

Bẫy transformers 5.16 (xem §13-§16 TK_103):
  • patch_zamba2_block_id() — bắt-buộc, không thì nạp checkpoint lỗi tie_weights.
  • forward khi train PHẢI use_cache=False (DynamicCache mặc-định không biết tầng 'mamba').

    # smoke CPU (model tí hon ngẫu-nhiên, tokenizer thật):
    PYTHONPATH=scripts:src .venv/bin/python scripts/train_zamba2_lora.py --tiny --steps 4
    # H100:
    PYTHONPATH=scripts python scripts/train_zamba2_lora.py --epochs 2 --out ckpt/zamba2_vi_lora
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
import torch

sys.path.insert(0, "scripts")
from test_zamba2 import patch_zamba2_block_id, MODEL

TARGETS = ["in_proj", "out_proj", "linear", "q_proj", "k_proj", "v_proj", "o_proj",
           "gate_up_proj", "down_proj"]


def build_examples(data_dir, replay_frac, n_phomt, seed=0):
    rng = random.Random(seed)
    ex = []
    for line in open(f"{data_dir}/distill.jsonl", encoding="utf-8"):
        r = json.loads(line)
        ex.append(("b2", r["q"], f"<en>\n{r['en']}\n</en>\n<vi>\n{r['vi']}\n</vi>"))
        if rng.random() < replay_frac:
            ex.append(("replay_en", r["q_en"], r["en"]))
    pairs = [json.loads(l) for l in open(f"{data_dir}/phomt.jsonl", encoding="utf-8")][:n_phomt]
    for p in pairs:
        if rng.random() < 0.7:
            ex.append(("mt_en2vi", f"Dịch sang tiếng Việt: {p['en']}", p["vi"]))
        else:
            ex.append(("mt_vi2en", f"Translate to English: {p['vi']}", p["en"]))
    rng.shuffle(ex)
    return ex


def encode(tok, user, assistant, max_len):
    u = [{"role": "user", "content": user}]
    p = tok.apply_chat_template(u, add_generation_prompt=True, tokenize=False)
    f = tok.apply_chat_template(u + [{"role": "assistant", "content": assistant}], tokenize=False)
    ids_p = tok(p, add_special_tokens=False)["input_ids"]
    ids = tok(f, add_special_tokens=False)["input_ids"]
    if ids[:len(ids_p)] != ids_p:
        return None
    eos = tok.eos_token_id
    if eos in ids[len(ids_p):]:
        ids = ids[:len(ids_p) + ids[len(ids_p):].index(eos) + 1]    # cắt sau <|im_end|>
    if len(ids) > max_len:
        return None                                                 # bỏ, không cắt cụt câu trả lời
    labels = [-100] * len(ids_p) + ids[len(ids_p):]
    return ids, labels


def batches(items, bs, pad_id, rng=None):
    idx = list(range(len(items)))
    if rng:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [items[j] for j in idx[i:i + bs]]
        L = max(len(a) for a, _ in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        lab = torch.full((len(chunk), L), -100, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for k, (a, b) in enumerate(chunk):                          # right-pad (nhân-quả → an-toàn)
            ids[k, :len(a)] = torch.tensor(a); lab[k, :len(b)] = torch.tensor(b); att[k, :len(a)] = 1
        yield ids, lab, att


def tiny_model():
    from transformers import AutoConfig, AutoModelForCausalLM
    c = AutoConfig.from_pretrained(MODEL)
    c.hidden_size, c.num_hidden_layers, c.hybrid_layer_ids = 64, 8, [2, 4, 6, 7]
    c.attention_hidden_size, c.num_attention_heads, c.num_key_value_heads = 128, 4, 4
    c.mamba_headdim, c.n_mamba_heads, c.intermediate_size, c.adapter_rank = 16, 8, 128, 8
    c.layers_block_type = ["hybrid" if i in c.hybrid_layer_ids else "mamba" for i in range(8)]
    return AutoModelForCausalLM.from_config(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pilot_b")
    ap.add_argument("--out", default="ckpt/zamba2_vi_lora")
    ap.add_argument("--replay-frac", type=float, default=0.3)
    ap.add_argument("--n-phomt", type=int, default=1500)
    ap.add_argument("--max-len", type=int, default=1536)   # 1024 bỏ 257 mẫu B2 (EN+VI dài)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--steps", type=int, default=0, help="dừng sau N bước (smoke)")
    ap.add_argument("--tiny", action="store_true", help="model tí hon ngẫu-nhiên (smoke CPU)")
    ap.add_argument("--use-kernels", action="store_true",
                    help="kernel Mamba2 build sẵn từ HF Hub (kernels-community/mamba-ssm, cần `pip install kernels`)")
    a = ap.parse_args()
    T0 = time.time()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    patch_zamba2_block_id()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    ex = build_examples(a.data, a.replay_frac, a.n_phomt)
    enc, kinds, dropped = [], {}, 0
    for kind, u, asst in ex:
        e = encode(tok, u, asst, a.max_len)
        if e is None:
            dropped += 1; continue
        enc.append(e); kinds[kind] = kinds.get(kind, 0) + 1
    nval = max(8, int(len(enc) * a.val_frac))
    val, train = enc[:nval], enc[nval:]
    ntok = sum(sum(1 for t in l if t != -100) for _, l in train)
    print(f"[{time.time()-T0:.0f}s] mẫu {len(enc)} (bỏ {dropped} quá dài) · {kinds} · train {len(train)} "
          f"· val {len(val)} · {ntok/1e6:.2f}M token-trả-lời", flush=True)

    if a.tiny:
        model = tiny_model()
    else:
        kw = {"use_kernels": True} if a.use_kernels else {}
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16 if dev == "cuda" else torch.float32, **kw)
    model.to(dev)
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps_ep = math.ceil(len(train) / (a.bs * a.grad_accum))
    total = a.steps or int(steps_ep * a.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(1, s / max(1, total)))))

    @torch.no_grad()
    def val_loss():
        model.eval(); tot = n = 0
        for ids, lab, att in batches(val, a.bs, pad_id):
            out = model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev), use_cache=False)
            k = int((lab != -100).sum()); tot += float(out.loss) * k; n += k
        model.train(); return tot / max(n, 1)

    print(f"[{time.time()-T0:.0f}s] {total} bước ({steps_ep}/epoch) · val đầu {val_loss():.3f}", flush=True)
    model.train(); rng = random.Random(1); step = micro = 0; seen = 0; t_log = time.time()
    while step < total:
        for ids, lab, att in batches(train, a.bs, pad_id, rng):
            out = model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev), use_cache=False)
            (out.loss / a.grad_accum).backward()
            micro += 1; seen += int(att.sum())
            if micro % a.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 10 == 0 or step == total:
                el = time.time() - t_log
                print(f"[{time.time()-T0:.0f}s] bước {step}/{total} · loss {float(out.loss):.3f} "
                      f"· lr {sched.get_last_lr()[0]:.1e} · {seen/el:.0f} tok/s", flush=True)
                seen, t_log = 0, time.time()
            if step % a.eval_every == 0 or step == total:
                print(f"[{time.time()-T0:.0f}s] ✦ val {val_loss():.3f} @bước {step}", flush=True)
            if step >= total:
                break
    if not a.tiny:
        model.save_pretrained(a.out)
        print(f"[{time.time()-T0:.0f}s] LƯU adapter → {a.out}", flush=True)
    print(f"[{time.time()-T0:.0f}s] ### xong", flush=True)


if __name__ == "__main__":
    main()
