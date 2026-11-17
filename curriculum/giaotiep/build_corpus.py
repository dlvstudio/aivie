"""TK_104 · Môn GIAO-TIẾP: dựng kho ngữ-liệu song-ngữ + tokenizer BPE (chạy trên box).

Nguồn:
  • nampdn-ai/tinystories-vietnamese (MIT): truyện đơn-giản, cột en + vi SONG-SONG (bản vi là dịch máy)
  • data/giaotiep/dd_envi.jsonl: DailyDialog (en) + bản dịch khẩu-ngữ của Gemma (vi) — translate_dd.py
Mỗi tài-liệu một loại (trộn NGẪU-NHIÊN khi train — bài học TK_104):
  story_en / story_vi        truyện một ngôn-ngữ
  dialog_en / dialog_vi      hội-thoại "A: …\\nB: …"
  dich_ev / dich_ve          dịch hai chiều (truyện ngắn + từng lượt nói)
Giữ riêng: 2.000 truyện + phần validation/test của DailyDialog (đánh-giá hội-thoại & dịch).

    python curriculum/giaotiep/build_corpus.py --stories 300000
    → data/giaotiep/train.jsonl · val.jsonl · test_dialog.jsonl · tok.json
"""
from __future__ import annotations
import argparse, json, os, random

OUT = "data/giaotiep"


def fmt_dialog(turns):
    return "\n".join(f"{'AB'[i % 2]}: {t}" for i, t in enumerate(turns))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", type=int, default=300000)
    ap.add_argument("--val-stories", type=int, default=2000)
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--extra", default="", help="hội-thoại SINH thêm (gen_dialog_grounded.py) — toàn-bộ vào train")
    ap.add_argument("--out", default=OUT, help="thư-mục ra (giữ v2 ở data/giaotiep, v3 ra thư-mục riêng)")
    ap.add_argument("--tok-from", default="", help="dùng LẠI tokenizer có sẵn (so-sánh công-bằng với v2)")
    ap.add_argument("--dd", default=f"{OUT}/dd_envi.jsonl", help="bản dịch DailyDialog (dùng bản chụp khi đang dịch tiếp)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(104)
    from datasets import load_dataset
    ts = load_dataset("nampdn-ai/tinystories-vietnamese", split="train")
    idx = list(range(len(ts)))
    rng.shuffle(idx)
    pick = idx[: a.stories + a.val_stories]
    stories = [(ts[i]["en"].strip(), ts[i]["vi"].strip()) for i in pick]
    stories = [(e, v) for e, v in stories if e and v and len(v) < 4 * len(e)]
    val_st, tr_st = stories[: a.val_stories], stories[a.val_stories:]
    dd = [json.loads(l) for l in open(a.dd, encoding="utf-8")]
    tr_dd = [d for d in dd if d["split"] == "train"]
    if a.extra:
        ex = [json.loads(l) for l in open(a.extra, encoding="utf-8")]
        tr_dd += ex
        print(f"+ {len(ex)} hội-thoại sinh thêm ({a.extra})")
    va_dd = [d for d in dd if d["split"] != "train"]
    print(f"truyện: train {len(tr_st)} · val {len(val_st)} · hội-thoại: train {len(tr_dd)} · giữ riêng {len(va_dd)}")

    def docs(st, dlg, with_trans=True):
        out = []
        for e, v in st:
            out += [{"type": "story_en", "text": e}, {"type": "story_vi", "text": v}]
            if with_trans and rng.random() < 0.15:          # dịch cả truyện (ngắn) — 15% số truyện
                if rng.random() < 0.5:
                    out.append({"type": "dich_ev", "text": f"Dịch sang tiếng Việt:\n{e}\n=>\n{v}"})
                else:
                    out.append({"type": "dich_ve", "text": f"Translate to English:\n{v}\n=>\n{e}"})
        for d in dlg:
            out += [{"type": "dialog_en", "text": fmt_dialog(d["en"])}, {"type": "dialog_vi", "text": fmt_dialog(d["vi"])}]
            if with_trans:
                for e, v in zip(d["en"], d["vi"]):
                    if rng.random() < 0.3:                   # dịch từng lượt nói
                        out.append({"type": "dich_ev", "text": f"Dịch sang tiếng Việt: {e}\n=> {v}"}
                                   if rng.random() < 0.5 else
                                   {"type": "dich_ve", "text": f"Translate to English: {v}\n=> {e}"})
        rng.shuffle(out)
        return out

    train = docs(tr_st, tr_dd)
    val = docs(val_st, va_dd[: len(va_dd) // 2])
    for name, rows in (("train", train), ("val", val)):
        with open(f"{a.out}/{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{a.out}/test_dialog.jsonl", "w", encoding="utf-8") as f:   # nửa còn lại: chấm hội-thoại & dịch
        for d in va_dd[len(va_dd) // 2:]:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    from collections import Counter
    print("train theo loại:", dict(Counter(r["type"] for r in train)))

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    if a.tok_from:
        import shutil
        shutil.copy(a.tok_from, f"{a.out}/tok.json")
        print(f"tokenizer: dùng lại {a.tok_from}")
        return
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=["<pad>", "<eos>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator((r["text"] for r in train), trainer=tr)
    tok.save(f"{a.out}/tok.json")
    n_tok = sum(len(tok.encode(r["text"]).ids) for r in train[:20000])
    n_chr = sum(len(r["text"]) for r in train[:20000])
    print(f"tokenizer {tok.get_vocab_size()} · {n_chr/n_tok:.2f} ký-tự/token · ước toàn kho "
          f"{n_tok/20000*len(train)/1e6:.0f}M token")


if __name__ == "__main__":
    main()
