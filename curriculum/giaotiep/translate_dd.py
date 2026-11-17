"""TK_104 · Môn GIAO-TIẾP: dịch DailyDialog (hội-thoại đời-thường tiếng Anh) sang tiếng Việt KHẨU-NGỮ
bằng Gemma 4 26B (Ollama, chạy trên box GPU, nhiều yêu-cầu song-song) → hội-thoại song-ngữ song-song.

Thiếu hụt: không có bộ hội-thoại đời-thường tiếng Việt chất-lượng → dịch bộ tiếng Anh kinh-điển.
- Chuẩn-hoá DailyDialog: dấu câu bị tách ("Say , Jim ?", "I ' m") → viết liền.
- Gemma dịch CẢ ĐOẠN (giữ ngữ-cảnh, xưng-hô nhất-quán), GIỮ ĐÚNG SỐ LƯỢT; lệch số lượt → thử lại 1 lần → bỏ.

    OLLAMA_NUM_PARALLEL=16 ollama serve &   # trên box
    python curriculum/giaotiep/translate_dd.py --workers 16
    → data/giaotiep/dd_envi.jsonl  {"id","split","en":[...],"vi":[...]}
"""
from __future__ import annotations
import argparse, json, os, re, time, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

OUT = "data/giaotiep/dd_envi.jsonl"
PROMPT = """Dịch đoạn hội thoại tiếng Anh dưới đây sang tiếng Việt KHẨU NGỮ, tự nhiên như người Việt nói chuyện hằng ngày
(không dịch word-by-word, không văn viết cứng nhắc). Chọn cách xưng hô hợp ngữ cảnh (anh/chị/em, bạn/mình, cậu/tớ,
ông/bà/cháu...) và GIỮ NHẤT QUÁN trong cả đoạn. Tên riêng giữ nguyên.
Đoạn hội thoại có ĐÚNG {n} lượt nói; trả về đúng {n} lượt đã dịch theo đúng thứ tự.

{dialog}

Trả về JSON: {{"turns": ["lượt 1 đã dịch", "lượt 2 đã dịch", ...]}}"""


def clean(u):
    u = u.strip()
    u = re.sub(r"\s+([,.!?;:%)])", r"\1", u)            # "Jim , how" → "Jim, how"
    u = re.sub(r"([(\$])\s+", r"\1", u)
    u = re.sub(r"\s*’\s*", "'", u)
    u = re.sub(r"(\w)\s+'\s*(s|m|re|ve|ll|d|t)\b", r"\1'\2", u)   # "I ' m" → "I'm"
    u = re.sub(r"\bn\s*'\s*t\b", "n't", u)
    return re.sub(r"\s{2,}", " ", u)


def load_dialogs():
    from datasets import load_dataset
    ds = load_dataset("pixelsandpointers/better_daily_dialog")
    out = []
    for split in ds:
        by = defaultdict(list)
        for r in ds[split]:
            by[r["dialog_id"]].append(clean(r["utterance"]))
        out += [{"id": f"{split}-{k}", "split": split, "en": v} for k, v in by.items() if 2 <= len(v) <= 14]
    return out


def gemma(prompt):
    body = {"model": "gemma4:26b-a4b-it-qat", "stream": False, "think": False, "format": "json",
            "options": {"temperature": 0.3, "num_ctx": 4096}, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["message"]["content"]


def work(d):
    dialog = "\n".join(f"{i+1}. {u}" for i, u in enumerate(d["en"]))
    for _ in range(2):
        try:
            turns = json.loads(gemma(PROMPT.format(n=len(d["en"]), dialog=dialog))).get("turns", [])
        except Exception:
            continue
        turns = [re.sub(r"^\s*\d+[.)]\s*", "", str(t)).strip() for t in turns]
        if len(turns) == len(d["en"]) and all(turns):
            return {**d, "vi": turns}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    dialogs = load_dialogs()
    done = {json.loads(l)["id"] for l in open(OUT, encoding="utf-8")} if os.path.exists(OUT) else set()
    todo = [d for d in dialogs if d["id"] not in done]
    todo.sort(key=lambda d: d["split"] == "train")             # phần GIỮ RIÊNG (validation/test) dịch TRƯỚC
    todo = todo[: a.limit or None]
    print(f"{len(dialogs)} hội-thoại · đã có {len(done)} · cần dịch {len(todo)} · {a.workers} luồng", flush=True)
    T0, ok, bad = time.time(), 0, 0
    with open(OUT, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(work, todo), 1):
            if r is None:
                bad += 1
            else:
                ok += 1
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if i % 200 == 0 or i == len(todo):
                el = time.time() - T0
                print(f"[{el:.0f}s] {i}/{len(todo)} · ok {ok} · bỏ {bad} · {i/el*60:.0f} đoạn/phút "
                      f"· còn ~{(len(todo)-i)/(i/el)/60:.0f} phút", flush=True)
    print(f"### xong · ok {ok} · bỏ {bad}", flush=True)


if __name__ == "__main__":
    main()
