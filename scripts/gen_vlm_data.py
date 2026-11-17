"""VLM V1 · dữ liệu ẢNH TỰ NHIÊN + biểu đồ/tài liệu cho model đọc ảnh (TK_106 §3). Ảnh màn hình: gen_screens.py.

Ổ box chật → đọc dạng LUỒNG (streaming), lưu lại ảnh JPEG nén (cạnh dài ≤768px; tài liệu/biểu đồ ≤1280px).

  fetch     coco (ảnh đời thường, chưa có chữ Việt) · ktvic (ảnh đời sống VN + chú thích Việt) · openvivqa (ảnh VN +
            hỏi–đáp Việt, MIT) · vichart (biểu đồ hỏi–đáp Việt) · cauldron (chartqa, textvqa, st_vqa, ai2d, scienceqa,
            docvqa, screen2words, tabmwp, infographic_vqa — hỏi–đáp tiếng Anh)
  vision    thầy Gemma-4-31B NHÌN ẢNH (coco/ktvic/openvivqa) → mô tả ngắn + mô tả chi tiết + 3–4 cặp hỏi–đáp tiếng Việt
            (có chú thích/hỏi–đáp gốc của người → đưa vào làm gợi ý để thầy bám đúng)
  translate thầy (chỉ chữ) dịch hỏi–đáp cauldron sang tiếng Việt, đáp thành câu trọn vẹn, GIỮ NGUYÊN chữ/số có trong ảnh
  build     → data/vlm/align.jsonl (giai đoạn 1: mô tả) · sft.jsonl (giai đoạn 2: hỏi–đáp + ảnh màn hình + ~10% chỉ chữ
            để expert thị giác học "đóng cổng" khi không có ảnh) · test_*.jsonl (giữ riêng)

    python scripts/gen_vlm_data.py fetch --only coco,ktvic,openvivqa,vichart,cauldron
    python scripts/gen_vlm_data.py vision --llm http://127.0.0.1:8000/v1 --workers 96
    python scripts/gen_vlm_data.py translate --llm http://127.0.0.1:8000/v1 --workers 96
    python scripts/gen_vlm_data.py build
"""
from __future__ import annotations
import argparse, ast, base64, io, json, os, random, re, sys, time, urllib.request, zipfile
import concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = "data/vlm"
NAT = f"{ROOT}/nat"
CAUL = {"chartqa": 8000, "textvqa": 8000, "st_vqa": 6000, "ai2d": 2434, "scienceqa": 4976, "docvqa": 5000,
        "screen2words": 6000, "tabmwp": 6000, "infographic_vqa": 2118}
TRANS_CAP = {"chartqa": 5000, "textvqa": 5000, "st_vqa": 4000, "ai2d": 2434, "scienceqa": 4000, "docvqa": 4000,
             "screen2words": 3000, "tabmwp": 2500, "infographic_vqa": 2000}
BIG = {"docvqa", "infographic_vqa", "chartqa", "screen2words", "tabmwp"}       # ảnh cần nét chữ → giữ ≤1280px


def save_img(img, path, max_side=768):
    img = img.convert("RGB")
    k = max_side / max(img.size)
    if k < 1:
        img = img.resize((round(img.width * k), round(img.height * k)))
    img.save(path, "JPEG", quality=88)


def jl_write(path, rows, mode="a"):
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def rows_of(path):
    out = []
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:                 # dòng cuối bị cắt dở khi dừng tiến trình giữa lúc ghi
                pass
    return out


# ── fetch ─────────────────────────────────────────────────────────────────────────────
def f_coco(a):
    from datasets import load_dataset
    out = []
    for i, r in enumerate(load_dataset("detection-datasets/coco", split="train", streaming=True)):
        if i >= a.n_coco:
            break
        p = f"{NAT}/img/coco_{r['image_id']}.jpg"
        save_img(r["image"], p)
        out.append({"id": f"coco_{r['image_id']}", "src": "coco", "image": p[len(ROOT) + 1:]})
        if len(out) % 1000 == 0:                                    # ghi dần → thầy nhìn ảnh bắt đầu sớm
            jl_write(f"{NAT}/raw.jsonl", out[-1000:])
            print(f"[coco] {i + 1}", flush=True)
    jl_write(f"{NAT}/raw.jsonl", out[len(out) // 1000 * 1000:])
    return len(out)


def f_ktvic(a):
    from datasets import load_dataset
    out = []
    for split in ("train", "test"):
        n = a.n_ktvic if split == "train" else 300
        for i, r in enumerate(load_dataset("Skyler215/KTVIC", split=split, streaming=True)):
            if i >= n:
                break
            p = f"{NAT}/img/ktvic_{split}_{i}.jpg"
            save_img(r["img_new"], p)
            out.append({"id": f"ktvic_{split}_{i}", "src": "ktvic", "split": split, "image": p[len(ROOT) + 1:],
                        "ref": r["labels_new"]})
    jl_write(f"{NAT}/raw.jsonl", out)
    return len(out)


def f_openvivqa(a):
    from huggingface_hub import hf_hub_download
    from PIL import Image
    out = []
    for split, js, zp in (("train", "vlsp2023_train_data.json", "train-images.zip"),
                          ("test", "vlsp2023_dev_data.json", "dev-images.zip")):
        d = json.load(open(hf_hub_download("uitnlp/OpenViVQA-dataset", js, repo_type="dataset")))
        z = zipfile.ZipFile(hf_hub_download("uitnlp/OpenViVQA-dataset", zp, repo_type="dataset"))
        names = {os.path.basename(n): n for n in z.namelist() if n.lower().endswith((".jpg", ".png", ".jpeg"))}
        qa = {}
        for v in d["annotations"].values():
            qa.setdefault(str(v["image_id"]), []).append({"q": v["question"], "a": v["answer"]})
        for iid, fn in d["images"].items():
            if str(iid) not in qa or fn not in names:
                continue
            p = f"{NAT}/img/ovqa_{split}_{iid}.jpg"
            save_img(Image.open(io.BytesIO(z.read(names[fn]))), p)
            out.append({"id": f"ovqa_{split}_{iid}", "src": "openvivqa", "split": split, "image": p[len(ROOT) + 1:],
                        "qa": qa[str(iid)]})
        z.close()
        os.remove(os.path.realpath(z.filename))                    # bỏ file zip gốc (ổ chật) — đã lưu bản nén
    jl_write(f"{NAT}/raw.jsonl", out)
    return len(out)


def f_vichart(a):
    from datasets import load_dataset
    out = []
    for split in ("train", "test"):
        for i, r in enumerate(load_dataset("Zenng2812/vqa-vietnamese-charts", split=split, streaming=True)):
            p = f"{ROOT}/chart/img/{split}_{i}.jpg"
            save_img(r["image"], p, 1280)
            out.append({"id": f"vichart_{split}_{i}", "src": "vichart", "split": split, "image": p[len(ROOT) + 1:],
                        "messages": [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}]})
    jl_write(f"{ROOT}/chart/raw.jsonl", out)
    return len(out)


def f_cauldron(a):
    from datasets import load_dataset
    out = []
    for sub, n in CAUL.items():
        k = 0
        for r in load_dataset("HuggingFaceM4/the_cauldron", sub, split="train", streaming=True):
            if k >= n:
                break
            if len(r["images"]) != 1:
                continue
            p = f"{ROOT}/caul/img/{sub}_{k}.jpg"
            save_img(r["images"][0], p, 1280 if sub in BIG else 768)
            texts = r["texts"] if isinstance(r["texts"], list) else ast.literal_eval(r["texts"])
            out.append({"id": f"{sub}_{k}", "src": "caul_" + sub, "image": p[len(ROOT) + 1:],
                        "en": [{"user": t["user"], "assistant": t["assistant"]} for t in texts[:6]]})
            k += 1
        print(f"[cauldron] {sub} {k}", flush=True)
    jl_write(f"{ROOT}/caul/raw.jsonl", out)
    return len(out)


def cmd_fetch(a):
    for d in (f"{NAT}/img", f"{ROOT}/chart/img", f"{ROOT}/caul/img"):
        os.makedirs(d, exist_ok=True)
    fns = {"coco": f_coco, "ktvic": f_ktvic, "openvivqa": f_openvivqa, "vichart": f_vichart, "cauldron": f_cauldron}
    for k in a.only.split(","):
        t = time.time()
        print(f"### {k}: {fns[k](a)} mẫu · {time.time()-t:.0f}s", flush=True)


# ── thầy ──────────────────────────────────────────────────────────────────────────────
MODEL = os.environ.get("LLM_MODEL", "gemma4-31b")


def chat(url, content, temp=0.6, max_tokens=1500, json_mode=True):
    body = {"model": MODEL, "temperature": temp, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}], "chat_template_kwargs": {"enable_thinking": False}}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["choices"][0]["message"]["content"]


def img_part(path):
    b = base64.b64encode(open(path, "rb").read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}


P_VISION = """Bạn là người tạo dữ liệu huấn luyện cho một trợ lý AI tiếng Việt biết NHÌN ẢNH. Hãy quan sát kỹ bức ảnh.
{hint}
Viết bằng tiếng Việt tự nhiên, CHỈ nêu điều thật sự thấy trong ảnh (không đoán bừa tên người, địa danh, thương hiệu nếu
không có chữ/dấu hiệu rõ; chữ xuất hiện trong ảnh thì chép đúng nguyên văn). Không dùng LaTeX, không dùng markdown đậm.
Trả về DUY NHẤT JSON:
{{"mo_ta_ngan": "1–2 câu nêu ảnh có gì",
  "mo_ta_chi_tiet": "4–7 câu: chủ thể, hành động, bối cảnh, màu sắc, vị trí, chữ trong ảnh (nếu có), không khí chung",
  "hoi_dap": [{{"hoi": "...", "dap": "..."}}, ...]}}
"hoi_dap": 4 cặp ĐA DẠNG như người dùng thật hỏi về ảnh (đếm số lượng, màu sắc, vị trí/không gian, đang làm gì,
chữ trong ảnh, suy luận hợp lý từ ảnh, gợi ý/ứng dụng thực tế). Câu đáp trọn vẹn 1–4 câu, đúng với ảnh."""

P_TRANS = """Dưới đây là các lượt hỏi–đáp (tiếng Anh) về MỘT bức ảnh ({kind}). Trợ lý tiếng Việt cần học cách trả lời.
{conv}
Chuyển thành tiếng Việt tự nhiên:
- câu hỏi: dịch tự nhiên (bỏ các dặn kiểu "Offer a terse response", "Answer with a single word").
- câu đáp: thành câu trọn vẹn tiếng Việt, GIỮ ĐÚNG giá trị/đáp án gốc (số, đơn vị, lựa chọn).
- màu sắc và từ ngữ thông thường PHẢI dịch (green → xanh lá, blue → xanh dương, orange → cam, bar → cột, graph/line → đường…).
  CHỈ giữ nguyên văn những chữ được VIẾT trong ảnh: tên riêng, chữ trên biển hiệu, tiêu đề, nhãn trục/chú giải, chữ trong tài
  liệu (đặt trong ngoặc kép), có thể giải thích nghĩa bằng tiếng Việt. Câu hỏi giữ đúng ý gốc, không đổi nội dung hỏi.
- câu hỏi trắc nghiệm: giữ các lựa chọn, đáp nêu lựa chọn đúng kèm nội dung.
Trả về DUY NHẤT JSON: {{"hoi_dap": [{{"hoi": "...", "dap": "..."}}, ...]}} (cùng số lượt, cùng thứ tự)."""
KIND = {"chartqa": "biểu đồ", "textvqa": "ảnh chụp có chữ", "st_vqa": "ảnh chụp có chữ", "ai2d": "sơ đồ khoa học",
        "scienceqa": "hình minh hoạ bài khoa học", "docvqa": "ảnh chụp tài liệu", "screen2words": "ảnh chụp màn hình điện thoại",
        "tabmwp": "bảng số liệu (bài toán)", "infographic_vqa": "đồ hoạ thông tin"}


def run_pool(fn, todo, out_path, workers):
    from gen_office import loads_loose, CJK
    ok = 0
    T0 = time.time()
    with cf.ThreadPoolExecutor(workers) as ex, open(out_path, "a", encoding="utf-8") as f:
        futs = {ex.submit(fn, r): r for r in todo}
        for k, fu in enumerate(cf.as_completed(futs)):
            try:
                res = fu.result()
            except Exception as e:
                res = None
            if res:
                f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush(); ok += 1
            if k % 500 == 0:
                print(f"  {k}/{len(todo)} · được {ok} · {time.time()-T0:.0f}s", flush=True)
    return ok


def cmd_vision(a):
    from gen_office import loads_loose, CJK
    import glob, zlib
    raw = rows_of(f"{NAT}/raw.jsonl")
    out = f"{NAT}/vision.jsonl"
    done = {r["id"] for p in [out] + glob.glob(f"{NAT}/vision_s*.jsonl") for r in rows_of(p)}
    coco = sorted(r["id"] for r in raw if r["src"] == "coco")[:a.max_coco]      # thầy ~1,5K token/s → giới hạn COCO
    keep = set(coco) | {r["id"] for r in raw if r["src"] != "coco"}
    todo = [r for r in raw if r["id"] not in done and r["id"] in keep]
    if a.shard:                                    # chia cho nhiều thầy: mỗi phần 1 file riêng (gộp vào vision.jsonl sau)
        i, n = map(int, a.shard.split("/"))
        todo = [r for r in todo if zlib.crc32(r["id"].encode()) % n == i]
        out = f"{NAT}/vision_s{i}.jsonl"
    if a.limit:
        todo = todo[:a.limit]
    print(f"[vision] {len(todo)} ảnh cần thầy nhìn", flush=True)

    def work(r):
        hint = ""
        if r.get("ref"):
            hint = f"Gợi ý (chú thích do người viết, đúng nhưng ngắn): \"{r['ref']}\"."
        elif r.get("qa"):
            hint = "Gợi ý (hỏi–đáp do người viết về ảnh này): " + "; ".join(f"{x['q']} → {x['a']}" for x in r["qa"][:5]) + "."
        txt = chat(a.llm, [img_part(f"{ROOT}/{r['image']}"), {"type": "text", "text": P_VISION.format(hint=hint)}])
        j = loads_loose(txt)
        qa = [x for x in j.get("hoi_dap", []) if x.get("hoi") and x.get("dap")]
        blob = json.dumps(j, ensure_ascii=False)
        if not j.get("mo_ta_chi_tiet") or len(qa) < 2 or CJK.search(blob) or "\\(" in blob:
            return None
        return {**r, "mo_ta_ngan": j["mo_ta_ngan"].strip(), "mo_ta_chi_tiet": j["mo_ta_chi_tiet"].strip(), "hoi_dap": qa}
    print(f"### vision: {run_pool(work, todo, out, a.workers)}/{len(todo)}", flush=True)


def cmd_translate(a):
    from gen_office import loads_loose, CJK
    raw = rows_of(f"{ROOT}/caul/raw.jsonl")
    out = f"{ROOT}/caul/vi.jsonl"
    done = {r["id"] for r in rows_of(out)}
    cnt = {}
    sel = []
    for r in raw:                                  # thầy ~1,5K token/s → mỗi loại tối đa TRANS_CAP mẫu, ≤3 lượt/mẫu
        sub = r["src"][5:]
        cnt[sub] = cnt.get(sub, 0) + 1
        if cnt[sub] <= TRANS_CAP.get(sub, 3000):
            sel.append({**r, "en": r["en"][:3]})
    todo = [r for r in sel if r["id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[translate] {len(todo)} mẫu", flush=True)

    def work(r):
        sub = r["src"][5:]
        conv = "\n".join(f"Q{i+1}: {t['user']}\nA{i+1}: {t['assistant']}" for i, t in enumerate(r["en"]))
        j = loads_loose(chat(a.llm, P_TRANS.format(kind=KIND.get(sub, "ảnh"), conv=conv[:6000]), temp=0.3))
        qa = [x for x in j.get("hoi_dap", []) if x.get("hoi") and x.get("dap")]
        if len(qa) != len(r["en"]) or CJK.search(json.dumps(qa, ensure_ascii=False)):
            return None
        return {**r, "hoi_dap": qa}
    print(f"### translate: {run_pool(work, todo, out, a.workers)}/{len(todo)}", flush=True)


# ── build ─────────────────────────────────────────────────────────────────────────────
ASK_SHORT = ["Mô tả ngắn gọn bức ảnh này.", "Ảnh này có gì?", "Trong ảnh là gì vậy?", "Tóm tắt nội dung ảnh trong một câu."]
ASK_LONG = ["Mô tả chi tiết bức ảnh này.", "Bạn hãy tả kỹ những gì thấy trong ảnh.", "Nhìn ảnh này bạn thấy những gì?",
            "Mô tả ảnh giúp mình nhé.", "Hãy mô tả bức ảnh thật đầy đủ."]


def conv(img, pairs, src, **kw):
    msgs = []
    for q, a_ in pairs:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a_}]
    return {"images": [img], "messages": msgs, "src": src, **kw}


def cmd_build(a):
    rng = random.Random(106)
    align, sft, tests = [], [], {"nat": [], "screen": [], "chart": [], "ovqa": []}
    # ảnh tự nhiên (thầy nhìn ảnh)
    seen_v = set()
    for r in rows_of(f"{NAT}/vision.jsonl"):
        if r["id"] in seen_v:
            continue
        seen_v.add(r["id"])
        test = r.get("split") == "test" or (r["src"] == "coco" and rng.random() < 0.015)
        if test:
            tests["nat"].append(conv(r["image"], [(rng.choice(ASK_LONG), r["mo_ta_chi_tiet"])] +
                                     [(x["hoi"], x["dap"]) for x in r["hoi_dap"][:2]], "test_" + r["src"], max_tiles=4))
            continue
        align.append(conv(r["image"], [(rng.choice(ASK_SHORT), r["mo_ta_ngan"])], "align_" + r["src"], max_tiles=1))
        align.append(conv(r["image"], [(rng.choice(ASK_LONG), r["mo_ta_chi_tiet"])], "align_" + r["src"], max_tiles=1))
        qa = [(x["hoi"], x["dap"]) for x in r["hoi_dap"]]
        rng.shuffle(qa)
        first = [(rng.choice(ASK_LONG), r["mo_ta_chi_tiet"])] if rng.random() < 0.4 else []
        sft.append(conv(r["image"], first + qa[:rng.choice([1, 2, 3, 4])], "nat_" + r["src"], max_tiles=4))
    # chú thích gốc tiếng Việt của người (KTVIC) → thêm vào căn chỉnh
    for r in rows_of(f"{NAT}/raw.jsonl"):
        if r["src"] == "ktvic" and r.get("split") == "train":
            cap = r["ref"].strip()
            align.append(conv(r["image"], [(rng.choice(ASK_SHORT), cap[:1].upper() + cap[1:] + ("" if cap.endswith(".") else "."))], "align_ktvic_ref", max_tiles=1))
        if r["src"] == "openvivqa":
            pairs = [(x["q"][:1].upper() + x["q"][1:], x["a"][:1].upper() + x["a"][1:]) for x in r["qa"][:3]]
            (tests["ovqa"] if r["split"] == "test" else sft).append(conv(r["image"], pairs, "ovqa", max_tiles=4))
    # biểu đồ tiếng Việt
    for r in rows_of(f"{ROOT}/chart/raw.jsonl"):
        (tests["chart"] if r["split"] == "test" else sft).append({"images": [r["image"]], "messages": r["messages"], "src": "vichart"})
    # cauldron đã dịch (khử trùng lặp theo id — từng có 2 tiến trình dịch chồng nhau)
    seen = set()
    for r in rows_of(f"{ROOT}/caul/vi.jsonl"):
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        pairs = [(x["hoi"], x["dap"]) for x in r["hoi_dap"]]
        if r["src"] == "caul_screen2words":
            align.append(conv(r["image"], pairs[:1], "align_screen2words", max_tiles=4))
        else:
            sft.append(conv(r["image"], pairs, r["src"]))
    # ảnh chụp màn hình tự dựng (đáp án chính xác) — đường dẫn ảnh tương đối theo data/vlm
    for r in rows_of(f"{ROOT}/screens/train.jsonl"):
        if any(not m["content"] for m in r["messages"]):           # còn lượt "teach" chưa có thầy viết
            continue
        r["images"] = ["screens/" + p for p in r["images"]]
        sft.append(r)
        if r["messages"][0]["content"].startswith("Chép lại") and rng.random() < 0.3:
            align.append({**r, "messages": r["messages"][:2], "src": "align_" + r["src"]})
    for r in rows_of(f"{ROOT}/screens/test.jsonl"):
        if all(m["content"] for m in r["messages"]):
            r["images"] = ["screens/" + p for p in r["images"]]
            tests["screen"].append(r)
    # ~10% chỉ chữ (không ảnh) — giữ kỹ năng cũ, dạy expert thị giác khép cổng khi không có ảnh
    txt = []
    for p, k in (("data/multiturn/train.jsonl", 4000), ("data/office/train.jsonl", 2000), ("data/sgk/train.jsonl", 3000),
                 ("data/code_tb/train.jsonl", 2000), ("data/calib/train.jsonl", 1500)):
        rs = rows_of(p); rng.shuffle(rs)
        txt += [{"images": [], "messages": x["messages"], "src": "text_" + p.split("/")[1]} for x in rs[:k]]
    n_txt = min(len(txt), len(sft) // 9)
    sft += txt[:n_txt]
    rng.shuffle(align); rng.shuffle(sft)
    jl_write(f"{ROOT}/align.jsonl", align, "w")
    jl_write(f"{ROOT}/sft.jsonl", sft, "w")
    for k, v in tests.items():
        jl_write(f"{ROOT}/test_{k}.jsonl", v, "w")
    import collections
    c = collections.Counter(r["src"].split("_")[0] if r["src"].startswith(("align", "text")) else r["src"] for r in sft)
    print(f"### align {len(align)} · sft {len(sft)} (chỉ chữ {n_txt}) · test " + " · ".join(f"{k} {len(v)}" for k, v in tests.items()))
    print("sft theo nguồn:", dict(c.most_common()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "vision", "translate", "build"])
    ap.add_argument("--only", default="coco,ktvic,openvivqa,vichart,cauldron")
    ap.add_argument("--n-coco", type=int, default=30000)
    ap.add_argument("--n-ktvic", type=int, default=6000)
    ap.add_argument("--max-coco", type=int, default=20000, help="số ảnh COCO tối đa cho thầy nhìn")
    ap.add_argument("--llm", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="", help="vision: phần i/n (vd 0/3) → ghi nat/vision_s<i>.jsonl")
    a = ap.parse_args()
    {"fetch": cmd_fetch, "vision": cmd_vision, "translate": cmd_translate, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
