"""§TK_103 · PIVOT QUA TIẾNG ANH: hỏi tiếng Việt → Zamba2 trả lời tiếng Anh → dịch sang tiếng Việt.

Zamba2-2.7B-Instruct HIỂU tiếng Việt và mạnh kiến-thức/code/văn-phòng bằng tiếng Anh, nhưng
KHÔNG VIẾT được tiếng Việt (§13). Pipeline này (cách A, 0 train) bù đúng chỗ yếu:
  [0] vinai/vinai-translate-vi2en-v2: dịch CÂU HỎI sang tiếng Anh (lần chạy đầu cho thấy Zamba2
      hiểu sai tên riêng/chi-tiết tiếng Việt: "Sao Hỏa" → không nhận ra là Mars). Tắt: --no-q-translate
  [1] Zamba2: "Answer in English only" → câu trả lời tiếng Anh
  [2] vinai/vinai-translate-en2vi-v2: dịch TỪNG DÒNG, giữ nguyên khối ```code```,
      ký-hiệu danh-sách (-, 1.) và **đậm**
  Yêu cầu "Dịch sang tiếng Việt: …" → đưa thẳng vào model dịch (bỏ bước 1).

    PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=scripts:src .venv/bin/python scripts/pivot_zamba2_vi.py
    ... --q "Sông Mê Kông chảy qua những nước nào?"
"""
from __future__ import annotations
import argparse, re, sys, time
import torch

sys.path.insert(0, "scripts")
from test_zamba2 import patch_zamba2_block_id, MODEL as ZAMBA

MT = "vinai/vinai-translate-en2vi-v2"
MT_VI2EN = "vinai/vinai-translate-vi2en-v2"   # bước 0: dịch CÂU HỎI sang tiếng Anh (Zamba2 hiểu
                                              # sai tên riêng tiếng Việt, vd "Sao Hỏa" ≠ Mars)

SUITE = [
    "Sao Hỏa có mấy vệ tinh? Tên là gì?",
    "Sông Mê Kông chảy qua những nước nào?",
    "Truyện Kiều do ai viết?",
    "Viết email xin nghỉ phép 2 ngày gửi sếp, lý do việc gia đình, giọng lịch sự.",
    "Tóm tắt biên bản họp thành 3 việc cần làm, có người phụ trách: 'Linh sẽ chốt ngân sách Q3 "
    "trước thứ Sáu. Minh cần sửa lỗi đăng nhập trước khi phát hành. Mọi người nên đọc lại tài liệu "
    "onboarding mới. Dự kiến phát hành ngày 15.'",
    "Viết hàm Python tính số Fibonacci thứ n bằng đệ quy có memoization.",
    "Dịch sang tiếng Việt: 'Artificial intelligence is transforming how businesses operate.'",
]

_LIST = re.compile(r"^(\s*(?:[-*•]|\d+[.)])?\s*)(.*)$")
_DICH = re.compile(r"^\s*Dịch sang tiếng Việt\s*:\s*(.+)$", re.S | re.I)
_FENCE = re.compile(r"```.*?```", re.S)

OLLAMA = "http://127.0.0.1:11434"
GEMMA = "gemma4:26b-a4b-it-qat"


class GemmaTranslator:
    """Dịch bằng Gemma 4 26B (Ollama, local). Dịch CẢ ĐOẠN (có ngữ-cảnh) thay vì từng dòng →
    văn tự-nhiên, giữ tên riêng. `think: False` bắt-buộc: mặc-định Gemma 4 'suy nghĩ' trước
    (~1.300 token, 30s) — tắt đi còn ~1-2s, chất-lượng như nhau. Khối ```code``` bị che bằng
    ⟦CODE_i⟧ trước khi dịch rồi trả lại nguyên-văn."""

    def _chat(self, prompt):
        import json, urllib.request
        body = {"model": GEMMA, "stream": False, "think": False, "options": {"temperature": 0},
                "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=900).read())["message"]["content"].strip()

    def unload(self):
        """Nhả Gemma (~15GB) khỏi RAM trước khi nạp Zamba2 — máy 26GB không giữ nổi cả hai."""
        import json, urllib.request
        req = urllib.request.Request(f"{OLLAMA}/api/generate", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"model": GEMMA, "keep_alive": 0}).encode())
        urllib.request.urlopen(req, timeout=120).read()

    def q2en(self, q):
        return self._chat("Translate this Vietnamese request into English. Use the established English "
                          "name for Vietnamese proper nouns and titles (e.g. 'Sao Hỏa' → 'Mars', "
                          "'Truyện Kiều' → 'The Tale of Kieu'). Keep people's names as they are. "
                          "Output only the translation.\n\n" + q)

    def batch(self, texts):
        return [self._chat("Translate the following text into natural, fluent Vietnamese as a native "
                           "speaker would write it. Output only the translation.\n\n" + t) for t in texts]

    def markdown(self, text):
        blocks = _FENCE.findall(text)
        masked = text
        for i, b in enumerate(blocks):
            masked = masked.replace(b, f"⟦CODE_{i}⟧", 1)
        vi = self._chat(
            "Translate the following text into natural, fluent Vietnamese as a native speaker would "
            "write it (office emails should sound like a Vietnamese office worker wrote them). Keep "
            "markdown formatting, `inline code`, placeholders in square brackets, and tokens like "
            "⟦CODE_0⟧ exactly unchanged. Output only the translation.\n\n" + masked)
        for i, b in enumerate(blocks):
            vi = vi.replace(f"⟦CODE_{i}⟧", b)
        return vi


class Translator:
    def __init__(self, dev, repo=MT, src="en_XX", tgt="vi_VN"):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.dev = dev
        self.tok = AutoTokenizer.from_pretrained(repo, src_lang=src)
        self.m = AutoModelForSeq2SeqLM.from_pretrained(repo).to(dev).eval()
        self.vi_id = self.tok.convert_tokens_to_ids(tgt)          # mã ngôn-ngữ ĐÍCH

    @torch.no_grad()
    def batch(self, texts, bs=8):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to(self.dev)
            ids = self.m.generate(**enc, decoder_start_token_id=self.vi_id, num_beams=5,
                                  early_stopping=True)          # max_length=1024 theo config
            out += self.tok.batch_decode(ids, skip_special_tokens=True)
        return out

    def markdown(self, text):
        """Dịch từng dòng; giữ khối code, dòng trống, ký-hiệu danh-sách, **đậm** bao cả dòng."""
        lines, jobs, in_code = text.split("\n"), [], False
        for i, ln in enumerate(lines):
            if ln.strip().startswith("```"):
                in_code = not in_code; continue
            if in_code or not ln.strip():
                continue
            pre, body = _LIST.match(ln).groups()
            bold = body.startswith("**") and body.endswith("**") and len(body) > 4
            core = body[2:-2] if bold else body
            if re.search(r"[A-Za-z]", core):
                jobs.append((i, pre, bold, core))
        vi = self.batch([c for *_, c in jobs]) if jobs else []
        for (i, pre, bold, _), v in zip(jobs, vi):
            v = re.sub(r"\*\*\s*(.+?)\s*\*\*", r"**\1**", v)     # model dịch chèn space: "** x **"
            lines[i] = pre + (f"**{v}**" if bold else v)
        return "\n".join(lines)


class Zamba:
    def __init__(self, dev, max_new):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        patch_zamba2_block_id()
        self.dev, self.max_new = dev, max_new
        self.tok = AutoTokenizer.from_pretrained(ZAMBA)
        dt = torch.float16 if dev == "mps" else torch.float32
        self.m = AutoModelForCausalLM.from_pretrained(ZAMBA, dtype=dt).to(dev).eval()

    @torch.no_grad()
    def answer_en(self, q):
        msgs = [{"role": "user", "content": f"Answer the following question in English only.\n\nQuestion: {q}"}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True)
        ids = enc["input_ids"].to(self.dev)
        out = self.m.generate(ids, max_new_tokens=self.max_new, do_sample=False,
                              pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def run_gemma(a):
    """Chạy THEO PHA để Gemma (~15GB, Ollama) và Zamba2 (~5GB, MPS) không cùng nằm trong RAM:
    [pha 0] Gemma dịch mọi câu hỏi → nhả Gemma · [pha 1] Zamba2 trả lời hết → nhả Zamba2 ·
    [pha 2] Gemma dịch mọi câu trả lời."""
    import gc
    T0 = time.time()
    g = GemmaTranslator()
    items = []
    for q in (a.q or SUITE):
        t = time.time()
        m = _DICH.match(q)
        if m:
            items.append({"q": q, "direct": True, "en": m.group(1).strip().strip("'\"“”"), "t_q": 0.0})
        else:
            q_en = q if a.no_q_translate else g.q2en(q)
            items.append({"q": q, "direct": False, "q_en": q_en, "t_q": time.time() - t})
    g.unload()
    print(f"[{time.time()-T0:.0f}s] pha 0: Gemma dịch {len(items)} câu hỏi xong → nhả Gemma", flush=True)

    zb = Zamba(a.device, a.max_new)
    print(f"[{time.time()-T0:.0f}s] nạp {ZAMBA}", flush=True)
    for it in items:
        if not it["direct"]:
            t = time.time()
            it["en"] = zb.answer_en(it["q_en"])
            it["t_en"] = time.time() - t
    del zb; gc.collect()
    if a.device == "mps":
        torch.mps.empty_cache()
    print(f"[{time.time()-T0:.0f}s] pha 1: Zamba2 trả lời xong → nhả Zamba2", flush=True)

    for it in items:
        t = time.time()
        it["vi"] = g.batch([it["en"]])[0] if it["direct"] else g.markdown(it["en"])
        it["t_vi"] = time.time() - t
        route = ("dịch-trực-tiếp" if it["direct"] else
                 f"pivot (hỏi→EN {it['t_q']:.0f}s + Zamba2 {it['t_en']:.0f}s + dịch {it['t_vi']:.0f}s)")
        qline = "" if it["direct"] or a.no_q_translate else f"\n🔁 {it['q_en']}"
        print(f"\n{'='*72}\n❓ {it['q']}{qline}\n[{route}]\n{'-'*72}"
              f"\n🇬🇧 {it['en']}\n{'-'*72}\n🇻🇳 {it['vi']}", flush=True)
    print(f"\n[{time.time()-T0:.0f}s] ### xong", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", action="append", default=None)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--max-new", type=int, default=220)
    ap.add_argument("--no-q-translate", action="store_true",
                    help="tắt bước 0 (đưa câu hỏi tiếng Việt thẳng cho Zamba2)")
    ap.add_argument("--translator", default="vinai", choices=["vinai", "gemma"],
                    help="vinai = mBART dịch từng dòng · gemma = Gemma 4 26B (Ollama) dịch cả đoạn")
    a = ap.parse_args()
    if a.translator == "gemma":
        return run_gemma(a)
    T0 = time.time()
    tr = Translator(a.device)
    q2en = None if a.no_q_translate else Translator(a.device, MT_VI2EN, "vi_VN", "en_XX")
    print(f"[{time.time()-T0:.0f}s] nạp {MT}" + ("" if a.no_q_translate else f" + {MT_VI2EN}"), flush=True)
    zb = Zamba(a.device, a.max_new)
    print(f"[{time.time()-T0:.0f}s] nạp {ZAMBA}", flush=True)
    for q in (a.q or SUITE):
        t = time.time()
        m = _DICH.match(q)
        if m:                                              # yêu cầu dịch → thẳng model dịch
            en = m.group(1).strip().strip("'\"“”")
            vi = tr.batch([en])[0]
            route = "dịch-trực-tiếp"
        else:
            q_en = q2en.batch([q])[0] if q2en else q
            en = zb.answer_en(q_en)
            t_en = time.time() - t
            vi = tr.markdown(en)
            route = f"pivot (EN {t_en:.0f}s + dịch {time.time()-t-t_en:.0f}s)"
        qline = f"\n🔁 {q_en}" if (not m and q2en) else ""
        print(f"\n{'='*72}\n❓ {q}{qline}\n[{route} · tổng {time.time()-t:.0f}s]\n{'-'*72}"
              f"\n🇬🇧 {en}\n{'-'*72}\n🇻🇳 {vi}", flush=True)
    print(f"\n[{time.time()-T0:.0f}s] ### xong", flush=True)


if __name__ == "__main__":
    main()
