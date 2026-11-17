"""TK_104 · Môn GIAO-TIẾP v3: Gemma SINH hội-thoại đời-thường song-ngữ MỚI, mỗi lượt phải ĂN NHẬP với câu trước.

Điểm yếu đo được ở v2: model nói câu nghe như hội-thoại nhưng hay lạc đề (hợp ngữ-cảnh 3,2/5). Giả-thuyết
"dữ-liệu ít mà chất": dữ-liệu ÉP mỗi lượt dùng lại CHI-TIẾT cụ-thể của lượt trước (tên, giờ, số, nơi chốn) dạy
được việc hiểu ý — thay cho tăng cỡ model.

Mỗi yêu-cầu bốc ngẫu-nhiên: chủ-đề × quan-hệ người nói × bộ chi-tiết (tên, giờ, giá, nơi) → đa-dạng, không trùng
DailyDialog. Gemma viết bản tiếng Việt khẩu-ngữ + bản tiếng Anh SONG-SONG (đúng số lượt).
Phần --probe: hội-thoại kết bằng câu HỎI LẠI một chi-tiết đã nêu + đáp-án ngắn → bài chấm KHÁCH-QUAN "nhớ chi-tiết".

    python curriculum/giaotiep/gen_dialog_grounded.py --n 8000 --workers 32            → data/giaotiep/grounded.jsonl
    python curriculum/giaotiep/gen_dialog_grounded.py --probe --n 400 --seed 7          → data/giaotiep/probe.jsonl
"""
from __future__ import annotations
import argparse, json, os, random, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

CHU_DE = ["hẹn đi ăn tối", "mua quần áo ở chợ", "hỏi đường đến bệnh viện", "đặt bàn nhà hàng", "đi khám răng",
          "mua vé xe khách về quê", "thuê phòng trọ", "sửa xe máy bị hỏng", "rủ đi xem phim", "lên kế hoạch du lịch Đà Lạt",
          "hỏi bài tập về nhà", "chuẩn bị sinh nhật bạn", "đi siêu thị mua đồ ăn", "gọi đồ ăn giao tận nhà",
          "phỏng vấn xin việc", "xin nghỉ phép với sếp", "họp nhóm làm dự án", "mượn sách thư viện", "đi tập gym",
          "nấu ăn cùng nhau", "trả lại hàng bị lỗi", "đổi tiền ở ngân hàng", "đặt phòng khách sạn", "đi cắt tóc",
          "hỏi thăm người ốm", "chọn quà cho mẹ", "đi đá bóng cuối tuần", "học lái xe", "chuyển nhà", "mua điện thoại mới",
          "đón người thân ở sân bay", "đi chợ hoa ngày Tết", "hỏi giá thuê xe", "đăng ký lớp học tiếng Anh",
          "mất ví trên xe buýt", "trời mưa hoãn buổi dã ngoại", "nhận hàng bưu điện", "đi khám sức khoẻ định kỳ",
          "góp tiền mua quà chung", "hỏi về món ăn trong thực đơn", "làm thủ tục ở phường", "đi cà phê nói chuyện công việc",
          "hướng dẫn dùng máy giặt", "đặt lịch sửa điều hoà", "chuẩn bị đồ đi cắm trại", "chăm cây trong vườn",
          "đi xem bóng đá ở sân", "mua thuốc ở hiệu thuốc", "đổi ca làm việc", "đi họp phụ huynh"]
QUAN_HE = ["hai người bạn thân (cậu/tớ)", "hai bạn cùng lớp (bạn/mình)", "mẹ và con (mẹ/con)", "khách hàng và người bán (anh/chị/em)",
           "hai đồng nghiệp (anh/em)", "sếp và nhân viên (anh/em)", "bác sĩ và bệnh nhân (bác sĩ/tôi)",
           "hai vợ chồng (anh/em)", "ông và cháu (ông/cháu)", "lễ tân và khách (quý khách/em)", "hàng xóm (bác/cháu)"]
TEN = ["Lan", "Minh", "Hùng", "Mai", "Tuấn", "Hà", "Nam", "Thảo", "Long", "Hương", "Đức", "Linh", "Quang", "Trang", "Phong"]
NOI = ["Quận 1", "phố Huế", "chợ Bến Thành", "Hồ Gươm", "đường Lê Lợi", "công viên Thống Nhất", "ga Hà Nội", "Vincom",
       "quán Phở Thìn", "trường Nguyễn Du", "bến xe Miền Đông", "hồ Tây", "cầu Rồng", "Nha Trang", "chợ Đà Lạt"]


def seed_details(rnd):
    return {"ten": rnd.sample(TEN, 2), "gio": f"{rnd.randint(6, 21)} giờ{rnd.choice(['', ' rưỡi', ' 15'])}",
            "gia": rnd.choice(["25 nghìn", "35 nghìn", "50 nghìn", "80 nghìn", "120 nghìn", "150 nghìn", "250 nghìn",
                           "400 nghìn", "600 nghìn", "1 triệu", "2 triệu rưỡi", "5 triệu"]),
            "noi": rnd.choice(NOI), "so": rnd.randint(2, 12), "thu": rnd.choice(["thứ Hai", "thứ Tư", "thứ Sáu", "thứ Bảy", "Chủ nhật"])}


PROMPT = """Viết MỘT đoạn hội thoại đời thường giữa {qh} về chủ đề: {cd}.
Dùng một vài chi tiết cụ thể sau (tự nhiên, không cần dùng hết): tên {t1}, {t2}; lúc {gio} {thu}; giá {gia}; ở {noi}; số lượng {so}.

YÊU CẦU QUAN TRỌNG:
- {n} lượt nói, hai người nói xen kẽ. Mỗi lượt 1-2 câu ngắn, khẩu ngữ tự nhiên như người Việt nói thật, xưng hô nhất quán.
- MỖI LƯỢT PHẢI ĂN NHẬP với lượt ngay trước: trả lời đúng câu hỏi được hỏi, hoặc phản hồi đúng điều vừa nói,
  và NHẮC LẠI hoặc dùng lại ít nhất một chi tiết cụ thể (tên, giờ, giá, nơi, số lượng, đồ vật) của lượt trước.
- Không lan man, không chào hỏi sáo rỗng kéo dài.
- Viết bản tiếng Việt và bản tiếng Anh tương ứng (dịch tự nhiên, đúng từng lượt, cùng số lượt).
{probe}
Trả về JSON: {{"vi": ["lượt 1", ...], "en": ["turn 1", ...]{probe_json}}}"""
PROBE = """- Lượt CUỐI CÙNG: người nói hỏi lại một chi tiết cụ thể đã được nhắc ở các lượt trước (ví dụ: "Mấy giờ hẹn nhỉ?",
  "Bao nhiêu tiền ấy nhỉ?", "Ở đâu cơ?"). KHÔNG viết câu trả lời cho câu hỏi cuối.
- Cho biết đáp án ngắn (chỉ chi tiết đó, đúng như trong hội thoại) bằng tiếng Việt và tiếng Anh."""


SPEAKER = re.compile(r"^\s*(?:[AB]|\d+|[^\W\d_][\w'’.]*(?: [^\W\d_][\w'’.]*){0,2})\s*[:.)]\s+")


def strip_speaker(t):
    """Gemma hay chèn TÊN người nói đầu lượt ('Sếp: …', 'Phong: …', 'A: …') — ~36% hội-thoại → gỡ."""
    t = str(t).strip()
    m = SPEAKER.match(t)
    return t[m.end():].strip() if m and len(m.group(0)) <= 25 else t


def strip_dialog(turns):
    """Chỉ gỡ khi ≥ nửa số lượt có dạng 'Tên: …' (chắc-chắn là kiểu ghi tên người nói) — tránh cắt nhầm câu
    thật có dấu hai chấm ('Tôi nghĩ là: được đấy')."""
    turns = [str(t).strip() for t in turns]
    hits = sum(bool(SPEAKER.match(t)) for t in turns)
    return [strip_speaker(t) for t in turns] if turns and hits * 2 >= len(turns) else turns


def clean_file(path):
    rows = []
    for l in open(path, encoding="utf-8"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        r["vi"], r["en"] = strip_dialog(r["vi"]), strip_dialog(r["en"])
        rows.append(r)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def gemma(prompt, temp=0.8):
    body = {"model": "gemma4:26b-a4b-it-qat", "stream": False, "think": False, "format": "json",
            "options": {"temperature": temp, "num_ctx": 4096}, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["message"]["content"]


def work(job):
    i, seed, probe = job
    rnd = random.Random(seed)
    d = seed_details(rnd)
    n = rnd.randint(5, 9) if probe else rnd.randint(6, 10)
    meta = {"chu_de": rnd.choice(CHU_DE), "quan_he": rnd.choice(QUAN_HE)}
    p = PROMPT.format(qh=meta["quan_he"], cd=meta["chu_de"], t1=d["ten"][0], t2=d["ten"][1], gio=d["gio"], thu=d["thu"],
                      gia=d["gia"], noi=d["noi"], so=d["so"], n=n, probe=PROBE if probe else "",
                      probe_json=', "dap_an_vi": "…", "dap_an_en": "…"' if probe else "")
    for _ in range(2):
        try:
            j = json.loads(gemma(p))
        except Exception:
            continue
        vi = strip_dialog(j.get("vi", []))
        en = strip_dialog(j.get("en", []))
        if len(vi) < 4 or len(vi) != len(en) or not all(vi) or not all(en):
            continue
        r = {"id": f"{'probe' if probe else 'g'}-{i}", "split": "probe" if probe else "train", **meta, "vi": vi, "en": en}
        if probe:
            a_vi, a_en = str(j.get("dap_an_vi", "")).strip(), str(j.get("dap_an_en", "")).strip()
            if not a_vi or not a_en:
                continue
            r.update(dap_an_vi=a_vi, dap_an_en=a_en)
        return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--seed", type=int, default=104)
    ap.add_argument("--out", default="")
    ap.add_argument("--clean", nargs="*", help="chỉ làm sạch (gỡ tên người nói) các file có sẵn rồi thoát")
    a = ap.parse_args()
    if a.clean:
        for p_ in a.clean:
            print(p_, clean_file(p_), "hội-thoại đã làm sạch")
        return
    out = a.out or ("data/giaotiep/probe.jsonl" if a.probe else "data/giaotiep/grounded.jsonl")
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
    jobs = [(i, a.seed * 1_000_003 + i, a.probe) for i in range(a.n)
            if f"{'probe' if a.probe else 'g'}-{i}" not in done]
    print(f"cần sinh {len(jobs)} (đã có {len(done)}) · {a.workers} luồng → {out}", flush=True)
    T0, ok, bad = time.time(), 0, 0
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(work, jobs), 1):
            if r is None:
                bad += 1
            else:
                ok += 1
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if k % 100 == 0 or k == len(jobs):
                el = time.time() - T0
                print(f"[{el:.0f}s] {k}/{len(jobs)} · ok {ok} · bỏ {bad} · {k/el*60:.0f}/phút · còn ~{(len(jobs)-k)/(k/el)/60:.0f} phút",
                      flush=True)
    print(f"### xong · ok {ok} · bỏ {bad}", flush=True)


if __name__ == "__main__":
    main()
