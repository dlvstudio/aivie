"""Thử nhanh model NỀN (chưa SFT) trên M4: viết tiếp nhiều prompt + trắc nghiệm kiến thức chấm tự động.

    .venv/bin/python scripts/probe_base_prompts.py --ckpts checkpoints/hf_hybrid/hybrid_2p6b_b9536_bf16.pt,checkpoints/hf_hybrid/hybrid_2p6b_b1500_bf16.pt
    → out/probe_base_<tên>.md (văn bản sinh) + bảng trắc nghiệm in ra cuối.

• Viết tiếp: model nền không có khuôn Hỏi/Đáp → prompt là đầu văn bản (tin, wiki, truyện, code, toán vài-mẫu…).
• Trắc nghiệm: mỗi câu có 1 đáp án đúng + 3 sai; chọn đáp án có log-xác-suất TRUNG BÌNH theo token cao nhất
  (không sinh → khách quan, so được giữa các checkpoint). Ngẫu nhiên = 25%.
"""
from __future__ import annotations
import argparse, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "src")]
os.chdir(ROOT)

import torch
import torch.nn.functional as F
from hybrid_infer import load, Runner
from nn_aivie.core.tokenizer import BPETokenizer
from nn_aivie.core.vocab import Vocabulary

PROMPTS = [
    # tin tức / văn xuôi
    ("tin", "Hà Nội, ngày 15/3 – Sở Giao thông vận tải thành phố cho biết"),
    ("tin", "Theo Tổng cục Thống kê, GDP của Việt Nam trong quý I năm nay"),
    ("tin", "Đội tuyển bóng đá Việt Nam đã có chiến thắng"),
    # bách khoa
    ("wiki", "Vịnh Hạ Long là một vịnh nhỏ thuộc phần bờ tây vịnh Bắc Bộ,"),
    ("wiki", "Chủ tịch Hồ Chí Minh sinh ngày 19 tháng 5 năm 1890 tại"),
    ("wiki", "Sông Mê Kông bắt nguồn từ"),
    ("wiki", "Quang hợp là quá trình"),
    ("wiki", "Chiến thắng Điện Biên Phủ năm 1954"),
    ("wiki", "Thành phố Đà Lạt nằm trên cao nguyên"),
    # văn học / sáng tác
    ("truyện", "Ngày xửa ngày xưa, ở một làng nhỏ ven sông, có hai anh em mồ côi"),
    ("truyện", "Mưa rơi lất phất trên phố cổ. Lan đứng nép dưới mái hiên, tay cầm lá thư"),
    ("thơ", "Trăm năm trong cõi người ta,\nChữ tài chữ mệnh khéo là ghét nhau.\n"),
    ("văn", "Mùa thu Hà Nội đẹp nhất là vào những buổi sáng"),
    # đời sống / hướng dẫn
    ("hướng dẫn", "Cách nấu phở bò Hà Nội:\nNguyên liệu:\n"),
    ("hướng dẫn", "5 mẹo giúp bạn ngủ ngon hơn mỗi đêm:\n1."),
    ("pháp luật", "Theo Bộ luật Lao động năm 2019, người lao động có quyền"),
    ("sức khoẻ", "Triệu chứng thường gặp của bệnh sốt xuất huyết gồm"),
    # hỏi đáp dạng văn bản
    ("hỏi đáp", "Câu hỏi: Thủ đô của Nhật Bản là gì?\nTrả lời:"),
    ("hỏi đáp", "Câu hỏi: Tại sao bầu trời có màu xanh?\nTrả lời:"),
    ("hội thoại", "A: Chào bạn, hôm nay bạn có khỏe không?\nB:"),
    # toán vài mẫu
    ("toán", "Bài 1: 12 + 7 = 19\nBài 2: 25 + 18 = 43\nBài 3: 36 + 47 ="),
    ("toán", "Một cửa hàng có 120 kg gạo, đã bán 45 kg. Hỏi cửa hàng còn lại bao nhiêu ki-lô-gam gạo?\nBài giải:\n"),
    ("toán", "Giải phương trình 2x + 6 = 0.\nLời giải:"),
    # dịch vài mẫu
    ("dịch", "Tiếng Anh: Good morning.\nTiếng Việt: Chào buổi sáng.\nTiếng Anh: I love my family.\nTiếng Việt:"),
    # tiếng Anh / code
    ("english", "The Industrial Revolution began in Britain in the late 18th century and"),
    ("english", "Photosynthesis is the process by which plants"),
    ("code", "def fibonacci(n):\n    \"\"\"Trả về số Fibonacci thứ n.\"\"\"\n"),
    ("code", "# Đọc file CSV và tính trung bình cột 'gia'\nimport pandas as pd\n"),
    ("code", "function isPrime(n) {\n"),
    # ngữ cảnh dài-ish: nhớ chi tiết đầu đoạn
    ("nhớ", "Anh Minh có ba con: Hoa 12 tuổi, Nam 9 tuổi và Bình 5 tuổi. Cả nhà sống ở Huế. "
            "Hằng ngày anh đi làm bằng xe máy, chị Thu ở nhà bán hàng tạp hoá. Cuối tuần cả nhà thường "
            "về quê thăm ông bà. Người con lớn nhất của anh Minh tên là"),
]

# (câu dẫn, đáp án đúng, [sai…])
MCQ = [
    ("Thủ đô của Việt Nam là", "Hà Nội", ["Huế", "Đà Nẵng", "Cần Thơ"]),
    ("Thủ đô của Pháp là", "Paris", ["Lyon", "Berlin", "Madrid"]),
    ("Thủ đô của Nhật Bản là", "Tokyo", ["Osaka", "Kyoto", "Seoul"]),
    ("Sông dài nhất Việt Nam chảy qua nhiều nước là sông", "Mê Kông", ["Hồng", "Đà", "Đồng Nai"]),
    ("Đỉnh núi cao nhất Việt Nam là", "Fansipan", ["Bạch Mã", "Langbiang", "Ngọc Linh"]),
    ("Chiến thắng Điện Biên Phủ diễn ra vào năm", "1954", ["1945", "1975", "1968"]),
    ("Ngày Quốc khánh của Việt Nam là ngày", "2 tháng 9", ["30 tháng 4", "1 tháng 5", "19 tháng 8"]),
    ("Tác giả của Truyện Kiều là", "Nguyễn Du", ["Nguyễn Trãi", "Hồ Xuân Hương", "Tố Hữu"]),
    ("Nước sôi ở nhiệt độ", "100 độ C", ["50 độ C", "0 độ C", "200 độ C"]),
    ("Công thức hoá học của nước là", "H2O", ["CO2", "NaCl", "O2"]),
    ("Hành tinh gần Mặt Trời nhất là", "Sao Thủy", ["Sao Hỏa", "Sao Mộc", "Trái Đất"]),
    ("Một năm có", "12 tháng", ["10 tháng", "13 tháng", "15 tháng"]),
    ("7 nhân 8 bằng", "56", ["54", "64", "48"]),
    ("Thành phố lớn nhất Việt Nam theo dân số là", "Thành phố Hồ Chí Minh", ["Hải Phòng", "Đà Nẵng", "Huế"]),
    ("Quốc gia đông dân nhất Đông Nam Á là", "Indonesia", ["Lào", "Campuchia", "Brunei"]),
    ("Con vật được gọi là chúa sơn lâm là", "con hổ", ["con mèo", "con thỏ", "con gà"]),
    ("Đơn vị đo cường độ dòng điện là", "ampe", ["vôn", "oát", "mét"]),
    ("Người đầu tiên đặt chân lên Mặt Trăng là", "Neil Armstrong", ["Yuri Gagarin", "Albert Einstein", "Isaac Newton"]),
    ("The capital of Germany is", "Berlin", ["Munich", "Paris", "Vienna"]),
    ("Water is made of hydrogen and", "oxygen", ["nitrogen", "carbon", "helium"]),
    ("Vịnh Hạ Long thuộc tỉnh", "Quảng Ninh", ["Khánh Hòa", "Kiên Giang", "Bình Định"]),
    ("Phở là món ăn nổi tiếng của", "Việt Nam", ["Thái Lan", "Hàn Quốc", "Ý"]),
    ("Trong Python, để in ra màn hình ta dùng hàm", "print", ["echo", "printf", "say"]),
    ("Tết Nguyên Đán là ngày đầu năm theo", "âm lịch", ["dương lịch", "lịch Hồi giáo", "lịch Do Thái"]),
    # khó hơn (bản 1.3B đã ~20/24 phần trên)
    ("Vua đầu tiên của nhà Nguyễn là", "Gia Long", ["Minh Mạng", "Tự Đức", "Quang Trung"]),
    ("Kinh đô của nhà Lý được dời từ Hoa Lư về Thăng Long vào năm", "1010", ["938", "1428", "1802"]),
    ("Trận Bạch Đằng năm 938 do", "Ngô Quyền", ["Trần Hưng Đạo", "Lê Lợi", "Lý Thường Kiệt"]),
    ("Tác giả của Bình Ngô đại cáo là", "Nguyễn Trãi", ["Nguyễn Du", "Lê Quý Đôn", "Phan Bội Châu"]),
    ("Tác phẩm Tắt đèn là của nhà văn", "Ngô Tất Tố", ["Nam Cao", "Vũ Trọng Phụng", "Tô Hoài"]),
    ("Tỉnh có diện tích lớn nhất Việt Nam là", "Nghệ An", ["Gia Lai", "Sơn La", "Đắk Lắk"]),
    ("Cố đô Huế nằm bên bờ sông", "Hương", ["Hàn", "Lam", "Sài Gòn"]),
    ("Cao nguyên đá Đồng Văn thuộc tỉnh", "Hà Giang", ["Lào Cai", "Cao Bằng", "Lai Châu"]),
    ("Định luật vạn vật hấp dẫn do nhà khoa học", "Newton", ["Darwin", "Pasteur", "Mendel"]),
    ("Thuyết tiến hóa bằng chọn lọc tự nhiên do", "Darwin", ["Newton", "Einstein", "Galileo"]),
    ("Kim loại ở thể lỏng ở nhiệt độ phòng là", "thủy ngân", ["sắt", "đồng", "nhôm"]),
    ("Cơ quan sản xuất insulin trong cơ thể người là", "tuyến tụy", ["gan", "thận", "dạ dày"]),
    ("Căn bậc hai của 144 là", "12", ["14", "16", "11"]),
    ("Tổng các góc trong của một tam giác bằng", "180 độ", ["90 độ", "360 độ", "270 độ"]),
    ("15% của 200 bằng", "30", ["15", "20", "45"]),
    ("Số nguyên tố nhỏ nhất là", "2", ["1", "3", "0"]),
    ("Trong SQL, câu lệnh để lấy dữ liệu từ bảng là", "SELECT", ["INSERT", "DROP", "UPDATE"]),
    ("Đồng tiền chính thức của Nhật Bản là", "yên", ["won", "nhân dân tệ", "baht"]),
    ("The largest ocean on Earth is the", "Pacific Ocean", ["Atlantic Ocean", "Indian Ocean", "Arctic Ocean"]),
    ("The author of Romeo and Juliet is", "William Shakespeare", ["Charles Dickens", "Mark Twain", "Jane Austen"]),
]


@torch.no_grad()
def seq_logprob(m, dev, ctx_ids, ans_ids):
    x = torch.tensor([ctx_ids + ans_ids], device=dev)
    lo = m(x)[0].float()
    lp = F.log_softmax(lo[len(ctx_ids) - 1:-1], -1)
    return lp.gather(1, torch.tensor(ans_ids, device=dev)[:, None]).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--mcq-only", action="store_true", help="chỉ chấm trắc nghiệm (không sinh)")
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = BPETokenizer.load("checkpoints/bpe_v1.json")
    vocab = Vocabulary(tok)
    eos = tok.token_to_id["<EOS>"]
    table = {}
    for ck in a.ckpts.split(","):
        name = os.path.basename(ck).replace("hybrid_", "").replace("_bf16.pt", "").replace(".pt", "")
        t0 = time.time()
        m, cfg, step = load(ck, dev)
        r = Runner(m, dev)
        print(f"[{name}] nạp {time.time()-t0:.0f}s · step {step}", flush=True)
        # trắc nghiệm
        ok, rows = 0, []
        for q, good, bad in MCQ:
            c = tok.encode(q)
            sc = {o: seq_logprob(m, dev, c, tok.encode(" " + o)) for o in [good] + bad}
            pick = max(sc, key=sc.get)
            ok += pick == good
            rows.append((q, good, pick))
        table[name] = (ok, rows)
        print(f"[{name}] trắc nghiệm {ok}/{len(MCQ)} (dễ {sum(r[1]==r[2] for r in rows[:24])}/24 · khó {sum(r[1]==r[2] for r in rows[24:])}/{len(MCQ)-24})", flush=True)
        if a.mcq_only:
            del r, m
            continue
        # viết tiếp
        out = [f"# {name} (step {step}) — viết tiếp, temp {a.temp}, {a.n} token\n",
               f"Trắc nghiệm kiến thức: **{ok}/{len(MCQ)}**\n"]
        ntok, tg = 0, time.time()
        for i, (kind, p) in enumerate(PROMPTS):
            ids = []
            for t in r.generate(tok.encode(p), n=a.n, temp=a.temp, top_p=0.9, rep_pen=1.1, eos=eos, seed=i):
                ids.append(t)
            ntok += len(ids)
            txt = vocab.detokenize(ids)
            out.append(f"\n## {i+1}. [{kind}]\n**Prompt:** {p}\n\n**→** {txt.strip()}\n")
            print(f"  {i+1}/{len(PROMPTS)} [{kind}] {len(ids)} tok", flush=True)
        out.insert(2, f"Tốc độ sinh trên {dev}: {ntok/(time.time()-tg):.1f} tok/s\n")
        os.makedirs("out", exist_ok=True)
        with open(f"out/probe_base_{name}.md", "w") as f:
            f.write("\n".join(out))
        del r, m
        if dev == "mps":
            torch.mps.empty_cache()
    print("\n### Trắc nghiệm (đúng/tổng)")
    names = list(table)
    print("câu | đúng | " + " | ".join(names))
    for j, (q, good, _) in enumerate(MCQ):
        print(f"{q} | {good} | " + " | ".join(("✓" if table[n][1][j][2] == good else "✗ " + table[n][1][j][2]) for n in names))
    print("TỔNG | | " + " | ".join(f"{table[n][0]}/{len(MCQ)}" for n in names))


if __name__ == "__main__":
    main()
