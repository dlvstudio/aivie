#!/usr/bin/env python3
"""
TK_024 nhánh C (C-targeted) — fetch bài Wikipedia tiếng Việt LỊCH SỬ theo TÊN, nguyên đoạn.

Densify đúng miền (triều đại / nhân vật / địa danh / sự kiện) để lấp data-gap entity-association
(vd "Văn Miếu" đồng-đoạn với "Lý") mà corpus câu-lẻ hiện tại thiếu. Giữ schema statements_doc:
{text, doc_id, para, title} → gộp (doc_id,para) = đoạn → mine ENTITY_LINK (B1).

Luật 0: chọn chủ đề = curriculum/data (được phép, README Mục 0). Không hardcode runtime.

Chạy:
  python scripts/fetch_history_wiki.py                 # full seed list
  python scripts/fetch_history_wiki.py --limit 5       # test nhanh
  python scripts/fetch_history_wiki.py --titles-file data/history_titles.txt
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from nn_aivie.data.quality import is_quality_vi_sentence, split_sentences
from nn_aivie.util.progress import FileProgressLog

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/real_vi/history_wiki_para.jsonl"
API = "https://vi.wikipedia.org/w/api.php"
UA = "nn-aivie-research/0.1 (educational; contact vietdung.luong@gmail.com)"

# Seed lịch sử VN — triều đại, nhân vật, địa danh/công trình, sự kiện.
SEED = [
    # Triều đại
    "Nhà Ngô", "Nhà Đinh", "Nhà Tiền Lê", "Nhà Lý", "Nhà Trần", "Nhà Hồ",
    "Nhà Hậu Lê", "Nhà Mạc", "Nhà Tây Sơn", "Nhà Nguyễn", "Nhà Lê sơ", "Chúa Trịnh", "Chúa Nguyễn",
    # Nhân vật
    "Ngô Quyền", "Đinh Bộ Lĩnh", "Lê Hoàn", "Lý Công Uẩn", "Lý Thường Kiệt", "Lý Chiêu Hoàng",
    "Trần Thái Tông", "Trần Nhân Tông", "Trần Hưng Đạo", "Trần Thủ Độ",
    "Hồ Quý Ly", "Lê Lợi", "Nguyễn Trãi", "Lê Thánh Tông", "Mạc Đăng Dung",
    "Nguyễn Huệ", "Nguyễn Nhạc", "Gia Long", "Minh Mạng", "Hai Bà Trưng", "Bà Triệu",
    "Lý Nam Đế", "Mai Hắc Đế", "Phùng Hưng", "Khúc Thừa Dụ",
    # Địa danh / kinh đô / công trình
    "Thăng Long", "Hoa Lư", "Cổ Loa", "Đại La", "Văn Miếu – Quốc Tử Giám", "Thành nhà Hồ",
    "Hoàng thành Thăng Long", "Phú Xuân", "Đại Việt", "Đại Cồ Việt", "Vạn Xuân", "Champa",
    # Sự kiện / chiến tranh
    "Chiến tranh Nguyên Mông–Đại Việt", "Khởi nghĩa Lam Sơn", "Trận Bạch Đằng (938)",
    "Trận Bạch Đằng (1288)", "Trận Như Nguyệt", "Khởi nghĩa Hai Bà Trưng", "Loạn 12 sứ quân",
    "Chiến tranh Đại Việt – Chiêm Thành", "Nhà Tống", "Nhà Nguyên", "Nhà Minh", "Nhà Thanh",
    # --- MỞ RỘNG (cụm dày) ---
    # Sử cận-hiện đại
    "Hồ Chí Minh", "Võ Nguyên Giáp", "Cách mạng Tháng Tám", "Chiến dịch Điện Biên Phủ",
    "Chiến tranh Đông Dương", "Chiến tranh Việt Nam", "Phan Bội Châu", "Phan Châu Trinh",
    "Trương Định", "Hoàng Hoa Thám", "Nguyễn Trung Trực", "Tôn Đức Thắng", "Trần Phú",
    # Văn hóa / giáo dục / danh nhân
    "Chu Văn An", "Lê Quý Đôn", "Nguyễn Du", "Truyện Kiều", "Nguyễn Bỉnh Khiêm",
    "Cao Bá Quát", "Nho giáo", "Phật giáo Việt Nam", "Khoa cử Nho học", "Hồ Xuân Hương",
    # Địa lý
    "Hà Nội", "Huế", "Thành phố Hồ Chí Minh", "Sông Hồng", "Sông Cửu Long", "Sông Bạch Đằng",
    "Dãy Trường Sơn", "Vịnh Hạ Long", "Đồng bằng sông Cửu Long", "Biển Đông", "Cố đô Huế",
    # Khoa học (cụm thiên văn — liên quan chặt)
    "Hệ Mặt Trời", "Trái Đất", "Mặt Trăng", "Mặt Trời", "Sao Hỏa", "Sao Mộc", "Sao Kim",
    "Thái Bình Dương", "Đại Tây Dương", "Dãy núi Himalaya",
]


def _fetch_extract(title: str, timeout: int = 25) -> tuple[str, str]:
    """Trả (resolved_title, plaintext). Rỗng nếu thiếu/lỗi."""
    q = urllib.parse.urlencode(
        {"action": "query", "prop": "extracts", "explaintext": 1,
         "redirects": 1, "titles": title, "format": "json"}
    )
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    pages = d.get("query", {}).get("pages", {})
    if not pages:
        return "", ""
    p = next(iter(pages.values()))
    if "missing" in p:
        return "", ""
    return p.get("title", title), (p.get("extract", "") or "")


def _slug(title: str) -> str:
    return title.replace(" ", "_").replace("/", "_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--titles-file", default=None, help="1 title/dòng; mặc định dùng SEED")
    ap.add_argument("--limit", type=int, default=0, help="chỉ lấy N bài đầu (test)")
    ap.add_argument("--delay", type=float, default=0.3, help="giây nghỉ giữa request (lịch sự)")
    ap.add_argument("--append", action="store_true", help="ghi tiếp, bỏ qua title đã có")
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()

    titles = SEED
    if args.titles_file:
        titles = [
            l.strip() for l in Path(args.titles_file).read_text("utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
    if args.limit:
        titles = titles[: args.limit]

    flog = FileProgressLog(Path(args.log_file) if args.log_file else (ROOT / "out" / "tk024_fetch_history.log"))
    flog.log(f"START titles={len(titles)} out={OUT.name} append={args.append}")

    done_titles: set[str] = set()
    mode = "a"
    if OUT.exists() and args.append:
        for line in OUT.read_text("utf-8").splitlines():
            if line.strip():
                try:
                    done_titles.add(json.loads(line).get("title", ""))
                except Exception:
                    pass
        flog.log(f"resume: {len(done_titles)} titles đã có")
    else:
        mode = "w"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_rec = n_doc = n_miss = 0
    with OUT.open(mode, encoding="utf-8") as out:
        for ti, title in enumerate(titles, 1):
            if title in done_titles:
                continue
            try:
                rtitle, text = _fetch_extract(title)
            except Exception as e:
                flog.log(f"  [{ti}/{len(titles)}] {title!r} FETCH-ERR {type(e).__name__}: {e}")
                continue
            if not text:
                n_miss += 1
                flog.log(f"  [{ti}/{len(titles)}] {title!r} MISSING")
                continue
            slug = _slug(rtitle)
            para_idx = 0
            rec_this = 0
            for line in text.split("\n"):
                line = line.strip()
                if not line or line.startswith("=="):  # bỏ header section
                    continue
                sents = [s for s in split_sentences(line) if is_quality_vi_sentence(s)]
                if not sents:
                    continue
                for s in sents:
                    out.write(json.dumps(
                        {"text": s, "doc_id": f"hist:{slug}", "para": para_idx, "title": rtitle},
                        ensure_ascii=False) + "\n")
                    n_rec += 1
                    rec_this += 1
                para_idx += 1
            n_doc += 1
            flog.log(f"  [{ti}/{len(titles)}] {rtitle!r} paras={para_idx} sents={rec_this}")
            time.sleep(max(0.0, args.delay))

    flog.log(f"DONE docs={n_doc} sentences={n_rec} missing={n_miss} → {OUT}")
    print(f"✓ {n_doc} docs, {n_rec} sentences → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
