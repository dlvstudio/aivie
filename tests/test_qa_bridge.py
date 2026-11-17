"""§183 — cầu nối QA→statement cho endpoint /api/gen."""
import re

# tái lập regex từ viz/server/main.py (thuần, không import server để khỏi kéo fastapi)
_QA = re.compile(
    r"^\s*(?:hỏi\s*:\s*)?(.+?)\s+là\s+(?:cái\s+)?gì\s*\?*\s*(?:đáp\s*:?.*)?$",
    re.IGNORECASE | re.DOTALL,
)


def bridge(p):
    m = _QA.match(p)
    if not m:
        return None
    ent = m.group(1).strip().strip("\"'“”")
    return f"{ent} là" if ent and len(ent) <= 40 else None


def test_qa_dinh_nghia_bridged():
    assert bridge("Hỏi: Sắt là gì? Đáp:") == "Sắt là"
    assert bridge("Chó là gì?") == "Chó là"
    assert bridge("Nước là gì") == "Nước là"
    assert bridge("hỏi: mặt trời là cái gì ? đáp:") == "mặt trời là"


def test_khong_phai_qa_thi_khong_bac_cau():
    assert bridge("sắt và") is None
    assert bridge("Sắt là kim loại") is None   # không phải "là gì"
    assert bridge("mùa xuân") is None
