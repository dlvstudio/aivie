"""Project paths — data/, checkpoints/, lessons/."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Repo root (nn-aivie/)."""
    return Path(__file__).resolve().parents[3]


def checkpoints_dir() -> Path:
    """checkpoints/ — GraphState sau teach/train."""
    path = project_root() / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_checkpoint(name: str = "network_v10.pkl") -> Path:
    p = checkpoints_dir() / name
    if p.exists():
        return p
    for fallback in (
        "network_v10.pkl",
        "network_v10_histwiki.pkl",
        "network_v9_histwiki.pkl",
        "network_v8_histwiki.pkl",
        "network_v7_history_segseed.pkl",
        "network_v4_full.pkl",
        "network_v4_convpool.pkl",
        "network_v4.pkl",
        "grade_v3_big_a2_t3_isg.pkl",
        "grade_v3.pkl",
    ):
        fb = checkpoints_dir() / fallback
        if fb.exists():
            return fb
    return p


# Thứ tự ưu tiên checkpoint cho Graph UI. Là HẰNG SỐ module (không phải tuple ẩn trong
# hàm) vì `tests/test_paths.py` từng CHÉP LẠI danh sách này thành một chuỗi `elif` song
# song — rồi trôi: thiếu 4 mục, đỏ trên máy có checkpoint mới. Một nguồn duy nhất thì
# test kiểm được VÒNG LẶP thay vì chép lại nội dung.
VIZ_CHECKPOINT_PREFERENCE: tuple[str, ...] = (
    "network_v10.pkl",
    "network_v10_histwiki_paragraph.pkl",
    "network_v10_histwiki.pkl",
    "network_v9_histwiki_paragraph.pkl",
    "network_v9_histwiki.pkl",
    "network_v8_histwiki.pkl",
    "network_v7_history_segseed.pkl",
    "network_v7_history.pkl",
    "network_v6_knowledge.pkl",
    "network_v5_entity.pkl",
    "network_v4_full.pkl",
    "network_v4_convpool.pkl",
    "network_v4.pkl",
    "grade_v3_big_a2_t3_isg.pkl",
    "grade_v3_big_a2.pkl",
    "grade_v3_big.pkl",
    "grade_v3.pkl",
)


# §180 — checkpoint MANG CẠNH LOẠI SUY (TK_054 §173–§179). Dò theo QUY ƯỚC TÊN:
# pipeline khái niệm lưu `net_ls_concept*` / `*_analogy` / `*_reinforced` / `loai_suy*`.
# Không nạp file để dò (mỗi checkpoint ~1GB) — tên là proxy đủ rẻ và ổn định.
_ANALOGY_CHECKPOINT_MARKERS: tuple[str, ...] = (
    "concept", "analogy", "reinforced", "loai_suy",
)


def analogy_checkpoints() -> list[Path]:
    """Mọi checkpoint mang cạnh loại suy (dò tên), MỚI NHẤT trước (theo mtime)."""
    out = [
        p for p in checkpoints_dir().glob("*.pkl")
        if any(m in p.name.lower() for m in _ANALOGY_CHECKPOINT_MARKERS)
    ]
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)


def latest_analogy_checkpoint() -> Path | None:
    """Checkpoint loại suy MỚI NHẤT, hoặc None nếu chưa có cái nào."""
    cks = analogy_checkpoints()
    return cks[0] if cks else None


def viz_default_checkpoint() -> Path:
    """Graph UI — ưu tiên checkpoint LOẠI SUY mới nhất (§180), rồi mới tới
    `VIZ_CHECKPOINT_PREFERENCE`, rồi mặc định. Để UI graph tự nhận diện đồ thị
    có cạnh loại suy khái niệm mà không cần đặt `AIVIE_CHECKPOINT` tay."""
    latest = latest_analogy_checkpoint()
    if latest is not None:
        return latest
    for name in VIZ_CHECKPOINT_PREFERENCE:
        p = checkpoints_dir() / name
        if p.exists():
            return p
    return default_checkpoint()


def grade_checkpoint(grade: int, suffix: str = "") -> Path:
    base = f"grade_{grade:02d}"
    if suffix:
        base = f"{base}_{suffix}"
    return checkpoints_dir() / f"{base}.pkl"
