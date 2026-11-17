from nn_aivie.core.paths import checkpoints_dir, default_checkpoint, project_root, viz_default_checkpoint


def test_checkpoints_dir_in_project():
    root = project_root()
    ckpt = checkpoints_dir()
    assert ckpt.parent == root
    assert ckpt.name == "checkpoints"


def test_default_checkpoint_path():
    p = default_checkpoint()
    assert p.parent.name == "checkpoints"
    assert p.name in {
        "network_v10.pkl",
        "network_v10_histwiki.pkl",
        "network_v9_histwiki.pkl",
        "network_v9_histwiki_paragraph.pkl",
        "network_v8_histwiki.pkl",
        "network_v7_history_segseed.pkl",
        "network_v4.pkl",
        "network_v4_full.pkl", "network_v4_convpool.pkl",
        "grade_v3_big_a2_t3_isg.pkl", "grade_v3.pkl",
    }


def test_viz_default_checkpoint_picks_first_existing():
    """Trả về checkpoint ĐẦU TIÊN tồn tại theo thứ tự ưu tiên — kiểm VÒNG LẶP.

    Bản cũ CHÉP LẠI danh sách ưu tiên thành một chuỗi `elif` song song, rồi trôi: thiếu
    `network_v6_knowledge` / `v5_entity` / `v4_full` / `grade_v3_big_a2_t3_isg` ⇒ đỏ trên
    máy có checkpoint mới, mà đỏ đó KHÔNG nói lên lỗi gì của mã. Nay đọc chung một nguồn.
    """
    from nn_aivie.core.paths import (
        VIZ_CHECKPOINT_PREFERENCE,
        checkpoints_dir,
        latest_analogy_checkpoint,
    )

    p = viz_default_checkpoint()
    assert p.parent.name == "checkpoints"
    ckpt_dir = checkpoints_dir()
    # §180 — checkpoint loại suy MỚI NHẤT thắng trước list ưu tiên tĩnh.
    latest_ls = latest_analogy_checkpoint()
    if latest_ls is not None:
        assert p == latest_ls, (
            f"phải lấy checkpoint loại suy mới nhất ({latest_ls.name}), nhận {p.name}"
        )
        return
    existing = [n for n in VIZ_CHECKPOINT_PREFERENCE if (ckpt_dir / n).exists()]
    if existing:
        assert p.name == existing[0], (
            f"phải lấy mục ĐẦU TIÊN tồn tại ({existing[0]}), nhận {p.name}"
        )
    else:
        from nn_aivie.core.paths import default_checkpoint
        assert p == default_checkpoint(), "không mục nào tồn tại ⇒ phải rơi về mặc định"
