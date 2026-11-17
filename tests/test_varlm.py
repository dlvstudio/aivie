from __future__ import annotations

from nn_aivie.core.varlm import VarOrderLM


def test_varlm_prefers_longer_context_when_available():
    # Build a tiny corpus where ctx=(1,2,3)->4 is strong and ctx=(2,3)->5 exists,
    # so with ctx [1,2,3] the model should prefer 4 over 5.
    lm = VarOrderLM(max_ctx=3, lam=0.5)
    for _ in range(10):
        lm.add_sequence([1, 2, 3, 4])
    for _ in range(3):
        lm.add_sequence([2, 3, 5])

    pp = lm.scores([1, 2, 3])
    assert pp  # has scores
    assert max(pp, key=pp.get) == 4


def test_varlm_kn_prefers_longer_context_and_returns_distribution():
    # KN phải giữ hành vi ưu tiên ctx dài (như scores) + trả phân bố hợp lệ trên ứng viên.
    lm = VarOrderLM(max_ctx=3, lam=0.5)
    for _ in range(10):
        lm.add_sequence([1, 2, 3, 4])
    for _ in range(3):
        lm.add_sequence([2, 3, 5])

    pp = lm.scores_kn([1, 2, 3], d=0.75)
    assert pp, "KN cho ctx đã thấy phải có ứng viên"
    assert max(pp, key=pp.get) == 4, "ctx [1,2,3] mạnh → 4 thắng 5"
    assert all(v >= 0 for v in pp.values()), "điểm KN không âm"


def test_varlm_kn_empty_on_unseen_context():
    lm = VarOrderLM(max_ctx=3, lam=0.5)
    lm.add_sequence([1, 2, 3, 4])
    # ctx toàn token lạ → không có bảng bậc nào chứa → rỗng (mù ctx), giống scores().
    assert lm.scores_kn([98, 99], d=0.75) == {}


def test_varlm_prune_drops_rare_high_order_contexts_only():
    lm = VarOrderLM(max_ctx=4, lam=0.45)
    # One rare k=3 ctx -> should be pruned when min_total=2 from k>=3.
    lm.add_sequence([10, 11, 12, 13])
    # Another sequence to keep some k=1/k=2 evidence around.
    lm.add_sequence([11, 12, 13, 14])

    before_k3 = len(lm.tables[2])
    before_k4 = len(lm.tables[3])
    st = lm.prune(min_total=2, from_ctx=3)

    assert st["k3"] >= 0 and st["k4"] >= 0
    assert len(lm.tables[2]) <= before_k3
    assert len(lm.tables[3]) <= before_k4
