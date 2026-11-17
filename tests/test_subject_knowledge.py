"""Subject knowledge corpus smoke tests."""

from __future__ import annotations

from nn_aivie.data.subject_knowledge import SUBJECT_META, all_grades, grade_subjects


def test_all_grades_have_five_subjects():
    for g in all_grades():
        subs = grade_subjects(g)
        assert set(subs) == set(SUBJECT_META)
        for bundle in subs.values():
            assert len(bundle.sentences) >= 20
            assert len(bundle.qa_pairs) >= 5


def test_grade3_geography_includes_hanoi():
    geo = grade_subjects(3)["dia_ly"]
    joined = " ".join(geo.sentences)
    assert "Hà Nội" in joined
    assert any("Thủ đô" in q or "Hà Nội" in q for q, _ in geo.qa_pairs)
