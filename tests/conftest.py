from __future__ import annotations

import sys
from pathlib import Path

import pytest


# Ensure repository root is importable during pytest runs.
_ROOT = Path(__file__).resolve().parents[1]
root_str = str(_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


@pytest.fixture(autouse=True)
def _unit_tests_keep_matrix_generate():
    """Production default is W_skip primary (§12.131); unit tests assert matrix/parent routes."""
    import nn_aivie.hca.generate_hebbian as gh

    prev = getattr(gh, "_FORCE_MATRIX_FOR_TESTS", False)
    gh._FORCE_MATRIX_FOR_TESTS = True
    try:
        yield
    finally:
        gh._FORCE_MATRIX_FOR_TESTS = prev
