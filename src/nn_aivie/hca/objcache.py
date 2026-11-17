"""Cache theo ĐỐI TƯỢNG, xác thực bằng weakref — TK_045 §12.30.

## Vì sao tồn tại

Khắp `hca/` có mẫu:

    _CACHE: dict[int, X] = {}
    def ensure_x(state):
        sid = id(state)
        if sid not in _CACHE:
            _CACHE[sid] = build(state)
        return _CACHE[sid]

`id()` chỉ duy nhất **trong thời gian đối tượng còn sống**. `GraphState` bị thu hồi rồi một
state MỚI cấp phát trúng cùng địa chỉ ⇒ nhận lại đồ dựng cho state CŨ (vocab/cha khác).

Triệu chứng bắt được (§12.29): suite chạy 3 lần ra **284 / 283+1 / 284** trên CÙNG cây mã,
lỗi `vocab.fold(7) IndexError`. Nhưng crash là trường hợp **MAY**: phần lớn id vẫn hợp lệ nên
bình thường nó **truy hồi sai ÂM THẦM** — đúng lớp lỗi nguy nhất của dự án này.

## Cách vá

Giữ `id()` làm khoá (tra nhanh), nhưng lưu kèm **weakref tới chính đối tượng** và so `is` lúc
đọc. Không heuristic hình dạng (`len(vocab)`, `n_parents`): hai state khác nhau hoàn toàn có thể
cùng kích thước — so `is` thì đúng tuyệt đối. Weakref chết ⇒ chủ cũ đã bị thu hồi ⇒ dựng lại.

    val = cache_get(_CACHE, state)
    if val is None:
        val = build(state)
        cache_put(_CACHE, state, val)
"""

from __future__ import annotations

import weakref
from typing import Any


def cache_get(cache: dict, owner: Any, sub: Any = None):
    """Giá trị đã cache CHO ĐÚNG `owner` này; None nếu chưa có / chủ cũ đã chết / id bị tái dùng."""
    key = (id(owner), sub) if sub is not None else id(owner)
    hit = cache.get(key)
    if hit is None:
        return None
    ref, val = hit
    if ref() is not owner:  # id tái dùng, hoặc chủ cũ đã bị thu hồi
        cache.pop(key, None)
        return None
    return val


def cache_put(cache: dict, owner: Any, val, sub: Any = None):
    """Lưu `val` kèm weakref tới `owner`. Trả lại `val` cho tiện `return cache_put(...)`."""
    key = (id(owner), sub) if sub is not None else id(owner)
    try:
        cache[key] = (weakref.ref(owner), val)
    except TypeError:  # đối tượng không weakref được → không cache còn hơn cache sai
        cache.pop(key, None)
    return val
