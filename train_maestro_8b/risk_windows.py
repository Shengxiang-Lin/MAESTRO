"""Pure helpers for fixed token windows inside one returned Sub-Agent report."""
from __future__ import annotations

def windows(start: int, end: int, size: int = 128, stride: int = 96):
    if not (0 <= start < end and size >= 1 and 1 <= stride <= size):
        raise ValueError("invalid token window")
    length = end - start
    if length <= size:
        return [(start, end)]
    offsets = list(range(0, length - size + 1, stride))
    if offsets[-1] != length - size:
        offsets.append(length - size)
    return [(start + o, start + o + size) for o in offsets]

def span_to_tokens(offsets, start: int, end: int):
    ids = [i for i, (left, right) in enumerate(offsets)
           if right > left and right > start and left < end]
    if not ids:
        raise ValueError("no token overlaps report span")
    return ids[0], ids[-1] + 1
