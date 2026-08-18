"""Утилиты работы с временными интервалами и скорингом пар."""

from __future__ import annotations

from typing import Any
import numpy as np


def intervals_overlap(a0: float, a1: float, b0: float, b1: float, eps: float = 1e-6) -> bool:
    """True, если интервалы пересекаются во времени.

    Соприкосновение границ [0,5] и [5,10] — не overlap.
    Точечный интервал t0==t1 в момент внутри/на границе другого — overlap.
    """
    lo = max(float(a0), float(b0))
    hi = min(float(a1), float(b1))
    if lo < hi - eps:
        return True
    if abs(lo - hi) > eps:
        return False
    a_point = abs(float(a0) - float(a1)) <= eps
    b_point = abs(float(b0) - float(b1)) <= eps
    return a_point or b_point


def pair_embed_score(emb_a: np.ndarray | None, emb_b: np.ndarray | None) -> float | None:
    """Лучшее косинусное сходство среди пар эмбеддингов."""
    if emb_a is None or emb_b is None or len(emb_a) == 0 or len(emb_b) == 0:
        return None
    return float(np.max(emb_a @ emb_b.T))


def time_spans(
    tracks_doc: dict[str, Any],
    tracking: dict[str, Any] | None = None,
) -> dict[int, tuple[float, float]]:
    """t0/t1 из tracks; иначе из tracking.frames."""
    tracks = list(tracks_doc.get("tracks") or [])
    spans: dict[int, tuple[float, float]] = {}
    for tr in tracks:
        try:
            tid = int(tr["track_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if tr.get("t0") is not None and tr.get("t1") is not None:
            spans[tid] = (float(tr["t0"]), float(tr["t1"]))
    if tracks and len(spans) >= len(tracks):
        return spans
    if tracking:
        for frame in tracking.get("frames") or []:
            t = float(frame.get("timestamp_sec") or 0.0)
            for det in frame.get("detections") or []:
                try:
                    tid = int(det["track_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                prev = spans.get(tid)
                spans[tid] = (t, t) if prev is None else (min(prev[0], t), max(prev[1], t))
    return spans
