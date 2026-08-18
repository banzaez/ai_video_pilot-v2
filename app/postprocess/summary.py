"""Сводка по трекам: когда и через какой край кадра трек пришёл и ушёл.

Нужна для сшивки треков между камерами: по краю выхода и времени видно,
на какой соседней камере искать продолжение.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from app.info import track_global_id

EDGE_MARGIN = 12


def _bottom_center(bbox: list[Any]) -> list[int]:
    x1, _y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return [int(round((x1 + x2) / 2)), int(round(y2))]


def frame_edges(
    bbox: list[Any],
    frame_w: int,
    frame_h: int,
    margin: int = EDGE_MARGIN,
) -> list[str]:
    """Какие края кадра задевает bbox. Пусто → трек целиком внутри кадра."""
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    edges: list[str] = []
    if x1 <= margin:
        edges.append("left")
    if frame_w > 0 and x2 >= frame_w - margin:
        edges.append("right")
    if y1 <= margin:
        edges.append("top")
    if frame_h > 0 and y2 >= frame_h - margin:
        edges.append("bottom")
    return edges


def build_track_summaries(
    tracked: dict[int, list[dict[str, Any]]],
    *,
    fps: float,
    frame_w: int,
    frame_h: int,
    margin: int = EDGE_MARGIN,
    camera_index: int | None = None,
) -> list[dict[str, Any]]:
    """tracked: 0-based кадр → детекции. Кадры в сводке 1-based, как в tracking.json."""
    first: dict[int, tuple[int, list[Any]]] = {}
    last: dict[int, tuple[int, list[Any]]] = {}
    widths: dict[int, list[float]] = defaultdict(list)
    heights: dict[int, list[float]] = defaultdict(list)
    confs: dict[int, list[float]] = defaultdict(list)
    counts: dict[int, int] = defaultdict(int)

    for frame_idx in sorted(tracked):
        one_based = int(frame_idx) + 1
        for det in tracked[frame_idx]:
            tid = int(det["track_id"])
            bbox = list(det["bbox"])
            if tid not in first:
                first[tid] = (one_based, bbox)
            last[tid] = (one_based, bbox)
            counts[tid] += 1
            widths[tid].append(max(0.0, float(bbox[2]) - float(bbox[0])))
            heights[tid].append(max(0.0, float(bbox[3]) - float(bbox[1])))
            confs[tid].append(float(det.get("confidence") or 0.0))

    def sec(frame_1b: int) -> float:
        return round(frame_1b / fps, 3) if fps > 0 else 0.0

    out: list[dict[str, Any]] = []
    for tid in sorted(first):
        f0, b0 = first[tid]
        f1, b1 = last[tid]
        rec: dict[str, Any] = {
            "id": track_global_id(camera_index, tid),
            "track_id": tid,
            "n_obs": counts[tid],
            "f0": f0,
            "f1": f1,
            "t0": sec(f0),
            "t1": sec(f1),
            "p0": _bottom_center(b0),
            "p1": _bottom_center(b1),
            "bbox0": [int(round(v)) for v in b0[:4]],
            "bbox1": [int(round(v)) for v in b1[:4]],
            "enter": frame_edges(b0, frame_w, frame_h, margin),
            "exit": frame_edges(b1, frame_w, frame_h, margin),
            "w": int(round(float(np.median(widths[tid])))),
            "h": int(round(float(np.median(heights[tid])))),
            "conf": round(float(np.median(confs[tid])), 3),
        }
        out.append(rec)
    return out
