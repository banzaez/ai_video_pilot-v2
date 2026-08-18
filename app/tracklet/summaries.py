"""Сводки по треклетам (аналог build_track_summaries)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from app.postprocess.summary import _bottom_center, frame_edges


def build_tracklet_summaries(
    tracked: dict[int, list[dict[str, Any]]],
    *,
    fps: float,
    frame_w: int,
    frame_h: int,
    margin: int = 12,
) -> list[dict[str, Any]]:
    """tracked: 0-based кадр → детекции с tracklet_id."""
    first: dict[int, tuple[int, list[Any]]] = {}
    last: dict[int, tuple[int, list[Any]]] = {}
    widths: dict[int, list[float]] = defaultdict(list)
    heights: dict[int, list[float]] = defaultdict(list)
    confs: dict[int, list[float]] = defaultdict(list)
    counts: dict[int, int] = defaultdict(int)

    for frame_idx in sorted(tracked):
        one_based = int(frame_idx) + 1
        for det in tracked[frame_idx]:
            tid = int(det.get("tracklet_id") or det.get("track_id"))
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
        out.append(
            {
                "tracklet_id": tid,
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
        )
    return out
