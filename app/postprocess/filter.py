"""Фильтры шума: крошечные bbox и короткие треки."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from app.util.bbox import bbox_wh

logger = logging.getLogger(__name__)


def _keep_ids(
    tracked: dict[int, list[dict[str, Any]]],
    keep: set[int],
) -> dict[int, list[dict[str, Any]]]:
    return {
        frame_idx: [det for det in dets if int(det["track_id"]) in keep]
        for frame_idx, dets in tracked.items()
    }


def surviving_track_ids(tracked: dict[int, list[dict[str, Any]]]) -> set[int]:
    ids: set[int] = set()
    for dets in tracked.values():
        for det in dets:
            ids.add(int(det["track_id"]))
    return ids


def apply_track_filters(
    tracked: dict[int, list[dict[str, Any]]],
    *,
    min_bbox_area: float = 0.0,
    min_bbox_side: float = 0.0,
    min_track_sec: float = 0.0,
    fps: float = 0.0,
    stage: str = "2.4",
) -> dict[int, list[dict[str, Any]]]:
    """Убирает track_id с крошечным медианным bbox или короткой длительностью."""
    if min_bbox_area <= 0 and min_bbox_side <= 0 and min_track_sec <= 0:
        return tracked

    areas: dict[int, list[float]] = defaultdict(list)
    widths: dict[int, list[float]] = defaultdict(list)
    heights: dict[int, list[float]] = defaultdict(list)
    first_f: dict[int, int] = {}
    last_f: dict[int, int] = {}
    for frame_idx, detections in tracked.items():
        fi = int(frame_idx)
        for det in detections:
            tid = int(det["track_id"])
            w, h = bbox_wh(det["bbox"])
            areas[tid].append(w * h)
            widths[tid].append(w)
            heights[tid].append(h)
            if tid not in first_f or fi < first_f[tid]:
                first_f[tid] = fi
            if tid not in last_f or fi > last_f[tid]:
                last_f[tid] = fi

    keep: set[int] = set()
    dropped = 0
    for tid, ar in areas.items():
        med_a = float(np.median(ar))
        med_w = float(np.median(widths[tid]))
        med_h = float(np.median(heights[tid]))
        if min_bbox_area > 0 and med_a < min_bbox_area:
            dropped += 1
            continue
        if min_bbox_side > 0 and med_w < min_bbox_side and med_h < min_bbox_side:
            dropped += 1
            continue
        if min_track_sec > 0 and fps > 0:
            dur = (last_f[tid] - first_f[tid]) / float(fps)
            if dur < min_track_sec:
                dropped += 1
                continue
        keep.add(tid)

    logger.info(
        "Фильтр (%s): min_area=%s min_side=%s min_sec=%s fps=%s, оставлено %s / %s (отброшено %s)",
        stage,
        min_bbox_area,
        min_bbox_side,
        min_track_sec,
        fps,
        len(keep),
        len(areas),
        dropped,
    )
    if not dropped:
        return tracked
    return _keep_ids(tracked, keep)
