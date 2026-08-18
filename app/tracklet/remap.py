"""Remap tracklet_id → global track_id для финального tracking.json."""

from __future__ import annotations

from typing import Any


def remap_tracklet_frames(
    frames_data: dict,
    tracklet_to_global: dict[str, int],
) -> dict[int, list[dict[str, Any]]]:
    """tracklet_frames + mapping → tracked dict (0-based frames, track_id).

    Если несколько треклетов одной группы попали в один кадр, оставляем один bbox:
    выше confidence, при равенстве — треклет с более поздним t0 (handover).
    """
    t0_by_tl: dict[int, float] = {}
    for frame in frames_data.get("frames") or []:
        t = float(frame.get("timestamp_sec") if frame.get("timestamp_sec") is not None else frame.get("frame_index") or 0)
        for det in frame.get("detections") or []:
            try:
                tl_id = int(det["tracklet_id"])
            except (KeyError, TypeError, ValueError):
                continue
            prev = t0_by_tl.get(tl_id)
            if prev is None or t < prev:
                t0_by_tl[tl_id] = t

    out: dict[int, list[dict[str, Any]]] = {}
    for frame in frames_data.get("frames") or []:
        fi = int(frame["frame_index"]) - 1
        best: dict[int, tuple[tuple[float, float], dict[str, Any]]] = {}
        for det in frame.get("detections") or []:
            try:
                tl_id = int(det["tracklet_id"])
            except (KeyError, TypeError, ValueError):
                continue
            global_id = int(tracklet_to_global.get(str(tl_id), tl_id))
            conf = det.get("confidence")
            conf_f = float(conf) if isinstance(conf, (int, float)) else 0.0
            t0 = float(t0_by_tl.get(tl_id, 0.0))
            rank = (conf_f, t0)
            rec = {
                "track_id": global_id,
                "confidence": det.get("confidence"),
                "bbox": det["bbox"],
            }
            prev = best.get(global_id)
            if prev is None or rank > prev[0]:
                best[global_id] = (rank, rec)
        dets = [item[1] for item in best.values()]
        dets.sort(key=lambda d: int(d["track_id"]))
        if dets:
            out[fi] = dets
    return out
