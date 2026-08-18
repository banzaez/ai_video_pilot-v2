"""Stage 1: сборка detections.json и пути session-частей."""

from __future__ import annotations

import os
from typing import Any


def resolve_part_path(part: dict[str, Any]) -> str:
    path = str(part.get("path") or "")
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    alt = os.path.join(os.getcwd(), path)
    if os.path.isfile(alt):
        return alt
    raise ValueError(f"Часть не найдена: {path}")


def build_detections_payload(
    merged: dict[int, list[dict[str, Any]]],
    *,
    fps: float,
    frame_count: int,
    width: int,
    height: int,
    conf: float,
    detect_every_n: int,
    nms_iou: float,
    video_source: str,
    session_key: str | None = None,
    sources_meta: list[dict[str, Any]] | None = None,
    detector_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Собрать JSON в формате Stage 1 (detect)."""
    frames: list[dict[str, Any]] = []
    for idx in sorted(i for i, dets in merged.items() if dets):
        part_time_off = 0.0
        local = idx
        if sources_meta:
            for sm in sources_meta:
                off = int(sm["frame_offset"])
                cnt = int(sm["frame_count"])
                if off <= idx < off + cnt:
                    part_time_off = float(sm["time_offset_sec"])
                    local = idx - off
                    break

        frames.append(
            {
                "frame_index": idx + 1,
                "timestamp_sec": round(part_time_off + (local + 1) / fps, 3) if fps else 0.0,
                "detections": [
                    {"bbox": det["bbox"], "confidence": det["confidence"]}
                    for det in merged[idx]
                ],
            }
        )

    payload: dict[str, Any] = {
        "stage": "detect",
        "video_source": video_source,
        "fps": round(float(fps), 3),
        "frame_count": int(frame_count),
        "width": int(width),
        "height": int(height),
        "conf_threshold": float(conf),
        "detect_every_n": int(detect_every_n),
        "nms_iou": float(nms_iou),
        "n_frames": len(frames),
        "frames": frames,
    }
    if session_key:
        payload["kind"] = "camera_day"
        payload["session_key"] = session_key
    if sources_meta:
        payload["sources"] = sources_meta
    if detector_meta:
        payload["detector"] = detector_meta
    return payload
