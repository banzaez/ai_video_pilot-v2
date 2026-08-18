"""Экспорт результатов трекинга в JSON."""

from __future__ import annotations

import logging
from typing import Any

from app.artifact_meta import attach_artifact_meta
from app.io.json_util import save_json

logger = logging.getLogger(__name__)


def _compact_detection(det: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "track_id": int(det["track_id"]),
        "confidence": round(float(det.get("confidence") or 0.0), 4),
        "bbox": [int(v) for v in det["bbox"][:4]],
    }
    return out


def frame_record(
    frame_index: int,
    fps: float,
    detections: list[dict[str, Any]],
) -> dict[str, Any]:
    compact = [_compact_detection(d) for d in detections]
    return {
        "frame_index": frame_index,
        "timestamp_sec": round(frame_index / fps, 3) if fps else 0.0,
        "detections": compact,
    }


class TrackingExporter:
    def __init__(
        self,
        *,
        fps: float,
        width: int,
        height: int,
        json_path: str | None,
        indent: int | None = None,
        detect_every_n: int = 1,
        video_source: str | None = None,
    ):
        self.json_path = json_path
        self.indent = indent
        self.data: dict[str, Any] = {
            "fps": round(fps, 2),
            "frame_count": 0,
            "width": width,
            "height": height,
            "detect_every_n": max(1, int(detect_every_n)),
            "frames": [],
        }
        if video_source:
            self.data["video_source"] = str(video_source)

    @property
    def enabled(self) -> bool:
        return bool(self.json_path)

    def add_frame(self, frame_index: int, fps: float, detections: list[dict[str, Any]]) -> None:
        if not self.enabled or not detections:
            return
        self.data["frames"].append(frame_record(frame_index, fps, detections))

    def save(self, frame_count: int) -> None:
        if not self.enabled or not self.json_path:
            return
        self.data["frame_count"] = frame_count
        attach_artifact_meta(self.data, stage="track", path=self.json_path)
        save_json(self.json_path, self.data, indent=self.indent)
        logger.info("JSON сохранён: %s", self.json_path)
