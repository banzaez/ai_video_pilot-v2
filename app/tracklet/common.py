"""Общие хелперы tracklet pipeline."""

from __future__ import annotations

import os
from typing import Any

from app.config import Settings, info_json_path
from app.io.json_util import load_tracking_json
from app.detections_io import detections_from_json
from app.io.video import VideoSource
from app.session.manifest import is_session_manifest


def session_manifest(settings: Settings) -> dict | None:
    info_path = info_json_path(settings)
    info = load_tracking_json(info_path) if os.path.isfile(info_path) else None
    return info if is_session_manifest(info) else None


def load_detection_meta(settings: Settings, det_path: str) -> dict[str, Any]:
    data = load_tracking_json(det_path)
    if not data:
        raise ValueError(f"Пустой detections JSON: {det_path}")
    total_frames = int(data.get("frame_count") or 0)
    fps = float(data.get("fps") or 0.0)
    width = int(data.get("width") or 0)
    height = int(data.get("height") or 0)
    detect_every_n = int(data.get("detect_every_n") or settings.detect_every_n)
    if not total_frames or not fps:
        manifest = session_manifest(settings)
        if manifest:
            total_frames = total_frames or int(manifest.get("frame_count") or 0)
            fps = fps or float(manifest.get("fps") or 0.0)
            width = width or int(manifest.get("width") or 0)
            height = height or int(manifest.get("height") or 0)
        elif os.path.isfile(str(settings.input_path)):
            with VideoSource(settings.input_path) as source:
                total_frames = total_frames or int(source.meta.frame_count or 0)
                fps = fps or float(source.meta.fps or 0.0)
                width = width or int(source.meta.width or 0)
                height = height or int(source.meta.height or 0)
    if not total_frames:
        raise ValueError("Не удалось определить число кадров видео")
    return {
        "data": data,
        "all_detections": detections_from_json(data),
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "detect_every_n": detect_every_n,
        "video_source": str(data.get("video_source") or settings.input_path),
    }


def tracklet_frames_to_tracked(frames_data: dict) -> dict[int, list[dict[str, Any]]]:
    """tracklet_frames.json → dict с track_id (для build_track_summaries)."""
    out: dict[int, list[dict[str, Any]]] = {}
    for frame in frames_data.get("frames") or []:
        fi = int(frame["frame_index"]) - 1
        dets = []
        for det in frame.get("detections") or []:
            item = dict(det)
            item["track_id"] = int(det.get("tracklet_id") or det.get("track_id"))
            dets.append(item)
        if dets:
            out[fi] = dets
    return out
