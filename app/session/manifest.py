"""Manifest camera-day session → info.json."""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.parse import quote

from app.io.video import VideoSource
from app.session.discover import Session

logger = logging.getLogger(__name__)


def _resolve_abs(path: str) -> str:
    if os.path.isabs(path):
        return path
    cwd = os.path.abspath(path)
    if os.path.isfile(cwd):
        return cwd
    alt = os.path.join(os.getcwd(), path)
    if os.path.isfile(alt):
        return alt
    return path


def build_session_manifest(session: Session) -> dict[str, Any]:
    if not session.parts:
        raise ValueError(f"Session {session.key}: нет частей")

    parts_out: list[dict[str, Any]] = []
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    frame_offset = 0
    time_offset = 0.0
    total_frames = 0

    for part in session.parts:
        abs_path = _resolve_abs(part.path)
        if not os.path.isfile(abs_path):
            raise ValueError(f"Часть не найдена: {part.path}")

        with VideoSource(abs_path) as src:
            meta = src.meta
            part_fps = float(meta.fps or 0.0)
            part_w = int(meta.width or 0)
            part_h = int(meta.height or 0)
            part_frames = int(meta.frame_count or 0)

        if part_fps <= 0 or part_frames <= 0:
            raise ValueError(f"Некорректные метаданные: {part.path}")

        if fps is None:
            fps = part_fps
            width = part_w
            height = part_h
        else:
            if abs(fps - part_fps) > 0.01:
                raise ValueError(
                    f"FPS не совпадает в session {session.key}: {fps} vs {part_fps} ({part.path})"
                )
            if width != part_w or height != part_h:
                raise ValueError(
                    f"Разрешение не совпадает в session {session.key}: "
                    f"{width}x{height} vs {part_w}x{part_h} ({part.path})"
                )

        part_duration = part_frames / part_fps
        parts_out.append(
            {
                "stem": part.stem,
                "name": part.name,
                "path": part.path,
                "url": "/media/" + quote(part.name),
                "started_at": part.started_at,
                "ended_at": part.ended_at,
                "frame_offset": frame_offset,
                "frame_count": part_frames,
                "time_offset_sec": round(time_offset, 3),
            }
        )
        frame_offset += part_frames
        time_offset += part_duration
        total_frames += part_frames

    duration_sec = round(total_frames / float(fps), 3) if fps else 0.0
    return {
        "stage": "info",
        "kind": "camera_day",
        "session_key": session.key,
        "camera_index": session.camera_index,
        "camera": f"Camera_{session.camera_index:03d}",
        "day": session.day,
        "parts": parts_out,
        "fps": round(float(fps), 3),
        "width": int(width or 0),
        "height": int(height or 0),
        "frame_count": total_frames,
        "duration_sec": duration_sec,
        "parsed": {
            "ok": True,
            "camera": f"Camera_{session.camera_index:03d}",
            "camera_index": session.camera_index,
            "started_at": parts_out[0]["started_at"],
            "ended_at": parts_out[-1]["ended_at"],
            "duration_sec": duration_sec,
        },
    }


def load_session_manifest(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("kind") != "camera_day":
        raise ValueError(f"Не session manifest: {path}")
    return data


def is_session_manifest(info: dict[str, Any] | None) -> bool:
    return bool(info and info.get("kind") == "camera_day")
