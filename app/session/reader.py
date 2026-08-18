"""Чтение кадров из multi-part session."""

from __future__ import annotations

import os
from typing import Any

import cv2

from app.parallel_tracker import open_video_capture
from app.session.discover import frame_to_part


def _resolve_abs(path: str) -> str:
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    alt = os.path.join(os.getcwd(), path)
    if os.path.isfile(alt):
        return alt
    return path


class SessionFrameReader:
    """Lazy VideoCapture per part; читает global frame index (0-based)."""

    def __init__(self, manifest: dict[str, Any]):
        self.manifest = manifest
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._last_part_path: str | None = None

    def _cap_for(self, part: dict[str, Any]) -> cv2.VideoCapture:
        path = _resolve_abs(str(part.get("path") or ""))
        if path not in self._caps:
            cap = open_video_capture(path)
            if not cap.isOpened():
                raise ValueError(f"Не удалось открыть видео: {path}")
            self._caps[path] = cap
        return self._caps[path]

    def _advance_to(self, cap: cv2.VideoCapture, path: str, local: int) -> tuple[bool, Any]:
        """Вперёд по файлу через grab(); seek только назад или при смене part."""
        if self._last_part_path != path:
            cap.set(cv2.CAP_PROP_POS_FRAMES, local)
            self._last_part_path = path
            return cap.read()
        cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        if local < cur:
            cap.set(cv2.CAP_PROP_POS_FRAMES, local)
            return cap.read()
        if local == cur:
            return cap.read()
        while cur < local:
            if not cap.grab():
                return False, None
            cur += 1
        return cap.retrieve()

    def read_frame(self, global_frame: int):
        part, local = frame_to_part(self.manifest, global_frame)
        path = _resolve_abs(str(part.get("path") or ""))
        cap = self._cap_for(part)
        ret, frame = self._advance_to(cap, path, local)
        if not ret or frame is None:
            raise ValueError(f"Кадр {global_frame} (local {local}) не прочитан из {path}")
        return frame

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
        self._last_part_path = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
