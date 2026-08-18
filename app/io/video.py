"""Работа с видео: открытие и метаданные."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import cv2

logger = logging.getLogger(__name__)


def ensure_parent_dir(path: str | None) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int | None
    source: str | int


class VideoSource:
    def __init__(self, input_path: str | int):
        self.raw = input_path
        self.source = input_path
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {input_path}")

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.meta = VideoMeta(
            width=width,
            height=height,
            fps=fps,
            frame_count=count if count > 0 else None,
            source=input_path,
        )

    def read(self):
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
