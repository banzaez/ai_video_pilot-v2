"""Базовый абстрактный класс для детекторов объектов (Stage 1: detect)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseDetector(ABC):
    """Абстрактный интерфейс детектора для инференса по батчам кадров."""

    @abstractmethod
    def detect_batch(
        self,
        frames: list[np.ndarray],
        *,
        conf: float = 0.25,
        nms_iou: float = 0.5,
        classes: list[int] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Инференс по списку кадров BGR.

        Возвращает список детекций для каждого кадра:
        [
            [{"bbox": [x1, y1, x2, y2], "confidence": float}, ...],  # кадр 0
            ...
        ]
        """
        pass
