"""Модули фильтрации краевых и аномальных кадров трека."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

from app.crops.filters.color_consistency import (
    compute_hist_similarity,
    extract_clothing_hsv_hist,
    filter_color_outliers,
)
from app.crops.filters.kinematic import filter_kinematic_outliers
from app.crops.filters.temporal_trim import trim_edge_items

T = TypeVar("T")


@dataclass(frozen=True)
class OutlierFilterConfig:
    """Конфигурация многоуровневой фильтрации выбросов в треке."""

    # Временной тримминг краев
    trim_enabled: bool = True
    trim_start: int = 2
    trim_end: int = 2
    trim_min_len: int = 8

    # Кинематический фильтр скачков BBox
    kinematic_enabled: bool = True
    kinematic_max_speed_ratio: float = 3.0
    kinematic_max_area_ratio: float = 2.2
    kinematic_min_candidates: int = 5

    # Фильтр цветовой самосогласованности HSV
    color_enabled: bool = True
    color_min_similarity: float = 0.50
    color_edge_only: bool = True
    color_edge_window: int = 2
    color_min_candidates: int = 3


def filter_track_outlier_candidates(
    candidates: Sequence[T],
    config: OutlierFilterConfig | None = None,
) -> list[T]:
    """Применяет конвейер фильтров (Trim -> Kinematics -> Color) к кандидатам одного трека."""
    if not candidates:
        return []

    cfg = config or OutlierFilterConfig()
    current: list[T] = list(candidates)

    # 1. Временной тримминг
    if cfg.trim_enabled and (cfg.trim_start > 0 or cfg.trim_end > 0):
        current = trim_edge_items(
            current,
            trim_start=cfg.trim_start,
            trim_end=cfg.trim_end,
            min_len=cfg.trim_min_len,
        )

    # 2. Кинематический фильтр
    if cfg.kinematic_enabled and len(current) >= cfg.kinematic_min_candidates:
        current = filter_kinematic_outliers(
            current,
            max_speed_ratio=cfg.kinematic_max_speed_ratio,
            max_area_ratio=cfg.kinematic_max_area_ratio,
            min_candidates=cfg.kinematic_min_candidates,
        )

    # 3. Фильтр цветовой согласованности
    if cfg.color_enabled and len(current) >= cfg.color_min_candidates:
        current = filter_color_outliers(
            current,
            min_similarity=cfg.color_min_similarity,
            edge_only=cfg.color_edge_only,
            edge_window=cfg.color_edge_window,
            min_candidates=cfg.color_min_candidates,
        )

    return current


__all__ = [
    "OutlierFilterConfig",
    "compute_hist_similarity",
    "extract_clothing_hsv_hist",
    "filter_color_outliers",
    "filter_kinematic_outliers",
    "filter_track_outlier_candidates",
    "trim_edge_items",
]
