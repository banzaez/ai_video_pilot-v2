"""Кинематический фильтр выбросов BBox (Kinematic Outlier Filter).

Обнаруживает резкие пространственные скачки центра BBox, скорости перемещения
и площади детекции на краях трека для исключения ошибочно прикрепившихся чужих детекций.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Protocol, Sequence, TypeVar

logger = logging.getLogger(__name__)


class BBoxCandidate(Protocol):
    frame_index: int
    target_det: dict[str, Any]


T = TypeVar("T", bound=BBoxCandidate)


def _get_bbox(cand: BBoxCandidate) -> list[float] | None:
    det = cand.target_det
    if not isinstance(det, dict):
        return None
    bb = det.get("bbox")
    if not bb or len(bb) < 4:
        return None
    return [float(v) for v in bb[:4]]


def filter_kinematic_outliers(
    candidates: Sequence[T],
    *,
    max_speed_ratio: float = 3.0,
    max_area_ratio: float = 2.2,
    min_jump_pixels: float = 35.0,
    edge_check_window: int = 2,
    min_candidates: int = 5,
) -> list[T]:
    """Фильтрует краевые кандидаты с аномальными кинематическими скачками (положение/площадь/скорость).

    Проверяет первые и последние edge_check_window кандидатов относительно медианных характеристик трека.
    """
    if len(candidates) < min_candidates:
        return list(candidates)

    sorted_cands = sorted(candidates, key=lambda c: c.frame_index)
    n = len(sorted_cands)

    boxes: list[list[float]] = []
    centers: list[tuple[float, float]] = []
    areas: list[float] = []
    aspects: list[float] = []

    for c in sorted_cands:
        bb = _get_bbox(c)
        if bb is None:
            # Если bbox некорректен, возвращаем без фильтрации
            return list(candidates)
        x1, y1, x2, y2 = bb
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        boxes.append(bb)
        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        areas.append(w * h)
        aspects.append(h / w)

    # 1. Медианная площадь и aspect ratio
    sorted_areas = sorted(areas)
    median_area = sorted_areas[n // 2]

    # 2. Вычисление скоростей перемещения между соседними кадрами
    speeds: list[float] = []
    for i in range(n - 1):
        df = max(1, sorted_cands[i + 1].frame_index - sorted_cands[i].frame_index)
        dx = centers[i + 1][0] - centers[i][0]
        dy = centers[i + 1][1] - centers[i][1]
        dist = math.sqrt(dx * dx + dy * dy)
        speeds.append(dist / float(df))

    if not speeds:
        return list(candidates)

    sorted_speeds = sorted(speeds)
    median_speed = sorted_speeds[len(sorted_speeds) // 2]
    speed_thresh = max(min_jump_pixels, median_speed * max(1.5, float(max_speed_ratio)))

    # 3. Проверка краевых кандидатов на выбросы
    outlier_indices: set[int] = set()

    # Начало трека (start window)
    for i in range(min(edge_check_window, n // 3)):
        is_outlier = False
        # Проверка скачка скорости к следующему кадру
        if i < len(speeds) and speeds[i] > speed_thresh:
            is_outlier = True

        # Проверка скачка площади
        if areas[i] > median_area * max_area_ratio or areas[i] < median_area / max_area_ratio:
            is_outlier = True

        if is_outlier:
            outlier_indices.add(i)
        else:
            # Если первый проверенный кадр в порядке, дальше вглубь не ищем
            break

    # Конец трека (end window)
    for i in range(n - 1, max(n - 1 - edge_check_window, n - 1 - n // 3), -1):
        is_outlier = False
        speed_idx = i - 1
        if 0 <= speed_idx < len(speeds) and speeds[speed_idx] > speed_thresh:
            is_outlier = True

        if areas[i] > median_area * max_area_ratio or areas[i] < median_area / max_area_ratio:
            is_outlier = True

        if is_outlier:
            outlier_indices.add(i)
        else:
            break

    if not outlier_indices:
        return list(candidates)

    result = [c for i, c in enumerate(sorted_cands) if i not in outlier_indices]
    return result if result else list(candidates)
