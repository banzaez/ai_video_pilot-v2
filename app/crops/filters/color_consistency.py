"""Фильтр цветовой самосогласованности трека (HSV Color Consistency).

Извлекает компактные HSV-гистограммы верхней части тела (одежда) и фильтрует
аномальные кадры, цвет одежды на которых резко отличается от медианного цвета трека.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, Sequence, TypeVar

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ColorCropCandidate(Protocol):
    frame_index: int
    target_det: dict[str, Any]
    crop_image: np.ndarray | None
    image: np.ndarray | None


T = TypeVar("T", bound=ColorCropCandidate)


def extract_clothing_hsv_hist(
    crop_img: np.ndarray | None,
    *,
    h_bins: int = 16,
    s_bins: int = 8,
    y_start_ratio: float = 0.12,
    y_end_ratio: float = 0.65,
    x_margin_ratio: float = 0.15,
) -> np.ndarray | None:
    """Извлекает 2D (H-S) нормированную гистограмму верхней части тела из кропа человека."""
    if crop_img is None or crop_img.size == 0:
        return None

    ch, cw = crop_img.shape[:2]
    if ch < 16 or cw < 12:
        return None

    y1 = int(round(ch * max(0.0, y_start_ratio)))
    y2 = int(round(ch * min(1.0, y_end_ratio)))
    x1 = int(round(cw * max(0.0, x_margin_ratio)))
    x2 = int(round(cw * (1.0 - max(0.0, x_margin_ratio))))

    if y2 <= y1 or x2 <= x1:
        return None

    body_roi = crop_img[y1:y2, x1:x2]
    if body_roi.size == 0:
        return None

    hsv = cv2.cvtColor(body_roi, cv2.COLOR_BGR2HSV)
    # H: [0..180], S: [0..256]
    hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.astype(np.float32)


def compute_hist_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Вычисляет сходство двух нормированных гистограмм в диапазоне [0..1].

    Используется комбинация Bhattacharyya distance и Correlation.
    """
    if hist_a is None or hist_b is None:
        return 1.0

    bhatt_dist = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))
    # Bhattacharyya distance: 0.0 = match, 1.0 = complete mismatch
    sim_bhatt = max(0.0, min(1.0, 1.0 - bhatt_dist))

    correl = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
    sim_correl = max(0.0, min(1.0, (correl + 1.0) / 2.0))

    return 0.6 * sim_bhatt + 0.4 * sim_correl


def filter_color_outliers(
    candidates: Sequence[T],
    *,
    min_similarity: float = 0.50,
    edge_only: bool = True,
    edge_window: int = 2,
    min_candidates: int = 4,
) -> list[T]:
    """Фильтрует кандидатов, цвет одежды которых сильно отличается от медианного цвета трека.

    Если edge_only=True, фильтрация применяется только к первым и последним edge_window кадрам,
    где чаще всего возникают ошибки переключения трекера.
    """
    if len(candidates) < min_candidates:
        return list(candidates)

    sorted_cands = sorted(candidates, key=lambda c: c.frame_index)
    n = len(sorted_cands)

    hists: list[np.ndarray | None] = []
    valid_hists: list[np.ndarray] = []

    for c in sorted_cands:
        crop = c.crop_image
        if (crop is None or crop.size == 0) and c.image is not None and c.image.size > 0:
            tb = c.target_det.get("bbox")
            if tb and len(tb) >= 4:
                x1, y1, x2, y2 = [int(round(float(v))) for v in tb[:4]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(c.image.shape[1], x2), min(c.image.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    crop = c.image[y1:y2, x1:x2]

        h = extract_clothing_hsv_hist(crop)
        hists.append(h)
        if h is not None:
            valid_hists.append(h)

    # Если для трека недостаточно валидных гистограмм, не фильтруем
    if len(valid_hists) < max(3, n // 2):
        return list(candidates)

    # 1. Построение медианной / центроидной гистограммы трека
    stacked = np.stack(valid_hists, axis=0)  # [N, H, S]
    median_hist = np.median(stacked, axis=0).astype(np.float32)
    cv2.normalize(median_hist, median_hist, alpha=1.0, norm_type=cv2.NORM_L1)

    # 2. Оценка сходства каждого кандидата с медианой
    outlier_indices: set[int] = set()

    for i in range(n):
        if edge_only:
            is_edge = (i < edge_window) or (i >= n - edge_window)
            if not is_edge:
                continue

        h = hists[i]
        if h is None:
            continue

        sim = compute_hist_similarity(h, median_hist)
        if sim < float(min_similarity):
            logger.debug(
                "Color consistency outlier: frame %s (sim=%.3f < %.3f)",
                sorted_cands[i].frame_index,
                sim,
                min_similarity,
            )
            outlier_indices.add(i)

    if not outlier_indices:
        return list(candidates)

    result = [c for i, c in enumerate(sorted_cands) if i not in outlier_indices]
    return result if result else list(candidates)
