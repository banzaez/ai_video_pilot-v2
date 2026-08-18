"""Вырезка ROI и координаты bbox в кропе."""

from __future__ import annotations

from typing import Any

from app.util.bbox import bbox_wh


def crop_roi_xyxy(
    bbox: list[float], frame_w: int, frame_h: int, pad: float = 0.0
) -> tuple[int, int, int, int]:
    """Область вырезки: по умолчанию ровно bbox (pad=0), без полей."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = bbox_wh(bbox)
    bw, bh = max(1.0, bw), max(1.0, bh)
    rx1 = int(max(0, x1 - bw * pad))
    ry1 = int(max(0, y1 - bh * pad * (1.4 if pad > 0 else 0)))
    rx2 = int(min(frame_w, x2 + bw * pad)) if frame_w > 0 else int(x2 + bw * pad)
    ry2 = int(min(frame_h, y2 + bh * pad * (0.3 if pad > 0 else 0))) if frame_h > 0 else int(y2 + bh * pad)
    if rx2 <= rx1 or ry2 <= ry1:
        return (0, 0, 1, 1)
    return (rx1, ry1, rx2, ry2)


def crop_person(
    frame: np.ndarray, bbox: list[float], pad: float = 0.0
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    fh, fw = frame.shape[:2]
    roi = crop_roi_xyxy(bbox, fw, fh, pad)
    x1, y1, x2, y2 = roi
    return frame[y1:y2, x1:x2], roi


def bbox_in_crop(
    bbox: list[Any] | None, crop_roi: list[Any] | None
) -> tuple[int, int, int, int] | None:
    """Bbox трека в координатах JPEG-кропа (crop_roi — вырезка из кадра)."""
    if not bbox or not crop_roi or len(bbox) < 4 or len(crop_roi) < 4:
        return None
    rx1, ry1 = int(crop_roi[0]), int(crop_roi[1])
    bx1, by1, bx2, by2 = (int(round(float(v))) for v in bbox[:4])
    return bx1 - rx1, by1 - ry1, bx2 - rx1, by2 - ry1
