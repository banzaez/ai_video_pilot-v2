"""Общие операции с bbox xyxy."""

from __future__ import annotations

from typing import Any


def bbox_wh(bbox: list[Any] | tuple[Any, ...]) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def bbox_area(bbox: list[float] | tuple[float, ...]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def bbox_center(bbox: list[Any] | tuple[Any, ...]) -> tuple[float, float]:
    """Центр bounding box (x_center, y_center)."""
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def bbox_bottom_center(bbox: list[Any] | tuple[Any, ...]) -> tuple[float, float] | None:
    """Точка опоры / ног: середина нижней грани (x_mid, y_bottom)."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return None
    return (x1 + x2) * 0.5, max(y1, y2)


def scale_bbox(
    bbox: Any,
    src_size: tuple[int, int] | None,
    dst_size: tuple[int, int] | None,
) -> list[float]:
    """Масштабирует bbox [x1, y1, x2, y2] из src_size (w, h) в dst_size (w, h)."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return list(bbox) if isinstance(bbox, (list, tuple)) else []
    try:
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return [float(v) for v in bbox[:4]]
    if not src_size or not dst_size:
        return [x1, y1, x2, y2]
    sw, sh = int(src_size[0]), int(src_size[1])
    dw, dh = int(dst_size[0]), int(dst_size[1])
    if sw <= 0 or sh <= 0 or (sw == dw and sh == dh):
        return [x1, y1, x2, y2]
    sx = dw / sw
    sy = dh / sh
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def xyxy_to_xywh(bbox: list[Any] | tuple[Any, ...]) -> list[float]:
    """[x1, y1, x2, y2] → [cx, cy, w, h]"""
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return [(x1 + x2) * 0.5, (y1 + y2) * 0.5, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def xywh_to_xyxy(xywh: list[Any] | tuple[Any, ...]) -> list[float]:
    """[cx, cy, w, h] → [x1, y1, x2, y2]"""
    cx, cy, w, h = (float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3]))
    hw, hh = w * 0.5, h * 0.5
    return [cx - hw, cy - hh, cx + hw, cy + hh]


def _bbox_inter(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: list[float], b: list[float]) -> float:
    inter = _bbox_inter(a, b)
    if inter <= 0:
        return 0.0
    ua = bbox_area(a) + bbox_area(b) - inter
    return float(inter / ua) if ua > 0 else 0.0


def bbox_ios(a: list[float], b: list[float]) -> float:
    """Intersection over smaller area. ~1.0 если меньший бокс целиком внутри большего."""
    inter = _bbox_inter(a, b)
    if inter <= 0:
        return 0.0
    smaller = min(bbox_area(a), bbox_area(b))
    return float(inter / smaller) if smaller > 0 else 0.0

