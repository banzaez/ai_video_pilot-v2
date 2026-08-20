"""Геометрические утилиты для работы с Bounding Box и NMS (автономный модуль)."""

from __future__ import annotations

from typing import Any


def bbox_wh(bbox: list[Any] | tuple[Any, ...]) -> tuple[float, float]:
    """Ширина и высота бокса [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def bbox_area(bbox: list[float] | tuple[float, ...]) -> float:
    """Площадь бокса [x1, y1, x2, y2]."""
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _bbox_inter(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    """Площадь пересечения двух боксов."""
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: list[float], b: list[float]) -> float:
    """Intersection over Union (IoU)."""
    inter = _bbox_inter(a, b)
    if inter <= 0:
        return 0.0
    ua = bbox_area(a) + bbox_area(b) - inter
    return float(inter / ua) if ua > 0 else 0.0


def bbox_ios(a: list[float], b: list[float]) -> float:
    """Intersection over Smaller Area (IoS).

    Возвращает ~1.0, если один из боксов почти целиком содержится внутри другого.
    """
    inter = _bbox_inter(a, b)
    if inter <= 0:
        return 0.0
    smaller = min(bbox_area(a), bbox_area(b))
    return float(inter / smaller) if smaller > 0 else 0.0


_NMS_CONTAIN_IOS = 0.70
_NMS_BLOB_WIDTH = 1.35
_NMS_BLOB_HEIGHT = 0.55
_NMS_BLOB_IOS = 0.50


def _is_two_person_blob(wide: list[float], inner: list[float]) -> bool:
    """True, если `wide` накрывает соседа, а не торс/голову того же человека."""
    ww, wh = bbox_wh(wide)
    iw, ih = bbox_wh(inner)
    if ww < _NMS_BLOB_WIDTH * iw:
        return False
    if ih < _NMS_BLOB_HEIGHT * wh:
        return False
    return bbox_ios(inner, wide) >= _NMS_BLOB_IOS


def nms_detections(raw_boxes: list[dict[str, Any]], iou_thresh: float) -> list[dict[str, Any]]:
    """Убрать дубли (торс + полный рост) и широкие ложные боксы «на двоих».

    1. Выкидываются широкие блобы, если внутри есть отдельный человек.
    2. Сортировка по confidence (убывание).
    3. Подавление по IoU >= iou_thresh и по IoS >= 0.70 (вложенные боксы).
    """
    if iou_thresh <= 0 or len(raw_boxes) < 2:
        return raw_boxes

    # Шаг 1: подавление ложных боксов на двоих
    blob_indices = {
        i
        for i, di in enumerate(raw_boxes)
        if any(
            j != i and _is_two_person_blob(di["bbox"], raw_boxes[j]["bbox"])
            for j in range(len(raw_boxes))
        )
    }
    filtered = [b for idx, b in enumerate(raw_boxes) if idx not in blob_indices]
    if len(filtered) < 2:
        return filtered

    # Шаг 2: сортировка по confidence
    order = sorted(range(len(filtered)), key=lambda k: float(filtered[k]["confidence"]), reverse=True)

    # Шаг 3: NMS c учетом IoU и вложенности IoS
    keep: list[int] = []
    for i in order:
        bi = filtered[i]["bbox"]
        if any(
            bbox_iou(bi, filtered[j]["bbox"]) >= iou_thresh
            or bbox_ios(bi, filtered[j]["bbox"]) >= _NMS_CONTAIN_IOS
            for j in keep
        ):
            continue
        keep.append(i)

    keep.sort()
    return [filtered[i] for i in keep]
