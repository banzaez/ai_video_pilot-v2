"""Точка «ног» на кадре и проекция на план."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.global_id.camera_pose import (
    DEFAULT_TORSO_H_M,
    CameraPose,
    load_camera_doc,
    parse_camera_pose,
    project_bbox_feet_to_map,
)
from app.global_id.spatial import load_camera_h


_COUNTERS_CACHE: dict[str, dict[str, list[list[list[float]]]]] = {}
_COUNTERS_MAP_CACHE: dict[str, list[list[list[float]]]] = {}


def reset_counters_cache() -> None:
    _COUNTERS_CACHE.clear()
    _COUNTERS_MAP_CACHE.clear()


@dataclass
class FeetResolve:
    xy: tuple[float, float]
    source: str
    confidence: float
    counter_blocked: bool = False


@dataclass
class MapFeetResult:
    map_xy: tuple[float, float]
    source: str
    confidence: float


def bbox_rel_offset(bbox: Any, img_feet: tuple[float, float]) -> tuple[float, float] | None:
    """Смещение точки ног относительно низа bbox в долях ширины/высоты."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    w = abs(x2 - x1) or 1e-6
    h = abs(y2 - y1) or 1e-6
    x_mid = (x1 + x2) * 0.5
    y_bot = max(y1, y2)
    return (img_feet[0] - x_mid) / w, (img_feet[1] - y_bot) / h


def apply_bbox_rel_offset(bbox: Any, dx_rel: float, dy_rel: float) -> tuple[float, float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    x_mid = (x1 + x2) * 0.5
    y_bot = max(y1, y2)
    return x_mid + dx_rel * w, y_bot + dy_rel * h


from app.util.bbox import bbox_bottom_center, scale_bbox

# Совместимость со старыми именами
feet_from_bbox = bbox_bottom_center
scale_bbox_to_image_size = scale_bbox


def load_counters_image_zones(maps_dir: str) -> dict[str, list[list[list[float]]]]:
    if maps_dir in _COUNTERS_CACHE:
        return _COUNTERS_CACHE[maps_dir]
    out: dict[str, list[list[list[float]]]] = {}
    path = os.path.join(maps_dir, "counters.json")
    if not maps_dir or not os.path.isfile(path):
        _COUNTERS_CACHE[maps_dir] = out
        return out
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        _COUNTERS_CACHE[maps_dir] = out
        return out
    for item in data.get("counters") or []:
        by_cam = item.get("image_by_camera") or {}
        if not isinstance(by_cam, dict):
            continue
        for cam, poly in by_cam.items():
            if isinstance(poly, list) and len(poly) >= 3:
                out.setdefault(str(cam), []).append(poly)
    _COUNTERS_CACHE[maps_dir] = out
    return out


def load_counters_map_polys(maps_dir: str) -> list[list[list[float]]]:
    if maps_dir in _COUNTERS_MAP_CACHE:
        return _COUNTERS_MAP_CACHE[maps_dir]
    path = os.path.join(maps_dir, "counters.json")
    if not maps_dir or not os.path.isfile(path):
        _COUNTERS_MAP_CACHE[maps_dir] = []
        return _COUNTERS_MAP_CACHE[maps_dir]
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        _COUNTERS_MAP_CACHE[maps_dir] = []
        return _COUNTERS_MAP_CACHE[maps_dir]
    out: list[list[list[float]]] = []
    for item in data.get("counters") or []:
        mp = item.get("map")
        if isinstance(mp, list) and len(mp) >= 3:
            out.append(mp)
    _COUNTERS_MAP_CACHE[maps_dir] = out
    return out


def point_in_polygon(x: float, y: float, poly: list[list[float]]) -> bool:
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _nearest_point_on_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[float, float]:
    dx = bx - ax
    dy = by - ay
    len2 = dx * dx + dy * dy
    if len2 < 1e-9:
        return ax, ay
    t = ((px - ax) * dx + (py - ay) * dy) / len2
    t = max(0.0, min(1.0, t))
    return ax + t * dx, ay + t * dy


def nearest_point_on_polygon_boundary(x: float, y: float, poly: list[list[float]]) -> tuple[float, float]:
    if len(poly) < 2:
        return x, y
    best = (float(poly[0][0]), float(poly[0][1]))
    best_d = float("inf")
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        qx, qy = _nearest_point_on_segment(x, y, float(a[0]), float(a[1]), float(b[0]), float(b[1]))
        d = float(np.hypot(qx - x, qy - y))
        if d < best_d:
            best_d = d
            best = (qx, qy)
    return best


def adjust_map_point_for_counters(
    map_xy: tuple[float, float],
    map_polys: list[list[list[float]]],
) -> tuple[float, float]:
    x, y = map_xy
    for poly in map_polys:
        if len(poly) < 3:
            continue
        if point_in_polygon(x, y, poly):
            return nearest_point_on_polygon_boundary(x, y, poly)
    return map_xy


def bbox_feet_in_counter(bbox: Any, camera_key: str, maps_dir: str) -> bool:
    feet = feet_from_bbox(bbox)
    if feet is None:
        return False
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False
    x1, _y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    mid = ((x1 + x2) * 0.5, y2)
    zones = load_counters_image_zones(maps_dir).get(camera_key) or []
    for poly in zones:
        if point_in_polygon(feet[0], feet[1], poly):
            return True
        if point_in_polygon(mid[0], mid[1], poly):
            return True
    return False


def resolve_feet(
    *,
    bbox: Any,
    camera_key: str = "",
    maps_dir: str = "",
    **_unused: Any,
) -> FeetResolve | None:
    pt_b = feet_from_bbox(bbox)
    if pt_b is None:
        return None
    return FeetResolve(xy=pt_b, source="bbox", confidence=0.4, counter_blocked=False)


def resolve_feet_xy(**kwargs: Any) -> tuple[float, float] | None:
    got = resolve_feet(**kwargs)
    return None if got is None else got.xy


def _image_size_from_doc(doc: dict[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(doc, dict):
        return None
    raw = doc.get("image_size")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        w, h = int(raw[0]), int(raw[1])
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return w, h


def _cameras_dir(maps_dir: str) -> str:
    if os.path.basename(maps_dir.rstrip("/")) == "cameras":
        return maps_dir
    return os.path.join(maps_dir, "cameras")


def project_feet_to_map(
    bbox: Any,
    *,
    camera_key: str = "",
    maps_dir: str = "",
    camera_doc: dict[str, Any] | None = None,
    H: np.ndarray | None = None,
    torso_height_m: float = DEFAULT_TORSO_H_M,
    person_height_m: float = 1.70,
    tracking_size: tuple[int, int] | None = None,
    kxy: list[list[float]] | None = None,
    kcf: list[float] | None = None,
    kpt_min: float = 0.25,
    h_loo_rms: float | None = None,
    ray_rms: float | None = None,
) -> MapFeetResult | None:
    doc = camera_doc
    if doc is None and maps_dir and camera_key:
        doc = load_camera_doc(_cameras_dir(maps_dir), camera_key)

    pose: CameraPose | None = parse_camera_pose(doc) if doc else None
    image_size = _image_size_from_doc(doc)
    scaled_bbox = scale_bbox_to_image_size(bbox, tracking_size, image_size)

    if H is None and maps_dir and camera_key:
        H = load_camera_h(_cameras_dir(maps_dir), camera_key)

    got = project_bbox_feet_to_map(
        scaled_bbox,
        pose=pose,
        image_size=image_size,
        H=H,
        torso_height_m=torso_height_m,
        person_height_m=person_height_m,
        h_loo_rms=h_loo_rms,
        ray_rms=ray_rms,
        kxy=kxy,
        kcf=kcf,
        kpt_min=kpt_min,
    )
    if got is None:
        return None
    map_xy, source, confidence = got
    return MapFeetResult(map_xy=map_xy, source=source, confidence=confidence)
