"""Гомография как предикат can_pair: слишком далеко на плане при одновременном появлении."""

from __future__ import annotations

import json
import math
import os
import numpy as np

from app.util.intervals import intervals_overlap

# Сетка плана: 80 px = 0.5 м → 160 px = 1 м (как admin/src/mapGrid.ts).
METER_PX = 160.0
DEFAULT_MAX_MAP_M = 3.0


def load_camera_h(cameras_dir: str, camera_key: str) -> np.ndarray | None:
    candidates = [f"{camera_key}.json"]
    try:
        idx = int(camera_key)
        candidates.append(f"{idx:03d}.json")
        candidates.append(f"{idx:02d}.json")
        candidates.append(f"{idx}.json")
    except ValueError:
        pass

    data = None
    for name in dict.fromkeys(candidates):
        path = os.path.join(cameras_dir, name)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                break
            except Exception:
                continue
    if not isinstance(data, dict):
        return None
    raw = data.get("H") if isinstance(data, dict) else None
    if not isinstance(raw, (list, tuple)) or len(raw) != 9:
        return None
    try:
        H = np.asarray(raw, dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError):
        return None
    return H


H_LOO_MIN_PAIRS = 6


def compute_homography(pairs: list[dict[str, Any]]) -> np.ndarray | None:
    """DLT image→map, ≥4 пар. Возвращает 3×3 или None."""
    if len(pairs) < 4:
        return None
    rows: list[list[float]] = []
    for pair in pairs:
        img = pair.get("image") if isinstance(pair, dict) else None
        mp = pair.get("map") if isinstance(pair, dict) else None
        if not isinstance(img, (list, tuple)) or not isinstance(mp, (list, tuple)):
            continue
        if len(img) < 2 or len(mp) < 2:
            continue
        x, y = float(img[0]), float(img[1])
        X, Y = float(mp[0]), float(mp[1])
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -X * x, -X * y, -X])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -Y * x, -Y * y, -Y])
    if len(rows) < 8:
        return None
    a = np.asarray(rows, dtype=np.float64)
    try:
        _, _, vt = np.linalg.svd(a)
    except np.linalg.LinAlgError:
        return None
    h = vt[-1].reshape(3, 3)
    scale = float(h[2, 2])
    if abs(scale) < 1e-12:
        return None
    return h / scale


def leave_one_out_h_rms(pairs: list[dict[str, Any]]) -> float | None:
    """Leave-one-out RMS гомографии. None, если пар < 6 (H нечем проверить)."""
    n = len(pairs)
    if n < H_LOO_MIN_PAIRS:
        return None
    errs: list[float] = []
    for i in range(n):
        rest = [p for j, p in enumerate(pairs) if j != i]
        h = compute_homography(rest)
        if h is None:
            continue
        img = pairs[i].get("image") if isinstance(pairs[i], dict) else None
        mp = pairs[i].get("map") if isinstance(pairs[i], dict) else None
        if not isinstance(img, (list, tuple)) or not isinstance(mp, (list, tuple)):
            continue
        mapped = apply_h(h, float(img[0]), float(img[1]))
        if mapped is None:
            continue
        errs.append(float(np.hypot(mapped[0] - float(mp[0]), mapped[1] - float(mp[1]))))
    if len(errs) < 2:
        return None
    return float(math.sqrt(sum(e * e for e in errs) / len(errs)))


def prefer_homography_over_ray(h_loo_rms: float | None, ray_rms: float | None) -> bool:
    if h_loo_rms is None or ray_rms is None:
        return False
    if not (h_loo_rms < 25.0):
        return False
    return ray_rms > max(2.5 * h_loo_rms, h_loo_rms + 15.0)


def apply_h(H: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
    v = H @ np.array([float(x), float(y), 1.0], dtype=np.float64)
    if abs(float(v[2])) < 1e-9:
        return None
    return float(v[0] / v[2]), float(v[1] / v[2])


def map_points_for_node(node: dict[str, Any], H: np.ndarray | None) -> list[tuple[float, float, float]]:
    """(t, map_x, map_y) из map_samples узла."""
    out: list[tuple[float, float, float]] = []
    for rec in node.get("map_samples") or []:
        xy = rec.get("xy")
        t = rec.get("t")
        if t is None or not isinstance(xy, (list, tuple)) or len(xy) < 2:
            continue
        source = str(rec.get("source") or "")
        if source and source != "bbox":
            out.append((float(t), float(xy[0]), float(xy[1])))
            continue
        if H is None:
            continue
        mapped = apply_h(H, float(xy[0]), float(xy[1]))
        if mapped is None:
            continue
        out.append((float(t), mapped[0], mapped[1]))
    return out


def too_far_on_map(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    points_a: list[tuple[float, float, float]],
    points_b: list[tuple[float, float, float]],
    max_map_m: float = DEFAULT_MAX_MAP_M,
) -> bool:
    """True, если при пересечении по времени ноги на плане дальше порога. Нет точек — не фильтруем."""
    if not points_a or not points_b:
        return False
    if not intervals_overlap(a["t0"], a["t1"], b["t0"], b["t1"]):
        return False
    lo = max(float(a["t0"]), float(b["t0"]))
    hi = min(float(a["t1"]), float(b["t1"]))
    pa = [p for p in points_a if lo <= p[0] <= hi] or points_a
    pb = [p for p in points_b if lo <= p[0] <= hi] or points_b
    best = None
    for _ta, xa, ya in pa:
        for _tb, xb, yb in pb:
            dist_m = float(np.hypot(xa - xb, ya - yb)) / METER_PX
            if best is None or dist_m < best:
                best = dist_m
    if best is None:
        return False
    return best > float(max_map_m)
