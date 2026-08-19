"""3D-поза камеры: луч через пиксель → пол (ground plane)."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from app.global_id.spatial import METER_PX, apply_h, prefer_homography_over_ray

FLOOR_ORIGIN = (120.0, 120.0)
DEFAULT_HEIGHT_M = 3.0
DEFAULT_PITCH_DEG = 35.0
# z=0 — пол; калибровка пар и проекция ног используют одну плоскость
DEFAULT_TORSO_H_M = 0.0
DEFAULT_PERSON_H_M = 1.70
MISS_PENALTY_PX = 5000.0
KPT_MIN_DEFAULT = 0.25
COCO_NOSE, COCO_L_HIP, COCO_R_HIP, COCO_L_ANKLE, COCO_R_ANKLE = 0, 11, 12, 15, 16
Z_FRAC_ANKLE, Z_FRAC_HIP, Z_FRAC_NOSE = 0.03, 0.55, 0.94
TRUNC_ABS_M = 0.4
TRUNC_REL = 0.15

_FIT_H_MIN, _FIT_H_MAX = 1.0, 6.0
_FIT_PITCH_MIN, _FIT_PITCH_MAX = 0.0, 85.0
_FIT_FOV_MIN, _FIT_FOV_MAX = 40.0, 140.0
_FIT_YAW_SPAN = 20.0
_FIT_POS_SPAN = 600.0
_FIT_MAX_ITER = 200


@dataclass
class CameraPose:
    position: tuple[float, float]
    yaw_deg: float
    fov_deg: float
    height_m: float = DEFAULT_HEIGHT_M
    pitch_deg: float = DEFAULT_PITCH_DEG


@dataclass
class RayPairStats:
    rms_px: float
    projected: int
    total: int


@dataclass
class FitRayPoseResult:
    height_m: float
    pitch_deg: float
    fov_deg: float
    yaw_deg: float
    position: tuple[float, float]
    rms_px: float
    projected: int
    total: int


def _map_px_to_meters(map_x: float, map_y: float) -> tuple[float, float]:
    ox, oy = FLOOR_ORIGIN
    return (map_x - ox) / METER_PX, (map_y - oy) / METER_PX


def _meters_to_map_px(mx: float, my: float) -> tuple[float, float]:
    ox, oy = FLOOR_ORIGIN
    return ox + mx * METER_PX, oy + my * METER_PX


def pinhole_k(image_size: tuple[int, int], fov_deg: float) -> tuple[float, float, float, float]:
    w, h = image_size
    fov = math.radians(max(20.0, min(160.0, float(fov_deg))))
    fx = (w / 2.0) / math.tan(fov / 2.0)
    fy = fx
    return fx, fy, w / 2.0, h / 2.0


def _look_basis(yaw_deg: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    fwd_x = math.cos(yaw)
    fwd_y = math.sin(yaw)
    look = np.array(
        [math.cos(pitch) * fwd_x, math.cos(pitch) * fwd_y, -math.sin(pitch)],
        dtype=np.float64,
    )
    right = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float64)
    down = np.cross(right, look)
    dn = float(np.linalg.norm(down))
    if dn > 1e-9:
        down /= dn
    return look, right, down


def parse_camera_pose(raw: dict[str, Any] | None) -> CameraPose | None:
    if not isinstance(raw, dict):
        return None
    pl = raw.get("placement")
    if not isinstance(pl, dict):
        return None
    pos = pl.get("position")
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return None
    try:
        x, y = float(pos[0]), float(pos[1])
        yaw = float(pl.get("yaw_deg") or 0.0)
        fov = float(pl.get("fov_deg") or 70.0)
        height = float(pl.get("height_m") if pl.get("height_m") is not None else DEFAULT_HEIGHT_M)
        pitch = float(pl.get("pitch_deg") if pl.get("pitch_deg") is not None else DEFAULT_PITCH_DEG)
    except (TypeError, ValueError):
        return None
    return CameraPose(
        position=(x, y),
        yaw_deg=((yaw % 360) + 360) % 360,
        fov_deg=max(20.0, min(160.0, fov)),
        height_m=max(0.5, height),
        pitch_deg=max(0.0, min(89.0, pitch)),
    )


def load_camera_doc(cameras_dir: str, camera_key: str) -> dict[str, Any] | None:
    candidates = [f"{camera_key}.json"]
    try:
        idx = int(camera_key)
        candidates.append(f"{idx:03d}.json")
        candidates.append(f"{idx:02d}.json")
        candidates.append(f"{idx}.json")
    except ValueError:
        pass

    for name in dict.fromkeys(candidates):
        path = os.path.join(cameras_dir, name)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else None
            except Exception:
                continue
    return None


def load_camera_pose(cameras_dir: str, camera_key: str) -> CameraPose | None:
    doc = load_camera_doc(cameras_dir, camera_key)
    return parse_camera_pose(doc)


def ray_to_ground_map(
    px: float,
    py: float,
    pose: CameraPose,
    image_size: tuple[int, int],
    *,
    torso_height_m: float = DEFAULT_TORSO_H_M,
) -> tuple[float, float] | None:
    fx, fy, cx, cy = pinhole_k(image_size, pose.fov_deg)
    cam_mx, cam_my = _map_px_to_meters(pose.position[0], pose.position[1])
    cam_z = pose.height_m
    look, right, down = _look_basis(pose.yaw_deg, pose.pitch_deg)
    u = (float(px) - cx) / fx
    v = (float(py) - cy) / fy
    direction = look + u * right + v * down
    dn = float(np.linalg.norm(direction))
    if dn < 1e-9:
        return None
    direction /= dn
    if direction[2] >= -1e-6:
        return None
    t_torso = (torso_height_m - cam_z) / direction[2]
    if t_torso <= 0:
        return None
    tx = cam_mx + t_torso * float(direction[0])
    ty = cam_my + t_torso * float(direction[1])
    mx, my = _meters_to_map_px(tx, ty)
    return float(mx), float(my)


def ray_pair_stats(
    pose: CameraPose,
    pairs: list[dict[str, Any]],
    image_size: tuple[int, int],
) -> RayPairStats | None:
    if not pairs or image_size[0] <= 0 or image_size[1] <= 0:
        return None
    total = 0
    projected = 0
    sum_sq = 0.0
    for pair in pairs:
        img = pair.get("image")
        mp = pair.get("map")
        if not isinstance(img, (list, tuple)) or not isinstance(mp, (list, tuple)):
            continue
        total += 1
        mapped = ray_to_ground_map(
            float(img[0]),
            float(img[1]),
            pose,
            image_size,
            torso_height_m=0.0,
        )
        if mapped is None:
            err = MISS_PENALTY_PX
        else:
            projected += 1
            err = float(math.hypot(mapped[0] - float(mp[0]), mapped[1] - float(mp[1])))
        sum_sq += err * err
    if total < 2:
        return None
    return RayPairStats(rms_px=math.sqrt(sum_sq / total), projected=projected, total=total)


def _wrap_deg(deg: float) -> float:
    return ((deg % 360.0) + 360.0) % 360.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _yaw_delta(a: float, b: float) -> float:
    d = ((a - b) % 360.0 + 360.0) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def _clamp_pose(trial: CameraPose, origin: CameraPose, *, fit_pose: bool) -> CameraPose:
    yaw = origin.yaw_deg
    pos = origin.position
    if fit_pose:
        d_yaw = _clamp(_yaw_delta(trial.yaw_deg, origin.yaw_deg), -_FIT_YAW_SPAN, _FIT_YAW_SPAN)
        yaw = _wrap_deg(origin.yaw_deg + d_yaw)
        pos = (
            _clamp(trial.position[0], origin.position[0] - _FIT_POS_SPAN, origin.position[0] + _FIT_POS_SPAN),
            _clamp(trial.position[1], origin.position[1] - _FIT_POS_SPAN, origin.position[1] + _FIT_POS_SPAN),
        )
    return CameraPose(
        position=pos,
        yaw_deg=yaw,
        fov_deg=_clamp(trial.fov_deg, _FIT_FOV_MIN, _FIT_FOV_MAX),
        height_m=_clamp(trial.height_m, _FIT_H_MIN, _FIT_H_MAX),
        pitch_deg=_clamp(trial.pitch_deg, _FIT_PITCH_MIN, _FIT_PITCH_MAX),
    )


def _nudge(pose: CameraPose, axis: str, delta: float) -> CameraPose:
    if axis == "height_m":
        return replace(pose, height_m=pose.height_m + delta)
    if axis == "pitch_deg":
        return replace(pose, pitch_deg=pose.pitch_deg + delta)
    if axis == "fov_deg":
        return replace(pose, fov_deg=pose.fov_deg + delta)
    if axis == "yaw_deg":
        return replace(pose, yaw_deg=_wrap_deg(pose.yaw_deg + delta))
    if axis == "x":
        return replace(pose, position=(pose.position[0] + delta, pose.position[1]))
    if axis == "y":
        return replace(pose, position=(pose.position[0], pose.position[1] + delta))
    return pose


def fit_ray_pose(
    pose: CameraPose,
    pairs: list[dict[str, Any]],
    image_size: tuple[int, int],
    *,
    fit_pose: bool = True,
) -> FitRayPoseResult | None:
    """Подбор height/pitch/FOV (и опционально yaw/position) по парам калибровки."""
    if not pairs or image_size[0] <= 0 or image_size[1] <= 0:
        return None
    origin = pose
    best = pose
    best_stats = ray_pair_stats(best, pairs, image_size)
    if best_stats is None:
        return None

    for fov in range(60, 121, 5):
        for hi in range(3, 11):  # 1.5 … 5.0 шаг 0.5
            h = hi * 0.5
            for p in range(10, 76, 5):
                trial = CameraPose(
                    position=origin.position,
                    yaw_deg=origin.yaw_deg,
                    fov_deg=float(fov),
                    height_m=float(h),
                    pitch_deg=float(p),
                )
                stats = ray_pair_stats(trial, pairs, image_size)
                if stats is None:
                    continue
                if stats.rms_px < best_stats.rms_px:
                    best = trial
                    best_stats = stats

    axes = ["height_m", "pitch_deg", "fov_deg"]
    steps: dict[str, float] = {"height_m": 0.4, "pitch_deg": 4.0, "fov_deg": 4.0}
    tols: dict[str, float] = {"height_m": 0.02, "pitch_deg": 0.25, "fov_deg": 0.5}
    if fit_pose:
        axes.extend(["yaw_deg", "x", "y"])
        steps.update({"yaw_deg": 4.0, "x": 160.0, "y": 160.0})
        tols.update({"yaw_deg": 0.25, "x": 10.0, "y": 10.0})

    for _ in range(_FIT_MAX_ITER):
        if all(steps[a] <= tols[a] for a in axes):
            break
        improved = False
        for axis in axes:
            for sign in (1.0, -1.0):
                trial = _clamp_pose(_nudge(best, axis, sign * steps[axis]), origin, fit_pose=fit_pose)
                stats = ray_pair_stats(trial, pairs, image_size)
                if stats is None:
                    continue
                if stats.rms_px < best_stats.rms_px:
                    best = trial
                    best_stats = stats
                    improved = True
        if not improved:
            for a in axes:
                steps[a] *= 0.5

    return FitRayPoseResult(
        height_m=best.height_m,
        pitch_deg=best.pitch_deg,
        fov_deg=best.fov_deg,
        yaw_deg=best.yaw_deg,
        position=best.position,
        rms_px=best_stats.rms_px,
        projected=best_stats.projected,
        total=best_stats.total,
    )


def ray_loo_rms(
    pose: CameraPose,
    pairs: list[dict[str, Any]],
    image_size: tuple[int, int],
) -> float | None:
    """Leave-one-out RMS 3D-луча: на каждой итерации fit по остальным парам."""
    n = len(pairs)
    if n < 6:
        return None
    errs: list[float] = []
    for i in range(n):
        rest = [p for j, p in enumerate(pairs) if j != i]
        fit = fit_ray_pose(pose, rest, image_size, fit_pose=True)
        if fit is None:
            continue
        trial = CameraPose(
            position=fit.position,
            yaw_deg=fit.yaw_deg,
            fov_deg=fit.fov_deg,
            height_m=fit.height_m,
            pitch_deg=fit.pitch_deg,
        )
        pair = pairs[i]
        img = pair.get("image") if isinstance(pair, dict) else None
        mp = pair.get("map") if isinstance(pair, dict) else None
        if not isinstance(img, (list, tuple)) or not isinstance(mp, (list, tuple)):
            continue
        mapped = ray_to_ground_map(float(img[0]), float(img[1]), trial, image_size, torso_height_m=0.0)
        if mapped is None:
            continue
        errs.append(float(math.hypot(mapped[0] - float(mp[0]), mapped[1] - float(mp[1]))))
    if len(errs) < 2:
        return None
    return math.sqrt(sum(e * e for e in errs) / len(errs))


def _bbox_xy(bbox: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return None
    return x1, y1, x2, y2


def cam_dist_m(map_xy: tuple[float, float], pose: CameraPose) -> float:
    return math.hypot(map_xy[0] - pose.position[0], map_xy[1] - pose.position[1]) / METER_PX


def is_truncated_dual(
    p_head: tuple[float, float] | None,
    p_feet: tuple[float, float] | None,
    pose: CameraPose,
) -> bool:
    if p_head is None:
        return False
    if p_feet is None:
        return True
    d_head = cam_dist_m(p_head, pose)
    d_feet = cam_dist_m(p_feet, pose)
    return d_feet > d_head + max(TRUNC_ABS_M, TRUNC_REL * d_head)


def dual_plane_from_bbox(
    bbox: Any,
    pose: CameraPose,
    image_size: tuple[int, int],
    person_height_m: float = DEFAULT_PERSON_H_M,
) -> tuple[tuple[float, float] | None, str, bool, tuple[float, float] | None, tuple[float, float] | None]:
    """Два луча: верх bbox на z=рост, низ на z=0. Возвращает (map, source, truncated, P_head, P_feet)."""
    xy = _bbox_xy(bbox)
    if xy is None:
        return None, "ray", False, None, None
    x1, y1, x2, y2 = xy
    x_mid = (x1 + x2) * 0.5
    y_top, y_bot = min(y1, y2), max(y1, y2)
    p_head = ray_to_ground_map(x_mid, y_top, pose, image_size, torso_height_m=float(person_height_m))
    p_feet = ray_to_ground_map(x_mid, y_bot, pose, image_size, torso_height_m=0.0)
    truncated = is_truncated_dual(p_head, p_feet, pose)
    if truncated and p_head is not None:
        return p_head, "ray_head", True, p_head, p_feet
    if p_feet is not None:
        return p_feet, "ray_feet", truncated, p_head, p_feet
    if p_head is not None:
        return p_head, "ray_head", truncated, p_head, p_feet
    return None, "ray", truncated, p_head, p_feet


def agree_confidence(
    p_head: tuple[float, float] | None,
    p_feet: tuple[float, float] | None,
    pose: CameraPose,
) -> float:
    if p_head is None or p_feet is None:
        return 0.55
    d_head = max(cam_dist_m(p_head, pose), 1.0)
    d_feet = cam_dist_m(p_feet, pose)
    agree = 1.0 - min(1.0, abs(d_feet - d_head) / d_head)
    return max(0.35, min(1.0, agree))


def _ray_direction(px: float, py: float, pose: CameraPose, image_size: tuple[int, int]) -> np.ndarray | None:
    fx, fy, cx, cy = pinhole_k(image_size, pose.fov_deg)
    look, right, down = _look_basis(pose.yaw_deg, pose.pitch_deg)
    u = (float(px) - cx) / fx
    v = (float(py) - cy) / fy
    direction = look + u * right + v * down
    dn = float(np.linalg.norm(direction))
    if dn < 1e-9:
        return None
    return direction / dn


def estimate_height_from_kpts(
    kxy: list[list[float]] | None,
    kcf: list[float] | None,
    pose: CameraPose,
    image_size: tuple[int, int],
    kpt_min: float = KPT_MIN_DEFAULT,
) -> float | None:
    """Рост по носу и двум лодыжкам: z носа над точкой лодыжек / 0.94."""
    if not kxy or not kcf:
        return None
    if len(kxy) <= COCO_R_ANKLE or len(kcf) <= COCO_R_ANKLE:
        return None
    if float(kcf[COCO_NOSE]) < kpt_min:
        return None
    if float(kcf[COCO_L_ANKLE]) < kpt_min or float(kcf[COCO_R_ANKLE]) < kpt_min:
        return None
    ax = (float(kxy[COCO_L_ANKLE][0]) + float(kxy[COCO_R_ANKLE][0])) * 0.5
    ay = (float(kxy[COCO_L_ANKLE][1]) + float(kxy[COCO_R_ANKLE][1])) * 0.5
    p_feet = ray_to_ground_map(ax, ay, pose, image_size, torso_height_m=0.0)
    if p_feet is None:
        return None
    feet_m = _map_px_to_meters(p_feet[0], p_feet[1])
    cam_m = _map_px_to_meters(pose.position[0], pose.position[1])
    direction = _ray_direction(float(kxy[COCO_NOSE][0]), float(kxy[COCO_NOSE][1]), pose, image_size)
    if direction is None:
        return None
    dxy = direction[:2]
    rel = np.array(feet_m, dtype=np.float64) - np.array(cam_m, dtype=np.float64)
    denom = float(dxy @ dxy)
    if denom < 1e-9:
        return None
    t = float(rel @ dxy) / denom
    if t <= 0:
        return None
    z_nose = pose.height_m + t * float(direction[2])
    height = z_nose / Z_FRAC_NOSE
    if height < 1.2 or height > 2.2:
        return None
    return float(height)


def image_feet_from_kpts(
    kxy: list[list[float]] | None,
    kcf: list[float] | None,
    kpt_min: float = KPT_MIN_DEFAULT,
) -> tuple[float, float] | None:
    if not kxy or not kcf:
        return None
    ankles: list[tuple[float, float]] = []
    for idx in (COCO_L_ANKLE, COCO_R_ANKLE):
        if idx < len(kxy) and idx < len(kcf) and float(kcf[idx]) >= kpt_min:
            ankles.append((float(kxy[idx][0]), float(kxy[idx][1])))
    if len(ankles) == 2:
        return (ankles[0][0] + ankles[1][0]) * 0.5, (ankles[0][1] + ankles[1][1]) * 0.5
    if len(ankles) == 1:
        return ankles[0]
    hips: list[tuple[float, float]] = []
    for idx in (COCO_L_HIP, COCO_R_HIP):
        if idx < len(kxy) and idx < len(kcf) and float(kcf[idx]) >= kpt_min:
            hips.append((float(kxy[idx][0]), float(kxy[idx][1])))
    if len(hips) == 2:
        return (hips[0][0] + hips[1][0]) * 0.5, (hips[0][1] + hips[1][1]) * 0.5
    if len(hips) == 1:
        return hips[0]
    return None


def project_keypoints_to_map(
    kxy: list[list[float]] | None,
    kcf: list[float] | None,
    pose: CameraPose,
    image_size: tuple[int, int],
    person_height_m: float = DEFAULT_PERSON_H_M,
    kpt_min: float = KPT_MIN_DEFAULT,
) -> tuple[tuple[float, float], str, float] | None:
    """Взвешенное среднее проекций уверенных keypoints на пол."""
    if not kxy or not kcf:
        return None
    samples: list[tuple[tuple[float, float], float, str]] = []

    def _add(idx: int, z_frac: float, kind: str) -> None:
        if idx >= len(kxy) or idx >= len(kcf):
            return
        cf = float(kcf[idx])
        if cf < kpt_min:
            return
        pt = kxy[idx]
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            return
        mapped = ray_to_ground_map(
            float(pt[0]),
            float(pt[1]),
            pose,
            image_size,
            torso_height_m=z_frac * person_height_m,
        )
        if mapped is not None:
            samples.append((mapped, cf, kind))

    _add(COCO_L_ANKLE, Z_FRAC_ANKLE, "ankle")
    _add(COCO_R_ANKLE, Z_FRAC_ANKLE, "ankle")
    _add(COCO_L_HIP, Z_FRAC_HIP, "hip")
    _add(COCO_R_HIP, Z_FRAC_HIP, "hip")
    _add(COCO_NOSE, Z_FRAC_NOSE, "nose")
    if not samples:
        return None
    wsum = sum(s[1] for s in samples)
    if wsum <= 0:
        return None
    mx = sum(s[0][0] * s[1] for s in samples) / wsum
    my = sum(s[0][1] * s[1] for s in samples) / wsum
    kinds = {s[2] for s in samples}
    n_ankle = sum(1 for s in samples if s[2] == "ankle")
    if n_ankle >= 2 and len(kinds) > 1:
        source = "kpt_lsq"
        base = 0.92
    elif n_ankle >= 1:
        source = "kpt_ankle"
        base = 0.88
    elif "hip" in kinds:
        source = "kpt_hip"
        base = 0.78
    else:
        source = "kpt_head"
        base = 0.72
    return (float(mx), float(my)), source, base


def project_bbox_feet_to_map(
    bbox: Any,
    *,
    pose: CameraPose | None,
    image_size: tuple[int, int] | None,
    H: np.ndarray | None,
    torso_height_m: float = DEFAULT_TORSO_H_M,
    person_height_m: float = DEFAULT_PERSON_H_M,
    h_loo_rms: float | None = None,
    ray_rms: float | None = None,
    kxy: list[list[float]] | None = None,
    kcf: list[float] | None = None,
    kpt_min: float = KPT_MIN_DEFAULT,
) -> tuple[tuple[float, float], str, float] | None:
    """Приоритет: keypoints → dual-plane; H только для полного bbox при честном LOO."""
    from app.global_id.feet import feet_from_bbox

    size = image_size if image_size and image_size[0] > 0 and image_size[1] > 0 else None
    if pose is not None and size is not None and kxy and kcf:
        kpt = project_keypoints_to_map(kxy, kcf, pose, size, person_height_m, kpt_min)
        if kpt is not None:
            return kpt

    if pose is not None and size is not None:
        mapped, source, truncated, p_head, p_feet = dual_plane_from_bbox(
            bbox, pose, size, person_height_m=person_height_m
        )
        agree = agree_confidence(p_head, p_feet, pose)
        use_h = (
            not truncated
            and H is not None
            and prefer_homography_over_ray(h_loo_rms, ray_rms)
        )
        if use_h:
            feet = feet_from_bbox(bbox)
            if feet is not None:
                h_mapped = apply_h(H, feet[0], feet[1])
                if h_mapped is not None:
                    return h_mapped, "h_bbox", 0.55 * agree
        if mapped is not None:
            base = 0.85 if source == "ray_feet" else 0.72
            return mapped, source, base * agree
        if torso_height_m and abs(torso_height_m) > 1e-9:
            feet = feet_from_bbox(bbox)
            if feet is not None:
                legacy = ray_to_ground_map(feet[0], feet[1], pose, size, torso_height_m=torso_height_m)
                if legacy is not None:
                    return legacy, "ray", 0.85

    feet = feet_from_bbox(bbox)
    if feet is None:
        return None
    if H is not None:
        mapped = apply_h(H, feet[0], feet[1])
        if mapped is not None:
            return mapped, "h_bbox", 0.55
    return None


# Обратная совместимость со старым именем
project_bbox_center_to_map = project_bbox_feet_to_map
