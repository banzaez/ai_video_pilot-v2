"""Map-координаты концов треклетов для склейки."""

from __future__ import annotations

import os
from typing import Any

from app.config import Settings, cameras_dir, info_json_path
from app.global_id.camera_pose import load_camera_doc, parse_camera_pose, ray_to_ground_map
from app.global_id.feet import MapFeetResult, project_feet_to_map
from app.global_id.spatial import apply_h, load_camera_h
from app.info import resolve_camera_key
from app.io.json_util import load_tracking_json


def _default_maps_dir(settings: Settings) -> str:
    cam_dir = cameras_dir(settings)
    return os.path.dirname(cam_dir)


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


def camera_key_for_settings(settings: Settings) -> str:
    info_path = info_json_path(settings)
    info = load_tracking_json(info_path) if os.path.isfile(info_path) else {}
    video_source = str((info or {}).get("video_source") or settings.input_path or "")
    stem = os.path.splitext(os.path.basename(video_source))[0] if video_source else None
    return resolve_camera_key(info if isinstance(info, dict) else None, stem=stem)


def is_kpt_map_source(source: str | None) -> bool:
    return str(source or "").startswith("kpt")


def accept_map_for_link(got: MapFeetResult | None, *, has_kpts: bool) -> MapFeetResult | None:
    """Bbox/dual-plane не должны перебивать ноги с позы: линкер всегда предпочитает map_*."""
    if got is None:
        return None
    if has_kpts and not is_kpt_map_source(got.source):
        return None
    return got


def _scale_xy(
    xy: Any,
    src_size: tuple[int, int] | None,
    dst_size: tuple[int, int] | None,
) -> tuple[float, float] | None:
    if not isinstance(xy, (list, tuple)) or len(xy) < 2:
        return None
    try:
        x, y = float(xy[0]), float(xy[1])
    except (TypeError, ValueError):
        return None
    if not src_size or not dst_size:
        return x, y
    sw, sh = int(src_size[0]), int(src_size[1])
    dw, dh = int(dst_size[0]), int(dst_size[1])
    if sw <= 0 or sh <= 0 or (sw == dw and sh == dh):
        return x, y
    return x * (dw / sw), y * (dh / sh)


def _scale_kxy(
    kxy: list[Any],
    src_size: tuple[int, int] | None,
    dst_size: tuple[int, int] | None,
) -> list[Any]:
    out: list[Any] = []
    for pt in kxy:
        scaled = _scale_xy(pt, src_size, dst_size)
        if scaled is None:
            out.append(pt)
        else:
            out.append([scaled[0], scaled[1]])
    return out


def project_image_xy_to_map(
    xy: Any,
    *,
    pose: Any,
    image_size: tuple[int, int] | None,
    H: Any,
    tracking_size: tuple[int, int] | None,
) -> MapFeetResult | None:
    """Проекция уже уточнённой image-точки ног (лодыжки) на план."""
    scaled = _scale_xy(xy, tracking_size, image_size)
    if scaled is None:
        return None
    x, y = scaled
    if pose is not None and image_size is not None:
        mapped = ray_to_ground_map(x, y, pose, image_size, torso_height_m=0.0)
        if mapped is not None:
            return MapFeetResult(map_xy=mapped, source="kpt_img", confidence=0.88)
    if H is not None:
        mapped = apply_h(H, x, y)
        if mapped is not None:
            return MapFeetResult(map_xy=mapped, source="kpt_h", confidence=0.80)
    return None


def _write_map_end(tracklet: dict[str, Any], suffix: str, got: MapFeetResult | None) -> bool:
    pt_key = f"map_p{suffix}"
    src_key = f"map_src{suffix}"
    if got is None:
        tracklet.pop(pt_key, None)
        tracklet.pop(src_key, None)
        return False
    tracklet[pt_key] = [round(got.map_xy[0], 2), round(got.map_xy[1], 2)]
    tracklet[src_key] = str(got.source)
    return True


def enrich_tracklets_map_coords(
    tracklets: list[dict[str, Any]],
    *,
    settings: Settings,
    torso_height_m: float,
    person_height_m: float = 1.70,
    kpt_min: float = 0.25,
) -> int:
    """Добавляет map_p0/map_p1 в tracklets in-place. Возвращает число треклетов с map-точками.

    Если на конце есть pose-keypoints, map берётся только из них (или из проекции
    ankle-p0/p1). Низ bbox на план не пишется — иначе линкер проигнорирует точные ноги.
    """
    maps_dir = _default_maps_dir(settings)
    camera_key = camera_key_for_settings(settings)
    cam_dir = cameras_dir(settings)
    camera_doc = load_camera_doc(cam_dir, camera_key) if camera_key else None
    pose = parse_camera_pose(camera_doc) if camera_doc else None
    image_size = _image_size_from_doc(camera_doc)
    H = load_camera_h(cam_dir, camera_key) if camera_key else None

    tracking_size: tuple[int, int] | None = None
    info_path = info_json_path(settings)
    if os.path.isfile(info_path):
        info = load_tracking_json(info_path)
        w, h = int(info.get("width") or 0), int(info.get("height") or 0)
        if w > 0 and h > 0:
            tracking_size = (w, h)
    n = 0
    for t in tracklets:
        ok = False
        for bbox_key, suffix in (("bbox0", "0"), ("bbox1", "1")):
            bbox = t.get(bbox_key)
            if not bbox:
                continue
            kxy = t.get(f"kxy{suffix}")
            kcf = t.get(f"kcf{suffix}")
            has_kpts = isinstance(kxy, list) and isinstance(kcf, list)
            kxy_scaled = _scale_kxy(kxy, tracking_size, image_size) if has_kpts else None
            got = project_feet_to_map(
                bbox,
                camera_key=camera_key,
                maps_dir=maps_dir,
                camera_doc=camera_doc,
                H=H,
                torso_height_m=torso_height_m,
                person_height_m=person_height_m,
                tracking_size=tracking_size,
                kxy=kxy_scaled,
                kcf=kcf if has_kpts else None,
                kpt_min=kpt_min,
            )
            got = accept_map_for_link(got, has_kpts=has_kpts)
            if got is None and has_kpts:
                got = project_image_xy_to_map(
                    t.get(f"p{suffix}"),
                    pose=pose,
                    image_size=image_size,
                    H=H,
                    tracking_size=tracking_size,
                )
            if _write_map_end(t, suffix, got):
                ok = True
        if ok:
            n += 1
    return n
