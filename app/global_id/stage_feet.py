"""Stage feet: проекция ног tracking.json → feet.json."""

from __future__ import annotations

import logging
import os
import statistics
from typing import Any

from app.artifact_meta import attach_artifact_meta
from app.config import (
    Settings,
    cameras_dir,
    feet_json_path,
    info_json_path,
    tracking_json_path,
    tracklet_frames_json_path,
    tracklets_json_path,
)
from app.io.json_util import load_tracking_json, save_debug_json
from app.global_id.calib_fingerprint import calib_fingerprint
from app.global_id.camera_pose import (
    estimate_height_from_kpts,
    image_feet_from_kpts,
    load_camera_doc,
    parse_camera_pose,
    ray_pair_stats,
    ray_to_ground_map,
)
from app.global_id.feet import apply_bbox_rel_offset, bbox_rel_offset, project_feet_to_map
from app.global_id.spatial import METER_PX, leave_one_out_h_rms, load_camera_h
from app.global_id.stage_pose import load_pose_lookup
from app.tracklet.map_coords import camera_key_for_settings

logger = logging.getLogger(__name__)


def _maps_dir(settings: Settings) -> str:
    return os.path.dirname(cameras_dir(settings))


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


def _pairs_from_doc(doc: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(doc, dict):
        return []
    raw = doc.get("pairs")
    if not isinstance(raw, list):
        return []
    return [p for p in raw if isinstance(p, dict)]


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _track_heights(
    pose_lookup: dict[int, dict[int, dict[str, Any]]],
    pose,
    image_size: tuple[int, int] | None,
    kpt_min: float,
    default_h: float,
) -> dict[int, float]:
    out: dict[int, float] = {}
    if pose is None or image_size is None:
        return out
    for tid, by_frame in pose_lookup.items():
        hs: list[float] = []
        for rec in by_frame.values():
            h = estimate_height_from_kpts(rec.get("kxy"), rec.get("kcf"), pose, image_size, kpt_min)
            if h is not None:
                hs.append(h)
        if hs:
            out[tid] = _median(hs)
        else:
            out[tid] = default_h
    return out


def _interp_offset(
    samples: list[tuple[int, float, float]],
    frame_index: int,
) -> tuple[float, float] | None:
    if not samples:
        return None
    samples = sorted(samples, key=lambda s: s[0])
    if frame_index <= samples[0][0]:
        if samples[0][0] == frame_index:
            return samples[0][1], samples[0][2]
        return None
    if frame_index >= samples[-1][0]:
        if samples[-1][0] == frame_index:
            return samples[-1][1], samples[-1][2]
        return None
    for i in range(1, len(samples)):
        f0, dx0, dy0 = samples[i - 1]
        f1, dx1, dy1 = samples[i]
        if f0 <= frame_index <= f1:
            if f1 == f0:
                return dx0, dy0
            t = (frame_index - f0) / (f1 - f0)
            return dx0 + (dx1 - dx0) * t, dy0 + (dy1 - dy0) * t
    return None


def smooth_track_xy(
    points: list[tuple[int, float, float]],
    *,
    fps: float,
    window: int,
    max_speed_mps: float,
    ema: float = 0.35,
) -> list[tuple[float, float]]:
    """Медиана по окну, отброс скачков быстрее max_speed, затем EMA. Короткие треки не трогаем."""
    n = len(points)
    if n < 3:
        return [(p[1], p[2]) for p in points]
    w = max(1, int(window))
    half = w // 2
    med: list[tuple[float, float]] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        xs = [points[j][1] for j in range(lo, hi)]
        ys = [points[j][2] for j in range(lo, hi)]
        med.append((_median(xs), _median(ys)))

    gated: list[tuple[float, float]] = [med[0]]
    fps_s = max(float(fps) or 25.0, 1e-6)
    for i in range(1, n):
        dt = abs(points[i][0] - points[i - 1][0]) / fps_s
        max_px = max_speed_mps * max(dt, 1e-6) * METER_PX
        dx = med[i][0] - gated[-1][0]
        dy = med[i][1] - gated[-1][1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > max_px:
            gated.append(gated[-1])
        else:
            gated.append(med[i])

    out: list[tuple[float, float]] = [gated[0]]
    alpha = min(1.0, max(0.05, float(ema)))
    for i in range(1, n):
        px, py = out[-1]
        cx, cy = gated[i]
        out.append((alpha * cx + (1.0 - alpha) * px, alpha * cy + (1.0 - alpha) * py))
    return out


def _enrich_tracklets_json(
    settings: Settings,
    raw_feet: dict[int, dict[int, dict[str, Any]]],
    pose_lookup: dict[int, dict[int, dict[str, Any]]],
    kpt_min: float,
) -> None:
    tl_path = tracklets_json_path(settings)
    if not os.path.isfile(tl_path):
        return
    tl_data = load_tracking_json(tl_path)
    tracklets = tl_data.get("tracklets") or []
    if not tracklets:
        return

    n_enriched = 0
    for t in tracklets:
        try:
            tid = int(t["tracklet_id"])
            f0 = int(t["f0"])
            f1 = int(t["f1"])
        except (KeyError, TypeError, ValueError):
            continue

        feet_by_f = raw_feet.get(tid, {})
        pose_by_f = pose_lookup.get(tid, {})

        # 1. Точные точки ног p0 и p1 по лодыжкам
        k0 = pose_by_f.get(f0)
        if k0 and k0.get("kxy"):
            ft0 = image_feet_from_kpts(k0["kxy"], k0.get("kcf"), kpt_min)
            if ft0 is not None:
                t["p0"] = [round(float(ft0[0]), 1), round(float(ft0[1]), 1)]
                t["kxy0"] = k0["kxy"]
                t["kcf0"] = k0.get("kcf")

        k1 = pose_by_f.get(f1)
        if k1 and k1.get("kxy"):
            ft1 = image_feet_from_kpts(k1["kxy"], k1.get("kcf"), kpt_min)
            if ft1 is not None:
                t["p1"] = [round(float(ft1[0]), 1), round(float(ft1[1]), 1)]
                t["kxy1"] = k1["kxy"]
                t["kcf1"] = k1.get("kcf")

        # 2. Map точки map_p0 и map_p1
        p0_map_rec = feet_by_f.get(f0)
        if p0_map_rec and p0_map_rec.get("map"):
            t["map_p0"] = p0_map_rec["map"]
            t["map_src0"] = p0_map_rec.get("source", "")
        p1_map_rec = feet_by_f.get(f1)
        if p1_map_rec and p1_map_rec.get("map"):
            t["map_p1"] = p1_map_rec["map"]
            t["map_src1"] = p1_map_rec.get("source", "")

        # 3. Оценка полноты позы (completeness)
        all_kcf = [rec["kcf"] for rec in pose_by_f.values() if rec.get("kcf")]
        if all_kcf:
            comps = [sum(1 for c in cfs if float(c) >= kpt_min) / max(1, len(cfs)) for cfs in all_kcf]
            t["completeness"] = round(float(statistics.median(comps)), 3)

        n_enriched += 1

    attach_artifact_meta(tl_data, stage="tracklets", path=tl_path)
    save_debug_json(tl_path, tl_data)
    logger.info("STAGE feet: обогащено %s треклетов в tracklets.json", n_enriched)


def run_feet(settings: Settings) -> None:
    track_path = tracking_json_path(settings)
    tl_frames_path = tracklet_frames_json_path(settings)
    if os.path.isfile(track_path):
        tracking = load_tracking_json(track_path)
    elif os.path.isfile(tl_frames_path):
        tracking = load_tracking_json(tl_frames_path)
    else:
        raise ValueError(
            f"Нет tracking JSON ({track_path}) и нет tracklet_frames JSON ({tl_frames_path}). "
            "Сначала --stage tracklets или --stage track"
        )

    frames_in = tracking.get("frames") or []
    tw = int(tracking.get("width") or 0)
    th = int(tracking.get("height") or 0)
    tracking_size: tuple[int, int] | None = (tw, th) if tw > 0 and th > 0 else None
    if tracking_size is None:
        info = load_tracking_json(info_json_path(settings)) if os.path.isfile(info_json_path(settings)) else {}
        iw, ih = int(info.get("width") or 0), int(info.get("height") or 0)
        if iw > 0 and ih > 0:
            tracking_size = (iw, ih)

    cam_dir = cameras_dir(settings)
    maps_dir = _maps_dir(settings)
    camera_key = camera_key_for_settings(settings)
    camera_doc = load_camera_doc(cam_dir, camera_key)
    H = load_camera_h(cam_dir, camera_key)
    torso = float(settings.feet_torso_height_m)
    person_h_default = float(settings.feet_person_height_m)
    kpt_min = float(settings.pose_kpt_min)
    fingerprint = calib_fingerprint(
        camera_key, camera_doc, torso, tracking_size, person_h_default
    )

    image_size = _image_size_from_doc(camera_doc)
    map_size = None
    if isinstance(camera_doc, dict):
        raw_ms = camera_doc.get("map_size")
        if isinstance(raw_ms, (list, tuple)) and len(raw_ms) >= 2:
            map_size = [int(raw_ms[0]), int(raw_ms[1])]

    pose = parse_camera_pose(camera_doc) if camera_doc else None
    pairs = _pairs_from_doc(camera_doc)
    h_loo = leave_one_out_h_rms(pairs)
    ray_rms = None
    if pose is not None and image_size is not None and len(pairs) >= 2:
        stats = ray_pair_stats(pose, pairs, image_size)
        if stats is not None:
            ray_rms = stats.rms_px

    work_dir = os.path.dirname(feet_json_path(settings))
    pose_lookup = load_pose_lookup(work_dir)
    heights = _track_heights(pose_lookup, pose, image_size, kpt_min, person_h_default)

    # Первый проход: keypoints / dual-plane, собираем смещения
    raw: dict[int, dict[int, dict[str, Any]]] = {}
    offsets: dict[int, list[tuple[int, float, float]]] = {}
    for fr in frames_in:
        try:
            frame_index = int(fr.get("frame_index"))
        except (TypeError, ValueError):
            continue
        for det in fr.get("detections") or []:
            if not isinstance(det, dict):
                continue
            raw_tid = det.get("track_id") if det.get("track_id") is not None else det.get("tracklet_id")
            try:
                tid = int(raw_tid)
            except (KeyError, TypeError, ValueError):
                continue
            kpt = pose_lookup.get(tid, {}).get(frame_index)
            person_h = heights.get(tid, person_h_default)
            got = project_feet_to_map(
                det.get("bbox"),
                camera_key=camera_key,
                maps_dir=maps_dir,
                camera_doc=camera_doc,
                H=H,
                torso_height_m=torso,
                person_height_m=person_h,
                tracking_size=tracking_size,
                kxy=kpt.get("kxy") if kpt else None,
                kcf=kpt.get("kcf") if kpt else None,
                kpt_min=kpt_min,
                h_loo_rms=h_loo,
                ray_rms=ray_rms,
            )
            if got is None:
                continue
            rec = {
                "track_id": tid,
                "map": [float(got.map_xy[0]), float(got.map_xy[1])],
                "source": str(got.source),
                "confidence": float(got.confidence),
                "bbox": det.get("bbox"),
            }
            raw.setdefault(tid, {})[frame_index] = rec
            if kpt and str(got.source).startswith("kpt"):
                img_ft = image_feet_from_kpts(kpt.get("kxy"), kpt.get("kcf"), kpt_min)
                if img_ft is not None:
                    off = bbox_rel_offset(det.get("bbox"), img_ft)
                    if off is not None:
                        offsets.setdefault(tid, []).append((frame_index, off[0], off[1]))

    # Второй проход: интерполяция смещений для кадров без позы
    for fr in frames_in:
        try:
            frame_index = int(fr.get("frame_index"))
        except (TypeError, ValueError):
            continue
        for det in fr.get("detections") or []:
            if not isinstance(det, dict):
                continue
            raw_tid = det.get("track_id") if det.get("track_id") is not None else det.get("tracklet_id")
            try:
                tid = int(raw_tid)
            except (KeyError, TypeError, ValueError):
                continue
            if frame_index in raw.get(tid, {}):
                continue
            kpt = pose_lookup.get(tid, {}).get(frame_index)
            if kpt:
                continue
            samples = offsets.get(tid)
            if not samples:
                continue
            off = _interp_offset(samples, frame_index)
            if off is None:
                continue
            img_pt = apply_bbox_rel_offset(det.get("bbox"), off[0], off[1])
            if img_pt is None:
                continue
            mapped = ray_to_ground_map(img_pt[0], img_pt[1], pose, image_size, torso_height_m=0.0)
            if mapped is None:
                continue
            rec = {
                "track_id": tid,
                "map": [float(mapped[0]), float(mapped[1])],
                "source": "kpt_interp",
                "confidence": 0.74,
                "bbox": det.get("bbox"),
            }
            raw.setdefault(tid, {})[frame_index] = rec

    # Сглаживание по трекам
    try:
        fps = float(tracking.get("fps") or 25.0)
    except (TypeError, ValueError):
        fps = 25.0
    window = int(settings.feet_smooth_window)
    max_speed = float(settings.feet_max_speed_mps)
    for tid, by_frame in raw.items():
        items = sorted(by_frame.items(), key=lambda kv: kv[0])
        pts = [(fi, rec["map"][0], rec["map"][1]) for fi, rec in items]
        smoothed = smooth_track_xy(pts, fps=fps, window=window, max_speed_mps=max_speed)
        for (fi, rec), (sx, sy) in zip(items, smoothed):
            rec["map"] = [round(float(sx), 2), round(float(sy), 2)]
            rec["confidence"] = round(float(rec["confidence"]), 3)

    frames_out: list[dict[str, Any]] = []
    n_points = 0
    for fr in frames_in:
        try:
            frame_index = int(fr.get("frame_index"))
        except (TypeError, ValueError):
            continue
        points: list[dict[str, Any]] = []
        for det in fr.get("detections") or []:
            if not isinstance(det, dict):
                continue
            raw_tid = det.get("track_id") if det.get("track_id") is not None else det.get("tracklet_id")
            try:
                tid = int(raw_tid)
            except (KeyError, TypeError, ValueError):
                continue
            rec = raw.get(tid, {}).get(frame_index)
            if rec is None:
                continue
            points.append(
                {
                    "track_id": tid,
                    "map": rec["map"],
                    "source": rec["source"],
                    "confidence": rec["confidence"],
                }
            )
            n_points += 1
        frames_out.append({"frame_index": frame_index, "points": points})

    payload: dict[str, Any] = {
        "stage": "feet",
        "camera_key": camera_key,
        "image_size": list(image_size) if image_size else None,
        "tracking_size": list(tracking_size) if tracking_size else None,
        "map_size": map_size,
        "torso_height_m": torso,
        "person_height_m": person_h_default,
        "calibration": {"fingerprint": fingerprint},
        "n_points": n_points,
        "frames": frames_out,
    }
    out_path = feet_json_path(settings)
    attach_artifact_meta(payload, stage="feet", path=out_path)
    save_debug_json(out_path, payload)
    logger.info("STAGE feet: %s точек, %s кадров → %s", n_points, len(frames_out), out_path)

    # Обогащаем tracklets.json точными точками ног и оценкой позы
    _enrich_tracklets_json(settings, raw, pose_lookup, kpt_min)


def load_feet_lookup(work_dir: str) -> tuple[dict[int, dict[int, dict[str, Any]]], dict[str, Any] | None]:
    """Индекс track_id → frame_index → {map, source, confidence} и сырой doc."""
    path = os.path.join(work_dir, "feet.json")
    if not os.path.isfile(path):
        return {}, None
    try:
        doc = load_tracking_json(path)
    except Exception:
        return {}, None
    if not isinstance(doc, dict):
        return {}, None
    index: dict[int, dict[int, dict[str, Any]]] = {}
    for fr in doc.get("frames") or []:
        if not isinstance(fr, dict):
            continue
        try:
            fi = int(fr.get("frame_index"))
        except (TypeError, ValueError):
            continue
        for p in fr.get("points") or []:
            if not isinstance(p, dict):
                continue
            try:
                tid = int(p["track_id"])
            except (KeyError, TypeError, ValueError):
                continue
            xy = p.get("map")
            if not isinstance(xy, (list, tuple)) or len(xy) < 2:
                continue
            index.setdefault(tid, {})[fi] = {
                "xy": [float(xy[0]), float(xy[1])],
                "source": str(p.get("source") or "ray"),
                "confidence": float(p.get("confidence") or 0.85),
            }
    return index, doc
