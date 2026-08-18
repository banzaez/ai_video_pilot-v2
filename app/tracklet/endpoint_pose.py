"""YOLO-pose на концах треклетов (f0/f1) для склейки."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any

from app.config import Settings, info_json_path
from app.global_id.camera_pose import image_feet_from_kpts
from app.io.json_util import load_tracking_json
from app.model_cache import get_model_cache, predict_batch_size, resolve_pt_path
from app.pose.types import select_pose_index_by_completeness
from app.progress import make_pbar
from app.tracklet.common import session_manifest
from app.tracklet.stage_reid import _advance_capture, _resolve_video_path

logger = logging.getLogger(__name__)

_MIN_IOU = 0.3


def apply_endpoint_kpts(
    tracklet: dict[str, Any],
    end: str,
    kxy: list[list[float]],
    kcf: list[float],
    *,
    kpt_min: float,
) -> bool:
    """Пишет kxy/kcf и уточняет p0/p1 по лодыжкам. end: '0' | '1'."""
    if end not in ("0", "1"):
        raise ValueError(end)
    tracklet[f"kxy{end}"] = kxy
    tracklet[f"kcf{end}"] = kcf
    feet = image_feet_from_kpts(kxy, kcf, kpt_min)
    if feet is None:
        return False
    tracklet[f"p{end}"] = [round(float(feet[0]), 1), round(float(feet[1]), 1)]
    return True


def match_pose_to_bbox(
    bbox: list[float],
    instances: list[dict[str, Any]],
    used: set[int],
    *,
    min_iou: float = _MIN_IOU,
    kpt_min: float = 0.25,
) -> dict[str, Any] | None:
    rows = [
        (
            inst.get("bbox") or [],
            inst.get("kcf") or [],
            float(inst.get("confidence") or 0.0),
        )
        for inst in instances
    ]
    best_i = select_pose_index_by_completeness(
        rows,
        bbox,
        min_iou=min_iou,
        kpt_min=kpt_min,
        skip=used,
    )
    if best_i is None:
        return None
    used.add(best_i)
    return instances[best_i]


def _endpoint_jobs(
    tracklets: list[dict[str, Any]],
) -> dict[int, list[tuple[dict[str, Any], str, list[float]]]]:
    """frame 0-based → [(tracklet, end, bbox), ...]. Если f0==f1 — только конец 0."""
    by_frame: dict[int, list[tuple[dict[str, Any], str, list[float]]]] = defaultdict(list)
    for t in tracklets:
        try:
            f0 = int(t["f0"]) - 1
            f1 = int(t["f1"]) - 1
        except (KeyError, TypeError, ValueError):
            continue
        b0 = t.get("bbox0")
        b1 = t.get("bbox1")
        if isinstance(b0, (list, tuple)) and len(b0) >= 4 and f0 >= 0:
            by_frame[f0].append((t, "0", [float(v) for v in b0[:4]]))
        if f1 != f0 and isinstance(b1, (list, tuple)) and len(b1) >= 4 and f1 >= 0:
            by_frame[f1].append((t, "1", [float(v) for v in b1[:4]]))
    return by_frame


def _copy_same_frame_ends(tracklets: list[dict[str, Any]]) -> None:
    for t in tracklets:
        try:
            if int(t["f0"]) != int(t["f1"]):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        if "kxy0" in t and "kxy1" not in t:
            t["kxy1"] = t["kxy0"]
            t["kcf1"] = t.get("kcf0")
            if "p0" in t:
                t["p1"] = list(t["p0"])


def enrich_tracklet_endpoint_pose(tracklets: list[dict[str, Any]], settings: Settings) -> int:
    """Pose на первом/последнем кадре треклета. Возвращает число концов с keypoints."""
    by_frame = _endpoint_jobs(tracklets)
    if not by_frame:
        return 0

    info: dict[str, Any] = {}
    info_path = info_json_path(settings)
    if os.path.isfile(info_path):
        info = load_tracking_json(info_path) or {}

    manifest = session_manifest(settings)
    video_path = None if manifest else _resolve_video_path(settings, info)
    if not manifest and not video_path:
        logger.warning("STAGE 2c: нет видео для pose на концах треклетов")
        return 0

    from app.pose import get_pose_service

    try:
        pose_service = get_pose_service(settings)
        infer_batch = pose_service.effective_batch_size()
    except Exception as exc:
        logger.warning("STAGE 2c: pose на концах пропущен: %s", exc)
        return 0

    kpt_min = float(settings.pose_kpt_min)
    n_hit = 0
    frame_ids = sorted(by_frame)

    def _flush(batch_frames: list, batch_idx: list[int]) -> None:
        nonlocal n_hit
        if not batch_frames:
            return
        batch_pose_results = pose_service.predict_batch(batch_frames)
        for fi, pose_items in zip(batch_idx, batch_pose_results):
            inst = [
                {
                    "bbox": r.bbox,
                    "confidence": r.confidence,
                    "kxy": r.kxy,
                    "kcf": r.kcf,
                }
                for r in pose_items
            ]
            used: set[int] = set()
            for tracklet, end, bbox in by_frame[fi]:
                hit = match_pose_to_bbox(bbox, inst, used, kpt_min=kpt_min)
                if hit is None:
                    continue
                apply_endpoint_kpts(tracklet, end, hit["kxy"], hit["kcf"], kpt_min=kpt_min)
                n_hit += 1

    pbar = make_pbar(total=max(1, len(frame_ids)), desc="[STAGE 2c: pose ends]", unit="frame")
    try:
        batch_frames: list = []
        batch_idx: list[int] = []
        if manifest:
            from app.session.reader import SessionFrameReader

            with SessionFrameReader(manifest) as reader:
                for fi in frame_ids:
                    batch_frames.append(reader.read_frame(fi))
                    batch_idx.append(fi)
                    if len(batch_frames) >= infer_batch:
                        _flush(batch_frames, batch_idx)
                        batch_frames.clear()
                        batch_idx.clear()
                    pbar.update(1)
        else:
            from app.parallel_tracker import open_video_capture

            cap = open_video_capture(str(video_path))
            last_pos: int | None = None
            try:
                for fi in frame_ids:
                    ret, frame, last_pos = _advance_capture(cap, fi, last_pos=last_pos)
                    if not ret or frame is None:
                        logger.warning("STAGE 2c: кадр %s не прочитан (pose ends)", fi + 1)
                        pbar.update(1)
                        continue
                    batch_frames.append(frame)
                    batch_idx.append(fi)
                    if len(batch_frames) >= infer_batch:
                        _flush(batch_frames, batch_idx)
                        batch_frames.clear()
                        batch_idx.clear()
                    pbar.update(1)
            finally:
                cap.release()
        _flush(batch_frames, batch_idx)
    finally:
        pbar.close()

    _copy_same_frame_ends(tracklets)
    return n_hit
