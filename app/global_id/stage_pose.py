"""Stage pose: YOLO-pose S на кадрах tracking → poses.json."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from app.artifact_meta import attach_artifact_meta
from app.config import (
    Settings,
    cameras_dir,
    info_json_path,
    poses_json_path,
    tracking_json_path,
    tracklet_frames_json_path,
)
from app.global_id.camera_pose import dual_plane_from_bbox, load_camera_doc, parse_camera_pose
from app.io.json_util import load_tracking_json, save_debug_json
from app.pose.types import select_pose_index_by_completeness
from app.model_cache import get_model_cache, predict_batch_size, resolve_pt_path
from app.parallel_tracker import FrameReaderThread, open_video_capture, reader_queue_size
from app.pose import get_pose_service
from app.progress import make_pbar
from app.tracklet.map_coords import camera_key_for_settings

logger = logging.getLogger(__name__)


def _session_manifest(settings: Settings) -> dict[str, Any] | None:
    from app.session.manifest import is_session_manifest

    path = info_json_path(settings)
    if not os.path.isfile(path):
        return None
    try:
        info = load_tracking_json(path)
    except Exception:
        return None
    return info if is_session_manifest(info) else None


def _resolve_video(settings: Settings) -> str:
    if os.path.isfile(str(settings.input_path)):
        return str(settings.input_path)
    raise ValueError(f"Видео не найдено: {settings.input_path}")


def _frame_wants_pose(
    dets: list[dict[str, Any]],
    pose,
    image_size: tuple[int, int] | None,
    person_height_m: float,
) -> bool:
    if pose is None or not image_size:
        return True
    if not dets:
        return False
    for det in dets:
        bbox = det.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        _mapped, _src, truncated, _h, _f = dual_plane_from_bbox(
            bbox, pose, image_size, person_height_m=person_height_m
        )
        if not truncated:
            return True
    return False


def _pose_to_dicts(
    pose_results: list[Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not pose_results:
        return out
    r0 = pose_results[0]
    boxes = getattr(r0, "boxes", None)
    keypoints = getattr(r0, "keypoints", None)
    if boxes is None or keypoints is None:
        return out
    xyxy = getattr(boxes, "xyxy", None)
    confs = getattr(boxes, "conf", None)
    kxy = getattr(keypoints, "xy", None)
    kcf = getattr(keypoints, "conf", None)
    if xyxy is None or confs is None or kxy is None or kcf is None:
        return out
    xyxy = xyxy.cpu().numpy()
    confs = confs.cpu().numpy()
    kxy = kxy.cpu().numpy()
    kcf = kcf.cpu().numpy()
    for i in range(len(xyxy)):
        box = [float(v) for v in xyxy[i][:4]]
        xy = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in kxy[i]]
        cf = [round(float(c), 3) for c in kcf[i]]
        out.append(
            {
                "bbox": box,
                "confidence": round(float(confs[i]), 4),
                "kxy": xy,
                "kcf": cf,
            }
        )
    return out


def _match_to_tracks(
    pose_inst: list[dict[str, Any]],
    track_dets: list[dict[str, Any]],
    frame_index: int,
    timestamp_sec: float,
    *,
    kpt_min: float = 0.25,
) -> list[dict[str, Any]]:
    used: set[int] = set()
    matched: list[dict[str, Any]] = []
    rows = [
        (
            inst.get("bbox") or [],
            inst.get("kcf") or [],
            float(inst.get("confidence") or 0.0),
        )
        for inst in pose_inst
    ]
    for det in track_dets:
        raw_tid = det.get("track_id") if det.get("track_id") is not None else det.get("tracklet_id")
        try:
            tid = int(raw_tid)
        except (KeyError, TypeError, ValueError):
            continue
        tb = det.get("bbox")
        if not isinstance(tb, (list, tuple)) or len(tb) < 4:
            continue
        tb4 = [float(tb[0]), float(tb[1]), float(tb[2]), float(tb[3])]
        best_i = select_pose_index_by_completeness(rows, tb4, kpt_min=kpt_min, skip=used)
        if best_i is None:
            continue
        used.add(best_i)
        inst = pose_inst[best_i]
        matched.append(
            {
                "track_id": tid,
                "frame_index": frame_index,
                "timestamp_sec": round(float(timestamp_sec), 3),
                "bbox": [int(round(v)) for v in tb4],
                "confidence": inst["confidence"],
                "kxy": inst["kxy"],
                "kcf": inst["kcf"],
            }
        )
    return matched


def load_pose_lookup(work_dir: str) -> dict[int, dict[int, dict[str, Any]]]:
    """track_id → frame_index → {kxy, kcf, bbox}."""
    path = os.path.join(work_dir, "poses.json")
    if not os.path.isfile(path):
        return {}
    try:
        doc = load_tracking_json(path)
    except Exception:
        return {}
    if not isinstance(doc, dict):
        return {}
    index: dict[int, dict[int, dict[str, Any]]] = {}
    for obs in doc.get("observations") or []:
        if not isinstance(obs, dict):
            continue
        try:
            tid = int(obs["track_id"])
            fi = int(obs["frame_index"])
        except (KeyError, TypeError, ValueError):
            continue
        kxy = obs.get("kxy")
        kcf = obs.get("kcf")
        if not isinstance(kxy, list) or not isinstance(kcf, list):
            continue
        index.setdefault(tid, {})[fi] = {
            "kxy": kxy,
            "kcf": kcf,
            "bbox": obs.get("bbox"),
        }
    return index


def run_pose(settings: Settings) -> None:
    track_path = tracking_json_path(settings)
    tl_frames_path = tracklet_frames_json_path(settings)
    source_name = "track"
    if os.path.isfile(track_path):
        tracking = load_tracking_json(track_path)
    elif os.path.isfile(tl_frames_path):
        tracking = load_tracking_json(tl_frames_path)
        source_name = "tracklets"
    else:
        raise ValueError(
            f"Нет tracking JSON ({track_path}) и нет tracklet_frames JSON ({tl_frames_path}). "
            "Сначала --stage tracklets или --stage track"
        )

    frames_in = tracking.get("frames") or []
    selected: list[dict[str, Any]] = list(frames_in)

    cam_dir = cameras_dir(settings)
    camera_key = camera_key_for_settings(settings)
    camera_doc = load_camera_doc(cam_dir, camera_key)
    cam_pose = parse_camera_pose(camera_doc) if camera_doc else None
    image_size = None
    if isinstance(camera_doc, dict):
        raw = camera_doc.get("image_size")
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            image_size = (int(raw[0]), int(raw[1]))
    if image_size is None:
        tw, th = int(tracking.get("width") or 0), int(tracking.get("height") or 0)
        if tw > 0 and th > 0:
            image_size = (tw, th)
    person_h = float(settings.feet_person_height_m)

    want: dict[int, dict[str, Any]] = {}
    skipped_full = 0
    for fr in selected:
        try:
            fi = int(fr.get("frame_index"))
        except (TypeError, ValueError):
            continue
        dets = [d for d in (fr.get("detections") or []) if isinstance(d, dict)]
        if not dets:
            continue
        if not _frame_wants_pose(dets, cam_pose, image_size, person_h):
            skipped_full += 1
            continue
        want[fi - 1] = fr

    observations: list[dict[str, Any]] = []
    n_with_pose = 0
    pose_service = get_pose_service(settings)
    model_path = pose_service.model_path
    if want:
        infer_batch = pose_service.effective_batch_size()

        def _flush(batch_frames: list, batch_meta: list[dict[str, Any]]) -> None:
            nonlocal n_with_pose
            if not batch_frames:
                return
            batch_pose_results = pose_service.predict_batch(batch_frames)
            for meta, pose_items in zip(batch_meta, batch_pose_results):
                inst = [
                    {
                        "bbox": r.bbox,
                        "confidence": r.confidence,
                        "kxy": r.kxy,
                        "kcf": r.kcf,
                    }
                    for r in pose_items
                ]
                dets = [d for d in (meta.get("detections") or []) if isinstance(d, dict)]
                matched = _match_to_tracks(
                    inst,
                    dets,
                    int(meta["frame_index"]),
                    float(meta.get("timestamp_sec") or 0.0),
                    kpt_min=float(settings.pose_kpt_min),
                )
                n_with_pose += sum(1 for m in matched if m.get("kxy"))
                observations.extend(matched)

        manifest = _session_manifest(settings)
        pbar = make_pbar(total=max(1, len(want)), desc="[STAGE pose]", unit="frame")
        try:
            if manifest:
                from app.session.reader import SessionFrameReader

                batch_frames: list = []
                batch_meta: list[dict[str, Any]] = []
                with SessionFrameReader(manifest) as reader:
                    for idx0 in sorted(want):
                        frame = reader.read_frame(idx0)
                        batch_frames.append(frame)
                        batch_meta.append(want[idx0])
                        if len(batch_frames) >= infer_batch:
                            _flush(batch_frames, batch_meta)
                            batch_frames.clear()
                            batch_meta.clear()
                        pbar.update(1)
                    _flush(batch_frames, batch_meta)
            else:
                video_path = _resolve_video(settings)
                cap = open_video_capture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"Не удалось открыть видео: {video_path}")
                total = int(tracking.get("frame_count") or 0)
                if not total:
                    import cv2

                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                reader = FrameReaderThread(
                    cap, 0, total, want=set(want), queue_size=reader_queue_size(max(8, infer_batch))
                )
                batch_frames = []
                batch_meta = []
                try:
                    for frame_idx, frame in reader:
                        fr = want.get(frame_idx)
                        if fr is None:
                            continue
                        batch_frames.append(frame)
                        batch_meta.append(fr)
                        if len(batch_frames) >= infer_batch:
                            _flush(batch_frames, batch_meta)
                            batch_frames.clear()
                            batch_meta.clear()
                        pbar.update(1)
                    _flush(batch_frames, batch_meta)
                finally:
                    reader.drain()
                    cap.release()
        finally:
            pbar.close()

    payload: dict[str, Any] = {
        "stage": "pose",
        "pose_model": model_path,
        "kpt_min": float(settings.pose_kpt_min),
        "frame_count": tracking.get("frame_count"),
        "width": tracking.get("width"),
        "height": tracking.get("height"),
        "n_obs": len(observations),
        "n_with_pose": n_with_pose,
        "n_skipped_full": skipped_full,
        "observations": observations,
    }
    out_path = poses_json_path(settings)
    attach_artifact_meta(payload, stage="pose", path=out_path)
    save_debug_json(out_path, payload)
    logger.info(
        "STAGE pose: %s наблюдений (%s с keypoints), пропуск полных кадров=%s → %s",
        len(observations),
        n_with_pose,
        skipped_full,
        out_path,
    )
