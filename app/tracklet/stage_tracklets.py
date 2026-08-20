"""Stage 2a: короткие треклеты (ByteTrack) → tracklet_frames.json / tracklets.json."""

from __future__ import annotations

import logging
import os

from app.artifact_meta import attach_artifact_meta
from app.config import (
    Settings,
    detections_json_path,
    tracklet_frames_json_path,
    tracklet_pose_cache_path,
    tracklets_json_path,
)
from app.io.json_util import save_debug_json
from app.parallel_tracker import associate_tracks
from app.tracklet.common import load_detection_meta
from app.tracklet.summaries import build_tracklet_summaries
from app.tracker_config import tracklet_tracker_params_dict

logger = logging.getLogger(__name__)


def _build_frames_payload(
    tracked: dict[int, list[dict]],
    *,
    fps: float,
    total_frames: int,
    width: int,
    height: int,
    detect_every_n: int,
    video_source: str,
) -> dict:
    frames = []
    for frame_idx in sorted(tracked):
        ts = round((frame_idx + 1) / fps, 3) if fps else 0.0
        cleaned = [
            {
                "tracklet_id": int(det["track_id"]),
                "confidence": det["confidence"],
                "bbox": det["bbox"],
            }
            for det in tracked[frame_idx]
        ]
        if cleaned:
            frames.append(
                {
                    "frame_index": frame_idx + 1,
                    "timestamp_sec": ts,
                    "detections": cleaned,
                }
            )
    return {
        "stage": "tracklets",
        "fps": round(float(fps), 3),
        "frame_count": int(total_frames),
        "width": int(width),
        "height": int(height),
        "detect_every_n": int(detect_every_n),
        "video_source": video_source,
        "n_frames": len(frames),
        "frames": frames,
    }


def run_tracklets(settings: Settings) -> None:
    if str(settings.input_path) == "0":
        raise ValueError("Нужен видеофайл: пайплайн только офлайн.")

    det_path = detections_json_path(settings)
    if not os.path.isfile(det_path):
        raise ValueError(f"Нет detections JSON: {det_path}. Сначала --stage detect")

    cache_path = tracklet_pose_cache_path(settings)
    if os.path.isfile(cache_path):
        try:
            os.remove(cache_path)
        except OSError:
            logger.warning("Не удалось удалить кэш поз %s", cache_path)

    meta = load_detection_meta(settings, det_path)
    logger.info(
        "STAGE 2a: треклеты (%s) из %s",
        settings.tracklet_local_tracker,
        det_path,
    )

    tracked = associate_tracks(
        meta["all_detections"],
        tracker_type=settings.tracklet_local_tracker,
        total_frames=meta["total_frames"],
        tracker_overrides=tracklet_tracker_params_dict(settings),
        nms_iou=settings.nms_iou,
        detect_every_n=meta["detect_every_n"],
    )

    summaries = build_tracklet_summaries(
        {
            fi: [{**det, "tracklet_id": int(det["track_id"])} for det in dets]
            for fi, dets in tracked.items()
        },
        fps=meta["fps"],
        frame_w=meta["width"],
        frame_h=meta["height"],
    )

    frames_path = tracklet_frames_json_path(settings)
    frames_payload = _build_frames_payload(
        tracked,
        fps=meta["fps"],
        total_frames=meta["total_frames"],
        width=meta["width"],
        height=meta["height"],
        detect_every_n=meta["detect_every_n"],
        video_source=meta["video_source"],
    )
    attach_artifact_meta(frames_payload, stage="tracklets", path=frames_path)
    save_debug_json(frames_path, frames_payload)

    summary_path = tracklets_json_path(settings)
    summary_payload = {
        "stage": "tracklets",
        "fps": round(meta["fps"], 3),
        "frame_count": meta["total_frames"],
        "width": meta["width"],
        "height": meta["height"],
        "n_tracklets": len(summaries),
        "tracklets": summaries,
    }
    attach_artifact_meta(summary_payload, stage="tracklets", path=summary_path)
    save_debug_json(summary_path, summary_payload)

    logger.info(
        "STAGE 2a: треклетов=%s, кадров=%s",
        len(summaries),
        frames_payload["n_frames"],
    )
