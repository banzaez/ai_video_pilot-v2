"""Офлайн: info → detect → tracklets → tracklet_reid → tracklet_link → track → pose → feet."""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import replace

from app.config import (
    Settings,
    detections_json_path,
    info_json_path,
    session_key,
    video_stem,
    json_output_path,
    tracking_json_path,
    tracks_json_path,
    tracklet_frames_json_path,
    tracklet_links_json_path,
)
from app.io.export import TrackingExporter
from app.io.video import VideoSource
from app.parallel_tracker import associate_tracks, default_workers
from app.detectors.payload import build_detections_payload, resolve_part_path
from app.detectors.stage_detect import detect_video_part, detector_meta
from app.info import (
    build_video_info,
    camera_meta,
    list_video_files,
    resolve_video_jobs,
    stamp_camera,
    video_stem_of,
)
from app.model_cache import ModelCache, set_model_cache
from app.artifact_meta import attach_artifact_meta
from app.io.json_util import load_tracking_json, save_debug_json
from app.postprocess import apply_track_filters, build_track_summaries, surviving_track_ids
from app.progress import make_pbar
from app.session.discover import SESSION_PREFIX, resolve_sessions_for_input
from app.session.manifest import build_session_manifest, is_session_manifest
from app.tracker_config import tracker_params_dict
from app.tracklet import run_tracklet_link, run_tracklet_reid, run_tracklets
from app.tracklet.common import load_detection_meta
from app.tracklet.remap import remap_tracklet_frames
from app.global_id.stage_feet import run_feet
from app.global_id.stage_pose import run_pose
from app.global_id.stage_camera_link import run_camera_link

_SESSION_KEY_RE = re.compile(r"^\d{2}_\d{8}$")

logger = logging.getLogger(__name__)


def _load_info(settings: Settings) -> dict | None:
    info_path = info_json_path(settings)
    return load_tracking_json(info_path) if os.path.isfile(info_path) else None


def _session_manifest(settings: Settings) -> dict | None:
    info = _load_info(settings)
    return info if is_session_manifest(info) else None


def _save_stage_json(
    path: str,
    payload: dict,
    *,
    stage: str,
) -> None:
    attach_artifact_meta(payload, stage=stage, path=path)
    save_debug_json(path, payload)


def _camera_meta_for(settings: Settings) -> dict:
    info_path = info_json_path(settings)
    info = load_tracking_json(info_path) if os.path.isfile(info_path) else None
    return camera_meta(info, stem=video_stem(settings))


def run_info(settings: Settings, video_dir: str | None = None, jobs: list[str] | None = None) -> None:
    """Stage 0: manifest session или info.json per-file (legacy)."""
    if str(settings.input_path) == "0":
        raise ValueError("Нужен видеофайл: пайплайн только офлайн (detect → ByteTrack).")

    mode, sessions, legacy_jobs = resolve_sessions_for_input(str(settings.input_path))
    if mode == "session":
        _run_info_sessions(settings, sessions)
        return

    if video_dir is None:
        video_dir, jobs_resolved = resolve_video_jobs(str(settings.input_path))
        if jobs is None:
            jobs = legacy_jobs or jobs_resolved

    results_root_path = str(settings.json_output_dir or "data/results")
    videos = list(jobs) if jobs is not None else list_video_files(video_dir)
    stems = {video_stem_of(path): path for path in videos}
    scoped = jobs is not None
    logger.info(
        "STAGE 0: папка %s, роликов %s%s",
        video_dir,
        len(videos),
        " (только выбранные)" if scoped else "",
    )

    if not scoped and os.path.isdir(results_root_path):
        for name in os.listdir(results_root_path):
            folder = os.path.join(results_root_path, name)
            if not os.path.isdir(folder) or name in stems or name.startswith("_"):
                continue
            info_fp = os.path.join(folder, "info.json")
            if os.path.isfile(info_fp):
                info = load_tracking_json(info_fp)
                if is_session_manifest(info):
                    continue
            logger.info("STAGE 0: видео нет, удаляем %s", folder)
            shutil.rmtree(folder, ignore_errors=True)

    n_fail = 0
    for path in videos:
        job = replace(settings, input_path=path)
        try:
            payload = build_video_info(path)
            out_path = info_json_path(job)
            _save_stage_json(out_path, payload, stage="info")
            logger.info(
                "STAGE 0: %s, %sx%s, %.1f сек, %s fps",
                payload.get("stem") or os.path.basename(path),
                payload.get("width"),
                payload.get("height"),
                float(payload.get("duration_sec") or 0),
                payload.get("fps"),
            )
        except Exception as exc:
            n_fail += 1
            logger.error("STAGE 0: %s — %s", path, exc)
    if n_fail:
        logger.warning("STAGE 0: ошибок %s / %s", n_fail, len(videos))


def _run_info_sessions(settings: Settings, sessions) -> None:
    res_root = str(settings.json_output_dir or "data/results")
    scoped = session_key(settings) is not None
    active_keys = {s.key for s in sessions}
    logger.info("STAGE 0: sessions %s%s", len(sessions), " (одна)" if scoped else "")

    if not scoped and os.path.isdir(res_root):
        for name in os.listdir(res_root):
            if name.startswith("_"):
                continue
            folder = os.path.join(res_root, name)
            if not os.path.isdir(folder):
                continue
            info_fp = os.path.join(folder, "info.json")
            is_session_dir = False
            if os.path.isfile(info_fp):
                info = load_tracking_json(info_fp)
                is_session_dir = is_session_manifest(info)
            elif _SESSION_KEY_RE.match(name):
                is_session_dir = True
            if is_session_dir and name not in active_keys:
                logger.info("STAGE 0: session без частей, удаляем %s", folder)
                shutil.rmtree(folder, ignore_errors=True)

    n_fail = 0
    for sess in sessions:
        job = replace(settings, input_path=f"{SESSION_PREFIX}{sess.key}")
        try:
            payload = build_session_manifest(sess)
            out_path = info_json_path(job)
            _save_stage_json(out_path, payload, stage="info")
            logger.info(
                "STAGE 0: session %s, %s частей, %sx%s, %.1f сек, %s fps",
                sess.key,
                len(sess.parts),
                payload.get("width"),
                payload.get("height"),
                float(payload.get("duration_sec") or 0),
                payload.get("fps"),
            )
        except Exception as exc:
            n_fail += 1
            logger.error("STAGE 0: session %s — %s", sess.key, exc)
    if n_fail:
        logger.warning("STAGE 0: ошибок %s / %s", n_fail, len(sessions))


from app.detections_io import detections_from_json
def run_detect(settings: Settings) -> None:
    """Stage 1: YOLO или RT-DETR → detections.json. Трекер не вызывается."""
    if str(settings.input_path) == "0":
        raise ValueError("Нужен видеофайл: пайплайн только офлайн (detect → ByteTrack).")
    manifest = _session_manifest(settings)
    if manifest:
        run_detect_session(settings, manifest)
        return

    with VideoSource(settings.input_path) as source:
        meta = source.meta
        total_frames = meta.frame_count
        if not total_frames:
            raise ValueError("Не удалось определить число кадров видео")

    meta_info = detector_meta(settings)
    workers = default_workers(settings.workers, settings.device)
    logger.info(
        "STAGE 1: %s (workers=%s, batch=%s, imgsz=%s, device=%s, quantize=%s, "
        "detect_every_n=%s, iou=%s, model=%s)",
        settings.detection_backend,
        workers,
        settings.batch_size,
        settings.imgsz,
        settings.device,
        settings.quantize,
        settings.detect_every_n,
        settings.nms_iou,
        settings.model_path,
    )

    all_detections = detect_video_part(settings, settings.input_path, total_frames)

    payload = build_detections_payload(
        all_detections,
        fps=float(meta.fps),
        frame_count=int(total_frames),
        width=int(meta.width),
        height=int(meta.height),
        conf=settings.conf,
        detect_every_n=settings.detect_every_n,
        nms_iou=settings.nms_iou,
        video_source=str(settings.input_path),
        detector_meta=meta_info,
    )
    out_path = detections_json_path(settings)
    _save_stage_json(out_path, payload, stage="detect")
    n_det = sum(len(fr["detections"]) for fr in payload["frames"])
    logger.info("STAGE 1: кадров с людьми=%s, детекций=%s", payload["n_frames"], n_det)


def run_detect_session(settings: Settings, manifest: dict) -> None:
    """Stage 1 для session: detect по частям → merged detections.json."""
    fps = float(manifest.get("fps") or 0.0)
    width = int(manifest.get("width") or 0)
    height = int(manifest.get("height") or 0)
    total_frames = int(manifest.get("frame_count") or 0)
    parts = manifest.get("parts") or []
    if not parts or fps <= 0:
        raise ValueError("Некорректный session manifest")

    meta_info = detector_meta(settings)
    workers = default_workers(settings.workers, settings.device)
    logger.info(
        "STAGE 1: session %s, %s частей, %s (workers=%s, batch=%s, model=%s)",
        manifest.get("session_key"),
        len(parts),
        settings.detection_backend,
        workers,
        settings.batch_size,
        settings.model_path,
    )

    merged: dict[int, list] = {}
    sources_meta: list[dict] = []
    for part in parts:
        part_path = resolve_part_path(part)
        offset = int(part.get("frame_offset") or 0)
        time_off = float(part.get("time_offset_sec") or 0.0)
        part_frames = int(part.get("frame_count") or 0)
        stem = part.get("stem") or os.path.basename(part_path)
        logger.info("STAGE 1: часть %s, кадров=%s, offset=%s", stem, part_frames, offset)

        dets = detect_video_part(
            settings,
            part_path,
            part_frames,
        )
        for local_idx, items in dets.items():
            if items:
                merged[offset + int(local_idx)] = items
        sources_meta.append(
            {
                "stem": stem,
                "path": part.get("path"),
                "frame_offset": offset,
                "frame_count": part_frames,
                "time_offset_sec": time_off,
            }
        )

    sk = manifest.get("session_key") or session_key(settings)
    payload = build_detections_payload(
        merged,
        fps=fps,
        frame_count=total_frames,
        width=width,
        height=height,
        conf=settings.conf,
        detect_every_n=settings.detect_every_n,
        nms_iou=settings.nms_iou,
        video_source=f"{SESSION_PREFIX}{sk}",
        session_key=sk,
        sources_meta=sources_meta,
        detector_meta=meta_info,
    )
    out_path = detections_json_path(settings)
    _save_stage_json(out_path, payload, stage="detect")
    n_det = sum(len(fr["detections"]) for fr in payload["frames"])
    logger.info("STAGE 1: session merged кадров=%s, детекций=%s", payload["n_frames"], n_det)


def run_track(settings: Settings) -> None:
    """Stage 2: direct tracker или финализация tracklet_global → tracking.json."""
    if str(settings.input_path) == "0":
        raise ValueError("Нужен видеофайл: пайплайн только офлайн (detect → ByteTrack).")

    if settings.tracklet_mode == "tracklet_global":
        _run_track_finalize(settings)
        return

    det_path = detections_json_path(settings)
    if not os.path.isfile(det_path):
        raise ValueError(f"Нет detections JSON: {det_path}. Сначала --stage detect")

    meta = load_detection_meta(settings, det_path)
    logger.info("STAGE 2: трекинг (%s) из %s", settings.tracker_type, det_path)
    tracked = associate_tracks(
        meta["all_detections"],
        tracker_type=settings.tracker_type,
        total_frames=meta["total_frames"],
        tracker_overrides=tracker_params_dict(settings),
        nms_iou=settings.nms_iou,
        detect_every_n=meta["detect_every_n"],
    )
    _write_tracking_outputs(settings, tracked, meta)


def _run_track_finalize(settings: Settings) -> None:
    frames_path = tracklet_frames_json_path(settings)
    links_path = tracklet_links_json_path(settings)
    if not os.path.isfile(frames_path):
        raise ValueError(f"Нет tracklet_frames JSON: {frames_path}. Сначала --stage tracklets")
    if not os.path.isfile(links_path):
        raise ValueError(f"Нет tracklet_links JSON: {links_path}. Сначала --stage tracklet_link")

    frames_data = load_tracking_json(frames_path)
    links_data = load_tracking_json(links_path)
    mapping = links_data.get("tracklet_to_global") or {}
    if not mapping:
        raise ValueError("tracklet_links.json: пустой tracklet_to_global")

    tracked = remap_tracklet_frames(frames_data, mapping)
    meta = {
        "fps": float(frames_data.get("fps") or 0),
        "total_frames": int(frames_data.get("frame_count") or 0),
        "width": int(frames_data.get("width") or 0),
        "height": int(frames_data.get("height") or 0),
        "detect_every_n": int(frames_data.get("detect_every_n") or settings.detect_every_n),
        "video_source": str(frames_data.get("video_source") or settings.input_path),
    }
    logger.info("STAGE 2: финализация tracklet_global (%s треклетов → группы)", len(mapping))
    _write_tracking_outputs(settings, tracked, meta)


def _write_tracking_outputs(
    settings: Settings,
    tracked: dict[int, list],
    meta: dict,
) -> None:
    fps = float(meta["fps"])
    total_frames = int(meta["total_frames"])
    width = int(meta["width"])
    height = int(meta["height"])
    detect_every_n = int(meta["detect_every_n"])

    logger.info(
        "STAGE 2.4: фильтр (min_area=%s, min_side=%s, min_sec=%s)",
        settings.min_bbox_area,
        settings.min_bbox_side,
        settings.min_track_sec,
    )
    n_tracks_before = len(surviving_track_ids(tracked))
    tracked = apply_track_filters(
        tracked,
        min_bbox_area=settings.min_bbox_area,
        min_bbox_side=settings.min_bbox_side,
        min_track_sec=settings.min_track_sec,
        fps=fps,
        stage="2.4",
    )

    exporter = TrackingExporter(
        fps=fps,
        width=width,
        height=height,
        json_path=json_output_path(settings),
        detect_every_n=detect_every_n,
        video_source=str(meta.get("video_source") or settings.input_path),
    )

    nonempty = sorted(idx for idx, dets in tracked.items() if dets)
    p_json = make_pbar(total=max(1, len(nonempty)), desc="[JSON]", unit="frame")
    try:
        for frame_idx in nonempty:
            dets = tracked[frame_idx]
            exporter.add_frame(frame_idx + 1, fps, dets)
            p_json.set_postfix(persons=len(dets))
            p_json.update(1)
    finally:
        p_json.close()
    exporter.save(total_frames)
    logger.info("STAGE 2: кадров с треками=%s", len(nonempty))

    cam = _camera_meta_for(settings)
    summaries = build_track_summaries(
        tracked,
        fps=fps,
        frame_w=width,
        frame_h=height,
        camera_index=cam.get("camera_index"),
    )
    summary_path = tracks_json_path(settings)
    summary = stamp_camera(
        {
            "stage": "track",
            "fps": round(fps, 3),
            "frame_count": int(total_frames),
            "width": width,
            "height": height,
            "n_tracks": len(summaries),
            "tracks": summaries,
        },
        cam,
    )
    _save_stage_json(summary_path, summary, stage="tracks")
    logger.info(
        "STAGE 2: треков после фильтра=%s (было %s, отброшено %s)",
        len(summaries),
        n_tracks_before,
        max(0, n_tracks_before - len(summaries)),
    )


PER_VIDEO_STAGES = (
    "detect",
    "tracklets",
    "tracklet_reid",
    "tracklet_link",
    "track",
    "pose",
    "feet",
    "camera_link",
)
PIPELINE_STAGES = PER_VIDEO_STAGES
TRACKLET_SUBSTAGES = ("tracklets", "tracklet_reid", "tracklet_link")


def _filter_tracklet_stages(stages: list[str], settings: Settings) -> list[str]:
    """В direct mode пропускаем 2a–2c, если пользователь явно не запросил tracklets."""
    if settings.tracklet_mode == "tracklet_global":
        return stages
    if settings.stage in TRACKLET_SUBSTAGES:
        return stages
    if settings.stage_onward and settings.stage in TRACKLET_SUBSTAGES:
        return stages
    return [s for s in stages if s not in TRACKLET_SUBSTAGES]


def stages_to_run(settings: Settings) -> list[str]:
    """Какие стадии обработки запустить. Stage 0 (info) всегда отдельно."""
    stage = settings.stage
    until = settings.stage_until

    def _slice(start_name: str, end_name: str | None) -> list[str]:
        start = 0 if start_name == "info" else PIPELINE_STAGES.index(start_name)
        if end_name is None:
            return list(PIPELINE_STAGES[start:])
        if end_name == "info":
            return []
        end = PIPELINE_STAGES.index(end_name)
        if end < start:
            raise ValueError(
                f"--to '{end_name}' раньше --from '{start_name}'. "
                f"Порядок: {' → '.join(PIPELINE_STAGES)}"
            )
        return list(PIPELINE_STAGES[start : end + 1])

    if stage in ("all", "no_merge"):
        return _filter_tracklet_stages(list(PIPELINE_STAGES), settings)
    if stage == "info" and not settings.stage_onward:
        return []
    if settings.stage_onward:
        return _filter_tracklet_stages(_slice(stage, until), settings)
    if stage not in PIPELINE_STAGES:
        raise ValueError(f"Неизвестный stage '{stage}'")
    return [stage]


def run(settings: Settings) -> None:
    mode, sessions, legacy_jobs = resolve_sessions_for_input(str(settings.input_path))
    run_info(settings)
    wanted = stages_to_run(settings)
    if not wanted:
        return
    per_video = [s for s in wanted if s in PER_VIDEO_STAGES]
    if per_video:
        if mode == "session":
            if not sessions:
                raise ValueError("Нет sessions для обработки")
        elif not legacy_jobs:
            raise ValueError(f"В папке нет видео: {settings.input_path}")
    if settings.stage_onward:
        how = "from…to" if settings.stage_until else "from"
        rng = f"{settings.stage}…{settings.stage_until or PIPELINE_STAGES[-1]}"
    else:
        how = "stage"
        rng = settings.stage
    n_jobs = len(sessions) if mode == "session" else len(legacy_jobs)
    logger.info("Обработка %s job(s), %s=%s → %s", n_jobs, how, rng, ",".join(wanted))
    cache = ModelCache()
    set_model_cache(cache)
    runners = {
        "detect": run_detect,
        "tracklets": run_tracklets,
        "tracklet_reid": run_tracklet_reid,
        "tracklet_link": run_tracklet_link,
        "track": run_track,
        "pose": run_pose,
        "feet": run_feet,
        "camera_link": run_camera_link,
    }
    fail_fast = os.environ.get("PIPELINE_FAIL_FAST", "").strip().lower() in ("1", "true", "yes")
    errors: list[str] = []
    if per_video:
        if mode == "session":
            for sess in sessions:
                job = replace(settings, input_path=f"{SESSION_PREFIX}{sess.key}")
                logger.info("=== session %s ===", sess.key)
                try:
                    for name in per_video:
                        runners[name](job)
                except Exception as exc:
                    msg = f"session {sess.key}: {exc}"
                    errors.append(msg)
                    logger.exception("Ошибка на session %s", sess.key)
                    if fail_fast:
                        raise
        else:
            for video in legacy_jobs:
                job = replace(settings, input_path=video)
                logger.info("=== %s ===", os.path.basename(video))
                try:
                    for name in per_video:
                        runners[name](job)
                except Exception as exc:
                    msg = f"{os.path.basename(video)}: {exc}"
                    errors.append(msg)
                    logger.exception("Ошибка на %s", os.path.basename(video))
                    if fail_fast:
                        raise
    if errors:
        raise RuntimeError(
            f"Пайплайн завершился с ошибками ({len(errors)}):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
