"""Stage 2b: ReID-эмбеддинги на треклеты с выбором лучших кадров через TrackBestFramesPicker."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from app.artifact_meta import attach_artifact_meta
from app.config import (
    Settings,
    tracklet_crops_dir,
    tracklet_frames_json_path,
    tracklet_pose_cache_path,
    tracklet_reid_json_path,
    tracklet_reid_npz_path,
    tracklets_json_path,
)
from app.crops import TrackBestFramesPicker, TrackFrameCandidate, crop_person
from app.io.json_util import load_tracking_json, save_debug_json
from app.pose.pose_service import PoseService
from app.progress import make_pbar
from app.reid import ReidExtractor, embed_with_cache_arrays, save_cache
from app.session.reader import SessionFrameReader
from app.tracklet.common import session_manifest

logger = logging.getLogger(__name__)


def _resolve_video_path(settings: Settings, meta: dict) -> str | None:
    manifest = session_manifest(settings)
    if manifest:
        return None
    name = os.path.basename(str(settings.input_path))
    folder = os.path.dirname(str(settings.input_path))
    for cand in (
        meta.get("video_source"),
        settings.input_path,
        os.path.join(folder, name) if folder else None,
        os.path.join("data", "video", name),
    ):
        if cand and os.path.isfile(str(cand)):
            return str(cand)
    raise ValueError(f"Видео не найдено: {settings.input_path}")


def _advance_capture(
    cap: cv2.VideoCapture, local: int, *, last_pos: int | None
) -> tuple[bool, np.ndarray | None, int]:
    """last_pos — индекс кадра, который cap отдаст на следующем read (CAP_PROP_POS_FRAMES)."""
    cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0) if last_pos is None else last_pos
    if local < cur:
        cap.set(cv2.CAP_PROP_POS_FRAMES, local)
        ret, frame = cap.read()
        return ret, frame, local + 1 if ret else local
    if local == cur:
        ret, frame = cap.read()
        return ret, frame, local + 1 if ret else cur
    while cur < local:
        if not cap.grab():
            return False, None, cur
        cur += 1
    ret, frame = cap.retrieve()
    return ret, frame, local + 1 if ret else local


def _preselect_candidate_frames(
    observations: Sequence[tuple[int, dict, list[dict]]],
    top_k: int,
    multiplier: int = 3,
) -> list[tuple[int, dict, list[dict]]]:
    """
    Предварительный выбор пула кандидатов для треклета (spread во времени).
    Если наблюдений мало (<= top_k * multiplier), берем все.
    """
    n = len(observations)
    max_cands = top_k * multiplier
    if n <= max_cands:
        return list(observations)

    # Равномерная сетка индексов по времени
    indices = np.linspace(0, n - 1, max_cands, dtype=int)
    seen = set()
    picked = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            picked.append(observations[idx])
    return picked


def run_tracklet_reid(settings: Settings) -> None:
    frames_path = tracklet_frames_json_path(settings)
    tl_path = tracklets_json_path(settings)
    if not os.path.isfile(frames_path):
        raise ValueError(f"Нет tracklet_frames JSON: {frames_path}. Сначала --stage tracklets")
    if not os.path.isfile(tl_path):
        raise ValueError(f"Нет tracklets JSON: {tl_path}")

    frames_data = load_tracking_json(frames_path)
    tl_data = load_tracking_json(tl_path)
    tracklets = list(tl_data.get("tracklets") or [])
    if not tracklets:
        raise ValueError("tracklets.json пуст")

    crops_out = tracklet_crops_dir(settings)
    save_crops = bool(settings.tracklet_reid_save_crops)
    if save_crops:
        if os.path.isdir(crops_out):
            for fname in os.listdir(crops_out):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    try:
                        os.remove(os.path.join(crops_out, fname))
                    except OSError:
                        pass
        os.makedirs(crops_out, exist_ok=True)

    # 1. Инициализация ReID экстрактора
    extractor = ReidExtractor(
        model_name=settings.tracklet_reid_model,
        weights=settings.tracklet_reid_weights,
        device=settings.tracklet_reid_device,
        backend=settings.tracklet_reid_backend,
        solider_weights=settings.tracklet_reid_solider_weights,
        solider_semantic_weight=settings.tracklet_reid_solider_semantic_weight,
        solider_image_size=settings.tracklet_reid_solider_image_size,
        solider_transformer=settings.tracklet_reid_solider_transformer,
    )
    ok, reason = extractor.available()
    if not ok:
        raise RuntimeError(f"ReID недоступен: {reason}")

    # 2. Инициализация PoseService для отбора кадров
    pose_service: PoseService | None = None
    try:
        pose_service = PoseService(
            model_name=settings.pose_model,
            models_dir=settings.models_dir,
            conf=settings.conf,
            imgsz=settings.imgsz,
            device=settings.device,
            quantize=settings.quantize,
        )
    except Exception as e:
        logger.warning("PoseService не удалось загрузить для отбора кадров: %s", e)
        pose_service = None

    picker = TrackBestFramesPicker(
        pose_service=pose_service,
        pose_weight=settings.tracklet_reid_pose_weight,
        crowd_crop_penalty=settings.tracklet_reid_crowd_penalty,
        kpt_min=settings.tracklet_reid_min_completeness,
        crop_pad=settings.tracklet_reid_pad,
    )

    # 3. Индексация детекций по кадрам и треклетам
    obs_by_tid: dict[int, list[tuple[int, dict, list[dict]]]] = defaultdict(list)
    for frame in frames_data.get("frames") or []:
        fi = int(frame["frame_index"]) - 1
        dets = frame.get("detections") or []
        for det in dets:
            tid = int(det["tracklet_id"])
            other_dets = [d for d in dets if int(d.get("tracklet_id", -1)) != tid]
            obs_by_tid[tid].append((fi, det, other_dets))

    # 4. Предварительный отбор кадров-кандидатов
    cands_by_frame: dict[int, list[tuple[int, dict, list[dict]]]] = defaultdict(list)
    top_k = max(1, int(settings.tracklet_reid_top_k))

    for rec in tracklets:
        tid = int(rec["tracklet_id"])
        obs = obs_by_tid.get(tid) or []
        if not obs:
            continue
        cands = _preselect_candidate_frames(obs, top_k=top_k, multiplier=3)
        for fi, det, other_dets in cands:
            cands_by_frame[fi].append((tid, det, other_dets))

    # 5. Чтение необходимых кадров из видео и вырезка кропов на лету (RAM-friendly)
    pad = float(getattr(settings, "tracklet_reid_pad", 0.04) or 0.04)
    all_candidates: list[TrackFrameCandidate] = []
    sorted_frames = sorted(cands_by_frame.keys())
    manifest = session_manifest(settings)
    video_path = _resolve_video_path(settings, frames_data) if not manifest else None

    def _process_frame_candidates(frame: np.ndarray, fi: int) -> None:
        fh, fw = frame.shape[:2]
        for tid, det, other_dets in cands_by_frame[fi]:
            crop, roi = crop_person(frame, det["bbox"], pad=pad)
            all_candidates.append(
                TrackFrameCandidate(
                    frame_index=fi,
                    target_det=det,
                    all_dets=other_dets,
                    crop_image=np.ascontiguousarray(crop),
                    crop_roi=roi,
                    frame_w=fw,
                    frame_h=fh,
                    tracklet_id=tid,
                )
            )

    pbar_read = make_pbar(
        total=len(sorted_frames), desc="[STAGE 2b: read & crop]", unit="frame"
    )
    try:
        if manifest:
            reader = SessionFrameReader(manifest)
            try:
                for fi in sorted_frames:
                    frame = reader.read_frame(fi)
                    if frame is not None:
                        _process_frame_candidates(frame, fi)
                    pbar_read.update(1)
            finally:
                reader.close()
        else:
            from app.parallel_tracker import open_video_capture

            cap = open_video_capture(str(video_path))
            last_pos: int | None = None
            try:
                for fi in sorted_frames:
                    ret, frame, last_pos = _advance_capture(cap, fi, last_pos=last_pos)
                    if ret and frame is not None:
                        _process_frame_candidates(frame, fi)
                    else:
                        logger.warning("STAGE 2b: кадр %s не прочитан", fi + 1)
                    pbar_read.update(1)
            finally:
                cap.release()
    finally:
        pbar_read.close()

    # 6. Единый пакетный скоринг всех кандидатов через TrackBestFramesPicker
    logger.info(
        "STAGE 2b: скоринг %s кропов кандидатов через TrackBestFramesPicker",
        len(all_candidates),
    )
    scored_candidates = picker.score_candidates_batch(
        all_candidates,
        batch_size=settings.tracklet_reid_batch_size,
        show_pbar=True,
        pbar_desc="[STAGE 2b: Pose scoring]",
        cache_path=tracklet_pose_cache_path(settings),
        prune_cache=True,
    )

    scored_by_tid: dict[int, list[ScoredTrackFrame]] = defaultdict(list)
    for sc in scored_candidates:
        tid = sc.tracklet_id if sc.tracklet_id is not None else 0
        scored_by_tid[tid].append(sc)

    # 7. Финальный отбор top_k лучших кропов на треклет
    all_keys: list[str] = []
    all_crop_images: list[np.ndarray] = []
    paths_by_tid: dict[int, list[str]] = defaultdict(list)

    for rec in tracklets:
        tid = int(rec["tracklet_id"])
        cands_scored = scored_by_tid.get(tid) or []
        if not cands_scored:
            continue

        best_frames = picker.pick_best_from_scored(cands_scored, top_k=top_k)
        for ki, scored_f in enumerate(best_frames):
            fi = scored_f.frame_index
            crop_path = os.path.join(crops_out, f"tl_{tid:04d}_k{ki}_f{fi + 1}.jpg")
            crop = scored_f.crop_image
            if crop is None:
                crop = np.zeros((10, 10, 3), dtype=np.uint8)

            if save_crops:
                cv2.imwrite(crop_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            all_keys.append(crop_path)
            all_crop_images.append(crop)
            paths_by_tid[tid].append(crop_path)

    # 7. Извлечение ReID-эмбеддингов
    logger.info(
        "STAGE 2b: ReID для %s треклетов (%s кропов, batch=%s, save_crops=%s)",
        len(tracklets),
        len(all_keys),
        settings.tracklet_reid_batch_size,
        settings.tracklet_reid_save_crops,
    )

    npz_path = tracklet_reid_npz_path(settings)
    cache = embed_with_cache_arrays(
        extractor,
        all_keys,
        all_crop_images,
        {},
        batch_size=settings.tracklet_reid_batch_size,
    )
    save_cache(npz_path, cache)

    # 8. Формирование результатов tracklet_reid.json
    tracklet_records: list[dict] = []
    dim = 0
    for rec in tracklets:
        tid = int(rec["tracklet_id"])
        paths_for_tid = paths_by_tid.get(tid) or []
        if not paths_for_tid:
            continue
        key = paths_for_tid[0]
        if key in cache:
            dim = int(cache[key].shape[-1])
        tracklet_records.append(
            {
                "tracklet_id": tid,
                "crop_paths": paths_for_tid,
                "embedding_key": key,
            }
        )

    out_path = tracklet_reid_json_path(settings)
    payload = {
        "stage": "tracklet_reid",
        "backend": settings.tracklet_reid_backend,
        "model": (
            settings.tracklet_reid_solider_transformer
            if settings.tracklet_reid_backend == "solider"
            else settings.tracklet_reid_model
        ),
        "dim": dim,
        "npz_path": npz_path,
        "tracklets": tracklet_records,
    }
    attach_artifact_meta(payload, stage="tracklet_reid", path=out_path)
    save_debug_json(out_path, payload)
    logger.info("STAGE 2b: кропов=%s, треклетов с ReID=%s", len(all_keys), len(tracklet_records))
