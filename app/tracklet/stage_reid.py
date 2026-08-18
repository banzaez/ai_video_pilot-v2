"""Stage 2b: ReID-эмбеддинги на треклеты."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from app.artifact_meta import attach_artifact_meta
from app.config import (
    Settings,
    tracklet_crops_dir,
    tracklet_frames_json_path,
    tracklet_reid_json_path,
    tracklet_reid_npz_path,
    tracklets_json_path,
)
from app.io.json_util import load_tracking_json, save_debug_json
from app.crops.geometry import crop_person
from app.progress import make_pbar
from app.reid import ReidExtractor, embed_with_cache_arrays, save_cache
from app.session.reader import SessionFrameReader
from app.tracklet.common import session_manifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CropJob:
    tracklet_id: int
    pick_index: int
    frame_index: int
    det: dict
    crop_path: str


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


from app.util.bbox import bbox_wh


def _observation_quality(det: dict) -> float:
    conf = float(det.get("confidence") or 0.0)
    bbox = det.get("bbox")
    if not bbox or len(bbox) < 4:
        return conf
    w, h = bbox_wh(bbox)
    if w <= 0 or h <= 0:
        return 0.0
    aspect = h / max(1.0, w)
    aspect_factor = 1.0
    if aspect < 1.5:
        aspect_factor = max(0.2, aspect / 1.5)
    elif aspect > 4.5:
        aspect_factor = max(0.5, 4.5 / aspect)
    area = w * h
    size_factor = min(1.2, max(0.8, (area / 10000.0) ** 0.1))
    return conf * aspect_factor * size_factor


def _pick_observations(
    frames_by_tid: dict[int, list[tuple[int, dict]]],
    tracklet_id: int,
    *,
    top_k: int,
    pick: str,
) -> list[tuple[int, dict]]:
    obs = list(frames_by_tid.get(tracklet_id) or [])
    if not obs:
        return []
    if len(obs) <= top_k:
        return obs
    if pick == "best_conf":
        return sorted(obs, key=lambda x: -_observation_quality(x[1]))[:top_k]

    # Гибридный spread: разбиваем интервал на top_k временных окон
    # и выбираем лучший по качеству кадр в каждом окне
    k = max(1, int(top_k))
    chunk_size = len(obs) / float(k)
    picked: list[tuple[int, dict]] = []
    for i in range(k):
        start_idx = int(round(i * chunk_size))
        end_idx = int(round((i + 1) * chunk_size)) if i < k - 1 else len(obs)
        window = obs[start_idx:end_idx]
        if window:
            best_in_window = max(window, key=lambda x: _observation_quality(x[1]))
            picked.append(best_in_window)
    return picked if picked else [obs[int(i)] for i in np.linspace(0, len(obs) - 1, top_k).astype(int)]


def _index_frames(frames_data: dict) -> dict[int, list[tuple[int, dict]]]:
    by_tid: dict[int, list[tuple[int, dict]]] = {}
    for frame in frames_data.get("frames") or []:
        fi = int(frame["frame_index"]) - 1
        for det in frame.get("detections") or []:
            tid = int(det["tracklet_id"])
            by_tid.setdefault(tid, []).append((fi, det))
    return by_tid


def _collect_jobs(
    tracklets: list[dict],
    by_tid: dict[int, list[tuple[int, dict]]],
    *,
    crops_out: str,
    top_k: int,
    pick: str,
) -> list[_CropJob]:
    jobs: list[_CropJob] = []
    for rec in tracklets:
        tid = int(rec["tracklet_id"])
        for ki, (fi, det) in enumerate(
            _pick_observations(by_tid, tid, top_k=top_k, pick=pick)
        ):
            crop_path = os.path.join(crops_out, f"tl_{tid:04d}_k{ki}_f{fi + 1}.jpg")
            jobs.append(_CropJob(tid, ki, fi, det, crop_path))
    return jobs


def _advance_capture(cap: cv2.VideoCapture, local: int, *, last_pos: int | None) -> tuple[bool, np.ndarray | None, int]:
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


def _extract_crops_by_frame(
    jobs: list[_CropJob],
    *,
    settings: Settings,
    frames_data: dict,
) -> tuple[list[str], list[np.ndarray]]:
    """Последовательный grab по видео; JPG опционально (ReID идёт из памяти)."""
    keys: list[str] = []
    images: list[np.ndarray] = []
    by_frame: dict[int, list[_CropJob]] = defaultdict(list)
    for job in jobs:
        by_frame[job.frame_index].append(job)

    save_crops = bool(settings.tracklet_reid_save_crops)
    if save_crops:
        out_crops = tracklet_crops_dir(settings)
        if os.path.isdir(out_crops):
            for fname in os.listdir(out_crops):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    try:
                        os.remove(os.path.join(out_crops, fname))
                    except OSError:
                        pass
        os.makedirs(out_crops, exist_ok=True)

    pad = float(getattr(settings, "tracklet_reid_pad", 0.04) or 0.04)
    manifest = session_manifest(settings)
    video_path = _resolve_video_path(settings, frames_data) if not manifest else None
    frame_ids = sorted(by_frame)
    pbar = make_pbar(total=len(jobs), desc="[STAGE 2b: crops]", unit="crop")

    def _keep_crops(frame: np.ndarray, fi: int) -> None:
        for job in by_frame[fi]:
            crop, _ = crop_person(frame, job.det["bbox"], pad=pad)
            crop = np.ascontiguousarray(crop)
            if save_crops:
                cv2.imwrite(job.crop_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            keys.append(job.crop_path)
            images.append(crop)
            pbar.update(1)

    try:
        if manifest:
            reader = SessionFrameReader(manifest)
            try:
                for fi in frame_ids:
                    _keep_crops(reader.read_frame(fi), fi)
            finally:
                reader.close()
        else:
            from app.parallel_tracker import open_video_capture

            cap = open_video_capture(str(video_path))
            last_pos: int | None = None
            try:
                for fi in frame_ids:
                    ret, frame, last_pos = _advance_capture(cap, fi, last_pos=last_pos)
                    if not ret or frame is None:
                        logger.warning("STAGE 2b: кадр %s не прочитан", fi + 1)
                        pbar.update(len(by_frame[fi]))
                        continue
                    _keep_crops(frame, fi)
            finally:
                cap.release()
    finally:
        pbar.close()

    return keys, images


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
    if settings.tracklet_reid_save_crops:
        os.makedirs(crops_out, exist_ok=True)

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

    by_tid = _index_frames(frames_data)
    jobs = _collect_jobs(
        tracklets,
        by_tid,
        crops_out=crops_out,
        top_k=settings.tracklet_reid_top_k,
        pick=settings.tracklet_reid_pick,
    )

    logger.info(
        "STAGE 2b: ReID для %s треклетов (%s кропов, batch=%s, save_crops=%s)",
        len(tracklets),
        len(jobs),
        settings.tracklet_reid_batch_size,
        settings.tracklet_reid_save_crops,
    )

    keys, images = _extract_crops_by_frame(jobs, settings=settings, frames_data=frames_data)

    npz_path = tracklet_reid_npz_path(settings)
    cache = embed_with_cache_arrays(
        extractor,
        keys,
        images,
        {},
        batch_size=settings.tracklet_reid_batch_size,
    )
    save_cache(npz_path, cache)

    paths_by_tid: dict[int, list[str]] = defaultdict(list)
    for job in jobs:
        if job.crop_path in cache:
            paths_by_tid[job.tracklet_id].append(job.crop_path)

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
    logger.info("STAGE 2b: кропов=%s, треклетов с ReID=%s", len(keys), len(tracklet_records))
