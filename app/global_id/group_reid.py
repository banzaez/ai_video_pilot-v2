"""ReID тела для узлов day_link: лучшие кадры трека камеры или группы треклетов."""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.config import Settings
from app.crops import TrackBestFramesPicker, TrackFrameCandidate, crop_person
from app.io.json_util import load_tracking_json
from app.pose.pose_service import PoseService
from app.progress import make_pbar
from app.reid import ReidExtractor, embed_with_cache_arrays
from app.session.manifest import is_session_manifest
from app.session.reader import SessionFrameReader
from app.tracklet.stage_reid import _advance_capture, _preselect_candidate_frames

logger = logging.getLogger(__name__)

DAY_GROUP_CROPS_DIR = "day_group_crops"
DAY_REID_CACHE_NAME = "day_reid.npz"


@dataclass
class TrackGroupReid:
    """ReID-результат одного узла day_link (track_id камеры)."""

    track_id: int
    embs: np.ndarray | None
    crop_files: list[str] = field(default_factory=list)
    n_tracklets: int = 1


@dataclass
class GroupReidContext:
    extractor: ReidExtractor | None
    picker: TrackBestFramesPicker
    available: bool


def make_group_reid_context(settings: Settings) -> GroupReidContext:
    extractor: ReidExtractor | None = None
    available = False
    try:
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
        if ok:
            available = True
        else:
            logger.warning("day_link group ReID недоступен: %s", reason)
            extractor = None
    except Exception as exc:
        logger.warning("day_link: ReidExtractor не загрузился: %s", exc)
        extractor = None

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
    except Exception as exc:
        logger.warning("day_link: PoseService не загрузился для отбора кадров: %s", exc)
        pose_service = None

    picker = TrackBestFramesPicker(
        pose_service=pose_service,
        pose_weight=settings.tracklet_reid_pose_weight,
        crowd_crop_penalty=settings.tracklet_reid_crowd_penalty,
        kpt_min=settings.tracklet_reid_min_completeness,
        crop_pad=settings.tracklet_reid_pad,
        feathering_enabled=settings.tracklet_reid_feathering_enabled,
        feathering_mode=settings.tracklet_reid_feathering_mode,
        feathering_sigma=settings.tracklet_reid_feathering_sigma,
        feathering_bone_thickness=settings.tracklet_reid_feathering_bone_thickness,
        feathering_bg_color=settings.tracklet_reid_feathering_bg_color,
    )
    return GroupReidContext(extractor=extractor, picker=picker, available=available)


def _load_session_manifest(session_root: str) -> dict[str, Any] | None:
    info_path = os.path.join(session_root, "info.json")
    if not os.path.isfile(info_path):
        return None
    info = load_tracking_json(info_path)
    return info if is_session_manifest(info) else None


def _track_ids_from_tracking(tracking_doc: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    for frame in tracking_doc.get("frames") or []:
        for det in frame.get("detections") or []:
            tid = int(det.get("track_id") or det.get("tracklet_id") or 0)
            if tid > 0:
                ids.add(tid)
    return ids


def _tracklet_groups(session_root: str, track_ids: set[int]) -> dict[int, list[int]]:
    """track_id → список исходных tracklet_id. Без links — соло [track_id]."""
    groups: dict[int, list[int]] = {tid: [tid] for tid in track_ids}
    links_path = os.path.join(session_root, "tracklet_links.json")
    if not os.path.isfile(links_path):
        return groups

    links_doc = load_tracking_json(links_path)
    raw_mapping = links_doc.get("tracklet_to_global") or {}
    if raw_mapping:
        inv: dict[int, list[int]] = {}
        for tl_str, gid in raw_mapping.items():
            try:
                inv.setdefault(int(gid), []).append(int(tl_str))
            except (TypeError, ValueError):
                continue
        for gid, tls in inv.items():
            if gid in groups and tls:
                groups[gid] = sorted(set(tls))
        return groups

    for gid, grp in enumerate(links_doc.get("groups") or [], start=1):
        tls = [int(x) for x in grp]
        if gid in groups and tls:
            groups[gid] = tls
    return groups


def _obs_from_tracklet_frames(
    frames_doc: dict[str, Any],
) -> dict[int, list[tuple[int, dict, list[dict]]]]:
    obs_by_tid: dict[int, list[tuple[int, dict, list[dict]]]] = defaultdict(list)
    for frame in frames_doc.get("frames") or []:
        fi = int(frame["frame_index"]) - 1
        dets = frame.get("detections") or []
        for det in dets:
            tid = int(det.get("tracklet_id") or det.get("track_id") or 0)
            if tid <= 0:
                continue
            other = [d for d in dets if int(d.get("tracklet_id") or d.get("track_id") or -1) != tid]
            obs_by_tid[tid].append((fi, det, other))
    return obs_by_tid


def _obs_from_tracking(
    tracking_doc: dict[str, Any],
) -> dict[int, list[tuple[int, dict, list[dict]]]]:
    obs_by_tid: dict[int, list[tuple[int, dict, list[dict]]]] = defaultdict(list)
    for frame in tracking_doc.get("frames") or []:
        fi = int(frame.get("frame_index", 0))
        dets = frame.get("detections") or []
        for det in dets:
            tid = int(det.get("track_id") or det.get("tracklet_id") or 0)
            if tid <= 0:
                continue
            other = [
                d
                for d in dets
                if int(d.get("track_id") or d.get("tracklet_id") or -1) != tid
            ]
            obs_by_tid[tid].append((fi, det, other))
    return obs_by_tid


def _resolve_video_path(session_root: str, meta: dict[str, Any]) -> str | None:
    for cand in (
        meta.get("video_source"),
        os.path.join(session_root, os.path.basename(str(meta.get("video_source") or ""))),
    ):
        if cand and os.path.isfile(str(cand)):
            return str(cand)
    return None


def _pick_best_frames(
    picker: TrackBestFramesPicker,
    candidates: list[TrackFrameCandidate],
    *,
    is_group: bool,
    top_k: int,
    batch_size: int,
    cache_path: str | None,
) -> list:
    if not candidates:
        return []
    if is_group:
        by_tid: dict[int, list[TrackFrameCandidate]] = defaultdict(list)
        for cand in candidates:
            by_tid[int(cand.tracklet_id or 0)].append(cand)
        return picker.pick_best_for_group(
            dict(by_tid),
            top_k=top_k,
            batch_size=batch_size,
            extract_faces=False,
            cache_path=cache_path,
        )
    return picker.pick_best_for_tracklet(
        candidates,
        top_k=top_k,
        batch_size=batch_size,
        extract_faces=False,
        cache_path=cache_path,
    )


def is_day_reid_cache_valid(
    session_root: str,
    settings: Settings,
    track_ids: set[int],
    top_k: int,
) -> tuple[bool, dict[str, Any] | None]:
    """Проверяет актуальность кэша day_reid.npz относительно tracking.json, links и настроек."""
    cache_path = os.path.join(session_root, DAY_REID_CACHE_NAME)
    if not os.path.isfile(cache_path):
        return False, None
    try:
        data = np.load(cache_path, allow_pickle=False)
        if "_meta" not in data:
            return False, None
        meta = json.loads(str(data["_meta"]))

        # 1. Проверяем состав треков
        cached_tids = set(int(x) for x in meta.get("track_ids", []))
        if cached_tids != track_ids:
            return False, None

        # 2. Проверяем параметры отбора кадров и модель ReID
        if int(meta.get("top_k", 0)) != top_k:
            return False, None
        if str(meta.get("backend", "")) != str(settings.tracklet_reid_backend):
            return False, None

        # 3. Проверяем mtime исходных файлов (tracking.json, tracklet_links.json, tracklet_frames.json)
        cache_mtime = os.path.getmtime(cache_path)
        for fname in ("tracking.json", "tracklet_links.json", "tracklet_frames.json", "feet.json"):
            fp = os.path.join(session_root, fname)
            if os.path.isfile(fp) and os.path.getmtime(fp) > cache_mtime:
                return False, None

        return True, meta
    except Exception as exc:
        logger.debug("Ошибка проверки кэша day_reid в %s: %s", session_root, exc)
        return False, None


def load_day_reid_cache(
    cache_path: str,
    meta: dict[str, Any],
    track_ids: set[int],
) -> dict[int, TrackGroupReid]:
    """Загружает TrackGroupReid из валидного day_reid.npz."""
    data = np.load(cache_path, allow_pickle=False)
    out: dict[int, TrackGroupReid] = {}
    crops_meta = meta.get("crops_by_tid", {})
    n_tracklets_meta = meta.get("n_tracklets_by_tid", {})
    for tid in sorted(track_ids):
        emb_key = f"emb_{tid}"
        embs = data[emb_key] if emb_key in data else None
        crops = crops_meta.get(str(tid), [])
        n_tl = n_tracklets_meta.get(str(tid), 1)
        out[tid] = TrackGroupReid(
            track_id=tid,
            embs=embs,
            crop_files=list(crops),
            n_tracklets=int(n_tl),
        )
    return out


def save_day_reid_cache(
    cache_path: str,
    out: dict[int, TrackGroupReid],
    track_ids: set[int],
    top_k: int,
    settings: Settings,
) -> None:
    """Сохраняет эмбеддинги и метаданные треков в day_reid.npz."""
    try:
        arrays_to_save: dict[str, Any] = {}
        crops_by_tid: dict[str, list[str]] = {}
        n_tracklets_by_tid: dict[str, int] = {}

        for tid, rec in out.items():
            if rec.embs is not None:
                arrays_to_save[f"emb_{tid}"] = rec.embs
            crops_by_tid[str(tid)] = rec.crop_files
            n_tracklets_by_tid[str(tid)] = rec.n_tracklets

        meta = {
            "track_ids": sorted(list(track_ids)),
            "top_k": top_k,
            "backend": str(settings.tracklet_reid_backend),
            "crops_by_tid": crops_by_tid,
            "n_tracklets_by_tid": n_tracklets_by_tid,
        }
        arrays_to_save["_meta"] = np.array(json.dumps(meta))
        np.savez_compressed(cache_path, **arrays_to_save)
    except Exception as exc:
        logger.warning("Не удалось сохранить кэш day_reid: %s", exc)


def embed_session_tracks(
    session_root: str,
    settings: Settings,
    ctx: GroupReidContext | None,
    *,
    session_key: str = "",
    force_recompute: bool = False,
) -> dict[int, TrackGroupReid]:
    """Для каждого track_id сессии выбирает лучшие кадры и считает ReID (без лиц)."""
    tracking_path = os.path.join(session_root, "tracking.json")
    if not os.path.isfile(tracking_path):
        return {}

    tracking_doc = load_tracking_json(tracking_path)
    track_ids = _track_ids_from_tracking(tracking_doc)
    if not track_ids:
        return {}

    top_k = max(1, int(settings.day_link_top_k) or int(settings.tracklet_reid_top_k) or 3)
    cache_path = os.path.join(session_root, DAY_REID_CACHE_NAME)

    if not force_recompute:
        is_valid, meta = is_day_reid_cache_valid(session_root, settings, track_ids, top_k)
        if is_valid and meta is not None:
            logger.info(
                "  - Session %s: ReID загружен из кэша (%d треков)",
                session_key or os.path.basename(session_root),
                len(track_ids),
            )
            return load_day_reid_cache(cache_path, meta, track_ids)

    if ctx is None or not ctx.available:
        logger.warning(
            "day_link: ReID контекст недоступен для пересчета сессии %s",
            session_key or os.path.basename(session_root),
        )
        return {tid: TrackGroupReid(track_id=tid, embs=None, crop_files=[], n_tracklets=1) for tid in track_ids}

    groups = _tracklet_groups(session_root, track_ids)
    frames_path = os.path.join(session_root, "tracklet_frames.json")
    if os.path.isfile(frames_path):
        obs_by_tid = _obs_from_tracklet_frames(load_tracking_json(frames_path))
        tracking_obs = _obs_from_tracking(tracking_doc)
    else:
        obs_by_tid = _obs_from_tracking(tracking_doc)
        tracking_obs = obs_by_tid

    top_k = max(1, int(settings.day_link_top_k) or int(settings.tracklet_reid_top_k) or 3)
    pad = float(getattr(settings, "tracklet_reid_pad", 0.04) or 0.04)
    batch_size = max(1, int(settings.tracklet_reid_batch_size))
    cache_path = os.path.join(session_root, "tracklet_pose_cache.json")
    save_crops = bool(getattr(settings, "day_link_save_crops", True))
    crops_out = os.path.join(session_root, DAY_GROUP_CROPS_DIR)
    if save_crops:
        os.makedirs(crops_out, exist_ok=True)

    obs_for_track: dict[int, list[tuple[int, dict, list[dict]]]] = {}
    for tid in sorted(track_ids):
        tls = groups.get(tid) or [tid]
        pooled: list[tuple[int, dict, list[dict]]] = []
        for tl in tls:
            pooled.extend(obs_by_tid.get(tl) or [])
        if not pooled:
            pooled = list(tracking_obs.get(tid) or [])
        obs_for_track[tid] = pooled

    cands_by_frame: dict[int, list[tuple[int, int, dict, list[dict]]]] = defaultdict(list)
    for tid, obs in obs_for_track.items():
        if not obs:
            continue
        tls = groups.get(tid) or [tid]
        if len(tls) > 1:
            by_tl: dict[int, list[tuple[int, dict, list[dict]]]] = defaultdict(list)
            for fi, det, other in obs:
                tl = int(det.get("tracklet_id") or det.get("track_id") or 0)
                by_tl[tl].append((fi, det, other))
            picked: list[tuple[int, dict, list[dict]]] = []
            for tl, tl_obs in by_tl.items():
                picked.extend(_preselect_candidate_frames(tl_obs, top_k=top_k, multiplier=3))
        else:
            picked = _preselect_candidate_frames(obs, top_k=top_k, multiplier=3)
        for fi, det, other in picked:
            tl = int(det.get("tracklet_id") or det.get("track_id") or tid)
            cands_by_frame[fi].append((tid, tl, det, other))

    candidates_by_track: dict[int, list[TrackFrameCandidate]] = defaultdict(list)
    sorted_frames = sorted(cands_by_frame.keys())
    manifest = _load_session_manifest(session_root)
    video_path = None if manifest else _resolve_video_path(session_root, tracking_doc)

    def _process_frame(frame: np.ndarray, fi: int) -> None:
        fh, fw = frame.shape[:2]
        for tid, tl, det, other_dets in cands_by_frame[fi]:
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            crop, roi = crop_person(frame, bbox, pad=pad)
            candidates_by_track[tid].append(
                TrackFrameCandidate(
                    frame_index=fi,
                    target_det=det,
                    all_dets=other_dets,
                    crop_image=np.ascontiguousarray(crop),
                    crop_roi=roi,
                    frame_w=fw,
                    frame_h=fh,
                    tracklet_id=tl,
                )
            )

    if sorted_frames:
        label = session_key or os.path.basename(session_root)
        pbar = make_pbar(
            total=len(sorted_frames),
            desc=f"[day_link ReID {label}: read]",
            unit="frame",
        )
        try:
            if manifest:
                reader = SessionFrameReader(manifest)
                try:
                    for fi in sorted_frames:
                        try:
                            frame = reader.read_frame(fi)
                        except Exception as exc:
                            logger.warning("day_link: кадр %s сессии %s: %s", fi, label, exc)
                            pbar.update(1)
                            continue
                        if frame is not None:
                            _process_frame(frame, fi)
                        pbar.update(1)
                finally:
                    reader.close()
            elif video_path:
                from app.parallel_tracker import open_video_capture

                cap = open_video_capture(str(video_path))
                last_pos: int | None = None
                try:
                    for fi in sorted_frames:
                        ret, frame, last_pos = _advance_capture(cap, fi, last_pos=last_pos)
                        if ret and frame is not None:
                            _process_frame(frame, fi)
                        pbar.update(1)
                finally:
                    cap.release()
            else:
                logger.warning("day_link: нет видео для сессии %s, ReID пропущен", label)
        finally:
            pbar.close()

    all_candidates: list[TrackFrameCandidate] = []
    for tid in sorted(candidates_by_track):
        all_candidates.extend(candidates_by_track[tid])
    if all_candidates:
        ctx.picker.score_candidates_batch(
            all_candidates,
            batch_size=batch_size,
            extract_faces=False,
            show_pbar=True,
            pbar_desc=f"[day_link ReID {session_key or os.path.basename(session_root)}: pose]",
            cache_path=cache_path,
            prune_cache=False,
        )

    all_keys: list[str] = []
    all_images: list[np.ndarray] = []
    keys_by_tid: dict[int, list[str]] = defaultdict(list)
    files_by_tid: dict[int, list[str]] = defaultdict(list)

    for tid in sorted(track_ids):
        cands = candidates_by_track.get(tid) or []
        tls = groups.get(tid) or [tid]
        best = _pick_best_frames(
            ctx.picker,
            cands,
            is_group=len(tls) > 1,
            top_k=top_k,
            batch_size=batch_size,
            cache_path=cache_path,
        )
        for ki, scored in enumerate(best):
            fi = scored.frame_index
            fname = f"g_{tid:04d}_k{ki}_f{fi + 1}.jpg"
            crop_path = os.path.join(crops_out, fname)
            crop = scored.crop_image
            if crop is None:
                crop = np.zeros((10, 10, 3), dtype=np.uint8)
            if save_crops:
                cv2.imwrite(crop_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            all_keys.append(crop_path)
            all_images.append(crop)
            keys_by_tid[tid].append(crop_path)
            files_by_tid[tid].append(fname)

    cache: dict[str, np.ndarray] = {}
    if ctx.available and ctx.extractor is not None and all_keys:
        cache = embed_with_cache_arrays(
            ctx.extractor,
            all_keys,
            all_images,
            {},
            batch_size=batch_size,
        )

    out: dict[int, TrackGroupReid] = {}
    for tid in sorted(track_ids):
        keys = keys_by_tid.get(tid) or []
        embs_rows: list[np.ndarray] = []
        for key in keys:
            arr = cache.get(key)
            if arr is None:
                continue
            vec = np.asarray(arr, dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(vec))
            if n <= 1e-9:
                continue
            embs_rows.append(vec / n)
        mat = np.stack(embs_rows, axis=0) if embs_rows else None
        out[tid] = TrackGroupReid(
            track_id=tid,
            embs=mat,
            crop_files=list(files_by_tid.get(tid) or []),
            n_tracklets=len(groups.get(tid) or [tid]),
        )

    save_day_reid_cache(cache_path, out, track_ids, top_k, settings)
    return out
