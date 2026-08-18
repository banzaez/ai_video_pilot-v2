"""Модуль извлечения лиц и эмбеддингов через InsightFace для групп и треклов."""

from __future__ import annotations

import logging
import os
import warnings
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

# Подавление устаревшего вызова estimate() внутри сторонней библиотеки insightface
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")

from app.config import Settings, face_crops_dir, video_work_dir
from app.global_id.stage_pose import load_pose_lookup
from app.pose import get_pose_service
from app.pose.types import PoseResult, pose_completeness
from app.progress import make_pbar

logger = logging.getLogger(__name__)

_FACE_APP_CACHE: dict[str, Any] = {}

_FACE_IOU_MIN = 0.05
_FACE_MIN_DET = 0.45
_POSE_CACHE_COMPLETE = 0.5


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-нормализация вектора или матрицы по последней оси."""
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return vec / norm


def face_models_for_settings(settings: Settings) -> list[str]:
    """Список моделей InsightFace для сравнения (первый — primary)."""
    models = list(getattr(settings, "camera_link_face_models", ()) or ())
    if models:
        return [str(m) for m in models if str(m).strip()]
    primary = str(getattr(settings, "camera_link_model", "") or "buffalo_l").strip()
    return [primary or "buffalo_l"]


def get_face_analysis(model_name: str = "buffalo_l") -> Any:
    """Ленивая инициализация FaceAnalysis из insightface."""
    if model_name in _FACE_APP_CACHE:
        return _FACE_APP_CACHE[model_name]

    import shutil
    from insightface.app import FaceAnalysis

    home_models = os.path.expanduser(f"~/.insightface/models/{model_name}")
    nested_sub = os.path.join(home_models, model_name)
    if os.path.isdir(nested_sub):
        try:
            for item in os.listdir(nested_sub):
                src = os.path.join(nested_sub, item)
                dst = os.path.join(home_models, item)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
            shutil.rmtree(nested_sub, ignore_errors=True)
        except Exception as exc:
            logger.debug("Не удалось переместить вложенные файлы модели %s: %s", model_name, exc)

    providers = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    app = None
    last_err: Exception | None = None
    for prov in [providers, ["CPUExecutionProvider"]]:
        try:
            app = FaceAnalysis(name=model_name, allowed_modules=["detection", "recognition"], providers=prov)
            app.prepare(ctx_id=0, det_size=(640, 640))
            break
        except Exception as e:
            last_err = e
            logger.warning("Не удалось инициализировать FaceAnalysis(%s) с %s: %r", model_name, prov, e)
            app = None

    if app is None:
        raise RuntimeError(f"Не удалось инициализировать InsightFace FaceAnalysis ({model_name}): {last_err}")

    _FACE_APP_CACHE[model_name] = app
    return app


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _face_center_in_bbox(face_bbox: tuple[float, float, float, float], person_bbox: list[float]) -> bool:
    fx1, fy1, fx2, fy2 = face_bbox
    px1, py1, px2, py2 = person_bbox[:4]
    cx = (fx1 + fx2) / 2.0
    cy = (fy1 + fy2) / 2.0
    return px1 <= cx <= px2 and py1 <= cy <= py2


def _pose_from_lookup(rec: dict[str, Any]) -> PoseResult | None:
    kxy = rec.get("kxy")
    kcf = rec.get("kcf")
    if not isinstance(kxy, list) or not isinstance(kcf, list) or not kxy:
        return None
    bbox = rec.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    return PoseResult(
        bbox=[float(v) for v in bbox[:4]],
        confidence=float(rec.get("confidence", 0.5)),
        kxy=kxy,
        kcf=[float(c) for c in kcf],
    )


def _pose_completeness_in_lookup(
    pose_lookup: dict[int, dict[int, dict[str, Any]]],
    track_id: int,
    frame_index: int,
    min_conf: float,
) -> float:
    rec = pose_lookup.get(track_id, {}).get(frame_index)
    if not rec:
        return 0.0
    kcf = rec.get("kcf")
    if not isinstance(kcf, list):
        return 0.0
    return pose_completeness(kcf, min_conf)


@dataclass
class FaceCandidate:
    group_id: int
    track_id: int
    frame_index: int
    bbox: list[float]  # [x1, y1, x2, y2]
    height: float
    is_solo: bool


@dataclass
class _DetectedFace:
    det_score: float
    embedding: np.ndarray
    face_bbox: tuple[float, float, float, float]


@dataclass
class _BufferedFace:
    quality: float
    group_id: int
    track_id: int
    frame_index: int
    det_score: float
    pose_face_score: float
    pose_conf: float
    face_bbox: tuple[float, float, float, float]
    embedding: np.ndarray
    model: str
    is_solo: bool
    jpeg_bytes: bytes | None


def _pick_spaced_candidates(
    cands: list[FaceCandidate],
    *,
    pose_lookup: dict[int, dict[int, dict[str, Any]]],
    min_gap_frames: int,
    kpt_min: float,
) -> list[FaceCandidate]:
    """Сортирует по полноте позы/высоте и жадно отбирает кадры с минимальным зазором."""
    ordered = sorted(
        cands,
        key=lambda c: (
            -_pose_completeness_in_lookup(pose_lookup, c.group_id, c.frame_index, kpt_min),
            -c.height,
        ),
    )
    picked: list[FaceCandidate] = []
    gap = max(1, int(min_gap_frames))
    for cand in ordered:
        if all(abs(cand.frame_index - p.frame_index) >= gap for p in picked):
            picked.append(cand)
    return picked


def _round_robin_quota(queues: list[list[FaceCandidate]], max_attempts: int) -> list[FaceCandidate]:
    """По одному кандидату из каждого трека по кругу; остаток уходит живым очередям."""
    live = [deque(q) for q in queues if q]
    out: list[FaceCandidate] = []
    qi = 0
    while len(out) < max_attempts and live:
        q = live[qi]
        out.append(q.popleft())
        if not q:
            live.pop(qi)
            if not live:
                break
            qi %= len(live)
        else:
            qi = (qi + 1) % len(live)
    return out


def _collect_group_candidates(
    frames_data: list[dict[str, Any]],
    *,
    solo_global_ids: set[int],
    max_attempts: int,
    pose_lookup: dict[int, dict[int, dict[str, Any]]] | None = None,
    min_gap_frames: int = 1,
    use_tracklet_ids: bool = False,
    tracklet_to_global: dict[int, int] | None = None,
    kpt_min: float = 0.25,
) -> dict[int, list[FaceCandidate]]:
    """Собирает кандидатов по global track_id; квота round-robin по трекам группы."""
    mapping = tracklet_to_global or {}
    lookup = pose_lookup or {}
    group_dets: dict[int, list[FaceCandidate]] = {}

    for f in frames_data:
        fi = int(f.get("frame_index", 0))
        for d in f.get("detections", []):
            tid = int(d.get("track_id") or d.get("tracklet_id") or 0)
            if tid <= 0:
                continue
            if use_tracklet_ids:
                gid = mapping.get(tid, tid)
                frag_id = tid
            else:
                gid = int(d.get("track_id") or tid)
                frag_id = int(d.get("tracklet_id") or gid)
            bbox = d.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            h = max(0.0, y2 - y1)
            if h <= 20.0:
                continue
            cand = FaceCandidate(
                group_id=gid,
                track_id=frag_id,
                frame_index=fi,
                bbox=[x1, y1, x2, y2],
                height=h,
                is_solo=gid in solo_global_ids,
            )
            group_dets.setdefault(gid, []).append(cand)

    selected: dict[int, list[FaceCandidate]] = {}
    for gid, cands in group_dets.items():
        by_track: dict[int, list[FaceCandidate]] = defaultdict(list)
        for cand in cands:
            by_track[cand.track_id].append(cand)
        queues = [
            _pick_spaced_candidates(
                track_cands,
                pose_lookup=lookup,
                min_gap_frames=min_gap_frames,
                kpt_min=kpt_min,
            )
            for track_cands in by_track.values()
        ]
        selected[gid] = _round_robin_quota(queues, max_attempts)
    return selected


def _person_head_crop(
    frame_img: np.ndarray,
    cand: FaceCandidate,
) -> tuple[np.ndarray, int, int] | None:
    img_h, img_w = frame_img.shape[:2]
    x1, y1, x2, y2 = cand.bbox
    bw = x2 - x1
    bh = y2 - y1
    head_y2 = y1 + bh * 0.65
    px = bw * 0.15
    py = bh * 0.10
    ix1 = max(0, int(round(x1 - px)))
    iy1 = max(0, int(round(y1 - py)))
    ix2 = min(img_w, int(round(x2 + px)))
    iy2 = min(img_h, int(round(head_y2 + py)))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    person_crop = frame_img[iy1:iy2, ix1:ix2]
    if person_crop.size == 0 or person_crop.shape[0] < 20 or person_crop.shape[1] < 20:
        return None
    return person_crop, ix1, iy1


def _pick_face_for_person(
    faces: list[Any],
    person_bbox: list[float],
    *,
    offset_x: int,
    offset_y: int,
    min_det: float = _FACE_MIN_DET,
) -> tuple[Any, tuple[float, float, float, float]] | None:
    if not faces:
        return None

    scored: list[tuple[float, Any, tuple[float, float, float, float]]] = []
    person_area = max(1.0, (person_bbox[2] - person_bbox[0]) * (person_bbox[3] - person_bbox[1]))

    for fc in faces:
        if fc.embedding is None or float(fc.det_score) < min_det:
            continue
        fb = (
            offset_x + float(fc.bbox[0]),
            offset_y + float(fc.bbox[1]),
            offset_x + float(fc.bbox[2]),
            offset_y + float(fc.bbox[3]),
        )
        iou = _bbox_iou(fb, person_bbox)
        if iou < _FACE_IOU_MIN and not _face_center_in_bbox(fb, person_bbox):
            continue
        area = max(0.0, (fb[2] - fb[0]) * (fb[3] - fb[1]))
        norm_area = min(1.0, area / person_area)
        score = float(fc.det_score) * 0.7 + norm_area * 0.3
        scored.append((score, fc, fb))

    if scored:
        _, best_face, fb = max(scored, key=lambda row: row[0])
        return best_face, fb

    valid = [fc for fc in faces if fc.embedding is not None and float(fc.det_score) >= min_det]
    if not valid:
        return None
    best_face = max(valid, key=lambda fc: float(fc.det_score))
    fb = (
        offset_x + float(best_face.bbox[0]),
        offset_y + float(best_face.bbox[1]),
        offset_x + float(best_face.bbox[2]),
        offset_y + float(best_face.bbox[3]),
    )
    return best_face, fb


def _detect_best_face(
    face_app: Any,
    person_crop: np.ndarray,
    person_bbox: list[float],
    *,
    offset_x: int,
    offset_y: int,
) -> _DetectedFace | None:
    faces = face_app.get(person_crop)
    picked = _pick_face_for_person(faces, person_bbox, offset_x=offset_x, offset_y=offset_y)
    if picked is None:
        return None
    best_face, (fx1, fy1, fx2, fy2) = picked
    emb = l2_normalize(best_face.embedding.astype(np.float32))
    return _DetectedFace(
        det_score=float(best_face.det_score),
        embedding=emb,
        face_bbox=(fx1, fy1, fx2, fy2),
    )


def _detect_faces_parallel(
    face_apps: dict[str, Any],
    person_crop: np.ndarray,
    person_bbox: list[float],
    *,
    offset_x: int,
    offset_y: int,
    executor: ThreadPoolExecutor | None = None,
) -> dict[str, _DetectedFace]:
    if len(face_apps) <= 1 or executor is None:
        out: dict[str, _DetectedFace] = {}
        for model_name, app in face_apps.items():
            hit = _detect_best_face(app, person_crop, person_bbox, offset_x=offset_x, offset_y=offset_y)
            if hit is not None:
                out[model_name] = hit
        return out

    out: dict[str, _DetectedFace] = {}
    futures = {
        model_name: executor.submit(
            _detect_best_face, app, person_crop, person_bbox, offset_x=offset_x, offset_y=offset_y
        )
        for model_name, app in face_apps.items()
    }
    for model_name, fut in futures.items():
        hit = fut.result()
        if hit is not None:
            out[model_name] = hit
    return out


def _encode_face_jpeg(frame_img: np.ndarray, face_bbox: tuple[float, float, float, float]) -> bytes | None:
    img_h, img_w = frame_img.shape[:2]
    fx1, fy1, fx2, fy2 = face_bbox
    fbw = fx2 - fx1
    fbh = fy2 - fy1
    f_ix1 = max(0, int(round(fx1 - fbw * 0.15)))
    f_iy1 = max(0, int(round(fy1 - fbh * 0.15)))
    f_ix2 = min(img_w, int(round(fx2 + fbw * 0.15)))
    f_iy2 = min(img_h, int(round(fy2 + fbh * 0.15)))
    if f_ix2 <= f_ix1 or f_iy2 <= f_iy1:
        return None
    face_crop_img = frame_img[f_iy1:f_iy2, f_ix1:f_ix2]
    if face_crop_img.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", face_crop_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


def _buffer_key(is_solo: bool, group_id: int, model: str) -> tuple[bool, int, str]:
    return (is_solo, group_id, model)


def _buffer_push(buf: list[_BufferedFace], slot: _BufferedFace, top_k: int, dup_cos: float) -> None:
    """Дедуп по косинусу эмбеддинга + покрытие всех треков в top_k."""
    for i, ex in enumerate(buf):
        if float(np.dot(ex.embedding, slot.embedding)) >= dup_cos:
            if slot.quality > ex.quality:
                buf[i] = slot
            return

    if len(buf) < top_k:
        buf.append(slot)
        return

    counts = Counter(s.track_id for s in buf)
    over = [i for i, s in enumerate(buf) if counts[s.track_id] > 1]

    if slot.track_id not in counts and over:
        buf[min(over, key=lambda i: buf[i].quality)] = slot
        return

    pool = over or list(range(len(buf)))
    worst_i = min(pool, key=lambda i: buf[i].quality)
    if slot.quality > buf[worst_i].quality:
        buf[worst_i] = slot


def _resolve_pose_for_candidates(
    pose_service: Any,
    frame_img: np.ndarray,
    cands: list[FaceCandidate],
    pose_lookup: dict[int, dict[int, dict[str, Any]]],
    kpt_min: float,
) -> list[tuple[FaceCandidate, PoseResult | None, float]]:
    """Сначала самая полная поза, затем face_confidence."""
    out: list[tuple[FaceCandidate, PoseResult | None, float]] = []
    need_yolo: list[FaceCandidate] = []
    need_indices: list[int] = []
    cached_poses: dict[int, PoseResult] = {}

    for cand in cands:
        cached = pose_lookup.get(cand.group_id, {}).get(cand.frame_index)
        pose = _pose_from_lookup(cached) if cached else None
        if pose is not None and pose.completeness(min_conf=kpt_min) >= _POSE_CACHE_COMPLETE:
            out.append((cand, pose, pose.face_confidence(min_conf=kpt_min)))
            continue
        if pose is not None:
            cached_poses[len(out)] = pose
        need_yolo.append(cand)
        need_indices.append(len(out))
        out.append((cand, pose, pose.face_confidence(min_conf=kpt_min) if pose is not None else 0.0))

    if need_yolo:
        rows = pose_service.pose_faces_for_bboxes(
            frame_img,
            [c.bbox for c in need_yolo],
            kpt_min=kpt_min,
        )
        for idx, (yolo_pose, yolo_score) in zip(need_indices, rows):
            cand = out[idx][0]
            cached = cached_poses.get(idx)
            best = yolo_pose
            score = yolo_score
            if cached is not None:
                if yolo_pose is None or cached.completeness(min_conf=kpt_min) >= yolo_pose.completeness(
                    min_conf=kpt_min
                ):
                    best = cached
                    score = cached.face_confidence(min_conf=kpt_min)
            out[idx] = (cand, best, score)

    return out


def extract_faces_for_groups(
    settings: Settings,
    *,
    frames_data: list[dict[str, Any]],
    solo_global_ids: set[int] | None = None,
    video_source_path: str | None = None,
    fps: float = 25.0,
    use_tracklet_ids: bool = False,
    tracklet_to_global: dict[int, int] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """
    Извлекает лица для групп (склейки) и solo-треков.
    tracking.json: track_id = global/group id; solo → tracks[t], группа → groups[g].
    """
    models = face_models_for_settings(settings)
    primary_model = models[0]
    face_apps = {m: get_face_analysis(m) for m in models}
    pose_service = get_pose_service(settings)
    top_k = max(1, int(settings.camera_link_face_top_k))
    max_attempts = max(1, int(settings.camera_link_face_max_attempts))
    save_crops = bool(settings.camera_link_save_face_crops)
    crops_dir = face_crops_dir(settings)
    multi_model = len(models) > 1
    solo_ids = solo_global_ids or set()
    kpt_min = float(settings.pose_kpt_min)
    min_gap_sec = float(settings.camera_link_face_min_gap_sec)
    min_gap_frames = max(1, int(round(min_gap_sec * max(1.0, fps))))
    dup_cos = float(settings.camera_link_face_dup_cos)

    pose_lookup = load_pose_lookup(video_work_dir(settings))

    if save_crops:
        if os.path.isdir(crops_dir):
            for fname in os.listdir(crops_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    try:
                        os.remove(os.path.join(crops_dir, fname))
                    except OSError:
                        pass
        os.makedirs(crops_dir, exist_ok=True)

    candidates_by_group = _collect_group_candidates(
        frames_data,
        solo_global_ids=solo_ids,
        max_attempts=max_attempts,
        pose_lookup=pose_lookup,
        min_gap_frames=min_gap_frames,
        use_tracklet_ids=use_tracklet_ids,
        tracklet_to_global=tracklet_to_global,
        kpt_min=kpt_min,
    )

    cands_by_frame: dict[int, list[FaceCandidate]] = {}
    for _gid, cands in candidates_by_group.items():
        for c in cands:
            cands_by_frame.setdefault(c.frame_index, []).append(c)

    face_buffers: dict[tuple[bool, int, str], list[_BufferedFace]] = {}

    from app.tracklet.common import session_manifest
    from app.session.reader import SessionFrameReader
    from app.parallel_tracker import open_video_capture

    manifest = session_manifest(settings)
    session_reader: SessionFrameReader | None = None
    cap: cv2.VideoCapture | None = None

    if manifest:
        session_reader = SessionFrameReader(manifest)
    elif video_source_path and os.path.isfile(video_source_path):
        cap = open_video_capture(video_source_path)
    else:
        logger.error("Видео не найдено для извлечения лиц (input=%s)", settings.input_path)
        return {"groups": {}, "tracks": {}, "groups_by_model": {}, "tracks_by_model": {}}, {}

    sorted_frame_indices = sorted(cands_by_frame.keys())
    current_frame_pos = -1
    pbar = make_pbar(total=len(sorted_frame_indices), desc="[STAGE Face: InsightFace]", unit="frame")
    executor = ThreadPoolExecutor(max_workers=len(face_apps)) if len(face_apps) > 1 else None

    try:
        for fi in sorted_frame_indices:
            pbar.update(1)
            target_0based = max(0, fi - 1)
            cands_on_frame = cands_by_frame[fi]

            frame_img: np.ndarray | None = None
            if session_reader is not None:
                try:
                    frame_img = session_reader.read_frame(target_0based)
                except Exception as e:
                    logger.debug("Кадр %s не прочитан из session: %s", target_0based, e)
                    continue
            elif cap is not None:
                if current_frame_pos >= 0 and 0 < (target_0based - current_frame_pos) <= 5:
                    while current_frame_pos < target_0based:
                        ret, frame_img = cap.read()
                        current_frame_pos += 1
                        if not ret:
                            break
                    if not ret or current_frame_pos != target_0based:
                        continue
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_0based)
                    ret, frame_img = cap.read()
                    current_frame_pos = target_0based
                    if not ret:
                        current_frame_pos = -1
                        continue

            if frame_img is None or frame_img.size == 0:
                continue

            # Дешёвая проверка head-crop до позы
            head_ok: list[tuple[FaceCandidate, np.ndarray, int, int]] = []
            for cand in cands_on_frame:
                head = _person_head_crop(frame_img, cand)
                if head is None:
                    continue
                person_crop, ix1, iy1 = head
                head_ok.append((cand, person_crop, ix1, iy1))

            if not head_ok:
                continue

            pose_rows = _resolve_pose_for_candidates(
                pose_service,
                frame_img,
                [c for c, _, _, _ in head_ok],
                pose_lookup,
                kpt_min,
            )

            for (cand, person_crop, ix1, iy1), (cand2, pose, pose_face_score) in zip(head_ok, pose_rows):
                if cand is not cand2:
                    continue
                if pose is None or pose_face_score <= 0.0:
                    continue

                detected = _detect_faces_parallel(
                    face_apps,
                    person_crop,
                    cand.bbox,
                    offset_x=ix1,
                    offset_y=iy1,
                    executor=executor,
                )
                if not detected:
                    continue

                for model_name, hit in detected.items():
                    quality = float(hit.det_score) * float(pose_face_score)
                    jpeg_bytes = _encode_face_jpeg(frame_img, hit.face_bbox) if save_crops else None
                    slot = _BufferedFace(
                        quality=quality,
                        group_id=cand.group_id,
                        track_id=cand.track_id,
                        frame_index=cand.frame_index,
                        det_score=float(hit.det_score),
                        pose_face_score=float(pose_face_score),
                        pose_conf=round(float(pose.confidence), 4),
                        face_bbox=hit.face_bbox,
                        embedding=hit.embedding,
                        model=model_name,
                        is_solo=cand.is_solo,
                        jpeg_bytes=jpeg_bytes,
                    )
                    key = _buffer_key(cand.is_solo, cand.group_id, model_name)
                    face_buffers.setdefault(key, [])
                    _buffer_push(face_buffers[key], slot, top_k, dup_cos)

    finally:
        if session_reader is not None:
            session_reader.close()
        if cap is not None:
            cap.release()
        pbar.close()
        if executor:
            executor.shutdown(wait=False)

    group_faces_by_model: dict[str, dict[str, list[dict[str, Any]]]] = {m: {} for m in models}
    track_faces_by_model: dict[str, dict[str, list[dict[str, Any]]]] = {m: {} for m in models}
    group_embeddings_by_model: dict[str, dict[str, list[np.ndarray]]] = {m: {} for m in models}
    track_embeddings_by_model: dict[str, dict[str, list[np.ndarray]]] = {m: {} for m in models}

    for (is_solo, gid, model_name), slots in face_buffers.items():
        slots.sort(key=lambda s: -s.quality)
        key_str = str(gid)
        prefix = "t" if is_solo else "g"
        suffix = f"_{model_name}" if multi_model else ""
        faces_store = track_faces_by_model if is_solo else group_faces_by_model
        embs_store = track_embeddings_by_model if is_solo else group_embeddings_by_model

        entries: list[dict[str, Any]] = []
        embs: list[np.ndarray] = []
        for rank_idx, slot in enumerate(slots):
            crop_filename = f"face_{prefix}{gid:04d}_k{rank_idx}_f{slot.frame_index}{suffix}.jpg"
            crop_out_path = os.path.join(crops_dir, crop_filename)
            if save_crops and slot.jpeg_bytes:
                with open(crop_out_path, "wb") as f:
                    f.write(slot.jpeg_bytes)

            fx1, fy1, fx2, fy2 = slot.face_bbox
            entry = {
                "group_id": slot.group_id,
                "track_id": slot.track_id,
                "rank": rank_idx,
                "frame_index": slot.frame_index,
                "model": model_name,
                "det_score": round(slot.det_score, 4),
                "pose_face_score": round(slot.pose_face_score, 4),
                "pose_conf": slot.pose_conf,
                "quality": round(slot.quality, 4),
                "face_bbox": [round(fx1, 1), round(fy1, 1), round(fx2, 1), round(fy2, 1)],
                "crop_file": crop_filename,
                "solo": slot.is_solo,
            }
            entries.append(entry)
            embs.append(slot.embedding)

        if entries:
            faces_store[model_name][key_str] = entries
            embs_store[model_name][key_str] = embs

    final_npz_arrays: dict[str, np.ndarray] = {}

    def _write_npz(
        bucket: dict[str, dict[str, list[np.ndarray]]],
        prefix: str,
    ) -> None:
        for model_name in models:
            for id_str, emb_list in bucket[model_name].items():
                if not emb_list:
                    continue
                key = f"{prefix}_{id_str}_{model_name}" if multi_model else f"{prefix}_{id_str}"
                final_npz_arrays[key] = np.vstack(emb_list).astype(np.float32)
                if model_name == primary_model:
                    final_npz_arrays[f"{prefix}_{id_str}"] = final_npz_arrays[key]

    _write_npz(group_embeddings_by_model, "group")
    _write_npz(track_embeddings_by_model, "track")

    primary_groups = group_faces_by_model.get(primary_model, {})
    primary_tracks = track_faces_by_model.get(primary_model, {})
    meta_payload = {
        "stage": "camera_face",
        "model": primary_model,
        "models": models,
        "n_groups_with_faces": len(primary_groups),
        "n_tracks_with_faces": len(primary_tracks),
        "groups": primary_groups,
        "tracks": primary_tracks,
        "groups_by_model": group_faces_by_model,
        "tracks_by_model": track_faces_by_model,
        "solo_global_ids": sorted(solo_ids),
    }

    return meta_payload, final_npz_arrays
