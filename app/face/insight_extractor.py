"""Модуль извлечения лиц и эмбеддингов через InsightFace для групп и треклов."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from app.config import Settings, face_crops_dir
from app.pose import get_pose_service
from app.progress import make_pbar

logger = logging.getLogger(__name__)

_FACE_APP_CACHE: dict[str, Any] = {}


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

    from insightface.app import FaceAnalysis

    providers = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    app = None
    for prov in [providers, ["CPUExecutionProvider"]]:
        try:
            app = FaceAnalysis(name=model_name, allowed_modules=["detection", "recognition"], providers=prov)
            app.prepare(ctx_id=0, det_size=(640, 640))
            break
        except Exception as e:
            logger.warning("Не удалось инициализировать FaceAnalysis(%s) с %s: %s", model_name, prov, e)
            app = None

    if app is None:
        raise RuntimeError(f"Не удалось инициализировать InsightFace FaceAnalysis ({model_name})")

    _FACE_APP_CACHE[model_name] = app
    return app


@dataclass
class FaceCandidate:
    group_id: int
    track_id: int
    frame_index: int
    bbox: list[float]  # [x1, y1, x2, y2]
    height: float


@dataclass
class _DetectedFace:
    det_score: float
    embedding: np.ndarray
    face_bbox: tuple[float, float, float, float]


def _collect_group_candidates(
    frames_data: list[dict[str, Any]],
    track_to_group: dict[int, int],
    *,
    max_attempts: int,
) -> dict[int, list[FaceCandidate]]:
    """Группирует детекции по group_id и сортирует по высоте bbox (от большего к меньшему)."""
    group_dets: dict[int, list[FaceCandidate]] = {}

    for f in frames_data:
        fi = int(f.get("frame_index", 0))
        for d in f.get("detections", []):
            tid = int(d.get("track_id") or d.get("tracklet_id") or 0)
            if tid <= 0:
                continue
            gid = track_to_group.get(tid, tid)
            bbox = d.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            h = max(0.0, y2 - y1)
            if h <= 20.0:
                continue
            cand = FaceCandidate(
                group_id=gid,
                track_id=tid,
                frame_index=fi,
                bbox=[x1, y1, x2, y2],
                height=h,
            )
            group_dets.setdefault(gid, []).append(cand)

    for gid in group_dets:
        group_dets[gid].sort(key=lambda c: -c.height)
        if len(group_dets[gid]) > max_attempts:
            group_dets[gid] = group_dets[gid][:max_attempts]

    return group_dets


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


def _detect_best_face(face_app: Any, person_crop: np.ndarray, *, offset_x: int, offset_y: int) -> _DetectedFace | None:
    faces = face_app.get(person_crop)
    if not faces:
        return None
    best_face = max(faces, key=lambda fc: (fc.bbox[2] - fc.bbox[0]) * (fc.bbox[3] - fc.bbox[1]))
    if best_face.det_score < 0.45 or best_face.embedding is None:
        return None
    fx1 = offset_x + float(best_face.bbox[0])
    fy1 = offset_y + float(best_face.bbox[1])
    fx2 = offset_x + float(best_face.bbox[2])
    fy2 = offset_y + float(best_face.bbox[3])
    emb = l2_normalize(best_face.embedding.astype(np.float32))
    return _DetectedFace(
        det_score=float(best_face.det_score),
        embedding=emb,
        face_bbox=(fx1, fy1, fx2, fy2),
    )


def _detect_faces_parallel(
    face_apps: dict[str, Any],
    person_crop: np.ndarray,
    *,
    offset_x: int,
    offset_y: int,
    executor: ThreadPoolExecutor | None = None,
) -> dict[str, _DetectedFace]:
    if len(face_apps) <= 1 or executor is None:
        out: dict[str, _DetectedFace] = {}
        for model_name, app in face_apps.items():
            hit = _detect_best_face(app, person_crop, offset_x=offset_x, offset_y=offset_y)
            if hit is not None:
                out[model_name] = hit
        return out

    out = {}
    futures = {
        model_name: executor.submit(_detect_best_face, app, person_crop, offset_x=offset_x, offset_y=offset_y)
        for model_name, app in face_apps.items()
    }
    for model_name, fut in futures.items():
        hit = fut.result()
        if hit is not None:
            out[model_name] = hit
    return out


def _person_full_crop(
    frame_img: np.ndarray,
    cand: FaceCandidate,
) -> np.ndarray | None:
    img_h, img_w = frame_img.shape[:2]
    x1, y1, x2, y2 = cand.bbox
    bw = x2 - x1
    bh = y2 - y1
    px = bw * 0.05
    py = bh * 0.05
    ix1 = max(0, int(round(x1 - px)))
    iy1 = max(0, int(round(y1 - py)))
    ix2 = min(img_w, int(round(x2 + px)))
    iy2 = min(img_h, int(round(y2 + py)))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    crop = frame_img[iy1:iy2, ix1:ix2]
    return crop if crop.size > 0 else None


def extract_faces_for_groups(
    settings: Settings,
    *,
    frames_data: list[dict[str, Any]],
    track_to_group: dict[int, int],
    video_source_path: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """
    Извлекает лица для каждой группы всеми моделями из camera_link_face_models.
    Возвращает (meta_dict, embeddings_dict).
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
        track_to_group,
        max_attempts=max_attempts,
    )

    cands_by_frame: dict[int, list[FaceCandidate]] = {}
    for _gid, cands in candidates_by_group.items():
        for c in cands:
            cands_by_frame.setdefault(c.frame_index, []).append(c)

    group_faces_by_model: dict[str, dict[str, list[dict[str, Any]]]] = {m: {} for m in models}
    group_embeddings_by_model: dict[str, dict[str, list[np.ndarray]]] = {m: {} for m in models}

    cap = cv2.VideoCapture(video_source_path)
    if not cap.isOpened():
        logger.error("Не удалось открыть видео для извлечения лиц: %s", video_source_path)
        return {"groups": {}, "groups_by_model": {}}, {}

    sorted_frame_indices = sorted(cands_by_frame.keys())
    current_frame_pos = -1
    pbar = make_pbar(total=len(sorted_frame_indices), desc="[STAGE Face: InsightFace]", unit="frame")
    executor = ThreadPoolExecutor(max_workers=len(face_apps)) if len(face_apps) > 1 else None

    try:
        for fi in sorted_frame_indices:
            pbar.update(1)
            target_0based = max(0, fi - 1)

            cands_on_frame = cands_by_frame[fi]
            active_cands = [
                c
                for c in cands_on_frame
                if any(
                    len(group_faces_by_model[m].get(str(c.group_id), [])) < top_k
                    for m in models
                )
            ]
            if not active_cands:
                continue

            if current_frame_pos >= 0 and 0 < (target_0based - current_frame_pos) <= 5:
                frame_img = None
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

            img_h, img_w = frame_img.shape[:2]

            for cand in active_cands:
                gid_str = str(cand.group_id)
                if all(len(group_faces_by_model[m].get(gid_str, [])) >= top_k for m in models):
                    continue

                # 1. Проверка позы через PoseService на лету
                full_crop = _person_full_crop(frame_img, cand)
                if not pose_service.is_facing_camera(full_crop):
                    continue

                # 2. Вырезание области головы и верхней части тела
                head = _person_head_crop(frame_img, cand)
                if head is None:
                    continue
                person_crop, ix1, iy1 = head

                # 3. Детекция и распознавание лица через InsightFace
                detected = _detect_faces_parallel(face_apps, person_crop, offset_x=ix1, offset_y=iy1, executor=executor)
                if not detected:
                    # Нет лица -> переходим к следующему BBox кандидата
                    continue

                for model_name, hit in detected.items():
                    if len(group_faces_by_model[model_name].get(gid_str, [])) >= top_k:
                        continue

                    rank_idx = len(group_faces_by_model[model_name].get(gid_str, []))
                    fx1, fy1, fx2, fy2 = hit.face_bbox
                    suffix = f"_{model_name}" if multi_model else ""
                    crop_filename = f"face_g{cand.group_id:04d}_k{rank_idx}_f{cand.frame_index}{suffix}.jpg"
                    crop_out_path = os.path.join(crops_dir, crop_filename)

                    if save_crops:
                        fbw = fx2 - fx1
                        fbh = fy2 - fy1
                        f_ix1 = max(0, int(round(fx1 - fbw * 0.15)))
                        f_iy1 = max(0, int(round(fy1 - fbh * 0.15)))
                        f_ix2 = min(img_w, int(round(fx2 + fbw * 0.15)))
                        f_iy2 = min(img_h, int(round(fy2 + fbh * 0.15)))
                        if f_ix2 > f_ix1 and f_iy2 > f_iy1:
                            face_crop_img = frame_img[f_iy1:f_iy2, f_ix1:f_ix2]
                            if face_crop_img.size > 0:
                                cv2.imwrite(crop_out_path, face_crop_img, [cv2.IMWRITE_JPEG_QUALITY, 85])

                    group_faces_by_model[model_name].setdefault(gid_str, []).append({
                        "group_id": cand.group_id,
                        "track_id": cand.track_id,
                        "rank": rank_idx,
                        "frame_index": cand.frame_index,
                        "model": model_name,
                        "det_score": round(hit.det_score, 4),
                        "face_bbox": [round(fx1, 1), round(fy1, 1), round(fx2, 1), round(fy2, 1)],
                        "crop_file": crop_filename,
                    })
                    group_embeddings_by_model[model_name].setdefault(gid_str, []).append(hit.embedding)

    finally:
        cap.release()
        pbar.close()
        if executor:
            executor.shutdown(wait=False)

    final_npz_arrays: dict[str, np.ndarray] = {}
    for model_name in models:
        for gid_str, embs in group_embeddings_by_model[model_name].items():
            key = f"group_{gid_str}_{model_name}" if multi_model else f"group_{gid_str}"
            final_npz_arrays[key] = np.vstack(embs).astype(np.float32)
            if model_name == primary_model:
                final_npz_arrays[f"group_{gid_str}"] = final_npz_arrays[key]

    primary_groups = group_faces_by_model.get(primary_model, {})
    meta_payload = {
        "stage": "camera_face",
        "model": primary_model,
        "models": models,
        "n_groups_with_faces": len(primary_groups),
        "groups": primary_groups,
        "groups_by_model": group_faces_by_model,
    }

    return meta_payload, final_npz_arrays
