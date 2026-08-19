"""Сервис интеллектуального отбора лучших кадров из трека (TrackBestFramesPicker)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from app.crops.filters import (
    OutlierFilterConfig,
    filter_track_outlier_candidates,
)
from app.crops.geometry import (
    apply_gaussian_feathering,
    create_pose_alpha_mask,
    crop_person,
)
from app.entity_id import tracklet as tracklet_eid
from app.pose.pose_service import PoseService
from app.pose.types import PoseResult, select_pose_index_by_completeness
from app.util.bbox import bbox_iou, bbox_wh

logger = logging.getLogger(__name__)

POSE_CACHE_VERSION = 2


def _make_candidate_cache_key(cand: TrackFrameCandidate) -> str:
    tb = cand.target_det.get("bbox") or [0, 0, 0, 0]
    tid = cand.tracklet_id if cand.tracklet_id is not None else 0
    eid = tracklet_eid(tid).format() if tid > 0 else "t0"
    return f"{cand.frame_index}:{eid}:{round(float(tb[0]), 1)}_{round(float(tb[1]), 1)}_{round(float(tb[2]), 1)}_{round(float(tb[3]), 1)}"


def _pose_to_dict(pose: PoseResult | None) -> dict | None:
    if pose is None:
        return None
    return {
        "bbox": [round(float(v), 1) for v in pose.bbox],
        "confidence": round(float(pose.confidence), 4),
        "kxy": [[round(float(x), 1), round(float(y), 1)] for x, y in pose.kxy],
        "kcf": [round(float(c), 3) for c in pose.kcf],
    }


def _pose_from_dict(d: dict | None) -> PoseResult | None:
    if not d or not isinstance(d, dict):
        return None
    return PoseResult(
        bbox=d.get("bbox", [0, 0, 0, 0]),
        confidence=float(d.get("confidence", 0.0)),
        kxy=d.get("kxy", []),
        kcf=d.get("kcf", []),
    )


def load_pose_cache(
    cache_path: str,
    *,
    pose_model: str | None = None,
    kpt_min: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Загружает кэш поз. Несовпадение fingerprint → пустой dict."""
    if not os.path.isfile(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        version = int(data.get("cache_version") or data.get("version") or 0)
        if version != POSE_CACHE_VERSION:
            return {}
        if pose_model is not None and str(data.get("pose_model") or "") != str(pose_model):
            return {}
        if kpt_min is not None:
            stored = data.get("kpt_min")
            if stored is None or abs(float(stored) - float(kpt_min)) > 1e-6:
                return {}
        entries = data.get("entries", data)
        return entries if isinstance(entries, dict) else {}
    except Exception as e:
        logger.warning("Не удалось прочитать кэш поз %s: %s", cache_path, e)
    return {}


def save_pose_cache(
    cache_path: str,
    entries: dict[str, dict[str, Any]],
    *,
    pose_model: str = "",
    kpt_min: float = 0.0,
    keep_keys: set[str] | None = None,
) -> None:
    """Атомарно сохраняет кэш поз. keep_keys — prune до ключей текущего прогона."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        stored = entries if keep_keys is None else {k: v for k, v in entries.items() if k in keep_keys}
        payload = {
            "version": POSE_CACHE_VERSION,
            "cache_version": POSE_CACHE_VERSION,
            "pose_model": pose_model,
            "kpt_min": kpt_min,
            "count": len(stored),
            "entries": stored,
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, cache_path)
    except Exception as e:
        logger.warning("Не удалось записать кэш поз %s: %s", cache_path, e)


@dataclass(frozen=True)
class TrackFrameCandidate:
    """Кандидат на оценку из трека."""

    frame_index: int  # 0-based глобальный номер кадра
    target_det: dict  # Детекция целевого человека: {"bbox": [...], "confidence": float, ...}
    all_dets: list[dict]  # Все детекции других людей на этом кадре (для проверки перекрытий)
    image: np.ndarray | None = None  # Полный кадр (если кроп еще не вырезан)
    crop_image: np.ndarray | None = None  # Уже вырезанный кроп человека (для экономии RAM)
    crop_roi: tuple[int, int, int, int] | None = None  # Координаты вырезки кропа из кадра (rx1, ry1, rx2, ry2)
    frame_w: int = 0
    frame_h: int = 0
    tracklet_id: int | None = None  # Опциональный ID треклета


@dataclass
class ScoredTrackFrame:
    """Оцененный кадр трека с детальной разбивкой качества."""

    frame_index: int
    target_det: dict
    score: float  # Итоговый скор качества [0..1]
    geom_score: float  # Геометрия: размер, пропорции, confidence, границы
    pose_score: float  # Скелет: полнота COCO-17 + ракурс/видимость лица
    completeness: float  # 0..1 полнота ключевых точек (PoseResult.completeness)
    face_conf: float  # 0..1 ракурс и видимость лица (PoseResult.face_confidence)
    crowd_penalty: float  # [0..1] штраф за присутствие других людей (1.0 = чисто)
    n_poses_in_crop: int  # Сколько людей/поз обнаружено в кропе
    pose_result: PoseResult | None
    crop_image: np.ndarray | None = None  # Кроп тела (для ReID)
    face_crop: np.ndarray | None = None  # Кроп лица/головы (для InsightFace/UI)
    face_bbox: list[float] | None = None  # Абсолютные координаты лица на кадре [fx1, fy1, fx2, fy2]
    tracklet_id: int | None = None


def extract_face_box_from_pose(
    pose: PoseResult | None,
    person_bbox: Sequence[float],
    *,
    frame_w: int = 0,
    frame_h: int = 0,
    kpt_min: float = 0.25,
) -> list[float]:
    """
    Вычисляет BBox лица/головы по 17 ключевым точкам позы (COCO-17).
    Точки: 0: нос, 1: левый глаз, 2: правый глаз, 3: левое ухо, 4: правое ухо, 5,6: плечи.
    Если точки позы недоступны — fallback на верхние 30% BBox человека.
    """
    x1, y1, x2, y2 = [float(v) for v in person_bbox[:4]]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    face_pts: list[list[float]] = []
    if pose is not None and pose.kxy and pose.kcf:
        for idx in range(min(5, len(pose.kxy))):
            if pose.kcf[idx] >= kpt_min:
                face_pts.append(pose.kxy[idx])

    if len(face_pts) >= 2:
        fx_min = min(p[0] for p in face_pts)
        fy_min = min(p[1] for p in face_pts)
        fx_max = max(p[0] for p in face_pts)
        fy_max = max(p[1] for p in face_pts)
        fw = max(10.0, fx_max - fx_min)
        fh = max(10.0, fy_max - fy_min)

        # Паддинг: шире по бокам, выше на лоб/волосы, чуть ниже на подбородок
        pad_x = fw * 0.40
        pad_y_top = fh * 0.50
        pad_y_bottom = fh * 0.35

        # Если видны плечи (5, 6), используем их уровень как ориентир низа подбородка/шеи
        if len(pose.kxy) >= 7 and len(pose.kcf) >= 7 and pose.kcf[5] >= kpt_min and pose.kcf[6] >= kpt_min:
            shoulders_y = (pose.kxy[5][1] + pose.kxy[6][1]) / 2.0
            # max(0,...) — плечи не могут дать отрицательный паддинг (если kcf ниже лица)
            pad_y_bottom = max(pad_y_bottom, min(fh * 0.8, max(0.0, (shoulders_y - fy_max) * 0.8)))

        bx1 = fx_min - pad_x
        by1 = fy_min - pad_y_top
        bx2 = fx_max + pad_x
        by2 = fy_max + pad_y_bottom
    elif pose is not None and len(pose.kxy) >= 7 and len(pose.kcf) >= 7 and pose.kcf[5] >= kpt_min and pose.kcf[6] >= kpt_min:
        # Точки лица скрыты, но видны плечи -> голова ровно над плечами
        sh_x = (pose.kxy[5][0] + pose.kxy[6][0]) / 2.0
        sh_y = (pose.kxy[5][1] + pose.kxy[6][1]) / 2.0
        sh_w = abs(pose.kxy[6][0] - pose.kxy[5][0])
        head_w = max(20.0, sh_w * 0.7)
        head_h = head_w * 1.2
        bx1 = sh_x - head_w / 2.0
        by1 = sh_y - head_h * 1.1
        bx2 = sh_x + head_w / 2.0
        by2 = sh_y + head_h * 0.1
    else:
        # Fallback: верхние 30% BBox человека
        bx1 = x1
        by1 = y1
        bx2 = x2
        by2 = y1 + bh * 0.30

    # Ограничение границами BBox человека с небольшим допуском
    bx1 = max(x1 - bw * 0.1, bx1)
    by1 = max(y1 - bh * 0.05, by1)
    bx2 = min(x2 + bw * 0.1, bx2)
    by2 = min(y1 + bh * 0.45, by2)

    # Ограничение границами кадра (если заданы)
    if frame_w > 0:
        bx1 = max(0.0, min(float(frame_w - 1), bx1))
        bx2 = max(0.0, min(float(frame_w), bx2))
    if frame_h > 0:
        by1 = max(0.0, min(float(frame_h - 1), by1))
        by2 = max(0.0, min(float(frame_h), by2))

    if bx2 <= bx1:
        bx2 = bx1 + max(1.0, bw * 0.3)
    if by2 <= by1:
        by2 = by1 + max(1.0, bh * 0.25)

    return [round(bx1, 1), round(by1, 1), round(bx2, 1), round(by2, 1)]


def extract_face_crop_from_person(
    crop_image: np.ndarray | None,
    crop_roi: tuple[int, int, int, int] | None,
    pose: PoseResult | None,
    target_det: dict,
    *,
    frame_w: int = 0,
    frame_h: int = 0,
    kpt_min: float = 0.25,
) -> tuple[np.ndarray | None, list[float] | None]:
    """
    Вырезает кроп лица/головы из уже существующего кропа человека или возвращает None.
    Возвращает (face_crop, face_bbox_abs).
    """
    tb = target_det.get("bbox")
    if not tb or len(tb) < 4:
        return None, None

    # Вычисляем абсолютный BBox лица на кадре
    face_box = extract_face_box_from_pose(
        pose, tb, frame_w=frame_w, frame_h=frame_h, kpt_min=kpt_min
    )

    if crop_image is None or crop_image.size == 0 or crop_roi is None:
        return None, face_box

    # Переводим координаты face_box в локальные координаты crop_image
    rx1, ry1, rx2, ry2 = crop_roi
    ch, cw = crop_image.shape[:2]
    fx1 = int(max(0, min(cw - 1, round(face_box[0] - rx1))))
    fy1 = int(max(0, min(ch - 1, round(face_box[1] - ry1))))
    fx2 = int(max(0, min(cw, round(face_box[2] - rx1))))
    fy2 = int(max(0, min(ch, round(face_box[3] - ry1))))

    if fx2 <= fx1 or fy2 <= fy1:
        return None, face_box

    face_crop = crop_image[fy1:fy2, fx1:fx2]
    if face_crop.size == 0 or face_crop.shape[0] < 8 or face_crop.shape[1] < 8:
        return None, face_box

    return np.ascontiguousarray(face_crop), face_box


def compute_geometry_score(
    det: dict,
    frame_w: int,
    frame_h: int,
    *,
    margin_threshold: float = 0.01,
) -> float:
    """
    Оценка геометрии детекции:
    1. conf детектора [0..1]
    2. Пропорции aspect ratio H/W (норма 1.8..3.8)
    3. Масштаб (log-площадь)
    4. Штраф за срез границей кадра
    """
    conf = float(det.get("confidence") or 0.0)
    bbox = det.get("bbox")
    if not bbox or len(bbox) < 4:
        return conf

    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    w, h = bbox_wh(bbox)
    if w <= 0 or h <= 0:
        return 0.0

    # 1. Aspect ratio factor (человек в полный рост / поясной)
    aspect = h / max(1.0, w)
    aspect_factor = 1.0
    if aspect < 1.5:
        aspect_factor = max(0.2, aspect / 1.5)
    elif aspect > 4.5:
        aspect_factor = max(0.5, 4.5 / aspect)

    # 2. Area factor (предпочтение более крупным планам)
    area = w * h
    area_factor = min(1.2, max(0.8, (area / 10000.0) ** 0.1))

    # 3. Штраф за касание границ кадра (человек обрезан)
    margin_penalty = 1.0
    if frame_w > 0 and frame_h > 0:
        near_left = x1 <= frame_w * margin_threshold
        near_top = y1 <= frame_h * margin_threshold
        near_right = x2 >= frame_w * (1.0 - margin_threshold)
        near_bottom = y2 >= frame_h * (1.0 - margin_threshold)
        cut_edges = sum([near_left, near_top, near_right, near_bottom])
        if cut_edges > 0:
            margin_penalty = max(0.4, 1.0 - cut_edges * 0.2)

    return round(float(conf * aspect_factor * area_factor * margin_penalty), 4)


def compute_frame_crowd_penalty(
    target_bbox: Sequence[float],
    all_other_dets: Sequence[dict],
    *,
    iou_penalty_k: float = 2.0,
) -> float:
    """
    Штраф за взаимное перекрытие BBox с другими обнаруженными людьми на исходном кадре.
    """
    if not all_other_dets or not target_bbox or len(target_bbox) < 4:
        return 1.0

    tb = [float(v) for v in target_bbox[:4]]
    max_iou = 0.0
    for other in all_other_dets:
        ob = other.get("bbox")
        if not ob or len(ob) < 4:
            continue
        iou = bbox_iou(tb, [float(v) for v in ob[:4]])
        if iou > max_iou:
            max_iou = iou

    if max_iou <= 0.05:
        return 1.0
    return round(max(0.2, 1.0 - max_iou * iou_penalty_k), 4)


class TrackBestFramesPicker:
    """
    Сервис интеллектуального отбора лучших кадров трека.
    """

    def __init__(
        self,
        pose_service: PoseService | None = None,
        *,
        pose_weight: float = 0.40,
        crowd_crop_penalty: float = 0.50,
        kpt_min: float = 0.25,
        w_completeness: float = 0.65,
        w_face: float = 0.35,
        crop_pad: float = 0.04,
        feathering_enabled: bool = False,
        feathering_mode: str = "pose",
        feathering_sigma: float = 15.0,
        feathering_bone_thickness: float = 0.18,
        feathering_bg_color: Sequence[int] | tuple[int, int, int] = (128, 128, 128),
        # Параметры фильтрации выбросов
        trim_enabled: bool = True,
        trim_start: int = 2,
        trim_end: int = 2,
        trim_min_len: int = 8,
        kinematic_enabled: bool = True,
        kinematic_max_speed_ratio: float = 3.0,
        kinematic_max_area_ratio: float = 2.2,
        kinematic_min_candidates: int = 5,
        color_consistency_enabled: bool = True,
        color_min_similarity: float = 0.50,
        color_edge_only: bool = True,
        color_edge_window: int = 2,
        outlier_filter_config: OutlierFilterConfig | None = None,
    ) -> None:
        self.pose_service = pose_service
        self.pose_weight = max(0.0, min(1.0, float(pose_weight)))
        self.crowd_crop_penalty = max(0.1, min(1.0, float(crowd_crop_penalty)))
        self.kpt_min = float(kpt_min)
        self.w_completeness = float(w_completeness)
        self.w_face = float(w_face)
        self.crop_pad = float(crop_pad)
        self.feathering_enabled = bool(feathering_enabled)
        self.feathering_mode = str(feathering_mode or "pose")
        self.feathering_sigma = max(1.0, float(feathering_sigma))
        self.feathering_bone_thickness = float(feathering_bone_thickness)
        self.feathering_bg_color = tuple(int(v) for v in feathering_bg_color[:3])

        if outlier_filter_config is not None:
            self.outlier_filter_config = outlier_filter_config
        else:
            self.outlier_filter_config = OutlierFilterConfig(
                trim_enabled=bool(trim_enabled),
                trim_start=int(trim_start),
                trim_end=int(trim_end),
                trim_min_len=int(trim_min_len),
                kinematic_enabled=bool(kinematic_enabled),
                kinematic_max_speed_ratio=float(kinematic_max_speed_ratio),
                kinematic_max_area_ratio=float(kinematic_max_area_ratio),
                kinematic_min_candidates=int(kinematic_min_candidates),
                color_enabled=bool(color_consistency_enabled),
                color_min_similarity=float(color_min_similarity),
                color_edge_only=bool(color_edge_only),
                color_edge_window=int(color_edge_window),
            )

    def filter_candidates(
        self, candidates: Sequence[TrackFrameCandidate]
    ) -> list[TrackFrameCandidate]:
        """Группирует кандидатов по tracklet_id и применяет многоуровневую фильтрацию выбросов."""
        if not candidates:
            return []

        cfg = self.outlier_filter_config
        if not (cfg.trim_enabled or cfg.kinematic_enabled or cfg.color_enabled):
            return list(candidates)

        by_tid: dict[int, list[TrackFrameCandidate]] = {}
        for c in candidates:
            tid = c.tracklet_id if c.tracklet_id is not None else 0
            by_tid.setdefault(tid, []).append(c)

        cleaned: list[TrackFrameCandidate] = []
        for tid, cands in by_tid.items():
            cleaned_t = filter_track_outlier_candidates(cands, cfg)
            cleaned.extend(cleaned_t)

        return sorted(cleaned, key=lambda c: (c.tracklet_id or 0, c.frame_index))

    def score_candidates_batch(
        self,
        candidates: Sequence[TrackFrameCandidate],
        *,
        batch_size: int = 16,
        extract_faces: bool = True,
        show_pbar: bool = False,
        pbar_desc: str = "[STAGE 2b: Pose scoring]",
        cache_path: str | None = None,
        prune_cache: bool = False,
        preloaded_poses: dict[int, dict[int, dict[str, Any]]] | None = None,
        filter_outliers: bool = False,
    ) -> list[ScoredTrackFrame]:
        """
        Батчевый скоринг списка кандидатов с поддержкой дискового кэша поз.
        Поддерживает как полные кадры (image), так и предварительно вырезанные кропы (crop_image).
        При extract_faces=True автоматически вырезает face_crop и face_bbox по ключевым точкам позы.
        При filter_outliers=True предварительно фильтрует аномальные кадры трека.
        """
        if not candidates:
            return []

        if filter_outliers:
            candidates = self.filter_candidates(candidates)
            if not candidates:
                return []

        pose_model = self.pose_service.model_name if self.pose_service is not None else ""
        cache_entries: dict[str, dict[str, Any]] = (
            load_pose_cache(cache_path, pose_model=pose_model, kpt_min=self.kpt_min)
            if cache_path
            else {}
        )
        cache_updated = False

        if preloaded_poses:
            for cand in candidates:
                k = _make_candidate_cache_key(cand)
                if k in cache_entries:
                    continue
                tid = cand.tracklet_id or 0
                fi_1 = cand.frame_index + 1
                p_info = preloaded_poses.get(tid, {}).get(fi_1) or preloaded_poses.get(tid, {}).get(cand.frame_index)
                if p_info and p_info.get("kxy"):
                    best_pose = PoseResult(
                        bbox=p_info.get("bbox") or cand.target_det.get("bbox") or [0, 0, 0, 0],
                        confidence=float(p_info.get("confidence", 0.8)),
                        kxy=p_info["kxy"],
                        kcf=p_info.get("kcf") or [1.0] * len(p_info["kxy"]),
                    )
                    cache_entries[k] = {
                        "pose_result": _pose_to_dict(best_pose),
                        "n_poses_in_crop": 1,
                        "crop_crowd_penalty": 1.0,
                    }
                    cache_updated = True

        # 2. Подготовка кропов и геометрии
        crops: list[np.ndarray] = []
        crop_rois: list[tuple[int, int, int, int]] = []
        geom_scores: list[float] = []
        frame_crowd_penalties: list[float] = []
        frame_dims: list[tuple[int, int]] = []
        cand_keys: list[str] = []

        # Индексы кандидатов, требующих инференса нейросети поз
        uncached_indices: list[int] = []
        uncached_crops: list[np.ndarray] = []

        for i, cand in enumerate(candidates):
            # Определение ширины/высоты кадра
            if cand.frame_w > 0 and cand.frame_h > 0:
                img_w, img_h = cand.frame_w, cand.frame_h
            elif cand.image is not None and cand.image.size > 0:
                img_h, img_w = cand.image.shape[:2]
            else:
                img_w, img_h = 0, 0
            frame_dims.append((img_w, img_h))

            tb = cand.target_det.get("bbox") or [0, 0, 1, 1]
            k = _make_candidate_cache_key(cand)
            cand_keys.append(k)

            # Геометрия и перекрытия на кадре
            gs = compute_geometry_score(cand.target_det, img_w, img_h)
            cp_frame = compute_frame_crowd_penalty(tb, cand.all_dets)
            geom_scores.append(gs)
            frame_crowd_penalties.append(cp_frame)

            # Получение кропа (из crop_image или вырезка из image)
            if cand.crop_image is not None and cand.crop_image.size > 0:
                crop_arr = np.ascontiguousarray(cand.crop_image)
                roi = cand.crop_roi or (
                    int(tb[0]),
                    int(tb[1]),
                    int(tb[2]),
                    int(tb[3]),
                )
                has_pixels = True
            elif cand.image is not None and cand.image.size > 0:
                crop_arr, roi = crop_person(cand.image, tb, pad=self.crop_pad)
                crop_arr = np.ascontiguousarray(crop_arr)
                has_pixels = crop_arr.size > 0
            else:
                crop_arr = np.zeros((10, 10, 3), dtype=np.uint8)
                roi = (0, 0, 10, 10)
                has_pixels = False

            crops.append(crop_arr)
            crop_rois.append(roi)

            if k not in cache_entries and has_pixels:
                uncached_indices.append(i)
                uncached_crops.append(crop_arr)

        if cache_path:
            cached_count = len(candidates) - len(uncached_indices)
            logger.info(
                "TrackBestFramesPicker: кэш %s/%s, инференс позы %s",
                cached_count,
                len(candidates),
                len(uncached_indices),
            )

        # 3. Батчевый инференс поз ТОЛЬКО для непокэшированных кандидатов
        uncached_poses: list[list[PoseResult]] = []
        if self.pose_service is not None and uncached_crops:
            uncached_poses = self.pose_service.predict_batch(
                uncached_crops,
                batch_size=batch_size,
                show_pbar=show_pbar,
                pbar_desc=pbar_desc,
            )

        # Карта результатов позы для каждого кандидата
        poses_by_cand_idx: dict[int, list[PoseResult]] = {}
        inferred_indices = set(uncached_indices)
        for idx_in_uncached, orig_idx in enumerate(uncached_indices):
            if idx_in_uncached < len(uncached_poses):
                poses_by_cand_idx[orig_idx] = uncached_poses[idx_in_uncached]

        # 4. Финализация скоров, сохранение в кэш и вырезка лиц
        scored_list: list[ScoredTrackFrame] = []
        for i, cand in enumerate(candidates):
            k = cand_keys[i]
            crop_roi = crop_rois[i]
            rx1, ry1 = crop_roi[0], crop_roi[1]
            tb = cand.target_det.get("bbox") or [0, 0, 1, 1]
            img_w, img_h = frame_dims[i]

            best_pose: PoseResult | None = None
            n_poses_in_crop = 0
            crop_crowd_penalty = 1.0

            if k in cache_entries:
                c_item = cache_entries[k]
                best_pose = _pose_from_dict(c_item.get("pose_result"))
                n_poses_in_crop = int(c_item.get("n_poses_in_crop", 0))
                crop_crowd_penalty = float(c_item.get("crop_crowd_penalty", 1.0))
            elif i in inferred_indices:
                raw_poses = poses_by_cand_idx.get(i, [])
                n_poses_in_crop = len(raw_poses)

                # Сопоставляем найденные позы в кропе с целевым человеком
                bx1, by1 = tb[0] - rx1, tb[1] - ry1
                bx2, by2 = tb[2] - rx1, tb[3] - ry1
                target_crop_bbox = [bx1, by1, bx2, by2]

                target_pose_idx: int | None = None
                if raw_poses:
                    target_pose_idx = select_pose_index_by_completeness(
                        [(p.bbox, p.kcf, p.confidence) for p in raw_poses],
                        target_crop_bbox,
                        min_iou=0.25,
                        kpt_min=self.kpt_min,
                        require_min_iou=False,
                    )
                    if target_pose_idx is not None:
                        # Смещаем позу обратно в координаты кадра
                        best_pose = raw_poses[target_pose_idx].with_offset(rx1, ry1)

                # Штраф за посторонних людей в кропе (фильтруем слабые шумы)
                other_confident_poses = [
                    p
                    for j, p in enumerate(raw_poses)
                    if j != target_pose_idx and (p.confidence >= 0.25 or p.completeness() >= 0.20)
                ]
                if len(other_confident_poses) > 0:
                    crop_crowd_penalty = self.crowd_crop_penalty

                # Сохраняем результат в кэш
                if cache_path:
                    cache_entries[k] = {
                        "n_poses_in_crop": n_poses_in_crop,
                        "crop_crowd_penalty": crop_crowd_penalty,
                        "pose_result": _pose_to_dict(best_pose),
                    }
                    cache_updated = True

            # Оценка позы
            if best_pose is not None:
                compl = best_pose.completeness(min_conf=self.kpt_min)
                face_c = best_pose.face_confidence(min_conf=self.kpt_min)
                pose_s = round(self.w_completeness * compl + self.w_face * face_c, 4)
            else:
                compl = 0.0
                face_c = 0.0
                pose_s = 0.0

            total_crowd_penalty = round(frame_crowd_penalties[i] * crop_crowd_penalty, 4)

            # Итоговый скор:
            if self.pose_service is not None and self.pose_weight > 0:
                pose_multiplier = (1.0 - self.pose_weight) + self.pose_weight * pose_s
            else:
                pose_multiplier = 1.0

            final_score = round(geom_scores[i] * pose_multiplier * total_crowd_penalty, 4)

            # Вырезка кропа лица и абсолютного face_bbox (если запрошено)
            face_crop_arr: np.ndarray | None = None
            face_bbox_arr: list[float] | None = None
            if extract_faces:
                face_crop_arr, face_bbox_arr = extract_face_crop_from_person(
                    crops[i],
                    crop_roi,
                    best_pose,
                    cand.target_det,
                    frame_w=img_w,
                    frame_h=img_h,
                    kpt_min=self.kpt_min,
                )

            final_crop = crops[i]
            if self.feathering_enabled and final_crop is not None and final_crop.size > 0:
                alpha_mask = create_pose_alpha_mask(
                    final_crop.shape[:2],
                    best_pose,
                    crop_roi=crop_roi,
                    kpt_min=self.kpt_min,
                    sigma=self.feathering_sigma,
                    bone_thickness_ratio=self.feathering_bone_thickness,
                    mode=self.feathering_mode,
                )
                final_crop = apply_gaussian_feathering(
                    final_crop,
                    alpha_mask,
                    bg_color=self.feathering_bg_color,
                )

            scored_list.append(
                ScoredTrackFrame(
                    frame_index=cand.frame_index,
                    target_det=cand.target_det,
                    score=final_score,
                    geom_score=geom_scores[i],
                    pose_score=pose_s,
                    completeness=compl,
                    face_conf=face_c,
                    crowd_penalty=total_crowd_penalty,
                    n_poses_in_crop=n_poses_in_crop,
                    pose_result=best_pose,
                    crop_image=final_crop,
                    face_crop=face_crop_arr,
                    face_bbox=face_bbox_arr,
                    tracklet_id=cand.tracklet_id,
                )
            )

        # 5. Сохраняем обновленный кэш на диск
        if cache_path and (cache_updated or prune_cache):
            save_pose_cache(
                cache_path,
                cache_entries,
                pose_model=pose_model,
                kpt_min=self.kpt_min,
                keep_keys=set(cand_keys) if prune_cache else None,
            )

        return scored_list

    def pick_best_from_scored(
        self,
        scored: Sequence[ScoredTrackFrame],
        top_k: int = 10,
    ) -> list[ScoredTrackFrame]:
        """
        Выбирает лучшие top_k кадров из уже оцененных ScoredTrackFrame с равномерным spread.
        """
        if not scored:
            return []
        if len(scored) <= top_k:
            return sorted(scored, key=lambda x: x.frame_index)

        k = max(1, int(top_k))
        chunk_size = len(scored) / float(k)
        picked: list[ScoredTrackFrame] = []

        for i in range(k):
            start_idx = int(round(i * chunk_size))
            end_idx = int(round((i + 1) * chunk_size)) if i < k - 1 else len(scored)
            window = scored[start_idx:end_idx]
            if window:
                best_in_window = max(window, key=lambda x: x.score)
                picked.append(best_in_window)

        return picked if picked else list(scored[:top_k])

    def pick_best_for_tracklet(
        self,
        candidates: Sequence[TrackFrameCandidate],
        top_k: int = 10,
        *,
        batch_size: int = 16,
        extract_faces: bool = True,
        cache_path: str | None = None,
        filter_outliers: bool = True,
    ) -> list[ScoredTrackFrame]:
        """
        Выбирает лучшие top_k кадров для одного треклета с равномерным распределением (spread).
        """
        if not candidates:
            return []
        cands = self.filter_candidates(candidates) if filter_outliers else candidates
        if not cands:
            cands = candidates
        scored = self.score_candidates_batch(
            cands,
            batch_size=batch_size,
            extract_faces=extract_faces,
            cache_path=cache_path,
        )
        return self.pick_best_from_scored(scored, top_k=top_k)

    def pick_best_for_group(
        self,
        candidates_by_tid: dict[int, list[TrackFrameCandidate]],
        top_k: int = 10,
        *,
        batch_size: int = 16,
        extract_faces: bool = True,
        cache_path: str | None = None,
        filter_outliers: bool = True,
    ) -> list[ScoredTrackFrame]:
        """
        Выбирает лучшие top_k кадров для группы склеенных треклетов
        (сбалансированное распределение по входящим трекам).
        """
        if not candidates_by_tid:
            return []

        all_candidates: list[TrackFrameCandidate] = []
        for cands in candidates_by_tid.values():
            if filter_outliers:
                cleaned = self.filter_candidates(cands)
                all_candidates.extend(cleaned if cleaned else cands)
            else:
                all_candidates.extend(cands)

        if not all_candidates:
            return []

        # Скорим всех кандидатов группы в одном батче
        scored = self.score_candidates_batch(
            all_candidates,
            batch_size=batch_size,
            extract_faces=extract_faces,
            cache_path=cache_path,
        )
        if len(scored) <= top_k:
            return sorted(scored, key=lambda x: x.frame_index)

        # Группируем оцененные кадры по tracklet_id
        scored_by_tid: dict[int, list[ScoredTrackFrame]] = {}
        for s in scored:
            tid = s.tracklet_id if s.tracklet_id is not None else 0
            scored_by_tid.setdefault(tid, []).append(s)

        # Выделяем квоту для каждого треклета пропорционально его длине
        picked: list[ScoredTrackFrame] = []
        tids = list(scored_by_tid.keys())
        total_obs = sum(len(obs) for obs in scored_by_tid.values())

        remaining_k = top_k
        for i, tid in enumerate(tids):
            obs = scored_by_tid[tid]
            if i == len(tids) - 1:
                k_for_tid = remaining_k
            else:
                k_for_tid = max(1, int(round(len(obs) / float(total_obs) * top_k)))
                k_for_tid = min(remaining_k, k_for_tid)

            if k_for_tid > 0:
                picked_tid = self.pick_best_from_scored(obs, top_k=k_for_tid)
                picked.extend(picked_tid)
                remaining_k -= len(picked_tid)

        # Если не добрали до top_k (из-за округления), добираем лучшие по скору
        if len(picked) < top_k:
            picked_keys = {(p.tracklet_id, p.frame_index) for p in picked}
            remaining = [s for s in scored if (s.tracklet_id, s.frame_index) not in picked_keys]
            remaining.sort(key=lambda x: -x.score)
            picked.extend(remaining[: top_k - len(picked)])

        return sorted(picked, key=lambda x: x.frame_index)

    def pick_best_faces_from_scored(
        self,
        scored: Sequence[ScoredTrackFrame],
        top_k: int = 5,
        *,
        min_face_conf: float = 0.20,
    ) -> list[ScoredTrackFrame]:
        """
        Специализированная выборка лучших кадров лиц из уже оцененных ScoredTrackFrame.
        Оптимизирует выборку по ракурсу и видимости лица (face_conf) с временным spread.
        """
        if not scored:
            return []

        # Фильтруем кадры, где лицо имеет достаточную уверенность
        valid = [s for s in scored if s.face_conf >= min_face_conf]
        pool = valid if valid else list(scored)
        if not pool:
            return []
        if len(pool) <= top_k:
            return sorted(pool, key=lambda x: x.frame_index)

        # Скоринг для лица: 70% ракурс лица + 30% общее качество кадра/чистота
        def _face_priority(s: ScoredTrackFrame) -> float:
            return s.face_conf * 0.70 + s.score * 0.30

        k = max(1, int(top_k))
        chunk_size = len(pool) / float(k)
        picked: list[ScoredTrackFrame] = []

        for i in range(k):
            start_idx = int(round(i * chunk_size))
            end_idx = int(round((i + 1) * chunk_size)) if i < k - 1 else len(pool)
            window = pool[start_idx:end_idx]
            if window:
                best_in_window = max(window, key=_face_priority)
                picked.append(best_in_window)

        return picked if picked else list(pool[:top_k])

    def pick_best_faces_for_group(
        self,
        candidates_by_tid: dict[int, list[TrackFrameCandidate]],
        top_k: int = 5,
        *,
        min_face_conf: float = 0.20,
        batch_size: int = 16,
        cache_path: str | None = None,
        filter_outliers: bool = True,
    ) -> list[ScoredTrackFrame]:
        """
        Выбирает лучшие top_k кадров лиц для группы склеенных треков (для InsightFace / Stage 10).
        """
        if not candidates_by_tid:
            return []

        all_candidates: list[TrackFrameCandidate] = []
        for cands in candidates_by_tid.values():
            if filter_outliers:
                cleaned = self.filter_candidates(cands)
                all_candidates.extend(cleaned if cleaned else cands)
            else:
                all_candidates.extend(cands)

        if not all_candidates:
            return []

        # Скорим всех кандидатов группы с извлечением кропов лиц и поддержкой кэша
        scored = self.score_candidates_batch(
            all_candidates,
            batch_size=batch_size,
            extract_faces=True,
            cache_path=cache_path,
        )
        return self.pick_best_faces_from_scored(
            scored, top_k=top_k, min_face_conf=min_face_conf
        )
