"""Вырезка ROI и координаты bbox в кропе."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.pose.types import PoseResult
from app.util.bbox import bbox_wh

# Анатомические связи скелета COCO-17 для построения силуэта человека
# (индексы пар точек: голова/шея, торс, руки, ноги)
COCO_LIMBS: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),  # лицо
    (5, 6),                           # плечи
    (5, 7), (7, 9),                   # левая рука (плечо-локоть-кисть)
    (6, 8), (8, 10),                  # правая рука
    (5, 11), (6, 12), (11, 12),       # торс (плечи к бедрам и линия бедер)
    (11, 13), (13, 15),               # левая нога (бедро-колено-лодыжка)
    (12, 14), (14, 16),               # правая нога
)


def crop_roi_xyxy(
    bbox: list[float], frame_w: int, frame_h: int, pad: float = 0.0
) -> tuple[int, int, int, int]:
    """Область вырезки: по умолчанию ровно bbox (pad=0), без полей."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = bbox_wh(bbox)
    bw, bh = max(1.0, bw), max(1.0, bh)
    rx1 = int(max(0, x1 - bw * pad))
    ry1 = int(max(0, y1 - bh * pad * (1.4 if pad > 0 else 0)))
    rx2 = int(min(frame_w, x2 + bw * pad)) if frame_w > 0 else int(x2 + bw * pad)
    ry2 = int(min(frame_h, y2 + bh * pad * (0.3 if pad > 0 else 0))) if frame_h > 0 else int(y2 + bh * pad)
    if rx2 <= rx1 or ry2 <= ry1:
        return (0, 0, 1, 1)
    return (rx1, ry1, rx2, ry2)


def crop_person(
    frame: np.ndarray, bbox: list[float], pad: float = 0.0
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    fh, fw = frame.shape[:2]
    roi = crop_roi_xyxy(bbox, fw, fh, pad)
    x1, y1, x2, y2 = roi
    return frame[y1:y2, x1:x2], roi


def bbox_in_crop(
    bbox: list[Any] | None, crop_roi: list[Any] | None
) -> tuple[int, int, int, int] | None:
    """Bbox трека в координатах JPEG-кропа (crop_roi — вырезка из кадра)."""
    if not bbox or not crop_roi or len(bbox) < 4 or len(crop_roi) < 4:
        return None
    rx1, ry1 = int(crop_roi[0]), int(crop_roi[1])
    bx1, by1, bx2, by2 = (int(round(float(v))) for v in bbox[:4])
    return bx1 - rx1, by1 - ry1, bx2 - rx1, by2 - ry1


def create_pose_alpha_mask(
    crop_shape: tuple[int, int],
    pose: PoseResult | None,
    crop_roi: tuple[int, int, int, int] | None = None,
    *,
    kpt_min: float = 0.20,
    sigma: float = 15.0,
    bone_thickness_ratio: float = 0.18,
    mode: str = "pose",
) -> np.ndarray:
    """
    Создает плавную 2D альфа-маску (float32 [0.0..1.0]) для кропа человека.

    Если mode='pose' и ключевые точки доступны:
      1. Отрисовывает анатомический скелет (кости) с пропорциональной толщиной.
      2. Добавляет выпуклую оболочку торса/головы.
      3. Применяет Gaussian Blur (или distanceTransform), формируя мягкое затухание (feathering) краев.
    Если точек недостаточно или mode='ellipse':
      Fallback на плавную радиально-эллиптическую маску от центра кропа.
    """
    h, w = crop_shape[:2]
    if h <= 0 or w <= 0:
        return np.ones((max(1, h), max(1, w)), dtype=np.float32)

    rx1 = float(crop_roi[0]) if crop_roi and len(crop_roi) >= 2 else 0.0
    ry1 = float(crop_roi[1]) if crop_roi and len(crop_roi) >= 2 else 0.0

    valid_pts: list[tuple[int, int]] = []
    pts_by_idx: dict[int, tuple[int, int]] = {}

    if mode == "pose" and pose is not None and pose.kxy and pose.kcf:
        for idx, (pt, cf) in enumerate(zip(pose.kxy, pose.kcf)):
            if cf >= kpt_min:
                px = int(round(pt[0] - rx1))
                py = int(round(pt[1] - ry1))
                # Допускаем точки внутри и в небольших окрестностях кропа
                if -w * 0.2 <= px <= w * 1.2 and -h * 0.2 <= py <= h * 1.2:
                    valid_pts.append((px, py))
                    pts_by_idx[idx] = (px, py)

    # Достаточно ли точек скелета для построения формы (хотя бы 3 точки)
    if mode == "pose" and len(valid_pts) >= 3:
        binary_mask = np.zeros((h, w), dtype=np.uint8)

        # Вычисляем толщину конечностей и торса пропорционально размерам кропа
        base_thickness = max(4, int(round(w * max(0.05, min(0.5, bone_thickness_ratio)))))
        torso_thickness = int(round(base_thickness * 1.5))
        head_radius = max(6, int(round(w * 0.22)))

        # 1. Линии скелета
        for idx1, idx2 in COCO_LIMBS:
            if idx1 in pts_by_idx and idx2 in pts_by_idx:
                p1 = pts_by_idx[idx1]
                p2 = pts_by_idx[idx2]
                is_torso = (idx1 in (5, 6, 11, 12)) and (idx2 in (5, 6, 11, 12))
                th = torso_thickness if is_torso else base_thickness
                cv2.line(binary_mask, p1, p2, 255, thickness=th, lineType=cv2.LINE_AA)

        # 2. Окружности в суставах и голове
        for idx, (px, py) in pts_by_idx.items():
            r = head_radius if idx in (0, 1, 2, 3, 4) else (base_thickness // 2 + 1)
            cv2.circle(binary_mask, (px, py), r, 255, thickness=-1)

        # 3. Полигон торса (если есть плечи и бедра)
        torso_pts = [pts_by_idx[i] for i in (5, 6, 12, 11) if i in pts_by_idx]
        if len(torso_pts) >= 3:
            hull = cv2.convexHull(np.array(torso_pts, dtype=np.int32))
            cv2.fillPoly(binary_mask, [hull], 255)

        # Мягкое морфологическое замыкание
        kernel_sz = max(3, (base_thickness // 2) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_sz, kernel_sz))
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

        # Gaussian Feathering от маски
        sig = max(1.0, float(sigma))
        ksize = int(round(sig * 3)) | 1
        mask_f = binary_mask.astype(np.float32) / 255.0
        smooth_mask = cv2.GaussianBlur(mask_f, (ksize, ksize), sigmaX=sig, sigmaY=sig)

        # Нормализация контраста маски (центр тела должен быть надежно 1.0)
        smooth_mask = np.clip(smooth_mask * 1.35, 0.0, 1.0)
        return smooth_mask.astype(np.float32)

    # Fallback: эллиптическое радиальное затухание от центра кропа
    yy, xx = np.ogrid[:h, :w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    rx, ry = max(1.0, w * 0.48), max(1.0, h * 0.48)
    dist_norm = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    # Сигмоида / Cosine затухание к краям
    mask = np.clip(1.0 - (dist_norm - 0.4) / 0.8, 0.0, 1.0)
    sig = max(1.0, float(sigma))
    ksize = int(round(sig * 2)) | 1
    smooth_mask = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), sigmaX=sig, sigmaY=sig)
    return np.clip(smooth_mask, 0.0, 1.0).astype(np.float32)


def apply_gaussian_feathering(
    crop_img: np.ndarray,
    alpha_mask: np.ndarray,
    bg_color: Sequence[int] | tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """
    Накладывает альфа-маску на кроп, плавно растворяя фон в нейтральный цвет (RGB/BGR).
    """
    if crop_img is None or crop_img.size == 0:
        return crop_img

    h, w = crop_img.shape[:2]
    if alpha_mask.shape[:2] != (h, w):
        alpha_mask = cv2.resize(alpha_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    if alpha_mask.ndim == 2:
        alpha = alpha_mask[:, :, np.newaxis]
    else:
        alpha = alpha_mask

    bg = np.array(bg_color[:3], dtype=np.float32).reshape(1, 1, 3)
    blended = crop_img.astype(np.float32) * alpha + bg * (1.0 - alpha)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)

