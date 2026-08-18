"""Типы данных и структуры результатов сервиса поз."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Sequence

from app.util.bbox import bbox_iou

# COCO-17: лицо / торс / руки / ноги. Веса для оценки полноты скелета.
_FACE_IDX = (0, 1, 2, 3, 4)
_TORSO_IDX = (5, 6, 11, 12)
_ARMS_IDX = (7, 8, 9, 10)
_LEGS_IDX = (13, 14, 15, 16)
_COMPLETENESS_GROUPS: tuple[tuple[tuple[int, ...], float], ...] = (
    (_FACE_IDX, 0.20),
    (_TORSO_IDX, 0.40),
    (_ARMS_IDX, 0.20),
    (_LEGS_IDX, 0.20),
)
_MIN_POSE_IOU = 0.3
_IOU_NEAR_BEST = 0.85


def pose_completeness(kcf: Sequence[float] | None, min_conf: float = 0.25) -> float:
    """
    0..1: насколько скелет полный (лицо, торс, руки, ноги).
    Не оценивает ракурс лица — только покрытие ключевых точек.
    """
    if not kcf:
        return 0.0
    total = 0.0
    for indices, weight in _COMPLETENESS_GROUPS:
        vis = 0.0
        n = 0
        for i in indices:
            if i < len(kcf):
                n += 1
                if float(kcf[i]) >= min_conf:
                    vis += 1.0
        if n:
            total += weight * (vis / n)
    return round(total, 4)


def select_pose_index_by_completeness(
    items: Sequence[tuple[Sequence[float], Sequence[float], float]],
    target_bbox: Sequence[float],
    *,
    min_iou: float = _MIN_POSE_IOU,
    kpt_min: float = 0.25,
    skip: AbstractSet[int] | None = None,
    require_min_iou: bool = True,
) -> int | None:
    """
    Среди детекций позы выбирает самую полную, совпадающую с tracking bbox.

    1) отсекает чужие боксы по IoU;
    2) среди близких к лучшему IoU берёт max полноты скелета;
    3) видимость лица считается отдельно (face_confidence) уже после выбора.
    items: (bbox, kcf, detector_confidence)
    """
    if not items or not target_bbox or len(target_bbox) < 4:
        return None
    skip_idx = skip or set()
    tb = [float(v) for v in target_bbox[:4]]
    scored: list[tuple[int, float, float, float]] = []
    for i, (bbox, kcf, conf) in enumerate(items):
        if i in skip_idx or not bbox or len(bbox) < 4:
            continue
        iou = bbox_iou(tb, [float(v) for v in bbox[:4]])
        scored.append((i, iou, pose_completeness(kcf, kpt_min), float(conf)))
    if not scored:
        return None
    best_iou = max(row[1] for row in scored)
    if best_iou < min_iou:
        if require_min_iou:
            return None
        return max(scored, key=lambda row: (row[1], row[2], row[3]))[0]
    thresh = max(min_iou, best_iou * _IOU_NEAR_BEST)
    near = [row for row in scored if row[1] >= thresh]
    return max(near, key=lambda row: (row[2], row[1], row[3]))[0]


def select_pose_by_completeness(
    items: Sequence[PoseResult],
    target_bbox: Sequence[float],
    *,
    min_iou: float = _MIN_POSE_IOU,
    kpt_min: float = 0.25,
    require_min_iou: bool = True,
) -> PoseResult | None:
    """Возвращает самую полную PoseResult, совпадающую с target_bbox."""
    idx = select_pose_index_by_completeness(
        [(r.bbox, r.kcf, r.confidence) for r in items],
        target_bbox,
        min_iou=min_iou,
        kpt_min=kpt_min,
        require_min_iou=require_min_iou,
    )
    return items[idx] if idx is not None else None


@dataclass
class PoseResult:
    """Результат детекции позы человека (COCO 17 keypoints)."""

    bbox: list[float]  # [x1, y1, x2, y2]
    confidence: float
    kxy: list[list[float]]  # 17 keypoints [[x, y], ...]
    kcf: list[float]  # 17 keypoints confidences

    def completeness(self, min_conf: float = 0.25) -> float:
        """0..1: полнота скелета (лицо + тело). Ракурс лица сюда не входит."""
        return pose_completeness(self.kcf, min_conf)

    def is_facing_camera(self, min_conf: float = 0.25) -> bool:
        """
        Определяет, повернут ли человек лицом (или ракурсом 3/4/профиль) к камере.
        Keypoints:
          0: нос, 1: левый глаз, 2: правый глаз, 3: левое ухо, 4: правое ухо,
          5: левое плечо, 6: правое плечо
        """
        if not self.kcf or len(self.kcf) < 5:
            return True

        c_nose = self.kcf[0]
        c_leye = self.kcf[1]
        c_reye = self.kcf[2]
        c_lear = self.kcf[3]
        c_rear = self.kcf[4]

        # Признак затылка: уши видны уверенно, а лица нет
        if (c_lear > 0.4 and c_rear > 0.4) and (c_nose < 0.2 and c_leye < 0.2 and c_reye < 0.2):
            return False

        # Анфас или 3/4: виден нос и хотя бы один глаз
        if c_nose >= min_conf and (c_leye >= min_conf or c_reye >= min_conf):
            return True

        # Профиль: видны хотя бы 2 лицевые точки
        face_pts_ok = sum(1 for c in (c_nose, c_leye, c_reye) if c >= min_conf)
        if face_pts_ok >= 2:
            return True

        # Если нос и глаза совсем не видны -> считаем ракурс неподходящим для лица
        if max(c_nose, c_leye, c_reye) < min_conf:
            return False

        return True

    def face_confidence(self, min_conf: float = 0.25) -> float:
        """
        0..1: видимость лица у уже выбранной позы (нос/глаза и ракурс).
        Полноту скелета сюда не смешиваем — её считает completeness().
        """
        if not self.is_facing_camera(min_conf=min_conf):
            return 0.0
        if not self.kcf or len(self.kcf) < 3:
            return round(float(self.confidence) * 0.5, 4)

        c_nose, c_leye, c_reye = self.kcf[0], self.kcf[1], self.kcf[2]
        visible = [c for c in (c_nose, c_leye, c_reye) if c >= min_conf]
        if not visible:
            return 0.0

        kpt_score = sum(visible) / 3.0
        both_eyes = c_leye >= min_conf and c_reye >= min_conf
        frontal_bonus = 0.12 if c_nose >= min_conf and both_eyes else 0.06 if c_nose >= min_conf else 0.0
        raw = min(1.0, kpt_score + frontal_bonus)
        return round(min(1.0, raw * 0.8 + float(self.confidence) * 0.2), 4)

    def feet_position(self, min_conf: float = 0.2) -> list[float] | None:
        """
        Возвращает координаты точки опоры ног (середина между лодыжками 15 и 16).
        Если лодыжки не видны, возвращает низ BBox.
        """
        if len(self.kxy) >= 17 and len(self.kcf) >= 17:
            p_l, c_l = self.kxy[15], self.kcf[15]
            p_r, c_r = self.kxy[16], self.kcf[16]
            if c_l >= min_conf and c_r >= min_conf:
                return [(p_l[0] + p_r[0]) / 2.0, (p_l[1] + p_r[1]) / 2.0]
            if c_l >= min_conf:
                return [p_l[0], p_l[1]]
            if c_r >= min_conf:
                return [p_r[0], p_r[1]]
        if self.bbox and len(self.bbox) >= 4:
            return [(self.bbox[0] + self.bbox[2]) / 2.0, self.bbox[3]]
        return None

    def torso_center(self, min_conf: float = 0.2) -> list[float] | None:
        """
        Возвращает центр торса (середина между плечами 5,6 и бедрами 11,12).
        """
        if len(self.kxy) >= 13 and len(self.kcf) >= 13:
            pts = []
            for idx in (5, 6, 11, 12):
                if self.kcf[idx] >= min_conf:
                    pts.append(self.kxy[idx])
            if pts:
                mx = sum(p[0] for p in pts) / len(pts)
                my = sum(p[1] for p in pts) / len(pts)
                return [mx, my]
        if self.bbox and len(self.bbox) >= 4:
            return [(self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0]
        return None

    def with_offset(self, offset_x: float, offset_y: float) -> PoseResult:
        """Смещает BBox и ключевые точки на заданный оффсет (из кропа в кадр)."""
        new_bbox = [
            round(self.bbox[0] + offset_x, 1),
            round(self.bbox[1] + offset_y, 1),
            round(self.bbox[2] + offset_x, 1),
            round(self.bbox[3] + offset_y, 1),
        ]
        new_kxy = [[round(p[0] + offset_x, 1), round(p[1] + offset_y, 1)] for p in self.kxy]
        return PoseResult(
            bbox=new_bbox,
            confidence=self.confidence,
            kxy=new_kxy,
            kcf=list(self.kcf),
        )
