"""Типы данных и структуры результатов сервиса поз."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class PoseResult:
    """Результат детекции позы человека (COCO 17 keypoints)."""

    bbox: list[float]  # [x1, y1, x2, y2]
    confidence: float
    kxy: list[list[float]]  # 17 keypoints [[x, y], ...]
    kcf: list[float]  # 17 keypoints confidences

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
