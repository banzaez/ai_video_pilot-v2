"""Сервис пространственно-временного анализа окружения и изоляции треков."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from app.global_id.spatial import METER_PX


@dataclass
class NeighborContact:
    """Информация о контакте / сближении трека с соседом."""

    other_track_id: int
    min_dist_m: float
    first_frame: int
    last_frame: int
    contact_frames_count: int


def _tracklet_endpoint(t: dict[str, Any], key: str) -> tuple[float, float, str]:
    """Извлекает точку конца треклета (x, y, 'map' | 'image')."""
    map_key = f"map_{key}"
    mp = t.get(map_key)
    if isinstance(mp, (list, tuple)) and len(mp) >= 2:
        return float(mp[0]), float(mp[1]), "map"
    p = t.get(key)
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        return float(p[0]), float(p[1]), "image"
    bbox_key = f"bbox{key[-1]}" if key and key[-1] in ("0", "1") else None
    if bbox_key and isinstance(t.get(bbox_key), (list, tuple)) and len(t[bbox_key]) >= 4:
        b = t[bbox_key]
        return 0.5 * (float(b[0]) + float(b[2])), float(b[3]), "image"
    return 0.0, 0.0, "image"


def calc_points_dist_m(
    p1: tuple[float, float, str] | tuple[float, float],
    p2: tuple[float, float, str] | tuple[float, float],
    *,
    scale_px_per_m: float = METER_PX,
    fallback_median_h_px: float = 150.0,
) -> float:
    """Вычисляет евклидово расстояние между точками в метрах."""
    x1, y1 = float(p1[0]), float(p1[1])
    s1 = p1[2] if len(p1) > 2 else "map"
    x2, y2 = float(p2[0]), float(p2[1])
    s2 = p2[2] if len(p2) > 2 else "map"

    d_px = math.hypot(x1 - x2, y1 - y2)
    if s1 == "map" and s2 == "map":
        return d_px / max(scale_px_per_m, 1e-6)
    px_per_m = max(10.0, fallback_median_h_px / 1.70)
    return d_px / px_per_m


def point_to_segment_dist(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    """Минимальное евклидово расстояние от точки p до отрезка [a, b]."""
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


class TrackNeighborhoodIndex:
    """Пространственно-временной индекс треков и их взаимного расположения."""

    def __init__(
        self,
        scale_px_per_m: float = METER_PX,
        fps: float = 25.0,
        median_h_px: float = 150.0,
    ) -> None:
        self.scale_px_per_m = float(scale_px_per_m or METER_PX)
        self.fps = max(float(fps or 25.0), 1e-6)
        self.median_h_px = float(median_h_px or 150.0)

        # frame_index -> dict[track_id, (x, y, space)]
        self._frames: dict[int, dict[int, tuple[float, float, str]]] = defaultdict(dict)
        # track_id -> (f0, f1)
        self._spans: dict[int, tuple[int, int]] = {}
        # track_id -> metadata
        self._meta: dict[int, dict[str, Any]] = {}

    @classmethod
    def from_tracklets(
        cls,
        tracklets: list[dict[str, Any]],
        frames_pos: dict[int, dict[int, tuple[float, float, str]]] | None = None,
        *,
        scale_px_per_m: float = METER_PX,
        fps: float = 25.0,
    ) -> TrackNeighborhoodIndex:
        """Строит индекс из списка треклетов и покадровых детекций."""
        heights = [float(t.get("h", 0) or 0) for t in tracklets if float(t.get("h", 0) or 0) > 1]
        med_h = float(sorted(heights)[len(heights) // 2]) if heights else 150.0

        index = cls(scale_px_per_m=scale_px_per_m, fps=fps, median_h_px=med_h)
        for t in tracklets:
            tid = int(t["tracklet_id"])
            f0 = int(t.get("f0", 1))
            f1 = int(t.get("f1", f0))
            p0 = _tracklet_endpoint(t, "p0")
            p1 = _tracklet_endpoint(t, "p1")
            index._spans[tid] = (f0, f1)
            index._meta[tid] = t

            # Заполняем кадры
            for fi in range(f0, f1 + 1):
                if frames_pos and fi in frames_pos and tid in frames_pos[fi]:
                    pos = frames_pos[fi][tid]
                else:
                    if f1 <= f0:
                        pos = p0
                    else:
                        alpha = max(0.0, min(1.0, (fi - f0) / float(f1 - f0)))
                        space = "map" if (p0[2] == "map" and p1[2] == "map") else "image"
                        pos = (p0[0] + alpha * (p1[0] - p0[0]), p0[1] + alpha * (p1[1] - p0[1]), space)
                index._frames[fi][tid] = pos

        return index

    @classmethod
    def from_feet_doc(
        cls,
        feet_doc: dict[str, Any],
        *,
        scale_px_per_m: float = METER_PX,
        fps: float = 25.0,
    ) -> TrackNeighborhoodIndex:
        """Строит индекс из документа feet.json."""
        index = cls(scale_px_per_m=scale_px_per_m, fps=fps)
        for fr in feet_doc.get("frames") or []:
            fi = int(fr.get("frame_index", 0))
            for pt in fr.get("points") or []:
                tid = int(pt.get("track_id", 0))
                if tid <= 0 or "map" not in pt:
                    continue
                xy = pt["map"]
                pos = (float(xy[0]), float(xy[1]), "map")
                index._frames[fi][tid] = pos
                if tid not in index._spans:
                    index._spans[tid] = (fi, fi)
                else:
                    cur_f0, cur_f1 = index._spans[tid]
                    index._spans[tid] = (min(cur_f0, fi), max(cur_f1, fi))

        return index

    def get_track_position(self, track_id: int, frame_index: int) -> tuple[float, float, str] | None:
        """Возвращает координаты трека на указанном кадре."""
        fr = self._frames.get(frame_index)
        if fr and track_id in fr:
            return fr[track_id]
        return None

    def get_neighbors(
        self,
        track_id: int,
        radius_m: float = 2.0,
        *,
        exclude_ids: set[int] | None = None,
    ) -> list[NeighborContact]:
        """Возвращает детальный список соседей, приближавшихся к треку ближе radius_m."""
        if track_id not in self._spans or radius_m <= 0:
            return []

        excluded = set(exclude_ids or ()) | {track_id}
        f0, f1 = self._spans[track_id]

        contacts_map: dict[int, dict[str, Any]] = {}

        for fi in range(f0, f1 + 1):
            pos_self = self._frames.get(fi, {}).get(track_id)
            if not pos_self:
                continue

            for other_id, pos_other in self._frames.get(fi, {}).items():
                if other_id in excluded:
                    continue

                dist = calc_points_dist_m(
                    pos_self,
                    pos_other,
                    scale_px_per_m=self.scale_px_per_m,
                    fallback_median_h_px=self.median_h_px,
                )
                if dist <= radius_m:
                    rec = contacts_map.get(other_id)
                    if rec is None:
                        contacts_map[other_id] = {
                            "min_dist_m": dist,
                            "first_frame": fi,
                            "last_frame": fi,
                            "count": 1,
                        }
                    else:
                        rec["min_dist_m"] = min(rec["min_dist_m"], dist)
                        rec["last_frame"] = fi
                        rec["count"] += 1

        contacts = [
            NeighborContact(
                other_track_id=oid,
                min_dist_m=round(float(data["min_dist_m"]), 2),
                first_frame=int(data["first_frame"]),
                last_frame=int(data["last_frame"]),
                contact_frames_count=int(data["count"]),
            )
            for oid, data in sorted(contacts_map.items(), key=lambda kv: kv[1]["min_dist_m"])
        ]
        return contacts

    def is_track_alone(
        self,
        track_id: int,
        radius_m: float = 2.0,
        *,
        exclude_ids: set[int] | None = None,
        max_violation_frames: int = 0,
    ) -> bool:
        """Проверяет, был ли трек изолирован (без соседей ближе radius_m) на протяжении всей жизни."""
        neighbors = self.get_neighbors(track_id, radius_m=radius_m, exclude_ids=exclude_ids)
        if not neighbors:
            return True
        total_violation_frames = sum(n.contact_frames_count for n in neighbors)
        return total_violation_frames <= max_violation_frames

    def is_area_clear_during_gap(
        self,
        p_exit: tuple[float, float, str] | tuple[float, float],
        p_entry: tuple[float, float, str] | tuple[float, float],
        f_start: int,
        f_end: int,
        radius_m: float = 2.0,
        *,
        exclude_ids: set[int] | None = None,
        max_violation_frames: int = 0,
    ) -> bool:
        """Проверяет, что в промежутке кадров (f_start, f_end) через зону перехода [p_exit, p_entry] не проходили посторонние."""
        if f_end <= f_start + 1 or radius_m <= 0:
            return True

        excluded = set(exclude_ids or ())
        seg_a = (float(p_exit[0]), float(p_exit[1]))
        seg_b = (float(p_entry[0]), float(p_entry[1]))
        is_map_space = (len(p_exit) > 2 and p_exit[2] == "map") and (len(p_entry) > 2 and p_entry[2] == "map")
        scale = self.scale_px_per_m if is_map_space else max(10.0, self.median_h_px / 1.70)

        violations = 0
        for fi in range(f_start + 1, f_end):
            for oid, pos_other in self._frames.get(fi, {}).items():
                if oid in excluded:
                    continue
                d_px = point_to_segment_dist((pos_other[0], pos_other[1]), seg_a, seg_b)
                dist_m = d_px / scale
                if dist_m < radius_m:
                    violations += 1
                    if violations > max_violation_frames:
                        return False

        return True

    def is_pair_transition_clear(
        self,
        track_a_id: int,
        track_b_id: int,
        radius_m: float = 2.0,
        *,
        group_a: set[int] | None = None,
        group_b: set[int] | None = None,
        max_violation_frames: int = 0,
    ) -> bool:
        """Проверяет, что пара A -> B изолирована от третьих лиц C not in (group_a | group_b)."""
        own_a = set(group_a or (track_a_id,))
        own_b = set(group_b or (track_b_id,))
        all_own = own_a | own_b

        # 1. Проверяем изоляцию треков группы A
        for tid in own_a:
            if not self.is_track_alone(
                tid,
                radius_m=radius_m,
                exclude_ids=all_own,
                max_violation_frames=max_violation_frames,
            ):
                return False

        # 2. Проверяем изоляцию треков группы B
        for tid in own_b:
            if not self.is_track_alone(
                tid,
                radius_m=radius_m,
                exclude_ids=all_own,
                max_violation_frames=max_violation_frames,
            ):
                return False

        # 3. Проверяем зазор между концом A и началом B
        if track_a_id in self._spans and track_b_id in self._spans:
            _, f1_a = self._spans[track_a_id]
            f0_b, _ = self._spans[track_b_id]
            if f0_b > f1_a + 1:
                p_exit = self.get_track_position(track_a_id, f1_a)
                p_entry = self.get_track_position(track_b_id, f0_b)
                if p_exit and p_entry:
                    if not self.is_area_clear_during_gap(
                        p_exit,
                        p_entry,
                        f1_a,
                        f0_b,
                        radius_m=radius_m,
                        exclude_ids=all_own,
                        max_violation_frames=max_violation_frames,
                    ):
                        return False

        return True
