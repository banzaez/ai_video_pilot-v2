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


class DayNeighborhoodIndex:
    """Пространственно-временной индекс треков дня на 2D-карте этажа для всех камер."""

    def __init__(
        self,
        scale_px_per_m: float = METER_PX,
        time_step_sec: float = 0.2,
    ) -> None:
        self.scale_px_per_m = float(scale_px_per_m or METER_PX)
        self.time_step_sec = max(0.05, float(time_step_sec or 0.2))

        # bin_index -> dict[uid, (x, y)] (точки на 2D-карте)
        self._time_bins: dict[int, dict[str, tuple[float, float]]] = defaultdict(dict)
        # uid -> (t0_sec, t1_sec)
        self._spans: dict[str, tuple[float, float]] = {}
        # uid -> camera_name / camera_idx
        self._camera_by_uid: dict[str, int] = {}

    def _t_to_bin(self, t_sec: float) -> int:
        return int(round(t_sec / self.time_step_sec))

    @classmethod
    def from_nodes_and_feet(
        cls,
        nodes: list[dict[str, Any]],
        feet_by_session: dict[str, dict[int, list[dict[str, Any]]]],
        *,
        scale_px_per_m: float = METER_PX,
        time_step_sec: float = 0.2,
    ) -> DayNeighborhoodIndex:
        """Строит глобальный индекс дня по всем узлам и покадровым точкам ног."""
        idx = cls(scale_px_per_m=scale_px_per_m, time_step_sec=time_step_sec)

        for node in nodes:
            uid = str(node["uid"])
            t0 = float(node["t0"])
            t1 = float(node["t1"])
            idx._spans[uid] = (t0, t1)
            idx._camera_by_uid[uid] = int(node.get("camera_index", 0))

            sk = str(node.get("session_key", ""))
            tid = int(node.get("track_id", 0))
            ft_pts = feet_by_session.get(sk, {}).get(tid, [])

            if ft_pts and len(ft_pts) >= 1:
                # Вычисляем t_sec по позиции кадра внутри трека через интерполяцию
                f0 = float(node.get("f0", 0))
                duration = max(0.01, t1 - t0)
                f_span = max(1.0, float(node.get("f1", f0)) - f0)

                for pt in ft_pts:
                    fi = float(pt.get("frame_index", f0))
                    # Вычисляем t_sec
                    frac = max(0.0, min(1.0, (fi - f0) / f_span))
                    t_pt = t0 + frac * duration
                    mp = pt.get("map")
                    if isinstance(mp, (list, tuple)) and len(mp) >= 2:
                        b = idx._t_to_bin(t_pt)
                        idx._time_bins[b][uid] = (float(mp[0]), float(mp[1]))
            else:
                # Fallback: интерполяция между map_p0 и map_p1
                p0 = node.get("map_p0")
                p1 = node.get("map_p1")
                if p0 and p1:
                    p0x, p0y = float(p0[0]), float(p0[1])
                    p1x, p1y = float(p1[0]), float(p1[1])
                    b0 = idx._t_to_bin(t0)
                    b1 = idx._t_to_bin(t1)
                    span_b = max(1, b1 - b0)
                    for b in range(b0, b1 + 1):
                        alpha = (b - b0) / float(span_b)
                        idx._time_bins[b][uid] = (p0x + alpha * (p1x - p0x), p0y + alpha * (p1y - p0y))

        return idx

    def get_position_at_bin(self, uid: str, b: int) -> tuple[float, float] | None:
        """Координаты трека в бине времени b."""
        return self._time_bins.get(b, {}).get(uid)

    def check_simultaneous_proximity_and_isolation(
        self,
        node_a: dict[str, Any],
        node_b: dict[str, Any],
        *,
        radius_m: float = 2.0,
        max_dist_m: float = 3.0,
        min_overlap_sec: float = 5.0,
        exclude_uids: set[str] | None = None,
        max_violation_bins: int = 2,
    ) -> tuple[bool, float, float]:
        """Проверяет одновременное нахождение рядом на 2D-карте и изоляцию от 3-х лиц на всех камерах.

        Возвращает: (is_valid, avg_dist_m, overlap_sec)
        """
        uid_a = str(node_a["uid"])
        uid_b = str(node_b["uid"])
        t0_a, t1_a = float(node_a["t0"]), float(node_a["t1"])
        t0_b, t1_b = float(node_b["t0"]), float(node_b["t1"])

        ov_start = max(t0_a, t0_b)
        ov_end = min(t1_a, t1_b)
        overlap_sec = ov_end - ov_start
        if overlap_sec < min_overlap_sec - 1e-6:
            return False, 0.0, overlap_sec

        b_start = self._t_to_bin(ov_start)
        b_end = self._t_to_bin(ov_end)
        if b_end <= b_start:
            return False, 0.0, overlap_sec

        excluded = set(exclude_uids or ()) | {uid_a, uid_b}
        scale = self.scale_px_per_m

        dists: list[float] = []
        violations = 0

        for b in range(b_start, b_end + 1):
            pos_a = self.get_position_at_bin(uid_a, b)
            pos_b = self.get_position_at_bin(uid_b, b)
            if not pos_a or not pos_b:
                continue

            d_px = math.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1])
            d_m = d_px / scale
            dists.append(d_m)

            # Проверяем, нет ли вокруг третьих лиц на любой камере
            if radius_m > 0:
                for other_uid, pos_other in self._time_bins.get(b, {}).items():
                    if other_uid in excluded:
                        continue
                    d_other_a = math.hypot(pos_other[0] - pos_a[0], pos_other[1] - pos_a[1]) / scale
                    d_other_b = math.hypot(pos_other[0] - pos_b[0], pos_other[1] - pos_b[1]) / scale
                    if min(d_other_a, d_other_b) < radius_m:
                        violations += 1
                        if violations > max_violation_bins:
                            return False, (sum(dists) / len(dists)) if dists else 0.0, overlap_sec

        if not dists:
            return False, 0.0, overlap_sec

        avg_dist = sum(dists) / len(dists)
        if avg_dist > max_dist_m:
            return False, avg_dist, overlap_sec

        return True, avg_dist, overlap_sec

    def check_transition_gap_and_isolation(
        self,
        node_a: dict[str, Any],
        node_b: dict[str, Any],
        *,
        radius_m: float = 2.0,
        max_dist_m: float = 3.0,
        max_gap_sec: float = 10.0,
        max_speed_mps: float = 2.5,
        exclude_uids: set[str] | None = None,
        max_violation_bins: int = 2,
    ) -> tuple[bool, float, float, float]:
        """Проверяет последовательный переход между камерами и чистоту зоны перехода.

        Возвращает: (is_valid, dist_m, gap_sec, speed_mps)
        """
        uid_a = str(node_a["uid"])
        uid_b = str(node_b["uid"])
        t1_a = float(node_a["t1"])
        t0_b = float(node_b["t0"])

        gap_sec = t0_b - t1_a
        if gap_sec < -0.5 or gap_sec > max_gap_sec:
            return False, 0.0, gap_sec, 0.0

        p1_a = node_a.get("map_p1")
        p0_b = node_b.get("map_p0")
        if not p1_a or not p0_b:
            return False, 0.0, gap_sec, 0.0

        p1x, p1y = float(p1_a[0]), float(p1_a[1])
        p0x, p0y = float(p0_b[0]), float(p0_b[1])
        scale = self.scale_px_per_m
        dist_m = math.hypot(p1x - p0x, p1y - p0y) / scale
        if max_dist_m > 0 and dist_m > max_dist_m:
            return False, dist_m, gap_sec, 0.0

        dt = max(0.5, gap_sec)
        speed_mps = dist_m / dt
        if max_speed_mps > 0 and speed_mps > max_speed_mps:
            return False, dist_m, gap_sec, speed_mps

        # Проверяем изоляцию в окне перехода на всех камерах
        excluded = set(exclude_uids or ()) | {uid_a, uid_b}
        b_start = self._t_to_bin(t1_a - 2.0)
        b_end = self._t_to_bin(t0_b + 2.0)
        violations = 0

        for b in range(b_start, b_end + 1):
            for other_uid, pos_other in self._time_bins.get(b, {}).items():
                if other_uid in excluded:
                    continue
                # Расстояние до отрезка перехода
                d_px = point_to_segment_dist((pos_other[0], pos_other[1]), (p1x, p1y), (p0x, p0y))
                d_m = d_px / scale
                if d_m < radius_m:
                    violations += 1
                    if violations > max_violation_bins:
                        return False, dist_m, gap_sec, speed_mps

        return True, dist_m, gap_sec, speed_mps
