"""Stage day_link: глобальная межкамерная склейка дня (Pass 1→2→4)."""

from __future__ import annotations

import logging
import math
import os
from bisect import bisect_right
from datetime import datetime
from typing import Any

from app.artifact_meta import attach_artifact_meta
from app.config import Settings, day_links_json_path, day_results_dir
from app.global_id.group_reid import TrackGroupReid, embed_session_tracks, make_group_reid_context
from app.global_id.spatial import METER_PX
from app.io.json_util import load_tracking_json, save_debug_json
from app.session.discover import parse_day_input
from app.tracklet.link_mcf import (
    _gap_score,
    _unique_groups,
    link_hungarian_chains,
)
from app.util.intervals import intervals_overlap, pair_embed_score
import numpy as np

logger = logging.getLogger(__name__)

_CANDIDATE_TOP_K = 50


def _as_xy(p: Any) -> tuple[float, float]:
    if not p or not isinstance(p, (list, tuple)) or len(p) < 2:
        return 0.0, 0.0
    return float(p[0]), float(p[1])


def _tracklet_spatial_point(tracklet: dict[str, Any], key: str) -> tuple[float, float, str]:
    suffix = key[-1] if key else ""
    map_key = f"map_{key}"
    mp = tracklet.get(map_key)
    has_kpts = isinstance(tracklet.get(f"kxy{suffix}"), list)
    src = str(tracklet.get(f"map_src{suffix}") or "")
    use_map = isinstance(mp, (list, tuple)) and len(mp) >= 2
    if use_map and has_kpts and not src.startswith("kpt"):
        use_map = False
    if use_map:
        return float(mp[0]), float(mp[1]), "map"
    x, y = _as_xy(tracklet.get(key))
    return x, y, "image"


def _pair_uses_map(a: dict[str, Any], b: dict[str, Any], key_a: str, key_b: str) -> bool:
    _, _, sa = _tracklet_spatial_point(a, key_a)
    _, _, sb = _tracklet_spatial_point(b, key_b)
    return sa == "map" and sb == "map"


def _velocity(tracklet: dict[str, Any]) -> tuple[float, float]:
    t0 = float(tracklet["t0"])
    t1 = float(tracklet["t1"])
    dt = t1 - t0
    if dt <= 1e-6:
        return 0.0, 0.0
    x0, y0, _ = _tracklet_spatial_point(tracklet, "p0")
    x1, y1, _ = _tracklet_spatial_point(tracklet, "p1")
    return (x1 - x0) / dt, (y1 - y0) / dt


def _predict_p1(tracklet: dict[str, Any], t: float) -> tuple[float, float]:
    x1, y1, _ = _tracklet_spatial_point(tracklet, "p1")
    vx, vy = _velocity(tracklet)
    dt = t - float(tracklet["t1"])
    return x1 + vx * dt, y1 + vy * dt


def _spatial_residual(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    sigma_px: float,
    sigma_m: float,
) -> tuple[float, float]:
    p1x, p1y, _ = _tracklet_spatial_point(a, "p1")
    b0x, b0y, space = _tracklet_spatial_point(b, "p0")
    dist_last = math.hypot(p1x - b0x, p1y - b0y)
    pred_x, pred_y = _predict_p1(a, float(b["t0"]))
    dist_pred = math.hypot(pred_x - b0x, pred_y - b0y)
    residual = min(dist_last, dist_pred)

    if space == "map" and sigma_m > 0:
        residual_m = residual / METER_PX
        if sigma_m <= 0:
            return 1.0, residual_m
        return float(math.exp(-residual_m / sigma_m)), residual_m
    if sigma_px <= 0:
        return 1.0, residual
    return float(math.exp(-residual / sigma_px)), residual


def _spatial_threshold_exceeded(
    residual: float,
    *,
    space: str,
    max_px: float,
    max_m: float,
) -> bool:
    if space == "map":
        if max_m <= 0:
            return False
        return residual > max_m
    if max_px <= 0:
        return False
    return residual > max_px


def _motion_score(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    sigma_px: float,
    sigma_m: float = 0.0,
) -> tuple[float, float]:
    motion, residual = _spatial_residual(a, b, sigma_px=sigma_px, sigma_m=sigma_m)
    gap = max(0.0, float(b["t0"]) - float(a["t1"]))
    if gap <= 1.5:
        vax, vay = _velocity(a)
        vbx, vby = _velocity(b)
        va_norm = math.hypot(vax, vay)
        vb_norm = math.hypot(vbx, vby)
        if va_norm > 20.0 and vb_norm > 20.0:
            cos_theta = (vax * vbx + vay * vby) / (va_norm * vb_norm)
            if cos_theta < -0.4:
                factor = max(0.4, (cos_theta + 1.0) / 0.6)
                motion *= factor
    return motion, residual


def _bbox_dim(tracklet: dict[str, Any], *, axis: str) -> float:
    key = "h" if axis == "h" else "w"
    raw = tracklet.get(key)
    if raw is not None and float(raw) > 1:
        return float(raw)
    idx0, idx1 = (1, 3) if axis == "h" else (0, 2)
    vals: list[float] = []
    for b in (tracklet.get("bbox0") or [], tracklet.get("bbox1") or []):
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            vals.append(max(0.0, float(b[idx1]) - float(b[idx0])))
    return float(np.median(vals)) if vals else 0.0


def _height(tracklet: dict[str, Any]) -> float:
    return _bbox_dim(tracklet, axis="h")


def _width(tracklet: dict[str, Any]) -> float:
    return _bbox_dim(tracklet, axis="w")


def _dim_sim(a: float, b: float, *, log_scale: float) -> float:
    if a <= 1 or b <= 1 or log_scale <= 0:
        return 1.0
    ratio = abs(math.log(a / b))
    return float(max(0.0, 1.0 - ratio / log_scale))


def _size_score(a: dict[str, Any], b: dict[str, Any], *, log_scale: float) -> float:
    ha, hb = _height(a), _height(b)
    wa, wb = _width(a), _width(b)
    h_sim = _dim_sim(ha, hb, log_scale=log_scale)
    if wa <= 1 or wb <= 1:
        return h_sim
    w_sim = _dim_sim(wa, wb, log_scale=log_scale)
    taller = max(ha, hb)
    shorter = min(ha, hb)
    if taller > 1 and shorter / taller <= 0.85 and w_sim >= 0.70:
        return w_sim
    return h_sim


def _combo_score(
    reid: float,
    motion: float,
    size: float,
    gap: float,
    *,
    w_reid: float,
    w_motion: float,
    w_size: float,
    w_gap: float,
) -> float:
    parts = (
        (reid, w_reid),
        (motion, w_motion),
        (size, w_size),
        (gap, w_gap),
    )
    num = 0.0
    den = 0.0
    for value, weight in parts:
        if weight <= 0:
            continue
        num += weight * float(value)
        den += weight
    if den <= 0:
        return float(reid)
    return float(num / den)


def _parse_iso_to_day_sec(iso_str: str | None) -> float:
    """Переводит ISO timestamp (e.g. '2026-04-01T11:30:50') в секунды от полуночи."""
    if not iso_str:
        return 0.0
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.hour * 3600.0 + dt.minute * 60.0 + dt.second + dt.microsecond / 1e6
    except Exception:
        return 0.0


def _load_feet_map_trajectories(feet_path: str) -> dict[int, list[dict[str, Any]]]:
    """Загружает упорядоченные по времени точки ног на 2D-карте этажа."""
    if not os.path.isfile(feet_path):
        return {}
    feet_doc = load_tracking_json(feet_path)
    trajs: dict[int, list[dict[str, Any]]] = {}
    for f in feet_doc.get("frames", []):
        fi = int(f.get("frame_index", 0))
        for p in f.get("points", []):
            tid = int(p.get("track_id", 0))
            if tid <= 0 or "map" not in p:
                continue
            trajs.setdefault(tid, []).append({
                "frame_index": fi,
                "map": [float(p["map"][0]), float(p["map"][1])],
                "confidence": float(p.get("confidence", 1.0)),
            })
    for tid in trajs:
        trajs[tid].sort(key=lambda pt: pt["frame_index"])
    return trajs


def _bbox_wh(bbox: Any) -> tuple[float, float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return 0.0, 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])), max(0.0, float(bbox[3]) - float(bbox[1]))


def _bbox_feet(bbox: Any) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    return [0.5 * (float(bbox[0]) + float(bbox[2])), float(bbox[3])]


def _camera_index_from_session(session_key: str, info_doc: dict[str, Any]) -> int:
    raw = info_doc.get("camera_index")
    if raw is not None:
        try:
            idx = int(raw)
            if idx > 0:
                return idx
        except (TypeError, ValueError):
            pass
    try:
        return int(session_key.split("_")[0])
    except (TypeError, ValueError, IndexError):
        return 0


def _extract_track_data(
    session_key: str,
    session_root: str,
    reid_by_track: dict[int, TrackGroupReid],
) -> list[dict[str, Any]]:
    """Извлекает и выравнивает все треки сессии по единому дневному таймлайну."""
    info_path = os.path.join(session_root, "info.json")
    tracking_path = os.path.join(session_root, "tracking.json")
    feet_path = os.path.join(session_root, "feet.json")

    if not os.path.isfile(info_path) or not os.path.isfile(tracking_path):
        return []

    info_doc = load_tracking_json(info_path)
    tracking_doc = load_tracking_json(tracking_path)
    camera_name = str(info_doc.get("camera") or f"Camera_{session_key.split('_')[0]}")
    camera_idx = _camera_index_from_session(session_key, info_doc)
    fps = float(info_doc.get("fps") or tracking_doc.get("fps") or 25.0)
    started_at_str = str(info_doc.get("parsed", {}).get("started_at") or info_doc.get("started_at") or "")
    if not started_at_str and info_doc.get("parts"):
        started_at_str = str(info_doc["parts"][0].get("started_at") or "")
    session_start_sec = _parse_iso_to_day_sec(started_at_str)

    feet_trajs = _load_feet_map_trajectories(feet_path)

    track_meta: dict[int, dict[str, Any]] = {}
    for f in tracking_doc.get("frames", []):
        fi = int(f.get("frame_index", 0))
        for d in f.get("detections", []):
            tid = int(d.get("track_id") or d.get("tracklet_id") or 0)
            if tid <= 0:
                continue
            bbox = d.get("bbox")
            rec = track_meta.get(tid)
            if rec is None:
                rec = {
                    "frames": [],
                    "bbox0": bbox,
                    "bbox1": bbox,
                    "f0": fi,
                    "f1": fi,
                }
                track_meta[tid] = rec
            rec["frames"].append(fi)
            if fi < rec["f0"]:
                rec["f0"] = fi
                rec["bbox0"] = bbox
            if fi >= rec["f1"]:
                rec["f1"] = fi
                rec["bbox1"] = bbox

    track_nodes: list[dict[str, Any]] = []

    for tid, meta in sorted(track_meta.items()):
        frames = meta["frames"]
        f0, f1 = int(meta["f0"]), int(meta["f1"])
        t0_sec = session_start_sec + (f0 / fps)
        t1_sec = session_start_sec + (f1 / fps)
        bbox0 = meta.get("bbox0")
        bbox1 = meta.get("bbox1")
        w0, h0 = _bbox_wh(bbox0)
        w1, h1 = _bbox_wh(bbox1)
        img_p0 = _bbox_feet(bbox0)
        img_p1 = _bbox_feet(bbox1)

        ft = feet_trajs.get(tid, [])
        map_p0 = ft[0]["map"] if ft else None
        map_p1 = ft[-1]["map"] if ft else None
        avg_speed_mps = 0.0
        if len(ft) >= 2 and map_p0 and map_p1:
            dt = max(0.1, (ft[-1]["frame_index"] - ft[0]["frame_index"]) / fps)
            dist_total_m = math.hypot(map_p1[0] - map_p0[0], map_p1[1] - map_p0[1]) / METER_PX
            avg_speed_mps = dist_total_m / dt

        reid_rec = reid_by_track.get(tid)
        reid_arr = reid_rec.embs if reid_rec is not None else None
        crop_files = list(reid_rec.crop_files) if reid_rec is not None else []

        track_nodes.append({
            "uid": f"{session_key}#{tid}",
            "session_key": session_key,
            "camera": camera_name,
            "camera_index": camera_idx,
            "track_id": tid,
            "f0": f0,
            "f1": f1,
            "t0": float(t0_sec),
            "t1": float(t1_sec),
            "duration_sec": round(t1_sec - t0_sec, 2),
            "p0": img_p0,
            "p1": img_p1,
            "map_p0": [float(map_p0[0]), float(map_p0[1])] if map_p0 else None,
            "map_p1": [float(map_p1[0]), float(map_p1[1])] if map_p1 else None,
            "map_src0": "kpt_map" if map_p0 else "",
            "map_src1": "kpt_map" if map_p1 else "",
            "bbox0": list(bbox0) if isinstance(bbox0, (list, tuple)) else None,
            "bbox1": list(bbox1) if isinstance(bbox1, (list, tuple)) else None,
            "h": float(h0 or h1 or 0.0),
            "w": float(w0 or w1 or 0.0),
            "avg_speed_mps": round(avg_speed_mps, 2),
            "n_frames": len(frames),
            "reid_embs": reid_arr,
            "has_reid": reid_arr is not None,
            "crops": crop_files,
            "best_crop": crop_files[0] if crop_files else None,
        })

    return track_nodes


def _score_pair(
    u: dict[str, Any],
    v: dict[str, Any],
    settings: Settings,
) -> dict[str, Any] | None:
    """Скор перехода u → v как tracklet_link: ReID + motion + size + gap, без лиц.

    На одной камере пересечение по времени запрещено. С разных камер персона
    может быть видна одновременно — overlap не отбрасывает пару.
    """
    is_same_camera = int(u["camera_index"]) == int(v["camera_index"])
    u_t0, u_t1 = float(u["t0"]), float(u["t1"])
    v_t0, v_t1 = float(v["t0"]), float(v["t1"])
    gap_sec = v_t0 - u_t1
    max_gap_sec = float(settings.day_link_max_gap_sec)
    is_overlap = intervals_overlap(u_t0, u_t1, v_t0, v_t1)

    if is_same_camera:
        if is_overlap or gap_sec < 0.0 or gap_sec > max_gap_sec:
            return None
    elif not is_overlap and (gap_sec < 0.0 or gap_sec > max_gap_sec):
        return None

    s_reid = pair_embed_score(u.get("reid_embs"), v.get("reid_embs"))
    min_reid_score = float(settings.day_link_min_reid_score)
    if s_reid is None or s_reid < min_reid_score:
        return None

    s_motion, residual = _motion_score(
        u,
        v,
        sigma_px=float(settings.day_link_motion_sigma_px),
        sigma_m=float(settings.day_link_motion_sigma_m),
    )
    space = "map" if _pair_uses_map(u, v, "p1", "p0") else "image"
    if _spatial_threshold_exceeded(
        residual,
        space=space,
        max_px=float(settings.day_link_max_spatial_px),
        max_m=float(settings.day_link_max_spatial_m),
    ):
        return None

    dist_m = float(residual) if space == "map" else float(residual) / METER_PX
    dt = max(0.5, abs(gap_sec) if is_overlap else max(gap_sec, 1e-6))
    p1x, p1y, _ = _tracklet_spatial_point(u, "p1")
    b0x, b0y, _ = _tracklet_spatial_point(v, "p0")
    dist_last = math.hypot(p1x - b0x, p1y - b0y)
    dist_last_m = dist_last / METER_PX if space == "map" else dist_last / METER_PX
    speed_mps = dist_last_m / dt

    s_size = _size_score(u, v, log_scale=float(settings.day_link_size_log_scale))
    gap_for_score = 0.0 if is_overlap else max(0.0, gap_sec)
    s_gap = _gap_score(gap_for_score, max_gap_sec)
    combo = _combo_score(
        float(s_reid),
        float(s_motion),
        float(s_size),
        float(s_gap),
        w_reid=float(settings.day_link_w_reid),
        w_motion=float(settings.day_link_w_motion),
        w_size=float(settings.day_link_w_size),
        w_gap=float(settings.day_link_w_gap),
    )

    reid_str = f"ReID={s_reid:.2f}"
    motion_str = (
        f"Motion={s_motion:.2f} (Δd={dist_m:.1f}м, v={speed_mps:.1f}м/с, Δt={gap_sec:.1f}с)"
    )
    size_str = f"Size={s_size:.2f}"

    return {
        "from": u["uid"],
        "to": v["uid"],
        "from_idx": int(u["_idx"]),
        "to_idx": int(v["_idx"]),
        "from_session": u["session_key"],
        "from_camera": u["camera"],
        "from_track": u["track_id"],
        "to_session": v["session_key"],
        "to_camera": v["camera"],
        "to_track": v["track_id"],
        "is_same_camera": is_same_camera,
        "is_overlap": is_overlap,
        "score": round(float(combo), 4),
        "reid": round(float(s_reid), 4),
        "motion": round(float(s_motion), 4),
        "size": round(float(s_size), 4),
        "dist_m": round(dist_m, 2),
        "gap_sec": round(gap_sec, 2),
        "speed_mps": round(speed_mps, 2),
        "reason": f"{reid_str}, {motion_str}, {size_str}",
    }


def _build_candidate_edges(
    nodes: list[dict[str, Any]],
    settings: Settings,
) -> list[dict[str, Any]]:
    """Пары A→B: same-cam только после конца A; cross-cam — и overlap, и пауза ≤ max_gap."""
    n = len(nodes)
    if n < 2:
        return []
    order = sorted(range(n), key=lambda i: float(nodes[i]["t0"]))
    t0s = [float(nodes[i]["t0"]) for i in order]
    max_gap = float(settings.day_link_max_gap_sec)
    edges: list[dict[str, Any]] = []
    for u in nodes:
        hi = float(u["t1"]) + max_gap
        right = bisect_right(t0s, hi)
        for pos in range(right):
            v = nodes[order[pos]]
            if v["_idx"] == u["_idx"]:
                continue
            edge = _score_pair(u, v, settings)
            if edge is not None:
                edges.append(edge)
    return edges


def _same_cam_overlap(
    ids: set[int],
    nodes: list[dict[str, Any]],
) -> bool:
    items = sorted(ids)
    for i, a in enumerate(items):
        na = nodes[a]
        for b in items[i + 1 :]:
            nb = nodes[b]
            if int(na["camera_index"]) != int(nb["camera_index"]):
                continue
            if intervals_overlap(float(na["t0"]), float(na["t1"]), float(nb["t0"]), float(nb["t1"])):
                return True
    return False


def _split_same_cam_overlap(
    groups: list[list[int]],
    nodes: list[dict[str, Any]],
) -> list[list[int]]:
    spans = {i: (float(nodes[i]["t0"]), float(nodes[i]["t1"])) for i in range(len(nodes))}
    final: list[list[int]] = []
    for group in groups:
        bucket: list[list[int]] = []
        for tid in sorted(group, key=lambda x: spans.get(x, (0.0, 0.0))[0]):
            placed = False
            for part in bucket:
                if not _same_cam_overlap(set(part) | {tid}, nodes):
                    part.append(tid)
                    placed = True
                    break
            if not placed:
                bucket.append([tid])
        final.extend(bucket)
    return final


def _group_span(group: list[int], nodes: list[dict[str, Any]]) -> tuple[float, float]:
    t0 = min(float(nodes[i]["t0"]) for i in group)
    t1 = max(float(nodes[i]["t1"]) for i in group)
    return t0, t1


def _exit_idx(group: list[int], nodes: list[dict[str, Any]]) -> int:
    return max(group, key=lambda i: (float(nodes[i]["t1"]), i))


def _entry_idx(group: list[int], nodes: list[dict[str, Any]]) -> int:
    return min(group, key=lambda i: (float(nodes[i]["t0"]), i))


def _public_edge(edge: dict[str, Any], *, pass_n: int, prefix: str) -> dict[str, Any]:
    skip = {"from_idx", "to_idx"}
    out = {k: v for k, v in edge.items() if k not in skip}
    out["pass"] = pass_n
    out["reason"] = f"{prefix}: {edge.get('reason') or ''}".strip()
    return out


def _merge_groups_by_endpoints(
    groups: list[list[int]],
    by_pair: dict[tuple[int, int], dict[str, Any]],
    nodes: list[dict[str, Any]],
    *,
    min_score: float,
    accept: Any,
    solver: str,
) -> tuple[list[list[int]], set[tuple[int, int]]]:
    """Hungarian: стык цепочек только выход A → вход B. Overlap разных камер можно."""
    if min_score <= 0 or len(groups) < 2:
        return groups, set()

    best: dict[tuple[int, int], tuple[float, int, int]] = {}
    for ga, group_a in enumerate(groups):
        if not group_a:
            continue
        for gb, group_b in enumerate(groups):
            if ga == gb or not group_b:
                continue
            if _same_cam_overlap(set(group_a) | set(group_b), nodes):
                continue
            exit_id = _exit_idx(group_a, nodes)
            entry_id = _entry_idx(group_b, nodes)
            rec = by_pair.get((exit_id, entry_id))
            if rec is None or not accept(rec):
                continue
            best[(ga, gb)] = (float(rec["score"]), exit_id, entry_id)

    if not best:
        return groups, set()

    group_ids = list(range(len(groups)))
    t0_g = {gi: _group_span(groups[gi], nodes)[0] for gi in group_ids if groups[gi]}
    fake_edges = [(ga, gb, val) for (ga, gb), (val, _, _) in best.items()]
    chains = link_hungarian_chains(group_ids, fake_edges, min_score=min_score, t0=t0_g)

    used: set[tuple[int, int]] = set()
    new_groups: list[list[int]] = []
    for chain in chains:
        merged: list[int] = []
        for i, gi in enumerate(chain):
            merged.extend(groups[gi])
            if i + 1 < len(chain):
                rec = best.get((gi, chain[i + 1]))
                if rec:
                    used.add((rec[1], rec[2]))
        new_groups.append(sorted(set(merged), key=lambda tid: (float(nodes[tid]["t0"]), tid)))

    new_groups = _split_same_cam_overlap(new_groups, nodes)
    return _unique_groups(new_groups), used


def _pass4_handover(
    groups: list[list[int]],
    by_pair: dict[tuple[int, int], dict[str, Any]],
    nodes: list[dict[str, Any]],
    *,
    max_overlap_sec: float,
    min_score: float,
    min_reid: float,
    solver: str,
) -> tuple[list[list[int]], set[tuple[int, int]]]:
    if max_overlap_sec <= 0 or min_score <= 0 or len(groups) < 2:
        return groups, set()

    work = [sorted(g, key=lambda tid: (float(nodes[tid]["t0"]), tid)) for g in groups if g]
    best: dict[tuple[int, int], tuple[float, int, int]] = {}
    for ga, group_a in enumerate(work):
        sa = _group_span(group_a, nodes)
        for gb, group_b in enumerate(work):
            if ga == gb:
                continue
            sb = _group_span(group_b, nodes)
            if not (sa[0] <= sb[0] < sa[1] < sb[1]):
                continue
            overlap = sa[1] - sb[0]
            if overlap > max_overlap_sec + 1e-9:
                continue
            exit_id = _exit_idx(group_a, nodes)
            entry_id = _entry_idx(group_b, nodes)
            if int(nodes[exit_id]["camera_index"]) == int(nodes[entry_id]["camera_index"]):
                continue
            rec = by_pair.get((exit_id, entry_id))
            if rec is None or not rec.get("is_overlap"):
                continue
            if float(rec["score"]) < min_score:
                continue
            reid = rec.get("reid")
            if min_reid > 0 and (reid is None or float(reid) < min_reid):
                continue
            if _same_cam_overlap(set(group_a) | set(group_b), nodes):
                continue
            best[(ga, gb)] = (float(rec["score"]), exit_id, entry_id)

    if not best:
        return work, set()

    group_ids = list(range(len(work)))
    t0_g = {gi: _group_span(work[gi], nodes)[0] for gi in group_ids if work[gi]}
    fake_edges = [(ga, gb, combo) for (ga, gb), (combo, _, _) in best.items()]
    chains = link_hungarian_chains(group_ids, fake_edges, min_score=min_score, t0=t0_g)

    used: set[tuple[int, int]] = set()
    new_groups: list[list[int]] = []
    for chain in chains:
        merged: list[int] = []
        for i, gi in enumerate(chain):
            merged.extend(work[gi])
            if i + 1 < len(chain):
                rec = best.get((gi, chain[i + 1]))
                if rec:
                    used.add((rec[1], rec[2]))
        new_groups.append(sorted(set(merged), key=lambda tid: (float(nodes[tid]["t0"]), tid)))

    new_groups = _split_same_cam_overlap(new_groups, nodes)
    return _unique_groups(new_groups), used


def _crop_ref(session_key: str, file: str | None) -> dict[str, str] | None:
    if not file:
        return None
    return {"session_key": session_key, "file": file}


def link_day_tracks(
    nodes: list[dict[str, Any]],
    settings: Settings,
) -> dict[str, Any]:
    """Межкамерный солвер дня: Pass 1 (стык по ReID) → 2 → 4."""
    if not nodes:
        return {
            "n_persons": 0,
            "persons": [],
            "edges": [],
            "candidate_edges": [],
            "stats": {},
        }

    for i, node in enumerate(nodes):
        node["_idx"] = i

    candidate_edges = _build_candidate_edges(nodes, settings)
    by_pair = {(int(e["from_idx"]), int(e["to_idx"])): e for e in candidate_edges}
    groups = [[i] for i in range(len(nodes))]

    def _pass1_ok(rec: dict[str, Any]) -> bool:
        if float(rec["score"]) < float(settings.day_link_pass1_min_score):
            return False
        reid = rec.get("reid")
        return reid is not None and float(reid) >= float(settings.day_link_pass1_min_reid)

    def _pass2_ok(rec: dict[str, Any]) -> bool:
        return float(rec["score"]) >= float(settings.day_link_pass2_min_score)

    pass1_used: set[tuple[int, int]] = set()
    if settings.day_link_pass1_min_reid > 0 or settings.day_link_pass1_min_score > 0:
        groups, pass1_used = _merge_groups_by_endpoints(
            groups,
            by_pair,
            nodes,
            min_score=float(settings.day_link_pass1_min_score),
            accept=_pass1_ok,
            solver="hungarian",
        )

    pass2_used: set[tuple[int, int]] = set()
    if settings.day_link_pass2_min_score > 0:
        groups, pass2_used = _merge_groups_by_endpoints(
            groups,
            by_pair,
            nodes,
            min_score=float(settings.day_link_pass2_min_score),
            accept=_pass2_ok,
            solver="hungarian",
        )

    pass4_used: set[tuple[int, int]] = set()
    if settings.day_link_pass4_max_overlap_sec > 0:
        groups, pass4_used = _pass4_handover(
            groups,
            by_pair,
            nodes,
            max_overlap_sec=float(settings.day_link_pass4_max_overlap_sec),
            min_score=float(settings.day_link_pass4_min_score),
            min_reid=float(settings.day_link_pass4_min_reid),
            solver="hungarian",
        )

    node_map = {node["uid"]: node for node in nodes}
    idx_to_person: dict[int, int] = {}
    sorted_groups = sorted(
        groups,
        key=lambda g: min(float(nodes[i]["t0"]) for i in g) if g else 0.0,
    )

    persons: list[dict[str, Any]] = []
    person_track_mapping: dict[str, int] = {}
    accepted_edges: list[dict[str, Any]] = []
    used_pairs: dict[tuple[int, int], int] = {}
    for pair in pass4_used:
        used_pairs[pair] = 4
    for pair in pass2_used:
        used_pairs.setdefault(pair, 2)
    for pair in pass1_used:
        used_pairs.setdefault(pair, 1)

    for gid, group in enumerate(sorted_groups, start=1):
        members = sorted((nodes[i] for i in group), key=lambda n: float(n["t0"]))
        for m in members:
            person_track_mapping[m["uid"]] = gid
            idx_to_person[int(m["_idx"])] = gid

        visited_cameras: list[str] = []
        for m in members:
            if not visited_cameras or visited_cameras[-1] != m["camera"]:
                visited_cameras.append(m["camera"])

        all_crops: list[dict[str, str]] = []
        best_crop = None
        for m in members:
            for cfile in m.get("crops") or []:
                ref = _crop_ref(m["session_key"], str(cfile))
                if ref and ref not in all_crops:
                    all_crops.append(ref)
            if best_crop is None and m.get("best_crop"):
                best_crop = _crop_ref(m["session_key"], str(m["best_crop"]))
        if best_crop is None and all_crops:
            best_crop = all_crops[0]

        p_t0 = min(float(m["t0"]) for m in members)
        p_t1 = max(float(m["t1"]) for m in members)
        n_transitions = sum(
            1 for i in range(len(members) - 1)
            if members[i]["camera_index"] != members[i + 1]["camera_index"]
        )

        persons.append({
            "person_id": gid,
            "label": f"Person #{gid}",
            "t0": round(p_t0, 2),
            "t1": round(p_t1, 2),
            "duration_sec": round(p_t1 - p_t0, 2),
            "n_tracks": len(members),
            "n_cameras": len(set(m["camera"] for m in members)),
            "n_transitions": n_transitions,
            "cameras": visited_cameras,
            "best_crop": best_crop,
            "crops": all_crops[:10],
            "tracks": [
                {
                    "uid": m["uid"],
                    "session_key": m["session_key"],
                    "camera": m["camera"],
                    "camera_index": m["camera_index"],
                    "track_id": m["track_id"],
                    "t0": round(float(m["t0"]), 3),
                    "t1": round(float(m["t1"]), 3),
                    "p0": (
                        [round(m["map_p0"][0], 2), round(m["map_p0"][1], 2)]
                        if m.get("map_p0")
                        else ([round(m["p0"][0], 2), round(m["p0"][1], 2)] if m.get("p0") else None)
                    ),
                    "p1": (
                        [round(m["map_p1"][0], 2), round(m["map_p1"][1], 2)]
                        if m.get("map_p1")
                        else ([round(m["p1"][0], 2), round(m["p1"][1], 2)] if m.get("p1") else None)
                    ),
                    "n_frames": m["n_frames"],
                    "has_reid": m["has_reid"],
                    "crops": list(m.get("crops") or []),
                }
                for m in members
            ],
        })

    pass_prefix = {
        1: "Pass 1 (Direct Match)",
        2: "Pass 2 (Chain Stitch)",
        4: "Pass 4 (Multi-Camera Handover)",
    }
    for pair, pass_n in used_pairs.items():
        rec = by_pair.get(pair)
        if rec is None:
            continue
        if idx_to_person.get(pair[0]) != idx_to_person.get(pair[1]):
            continue
        accepted_edges.append(_public_edge(rec, pass_n=pass_n, prefix=pass_prefix.get(pass_n, "Link")))

    accepted_edges.sort(key=lambda e: node_map.get(e["from"], {}).get("t1", 0.0))
    pass1_count = sum(1 for e in accepted_edges if e.get("pass") == 1)
    pass2_count = sum(1 for e in accepted_edges if e.get("pass") == 2)
    pass4_count = sum(1 for e in accepted_edges if e.get("pass") == 4)

    accepted_set = set(used_pairs.keys())
    leftover = [
        _public_edge(e, pass_n=-1, prefix="Candidate")
        for e in sorted(candidate_edges, key=lambda x: -float(x["score"]))
        if (int(e["from_idx"]), int(e["to_idx"])) not in accepted_set
    ][:_CANDIDATE_TOP_K]
    for e in leftover:
        e.pop("pass", None)
        e["reason"] = e.get("reason", "").split(": ", 1)[-1]

    stats = {
        "n_tracks_total": len(nodes),
        "n_persons": len(persons),
        "n_multi_cam_persons": sum(1 for p in persons if p["n_cameras"] > 1),
        "n_solo_persons": sum(1 for p in persons if p["n_cameras"] == 1 and p["n_tracks"] == 1),
        "n_merges_total": len(accepted_edges),
        "pass1_merges": pass1_count,
        "pass2_merges": pass2_count,
        "pass4_merges": pass4_count,
    }

    return {
        "n_persons": len(persons),
        "persons": persons,
        "edges": accepted_edges,
        "candidate_edges": leftover,
        "person_track_mapping": person_track_mapping,
        "stats": stats,
    }


def run_day_link(settings: Settings, target_day: str | None = None) -> None:
    """Точка входа для глобальной стадии day_link."""
    if not settings.day_link_enabled:
        logger.info("STAGE day_link: отключена в настройках")
        return

    day_clean = target_day or parse_day_input(str(settings.input_path))
    if not day_clean:
        from app.session.discover import resolve_sessions_for_input
        mode, sessions, _ = resolve_sessions_for_input(str(settings.input_path))
        _ = mode
        if sessions and sessions[0].day:
            day_clean = sessions[0].day.replace("-", "")

    if not day_clean:
        logger.warning(
            "STAGE day_link: день не определён из '%s', пропуск",
            settings.input_path,
        )
        return

    logger.info("=== STAGE day_link: глобальная склейка дня %s ===", day_clean)

    results_root = str(settings.json_output_dir or "data/results")
    sessions_for_day: list[str] = []
    if os.path.isdir(results_root):
        for name in sorted(os.listdir(results_root)):
            if name.startswith("_") or name.startswith("day_"):
                continue
            sess_path = os.path.join(results_root, name)
            info_fp = os.path.join(sess_path, "info.json")
            if os.path.isfile(info_fp):
                try:
                    info_data = load_tracking_json(info_fp)
                    d = str(info_data.get("day") or "").replace("-", "")
                    if d == day_clean:
                        sessions_for_day.append(name)
                except Exception:
                    pass

    if not sessions_for_day:
        logger.warning("STAGE day_link: нет обработанных сессий для дня %s в %s", day_clean, results_root)
        return

    logger.info("STAGE day_link: найдено сессий за день %s: %s", day_clean, ", ".join(sessions_for_day))

    reid_ctx = make_group_reid_context(settings)
    if not reid_ctx.available:
        logger.warning("STAGE day_link: ReID недоступен, пары без эмбеддингов не склеятся")

    all_nodes: list[dict[str, Any]] = []
    cameras_list: list[str] = []
    for sk in sessions_for_day:
        s_root = os.path.join(results_root, sk)
        reid_by_track = embed_session_tracks(s_root, settings, reid_ctx, session_key=sk)
        nodes = _extract_track_data(sk, s_root, reid_by_track)
        if nodes:
            cam_name = nodes[0]["camera"]
            if cam_name not in cameras_list:
                cameras_list.append(cam_name)
            all_nodes.extend(nodes)
            logger.info("  - Session %s (%s): %s треков", sk, cam_name, len(nodes))

    if not all_nodes:
        logger.warning("STAGE day_link: нет треков для объединения в день %s", day_clean)
        return

    logger.info(
        "STAGE day_link: всего %s треков по %s камерам (%s)",
        len(all_nodes),
        len(cameras_list),
        ", ".join(cameras_list),
    )

    result = link_day_tracks(all_nodes, settings)

    out_dir = day_results_dir(settings, day_clean)
    os.makedirs(out_dir, exist_ok=True)
    out_json = day_links_json_path(settings, day_clean)

    payload = {
        "stage": "day_link",
        "day": f"{day_clean[:4]}-{day_clean[4:6]}-{day_clean[6:8]}" if len(day_clean) == 8 else day_clean,
        "day_clean": day_clean,
        "cameras": cameras_list,
        "sessions": sessions_for_day,
        "solver": "hungarian",
        "n_persons": result["n_persons"],
        "persons": result["persons"],
        "edges": result["edges"],
        "candidate_edges": result["candidate_edges"],
        "person_track_mapping": result["person_track_mapping"],
        "stats": result["stats"],
    }

    attach_artifact_meta(payload, stage="day_link", path=out_json)
    save_debug_json(out_json, payload)

    logger.info(
        "STAGE day_link: готово! Персон=%s (мультикамерных=%s), склеек=%s (Pass1=%s, Pass2=%s, Pass4=%s) → %s",
        result["stats"]["n_persons"],
        result["stats"]["n_multi_cam_persons"],
        result["stats"]["n_merges_total"],
        result["stats"]["pass1_merges"],
        result["stats"]["pass2_merges"],
        result["stats"]["pass4_merges"],
        out_json,
    )
