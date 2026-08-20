"""Stage day_link: глобальная межкамерная склейка дня (Pass 0: Alone Geo, Pass 1: ReID >= 0.96)."""

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
from app.global_id.isolation import DayNeighborhoodIndex
from app.global_id.spatial import METER_PX
from app.io.json_util import load_tracking_json, save_debug_json
from app.session.discover import parse_day_input
from app.tracklet.link_mcf import _unique_groups, link_hungarian_chains
from app.util.intervals import intervals_overlap, pair_embed_score

logger = logging.getLogger(__name__)

_CANDIDATE_TOP_K = 50


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
    feet_trajs: dict[int, list[dict[str, Any]]] | None = None,
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

    if feet_trajs is None:
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


def _same_cam_overlap(
    ids: set[int],
    nodes: list[dict[str, Any]],
) -> bool:
    """Проверяет, есть ли перекрытие во времени между треками одной и той же камеры."""
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
    """Разбивает группы, если в них случайно оказались перекрывающиеся треки одной камеры."""
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


def _public_edge(
    edge: dict[str, Any],
    *,
    pass_n: int,
    prefix: str,
) -> dict[str, Any]:
    skip = {"from_idx", "to_idx"}
    out = {k: v for k, v in edge.items() if k not in skip}
    out["pass"] = pass_n
    reason = str(edge.get("reason") or "").strip()
    if reason.startswith("Pass "):
        out["reason"] = reason
    else:
        out["reason"] = f"{prefix}: {reason}".strip()
    return out


def _crop_ref(session_key: str, file: str | None) -> dict[str, str] | None:
    if not file:
        return None
    return {"session_key": session_key, "file": file}


def _pass0_alone_geo_stitch(
    groups: list[list[int]],
    nodes: list[dict[str, Any]],
    spatial_index: DayNeighborhoodIndex,
    settings: Settings,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Pass 0: Склейка одиноких треков/групп с разных камер по 2D-карте и изоляции."""
    if not settings.day_link_pass0_enabled or len(groups) < 2:
        return groups, []

    radius_m = float(settings.day_link_pass0_radius_m)
    min_overlap_sec = float(settings.day_link_pass0_min_overlap_sec)
    max_gap_sec = float(settings.day_link_pass0_max_gap_sec)
    max_dist_m = float(settings.day_link_pass0_max_dist_m)
    max_speed_mps = float(settings.day_link_pass0_max_speed_mps)
    min_reid = float(settings.day_link_pass0_min_reid)

    best_pairs: dict[tuple[int, int], tuple[float, int, int, dict[str, Any]]] = {}

    for ga, group_a in enumerate(groups):
        if not group_a:
            continue
        for gb, group_b in enumerate(groups):
            if ga == gb or not group_b:
                continue

            # Проверяем, нет ли пересечения по времени на одной камере
            if _same_cam_overlap(set(group_a) | set(group_b), nodes):
                continue

            # Проверяем пары узлов между группами
            uids_a = {str(nodes[i]["uid"]) for i in group_a}
            uids_b = {str(nodes[i]["uid"]) for i in group_b}
            all_uids = uids_a | uids_b

            best_edge: dict[str, Any] | None = None
            best_score = -1.0
            best_from_idx = -1
            best_to_idx = -1

            for ia in group_a:
                node_a = nodes[ia]
                for ib in group_b:
                    node_b = nodes[ib]
                    if int(node_a["camera_index"]) == int(node_b["camera_index"]):
                        continue

                    # Проверка 1: Одновременное нахождение рядом на 2D-карте
                    ok_sim, avg_d, ov_sec = spatial_index.check_simultaneous_proximity_and_isolation(
                        node_a,
                        node_b,
                        radius_m=radius_m,
                        max_dist_m=max_dist_m,
                        min_overlap_sec=min_overlap_sec,
                        exclude_uids=all_uids,
                    )
                    if ok_sim:
                        reid = pair_embed_score(node_a.get("reid_embs"), node_b.get("reid_embs"))
                        if min_reid <= 0 or (reid is not None and float(reid) >= min_reid):
                            r_val = float(reid) if reid is not None else 1.0
                            geo_score = 0.5 * max(0.0, 1.0 - avg_d / max(max_dist_m, 1.0)) + 0.5 * r_val
                            if geo_score > best_score:
                                best_score = geo_score
                                best_from_idx = ia
                                best_to_idx = ib
                                best_edge = {
                                    "from": node_a["uid"],
                                    "to": node_b["uid"],
                                    "from_idx": ia,
                                    "to_idx": ib,
                                    "from_session": node_a["session_key"],
                                    "from_camera": node_a["camera"],
                                    "from_track": node_a["track_id"],
                                    "to_session": node_b["session_key"],
                                    "to_camera": node_b["camera"],
                                    "to_track": node_b["track_id"],
                                    "is_same_camera": False,
                                    "is_overlap": True,
                                    "score": round(float(geo_score), 4),
                                    "reid": round(float(r_val), 4),
                                    "dist_m": round(float(avg_d), 2),
                                    "gap_sec": round(-float(ov_sec), 2),
                                    "pass": 0,
                                    "reason": (
                                        f"Pass 0 (Alone Geo): совпадение {ov_sec:.1f}с "
                                        f"(d={avg_d:.2f}м, ReID={r_val:.2f}, чистота R={radius_m:.1f}м)"
                                    ),
                                }

                    # Проверка 2: Последовательный переход между камерами
                    ok_gap, dist_m, gap_sec, speed_mps = spatial_index.check_transition_gap_and_isolation(
                        node_a,
                        node_b,
                        radius_m=radius_m,
                        max_dist_m=max_dist_m,
                        max_gap_sec=max_gap_sec,
                        max_speed_mps=max_speed_mps,
                        exclude_uids=all_uids,
                    )
                    if ok_gap:
                        reid = pair_embed_score(node_a.get("reid_embs"), node_b.get("reid_embs"))
                        if min_reid <= 0 or (reid is not None and float(reid) >= min_reid):
                            r_val = float(reid) if reid is not None else 1.0
                            geo_score = 0.5 * max(0.0, 1.0 - dist_m / max(max_dist_m, 1.0)) + 0.5 * r_val
                            if geo_score > best_score:
                                best_score = geo_score
                                best_from_idx = ia
                                best_to_idx = ib
                                best_edge = {
                                    "from": node_a["uid"],
                                    "to": node_b["uid"],
                                    "from_idx": ia,
                                    "to_idx": ib,
                                    "from_session": node_a["session_key"],
                                    "from_camera": node_a["camera"],
                                    "from_track": node_a["track_id"],
                                    "to_session": node_b["session_key"],
                                    "to_camera": node_b["camera"],
                                    "to_track": node_b["track_id"],
                                    "is_same_camera": False,
                                    "is_overlap": False,
                                    "score": round(float(geo_score), 4),
                                    "reid": round(float(r_val), 4),
                                    "dist_m": round(float(dist_m), 2),
                                    "gap_sec": round(float(gap_sec), 2),
                                    "speed_mps": round(float(speed_mps), 2),
                                    "pass": 0,
                                    "reason": (
                                        f"Pass 0 (Alone Geo): переход Δt={gap_sec:.1f}с "
                                        f"(d={dist_m:.2f}м, v={speed_mps:.2f}м/с, ReID={r_val:.2f}, чистота R={radius_m:.1f}м)"
                                    ),
                                }

            if best_edge is not None and best_score > 0:
                best_pairs[(ga, gb)] = (best_score, best_from_idx, best_to_idx, best_edge)

    if not best_pairs:
        return groups, []

    group_ids = list(range(len(groups)))
    t0_g = {gi: _group_span(groups[gi], nodes)[0] for gi in group_ids if groups[gi]}
    fake_edges = [(ga, gb, val) for (ga, gb), (val, _, _, _) in best_pairs.items()]
    chains = link_hungarian_chains(group_ids, fake_edges, min_score=0.1, t0=t0_g)

    accepted_edges: list[dict[str, Any]] = []
    new_groups: list[list[int]] = []
    for chain in chains:
        merged: list[int] = []
        for i, gi in enumerate(chain):
            merged.extend(groups[gi])
            if i + 1 < len(chain):
                rec = best_pairs.get((gi, chain[i + 1]))
                if rec:
                    accepted_edges.append(rec[3])
        new_groups.append(sorted(set(merged), key=lambda tid: (float(nodes[tid]["t0"]), tid)))

    new_groups = _split_same_cam_overlap(new_groups, nodes)
    return _unique_groups(new_groups), accepted_edges


def _pass1_strict_reid_stitch(
    groups: list[list[int]],
    nodes: list[dict[str, Any]],
    settings: Settings,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Pass 1: Строгая склейка по ReID тела (ReID >= min_reid)."""
    if not settings.day_link_pass1_enabled or len(groups) < 2:
        return groups, []

    min_reid = float(settings.day_link_pass1_min_reid)
    max_gap_sec = float(settings.day_link_pass1_max_gap_sec)

    best_pairs: dict[tuple[int, int], tuple[float, int, int, dict[str, Any]]] = {}

    for ga, group_a in enumerate(groups):
        if not group_a:
            continue
        for gb, group_b in enumerate(groups):
            if ga == gb or not group_b:
                continue

            if _same_cam_overlap(set(group_a) | set(group_b), nodes):
                continue

            # Стык: выход group_a -> вход group_b
            exit_id = _exit_idx(group_a, nodes)
            entry_id = _entry_idx(group_b, nodes)
            node_a = nodes[exit_id]
            node_b = nodes[entry_id]

            t1_a = float(node_a["t1"])
            t0_b = float(node_b["t0"])
            gap_sec = t0_b - t1_a
            is_same_cam = int(node_a["camera_index"]) == int(node_b["camera_index"])

            # На одной камере overlap запрещен
            if is_same_cam and gap_sec < 0.0:
                continue
            if not is_same_cam and gap_sec < -60.0:
                continue
            if gap_sec > max_gap_sec:
                continue

            reid = pair_embed_score(node_a.get("reid_embs"), node_b.get("reid_embs"))
            if reid is None or float(reid) < min_reid:
                continue

            score = float(reid)
            edge_info = {
                "from": node_a["uid"],
                "to": node_b["uid"],
                "from_idx": exit_id,
                "to_idx": entry_id,
                "from_session": node_a["session_key"],
                "from_camera": node_a["camera"],
                "from_track": node_a["track_id"],
                "to_session": node_b["session_key"],
                "to_camera": node_b["camera"],
                "to_track": node_b["track_id"],
                "is_same_camera": is_same_cam,
                "is_overlap": gap_sec < 0.0,
                "score": round(score, 4),
                "reid": round(score, 4),
                "gap_sec": round(float(gap_sec), 2),
                "pass": 1,
                "reason": f"Pass 1 (ReID): ReID={score:.2f}, Δt={gap_sec:.1f}с",
            }
            best_pairs[(ga, gb)] = (score, exit_id, entry_id, edge_info)

    if not best_pairs:
        return groups, []

    group_ids = list(range(len(groups)))
    t0_g = {gi: _group_span(groups[gi], nodes)[0] for gi in group_ids if groups[gi]}
    fake_edges = [(ga, gb, val) for (ga, gb), (val, _, _, _) in best_pairs.items()]
    chains = link_hungarian_chains(group_ids, fake_edges, min_score=min_reid, t0=t0_g)

    accepted_edges: list[dict[str, Any]] = []
    new_groups: list[list[int]] = []
    for chain in chains:
        merged: list[int] = []
        for i, gi in enumerate(chain):
            merged.extend(groups[gi])
            if i + 1 < len(chain):
                rec = best_pairs.get((gi, chain[i + 1]))
                if rec:
                    accepted_edges.append(rec[3])
        new_groups.append(sorted(set(merged), key=lambda tid: (float(nodes[tid]["t0"]), tid)))

    new_groups = _split_same_cam_overlap(new_groups, nodes)
    return _unique_groups(new_groups), accepted_edges


def _build_debug_candidate_edges(
    nodes: list[dict[str, Any]],
    settings: Settings,
) -> list[dict[str, Any]]:
    """Строит список кандидатных связей между треками для отображения в UI/отладки."""
    n = len(nodes)
    if n < 2:
        return []
    order = sorted(range(n), key=lambda i: float(nodes[i]["t0"]))
    t0s = [float(nodes[i]["t0"]) for i in order]
    max_gap = float(settings.day_link_max_gap_sec)
    candidates: list[dict[str, Any]] = []

    for u in nodes:
        hi = float(u["t1"]) + max_gap
        right = bisect_right(t0s, hi)
        for pos in range(right):
            v = nodes[order[pos]]
            if v["_idx"] == u["_idx"]:
                continue
            is_same_cam = int(u["camera_index"]) == int(v["camera_index"])
            u_t0, u_t1 = float(u["t0"]), float(u["t1"])
            v_t0, v_t1 = float(v["t0"]), float(v["t1"])
            gap_sec = v_t0 - u_t1
            is_overlap = intervals_overlap(u_t0, u_t1, v_t0, v_t1)

            if is_same_cam and (is_overlap or gap_sec < 0.0 or gap_sec > max_gap):
                continue
            if not is_same_cam and not is_overlap and (gap_sec < 0.0 or gap_sec > max_gap):
                continue

            s_reid = pair_embed_score(u.get("reid_embs"), v.get("reid_embs"))
            if s_reid is None or s_reid < 0.50:
                continue

            candidates.append({
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
                "is_same_camera": is_same_cam,
                "is_overlap": is_overlap,
                "score": round(float(s_reid), 4),
                "reid": round(float(s_reid), 4),
                "gap_sec": round(gap_sec, 2),
                "reason": f"Candidate: ReID={s_reid:.2f}, Δt={gap_sec:.1f}с",
            })

    return candidates


def link_day_tracks(
    nodes: list[dict[str, Any]],
    settings: Settings,
    feet_by_session: dict[str, dict[int, list[dict[str, Any]]]] | None = None,
) -> dict[str, Any]:
    """Межкамерный солвер дня: Pass 0 (Alone Geo по 2D-карте) → Pass 1 (ReID >= 0.96)."""
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

    feet_data = feet_by_session or {}
    spatial_index = DayNeighborhoodIndex.from_nodes_and_feet(nodes, feet_data)

    initial_groups = [[i] for i in range(len(nodes))]

    # --- Pass 0: Alone Geo ---
    groups_p0, p0_edges = _pass0_alone_geo_stitch(
        initial_groups,
        nodes,
        spatial_index,
        settings,
    )
    pass0_count = len(p0_edges)
    logger.info("STAGE day_link Pass 0 (Alone Geo): склеено %d ребер", pass0_count)

    # --- Pass 1: Strict ReID ---
    groups_p1, p1_edges = _pass1_strict_reid_stitch(
        groups_p0,
        nodes,
        settings,
    )
    pass1_count = len(p1_edges)
    logger.info(
        "STAGE day_link Pass 1 (ReID >= %.2f): склеено %d ребер",
        settings.day_link_pass1_min_reid,
        pass1_count,
    )

    all_accepted_edges: list[dict[str, Any]] = []
    for e in p0_edges:
        all_accepted_edges.append(_public_edge(e, pass_n=0, prefix="Pass 0 (Alone Geo)"))
    for e in p1_edges:
        all_accepted_edges.append(_public_edge(e, pass_n=1, prefix="Pass 1 (ReID)"))

    # Формирование финальных персон
    node_map = {node["uid"]: node for node in nodes}
    idx_to_person: dict[int, int] = {}
    sorted_groups = sorted(
        groups_p1,
        key=lambda g: min(float(nodes[i]["t0"]) for i in g) if g else 0.0,
    )

    persons: list[dict[str, Any]] = []
    person_track_mapping: dict[str, int] = {}

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

    all_accepted_edges.sort(key=lambda e: node_map.get(e["from"], {}).get("t1", 0.0))

    candidate_edges = _build_debug_candidate_edges(nodes, settings)
    accepted_pairs = {(e["from"], e["to"]) for e in all_accepted_edges}
    leftover = [
        _public_edge(e, pass_n=-1, prefix="Candidate")
        for e in sorted(candidate_edges, key=lambda x: -float(x["score"]))
        if (e["from"], e["to"]) not in accepted_pairs
    ][:_CANDIDATE_TOP_K]
    for e in leftover:
        e.pop("pass", None)
        e["reason"] = e.get("reason", "").split(": ", 1)[-1]

    stats = {
        "n_tracks_total": len(nodes),
        "n_persons": len(persons),
        "n_multi_cam_persons": sum(1 for p in persons if p["n_cameras"] > 1),
        "n_solo_persons": sum(1 for p in persons if p["n_cameras"] == 1 and p["n_tracks"] == 1),
        "n_merges_total": len(all_accepted_edges),
        "pass0_merges": pass0_count,
        "pass1_merges": pass1_count,
    }

    logger.info(
        "STAGE day_link ИТОГО: персон=%d (мультикамерных=%d), склеек=%d (Pass 0=%d, Pass 1=%d)",
        stats["n_persons"],
        stats["n_multi_cam_persons"],
        stats["n_merges_total"],
        stats["pass0_merges"],
        stats["pass1_merges"],
    )

    return {
        "n_persons": len(persons),
        "persons": persons,
        "edges": all_accepted_edges,
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
    feet_by_session: dict[str, dict[int, list[dict[str, Any]]]] = {}

    for sk in sessions_for_day:
        s_root = os.path.join(results_root, sk)
        feet_fp = os.path.join(s_root, "feet.json")
        feet_trajs = _load_feet_map_trajectories(feet_fp)
        feet_by_session[sk] = feet_trajs

        # Пересчет ReID по лучшим кадрам группы/трека
        reid_by_track = embed_session_tracks(s_root, settings, reid_ctx, session_key=sk)
        nodes = _extract_track_data(sk, s_root, reid_by_track, feet_trajs=feet_trajs)
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

    result = link_day_tracks(all_nodes, settings, feet_by_session=feet_by_session)

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
        "STAGE day_link: готово! Персон=%s (мультикамерных=%s), склеек=%s (Pass 0=%s, Pass 1=%s) → %s",
        result["stats"]["n_persons"],
        result["stats"]["n_multi_cam_persons"],
        result["stats"]["n_merges_total"],
        result["stats"]["pass0_merges"],
        result["stats"]["pass1_merges"],
        out_json,
    )
