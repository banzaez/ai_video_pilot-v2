"""Stage day_link: глобальная межкамерная склейка дня (Pass 0–4)."""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.artifact_meta import attach_artifact_meta
from app.config import (
    Settings,
    camera_face_json_path,
    camera_face_npz_path,
    day_links_json_path,
    day_results_dir,
    feet_json_path,
    info_json_path,
    tracking_json_path,
    tracklet_reid_npz_path,
)
from app.entity_id import group as group_eid
from app.io.json_util import load_tracking_json, save_debug_json
from app.session.discover import Session, discover_sessions, parse_day_input
from app.util.union_find import UnionFind

logger = logging.getLogger(__name__)


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


def _max_face_similarity_weighted(
    embs_a: np.ndarray | None,
    weights_a: np.ndarray | None,
    embs_b: np.ndarray | None,
    weights_b: np.ndarray | None,
) -> tuple[float, float]:
    """Выбор пары по cos * w_a * w_b; возвращает (raw_cos, min(w_a, w_b))."""
    if embs_a is None or embs_b is None or len(embs_a) == 0 or len(embs_b) == 0:
        return 0.0, 0.0
    sim_raw = np.dot(embs_a, embs_b.T)
    sim_weighted = sim_raw.copy()
    if weights_a is not None and len(weights_a) == len(embs_a):
        sim_weighted = sim_weighted * weights_a[:, None]
    if weights_b is not None and len(weights_b) == len(embs_b):
        sim_weighted = sim_weighted * weights_b[None, :]
    flat_idx = int(np.argmax(sim_weighted))
    i, j = divmod(flat_idx, sim_weighted.shape[1])
    w_a = float(weights_a[i]) if weights_a is not None and len(weights_a) == len(embs_a) else 1.0
    w_b = float(weights_b[j]) if weights_b is not None and len(weights_b) == len(embs_b) else 1.0
    return float(sim_raw[i, j]), min(w_a, w_b)


def _get_face_embs_for_id(
    face_npz: dict[str, np.ndarray],
    track_id: int,
    model: str,
) -> np.ndarray | None:
    eid = group_eid(track_id)
    keys = [
        eid.npz_key(model),
        eid.npz_key(),
        f"group_{track_id}_{model}",
        f"track_{track_id}_{model}",
        f"group_{track_id}",
        f"track_{track_id}",
    ]
    for k in keys:
        arr = face_npz.get(k)
        if arr is not None and len(arr) > 0:
            return arr
    return None


def _get_face_entries(
    face_meta: dict[str, Any],
    track_id: int,
    model: str,
) -> list[dict[str, Any]]:
    key = group_eid(track_id).format()
    by_model = face_meta.get("faces_by_model") or {}
    bucket = by_model.get(model) or face_meta.get("faces") or {}
    entries = bucket.get(key) or bucket.get(str(track_id)) or []
    if entries:
        return entries
    return []


def _load_reid_by_track_id(session_root: str) -> dict[int, np.ndarray]:
    """Загружает ReID-эмбеддинги для каждого track_id (группы треклетов)."""
    reid_json_path = os.path.join(session_root, "tracklet_reid.json")
    reid_npz_path = os.path.join(session_root, "tracklet_reid.npz")
    links_json_path = os.path.join(session_root, "tracklet_links.json")

    if not (os.path.isfile(reid_json_path) and os.path.isfile(reid_npz_path)):
        return {}

    reid_doc = load_tracking_json(reid_json_path)
    tracklet_crops: dict[int, list[str]] = {}
    for t in reid_doc.get("tracklets", []):
        tl_id = int(t.get("tracklet_id", 0))
        crops = t.get("crop_paths", [])
        if tl_id > 0 and crops:
            tracklet_crops[tl_id] = crops

    track_to_tls: dict[int, list[int]] = {}
    if os.path.isfile(links_json_path):
        links_doc = load_tracking_json(links_json_path)
        groups = links_doc.get("groups", [])
        for gid, grp in enumerate(groups, start=1):
            track_to_tls[gid] = [int(x) for x in grp]
    else:
        for tl_id in tracklet_crops:
            track_to_tls[tl_id] = [tl_id]

    npz_data: dict[str, np.ndarray] = {}
    try:
        with np.load(reid_npz_path) as npz:
            for k in npz.files:
                npz_data[k] = npz[k]
    except Exception as e:
        logger.warning("Session %s: ошибка чтения %s: %s", session_root, reid_npz_path, e)
        return {}

    track_reids: dict[int, np.ndarray] = {}
    for tid, tls in track_to_tls.items():
        embs: list[np.ndarray] = []
        for tl in tls:
            for cpath in tracklet_crops.get(tl, []):
                arr = npz_data.get(cpath)
                if arr is not None:
                    if arr.ndim == 1:
                        embs.append(arr)
                    else:
                        for row in arr:
                            embs.append(row)
        if embs:
            mat = np.stack(embs, axis=0)
            norm = np.linalg.norm(mat, axis=-1, keepdims=True)
            mat = mat / np.maximum(norm, 1e-9)
            track_reids[tid] = mat
    return track_reids


def _extract_track_data(
    session_key: str,
    session_root: str,
    settings: Settings,
) -> list[dict[str, Any]]:
    """Извлекает и выравнивает все треки сессии по единому дневному таймлайну."""
    info_path = os.path.join(session_root, "info.json")
    tracking_path = os.path.join(session_root, "tracking.json")
    feet_path = os.path.join(session_root, "feet.json")
    face_json_path = os.path.join(session_root, "camera_face.json")
    face_npz_path = os.path.join(session_root, "camera_face.npz")

    if not os.path.isfile(info_path) or not os.path.isfile(tracking_path):
        return []

    info_doc = load_tracking_json(info_path)
    tracking_doc = load_tracking_json(tracking_path)
    camera_name = str(info_doc.get("camera") or f"Camera_{session_key.split('_')[0]}")
    camera_idx = int(info_doc.get("camera_index") or 0)
    fps = float(info_doc.get("fps") or tracking_doc.get("fps") or 25.0)
    started_at_str = str(info_doc.get("parsed", {}).get("started_at") or info_doc.get("started_at") or "")
    if not started_at_str and info_doc.get("parts"):
        started_at_str = str(info_doc["parts"][0].get("started_at") or "")
    session_start_sec = _parse_iso_to_day_sec(started_at_str)

    # Точки ног на 2D-карте
    feet_trajs = _load_feet_map_trajectories(feet_path)

    # ReID эмбеддинги по каждому track_id
    reid_by_track = _load_reid_by_track_id(session_root)

    # Лица InsightFace
    face_meta: dict[str, Any] = load_tracking_json(face_json_path) if os.path.isfile(face_json_path) else {}
    face_npz_dict: dict[str, np.ndarray] = {}
    if os.path.isfile(face_npz_path):
        try:
            with np.load(face_npz_path) as npz:
                for k in npz.files:
                    face_npz_dict[k] = npz[k]
        except Exception as e:
            logger.warning("Session %s: ошибка чтения %s: %s", session_key, face_npz_path, e)

    # Сбор спанов треков по кадрам
    track_frames: dict[int, list[int]] = {}
    for f in tracking_doc.get("frames", []):
        fi = int(f.get("frame_index", 0))
        for d in f.get("detections", []):
            tid = int(d.get("track_id") or d.get("tracklet_id") or 0)
            if tid <= 0:
                continue
            track_frames.setdefault(tid, []).append(fi)

    face_models = list(settings.day_link_face_models)
    track_nodes: list[dict[str, Any]] = []

    for tid, frames in sorted(track_frames.items()):
        f0, f1 = min(frames), max(frames)
        t0_sec = session_start_sec + (f0 / fps)
        t1_sec = session_start_sec + (f1 / fps)

        # Траектория ног на 2D-карте
        ft = feet_trajs.get(tid, [])
        p0 = ft[0]["map"] if ft else None
        p1 = ft[-1]["map"] if ft else None
        avg_speed_mps = 0.0
        v_out = (0.0, 0.0)
        if len(ft) >= 2:
            dt = max(0.1, (ft[-1]["frame_index"] - ft[0]["frame_index"]) / fps)
            dist_total_px = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            dist_total_m = dist_total_px / 160.0
            avg_speed_mps = dist_total_m / dt
            # Выходной вектор скорости по последним 3-5 точкам (в м/с)
            recent = ft[-min(len(ft), 5):]
            dt_r = max(0.04, (recent[-1]["frame_index"] - recent[0]["frame_index"]) / fps)
            v_out = (
                ((recent[-1]["map"][0] - recent[0]["map"][0]) / 160.0) / dt_r,
                ((recent[-1]["map"][1] - recent[0]["map"][1]) / 160.0) / dt_r,
            )

        # ReID векторы
        reid_arr = reid_by_track.get(tid)

        # Лица по моделям
        face_embs_by_model: dict[str, np.ndarray | None] = {}
        face_pose_weights_by_model: dict[str, np.ndarray | None] = {}
        face_crops_list: list[str] = []
        best_face_det_score = 0.0

        for model in face_models:
            fembs = _get_face_embs_for_id(face_npz_dict, tid, model)
            if fembs is not None:
                if fembs.ndim == 1:
                    fembs = fembs.reshape(1, -1)
                norm = np.linalg.norm(fembs, axis=1, keepdims=True)
                fembs = fembs / np.maximum(norm, 1e-9)
            face_embs_by_model[model] = fembs

            entries = _get_face_entries(face_meta, tid, model)
            if entries:
                pw = np.array(
                    [max(0.0, min(1.0, float(e.get("pose_face_score", 1.0)))) for e in entries],
                    dtype=np.float32,
                )
                face_pose_weights_by_model[model] = pw
                for e in entries:
                    cfile = e.get("crop_file")
                    if cfile and cfile not in face_crops_list:
                        face_crops_list.append(cfile)
                    dscore = float(e.get("det_score") or 0.0)
                    if dscore > best_face_det_score:
                        best_face_det_score = dscore
            else:
                face_pose_weights_by_model[model] = None

        has_any_face = any(
            arr is not None and len(arr) > 0 for arr in face_embs_by_model.values()
        )

        track_nodes.append({
            "uid": f"{session_key}#{tid}",
            "session_key": session_key,
            "camera": camera_name,
            "camera_index": camera_idx,
            "track_id": tid,
            "f0": f0,
            "f1": f1,
            "t0": round(t0_sec, 3),
            "t1": round(t1_sec, 3),
            "duration_sec": round(t1_sec - t0_sec, 2),
            "p0": [round(p0[0], 2), round(p0[1], 2)] if p0 else None,
            "p1": [round(p1[0], 2), round(p1[1], 2)] if p1 else None,
            "v_out": (round(v_out[0], 2), round(v_out[1], 2)),
            "avg_speed_mps": round(avg_speed_mps, 2),
            "n_frames": len(frames),
            "reid_embs": reid_arr,
            "has_reid": reid_arr is not None,
            "has_face": has_any_face,
            "face_embs_by_model": face_embs_by_model,
            "face_pose_weights_by_model": face_pose_weights_by_model,
            "face_crops": face_crops_list,
            "best_face_score": round(best_face_det_score, 4),
        })

    return track_nodes


def _score_pair(
    u: dict[str, Any],
    v: dict[str, Any],
    settings: Settings,
) -> dict[str, Any] | None:
    """Вычисляет многомодальный скор перехода u -> v."""
    is_same_camera = (u["camera_index"] == v["camera_index"])

    # Временной интервал: u должен завершиться до старта v (или небольшой оверлап для handover)
    gap_sec = v["t0"] - u["t1"]
    max_gap_sec = float(settings.day_link_max_gap_sec)
    max_overlap_sec = float(settings.day_link_max_overlap_sec)

    if gap_sec < -max_overlap_sec or gap_sec > max_gap_sec:
        return None

    is_overlap = (gap_sec < 0.0)

    # Пространственная метрика на 2D карте (160 px = 1 метр)
    pA1 = u["p1"]
    pB0 = v["p0"]
    dist_m = 0.0
    s_motion = 0.5
    speed_mps = 0.0
    motion_sigma = float(settings.day_link_motion_sigma_m)
    max_speed = float(settings.day_link_max_speed_mps)

    if pA1 and pB0:
        dist_px = float(math.hypot(pB0[0] - pA1[0], pB0[1] - pA1[1]))
        dist_m = dist_px / 160.0
        dt = max(0.5, abs(gap_sec) if is_overlap else gap_sec)
        speed_mps = dist_m / dt
        if not is_overlap and speed_mps > max_speed:
            # Превышение физически возможной скорости перемещения человека
            return None
        if is_overlap and dist_m > 3.0:
            # При одновременной видимости на разных камерах человек не может быть дальше 3м
            return None
        s_motion = float(math.exp(- (dist_m ** 2) / (2.0 * (motion_sigma ** 2))))

    # Сходство лиц (InsightFace)
    face_models = list(settings.day_link_face_models)
    face_scores: dict[str, float] = {}
    best_raw_face = 0.0
    pose_face_weight = 0.0

    for model in face_models:
        fembs_u = u["face_embs_by_model"].get(model)
        fembs_v = v["face_embs_by_model"].get(model)
        pw_u = u["face_pose_weights_by_model"].get(model)
        pw_v = v["face_pose_weights_by_model"].get(model)
        raw_cos, min_w = _max_face_similarity_weighted(fembs_u, pw_u, fembs_v, pw_v)
        if raw_cos > 0.0:
            face_scores[model] = round(raw_cos, 4)
            if raw_cos > best_raw_face:
                best_raw_face = raw_cos
                pose_face_weight = min_w

    s_face = max(face_scores.values()) if face_scores else 0.0
    has_face = bool(face_scores)
    s_face_eff = s_face * max(0.1, pose_face_weight)

    # Сходство ReID тела (SOLIDER)
    s_reid = 0.0
    has_reid = False
    if u["reid_embs"] is not None and v["reid_embs"] is not None:
        reid_sim = np.dot(u["reid_embs"], v["reid_embs"].T)
        s_reid = float(np.max(reid_sim))
        has_reid = True

    # Штраф за временной зазор
    s_gap = max(0.0, 1.0 - abs(gap_sec) / max(1.0, max_gap_sec))

    w_face = float(settings.day_link_w_face)
    w_reid = float(settings.day_link_w_reid)
    w_motion = float(settings.day_link_w_motion)
    w_gap = float(settings.day_link_w_gap)

    # Вычисление комбинированного скора
    combo = (
        w_face * (s_face_eff if has_face else 0.0)
        + w_reid * (s_reid if has_reid else 0.0)
        + w_motion * s_motion
        + w_gap * s_gap
    )

    # Если лицо четкое и высококачественное — повышаем уверенность
    min_face_score = float(settings.day_link_min_face_score)
    if has_face and s_face >= min_face_score and pose_face_weight >= 0.35:
        combo = max(combo, s_face_eff * 0.85 + s_motion * 0.15)

    min_reid_score = float(settings.day_link_min_reid_score)
    if not has_face and has_reid and s_reid >= min_reid_score:
        combo = max(combo, s_reid * 0.70 + s_motion * 0.30)

    face_parts = [f"{m}={sc:.2f}" for m, sc in sorted(face_scores.items())]
    face_str = f"Face[{', '.join(face_parts)}]" if face_parts else "Face=—"
    reid_str = f"ReID={s_reid:.2f}" if has_reid else "ReID=—"
    motion_str = f"Motion={s_motion:.2f} (Δd={dist_m:.1f}м, v={speed_mps:.1f}м/с, Δt={gap_sec:.1f}с)"

    reason = f"{face_str}, {reid_str}, {motion_str}"

    return {
        "from": u["uid"],
        "to": v["uid"],
        "from_session": u["session_key"],
        "from_camera": u["camera"],
        "from_track": u["track_id"],
        "to_session": v["session_key"],
        "to_camera": v["camera"],
        "to_track": v["track_id"],
        "is_same_camera": is_same_camera,
        "is_overlap": is_overlap,
        "score": round(combo, 4),
        "face": round(s_face, 4) if has_face else None,
        "face_scores": face_scores or None,
        "pose_face": round(pose_face_weight, 4) if pose_face_weight > 0 else None,
        "reid": round(s_reid, 4) if has_reid else None,
        "motion": round(s_motion, 4),
        "dist_m": round(dist_m, 2),
        "gap_sec": round(gap_sec, 2),
        "speed_mps": round(speed_mps, 2),
        "reason": reason,
    }


def link_day_tracks(
    nodes: list[dict[str, Any]],
    settings: Settings,
) -> dict[str, Any]:
    """5-проходный межкамерный солвер дня."""
    if not nodes:
        return {
            "n_persons": 0,
            "persons": [],
            "edges": [],
            "candidate_edges": [],
            "stats": {},
        }

    # 1. Попарный скоринг всех потенциальных кандидатов
    candidate_edges: list[dict[str, Any]] = []
    n = len(nodes)
    for i in range(n):
        u = nodes[i]
        for j in range(n):
            if i == j:
                continue
            v = nodes[j]
            edge = _score_pair(u, v, settings)
            if edge is not None:
                candidate_edges.append(edge)

    # 2. Инициализация системы связей (Union-Find)
    uids = [node["uid"] for node in nodes]
    uf = UnionFind(uids)
    used_from: set[str] = set()
    used_to: set[str] = set()
    accepted_edges: list[dict[str, Any]] = []

    # -------------------------------------------------------------
    # Pass 0: Сверхнадежные совпадения (Direct Face/ReID Match)
    # -------------------------------------------------------------
    pass0_min_face = float(settings.day_link_pass0_min_face)
    pass0_min_reid = float(settings.day_link_pass0_min_reid)
    pass0_min_score = float(settings.day_link_pass0_min_score)

    pass0_candidates = [
        e for e in candidate_edges
        if not e["is_overlap"]
        and e["score"] >= pass0_min_score
        and (
            (e.get("face") and e["face"] >= pass0_min_face)
            or (e.get("reid") and e["reid"] >= pass0_min_reid)
        )
    ]
    pass0_candidates.sort(key=lambda e: -e["score"])

    pass0_count = 0
    for e in pass0_candidates:
        u, v = e["from"], e["to"]
        if u in used_from or v in used_to or uf.find(u) == uf.find(v):
            continue
        used_from.add(u)
        used_to.add(v)
        uf.union(u, v)
        e_copy = dict(e)
        e_copy["pass"] = 0
        e_copy["reason"] = f"Pass 0 (Direct Match): {e['reason']}"
        accepted_edges.append(e_copy)
        pass0_count += 1

    # -------------------------------------------------------------
    # Pass 1: Оконное венгерское сопоставление (Windowed Temporal Matching)
    # -------------------------------------------------------------
    pass1_min_score = float(settings.day_link_pass1_min_score)
    window_sec = float(settings.day_link_window_sec)
    window_overlap_sec = float(settings.day_link_window_overlap_sec)

    min_t = min(node["t0"] for node in nodes)
    max_t = max(node["t1"] for node in nodes)
    cur_t = min_t
    step_sec = max(10.0, window_sec - window_overlap_sec)

    pass1_count = 0
    while cur_t < max_t:
        w_start = cur_t
        w_end = cur_t + window_sec
        # Собираем доступные узлы и ребра в текущем окне
        w_nodes = [node for node in nodes if w_start <= node["t0"] <= w_end or w_start <= node["t1"] <= w_end]
        w_uids = [node["uid"] for node in w_nodes]

        w_edges = [
            e for e in candidate_edges
            if e["from"] in w_uids and e["to"] in w_uids
            and e["from"] not in used_from and e["to"] not in used_to
            and not e["is_overlap"]
            and e["score"] >= pass1_min_score
            and uf.find(e["from"]) != uf.find(e["to"])
        ]

        if w_edges:
            # Формируем матрицу стоимостей для Linear Sum Assignment
            u_srcs = sorted(set(e["from"] for e in w_edges))
            v_dsts = sorted(set(e["to"] for e in w_edges))
            src_idx = {uid: i for i, uid in enumerate(u_srcs)}
            dst_idx = {uid: j for j, uid in enumerate(v_dsts)}
            cost_matrix = np.full((len(u_srcs), len(v_dsts)), 10.0, dtype=np.float32)

            edge_map: dict[tuple[int, int], dict[str, Any]] = {}
            for e in w_edges:
                i, j = src_idx[e["from"]], dst_idx[e["to"]]
                cost = 1.0 - float(e["score"])
                if cost < cost_matrix[i, j]:
                    cost_matrix[i, j] = cost
                    edge_map[(i, j)] = e

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < (1.0 - pass1_min_score + 1e-4):
                    e = edge_map.get((r, c))
                    if not e:
                        continue
                    u, v = e["from"], e["to"]
                    if u in used_from or v in used_to or uf.find(u) == uf.find(v):
                        continue
                    used_from.add(u)
                    used_to.add(v)
                    uf.union(u, v)
                    e_copy = dict(e)
                    e_copy["pass"] = 1
                    e_copy["reason"] = f"Pass 1 (Window {int(window_sec)}s): {e['reason']}"
                    accepted_edges.append(e_copy)
                    pass1_count += 1

        cur_t += step_sec

    # -------------------------------------------------------------
    # Pass 2: Склейка цепочек (Chain-to-Chain Stitching)
    # -------------------------------------------------------------
    pass2_min_score = float(settings.day_link_pass2_min_score)
    pass2_candidates = [
        e for e in candidate_edges
        if e["from"] not in used_from and e["to"] not in used_to
        and not e["is_overlap"]
        and e["score"] >= pass2_min_score
        and uf.find(e["from"]) != uf.find(e["to"])
    ]
    pass2_candidates.sort(key=lambda e: -e["score"])

    pass2_count = 0
    for e in pass2_candidates:
        u, v = e["from"], e["to"]
        if u in used_from or v in used_to or uf.find(u) == uf.find(v):
            continue
        used_from.add(u)
        used_to.add(v)
        uf.union(u, v)
        e_copy = dict(e)
        e_copy["pass"] = 2
        e_copy["reason"] = f"Pass 2 (Chain Stitch): {e['reason']}"
        accepted_edges.append(e_copy)
        pass2_count += 1

    # -------------------------------------------------------------
    # Pass 4: Handover / Co-visibility Overlap (Мультикамерный оверлап)
    # -------------------------------------------------------------
    pass4_min_score = float(settings.day_link_pass4_min_score)
    pass4_candidates = [
        e for e in candidate_edges
        if e["is_overlap"]
        and not e["is_same_camera"]
        and e["score"] >= pass4_min_score
        and e["from"] not in used_from and e["to"] not in used_to
        and uf.find(e["from"]) != uf.find(e["to"])
    ]
    pass4_candidates.sort(key=lambda e: -e["score"])

    pass4_count = 0
    for e in pass4_candidates:
        u, v = e["from"], e["to"]
        if u in used_from or v in used_to or uf.find(u) == uf.find(v):
            continue
        used_from.add(u)
        used_to.add(v)
        uf.union(u, v)
        e_copy = dict(e)
        e_copy["pass"] = 4
        e_copy["reason"] = f"Pass 4 (Multi-Camera Handover): {e['reason']}"
        accepted_edges.append(e_copy)
        pass4_count += 1

    # -------------------------------------------------------------
    # Сборка финальных глобальных персон (Global Person Entities)
    # -------------------------------------------------------------
    node_map = {node["uid"]: node for node in nodes}
    person_clusters: dict[str, list[dict[str, Any]]] = {}

    for uid in uids:
        root = uf.find(uid)
        person_clusters.setdefault(root, []).append(node_map[uid])

    # Сортировка кластеров по времени первого появления
    sorted_roots = sorted(
        person_clusters.keys(),
        key=lambda r: min(node["t0"] for node in person_clusters[r]),
    )

    persons: list[dict[str, Any]] = []
    person_track_mapping: dict[str, int] = {}

    for gid, root in enumerate(sorted_roots, start=1):
        members = sorted(person_clusters[root], key=lambda n: n["t0"])
        for m in members:
            person_track_mapping[m["uid"]] = gid

        visited_cameras = []
        for m in members:
            if not visited_cameras or visited_cameras[-1] != m["camera"]:
                visited_cameras.append(m["camera"])

        all_face_crops = []
        best_face_crop = None
        best_face_score = 0.0

        for m in members:
            for cfile in m["face_crops"]:
                crop_path = f"/api/face_crop/{m['session_key']}/{cfile}"
                if crop_path not in all_face_crops:
                    all_face_crops.append(crop_path)
                if m["best_face_score"] > best_face_score:
                    best_face_score = m["best_face_score"]
                    best_face_crop = crop_path

        p_t0 = min(m["t0"] for m in members)
        p_t1 = max(m["t1"] for m in members)

        # Вычисляем количество межкамерных переходов внутри персоны
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
            "best_face_crop": best_face_crop,
            "face_crops": all_face_crops[:10],
            "tracks": [
                {
                    "uid": m["uid"],
                    "session_key": m["session_key"],
                    "camera": m["camera"],
                    "camera_index": m["camera_index"],
                    "track_id": m["track_id"],
                    "t0": m["t0"],
                    "t1": m["t1"],
                    "p0": m["p0"],
                    "p1": m["p1"],
                    "n_frames": m["n_frames"],
                    "has_face": m["has_face"],
                    "has_reid": m["has_reid"],
                    "best_face_score": m["best_face_score"],
                    "crops": [f"/api/face_crop/{m['session_key']}/{c}" for c in m["face_crops"]],
                }
                for m in members
            ],
        })

    # Сортируем ребра по времени перехода
    accepted_edges.sort(key=lambda e: node_map.get(e["from"], {}).get("t1", 0.0))

    stats = {
        "n_tracks_total": len(nodes),
        "n_persons": len(persons),
        "n_multi_cam_persons": sum(1 for p in persons if p["n_cameras"] > 1),
        "n_solo_persons": sum(1 for p in persons if p["n_cameras"] == 1 and p["n_tracks"] == 1),
        "n_merges_total": len(accepted_edges),
        "pass0_merges": pass0_count,
        "pass1_merges": pass1_count,
        "pass2_merges": pass2_count,
        "pass4_merges": pass4_count,
    }

    return {
        "n_persons": len(persons),
        "persons": persons,
        "edges": accepted_edges,
        "candidate_edges": candidate_edges,
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
        # Проверяем, может передана сессия или видео — извлекаем день из info.json
        from app.session.discover import resolve_sessions_for_input
        mode, sessions, _ = resolve_sessions_for_input(str(settings.input_path))
        if sessions:
            day_clean = sessions[0].day.replace("-", "")

    if not day_clean:
        raise ValueError(f"Не удалось определить день для day_link из '{settings.input_path}'. Укажите --day YYYYMMDD")

    logger.info("=== STAGE day_link: глобальная склейка дня %s ===", day_clean)

    # 1. Поиск всех сессий этого дня
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
        raise ValueError(f"Нет обработанных сессий для дня {day_clean} в {results_root}")

    logger.info("STAGE day_link: найдено сессий за день %s: %s", day_clean, ", ".join(sessions_for_day))

    # 2. Сбор треков по всем сессиям дня
    all_nodes: list[dict[str, Any]] = []
    cameras_list: list[str] = []
    for sk in sessions_for_day:
        s_root = os.path.join(results_root, sk)
        nodes = _extract_track_data(sk, s_root, settings)
        if nodes:
            cam_name = nodes[0]["camera"]
            if cam_name not in cameras_list:
                cameras_list.append(cam_name)
            all_nodes.extend(nodes)
            logger.info("  - Session %s (%s): %s треков", sk, cam_name, len(nodes))

    if not all_nodes:
        logger.warning("STAGE day_link: нет треков для объединения в день %s", day_clean)
        return

    logger.info("STAGE day_link: всего %s треков по %s камерам (%s)", len(all_nodes), len(cameras_list), ", ".join(cameras_list))

    # 3. Запуск межкамерного солвера дня
    result = link_day_tracks(all_nodes, settings)

    # 4. Сохранение артефакта day_links.json
    out_dir = day_results_dir(settings, day_clean)
    os.makedirs(out_dir, exist_ok=True)
    out_json = day_links_json_path(settings, day_clean)

    payload = {
        "stage": "day_link",
        "day": f"{day_clean[:4]}-{day_clean[4:6]}-{day_clean[6:8]}" if len(day_clean) == 8 else day_clean,
        "day_clean": day_clean,
        "cameras": cameras_list,
        "sessions": sessions_for_day,
        "solver": settings.day_link_solver,
        "face_models": list(settings.day_link_face_models),
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
        "STAGE day_link: готово! Персон=%s (мультикамерных=%s), склеек=%s (Pass0=%s, Pass1=%s, Pass2=%s, Pass4=%s) → %s",
        result["stats"]["n_persons"],
        result["stats"]["n_multi_cam_persons"],
        result["stats"]["n_merges_total"],
        result["stats"]["pass0_merges"],
        result["stats"]["pass1_merges"],
        result["stats"]["pass2_merges"],
        result["stats"]["pass4_merges"],
        out_json,
    )
