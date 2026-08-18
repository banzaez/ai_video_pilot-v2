"""Stage camera_link (Pass 10): верхнеуровневая склейка по InsightFace и координатам ног."""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import numpy as np

from app.artifact_meta import attach_artifact_meta
from app.config import (
    Settings,
    camera_face_json_path,
    camera_face_npz_path,
    camera_links_json_path,
    feet_json_path,
    info_json_path,
    tracking_json_path,
    tracklet_links_json_path,
    tracklet_reid_npz_path,
    tracks_json_path,
)
from app.face.insight_extractor import extract_faces_for_groups, face_models_for_settings
from app.io.json_util import load_tracking_json, save_debug_json

logger = logging.getLogger(__name__)


def _load_feet_trajectories(feet_path: str) -> dict[int, list[dict[str, Any]]]:
    """Загружает упорядоченные по времени точки ног для каждого track_id."""
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


def _resolve_video_path(settings: Settings, meta: dict | None) -> str | None:
    cand = str(settings.input_path)
    if os.path.isfile(cand):
        return cand
    name = os.path.basename(cand)
    folder = os.path.dirname(cand)
    for p in (
        meta.get("video_source") if meta else None,
        os.path.join(folder, name) if folder else None,
        os.path.join("data", "video", name),
    ):
        if p and os.path.isfile(str(p)):
            return str(p)
    return None


def _max_face_similarity(embs_a: np.ndarray | None, embs_b: np.ndarray | None) -> float:
    if embs_a is None or embs_b is None or len(embs_a) == 0 or len(embs_b) == 0:
        return 0.0
    return float(np.max(np.dot(embs_a, embs_b.T)))


def _face_embs_for_group(face_npz: dict[str, np.ndarray], gid: int, model: str, *, multi: bool) -> np.ndarray | None:
    if multi:
        return face_npz.get(f"group_{gid}_{model}")
    return face_npz.get(f"group_{gid}")


def run_camera_link(settings: Settings) -> None:
    """Точка входа стадии camera_link (Pass 10)."""
    if not settings.camera_link_enabled:
        logger.info("STAGE camera_link: отключена в настройках")
        return

    tracking_path = tracking_json_path(settings)
    if not os.path.isfile(tracking_path):
        logger.warning("STAGE camera_link: нет %s, пропуск", tracking_path)
        return

    tracking_doc = load_tracking_json(tracking_path)
    frames_data = tracking_doc.get("frames", [])
    fps = float(tracking_doc.get("fps") or 25.0)

    # 1. Определяем базовые группы из tracklet_links.json (или одиночные треки)
    links_path = tracklet_links_json_path(settings)
    track_to_group: dict[int, int] = {}
    base_groups: list[list[int]] = []
    if os.path.isfile(links_path):
        links_doc = load_tracking_json(links_path)
        base_groups = [list(g) for g in links_doc.get("groups", []) if g]
        raw_mapping = links_doc.get("tracklet_to_global", {})
        track_to_group = {int(k): int(v) for k, v in raw_mapping.items()}

    # Если групп нет — строим группы из уникальных track_id в tracking.json
    all_track_ids = set()
    track_spans: dict[int, tuple[float, float]] = {}
    for f in frames_data:
        fi = int(f.get("frame_index", 0))
        t_sec = fi / fps
        for d in f.get("detections", []):
            tid = int(d.get("track_id") or d.get("tracklet_id") or 0)
            if tid <= 0:
                continue
            all_track_ids.add(tid)
            if tid not in track_spans:
                track_spans[tid] = (t_sec, t_sec)
            else:
                track_spans[tid] = (min(track_spans[tid][0], t_sec), max(track_spans[tid][1], t_sec))

    if not base_groups:
        base_groups = [[tid] for tid in sorted(all_track_ids)]
        track_to_group = {tid: tid for tid in all_track_ids}

    # 2. Загружаем координаты ног
    feet_path = feet_json_path(settings)
    feet_trajs = _load_feet_trajectories(feet_path)

    # 3. Извлечение лиц через InsightFace
    video_path = _resolve_video_path(settings, tracking_doc)
    face_meta: dict[str, Any] = {}
    face_npz: dict[str, np.ndarray] = {}

    if video_path and os.path.isfile(video_path):
        face_meta, face_npz = extract_faces_for_groups(
            settings,
            frames_data=frames_data,
            track_to_group=track_to_group,
            video_source_path=video_path,
        )
        face_json_out = camera_face_json_path(settings)
        attach_artifact_meta(face_meta, stage="camera_face", path=face_json_out)
        save_debug_json(face_json_out, face_meta)
        face_npz_out = camera_face_npz_path(settings)
        np.savez_compressed(face_npz_out, **face_npz)
        logger.info("STAGE camera_link: извлечено лиц для %s групп", len(face_npz))

    # 4. Загружаем ReID эмбеддинги тела (если есть)
    reid_npz_path = tracklet_reid_npz_path(settings)
    reid_embeddings: dict[str, np.ndarray] = {}
    if os.path.isfile(reid_npz_path):
        try:
            with np.load(reid_npz_path) as npz:
                for k in npz.files:
                    reid_embeddings[k] = npz[k]
        except Exception as e:
            logger.warning("Не удалось загрузить %s: %s", reid_npz_path, e)

    # 5. Формируем метаданные групп для склейки Pass 10
    face_models = face_models_for_settings(settings)
    multi_face = len(face_models) > 1
    group_info: dict[int, dict[str, Any]] = {}
    for gid_idx, g in enumerate(base_groups, start=1):
        gid = gid_idx
        t0 = min((track_spans[tid][0] for tid in g if tid in track_spans), default=0.0)
        t1 = max((track_spans[tid][1] for tid in g if tid in track_spans), default=0.0)

        # Сбор точек ног для группы
        all_feet: list[dict[str, Any]] = []
        for tid in g:
            all_feet.extend(feet_trajs.get(tid, []))
        all_feet.sort(key=lambda pt: pt["frame_index"])

        p0 = all_feet[0]["map"] if all_feet else None
        p1 = all_feet[-1]["map"] if all_feet else None
        f0_idx = all_feet[0]["frame_index"] if all_feet else int(t0 * fps)
        f1_idx = all_feet[-1]["frame_index"] if all_feet else int(t1 * fps)

        face_embs_by_model = {
            model: _face_embs_for_group(face_npz, gid, model, multi=multi_face)
            for model in face_models
        }
        group_info[gid] = {
            "group_id": gid,
            "tracks": g,
            "t0": t0,
            "t1": t1,
            "f0": f0_idx,
            "f1": f1_idx,
            "p0": p0,
            "p1": p1,
            "face_embs_by_model": face_embs_by_model,
        }

    # 6. Попарный скоринг и поиск кандидатов (Pass 10)
    gids = sorted(group_info.keys())
    candidate_edges: list[dict[str, Any]] = []
    max_gap_sec = float(settings.camera_link_max_gap_sec)
    max_speed = float(settings.camera_link_max_speed_mps)
    motion_sigma = float(settings.camera_link_motion_sigma_m)
    min_face_score = float(settings.camera_link_min_face_score)
    w_face = float(settings.camera_link_w_face)
    w_reid = float(settings.camera_link_w_reid)
    w_motion = float(settings.camera_link_w_motion)

    for i in range(len(gids)):
        gA = group_info[gids[i]]
        for j in range(len(gids)):
            if i == j:
                continue
            gB = group_info[gids[j]]

            # Временной порядок: A завершился до старта B
            gap_sec = gB["t0"] - gA["t1"]
            if gap_sec < 0.0 or gap_sec > max_gap_sec:
                continue

            # Физическая позиция ног на карте
            pA1 = gA["p1"]
            pB0 = gB["p0"]
            dist_m = 0.0
            s_motion = 0.5
            v = 0.0

            if pA1 and pB0:
                dist_m = float(math.hypot(pB0[0] - pA1[0], pB0[1] - pA1[1]))
                dt = max(0.1, gap_sec)
                v = dist_m / dt
                if v > max_speed:
                    # Физически невозможная скорость перемещения
                    continue
                s_motion = float(math.exp(- (dist_m ** 2) / (2.0 * (motion_sigma ** 2))))

            # Сходство лиц (отдельно по каждой модели InsightFace)
            face_scores: dict[str, float] = {}
            for model in face_models:
                s_model = _max_face_similarity(
                    gA["face_embs_by_model"].get(model),
                    gB["face_embs_by_model"].get(model),
                )
                if s_model > 0:
                    face_scores[model] = round(s_model, 4)
            s_face = max(face_scores.values()) if face_scores else 0.0
            has_face = bool(face_scores)

            # Сходство ReID тела (по всем трекам группы)
            s_reid = 0.0
            has_reid = False
            reid_pairs = []
            for tidA in gA["tracks"]:
                for tidB in gB["tracks"]:
                    vecA = reid_embeddings.get(f"tracklet_{tidA}") or reid_embeddings.get(f"track_{tidA}")
                    vecB = reid_embeddings.get(f"tracklet_{tidB}") or reid_embeddings.get(f"track_{tidB}")
                    if vecA is not None and vecB is not None:
                        if vecA.ndim == 1:
                            vecA = vecA.reshape(1, -1)
                        if vecB.ndim == 1:
                            vecB = vecB.reshape(1, -1)
                        reid_pairs.append(float(np.max(np.dot(vecA, vecB.T))))
            if reid_pairs:
                has_reid = True
                s_reid = max(reid_pairs)

            # Вычисление комбинированного скора
            if has_face:
                combo = w_face * s_face + w_reid * s_reid + w_motion * s_motion
                # Высокая уверенность по лицу при валидной скорости ног
                if s_face >= min_face_score:
                    combo = max(combo, s_face)
            elif has_reid:
                combo = (w_reid / (w_reid + w_motion)) * s_reid + (w_motion / (w_reid + w_motion)) * s_motion
            else:
                combo = s_motion

            min_combo_score = float(settings.camera_link_min_combo_score)
            if combo < min_combo_score:
                continue

            face_parts = [f"{model}={score:.2f}" for model, score in sorted(face_scores.items())]
            face_reason = f"Face[{', '.join(face_parts)}]" if face_parts else "Face=—"
            reason_str = (
                f"Pass 10 (Camera Link / Face + Feet): "
                f"{face_reason}, ReID={s_reid:.2f}, Motion={s_motion:.2f} "
                f"(Δd={dist_m:.1f}м, v={v:.1f}м/с, Δt={gap_sec:.1f}с)"
            )

            candidate_edges.append({
                "from": gA["group_id"],
                "to": gB["group_id"],
                "score": round(combo, 4),
                "face": round(s_face, 4) if has_face else None,
                "face_scores": face_scores or None,
                "reid": round(s_reid, 4) if has_reid else None,
                "motion": round(s_motion, 4),
                "dist_m": round(dist_m, 2),
                "gap_sec": round(gap_sec, 2),
                "speed_mps": round(v, 2),
                "pass": 10,
                "reason": reason_str,
            })

    # 7. Глобальное 1-1 связывание (Greedy best-first)
    candidate_edges.sort(key=lambda e: -e["score"])
    best_edges: list[dict[str, Any]] = []
    used_from = set()
    used_to = set()

    for e in candidate_edges:
        u, v = e["from"], e["to"]
        if u in used_from or v in used_to:
            continue
        used_from.add(u)
        used_to.add(v)
        best_edges.append(e)

    # 8. Сборка итоговых объединенных групп (Union-Find)
    parent = {gid: gid for gid in gids}

    def _find(x: int) -> int:
        if parent[x] != x:
            parent[x] = _find(parent[x])
        return parent[x]

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for e in best_edges:
        _union(e["from"], e["to"])

    final_group_map: dict[int, list[int]] = {}
    for gid in gids:
        root = _find(gid)
        # Объединяем входящие треки
        final_group_map.setdefault(root, []).extend(group_info[gid]["tracks"])

    final_groups = [sorted(set(members)) for root, members in sorted(final_group_map.items())]

    # Маппинг track_id -> camera_global_id
    track_to_camera_global: dict[str, int] = {}
    for new_gid, members in enumerate(final_groups, start=1):
        for tid in members:
            track_to_camera_global[str(tid)] = new_gid

    out_links_path = camera_links_json_path(settings)
    payload = {
        "stage": "camera_link",
        "pass": 10,
        "solver": settings.camera_link_solver,
        "face_models": face_models,
        "n_groups": len(final_groups),
        "groups": final_groups,
        "edges": best_edges,
        "candidate_edges": candidate_edges,
        "track_to_global": track_to_camera_global,
    }
    attach_artifact_meta(payload, stage="camera_link", path=out_links_path)
    save_debug_json(out_links_path, payload)

    n_pass10_merges = sum(1 for e in best_edges if e.get("pass") == 10)
    logger.info(
        "STAGE camera_link (Pass 10): групп=%s (склеек Pass 10: %s), рёбер=%s",
        len(final_groups),
        n_pass10_merges,
        len(best_edges),
    )
