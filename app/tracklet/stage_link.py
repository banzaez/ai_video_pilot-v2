"""Stage 2c: глобальная склейка треклетов → tracklet_links.json."""

from __future__ import annotations

import logging
import os
import statistics

from app.artifact_meta import attach_artifact_meta
from app.config import (
    Settings,
    tracklet_frames_json_path,
    tracklet_links_json_path,
    tracklet_reid_json_path,
    tracklets_json_path,
)
from app.io.json_util import load_tracking_json, save_debug_json
from app.reid import load_cache
from app.tracklet.embeddings import build_tracklet_crop_embeddings
from app.tracklet.link_mcf import link_tracklets
from app.tracklet.map_coords import enrich_tracklets_map_coords

logger = logging.getLogger(__name__)


def run_tracklet_link(settings: Settings) -> None:
    tl_path = tracklets_json_path(settings)
    reid_path = tracklet_reid_json_path(settings)
    if not os.path.isfile(tl_path):
        raise ValueError(f"Нет tracklets JSON: {tl_path}. Сначала --stage tracklets")
    if not os.path.isfile(reid_path):
        raise ValueError(f"Нет tracklet_reid JSON: {reid_path}. Сначала --stage tracklet_reid")

    tl_data = load_tracking_json(tl_path)
    reid_data = load_tracking_json(reid_path)
    tracklets = list(tl_data.get("tracklets") or [])
    if not tracklets:
        raise ValueError("tracklets.json пуст")

    # Обогащаем tracklets точками на карте этажа
    enrich_tracklets_map_coords(
        tracklets,
        settings=settings,
        torso_height_m=settings.feet_torso_height_m,
        person_height_m=settings.feet_person_height_m,
        kpt_min=settings.pose_kpt_min,
    )

    # Загружаем покадровые позиции треклетов
    frames_pos: dict[int, dict[int, tuple[float, float, str]]] = {}
    tf_path = tracklet_frames_json_path(settings)
    if os.path.isfile(tf_path):
        tf_data = load_tracking_json(tf_path)
        for fr in tf_data.get("frames") or []:
            fi = int(fr.get("frame_index", 0))
            for det in fr.get("detections") or []:
                tid = int(det.get("tracklet_id") or det.get("track_id") or 0)
                if tid <= 0:
                    continue
                bbox = det.get("bbox")
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    px = 0.5 * (float(bbox[0]) + float(bbox[2]))
                    py = float(bbox[3])
                    frames_pos.setdefault(fi, {})[tid] = (px, py, "image")

    npz_path = str(reid_data.get("npz_path") or "")
    if not npz_path or not os.path.isfile(npz_path):
        from app.config import tracklet_reid_npz_path

        npz_path = tracklet_reid_npz_path(settings)
    cache = load_cache(npz_path)

    embeddings = build_tracklet_crop_embeddings(reid_data.get("tracklets") or [], cache)

    if not embeddings:
        raise ValueError("Нет ReID-эмбеддингов для tracklet_link")

    logger.info("STAGE 2c: link %s треклетов", len(tracklets))
    result = link_tracklets(
        tracklets,
        embeddings,
        max_gap_sec=settings.tracklet_link_max_gap_sec,
        max_overlap_sec=settings.tracklet_link_max_overlap_sec,
        min_reid_score=settings.tracklet_link_min_reid_score,
        w_reid=settings.tracklet_link_w_reid,
        w_gap=settings.tracklet_link_w_gap,
        pass0_min_reid=settings.tracklet_link_pass0_min_reid,
        pass0_min_score=settings.tracklet_link_pass0_min_score,
        pass_alone_enabled=settings.tracklet_link_pass_alone_enabled,
        pass_alone_radius_m=settings.tracklet_link_pass_alone_radius_m,
        pass_alone_max_gap_sec=settings.tracklet_link_pass_alone_max_gap_sec,
        pass_alone_max_dist_m=settings.tracklet_link_pass_alone_max_dist_m,
        pass_alone_max_speed_mps=settings.tracklet_link_pass_alone_max_speed_mps,
        pass_alone_min_reid=settings.tracklet_link_pass_alone_min_reid,
        frames_pos=frames_pos if frames_pos else None,
    )

    groups = result["groups"]
    edges = result["edges"]
    mapping = result["tracklet_to_global"]
    n_multi = sum(1 for g in groups if len(g) > 1)
    chain_lens = [len(g) for g in groups if len(g) > 1]
    median_chain = statistics.median(chain_lens) if chain_lens else 1

    missed_high = [
        e
        for e in edges
        if float(e.get("reid") or 0) >= 0.95
        and mapping.get(str(e["from"])) != mapping.get(str(e["to"]))
    ]

    out_path = tracklet_links_json_path(settings)
    payload = {
        "stage": "tracklet_link",
        "solver": result["solver"],
        "pass0_min_reid": settings.tracklet_link_pass0_min_reid,
        "pass0_min_score": settings.tracklet_link_pass0_min_score,
        "pass_alone_enabled": settings.tracklet_link_pass_alone_enabled,
        "pass_alone_radius_m": settings.tracklet_link_pass_alone_radius_m,
        "n_groups": len(groups),
        "groups": groups,
        "edges": edges,
        "tracklet_to_global": mapping,
        "pass0_merged": int(result.get("pass0_merged") or 0),
        "pass_alone_merged": int(result.get("pass_alone_merged") or 0),
    }
    attach_artifact_meta(payload, stage="tracklet_link", path=out_path)
    save_debug_json(out_path, payload)
    logger.info(
        "STAGE 2c: групп=%s (склеено %s, median_chain=%.1f), рёбер=%s, reid>=0.95 не взяты=%s, pass0=%s, pass_alone_geo=%s",
        payload["n_groups"],
        n_multi,
        float(median_chain),
        len(edges),
        len(missed_high),
        int(result.get("pass0_merged") or 0),
        int(result.get("pass_alone_merged") or 0),
    )
