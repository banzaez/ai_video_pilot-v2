"""Stage 2c: глобальная склейка треклетов → tracklet_links.json."""

from __future__ import annotations

import logging
import os
import statistics

from app.artifact_meta import attach_artifact_meta
from app.config import Settings, tracklet_links_json_path, tracklet_reid_json_path, tracklets_json_path
from app.io.json_util import load_tracking_json, save_debug_json
from app.reid import load_cache
from app.tracklet.embeddings import build_tracklet_crop_embeddings
from app.tracklet.endpoint_pose import enrich_tracklet_endpoint_pose
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

    npz_path = str(reid_data.get("npz_path") or "")
    if not npz_path or not os.path.isfile(npz_path):
        from app.config import tracklet_reid_npz_path

        npz_path = tracklet_reid_npz_path(settings)
    cache = load_cache(npz_path)

    embeddings = build_tracklet_crop_embeddings(reid_data.get("tracklets") or [], cache)

    if not embeddings:
        raise ValueError("Нет ReID-эмбеддингов для tracklet_link")

    n_pose = enrich_tracklet_endpoint_pose(tracklets, settings)
    if n_pose:
        logger.info("STAGE 2c: pose на концах %s точек", n_pose)

    n_map = enrich_tracklets_map_coords(
        tracklets,
        settings=settings,
        torso_height_m=settings.tracklet_link_torso_height_m,
        person_height_m=settings.feet_person_height_m,
        kpt_min=settings.pose_kpt_min,
    )
    if n_map:
        logger.info("STAGE 2c: map-координаты для %s/%s треклетов", n_map, len(tracklets))

    logger.info("STAGE 2c: link %s треклетов (solver=%s)", len(tracklets), settings.tracklet_link_solver)
    result = link_tracklets(
        tracklets,
        embeddings,
        max_gap_sec=settings.tracklet_link_max_gap_sec,
        min_reid_score=settings.tracklet_link_min_reid_score,
        pass1_min_score=settings.tracklet_link_pass1_min_score,
        max_spatial_px=settings.tracklet_link_max_spatial_px,
        max_spatial_m=settings.tracklet_link_max_spatial_m,
        window_sec=settings.tracklet_link_window_sec,
        window_overlap_sec=settings.tracklet_link_window_overlap_sec,
        solver=settings.tracklet_link_solver,
        motion_sigma_px=settings.tracklet_link_motion_sigma_px,
        motion_sigma_m=settings.tracklet_link_motion_sigma_m,
        size_log_scale=settings.tracklet_link_size_log_scale,
        w_reid=settings.tracklet_link_w_reid,
        w_motion=settings.tracklet_link_w_motion,
        w_size=settings.tracklet_link_w_size,
        w_gap=settings.tracklet_link_w_gap,
        pass0_min_reid=settings.tracklet_link_pass0_min_reid,
        pass0_min_score=settings.tracklet_link_pass0_min_score,
        pass2_min_score=settings.tracklet_link_pass2_min_score,
        pass4_max_overlap_sec=settings.tracklet_link_pass4_max_overlap_sec,
        pass4_min_reid=settings.tracklet_link_pass4_min_reid,
        pass4_min_score=settings.tracklet_link_pass4_min_score,
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
        "pass1_min_score": settings.tracklet_link_pass1_min_score,
        "pass2_min_score": settings.tracklet_link_pass2_min_score,
        "pass4_max_overlap_sec": settings.tracklet_link_pass4_max_overlap_sec,
        "pass4_min_reid": settings.tracklet_link_pass4_min_reid,
        "pass4_min_score": settings.tracklet_link_pass4_min_score,
        "n_groups": len(groups),
        "groups": groups,
        "edges": edges,
        "tracklet_to_global": mapping,
        "pass0_merged": int(result.get("pass0_merged") or 0),
        "pass2_merged": int(result.get("pass2_merged") or 0),
        "pass3_spliced": int(result.get("pass3_spliced") or 0),
        "pass4_merged": int(result.get("pass4_merged") or 0),
    }
    attach_artifact_meta(payload, stage="tracklet_link", path=out_path)
    save_debug_json(out_path, payload)
    logger.info(
        "STAGE 2c: групп=%s (склеено %s, median_chain=%.1f), рёбер=%s, reid>=0.95 не взяты=%s, pass0=%s, pass2=%s, pass3=%s, pass4=%s",
        payload["n_groups"],
        n_multi,
        float(median_chain),
        len(edges),
        len(missed_high),
        int(result.get("pass0_merged") or 0),
        int(result.get("pass2_merged") or 0),
        int(result.get("pass3_spliced") or 0),
        int(result.get("pass4_merged") or 0),
    )
