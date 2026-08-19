"""Глобальная склейка треклетов: Pass 0 (жадный матчинг по ReID + Gap)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from app.global_id.isolation import TrackNeighborhoodIndex, _tracklet_endpoint, calc_points_dist_m
from app.global_id.spatial import METER_PX
from app.util.intervals import intervals_overlap, intervals_overlap_sec, pair_embed_score


def _crop_matrix(emb: np.ndarray | None) -> np.ndarray | None:
    if emb is None:
        return None
    arr = np.asarray(emb, dtype=np.float32)
    if arr.ndim == 1:
        n = float(np.linalg.norm(arr))
        if n <= 1e-9:
            return None
        return (arr / n).reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    return arr / norms


def _gap_score(gap: float, max_gap_sec: float) -> float:
    if max_gap_sec <= 0:
        return 1.0
    return float(max(0.0, 1.0 - gap / max_gap_sec))


def _combo_score(
    reid: float,
    gap: float,
    *,
    w_reid: float = 0.85,
    w_gap: float = 0.15,
) -> float:
    parts = (
        (reid, w_reid),
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


def build_candidate_edges(
    tracklets: list[dict[str, Any]],
    embeddings: dict[int, np.ndarray],
    *,
    max_gap_sec: float,
    max_overlap_sec: float = 2.0,
    min_reid_score: float,
    w_reid: float = 0.85,
    w_gap: float = 0.15,
    **_kwargs: Any,
) -> list[tuple[int, int, float, float, float]]:
    """Рёбра A→B: (from, to, combo, reid, gap)."""
    edges: list[tuple[int, int, float, float, float]] = []
    sorted_tl = sorted(tracklets, key=lambda r: (float(r["t0"]), int(r["tracklet_id"])))
    mats = {int(t["tracklet_id"]): _crop_matrix(embeddings.get(int(t["tracklet_id"]))) for t in sorted_tl}

    for i, a in enumerate(sorted_tl):
        aid = int(a["tracklet_id"])
        mat_a = mats.get(aid)
        if mat_a is None:
            continue
        t0_a, t1_a = float(a["t0"]), float(a["t1"])
        for b in sorted_tl:
            bid = int(b["tracklet_id"])
            if aid == bid:
                continue
            t0_b = float(b["t0"])
            # Ребро строится только вперед по времени старта
            if t0_b < t0_a:
                continue
            gap = t0_b - t1_a
            # Ограничение по наложению (overlap) и по максимальной паузе (gap)
            if gap < -max_overlap_sec:
                continue
            if gap > max_gap_sec:
                continue
            mat_b = mats.get(bid)
            if mat_b is None:
                continue
            reid = pair_embed_score(mat_a, mat_b)
            if reid is None or reid < min_reid_score:
                continue

            gap_score = _gap_score(max(0.0, gap), max_gap_sec)
            combo = _combo_score(
                float(reid),
                gap_score,
                w_reid=w_reid,
                w_gap=w_gap,
            )
            # Короткий разрыв + почти идентичный ReID: гарантируем высокий скор.
            if float(reid) >= 0.97 and gap <= 3.0:
                combo = max(combo, 0.78)
            edges.append(
                (
                    aid,
                    bid,
                    combo,
                    float(reid),
                    float(gap),
                )
            )
    return edges


def _filter_edges(
    tracklet_ids: list[int],
    edges: list[tuple],
    *,
    min_score: float,
    min_reid: float = 0.0,
    t0: dict[int, float] | None,
) -> list[tuple[int, int, float]]:
    id_set = {int(tid) for tid in tracklet_ids}
    filtered: list[tuple[int, int, float]] = []
    for edge in edges:
        a, b, combo = int(edge[0]), int(edge[1]), float(edge[2])
        reid = float(edge[3]) if len(edge) > 3 else 1.0
        if combo < min_score or (min_reid > 0 and reid < min_reid):
            continue
        if a not in id_set or b not in id_set:
            continue
        if t0 is not None and t0.get(b, 0) < t0.get(a, 0):
            continue
        filtered.append((a, b, combo))
    return filtered


def _chains_from_succ(
    tracklet_ids: list[int],
    succ: dict[int, int],
    *,
    t0: dict[int, float] | None,
) -> list[list[int]]:
    ids = [int(tid) for tid in tracklet_ids]
    pred = {b: a for a, b in succ.items()}
    visited: set[int] = set()
    chains: list[list[int]] = []
    starts = [tid for tid in sorted(ids, key=lambda x: t0.get(x, 0) if t0 else x) if tid not in pred]
    for tid in starts:
        if tid in visited:
            continue
        chain = [tid]
        visited.add(tid)
        cur = tid
        while cur in succ:
            nxt = succ[cur]
            if nxt in visited:
                break
            chain.append(nxt)
            visited.add(nxt)
            cur = nxt
        chains.append(chain)
    for tid in ids:
        if tid not in visited:
            chains.append([tid])
    return chains


def link_greedy_chains(
    tracklet_ids: list[int],
    edges: list[tuple],
    *,
    min_score: float,
    min_reid: float = 0.0,
    t0: dict[int, float] | None = None,
) -> list[list[int]]:
    """Жадный path cover: <=1 предшественник и <=1 преемник, рёбра по combo ↓."""
    filtered = _filter_edges(tracklet_ids, edges, min_score=min_score, min_reid=min_reid, t0=t0)
    succ: dict[int, int] = {}
    pred: dict[int, int] = {}
    for a, b, _score in sorted(filtered, key=lambda e: -e[2]):
        if a in succ or b in pred:
            continue
        succ[a] = b
        pred[b] = a
    return _chains_from_succ(tracklet_ids, succ, t0=t0)


link_hungarian_chains = link_greedy_chains
link_mcf_chains = link_greedy_chains


def _split_overlap_groups(
    groups: list[list[int]],
    spans: dict[int, tuple[float, float]],
    *,
    max_overlap_sec: float = 0.0,
) -> list[list[int]]:
    final_groups: list[list[int]] = []
    for group in groups:
        bucket: list[list[int]] = []
        for tid in sorted(group, key=lambda x: spans.get(x, (0.0, 0.0))[0]):
            placed = False
            for part in bucket:
                if all(
                    intervals_overlap_sec(
                        spans[tid][0],
                        spans[tid][1],
                        spans[other][0],
                        spans[other][1],
                    ) <= max_overlap_sec
                    for other in part
                ):
                    part.append(tid)
                    placed = True
                    break
            if not placed:
                bucket.append([tid])
        final_groups.extend(bucket)
    return final_groups


def _unique_groups(groups: list[list[int]]) -> list[list[int]]:
    """Каждый tracklet_id ровно в одной группе (длинная цепочка побеждает)."""
    ranked = sorted(groups, key=lambda g: (-len(g), g[0] if g else 0))
    used: set[int] = set()
    out: list[list[int]] = []
    for group in ranked:
        kept = [tid for tid in group if tid not in used]
        if not kept:
            continue
        used.update(kept)
        out.append(kept)
    out.sort(key=lambda g: min(g) if g else 0)
    return out



def _pass_alone_geo_stitch(
    groups: list[list[int]],
    tracklets: list[dict[str, Any]],
    embeddings: dict[int, np.ndarray],
    *,
    radius_m: float,
    max_gap_sec: float,
    max_overlap_sec: float = 2.0,
    max_dist_m: float,
    max_speed_mps: float,
    min_reid: float = 0.0,
    frames_pos: dict[int, dict[int, tuple[float, float, str]]] | None = None,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Pass Alone Geo (Pass 1): связывание изолированных треклетов с помощью TrackNeighborhoodIndex."""
    if len(groups) < 2 or not tracklets:
        return groups, []

    tracklets_map = {int(t["tracklet_id"]): t for t in tracklets}
    mats = {int(t["tracklet_id"]): _crop_matrix(embeddings.get(int(t["tracklet_id"]))) for t in tracklets}

    # Строим пространственно-временной индекс треклетов
    spatial_index = TrackNeighborhoodIndex.from_tracklets(
        tracklets,
        frames_pos=frames_pos,
    )

    candidates: list[tuple[int, int, float, dict[str, Any]]] = []
    for ga, group_a in enumerate(groups):
        if not group_a:
            continue
        aid = group_a[-1]
        ta = tracklets_map.get(aid)
        if not ta:
            continue
        t1_a = float(ta.get("t1", 0.0))
        p1_a = _tracklet_endpoint(ta, "p1")

        for gb, group_b in enumerate(groups):
            if ga == gb or not group_b:
                continue
            bid = group_b[0]
            tb = tracklets_map.get(bid)
            if not tb:
                continue
            t0_b = float(tb.get("t0", 0.0))
            if t0_b < t1_a:
                continue
            gap_sec = t0_b - t1_a
            if gap_sec > max_gap_sec:
                continue

            p0_b = _tracklet_endpoint(tb, "p0")
            dist_m = calc_points_dist_m(
                p1_a,
                p0_b,
                scale_px_per_m=spatial_index.scale_px_per_m,
                fallback_median_h_px=spatial_index.median_h_px,
            )
            if max_dist_m > 0 and dist_m > max_dist_m:
                continue

            dt = max(gap_sec, 0.1)
            speed_mps = dist_m / dt
            if max_speed_mps > 0 and speed_mps > max_speed_mps:
                continue

            reid_val: float | None = None
            if mats.get(aid) is not None and mats.get(bid) is not None:
                reid_val = pair_embed_score(mats[aid], mats[bid])

            if min_reid > 0:
                if reid_val is None or float(reid_val) < min_reid:
                    continue

            # Проверяем изоляцию пары A -> B через пространственно-временной индекс
            if not spatial_index.is_pair_transition_clear(
                aid,
                bid,
                radius_m=radius_m,
                group_a=set(group_a),
                group_b=set(group_b),
            ):
                continue

            # Скор связи
            effective_gap = max(0.0, gap_sec)
            geo_score = math.exp(-dist_m / max(max_dist_m, 1.0)) * (1.0 - effective_gap / max(max_gap_sec, 1.0))
            if reid_val is not None:
                geo_score = 0.5 * geo_score + 0.5 * float(reid_val)

            edge_info = {
                "from": aid,
                "to": bid,
                "score": round(float(geo_score), 4),
                "reid": round(float(reid_val), 4) if reid_val is not None else None,
                "gap": round(float(gap_sec), 2),
                "dist_m": round(float(dist_m), 2),
                "speed_mps": round(float(speed_mps), 2),
                "pass": 1,
                "reason": (
                    f"Pass 1 (Alone Geo): чистота R={radius_m:.1f}м "
                    f"(Δd={dist_m:.2f}м, Δt={gap_sec:.1f}с, v={speed_mps:.2f}м/с)"
                ),
            }
            candidates.append((ga, gb, geo_score, edge_info))

    if not candidates:
        return groups, []

    # Жадный выбор лучших цепочек групп
    candidates.sort(key=lambda x: -x[2])
    parent = list(range(len(groups)))

    def _find(i: int) -> int:
        if parent[i] != i:
            parent[i] = _find(parent[i])
        return parent[i]

    def _union(i: int, j: int) -> bool:
        ri, rj = _find(i), _find(j)
        if ri == rj:
            return False
        parent[rj] = ri
        return True

    chosen_edges: dict[tuple[int, int], dict[str, Any]] = {}
    for ga, gb, _score, edge_info in candidates:
        if _union(ga, gb):
            chosen_edges[(edge_info["from"], edge_info["to"])] = edge_info

    group_map: dict[int, list[int]] = {}
    for i, g in enumerate(groups):
        root = _find(i)
        group_map.setdefault(root, []).extend(g)

    spans = {int(t["tracklet_id"]): (float(t["t0"]), float(t["t1"])) for t in tracklets}
    new_groups = list(group_map.values())
    new_groups = _split_overlap_groups(new_groups, spans, max_overlap_sec=max_overlap_sec)
    new_groups = _unique_groups(new_groups)

    return new_groups, list(chosen_edges.values())


def link_tracklets(
    tracklets: list[dict[str, Any]],
    embeddings: dict[int, np.ndarray],
    *,
    max_gap_sec: float,
    max_overlap_sec: float = 2.0,
    min_reid_score: float,
    w_reid: float = 0.85,
    w_gap: float = 0.15,
    pass0_min_reid: float = 0.0,
    pass0_min_score: float = 0.0,
    pass_alone_enabled: bool = False,
    pass_alone_radius_m: float = 2.0,
    pass_alone_max_gap_sec: float = 15.0,
    pass_alone_max_dist_m: float = 3.0,
    pass_alone_max_speed_mps: float = 2.0,
    pass_alone_min_reid: float = 0.0,
    frames_pos: dict[int, dict[int, tuple[float, float, str]]] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Склейка треклетов: Pass 0 (жадный матчинг по ReID + Gap) + Pass 1 (Alone Geo)."""
    t0 = {int(t["tracklet_id"]): float(t["t0"]) for t in tracklets}
    spans = {int(t["tracklet_id"]): (float(t["t0"]), float(t["t1"])) for t in tracklets}
    ids = [int(t["tracklet_id"]) for t in tracklets]

    effective_min_reid = pass0_min_reid if pass0_min_reid > 0 else min_reid_score
    effective_candidate_reid = min(min_reid_score, effective_min_reid)

    edges = build_candidate_edges(
        tracklets,
        embeddings,
        max_gap_sec=max_gap_sec,
        max_overlap_sec=max_overlap_sec,
        min_reid_score=effective_candidate_reid,
        w_reid=w_reid,
        w_gap=w_gap,
    )

    chains = link_greedy_chains(
        ids,
        edges,
        min_score=pass0_min_score,
        min_reid=pass0_min_reid,
        t0=t0,
    )

    pass0_used: set[tuple[int, int]] = set()
    for chain in chains:
        for i in range(len(chain) - 1):
            pass0_used.add((int(chain[i]), int(chain[i + 1])))

    groups = _split_overlap_groups(chains, spans, max_overlap_sec=max_overlap_sec)
    groups = _unique_groups(groups)

    assigned = {tid for g in groups for tid in g}
    for t in tracklets:
        tid = int(t["tracklet_id"])
        if tid not in assigned:
            groups.append([tid])

    # Pass Alone Geo (Pass 1)
    pass_alone_edges: list[dict[str, Any]] = []
    if pass_alone_enabled:
        groups, pass_alone_edges = _pass_alone_geo_stitch(
            groups,
            tracklets,
            embeddings,
            radius_m=pass_alone_radius_m,
            max_gap_sec=pass_alone_max_gap_sec,
            max_overlap_sec=max_overlap_sec,
            max_dist_m=pass_alone_max_dist_m,
            max_speed_mps=pass_alone_max_speed_mps,
            min_reid=pass_alone_min_reid,
            frames_pos=frames_pos,
        )

    all_edges: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, int]] = set()

    for edge in edges:
        a, b = int(edge[0]), int(edge[1])
        combo, reid = float(edge[2]), float(edge[3])
        gap = float(edge[4]) if len(edge) > 4 else None

        parts = [f"ReID={reid:.4f}"]
        if gap is not None:
            parts.append(f"Δt={gap:.1f}с")
        reason_str = "Pass 0: " + ", ".join(parts)

        is_used = (a, b) in pass0_used
        edge_dict: dict[str, Any] = {
            "from": a,
            "to": b,
            "score": round(combo, 4),
            "reid": round(reid, 4),
            "gap": round(gap, 2) if gap is not None else None,
            "reason": reason_str,
        }
        if is_used:
            edge_dict["pass"] = 0
        all_edges.append(edge_dict)
        seen_pairs.add((a, b))

    # Добавляем ребра Pass Alone Geo, если их не было среди candidate_edges
    for pe in pass_alone_edges:
        pair = (int(pe["from"]), int(pe["to"]))
        found = False
        for e in all_edges:
            if (int(e["from"]), int(e["to"])) == pair:
                e["pass"] = 1
                e["reason"] = pe["reason"]
                e["dist_m"] = pe.get("dist_m")
                e["speed_mps"] = pe.get("speed_mps")
                found = True
                break
        if not found:
            all_edges.append(pe)
            seen_pairs.add(pair)

    tracklet_to_global: dict[str, int] = {}
    for gi, group in enumerate(groups, start=1):
        for tid in group:
            tracklet_to_global[str(int(tid))] = gi

    return {
        "solver": "greedy",
        "groups": groups,
        "edges": all_edges,
        "tracklet_to_global": tracklet_to_global,
        "pass0_merged": len(pass0_used),
        "pass_alone_merged": len(pass_alone_edges),
    }
