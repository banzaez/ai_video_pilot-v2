"""Глобальная склейка треклетов: Pass 0–4 + Hungarian/greedy."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from app.util.union_find import _UF
from app.util.intervals import intervals_overlap, pair_embed_score
from app.global_id.spatial import METER_PX


def _as_xy(p: Any) -> tuple[float, float]:
    if not p or not isinstance(p, (list, tuple)) or len(p) < 2:
        return 0.0, 0.0
    return float(p[0]), float(p[1])


def _dist(p0: Any, p1: Any) -> float:
    x0, y0 = _as_xy(p0)
    x1, y1 = _as_xy(p1)
    return math.hypot(x1 - x0, y1 - y0)


def _tracklet_spatial_point(tracklet: dict[str, Any], key: str) -> tuple[float, float, str]:
    """key: p0 | p1. Возвращает (x, y, space) где space = map|image.

    Map используется для motion, если точка с позы (map_src=kpt_*) либо позы нет.
    Иначе — image p0/p1 (лодыжки): bbox-map не должен перебивать точные ноги.
    """
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
    """Похожесть масштаба. Обрезанный bbox (та же ширина, меньше высота) не штрафуем."""
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


def _gap_score(gap: float, max_gap_sec: float) -> float:
    if max_gap_sec <= 0:
        return 1.0
    return float(max(0.0, 1.0 - gap / max_gap_sec))


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


def build_candidate_edges(
    tracklets: list[dict[str, Any]],
    embeddings: dict[int, np.ndarray],
    *,
    max_gap_sec: float,
    min_reid_score: float,
    max_spatial_px: float,
    motion_sigma_px: float = 180.0,
    motion_sigma_m: float = 0.0,
    max_spatial_m: float = 0.0,
    size_log_scale: float = 0.45,
    w_reid: float = 0.55,
    w_motion: float = 0.25,
    w_size: float = 0.10,
    w_gap: float = 0.10,
    spatial_relax_px: float = 0.0,
    spatial_relax_min_score: float = 0.0,
) -> list[tuple[int, int, float, float]]:
    """Рёбра A→B: (from, to, combo, reid). spatial_relax_* игнорируются (совместимость)."""
    del spatial_relax_px, spatial_relax_min_score
    edges: list[tuple[int, int, float, float]] = []
    sorted_tl = sorted(tracklets, key=lambda r: (float(r["t0"]), int(r["tracklet_id"])))
    mats = {int(t["tracklet_id"]): _crop_matrix(embeddings.get(int(t["tracklet_id"]))) for t in sorted_tl}

    for i, a in enumerate(sorted_tl):
        aid = int(a["tracklet_id"])
        mat_a = mats.get(aid)
        if mat_a is None:
            continue
        for b in sorted_tl[i + 1 :]:
            bid = int(b["tracklet_id"])
            if float(b["t0"]) < float(a["t1"]):
                continue
            gap = float(b["t0"]) - float(a["t1"])
            if gap > max_gap_sec:
                break
            mat_b = mats.get(bid)
            if mat_b is None:
                continue
            reid = pair_embed_score(mat_a, mat_b)
            if reid is None or reid < min_reid_score:
                continue
            motion, residual = _motion_score(
                a,
                b,
                sigma_px=motion_sigma_px,
                sigma_m=motion_sigma_m if _pair_uses_map(a, b, "p1", "p0") else 0.0,
            )
            b_space = _tracklet_spatial_point(b, "p0")[2]
            if _spatial_threshold_exceeded(
                residual,
                space=b_space,
                max_px=max_spatial_px,
                max_m=max_spatial_m if _pair_uses_map(a, b, "p1", "p0") else 0.0,
            ):
                continue
            size = _size_score(a, b, log_scale=size_log_scale)
            combo = _combo_score(
                float(reid),
                motion,
                size,
                _gap_score(gap, max_gap_sec),
                w_reid=w_reid,
                w_motion=w_motion,
                w_size=w_size,
                w_gap=w_gap,
            )
            # Короткий разрыв + почти идентичный ReID: не режем size/motion (bbox скачет).
            if float(reid) >= 0.97 and gap <= 3.0:
                combo = max(combo, 0.78)
            edges.append(
                (
                    aid,
                    bid,
                    combo,
                    float(reid),
                    float(motion),
                    float(size),
                    float(gap),
                    float(residual),
                    str(b_space),
                )
            )
    return edges


def _time_windows(
    tracklets: list[dict[str, Any]],
    *,
    window_sec: float,
    overlap_sec: float,
) -> list[tuple[float, float]]:
    if not tracklets or window_sec <= 0:
        t0 = min(float(t["t0"]) for t in tracklets) if tracklets else 0.0
        t1 = max(float(t["t1"]) for t in tracklets) if tracklets else 0.0
        return [(t0, t1)]
    t_min = min(float(t["t0"]) for t in tracklets)
    t_max = max(float(t["t1"]) for t in tracklets)
    step = max(1.0, window_sec - overlap_sec)
    windows: list[tuple[float, float]] = []
    start = t_min
    while start <= t_max:
        windows.append((start, start + window_sec))
        start += step
    return windows


def _tracklets_in_window(
    tracklets: list[dict[str, Any]],
    win: tuple[float, float],
) -> list[dict[str, Any]]:
    w0, w1 = win
    return [t for t in tracklets if float(t["t1"]) >= w0 and float(t["t0"]) <= w1]


def _filter_edges(
    tracklet_ids: list[int],
    edges: list[tuple],
    *,
    min_score: float,
    t0: dict[int, float] | None,
) -> list[tuple[int, int, float]]:
    id_set = {int(tid) for tid in tracklet_ids}
    filtered: list[tuple[int, int, float]] = []
    for edge in edges:
        a, b, combo = int(edge[0]), int(edge[1]), float(edge[2])
        if combo < min_score or a not in id_set or b not in id_set:
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
    t0: dict[int, float] | None = None,
) -> list[list[int]]:
    """Жадный path cover: ≤1 предшественник и ≤1 преемник, рёбра по combo ↓."""
    filtered = _filter_edges(tracklet_ids, edges, min_score=min_score, t0=t0)
    succ: dict[int, int] = {}
    pred: dict[int, int] = {}
    for a, b, _score in sorted(filtered, key=lambda e: -e[2]):
        if a in succ or b in pred:
            continue
        succ[a] = b
        pred[b] = a
    return _chains_from_succ(tracklet_ids, succ, t0=t0)


def link_hungarian_chains(
    tracklet_ids: list[int],
    edges: list[tuple],
    *,
    min_score: float,
    t0: dict[int, float] | None = None,
) -> list[list[int]]:
    """1-1 assignment: max сумма combo, у узла ≤1 вход и ≤1 выход."""
    ids = [int(tid) for tid in tracklet_ids]
    filtered = _filter_edges(ids, edges, min_score=min_score, t0=t0)
    if not filtered:
        return [[tid] for tid in ids]
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return link_greedy_chains(ids, edges, min_score=min_score, t0=t0)

    n = len(ids)
    idx = {tid: i for i, tid in enumerate(ids)}
    cost = np.zeros((n, n), dtype=np.float64)
    for a, b, combo in filtered:
        i, j = idx[a], idx[b]
        if i == j:
            continue
        cost[i, j] = -float(combo)

    row_ind, col_ind = linear_sum_assignment(cost)
    succ: dict[int, int] = {}
    pred: dict[int, int] = {}
    for i, j in zip(row_ind, col_ind):
        if i == j or cost[i, j] >= 0:
            continue
        a, b = ids[i], ids[j]
        if a in succ or b in pred:
            continue
        succ[a] = b
        pred[b] = a
    return _chains_from_succ(ids, succ, t0=t0)


def link_mcf_chains(
    tracklet_ids: list[int],
    edges: list[tuple],
    *,
    min_score: float,
    t0: dict[int, float] | None = None,
) -> list[list[int]]:
    """MCF в конфиге = Hungarian (настоящий LP на ~10² узлах не нужен)."""
    return link_hungarian_chains(
        tracklet_ids,
        edges,
        min_score=min_score,
        t0=t0,
    )


def _groups_have_edge(sa: set[int], sb: set[int], edge_pairs: set[tuple[int, int]]) -> bool:
    for a in sa:
        for b in sb:
            if (a, b) in edge_pairs or (b, a) in edge_pairs:
                return True
    return False


def _merge_chain_sets(
    chain_sets: list[list[list[int]]],
    *,
    edge_pairs: set[tuple[int, int]],
    spans: dict[int, tuple[float, float]],
) -> list[list[int]]:
    """Стык окон только если есть ребро и нет temporal overlap в объединении."""
    if not chain_sets:
        return []
    merged: list[set[int]] = [set(c) for c in chain_sets[0] if c]
    for win_chains in chain_sets[1:]:
        next_sets = [set(c) for c in win_chains if c]
        combined: list[set[int]] = []
        used_b: set[int] = set()
        for sa in merged:
            candidates = []
            for bi, sb in enumerate(next_sets):
                if bi in used_b:
                    continue
                intersection_len = len(sa & sb)
                has_edge = _groups_have_edge(sa, sb, edge_pairs)
                if intersection_len == 0 and not has_edge:
                    continue
                candidates.append((intersection_len, 1 if has_edge else 0, bi, sb))
            # Сортируем кандидатов по силе совпадения (сначала общие tracklet_id, затем ребра)
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

            hit = False
            for _, _, bi, sb in candidates:
                union = sa | sb
                if _group_has_overlap(union, spans):
                    continue
                combined.append(union)
                used_b.add(bi)
                hit = True
                break
            if not hit:
                combined.append(set(sa))
        for bi, sb in enumerate(next_sets):
            if bi not in used_b:
                combined.append(set(sb))
        merged = combined
    return [sorted(s) for s in merged]


def _overlap_sec(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(float(a0), float(b0))
    hi = min(float(a1), float(b1))
    return max(0.0, hi - lo)


def _pair_overlap_too_long(
    a0: float,
    a1: float,
    b0: float,
    b1: float,
    max_overlap_sec: float,
) -> bool:
    if max_overlap_sec <= 0:
        return intervals_overlap(a0, a1, b0, b1)
    return _overlap_sec(a0, a1, b0, b1) > float(max_overlap_sec) + 1e-9


def _group_has_overlap(group: set[int], spans: dict[int, tuple[float, float]]) -> bool:
    items = sorted(group)
    for i, a in enumerate(items):
        a0, a1 = spans[a]
        for b in items[i + 1 :]:
            b0, b1 = spans[b]
            if intervals_overlap(a0, a1, b0, b1):
                return True
    return False


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
                    not _pair_overlap_too_long(
                        spans[tid][0],
                        spans[tid][1],
                        spans[other][0],
                        spans[other][1],
                        max_overlap_sec,
                    )
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


def _group_span(group: list[int], spans: dict[int, tuple[float, float]]) -> tuple[float, float]:
    t0 = min(spans[tid][0] for tid in group)
    t1 = max(spans[tid][1] for tid in group)
    return t0, t1


def _span_gap(sa: tuple[float, float], sb: tuple[float, float]) -> float | None:
    if intervals_overlap(sa[0], sa[1], sb[0], sb[1]):
        return None
    if sa[1] <= sb[0]:
        return sb[0] - sa[1]
    return sa[0] - sb[1]


def _ends_from_tracklets(tracklets: list[dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    if not tracklets:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for t in tracklets:
        tid = int(t["tracklet_id"])
        vx, vy = _velocity(t)
        p0x, p0y, _ = _tracklet_spatial_point(t, "p0")
        p1x, p1y, _ = _tracklet_spatial_point(t, "p1")
        out[tid] = {
            "t0": float(t["t0"]),
            "t1": float(t["t1"]),
            "p0": (p0x, p0y),
            "p1": (p1x, p1y),
            "vx": vx,
            "vy": vy,
            "space": _tracklet_spatial_point(t, "p0")[2],
        }
    return out


def _exit_tid(group: list[int], spans: dict[int, tuple[float, float]], ends: dict[int, dict[str, Any]]) -> int:
    return max(group, key=lambda tid: (ends[tid]["t1"] if tid in ends else spans[tid][1], tid))


def _entry_tid(group: list[int], spans: dict[int, tuple[float, float]], ends: dict[int, dict[str, Any]]) -> int:
    return min(group, key=lambda tid: (ends[tid]["t0"] if tid in ends else spans[tid][0], tid))


def _handoff_gap_dist(
    ga: list[int],
    gb: list[int],
    spans: dict[int, tuple[float, float]],
    ends: dict[int, dict[str, Any]],
) -> tuple[float | None, float | None]:
    """Разрыв и дистанция: конец ранней цепочки → старт поздней (где пропал / где появился)."""
    gap_span = _span_gap(_group_span(ga, spans), _group_span(gb, spans))
    if gap_span is None:
        return None, None
    if not ends:
        return gap_span, None

    xa = _exit_tid(ga, spans, ends)
    ea = _entry_tid(ga, spans, ends)
    xb = _exit_tid(gb, spans, ends)
    eb = _entry_tid(gb, spans, ends)
    options: list[tuple[float, float]] = []
    for exit_id, entry_id in ((xa, eb), (xb, ea)):
        if exit_id not in ends or entry_id not in ends:
            continue
        t1 = float(ends[exit_id]["t1"])
        t0 = float(ends[entry_id]["t0"])
        if t0 < t1:
            continue
        gap = t0 - t1
        p1x, p1y = ends[exit_id]["p1"]
        p0x, p0y = ends[entry_id]["p0"]
        dist_last = math.hypot(p0x - p1x, p0y - p1y)
        pred_x = p1x + float(ends[exit_id]["vx"]) * gap
        pred_y = p1y + float(ends[exit_id]["vy"]) * gap
        dist_pred = math.hypot(p0x - pred_x, p0y - pred_y)
        dist = min(dist_last, dist_pred)
        if ends[entry_id].get("space") == "map":
            dist /= METER_PX
        options.append((gap, dist))
    if not options:
        return gap_span, None
    options.sort(key=lambda item: item[0])
    return options[0]


def _tracklets_use_map(tracklets: list[dict[str, Any]]) -> bool:
    return any(isinstance(t.get("map_p0"), (list, tuple)) for t in tracklets)


def _pass0_merge_groups(
    groups: list[list[int]],
    edges: list[dict[str, Any]],
    *,
    spans: dict[int, tuple[float, float]],
    min_reid: float,
    tracklets: list[dict[str, Any]] | None = None,
) -> tuple[list[list[int]], set[tuple[int, int]]]:
    """Pass 0: склеить готовые цепочки только по ReID ≥ min_reid (выход A → вход B), 1-1, без overlap."""
    if min_reid <= 0 or len(groups) < 2:
        return groups, set()

    ends = _ends_from_tracklets(tracklets)
    by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for e in edges:
        a, b = int(e["from"]), int(e["to"])
        reid = float(e.get("reid") or 0)
        if reid < min_reid:
            continue
        prev = by_pair.get((a, b))
        if prev is None or reid > float(prev.get("reid") or 0):
            by_pair[(a, b)] = e

    best: dict[tuple[int, int], tuple[float, int, int]] = {}
    for ga, group_a in enumerate(groups):
        if not group_a:
            continue
        for gb, group_b in enumerate(groups):
            if ga == gb or not group_b:
                continue
            if _group_has_overlap(set(group_a) | set(group_b), spans):
                continue
            sa = _group_span(group_a, spans)
            sb = _group_span(group_b, spans)
            if sb[0] < sa[1]:
                continue
            exit_id = _exit_tid(group_a, spans, ends)
            entry_id = _entry_tid(group_b, spans, ends)
            rec = by_pair.get((exit_id, entry_id))
            if rec is None:
                continue
            reid = float(rec.get("reid") or 0)
            best[(ga, gb)] = (reid, exit_id, entry_id)

    if not best:
        return groups, set()

    group_ids = list(range(len(groups)))
    t0_g = {gi: _group_span(groups[gi], spans)[0] for gi in group_ids if groups[gi]}
    fake_edges = [(ga, gb, reid) for (ga, gb), (reid, _, _) in best.items()]
    chains = link_hungarian_chains(group_ids, fake_edges, min_score=min_reid, t0=t0_g)

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
        new_groups.append(sorted(set(merged), key=lambda tid: (spans[tid][0], tid)))

    new_groups = _split_overlap_groups(new_groups, spans)
    return _unique_groups(new_groups), used


def _pass2_merge_groups(
    groups: list[list[int]],
    edges: list[dict[str, Any]],
    *,
    spans: dict[int, tuple[float, float]],
    min_combo: float,
    tracklets: list[dict[str, Any]] | None = None,
) -> tuple[list[list[int]], set[tuple[int, int]]]:
    """Pass 2: склеить готовые цепочки по combo ≥ min_combo (выход A → вход B), 1-1, без overlap."""
    if min_combo <= 0 or len(groups) < 2:
        return groups, set()

    ends = _ends_from_tracklets(tracklets)
    by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for e in edges:
        a, b = int(e["from"]), int(e["to"])
        combo = float(e.get("score") or 0)
        if combo < min_combo:
            continue
        prev = by_pair.get((a, b))
        if prev is None or combo > float(prev.get("score") or 0):
            by_pair[(a, b)] = e

    best: dict[tuple[int, int], tuple[float, int, int]] = {}
    for ga, group_a in enumerate(groups):
        if not group_a:
            continue
        for gb, group_b in enumerate(groups):
            if ga == gb or not group_b:
                continue
            if _group_has_overlap(set(group_a) | set(group_b), spans):
                continue
            sa = _group_span(group_a, spans)
            sb = _group_span(group_b, spans)
            if sb[0] < sa[1]:
                continue
            exit_id = _exit_tid(group_a, spans, ends)
            entry_id = _entry_tid(group_b, spans, ends)
            rec = by_pair.get((exit_id, entry_id))
            if rec is None:
                continue
            combo = float(rec["score"])
            best[(ga, gb)] = (combo, exit_id, entry_id)

    if not best:
        return groups, set()

    group_ids = list(range(len(groups)))
    t0_g = {gi: _group_span(groups[gi], spans)[0] for gi in group_ids if groups[gi]}
    fake_edges = [(ga, gb, combo) for (ga, gb), (combo, _, _) in best.items()]
    chains = link_hungarian_chains(group_ids, fake_edges, min_score=min_combo, t0=t0_g)

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
        new_groups.append(sorted(set(merged), key=lambda tid: (spans[tid][0], tid)))

    new_groups = _split_overlap_groups(new_groups, spans)
    return _unique_groups(new_groups), used


def _pass3_splice_gaps(
    groups: list[list[int]],
    edges: list[dict[str, Any]],
    *,
    spans: dict[int, tuple[float, float]],
    min_combo: float,
) -> tuple[list[list[int]], set[tuple[int, int]]]:
    """Pass 3: вставить цепочку в дырку A→…→B, если оба стыка combo ≥ min_combo."""
    if min_combo <= 0 or len(groups) < 2:
        return groups, set()

    by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for e in edges:
        a, b = int(e["from"]), int(e["to"])
        combo = float(e.get("score") or 0)
        if combo < min_combo:
            continue
        prev = by_pair.get((a, b))
        if prev is None or combo > float(prev.get("score") or 0):
            by_pair[(a, b)] = e
    if not by_pair:
        return groups, set()

    work = [sorted(g, key=lambda tid: (spans[tid][0], tid)) for g in groups if g]
    used: set[tuple[int, int]] = set()

    def _best_insert() -> tuple[float, int, int, int, int, int, int, int] | None:
        best: tuple[float, int, int, int, int, int, int, int] | None = None
        for hi, host in enumerate(work):
            if len(host) < 2:
                continue
            for k in range(len(host) - 1):
                left, right = host[k], host[k + 1]
                hole0, hole1 = spans[left][1], spans[right][0]
                if hole1 < hole0:
                    continue
                for gi, guest in enumerate(work):
                    if gi == hi or not guest:
                        continue
                    g0, g1 = _group_span(guest, spans)
                    if g0 < hole0 or g1 > hole1:
                        continue
                    if _group_has_overlap(set(host) | set(guest), spans):
                        continue
                    entry, exit_id = guest[0], guest[-1]
                    e_left = by_pair.get((left, entry))
                    e_right = by_pair.get((exit_id, right))
                    if e_left is None or e_right is None:
                        continue
                    score = min(float(e_left["score"]), float(e_right["score"]))
                    cand = (score, hi, gi, k, left, entry, exit_id, right)
                    if best is None or cand[0] > best[0]:
                        best = cand
        return best

    while True:
        hit = _best_insert()
        if hit is None:
            break
        _, hi, gi, k, left, entry, exit_id, right = hit
        work[hi] = work[hi][: k + 1] + work[gi] + work[hi][k + 1 :]
        work[gi] = []
        used.add((left, entry))
        used.add((exit_id, right))

    new_groups = [g for g in work if g]
    new_groups = _split_overlap_groups(new_groups, spans)
    return _unique_groups(new_groups), used


def _pair_metrics(
    a: dict[str, Any],
    b: dict[str, Any],
    embeddings: dict[int, np.ndarray],
    *,
    min_reid_score: float,
    max_spatial_px: float,
    max_spatial_m: float,
    motion_sigma_px: float,
    motion_sigma_m: float,
    size_log_scale: float,
    max_gap_sec: float,
    w_reid: float,
    w_motion: float,
    w_size: float,
    w_gap: float,
    gap: float,
) -> dict[str, Any] | None:
    mat_a = _crop_matrix(embeddings.get(int(a["tracklet_id"])))
    mat_b = _crop_matrix(embeddings.get(int(b["tracklet_id"])))
    reid = pair_embed_score(mat_a, mat_b)
    if reid is None or reid < min_reid_score:
        return None
    motion, residual = _motion_score(
        a,
        b,
        sigma_px=motion_sigma_px,
        sigma_m=motion_sigma_m if _pair_uses_map(a, b, "p1", "p0") else 0.0,
    )
    b_space = _tracklet_spatial_point(b, "p0")[2]
    if _spatial_threshold_exceeded(
        residual,
        space=b_space,
        max_px=max_spatial_px,
        max_m=max_spatial_m if _pair_uses_map(a, b, "p1", "p0") else 0.0,
    ):
        return None
    size = _size_score(a, b, log_scale=size_log_scale)
    combo = _combo_score(
        float(reid),
        motion,
        size,
        _gap_score(gap, max_gap_sec),
        w_reid=w_reid,
        w_motion=w_motion,
        w_size=w_size,
        w_gap=w_gap,
    )
    return {
        "combo": float(combo),
        "reid": float(reid),
        "motion": float(motion),
        "size": float(size),
        "gap": float(gap),
        "dist": float(residual),
        "space": str(b_space),
    }


def _pass4_handover_groups(
    groups: list[list[int]],
    tracklets: list[dict[str, Any]],
    embeddings: dict[int, np.ndarray],
    *,
    spans: dict[int, tuple[float, float]],
    max_overlap_sec: float,
    min_reid: float,
    min_combo: float,
    max_spatial_px: float,
    max_spatial_m: float,
    motion_sigma_px: float,
    motion_sigma_m: float,
    size_log_scale: float,
    max_gap_sec: float,
    w_reid: float,
    w_motion: float,
    w_size: float,
    w_gap: float,
) -> tuple[list[list[int]], list[dict[str, Any]], set[tuple[int, int]]]:
    """Pass 4: B стартует в хвосте A (короткий overlap), ReID+combo свои пороги."""
    if max_overlap_sec <= 0 or len(groups) < 2:
        return groups, [], set()

    by_id = {int(t["tracklet_id"]): t for t in tracklets}
    work = [sorted(g, key=lambda tid: (spans[tid][0], tid)) for g in groups if g]
    best: dict[tuple[int, int], tuple[float, int, int, dict[str, Any]]] = {}

    for ga, group_a in enumerate(work):
        if not group_a:
            continue
        sa = _group_span(group_a, spans)
        for gb, group_b in enumerate(work):
            if ga == gb or not group_b:
                continue
            sb = _group_span(group_b, spans)
            if not (sa[0] <= sb[0] < sa[1] < sb[1]):
                continue
            overlap = sa[1] - sb[0]
            if overlap > max_overlap_sec + 1e-9:
                continue
            exit_id = max(group_a, key=lambda tid: (spans[tid][1], tid))
            entry_id = min(group_b, key=lambda tid: (spans[tid][0], tid))
            if not intervals_overlap(*spans[exit_id], *spans[entry_id]):
                continue
            ta, tb = by_id.get(exit_id), by_id.get(entry_id)
            if ta is None or tb is None:
                continue
            metrics = _pair_metrics(
                ta,
                tb,
                embeddings,
                min_reid_score=min_reid,
                max_spatial_px=max_spatial_px,
                max_spatial_m=max_spatial_m,
                motion_sigma_px=motion_sigma_px,
                motion_sigma_m=motion_sigma_m,
                size_log_scale=size_log_scale,
                max_gap_sec=max_gap_sec,
                w_reid=w_reid,
                w_motion=w_motion,
                w_size=w_size,
                w_gap=w_gap,
                gap=0.0,
            )
            if metrics is None or float(metrics["combo"]) < min_combo:
                continue
            best[(ga, gb)] = (float(metrics["combo"]), exit_id, entry_id, metrics)

    if not best:
        return work, [], set()

    group_ids = list(range(len(work)))
    t0_g = {gi: _group_span(work[gi], spans)[0] for gi in group_ids if work[gi]}
    fake_edges = [(ga, gb, combo) for (ga, gb), (combo, _, _, _) in best.items()]
    chains = link_hungarian_chains(group_ids, fake_edges, min_score=min_combo, t0=t0_g)

    used: set[tuple[int, int]] = set()
    synth: list[dict[str, Any]] = []
    new_groups: list[list[int]] = []
    for chain in chains:
        merged: list[int] = []
        for i, gi in enumerate(chain):
            merged.extend(work[gi])
            if i + 1 < len(chain):
                rec = best.get((gi, chain[i + 1]))
                if rec is None:
                    continue
                _combo, exit_id, entry_id, metrics = rec
                used.add((exit_id, entry_id))
                space = str(metrics.get("space") or "image")
                unit = "м" if space == "map" else "px"
                dist = metrics.get("dist")
                dist_s = f" ({float(dist):.1f}{unit})" if dist is not None else ""
                ov = _overlap_sec(*spans[exit_id], *spans[entry_id])
                synth.append(
                    {
                        "from": exit_id,
                        "to": entry_id,
                        "score": round(float(metrics["combo"]), 4),
                        "reid": round(float(metrics["reid"]), 4),
                        "motion": round(float(metrics["motion"]), 4),
                        "size": round(float(metrics["size"]), 4),
                        "gap": 0.0,
                        "dist": round(float(dist), 2) if dist is not None else None,
                        "space": space,
                        "reason": (
                            f"Pass 4: handover, ReID={float(metrics['reid']):.2f}, "
                            f"Motion={float(metrics['motion']):.2f}{dist_s}, "
                            f"overlap={ov:.2f}с, Size={float(metrics['size']):.2f}"
                        ),
                        "pass": 4,
                    }
                )
        new_groups.append(sorted(set(merged), key=lambda tid: (spans[tid][0], tid)))

    new_groups = _split_overlap_groups(new_groups, spans, max_overlap_sec=max_overlap_sec)
    return _unique_groups(new_groups), synth, used


def link_tracklets(
    tracklets: list[dict[str, Any]],
    embeddings: dict[int, np.ndarray],
    *,
    max_gap_sec: float,
    min_reid_score: float,
    pass1_min_score: float,
    max_spatial_px: float,
    window_sec: float,
    window_overlap_sec: float,
    solver: str,
    spatial_relax_px: float = 0.0,
    spatial_relax_min_score: float = 0.0,
    motion_sigma_px: float = 180.0,
    motion_sigma_m: float = 0.0,
    max_spatial_m: float = 0.0,
    size_log_scale: float = 0.45,
    w_reid: float = 0.55,
    w_motion: float = 0.25,
    w_size: float = 0.10,
    w_gap: float = 0.10,
    pass0_min_reid: float = 0.0,
    pass2_min_score: float = 0.0,
    pass4_max_overlap_sec: float = 0.0,
    pass4_min_reid: float = 0.95,
    pass4_min_score: float = 0.85,
) -> dict[str, Any]:
    windows = _time_windows(
        tracklets,
        window_sec=window_sec,
        overlap_sec=window_overlap_sec,
    )
    chain_sets: list[list[list[int]]] = []
    all_edges: list[dict[str, Any]] = []
    edge_pairs: set[tuple[int, int]] = set()
    t0 = {int(t["tracklet_id"]): float(t["t0"]) for t in tracklets}
    spans = {int(t["tracklet_id"]): (float(t["t0"]), float(t["t1"])) for t in tracklets}
    if solver in ("hungarian", "mcf"):
        link_fn = link_hungarian_chains
    else:
        link_fn = link_greedy_chains

    for win in windows:
        subset = _tracklets_in_window(tracklets, win)
        if not subset:
            continue
        ids = [int(t["tracklet_id"]) for t in subset]
        edges = build_candidate_edges(
            subset,
            embeddings,
            max_gap_sec=max_gap_sec,
            min_reid_score=min_reid_score,
            max_spatial_px=max_spatial_px,
            motion_sigma_px=motion_sigma_px,
            motion_sigma_m=motion_sigma_m,
            max_spatial_m=max_spatial_m,
            size_log_scale=size_log_scale,
            w_reid=w_reid,
            w_motion=w_motion,
            w_size=w_size,
            w_gap=w_gap,
            spatial_relax_px=spatial_relax_px,
            spatial_relax_min_score=spatial_relax_min_score,
        )
        for edge in edges:
            a, b = int(edge[0]), int(edge[1])
            combo, reid = float(edge[2]), float(edge[3])
            motion = float(edge[4]) if len(edge) > 4 else None
            size = float(edge[5]) if len(edge) > 5 else None
            gap = float(edge[6]) if len(edge) > 6 else None
            dist = float(edge[7]) if len(edge) > 7 else None
            space = str(edge[8]) if len(edge) > 8 else "image"

            unit = "м" if space == "map" else "px"
            parts = [f"ReID={reid:.2f}"]
            if motion is not None:
                parts.append(f"Motion={motion:.2f}" + (f" ({dist:.1f}{unit})" if dist is not None else ""))
            if gap is not None:
                parts.append(f"Δt={gap:.1f}с")
            if size is not None:
                parts.append(f"Size={size:.2f}")
            reason_str = "Hungarian: " + ", ".join(parts)

            all_edges.append({
                "from": a,
                "to": b,
                "score": round(combo, 4),
                "reid": round(reid, 4),
                "motion": round(motion, 4) if motion is not None else None,
                "size": round(size, 4) if size is not None else None,
                "gap": round(gap, 2) if gap is not None else None,
                "dist": round(dist, 2) if dist is not None else None,
                "space": space,
                "reason": reason_str,
            })
            if combo >= pass1_min_score:
                edge_pairs.add((a, b))
        chain_sets.append(
            link_fn(ids, edges, min_score=pass1_min_score, t0=t0)
        )

    pass1_used: set[tuple[int, int]] = set()
    for chains in chain_sets:
        for chain in chains:
            for i in range(len(chain) - 1):
                pass1_used.add((int(chain[i]), int(chain[i + 1])))

    groups = _merge_chain_sets(chain_sets, edge_pairs=edge_pairs, spans=spans)
    if not groups:
        groups = [[int(t["tracklet_id"])] for t in tracklets]

    groups = _split_overlap_groups(groups, spans)
    groups = _unique_groups(groups)

    assigned = {tid for g in groups for tid in g}
    for t in tracklets:
        tid = int(t["tracklet_id"])
        if tid not in assigned:
            groups.append([tid])

    pass0_used: set[tuple[int, int]] = set()
    pass2_used: set[tuple[int, int]] = set()
    pass3_used: set[tuple[int, int]] = set()
    pass4_used: set[tuple[int, int]] = set()
    if pass0_min_reid > 0:
        groups, pass0_used = _pass0_merge_groups(
            groups,
            all_edges,
            spans=spans,
            min_reid=pass0_min_reid,
            tracklets=tracklets,
        )
    if pass2_min_score > 0:
        groups, pass2_used = _pass2_merge_groups(
            groups,
            all_edges,
            spans=spans,
            min_combo=pass2_min_score,
            tracklets=tracklets,
        )
        groups, pass3_used = _pass3_splice_gaps(
            groups,
            all_edges,
            spans=spans,
            min_combo=pass2_min_score,
        )
    if pass4_max_overlap_sec > 0:
        groups, pass4_edges, pass4_used = _pass4_handover_groups(
            groups,
            tracklets,
            embeddings,
            spans=spans,
            max_overlap_sec=pass4_max_overlap_sec,
            min_reid=pass4_min_reid,
            min_combo=pass4_min_score,
            max_spatial_px=max_spatial_px,
            max_spatial_m=max_spatial_m,
            motion_sigma_px=motion_sigma_px,
            motion_sigma_m=motion_sigma_m,
            size_log_scale=size_log_scale,
            max_gap_sec=max_gap_sec,
            w_reid=w_reid,
            w_motion=w_motion,
            w_size=w_size,
            w_gap=w_gap,
        )
        all_edges.extend(pass4_edges)

    tracklet_to_global: dict[str, int] = {}
    for gi, group in enumerate(groups, start=1):
        for tid in group:
            tracklet_to_global[str(int(tid))] = gi

    for e in all_edges:
        pair = (int(e["from"]), int(e["to"]))
        reason = str(e.get("reason") or "")
        if reason.startswith("Hungarian:"):
            reason = reason[len("Hungarian:") :].strip()
        if pair in pass4_used or e.get("pass") == 4:
            e["pass"] = 4
            if not str(e.get("reason") or "").startswith("Pass 4:"):
                e["reason"] = f"Pass 4: {reason}" if reason else "Pass 4"
        elif pair in pass3_used:
            e["pass"] = 3
            e["reason"] = f"Pass 3: {reason}" if reason else "Pass 3"
        elif pair in pass2_used:
            e["pass"] = 2
            e["pass2"] = True
            e["reason"] = f"Pass 2: {reason}" if reason else "Pass 2"
        elif pair in pass1_used:
            e["pass"] = 1
            e["reason"] = f"Pass 1: {reason}" if reason else "Pass 1"
        elif pair in pass0_used:
            e["pass"] = 0
            e["reason"] = f"Pass 0: {reason}" if reason else "Pass 0"

    return {
        "solver": solver,
        "groups": groups,
        "edges": all_edges,
        "tracklet_to_global": tracklet_to_global,
        "pass0_merged": len(pass0_used),
        "pass2_merged": len(pass2_used),
        "pass3_spliced": len(pass3_used),
        "pass4_merged": len(pass4_used),
    }
