"""ReID-эмбеддинги треклетов для global link."""

from __future__ import annotations

from typing import Any

import numpy as np


def _normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        return v
    return v / n


def crop_vectors(
    crop_paths: list[str],
    cache: dict[str, np.ndarray],
) -> np.ndarray | None:
    """Матрица (k, dim) L2-нормированных кропов треклета."""
    vecs: list[np.ndarray] = []
    for path in crop_paths:
        if path and path in cache:
            vecs.append(_normalize(cache[path]))
    if not vecs:
        return None
    return np.stack(vecs, axis=0)


def mean_tracklet_embedding(
    crop_paths: list[str],
    cache: dict[str, np.ndarray],
) -> np.ndarray | None:
    """L2-normalized mean по top_k кропам треклета."""
    mat = crop_vectors(crop_paths, cache)
    if mat is None:
        return None
    return _normalize(np.mean(mat, axis=0))


def _paths_for_record(rec: dict[str, Any]) -> list[str]:
    paths = list(rec.get("crop_paths") or [])
    if paths:
        return paths
    key = str(rec.get("embedding_key") or "")
    return [key] if key else []


def build_tracklet_embeddings(
    reid_records: list[dict[str, Any]],
    cache: dict[str, np.ndarray],
) -> dict[int, np.ndarray]:
    """tracklet_id → mean embedding (совместимость)."""
    out: dict[int, np.ndarray] = {}
    for rec in reid_records:
        tid = int(rec["tracklet_id"])
        emb = mean_tracklet_embedding(_paths_for_record(rec), cache)
        if emb is not None:
            out[tid] = emb
    return out


def build_tracklet_crop_embeddings(
    reid_records: list[dict[str, Any]],
    cache: dict[str, np.ndarray],
) -> dict[int, np.ndarray]:
    """tracklet_id → (k, dim) кропы для max pairwise cosine."""
    out: dict[int, np.ndarray] = {}
    for rec in reid_records:
        tid = int(rec["tracklet_id"])
        mat = crop_vectors(_paths_for_record(rec), cache)
        if mat is not None:
            out[tid] = mat
    return out
