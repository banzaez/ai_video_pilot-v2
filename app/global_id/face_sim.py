"""Общий скоринг лиц InsightFace (camera_link / day_link)."""

from __future__ import annotations

import numpy as np


def max_face_similarity_weighted(
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
