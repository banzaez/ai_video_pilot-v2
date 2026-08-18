"""Чтение body_calib и pick k — без импорта nodes/feet."""

from __future__ import annotations

import json
import os
from typing import Any

# px-отношение (плечи→ноги) / (плечи→бёдра); ~1.7 м в типичной перспективе, без метров на полу.
FALLBACK_K = 2.0
MIN_SAMPLES = 30


def body_calib_trusted(calib: dict[str, Any] | None) -> bool:
    if not calib:
        return False
    n = calib.get("n_samples")
    return isinstance(n, (int, float)) and int(n) >= MIN_SAMPLES


def pick_k_for_shoulder(y_shoulder: float, calib: dict[str, Any] | None) -> float:
    if not body_calib_trusted(calib):
        if calib and isinstance(calib.get("fallback_k"), (int, float)):
            return float(calib["fallback_k"])
        return FALLBACK_K
    assert calib is not None
    bands = calib.get("bands")
    if isinstance(bands, dict) and bands:
        near = bands.get("near")
        mid = bands.get("mid")
        far = bands.get("far")
        if isinstance(near, dict) and y_shoulder <= float(near.get("y_max", 0)):
            return float(near.get("k", FALLBACK_K))
        if isinstance(mid, dict) and y_shoulder <= float(mid.get("y_max", 0)):
            return float(mid.get("k", FALLBACK_K))
        if isinstance(far, dict):
            return float(far.get("k", FALLBACK_K))
    k = calib.get("k_shoulder_to_feet")
    if isinstance(k, (int, float)) and k > 0:
        return float(k)
    fb = calib.get("fallback_k")
    return float(fb) if isinstance(fb, (int, float)) else FALLBACK_K


def load_body_calib(maps_dir: str, camera_key: str) -> dict[str, Any] | None:
    path = os.path.join(maps_dir, "cameras", f"{camera_key}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    bc = data.get("body_calib") if isinstance(data, dict) else None
    return bc if isinstance(bc, dict) else None
