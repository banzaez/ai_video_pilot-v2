"""Параметры трекеров: inline в config.yaml (`tracklet_pipeline.tracker`)."""

from __future__ import annotations

from typing import Any

from app.config.settings import TRACKER_CHOICES, Settings

# Ключи, которые не передаём в IterableSimpleNamespace трекера
_SKIP_KEYS = frozenset({"type", "config"})


def _normalize_tracker_params(
    tracker_type: str,
    raw: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _SKIP_KEYS or value is None:
            continue
        params[str(key)] = value
    params["tracker_type"] = tracker_type
    if "gmc_method" in params:
        params["gmc_method"] = "none"
    params["_config_path"] = source
    return params


def load_tracker_params(
    tracker_type: str,
    *,
    inline_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Параметры трекера из inline-блока YAML. Без файла — только тип (дефолты Ultralytics)."""
    tracker_type = str(tracker_type).lower().strip()
    if tracker_type not in TRACKER_CHOICES:
        raise ValueError(
            f"Неизвестный tracker '{tracker_type}'. Доступны: {', '.join(TRACKER_CHOICES)}"
        )
    return _normalize_tracker_params(
        tracker_type,
        inline_params or {},
        source=f"inline:{tracker_type}",
    )


def tracker_params_dict(settings: Settings) -> dict[str, Any]:
    """Overrides для create_tracker (без служебных ключей)."""
    params: dict[str, Any] = {"tracker_type": settings.tracker_type}
    for key, value in (settings.tracker_params or {}).items():
        if key in _SKIP_KEYS or key.startswith("_") or key == "tracker_type":
            continue
        if value is not None:
            params[key] = value
    if "gmc_method" in params:
        params["gmc_method"] = "none"
    return params


def tracklet_tracker_params_dict(settings: Settings) -> dict[str, Any]:
    """Overrides для локального ByteTrack (Stage 2a)."""
    params: dict[str, Any] = {"tracker_type": settings.tracklet_local_tracker}
    for key, value in (settings.tracklet_local_params or {}).items():
        if key in _SKIP_KEYS or key.startswith("_") or key == "tracker_type":
            continue
        if value is not None:
            params[key] = value
    if "gmc_method" in params:
        params["gmc_method"] = "none"
    return params
