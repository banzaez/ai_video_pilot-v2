"""Tracklet pipeline: 2a tracklets → 2b reid → 2c link → track."""

from __future__ import annotations

from typing import Any

__all__ = [
    "run_tracklet_link",
    "run_tracklet_reid",
    "run_tracklets",
]


def __getattr__(name: str) -> Any:
    if name == "run_tracklet_link":
        from app.tracklet.stage_link import run_tracklet_link

        return run_tracklet_link
    if name == "run_tracklet_reid":
        from app.tracklet.stage_reid import run_tracklet_reid

        return run_tracklet_reid
    if name == "run_tracklets":
        from app.tracklet.stage_tracklets import run_tracklets

        return run_tracklets
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
