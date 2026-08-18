"""Pose / feet / проекции на карту (пакет оставлен как app.global_id)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "run_feet",
    "run_pose",
]


def __getattr__(name: str) -> Any:
    if name == "run_feet":
        from app.global_id.stage_feet import run_feet

        return run_feet
    if name == "run_pose":
        from app.global_id.stage_pose import run_pose

        return run_pose
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
