"""Пакет сервиса поз человека (PoseService)."""

from app.pose.pose_service import PoseService, get_pose_service
from app.pose.types import PoseResult

__all__ = [
    "PoseResult",
    "PoseService",
    "get_pose_service",
]
