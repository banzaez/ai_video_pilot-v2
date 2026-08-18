"""Модуль работы с лицами InsightFace."""

from app.face.insight_extractor import (
    extract_faces_for_groups,
    face_models_for_settings,
    get_face_analysis,
    l2_normalize,
)

__all__ = [
    "extract_faces_for_groups",
    "face_models_for_settings",
    "get_face_analysis",
    "l2_normalize",
]
