"""Вырезка ROI для tracklet ReID."""

from app.crops.filters import (
    OutlierFilterConfig,
    filter_color_outliers,
    filter_kinematic_outliers,
    filter_track_outlier_candidates,
    trim_edge_items,
)
from app.crops.geometry import bbox_in_crop, crop_person, crop_roi_xyxy
from app.crops.track_best_frames import (
    ScoredTrackFrame,
    TrackBestFramesPicker,
    TrackFrameCandidate,
    extract_face_box_from_pose,
    extract_face_crop_from_person,
)

__all__ = [
    "OutlierFilterConfig",
    "bbox_in_crop",
    "crop_person",
    "crop_roi_xyxy",
    "extract_face_box_from_pose",
    "extract_face_crop_from_person",
    "filter_color_outliers",
    "filter_kinematic_outliers",
    "filter_track_outlier_candidates",
    "trim_edge_items",
    "ScoredTrackFrame",
    "TrackBestFramesPicker",
    "TrackFrameCandidate",
]

