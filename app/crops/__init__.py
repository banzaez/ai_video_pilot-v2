"""Вырезка ROI для tracklet ReID."""

from app.crops.geometry import bbox_in_crop, crop_person, crop_roi_xyxy
from app.crops.track_best_frames import (
    ScoredTrackFrame,
    TrackBestFramesPicker,
    TrackFrameCandidate,
    extract_face_box_from_pose,
    extract_face_crop_from_person,
)

__all__ = [
    "bbox_in_crop",
    "crop_person",
    "crop_roi_xyxy",
    "extract_face_box_from_pose",
    "extract_face_crop_from_person",
    "ScoredTrackFrame",
    "TrackBestFramesPicker",
    "TrackFrameCandidate",
]
