"""Вырезка ROI для tracklet ReID."""

from app.crops.geometry import bbox_in_crop, crop_person, crop_roi_xyxy

__all__ = [
    "bbox_in_crop",
    "crop_person",
    "crop_roi_xyxy",
]
