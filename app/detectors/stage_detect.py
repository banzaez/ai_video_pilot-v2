"""Stage 1: YOLO, RT-DETR (Ultralytics) или RT-DETRv2 (Hugging Face)."""

from __future__ import annotations

import logging
from typing import Any

from app.config.settings import DETECTION_BACKEND_CHOICES, Settings
from app.parallel_tracker import default_workers, detect_video_frames

logger = logging.getLogger(__name__)


def detector_meta(settings: Settings) -> dict[str, Any]:
    return {
        "backend": settings.detection_backend,
        "path": settings.model_path,
        "classes": settings.classes,
        "imgsz": settings.imgsz,
        "batch_size": settings.batch_size,
        "quantize": settings.quantize,
        "device": settings.device,
    }


def detect_video_part(
    settings: Settings,
    video_path: str,
    total_frames: int,
) -> dict[int, list[dict[str, Any]]]:
    """Детекция по одному видеофрагменту. Ключ — локальный 0-based индекс кадра."""
    backend = settings.detection_backend
    if backend not in DETECTION_BACKEND_CHOICES:
        known = ", ".join(DETECTION_BACKEND_CHOICES)
        raise ValueError(f"Неизвестный detection.backend '{backend}'. Доступны: {known}")

    if backend == "rtdetr_v2":
        from app.detectors.rtdetr_v2 import detect_video_frames as detect_rtdetr_v2

        return detect_rtdetr_v2(
            video_path,
            model_path=settings.model_path,
            conf=settings.conf,
            classes=settings.classes,
            device=settings.device,
            batch_size=settings.batch_size,
            total_frames=total_frames,
            imgsz=settings.imgsz,
            detect_every_n=settings.detect_every_n,
            nms_iou=settings.nms_iou,
            quantize=settings.quantize,
        )

    workers = default_workers(settings.workers, settings.device)
    return detect_video_frames(
        video_path,
        model_path=settings.model_path,
        conf=settings.conf,
        classes=settings.classes,
        device=settings.device,
        batch_size=settings.batch_size,
        num_workers=workers,
        total_frames=total_frames,
        imgsz=settings.imgsz,
        detect_every_n=settings.detect_every_n,
        nms_iou=settings.nms_iou,
        quantize=settings.quantize,
        detector_backend=backend,
    )
