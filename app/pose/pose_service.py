"""Сервис работы с моделью YOLO-Pose (одиночные кадры/кропы и батчи)."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from app.config import Settings
from app.model_cache import get_model_cache, predict_batch_size, resolve_pt_path
from app.pose.types import PoseResult, select_pose_by_completeness
from app.progress import make_pbar

logger = logging.getLogger(__name__)

_POSE_SERVICE_CACHE: dict[str, PoseService] = {}


def _parse_pose_results(raw_results: Any) -> list[list[PoseResult]]:
    """Парсит вывод Ultralytics Results в структурированный PoseResult."""
    if not isinstance(raw_results, (list, tuple)):
        raw_results = [raw_results]

    out_batches: list[list[PoseResult]] = []
    for res in raw_results:
        items: list[PoseResult] = []
        boxes = getattr(res, "boxes", None)
        kpts = getattr(res, "keypoints", None)
        if boxes is not None and kpts is not None:
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            kxy = kpts.xy.cpu().numpy() if hasattr(kpts.xy, "cpu") else np.asarray(kpts.xy)
            if getattr(kpts, "conf", None) is not None:
                kcf = kpts.conf.cpu().numpy() if hasattr(kpts.conf, "cpu") else np.asarray(kpts.conf)
            else:
                kcf = np.ones((len(kxy), kxy.shape[1] if len(kxy) else 0), dtype=np.float32)

            for i in range(len(xyxy)):
                box = [round(float(v), 1) for v in xyxy[i][:4]]
                xy = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in kxy[i]]
                cf = [round(float(c), 3) for c in kcf[i]]
                items.append(
                    PoseResult(
                        bbox=box,
                        confidence=round(float(confs[i]), 4),
                        kxy=xy,
                        kcf=cf,
                    )
                )
        out_batches.append(items)
    return out_batches


class PoseService:
    """Универсальный сервис оценки поз человека."""

    def __init__(
        self,
        *,
        model_name: str = "yolo26s-pose.pt",
        models_dir: str = "models",
        conf: float = 0.25,
        imgsz: int = 640,
        batch_size: int = 16,
        device: str | None = None,
        quantize: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.models_dir = models_dir
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.batch_size = max(1, int(batch_size))
        self.device = str(device or "").strip() or None
        self.quantize = quantize
        self._model = None
        self._resolved_model_path: str | None = None

    @property
    def model_path(self) -> str:
        if self._resolved_model_path is None:
            self._resolved_model_path = resolve_pt_path(self.model_name, models_dir=self.models_dir)
        return self._resolved_model_path

    def _ensure_model(self) -> Any:
        if self._model is None:
            cache = get_model_cache()
            self._model = cache.get_yolo(self.model_path, kind="pose")
        return self._model

    def effective_batch_size(self, requested: int | None = None) -> int:
        """Возвращает размер батча с учетом ограничений модели/бэкенда (напр., CoreML batch=1)."""
        base_batch = requested if requested is not None else self.batch_size
        return predict_batch_size(self.model_path, base_batch)

    def _build_predict_kwargs(self, conf: float | None = None) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "conf": float(conf if conf is not None else self.conf),
            "imgsz": self.imgsz,
            "verbose": False,
            "classes": [0],
        }
        if self.device:
            kw["device"] = self.device
        if self.quantize is not None:
            kw["quantize"] = self.quantize
        return kw

    def predict_single(
        self,
        image: np.ndarray,
        *,
        conf: float | None = None,
    ) -> list[PoseResult]:
        """Инференс позы для одного изображения/кропа."""
        if image is None or image.size == 0:
            return []
        model = self._ensure_model()
        kw = self._build_predict_kwargs(conf)
        raw_res = model.predict(source=image, **kw)
        parsed = _parse_pose_results(raw_res)
        return parsed[0] if parsed else []

    def predict_batch(
        self,
        images: Sequence[np.ndarray],
        *,
        conf: float | None = None,
        batch_size: int | None = None,
        show_pbar: bool = False,
        pbar_desc: str = "[Pose batch]",
    ) -> list[list[PoseResult]]:
        """
        Батчевый инференс списка изображений/кадров с автоматическим чанкованием.
        """
        if not images:
            return []
        model = self._ensure_model()
        kw = self._build_predict_kwargs(conf)
        eff_batch = self.effective_batch_size(batch_size)

        out: list[list[PoseResult]] = []
        pbar = (
            make_pbar(total=len(images), desc=pbar_desc, unit="crop")
            if show_pbar
            else None
        )
        try:
            for i in range(0, len(images), eff_batch):
                chunk = list(images[i : i + eff_batch])
                raw_res = model.predict(source=chunk, **kw)
                parsed = _parse_pose_results(raw_res)
                out.extend(parsed)
                if pbar is not None:
                    pbar.update(len(chunk))
        finally:
            if pbar is not None:
                pbar.close()
        return out

    def pose_faces_for_bboxes(
        self,
        frame_img: np.ndarray,
        bboxes: Sequence[Sequence[float]],
        *,
        kpt_min: float = 0.25,
    ) -> list[tuple[PoseResult | None, float]]:
        """Самая полная поза на bbox, затем face_confidence (conf детектора = self.conf)."""
        if frame_img is None or frame_img.size == 0 or not bboxes:
            return [(None, 0.0)] * len(bboxes)
        poses = self.predict_crops_from_frame(frame_img, bboxes, kpt_min=kpt_min)
        out: list[tuple[PoseResult | None, float]] = []
        for pose in poses:
            if pose is None:
                out.append((None, 0.0))
            else:
                out.append((pose, pose.face_confidence(min_conf=kpt_min)))
        return out

    def pose_face_for_bbox(
        self,
        frame_img: np.ndarray,
        bbox: Sequence[float],
        *,
        kpt_min: float = 0.25,
    ) -> tuple[PoseResult | None, float]:
        """Поза и оценка пригодности лица для одного tracking bbox."""
        rows = self.pose_faces_for_bboxes(frame_img, [bbox], kpt_min=kpt_min)
        return rows[0] if rows else (None, 0.0)

    def predict_crops_from_frame(
        self,
        frame_img: np.ndarray,
        bboxes: Sequence[Sequence[float]],
        *,
        conf: float | None = None,
        padding: float = 0.05,
        batch_size: int | None = None,
        kpt_min: float = 0.25,
    ) -> list[PoseResult | None]:
        """
        Вырезает кропы для списка BBox с кадра, прогоняет через модель и
        возвращает PoseResult с абсолютными координатами на исходном кадре.
        """
        if frame_img is None or frame_img.size == 0 or not bboxes:
            return [None] * len(bboxes)

        img_h, img_w = frame_img.shape[:2]
        valid_crops: list[np.ndarray] = []
        crop_offsets: list[tuple[int, int]] = []
        bbox_map: list[int | None] = []

        for bbox in bboxes:
            if not bbox or len(bbox) < 4:
                bbox_map.append(None)
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            bw = x2 - x1
            bh = y2 - y1
            px = bw * padding
            py = bh * padding
            ix1 = max(0, int(round(x1 - px)))
            iy1 = max(0, int(round(y1 - py)))
            ix2 = min(img_w, int(round(x2 + px)))
            iy2 = min(img_h, int(round(y2 + py)))
            if ix2 <= ix1 or iy2 <= iy1:
                bbox_map.append(None)
                continue
            crop = frame_img[iy1:iy2, ix1:ix2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                bbox_map.append(None)
                continue

            crop_idx = len(valid_crops)
            valid_crops.append(crop)
            crop_offsets.append((ix1, iy1))
            bbox_map.append(crop_idx)

        if not valid_crops:
            return [None] * len(bboxes)

        batch_results = self.predict_batch(valid_crops, conf=conf, batch_size=batch_size)

        out: list[PoseResult | None] = []
        for bbox_idx, mapped_idx in enumerate(bbox_map):
            if mapped_idx is None:
                out.append(None)
                continue
            if mapped_idx >= len(batch_results):
                out.append(None)
                continue
            items = batch_results[mapped_idx]
            if not items:
                out.append(None)
                continue
            orig_bbox = [float(v) for v in bboxes[bbox_idx][:4]]
            ox, oy = crop_offsets[mapped_idx]
            local_orig = [orig_bbox[0] - ox, orig_bbox[1] - oy, orig_bbox[2] - ox, orig_bbox[3] - oy]
            best = select_pose_by_completeness(
                items, local_orig, kpt_min=kpt_min, require_min_iou=False
            )
            if best is None:
                out.append(None)
                continue
            out.append(best.with_offset(ox, oy))

        return out


def get_pose_service(settings: Settings | None = None) -> PoseService:
    """Фабрика PoseService с кэшированием."""
    if settings is None:
        key = "default"
        if key not in _POSE_SERVICE_CACHE:
            _POSE_SERVICE_CACHE[key] = PoseService()
        return _POSE_SERVICE_CACHE[key]

    key = f"{settings.pose_model}_{settings.models_dir}_{settings.device}_{settings.batch_size}_{settings.pose_conf}_{settings.imgsz}_{settings.quantize}"
    if key not in _POSE_SERVICE_CACHE:
        quantize = None if settings.quantize in (None, 32, 0) else int(settings.quantize)
        _POSE_SERVICE_CACHE[key] = PoseService(
            model_name=settings.pose_model,
            models_dir=settings.models_dir,
            conf=float(settings.pose_conf),
            imgsz=int(settings.imgsz),
            batch_size=int(settings.batch_size),
            device=settings.device,
            quantize=quantize,
        )
    return _POSE_SERVICE_CACHE[key]
