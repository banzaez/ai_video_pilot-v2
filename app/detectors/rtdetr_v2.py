"""RT-DETRv2 (Hugging Face transformers) для Stage 1."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from app.parallel_tracker import FrameReaderThread, nms_detections, open_video_capture, reader_queue_size
from app.progress import make_pbar

logger = logging.getLogger(__name__)

_CACHE: dict[str, tuple[Any, Any]] = {}
_POS_EMBED_PATCHED = False


def _require_transformers():
    try:
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
    except ImportError as exc:
        raise ImportError(
            "RT-DETRv2 требует transformers. Установите: pip install 'transformers>=5.15.0,<6'"
        ) from exc
    patch_rtdetr_v2_mps_pos_embed()
    return RTDetrImageProcessor, RTDetrV2ForObjectDetection


def patch_rtdetr_v2_mps_pos_embed() -> None:
    """HF считает pos-embed в float64 на device модели; MPS float64 не умеет → CPU, потом .to(mps)."""
    global _POS_EMBED_PATCHED
    if _POS_EMBED_PATCHED:
        return
    from transformers.models.rt_detr_v2 import modeling_rt_detr_v2 as mod

    orig = mod.build_2d_sinusoidal_position_embedding
    if getattr(orig, "_mps_float64_patched", False):
        _POS_EMBED_PATCHED = True
        return

    def _safe(*args: Any, **kwargs: Any):
        device = kwargs.get("device")
        if device is None:
            return orig(*args, **kwargs)
        try:
            dev = torch.device(device)
        except (TypeError, RuntimeError):
            return orig(*args, **kwargs)
        if dev.type != "mps":
            return orig(*args, **kwargs)
        patched = dict(kwargs)
        patched["device"] = torch.device("cpu")
        out = orig(*args, **patched)
        dtype = patched.get("dtype", out.dtype)
        return out.to(device=dev, dtype=dtype)

    _safe._mps_float64_patched = True  # type: ignore[attr-defined]
    mod.build_2d_sinusoidal_position_embedding = _safe
    _POS_EMBED_PATCHED = True
    logger.info("RT-DETRv2: pos-embed float64 на CPU (MPS не поддерживает float64)")


def _resolve_torch_device(device: str) -> torch.device:
    dev = (device or "auto").lower()
    if dev == "auto":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if dev == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        logger.warning("MPS недоступен → CPU")
        return torch.device("cpu")
    return torch.device(dev)


def _load_model(model_id: str, device: str, *, quantize: int | None):
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

    patch_rtdetr_v2_mps_pos_embed()
    key = f"{model_id}|{device}|{quantize}"
    if key in _CACHE:
        return _CACHE[key]

    torch_device = _resolve_torch_device(device)
    logger.info("STAGE load: rtdetr_v2 ← %s (device=%s)", model_id, torch_device)
    processor = RTDetrImageProcessor.from_pretrained(model_id)
    model = RTDetrV2ForObjectDetection.from_pretrained(model_id)
    model.to(torch_device)
    model.eval()
    if quantize == 16 and torch_device.type in ("cuda", "mps"):
        model = model.half()
    _CACHE[key] = (processor, model)
    return processor, model


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _boxes_from_hf_result(
    result: dict[str, Any],
    *,
    classes: list[int],
    conf: float,
    nms_iou: float,
) -> list[dict[str, Any]]:
    allowed = set(int(c) for c in classes)
    frame_boxes: list[dict[str, Any]] = []
    scores = result["scores"]
    labels = result["labels"]
    boxes = result["boxes"]
    for score, label_id, box in zip(scores, labels, boxes):
        label = int(label_id.item())
        if allowed and label not in allowed:
            continue
        conf_v = float(score.item())
        if conf_v < conf:
            continue
        xyxy = [int(round(v)) for v in box.tolist()]
        frame_boxes.append({"bbox": xyxy, "confidence": round(conf_v, 4)})
    return nms_detections(frame_boxes, nms_iou)


def detect_video_frames(
    video_path: str,
    *,
    model_path: str,
    conf: float,
    classes: list[int],
    device: str,
    batch_size: int,
    total_frames: int,
    imgsz: int = 640,
    detect_every_n: int = 1,
    nms_iou: float = 0.5,
    quantize: int | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Детекция RT-DETRv2 по видео. Один процесс — HF-модель тяжёлая для fork."""
    _require_transformers()
    detect_every_n = max(1, int(detect_every_n))
    infer_batch = max(1, int(batch_size))
    processor, model = _load_model(model_path, device, quantize=quantize)
    torch_device = next(model.parameters()).device
    use_half = quantize == 16 and torch_device.type in ("cuda", "mps")

    cap = open_video_capture(video_path)
    total = max(1, int(total_frames or 0) or int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    chunk: dict[int, list[dict[str, Any]]] = {}
    pbar = make_pbar(total=total, desc="[STAGE 1: RTDETR_V2]", unit="frame", leave=True)
    boxes_total = 0
    skipped_per_batch = detect_every_n - 1 if detect_every_n > 1 else 0

    reader = FrameReaderThread(
        cap,
        start_frame=0,
        end_frame=total,
        detect_every_n=detect_every_n,
        queue_size=reader_queue_size(infer_batch),
    )

    try:
        with torch.inference_mode():
            batch_frames: list[np.ndarray] = []
            batch_indices: list[int] = []
            batch_sizes: list[tuple[int, int]] = []

            def flush_batch() -> None:
                nonlocal boxes_total
                if not batch_frames:
                    return
                images = [_frame_to_pil(f) for f in batch_frames]
                inputs = processor(images=images, return_tensors="pt")
                inputs = {k: v.to(torch_device) for k, v in inputs.items()}
                if use_half and "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].half()
                outputs = model(**inputs)
                target_sizes = torch.tensor(batch_sizes, device=torch_device)
                results = processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=float(conf),
                )
                for idx, result, (h, w) in zip(batch_indices, results, batch_sizes):
                    frame_boxes = _boxes_from_hf_result(
                        result,
                        classes=classes,
                        conf=conf,
                        nms_iou=nms_iou,
                    )
                    chunk[idx] = frame_boxes
                    boxes_total += len(frame_boxes)
                    pbar.set_postfix(boxes=boxes_total, batch=len(batch_frames))
                    pbar.update(1 + skipped_per_batch)
                batch_frames.clear()
                batch_indices.clear()
                batch_sizes.clear()

            for frame_idx, frame in reader:
                batch_frames.append(frame)
                batch_indices.append(frame_idx)
                batch_sizes.append((frame.shape[0], frame.shape[1]))
                if len(batch_frames) >= infer_batch:
                    flush_batch()
            flush_batch()
    finally:
        reader.drain()
        pbar.close()
        cap.release()

    return chunk
