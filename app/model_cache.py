"""Кэш YOLO-моделей и выбор runtime-весов (CoreML / ONNX / .pt)."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

_CACHE: "ModelCache | None" = None
DETECT_MODELS_DIR = "data/models/detect"
REID_MODELS_DIR = "data/models/reid"


def join_models_path(models_dir: str, name: str) -> str:
    """Собрать путь: basename → models_dir/name; полный/существующий — как есть."""
    raw = (name or "").strip()
    if not raw:
        return raw
    if os.path.isfile(raw):
        return os.path.abspath(raw)
    if os.path.isabs(raw):
        return raw
    if "/" in raw.replace("\\", "/"):
        return raw
    base = (models_dir or DETECT_MODELS_DIR).strip().rstrip("/\\")
    return os.path.join(base, raw) if base else raw


def resolve_pt_path(path: str, *, models_dir: str | None = None) -> str:
    """Найти .pt (или runtime) в models_dir, data/models/detect/, data/models/ или как указано."""
    raw = (path or "").strip()
    if not raw:
        return raw
    if os.path.isfile(raw):
        return os.path.abspath(raw)
    base_dir = models_dir or DETECT_MODELS_DIR
    base = os.path.basename(raw)
    for candidate in (
        os.path.join(base_dir, base),
        os.path.join(base_dir, raw),
        os.path.join(DETECT_MODELS_DIR, base),
        os.path.join(DETECT_MODELS_DIR, raw),
        os.path.join("data", "models", base),
        os.path.join("data", "models", raw),
        base,
    ):
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return join_models_path(base_dir, raw)


def resolve_runtime_weights(path: str) -> tuple[str, str]:
    """Вернуть (путь_к_весам, формат): coreml | onnx | pt.

    YAML указывает .pt; если рядом есть .mlpackage (macOS) или .onnx — берём их.
    """
    resolved = resolve_pt_path(path)
    stem, ext = os.path.splitext(resolved)
    # Уже передан runtime-путь
    if resolved.endswith(".mlpackage") or os.path.isdir(resolved) and resolved.endswith(".mlpackage"):
        return resolved, "coreml"
    if ext.lower() == ".onnx":
        return resolved, "onnx"

    mlpackage = stem + ".mlpackage"
    onnx = stem + ".onnx"
    if sys.platform == "darwin" and (os.path.isdir(mlpackage) or os.path.isfile(mlpackage)):
        return mlpackage, "coreml"
    if os.path.isfile(onnx):
        return onnx, "onnx"
    return resolved, "pt"


_WARNED_COREML_BATCH = False


def predict_batch_size(path: str, requested: int) -> int:
    """Эффективный batch для predict.

    Статический CoreML (экспорт без dynamic=True) падает на batch>1
    (IndexError в ultralytics stream_inference). Пока не проверен dynamic-пакет —
    всегда clamp к 1.
    """
    global _WARNED_COREML_BATCH
    req = max(1, int(requested))
    _, fmt = resolve_runtime_weights(path)
    if fmt == "coreml" and req > 1:
        if not _WARNED_COREML_BATCH:
            _WARNED_COREML_BATCH = True
            logger.warning(
                "CoreML: batch>1 не поддерживается → batch=1 (медленнее батча на .pt; это ожидаемо)"
            )
        return 1
    return req

def load_detector_model(path: str, *, backend: str = "yolo"):
    """YOLO или RT-DETR (Ultralytics) для Stage 1."""
    runtime, fmt = resolve_runtime_weights(path)
    if backend == "rtdetr":
        from ultralytics import RTDETR

        logger.info("STAGE load: rtdetr ← %s (%s)", runtime, fmt)
        return RTDETR(runtime)
    from ultralytics import YOLO

    logger.info("STAGE load: yolo ← %s (%s)", runtime, fmt)
    return YOLO(runtime, task="detect")


class ModelCache:
    """Кэш YOLO в памяти процесса (между роликами одного прогона)."""

    def __init__(self) -> None:
        self._models: dict[tuple[str, str], Any] = {}

    def get_yolo(self, path: str, *, kind: str = "detect") -> Any:
        from ultralytics import YOLO

        runtime, fmt = resolve_runtime_weights(path)
        key = (runtime, kind)
        hit = key in self._models
        if not hit:
            logger.info("STAGE load: %s ← %s (%s)", kind, runtime, fmt)
            task = "pose" if kind == "pose" else "detect"
            self._models[key] = YOLO(runtime, task=task)
        else:
            logger.info("STAGE load: %s reuse %s (%s)", kind, runtime, fmt)
        return self._models[key]

    def get_detector(self, path: str, *, backend: str = "yolo", kind: str = "detect") -> Any:
        if kind != "detect":
            return self.get_yolo(path, kind=kind)
        runtime, fmt = resolve_runtime_weights(path)
        key = (runtime, kind, backend)
        if key not in self._models:
            self._models[key] = load_detector_model(path, backend=backend)
        else:
            logger.info("STAGE load: %s reuse %s (%s)", backend, runtime, fmt)
        return self._models[key]

    def clear(self) -> None:
        self._models.clear()


def get_model_cache() -> ModelCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = ModelCache()
    return _CACHE


def set_model_cache(cache: ModelCache | None) -> None:
    global _CACHE
    _CACHE = cache
