"""SOLIDER ReID backend helpers (preprocess + lazy model load)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TRANSFORMER = "swin_small_patch4_window7_224"
DEFAULT_IMAGE_SIZE = (384, 128)
PIXEL_MEAN = (0.5, 0.5, 0.5)
PIXEL_STD = (0.5, 0.5, 0.5)


def load_solider_model(
    weights: str,
    *,
    device: str = "cpu",
    semantic_weight: float = 0.2,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    transformer_type: str = DEFAULT_TRANSFORMER,
) -> Any:
    from app.third_party.solider_reid import build_solider_reid

    logger.info(
        "SOLIDER: loading %s (%s, semantic_weight=%s) on %s",
        weights,
        transformer_type,
        semantic_weight,
        device,
    )
    return build_solider_reid(
        weights,
        transformer_type=transformer_type,
        image_size=image_size,
        semantic_weight=semantic_weight,
        device=device,
    )


def _normalize_nchw(arr_hwc_rgb: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    import cv2

    h, w = int(image_size[0]), int(image_size[1])
    resized = cv2.resize(arr_hwc_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    mean = np.asarray(PIXEL_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(PIXEL_STD, dtype=np.float32).reshape(3, 1, 1)
    return (arr - mean) / std


def preprocess_paths(
    paths: list[str],
    image_size: tuple[int, int],
    device: str,
) -> Any:
    """Load crop images → NCHW float tensor normalized like SOLIDER-REID."""
    import torch
    from PIL import Image

    batch: list[np.ndarray] = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        batch.append(_normalize_nchw(np.asarray(img), image_size))
    x = np.stack(batch, axis=0)
    return torch.from_numpy(x).to(device)


def preprocess_arrays(
    images: list[np.ndarray],
    image_size: tuple[int, int],
    device: str,
) -> Any:
    """BGR numpy (H,W,3) → NCHW float tensor, как SOLIDER-REID."""
    import torch

    batch: list[np.ndarray] = []
    for img in images:
        rgb = img[:, :, ::-1] if img.ndim == 3 and img.shape[2] == 3 else img
        batch.append(_normalize_nchw(np.asarray(rgb), image_size))
    x = np.stack(batch, axis=0)
    return torch.from_numpy(x).to(device)


def embed_paths(
    model: Any,
    paths: list[str],
    *,
    device: str,
    image_size: tuple[int, int],
    batch_size: int = 32,
) -> np.ndarray:
    import torch

    from app.progress import make_pbar

    if not paths:
        return np.zeros((0, 0), dtype=np.float32)
    outs: list[np.ndarray] = []
    bs = max(1, int(batch_size))
    model.eval()
    pbar = make_pbar(
        total=len(paths),
        desc="[ReID solider]",
        unit="crop",
    )
    try:
        with torch.inference_mode():
            for i in range(0, len(paths), bs):
                chunk = paths[i : i + bs]
                x = preprocess_paths(chunk, image_size, device)
                feat = model(x)
                if isinstance(feat, (tuple, list)):
                    feat = feat[0]
                outs.append(feat.detach().float().cpu().numpy())
                pbar.update(len(chunk))
    finally:
        pbar.close()
    return np.concatenate(outs, axis=0).astype(np.float32)


def embed_arrays(
    model: Any,
    images: list[np.ndarray],
    *,
    device: str,
    image_size: tuple[int, int],
    batch_size: int = 8,
) -> np.ndarray:
    import torch

    from app.progress import make_pbar

    if not images:
        return np.zeros((0, 0), dtype=np.float32)
    outs: list[np.ndarray] = []
    bs = max(1, int(batch_size))
    model.eval()
    pbar = make_pbar(
        total=len(images),
        desc="[ReID solider]",
        unit="crop",
    )
    try:
        with torch.inference_mode():
            for i in range(0, len(images), bs):
                chunk = images[i : i + bs]
                x = preprocess_arrays(chunk, image_size, device)
                feat = model(x)
                if isinstance(feat, (tuple, list)):
                    feat = feat[0]
                outs.append(feat.detach().float().cpu().numpy())
                pbar.update(len(chunk))
    finally:
        pbar.close()
    return np.concatenate(outs, axis=0).astype(np.float32)
