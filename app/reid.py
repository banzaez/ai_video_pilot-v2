"""ReID-эмбеддинги кропов. Backend: torchreid (OSNet) или solider (Swin).

Используются офлайн в tracklet_reid / link, в трекинг не входят.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "osnet_x1_0"
DEFAULT_WEIGHTS = "data/models/reid/osnet_x1_0_msmt17.pth"
DEFAULT_SOLIDER_WEIGHTS = "data/models/reid/solider_swin_base_msmt17.pth"
DEFAULT_SOLIDER_TRANSFORMER = "swin_base_patch4_window7_224"
IMAGE_SIZE = (256, 128)
SOLIDER_IMAGE_SIZE = (384, 128)

BACKEND_TORCHREID = "torchreid"
BACKEND_SOLIDER = "solider"
VALID_BACKENDS = (BACKEND_TORCHREID, BACKEND_SOLIDER)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def normalize_backend(backend: str | None) -> str:
    name = (backend or BACKEND_TORCHREID).strip().lower()
    if name not in VALID_BACKENDS:
        raise ValueError(f"unknown reid_backend '{backend}', expected {VALID_BACKENDS}")
    return name


def cache_filename(backend: str | None = BACKEND_TORCHREID) -> str:
    return "reid_solider.npz" if normalize_backend(backend) == BACKEND_SOLIDER else "reid.npz"


def cache_path_for(json_path: str, backend: str | None = BACKEND_TORCHREID) -> str:
    """JSON в папке видео → reid.npz / reid_solider.npz рядом."""
    return os.path.join(os.path.dirname(json_path) or ".", cache_filename(backend))


def load_cache(path: str) -> dict[str, np.ndarray]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception as exc:  # повреждённый кэш не должен ронять стадию
        logger.warning("ReID: кэш не прочитан (%s), считаем заново", exc)
        return {}


def save_cache(path: str, data: dict[str, np.ndarray]) -> None:
    if not path or not data:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        np.savez_compressed(path, **data)
    except Exception as exc:
        logger.warning("ReID: кэш не сохранён (%s)", exc)


class ReidExtractor:
    """Ленивая обёртка: torchreid FeatureExtractor или SOLIDER Swin."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        weights: str = DEFAULT_WEIGHTS,
        device: str = "cpu",
        backend: str = BACKEND_TORCHREID,
        solider_weights: str | None = None,
        solider_semantic_weight: float = 0.2,
        solider_image_size: Sequence[int] | None = None,
        solider_transformer: str = DEFAULT_SOLIDER_TRANSFORMER,
    ) -> None:
        self.backend = normalize_backend(backend)
        self.model_name = model_name
        self.weights = weights
        self.device = "cpu" if device == "auto" else device
        self.solider_weights = solider_weights or DEFAULT_SOLIDER_WEIGHTS
        self.solider_semantic_weight = float(solider_semantic_weight)
        size = solider_image_size or SOLIDER_IMAGE_SIZE
        self.solider_image_size = (int(size[0]), int(size[1]))
        self.solider_transformer = solider_transformer or DEFAULT_SOLIDER_TRANSFORMER
        self._extractor: Any = None
        self.model_id = (
            f"solider:{self.solider_transformer}"
            if self.backend == BACKEND_SOLIDER
            else f"torchreid:{self.model_name}"
        )

    @property
    def cache_name(self) -> str:
        return cache_filename(self.backend)

    def available(self) -> tuple[bool, str]:
        if self.backend == BACKEND_SOLIDER:
            if not self.solider_weights or not os.path.isfile(self.solider_weights):
                return False, f"нет весов SOLIDER {self.solider_weights}"
            try:
                import torch  # noqa: F401
                from app.third_party.solider_reid import build_solider_reid  # noqa: F401
            except Exception as exc:
                return False, f"SOLIDER недоступен ({exc})"
            return True, ""

        if self.weights and not os.path.isfile(self.weights):
            return False, f"нет весов {self.weights}"
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Cython evaluation*",
                    category=UserWarning,
                    module=r"torchreid\.reid\.metrics\.rank",
                )
                import torchreid  # noqa: F401
        except Exception as exc:
            return False, f"нет torchreid ({exc})"
        return True, ""

    def _ensure(self) -> Any:
        if self._extractor is not None:
            return self._extractor
        if self.backend == BACKEND_SOLIDER:
            from app.reid_solider import load_solider_model

            self._extractor = load_solider_model(
                self.solider_weights,
                device=self.device,
                semantic_weight=self.solider_semantic_weight,
                image_size=self.solider_image_size,
                transformer_type=self.solider_transformer,
            )
            return self._extractor

        from torchreid.reid.utils import FeatureExtractor

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Cython evaluation*",
                category=UserWarning,
                module=r"torchreid\.reid\.metrics\.rank",
            )
            self._extractor = FeatureExtractor(
                model_name=self.model_name,
                model_path=self.weights or "",
                device=self.device,
                image_size=IMAGE_SIZE,
                verbose=False,
            )
        return self._extractor

    def _embed_batched(self, items: list[Any], *, batch_size: int) -> np.ndarray:
        if not items:
            return np.zeros((0, 512), dtype=np.float32)
        from app.progress import make_pbar

        extractor = self._ensure()
        outs: list[np.ndarray] = []
        bs = max(1, int(batch_size))
        pbar = make_pbar(
            total=len(items),
            desc=f"[ReID {self.model_id}]",
            unit="crop",
        )
        try:
            for i in range(0, len(items), bs):
                chunk = items[i : i + bs]
                feats = extractor(chunk).cpu().numpy().astype(np.float32)
                outs.append(feats)
                pbar.update(len(chunk))
        finally:
            pbar.close()
        return l2_normalize(np.concatenate(outs, axis=0))

    def embed_arrays(self, images: list[np.ndarray], *, batch_size: int = 32) -> np.ndarray:
        """L2-нормированные векторы для BGR numpy (H,W,3) без записи на диск."""
        if not images:
            return np.zeros((0, 512), dtype=np.float32)
        if self.backend == BACKEND_SOLIDER:
            from app.reid_solider import embed_arrays as solider_embed_arrays

            model = self._ensure()
            feats = solider_embed_arrays(
                model,
                images,
                device=self.device,
                image_size=self.solider_image_size,
                batch_size=batch_size,
            )
            return l2_normalize(feats)
        return self._embed_batched(images, batch_size=batch_size)

    def embed(self, paths: list[str], *, batch_size: int = 32) -> np.ndarray:
        """L2-нормированные векторы для списка путей к кропам."""
        if not paths:
            return np.zeros((0, 512), dtype=np.float32)
        if self.backend == BACKEND_SOLIDER:
            from app.reid_solider import embed_paths

            model = self._ensure()
            feats = embed_paths(
                model,
                paths,
                device=self.device,
                image_size=self.solider_image_size,
                batch_size=batch_size,
            )
            return l2_normalize(feats)

        return self._embed_batched(paths, batch_size=batch_size)


def embed_with_cache_arrays(
    extractor: ReidExtractor,
    keys: list[str],
    images: list[np.ndarray],
    cache: dict[str, np.ndarray],
    *,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Эмбеддинги из памяти; ключи — пути (для npz), без повторного чтения JPG."""
    if len(keys) != len(images):
        raise ValueError("keys и images должны быть одной длины")
    todo_keys: list[str] = []
    todo_images: list[np.ndarray] = []
    for key, img in zip(keys, images):
        if key and key not in cache:
            todo_keys.append(key)
            todo_images.append(img)
    if todo_keys:
        logger.info(
            "ReID[%s]: считаем %s новых кропов (%s из кэша)",
            extractor.model_id,
            len(todo_keys),
            len(keys) - len(todo_keys),
        )
        feats = extractor.embed_arrays(todo_images, batch_size=batch_size)
        for key, feat in zip(todo_keys, feats):
            cache[key] = feat
    return cache


def embed_with_cache(
    extractor: ReidExtractor,
    paths: list[str],
    cache: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Считает эмбеддинги только для новых кропов, остальное берёт из кэша."""
    todo = [p for p in paths if p and os.path.isfile(p) and p not in cache]
    if todo:
        logger.info(
            "ReID[%s]: считаем %s новых кропов (%s из кэша)",
            extractor.model_id,
            len(todo),
            len(paths) - len(todo),
        )
        feats = extractor.embed(todo)
        for p, f in zip(todo, feats):
            cache[p] = f
    return cache
