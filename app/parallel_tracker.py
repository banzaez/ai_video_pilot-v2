"""Двухстадийный офлайн-пайплайн: детекция → ByteTrack.

Оптимизации:
- Threaded FrameReader (I/O перекрывается с GPU-инференсом)
- grab() для пропущенных кадров при detect_every_n > 1
- torch.inference_mode() для Stage 1
- quantize=16 (FP16) в predict
- Единый .data.cpu().numpy() вместо 3 отдельных вызовов
- Кешированный xywh в DetectionSet
- Vectorized tracks_to_detections
"""

from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing
import os
import queue
import sys
import threading
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from app.progress import make_pbar
from app.util.bbox import bbox_area, bbox_ios, bbox_iou, bbox_wh

logger = logging.getLogger(__name__)

_SENTINEL = None  # маркер конца потока


def reader_queue_size(batch_size: int) -> int:
    return min(64, max(8, int(batch_size) * 2))


def open_video_capture(path: str) -> cv2.VideoCapture:
    """VideoCapture: на macOS пробуем AVFoundation (HW decode), иначе дефолтный backend."""
    path = str(path)
    cap: cv2.VideoCapture | None = None
    if sys.platform == "darwin":
        cand = cv2.VideoCapture(path, cv2.CAP_AVFOUNDATION)
        if cand.isOpened() and int(cand.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0:
            cap = cand
        else:
            cand.release()
    if cap is None:
        cap = cv2.VideoCapture(path)
    if cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        except Exception:
            pass
    return cap


# ---------------------------------------------------------------------------
# Threaded Frame Reader (Оптимизация #1)
# ---------------------------------------------------------------------------

class FrameReaderThread:
    """Фоновое декодирование видео в отдельном потоке.

    Позволяет GPU обрабатывать текущий батч, пока CPU декодирует следующий.
    """

    def __init__(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        end_frame: int,
        detect_every_n: int = 1,
        queue_size: int = 8,
        want: set[int] | None = None,
    ):
        self._cap = cap
        self._start = start_frame
        self._end = end_frame
        self._stride = max(1, detect_every_n)
        self._want = want
        self._q: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue(maxsize=max(2, int(queue_size)))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        current = self._start
        try:
            while current < self._end:
                decode = (
                    current in self._want
                    if self._want is not None
                    else current % self._stride == 0
                )
                if decode:
                    ret, frame = self._cap.read()
                    if not ret:
                        break
                    self._q.put((current, frame))
                else:
                    # Пропуск: только grab() без decode BGR
                    if not self._cap.grab():
                        break
                current += 1
        finally:
            self._q.put(_SENTINEL)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, np.ndarray]:
        item = self._q.get()
        if item is _SENTINEL:
            raise StopIteration
        return item

    def drain(self) -> None:
        """Вычитать остатки, если итерация была прервана."""
        try:
            while True:
                item = self._q.get_nowait()
                if item is _SENTINEL:
                    break
        except queue.Empty:
            pass


# ---------------------------------------------------------------------------
# DetectionSet (Оптимизация #4 — кешированный xywh)
# ---------------------------------------------------------------------------

class DetectionSet:
    """Минимальный Results-like объект для BYTETracker.update."""

    __slots__ = ("_xyxy", "conf", "cls", "xywh")

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self._xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)
        # Кешируем xywh при создании — трекер обращается многократно
        if len(self._xyxy) == 0:
            self.xywh = np.zeros((0, 4), dtype=np.float32)
        else:
            x1, y1, x2, y2 = self._xyxy.T
            w = x2 - x1
            h = y2 - y1
            self.xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, w, h], axis=1)

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __getitem__(self, idx) -> "DetectionSet":
        return DetectionSet(self._xyxy[idx], self.conf[idx], self.cls[idx])


# Меньший бокс почти целиком внутри большего (торс/голова vs полный рост).
# У RT-DETRv2 IoU такой пары часто 0.25–0.49 — одного detection.iou мало.
_NMS_CONTAIN_IOS = 0.7
# Широкий бокс на двоих: шире соседа и не «высокий полный рост над головой».
_NMS_BLOB_WIDTH = 1.35
_NMS_BLOB_HEIGHT = 0.55
_NMS_BLOB_IOS = 0.5


def _is_two_person_blob(wide: list[float], inner: list[float]) -> bool:
    """True если `wide` накрывает соседа, а не торс/голову того же человека."""
    ww, wh = bbox_wh(wide)
    iw, ih = bbox_wh(inner)
    if ww < _NMS_BLOB_WIDTH * iw:
        return False
    if ih < _NMS_BLOB_HEIGHT * wh:
        return False
    return bbox_ios(inner, wide) >= _NMS_BLOB_IOS


def nms_detections(raw_boxes: list[dict[str, Any]], iou_thresh: float) -> list[dict[str, Any]]:
    """Убрать дубли (торс + полный рост) и широкие боксы «на двоих».

    YOLO NMS по умолчанию 0.7 — пара с IoU 0.6 проходит и даёт второй track_id.
    DETR: меньший бокс внутри большего с низким IoU (IoS) и запросы, которые
    расползаются на соседа — такой жирный бокс выкидываем, узкий оставляем.
    """
    if iou_thresh <= 0 or len(raw_boxes) < 2:
        return raw_boxes
    blob = {
        i
        for i, di in enumerate(raw_boxes)
        if any(
            j != i and _is_two_person_blob(di["bbox"], raw_boxes[j]["bbox"])
            for j in range(len(raw_boxes))
        )
    }
    candidates = [i for i in range(len(raw_boxes)) if i not in blob]
    order = sorted(candidates, key=lambda i: bbox_area(raw_boxes[i]["bbox"]), reverse=True)
    keep: list[int] = []
    for i in order:
        bi = raw_boxes[i]["bbox"]
        if any(
            bbox_iou(bi, raw_boxes[j]["bbox"]) >= iou_thresh
            or bbox_ios(bi, raw_boxes[j]["bbox"]) >= _NMS_CONTAIN_IOS
            for j in keep
        ):
            continue
        keep.append(i)
    keep.sort()
    return [raw_boxes[i] for i in keep]


def boxes_to_detection_set(raw_boxes: list[dict[str, Any]], default_cls: int = 0) -> DetectionSet:
    if not raw_boxes:
        return DetectionSet(np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,)))
    xyxy = np.array([b["bbox"] for b in raw_boxes], dtype=np.float32)
    conf = np.array([b["confidence"] for b in raw_boxes], dtype=np.float32)
    cls = np.array([b.get("cls", default_cls) for b in raw_boxes], dtype=np.float32)
    return DetectionSet(xyxy, conf, cls)


# ---------------------------------------------------------------------------
# tracks_to_detections (Оптимизация #6 — vectorized)
# ---------------------------------------------------------------------------

def tracks_to_detections(tracks: np.ndarray) -> list[dict[str, Any]]:
    """tracks: N×8 → xyxy, track_id, score, cls, idx."""
    if tracks is None or len(tracks) == 0:
        return []
    data = np.asarray(tracks)
    coords = data[:, :4].astype(np.int32)
    track_ids = data[:, 4].astype(np.int32)
    confs = np.round(data[:, 5], 4)

    detections: list[dict[str, Any]] = []
    for i in range(len(data)):
        detections.append(
            {
                "track_id": int(track_ids[i]),
                "confidence": float(confs[i]),
                "bbox": coords[i].tolist(),
            }
        )
    return detections


def _resolve_tracker_device(device: Any) -> str:
    if device is not None and str(device).lower() not in ("", "auto", "none"):
        return str(device).lower()
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class TickFrameReader:
    """Последовательное чтение кадров для списка целевых индексов ticks с cap.grab()."""

    def __init__(self, video_source: str):
        self.video_source = str(video_source)
        self.cap: cv2.VideoCapture | None = open_video_capture(self.video_source)
        self.current_idx = 0
        if self.cap is None or not self.cap.isOpened():
            logger.warning("ReID frame reader: не удалось открыть видео %s", self.video_source)

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def get_frame(self, target_idx: int) -> np.ndarray | None:
        if self.cap is None or not self.cap.isOpened():
            return None
        if target_idx < self.current_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            self.current_idx = target_idx
        elif target_idx - self.current_idx > 30:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            self.current_idx = target_idx

        while self.current_idx < target_idx:
            if not self.cap.grab():
                return None
            self.current_idx += 1

        ret, frame = self.cap.read()
        if not ret:
            return None
        self.current_idx += 1
        return frame

    def close(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


def create_tracker(tracker_type: str = "bytetrack", overrides: dict | None = None):
    """Создать трекер Ultralytics по типу + overrides из config.yaml."""
    tracker_type = str(tracker_type).lower()
    if tracker_type not in TRACKER_MAP:
        raise ValueError(
            f"Неизвестный tracker '{tracker_type}'. "
            f"Доступны: {', '.join(sorted(TRACKER_MAP))}"
        )
    # База — дефолты Ultralytics, поверх — наш YAML / overrides
    cfg_path = check_yaml(f"{tracker_type}.yaml")
    cfg = IterableSimpleNamespace(**YAML.load(cfg_path))
    if overrides:
        for key, value in overrides.items():
            if value is None or str(key).startswith("_"):
                continue
            setattr(cfg, key, value)
    cfg.tracker_type = tracker_type

    # Нормализация ReID настроек
    if hasattr(cfg, "with_reid"):
        cfg.with_reid = bool(cfg.with_reid)
        if cfg.with_reid:
            if not getattr(cfg, "model", None) or str(cfg.model).strip().lower() in ("auto", "none", ""):
                cfg.model = "yolo26n-reid.onnx"
            model_str = str(cfg.model).strip()
            if not os.path.isfile(model_str):
                cand = os.path.join("data", "models", "reid", model_str)
                if os.path.isfile(cand):
                    cfg.model = cand
            cfg.device = _resolve_tracker_device(getattr(cfg, "device", None))

    # Пайплайн без GMC
    if hasattr(cfg, "gmc_method"):
        cfg.gmc_method = "none"
    return TRACKER_MAP[tracker_type](args=cfg)


def create_byte_tracker(tracker_type: str = "bytetrack", overrides: dict | None = None):
    """Alias для обратной совместимости тестов."""
    return create_tracker(tracker_type, overrides=overrides)


def _observation_ticks(
    det_indices: list[int],
    *,
    detect_every_n: int,
) -> list[int]:
    """Кадры для tracker.update: наблюдения + пустые тики между ними с шагом detect_every_n.

    Без пустых update Kalman/track_buffer считают соседние наблюдения соседними шагами,
    хотя по видео прошло N кадров — отсюда ломается IoU и рвутся ID.
    В выход tracking пишем только исходные наблюдения (см. associate_tracks).
    """
    stride = max(1, int(detect_every_n))
    if not det_indices:
        return []
    if stride <= 1:
        if not det_indices:
            return []
        lo, hi = det_indices[0], det_indices[-1]
        return list(range(lo, hi + 1))

    ticks: list[int] = []
    prev: int | None = None
    for idx in det_indices:
        if prev is not None and idx - prev > stride:
            t = prev + stride
            while t < idx:
                ticks.append(t)
                t += stride
        ticks.append(idx)
        prev = idx
    return ticks


class SessionTickReader:
    """Адаптер SessionFrameReader для associate_tracks."""

    def __init__(self, manifest: dict[str, Any]):
        from app.session.reader import SessionFrameReader
        self.manifest = manifest
        self._reader = SessionFrameReader(manifest)

    def is_opened(self) -> bool:
        return True

    def get_frame(self, target_idx: int) -> np.ndarray | None:
        try:
            return self._reader.read_frame(target_idx)
        except Exception as exc:
            logger.debug("SessionTickReader: ошибка чтения кадра %s (%s)", target_idx, exc)
            return None

    def close(self) -> None:
        try:
            self._reader.close()
        except Exception:
            pass


def _resolve_reader(
    video_source: str | None = None,
    manifest: dict[str, Any] | None = None,
) -> Any:
    """Создать читатель кадров: SessionFrameReader (если сессия/манифест) или TickFrameReader (видеофайл)."""
    if manifest and isinstance(manifest, dict) and "parts" in manifest:
        return SessionTickReader(manifest)

    if not video_source:
        return None

    v_str = str(video_source).strip()

    # 1. Проверяем, не ключ ли это сессии (session:<key> или <key>)
    sess_key = v_str.split("session:", 1)[1].strip() if v_str.startswith("session:") else v_str
    cand_info = os.path.join("data", "results", sess_key, "info.json")
    if os.path.isfile(cand_info):
        try:
            from app.io.json_util import load_tracking_json
            info_data = load_tracking_json(cand_info)
            if isinstance(info_data, dict) and "parts" in info_data:
                return SessionTickReader(info_data)
        except Exception as exc:
            logger.warning("ReID: не удалось загрузить info.json для %s (%s)", sess_key, exc)

    # 2. Одиночный файл
    for cand in (
        v_str,
        os.path.join(os.getcwd(), v_str),
        os.path.join("data", "video", v_str),
        os.path.join("data", "video", os.path.basename(v_str)),
    ):
        if cand and os.path.isfile(cand):
            reader = TickFrameReader(cand)
            if reader.is_opened():
                return reader

    logger.warning("ReID: video_source не найден (%s). Трекинг без кадров.", video_source)
    return None


def associate_tracks(
    all_detections: dict[int, list[dict[str, Any]]],
    *,
    tracker_type: str,
    total_frames: int,
    tracker_overrides: dict | None = None,
    nms_iou: float = 0.5,
    detect_every_n: int = 1,
    fill_empty_ticks: bool = False,
    video_source: str | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Stage 2: трекинг по готовым боксам (с опциональным ReID по кадрам видео при with_reid=True).

    fill_empty_ticks: если True — между наблюдениями подаём пустые update
    с шагом detect_every_n (эксперимент; на TrackTrack/ByteTrack A/B не улучшил ID).
    """
    tracker = create_tracker(tracker_type, overrides=tracker_overrides)
    det_indices = sorted(int(i) for i in all_detections.keys())
    obs_set = set(det_indices)
    if fill_empty_ticks:
        ticks = _observation_ticks(det_indices, detect_every_n=detect_every_n)
    else:
        ticks = det_indices
    n_empty = max(0, len(ticks) - len(det_indices))

    needs_reid = bool(
        getattr(getattr(tracker, "args", None), "with_reid", False)
        and getattr(tracker, "encoder", None) is not None
    )

    if total_frames > len(det_indices):
        logger.info(
            "STAGE 2: %s наблюдений%s (видеокадров %s, detect_every_n=%s%s)",
            len(det_indices),
            f" + {n_empty} пустых тиков" if n_empty else "",
            total_frames,
            max(1, int(detect_every_n)),
            ", with_reid=True" if needs_reid else "",
        )

    reader: Any = None
    if needs_reid:
        reader = _resolve_reader(video_source=video_source, manifest=manifest)

    empty_ds = boxes_to_detection_set([])
    tracked: dict[int, list[dict[str, Any]]] = {}
    pbar = make_pbar(total=max(1, len(det_indices)), desc="[STAGE 2: Tracking]", unit="obs")
    try:
        for frame_idx in ticks:
            raw = all_detections.get(frame_idx) or []
            if raw:
                raw = nms_detections(raw, nms_iou)
                ds = boxes_to_detection_set(raw)
            else:
                ds = empty_ds

            img: np.ndarray | None = None
            if reader is not None and (raw or frame_idx in obs_set):
                img = reader.get_frame(frame_idx)

            dets = tracks_to_detections(tracker.update(ds, img=img))
            if frame_idx in obs_set:
                tracked[frame_idx] = dets
                pbar.set_postfix(dets=len(raw), tracks=len(dets))
                pbar.update(1)
    finally:
        if reader is not None:
            reader.close()
        pbar.close()

    return tracked


# ---------------------------------------------------------------------------
# _detect_chunk (Оптимизации #1 threaded read, #2 inference_mode, #3 single cpu().numpy())
# ---------------------------------------------------------------------------

def _boxes_from_result(res: Any, nms_iou: float) -> list[dict[str, Any]]:
    frame_boxes: list[dict[str, Any]] = []
    if res.boxes is not None and len(res.boxes) > 0:
        boxes_data = res.boxes.data.cpu().numpy()
        coords = boxes_data[:, :4].astype(int)
        confs_arr = boxes_data[:, 4]
        for i in range(len(coords)):
            frame_boxes.append(
                {
                    "bbox": coords[i].tolist(),
                    "confidence": round(float(confs_arr[i]), 4),
                }
            )
    return nms_detections(frame_boxes, nms_iou)


def _detect_chunk(args: tuple, model=None) -> dict[int, list[dict[str, Any]]]:
    (
        video_path,
        start_frame,
        end_frame,
        model_path,
        conf_thresh,
        classes,
        device,
        batch_size,
        imgsz,
        detect_every_n,
        process_idx,
        nms_iou,
        quantize,
        detector_backend,
    ) = args
    detect_every_n = max(1, int(detect_every_n))
    nms_iou = float(nms_iou)
    detector_backend = str(detector_backend or "yolo").lower()
    # Ultralytics: 16 = FP16, None = FP32
    quantize = None if quantize in (None, 32, 0) else int(quantize)

    import torch  # локальный импорт — в subprocess может не быть заранее
    from app.model_cache import load_detector_model, predict_batch_size, resolve_runtime_weights

    runtime, fmt = resolve_runtime_weights(model_path)
    if model is None:
        logger.info("STAGE 1: %s model=%s (%s)", detector_backend, runtime, fmt)
        model = load_detector_model(model_path, backend=detector_backend)
    else:
        logger.info("STAGE 1: %s model=%s (%s)", detector_backend, runtime, fmt)

    # CoreML без dynamic не умеет batch>1
    infer_batch = predict_batch_size(model_path, batch_size)

    cap = open_video_capture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    chunk: dict[int, list[dict[str, Any]]] = {}
    total = max(0, end_frame - start_frame)

    pbar = make_pbar(
        total=total,
        desc=f"[STAGE 1: {detector_backend.upper()} #{process_idx + 1}]",
        position=process_idx,
        leave=True,
        unit="frame",
    )
    boxes_total = 0

    reader = FrameReaderThread(
        cap,
        start_frame=start_frame,
        end_frame=end_frame,
        detect_every_n=detect_every_n,
        queue_size=reader_queue_size(max(batch_size, infer_batch)),
    )

    skipped_per_batch = detect_every_n - 1 if detect_every_n > 1 else 0
    predict_kw = dict(
        classes=classes,
        conf=conf_thresh,
        iou=nms_iou if nms_iou > 0 else 0.7,
        device=device,
        imgsz=imgsz,
        quantize=quantize,
        verbose=False,
        stream=True,
    )

    try:
        with torch.inference_mode():
            batch_frames: list[np.ndarray] = []
            batch_indices: list[int] = []

            def flush_batch() -> None:
                nonlocal boxes_total
                if not batch_frames:
                    return
                results = model.predict(source=batch_frames, **predict_kw)
                for idx, res in zip(batch_indices, results):
                    frame_boxes = _boxes_from_result(res, nms_iou)
                    chunk[idx] = frame_boxes
                    boxes_total += len(frame_boxes)
                    pbar.set_postfix(boxes=boxes_total, batch=len(batch_frames))
                    pbar.update(1 + skipped_per_batch)
                batch_frames.clear()
                batch_indices.clear()

            for frame_idx, frame in reader:
                batch_frames.append(frame)
                batch_indices.append(frame_idx)
                if len(batch_frames) >= infer_batch:
                    flush_batch()
            flush_batch()

    finally:
        reader.drain()
        pbar.close()
        cap.release()

    return chunk


def detect_video_frames(
    video_path: str,
    *,
    model_path: str,
    conf: float,
    classes: list[int],
    device: str,
    batch_size: int,
    num_workers: int,
    total_frames: int,
    imgsz: int = 640,
    detect_every_n: int = 1,
    nms_iou: float = 0.5,
    quantize: int | None = None,
    model_cache=None,
    detector_backend: str = "yolo",
) -> dict[int, list[dict[str, Any]]]:
    """
    Stage 1: детекции по кадрам.
    На MPS/одиночном воркере — один процесс с батчами (без конкуренции за GPU).
    Иначе — ProcessPool по чанкам.
    """
    effective_workers = num_workers
    if device == "mps" and num_workers > 1:
        logger.info("MPS: параллельные воркеры отключены, используем batch_size=%s", batch_size)
        effective_workers = 1

    quantize = None if quantize in (None, 32, 0) else int(quantize)
    detector_backend = str(detector_backend or "yolo").lower()

    args = (
        video_path,
        0,
        total_frames,
        model_path,
        conf,
        classes,
        device,
        batch_size,
        imgsz,
        detect_every_n,
        0,
        nms_iou,
        quantize,
        detector_backend,
    )

    if effective_workers <= 1:
        model = None
        if model_cache is not None:
            model = model_cache.get_detector(model_path, backend=detector_backend, kind="detect")
        else:
            from app.model_cache import get_model_cache

            model = get_model_cache().get_detector(model_path, backend=detector_backend, kind="detect")
        return _detect_chunk(args, model=model)

    chunk_size = int(np.ceil(total_frames / effective_workers))
    tasks = []
    for i in range(effective_workers):
        start_f = i * chunk_size
        end_f = min((i + 1) * chunk_size, total_frames)
        if start_f < total_frames:
            tasks.append(
                (
                    video_path,
                    start_f,
                    end_f,
                    model_path,
                    conf,
                    classes,
                    device,
                    batch_size,
                    imgsz,
                    detect_every_n,
                    i,
                    nms_iou,
                    quantize,
                    detector_backend,
                )
            )

    all_detections: dict[int, list[dict[str, Any]]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = [executor.submit(_detect_chunk, t) for t in tasks]
        p_chunks = make_pbar(
            total=len(futures),
            desc="[STAGE 1: chunks]",
            unit="chunk",
        )
        try:
            for future in concurrent.futures.as_completed(futures):
                part = future.result()
                all_detections.update(part)
                p_chunks.set_postfix(frames=len(all_detections))
                p_chunks.update(1)
        finally:
            p_chunks.close()

    return all_detections


def default_workers(requested: int | None, device: str) -> int:
    if requested is not None:
        return max(1, requested)
    if device == "mps":
        return 1
    return max(1, multiprocessing.cpu_count() - 1)
