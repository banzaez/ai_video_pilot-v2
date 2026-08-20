#!/usr/bin/env python3
"""Автономный worker детекции RT-DETR (rtdetr-l.pt) для запуска на GPU-сервере.

Принимает видеофайл(ы) или директорию, выполняет инференс RT-DETR с NMS-подавлением
и сохраняет результат в стандартный detections.json проекта.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import queue
import re
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import RTDETR

# Относительный импорт утилит геометрии
try:
    from bbox_util import bbox_wh, nms_detections
except ImportError:
    from tools.remote_detector.bbox_util import bbox_wh, nms_detections

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("remote_detector")

_SENTINEL = None

# Регулярка для определения session_key (Camera_024_..._20260817... -> 024_20260817)
_PROD_CAM_RE = re.compile(
    r"^Camera_(?P<idx>\d+)_(?P<source>.+)_(?P<started>\d{14})_(?P<ended>\d{14})_(?P<seg>.+)$",
    re.IGNORECASE,
)


def extract_session_key(file_path: str) -> str:
    """Извлекает session_key из имени файла (например, 024_20260817)."""
    stem = os.path.splitext(os.path.basename(file_path))[0]
    match = _PROD_CAM_RE.match(stem)
    if match:
        idx_str = match.group("idx").zfill(3)
        day_str = match.group("started")[:8]
        return f"{idx_str}_{day_str}"
    
    # Фолбэк по паттерну Camera_XX и даты YYYYMMDD
    cam_match = re.search(r"Camera_?(\d+)", stem, re.IGNORECASE)
    date_match = re.search(r"(\d{8})", stem)
    if cam_match and date_match:
        return f"{cam_match.group(1).zfill(3)}_{date_match.group(1)}"
    
    return stem


class FrameReaderThread:
    """Фоновое декодирование видео в отдельном потоке (I/O параллельно с GPU)."""

    def __init__(
        self,
        cap: cv2.VideoCapture,
        start_frame: int = 0,
        end_frame: int | None = None,
        detect_every_n: int = 1,
        queue_size: int = 64,
    ):
        self._cap = cap
        self._start = max(0, int(start_frame))
        self._end = int(end_frame) if end_frame is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._stride = max(1, detect_every_n)
        self._q: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue(maxsize=max(8, queue_size))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        current = self._start
        try:
            while current < self._end:
                decode = (current % self._stride == 0)
                if decode:
                    ret, frame = self._cap.read()
                    if not ret or frame is None:
                        break
                    self._q.put((current, frame))
                else:
                    ret = self._cap.grab()
                    if not ret:
                        break
                current += 1
        except Exception as exc:
            logger.warning("Ошибка чтения кадров: %s", exc)
        finally:
            self._q.put(_SENTINEL)

    def get(self) -> tuple[int, np.ndarray] | None:
        return self._q.get()

    def drain(self) -> None:
        try:
            while True:
                item = self._q.get_nowait()
                if item is _SENTINEL:
                    break
        except queue.Empty:
            pass


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


def _detect_video_chunk(
    video_path: str,
    start_frame: int,
    end_frame: int,
    model: Any,
    predict_kw: dict[str, Any],
    batch_size: int,
    stride: int,
    nms_iou: float,
    pbar: tqdm,
    lock: threading.Lock,
) -> dict[int, list[dict[str, Any]]]:
    """Обработка диапазона кадров [start_frame, end_frame) в отдельном потоке."""
    cap = cv2.VideoCapture(video_path)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    reader = FrameReaderThread(
        cap,
        start_frame=start_frame,
        end_frame=end_frame,
        detect_every_n=stride,
        queue_size=max(64, batch_size * 2),
    )

    chunk_dets: dict[int, list[dict[str, Any]]] = {}
    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []

    def flush() -> None:
        if not batch_frames:
            return
        results = model.predict(source=batch_frames, **predict_kw)
        with lock:
            for idx, res in zip(batch_indices, results):
                boxes = _boxes_from_result(res, nms_iou)
                if boxes:
                    chunk_dets[idx] = boxes
            pbar.update(len(batch_frames) * stride)
            pbar.set_postfix(
                dets=len(chunk_dets),
                batch=len(batch_frames),
            )
        batch_frames.clear()
        batch_indices.clear()

    try:
        with torch.inference_mode():
            while True:
                item = reader.get()
                if item is _SENTINEL:
                    break
                frame_idx, frame = item
                batch_frames.append(frame)
                batch_indices.append(frame_idx)
                if len(batch_frames) >= batch_size:
                    flush()
            flush()
    finally:
        reader.drain()
        cap.release()

    return chunk_dets


def build_detections_payload(
    detections_by_frame: dict[int, list[dict[str, Any]]],
    *,
    video_source: str,
    fps: float,
    frame_count: int,
    width: int,
    height: int,
    conf_thresh: float,
    detect_every_n: int,
    nms_iou: float,
    session_key: str | None = None,
    detector_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Сборка итогового payload detections.json."""
    frames: list[dict[str, Any]] = []
    for idx in sorted(i for i, dets in detections_by_frame.items() if dets):
        frames.append(
            {
                "frame_index": idx + 1,
                "timestamp_sec": round((idx + 1) / fps, 3) if fps > 0 else 0.0,
                "detections": [
                    {"bbox": det["bbox"], "confidence": det["confidence"]}
                    for det in detections_by_frame[idx]
                ],
            }
        )

    payload: dict[str, Any] = {
        "stage": "detect",
        "video_source": video_source,
        "fps": round(float(fps), 3),
        "frame_count": int(frame_count),
        "width": int(width),
        "height": int(height),
        "conf_threshold": float(conf_thresh),
        "detect_every_n": int(detect_every_n),
        "nms_iou": float(nms_iou),
        "n_frames": len(frames),
        "frames": frames,
    }
    if session_key:
        payload["kind"] = "camera_day"
        payload["session_key"] = session_key
    if detector_meta:
        payload["detector"] = detector_meta
    return payload


def load_detector_model(weights_path: str) -> Any:
    """Загрузка RT-DETR или YOLO в зависимости от весов."""
    basename = os.path.basename(weights_path).lower()
    if "rtdetr" in basename:
        from ultralytics import RTDETR
        logger.info("Загрузка модели RT-DETR (%s)...", weights_path)
        return RTDETR(weights_path)
    from ultralytics import YOLO
    logger.info("Загрузка модели YOLO (%s)...", weights_path)
    return YOLO(weights_path, task="detect")


def process_video_file(
    video_path: str,
    model: Any,
    args: argparse.Namespace,
) -> str:
    """Обработка одного видеофайла с детекцией RT-DETR/YOLO."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Видеофайл не найден: {video_path}")

    stem = os.path.splitext(os.path.basename(video_path))[0]
    session_key = extract_session_key(video_path)

    # Путь для сохранения detections.json
    out_dir = os.path.join(args.output_dir, session_key)
    os.makedirs(out_dir, exist_ok=True)
    out_json_path = os.path.join(out_dir, "detections.json")

    if os.path.isfile(out_json_path) and not args.overwrite:
        logger.info("[SKIP] %s уже обработан -> %s", session_key, out_json_path)
        return out_json_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видеофайл через OpenCV: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    workers = max(1, int(getattr(args, "workers", 2) or 2))
    stride = max(1, int(args.detect_every_n))
    batch_size = max(1, int(args.batch_size))
    conf_thresh = float(args.conf)
    nms_iou = float(args.iou)
    classes = [int(c) for c in args.classes.split(",")] if args.classes else [0]

    logger.info(
        "Начало детекции: session=%s, файл=%s (%sx%s, %.1f сек, %d кадров, fps=%.2f, потоков=%d)",
        session_key,
        os.path.basename(video_path),
        width,
        height,
        total_frames / fps if fps > 0 else 0,
        total_frames,
        fps,
        workers,
    )

    predict_kw = dict(
        classes=classes,
        conf=conf_thresh,
        iou=nms_iou if nms_iou > 0 else 0.7,
        device=args.device,
        imgsz=args.imgsz,
        quantize=16 if args.fp16 else None,
        verbose=False,
        stream=True,
    )

    t_start = time.time()
    pbar = tqdm(
        total=total_frames,
        desc=f"[{session_key}] RT-DETR x{workers}",
        unit="fr",
        dynamic_ncols=True,
        leave=True,
    )

    detections_by_frame: dict[int, list[dict[str, Any]]] = {}
    lock = threading.Lock()

    if workers > 1 and total_frames > workers * batch_size * 2:
        chunk_size = int(np.ceil(total_frames / workers))
        chunks = []
        for w in range(workers):
            sf = w * chunk_size
            ef = min((w + 1) * chunk_size, total_frames)
            if sf < total_frames:
                chunks.append((sf, ef))

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(
                    _detect_video_chunk,
                    video_path,
                    sf,
                    ef,
                    model,
                    predict_kw,
                    batch_size,
                    stride,
                    nms_iou,
                    pbar,
                    lock,
                )
                for sf, ef in chunks
            ]
            for fut in concurrent.futures.as_completed(futures):
                chunk_res = fut.result()
                detections_by_frame.update(chunk_res)
    else:
        detections_by_frame = _detect_video_chunk(
            video_path,
            0,
            total_frames,
            model,
            predict_kw,
            batch_size,
            stride,
            nms_iou,
            pbar,
            lock,
        )

    pbar.close()

    elapsed = time.time() - t_start
    fps_infer = total_frames / elapsed if elapsed > 0 else 0.0
    boxes_total = sum(len(dets) for dets in detections_by_frame.values())

    detector_meta = {
        "backend": "rtdetr",
        "path": args.weights,
        "classes": classes,
        "imgsz": args.imgsz,
        "batch_size": batch_size,
        "quantize": 16 if args.fp16 else None,
        "device": str(args.device),
    }

    payload = build_detections_payload(
        detections_by_frame,
        video_source=f"session:{session_key}",
        fps=fps,
        frame_count=total_frames,
        width=width,
        height=height,
        conf_thresh=conf_thresh,
        detect_every_n=stride,
        nms_iou=nms_iou,
        session_key=session_key,
        detector_meta=detector_meta,
    )

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=None)

    logger.info(
        "Завершено %s: кадров с людьми=%d/%d, всего боксов=%d, время=%.1fс (скорость: %.1f FPS) -> %s",
        session_key,
        payload["n_frames"],
        total_frames,
        boxes_total,
        elapsed,
        fps_infer,
        out_json_path,
    )
    return out_json_path


def load_yaml_config(config_path: str | None = None) -> dict[str, Any]:
    """Загрузка config.yaml (по умолчанию рядом с worker.py)."""
    target = config_path
    if not target:
        default_cfg = os.path.join(os.path.dirname(__file__), "config.yaml")
        if os.path.isfile(default_cfg):
            target = default_cfg
        elif os.path.isfile("config.yaml"):
            target = "config.yaml"

    if not target or not os.path.isfile(target):
        return {}

    try:
        import yaml
        with open(target, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        logger.info("Загружена конфигурация из: %s", target)
        return cfg
    except Exception as exc:
        logger.warning("Не удалось прочитать YAML конфигурацию (%s): %s", target, exc)
        return {}


def list_video_files(directory: str) -> list[str]:
    """Рекурсивный поиск видео с игнорированием папок '_orig', 'lite', '_' и '.'."""
    if not directory or not os.path.isdir(directory):
        return []
    valid_exts = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
    videos: list[str] = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in sorted(dirs) if not d.startswith(("_", ".")) and d.lower() != "lite"]
        for name in sorted(files):
            if name.startswith(("_", ".")):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in valid_exts:
                videos.append(os.path.abspath(os.path.join(root, name)))
    return videos


def main() -> None:
    parser = argparse.ArgumentParser(description="Автономный worker детекции RT-DETR на GPU.")
    parser.add_argument("--config", type=str, help="Путь к config.yaml (по умолчанию tools/remote_detector/config.yaml)")
    parser.add_argument("--video", type=str, help="Путь к конкретному видеофайлу")
    parser.add_argument("--input-dir", type=str, help="Путь к директории с видео (поиск mp4/mov/mkv/avi)")
    parser.add_argument("--day", type=str, help="День в формате YYYYMMDD (поиск внутри input-dir или data/video)")
    parser.add_argument("--weights", type=str, help="Веса RT-DETR (rtdetr-l.pt или rtdetr-x.pt)")
    parser.add_argument("--output-dir", type=str, help="Корневая директория сохранения results")
    parser.add_argument("--batch-size", type=int, help="Размер батча для predict")
    parser.add_argument("--imgsz", type=int, help="Размер длинной стороны кадра")
    parser.add_argument("--conf", type=float, help="Порог уверенности детекции")
    parser.add_argument("--iou", type=float, help="NMS IoU порог")
    parser.add_argument("--detect-every-n", type=int, help="Шаг прореживания кадров (1 = все кадры)")
    parser.add_argument("--classes", type=str, help="Классы COCO через запятую (0 = person)")
    parser.add_argument("--device", type=str, help="Устройство инференса (0, cuda:0, cpu, mps)")
    parser.add_argument("--workers", type=int, help="Число параллельных потоков обработки одного видео (по умолчанию: 2)")
    parser.add_argument("--fp16", action="store_true", default=None, help="Использовать FP16 точность")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Перезаписывать существующие detections.json (по умолчанию: всегда True)")

    args = parser.parse_args()

    # Загружаем настройки из config.yaml
    cfg = load_yaml_config(args.config)
    det_cfg = cfg.get("detection", {})
    paths_cfg = cfg.get("paths", {})
    cuda_cfg = cfg.get("cuda_optimizations", {})

    # Применение значений: CLI > config.yaml > hardcoded defaults
    args.overwrite = True if args.overwrite else bool(det_cfg.get("overwrite", True))
    if getattr(args, "workers", None) is None:
        args.workers = int(det_cfg.get("workers") or 2)
    if args.weights is None:
        args.weights = str(det_cfg.get("model") or "rtdetr-l.pt")
    if args.output_dir is None:
        args.output_dir = str(paths_cfg.get("results_dir") or "data/results")
    if args.input_dir is None and not args.video:
        args.input_dir = str(paths_cfg.get("video_dir") or "data/video")
    if args.batch_size is None:
        args.batch_size = int(det_cfg.get("batch_size") or 32)
    if args.imgsz is None:
        args.imgsz = int(det_cfg.get("imgsz") or 1280)
    if args.conf is None:
        args.conf = float(det_cfg.get("conf") if det_cfg.get("conf") is not None else 0.10)
    if args.iou is None:
        args.iou = float(det_cfg.get("iou") if det_cfg.get("iou") is not None else 0.50)
    if args.detect_every_n is None:
        args.detect_every_n = int(det_cfg.get("detect_every_n") or 1)
    if args.classes is None:
        raw_cls = det_cfg.get("classes", [0])
        args.classes = ",".join(str(c) for c in raw_cls) if isinstance(raw_cls, list) else str(raw_cls)
    if args.device is None:
        args.device = str(det_cfg.get("device") or "0")
    if args.fp16 is None:
        args.fp16 = bool(det_cfg.get("fp16", True))

    # Сбор списка видеофайлов
    video_files: list[str] = []
    if args.video:
        video_files.append(os.path.abspath(args.video))
    elif args.day or args.input_dir:
        base_dir = args.input_dir or "data/video"
        search_dirs: list[str] = []
        if args.day:
            day_clean = args.day.replace("-", "").strip()
            cand_dirs = [
                os.path.join(base_dir, day_clean),
                os.path.join(base_dir, f"day_{day_clean}"),
            ]
            search_dirs.extend([d for d in cand_dirs if os.path.isdir(d)])

        if not search_dirs and os.path.isdir(base_dir):
            search_dirs.append(base_dir)

        for sdir in search_dirs:
            video_files.extend(list_video_files(sdir))

        if args.day:
            day_clean = args.day.replace("-", "").strip()
            video_files = [f for f in set(video_files) if day_clean in os.path.basename(f)]
        else:
            video_files = list(set(video_files))

    video_files.sort()

    if not video_files:
        logger.error("Не найдено видеофайлов для обработки по указанным параметрам.")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("Удаленный воркер детекции RT-DETR")
    logger.info("Найдено видеофайлов: %d", len(video_files))
    logger.info("Модель весов: %s", args.weights)
    # Оптимизации PyTorch/CUDA под архитектуру Ada Lovelace (RTX 4090)
    if torch.cuda.is_available() and str(args.device).lower() not in ("cpu", "mps"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            cv2.setNumThreads(min(8, os.cpu_count() or 4))
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass

        dev_idx = int(args.device) if str(args.device).isdigit() else 0
        gpu_name = torch.cuda.get_device_name(dev_idx)
        gpu_vram = torch.cuda.get_device_properties(dev_idx).total_memory / (1024**3)
        logger.info("CUDA GPU: %s (%.1f ГБ VRAM) | Режим: FP16=%s, TF32=True, cuDNN benchmark=True",
                    gpu_name, gpu_vram, bool(args.fp16))

    # Поиск весов
    weights_path = args.weights
    candidates = [
        weights_path,
        os.path.join("data", "models", "detect", os.path.basename(weights_path)),
        os.path.join("data", "models", os.path.basename(weights_path)),
        os.path.join(os.path.dirname(__file__), os.path.basename(weights_path)),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            weights_path = cand
            break

    model = load_detector_model(weights_path)
    logger.info("Модель детекции успешно загружена!")

    # Прогрев CUDA ядер и выделение видеопамяти
    if torch.cuda.is_available() and str(args.device).lower() not in ("cpu", "mps"):
        logger.info("Прогрев CUDA-ядер на GPU...")
        warmup_batch = min(4, max(1, args.batch_size))
        dummy = [np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8) for _ in range(warmup_batch)]
        with torch.inference_mode():
            model.predict(
                source=dummy,
                device=args.device,
                imgsz=args.imgsz,
                quantize=16 if args.fp16 else None,
                verbose=False,
            )
        torch.cuda.synchronize()
        logger.info("Прогрев CUDA завершен.")

    success_count = 0
    fail_count = 0
    total_start = time.time()

    for idx, vpath in enumerate(video_files, start=1):
        logger.info("\n[%d/%d] Обработка: %s", idx, len(video_files), os.path.basename(vpath))
        try:
            process_video_file(vpath, model, args)
            success_count += 1
        except Exception as exc:
            fail_count += 1
            logger.error("Ошибка при обработке %s: %s", vpath, exc, exc_info=True)

    total_time = time.time() - total_start
    logger.info("\n" + "=" * 70)
    logger.info(
        "ИТОГ ДЕТЕКЦИИ: успешно %d/%d, ошибок %d, общее время: %.1f сек",
        success_count,
        len(video_files),
        fail_count,
        total_time,
    )
    logger.info("=" * 70)

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
