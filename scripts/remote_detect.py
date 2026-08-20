#!/usr/bin/env python3
"""Локальный скрипт для удаленной детекции RT-DETR (rtdetr-l.pt) на SSH хосте.

Пайплайн:
1. Проверка SSH подключения к хосту (по умолчанию: aivideo).
2. Синхронизация автономного мини-воркера (tools/remote_detector) на сервер.
3. Отправка видеофайлов с прогресс-баром (rsync --ignore-existing, без повторной загрузки).
4. Запуск инференса RT-DETR на GPU через SSH со стримингом прогресс-баров (tqdm) в реальном времени.
5. Скачивание готовых detections.json в локальную папку data/results/{session_key}/.
6. Валидация структуры скачанных detections.json.

Примеры использования:
    ./venv/bin/python3 scripts/remote_detect.py --day 20260817
    ./venv/bin/python3 scripts/remote_detect.py --input data/video/20260817 --batch-size 32
    ./venv/bin/python3 scripts/remote_detect.py --session 024_20260817 --host aivideo
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any

# Настройка локального логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("remote_detect")


def run_streaming_cmd(cmd: list[str] | str, *, shell: bool = False, desc: str = "") -> int:
    """Выполняет команду с пробросом вывода в реальном времени (без буферизации)."""
    if desc:
        logger.info(">>> %s", desc)
    
    cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
    logger.debug("Команда: %s", cmd_str)

    process = subprocess.Popen(
        cmd,
        shell=shell,
        stdout=sys.stdout,
        stderr=sys.stderr,
        bufsize=1,
        universal_newlines=True,
    )
    return process.wait()


def check_ssh_connection(host: str) -> bool:
    """Проверка доступности SSH хоста."""
    logger.info("Проверка SSH подключения к хосту '%s'...", host)
    cmd = ["ssh", host, "echo 'SSH_CONNECTION_OK'"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if proc.returncode == 0 and "SSH_CONNECTION_OK" in proc.stdout:
            logger.info("SSH подключение к '%s' успешно установлено.", host)
            return True
    except subprocess.TimeoutExpired:
        logger.warning("Быстрая фоновая проверка превысила таймаут (возможно, требуется ввод пароля/passphrase ключа).")
    except Exception as exc:
        logger.debug("Фоновая проверка завершилась с ошибкой: %s", exc)

    # Интерактивная попытка (если требуется ввод passphrase или подтверждение ключа)
    ret = run_streaming_cmd(["ssh", host, "echo 'SSH_CONNECTION_OK'"], desc=f"Проверка подключения к {host}")
    if ret == 0:
        logger.info("SSH подключение к '%s' успешно установлено.", host)
        return True

    logger.error("Не удалось подключиться к SSH хосту '%s'.", host)
    return False


def list_video_files(directory: str) -> list[str]:
    """Рекурсивный поиск видео с игнорированием папок '_orig', 'lite', '_' и '.'."""
    if not directory or not os.path.isdir(directory):
        return []
    valid_exts = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
    videos: list[str] = []
    for root, dirs, files in os.walk(directory):
        # Игнорируем служебные папки (_orig, .git, lite и т.д.)
        dirs[:] = [d for d in sorted(dirs) if not d.startswith(("_", ".")) and d.lower() != "lite"]
        for name in sorted(files):
            if name.startswith(("_", ".")):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in valid_exts:
                videos.append(os.path.abspath(os.path.join(root, name)))
    return videos


def find_local_videos(input_arg: str, day_arg: str | None) -> list[str]:
    """Поиск локальных видеофайлов по аргументам."""
    # Если передан конкретный файл
    if os.path.isfile(input_arg):
        return [os.path.abspath(input_arg)]

    # Разбор дня
    day_clean = None
    if day_arg:
        day_clean = day_arg.replace("-", "").replace("day:", "").strip()
    elif input_arg.startswith("day:"):
        day_clean = input_arg.replace("day:", "").replace("-", "").strip()

    search_dirs: list[str] = []
    if day_clean:
        cand_day_dirs = [
            os.path.join("data", "video", day_clean),
            os.path.join("data", "video", f"day_{day_clean}"),
        ]
        search_dirs.extend([d for d in cand_day_dirs if os.path.isdir(d)])

    if not search_dirs and os.path.isdir(input_arg):
        search_dirs.append(input_arg)

    if not search_dirs and os.path.isdir("data/video"):
        search_dirs.append("data/video")

    videos: list[str] = []
    for sdir in search_dirs:
        videos.extend(list_video_files(sdir))

    unique_videos = sorted(list(set(videos)))
    if day_clean:
        unique_videos = [v for v in unique_videos if day_clean in os.path.basename(v)]

    return unique_videos


def human_size(size_bytes: int) -> str:
    """Форматирование размера в читаемый вид (КБ, МБ, ГБ)."""
    for unit in ["Б", "КБ", "МБ", "ГБ", "ТБ"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:3.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} ПБ"


def validate_detection_json(json_path: str) -> dict[str, Any] | None:
    """Проверка структуры полученного detections.json."""
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        frames = data.get("frames", [])
        total_boxes = sum(len(fr.get("detections", [])) for fr in frames)
        return {
            "n_frames": data.get("n_frames", len(frames)),
            "total_frames": data.get("frame_count", 0),
            "boxes": total_boxes,
            "fps": data.get("fps", 0),
            "stage": data.get("stage", ""),
        }
    except Exception as exc:
        logger.warning("Ошибка валидации JSON %s: %s", json_path, exc)
        return None


def load_remote_detector_config() -> dict[str, Any]:
    """Чтение конфигурации из tools/remote_detector/config.yaml."""
    cfg_path = os.path.join("tools", "remote_detector", "config.yaml")
    if os.path.isfile(cfg_path):
        try:
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


def main() -> None:
    # Загружаем базовые значения из tools/remote_detector/config.yaml
    cfg = load_remote_detector_config()
    det_cfg = cfg.get("detection", {})
    paths_cfg = cfg.get("paths", {})

    cfg_model = str(det_cfg.get("model") or "rtdetr-l.pt")
    cfg_batch = int(det_cfg.get("batch_size") or 32)
    cfg_imgsz = int(det_cfg.get("imgsz") or 1280)
    cfg_conf = float(det_cfg.get("conf") if det_cfg.get("conf") is not None else 0.10)
    cfg_iou = float(det_cfg.get("iou") if det_cfg.get("iou") is not None else 0.50)
    cfg_stride = int(det_cfg.get("detect_every_n") or 1)
    cfg_device = str(det_cfg.get("device") or "0")
    cfg_workers = int(det_cfg.get("workers") or 2)
    cfg_remote_dir = str(paths_cfg.get("remote_dir") or "/workspace/remote_detector")

    parser = argparse.ArgumentParser(
        description="Скрипт для удаленной детекции RT-DETR на GPU-сервере (SSH aivideo)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/video",
        help="Путь к видео / папке / day:20260817 / session:024_20260817",
    )
    parser.add_argument(
        "--day",
        type=str,
        help="День в формате YYYYMMDD (например, 20260817)",
    )
    parser.add_argument(
        "--session",
        type=str,
        help="Сессия (например, 024_20260817)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="aivideo",
        help="SSH хост из ~/.ssh/config (по умолчанию: aivideo)",
    )
    parser.add_argument(
        "--remote-dir",
        type=str,
        default=cfg_remote_dir,
        help=f"Путь к директории мини-проекта на сервере (по умолчанию: {cfg_remote_dir})",
    )
    parser.add_argument(
        "--remote-python",
        type=str,
        default="/venv/main/bin/python3",
        help="Интерпретатор Python на удаленном сервере (по умолчанию: /venv/main/bin/python3)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=cfg_model,
        help=f"Веса модели детекции (по умолчанию из config.yaml: {cfg_model})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=cfg_batch,
        help=f"Размер батча для инференса на GPU (по умолчанию из config.yaml: {cfg_batch})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=cfg_imgsz,
        help=f"Размер длинной стороны кадра (по умолчанию: {cfg_imgsz})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=cfg_conf,
        help=f"Порог уверенности детекции (по умолчанию: {cfg_conf})",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=cfg_iou,
        help=f"NMS IoU порог (по умолчанию: {cfg_iou})",
    )
    parser.add_argument(
        "--detect-every-n",
        type=int,
        default=cfg_stride,
        help=f"Шаг кадров (по умолчанию: {cfg_stride})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cfg_workers,
        help=f"Число параллельных потоков обработки одного видео (по умолчанию: {cfg_workers})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=cfg_device,
        help=f"Устройство инференса на GPU сервере (по умолчанию: {cfg_device})",
    )
    parser.add_argument(
        "--skip-sync-code",
        action="store_true",
        help="Пропустить шаг синхронизации кода воркера",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Пропустить шаг загрузки видео на сервер",
    )
    parser.add_argument(
        "--skip-detect",
        action="store_true",
        help="Пропустить шаг выполнения инференса на сервере",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Пропустить шаг скачивания результатов",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Перезаписывать существующие результаты детекции (по умолчанию: True)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Вывести планируемые действия и команды без выполнения",
    )

    args = parser.parse_args()

    # 1. Определение списка видео
    input_target = args.session or args.input
    videos = find_local_videos(input_target, args.day)

    if not videos and not args.skip_upload:
        logger.error("Не найдено локальных видеофайлов для отправки.")
        sys.exit(1)

    total_size = sum(os.path.getsize(v) for v in videos) if videos else 0

    logger.info("=" * 75)
    logger.info("  AI VIDEO PILOT: УДАЛЕННАЯ ДЕТЕКЦИЯ RT-DETR ЧЕРЕЗ SSH")
    logger.info("=" * 75)
    logger.info("SSH хост:             %s", args.host)
    logger.info("Удаленная директория: %s", args.remote_dir)
    logger.info("Модель / Веса:        %s", args.weights)
    logger.info("Параметры инференса:  batch=%d, imgsz=%d, conf=%.2f, iou=%.2f, workers=%d, device=%s",
                args.batch_size, args.imgsz, args.conf, args.iou, args.workers, args.device)
    logger.info("Найдено видеофайлов:  %d (%s)", len(videos), human_size(total_size))
    for idx, v in enumerate(videos, 1):
        logger.info("  [%d] %s (%s)", idx, os.path.basename(v), human_size(os.path.getsize(v)))
    logger.info("=" * 75)

    if args.dry_run:
        logger.info("[DRY-RUN] Режим симуляции. Завершение работы без изменений.")
        return

    # 2. Проверка связи
    if not check_ssh_connection(args.host):
        logger.error("Не удалось подключиться к SSH хосту '%s'. Проверьте ~/.ssh/config и доступность сервера.", args.host)
        sys.exit(1)

    # Создание удаленных директорий
    cmd_mkdir = ["ssh", args.host, f"mkdir -p {args.remote_dir}/tools/remote_detector {args.remote_dir}/data/video {args.remote_dir}/data/models/detect {args.remote_dir}/data/results"]
    run_streaming_cmd(cmd_mkdir, desc=f"Создание директорий на сервере {args.host}")

    # 3. Синхронизация кода мини-воркера и весов
    if not args.skip_sync_code:
        local_tool_dir = os.path.abspath("tools/remote_detector")
        if not os.path.isdir(local_tool_dir):
            logger.error("Локальная директория %s не найдена!", local_tool_dir)
            sys.exit(1)

        rsync_code_cmd = [
            "rsync",
            "-avz",
            "--progress",
            "--delete",
            f"{local_tool_dir}/",
            f"{args.host}:{args.remote_dir}/tools/remote_detector/",
        ]
        ret = run_streaming_cmd(rsync_code_cmd, desc="Синхронизация кода мини-воркера на сервер")
        if ret != 0:
            logger.error("Ошибка синхронизации кода воркера (код %d)", ret)
            sys.exit(1)

        # Проверяем локальное наличие файла весов и синхронизируем при необходимости
        local_weights_cands = [
            args.weights,
            os.path.join("data", "models", "detect", os.path.basename(args.weights)),
            os.path.join("data", "models", os.path.basename(args.weights)),
        ]
        for w_cand in local_weights_cands:
            if os.path.isfile(w_cand):
                rsync_w_cmd = [
                    "rsync",
                    "-avz",
                    "--progress",
                    "--ignore-existing",
                    w_cand,
                    f"{args.host}:{args.remote_dir}/data/models/detect/{os.path.basename(args.weights)}",
                ]
                run_streaming_cmd(rsync_w_cmd, desc=f"Синхронизация весов модели ({os.path.basename(args.weights)})")
                break

    # 4. Отправка видеофайлов с прогресс-баром (без повторной загрузки)
    if not args.skip_upload and videos:
        logger.info("\n--- ШАГ: Отправка видеофайлов на сервер (rsync --progress --ignore-existing) ---")
        
        # Определяем подпапку на сервере (например, data/video/20260817 или data/video)
        day_sub = None
        if args.day:
            day_sub = args.day.replace("-", "").replace("day:", "").strip()
        elif args.input and "2026" in args.input:
            match = re.search(r"(\d{8})", args.input)
            if match:
                day_sub = match.group(1)

        remote_video_dest = f"{args.remote_dir}/data/video"
        if day_sub:
            remote_video_dest += f"/{day_sub}"
            run_streaming_cmd(["ssh", args.host, f"mkdir -p {remote_video_dest}"], desc=f"Создание папки {remote_video_dest}")

        # Формируем rsync команду
        rsync_video_cmd = [
            "rsync",
            "-avz",
            "--progress",
            "--ignore-existing",
        ] + videos + [f"{args.host}:{remote_video_dest}/"]

        t0_up = time.time()
        ret = run_streaming_cmd(rsync_video_cmd, desc=f"Загрузка {len(videos)} видеофайлов на {args.host}:{remote_video_dest}")
        if ret != 0:
            logger.error("Ошибка при передаче видеофайлов (код %d)", ret)
            sys.exit(1)
        logger.info("Загрузка видео завершена за %.1f сек.", time.time() - t0_up)

    # 5. Запуск инференса RT-DETR на удаленном GPU
    if not args.skip_detect:
        logger.info("\n--- ШАГ: Запуск удаленной детекции RT-DETR на GPU ---")
        
        day_param = ""
        if args.day:
            day_param = f"--day {args.day.replace('-', '').strip()}"
        elif args.input and "2026" in args.input:
            match = re.search(r"(\d{8})", args.input)
            if match:
                day_param = f"--day {match.group(1)}"

        overwrite_param = "--overwrite" if args.overwrite else ""
        remote_cmd = (
            f"cd {args.remote_dir} && "
            f"{args.remote_python} -u tools/remote_detector/worker.py "
            f"--input-dir data/video "
            f"{day_param} "
            f"--weights {args.weights} "
            f"--output-dir data/results "
            f"--batch-size {args.batch_size} "
            f"--imgsz {args.imgsz} "
            f"--conf {args.conf} "
            f"--iou {args.iou} "
            f"--detect-every-n {args.detect_every_n} "
            f"--workers {args.workers} "
            f"--device {args.device} "
            f"{overwrite_param}"
        )

        ssh_exec_cmd = ["ssh", "-t", args.host, remote_cmd]
        t0_detect = time.time()
        ret = run_streaming_cmd(ssh_exec_cmd, desc="Выполнение инференса RT-DETR на GPU")
        if ret != 0:
            logger.error("Ошибка выполнения детекции на удаленном сервере (код %d)", ret)
            sys.exit(1)
        logger.info("Удаленная детекция успешно завершена за %.1f сек.", time.time() - t0_detect)

    # 6. Скачивание результатов detections.json
    if not args.skip_download:
        logger.info("\n--- ШАГ: Скачивание detections.json в local data/results ---")
        local_results_dir = os.path.abspath("data/results")
        os.makedirs(local_results_dir, exist_ok=True)

        rsync_down_cmd = [
            "rsync",
            "-avz",
            "--progress",
            "--include=*/",
            "--include=detections.json",
            "--exclude=*",
            f"{args.host}:{args.remote_dir}/data/results/",
            f"{local_results_dir}/",
        ]
        ret = run_streaming_cmd(rsync_down_cmd, desc="Скачивание detections.json с сервера")
        if ret != 0:
            logger.error("Ошибка при скачивании результатов (код %d)", ret)
            sys.exit(1)

        # 7. Валидация и сводка
        logger.info("\n" + "=" * 75)
        logger.info("  РЕЗУЛЬТАТЫ ДЕТЕКЦИИ (data/results/):")
        logger.info("=" * 75)
        found_results = 0
        for root, dirs, files in os.walk(local_results_dir):
            if "detections.json" in files:
                json_fp = os.path.join(root, "detections.json")
                session_name = os.path.basename(root)
                meta = validate_detection_json(json_fp)
                if meta:
                    found_results += 1
                    logger.info(
                        "  [OK] Session %-16s | Кадров с людьми: %5d / %5d | Всего боксов: %6d | FPS: %.1f -> %s",
                        session_name,
                        meta["n_frames"],
                        meta["total_frames"],
                        meta["boxes"],
                        meta["fps"],
                        os.path.relpath(json_fp),
                    )
                else:
                    logger.warning("  [WARN] Session %-16s | Некорректный JSON -> %s", session_name, json_fp)

        logger.info("=" * 75)
        logger.info("Всего валидных сессий с детекциями: %d", found_results)
        logger.info("Готово к следующему этапу: ./venv/bin/python3 -m app.main --input %s --stage tracklets", args.input)
        logger.info("=" * 75)


if __name__ == "__main__":
    main()
