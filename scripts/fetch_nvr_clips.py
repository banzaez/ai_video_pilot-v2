#!/usr/bin/env python
"""Скачивание фрагментов с NVR по времени — без БД и пайплайна.

Использует тот же ISAPI-клиент, что и админка (кнопка «Загрузить видео»):
поиск ``/ISAPI/ContentMgmt/search`` → скачивание ``/ISAPI/ContentMgmt/download``.

Примеры::

    # что есть на 1 июня 10:20 по камерам 14 и 15 (только список)
    python scratch/fetch_nvr_clips.py --at "2026-06-01 10:20" --list

    # скачать их в текущий каталог
    python scratch/fetch_nvr_clips.py --at "2026-06-01 10:20"

    # другие две камеры на другом NVR
    python scratch/fetch_nvr_clips.py --at "2026-06-01 10:20" \
        --tracks 1601,1701 --host 1.2.3.4 --port 8003 --user admin --password secret

Время ``--at`` трактуется как время NVR (как его отдаёт поиск), без пересчёта
таймзон — ровно так же, как это делает ingest в пайплайне.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

env_file = REPO_ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from app.nvr_client import (  # noqa: E402
    HikvisionNvrClient,
    NvrApiError,
    RecordingSegment,
)


def canonical_nvr_video_filename(seg: RecordingSegment, camera_map: dict[str, str]) -> str:
    cam = camera_map.get(seg.track_id, f"Camera_{seg.track_id[:2]}")
    t_str = seg.start_time.strftime("%Y%m%d_%H%M%S")
    return f"{cam}_{t_str}.mp4"


def _parse_at(value: str) -> dt.datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            return dt.datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"не разобрал время {value!r}, ожидается 'YYYY-MM-DD HH:MM[:SS]'"
    )


def _covers(seg: RecordingSegment, at: dt.datetime) -> bool:
    start = seg.start_time.replace(tzinfo=None)
    end = seg.end_time.replace(tzinfo=None)
    return start <= at <= end


def _human_size(n: int | None) -> str:
    if not n:
        return "?"
    return f"{n / (1024 * 1024):.1f} MB"


def _build_client(args: argparse.Namespace) -> tuple[HikvisionNvrClient, dict[str, str]]:
    """Клиент + маппинг track_id→Camera_XX (из настроек, если не переопределено)."""
    host, port = args.host, args.port
    user, password = args.user, args.password
    camera_map: dict[str, str] = {}
    timeouts = dict(connect_timeout_sec=30.0, read_timeout_search_sec=120.0,
                    read_timeout_download_sec=1800.0)

    if not (host and user and password):
        from app.config.loader import settings_from_sources

        nv = settings_from_sources().nvr
        host = host or nv.host
        port = port or nv.port
        user = user or nv.username
        password = password or nv.password
        camera_map = dict(nv.track_camera_map)
        timeouts = dict(
            connect_timeout_sec=nv.connect_timeout_sec,
            read_timeout_search_sec=nv.read_timeout_search_sec,
            read_timeout_download_sec=max(nv.read_timeout_download_sec, 1800.0),
        )

    if args.no_camera_map:
        camera_map = {}

    client = HikvisionNvrClient(
        host=host, username=user, password=password, port=port or 80, **timeouts
    )
    return client, camera_map


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--at", type=_parse_at, required=True,
                   help="момент времени, который должен перекрывать фрагмент: 'YYYY-MM-DD HH:MM'")
    p.add_argument("--tracks", default="1401,1501",
                   help="track_id камер через запятую (по умолчанию 1401,1501 = камеры 14 и 15)")
    p.add_argument("--out", default=".", help="куда складывать файлы (по умолчанию текущий каталог)")
    p.add_argument("--window-min", type=int, default=90,
                   help="полуокно поиска в минутах вокруг --at (по умолчанию 90)")
    p.add_argument("--list", action="store_true", help="только показать найденное, не скачивать")
    p.add_argument("--all", action="store_true",
                   help="качать все фрагменты из окна поиска, а не только перекрывающие --at")
    p.add_argument("--no-camera-map", action="store_true",
                   help="не применять track_camera_map из конфига (имя будет Camera_14/Camera_15)")
    p.add_argument("--host", default=os.environ.get("NVR_HOST"))
    p.add_argument("--port", type=int, default=int(os.environ.get("NVR_PORT") or 0) or None)
    p.add_argument("--user", default=os.environ.get("NVR__USERNAME"))
    p.add_argument("--password", default=os.environ.get("NVR__PASSWORD"))
    args = p.parse_args()

    at: dt.datetime = args.at
    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    out_dir = pathlib.Path(args.out).resolve()
    start = at - dt.timedelta(minutes=args.window_min)
    end = at + dt.timedelta(minutes=args.window_min)

    client, camera_map = _build_client(args)
    print(f"NVR {client.base_url}  окно поиска {start:%Y-%m-%d %H:%M} .. {end:%Y-%m-%d %H:%M} "
          f"(цель {at:%Y-%m-%d %H:%M:%S})")

    selected: list[RecordingSegment] = []
    for tid in tracks:
        try:
            segments = client.search_recordings(tid, start, end)
        except NvrApiError as exc:
            print(f"  track {tid}: ОШИБКА поиска: {exc}")
            continue

        print(f"  track {tid}: найдено {len(segments)} фрагмент(ов) в окне")
        hit = 0
        for seg in segments:
            mark = "*" if _covers(seg, at) else " "
            print(f"    {mark} {seg.start_time:%Y-%m-%d %H:%M:%S} .. {seg.end_time:%H:%M:%S}"
                  f"  {seg.codec_type}  {_human_size(seg.size_bytes)}")
            if _covers(seg, at) or args.all:
                selected.append(seg)
                hit += 1
        if not hit:
            print(f"    track {tid}: НЕТ фрагмента, перекрывающего {at:%H:%M:%S} "
                  f"(возможно, смещение таймзоны — смотри времена выше)")

    if not selected:
        print("\nНечего качать.")
        return 1

    total = sum(s.size_bytes or 0 for s in selected)
    print(f"\nК загрузке: {len(selected)} файл(ов), ~{_human_size(total)} → {out_dir}")
    if args.list:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for i, seg in enumerate(selected, 1):
        filename = canonical_nvr_video_filename(seg, camera_map)
        print(f"[{i}/{len(selected)}] {filename} ...", flush=True)
        try:
            path = client.download_segment(seg, out_dir, filename=filename)
        except NvrApiError as exc:
            failed += 1
            print(f"    ОШИБКА: {exc}")
            continue
        print(f"    OK {_human_size(path.stat().st_size)}")

    print(f"\nГотово: {len(selected) - failed} ок, {failed} с ошибкой.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
