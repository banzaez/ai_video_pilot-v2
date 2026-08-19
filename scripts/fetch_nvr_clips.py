#!/usr/bin/env python
"""Скачивание фрагментов с NVR по времени — без БД и пайплайна.

УЛИЦА
    На вход -> 033
    От входа -> 034
    Парковка -> 003
    Скамейки -> 004
    Ворота на парковку -> 007
    Калитка -> 008

ТЕХНИЧКА: 005
КОМНАТА ОЖИДАНИЯ/ОТДЫХА?: 006
СКЛАД -> 009
КУХНЯ -> 010
СЕЙФ -> 011
СЛУЖЕБНОЕ ПОМЕЩЕНИЕ? -> 012

РЕСЕПШЕН: 035, 036
ШОУ РУМ: 037, 038, 039, 040
ЗАЛ:
    БЕСЕДКИ? -> 001, 002
    ВХОД РЕСЕПШЕН -> 048
    ??? -> 017, 018, 019, 020, 021, 022
    ??? -> 023, 024, 025, 026, 027, 028
    ??? -> 029, 030, 031, 032, 045, 
    046, 047
СТОЙКА АДМИНИСТРАТОРА 
    СТОЛ ADM  -> 041
    ПРОХОД ЗА СТОЙКОЙ -> 042, 043, 044

ДЛЯ ТЕСТА
    046 и 047
    024 и 026 и 029


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

./venv/bin/python3 scripts/fetch_nvr_clips.py --at "2026-08-17 12:00" --window-min 30 --out ./downloads --port 8002 

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


def canonical_nvr_video_filename(
    seg: RecordingSegment, camera_map: dict[str, str]
) -> str:
    # Имя камеры: Camera_01, Camera_017 или по маппингу
    if seg.track_id in camera_map:
        cam = camera_map[seg.track_id]
    else:
        try:
            chan = int(seg.track_id) // 100
            cam = f"Camera_{chan:02d}"
        except ValueError:
            cam = f"Camera_{seg.track_id}"

    t_start = seg.start_time.strftime("%Y%m%d%H%M%S")
    t_end = seg.end_time.strftime("%Y%m%d%H%M%S")
    tid = seg.track_id

    # Формат совместим с пайплайном: Camera_<index>_<source/track>_<started>_<ended>_<seg>.mp4
    return f"{cam}_tid{tid}_{t_start}_{t_end}_seg.mp4"


def _parse_at(value: str) -> dt.datetime:
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
    ):
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


def _build_client(
    args: argparse.Namespace,
) -> tuple[HikvisionNvrClient, dict[str, str]]:
    """Клиент + маппинг track_id→Camera_XX (из настроек или NVR, если не переопределено)."""
    from app.config.loader import settings_from_sources

    nv = settings_from_sources().nvr
    host = args.host or nv.host
    port = args.port or nv.port
    user = args.user or nv.username
    password = args.password or nv.password
    camera_map = dict(nv.track_camera_map) if not args.no_camera_map else {}

    timeouts = dict(
        connect_timeout_sec=nv.connect_timeout_sec,
        read_timeout_search_sec=nv.read_timeout_search_sec,
        read_timeout_download_sec=max(nv.read_timeout_download_sec, 1800.0),
    )

    client = HikvisionNvrClient(
        host=host, username=user, password=password, port=port or 80, **timeouts
    )
    return client, camera_map


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--probe",
        action="store_true",
        help="опросить NVR и вывести список всех каналов/камер со статусом",
    )
    p.add_argument(
        "--at",
        type=_parse_at,
        default=None,
        help="момент времени, который должен перекрывать фрагмент: 'YYYY-MM-DD HH:MM'",
    )
    p.add_argument(
        "--tracks",
        default=None,
        help="track_id камер через запятую (по умолчанию опрашиваются все каналы NVR)",
    )
    p.add_argument(
        "--out",
        default=".",
        help="куда складывать файлы (по умолчанию текущий каталог)",
    )
    p.add_argument(
        "--window-min",
        type=int,
        default=60,
        help="полуокно поиска в минутах вокруг --at (по умолчанию 60)",
    )
    p.add_argument(
        "--list", action="store_true", help="только показать найденное, не скачивать"
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="качать все фрагменты из окна поиска, а не только перекрывающие --at",
    )
    p.add_argument(
        "--no-camera-map",
        action="store_true",
        help="не применять track_camera_map из конфига (использовать Camera_XX по каналу)",
    )
    p.add_argument(
        "--concurrency",
        "-j",
        type=int,
        default=2,
        help="количество параллельных потоков скачивания (по умолчанию 2)",
    )
    p.add_argument("--host", default=os.environ.get("NVR_HOST"))
    p.add_argument(
        "--port", type=int, default=int(os.environ.get("NVR_PORT") or 0) or None
    )
    p.add_argument("--user", default=os.environ.get("NVR__USERNAME"))
    p.add_argument("--password", default=os.environ.get("NVR__PASSWORD"))
    args = p.parse_args()

    client, camera_map = _build_client(args)

    if args.probe:
        print(f"NVR {client.base_url}")
        channels = client.probe_channels()
        if not channels:
            print("Не удалось получить каналы NVR (проверьте порт, логин и пароль).")
            return 1
        print(f"{'ch':>2} {'track':>6} {'online':>7} {'ip':<15} {'name'}")
        for ch in channels:
            st = "true" if ch.online else "false"
            ip_str = ch.ip or "-"
            print(f"{ch.channel_id:>2} {ch.track_id:>6} {st:>7} {ip_str:<15} {ch.name}")
        return 0

    if not args.at:
        p.error("параметр --at обязателен, если не указан --probe")

    at: dt.datetime = args.at

    # Автоматически опрашиваем NVR и заполняем имена камер (Camera_001, Camera_017 и т.д.)
    probed_channels = []
    try:
        probed_channels = client.probe_channels()
        if probed_channels and not args.no_camera_map:
            for ch in probed_channels:
                cam_name = ch.name.replace(" ", "_")
                camera_map[ch.track_id] = cam_name
                camera_map[str(int(ch.track_id))] = cam_name
                camera_map[f"{ch.channel_id:04d}"] = cam_name
                camera_map[f"{ch.channel_id * 100 + 1:04d}"] = cam_name
    except Exception as exc:
        print(f"[warning] не удалось получить имена камер из NVR: {exc}")

    # Если треки не указаны явно — берем все онлайн-каналы (или 1..16 по умолчанию)
    if args.tracks:
        tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    else:
        if probed_channels:
            tracks = [ch.track_id for ch in probed_channels if ch.online]
            if not tracks:
                tracks = [ch.track_id for ch in probed_channels]
        else:
            tracks = [f"{i * 100 + 1}" for i in range(1, 17)]

    out_dir = pathlib.Path(args.out).resolve()
    start = at - dt.timedelta(minutes=args.window_min)
    end = at + dt.timedelta(minutes=args.window_min)

    print(
        f"NVR {client.base_url}  окно поиска {start:%Y-%m-%d %H:%M} .. {end:%Y-%m-%d %H:%M} "
        f"(цель {at:%Y-%m-%d %H:%M:%S})"
    )

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
            print(
                f"    {mark} {seg.start_time:%Y-%m-%d %H:%M:%S} .. {seg.end_time:%H:%M:%S}"
                f"  {seg.codec_type}  {_human_size(seg.size_bytes)}"
            )
            if _covers(seg, at) or args.all:
                selected.append(seg)
                hit += 1
        if not hit:
            print(
                f"    track {tid}: НЕТ фрагмента, перекрывающего {at:%H:%M:%S} "
                f"(возможно, смещение таймзоны — смотри времена выше)"
            )

    if not selected:
        print("\nНечего качать.")
        return 1

    total = sum(s.size_bytes or 0 for s in selected)
    concurrency = max(1, args.concurrency)
    print(
        f"\nК загрузке: {len(selected)} файл(ов), ~{_human_size(total)} → {out_dir} "
        f"(потоков: {concurrency})"
    )
    if args.list:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    def _download_task(idx: int, seg: RecordingSegment, pos: int) -> bool:
        filename = canonical_nvr_video_filename(seg, camera_map)
        target_path = out_dir / filename
        if target_path.exists():
            tqdm.write(
                f"[{idx}/{len(selected)}] {filename} — уже существует, пропускаем."
            )
            return True

        total_bytes = seg.size_bytes or None
        with tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"[{idx}/{len(selected)}] {filename}",
            position=pos,
            leave=True,
        ) as pbar:
            try:
                client.download_segment(
                    seg,
                    out_dir,
                    filename=filename,
                    progress_callback=pbar.update,
                )
                return True
            except NvrApiError as exc:
                pbar.write(f"[{idx}/{len(selected)}] {filename} — ОШИБКА: {exc}")
                return False

    failed = 0
    if concurrency == 1 or len(selected) == 1:
        for i, seg in enumerate(selected, 1):
            ok = _download_task(i, seg, pos=0)
            if not ok:
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(_download_task, i, seg, pos=i - 1)
                for i, seg in enumerate(selected, 1)
            ]
            for f in futures:
                if not f.result():
                    failed += 1

    print(f"\nГотово: {len(selected) - failed} ок, {failed} с ошибкой.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
