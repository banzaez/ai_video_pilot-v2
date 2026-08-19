"""Stage 0: метаданные ролика → info.json (имя, путь, разбор имени, fps, кадры)."""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any
from urllib.parse import quote

import cv2

from app.io.video import VideoSource

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv"}

# Camera_01_10.12.0.35_10.12.0.235_20260401113050_20260401113550_3084159
_NVR_NAME = re.compile(
    r"^(?P<camera>Cam(?:era)?[_-]?(?P<camera_index>\d+))"
    r"_(?P<ip>\d{1,3}(?:\.\d{1,3}){3})"
    r"_(?P<peer_ip>\d{1,3}(?:\.\d{1,3}){3})"
    r"_(?P<started>\d{14})"
    r"_(?P<ended>\d{14})"
    r"_(?P<recording_id>\d+)$",
    re.IGNORECASE,
)
_CAM_ONLY = re.compile(
    r"^(?P<camera>Cam(?:era)?[_-]?(?P<camera_index>\d+))$",
    re.IGNORECASE,
)
# Camera_01_<source>_<started>_<ended>_<seg> — source: nvr_local | IP-пара | другое
_PROD_NVR = re.compile(
    r"^Camera_(?P<camera_index>\d+)_(?P<source>.+)_"
    r"(?P<started>\d{14})_(?P<ended>\d{14})_"
    r"(?P<seg>.+)$",
    re.IGNORECASE,
)


def _iso_from_nvr(raw: str) -> str | None:
    try:
        return datetime.strptime(raw, "%Y%m%d%H%M%S").isoformat()
    except ValueError:
        return None


def _fourcc_name(cap: cv2.VideoCapture) -> str | None:
    raw = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    if raw <= 0:
        return None
    name = "".join(chr((raw >> (8 * i)) & 0xFF) for i in range(4))
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    return name or None


def list_video_files(directory: str) -> list[str]:
    """Видеофайлы в папке, по имени (файлы, начинающиеся с '_' или '.', игнорируются)."""
    if not directory or not os.path.isdir(directory):
        return []
    out: list[str] = []
    for name in sorted(os.listdir(directory)):
        if name.startswith(("_", ".")):
            continue
        path = os.path.join(directory, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in VIDEO_EXTS:
            out.append(path)
    return out


def video_stem_of(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def resolve_video_jobs(raw: str, search_dir: str | None = None) -> tuple[str, list[str]]:
    """Папка роликов + список файлов к обработке.

    Файл — один ролик; папка — все видео в ней.
    """
    raw = (raw or "").strip()
    if raw == "0":
        raise ValueError("Нужен видеофайл: пайплайн только офлайн (detect → ByteTrack).")
    if os.path.isfile(raw):
        folder = os.path.dirname(os.path.abspath(raw)) or "."
        return folder, [raw]
    if os.path.isdir(raw):
        return raw, list_video_files(raw)
    if search_dir and os.path.isdir(search_dir):
        cand = os.path.join(search_dir, os.path.basename(raw))
        if os.path.isfile(cand):
            return search_dir, [cand]
    raise ValueError(f"Видео не найдено: {raw}")


def parse_video_name(stem: str) -> dict[str, Any]:
    """Разобрать имя файла камеры. Неизвестное имя → ok=false."""
    empty: dict[str, Any] = {
        "ok": False,
        "camera": None,
        "camera_index": None,
        "ip": None,
        "peer_ip": None,
        "started_at": None,
        "ended_at": None,
        "recording_id": None,
        "duration_sec": None,
    }
    m = _NVR_NAME.match(stem) or _PROD_NVR.match(stem) or _CAM_ONLY.match(stem)
    if not m:
        return empty
    gd = m.groupdict()
    started_raw = gd.get("started")
    ended_raw = gd.get("ended")
    if m.re is _PROD_NVR:
        idx_raw = gd.get("camera_index")
        cam = f"Camera_{int(idx_raw):03d}" if idx_raw else None
        started = _iso_from_nvr(started_raw) if started_raw else None
        ended = _iso_from_nvr(ended_raw) if ended_raw else None
        duration = None
        if started_raw and ended_raw:
            try:
                t0 = datetime.strptime(started_raw, "%Y%m%d%H%M%S")
                t1 = datetime.strptime(ended_raw, "%Y%m%d%H%M%S")
                duration = round((t1 - t0).total_seconds(), 3)
            except ValueError:
                duration = None
        return {
            "ok": True,
            "camera": cam,
            "camera_index": int(idx_raw) if idx_raw is not None else None,
            "ip": None,
            "peer_ip": None,
            "started_at": started,
            "ended_at": ended,
            "recording_id": gd.get("seg"),
            "duration_sec": duration,
        }
    started = _iso_from_nvr(started_raw) if started_raw else None
    ended = _iso_from_nvr(ended_raw) if ended_raw else None
    duration = None
    if started_raw and ended_raw:
        try:
            t0 = datetime.strptime(started_raw, "%Y%m%d%H%M%S")
            t1 = datetime.strptime(ended_raw, "%Y%m%d%H%M%S")
            duration = round((t1 - t0).total_seconds(), 3)
        except ValueError:
            duration = None
    idx_raw = gd.get("camera_index")
    return {
        "ok": True,
        "camera": gd.get("camera"),
        "camera_index": int(idx_raw) if idx_raw is not None else None,
        "ip": gd.get("ip"),
        "peer_ip": gd.get("peer_ip"),
        "started_at": started,
        "ended_at": ended,
        "recording_id": gd.get("recording_id"),
        "duration_sec": duration,
    }


def build_video_info(video_path: str) -> dict[str, Any]:
    if not os.path.isfile(video_path):
        raise ValueError(f"Видео не найдено: {video_path}")

    abs_path = os.path.abspath(video_path)
    name = os.path.basename(abs_path)
    stem = os.path.splitext(name)[0]
    try:
        stored = os.path.relpath(abs_path, os.getcwd()).replace("\\", "/")
    except ValueError:
        stored = name
    if os.path.isabs(stored) or stored.startswith(".."):
        candidate = os.path.join("data", "video", name).replace("\\", "/")
        stored = candidate if os.path.isfile(candidate) else name

    stat = os.stat(abs_path)
    mtime = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")

    with VideoSource(abs_path) as src:
        meta = src.meta
        codec = _fourcc_name(src.cap)
        bitrate = int(src.cap.get(cv2.CAP_PROP_BITRATE) or 0)

    fps = float(meta.fps or 0.0)
    frames = int(meta.frame_count or 0)
    duration = round(frames / fps, 3) if fps > 0 and frames > 0 else None

    return {
        "stage": "info",
        "name": name,
        "stem": stem,
        "path": stored,
        "url": "/media/" + quote(name),
        "size_bytes": int(stat.st_size),
        "mtime": mtime,
        "width": int(meta.width or 0),
        "height": int(meta.height or 0),
        "fps": round(fps, 3) if fps else 0.0,
        "frame_count": frames,
        "duration_sec": duration,
        "codec": codec,
        "bitrate": bitrate if bitrate > 0 else None,
        "parsed": parse_video_name(stem),
    }


def track_global_id(camera_index: int | None, track_id: int) -> str:
    """Короткий id без букв: Camera_01, track 12 → '01#12'."""
    cam = 0 if camera_index is None else int(camera_index)
    return f"{cam:02d}#{int(track_id)}"


def camera_meta(info: dict[str, Any] | None = None, *, stem: str | None = None) -> dict[str, Any]:
    """Поля камеры для шапки tracks/crops/similar/merge. Пустые не включаем."""
    parsed = dict((info or {}).get("parsed") or {})
    use_stem = (info or {}).get("stem") or stem
    if parsed.get("camera_index") is None and use_stem:
        parsed = parse_video_name(str(use_stem))
    out: dict[str, Any] = {}
    if parsed.get("camera"):
        out["camera"] = parsed["camera"]
    if parsed.get("camera_index") is not None:
        out["camera_index"] = int(parsed["camera_index"])
    if use_stem:
        out["stem"] = use_stem
    if parsed.get("started_at"):
        out["started_at"] = parsed["started_at"]
    return out


def resolve_camera_key(
    info: dict[str, Any] | None = None,
    *,
    stem: str | None = None,
    default: str = "001",
) -> str:
    """Возвращает ключ камеры в формате 3 цифр ('001', '046') или исходном ключе."""
    meta = camera_meta(info, stem=stem)
    idx = meta.get("camera_index")
    if idx is not None:
        return f"{int(idx):03d}"
    return default


def stamp_camera(payload: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """Шапка камеры + id на каждом треке в payload['tracks']."""
    payload.update(meta)
    idx = meta.get("camera_index")
    for tr in payload.get("tracks") or []:
        tid = tr.get("track_id")
        if tid is None:
            continue
        tr["id"] = track_global_id(idx, int(tid))
    return payload
