"""Discovery camera-day sessions из плоских prod-имён в data/video/."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.info import VIDEO_EXTS, list_video_files

logger = logging.getLogger(__name__)

SESSION_PREFIX = "session:"
DAY_PREFIX = "day:"

# Camera_01_<source>_<started>_<ended>_<seg>
# source: nvr_local | 10.12.0.35_10.12.0.235 | любое другое без обязательного формата
# seg: tid1401s… | 3084159 | любой хвост
_PROD_STEM = re.compile(
    r"^Camera_(?P<idx>\d+)_(?P<source>.+)_"
    r"(?P<started>\d{14})_(?P<ended>\d{14})_"
    r"(?P<seg>.+)$",
    re.IGNORECASE,
)

_SESSION_KEY = re.compile(r"^\d{2,3}_\d{8}$")
_DAY_KEY = re.compile(r"^(\d{8}|\d{4}-\d{2}-\d{2})$")


def parse_session_input(raw: str) -> str | None:
    raw = str(raw or "").strip()
    if raw.startswith(SESSION_PREFIX):
        key = raw[len(SESSION_PREFIX) :].strip()
        return key if key else None
    if _SESSION_KEY.match(raw):
        return raw
    return None


def parse_day_input(raw: str) -> str | None:
    raw = str(raw or "").strip()
    if raw.startswith(DAY_PREFIX):
        val = raw[len(DAY_PREFIX) :].strip()
        clean = val.replace("-", "")
        if len(clean) == 8 and clean.isdigit():
            return clean
    if _DAY_KEY.match(raw):
        clean = raw.replace("-", "")
        if len(clean) == 8 and clean.isdigit():
            return clean
    return None


def is_lite_subdir(path: str) -> bool:
    norm = os.path.normpath(str(path))
    return os.path.basename(norm) == "lite" or norm.endswith(f"{os.sep}lite")


def _iso_from_nvr(raw: str) -> str | None:
    try:
        return datetime.strptime(raw, "%Y%m%d%H%M%S").isoformat(timespec="seconds")
    except ValueError:
        return None


def _day_from_started(started_raw: str) -> str | None:
    try:
        return datetime.strptime(started_raw[:8], "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


@dataclass(frozen=True)
class ParsedPart:
    path: str
    stem: str
    name: str
    camera_index: int
    started_raw: str
    ended_raw: str
    started_at: str
    ended_at: str
    day: str
    session_key: str


def session_key_from_part(camera_index: int, started_raw: str) -> str:
    day = started_raw[:8]
    return f"{int(camera_index):03d}_{day}"


def parse_prod_stem(stem: str) -> ParsedPart | None:
    m = _PROD_STEM.match(stem)
    if not m:
        return None
    gd = m.groupdict()
    idx = int(gd["idx"])
    started_raw = gd["started"]
    ended_raw = gd["ended"]
    started_at = _iso_from_nvr(started_raw)
    ended_at = _iso_from_nvr(ended_raw)
    day = _day_from_started(started_raw)
    if not started_at or not ended_at or not day:
        return None
    name = f"{stem}.mp4"
    return ParsedPart(
        path="",
        stem=stem,
        name=name,
        camera_index=idx,
        started_raw=started_raw,
        ended_raw=ended_raw,
        started_at=started_at,
        ended_at=ended_at,
        day=day,
        session_key=session_key_from_part(idx, started_raw),
    )


def discover_prod_parts(video_dir: str) -> list[ParsedPart]:
    """Сканирование video_dir, включая подпапки по датам (data/video/20260817/...)."""
    if not video_dir or not os.path.isdir(video_dir):
        return []
    out: list[ParsedPart] = []
    video_dir_abs = os.path.abspath(video_dir)
    for path in list_video_files(video_dir, recursive=True):
        stem = os.path.splitext(os.path.basename(path))[0]
        parsed = parse_prod_stem(stem)
        if not parsed:
            continue
        abs_path = os.path.abspath(path)
        try:
            rel_to_root = os.path.relpath(abs_path, os.getcwd()).replace("\\", "/")
        except ValueError:
            rel_to_root = path
        
        # Относительный путь от videoDir для URL в админке (/media/<rel_path>)
        try:
            rel_to_video = os.path.relpath(abs_path, video_dir_abs).replace("\\", "/")
        except ValueError:
            rel_to_video = os.path.basename(path)

        out.append(
            ParsedPart(
                path=rel_to_root,
                stem=parsed.stem,
                name=rel_to_video,
                camera_index=parsed.camera_index,
                started_raw=parsed.started_raw,
                ended_raw=parsed.ended_raw,
                started_at=parsed.started_at,
                ended_at=parsed.ended_at,
                day=parsed.day,
                session_key=parsed.session_key,
            )
        )
    return out


@dataclass
class Session:
    key: str
    camera_index: int
    day: str
    parts: list[ParsedPart] = field(default_factory=list)


def group_by_session_key(parts: list[ParsedPart]) -> list[Session]:
    by_key: dict[str, list[ParsedPart]] = {}
    for p in parts:
        by_key.setdefault(p.session_key, []).append(p)
    sessions: list[Session] = []
    for key in sorted(by_key):
        group = sorted(by_key[key], key=lambda p: p.started_raw)
        for i in range(1, len(group)):
            prev, cur = group[i - 1], group[i]
            if prev.ended_raw != cur.started_raw:
                logger.warning(
                    "Session %s: gap между частями %s → %s (ended=%s started=%s)",
                    key,
                    prev.stem,
                    cur.stem,
                    prev.ended_raw,
                    cur.started_raw,
                )
        first = group[0]
        sessions.append(
            Session(
                key=key,
                camera_index=first.camera_index,
                day=first.day,
                parts=group,
            )
        )
    return sessions


def discover_sessions(video_dir: str) -> list[Session]:
    return group_by_session_key(discover_prod_parts(video_dir))


def discover_days(video_dir: str) -> dict[str, list[Session]]:
    """Группирует сессии по дням (день = '20260401' или '2026-04-01')."""
    sessions = discover_sessions(video_dir)
    by_day: dict[str, list[Session]] = {}
    for s in sessions:
        clean = s.day.replace("-", "")
        by_day.setdefault(clean, []).append(s)
    return by_day


def frame_to_part(manifest: dict[str, Any], global_frame: int) -> tuple[dict[str, Any], int]:
    """global_frame 0-based → (part dict, local_frame 0-based)."""
    parts = manifest.get("parts") or []
    gf = int(global_frame)
    for part in parts:
        offset = int(part.get("frame_offset") or 0)
        count = int(part.get("frame_count") or 0)
        if offset <= gf < offset + count:
            return part, gf - offset
    if parts:
        last = parts[-1]
        offset = int(last.get("frame_offset") or 0)
        count = int(last.get("frame_count") or 0)
        local = max(0, min(gf - offset, max(0, count - 1)))
        return last, local
    raise ValueError(f"frame {global_frame} вне session manifest")


def resolve_sessions_for_input(raw: str, search_dir: str | None = None) -> tuple[str, list[Session], list[str]]:
    """(mode, sessions, legacy_files).

    mode: 'session' | 'day' | 'legacy'
    """
    raw = str(raw or "").strip()

    # Проверяем день: day:20260401 или 20260401
    day_key = parse_day_input(raw)
    if day_key:
        video_dir = search_dir or "data/video"
        all_sessions = discover_sessions(video_dir)
        hit = [s for s in all_sessions if s.day.replace("-", "") == day_key]
        if hit:
            return "day", sorted(hit, key=lambda s: s.camera_index), [day_key]
        # Если в video_dir нет файлов, но есть сохраненные сессии в data/results:
        from app.io.json_util import load_tracking_json
        results_root = "data/results"
        if os.path.isdir(results_root):
            recovered = []
            for name in os.listdir(results_root):
                info_path = os.path.join(results_root, name, "info.json")
                if os.path.isfile(info_path):
                    try:
                        info = load_tracking_json(info_path)
                        sess_day = str(info.get("day") or "").replace("-", "")
                        if sess_day == day_key:
                            cam_idx = int(info.get("camera_index") or 0)
                            recovered.append(Session(
                                key=name,
                                camera_index=cam_idx,
                                day=str(info.get("day")),
                                parts=[],
                            ))
                    except Exception:
                        pass
            if recovered:
                return "day", sorted(recovered, key=lambda s: s.camera_index), [day_key]
        raise ValueError(f"Сессии за день {day_key} не найдены")

    sk = parse_session_input(raw)
    if sk:
        video_dir = search_dir or "data/video"
        all_sessions = discover_sessions(video_dir)
        hit = [s for s in all_sessions if s.key == sk]
        if not hit:
            raise ValueError(f"Session не найдена: {sk}")
        return "session", hit, []

    if os.path.isfile(raw):
        if is_lite_subdir(os.path.dirname(raw)) or parse_prod_stem(os.path.splitext(os.path.basename(raw))[0]):
            # prod single file still goes legacy unless explicitly session:
            if parse_prod_stem(os.path.splitext(os.path.basename(raw))[0]):
                part = parse_prod_stem(os.path.splitext(os.path.basename(raw))[0])
                if part:
                    video_dir = search_dir or os.path.dirname(os.path.abspath(raw)) or "data/video"
                    sessions = discover_sessions(video_dir)
                    if sessions:
                        hit = [s for s in sessions if any(p.stem == part.stem for p in s.parts)]
                        if hit:
                            return "session", hit, []
            return "legacy", [], [raw]
        return "legacy", [], [raw]

    if os.path.isdir(raw):
        if is_lite_subdir(raw):
            return "legacy", [], list_video_files(raw)
        sessions = discover_sessions(raw)
        if sessions:
            return "session", sessions, []
        return "legacy", [], list_video_files(raw)

    if search_dir and os.path.isdir(search_dir):
        cand = os.path.join(search_dir, os.path.basename(raw))
        if os.path.isfile(cand):
            return resolve_sessions_for_input(cand, search_dir)

    raise ValueError(f"Видео не найдено: {raw}")

