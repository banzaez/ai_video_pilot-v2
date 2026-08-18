"""Версии артефактов стадий + отчёт «что пересчитать»."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

# Не импортируем app.io здесь — иначе цикл: artifact_meta ↔ io.export

# Порядок стадий пайплайна (как в CLI --from / --to)
VIDEO_STAGE_ORDER = (
    "info",
    "detect",
    "tracklets",
    "tracklet_reid",
    "tracklet_link",
    "track",
    "pose",
    "feet",
    "camera_face",
    "camera_link",
    "day_link",
)
STAGE_ORDER = VIDEO_STAGE_ORDER

# Основной JSON каждой стадии
STAGE_FILES: dict[str, str] = {
    "info": "info.json",
    "detect": "detections.json",
    "tracklets": "tracklet_frames.json",
    "tracklet_reid": "tracklet_reid.json",
    "tracklet_link": "tracklet_links.json",
    "track": "tracking.json",
    "pose": "poses.json",
    "feet": "feet.json",
    "camera_face": "camera_face.json",
    "camera_link": "camera_links.json",
    "day_link": "day_links.json",
}

# Прямой родитель (для inputs). info и detect — корни цепочки обработки.
STAGE_PARENT: dict[str, str | None] = {
    "info": None,
    "detect": None,
    "tracklets": "detect",
    "tracklet_reid": "tracklets",
    "tracklet_link": "tracklet_reid",
    "track": "tracklet_link",
    "pose": "track",
    "feet": "pose",
    "camera_face": "track",
    "camera_link": "feet",
    "day_link": None,
}

# Версия формата файла стадии (поднять при несовместимом изменении JSON)
STAGE_FILE_VERSION: dict[str, int] = {s: 1 for s in STAGE_ORDER}
STAGE_FILE_VERSION["day_link"] = 3

ARTIFACT_KEY = "artifact"


def _utc_now() -> str:
    # С микросекундами — иначе два сохранения в одну секунду не различить
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stage_json_path(work_dir: str, stage: str) -> str:
    name = STAGE_FILES.get(stage)
    if not name:
        raise ValueError(f"Неизвестная стадия: {stage}")
    return os.path.join(work_dir, name)


def read_artifact(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get(ARTIFACT_KEY)
    return raw if isinstance(raw, dict) else None


def _parent_ref(work_dir: str, parent: str) -> dict[str, Any] | None:
    path = stage_json_path(work_dir, parent)
    art = read_artifact(path)
    if art and art.get("written_at"):
        return {
            "stage": parent,
            "file_version": int(art.get("file_version") or STAGE_FILE_VERSION.get(parent, 1)),
            "written_at": str(art["written_at"]),
        }
    # Старый файл без artifact — фиксируем mtime/size как слабый отпечаток
    if os.path.isfile(path):
        st = os.stat(path)
        return {
            "stage": parent,
            "file_version": 0,
            "written_at": None,
            "mtime": round(st.st_mtime, 3),
            "size": int(st.st_size),
        }
    return None


def _resolve_parent(stage: str, work_dir: str) -> str | None:
    """Родитель стадии с fallback для direct mode."""
    parent = STAGE_PARENT.get(stage)
    if stage == "track" and parent == "tracklet_link":
        if not os.path.isfile(stage_json_path(work_dir, "tracklet_link")):
            parent = "detect"
    return parent


def build_artifact_meta(stage: str, work_dir: str) -> dict[str, Any]:
    """Метаданные для записи в JSON стадии."""
    stage = str(stage).lower().strip()
    if stage not in STAGE_FILES and stage != "tracks":
        raise ValueError(f"Неизвестная стадия для artifact: {stage}")
    meta: dict[str, Any] = {
        "stage": "track" if stage == "tracks" else stage,
        "file_version": int(STAGE_FILE_VERSION.get("track" if stage == "tracks" else stage, 1)),
        "written_at": _utc_now(),
    }
    parent_key = "track" if stage == "tracks" else stage
    parent = _resolve_parent(parent_key, work_dir)
    inputs: dict[str, Any] = {}
    if parent:
        ref = _parent_ref(work_dir, parent)
        if ref:
            inputs[parent] = ref
    meta["inputs"] = inputs
    return meta


def attach_artifact_meta(
    payload: dict[str, Any],
    *,
    stage: str,
    work_dir: str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    """Вшить artifact в payload (на месте) и вернуть его."""
    wd = work_dir or (os.path.dirname(path) if path else None)
    if not wd:
        raise ValueError("нужен work_dir или path")
    payload[ARTIFACT_KEY] = build_artifact_meta(stage, wd)
    return payload


def _ref_matches(recorded: dict[str, Any] | None, parent_path: str, parent_art: dict[str, Any] | None) -> bool:
    if not recorded:
        return False
    if parent_art and parent_art.get("written_at"):
        return (
            str(recorded.get("written_at") or "") == str(parent_art.get("written_at") or "")
            and int(recorded.get("file_version") or 0) == int(parent_art.get("file_version") or 0)
        )
    # Родитель без artifact: сверяем mtime/size, если ребёнок их запомнил
    if not os.path.isfile(parent_path):
        return False
    st = os.stat(parent_path)
    if recorded.get("mtime") is not None:
        return (
            abs(float(recorded["mtime"]) - st.st_mtime) < 0.01
            and int(recorded.get("size") or -1) == int(st.st_size)
        )
    # Ребёнок ссылался на written_at, а у родителя его нет → устарело / неизвестно
    return False


def stale_stages_report(work_dir: str) -> dict[str, Any]:
    """Какие стадии устарели относительно родителей + CLI подсказка."""
    stages: dict[str, Any] = {}
    stale: list[str] = []

    for stage in VIDEO_STAGE_ORDER:
        path = stage_json_path(work_dir, stage)
        exists = os.path.isfile(path)
        art = read_artifact(path) if exists else None
        entry: dict[str, Any] = {
            "stage": stage,
            "file": STAGE_FILES[stage],
            "exists": exists,
            "file_version": art.get("file_version") if art else None,
            "written_at": art.get("written_at") if art else None,
            "stale": False,
            "reason": None,
        }
        if not exists:
            stages[stage] = entry
            continue

        parent = _resolve_parent(stage, work_dir)
        if parent:
            parent_path = stage_json_path(work_dir, parent)
            if not os.path.isfile(parent_path):
                entry["stale"] = True
                entry["reason"] = f"нет родителя {STAGE_FILES[parent]}"
            else:
                parent_art = read_artifact(parent_path)
                recorded = (art or {}).get("inputs", {}).get(parent) if art else None
                if isinstance(recorded, dict) and _ref_matches(recorded, parent_path, parent_art):
                    pass
                elif art is None:
                    # Старый JSON без meta: эвристика по mtime
                    if os.path.getmtime(parent_path) > os.path.getmtime(path) + 0.01:
                        entry["stale"] = True
                        entry["reason"] = f"{STAGE_FILES[parent]} новее (mtime)"
                else:
                    entry["stale"] = True
                    entry["reason"] = f"не совпадает с {STAGE_FILES[parent]} — пересчитать"

        if entry["stale"]:
            stale.append(stage)
        stages[stage] = entry

    # Если родитель stale — все существующие потомки тоже помечаем
    stale_set = set(stale)
    for stage in VIDEO_STAGE_ORDER:
        parent = _resolve_parent(stage, work_dir)
        if parent and parent in stale_set and stages[stage]["exists"]:
            if stage not in stale_set:
                stages[stage]["stale"] = True
                stages[stage]["reason"] = stages[stage]["reason"] or f"устарел родитель {parent}"
                stale_set.add(stage)

    # Всегда в порядке пайплайна (каскад иначе дописывал хвост не по порядку)
    stale = [s for s in VIDEO_STAGE_ORDER if stages[s]["stale"]]

    recompute_from = stale[0] if stale else None
    recompute_to = stale[-1] if stale else None
    cli = None
    if recompute_from and recompute_to:
        if recompute_from == recompute_to:
            cli = f"python -m app.main --stage {recompute_from}"
        else:
            cli = f"python -m app.main --from {recompute_from} --to {recompute_to}"

    return {
        "stages": stages,
        "stale": stale,
        "recompute_from": recompute_from,
        "recompute_to": recompute_to,
        "cli": cli,
    }
