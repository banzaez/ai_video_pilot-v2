"""JSON I/O: tracking/debug артефакты и merge-кэш."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from app.io.video import ensure_parent_dir

try:
    import orjson

    _HAS_ORJSON = True
except ImportError:
    _HAS_ORJSON = False


def load_json(path: str) -> dict[str, Any]:
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as exc:
        raise ValueError(f"Не удалось прочитать JSON: {path} ({exc})") from exc
    try:
        if _HAS_ORJSON:
            data = orjson.loads(raw)
        else:
            data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Битый JSON: {path} ({exc})") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Ожидали JSON-объект в {path}")
    return data


def _atomic_write_bytes(path: str, data: bytes) -> None:
    ensure_parent_dir(path)
    parent = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_json(path: str, payload: dict[str, Any], *, indent: int | None = None) -> None:
    if _HAS_ORJSON:
        opts = orjson.OPT_NON_STR_KEYS
        if indent is not None:
            opts |= orjson.OPT_INDENT_2
        raw = orjson.dumps(payload, option=opts)
    else:
        if indent is None:
            text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        else:
            text = json.dumps(payload, ensure_ascii=False, indent=indent)
        raw = text.encode("utf-8")
    _atomic_write_bytes(path, raw)


def load_json_cache(path: str) -> dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_json_cache(path: str, data: dict[str, Any]) -> None:
    if not path:
        return
    raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
    _atomic_write_bytes(path, raw)


# Совместимость со старыми именами
load_tracking_json = load_json
save_debug_json = save_json
