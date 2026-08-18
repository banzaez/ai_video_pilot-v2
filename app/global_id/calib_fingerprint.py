"""Канонический фингерпринт калибровки камеры (зеркало admin/src/calibFingerprint.ts)."""

from __future__ import annotations

from typing import Any

FNV_OFFSET = 2166136261
FNV_PRIME = 16777619
DEFAULT_PERSON_H_M = 1.70


def _fmt(v: Any) -> str:
    try:
        return f"{float(v):.6f}"
    except (TypeError, ValueError):
        return "0.000000"


def _fmt_pt(pt: Any) -> str:
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        return "0.000000,0.000000"
    return f"{_fmt(pt[0])},{_fmt(pt[1])}"


def _fmt_size(size: Any) -> str:
    if not isinstance(size, (list, tuple)) or len(size) < 2:
        return "0,0"
    try:
        return f"{int(size[0])},{int(size[1])}"
    except (TypeError, ValueError):
        return "0,0"


def canonical_calib_string(
    camera_key: str,
    camera_doc: dict[str, Any] | None,
    torso_height_m: float,
    tracking_size: tuple[int, int] | list[int] | None,
    person_height_m: float = DEFAULT_PERSON_H_M,
) -> str:
    doc = camera_doc if isinstance(camera_doc, dict) else {}
    pl = doc.get("placement") if isinstance(doc.get("placement"), dict) else {}
    pos = pl.get("position") if isinstance(pl.get("position"), (list, tuple)) else (0, 0)
    lines = [
        "v2",
        f"camera_key={camera_key}",
        f"image_size={_fmt_size(doc.get('image_size'))}",
        f"tracking_size={_fmt_size(tracking_size)}",
        f"torso_height_m={_fmt(torso_height_m)}",
        f"person_height_m={_fmt(person_height_m)}",
        "placement="
        + "|".join(
            [
                _fmt_pt(pos),
                _fmt(pl.get("yaw_deg") or 0),
                _fmt(pl.get("fov_deg") or 70),
                _fmt(pl.get("height_m") if pl.get("height_m") is not None else 3),
                _fmt(pl.get("pitch_deg") if pl.get("pitch_deg") is not None else 35),
            ]
        ),
    ]
    h = doc.get("H")
    if isinstance(h, (list, tuple)) and len(h) >= 9:
        lines.append("H=" + ",".join(_fmt(x) for x in h[:9]))
    else:
        lines.append("H=")
    return "\n".join(lines)


def fnv1a32(text: str) -> str:
    h = FNV_OFFSET
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFF
    return f"{h:08x}"


def calib_fingerprint(
    camera_key: str,
    camera_doc: dict[str, Any] | None,
    torso_height_m: float,
    tracking_size: tuple[int, int] | list[int] | None,
    person_height_m: float = DEFAULT_PERSON_H_M,
) -> str:
    return fnv1a32(
        canonical_calib_string(camera_key, camera_doc, torso_height_m, tracking_size, person_height_m)
    )
