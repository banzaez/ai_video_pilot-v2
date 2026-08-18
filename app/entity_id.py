"""Пространства сущностей: t (tracklet), g (группа камеры), p (человек)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

EntitySpace = Literal["t", "g", "p"]
SPACES: frozenset[str] = frozenset({"t", "g", "p"})
_TOKEN = re.compile(r"^([tgp])(\d+)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True, order=True)
class EntityId:
    space: EntitySpace
    n: int

    def __post_init__(self) -> None:
        if self.space not in SPACES:
            raise ValueError(f"Неизвестное пространство id: {self.space!r}")
        if int(self.n) <= 0:
            raise ValueError(f"id должен быть > 0: {self.n}")
        object.__setattr__(self, "n", int(self.n))

    def format(self) -> str:
        return f"{self.space}{self.n}"

    def npz_key(self, model: str | None = None) -> str:
        base = f"{self.space}_{self.n}"
        if model:
            return f"{base}_{model}"
        return base

    def crop_stem(self) -> str:
        return f"{self.space}{self.n:04d}"

    def __str__(self) -> str:
        return self.format()


def tracklet(n: int) -> EntityId:
    return EntityId("t", n)


def group(n: int) -> EntityId:
    return EntityId("g", n)


def person(n: int) -> EntityId:
    return EntityId("p", n)


def parse(raw: str) -> EntityId:
    """Разбор канонической строки `t12` / `g3` / `p1`. Голое число — ошибка."""
    text = str(raw).strip()
    m = _TOKEN.match(text)
    if not m:
        raise ValueError(f"Ожидался EntityId вида t1/g1/p1, получено {raw!r}")
    space = m.group(1).lower()
    n = int(m.group(2))
    if space not in SPACES:
        raise ValueError(f"Неизвестное пространство id: {space!r}")
    return EntityId(space, n)  # type: ignore[arg-type]


def parse_optional(raw: Any) -> EntityId | None:
    if raw is None:
        return None
    try:
        return parse(str(raw))
    except (TypeError, ValueError):
        return None


def _positive_int(raw: Any) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def ids_from_detection(
    det: dict[str, Any],
    mapping: dict[int, int] | None = None,
) -> tuple[EntityId | None, EntityId | None]:
    """(group, tracklet) из детекции tracking.json / tracklet_frames.

    После remap ``track_id`` — группа (g), ``tracklet_id`` — фрагмент (t).
    ``tracklet_to_global`` применяется только к tracklet_id, никогда к track_id.
    """
    frag_n = _positive_int(det.get("tracklet_id"))
    track_n = _positive_int(det.get("track_id"))
    t_id = tracklet(frag_n) if frag_n else None

    gid_n = 0
    if t_id is not None and mapping:
        mapped = mapping.get(t_id.n)
        if mapped is not None:
            gid_n = _positive_int(mapped)
    if gid_n <= 0:
        gid_n = track_n
    if gid_n <= 0 and t_id is not None and not mapping:
        gid_n = t_id.n
    g_id = group(gid_n) if gid_n else None
    return g_id, t_id
