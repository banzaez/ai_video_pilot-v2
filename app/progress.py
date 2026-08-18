"""Единый стиль tqdm-прогрессбаров для пайплайна."""

from __future__ import annotations

from typing import Any, Iterable, TypeVar

from tqdm import tqdm

T = TypeVar("T")


def make_pbar(
    iterable: Iterable[T] | None = None,
    *,
    total: int | None = None,
    desc: str,
    unit: str = "it",
    leave: bool = True,
    position: int | None = None,
    **kwargs: Any,
) -> Any:
    """Создаёт tqdm с едиными настройками (ncols, ascii-safe postfix)."""
    opts: dict[str, Any] = {
        "total": total,
        "desc": desc,
        "unit": unit,
        "leave": leave,
        "dynamic_ncols": True,
        "mininterval": 0.2,
        "smoothing": 0.05,
    }
    if position is not None:
        opts["position"] = position
    opts.update(kwargs)
    if iterable is None:
        return tqdm(**opts)
    return tqdm(iterable, **opts)
