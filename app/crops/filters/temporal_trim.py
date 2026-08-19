"""Фильтр временного тримминга краев трека (Temporal Edge Trimming).

Исключает начальные и конечные кадры трека, если трек достаточно длинный,
для защиты от ошибок инициализации и завершения MOT-трекера.
"""

from __future__ import annotations

import logging
from typing import Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def trim_edge_items(
    items: Sequence[T],
    *,
    trim_start: int = 2,
    trim_end: int = 2,
    min_len: int = 8,
) -> list[T]:
    """Исключает первые trim_start и последние trim_end элементов из последовательности.

    Если длина последовательности меньше min_len, элементы не отбрасываются.
    Если после обрезки остается меньше 1 элемента, возвращается исходная последовательность.
    """
    n = len(items)
    if n < min_len or (trim_start <= 0 and trim_end <= 0):
        return list(items)

    start_idx = max(0, int(trim_start))
    end_idx = n - max(0, int(trim_end))

    if end_idx <= start_idx:
        return list(items)

    return list(items[start_idx:end_idx])
