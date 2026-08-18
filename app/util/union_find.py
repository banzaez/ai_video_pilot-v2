"""Структура Union-Find (система непересекающихся множеств) и жадная кластеризация."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import TypeVar

from app.util.intervals import intervals_overlap

K = TypeVar("K", bound=Hashable)
PairAllowed = Callable[[K, K], bool]


class _UF:
    def __init__(self, ids: list[K]) -> None:
        self.p: dict[K, K] = {i: i for i in ids}

    def find(self, x: K) -> K:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: K, b: K) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


UnionFind = _UF


def pair_key(a: K, b: K) -> tuple[K, K]:
    return (a, b) if a <= b else (b, a)  # type: ignore[operator]


def _overlap_forbidden(spans: dict[K, tuple[float, float]]) -> PairAllowed[K]:
    def allowed(x: K, y: K) -> bool:
        sx, sy = spans.get(x), spans.get(y)
        if sx is None or sy is None:
            return False
        return not intervals_overlap(sx[0], sx[1], sy[0], sy[1])

    return allowed


def greedy_groups(
    ids: list[K],
    edges: list[tuple[K, K, float]],
    spans: dict[K, tuple[float, float]],
    complete_link: bool = True,
    can_merge: PairAllowed[K] | None = None,
) -> list[list[K]]:
    """Жадное слияние пар с проверкой временных пересечений."""
    uf = _UF(ids)
    members: dict[K, set[K]] = {i: {i} for i in ids}
    linked = {pair_key(a, b) for a, b, _ in edges}
    pair_ok = can_merge or _overlap_forbidden(spans)

    def ok(a: K, b: K) -> bool:
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            return False
        for x in members[ra]:
            for y in members[rb]:
                if complete_link and pair_key(x, y) not in linked:
                    return False
                if not pair_ok(x, y):
                    return False
        return True

    for a, b, _score in sorted(edges, key=lambda e: -e[2]):
        if not ok(a, b):
            continue
        ra, rb = uf.find(a), uf.find(b)
        uf.union(a, b)
        root = uf.find(a)
        other = rb if root == ra else ra
        members[root] = members[ra] | members[rb]
        members[other] = set()

    groups = [sorted(s) for s in members.values() if len(s) >= 2]
    groups.sort(key=lambda g: (-len(g), g[0]))
    return groups
