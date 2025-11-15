"""Leaf clustering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class LeafMeasurement:
    """Simple container that describes a detected leaf."""

    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    polygons: List[np.ndarray] | None = None


def _union_find(n: int):
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        root_x, root_y = find(x), find(y)
        if root_x == root_y:
            return
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1

    return parent, find, union


def cluster_leaves(
    leaves: Sequence[LeafMeasurement], distance_threshold: float
) -> List[List[int]]:
    """Cluster leaves based on Euclidean distance between centers."""

    if not leaves:
        return []

    n = len(leaves)
    parent, find, union = _union_find(n)

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(
                np.subtract(leaves[i].center, leaves[j].center, dtype=float)
            )
            if dist <= distance_threshold:
                union(i, j)

    clusters = {}
    for idx in range(n):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    return list(clusters.values())
