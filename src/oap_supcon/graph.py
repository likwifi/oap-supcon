"""Skeleton graph topology and ST-GCN spatial partitioning.

The reference encoder in `model.py` is permutation-equivariant over joints: it
applies the same per-joint function everywhere and averages, so it cannot tell a
left ankle from a right wrist. A part-level anatomical objective has nothing to
exploit under such a trunk. These helpers supply the skeleton connectivity that
graph and attention backbones need to represent anatomy.
"""
from __future__ import annotations

from collections import deque

import torch


def skeleton_edges(joints: int) -> list[tuple[int, int]]:
    """Undirected bone list for the supported joint formats."""
    if joints == 17:  # COCO-17
        return [
            (0, 1), (0, 2), (1, 3), (2, 4),          # head
            (0, 5), (0, 6), (5, 6),                   # neck / shoulders
            (5, 7), (7, 9), (6, 8), (8, 10),          # arms
            (5, 11), (6, 12), (11, 12),               # torso
            (11, 13), (13, 15), (12, 14), (14, 16),   # legs
        ]
    if joints == 18:  # OpenPose BODY-18
        return [
            (1, 0), (1, 2), (2, 3), (3, 4),
            (1, 5), (5, 6), (6, 7),
            (1, 8), (8, 9), (9, 10),
            (1, 11), (11, 12), (12, 13),
            (0, 14), (14, 16), (0, 15), (15, 17),
        ]
    if joints == 25:  # NTU / Kinect v2
        return [
            (0, 1), (1, 20), (20, 2), (2, 3),
            (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
            (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
            (0, 12), (12, 13), (13, 14), (14, 15),
            (0, 16), (16, 17), (17, 18), (18, 19),
        ]
    # Unknown format: fall back to a chain so the graph stays connected.
    return [(i, i + 1) for i in range(joints - 1)]


def center_joint(joints: int) -> int:
    """Body-centre joint used as the reference for spatial partitioning."""
    if joints == 17:
        return 11  # COCO has no pelvis joint; left hip is the closest anchor
    if joints == 18:
        return 1   # OpenPose neck
    if joints == 25:
        return 20  # Kinect spine-shoulder
    return 0


def _hop_distances(joints: int, edges: list[tuple[int, int]]) -> list[int]:
    """Breadth-first hop distance from the centre joint to every other joint."""
    neighbours: list[list[int]] = [[] for _ in range(joints)]
    for a, b in edges:
        neighbours[a].append(b)
        neighbours[b].append(a)
    distance = [-1] * joints
    root = center_joint(joints)
    distance[root] = 0
    queue = deque([root])
    while queue:
        current = queue.popleft()
        for neighbour in neighbours[current]:
            if distance[neighbour] < 0:
                distance[neighbour] = distance[current] + 1
                queue.append(neighbour)
    # A disconnected joint is treated as maximally distant rather than dropped.
    longest = max(distance)
    return [d if d >= 0 else longest + 1 for d in distance]


def spatial_adjacency(joints: int) -> torch.Tensor:
    """ST-GCN spatial partitioning as a [3, J, J] row-normalised tensor.

    Subset 0 is the joint itself, subset 1 its neighbours nearer the body
    centre (centripetal), subset 2 its neighbours further out (centrifugal).
    Splitting the neighbourhood this way is what lets a graph convolution treat
    inward and outward motion differently instead of averaging them.
    """
    edges = skeleton_edges(joints)
    distance = _hop_distances(joints, edges)
    adjacency = torch.zeros(3, joints, joints)
    for j in range(joints):
        adjacency[0, j, j] = 1.0
    for a, b in edges:
        for source, target in ((a, b), (b, a)):
            # target aggregates from source; subset by their relative depth.
            if distance[source] < distance[target]:
                adjacency[1, target, source] = 1.0
            elif distance[source] > distance[target]:
                adjacency[2, target, source] = 1.0
            else:
                adjacency[0, target, source] = 1.0
    # Row-normalise each subset so aggregation is a mean, not a sum: node
    # degrees vary across the skeleton and unnormalised sums would scale
    # features by connectivity.
    totals = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adjacency / totals
