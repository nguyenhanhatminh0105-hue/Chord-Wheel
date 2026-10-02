"""Estimates that make HaGRID usable as pixel-proportional data (design section 3, step 1).

HaGRID stores x / image width and y / image height without the sizes. A hand's wrist-to-knuckle
length relative to its knuckle span does not depend on how the hand is turned, so the right aspect
factor is the one that makes upright and sideways hands agree on that ratio.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .baseline import raised_fingers
from .hagrid import iter_hands

ASPECT_GRID = np.round(np.arange(0.40, 2.5001, 0.01), 2)
MIN_HANDS = 200


def aspect_cost(points: np.ndarray, aspect: float) -> float:
    q = np.array(points, dtype=np.float64)
    q[..., 0] *= aspect
    v = q[:, 9] - q[:, 0]
    span = np.hypot(*(q[:, 17] - q[:, 5]).T)
    ratio = np.hypot(v[:, 0], v[:, 1]) / np.maximum(span, 1e-9)
    tilt = np.degrees(np.arctan2(np.abs(v[:, 0]), np.abs(v[:, 1])))   # 0 upright, 90 sideways
    upright, sideways = ratio[tilt < 20], ratio[tilt > 70]
    if len(upright) < MIN_HANDS or len(sideways) < MIN_HANDS:
        return float("nan")
    return abs(float(np.median(upright)) - float(np.median(sideways)))


def estimate_aspect(points: np.ndarray) -> float:
    costs = np.array([aspect_cost(points, a) for a in ASPECT_GRID])
    if np.all(np.isnan(costs)):
        print("too few upright or sideways hands to estimate the aspect; using 1.0")
        return 1.0
    return float(ASPECT_GRID[np.nanargmin(costs)])


def three_fraction(root: Path, name: str, aspect: float) -> float:
    """Share of a HaGRID class's training hands whose raised fingers are exactly index, middle, ring."""
    f = Path(root) / "train" / f"{name}.json"
    if not f.exists():
        return 0.0
    hands = list(iter_hands(f, {name: "three"}))
    if not hands:
        return 0.0
    want = frozenset({"index", "middle", "ring"})
    hits = sum((raised_fingers(h.points, width=aspect) - {"thumb"}) == want for h in hands)
    return hits / len(hands)
