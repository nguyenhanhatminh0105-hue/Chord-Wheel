"""Finger-counting rules: the baseline the model must beat (design section 6)."""
from __future__ import annotations

import numpy as np

# finger -> (middle joint, tip) landmark indices
FINGERS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "little": (18, 20)}
POSES = {
    frozenset(): "fist",
    frozenset({"thumb", "index", "middle", "ring", "little"}): "palm",
    frozenset({"index"}): "one",
    frozenset({"index", "middle"}): "peace",
    frozenset({"index", "middle", "ring"}): "three",
    frozenset({"index", "middle", "ring", "little"}): "four",
}


def raised_fingers(points, width: float = 1.0, height: float = 1.0) -> frozenset[str]:
    p = np.array(points, dtype=np.float64)[:, :2]
    p[:, 0] *= width
    p[:, 1] *= height

    def dist(a: int, b: int) -> float:
        return float(np.hypot(*(p[a] - p[b])))

    up = {name for name, (pip, tip) in FINGERS.items() if dist(tip, 0) > dist(pip, 0)}
    if dist(4, 5) > dist(3, 5):        # thumb tip farther from the index knuckle than its IP joint
        up.add("thumb")
    return frozenset(up)


def classify(points, width: float = 1.0, height: float = 1.0) -> str:
    return POSES.get(raised_fingers(points, width, height), "other")
