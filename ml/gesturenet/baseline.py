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
    # A folded thumb crosses the palm toward the little finger, so its tip ends up nearer the little
    # finger's knuckle than its IP joint is; a raised thumb points away from it.
    if dist(4, 17) > dist(3, 17):
        up.add("thumb")
    return frozenset(up)


def classify(points, width: float = 1.0, height: float = 1.0) -> str:
    return POSES.get(raised_fingers(points, width, height), "other")
