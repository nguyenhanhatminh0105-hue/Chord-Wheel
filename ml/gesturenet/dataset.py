"""Arrays, augmentation and class weights for training (design section 4)."""
from __future__ import annotations

import numpy as np

from . import CLASSES
from .features import features_batch
from .hagrid import Hand

CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}


def to_arrays(hands: list[Hand]) -> tuple[np.ndarray, np.ndarray]:
    if not hands:
        raise ValueError("no hands to convert")
    points = np.stack([h.points for h in hands]).astype(np.float64)
    labels = np.array([CLASS_INDEX[h.label] for h in hands], dtype=np.int64)
    return points, labels


def to_pixel_proportional(points: np.ndarray, aspect: float) -> np.ndarray:
    out = np.array(points, dtype=np.float64)
    out[..., 0] *= aspect
    return out


def augment(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random flip, rotation, scale, stretch and noise about the wrist; points are pixel-proportional."""
    p = np.array(points, dtype=np.float64)
    n = p.shape[0]
    p -= p[:, :1, :]
    p[rng.random(n) < 0.5, :, 0] *= -1                       # horizontal flip, p = 0.5
    theta = np.radians(rng.uniform(-15.0, 15.0, n))           # rotation +-15 degrees
    c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
    x, y = p[..., 0].copy(), p[..., 1].copy()
    p[..., 0], p[..., 1] = c * x - s * y, s * x + c * y
    p *= rng.uniform(0.9, 1.1, n)[:, None, None]              # scale +-10 %
    p[..., 0] *= rng.uniform(0.75, 1.25, n)[:, None]           # independent stretch +-25 %
    p[..., 1] *= rng.uniform(0.75, 1.25, n)[:, None]
    size = np.hypot(p[:, 9, 0], p[:, 9, 1])
    p += rng.normal(0.0, 1.0, p.shape) * (0.01 * size)[:, None, None]   # noise: 1 % of s
    return p


def class_weights(labels: np.ndarray, num_classes: int = len(CLASSES)) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = np.zeros(num_classes)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return weights


def make_features(points: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feats, valid = features_batch(points)
    return feats[valid].astype(np.float32), labels[valid]
