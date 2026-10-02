"""Hand landmarks to the 42 numbers the classifier sees (design section 3).

Must match gestures/features.js exactly; test/gestures.features.test.js checks it.
"""
from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
NUM_FEATURES = 2 * NUM_LANDMARKS
MIN_SCALE = 1e-6


def features_batch(points, width: float = 1.0, height: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """points: (N, 21, 2 or 3) image-relative landmarks. Returns (features (N, 42), valid (N,))."""
    p = np.array(points, dtype=np.float64)[..., :2]
    if p.ndim != 3 or p.shape[1:] != (NUM_LANDMARKS, 2):
        raise ValueError(f"expected (N, 21, 2) landmarks, got shape {np.shape(points)}")
    p[..., 0] *= width                      # 1. pixel-proportional coordinates
    p[..., 1] *= height
    p -= p[:, :1, :]                        # 2. wrist at the origin
    vx, vy = p[:, 9, 0], p[:, 9, 1]
    s = np.hypot(vx, vy)
    valid = s >= MIN_SCALE                  # NaN compares False, so NaN hands are invalid too
    s = np.where(valid, s, 1.0)[:, None]
    x, y = p[..., 0], p[..., 1]
    rx = (-vy[:, None] * x + vx[:, None] * y) / s     # 3. rotate landmark 9 straight up
    ry = (-vx[:, None] * x - vy[:, None] * y) / s
    out = np.empty((p.shape[0], NUM_FEATURES))
    out[:, 0::2] = rx / s                   # 4. scale so landmark 9 sits at (0, -1)
    out[:, 1::2] = ry / s
    out[~valid] = 0.0
    return out, valid


def features(points, width: float = 1.0, height: float = 1.0) -> np.ndarray | None:
    """One hand: 21 (x, y[, z]) landmarks. Returns 42 features, or None for a degenerate hand."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != NUM_LANDMARKS:
        raise ValueError(f"expected 21 landmarks, got shape {p.shape}")
    out, valid = features_batch(p[None, ...], width, height)
    return out[0] if valid[0] else None
