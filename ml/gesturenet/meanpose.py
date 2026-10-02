"""Each class's mean normalised pose (design section 2): a check that every class has the expected shape."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import CLASSES  # noqa: E402

# MediaPipe's hand skeleton: thumb, index, middle, ring, little finger, and the palm's edges.
CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
               (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20))


def same_side(feats: np.ndarray) -> np.ndarray:
    """Feature rows as (N, 21, 2) points, mirrored where needed so every index knuckle is left of the
    little finger's: left and right hands then line up."""
    p = np.array(feats, dtype=np.float64).reshape(-1, 21, 2)
    p[p[:, 5, 0] > p[:, 17, 0], :, 0] *= -1
    return p


def mean_poses(feats: np.ndarray, labels: np.ndarray, num_classes: int = len(CLASSES)) -> np.ndarray:
    """(num_classes, 21, 2) mean of each class's same-side hands; NaN for a class without hands."""
    p = same_side(feats)
    out = np.full((num_classes, 21, 2), np.nan)
    for k in range(num_classes):
        if np.any(labels == k):
            out[k] = p[labels == k].mean(axis=0)
    return out


def plot_mean_poses(means: np.ndarray, path: Path, names=CLASSES) -> None:
    lo, hi = np.nanmin(means, axis=(0, 1)) - 0.15, np.nanmax(means, axis=(0, 1)) + 0.15
    fig, axes = plt.subplots(1, len(names), figsize=(1.5 * len(names), 2.6), dpi=150)
    for ax, name, pose in zip(axes, names, means):
        for a, b in CONNECTIONS:
            ax.plot(pose[[a, b], 0], pose[[a, b], 1], color="#1f4e79", linewidth=2, solid_capstyle="round")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(hi[1], lo[1])   # image convention: y grows downward, fingers point up
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
