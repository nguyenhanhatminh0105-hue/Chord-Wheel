import math

import numpy as np

from gesturenet.calibration import estimate_aspect
from synthetic import synthetic_hand


def rotated(p, deg):
    t = math.radians(deg)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return (p - p[0]) @ rot.T + p[0]


def test_recovers_a_known_aspect():
    rng = np.random.default_rng(0)
    stored = np.stack([rotated(synthetic_hand(("index",)), rng.uniform(0, 360)) for _ in range(3000)])
    stored[..., 0] /= 1.6           # as if x had been normalised by a 1.6x wider image
    assert abs(estimate_aspect(stored) - 1.6) <= 0.02


def test_falls_back_to_one_without_enough_sideways_hands():
    upright = np.stack([synthetic_hand(()) for _ in range(500)])
    assert estimate_aspect(upright) == 1.0
