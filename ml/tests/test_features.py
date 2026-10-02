import math

import numpy as np
import pytest

from gesturenet.features import NUM_FEATURES, features, features_batch


def make_hand(seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.2, 0.8, size=(21, 2))
    pts[9] = pts[0] + np.array([0.02, -0.15])  # a clear wrist-to-knuckle vector
    return pts


def test_shape_wrist_and_knuckle_position():
    f = features(make_hand())
    assert f.shape == (NUM_FEATURES,)
    assert f[0] == pytest.approx(0.0) and f[1] == pytest.approx(0.0)      # wrist at the origin
    assert f[18] == pytest.approx(0.0, abs=1e-12) and f[19] == pytest.approx(-1.0)  # landmark 9 straight up


def test_invariant_to_translation_scale_and_rotation():
    p = make_hand(1)
    t = math.radians(35)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    moved = (p - p[0]) @ rot.T * 1.7 + np.array([0.31, -0.12])
    assert np.allclose(features(moved), features(p), atol=1e-12)


def test_mirroring_negates_x_features_only():
    p = make_hand(2)
    mirrored = p.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    a, b = features(p), features(mirrored)
    assert np.allclose(b[0::2], -a[0::2], atol=1e-12)
    assert np.allclose(b[1::2], a[1::2], atol=1e-12)


def test_pixel_proportional_coordinates():
    p = make_hand(3)
    scaled = p.copy()
    scaled[:, 0] *= 640
    scaled[:, 1] *= 480
    assert np.allclose(features(p, 640, 480), features(scaled), atol=1e-12)


def test_degenerate_hand_returns_none():
    assert features(np.full((21, 2), 0.5)) is None


def test_accepts_xyz_and_rejects_wrong_count():
    p3 = np.column_stack([make_hand(4), np.zeros(21)])
    assert np.allclose(features(p3), features(make_hand(4)))
    with pytest.raises(ValueError):
        features(np.zeros((20, 2)))


def test_batch_marks_invalid_rows():
    batch = np.stack([make_hand(5), np.full((21, 2), 0.5)])
    out, valid = features_batch(batch)
    assert valid.tolist() == [True, False]
    assert np.allclose(out[0], features(make_hand(5)))
    assert not out[1].any()
