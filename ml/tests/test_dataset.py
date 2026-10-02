import numpy as np

from gesturenet import CLASSES
from gesturenet.baseline import classify
from gesturenet.dataset import augment, class_weights, make_features, to_arrays, to_pixel_proportional
from gesturenet.features import features
from gesturenet.hagrid import Hand
from synthetic import synthetic_hand


def test_to_arrays():
    hands = [Hand(synthetic_hand(()).astype(np.float32), "fist", "u", "i"),
             Hand(synthetic_hand(("index",)).astype(np.float32), "one", "u", "j")]
    pts, y = to_arrays(hands)
    assert pts.shape == (2, 21, 2) and pts.dtype == np.float64
    assert y.tolist() == [CLASSES.index("fist"), CLASSES.index("one")]


def test_pixel_proportional_scales_x_only_and_copies():
    p = np.ones((1, 21, 2))
    q = to_pixel_proportional(p, 0.5)
    assert np.allclose(q[..., 0], 0.5) and np.allclose(q[..., 1], 1.0) and np.allclose(p, 1.0)


def test_augment_is_seeded_and_changes_the_hands():
    p = np.stack([synthetic_hand(("index", "middle"))] * 4)
    a, b = augment(p, np.random.default_rng(1)), augment(p, np.random.default_rng(1))
    assert a.shape == p.shape and np.array_equal(a, b)
    assert not np.allclose(a, p - p[:, :1])


def test_augment_keeps_the_pose_readable():
    out = augment(np.stack([synthetic_hand(("index", "middle"))] * 400), np.random.default_rng(2))
    assert np.mean([classify(h) == "peace" for h in out]) >= 0.95


def test_augment_flips_about_half_the_hands():
    out = augment(np.stack([synthetic_hand(("thumb",))] * 2000), np.random.default_rng(3))
    flipped = np.mean(out[:, 4, 0] > 0)      # the unflipped raised thumb tip sits left of the wrist
    assert 0.45 < flipped < 0.55


def test_class_weights_are_inverse_frequency():
    assert np.allclose(class_weights(np.array([0, 1, 1, 1]), num_classes=3), [2.0, 4 / 6, 0.0])


def test_make_features_drops_degenerate_hands():
    pts = np.stack([synthetic_hand(()), np.full((21, 2), 0.5)])
    x, y = make_features(pts, np.array([0, 1]))
    assert x.shape == (1, 42) and x.dtype == np.float32 and y.tolist() == [0]
    assert np.allclose(x[0], features(synthetic_hand(())), atol=1e-6)
