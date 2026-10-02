import numpy as np

from gesturenet.features import features
from gesturenet.meanpose import mean_poses, plot_mean_poses, same_side
from synthetic import synthetic_hand


def test_a_hand_and_its_mirror_image_line_up():
    hand = synthetic_hand(("index",))
    mirrored = hand * np.array([-1.0, 1.0]) + np.array([1.0, 0.0])
    p = same_side(np.stack([features(hand), features(mirrored)]))
    np.testing.assert_allclose(p[0], p[1], atol=1e-9)


def test_mean_pose_of_each_class_and_nan_for_a_missing_one():
    fist, peace = features(synthetic_hand(())), features(synthetic_hand(("index", "middle")))
    means = mean_poses(np.stack([fist, fist, peace]), np.array([0, 0, 3]), num_classes=7)
    np.testing.assert_allclose(means[0], same_side(fist[None])[0])
    np.testing.assert_allclose(means[3], same_side(peace[None])[0])
    assert means.shape == (7, 21, 2) and np.isnan(means[1]).all()


def test_plot_writes_an_image(tmp_path):
    means = mean_poses(features(synthetic_hand(("index",)))[None], np.array([2]))
    plot_mean_poses(means, tmp_path / "poses.png")
    assert (tmp_path / "poses.png").stat().st_size > 1000
