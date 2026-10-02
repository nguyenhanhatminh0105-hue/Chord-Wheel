import math

import numpy as np
import pytest

from gesturenet.baseline import classify, raised_fingers
from synthetic import synthetic_hand

CASES = {
    "fist": (),
    "palm": ("thumb", "index", "middle", "ring", "little"),
    "one": ("index",),
    "peace": ("index", "middle"),
    "three": ("index", "middle", "ring"),
    "four": ("index", "middle", "ring", "little"),
}


@pytest.mark.parametrize("pose,raised", CASES.items())
def test_each_pose(pose, raised):
    assert classify(synthetic_hand(raised)) == pose


def test_rotation_does_not_change_the_pose():
    p = synthetic_hand(("index", "middle"))
    t = math.radians(90)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    assert classify((p - p[0]) @ rot.T + p[0]) == "peace"


def test_a_thumb_folded_across_the_palm_is_down():
    p = synthetic_hand(("index", "middle", "ring", "little"))
    p[4] = np.array([0.5, 0.7]) + 0.1 * np.array([0.25, -0.55])   # tip over the ring finger's knuckle
    assert classify(p) == "four"


def test_unlisted_combination_is_other():
    assert classify(synthetic_hand(("thumb", "index"))) == "other"


def test_raised_fingers_names():
    assert raised_fingers(synthetic_hand(("thumb", "little"))) == frozenset({"thumb", "little"})


def test_respects_image_aspect():
    squashed = synthetic_hand(("index",))
    squashed[:, 1] *= 0.5           # stored as if the image were twice as tall
    assert classify(squashed, width=1.0, height=2.0) == "one"
