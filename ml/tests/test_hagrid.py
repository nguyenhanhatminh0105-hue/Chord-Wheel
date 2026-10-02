import json

import pytest

from gesturenet.hagrid import (BASE_MAPPING, Calibration, iter_hands, load_calibration, load_split, mapping,
                               users_in_several_splits)
from synthetic import synthetic_hand

PALM = ("thumb", "index", "middle", "ring", "little")


def write_split(root, split, cls, images):
    d = root / split
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{cls}.json").write_text(json.dumps(images), encoding="utf-8")


def image(labels, user, hands):
    return {"bboxes": [[0, 0, 1, 1]] * len(labels), "labels": labels, "united_bbox": None,
            "united_label": None, "user_id": user, "hand_landmarks": hands, "meta": {}}


@pytest.fixture
def mini(tmp_path):
    fist, palm = synthetic_hand(()).tolist(), synthetic_hand(PALM).tolist()
    write_split(tmp_path, "train", "fist", {"a1": image(["fist", "no_gesture"], "u1", [fist, palm]),
                                             "a2": image(["fist"], "u2", [[]])})       # no landmarks: skipped
    write_split(tmp_path, "train", "like", {"b1": image(["like"], "u3", [fist])})
    write_split(tmp_path, "train", "hand_heart", {"c1": image(["hand_heart"], "u4", [fist])})  # unmapped
    write_split(tmp_path, "test", "palm", {"d1": image(["palm"], "u9", [palm])})
    return tmp_path


def test_iter_hands_pairs_labels_with_landmarks(mini):
    hands = list(iter_hands(mini / "train" / "fist.json", mapping(Calibration())))
    assert [(h.label, h.user_id, h.image_id) for h in hands] == [("fist", "u1", "a1"), ("other", "u1", "a1")]
    assert hands[0].points.shape == (21, 2)


def test_load_split_reads_mapped_files_only(mini):
    hands = load_split(mini, "train", mapping(Calibration()))
    assert sorted(h.label for h in hands) == ["fist", "other", "other"]


def test_out_of_vocabulary_gestures_map_to_other():
    for name in ("like", "ok", "rock", "call", "no_gesture"):
        assert BASE_MAPPING[name] == "other"


def test_three_class_comes_from_calibration():
    m = mapping(Calibration(three_class="three2"))
    assert m["three2"] == "three" and "three" not in m


def test_calibration_defaults_and_file(tmp_path):
    assert load_calibration(tmp_path / "missing.json") == Calibration()
    f = tmp_path / "cal.json"
    f.write_text(json.dumps({"aspect": 0.5625, "three_class": "three3"}))
    assert load_calibration(f) == Calibration(aspect=0.5625, three_class="three3")


def test_users_in_several_splits(mini):
    m = mapping(Calibration())
    assert users_in_several_splits({s: load_split(mini, s, m) for s in ("train", "test")}) == set()
    write_split(mini, "test", "fist", {"e1": image(["fist"], "u1", [synthetic_hand(()).tolist()])})
    assert users_in_several_splits({s: load_split(mini, s, m) for s in ("train", "test")}) == {"u1"}
