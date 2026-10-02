"""Read HaGRID v2 annotation files (design section 2).

Layout after ml/scripts/download_hagrid.py: <root>/<split>/<class>.json, each a dict from
image id to {"labels": [...], "hand_landmarks": [[[x, y] x 21] or [], ...], "user_id": str, ...};
hand_landmarks[i] belongs to labels[i].
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

SPLITS = ("train", "val", "test")
CALIBRATION_FILE = Path(__file__).with_name("hagrid_calibration.json")
# HaGRID label -> our class. The three-finger class is added from the calibration.
BASE_MAPPING = {
    "fist": "fist", "palm": "palm", "one": "one", "peace": "peace", "four": "four",
    "no_gesture": "other", "like": "other", "ok": "other", "rock": "other", "call": "other",
}
THREE_CANDIDATES = ("three", "three2", "three3")


@dataclass(frozen=True)
class Calibration:
    aspect: float = 1.0          # multiply HaGRID x by this for pixel-proportional coordinates
    three_class: str = "three"   # the HaGRID class whose raised fingers are index, middle and ring


def load_calibration(path: Path = CALIBRATION_FILE) -> Calibration:
    if not Path(path).exists():
        return Calibration()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Calibration(aspect=float(data["aspect"]), three_class=str(data["three_class"]))


def mapping(cal: Calibration) -> dict[str, str]:
    return {**BASE_MAPPING, cal.three_class: "three"}


@dataclass(frozen=True)
class Hand:
    points: np.ndarray   # (21, 2) image-relative x, y
    label: str           # one of gesturenet.CLASSES
    user_id: str
    image_id: str


def iter_hands(path: Path, hagrid_to_class: dict[str, str]) -> Iterator[Hand]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for image_id, ann in data.items():
        for label, pts in zip(ann.get("labels", []), ann.get("hand_landmarks", [])):
            cls = hagrid_to_class.get(label)
            if cls is None or not pts or len(pts) != 21:
                continue
            yield Hand(np.asarray(pts, dtype=np.float32)[:, :2], cls, str(ann["user_id"]), str(image_id))


def load_split(root: Path, split: str, hagrid_to_class: dict[str, str]) -> list[Hand]:
    """Every mapped hand in one split, from each <class>.json the mapping names."""
    hands: list[Hand] = []
    for name in sorted(hagrid_to_class):
        f = Path(root) / split / f"{name}.json"
        if f.exists():
            hands.extend(iter_hands(f, hagrid_to_class))
    return hands


def users_in_several_splits(splits: dict[str, list[Hand]]) -> set[str]:
    first_split: dict[str, str] = {}
    shared: set[str] = set()
    for split, hands in splits.items():
        for h in hands:
            if first_split.setdefault(h.user_id, split) != split:
                shared.add(h.user_id)
    return shared
