"""Writes test/fixtures/e2e_hands.json: the most confidently recognised HaGRID test hand per pose.
    python ml/scripts/make_e2e_hands.py ml/data/hagrid/annotations
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from gesturenet import CLASSES
from gesturenet.dataset import to_arrays, to_pixel_proportional
from gesturenet.features import features_batch
from gesturenet.hagrid import load_calibration, load_split, mapping

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "test" / "fixtures" / "e2e_hands.json"
POSES = ("fist", "palm", "peace")


def main(root: Path) -> None:
    cal = load_calibration()
    pts, y = to_arrays(load_split(root, "test", mapping(cal)))
    pts = to_pixel_proportional(pts, cal.aspect)
    feats, valid = features_batch(pts)
    sess = ort.InferenceSession(str(REPO / "models" / "gestures.onnx"), providers=["CPUExecutionProvider"])
    probs = np.zeros((len(pts), len(CLASSES)), dtype=np.float32)
    probs[valid] = sess.run(["probs"], {"landmarks": feats[valid].astype(np.float32)})[0]
    out = {}
    for name in POSES:
        k = CLASSES.index(name)
        idx = np.where((y == k) & valid)[0]
        best = idx[np.argmax(probs[idx, k])]
        hand = pts[best] - pts[best, 0]
        hand = hand / float(np.hypot(*hand[9])) * 0.12 + np.array([0.5, 0.6])   # wrist-to-knuckle 0.12
        out[name] = np.round(hand, 6).tolist()
        print(f"{name}: test hand {best}, model probability {probs[best, k]:.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
