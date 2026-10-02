"""Writes ml/tests/fixtures/features.json: hands and their expected features (JS parity test)."""
import json
from pathlib import Path

import numpy as np

from gesturenet.features import features

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "features.json"
SIZES = [(1, 1), (640, 480), (1280, 720), (480, 640)]


def main() -> None:
    rng = np.random.default_rng(42)
    cases = []
    for i in range(50):
        pts = rng.uniform(0.0, 1.0, size=(21, 3))
        width, height = SIZES[i % len(SIZES)]
        f = features(pts, width, height)
        cases.append({"landmarks": pts.tolist(), "width": width, "height": height,
                      "expected": None if f is None else f.tolist()})
    cases.append({"landmarks": [[0.5, 0.5, 0.0]] * 21, "width": 1, "height": 1, "expected": None})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
