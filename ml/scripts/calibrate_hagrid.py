"""Estimate HaGRID's aspect factor, pick its three-finger class, check the splits are person-disjoint.
    python ml/scripts/calibrate_hagrid.py ml/data/hagrid/annotations   -> gesturenet/hagrid_calibration.json
"""
import json
import sys
from pathlib import Path

import numpy as np

from gesturenet.calibration import estimate_aspect, three_fraction
from gesturenet.hagrid import (BASE_MAPPING, CALIBRATION_FILE, SPLITS, THREE_CANDIDATES, Calibration,
                               load_split, mapping, users_in_several_splits)


def main(root: Path) -> None:
    gestures_only = {k: v for k, v in BASE_MAPPING.items() if v != "other"}
    points = np.stack([h.points for h in load_split(root, "train", gestures_only)])
    aspect = estimate_aspect(points)
    fractions = {name: three_fraction(root, name, aspect) for name in THREE_CANDIDATES}
    for name, frac in fractions.items():
        print(f"{name:7} index+middle+ring raised in {frac:6.1%} of its hands")
    three = max(fractions, key=fractions.get)
    m = mapping(Calibration(aspect=aspect, three_class=three))
    shared = users_in_several_splits({s: load_split(root, s, m) for s in SPLITS})
    if shared:
        raise SystemExit(f"{len(shared)} people appear in more than one split")
    CALIBRATION_FILE.write_text(json.dumps(
        {"aspect": aspect, "three_class": three, "hands_used_for_aspect": int(len(points))}, indent=2),
        encoding="utf-8")
    print(f"aspect {aspect:.2f}, three-finger class '{three}', splits person-disjoint -> {CALIBRATION_FILE}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
