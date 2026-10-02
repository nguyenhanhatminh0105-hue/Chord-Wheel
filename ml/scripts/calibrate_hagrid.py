"""Measure HaGRID's aspect factor, pick its three-finger class, check the splits are person-disjoint.
    python ml/scripts/calibrate_hagrid.py ml/data/hagrid/annotations   -> gesturenet/hagrid_calibration.json
Image sizes come from HaGRID's 512-pixel release through the Hugging Face Dataset Viewer (a few minutes).
"""
import json
import sys
from pathlib import Path

from gesturenet.calibration import (SIZES_DATASET, SIZES_SPLITS, dominant_aspect, sample_image_sizes,
                                    three_fraction, viewer_get)
from gesturenet.hagrid import (CALIBRATION_FILE, SPLITS, THREE_CANDIDATES, Calibration, load_split,
                               mapping, users_in_several_splits)


def main(root: Path) -> None:
    sizes = [size for split in SIZES_SPLITS for size in sample_image_sizes(viewer_get, split)]
    aspect, share = dominant_aspect(sizes)
    print(f"{len(sizes)} images sampled from {SIZES_DATASET}: width/height {aspect:.2f} in {share:.1%}")
    fractions = {name: three_fraction(root, name, aspect) for name in THREE_CANDIDATES}
    for name, frac in fractions.items():
        print(f"{name:7} index+middle+ring raised in {frac:6.1%} of its hands")
    three = max(fractions, key=fractions.get)
    m = mapping(Calibration(aspect=aspect, three_class=three))
    shared = users_in_several_splits({s: load_split(root, s, m) for s in SPLITS})
    if shared:
        raise SystemExit(f"{len(shared)} people appear in more than one split")
    CALIBRATION_FILE.write_text(json.dumps(
        {"aspect": aspect, "three_class": three, "aspect_share": round(share, 3),
         "images_sampled": len(sizes), "sizes_from": SIZES_DATASET}, indent=2) + "\n", encoding="utf-8")
    print(f"aspect {aspect:.2f}, three-finger class '{three}', splits person-disjoint -> {CALIBRATION_FILE}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
