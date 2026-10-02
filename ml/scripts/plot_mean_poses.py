"""Plot each class's mean normalised pose over HaGRID's training split -> docs/mean-poses.png
    python ml/scripts/plot_mean_poses.py ml/data/hagrid/annotations
"""
import sys
from pathlib import Path

from gesturenet.dataset import make_features, to_arrays, to_pixel_proportional
from gesturenet.hagrid import load_calibration, load_split, mapping
from gesturenet.meanpose import mean_poses, plot_mean_poses

OUT = Path(__file__).resolve().parents[2] / "docs" / "mean-poses.png"


def main(root: Path) -> None:
    cal = load_calibration()
    points, labels = to_arrays(load_split(root, "train", mapping(cal)))
    feats, labels = make_features(to_pixel_proportional(points, cal.aspect), labels)
    plot_mean_poses(mean_poses(feats, labels), OUT)
    print(f"wrote {OUT} from {len(labels)} training hands")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
