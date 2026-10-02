"""Export the trained network to ONNX and check it against PyTorch (design section 5).
    python -m gesturenet.export --hagrid ml/data/hagrid/annotations [--weights ml/artifacts/gestures.pt]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from . import CLASSES
from .dataset import make_features, to_arrays, to_pixel_proportional
from .features import NUM_FEATURES
from .hagrid import load_calibration, load_split, mapping
from .model import GestureNet, WithSoftmax

REPO = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO / "models" / "gestures.onnx"
LABELS_PATH = REPO / "models" / "labels.json"


OPSET = 18   # the torch.export-based exporter (PyTorch's default since 2.9) starts at opset 18


def export_onnx(net: GestureNet, path: Path, opset: int = OPSET) -> None:
    wrapped = WithSoftmax(net.eval()).eval()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    batch = torch.export.Dim("N", min=1)
    # A batch of 2 in the example input keeps the exported batch dimension dynamic. verbose=False
    # silences progress lines whose emoji a redirected Windows stdout cannot encode.
    torch.onnx.export(wrapped, (torch.zeros(2, NUM_FEATURES),), str(path), dynamo=True,
                      opset_version=opset, input_names=["landmarks"], output_names=["probs"],
                      dynamic_shapes={"x": {0: batch}}, external_data=False, verbose=False)


def max_abs_diff(net: GestureNet, path: Path, feats: np.ndarray) -> float:
    x = np.asarray(feats, dtype=np.float32)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(["probs"], {"landmarks": x})[0]
    with torch.no_grad():
        want = torch.softmax(net.eval()(torch.from_numpy(x)), dim=-1).numpy()
    return float(np.max(np.abs(got - want)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hagrid", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=Path("ml/artifacts/gestures.pt"))
    args = ap.parse_args(argv)
    net = GestureNet()
    net.load_state_dict(torch.load(args.weights, map_location="cpu"))
    net.eval()
    export_onnx(net, MODEL_PATH)
    LABELS_PATH.write_text(json.dumps(list(CLASSES)), encoding="utf-8")
    cal = load_calibration()
    pts, y = to_arrays(load_split(args.hagrid, "test", mapping(cal)))
    feats, _ = make_features(to_pixel_proportional(pts, cal.aspect), y)
    pick = np.random.default_rng(0).choice(len(feats), size=min(1000, len(feats)), replace=False)
    diff = max_abs_diff(net, MODEL_PATH, feats[pick])
    print(f"wrote {MODEL_PATH} ({MODEL_PATH.stat().st_size} bytes); "
          f"max |onnx - torch| = {diff:.2e} on {len(pick)} test hands")
    if diff >= 1e-5:
        raise SystemExit("ONNX output differs from PyTorch")


if __name__ == "__main__":
    main()
