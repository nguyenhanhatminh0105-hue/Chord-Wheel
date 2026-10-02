"""Score the ONNX model and the rules baseline (design section 6); writes ml/reports/.
    python -m gesturenet.evaluate --hagrid ml/data/hagrid/annotations [--recordings ml/data/recordings]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from . import CLASSES, baseline  # noqa: E402
from .dataset import CLASS_INDEX, to_arrays, to_pixel_proportional  # noqa: E402
from .features import features_batch  # noqa: E402
from .hagrid import load_calibration, load_split, mapping  # noqa: E402
from .recordings import load_recordings  # noqa: E402
from .train import macro_f1  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "ml" / "reports"
MODEL = REPO / "models" / "gestures.onnx"
OTHER = CLASSES.index("other")


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int = len(CLASSES)) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion(y_true, y_pred)
    per_class = {}
    for k, name in enumerate(CLASSES):
        tp, predicted, actual = int(cm[k, k]), int(cm[:, k].sum()), int(cm[k, :].sum())
        per_class[name] = {"precision": tp / predicted if predicted else 0.0,
                           "recall": tp / actual if actual else 0.0, "support": actual}
    return {"accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
            "macro_f1": macro_f1(y_true, y_pred), "per_class": per_class, "confusion": cm.tolist()}


def model_predict(points_pp: np.ndarray, model_path: Path = MODEL) -> np.ndarray:
    """Pixel-proportional points in; degenerate hands are predicted 'other'."""
    feats, valid = features_batch(points_pp)
    pred = np.full(len(points_pp), OTHER, dtype=np.int64)
    if valid.any():
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        pred[valid] = sess.run(["probs"], {"landmarks": feats[valid].astype(np.float32)})[0].argmax(axis=1)
    return pred


def rules_predict(points_pp: np.ndarray) -> np.ndarray:
    return np.array([CLASSES.index(baseline.classify(p)) for p in points_pp], dtype=np.int64)


def plot_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    rows = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, rows, out=np.zeros(cm.shape), where=rows > 0)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    image = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j]:
                ax.text(j, i, f"{share[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if share[i, j] > 0.5 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, label="share of the true class")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def evaluate(name: str, points_pp: np.ndarray, labels: np.ndarray, model_path: Path = MODEL,
             reports: Path = REPORTS) -> dict:
    out = {"hands": int(len(labels))}
    for who, pred in (("model", model_predict(points_pp, model_path)), ("rules", rules_predict(points_pp))):
        rep = report(labels, pred)
        out[who] = rep
        plot_confusion(np.array(rep["confusion"]), f"{name}: {who}", Path(reports) / f"confusion_{name}_{who}.png")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hagrid", type=Path, required=True)
    ap.add_argument("--recordings", type=Path)
    args = ap.parse_args(argv)
    cal = load_calibration()
    pts, y = to_arrays(load_split(args.hagrid, "test", mapping(cal)))
    metrics = {"hagrid_test": evaluate("hagrid_test", to_pixel_proportional(pts, cal.aspect), y)}
    if args.recordings:
        recs = load_recordings(args.recordings)
        rec_pts = np.stack([r.points[:, :2] * np.array([r.width, r.height]) for r in recs])
        rec_y = np.array([CLASS_INDEX[r.label] for r in recs], dtype=np.int64)
        metrics["webcam"] = evaluate("webcam", rec_pts, rec_y)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for set_name, m in metrics.items():
        print(f"{set_name:12} hands {m['hands']:7d}   model acc {m['model']['accuracy']:.3f} "
              f"F1 {m['model']['macro_f1']:.3f}   rules acc {m['rules']['accuracy']:.3f} "
              f"F1 {m['rules']['macro_f1']:.3f}   model 'other' recall "
              f"{m['model']['per_class']['other']['recall']:.3f}")


if __name__ == "__main__":
    main()
