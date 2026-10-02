"""Train the gesture network on HaGRID (design section 4).
    python -m gesturenet.train --hagrid ml/data/hagrid/annotations [--out ml/artifacts/gestures.pt] [--track]
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from . import CLASSES
from .dataset import augment, class_weights, make_features, to_arrays, to_pixel_proportional
from .hagrid import load_calibration, load_split, mapping
from .model import GestureNet


@dataclass
class TrainConfig:
    epochs: int = 60
    patience: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    seed: int = 0


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = len(CLASSES)) -> float:
    scores = []
    for k in range(num_classes):
        tp = int(np.sum((y_pred == k) & (y_true == k)))
        fp = int(np.sum((y_pred == k) & (y_true != k)))
        fn = int(np.sum((y_pred != k) & (y_true == k)))
        if tp + fp + fn:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else 0.0


def predict(model: GestureNet, feats: np.ndarray, batch: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(feats), batch):
            out.append(model(torch.from_numpy(feats[i:i + batch])).argmax(dim=1).numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def train(train_points, train_labels, val_points, val_labels, cfg: TrainConfig | None = None,
          log: Callable[[dict], None] | None = None) -> tuple[GestureNet, list[dict]]:
    """Points are pixel-proportional (N, 21, 2). Returns the best model by validation macro-F1."""
    cfg = cfg or TrainConfig()
    set_seeds(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = GestureNet()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights(train_labels), dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    val_x, val_y = make_features(val_points, val_labels)
    best_f1, best_state, since_best, history = -1.0, None, 0, []
    for epoch in range(1, cfg.epochs + 1):
        x, y = make_features(augment(train_points, rng), train_labels)
        order = rng.permutation(len(x))
        x, y = x[order], y[order]
        model.train()
        total = 0.0
        for i in range(0, len(x), cfg.batch_size):
            xb, yb = torch.from_numpy(x[i:i + cfg.batch_size]), torch.from_numpy(y[i:i + cfg.batch_size])
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        val_pred = predict(model, val_x)
        row = {"epoch": epoch, "train_loss": total / max(1, len(x)),
               "val_accuracy": float(np.mean(val_pred == val_y)) if len(val_y) else 0.0,
               "val_macro_f1": macro_f1(val_y, val_pred)}
        history.append(row)
        if log:
            log(row)
        if row["val_macro_f1"] > best_f1:
            best_f1, since_best = row["val_macro_f1"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hagrid", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("ml/artifacts/gestures.pt"))
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--track", action="store_true", help="log every epoch with Trackio")
    args = ap.parse_args(argv)

    cal = load_calibration()
    m = mapping(cal)
    tr_p, tr_y = to_arrays(load_split(args.hagrid, "train", m))
    va_p, va_y = to_arrays(load_split(args.hagrid, "val", m))
    tr_p, va_p = to_pixel_proportional(tr_p, cal.aspect), to_pixel_proportional(va_p, cal.aspect)
    cfg = TrainConfig(epochs=args.epochs)
    print(f"train {len(tr_y)} hands, val {len(va_y)} hands, aspect {cal.aspect:.2f}")

    tracker = None
    if args.track:
        import trackio
        trackio.init(project="chord-wheel-gestures",
                     config={**asdict(cfg), "aspect": cal.aspect, "train_hands": int(len(tr_y))})
        tracker = trackio

    def report(row: dict) -> None:
        print(f"epoch {row['epoch']:2d}  loss {row['train_loss']:.4f}  "
              f"val acc {row['val_accuracy']:.4f}  val macro-F1 {row['val_macro_f1']:.4f}")
        if tracker:
            tracker.log(row)

    model, history = train(tr_p, tr_y, va_p, va_y, cfg, log=report)
    if tracker:
        tracker.finish()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    args.out.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved {args.out}; best val macro-F1 {max(r['val_macro_f1'] for r in history):.4f}")


if __name__ == "__main__":
    main()
