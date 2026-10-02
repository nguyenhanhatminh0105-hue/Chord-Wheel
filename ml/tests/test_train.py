import numpy as np
import pytest
import torch

from gesturenet import CLASSES
from gesturenet.model import GestureNet, WithSoftmax
from gesturenet.train import TrainConfig, macro_f1, train
from synthetic import synthetic_hand

POSES = {"fist": (), "palm": ("thumb", "index", "middle", "ring", "little"), "one": ("index",),
         "peace": ("index", "middle"), "three": ("index", "middle", "ring"),
         "four": ("index", "middle", "ring", "little"), "other": ("thumb", "index")}


def dataset(n_per_class, seed):
    rng = np.random.default_rng(seed)
    pts, ys = [], []
    for k, name in enumerate(CLASSES):
        for _ in range(n_per_class):
            pts.append(synthetic_hand(POSES[name]) + rng.normal(0, 0.002, (21, 2)))
            ys.append(k)
    return np.stack(pts), np.array(ys, dtype=np.int64)


def test_network_shape_and_size():
    net = GestureNet()
    assert net(torch.zeros(3, 42)).shape == (3, len(CLASSES))
    assert sum(p.numel() for p in net.parameters()) == 14215
    probs = WithSoftmax(net).eval()(torch.zeros(2, 42))
    assert torch.allclose(probs.sum(dim=1), torch.ones(2))


def test_macro_f1():
    y = np.array([0, 0, 1, 1])
    assert macro_f1(y, y, num_classes=2) == 1.0
    assert macro_f1(y, np.array([0, 1, 1, 1]), num_classes=2) == pytest.approx((2 / 3 + 0.8) / 2)


def test_training_learns_synthetic_poses():
    tr_p, tr_y = dataset(120, seed=0)
    va_p, va_y = dataset(20, seed=1)
    _, history = train(tr_p, tr_y, va_p, va_y, TrainConfig(epochs=30, patience=30, batch_size=64))
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert max(r["val_accuracy"] for r in history) >= 0.95


def test_training_is_reproducible():
    tr_p, tr_y = dataset(10, seed=0)
    va_p, va_y = dataset(5, seed=1)
    cfg = TrainConfig(epochs=3, patience=3, batch_size=32)
    assert train(tr_p, tr_y, va_p, va_y, cfg)[1] == train(tr_p, tr_y, va_p, va_y, cfg)[1]
