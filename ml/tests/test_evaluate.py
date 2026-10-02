import json

import numpy as np
import pytest
import torch

from gesturenet import CLASSES
from gesturenet.evaluate import confusion, evaluate, plot_confusion, report
from gesturenet.export import export_onnx
from gesturenet.model import GestureNet
from gesturenet.recordings import load_recordings
from synthetic import synthetic_hand


def test_confusion_counts_true_rows_and_predicted_columns():
    cm = confusion(np.array([0, 0, 1]), np.array([0, 1, 1]), k=2)
    assert cm.tolist() == [[1, 1], [0, 1]]


def test_report_per_class_numbers():
    rep = report(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert rep["accuracy"] == 0.75
    assert rep["per_class"]["fist"] == {"precision": 1.0, "recall": 0.5, "support": 2}
    assert rep["per_class"]["palm"]["precision"] == pytest.approx(2 / 3)


def test_plot_writes_a_png(tmp_path):
    out = tmp_path / "cm.png"
    plot_confusion(np.eye(len(CLASSES), dtype=int) * 5, "test", out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_evaluate_scores_model_and_rules(tmp_path):
    torch.manual_seed(0)
    model = tmp_path / "g.onnx"
    export_onnx(GestureNet().eval(), model)
    pts = np.stack([synthetic_hand(()), synthetic_hand(("index",)), np.full((21, 2), 0.5)])
    labels = np.array([CLASSES.index("fist"), CLASSES.index("one"), CLASSES.index("other")])
    out = evaluate("toy", pts, labels, model_path=model, reports=tmp_path)
    assert out["hands"] == 3 and out["rules"]["accuracy"] == pytest.approx(2 / 3)
    assert set(out) == {"hands", "model", "rules"}
    assert (tmp_path / "confusion_toy_rules.png").exists()


def test_load_recordings(tmp_path):
    line = {"session": "s1", "t": 1, "label": "fist", "handedness": "Right", "width": 640, "height": 480,
            "landmarks": [[0.5, 0.5, 0.0]] * 21}
    (tmp_path / "s1.jsonl").write_text(json.dumps(line) + "\n\n" + json.dumps({**line, "label": "palm"}) + "\n")
    recs = load_recordings(tmp_path)
    assert [r.label for r in recs] == ["fist", "palm"]
    assert recs[0].points.shape == (21, 3) and recs[0].width == 640


def test_bad_record_names_the_line(tmp_path):
    bad = {"session": "s2", "t": 1, "label": "wave", "handedness": "Left", "width": 1, "height": 1,
           "landmarks": [[0, 0, 0]] * 21}
    (tmp_path / "s2.jsonl").write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="s2.jsonl:1"):
        load_recordings(tmp_path)
