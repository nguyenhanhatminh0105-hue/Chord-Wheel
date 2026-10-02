import numpy as np
import onnx
import onnxruntime as ort
import torch

from gesturenet.export import export_onnx, max_abs_diff
from gesturenet.model import GestureNet


def test_export_matches_pytorch(tmp_path):
    torch.manual_seed(0)
    net = GestureNet().eval()
    path = tmp_path / "g.onnx"
    export_onnx(net, path)
    onnx.checker.check_model(onnx.load(str(path)))
    feats = np.random.default_rng(0).normal(size=(1000, 42))
    assert max_abs_diff(net, path, feats) < 1e-5


def test_export_writes_nothing_to_stdout(tmp_path, capsys):
    # The exporter's progress lines contain emoji that a redirected Windows stdout (cp1252) cannot encode.
    export_onnx(GestureNet().eval(), tmp_path / "g.onnx")
    assert capsys.readouterr().out == ""


def test_onnx_io_names_and_dynamic_batch(tmp_path):
    path = tmp_path / "g.onnx"
    export_onnx(GestureNet().eval(), path)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["landmarks"]
    assert [o.name for o in sess.get_outputs()] == ["probs"]
    out = sess.run(["probs"], {"landmarks": np.zeros((3, 42), dtype=np.float32)})[0]
    assert out.shape == (3, 7) and np.allclose(out.sum(axis=1), 1.0, atol=1e-6)
