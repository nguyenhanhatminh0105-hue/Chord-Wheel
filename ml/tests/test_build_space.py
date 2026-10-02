import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("build_space", REPO / "ml" / "scripts" / "build_space.py")
build_space = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_space)


def test_the_space_holds_the_page_the_app_code_and_the_model(tmp_path):
    assert build_space.build(tmp_path / "space") == [
        "README.md", "classifier.js", "demo.js", "features.js", "gestures.onnx", "index.html", "labels.json",
        "smoother.js"]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_the_app_modules_load_side_by_side_as_page_scripts(tmp_path):
    # Page scripts share one global scope, as vm.runInThisContext scripts do; two files declaring the
    # same top-level const would fail to load.
    out = tmp_path / "space"
    build_space.build(out)
    script = ("const vm = require('vm'), fs = require('fs'); globalThis.window = globalThis;"
              "for (const f of ['features', 'classifier', 'smoother'])"
              "  vm.runInThisContext(fs.readFileSync(process.argv[1] + '/' + f + '.js', 'utf8'));"
              "const m = window.chordWheel;"
              "console.log(typeof m.features.exports.handFeatures, typeof m.classifier.exports.GestureClassifier,"
              "            typeof m.smoother.exports.PoseSmoother);")
    run = subprocess.run(["node", "-e", script, str(out)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert run.stdout.split() == ["function", "function", "function"]
