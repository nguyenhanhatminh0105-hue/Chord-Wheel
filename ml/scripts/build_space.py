"""Assemble the Hugging Face Space in ml/data/space: the page, the app's gesture code, the model and labels.
    python ml/scripts/build_space.py
"""
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "ml" / "data" / "space"
COPIED = {
    "ml/space/index.html": "index.html",
    "ml/space/demo.js": "demo.js",
    "ml/space/README.md": "README.md",
    "models/gestures.onnx": "gestures.onnx",
    "models/labels.json": "labels.json",
}
APP_MODULES = ("features", "classifier", "smoother")   # gestures/<name>.js, unchanged inside a wrapper


def wrap(name: str, code: str) -> str:
    """A CommonJS module as a page script: page scripts share one global scope, so the module gets a
    function scope of its own, and its exports land at window.chordWheel.<name>.exports."""
    return (f"// gestures/{name}.js from Chord-Wheel, wrapped by ml/scripts/build_space.py.\n"
            f"(function (module) {{\n{code}\n}})((window.chordWheel = window.chordWheel || {{}}).{name} = "
            "{ exports: {} });\n")


def build(out: Path = OUT) -> list[str]:
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for src, dst in COPIED.items():
        shutil.copy(REPO / src, out / dst)
    for name in APP_MODULES:
        code = (REPO / "gestures" / f"{name}.js").read_text(encoding="utf-8")
        (out / f"{name}.js").write_text(wrap(name, code), encoding="utf-8")
    return sorted(p.name for p in out.iterdir())


if __name__ == "__main__":
    print(f"Space assembled in {OUT}: {build()}")
