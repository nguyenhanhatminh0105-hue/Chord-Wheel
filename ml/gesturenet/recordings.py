"""Read webcam recordings written by the app's recording mode (design section 2)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import CLASSES


@dataclass(frozen=True)
class Recorded:
    points: np.ndarray   # (21, 3) MediaPipe x, y, z
    label: str
    handedness: str
    width: int
    height: int
    session: str


def load_recordings(directory: Path) -> list[Recorded]:
    out: list[Recorded] = []
    for f in sorted(Path(directory).glob("*.jsonl")):
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("label") not in CLASSES or len(d.get("landmarks", [])) != 21:
                raise ValueError(f"{f.name}:{n}: bad record")
            out.append(Recorded(np.asarray(d["landmarks"], dtype=np.float64), d["label"], d["handedness"],
                                int(d["width"]), int(d["height"]), d["session"]))
    return out
