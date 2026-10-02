"""Estimates that make HaGRID usable as pixel-proportional data (design section 3, step 1).

HaGRID stores x / image width and y / image height without the sizes, and nearly every hand in it
is upright, so the hands cannot reveal the aspect. HaGRID's 512-pixel release keeps each image's
proportions: the aspect factor is the most common width / height in a sample of those images, read
through the Hugging Face Dataset Viewer.
"""
from __future__ import annotations

import json
import time
import urllib.request
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError

from .baseline import raised_fingers
from .hagrid import iter_hands

VIEWER = "https://datasets-server.huggingface.co"
SIZES_DATASET = "testdummyvt/hagRIDv2_512px"   # HaGRID v2, shortest side 512, proportions kept
SIZES_SPLITS = ("train", "validation", "test")
RETRY_CODES = (429, 500, 502, 503, 504)


def viewer_get(path: str, urlopen=urllib.request.urlopen, sleep=time.sleep, tries: int = 8,
               pause: float = 4.0) -> dict:
    """GET a Dataset Viewer path as JSON, pausing after each answer and waiting out rate limits."""
    wait = 30.0
    for _ in range(tries):
        try:
            with urlopen(f"{VIEWER}/{path}", timeout=120) as r:
                data = json.load(r)
            sleep(pause)
            return data
        except HTTPError as e:
            if e.code not in RETRY_CODES:
                raise
            sleep(float(e.headers.get("Retry-After") or wait))
            wait = min(wait * 2, 240.0)
    raise RuntimeError(f"the Dataset Viewer kept refusing {path}")


def sample_image_sizes(get, split: str, pages: int = 8, page_rows: int = 100,
                       dataset: str = SIZES_DATASET) -> list[tuple[int, int]]:
    """(width, height) of the images on `pages` evenly spread pages of one split."""
    base = f"rows?dataset={dataset}&config=default&split={split}"
    total = get(f"{base}&offset=0&length=1")["num_rows_total"]
    sizes = []
    for i in range(pages):
        page = get(f"{base}&offset={int(total * (i + 0.5) / pages)}&length={page_rows}")
        sizes += [(r["row"]["image"]["width"], r["row"]["image"]["height"]) for r in page["rows"]]
    return sizes


def dominant_aspect(sizes: list[tuple[int, int]]) -> tuple[float, float]:
    """The most common width / height, to two decimals, and the share of images that have it."""
    ratios = Counter(round(w / h, 2) for w, h in sizes)
    if not ratios:
        raise ValueError("no image sizes")
    aspect, count = ratios.most_common(1)[0]
    return aspect, count / sum(ratios.values())


def three_fraction(root: Path, name: str, aspect: float) -> float:
    """Share of a HaGRID class's training hands whose raised fingers are exactly index, middle, ring."""
    f = Path(root) / "train" / f"{name}.json"
    if not f.exists():
        return 0.0
    hands = list(iter_hands(f, {name: "three"}))
    if not hands:
        return 0.0
    want = frozenset({"index", "middle", "ring"})
    hits = sum((raised_fingers(h.points, width=aspect) - {"thumb"}) == want for h in hands)
    return hits / len(hands)
