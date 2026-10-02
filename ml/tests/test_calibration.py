import io
from urllib.error import HTTPError

import pytest

from gesturenet.calibration import dominant_aspect, sample_image_sizes, viewer_get


def test_dominant_aspect_is_the_most_common_shape_and_its_share():
    sizes = [(512, 683)] * 6 + [(512, 686), (512, 910), (910, 512), (683, 512)]
    assert dominant_aspect(sizes) == (0.75, 0.7)


def test_dominant_aspect_needs_sizes():
    with pytest.raises(ValueError):
        dominant_aspect([])


def test_samples_evenly_spread_pages_of_a_split():
    asked = []

    def get(path):
        asked.append(path)
        if "length=1" in path:
            return {"num_rows_total": 1000}
        return {"rows": [{"row": {"image": {"width": 512, "height": 683}, "label": 3}}] * 2}

    sizes = sample_image_sizes(get, "validation", pages=4, page_rows=2, dataset="org/set")
    assert sizes == [(512, 683)] * 8
    assert asked[0] == "rows?dataset=org/set&config=default&split=validation&offset=0&length=1"
    assert [p.split("&offset=")[1] for p in asked[1:]] == ["125&length=2", "375&length=2",
                                                            "625&length=2", "875&length=2"]


def test_viewer_get_waits_out_a_rate_limit():
    opened, waits = [], []

    def urlopen(url, timeout):
        opened.append(url)
        if len(opened) == 1:
            raise HTTPError(url, 429, "Too Many Requests", {"Retry-After": "7"}, None)
        return io.BytesIO(b'{"ok": true}')

    assert viewer_get("rows?x", urlopen=urlopen, sleep=waits.append) == {"ok": True}
    assert opened == ["https://datasets-server.huggingface.co/rows?x"] * 2
    assert waits[0] == 7


def test_viewer_get_does_not_retry_a_missing_page():
    def urlopen(url, timeout):
        raise HTTPError(url, 404, "Not Found", {}, None)

    with pytest.raises(HTTPError):
        viewer_get("rows?x", urlopen=urlopen, sleep=lambda s: None)
