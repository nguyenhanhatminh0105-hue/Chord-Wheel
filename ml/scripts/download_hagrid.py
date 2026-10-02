"""Download HaGRID v2's annotations (landmarks included, no images) and extract the class files used.
    python ml/scripts/download_hagrid.py        -> ml/data/hagrid/annotations/<split>/<class>.json
"""
import re
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/"
       "annotations_with_landmarks/annotations.zip")
WANTED = {"fist", "palm", "one", "peace", "three", "three2", "three3", "four",
          "no_gesture", "like", "ok", "rock", "call"}
ML = Path(__file__).resolve().parents[1]
ZIP = ML / "data" / "hagrid" / "annotations.zip"
OUT = ML / "data" / "hagrid" / "annotations"
MEMBER = re.compile(r"(?:^|/)(train|val|test)/([a-z0-9_]+)\.json$")


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        if dest.exists() and total and dest.stat().st_size == total:
            print(f"already downloaded: {dest}")
            return
        part, done = dest.with_suffix(".part"), 0
        with open(part, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r{done / total:6.1%} of {total / 1e6:.0f} MB", end="", flush=True)
        print()
    if total and done != total:
        raise SystemExit(f"incomplete download: {done} of {total} bytes")
    part.replace(dest)


def extract(zip_path: Path, out: Path) -> int:
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            m = MEMBER.search(info.filename)
            if not m or m.group(2) not in WANTED:
                continue
            target = out / m.group(1) / f"{m.group(2)}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
            n += 1
        if n == 0:
            raise SystemExit(f"no class files matched; first members: {[i.filename for i in zf.infolist()][:20]}")
    return n


if __name__ == "__main__":
    download(URL, ZIP)
    print(f"extracted {extract(ZIP, OUT)} files into {OUT}")
