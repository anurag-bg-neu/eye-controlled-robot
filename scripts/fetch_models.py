"""Download the dlib 68-point shape predictor into models/.

The tracker uses the stock dlib model, so it is fetched from dlib.net rather
than committed. That keeps the repo at a few megabytes instead of a hundred.
"""

import argparse
import bz2
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

TARGET = MODELS_DIR / "sp.dat"
SIZE = 99_693_937
SHA256 = "fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f"

# Official source first, GitHub mirror second in case dlib.net is unreachable.
URLS = [
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2",
]


def sha256(path):
    """Hex digest of a file, read in chunks so the 96 MB model stays off the heap."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report(count, block, total):
    """urlretrieve progress hook."""
    if total > 0:
        done = min(count * block, total)
        sys.stdout.write(f"\r  {done / total:6.1%}  {done / 1e6:.1f} MB")
        sys.stdout.flush()


def download(url, archive):
    """Fetch the compressed model, returning True on success."""
    print(f"Downloading from {url}")
    try:
        urllib.request.urlretrieve(url, archive, reporthook=report)
        print()
        return True
    except OSError as error:
        print(f"\n  failed: {error}")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    if TARGET.exists() and not args.force and TARGET.stat().st_size == SIZE:
        print(f"{TARGET.name} already present and the right size, nothing to do")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    archive = TARGET.with_suffix(".dat.bz2")

    if not any(download(url, archive) for url in URLS):
        raise SystemExit("Every source failed. Check your connection and retry.")

    print("Decompressing")
    with bz2.open(archive, "rb") as source, TARGET.open("wb") as out:
        shutil.copyfileobj(source, out)
    archive.unlink()

    if sha256(TARGET) != SHA256:
        TARGET.unlink()
        raise SystemExit("Checksum mismatch, download discarded")

    print(f"Verified and saved to {TARGET}")


if __name__ == "__main__":
    main()
