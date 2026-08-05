#!/usr/bin/env python3
"""Download the required WM793b CSV files from the authors' repository.

The files are stored locally in data/ but are excluded from Git tracking.
No third-party file is relicensed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).resolve().parent

FILES = {
    "WM793b_outer.csv": {
        "url": (
            "https://raw.githubusercontent.com/"
            "ryanmurphy42/4DSpheroids_Murphy2021/main/"
            "Data%20files%20%28csv%29/"
            "WM793b_IncuCyte_confocal_outer.csv"
        ),
        "expected_lines": 7807,
    },
    "WM793b_inhibited.csv": {
        "url": (
            "https://raw.githubusercontent.com/"
            "ryanmurphy42/4DSpheroids_Murphy2021/main/"
            "Data%20files%20%28csv%29/"
            "WM793b_confocal_inhibited.csv"
        ),
        "expected_lines": 668,
    },
    "WM793b_necrotic.csv": {
        "url": (
            "https://raw.githubusercontent.com/"
            "ryanmurphy42/4DSpheroids_Murphy2021/main/"
            "Data%20files%20%28csv%29/"
            "WM793b_confocal_necrotic.csv"
        ),
        "expected_lines": 668,
    },
}


def download(name: str, url: str, expected_lines: int, force: bool) -> None:
    destination = DATA_DIR / name

    if destination.exists() and not force:
        print(f"Exists: {destination}")
        return

    request = Request(
        url,
        headers={"User-Agent": "CMC-Evolving-Necrotic-Front-data-fetcher"},
    )

    try:
        with urlopen(request, timeout=60) as response:
            payload = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc

    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{name} is not valid UTF-8 text") from exc

    lines = text.splitlines()

    if len(lines) != expected_lines:
        raise RuntimeError(
            f"{name}: expected {expected_lines} lines, received {len(lines)}"
        )

    if not lines or not lines[0].startswith("Day,InitialCondition,Radius"):
        raise RuntimeError(f"{name}: unexpected CSV header")

    destination.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    print(f"Downloaded: {destination}")
    print(f"  lines:  {len(lines)}")
    print(f"  sha256: {digest}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing downloaded files",
    )
    args = parser.parse_args()

    try:
        for name, metadata in FILES.items():
            download(
                name=name,
                url=str(metadata["url"]),
                expected_lines=int(metadata["expected_lines"]),
                force=args.force,
            )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
