#!/usr/bin/env python3
"""Validate the release EPUB with a pinned, checksum-verified EPUBCheck."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent.parent
EPUB = ROOT / "dist" / "open-llm.epub"
CACHE = ROOT / ".cache" / "epubcheck-5.2.1"
ARCHIVE = ROOT / ".cache" / "epubcheck-5.2.1.zip"
URL = "https://github.com/w3c/epubcheck/releases/download/v5.2.1/epubcheck-5.2.1.zip"
SHA256 = "0532f6291faa2bb729dd253f958868a2a57dbd2c32f881a97c7c980c5940309e"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe EPUBCheck archive member: {member.filename}")
        zipped.extractall(destination)


def ensure_validator() -> Path:
    jars = list(CACHE.rglob("epubcheck.jar")) if CACHE.exists() else []
    if jars:
        return jars[0]
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists() or digest(ARCHIVE) != SHA256:
        print(f"Downloading pinned EPUBCheck 5.2.1 from {URL}")
        request = urllib.request.Request(URL, headers={"User-Agent": "open-llm-release-builder/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, ARCHIVE.open("wb") as output:
            shutil.copyfileobj(response, output)
    actual = digest(ARCHIVE)
    if actual != SHA256:
        ARCHIVE.unlink(missing_ok=True)
        raise RuntimeError(f"EPUBCheck checksum mismatch: expected {SHA256}, received {actual}")
    CACHE.mkdir(parents=True, exist_ok=True)
    safe_extract(ARCHIVE, CACHE)
    jars = list(CACHE.rglob("epubcheck.jar"))
    if len(jars) != 1:
        raise RuntimeError("Unable to locate epubcheck.jar after extraction")
    return jars[0]


def main() -> None:
    if not EPUB.is_file():
        raise RuntimeError("Missing dist/open-llm.epub; run node scripts/build-epub.js")
    jar = ensure_validator()
    completed = subprocess.run(["java", "-jar", str(jar), str(EPUB)], cwd=ROOT, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print("PASS — EPUBCheck 5.2.1 reported no errors or warnings")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL — {error}", file=sys.stderr)
        raise SystemExit(1)
