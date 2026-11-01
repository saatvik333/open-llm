#!/usr/bin/env python3
"""Stage the versioned PDF and EPUB release files with SHA-256 checksums."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
RELEASE = DIST / "release"
PDF = DIST / "open-llm.pdf"
EPUB = DIST / "open-llm.epub"
VERSION = json.loads((ROOT / "book.json").read_text(encoding="utf-8"))["version"]
FILE_BASE = f"Open-LLM-v{VERSION}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not PDF.is_file() or not EPUB.is_file():
        raise RuntimeError("Missing PDF or EPUB; build both editions before staging release assets")
    shutil.rmtree(RELEASE, ignore_errors=True)
    RELEASE.mkdir(parents=True)

    files = [RELEASE / f"{FILE_BASE}.pdf", RELEASE / f"{FILE_BASE}.epub"]
    shutil.copy2(PDF, files[0])
    shutil.copy2(EPUB, files[1])

    sums = RELEASE / "SHA256SUMS"
    sums.write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files), encoding="utf-8")

    print(f"Staged release v{VERSION} in {RELEASE.relative_to(ROOT)}")
    for path in [*files, sums]:
        print(f"  {path.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL — {error}")
        raise SystemExit(1)
