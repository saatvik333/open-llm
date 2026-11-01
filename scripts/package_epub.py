#!/usr/bin/env python3
"""Create a deterministic EPUB ZIP from a prepared staging directory."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parent.parent
_released = date.fromisoformat(json.loads((ROOT / "book.json").read_text(encoding="utf-8"))["releaseDate"])
FIXED_TIME = (_released.year, _released.month, _released.day, 0, 0, 0)


def add_file(archive: ZipFile, path: Path, name: str, compression: int) -> None:
    info = ZipInfo(name, FIXED_TIME)
    info.compress_type = compression
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    archive.writestr(info, path.read_bytes())


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: package_epub.py STAGING_DIR OUTPUT.epub")
    staging = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    mimetype = staging / "mimetype"
    if not mimetype.is_file() or mimetype.read_bytes() != b"application/epub+zip":
        raise SystemExit("staging/mimetype must contain exactly application/epub+zip")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    files = sorted(path for path in staging.rglob("*") if path.is_file() and path != mimetype)
    with ZipFile(output, "w") as archive:
        add_file(archive, mimetype, "mimetype", ZIP_STORED)
        for path in files:
            add_file(archive, path, path.relative_to(staging).as_posix(), ZIP_DEFLATED)

    with ZipFile(output) as archive:
        names = archive.namelist()
        if not names or names[0] != "mimetype":
            raise SystemExit("EPUB mimetype is not the first archive member")
        if archive.getinfo("mimetype").compress_type != ZIP_STORED:
            raise SystemExit("EPUB mimetype must be stored without compression")
        bad = archive.testzip()
        if bad:
            raise SystemExit(f"corrupt EPUB archive member: {bad}")
    print(f"Packaged {output} ({output.stat().st_size:,} bytes, {len(files) + 1} members)")


if __name__ == "__main__":
    main()
