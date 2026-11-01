#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

printf '\n[1/8] Building A5 PDF and standalone HTML\n'
node scripts/build.js

printf '\n[2/8] Validating PDF publication\n'
python3 scripts/preflight.py

printf '\n[3/8] Building cover and social card\n'
node scripts/build-cover.js

printf '\n[4/8] Building reflowable EPUB 3\n'
node scripts/build-epub.js

printf '\n[5/8] Running pinned EPUBCheck\n'
python3 scripts/validate_epub.py

printf '\n[6/8] Staging versioned release files and checksums\n'
python3 scripts/build_release_assets.py

printf '\n[7/8] Building the deployable site (dist/site)\n'
node scripts/build-site.js

printf '\n[8/8] Running free-release preflight\n'
python3 scripts/preflight_release.py

printf '\nRelease package complete. Nothing was published or deployed; see PUBLISHING.md.\n'
