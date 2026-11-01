#!/usr/bin/env python3
"""Offline publication preflight for the Open LLM release artifacts."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANUSCRIPT = ROOT / "manuscript"
PDF = ROOT / "dist" / "open-llm.pdf"
HTML = ROOT / "dist" / "open-llm.html"
BOOK = json.loads((ROOT / "book.json").read_text(encoding="utf-8"))


def fail(message: str) -> None:
    raise RuntimeError(message)


def run(*args: str) -> str:
    completed = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        fail(f"Command failed ({' '.join(args)}):\n{completed.stdout}")
    return completed.stdout


def flatten_outlines(items: list[dict]) -> list[dict]:
    result: list[dict] = []
    for item in items:
        result.append(item)
        result.extend(flatten_outlines(item.get("kids", [])))
    return result


def main() -> None:
    if not PDF.is_file() or not HTML.is_file():
        fail("Build artifacts are missing; run: node scripts/build.js")

    files = sorted(MANUSCRIPT.glob("*.md"))
    if not files:
        fail("No manuscript files found")
    source = "\n".join(path.read_text(encoding="utf-8") for path in files)

    definition_ids = re.findall(r"^\*\*\[R(\d+)\]\*\*", source, re.M)
    if any(value != f"{int(value):02d}" for value in definition_ids):
        fail("Reference IDs must be zero-padded to two digits (R01, …, R99, R100, …)")
    definitions = [int(value) for value in definition_ids]
    if not definitions or definitions != list(range(1, len(definitions) + 1)):
        fail(f"Reference definitions are not a continuous R01–Rnn sequence: {definitions}")
    without_definitions = re.sub(r"^\*\*\[R\d{2,3}\]\*\*.*$", "", source, flags=re.M)
    citations = {int(value) for value in re.findall(r"\bR(\d{2,3})\b", without_definitions)}
    if citations != set(definitions):
        fail(f"Citation mismatch: missing={sorted(set(definitions)-citations)}, unknown={sorted(citations-set(definitions))}")
    if source.count("```") % 2:
        fail("Unbalanced fenced code blocks")
    if len(re.findall(r"^:::", source, re.M)) % 2:
        fail("Unbalanced callout markers")

    prose = re.sub(r"^```.*?^```\s*$", "", source, flags=re.M | re.S)
    heading_matches = list(re.finditer(r"^(#{1,6})\s+(.+)$", prose, re.M))
    levels = [len(match.group(1)) for match in heading_matches]
    if any(current > previous + 1 for previous, current in zip(levels, levels[1:])):
        fail("Heading hierarchy skips a level")
    heading_titles = [re.sub(r"[`*_]", "", match.group(2).strip()) for match in heading_matches]
    h1_titles = [
        re.sub(r"[`*_]", "", match.group(1).strip())
        for match in re.finditer(r"^#\s+(.+)$", prose, re.M)
    ]

    for path in files:
        text = path.read_text(encoding="utf-8")
        for index, match in enumerate(re.finditer(r"^```json\n(.*?)^```\s*$", text, re.M | re.S), 1):
            try:
                json.loads(match.group(1))
            except Exception as error:
                fail(f"Invalid JSON example in {path.name} block {index}: {error}")
        for index, match in enumerate(re.finditer(r"^```python\n(.*?)^```\s*$", text, re.M | re.S), 1):
            try:
                ast.parse(match.group(1))
            except Exception as error:
                fail(f"Invalid Python example in {path.name} block {index}: {error}")

    html_text = HTML.read_text(encoding="utf-8")
    ids = set(re.findall(r'\bid="([^"]+)"', html_text))
    internal_targets = re.findall(r'href="#([^"]+)"', html_text)
    missing_targets = sorted(set(internal_targets) - ids)
    if missing_targets:
        fail(f"HTML has missing internal targets: {missing_targets}")
    reference_anchors = set(re.findall(r'id="ref-(r\d{2,3})"', html_text))
    citation_targets = set(re.findall(r'href="#ref-(r\d{2,3})"', html_text))
    if reference_anchors != citation_targets or len(reference_anchors) != len(definitions):
        fail("HTML citation anchors do not match citation links")
    if "<span class=\"toc-page\"" not in html_text or ">—</span>" in html_text:
        fail("Contents page numbers are missing or unresolved")

    run("qpdf", "--check", str(PDF))
    info = run("pdfinfo", str(PDF))
    for key, expected in (("Title", BOOK["fullTitle"]), ("Author", BOOK["author"])):
        match = re.search(rf"^{key}:\s+(.*)$", info, re.M)
        if not match or match.group(1).strip() != expected:
            fail(f"PDF {key} metadata does not match book.json")
    if not re.search(r"^Tagged:\s+yes$", info, re.M):
        fail("PDF is not tagged")
    if not re.search(r"^Optimized:\s+yes$", info, re.M):
        fail("PDF is not linearized")
    if not re.search(r"^Encrypted:\s+no$", info, re.M):
        fail("Unexpected PDF encryption")
    page_match = re.search(r"^Pages:\s+(\d+)$", info, re.M)
    if not page_match:
        fail("Unable to read PDF page count")
    pages = int(page_match.group(1))
    if not re.search(r"^Page size:.*\(A5\)$", info, re.M):
        fail("PDF page size is not A5")

    fonts = run("pdffonts", str(PDF))
    font_lines = [line for line in fonts.splitlines()[2:] if line.strip()]
    if not font_lines or any(not re.search(r"\byes\s+yes\s+yes\s+\d+\s+\d+\s*$", line) for line in font_lines):
        fail("One or more fonts are not embedded, subset, and Unicode-mapped")

    extracted = run("pdftotext", str(PDF), "-")
    forbidden = ["@@CODE", "&amp;", "&lt;", "[object Object]", "TODO", "TBD", "FIXME", "\ufffd"]
    present = [marker for marker in forbidden if marker in extracted]
    if present:
        fail(f"PDF extraction contains rendering/editorial artifacts: {present}")
    for label, value in (("edition", BOOK["edition"]), ("technology snapshot", BOOK["snapshot"])):
        if value not in extracted:
            fail(f"PDF text does not state the current {label}: {value!r}")
    for number in definitions:
        if f"[R{number:02d}]" not in extracted:
            fail(f"PDF text is missing reference R{number:02d}")

    outline_json = json.loads(run("qpdf", "--json", "--json-key=outlines", str(PDF)))
    top_level = outline_json.get("outlines", [])
    flattened = flatten_outlines(top_level)
    expected_titles = ["OPEN LLM", "Open LLM", "Contents", *heading_titles]
    actual_titles = [item.get("title") for item in flattened]
    if actual_titles != expected_titles:
        fail("PDF bookmark titles do not exactly match the manuscript")
    manuscript_top = top_level[-len(h1_titles):]
    physical_pages = [item.get("destpageposfrom1") for item in manuscript_top]
    toc_pages = [int(value) for value in re.findall(r'<span class="toc-page" aria-label="page \d+">(\d+)</span>', html_text)]
    if toc_pages != physical_pages:
        fail("Contents page numbers do not match PDF destinations")

    digest = hashlib.sha256(PDF.read_bytes()).hexdigest()
    manuscript_words = len(re.findall(r"\b[\w’'-]+\b", source))
    print("PASS — Open LLM publication preflight")
    print(f"  manuscript files: {len(files)}")
    print(f"  manuscript words: {manuscript_words}")
    print(f"  references: {len(definitions)}")
    print(f"  headings/bookmarks: {len(heading_titles)} / {len(flattened)-3}")
    print(f"  contents entries: {len(h1_titles)}")
    print("  PDF format: A5 (148 × 210 mm)")
    print(f"  PDF pages: {pages}")
    print(f"  embedded fonts: {len(font_lines)}")
    print(f"  SHA-256: {digest}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL — {error}", file=sys.stderr)
        raise SystemExit(1)
