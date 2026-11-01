#!/usr/bin/env python3
"""Verify the free release: PDF, EPUB, versioned names, checksums, and the deployable site."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
RELEASE = DIST / "release"
SITE = DIST / "site"
BOOK = json.loads((ROOT / "book.json").read_text(encoding="utf-8"))
FILE_BASE = f"Open-LLM-v{BOOK['version']}"
REPO_URL = f"https://github.com/{BOOK['repository']}"

SITE_FILES = [
    "index.html", "404.html", "styles.css", "_headers", "robots.txt",
    "assets/favicon.svg", "assets/cover-web.jpg", "assets/social-card.jpg", "read/index.html",
]
REQUIRED_HEADERS = [
    "content-security-policy", "cross-origin-opener-policy", "cross-origin-resource-policy",
    "permissions-policy", "referrer-policy", "strict-transport-security",
    "x-content-type-options", "x-frame-options",
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def run(*args: str) -> str:
    completed = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        fail(f"Command failed ({' '.join(args)}):\n{completed.stdout}")
    return completed.stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_size(path: Path) -> tuple[int, int]:
    width, height = run("magick", "identify", "-format", "%w %h", str(path)).split()
    return int(width), int(height)


def cff_value(text: str, key: str) -> str:
    match = re.search(rf'^{re.escape(key)}:\s*"?([^"\n]+?)"?\s*$', text, re.M)
    return match.group(1) if match else ""


def check_metadata() -> None:
    """book.json is the source of truth; every other copy of it must agree."""
    if not BOOK["version"].startswith(f"{BOOK['release']}."):
        fail(f"book.json version {BOOK['version']} does not belong to release {BOOK['release']}")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", BOOK["releaseDate"]):
        fail("book.json releaseDate must be YYYY-MM-DD")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    expected = {
        "title": BOOK["fullTitle"],
        "version": BOOK["release"],
        "date-released": BOOK["releaseDate"],
        "repository-code": REPO_URL,
    }
    for key, value in expected.items():
        if cff_value(citation, key) != value:
            fail(f"CITATION.cff {key} is {cff_value(citation, key)!r}; book.json requires {value!r}")
    for name in ("README.md", "PUBLISHING.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for value in (f"{BOOK['edition']}, release {BOOK['release']}", f"v{BOOK['version']}", BOOK["snapshot"], FILE_BASE):
            if value not in text:
                fail(f"{name} does not state {value!r}; update it to match book.json")


def check_cloudflare_config() -> None:
    config = (ROOT / "cloudflare.config.ts").read_text(encoding="utf-8")
    build = (ROOT / "wrangler.config.ts").read_text(encoding="utf-8")
    if not re.search(r'name:\s*"open-llm"', config):
        fail('cloudflare.config.ts must name the Worker "open-llm"')
    if re.search(r"\bentrypoint\b\s*:", config):
        fail("cloudflare.config.ts must be static-assets only (no entrypoint)")
    if not re.search(r'assetsDirectory:\s*"\./dist/site"', build):
        fail("wrangler.config.ts must serve ./dist/site")


def check_epub() -> None:
    epub = DIST / "open-llm.epub"
    with ZipFile(epub) as archive:
        names = archive.namelist()
        if not names or names[0] != "mimetype" or archive.getinfo("mimetype").compress_type != ZIP_STORED:
            fail("EPUB mimetype must be first and uncompressed")
        if archive.read("mimetype") != b"application/epub+zip":
            fail("Invalid EPUB mimetype")
        for name in names:
            if name.endswith((".xhtml", ".opf", ".ncx", ".xml")):
                try:
                    ElementTree.fromstring(archive.read(name))
                except ElementTree.ParseError as error:
                    fail(f"Malformed XML in EPUB member {name}: {error}")
        if archive.testzip():
            fail("EPUB archive integrity test failed")
    run("python3", "scripts/validate_epub.py")


def check_release_files() -> list[Path]:
    expected = {f"{FILE_BASE}.pdf", f"{FILE_BASE}.epub", "SHA256SUMS"}
    present = {path.name for path in RELEASE.iterdir()} if RELEASE.is_dir() else set()
    if present != expected:
        fail(f"dist/release must contain exactly {sorted(expected)}; found {sorted(present)}")
    for versioned, canonical in ((f"{FILE_BASE}.pdf", "open-llm.pdf"), (f"{FILE_BASE}.epub", "open-llm.epub")):
        if sha256(RELEASE / versioned) != sha256(DIST / canonical):
            fail(f"{versioned} differs from dist/{canonical}")
    listed = {}
    for line in (RELEASE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (\S+)", line)
        if not match:
            fail(f"Malformed SHA256SUMS line: {line!r}")
        listed[match.group(2)] = match.group(1)
    if set(listed) != expected - {"SHA256SUMS"}:
        fail(f"SHA256SUMS must list exactly the PDF and EPUB; lists {sorted(listed)}")
    for name, digest in listed.items():
        if sha256(RELEASE / name) != digest:
            fail(f"SHA256SUMS mismatch for {name}")
    return [RELEASE / f"{FILE_BASE}.pdf", RELEASE / f"{FILE_BASE}.epub"]


def parse_header_rules(text: str) -> list[tuple[str, list[tuple[str, str | None]]]]:
    rules: list[tuple[str, list[tuple[str, str | None]]]] = []
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if not line[0].isspace():
            rules.append((line.strip(), []))
        elif line.strip().startswith("!"):
            rules[-1][1].append((line.strip()[1:].strip().lower(), None))
        else:
            name, _, value = line.strip().partition(":")
            rules[-1][1].append((name.strip().lower(), value.strip()))
    return rules


def headers_for(rules: list[tuple[str, list[tuple[str, str | None]]]], path: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for pattern, headers in rules:
        if not re.fullmatch(".*".join(re.escape(part) for part in pattern.split("*")), path):
            continue
        for name, value in headers:
            if value is None:
                result.pop(name, None)
            elif name in result:
                result[name] = f"{result[name]}, {value}"
            else:
                result[name] = value
    return result


def style_hash(html: str) -> str:
    styles = re.findall(r"<style>(.*?)</style>", html, re.S)
    if len(styles) != 1:
        fail(f"Expected exactly one inline <style> in the book edition, found {len(styles)}")
    return base64.b64encode(hashlib.sha256(styles[0].encode("utf-8")).digest()).decode("ascii")


def check_headers(text: str, book_html: str) -> None:
    rules = parse_header_rules(text)
    if "{{" in text:
        fail("dist/site/_headers contains an unresolved placeholder")
    for path in ["/", "/index.html", "/404.html", "/no-such-page", "/read/", "/assets/cover-web.jpg"]:
        headers = headers_for(rules, path)
        missing = [name for name in REQUIRED_HEADERS if name not in headers]
        if missing:
            fail(f"_headers leaves {path} without: {', '.join(missing)}")
        policy = headers["content-security-policy"]
        if policy.count("default-src") != 1 or "default-src 'none'" not in policy:
            fail(f"{path} must have exactly one default-deny Content-Security-Policy: {policy}")
        if re.search(r"unsafe-(inline|eval)|script-src|https?:|\*", policy):
            fail(f"{path} Content-Security-Policy is looser than intended: {policy}")
        if "frame-ancestors 'none'" not in policy or headers["x-frame-options"] != "DENY":
            fail(f"{path} must not be frameable")
        if "includesubdomains" in headers["strict-transport-security"].lower():
            fail("Strict-Transport-Security must not use includeSubDomains on a site that does not own the whole domain")
    expected_hash = f"'sha256-{style_hash(book_html)}'"
    for path in ["/", "/read/", "/read/index.html", "/no-such-page"]:
        if f"style-src 'self' {expected_hash};" not in headers_for(rules, path)["content-security-policy"]:
            fail(f"{path} Content-Security-Policy does not allow the inline stylesheet by its current hash")


def check_site() -> None:
    if not SITE.is_dir():
        fail("dist/site is missing; run node scripts/build-site.js")
    for relative in SITE_FILES:
        if not (SITE / relative).is_file():
            fail(f"Site file is missing: {relative}")
    banned = [path.relative_to(SITE).as_posix() for path in SITE.rglob("*") if path.suffix.lower() in {".pdf", ".epub", ".zip"}]
    if banned:
        fail(f"Downloads belong in the GitHub Release, not the site: {banned}")
    book_html = (SITE / "read" / "index.html").read_bytes()
    if book_html != (DIST / "open-llm.html").read_bytes():
        fail("dist/site/read/index.html differs from dist/open-llm.html")

    index = (SITE / "index.html").read_text(encoding="utf-8")
    if "{{" in index:
        fail("dist/site/index.html contains an unresolved placeholder")
    download = f"{REPO_URL}/releases/latest/download/{FILE_BASE}"
    for url in [f"{download}.pdf", f"{download}.epub", REPO_URL]:
        if f'href="{url}"' not in index:
            fail(f"Landing page does not link to {url}")
    if 'href="read/"' not in index:
        fail("Landing page does not link to the online edition")
    for text in [BOOK["edition"], BOOK["snapshot"], BOOK["version"], "CC BY 4.0"]:
        if text not in index:
            fail(f"Landing page does not state {text!r}")

    for relative in ["index.html", "read/index.html", "404.html"]:
        if re.search(r"noindex|nofollow", (SITE / relative).read_text(encoding="utf-8"), re.I):
            fail(f"{relative} blocks indexing")
    robots = (SITE / "robots.txt").read_text(encoding="utf-8")
    if re.search(r"^Disallow:\s*/\s*$", robots, re.M):
        fail("robots.txt blocks the whole site")

    for relative in ["index.html", "404.html"]:
        html = (SITE / relative).read_text(encoding="utf-8")
        scripts = [tag for tag in re.findall(r"<script\b[^>]*>", html) if 'type="application/ld+json"' not in tag]
        if scripts:
            fail(f"{relative} contains executable scripts: {scripts}")
        remote = [
            tag for tag in re.findall(r"<(?:link|img|source|iframe|video|audio)\b[^>]*>", html)
            if re.search(r'\s(?:href|src|srcset)="(?:https?:)?//', tag) and not re.search(r'\srel="(?:canonical|license|author|me)"', tag)
        ]
        if remote:
            fail(f"{relative} loads a remote resource: {remote}")
    css = (SITE / "styles.css").read_text(encoding="utf-8")
    if re.search(r"url\(|@import", css):
        fail("styles.css must not load external resources")

    check_headers((SITE / "_headers").read_text(encoding="utf-8"), book_html.decode("utf-8"))
    expected_images = {
        DIST / "open-llm-cover.jpg": (1748, 2480),
        SITE / "assets" / "cover-web.jpg": (1200, 1703),
        SITE / "assets" / "social-card.jpg": (1200, 630),
    }
    for path, size in expected_images.items():
        if image_size(path) != size:
            fail(f"Unexpected image dimensions for {path.relative_to(ROOT)}: {image_size(path)} (expected {size})")


def check_secret_patterns() -> None:
    patterns = [
        r"CLOUDFLARE_API_TOKEN\s*=\s*[^\s]+",
        r"\bgh[pousr]_[A-Za-z0-9]{36,}",
        r"github_pat_[A-Za-z0-9_]{20,}",
        r"-----BEGIN [A-Z ]*KEY-----",
    ]
    skipped = {".cache", ".cloudflare", ".git", "dist", "node_modules"}
    suffixes = {".md", ".json", ".js", ".ts", ".py", ".sh", ".html", ".css", ".txt", ".cff", ".svg"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or skipped & set(path.relative_to(ROOT).parts) or path.suffix.lower() not in suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            if re.search(pattern, text):
                fail(f"Possible secret in {path.relative_to(ROOT)}")


def check_scripts_parse() -> None:
    for script in sorted((ROOT / "scripts").glob("*.js")):
        run("node", "--check", str(script))


def main() -> None:
    check_metadata()
    check_cloudflare_config()
    check_scripts_parse()
    check_secret_patterns()
    run("python3", "scripts/preflight.py")
    check_epub()
    files = check_release_files()
    check_site()
    run("node", "scripts/audit-site.js")

    print("PASS — free release preflight")
    print(f"  {BOOK['edition']}, release {BOOK['release']}, v{BOOK['version']}, snapshot {BOOK['snapshot']}")
    print("  PDF: validated A5 publication")
    print("  EPUB: EPUBCheck 5.2.1, 0 errors / 0 warnings")
    for path in files:
        print(f"  {path.name}: {path.stat().st_size:,} bytes, SHA-256 verified")
    print("  site: dist/site is static, script-free, indexable, with restrictive headers")
    print("  browser audit: no remote requests, CSP violations, broken links, or heading jumps")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL — {error}", file=sys.stderr)
        raise SystemExit(1)
