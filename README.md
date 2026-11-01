# Open LLM — From Nothing to the Frontier of Language Models

A free and open, zero-prerequisite book about artificial intelligence and large language models: foundations, neural networks, transformers, data and training, post-training, reasoning, retrieval, agents, multimodality, inference systems, evaluation, safety, and hands-on implementation.

The book is intentionally a **connected map and working mental model**, not a claim to contain literally every fact. AI changes too quickly for a static artifact to be the final word. Durable concepts are explained in depth; named products are clearly marked as a dated snapshot.

**How it was made:** the text was written by AI language models under the direction of the Open LLM Project, then checked by separate AI fact-checking passes against its 422 cited primary sources, with every number recomputed and every code example run. [`VERIFICATION.md`](VERIFICATION.md) describes the method and its limits. Corrections are welcome.

**Second edition, release 2.1 (v2.1.0). Technology snapshot and release date: 2 November 2026.**

## Read the book

- **Read online:** the project site serves the standalone, searchable edition at `/read/`.
- **Download:** every [release](https://github.com/saatvik333/open-llm/releases/latest) carries the A5 PDF (`Open-LLM-v2.1.0.pdf`), the reflowable EPUB 3 (`Open-LLM-v2.1.0.epub`), and `SHA256SUMS`. Verify a download with `sha256sum -c SHA256SUMS --ignore-missing`.
- **Cite:** see [`CITATION.cff`](CITATION.cff).
- **Accessibility:** see [`ACCESSIBILITY.md`](ACCESSIBILITY.md).
- **Corrections:** open an issue on this repository with the edition, page or section, and the proposed fix.
- **Verification record:** [`RELEASE_NOTES.md`](RELEASE_NOTES.md).

## Rebuild

Requirements:

- Node.js 22.23+, 24.22+, or 26.12+ (the range the pinned Cloudflare CLI supports)
- qpdf 11+
- Python 3 and Poppler tools (`pdfinfo`, `pdffonts`, `pdftotext`, `pdftoppm`)
- ImageMagick 7 (`magick`) for the web cover
- Java 17+ for EPUBCheck, which is downloaded into `.cache/` and checked against a pinned SHA-256 on first use
- The Noto Sans font for the cover typography (Arial is the fallback)

Install the pinned Node toolchain once:

```bash
npm ci
npx playwright install chromium
```

`npm ci` installs Playwright (PDF, cover, and audit rendering), the Cloudflare CLI, and Wrangler (deployment only) at the exact versions in `package-lock.json`.

Build and verify the complete release:

```bash
./scripts/build-digital-release.sh
```

It writes the PDF, HTML, and EPUB to `dist/`, the versioned release files and checksums to `dist/release/`, and the deployable site to `dist/site/`, then runs the free-release preflight. Nothing is published or deployed.

PDF-only development remains available with:

```bash
node scripts/build.js
python3 scripts/preflight.py
```

The builder has no Markdown package dependency. It:

1. reads every `manuscript/*.md` file in lexical order;
2. converts the supported Markdown subset to accessible HTML;
3. renders and measures a draft PDF;
4. generates a concise table of contents with stable page numbers;
5. rerenders until pagination stabilizes;
6. adds publication metadata and correct bookmark titles;
7. linearizes and validates the final PDF with qpdf.

## Publish

[`PUBLISHING.md`](PUBLISHING.md) covers the release checklist, publishing a GitHub Release with `gh`, and deploying the site to Cloudflare with `cf`.

## Project map

```text
book.json                         Title, edition, version, release date, snapshot (single source of truth)
manuscript/                       Book source, ordered by filename
assets/book.css                   Print and screen typography
assets/epub.css                   Reflowable-reader typography
scripts/build.js                  Markdown → HTML → publication PDF
scripts/build-epub.js             Semantic EPUB 3 builder
scripts/build-cover.js            Cover and social card
scripts/build-site.js             dist/site: landing page, online edition, headers
scripts/build_release_assets.py   Versioned PDF/EPUB and SHA256SUMS
scripts/build-digital-release.sh  Full build and verification
scripts/preflight.py              Manuscript and PDF verification
scripts/preflight_release.py      Release, checksum, metadata, and site verification
scripts/audit-site.js             Browser audit of the built site, with its headers applied
scripts/validate_epub.py          Pinned, checksum-verified EPUBCheck
site/                             Landing-page source, headers, styles
cloudflare.config.ts              Cloudflare Worker (static assets only)
wrangler.config.ts                Static assets directory for the Cloudflare build
sources/                          Research and currency notes
CITATION.cff                      Machine-readable citation metadata
PUBLISHING.md                     Release and deployment guide
RELEASE_NOTES.md                  Preflight and verification record
dist/                             Generated output (not committed)
```

## Publication identity

- Title: **Open LLM — From Nothing to the Frontier of Language Models**
- Author and publisher: **Open LLM Project**
- Edition: **Second edition, release 2.1 (v2.1.0)**
- Technology snapshot: **2 November 2026**
- Formats: **A5 PDF, reflowable EPUB 3, and standalone HTML**

Edit `book.json` to publish under a different author or imprint, then update `CITATION.cff`, rebuild, and rerun the preflight. The preflight fails when `CITATION.cff` disagrees with `book.json`.

## Editorial promises

- Start from no assumed AI knowledge.
- Separate definitions, derivations, empirical rules, and open questions.
- Show equations in words before symbols.
- Explain both capabilities and failure modes.
- Prefer primary sources and stable official references.
- Never equate fluent language with guaranteed truth or understanding.
- Treat safety, security, labor, law, and environmental cost as core engineering topics.
- Label fast-changing product information as a dated snapshot.

## License

Book prose and original diagrams: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Code samples and build scripts: [MIT](https://opensource.org/license/mit/). Full terms are in [`LICENSE.md`](LICENSE.md).
