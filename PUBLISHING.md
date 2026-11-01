# Publishing Open LLM

Open LLM is free and open. The book files are published as GitHub Release assets; the reading site and landing page are a static site on Cloudflare. Nothing in the build publishes or deploys by itself.

## Release artifacts

| Artifact | Where it is built | Where it is published |
|---|---|---|
| `Open-LLM-v2.0.0.pdf` — A5, tagged, bookmarked, linearized, embedded fonts | `dist/release/` | GitHub Release asset |
| `Open-LLM-v2.0.0.epub` — reflowable EPUB 3, validated with EPUBCheck | `dist/release/` | GitHub Release asset |
| `SHA256SUMS` — checksums of the two files above | `dist/release/` | GitHub Release asset |
| Standalone searchable HTML edition (`dist/open-llm.html`) | `dist/site/read/index.html` | Cloudflare, at `/read/` |
| Landing page, cover, social card, favicon, headers | `dist/site/` | Cloudflare |

The canonical fixed-layout publication is the PDF. The site never hosts the PDF or EPUB; its download buttons point at `https://github.com/saatvik333/open-llm/releases/latest/download/<versioned file name>`.

## Publication identity

`book.json` is the single source of truth for the title, edition, version, release date, and snapshot date. The scripts, the landing page, the file names, and the PDF and EPUB metadata all read it.

- **Title:** Open LLM — From Nothing to the Frontier of Language Models
- **Author and publisher:** Open LLM Project
- **Edition:** Second edition, release 2.0 (v2.0.0)
- **Technology snapshot and release date:** 2 November 2026
- **Book license:** Creative Commons Attribution 4.0 International
- **Code license:** MIT

Suggested citation:

> Open LLM Project. *Open LLM — From Nothing to the Frontier of Language Models*. Second edition, release 2.0, 2026.

`CITATION.cff`, this file, and `README.md` repeat the edition, version, and snapshot. `scripts/preflight_release.py` fails if any of them disagrees with `book.json`.

Do not add an ISBN until one has actually been assigned by the appropriate ISBN agency or publishing service.

## 1. Build and verify

```bash
npm ci
npx playwright install chromium
./scripts/build-digital-release.sh
```

Set `SITE_URL` first if the site already has a final address. It adds the canonical link, absolute Open Graph URLs, and a sitemap:

```bash
SITE_URL=https://open-llm.example.org ./scripts/build-digital-release.sh
```

The script ends with `scripts/preflight_release.py`, which checks the manuscript and references, the PDF structure and metadata, the EPUB (including EPUBCheck), the versioned file names and `SHA256SUMS`, `CITATION.cff` against `book.json`, the `cloudflare.config.ts` and `wrangler.config.ts` settings, the site contents, and the security headers. It then runs `scripts/audit-site.js`, which serves `dist/site` with its `_headers` rules applied and loads the pages in Chromium.

Then do the checks a machine cannot:

1. open the PDF in at least two independent viewers;
2. import the EPUB into at least two independent EPUB 3 readers;
3. test contents, citations, external links, keyboard navigation, and a representative screen-reader pass;
4. inspect the cover, contents, code pages, tables, references, and final page;
5. recheck fast-changing product names, laws, and capabilities against official primary sources;
6. update `RELEASE_NOTES.md` with the verification record and changes.

## 2. Publish the GitHub Release

Create the repository once:

```bash
gh repo create saatvik333/open-llm --public --source=. --push
```

Publish the release. The `latest/download` links on the site resolve only for a published release that is not a draft or a pre-release, so publish before you deploy:

```bash
git tag -a v2.0.0 -m "Open LLM, second edition, release 2.0"
git push origin v2.0.0
gh release create v2.0.0 \
  dist/release/Open-LLM-v2.0.0.pdf \
  dist/release/Open-LLM-v2.0.0.epub \
  dist/release/SHA256SUMS \
  --title "Open LLM — Second edition, release 2.0 (v2.0.0)" \
  --notes-file RELEASE_NOTES.md \
  --verify-tag
```

Check the result with `gh release view v2.0.0` and by downloading the PDF through its `latest/download` URL.

## 3. Deploy the site to Cloudflare

The Worker is named `open-llm` and serves static assets only: no Worker code runs. `cloudflare.config.ts` defines it for the `cf` CLI, and `wrangler.config.ts` sets the served directory to `dist/site` (the `cf` configuration has no field for it). `npm ci` installs the `cf` CLI and Wrangler at the pinned versions in `package-lock.json`; use them through `npx` or the npm scripts rather than a globally installed `cf`, which may be older and lack the project commands.

Sign in once (use `cf auth login --no-browser` on a remote machine, or set `CLOUDFLARE_API_TOKEN` in CI):

```bash
npx cf auth login
```

Validate without uploading, then deploy:

```bash
npx cf deploy --dry-run
npx cf deploy --message "Open LLM v2.0.0"
```

`cf` prints the `workers.dev` address when the deploy finishes. Node prints a `MODULE_TYPELESS_PACKAGE_JSON` warning while loading `cloudflare.config.ts`; it is harmless. `package.json` stays a CommonJS package because the build scripts are CommonJS.

To serve the site from your own domain, add `domains: ["your.domain.example"]` to the `worker` object in `cloudflare.config.ts` and rebuild with the matching `SITE_URL`. See <https://developers.cloudflare.com/cf/projects/cloudflare-config/>.

### Headers

`site/_headers` applies one policy to every response: a default-deny Content-Security-Policy (`default-src 'none'`, same-origin images and styles, no scripts), `frame-ancestors 'none'`, HSTS, `nosniff`, a strict referrer policy, and a minimal Permissions-Policy. The standalone book at `/read/` contains one inline stylesheet, so `scripts/build-site.js` adds that stylesheet's SHA-256 to `style-src`. Rebuild the site whenever the book HTML changes; the preflight fails if the hash is stale. Assets under `/assets/` revalidate after an hour.

## Corrections and new editions

For a factual correction that does not alter the technology snapshot, increment the release (for example 2.0 → 2.1, version 2.1.0). For substantial new chapters or a refreshed frontier landscape, create a new edition or a dated revision. In both cases:

1. edit `book.json` (`edition`, `release`, `version`, `releaseDate`, `snapshot`);
2. update `CITATION.cff`, `README.md`, and this file to match;
3. rebuild, publish the new release, then deploy the site so its download links name the new files.

Earlier releases stay available under their tags. The `latest/download` links always point at the newest release.

## Print-on-demand note

The included PDF is a complete A5 interior-and-digital-cover edition. A print-on-demand platform may require a different trim size, inside/outside gutter margins, bleed, color profile, and a separate full-wrap cover calculated from final paper stock and spine width. Generate those platform-specific files only after selecting the printer; do not stretch this PDF to fit an unrelated template.
