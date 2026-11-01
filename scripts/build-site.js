#!/usr/bin/env node
'use strict';

// Assembles dist/site/, the static directory Cloudflare serves:
//   index.html   landing page rendered from site/index.html
//   read/        the standalone HTML edition (dist/open-llm.html)
//   assets/      web cover, social card, favicon
//   _headers, robots.txt, 404.html, styles.css
// PDF and EPUB files are never copied here; the page links to the GitHub Release assets.
//
// Optional: SITE_URL=https://example.org adds canonical, Open Graph, and sitemap URLs.

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execFileSync } = require('child_process');
const { BOOK, escapeHtml } = require('./build.js');

const ROOT = path.resolve(__dirname, '..');
const DIST = path.join(ROOT, 'dist');
const SOURCE = path.join(ROOT, 'site');
const OUTPUT = path.join(DIST, 'site');
const MANUSCRIPT = path.join(ROOT, 'manuscript');
const TEMPLATED = new Set(['index.html', '_headers', 'robots.txt']);

function siteUrl() {
  const value = (process.env.SITE_URL || '').trim();
  if (!value) return '';
  let url;
  try {
    url = new URL(value);
  } catch (_) {
    throw new Error('SITE_URL must be an absolute https URL such as https://open-llm.example.org');
  }
  if (url.protocol !== 'https:' || url.pathname !== '/' || url.search || url.hash) {
    throw new Error('SITE_URL must be an https origin without a path, query, or fragment');
  }
  return url.origin;
}

function requireFile(file, hint) {
  if (!fs.existsSync(file)) throw new Error(`Missing ${path.relative(ROOT, file)}; ${hint}`);
}

function manuscriptStats() {
  const source = fs.readdirSync(MANUSCRIPT)
    .filter(name => name.endsWith('.md'))
    .sort()
    .map(name => fs.readFileSync(path.join(MANUSCRIPT, name), 'utf8'))
    .join('\n');
  const references = (source.match(/^\*\*\[R\d{2,3}\]\*\*/gm) || []).length;
  const labs = (source.match(/^# Lab \d+ —/gm) || []).length;
  const words = (source.match(/\b[\w’'-]+\b/gu) || []).length; // same rule as scripts/preflight.py
  const pages = Number(execFileSync('qpdf', ['--show-npages', path.join(DIST, 'open-llm.pdf')], { encoding: 'utf8' }).trim());
  if (!references || !labs || !words || !Number.isInteger(pages) || pages < 1) throw new Error('Unable to compute book statistics.');
  return { references, labs, pages, words: Math.round(words / 1000) * 1000 };
}

// Decimal units, as file managers and browsers display them.
function fileSize(file) {
  const bytes = fs.statSync(file).size;
  return bytes >= 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.round(bytes / 1e3)} KB`;
}

function render(template, values, label) {
  return template.replace(/\{\{([A-Z_]+)\}\}/g, (match, key) => {
    if (!(key in values)) throw new Error(`${label}: unknown placeholder ${match}`);
    return values[key];
  });
}

function main() {
  requireFile(path.join(DIST, 'open-llm.html'), 'run node scripts/build.js first');
  requireFile(path.join(DIST, 'open-llm.pdf'), 'run node scripts/build.js first');
  requireFile(path.join(DIST, 'open-llm.epub'), 'run node scripts/build-epub.js first');
  requireFile(path.join(DIST, 'open-llm-cover.jpg'), 'run node scripts/build-cover.js first');
  requireFile(path.join(DIST, 'open-llm-social-card.jpg'), 'run node scripts/build-cover.js first');

  const origin = siteUrl();
  const stats = manuscriptStats();
  const releaseBase = `${BOOK.repositoryUrl}/releases/latest/download`;
  const pdfUrl = `${releaseBase}/${BOOK.fileBase}.pdf`;
  const epubUrl = `${releaseBase}/${BOOK.fileBase}.epub`;

  fs.rmSync(OUTPUT, { recursive: true, force: true });
  fs.cpSync(SOURCE, OUTPUT, {
    recursive: true,
    filter: source => !TEMPLATED.has(path.relative(SOURCE, source)),
  });

  // Landing page.
  const jsonLd = JSON.stringify({
    '@context': 'https://schema.org',
    '@type': 'Book',
    name: BOOK.fullTitle,
    author: { '@type': 'Organization', name: BOOK.author },
    publisher: { '@type': 'Organization', name: BOOK.publisher },
    bookEdition: BOOK.edition,
    datePublished: BOOK.releaseDate,
    inLanguage: BOOK.language,
    numberOfPages: stats.pages,
    isAccessibleForFree: true,
    license: BOOK.licenseUrl,
    description: BOOK.description,
  }).replace(/</g, '\\u003c');
  const values = {
    LABS: String(stats.labs),
    PDF_SIZE: fileSize(path.join(DIST, 'open-llm.pdf')),
    EPUB_SIZE: fileSize(path.join(DIST, 'open-llm.epub')),
    FULL_TITLE: escapeHtml(BOOK.fullTitle),
    TITLE: escapeHtml(BOOK.title),
    SUBTITLE: escapeHtml(BOOK.subtitle),
    AUTHOR: escapeHtml(BOOK.author),
    EDITION: escapeHtml(BOOK.edition),
    VERSION: escapeHtml(BOOK.version),
    SNAPSHOT: escapeHtml(BOOK.snapshot),
    YEAR: escapeHtml(BOOK.year),
    PDF_URL: escapeHtml(pdfUrl),
    EPUB_URL: escapeHtml(epubUrl),
    REPO_URL: escapeHtml(BOOK.repositoryUrl),
    RELEASES_URL: escapeHtml(`${BOOK.repositoryUrl}/releases`),
    ISSUES_URL: escapeHtml(`${BOOK.repositoryUrl}/issues`),
    CITATION_URL: escapeHtml(`${BOOK.repositoryUrl}/blob/HEAD/CITATION.cff`),
    LICENSE_FILE_URL: escapeHtml(`${BOOK.repositoryUrl}/blob/HEAD/LICENSE.md`),
    ACCESSIBILITY_URL: escapeHtml(`${BOOK.repositoryUrl}/blob/HEAD/ACCESSIBILITY.md`),
    VERIFICATION_URL: escapeHtml(`${BOOK.repositoryUrl}/blob/HEAD/VERIFICATION.md`),
    PAGES: String(stats.pages),
    REFERENCES: String(stats.references),
    ORIGIN: escapeHtml(origin),
    CANONICAL: origin
      ? `<link rel="canonical" href="${escapeHtml(origin)}/">\n  <meta property="og:url" content="${escapeHtml(origin)}/">`
      : '',
    JSON_LD: jsonLd,
  };
  const index = render(fs.readFileSync(path.join(SOURCE, 'index.html'), 'utf8'), values, 'site/index.html');
  fs.writeFileSync(path.join(OUTPUT, 'index.html'), index);

  // Web cover and social card.
  fs.mkdirSync(path.join(OUTPUT, 'assets'), { recursive: true });
  execFileSync('magick', [
    path.join(DIST, 'open-llm-cover.jpg'),
    '-resize', '1200x1703!', '-strip', '-colorspace', 'sRGB', '-quality', '91',
    path.join(OUTPUT, 'assets', 'cover-web.jpg'),
  ]);
  fs.copyFileSync(path.join(DIST, 'open-llm-social-card.jpg'), path.join(OUTPUT, 'assets', 'social-card.jpg'));

  // Standalone HTML edition.
  const book = fs.readFileSync(path.join(DIST, 'open-llm.html'), 'utf8');
  const styles = [...book.matchAll(/<style>([\s\S]*?)<\/style>/g)];
  if (styles.length !== 1) throw new Error(`Expected exactly one inline <style> in dist/open-llm.html, found ${styles.length}.`);
  const styleHash = crypto.createHash('sha256').update(styles[0][1], 'utf8').digest('base64');
  fs.mkdirSync(path.join(OUTPUT, 'read'), { recursive: true });
  fs.writeFileSync(path.join(OUTPUT, 'read', 'index.html'), book);

  // Headers and crawler files.
  const headers = render(fs.readFileSync(path.join(SOURCE, '_headers'), 'utf8'), { READ_STYLE_HASH: styleHash }, 'site/_headers');
  fs.writeFileSync(path.join(OUTPUT, '_headers'), headers);
  let robots = fs.readFileSync(path.join(SOURCE, 'robots.txt'), 'utf8').replace(/\s*$/, '\n');
  if (origin) {
    robots += `\nSitemap: ${origin}/sitemap.xml\n`;
    const urls = ['/', '/read/'].map(route => `  <url><loc>${escapeHtml(origin + route)}</loc><lastmod>${BOOK.releaseDate}</lastmod></url>`);
    fs.writeFileSync(path.join(OUTPUT, 'sitemap.xml'), `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls.join('\n')}\n</urlset>\n`);
  }
  fs.writeFileSync(path.join(OUTPUT, 'robots.txt'), robots);

  console.log(`Built ${path.relative(ROOT, OUTPUT)} for ${BOOK.fileBase} (${stats.pages} pages, ${stats.references} references, ~${stats.words.toLocaleString('en-US')} words)`);
  console.log(origin ? `  canonical origin: ${origin}` : '  canonical origin: not set (export SITE_URL to add canonical, Open Graph, and sitemap URLs)');
}

try {
  main();
} catch (error) {
  console.error(error.stack || error.message || String(error));
  process.exitCode = 1;
}
