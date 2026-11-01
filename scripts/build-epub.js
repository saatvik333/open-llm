#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execFileSync } = require('child_process');
const { BOOK, escapeHtml, markdownToHtml } = require('./build.js');

const ROOT = path.resolve(__dirname, '..');
const MANUSCRIPT = path.join(ROOT, 'manuscript');
const DIST = path.join(ROOT, 'dist');
const STYLE = path.join(ROOT, 'assets', 'epub.css');
const STAGING = path.join(DIST, '.epub-staging');
const EPUB = path.join(DIST, 'open-llm.epub');
const PDF = path.join(DIST, 'open-llm.pdf');
const RELEASE_DATE = BOOK.releaseDate;
const MODIFIED = `${RELEASE_DATE}T00:00:00Z`;

function write(relativePath, content) {
  const target = path.join(STAGING, relativePath);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, content);
}

function deterministicUuid(value) {
  const bytes = crypto.createHash('sha256').update(value).digest().subarray(0, 16);
  bytes[6] = (bytes[6] & 0x0f) | 0x50;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = bytes.toString('hex');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function xhtmlDocument(title, body, bodyClass = '') {
  const klass = bodyClass ? ` class="${bodyClass}"` : '';
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="${BOOK.language}" lang="${BOOK.language}">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${escapeHtml(title)}</title>
  <link rel="stylesheet" type="text/css" href="../css/epub.css" />
</head>
<body${klass}>
${body}
</body>
</html>
`;
}

function remapInternalLinks(html, currentFile, targetFiles) {
  return html
    .replace(/href="#([^"]+)"/g, (match, id) => {
      const target = targetFiles.get(id);
      if (!target || target === currentFile) return match;
      return `href="${target}#${id}"`;
    })
    .replace(/<hr>/g, '<hr />')
    .replace(/<a class="external-link"/g, '<a rel="noopener noreferrer" class="external-link"');
}

function navList(headings) {
  const h1s = [];
  let current = null;
  for (const heading of headings) {
    if (heading.level === 1) {
      current = { ...heading, children: [] };
      h1s.push(current);
    } else if (heading.level === 2 && current) {
      current.children.push(heading);
    }
  }
  return h1s.map(item => {
    const children = item.children.length
      ? `<ol>${item.children.map(child => `<li><a href="text/${child.file}#${child.id}">${escapeHtml(child.title)}</a></li>`).join('')}</ol>`
      : '';
    return `<li><a href="text/${item.file}#${item.id}">${escapeHtml(item.title)}</a>${children}</li>`;
  }).join('\n');
}

function ncxPoints(headings) {
  let order = 1;
  return headings.filter(item => item.level === 1).map(item => {
    const point = `<navPoint id="nav-${order}" playOrder="${order}"><navLabel><text>${escapeHtml(item.title)}</text></navLabel><content src="text/${item.file}#${item.id}" /></navPoint>`;
    order += 1;
    return point;
  }).join('\n');
}

function buildCover() {
  const sourceCover = path.join(DIST, 'open-llm-cover.jpg');
  const cover = path.join(STAGING, 'EPUB', 'images', 'cover.jpg');
  fs.mkdirSync(path.dirname(cover), { recursive: true });
  if (fs.existsSync(sourceCover)) {
    fs.copyFileSync(sourceCover, cover);
    return;
  }
  if (!fs.existsSync(PDF)) throw new Error('Missing cover and PDF; run node scripts/build.js and node scripts/build-cover.js first.');
  const outputBase = cover.replace(/\.jpg$/, '');
  execFileSync('pdftoppm', ['-f', '1', '-l', '1', '-r', '240', '-jpeg', '-singlefile', PDF, outputBase], { stdio: 'ignore' });
  if (!fs.existsSync(cover)) throw new Error('Failed to render EPUB cover image.');
}

function main() {
  if (!fs.existsSync(STYLE)) throw new Error(`Missing EPUB stylesheet: ${STYLE}`);
  const sourceFiles = fs.readdirSync(MANUSCRIPT).filter(name => name.endsWith('.md')).sort();
  if (!sourceFiles.length) throw new Error('No manuscript files found.');

  fs.rmSync(STAGING, { recursive: true, force: true });
  fs.mkdirSync(STAGING, { recursive: true });
  write('mimetype', 'application/epub+zip');
  write('META-INF/container.xml', `<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="EPUB/package.opf" media-type="application/oebps-package+xml" /></rootfiles>
</container>
`);
  write('EPUB/css/epub.css', fs.readFileSync(STYLE, 'utf8'));
  buildCover();

  const headings = [];
  const usedSlugs = new Set();
  const sections = [];
  const targetFiles = new Map();

  sourceFiles.forEach((sourceName, index) => {
    const source = fs.readFileSync(path.join(MANUSCRIPT, sourceName), 'utf8');
    const start = headings.length;
    const body = markdownToHtml(source, headings, usedSlugs);
    const file = `chapter-${String(index + 1).padStart(2, '0')}.xhtml`;
    const title = headings[start] ? headings[start].title : sourceName;
    for (let cursor = start; cursor < headings.length; cursor += 1) {
      headings[cursor].file = file;
      targetFiles.set(headings[cursor].id, file);
    }
    for (const match of body.matchAll(/\bid="([^"]+)"/g)) targetFiles.set(match[1], file);
    sections.push({ file, title, body });
  });

  sections.forEach(section => {
    const body = remapInternalLinks(section.body, section.file, targetFiles);
    write(`EPUB/text/${section.file}`, xhtmlDocument(section.title, `<main epub:type="bodymatter">${body}</main>`));
  });

  write('EPUB/text/cover.xhtml', xhtmlDocument('Cover', `<section epub:type="cover"><img src="../images/cover.jpg" alt="Cover of ${escapeHtml(BOOK.fullTitle)}" /></section>`, 'cover-page'));
  write('EPUB/text/title.xhtml', xhtmlDocument('Title page', `<main epub:type="frontmatter titlepage">
  <section class="title-page">
    <h1>${escapeHtml(BOOK.title)}</h1>
    <p class="subtitle">${escapeHtml(BOOK.subtitle)}</p>
    <p class="author">${escapeHtml(BOOK.author)}</p>
    <p class="edition">${escapeHtml(BOOK.edition)} · Release ${escapeHtml(BOOK.release)} · ${escapeHtml(BOOK.year)}</p>
    <p>Technology snapshot: ${escapeHtml(BOOK.snapshot)}</p>
    <p>Written by AI language models under the direction of the Open LLM Project and checked by separate AI fact-checking passes against the cited primary sources. Report mistakes at <a href="${BOOK.repositoryUrl}">github.com/${escapeHtml(BOOK.repository)}</a>.</p>
    <p><a href="${BOOK.licenseUrl}">Creative Commons Attribution 4.0 International</a></p>
  </section>
</main>`, 'title-page'));

  write('EPUB/nav.xhtml', `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="${BOOK.language}" lang="${BOOK.language}">
<head><meta charset="UTF-8" /><title>Contents</title><link rel="stylesheet" type="text/css" href="css/epub.css" /></head>
<body>
<nav epub:type="toc" id="toc" aria-labelledby="toc-title"><h1 id="toc-title">Contents</h1><ol>${navList(headings)}</ol></nav>
<nav epub:type="landmarks" aria-label="Landmarks"><h2>Landmarks</h2><ol>
<li><a epub:type="cover" href="text/cover.xhtml">Cover</a></li>
<li><a epub:type="titlepage" href="text/title.xhtml">Title page</a></li>
<li><a epub:type="bodymatter" href="text/${sections[0].file}">Start of book</a></li>
<li><a epub:type="bibliography" href="text/${sections[sections.length - 1].file}#references-and-primary-reading">References</a></li>
</ol></nav>
</body></html>
`);

  const identifier = deterministicUuid(`${BOOK.fullTitle}|${BOOK.edition}|${BOOK.release}`);
  write('EPUB/toc.ncx', `<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1" xml:lang="${BOOK.language}">
<head><meta name="dtb:uid" content="urn:uuid:${identifier}" /><meta name="dtb:depth" content="1" /><meta name="dtb:totalPageCount" content="0" /><meta name="dtb:maxPageNumber" content="0" /></head>
<docTitle><text>${escapeHtml(BOOK.fullTitle)}</text></docTitle><docAuthor><text>${escapeHtml(BOOK.author)}</text></docAuthor>
<navMap>${ncxPoints(headings)}</navMap>
</ncx>
`);

  const chapterManifest = sections.map((section, index) => `<item id="chapter-${index + 1}" href="text/${section.file}" media-type="application/xhtml+xml" />`).join('\n    ');
  const chapterSpine = sections.map((_, index) => `<itemref idref="chapter-${index + 1}" />`).join('\n    ');
  write('EPUB/package.opf', `<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book-id" xml:lang="${BOOK.language}" prefix="rendition: http://www.idpf.org/vocab/rendition/#">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/">
    <dc:identifier id="book-id">urn:uuid:${identifier}</dc:identifier>
    <dc:title id="title">${escapeHtml(BOOK.fullTitle)}</dc:title>
    <dc:creator id="creator">${escapeHtml(BOOK.author)}</dc:creator>
    <meta refines="#creator" property="role" scheme="marc:relators">aut</meta>
    <dc:publisher>${escapeHtml(BOOK.publisher)}</dc:publisher>
    <dc:language>${BOOK.language}</dc:language>
    <dc:date>${RELEASE_DATE}</dc:date>
    <dc:description>${escapeHtml(BOOK.description)}</dc:description>
    <dc:rights>Creative Commons Attribution 4.0 International; code examples under the MIT License.</dc:rights>
    <meta property="dcterms:modified">${MODIFIED}</meta>
    <meta property="rendition:layout">reflowable</meta>
    <meta property="rendition:orientation">auto</meta>
    <meta property="rendition:spread">auto</meta>
    <meta property="schema:accessMode">textual</meta>
    <meta property="schema:accessMode">visual</meta>
    <meta property="schema:accessModeSufficient">textual</meta>
    <meta property="schema:accessibilityFeature">structuralNavigation</meta>
    <meta property="schema:accessibilityFeature">tableOfContents</meta>
    <meta property="schema:accessibilityFeature">readingOrder</meta>
    <meta property="schema:accessibilityFeature">displayTransformability</meta>
    <meta property="schema:accessibilityHazard">none</meta>
    <meta property="schema:accessibilitySummary">This reflowable edition provides semantic headings, structural navigation, a table of contents, descriptive link text, and a text alternative for the cover. Mathematical expressions are represented as readable Unicode text.</meta>
    <meta name="cover" content="cover-image" />
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav" />
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml" />
    <item id="css" href="css/epub.css" media-type="text/css" />
    <item id="cover-image" href="images/cover.jpg" media-type="image/jpeg" properties="cover-image" />
    <item id="cover" href="text/cover.xhtml" media-type="application/xhtml+xml" />
    <item id="title-page" href="text/title.xhtml" media-type="application/xhtml+xml" />
    ${chapterManifest}
  </manifest>
  <spine toc="ncx" page-progression-direction="ltr">
    <itemref idref="cover" />
    <itemref idref="title-page" />
    ${chapterSpine}
  </spine>
</package>
`);

  execFileSync('python3', [path.join(__dirname, 'package_epub.py'), STAGING, EPUB], { stdio: 'inherit' });
  fs.rmSync(STAGING, { recursive: true, force: true });
  console.log(`Built ${path.relative(ROOT, EPUB)} with ${sections.length} manuscript documents and ${headings.length} headings.`);
}

try {
  main();
} catch (error) {
  fs.rmSync(STAGING, { recursive: true, force: true });
  console.error(error.stack || error.message || String(error));
  process.exitCode = 1;
}
