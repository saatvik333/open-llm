#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..');
const MANUSCRIPT = path.join(ROOT, 'manuscript');
const DIST = path.join(ROOT, 'dist');
const STYLE = path.join(ROOT, 'assets', 'book.css');

// Single source of truth for edition, version, and release-date metadata.
// Python tooling, the site builder, and the release preflight read the same file.
const META = JSON.parse(fs.readFileSync(path.join(ROOT, 'book.json'), 'utf8'));

const BOOK = Object.freeze({
  ...META,
  fileBase: `Open-LLM-v${META.version}`,
  repositoryUrl: `https://github.com/${META.repository}`,
});

// PDF date strings (D:YYYYMMDDHHmmSSZ) are fixed to the release date so rebuilds are reproducible.
const PDF_DATE = `D:${BOOK.releaseDate.replace(/-/g, '')}000000Z`;

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function slugify(text, used) {
  const base = text
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '') || 'section';
  let slug = base;
  let n = 2;
  while (used.has(slug)) slug = `${base}-${n++}`;
  used.add(slug);
  return slug;
}

function inline(text, options = {}) {
  const { linkCitations = true } = options;
  const code = [];
  let value = escapeHtml(text).replace(/`([^`]+)`/g, (_, body) => {
    const key = `@@CODE${code.length}@@`;
    code.push(`<code>${body}</code>`);
    return key;
  });

  value = value
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+|#[^\s)]+)\)/g, (_, label, url) => {
      const klass = url.startsWith('#') ? 'internal-link' : 'external-link';
      return `<a class="${klass}" href="${url}">${label}</a>`;
    })
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/__([^_]+)__/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*]+)\*/g, '$1<em>$2</em>')
    .replace(/~~([^~]+)~~/g, '<s>$1</s>');

  if (linkCitations) {
    value = value.replace(/\[((?:R\d{2,3})(?:,\s*R\d{2,3})*)\]/g, (_, references) => {
      const links = references.split(/,\s*/).map(reference => {
        const id = reference.toLowerCase();
        return `<a class="citation" href="#ref-${id}">${reference}</a>`;
      });
      return `[${links.join(', ')}]`;
    });
  }

  code.forEach((html, index) => {
    value = value.replace(`@@CODE${index}@@`, html);
  });
  return value;
}

function tableCells(line) {
  let value = line.trim();
  if (value.startsWith('|')) value = value.slice(1);
  if (value.endsWith('|')) value = value.slice(0, -1);

  const cells = [];
  let cell = '';
  let codeFenceLength = 0;
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index];
    if (character === '`') {
      let runLength = 1;
      while (value[index + runLength] === '`') runLength += 1;
      if (codeFenceLength === 0) codeFenceLength = runLength;
      else if (codeFenceLength === runLength) codeFenceLength = 0;
      cell += '`'.repeat(runLength);
      index += runLength - 1;
      continue;
    }
    if (character === '|' && codeFenceLength === 0 && value[index - 1] !== '\\') {
      cells.push(cell.trim());
      cell = '';
      continue;
    }
    cell += character;
  }
  cells.push(cell.trim());
  return cells;
}

function markdownToHtml(source, headingIndex, usedSlugs) {
  const lines = source.replace(/\r\n?/g, '\n').split('\n');
  const out = [];
  let i = 0;
  let listType = null;
  let inAside = false;
  let currentChapterId = '';

  function closeList() {
    if (listType) out.push(`</${listType}>`);
    listType = null;
  }

  function closeAside() {
    closeList();
    if (inAside) out.push('</aside>');
    inAside = false;
  }

  function isBlockStart(line, next) {
    return /^#{1,6}\s+/.test(line) || /^```/.test(line) || /^>\s?/.test(line) ||
      /^[-*+]\s+/.test(line) || /^\d+\.\s+/.test(line) || /^---+$/.test(line.trim()) ||
      /^:::\s*/.test(line) || /^<!--\s*pagebreak\s*-->$/.test(line.trim()) ||
      (line.includes('|') && next && /^\s*\|?\s*:?-{3,}/.test(next));
  }

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      closeList();
      i += 1;
      continue;
    }

    if (/^```/.test(trimmed)) {
      closeList();
      const language = trimmed.slice(3).trim();
      const body = [];
      i += 1;
      while (i < lines.length && !/^```/.test(lines[i].trim())) body.push(lines[i++]);
      if (i < lines.length) i += 1;
      const preClass = body.length > 32 ? ' class="long-code"' : '';
      const languageClass = language ? ` class="language-${escapeHtml(language)}"` : '';
      out.push(`<pre${preClass}><code${languageClass}>${escapeHtml(body.join('\n'))}</code></pre>`);
      continue;
    }

    const heading = /^(#{1,6})\s+(.+)$/.exec(trimmed);
    if (heading) {
      closeList();
      const level = heading[1].length;
      const title = heading[2].replace(/\s+#+$/, '');
      const plainTitle = title.replace(/[`*_]/g, '');
      const id = slugify(plainTitle, usedSlugs);
      const isPart = level === 1 && /^Part\s+[IVX]+\b/.test(plainTitle);
      const isAppendix = level === 1 && /^Appendix\s+[A-Z]\b/.test(plainTitle);
      headingIndex.push({ level, title: plainTitle, id, isPart, isAppendix });
      if (level === 1) currentChapterId = id;
      const classes = [];
      if (level === 1) classes.push('chapter-title');
      if (isPart) classes.push('part-title');
      if (isAppendix) classes.push('appendix-title');
      if (level === 2 && currentChapterId === 'appendix-e-a-one-year-curriculum') classes.push('curriculum-heading');
      if (level === 2 && currentChapterId === 'distributed-training-systems') classes.push('compact-heading');
      if (level === 1 && plainTitle.length >= 43) classes.push('long-title');
      const klass = classes.length ? ` class="${classes.join(' ')}"` : '';
      out.push(`<h${level}${klass} id="${id}" aria-label="${escapeHtml(plainTitle)}">${inline(title)}</h${level}>`);
      i += 1;
      continue;
    }

    const aside = /^:::\s*([a-z-]+)?\s*(.*)$/.exec(trimmed);
    if (aside) {
      closeList();
      if (inAside) {
        out.push('</aside>');
        inAside = false;
      } else {
        const type = aside[1] || 'note';
        const title = aside[2] || type[0].toUpperCase() + type.slice(1);
        out.push(`<aside class="callout ${escapeHtml(type)}" aria-label="${escapeHtml(title)}"><div class="callout-title">${inline(title)}</div>`);
        inAside = true;
      }
      i += 1;
      continue;
    }

    if (/^<!--\s*pagebreak\s*-->$/.test(trimmed)) {
      closeList();
      out.push('<div class="pagebreak" aria-hidden="true"></div>');
      i += 1;
      continue;
    }

    if (/^---+$/.test(trimmed)) {
      closeList();
      out.push('<hr>');
      i += 1;
      continue;
    }

    if (line.includes('|') && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1])) {
      closeList();
      const headers = tableCells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) rows.push(tableCells(lines[i++]));
      if (rows.some(row => row.length !== headers.length)) {
        throw new Error(`Malformed Markdown table near: ${line}`);
      }
      const compactTable = /^the-.+-landscape$/.test(currentChapterId) ? ' compact-table' : '';
      out.push(`<div class="table-wrap${compactTable}"><table><thead><tr>` + headers.map(cell => `<th scope="col">${inline(cell)}</th>`).join('') + '</tr></thead><tbody>');
      for (const row of rows) out.push('<tr>' + row.map(cell => `<td>${inline(cell)}</td>`).join('') + '</tr>');
      out.push('</tbody></table></div>');
      continue;
    }

    const bullet = /^[-*+]\s+(.+)$/.exec(trimmed);
    const number = /^\d+\.\s+(.+)$/.exec(trimmed);
    if (bullet || number) {
      const wanted = bullet ? 'ul' : 'ol';
      if (listType !== wanted) {
        closeList();
        listType = wanted;
        out.push(`<${wanted}>`);
      }
      out.push(`<li>${inline((bullet || number)[1])}</li>`);
      i += 1;
      continue;
    }

    if (/^>\s?/.test(trimmed)) {
      closeList();
      const quote = [];
      while (i < lines.length && /^>\s?/.test(lines[i].trim())) {
        quote.push(lines[i].trim().replace(/^>\s?/, ''));
        i += 1;
      }
      out.push(`<blockquote>${inline(quote.join(' '))}</blockquote>`);
      continue;
    }

    closeList();
    const paragraph = [trimmed];
    i += 1;
    while (i < lines.length && lines[i].trim() && !isBlockStart(lines[i].trim(), lines[i + 1])) {
      paragraph.push(lines[i].trim());
      i += 1;
    }
    const text = paragraph.join(' ');
    const reference = /^\*\*\[R(\d{2,3})\]\*\*/.exec(text);
    const paragraphClasses = [];
    if (reference) paragraphClasses.push('reference');
    if (currentChapterId === 'appendix-b-glossary') paragraphClasses.push('glossary-entry');
    if (currentChapterId === 'appendix-e-a-one-year-curriculum') paragraphClasses.push('curriculum-entry');
    const classAttribute = paragraphClasses.length ? ` class="${paragraphClasses.join(' ')}"` : '';
    const idAttribute = reference ? ` id="ref-r${reference[1]}"` : '';
    out.push(`<p${classAttribute}${idAttribute}>${inline(text, { linkCitations: !reference })}</p>`);
  }

  closeAside();
  return out.join('\n');
}

function locatePlaywright() {
  try {
    return require('playwright');
  } catch (error) {
    if (error.code !== 'MODULE_NOT_FOUND') throw error;
    throw new Error('Playwright is not installed. Run `npm ci` and `npx playwright install chromium` in the project root.');
  }
}

function ensureQpdf() {
  try {
    execFileSync('qpdf', ['--version'], { stdio: 'ignore' });
  } catch (_) {
    throw new Error('qpdf is required for page-numbered contents, metadata, bookmarks, and linearization.');
  }
}

function tocHtml(headings, pageNumbers = {}) {
  return headings.filter(item => item.level === 1).map(item => {
    const klass = item.isPart ? 'toc-entry toc-part' : 'toc-entry toc-chapter';
    const pageNumber = pageNumbers[item.id] || '—';
    return `<a class="${klass}" href="#${item.id}"><span class="toc-label">${inline(item.title)}</span><span class="toc-dots" aria-hidden="true"></span><span class="toc-page" aria-label="page ${pageNumber}">${pageNumber}</span></a>`;
  }).join('\n');
}

function makeHtml(css, sections, headings, pageNumbers) {
  const structuredData = JSON.stringify({
    '@context': 'https://schema.org',
    '@type': 'Book',
    name: BOOK.fullTitle,
    author: { '@type': 'Organization', name: BOOK.author },
    publisher: { '@type': 'Organization', name: BOOK.publisher },
    datePublished: BOOK.year,
    inLanguage: BOOK.language,
    license: BOOK.licenseUrl,
    bookEdition: BOOK.edition,
    description: BOOK.description,
  }).replace(/</g, '\\u003c');

  return `<!doctype html>
<html lang="${BOOK.language}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="author" content="${escapeHtml(BOOK.author)}">
<meta name="description" content="${escapeHtml(BOOK.description)}">
<meta name="keywords" content="${escapeHtml(BOOK.keywords)}">
<meta name="color-scheme" content="light">
<title>${escapeHtml(BOOK.fullTitle)}</title>
<script type="application/ld+json">${structuredData}</script>
<style>${css}</style>
</head>
<body>
<a class="skip-link" href="#main-content">Skip to the book</a>
<section class="cover" role="doc-cover" aria-label="Book cover">
  <div class="cover-kicker">AN OPEN, ZERO-PREREQUISITE FIELD GUIDE</div>
  <h1>${escapeHtml(BOOK.title.toUpperCase())}</h1>
  <div class="cover-subtitle">${escapeHtml(BOOK.subtitle)}</div>
  <div class="cover-rule" aria-hidden="true"></div>
  <p class="cover-deck">Math • neural networks • transformers • training • reasoning • agents • retrieval • multimodality • systems • evaluation • safety • hands-on labs</p>
  <div class="cover-author">${escapeHtml(BOOK.author)}</div>
  <div class="cover-meta">${escapeHtml(BOOK.edition)} · Release ${escapeHtml(BOOK.release)}<br>Technology snapshot: ${escapeHtml(BOOK.snapshot)}</div>
</section>
<section class="imprint" role="doc-imprint" aria-label="Publication information">
  <h1>${escapeHtml(BOOK.title)}</h1>
  <p class="imprint-subtitle"><strong>${escapeHtml(BOOK.subtitle)}</strong></p>
  <dl class="publication-data">
    <div><dt>Author and publisher</dt><dd>${escapeHtml(BOOK.author)}</dd></div>
    <div><dt>Edition</dt><dd>${escapeHtml(BOOK.edition)}, release ${escapeHtml(BOOK.release)} (${escapeHtml(BOOK.year)})</dd></div>
    <div><dt>Technology snapshot</dt><dd>${escapeHtml(BOOK.snapshot)}</dd></div>
  </dl>
  <p><strong>How this book was made.</strong> The text was written by AI language models under the direction of the Open LLM Project, then checked by separate AI fact-checking passes that re-opened the cited primary source for each claim, recomputed every number, and ran every code example. The publisher is responsible for the released text. Automated checking reduces errors but cannot guarantee their absence; please report any mistake you find on the project’s issue tracker.</p>
  <p>The book prose and original diagrams are licensed under <a class="external-link" rel="license" href="${BOOK.licenseUrl}">Creative Commons Attribution 4.0 International</a>. Code samples and build scripts are licensed under the MIT License. Third-party names and cited works remain the property of their respective owners.</p>
  <p><strong>Suggested citation:</strong> Open LLM Project. <em>${escapeHtml(BOOK.fullTitle)}</em>. ${escapeHtml(BOOK.edition)}, release ${escapeHtml(BOOK.release)}, ${escapeHtml(BOOK.year)}.</p>
  <p>The field changes quickly. Product names and reported capabilities are a dated snapshot, not endorsements or permanent rankings. Verify high-stakes decisions against current primary documentation. This book is educational material, not legal, medical, financial, safety, or other professional advice.</p>
  <p class="small">Free to read, download, and adapt. Source, releases, and corrections: <a class="external-link" href="${BOOK.repositoryUrl}">github.com/${escapeHtml(BOOK.repository)}</a>. Links and reference markers are clickable in the digital edition.</p>
</section>
<nav class="toc" role="doc-toc" aria-labelledby="contents-title">
  <h1 id="contents-title">Contents</h1>
  ${tocHtml(headings, pageNumbers)}
</nav>
<main id="main-content">${sections}</main>
</body>
</html>`;
}

async function renderPdf(page, htmlPath, pdfPath) {
  await page.goto(`file://${htmlPath}`, { waitUntil: 'load' });
  await page.emulateMedia({ media: 'print' });
  await page.pdf({
    path: pdfPath,
    format: 'A5',
    printBackground: true,
    preferCSSPageSize: true,
    displayHeaderFooter: true,
    headerTemplate: '<div></div>',
    footerTemplate: '<div style="font-size:8px;color:#6b7280;width:100%;padding:0 18mm;text-align:center;font-family:Arial,sans-serif"><span class="pageNumber"></span></div>',
    margin: { top: '12mm', right: '0', bottom: '12mm', left: '0' },
    outline: true,
    tagged: true,
  });
}

function qpdfJson(pdfPath, options) {
  const stdout = execFileSync('qpdf', ['--json', ...options, pdfPath], {
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
  });
  return JSON.parse(stdout);
}

function flattenOutlines(items) {
  const flattened = [];
  for (const item of items || []) {
    flattened.push(item);
    flattened.push(...flattenOutlines(item.kids));
  }
  return flattened;
}

function readOutlineTree(pdfPath) {
  return qpdfJson(pdfPath, ['--json-key=outlines']).outlines || [];
}

function manuscriptPageNumbers(headings, topLevelOutlines) {
  const chapters = headings.filter(item => item.level === 1);
  if (topLevelOutlines.length < chapters.length) {
    throw new Error(`Expected at least ${chapters.length} top-level PDF outlines but found ${topLevelOutlines.length}.`);
  }
  const manuscriptOutlines = topLevelOutlines.slice(-chapters.length);
  const pageNumbers = {};
  for (let i = 0; i < chapters.length; i += 1) {
    const page = manuscriptOutlines[i].destpageposfrom1;
    if (!Number.isInteger(page) || page < 1) throw new Error(`Missing page destination for ${chapters[i].title}.`);
    pageNumbers[chapters[i].id] = page;
  }
  return pageNumbers;
}

function samePageNumbers(a, b) {
  const keys = Object.keys(a);
  return keys.length === Object.keys(b).length && keys.every(key => a[key] === b[key]);
}

function objectOption(reference) {
  const match = /^(\d+)\s+(\d+)\s+R$/.exec(reference);
  if (!match) throw new Error(`Unexpected PDF object reference: ${reference}`);
  return `${match[1]},${match[2]}`;
}

function polishPdf(pdfPath, headings) {
  const outlines = flattenOutlines(readOutlineTree(pdfPath));
  const desiredOutlineTitles = [BOOK.title.toUpperCase(), BOOK.title, 'Contents', ...headings.map(item => item.title)];
  if (outlines.length !== desiredOutlineTitles.length) {
    throw new Error(`Expected ${desiredOutlineTitles.length} bookmarks but found ${outlines.length}.`);
  }

  const trailerJson = qpdfJson(pdfPath, ['--json-object=trailer']);
  const trailer = trailerJson.qpdf[1].trailer;
  const infoReference = trailer && trailer.value && trailer.value['/Info'];
  if (!infoReference) throw new Error('The generated PDF has no information dictionary.');

  const references = [infoReference, ...outlines.map(item => item.object)];
  const objectArgs = references.map(reference => `--json-object=${objectOption(reference)}`);
  const objectJson = qpdfJson(pdfPath, objectArgs);
  const objects = objectJson.qpdf[1];

  const infoKey = `obj:${infoReference}`;
  if (!objects[infoKey] || !objects[infoKey].value) throw new Error(`Missing metadata object ${infoReference}.`);
  Object.assign(objects[infoKey].value, {
    '/Title': `u:${BOOK.fullTitle}`,
    '/Author': `u:${BOOK.author}`,
    '/Subject': `u:${BOOK.description}`,
    '/Keywords': `u:${BOOK.keywords}`,
    '/Creator': 'u:Open LLM build system',
    '/Producer': 'u:Chromium + qpdf',
    '/CreationDate': `u:${PDF_DATE}`,
    '/ModDate': `u:${PDF_DATE}`,
  });

  outlines.forEach((outline, index) => {
    const key = `obj:${outline.object}`;
    if (!objects[key] || !objects[key].value) throw new Error(`Missing bookmark object ${outline.object}.`);
    objects[key].value['/Title'] = `u:${desiredOutlineTitles[index]}`;
  });

  const update = {
    version: 2,
    qpdf: [objectJson.qpdf[0], objects],
  };
  const updatePath = path.join(DIST, '.open-llm-pdf-update.json');
  const polishedPath = path.join(DIST, '.open-llm-polished.pdf');
  fs.writeFileSync(updatePath, JSON.stringify(update));
  try {
    execFileSync('qpdf', [pdfPath, polishedPath, '--linearize', '--deterministic-id', `--update-from-json=${updatePath}`], {
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
    });
    fs.renameSync(polishedPath, pdfPath);
  } finally {
    if (fs.existsSync(updatePath)) fs.unlinkSync(updatePath);
    if (fs.existsSync(polishedPath)) fs.unlinkSync(polishedPath);
  }
}

async function main() {
  ensureQpdf();
  if (!fs.existsSync(MANUSCRIPT)) throw new Error(`Missing manuscript directory: ${MANUSCRIPT}`);
  const files = fs.readdirSync(MANUSCRIPT).filter(name => name.endsWith('.md')).sort();
  if (!files.length) throw new Error('No Markdown files found in manuscript/.');

  const headings = [];
  const usedSlugs = new Set();
  const sections = files.map(name => {
    const source = fs.readFileSync(path.join(MANUSCRIPT, name), 'utf8');
    return markdownToHtml(source, headings, usedSlugs);
  }).join('\n');
  const css = fs.readFileSync(STYLE, 'utf8');

  fs.mkdirSync(DIST, { recursive: true });
  const htmlPath = path.join(DIST, 'open-llm.html');
  const pdfPath = path.join(DIST, 'open-llm.pdf');

  const { chromium } = locatePlaywright();
  const browser = await chromium.launch({ headless: true });
  let pageNumbers = {};
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1 });
    let stable = false;
    for (let pass = 1; pass <= 4; pass += 1) {
      fs.writeFileSync(htmlPath, makeHtml(css, sections, headings, pageNumbers));
      await renderPdf(page, htmlPath, pdfPath);
      const measured = manuscriptPageNumbers(headings, readOutlineTree(pdfPath));
      if (samePageNumbers(pageNumbers, measured)) {
        stable = true;
        break;
      }
      pageNumbers = measured;
    }
    if (!stable) throw new Error('Table-of-contents pagination did not stabilize after four passes.');
  } finally {
    await browser.close();
  }

  polishPdf(pdfPath, headings);
  execFileSync('qpdf', ['--check', pdfPath], { stdio: 'ignore' });

  const stats = fs.statSync(pdfPath);
  const pages = Number(execFileSync('qpdf', ['--show-npages', pdfPath], { encoding: 'utf8' }).trim());
  console.log(`Built ${path.relative(ROOT, pdfPath)} (${pages} pages, ${(stats.size / 1024 / 1024).toFixed(2)} MiB)`);
  console.log(`Built ${path.relative(ROOT, htmlPath)}`);
  console.log(`Included ${files.length} manuscript files, ${headings.length} headings, and ${headings.filter(item => item.level === 1).length} contents entries.`);
}

if (require.main === module) {
  main().catch(error => {
    console.error(error.stack || error.message || String(error));
    process.exitCode = 1;
  });
}

module.exports = {
  BOOK,
  escapeHtml,
  inline,
  markdownToHtml,
  slugify,
  locatePlaywright,
};
