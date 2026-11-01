#!/usr/bin/env node
'use strict';

// Renders the retail cover (1748 × 2480) and the social card (1200 × 630) in the book's
// design: dark panel, Instrument Serif display, JetBrains Mono labels, one blue accent.

const fs = require('fs');
const path = require('path');
const { BOOK, escapeHtml, locatePlaywright, fontFaces, FONTS } = require('./build.js');

const ROOT = path.resolve(__dirname, '..');
const DIST = path.join(ROOT, 'dist');
const OUTPUT = path.join(DIST, 'open-llm-cover.jpg');
const SOCIAL_OUTPUT = path.join(DIST, 'open-llm-social-card.jpg');

const BASE_CSS = `
${fontFaces(file => `file://${path.join(FONTS, file)}`)}
:root {
  --ink: #ece8df; --muted: #9a9ea6; --line: #2b2f36; --accent: #8ea4ff; --panel: #0e1116;
  --serif: "Instrument Serif", Georgia, serif; --sans: "Inter Tight", Arial, sans-serif; --mono: "JetBrains Mono", monospace;
}
* { box-sizing: border-box; }
body { margin: 0; overflow: hidden; color: var(--ink); background: var(--panel); font-family: var(--sans); }
em { color: var(--accent); }`;

function coverHtml() {
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><style>${BASE_CSS}
html, body { width: 100%; height: 100%; }
body { padding: 62px 62px 56px; display: flex; flex-direction: column; }
.row { display: flex; justify-content: space-between; color: var(--muted); font: 500 14px/1.4 var(--mono); letter-spacing: .06em; text-transform: uppercase; }
.top { padding-bottom: 16px; border-bottom: 1px solid var(--line); }
.bottom { padding-top: 16px; border-top: 1px solid var(--line); }
.display { margin: 150px 0 0; font: 400 112px/.92 var(--serif); letter-spacing: -.03em; }
.deck { max-width: 600px; margin: 46px 0 0; color: var(--muted); font-size: 20px; line-height: 1.55; }
.mark { margin: auto 0 18px; font: italic 400 190px/.9 var(--serif); letter-spacing: -.04em; }
.mark span { margin-left: 10px; color: var(--accent); font: 600 32px/1 var(--mono); letter-spacing: .14em; vertical-align: 92px; }
</style></head><body>
<div class="row top"><span>A free and open book</span><span>${escapeHtml(BOOK.edition)} · ${escapeHtml(BOOK.year)}</span></div>
<p class="display">From nothing to the <em>frontier</em> of language models.</p>
<p class="deck">Mathematics · neural networks · transformers · training · reinforcement learning · reasoning · retrieval · agents · serving · evaluation · safety · hands-on labs</p>
<div class="mark">Open<span>LLM</span></div>
<div class="row bottom"><span>${escapeHtml(BOOK.author)}</span><span>Technology snapshot ${escapeHtml(BOOK.snapshot)}</span></div>
</body></html>`;
}

function socialHtml(coverDataUri) {
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><style>${BASE_CSS}
html, body { width: 1200px; height: 630px; }
body { position: relative; padding: 64px; }
.cover { position: absolute; right: 74px; top: 66px; width: 350px; height: 497px; border-radius: 3px; box-shadow: 0 0 0 1px var(--line); transform: rotate(2deg); }
.kicker { margin: 0; color: var(--muted); font: 500 16px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; }
.display { max-width: 660px; margin: 42px 0 0; font: 400 78px/.94 var(--serif); letter-spacing: -.03em; }
.meta { position: absolute; left: 64px; bottom: 58px; margin: 0; color: var(--muted); font: 500 16px/1.4 var(--mono); letter-spacing: .04em; }
</style></head><body>
<img class="cover" src="${coverDataUri}" alt="">
<p class="kicker">Open LLM · a free and open book</p>
<p class="display">From nothing to the <em>frontier</em> of language models.</p>
<p class="meta">Read online · PDF · EPUB — ${escapeHtml(BOOK.edition)}, ${escapeHtml(BOOK.snapshot)}</p>
</body></html>`;
}

// Pages are written to disk and opened with file:// so the file:// font URLs load.
async function shoot(browser, html, viewport, scale, output, quality) {
  const file = path.join(DIST, `.${path.basename(output, '.jpg')}.html`);
  fs.writeFileSync(file, html);
  const page = await browser.newPage({ viewport, deviceScaleFactor: scale });
  try {
    await page.goto(`file://${file}`, { waitUntil: 'load' });
    await page.evaluate(() => document.fonts.ready);
    const failed = await page.evaluate(() => [...document.fonts].filter(face => face.status === 'error').length);
    if (failed) throw new Error(`${path.basename(output)}: ${failed} font face(s) failed to load`);
    await page.screenshot({ path: output, type: 'jpeg', quality, fullPage: false });
  } finally {
    await page.close();
    fs.rmSync(file, { force: true });
  }
}

async function main() {
  fs.mkdirSync(DIST, { recursive: true });
  const { chromium } = locatePlaywright();
  const browser = await chromium.launch({ headless: true });
  try {
    await shoot(browser, coverHtml(), { width: 874, height: 1240 }, 2, OUTPUT, 95);
    const coverDataUri = `data:image/jpeg;base64,${fs.readFileSync(OUTPUT).toString('base64')}`;
    await shoot(browser, socialHtml(coverDataUri), { width: 1200, height: 630 }, 1, SOCIAL_OUTPUT, 92);
  } finally {
    await browser.close();
  }
  console.log(`Built ${path.relative(ROOT, OUTPUT)} (1748 × 2480 px)`);
  console.log(`Built ${path.relative(ROOT, SOCIAL_OUTPUT)} (1200 × 630 px)`);
}

main().catch(error => {
  console.error(error.stack || error.message || String(error));
  process.exitCode = 1;
});
