#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const { BOOK, escapeHtml, locatePlaywright } = require('./build.js');

const ROOT = path.resolve(__dirname, '..');
const DIST = path.join(ROOT, 'dist');
const OUTPUT = path.join(DIST, 'open-llm-cover.jpg');
const SOCIAL_OUTPUT = path.join(DIST, 'open-llm-social-card.jpg');

function coverHtml() {
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><style>
* { box-sizing: border-box; }
html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; }
body {
  position: relative;
  padding: 92px 80px 76px;
  color: white;
  background:
    radial-gradient(circle at 88% 16%, rgba(19,184,166,.55), transparent 27%),
    radial-gradient(circle at 6% 92%, rgba(21,94,239,.68), transparent 36%),
    linear-gradient(145deg, #071224 0%, #10274a 52%, #091728 100%);
  display: flex;
  flex-direction: column;
  font-family: "Noto Sans", Arial, sans-serif;
}
body::after {
  content: "";
  position: absolute;
  width: 570px;
  height: 570px;
  right: -290px;
  bottom: -300px;
  border: 2px solid rgba(255,255,255,.3);
  border-radius: 50%;
  box-shadow: 0 0 0 65px rgba(255,255,255,.026), 0 0 0 130px rgba(255,255,255,.018);
}
.kicker { margin-top: 22px; color: #a5f3e9; font-size: 15px; font-weight: 800; letter-spacing: .18em; }
h1 { margin: 145px 0 14px; font-size: 105px; line-height: .9; letter-spacing: -.06em; }
.accent { width: 165px; height: 8px; margin: 0 0 28px; background: linear-gradient(90deg, #155eef, #13b8a6); }
.subtitle { max-width: 690px; margin: 0; color: #dce8ff; font-size: 48px; line-height: 1.16; letter-spacing: -.025em; }
.rule { width: 132px; height: 8px; margin: 62px 0 28px; background: #f5b942; }
.deck { max-width: 700px; margin: 0; color: #cbd8ee; font-size: 21px; line-height: 1.55; }
.author { margin-top: auto; padding-bottom: 18px; font-size: 24px; font-weight: 800; }
.meta { color: #a9bad5; font-size: 17px; line-height: 1.5; }
</style></head><body>
<div class="kicker">AN OPEN, ZERO-PREREQUISITE FIELD GUIDE</div>
<h1>${escapeHtml(BOOK.title.toUpperCase())}</h1>
<div class="accent"></div>
<p class="subtitle">${escapeHtml(BOOK.subtitle)}</p>
<div class="rule"></div>
<p class="deck">Math • neural networks • transformers • training • reasoning • agents • retrieval • multimodality • systems • evaluation • safety • hands-on labs</p>
<div class="author">${escapeHtml(BOOK.author)}</div>
<div class="meta">${escapeHtml(BOOK.edition)} · Release ${escapeHtml(BOOK.release)}<br>Technology snapshot: ${escapeHtml(BOOK.snapshot)}</div>
</body></html>`;
}

function socialHtml(coverDataUri) {
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><style>
* { box-sizing: border-box; }
html, body { width: 1200px; height: 630px; margin: 0; overflow: hidden; }
body {
  position: relative;
  padding: 72px 65px;
  color: white;
  background: linear-gradient(135deg, #071224, #123563);
  font-family: "Noto Sans", Arial, sans-serif;
}
.cover { position: absolute; left: 800px; top: 81px; width: 330px; height: 468px; }
.kicker { color: #a5f3e9; font-size: 17px; font-weight: 800; letter-spacing: .14em; }
h1 { margin: 22px 0 0; font-size: 82px; line-height: 1; letter-spacing: -.05em; }
.subtitle { max-width: 640px; margin: 18px 0 0; color: #dce8ff; font-size: 29px; line-height: 1.3; }
.rule { width: 150px; height: 8px; margin: 40px 0 28px; background: #f5b942; }
.formats { margin: 0; color: #cbd8ee; font-size: 20px; }
.meta { position: absolute; left: 65px; bottom: 52px; margin: 0; color: #a9bad5; font-size: 17px; }
</style></head><body>
<img class="cover" src="${coverDataUri}" alt="">
<div class="kicker">A FREE AND OPEN BOOK</div>
<h1>${escapeHtml(BOOK.title.toUpperCase())}</h1>
<p class="subtitle">${escapeHtml(BOOK.subtitle)}</p>
<div class="rule"></div>
<p class="formats">Read online • PDF • EPUB 3</p>
<p class="meta">${escapeHtml(BOOK.author)} • ${escapeHtml(BOOK.edition)} • ${escapeHtml(BOOK.snapshot)}</p>
</body></html>`;
}

async function main() {
  fs.mkdirSync(DIST, { recursive: true });
  const { chromium } = locatePlaywright();
  const browser = await chromium.launch({ headless: true });
  try {
    const coverPage = await browser.newPage({ viewport: { width: 874, height: 1240 }, deviceScaleFactor: 2 });
    await coverPage.setContent(coverHtml(), { waitUntil: 'load' });
    await coverPage.screenshot({ path: OUTPUT, type: 'jpeg', quality: 95, fullPage: false });

    const coverDataUri = `data:image/jpeg;base64,${fs.readFileSync(OUTPUT).toString('base64')}`;
    const socialPage = await browser.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1 });
    await socialPage.setContent(socialHtml(coverDataUri), { waitUntil: 'load' });
    await socialPage.screenshot({ path: SOCIAL_OUTPUT, type: 'jpeg', quality: 92, fullPage: false });
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
