#!/usr/bin/env node
'use strict';

// Print edition for Pothi.com hardcover (case binding), A5, black-and-white interior.
// Writes dist/print/<fileBase>-interior.pdf (grayscale, fonts embedded, no cover page) and
// dist/print/<fileBase>-hardcover.pdf (CMYK case wrap: flap, back, spine, front, flap).
// Cover geometry follows https://pothi.com/print-book-cover-specification/ for A5 hardcover.

const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const { BOOK, STYLE, FONTS, escapeHtml, fontFaces, buildPdf, locatePlaywright } = require('./build.js');

const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'dist', 'print');
const INTERIOR = path.join(OUT, `${BOOK.fileBase}-interior.pdf`);
const COVER = path.join(OUT, `${BOOK.fileBase}-hardcover.pdf`);

// Inches, from Pothi's A5 hardcover formula.
const FLAP = 0.787;
const BOARD_W = 6.106;
const BOARD_H = 8.664;
const spineWidth = pages => 0.157 + 0.001968 * pages;

// The page number sits about 14mm above the bottom edge, inside Pothi's 0.5in minimum margin.
const FOOTER = '<div style="font-size:7px;letter-spacing:.06em;color:#000;width:100%;text-align:center;margin-bottom:9mm;font-family:\'DejaVu Sans Mono\',monospace"><span class="pageNumber"></span></div>';

const fontCss = fontFaces(file => `file://${path.join(FONTS, file)}`);

function ghostscript(input, output, colour) {
  const strategy = colour === 'gray' ? ['-sColorConversionStrategy=Gray', '-dProcessColorModel=/DeviceGray']
    : ['-sColorConversionStrategy=CMYK', '-dProcessColorModel=/DeviceCMYK'];
  execFileSync('gs', ['-q', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.6',
    '-dEmbedAllFonts=true', '-dSubsetFonts=true', '-dAutoRotatePages=/None', ...strategy, `-sOutputFile=${output}`, input]);
}

function coverHtml(pages) {
  const spine = spineWidth(pages);
  const width = 2 * FLAP + 2 * BOARD_W + spine;
  const height = 2 * FLAP + BOARD_H;
  return { width, height, spine, html: `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><style>${fontCss}
@page { size: ${width}in ${height}in; margin: 0; }
:root { --ink: #ece8df; --muted: #9a9ea6; --line: #2b2f36; --accent: #8ea4ff; --panel: #0e1116;
  --serif: "Instrument Serif", serif; --sans: "Inter Tight", sans-serif; --mono: "JetBrains Mono", monospace; }
* { box-sizing: border-box; margin: 0; }
html, body { width: ${width}in; height: ${height}in; background: var(--panel); color: var(--ink); font-family: var(--sans);
  print-color-adjust: exact; -webkit-print-color-adjust: exact; }
.panel { position: absolute; top: ${FLAP}in; height: ${BOARD_H}in; display: flex; flex-direction: column; }
/* Keep text 12mm inside each board edge, and 16mm clear of the spine hinge. */
.back { left: ${FLAP}in; width: ${BOARD_W}in; padding: 14mm 16mm 13mm 12mm; }
.front { left: ${FLAP + BOARD_W + spine}in; width: ${BOARD_W}in; padding: 14mm 12mm 13mm 16mm; }
.spine { left: ${FLAP + BOARD_W}in; width: ${spine}in; padding: 14mm 0; align-items: center; justify-content: space-between; }
.row { display: flex; justify-content: space-between; gap: 4mm; color: var(--muted); font: 500 6.4pt/1.4 var(--mono); letter-spacing: .06em; text-transform: uppercase; }
.top { padding-bottom: 2.6mm; border-bottom: .25mm solid var(--line); }
.bottom { margin-top: auto; padding-top: 2.6mm; border-top: .25mm solid var(--line); }
em { color: var(--accent); }
.display { margin-top: 24mm; font: 400 41pt/.92 var(--serif); letter-spacing: -.03em; }
.deck { max-width: 104mm; margin-top: 8mm; color: var(--muted); font-size: 9pt; line-height: 1.55; }
.mark { margin: auto 0 3mm; font: italic 400 70pt/.9 var(--serif); letter-spacing: -.04em; }
.mark span { margin-left: 1.5mm; color: var(--accent); font: 600 11pt/1 var(--mono); letter-spacing: .14em; vertical-align: 34pt; font-style: normal; }
.kicker { margin-top: 18mm; color: var(--muted); font: 500 6.4pt/1.4 var(--mono); letter-spacing: .08em; text-transform: uppercase; }
.blurb { margin-top: 5mm; font: 400 19pt/1.12 var(--serif); letter-spacing: -.01em; }
.body { margin-top: 6mm; color: #c9c5bc; font-size: 8.6pt; line-height: 1.58; }
.body p + p { margin-top: 3mm; }
.spine-title, .spine-meta { writing-mode: vertical-rl; }
.spine-title { font: italic 400 22pt/1 var(--serif); letter-spacing: -.02em; }
.spine-title span { color: var(--accent); font: 600 7pt/1 var(--mono); letter-spacing: .14em; font-style: normal; margin-top: 2mm; }
.spine-meta { color: var(--muted); font: 500 6pt/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; }
</style></head><body>
<div class="panel back">
  <div class="row top"><span>Open LLM</span><span>${escapeHtml(BOOK.edition)}</span></div>
  <p class="kicker">About this book</p>
  <p class="blurb">A zero-prerequisite path from how a computer stores a number to how today’s language models are built, trained, served, and kept safe.</p>
  <div class="body">
    <p>Bits, vectors, probability, and gradients first. Then the transformer, piece by piece, and everything built on it: training at scale, reinforcement learning for reasoning, retrieval, agents, serving, evaluation, safety, and the field as it stood on ${escapeHtml(BOOK.snapshot)}. Hands-on labs throughout build each idea in code.</p>
    <p>Written by AI language models under the direction of the Open LLM Project and checked by separate fact-checking passes against the cited primary sources.</p>
    <p>Free to read and share. Text CC BY 4.0, code MIT. Source, downloads, and corrections: github.com/${escapeHtml(BOOK.repository)}</p>
  </div>
  <div class="row bottom"><span>${escapeHtml(BOOK.author)}</span><span>Release ${escapeHtml(BOOK.release)}</span></div>
</div>
<div class="panel spine">
  <div class="spine-title">Open <span>LLM</span></div>
  <div class="spine-meta">${escapeHtml(BOOK.edition)} · ${escapeHtml(BOOK.year)}</div>
</div>
<div class="panel front">
  <div class="row top"><span>A free and open book</span><span>${escapeHtml(BOOK.edition)} · ${escapeHtml(BOOK.year)}</span></div>
  <p class="display">From nothing to the <em>frontier</em> of language models.</p>
  <p class="deck">Mathematics · neural networks · transformers · training · reinforcement learning · reasoning · retrieval · agents · serving · evaluation · safety · hands-on labs</p>
  <div class="mark">Open<span>LLM</span></div>
  <div class="row bottom"><span>${escapeHtml(BOOK.author)}</span><span>Technology snapshot ${escapeHtml(BOOK.snapshot)}</span></div>
</div>
</body></html>` };
}

async function buildCover(pages) {
  const { width, height, spine, html } = coverHtml(pages);
  const htmlPath = path.join(OUT, '.hardcover.html');
  const rgbPath = path.join(OUT, '.hardcover-rgb.pdf');
  fs.writeFileSync(htmlPath, html);
  const { chromium } = locatePlaywright();
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.goto(`file://${htmlPath}`, { waitUntil: 'load' });
    await page.evaluate(() => document.fonts.ready);
    const failed = await page.evaluate(() => [...document.fonts].filter(face => face.status === 'error').length);
    if (failed) throw new Error(`Cover: ${failed} font face(s) failed to load`);
    await page.pdf({ path: rgbPath, width: `${width}in`, height: `${height}in`, printBackground: true, preferCSSPageSize: true });
  } finally {
    await browser.close();
  }
  ghostscript(rgbPath, COVER, 'cmyk');
  fs.rmSync(htmlPath);
  fs.rmSync(rgbPath);
  return { width, height, spine };
}

async function main() {
  fs.mkdirSync(OUT, { recursive: true });
  const css = fontCss + fs.readFileSync(STYLE, 'utf8') + fs.readFileSync(path.join(ROOT, 'assets', 'print.css'), 'utf8');
  const htmlPath = path.join(OUT, '.interior.html');
  const rgbPath = path.join(OUT, '.interior-rgb.pdf');
  const { pages } = await buildPdf({ css, htmlPath, pdfPath: rgbPath, cover: false, footerTemplate: FOOTER });
  ghostscript(rgbPath, INTERIOR, 'gray');
  fs.rmSync(htmlPath);
  fs.rmSync(rgbPath);

  const { width, height, spine } = await buildCover(pages);
  const inch = value => `${value.toFixed(3)} in (${(value * 25.4).toFixed(1)} mm)`;
  console.log(`Built ${path.relative(ROOT, INTERIOR)} (A5, ${pages} pages, grayscale)`);
  console.log(`Built ${path.relative(ROOT, COVER)} (CMYK, ${inch(width)} × ${inch(height)}, spine ${inch(spine)})`);
}

main().catch(error => {
  console.error(error.stack || error.message || String(error));
  process.exitCode = 1;
});
