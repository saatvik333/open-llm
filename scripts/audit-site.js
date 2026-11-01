#!/usr/bin/env node
'use strict';

// Serves dist/site on a loopback port with its `_headers` rules applied, then loads the landing
// page, the standalone book edition, and the 404 page in Chromium. It fails on remote requests,
// browser errors, Content-Security-Policy violations, broken internal links, missing alt text,
// unnamed links, and heading-level jumps.

const fs = require('fs');
const http = require('http');
const path = require('path');
const { BOOK, locatePlaywright } = require('./build.js');

const ROOT = path.resolve(__dirname, '..');
const SITE = path.join(ROOT, 'dist', 'site');
const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.jpg': 'image/jpeg',
  '.txt': 'text/plain; charset=utf-8',
  '.xml': 'application/xml',
};

function parseHeaderRules(text) {
  const rules = [];
  let current = null;
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim() || line.trim().startsWith('#')) continue;
    if (!/^\s/.test(line)) {
      current = { pattern: line.trim(), headers: [] };
      rules.push(current);
    } else if (line.trim().startsWith('!')) {
      current.headers.push([line.trim().slice(1).trim().toLowerCase(), null]);
    } else {
      const [name, ...value] = line.trim().split(':');
      current.headers.push([name.trim().toLowerCase(), value.join(':').trim()]);
    }
  }
  return rules;
}

function patternToRegExp(pattern) {
  const escape = part => part.replace(/[.+?^$|(){}[\]\\]/g, '\\$&');
  return new RegExp(`^${pattern.split('*').map(escape).join('.*')}$`);
}

function headersFor(rules, pathname) {
  const result = {};
  for (const rule of rules) {
    if (!patternToRegExp(rule.pattern).test(pathname)) continue;
    for (const [name, value] of rule.headers) {
      if (value === null) delete result[name];
      else result[name] = name in result ? `${result[name]}, ${value}` : value;
    }
  }
  return result;
}

function startServer(rules) {
  const server = http.createServer((request, response) => {
    const { pathname } = new URL(request.url, 'http://localhost');
    let status = 200;
    let file = path.join(SITE, decodeURIComponent(pathname));
    if (!file.startsWith(SITE)) file = '';
    if (file && pathname.endsWith('/')) file = path.join(file, 'index.html');
    if (!file || !fs.existsSync(file) || !fs.statSync(file).isFile() || path.basename(file) === '_headers') {
      if (file && fs.existsSync(file) && fs.statSync(file).isDirectory()) {
        response.writeHead(301, { location: `${pathname}/` });
        response.end();
        return;
      }
      status = 404;
      file = path.join(SITE, '404.html');
    }
    const headers = headersFor(rules, pathname);
    // Chromium upgrades loopback subresources to https under this directive, which has nothing to
    // upgrade on the real https site; drop it for the plain-http local server only.
    if (headers['content-security-policy']) {
      headers['content-security-policy'] = headers['content-security-policy'].replace(/;?\s*upgrade-insecure-requests/, '');
    }
    headers['content-type'] = TYPES[path.extname(file)] || 'application/octet-stream';
    response.writeHead(status, headers);
    response.end(fs.readFileSync(file));
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server)));
}

async function loadPage(browser, origin, route, failures) {
  const page = await browser.newPage({ viewport: { width: 1100, height: 800 } });
  const label = route;
  page.on('request', request => {
    if (!request.url().startsWith(origin) && !request.url().startsWith('data:')) failures.push(`${label}: remote request ${request.url()}`);
  });
  page.on('pageerror', error => failures.push(`${label}: page error ${error}`));
  page.on('console', message => {
    if (message.type() === 'error') failures.push(`${label}: console error ${message.text()}`);
  });
  await page.addInitScript(() => {
    window.__violations = [];
    document.addEventListener('securitypolicyviolation', event => {
      window.__violations.push(`${event.violatedDirective} ${event.blockedURI}`);
    });
  });
  const response = await page.goto(`${origin}${route}`, { waitUntil: 'load' });
  const violations = await page.evaluate(() => window.__violations);
  if (violations.length) failures.push(`${label}: CSP violations ${violations.join('; ')}`);
  return { page, response };
}

async function auditLanding(browser, origin, failures) {
  const { page, response } = await loadPage(browser, origin, '/', failures);
  try {
    if (response.status() !== 200) failures.push(`/: HTTP ${response.status()}`);
    const csp = response.headers()['content-security-policy'] || '';
    if (!csp.includes("default-src 'none'")) failures.push("/: missing restrictive Content-Security-Policy");
    const audit = await page.evaluate(() => ({
      title: document.title,
      lang: document.documentElement.lang,
      description: document.querySelector('meta[name="description"]')?.content || '',
      robots: document.querySelector('meta[name="robots"]')?.content || '',
      scripts: [...document.scripts].filter(script => script.type !== 'application/ld+json').length,
      inlineHandlers: document.querySelectorAll('[onclick],[onload],[onerror]').length,
      brokenImages: [...document.images].filter(image => !image.complete || !image.naturalWidth).length,
      missingImageAlt: [...document.images].filter(image => !image.alt).length,
      emptyLinks: [...document.links].filter(link => !link.textContent.trim() && !link.getAttribute('aria-label')).length,
      headings: [...document.querySelectorAll('h1,h2,h3,h4,h5,h6')].map(heading => Number(heading.tagName[1])),
      landmarks: ['header', 'main', 'footer'].filter(tag => !document.querySelector(tag)),
      links: [...document.links].map(link => link.href),
      placeholders: /\{\{[A-Z_]+\}\}/.test(document.documentElement.outerHTML),
    }));
    const headingJumps = audit.headings.filter((level, index, levels) => index && level > levels[index - 1] + 1);
    if (!audit.title) failures.push('/: missing <title>');
    if (audit.lang !== BOOK.language) failures.push(`/: html lang is "${audit.lang}"`);
    if (!audit.description) failures.push('/: missing meta description');
    if (/noindex|nofollow/i.test(audit.robots)) failures.push('/: page blocks indexing');
    if (audit.scripts) failures.push(`/: ${audit.scripts} executable script element(s)`);
    if (audit.inlineHandlers) failures.push(`/: ${audit.inlineHandlers} inline event handler(s)`);
    if (audit.brokenImages) failures.push(`/: ${audit.brokenImages} image(s) failed to load`);
    if (audit.missingImageAlt) failures.push(`/: ${audit.missingImageAlt} image(s) lack alt text`);
    if (audit.emptyLinks) failures.push(`/: ${audit.emptyLinks} link(s) lack accessible names`);
    if (audit.headings.filter(level => level === 1).length !== 1) failures.push('/: page must have exactly one h1');
    if (headingJumps.length) failures.push('/: heading hierarchy skips levels');
    if (audit.landmarks.length) failures.push(`/: missing landmark(s): ${audit.landmarks.join(', ')}`);
    if (audit.placeholders) failures.push('/: unresolved template placeholder');

    const download = `${BOOK.repositoryUrl}/releases/latest/download/${BOOK.fileBase}`;
    for (const expected of [`${download}.pdf`, `${download}.epub`, BOOK.repositoryUrl, `${origin}/read/`]) {
      if (!audit.links.some(link => link.replace(/\/$/, '') === expected.replace(/\/$/, ''))) failures.push(`/: missing link to ${expected}`);
    }
    for (const link of new Set(audit.links)) {
      if (link.startsWith(origin)) {
        const reply = await fetch(link.split('#')[0], { redirect: 'manual' });
        if (reply.status !== 200) failures.push(`/: internal link ${link} returned HTTP ${reply.status}`);
      } else if (!link.startsWith('https://')) {
        failures.push(`/: non-https external link ${link}`);
      }
    }
    return audit;
  } finally {
    await page.close();
  }
}

async function auditBook(browser, origin, failures) {
  const { page, response } = await loadPage(browser, origin, '/read/', failures);
  try {
    if (response.status() !== 200) failures.push(`/read/: HTTP ${response.status()}`);
    const audit = await page.evaluate(() => ({
      title: document.title,
      toc: document.querySelectorAll('nav.toc .toc-entry').length,
      main: Boolean(document.querySelector('main#main-content')),
      styled: document.styleSheets.length === 1 && document.styleSheets[0].cssRules.length > 0,
      scripts: [...document.scripts].filter(script => script.type !== 'application/ld+json').length,
    }));
    if (audit.title !== BOOK.fullTitle) failures.push('/read/: title does not match book.json');
    if (!audit.toc) failures.push('/read/: contents are missing');
    if (!audit.main) failures.push('/read/: main landmark is missing');
    if (!audit.styled) failures.push('/read/: stylesheet was blocked or empty');
    if (audit.scripts) failures.push(`/read/: ${audit.scripts} executable script element(s)`);
    return audit;
  } finally {
    await page.close();
  }
}

async function auditNotFound(browser, origin, failures) {
  const page = await browser.newPage();
  try {
    const response = await page.goto(`${origin}/no-such-page`, { waitUntil: 'load' });
    if (response.status() !== 404) failures.push(`/no-such-page: expected HTTP 404, received ${response.status()}`);
    if (!(await page.textContent('body')).includes('Page not found')) failures.push('/no-such-page: 404 page was not served');
  } finally {
    await page.close();
  }
}

async function main() {
  if (!fs.existsSync(path.join(SITE, 'index.html'))) throw new Error('Missing dist/site; run node scripts/build-site.js first.');
  const rules = parseHeaderRules(fs.readFileSync(path.join(SITE, '_headers'), 'utf8'));
  const server = await startServer(rules);
  const origin = `http://127.0.0.1:${server.address().port}`;
  const { chromium } = locatePlaywright();
  const browser = await chromium.launch({ headless: true });
  const failures = [];
  try {
    const landing = await auditLanding(browser, origin, failures);
    const book = await auditBook(browser, origin, failures);
    await auditNotFound(browser, origin, failures);
    if (failures.length) throw new Error(failures.join('\n'));
    console.log('PASS — site browser audit');
    console.log('  remote requests: 0');
    console.log('  browser errors and CSP violations: 0');
    console.log(`  landing page: ${landing.links.length} links resolved, no scripts, one h1, no heading jumps`);
    console.log(`  online edition: ${book.toc} contents entries, stylesheet allowed by hash`);
    console.log('  404 page: served with HTTP 404');
  } finally {
    await browser.close();
    server.close();
  }
}

main().catch(error => {
  console.error(`FAIL — ${error.stack || error.message || String(error)}`);
  process.exitCode = 1;
});
